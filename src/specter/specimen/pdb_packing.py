"""
SpherePackingSpecimenGenerator: builds a specimen volume by packing several
PDB-derived protein species -- each at its own real physical size, from
``PDB.max_diameter`` -- via hard-sphere Random Sequential Addition
(:func:`specter.crowding.pack_hard_spheres_3d`), then rendering each
accepted instance's real high-resolution scattering potential
(``PotentialBuilder``) at its packed position and a random rotation.

A third, independent alternative alongside specter's other two specimen
generators -- ``specimen/cryoet.py``'s ``CryoETSpecimenGenerator`` (polnet
SAWLC placement + membranes) and ``specimen/cryotomosim.py``'s
``CryoTomoSimSpecimenGenerator`` (a from-scratch CTS port: reject-and-retry
Monte Carlo with direct density-overlap testing). This one approximates
each species as a bounding sphere for placement (not the true molecular
shape, unlike the other two -- a real cost in packing fidelity for oddly
shaped molecules), in exchange for speed: ``pack_hard_spheres_3d`` resolves
many candidates' accept/reject decisions per pass instead of one at a time,
roughly 90x faster than a naive one-candidate-at-a-time loop at a few
thousand instances (see ``dev/packing_algorithms.py`` for the full
algorithm comparison -- RSA vs. force-biased vs. Lubachevsky-Stillinger,
sequential vs. batched vs. voxel-grid -- this was promoted from). No
membranes, ice, or carbon film here -- just densely packed protein species
at whatever occupancy fraction and species-ratio mix is requested; layer
ice on top separately (e.g. ``specter.ice.blend_ice_into_volume``) if
needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..crowding import insert_particles_into_micrograph, pack_hard_spheres_3d
from ..pdb import PDB
from ..potential import PotentialBuilder
from ..rotations import build_affine_matrix, random_rotation_matrix, rotate_volume

# specter.potential.compute_supersampling_parameters's own default atomic
# kernel width (potential.py: width_atom=5.0) -- each atom's own potential
# is evaluated over a +/-2.5 A box around it, so a molecule's bounding box
# needs at least this much margin beyond its outermost atom or that atom's
# kernel gets truncated by convolution. Same value as
# specimen/polnet_bridge.py's ATOM_KERNEL_HALF_WIDTH_A and
# specimen/cryotomosim.py's own copy -- kept as an independent local
# constant (not imported) to keep this generator's dependencies limited to
# crowding/pdb/potential/rotations, matching cryotomosim.py's own
# zero-cross-generator-coupling convention.
ATOM_KERNEL_HALF_WIDTH_A = 2.5


def estimate_protein_box_size(max_diameter: float, v_size: float) -> int:
    """
    Grid size (voxels, per axis) for a molecule with the given max diameter
    (Angstrom, from ``PDB.max_diameter``) at voxel size ``v_size``.

    Parameters
    ----------
    max_diameter : float
    v_size : float

    Returns
    -------
    int
        Even grid size in voxels.
    """
    margin_a = 2 * ATOM_KERNEL_HALF_WIDTH_A
    n = int(np.ceil((max_diameter + 2 * margin_a) / v_size))
    n += n % 2
    return n


@dataclass
class SphereProteinSpec:
    """
    One placeable protein species for `SpherePackingSpecimenGenerator`.

    Attributes
    ----------
    pdb_source : str
        PDB ID (4-character code, fetched from RCSB) or a local PDB/mmCIF
        file path.
    ratio : float, optional
        Relative abundance weight: species are drawn into the candidate
        pool with probability proportional to `ratio` across all specs --
        only the RATIO between species matters, not the absolute value
        (matches how `specter.specimen.cytosolic_filler.
        PEI2016_CROWDING_TABLE`'s `occurrence_freq` is meant to be used).
        Default 1.0 (uniform across species if left at default for all).
    """

    pdb_source: str
    ratio: float = 1.0


@dataclass
class SpherePlacement:
    """One placed instance, for ground-truth bookkeeping."""

    species_id: str  # the owning spec's pdb_source
    position_xyz: torch.Tensor  # (3,) physical Angstrom, box-centered
    rotation_matrix: torch.Tensor  # (3, 3)


class SpherePackingSpecimenGenerator:
    """
    Packs several PDB-derived protein species into a volume via hard-sphere
    RSA, sized to each species' own real physical radius, then renders each
    placed instance's real scattering potential.

    Parameters
    ----------
    protein_specs : list of SphereProteinSpec
        Species to place.
    target_shape : tuple of int, optional
        Output volume shape (Z, Y, X), voxels. Default (128, 256, 256).
    v_size : float, optional
        Voxel size, Angstrom. Default 5.0.
    occupancy_fraction : float, optional
        Target packing density: candidate instances are drawn (species-
        ratio-weighted) until their combined bare-sphere volume reaches
        this fraction of the box volume. The box's real, gap-inflated
        excluded-volume requirement is higher than this number (see
        `gap_angstrom`) -- an `occupancy_fraction` whose implied
        excluded-volume fraction exceeds the random-close-packing ceiling
        for spheres (~0.64) is not physically achievable by any packing
        algorithm; `pack_hard_spheres_3d` will simply place as many
        candidates as fit rather than erroring -- compare
        `len(placements)` to `n_candidates` below to see how much of the
        target was actually reached. Default 0.2.
    gap_angstrom : float, optional
        Minimum clearance between placed spheres' surfaces, Angstrom
        (steric/hydration-shell buffer). Default 5.0.
    pdb_cache_dir : str, optional
        Directory for downloaded PDB/mmCIF files. Default "../pdb-data/".
    parameterization : str, optional
        Atomic scattering-factor parameterization for `PotentialBuilder`
        ("kirkland" or "lobato"). Default "kirkland".
    seed : int, optional
        Random seed.
    device : str or torch.device, optional
        Device for the packing step (`pack_hard_spheres_3d`) -- see that
        function's own docstring on why "cpu" (the default) has
        outperformed "cuda" at every scale tested so far. Rotation/
        rendering always happens on CPU regardless (this generator doesn't
        move volumes to GPU).
    chunk_size : int, optional
        Number of instances to rotate per batch, per species -- caps peak
        memory when a species has many placed instances (mirrors
        `specter.crowding.CrowdWithDuplicates`'s own `chunk_size`). Default
        None (rotate all of a species' instances at once).

    Attributes
    ----------
    placements : list of SpherePlacement
        Every successfully placed instance, set after `generate()` runs.
    n_candidates : int
        Size of the drawn candidate pool, set after `generate()` runs.
    """

    def __init__(
        self,
        protein_specs: list[SphereProteinSpec],
        target_shape: tuple[int, int, int] = (128, 256, 256),
        v_size: float = 5.0,
        occupancy_fraction: float = 0.2,
        gap_angstrom: float = 5.0,
        pdb_cache_dir: str = "../pdb-data/",
        parameterization: str = "kirkland",
        seed: int | None = None,
        device: str | torch.device = "cpu",
        chunk_size: int | None = None,
    ):
        if not protein_specs:
            raise ValueError("protein_specs must be non-empty")
        self.protein_specs = protein_specs
        self.target_shape = target_shape
        self.v_size = v_size
        self.occupancy_fraction = occupancy_fraction
        self.gap_angstrom = gap_angstrom
        self.pdb_cache_dir = pdb_cache_dir
        self.parameterization = parameterization
        self.seed = seed
        self.device = device
        self.chunk_size = chunk_size

        self.placements: list[SpherePlacement] = []
        self.n_candidates = 0

    def _draw_instance_pool(
        self,
        species_radii: torch.Tensor,
        species_ratios: torch.Tensor,
        box_volume: float,
        gen: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a candidate instance pool (one row per sphere to attempt
        placing) whose combined bare-sphere volume reaches
        `occupancy_fraction * box_volume`, species drawn with probability
        proportional to `species_ratios`. Returns (radii, species_idx),
        both length N, sorted largest-radius-first (matches
        `pack_hard_spheres_3d`'s own largest-first staging, though it
        re-groups by radius internally regardless of input order)."""
        target_volume = self.occupancy_fraction * box_volume
        probs = species_ratios / species_ratios.sum()

        radii_list: list[float] = []
        species_list: list[int] = []
        accumulated = 0.0
        for _ in range(1_000_000):  # defensive cap, see draw_instance_pool precedent
            if accumulated >= target_volume:
                break
            k = int(torch.multinomial(probs, 1, generator=gen))
            r = float(species_radii[k])
            radii_list.append(r)
            species_list.append(k)
            accumulated += (4.0 / 3.0) * torch.pi * r**3

        radii = torch.tensor(radii_list)
        species_idx = torch.tensor(species_list, dtype=torch.long)
        if radii.numel() > 0:
            perm = torch.argsort(radii, descending=True)
            radii, species_idx = radii[perm], species_idx[perm]
        return radii, species_idx

    def generate(self) -> torch.Tensor:
        """
        Run the full pipeline and return the assembled specimen volume.

        Returns
        -------
        torch.Tensor
            Shape `target_shape`, dtype float32.
        """
        gen = torch.Generator()
        if self.seed is not None:
            torch.manual_seed(self.seed)
            gen.manual_seed(self.seed)

        pdbs = [
            PDB(spec.pdb_source, savefolder=self.pdb_cache_dir, verbose=False)
            for spec in self.protein_specs
        ]
        species_radii = torch.tensor([float(pdb.max_diameter) / 2.0 for pdb in pdbs])
        species_ratios = torch.tensor([spec.ratio for spec in self.protein_specs])

        box = (
            self.target_shape[0] * self.v_size,
            self.target_shape[1] * self.v_size,
            self.target_shape[2] * self.v_size,
        )
        box_volume = box[0] * box[1] * box[2]

        pool_radii, pool_species_idx = self._draw_instance_pool(
            species_radii, species_ratios, box_volume, gen
        )
        self.n_candidates = int(pool_radii.numel())

        coords, accepted_idx = pack_hard_spheres_3d(
            pool_radii, box, gap=self.gap_angstrom, seed=self.seed, device=self.device
        )
        accepted_species_idx = pool_species_idx[accepted_idx]

        volume = torch.zeros(self.target_shape, dtype=torch.float32)
        self.placements = []

        for species_i, spec in enumerate(self.protein_specs):
            mask = accepted_species_idx == species_i
            if not bool(mask.any()):
                continue
            pdb = pdbs[species_i]
            n = estimate_protein_box_size(pdb.max_diameter, self.v_size)
            builder = PotentialBuilder(
                n_xyz=n,
                dx=self.v_size,
                atomic_numbers=pdb.atomic_numbers,
                progressbars=False,
                parameterization=self.parameterization,
            )
            template = builder.forward(pdb.coordinates, method="3d")

            species_coords = coords[mask]
            n_instances = species_coords.shape[0]
            R = random_rotation_matrix(n_instances)
            if R.dim() == 2:
                R = R.unsqueeze(0)
            theta = build_affine_matrix(R)

            step = self.chunk_size or n_instances
            for start in range(0, n_instances, step):
                end = min(start + step, n_instances)
                rotated = rotate_volume(
                    template, theta[start:end], padding_mode="zeros"
                )
                volume = insert_particles_into_micrograph(
                    rotated,
                    species_coords[start:end],
                    pixel_size=self.v_size,
                    micrograph=volume,
                )

            for i in range(n_instances):
                self.placements.append(
                    SpherePlacement(
                        species_id=spec.pdb_source,
                        position_xyz=species_coords[i],
                        rotation_matrix=R[i],
                    )
                )

        return volume

    def export_picks(
        self,
        output_dir: str | Path,
        annotation_version: str = "1.0",
        oriented: bool = True,
    ) -> dict[str, Path]:
        """
        Write one copick/CryoET-Data-Portal-style .ndjson pick file per
        placed species -- same schema as `specimen.cryoet.
        CryoETSpecimenGenerator.export_picks` (one JSON object per line:
        ``{"type": "point"|"orientedPoint", "location": {"x", "y", "z"}[,
        "xyz_rotation_matrix"]}``), so picks from either generator are
        interchangeable downstream.

        Coordinates are converted from this generator's box-centered
        convention (`SpherePlacement.position_xyz`, origin at the volume's
        center, matching `crowding.pack_hard_spheres_3d`) to the
        corner-relative (``0..extent``) convention copick/the portal
        actually use -- the same conversion `CryoETSpecimenGenerator.
        export_picks` performs, just starting from a different internal
        coordinate origin (see that method's own docstring for the
        underlying axis-convention note).

        Must be called after `generate()`.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Directory to write the .ndjson files into.
        annotation_version : str, optional
            Used only in the output filename
            (``"{name}-{version}_{type}.ndjson"``). Default "1.0".
        oriented : bool, optional
            If True (default), picks are written as ``"orientedPoint"``
            with each instance's rotation matrix included; if False, as
            plain ``"point"`` (location only).

        Returns
        -------
        dict[str, pathlib.Path]
            Mapping of species id (`SphereProteinSpec.pdb_source`) to
            written file path.
        """
        if not self.placements:
            raise RuntimeError("call generate() before export_picks()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        extent_xyz = (
            torch.tensor(
                [self.target_shape[2], self.target_shape[1], self.target_shape[0]],
                dtype=torch.float32,
            )
            * self.v_size
        )

        by_species: dict[str, list[SpherePlacement]] = {}
        for placed in self.placements:
            by_species.setdefault(placed.species_id, []).append(placed)

        point_type = "orientedPoint" if oriented else "point"
        for species_id, placed_list in by_species.items():
            name = Path(species_id).stem  # strip dir/ext if a file path was used
            path = (
                output_dir / f"{name}-{annotation_version}_{point_type.lower()}.ndjson"
            )
            with open(path, "w") as f:
                for placed in placed_list:
                    corner_xyz = placed.position_xyz + extent_xyz / 2
                    x, y, z = (float(v) for v in corner_xyz)
                    row: dict = {
                        "type": point_type,
                        "location": {"x": x, "y": y, "z": z},
                    }
                    if oriented:
                        row["xyz_rotation_matrix"] = (
                            placed.rotation_matrix.numpy().tolist()
                        )
                    f.write(json.dumps(row) + "\n")
            written[species_id] = path

        return written
