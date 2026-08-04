"""
MembraneTomogramGenerator: assembles a full specimen tomogram around an
organic membrane -- transmembrane proteins on the bilayer, plus densely
packed cytosolic and vesicle-lumen protein populations, region-gated
against the membrane's own geometry so a "lumen" species can only land
inside an enclosed compartment and a "cytosol" species can only land
outside one.

Composes three independently-developed pieces rather than reimplementing
any of them: ``specimen.membrane.MembraneGenerator`` (organic shape +
transmembrane placement, unmodified), :func:`.classify_membrane_regions`
(shell/lumen/cytosol masks via connected-components flood-fill, this
subpackage), and ``specimen.packing.pack_hard_spheres_3d``'s
`exclusion_distance_field` (obstacle- and region-aware RSA, this session).
Deliberately uses the RSA backend, not `pack_hard_spheres_3d_dense`: a
production-scale tomogram (hundreds of voxels per axis) draws candidate
pools far too large for the force-biased backend's per-iteration Python-
loop cost to stay practical (verified to run into the hours at that scale)
-- RSA's own ~28-41% ceiling, reached in seconds, is the actual target
here, not `pack_hard_spheres_3d_dense`'s higher-but-impractical one.

Independent of, and not integrated with, the older CTS-derived generators
(``cryotomosim.py``, ``_cts_membrane.py``, ``_cts_placement.py``) -- this is
a clean-room second approach, meant to be benchmarked against CTS later,
not merged with it now.

v1 scope: single-instance placement only (no CellPACK-style cluster/bundle
correlated placement -- a separate, later feature). Multiple disjoint
membrane compartments (e.g. several vesicles) are supported for free, since
:func:`.classify_membrane_regions`'s connected-components approach doesn't
special-case a single compartment. Transmembrane placements are NOT given
per-instance voxel labels in `instance_labels` (their density is correctly
present in the volume via `MembraneGenerator.place_transmembrane` itself,
which already exists and is unmodified here; only cytosol/lumen instances
get per-instance labels in v1) -- a documented gap, not an oversight.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import torch
from scipy import ndimage

from ...arrays import clip_insert_bounds
from ...crowding import insert_particles_into_micrograph
from ...pdb import PDB
from ...potential import PotentialBuilder
from ...rotations import build_affine_matrix, random_rotation_matrix, rotate_volume
from ..membrane import MembraneGenerator, TransmembranePlacement
from ..packing import draw_species_pool, estimate_protein_box_size, pack_hard_spheres_3d
from ._regions import classify_membrane_regions

# Same convention as specimen/packing/pdb_packing.py's own
# _INSTANCE_LABEL_REL_THRESHOLD -- kept as an independent copy rather than
# imported, matching this codebase's established per-generator
# zero-cross-coupling convention (see e.g. cryotomosim.py's own docstring).
_INSTANCE_LABEL_REL_THRESHOLD = 0.01


def _insert_instance_labels(
    binarized: torch.Tensor,
    positions: torch.Tensor,
    pixel_size: float,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Stamp per-instance integer labels into a shared label volume --
    identical mechanism to specimen/packing/pdb_packing.py's own helper of
    the same name, duplicated rather than imported (see module docstring)."""
    N, Zp, Yp, Xp = binarized.shape
    device = binarized.device
    Z, Y, X = labels.shape
    positions = positions.to(device)
    positions_int = (positions / pixel_size).round().long()
    cz_center, cy_center, cx_center = Z // 2, Y // 2, X // 2

    for i in range(N):
        cx_index = cx_center + int(positions_int[i, 0].item())
        cy_index = cy_center + int(positions_int[i, 1].item())
        cz_index = cz_center + int(positions_int[i, 2].item())
        bounds = clip_insert_bounds(
            (cz_index, cy_index, cx_index), (Zp, Yp, Xp), (Z, Y, X)
        )
        if bounds is None:
            continue
        dst, src = bounds
        chunk = binarized[i][src]
        labels[dst] = torch.where(chunk > 0, chunk, labels[dst])

    return labels


@dataclass
class TomogramProteinSpec:
    """
    One cytosolic- or lumen-dwelling protein species to densely pack.

    Attributes
    ----------
    pdb_source : str
        PDB ID or local PDB/mmCIF file path.
    location : {"cytosol", "lumen"}, optional
        Which region this species is restricted to -- "cytosol" (outside
        every membrane compartment) or "lumen" (inside an enclosed
        compartment, e.g. a vesicle's interior). Default "cytosol".
    ratio : float, optional
        Relative abundance weight among OTHER specs sharing the same
        `location` (species at different locations are packed
        independently, so ratios don't compare across locations). Default
        1.0.
    """

    pdb_source: str
    location: Literal["cytosol", "lumen"] = "cytosol"
    ratio: float = 1.0


@dataclass
class TomogramPlacement:
    """One placed cytosolic/lumen instance, for ground-truth bookkeeping."""

    species_id: str
    location: str
    position_xyz: torch.Tensor
    rotation_matrix: torch.Tensor
    instance_id: int


class MembraneTomogramGenerator:
    """
    Assemble a tomogram from a pre-configured membrane plus densely packed
    cytosolic and vesicle-lumen protein populations.

    Parameters
    ----------
    membrane_generator : MembraneGenerator
        Already-configured (not yet `.generate()`-called) membrane
        generator -- its own shape/transmembrane_specs/target_shape_zyx/
        v_size parameters are used as-is and not duplicated here.
    protein_specs : list of TomogramProteinSpec
        Cytosolic/lumen species to pack.
    occupancy_fraction : float, optional
        Target packing density (see `pack_hard_spheres_3d`/
        `draw_species_pool`), applied independently per region -- e.g. 0.2
        for `"lumen"` species targets 20% of the LUMEN's own volume, not
        20% of the whole tomogram. Default 0.2.
    gap_angstrom : float, optional
        Minimum clearance between placed spheres' surfaces, and between a
        placed sphere and the membrane shell (same value used for both).
        Default 5.0.
    region_density_threshold : float, optional
        Passed to `classify_membrane_regions`. Default None (that
        function's own default).
    region_max_passes : int, optional
        `max_passes` (and `stall_patience`, set equal to it -- see
        `pack_hard_spheres_3d`'s `sampling_mask` docstring for why a small
        region needs the early-exit heuristic disabled) for cytosol/lumen
        packing. Default 300, higher than `pack_hard_spheres_3d`'s own
        default 200 since a tight region (e.g. a small vesicle lumen) can
        need more attempts before a geometrically valid spot turns up.
    min_transmembrane_spacing_a : float, optional
        Passed to `MembraneGenerator.place_transmembrane`. Default 40.0.
    pdb_cache_dir : str, optional
        Directory for downloaded PDB/mmCIF files. Default "../pdb-data/".
    parameterization : str, optional
        Atomic scattering-factor parameterization for `PotentialBuilder`.
        Default "shtyrov", matching `PotentialBuilder`'s own default.
    seed : int, optional
        Random seed.
    device : str or torch.device, optional
        Device for the packing step. Default "cpu" -- see
        `pack_hard_spheres_3d`'s own docstring on why.
    chunk_size : int, optional
        Instances rotated per batch, per species. Default None (all of a
        species' instances at once).

    Attributes
    ----------
    regions : dict of str to torch.Tensor
        ``{"shell", "lumen", "cytosol"}`` boolean masks, set after
        `generate()` runs (see `classify_membrane_regions`).
    transmembrane_placements : list of TransmembranePlacement
        From `MembraneGenerator.place_transmembrane`, set after
        `generate()` runs.
    placements : list of TomogramPlacement
        Every placed cytosolic/lumen instance, set after `generate()` runs.
    instance_labels : torch.Tensor
        Per-instance integer label volume for cytosol/lumen instances only
        (see module docstring), shape matching `membrane_generator.
        target_shape_zyx`, dtype int32. Set after `generate()` runs.
    """

    def __init__(
        self,
        membrane_generator: MembraneGenerator,
        protein_specs: list[TomogramProteinSpec],
        occupancy_fraction: float = 0.2,
        gap_angstrom: float = 5.0,
        region_density_threshold: float | None = None,
        region_max_passes: int = 300,
        min_transmembrane_spacing_a: float = 40.0,
        pdb_cache_dir: str = "../pdb-data/",
        parameterization: str = "shtyrov",
        seed: int | None = None,
        device: str | torch.device = "cpu",
        chunk_size: int | None = None,
    ):
        if not protein_specs:
            raise ValueError("protein_specs must be non-empty")
        self.membrane_generator = membrane_generator
        self.protein_specs = protein_specs
        self.occupancy_fraction = occupancy_fraction
        self.gap_angstrom = gap_angstrom
        self.region_density_threshold = region_density_threshold
        self.region_max_passes = region_max_passes
        self.min_transmembrane_spacing_a = min_transmembrane_spacing_a
        self.pdb_cache_dir = pdb_cache_dir
        self.parameterization = parameterization
        self.seed = seed
        self.device = device
        self.chunk_size = chunk_size

        self.regions: dict[str, torch.Tensor] | None = None
        self.transmembrane_placements: list[TransmembranePlacement] = []
        self.placements: list[TomogramPlacement] = []
        self.instance_labels: torch.Tensor | None = None

    def generate(self) -> torch.Tensor:
        """
        Run the full pipeline and return the assembled specimen volume.

        Returns
        -------
        torch.Tensor
            Shape `membrane_generator.target_shape_zyx`, dtype float32.
        """
        if self.seed is not None:
            torch.manual_seed(
                self.seed
            )  # random_rotation_matrix has no generator= param

        mgen = self.membrane_generator
        v_size = mgen.v_size
        target_shape = mgen.target_shape_zyx
        box = (
            target_shape[0] * v_size,
            target_shape[1] * v_size,
            target_shape[2] * v_size,
        )

        mgen.generate()
        self.transmembrane_placements = mgen.place_transmembrane(
            min_spacing_a=self.min_transmembrane_spacing_a
        )
        volume = mgen.volume
        assert volume is not None

        self.regions = classify_membrane_regions(volume, self.region_density_threshold)

        instance_labels = torch.zeros(target_shape, dtype=torch.int32)
        self.placements = []
        next_instance_id = 1
        pdb_cache: dict[str, PDB] = {}

        for location in ("cytosol", "lumen"):
            specs_here = [s for s in self.protein_specs if s.location == location]
            if not specs_here:
                continue

            region_mask = self.regions[location]
            region_voxels = int(region_mask.sum())
            if region_voxels == 0:
                warnings.warn(
                    f"MembraneTomogramGenerator: no '{location}' region found "
                    f"(0 voxels) -- {len(specs_here)} species declared for it "
                    "will not be placed. For 'lumen', this means the membrane "
                    "has no enclosed compartment.",
                    stacklevel=2,
                )
                continue
            region_volume_a3 = region_voxels * v_size**3

            pdbs = []
            for spec in specs_here:
                if spec.pdb_source not in pdb_cache:
                    pdb_cache[spec.pdb_source] = PDB(
                        spec.pdb_source, savefolder=self.pdb_cache_dir, verbose=False
                    )
                pdbs.append(pdb_cache[spec.pdb_source])
            species_radii = torch.tensor(
                [float(pdb.max_diameter) / 2.0 for pdb in pdbs]
            )
            species_ratios = torch.tensor([s.ratio for s in specs_here])

            pool_radii, pool_species_idx = draw_species_pool(
                species_radii,
                species_ratios,
                self.occupancy_fraction,
                region_volume_a3,
                seed=self.seed,
            )

            exclusion_field = (
                torch.from_numpy(
                    ndimage.distance_transform_edt(region_mask.cpu().numpy())
                ).to(device=self.device, dtype=torch.float32)
                * v_size
            )

            coords, accepted_idx = pack_hard_spheres_3d(
                pool_radii,
                box,
                gap=self.gap_angstrom,
                seed=self.seed,
                device=self.device,
                exclusion_distance_field=exclusion_field,
                field_v_size=v_size,
                sampling_mask=region_mask,
                max_passes=self.region_max_passes,
                # region-restricted sampling can need many more consecutive
                # misses than a "box is saturated" heuristic expects before
                # finding a geometrically valid spot (see pack_hard_spheres_3d's
                # own sampling_mask docstring) -- exhaust max_passes instead
                # of bailing out early on a run of misses.
                stall_patience=self.region_max_passes,
            )
            accepted_species_idx = pool_species_idx[accepted_idx]

            volume, instance_labels, next_instance_id = self._render_species_pool(
                specs_here,
                pdbs,
                coords,
                accepted_species_idx,
                volume,
                instance_labels,
                next_instance_id,
                location,
                v_size,
            )

        self.instance_labels = instance_labels
        return volume

    def _render_species_pool(
        self,
        specs: list[TomogramProteinSpec],
        pdbs: list[PDB],
        coords: torch.Tensor,
        accepted_species_idx: torch.Tensor,
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        location: str,
        v_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        for species_i, spec in enumerate(specs):
            mask = accepted_species_idx == species_i
            if not bool(mask.any()):
                continue
            pdb = pdbs[species_i]
            n = estimate_protein_box_size(pdb.max_diameter, v_size)
            builder = PotentialBuilder(
                n_xyz=n,
                dx=v_size,
                atomic_numbers=pdb.atomic_numbers,
                progressbars=False,
                parameterization=self.parameterization,
            )
            template = builder.forward(pdb.coordinates, method="analytic")
            label_threshold = _INSTANCE_LABEL_REL_THRESHOLD * float(template.max())

            species_coords = coords[mask]
            n_instances = species_coords.shape[0]
            R = random_rotation_matrix(n_instances)
            if R.dim() == 2:
                R = R.unsqueeze(0)
            theta = build_affine_matrix(R)

            instance_ids = torch.arange(
                next_instance_id, next_instance_id + n_instances, dtype=torch.int32
            )
            next_instance_id += n_instances

            step = self.chunk_size or n_instances
            for start in range(0, n_instances, step):
                end = min(start + step, n_instances)
                rotated = rotate_volume(
                    template, theta[start:end], padding_mode="zeros"
                )
                volume = insert_particles_into_micrograph(
                    rotated,
                    species_coords[start:end],
                    pixel_size=v_size,
                    micrograph=volume,
                )
                binarized = (rotated > label_threshold).to(torch.int32) * instance_ids[
                    start:end
                ].view(-1, 1, 1, 1)
                instance_labels = _insert_instance_labels(
                    binarized,
                    species_coords[start:end],
                    pixel_size=v_size,
                    labels=instance_labels,
                )

            for i in range(n_instances):
                self.placements.append(
                    TomogramPlacement(
                        species_id=spec.pdb_source,
                        location=location,
                        position_xyz=species_coords[i],
                        rotation_matrix=R[i],
                        instance_id=int(instance_ids[i]),
                    )
                )

        return volume, instance_labels, next_instance_id
