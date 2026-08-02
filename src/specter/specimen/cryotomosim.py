"""
CryoTomoSimSpecimenGenerator: a from-scratch, self-contained specimen
generator that replicates CryoTomoSim (CTS)'s own algorithms -- particle
placement, organic (alpha-shape-based) membrane shapes, carbon support
film, gold fiducial beads -- entirely independently of polnet/VTK.

This has NO relationship to ``specimen/cryoet.py``'s polnet-based
``CryoETSpecimenGenerator``: it does not import from it or from
``polnet_bridge.py``, and does not use polnet or VTK anywhere. Its only
connection to the rest of specter is the shared specimen-volume contract
already established by ``specimen/from_volume.py``'s
``load_specimen_volume`` -- produce a plain ``torch.Tensor`` of shape
(Z, Y, X) in scattering-potential units, usable directly by
``TiltSeriesGenerator``/``MicrographGenerator``/``ImageGenerator``.

Two pieces of specter's existing, physically-accurate infrastructure ARE
reused (both plain physics utilities, neither polnet-related):

- ``specter.potential.PotentialBuilder`` -- every real protein species'
  density is built from its actual PDB atomic coordinates via Kirkland/
  Lobato scattering potentials, instead of CTS's own hand-tuned
  Z-plus-fractional-hydrogen atomic table (``helper_pdb2vol.m``'s ``edat``).
- ``specter.ice`` (``RandomIcemaker`` by default, or any ``IceBank``
  instance) -- the ice/solvent layer uses real Kirkland-oxygen-potential-
  convolved amorphous ice, instead of CTS's ``gen_ice.m`` flat-background-
  plus-scattered-point-molecule blend.

Membrane bilayer geometry (``_cts_membrane.MembraneBlobGenerator``) and
carbon film/bead bulk density (``_cts_grid.py``) are new, self-contained
code with no such existing Specter equivalent -- their output magnitude is
calibrated against the actual placed protein densities (see
``_estimate_organic_potential_reference``) rather than an arbitrary
constant, mirroring the same calibration idea already used elsewhere in
specter for exactly this "what should a membrane's raw density value be"
question -- reimplemented locally here rather than imported, to keep this
module's only outside dependencies limited to ``potential.py``/``ice/``/
``pdb.py``/``rotations/``/``arrays.py`` (no import from ``specimen.cryoet``
or ``specimen.polnet_bridge``).

Placement is a HYBRID of two mechanisms, chosen per placement request:

- **Bulk hard-sphere RSA** (``specter.crowding.pack_hard_spheres_3d`` +
  ``insert_particles_into_micrograph``, the same fully-vectorized,
  ~90x-faster-than-one-at-a-time machinery ``specimen/pdb_packing.py``'s
  ``SpherePackingSpecimenGenerator`` uses) handles every membrane instance
  and every ``"any"``-location, ``"single"``-mode protein species together
  in ONE packing pass -- these are a genuine "pack non-overlapping spheres
  into open space" problem, and don't need per-attempt density rendering
  (only accepted instances get rendered, once). Membranes use their own
  true outer bounding-sphere radius (from the actual generated blob
  geometry, not an arbitrary constant); proteins use ``pdb.max_diameter/2``,
  matching ``pdb_packing.py``'s own convention. This is a real accuracy
  tradeoff vs. the old one-at-a-time voxel-overlap test: a bounding sphere
  is more conservative than the true (irregular, rotated) shape, so
  RSA-packed content won't pack QUITE as tightly as exact voxel-overlap
  placement would -- the same kind of documented approximation CTS's own
  MATLAB source makes elsewhere in this port.
- **Sequential reject-and-retry** (``_cts_placement.ParticlePlacer``,
  unchanged) still handles anything that genuinely needs true-shape
  awareness rather than a bounding sphere: ``"membrane"``/``"vesicle"``/
  ``"cytosol"``-flagged proteins (these need the membrane's *actual*
  irregular shell/interior mask and surface normal field -- a bare sphere
  pack has no notion of "inside vs. outside this specific blob's real
  surface"), and ``"cluster"``/``"bundle"``-mode proteins (deliberately
  correlated/clustered placements around a primary, not an independent-
  sphere-packing problem). This pass runs AFTER the RSA pass, against the
  already-RSA-populated volume, so it still correctly avoids overlapping
  whatever RSA placed.

Known limitation of the hybrid scheme: RSA has no notion of the carbon
support film (added directly to the volume before either pass runs), so if
`grid_spec` is set, RSA-packed membranes/proteins could in principle
overlap it -- not an issue for any run without a carbon film. Bead
placement is NOT yet folded into the RSA pool (beads are literally already
spheres, so this would be a natural, low-effort future addition) -- left on
the sequential path for now.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..arrays import clip_insert_bounds
from ..crowding import insert_particles_into_micrograph, pack_hard_spheres_3d
from ..ice import IceBank, RandomIcemaker, blend_ice_into_volume
from ..pdb import PDB
from ..potential import PotentialBuilder
from ..rotations import build_affine_matrix, random_rotation_matrix, rotate_volume
from ._cts_grid import BeadGenerator, CarbonFilmGenerator
from ._cts_membrane import BlobMembraneInstance, MembraneBlobGenerator
from ._cts_placement import ParticlePlacer, ParticleSpec, PlacedInstance, PlacementMode

# specter.potential.compute_supersampling_parameters's own default atomic
# kernel width (potential.py: width_atom=5.0) -- each atom's own potential
# is evaluated over a +/-2.5 A box around it, so a molecule's own bounding
# box needs at least this much margin beyond its outermost atom or that
# atom's kernel gets truncated by convolution. Same reasoning (and value)
# as ``specimen/polnet_bridge.py``'s ``ATOM_KERNEL_HALF_WIDTH_A`` -- kept
# as an independent local constant rather than importing from that module,
# per this generator's zero-polnet-coupling design.
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


def _estimate_organic_potential_reference(templates: list[torch.Tensor]) -> float:
    """
    Estimate a typical scattering-potential magnitude for ordinary,
    densely-packed organic matter (protein interior), from the actual
    high-res protein templates already built for this specimen.

    Per species: mean potential over voxels at or above half that
    species' own peak value. Averaged across species if more than one is
    present. A lipid bilayer is also ordinary organic matter (mostly
    C/H/N/O, packed at broadly similar density to protein interior), so
    this is a physically-motivated approximation for the membrane's
    baseline voxel value -- not exact, but far better grounded than an
    arbitrary constant.

    Returns 1.0 if no templates are available (e.g. a membrane-only
    specimen with no proteins placed at all) -- an explicit, documented
    fallback, not a claim of physical accuracy in that case.
    """
    if not templates:
        return 1.0
    per_species = []
    for template in templates:
        peak = template.max()
        if peak <= 0:
            continue
        occupied = template[template >= 0.5 * peak]
        if occupied.numel() > 0:
            per_species.append(occupied.mean().item())
    return float(np.mean(per_species)) if per_species else 1.0


def _bounding_radius_voxels(density: torch.Tensor) -> float:
    """
    Conservative bounding-sphere radius (voxels) of `density`'s own nonzero
    content, measured from the density array's geometric center (matching
    where ``rotate_volume``/``clip_insert_bounds`` treat the array's own
    center as "the point that lands at a given placement position" --
    rotation is distance-preserving from that center, so this radius,
    computed once on the UNROTATED template, stays a valid bound regardless
    of whatever rotation gets applied later).

    Used only for membrane instances here -- real protein species use
    ``pdb.max_diameter / 2`` instead (matching
    ``specimen/pdb_packing.py``'s convention), since membranes have no PDB
    structure to measure a diameter from.
    """
    coords = torch.nonzero(density > 0, as_tuple=False).float()
    if coords.shape[0] == 0:
        return 0.0
    center = torch.tensor(density.shape, dtype=torch.float32) / 2.0
    return float((coords - center).norm(dim=1).max())


def _physical_xyz_to_voxel_zyx(
    pos_xyz: torch.Tensor, target_shape: tuple[int, ...], v_size: float
) -> torch.Tensor:
    """
    Convert a box-centered physical (x, y, z) position (Angstrom, the
    convention ``specter.crowding.pack_hard_spheres_3d``/
    ``insert_particles_into_micrograph`` use) to a corner-relative voxel
    index (z, y, x) (the convention ``_cts_placement.PlacedInstance``/
    ``ParticlePlacer`` use) -- the exact same rounding/center convention
    ``insert_particles_into_micrograph`` itself uses internally, so a
    recorded ``PlacedInstance.center_zyx`` always matches where the density
    was actually rendered.
    """
    nz, ny, nx = target_shape
    center_xyz = torch.tensor([nx // 2, ny // 2, nz // 2], dtype=torch.float32)
    vox_xyz = (pos_xyz / v_size).round() + center_xyz
    return vox_xyz.flip(0)


def _stamp_max_merge(
    volume: torch.Tensor, local_density: torch.Tensor, center_zyx: torch.Tensor
) -> None:
    """Max-merge `local_density` into `volume` at `center_zyx` (voxel
    index) -- used for RSA-accepted membranes, which are few enough per
    specimen that a plain per-instance loop (rather than
    ``insert_particles_into_micrograph``'s batched-additive approach, used
    for the much-more-numerous proteins instead) is simple and fast enough,
    and avoids double-counting density at any grazing boundary voxel."""
    bounds = clip_insert_bounds(center_zyx.tolist(), local_density.shape, volume.shape)
    if bounds is None:
        return
    dst, src = bounds
    volume[dst] = torch.maximum(volume[dst], local_density[src])


def _pack_membranes_and_free_proteins(
    membrane_specs: list["MembraneSpec"],
    rsa_protein_specs: list["ProteinSpec"],
    protein_templates: dict[str, torch.Tensor],
    protein_pdbs: dict[str, PDB],
    reference: float,
    volume: torch.Tensor,
    v_size: float,
    gap_angstrom: float,
    chunk_size: int | None,
    seed: int | None,
) -> tuple[
    list[PlacedInstance], dict[str, BlobMembraneInstance], list[BlobMembraneInstance]
]:
    """
    Pack every membrane instance and every RSA-eligible ("any"-location,
    "single"-mode) protein species instance into `volume` in ONE hard-
    sphere RSA pass (``specter.crowding.pack_hard_spheres_3d``), then
    render only the accepted instances -- membranes individually (their
    real bilayer geometry, not a sphere -- the sphere was only a collision-
    detection proxy), proteins batched per species via
    ``insert_particles_into_micrograph`` (mirrors
    ``specimen/pdb_packing.py``'s own per-species chunked rendering loop).

    Mutates `volume` in place (and also returns it implicitly via the
    caller's own reference, same object).

    Returns
    -------
    placements : list of PlacedInstance
        One entry per accepted membrane or protein instance.
    membrane_local_by_id : dict
        Maps each accepted membrane's `species_id` (e.g. "membrane_0_3") to
        its `BlobMembraneInstance`, for `_build_membrane_location_fields`.
    membrane_instances : list of BlobMembraneInstance
        Every accepted membrane instance, for `self.membrane_instances`
        bookkeeping.
    """
    target_shape = tuple(volume.shape)
    box = (
        target_shape[0] * v_size,
        target_shape[1] * v_size,
        target_shape[2] * v_size,
    )

    all_radii: list[float] = []
    # ("membrane", index into membrane_pool) or ("protein", pdb_source)
    candidate_kind: list[tuple[str, object]] = []
    membrane_pool: list[tuple[str, BlobMembraneInstance]] = []

    if membrane_specs:
        mem_gen = MembraneBlobGenerator(v_size=v_size, seed=seed)
        for i, mspec in enumerate(membrane_specs):
            for j in range(mspec.count):
                inst = mem_gen.generate(
                    size=mspec.size,
                    roughness=mspec.roughness,
                    thickness=mspec.thickness,
                )
                sid = f"membrane_{i}_{j}"
                idx = len(membrane_pool)
                membrane_pool.append((sid, inst))
                radius_a = _bounding_radius_voxels(inst.density) * v_size
                all_radii.append(radius_a)
                candidate_kind.append(("membrane", idx))

    for spec in rsa_protein_specs:
        pdb = protein_pdbs[spec.pdb_source]
        radius_a = float(pdb.max_diameter) / 2.0
        for _ in range(spec.max_count):
            all_radii.append(radius_a)
            candidate_kind.append(("protein", spec.pdb_source))

    placements: list[PlacedInstance] = []
    membrane_local_by_id: dict[str, BlobMembraneInstance] = {}
    membrane_instances: list[BlobMembraneInstance] = []

    if not all_radii:
        return placements, membrane_local_by_id, membrane_instances

    radii_t = torch.tensor(all_radii)
    coords, accepted_idx = pack_hard_spheres_3d(
        radii_t, box, gap=gap_angstrom, seed=seed
    )
    accepted_kinds = [candidate_kind[i] for i in accepted_idx.tolist()]

    # Render accepted membranes individually, real bilayer geometry.
    for coord, (kind, ref) in zip(coords, accepted_kinds):
        if kind != "membrane":
            continue
        sid, inst = membrane_pool[ref]
        R = random_rotation_matrix(1)
        theta = build_affine_matrix(R.unsqueeze(0))
        rotated = rotate_volume(inst.density * reference, theta, padding_mode="zeros")[
            0
        ]
        center_zyx = _physical_xyz_to_voxel_zyx(coord, target_shape, v_size)
        _stamp_max_merge(volume, rotated, center_zyx)
        placements.append(
            PlacedInstance(species_id=sid, center_zyx=center_zyx, rotation_matrix=R)
        )
        membrane_local_by_id[sid] = inst
        membrane_instances.append(inst)

    # Render accepted proteins, batched per species.
    protein_coords_by_species: dict[str, list[torch.Tensor]] = {}
    for coord, (kind, ref) in zip(coords, accepted_kinds):
        if kind == "protein":
            protein_coords_by_species.setdefault(ref, []).append(coord)

    for species_id, coord_list in protein_coords_by_species.items():
        template = protein_templates[species_id]
        species_coords = torch.stack(coord_list)
        n_instances = species_coords.shape[0]
        R = random_rotation_matrix(n_instances)
        if R.dim() == 2:
            R = R.unsqueeze(0)
        theta = build_affine_matrix(R)

        step = chunk_size or n_instances
        for start in range(0, n_instances, step):
            end = min(start + step, n_instances)
            rotated = rotate_volume(template, theta[start:end], padding_mode="zeros")
            insert_particles_into_micrograph(
                rotated,
                species_coords[start:end],
                pixel_size=v_size,
                micrograph=volume,
            )

        for i in range(n_instances):
            center_zyx = _physical_xyz_to_voxel_zyx(
                species_coords[i], target_shape, v_size
            )
            placements.append(
                PlacedInstance(
                    species_id=species_id, center_zyx=center_zyx, rotation_matrix=R[i]
                )
            )

    return placements, membrane_local_by_id, membrane_instances


def _build_membrane_location_fields(
    placements: list,
    membrane_local_by_id: dict[str, BlobMembraneInstance],
    target_shape: tuple[int, int, int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Build full-volume ``"membrane"``/``"vesicle"``/``"cytosol"`` location
    (candidate-SAMPLING) masks, a ``"membrane"`` outward-normal field, and
    a separate ``"membrane"`` overlap-IGNORE mask, from the already-placed
    membrane instances' local geometry (``BlobMembraneInstance.shell_mask``/
    ``skeleton_mask``/``interior_mask``/``normal_field``, see
    ``_cts_membrane.py``), rotated and positioned exactly as each
    membrane's own density was.

    The `"membrane"` entry is deliberately split across TWO different
    return dicts, built from TWO different `BlobMembraneInstance` masks:

    - `location_masks["membrane"]` (candidate SAMPLING region) is built
      from `skeleton_mask` -- the thin mid-thickness ridge -- so a
      membrane-embedded protein's insertion depth lands consistently at
      mid-bilayer, not uniformly random anywhere across the full
      thickness.
    - `ignore_masks["membrane"]` (overlap-IGNORE region) is built from
      `shell_mask` -- the FULL bilayer thickness footprint -- so the
      overlap test doesn't wrongly reject a candidate whose rotated
      density genuinely spans the whole membrane thickness once centered
      on the thin skeleton.

    These must NOT be swapped or merged: sampling from the full shell
    would reintroduce random insertion depth; ignoring overlap against
    only the thin skeleton would leave most of the real bilayer thickness
    still treated as a blocking obstacle, collapsing membrane-embedded
    placement almost to zero (the candidate's own rotated footprint
    genuinely overlaps that material once centered on the skeleton).
    `"vesicle"`/`"cytosol"` are unaffected by this split -- they still use
    `interior_mask`/the full shell as before (cytosol correctly excludes
    the membrane's ENTIRE material footprint, not just the thin ridge).

    Parameters
    ----------
    placements : list of PlacedInstance
        `ParticlePlacer.placements` after membranes (and only membranes)
        have been placed -- entries whose `species_id` isn't in
        `membrane_local_by_id` are ignored (defensive; in practice this is
        called before any non-membrane species are placed).
    membrane_local_by_id : dict
        Maps each membrane `ParticleSpec.species_id` to the
        `BlobMembraneInstance` it was built from.
    target_shape : tuple of int
        The specimen volume's shape (Z, Y, X).

    Returns
    -------
    location_masks : dict
        Keys ``"membrane"`` (skeleton-based), ``"vesicle"``, ``"cytosol"``
        -> bool tensors, shape `target_shape`. If no membranes were
        placed, `"membrane"`/`"vesicle"` are all-False and `"cytosol"` is
        all-True (equivalent to unrestricted "any" placement).
    normal_fields : dict
        Key ``"membrane"`` -> float tensor, shape (3,) + `target_shape`,
        physical (x, y, z) component order (see
        ``_cts_membrane._compute_geometry_fields``). Zero away from any
        placed membrane's shell (never queried there by
        ``ParticlePlacer``, since candidate voxels are already restricted
        to the `"membrane"` mask).
    ignore_masks : dict
        Key ``"membrane"`` (full-shell-based) -> bool tensor, shape
        `target_shape`, for ``ParticlePlacer``/``HierarchicalParticlePlacer``'s
        overlap-ignore mechanism.
    """
    membrane_shell_mask = torch.zeros(target_shape, dtype=torch.bool)
    membrane_skeleton_mask = torch.zeros(target_shape, dtype=torch.bool)
    vesicle_mask = torch.zeros(target_shape, dtype=torch.bool)
    normal_field_full = torch.zeros((3, *target_shape), dtype=torch.float32)

    for placed in placements:
        inst = membrane_local_by_id.get(placed.species_id)
        if inst is None:
            continue
        R = placed.rotation_matrix
        center = placed.center_zyx
        theta = build_affine_matrix(R.unsqueeze(0))

        rotated_shell = (
            rotate_volume(inst.shell_mask.float(), theta, padding_mode="zeros")[0] > 0.5
        )
        rotated_skeleton = (
            rotate_volume(inst.skeleton_mask.float(), theta, padding_mode="zeros")[0]
            > 0.5
        )
        rotated_interior = (
            rotate_volume(inst.interior_mask.float(), theta, padding_mode="zeros")[0]
            > 0.5
        )
        bounds = clip_insert_bounds(
            center.tolist(), rotated_shell.shape, membrane_shell_mask.shape
        )
        if bounds is not None:
            dst, src = bounds
            membrane_shell_mask[dst] |= rotated_shell[src]
            membrane_skeleton_mask[dst] |= rotated_skeleton[src]
            vesicle_mask[dst] |= rotated_interior[src]

        resampled_normals = torch.stack(
            [
                rotate_volume(inst.normal_field[c], theta, padding_mode="zeros")[0]
                for c in range(3)
            ],
            dim=0,
        )
        rotated_normals = torch.einsum("ij,jzyx->izyx", R, resampled_normals)
        bounds3 = clip_insert_bounds(
            center.tolist(), rotated_normals.shape[1:], normal_field_full.shape[1:]
        )
        if bounds3 is not None:
            dst3, src3 = bounds3
            normal_field_full[(slice(None), *dst3)] = rotated_normals[
                (slice(None), *src3)
            ]

    cytosol_mask = ~(membrane_shell_mask | vesicle_mask)
    location_masks = {
        "membrane": membrane_skeleton_mask,
        "vesicle": vesicle_mask,
        "cytosol": cytosol_mask,
    }
    normal_fields = {"membrane": normal_field_full}
    ignore_masks = {"membrane": membrane_shell_mask}
    return location_masks, normal_fields, ignore_masks


@dataclass
class ProteinSpec:
    """
    One real protein species to place, built from an actual PDB structure.

    Attributes
    ----------
    pdb_source : str
        PDB ID (fetched from RCSB) or local file path, passed directly to
        ``specter.pdb.PDB``.
    max_count : int
        Maximum number of copies to attempt to place.
    location : {"any", "membrane", "vesicle", "cytosol"}, optional
        See ``ParticleSpec.location``. Default "any". A `"membrane"`
        location assumes `pdb_source`'s own coordinate frame is already
        oriented with its membrane-insertion axis along local +Z (see
        ``_cts_placement``'s module docstring) -- most ordinary PDB
        structures are NOT pre-oriented this way, so `"membrane"` is
        really only meaningful for a structure you've deliberately
        prepared that way.
    mode : {"single", "cluster", "bundle"}, optional
        See ``ParticleSpec.mode``. Default "single".
    cluster_size, bundle_size, bundle_length : optional
        Forwarded to ``ParticleSpec`` when `mode` is `"cluster"`/`"bundle"`.
    """

    pdb_source: str
    max_count: int
    location: str = "any"
    mode: PlacementMode = "single"
    cluster_size: int = 8
    bundle_size: int = 8
    bundle_length: float | None = None


@dataclass
class MembraneSpec:
    """One organic-blob membrane population to generate and place.

    Attributes
    ----------
    count : int
        Number of independent vesicles to generate and place.
    size, roughness, thickness : float
        Forwarded to ``MembraneBlobGenerator.generate``.
    """

    count: int = 1
    size: float = 300.0
    roughness: float = 0.7
    thickness: float = 40.0


@dataclass
class BeadSpec:
    """Gold fiducial beads to place, one population per radius.

    Attributes
    ----------
    radii : list of float
        One bead radius (Angstrom) per requested bead population.
    count_per_radius : int, optional
        Number of copies to place per radius. Default 1.
    """

    radii: list[float]
    count_per_radius: int = 1


@dataclass
class GridSpec:
    """Carbon support film, forwarded to ``CarbonFilmGenerator.generate``."""

    thickness: float = 150.0
    hole_radius: float = 400.0


class CryoTomoSimSpecimenGenerator:
    """
    Self-contained CTS-replica specimen generator (no polnet/VTK).

    Parameters
    ----------
    protein_specs : list of ProteinSpec
        Real protein species to place (built via ``PotentialBuilder``).
    membrane_specs : list of MembraneSpec, optional
        Organic-blob membrane populations to generate and place.
    bead_spec : BeadSpec, optional
        Gold fiducial beads to place.
    grid_spec : GridSpec, optional
        Carbon support film to lay down first.
    ice_opacity : float, optional
        Scales the ice/solvent layer's contribution; 0 disables it.
        Default 1.0.
    icemaker : RandomIcemaker or IceBank, optional
        Pre-constructed ice source; if not given, a fresh
        ``RandomIcemaker`` sized to `target_shape`/`v_size` is used.
    ice_method : str, optional
        Forwarded to ``blend_ice_into_volume``'s own ``method``: ``'convolve'``
        (default, unchanged behaviour) or ``'analytic'`` (occupancy-aware,
        in-place per-tile analytic insertion -- see
        ``IceBank.insert_analytic_ice``; only meaningful when `icemaker` is
        an ``IceBank``, since a default/passed ``RandomIcemaker`` ignores
        `method` either way).
    target_shape : tuple of int
        Output volume shape (Z, Y, X), voxels.
    v_size : float
        Voxel size, Angstrom.
    pdb_cache_dir : str, optional
        Directory for downloaded PDB/mmCIF files. Default "../pdb-data/".
    parameterization : str, optional
        Atomic scattering-factor parameterization for `PotentialBuilder`
        ("kirkland" or "lobato"). Default "kirkland".
    density_cutoff : float, optional
        Shared occupancy cutoff across all particle/bead placement,
        matching CTS's own default (``param_model.m``'s `density`).
        Default 0.4.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        protein_specs: list[ProteinSpec],
        membrane_specs: list[MembraneSpec] | None = None,
        bead_spec: BeadSpec | None = None,
        grid_spec: GridSpec | None = None,
        ice_opacity: float = 1.0,
        icemaker: RandomIcemaker | IceBank | None = None,
        ice_method: str = "convolve",
        target_shape: tuple[int, int, int] = (128, 256, 256),
        v_size: float = 5.0,
        pdb_cache_dir: str = "../pdb-data/",
        parameterization: str = "kirkland",
        density_cutoff: float = 0.4,
        seed: int | None = None,
    ):
        self.protein_specs = protein_specs
        self.membrane_specs = membrane_specs or []
        self.bead_spec = bead_spec
        self.grid_spec = grid_spec
        self.ice_opacity = ice_opacity
        self.icemaker = icemaker
        self.ice_method = ice_method
        self.target_shape = target_shape
        self.v_size = v_size
        self.pdb_cache_dir = pdb_cache_dir
        self.parameterization = parameterization
        self.density_cutoff = density_cutoff
        self.seed = seed

        self.placements: list = []
        self.membrane_instances: list = []
        self._generated = False

    def generate(self) -> torch.Tensor:
        """
        Run the full pipeline and return the assembled specimen volume.

        Returns
        -------
        torch.Tensor
            Shape `target_shape`, dtype float32.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np_rng_seed = self.seed
        else:
            np_rng_seed = None

        volume = torch.zeros(self.target_shape, dtype=torch.float32)

        if self.grid_spec is not None:
            carbon_gen = CarbonFilmGenerator(v_size=self.v_size, seed=np_rng_seed)
            film = carbon_gen.generate(
                target_shape=self.target_shape,
                thickness=self.grid_spec.thickness,
                hole_radius=self.grid_spec.hole_radius,
            )
            volume += film.density

        # Build every real protein species' density once via PotentialBuilder,
        # regardless of which placement path (RSA or sequential) it ends up
        # on -- _estimate_organic_potential_reference needs every species'
        # template either way.
        protein_templates: dict[str, torch.Tensor] = {}
        protein_pdbs: dict[str, PDB] = {}
        for spec in self.protein_specs:
            pdb = PDB(spec.pdb_source, savefolder=self.pdb_cache_dir, verbose=False)
            n = estimate_protein_box_size(pdb.max_diameter, self.v_size)
            builder = PotentialBuilder(
                n_xyz=n,
                dx=self.v_size,
                atomic_numbers=pdb.atomic_numbers,
                progressbars=False,
                parameterization=self.parameterization,
            )
            protein_templates[spec.pdb_source] = builder.forward(
                pdb.coordinates, method="3d"
            )
            protein_pdbs[spec.pdb_source] = pdb

        reference = _estimate_organic_potential_reference(
            list(protein_templates.values())
        )

        # Split protein species: "any"-location, "single"-mode instances
        # are a genuine sphere-packing problem (no shape/location awareness
        # needed) and go through the fast RSA pass below alongside every
        # membrane instance. Everything else -- "membrane"/"vesicle"/
        # "cytosol"-flagged (needs the membrane's real shape) or "cluster"/
        # "bundle"-mode (deliberately correlated placements, not
        # independent packing) -- stays on the sequential ParticlePlacer
        # path, run afterwards against the already-RSA-populated volume.
        rsa_protein_specs = [
            spec
            for spec in self.protein_specs
            if spec.location == "any" and spec.mode == "single"
        ]
        # Identity-based (not ==) so two specs with coincidentally-equal
        # field values (e.g. the same pdb_source added twice) don't get
        # conflated -- ProteinSpec is a plain (non-frozen) dataclass, so
        # `==` compares field values, not object identity.
        rsa_spec_ids = {id(spec) for spec in rsa_protein_specs}
        sequential_protein_specs = [
            spec for spec in self.protein_specs if id(spec) not in rsa_spec_ids
        ]

        rsa_placements, membrane_local_by_id, membrane_instances = (
            _pack_membranes_and_free_proteins(
                membrane_specs=self.membrane_specs,
                rsa_protein_specs=rsa_protein_specs,
                protein_templates=protein_templates,
                protein_pdbs=protein_pdbs,
                reference=reference,
                volume=volume,
                v_size=self.v_size,
                gap_angstrom=5.0,
                chunk_size=256,
                seed=np_rng_seed,
            )
        )
        self.membrane_instances.extend(membrane_instances)

        placer = ParticlePlacer(volume=volume, density_cutoff=self.density_cutoff)
        placer.placements.extend(rsa_placements)

        location_masks, normal_fields, ignore_masks = _build_membrane_location_fields(
            rsa_placements, membrane_local_by_id, self.target_shape
        )

        sequential_specs_for_placer = [
            ParticleSpec(
                species_id=spec.pdb_source,
                density=protein_templates[spec.pdb_source],
                max_count=spec.max_count,
                location=spec.location,
                mode=spec.mode,
                cluster_size=spec.cluster_size,
                bundle_size=spec.bundle_size,
                bundle_length=spec.bundle_length,
            )
            for spec in sequential_protein_specs
        ]
        placer.run(
            sequential_specs_for_placer,
            location_masks=location_masks,
            normal_fields=normal_fields,
            ignore_masks=ignore_masks,
        )

        if self.bead_spec is not None:
            # TODO: beads are literally already spheres and could be folded
            # into the RSA pool above like membranes/proteins are -- left on
            # the slower sequential path for now (not needed for the runs
            # this generator has been used for so far).
            bead_gen = BeadGenerator(v_size=self.v_size)
            bead_specs_for_placer = []
            for radius in self.bead_spec.radii:
                bead = bead_gen.generate(radius=radius)
                bead_specs_for_placer.append(
                    ParticleSpec(
                        species_id=f"bead_{radius:g}",
                        density=bead.density,
                        max_count=self.bead_spec.count_per_radius,
                        location="any",
                    )
                )
            placer.run(bead_specs_for_placer)

        self.placements = placer.placements

        if self.ice_opacity > 0:
            icemaker = self.icemaker
            if icemaker is None:
                nz, ny, nx = self.target_shape
                if ny != nx:
                    raise ValueError(
                        "default RandomIcemaker requires a square XY "
                        f"footprint; got target_shape={self.target_shape}. "
                        "Pass a pre-built `icemaker` for non-square volumes."
                    )
                icemaker = RandomIcemaker(
                    dx=self.v_size, n=nx, nz=nz, progressbars=False
                )
            pre_ice = volume.clone()
            iced = blend_ice_into_volume(
                volume.unsqueeze(0),
                icemaker,
                pixel_size=self.v_size,
                method=self.ice_method,
            )[0]
            # blend_ice_into_volume adds ice at full strength; scale only
            # the ice-only delta by ice_opacity, not the whole volume.
            volume = pre_ice + (iced - pre_ice) * self.ice_opacity

        self._generated = True
        return volume

    def export_picks(
        self,
        output_dir: str | Path,
        annotation_version: str = "1.0",
        oriented: bool = True,
    ) -> dict[str, Path]:
        """
        Write one copick/CryoET-Data-Portal-style ``.ndjson`` pick file per
        placed species, matching ``CryoETSpecimenGenerator.export_picks``'s
        schema exactly (one JSON object per line:
        ``{"type": "point"|"orientedPoint", "location": {"x", "y", "z"}[,
        "xyz_rotation_matrix"]}``) so downstream pick consumers see the same
        format regardless of which generator produced the specimen.

        Coordinates are physical Angstrom (x, y, z), converted from each
        placement's voxel-index center via ``center_zyx.flip(0) * v_size``
        -- the same voxel-index-(z,y,x)-equals-physical-(x,y,z)[::-1]/v_size
        convention used throughout this generator (see e.g.
        ``_build_membrane_location_fields``).

        Must be called after ``generate()``.

        Parameters
        ----------
        output_dir : str or Path
            Directory to write the ``.ndjson`` files into.
        annotation_version : str, optional
            Used only in the output filename, matching the portal's
            ``"{name}-{version}_{point_type}.ndjson"`` convention. Default
            "1.0".
        oriented : bool, optional
            If True, non-membrane picks are written as ``"orientedPoint"``
            with each instance's rotation matrix included (already a plain
            3x3 matrix on ``PlacedInstance`` -- no quaternion round-trip
            needed, unlike ``CryoETSpecimenGenerator``'s stored
            ``quat_wxyz``); if False, as plain ``"point"``. Membrane picks
            are always written as plain ``"point"`` regardless of this
            flag -- a membrane's own rotation isn't a meaningful
            per-instance orientation the way a protein's is (matches
            ``CryoETSpecimenGenerator``'s own membrane handling).

        Returns
        -------
        dict[str, Path]
            Mapping of species name (or "membrane") to written file path.
        """
        if not self._generated:
            raise RuntimeError("call generate() before export_picks()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        # Membrane placements (species_id like "membrane_0_0", one per
        # generated vesicle instance -- see generate()) are collapsed into
        # a single shared "membrane" group, matching
        # CryoETSpecimenGenerator.export_picks's own membrane handling
        # (one file for all membranes, not one per instance).
        by_name: dict[str, list] = {}
        membrane_placements = []
        for inst in self.placements:
            if inst.species_id.startswith("membrane_"):
                membrane_placements.append(inst)
            else:
                # species_id for real proteins is `ProteinSpec.pdb_source`,
                # which may be a full file path rather than a clean short
                # code -- sanitize into a filename-safe stem. Bead
                # species_id ("bead_60", from cryotomosim.py's own
                # f"bead_{radius:g}") already IS a plain stem, so this is a
                # no-op for those.
                name = Path(inst.species_id).stem
                by_name.setdefault(name, []).append(inst)

        point_type = "orientedPoint" if oriented else "point"
        for name, insts in by_name.items():
            path = (
                output_dir / f"{name}-{annotation_version}_{point_type.lower()}.ndjson"
            )
            with open(path, "w") as f:
                for inst in insts:
                    x, y, z = (
                        float(v) for v in (inst.center_zyx.flip(0) * self.v_size)
                    )
                    row: dict = {
                        "type": point_type,
                        "location": {"x": x, "y": y, "z": z},
                    }
                    if oriented:
                        row["xyz_rotation_matrix"] = (
                            inst.rotation_matrix.numpy().tolist()
                        )
                    f.write(json.dumps(row) + "\n")
            written[name] = path

        if membrane_placements:
            path = output_dir / f"membrane-{annotation_version}_point.ndjson"
            with open(path, "w") as f:
                for inst in membrane_placements:
                    x, y, z = (
                        float(v) for v in (inst.center_zyx.flip(0) * self.v_size)
                    )
                    f.write(
                        json.dumps(
                            {"type": "point", "location": {"x": x, "y": y, "z": z}}
                        )
                        + "\n"
                    )
            written["membrane"] = path

        return written
