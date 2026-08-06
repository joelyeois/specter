"""
SpherePackingSpecimenGenerator: builds a specimen volume by packing several
PDB-derived protein species -- each at its own real physical size, from
``PDB.max_diameter`` -- via hard-sphere packing (:mod:`.algorithms`), then
rendering each accepted instance's real high-resolution scattering
potential (``PotentialBuilder``) at its packed position and a random
rotation.

A third, independent alternative alongside specter's other two specimen
generators -- ``specimen/cryoet.py``'s ``CryoETSpecimenGenerator`` (polnet
SAWLC placement + membranes) and ``specimen/cryotomosim.py``'s
``CryoTomoSimSpecimenGenerator`` (a from-scratch CTS port: reject-and-retry
Monte Carlo with direct density-overlap testing). This one approximates
each species as a bounding sphere for placement (not the true molecular
shape, unlike the other two -- a real cost in packing fidelity for oddly
shaped molecules), in exchange for speed and density.

Species split into two priority groups, packed in two sequential stages:

- `target_specs` -- placed FIRST, each at an exact `SphereProteinSpec.
  n_copies` count, via :func:`.algorithms.pack_hard_spheres_3d`. These are
  the ground-truth particles you actually want annotated -- exact,
  reproducible counts matter more here than crowding density.
- `filler_specs` -- placed SECOND, drawn with equal attempt-weight across
  species (no per-species knob -- see `SphereProteinSpec`'s own docstring
  for why) and budget-driven by `filler_occupancy_fraction` (default a
  generously high ceiling, so the backend naturally jams out rather than
  needing a hand-tuned target -- see that parameter), avoiding every
  already-placed target via an
  `exclusion_distance_field` built from their positions/radii
  (`_build_sphere_exclusion_field`). `packing_method="rsa"` (default)
  resolves many candidates' accept/reject decisions per pass instead of
  one at a time, roughly 90x faster than a naive one-candidate-at-a-time
  loop at a few thousand instances, and is the only backend that can honor
  an exclusion field; `packing_method="dense"`
  (:func:`.algorithms.pack_hard_spheres_3d_dense`) trades that for
  substantially higher occupancy via periodic force-biased relaxation, but
  has no obstacle-avoidance mechanism -- it's only usable when
  `target_specs` is empty (see the error raised otherwise). See
  ``dev/packing_algorithms.py`` for the full algorithm comparison this
  pair was promoted from.

No membranes, ice, or carbon film here -- just densely packed protein
species; layer ice on top separately (e.g.
``specter.ice.blend_ice_into_volume``) if needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy import ndimage

from ...arrays import clip_insert_bounds
from ...crowding import insert_particles_into_micrograph
from ...pdb import DEFAULT_PDB_SAVEFOLDER, PDB
from ...potential import PotentialBuilder
from ...progress import TqdmProgress, status
from ...rotations import build_affine_matrix, random_rotation_matrix, rotate_volume
from .algorithms import (
    draw_species_pool,
    pack_hard_spheres_3d,
    pack_hard_spheres_3d_dense,
)

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

# Binarization threshold for per-instance label masks, as a fraction of
# that species' own unrotated template peak. `rotate_volume`'s trilinear
# resampling leaves tiny non-zero "bleed" density just outside a shape's
# true edge (see specimen/packing/tetris.py's docstring on the same
# issue) -- thresholding relative to each species' own peak (rather than a
# fixed absolute value) keeps the bar equally strict across species whose
# absolute potential scale differs with atomic composition.
_INSTANCE_LABEL_REL_THRESHOLD = 0.01


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


def _insert_instance_labels(
    binarized: torch.Tensor,
    positions: torch.Tensor,
    pixel_size: float,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Stamp per-instance integer labels into a shared label volume. Same
    clip-insert bookkeeping as `insert_particles_into_micrograph`, but
    overwriting each destination voxel with the instance id (not summing
    density) -- `labels[dst]` only changes where `binarized[src] > 0`, so
    already-labeled voxels from earlier instances are preserved (placement
    already keeps instances non-overlapping given `gap_angstrom`, so this
    should be a formality, not a real conflict resolution)."""
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


def _build_sphere_exclusion_field(
    coords_xyz: torch.Tensor,
    radii: torch.Tensor,
    target_shape: tuple[int, int, int],
    v_size: float,
) -> tuple[torch.Tensor, float]:
    """
    Rasterize already-placed spheres into a boolean occupied grid and
    return its Euclidean distance transform, ready to pass straight through
    as `pack_hard_spheres_3d`'s `exclusion_distance_field` (so a later
    packing stage can avoid overlapping these fixed spheres). Only the raw
    sphere volume is marked occupied -- `gap` clearance is applied later, at
    comparison time, by the caller's own `gap` argument to that function.

    Parameters
    ----------
    coords_xyz : torch.Tensor, shape (N, 3)
        Sphere centers (x, y, z), Angstrom, box-centered -- same convention
        `pack_hard_spheres_3d` returns.
    radii : torch.Tensor, shape (N,)
        Sphere radius per instance, Angstrom.
    target_shape : tuple of int
        (Z, Y, X) voxels.
    v_size : float
        Grid spacing, Angstrom -- shared by the occupied grid and the
        returned distance field.

    Returns
    -------
    field : torch.Tensor, shape target_shape
        Distance (Angstrom) from each voxel to the nearest occupied voxel.
    field_v_size : float
        Equal to `v_size` -- returned alongside the field only to match
        `pack_hard_spheres_3d`'s `exclusion_distance_field`/`field_v_size`
        calling convention.
    """
    Z, Y, X = target_shape
    occupied = np.zeros((Z, Y, X), dtype=bool)
    cz, cy, cx = Z / 2.0, Y / 2.0, X / 2.0

    coords_np = coords_xyz.numpy()
    radii_np = radii.numpy()
    for (x, y, z), r in zip(coords_np, radii_np):
        vz, vy, vx = cz + z / v_size, cy + y / v_size, cx + x / v_size
        r_vox = r / v_size
        z0, z1 = max(0, int(np.floor(vz - r_vox))), min(Z, int(np.ceil(vz + r_vox)) + 1)
        y0, y1 = max(0, int(np.floor(vy - r_vox))), min(Y, int(np.ceil(vy + r_vox)) + 1)
        x0, x1 = max(0, int(np.floor(vx - r_vox))), min(X, int(np.ceil(vx + r_vox)) + 1)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue
        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
        dist2 = (zz - vz) ** 2 + (yy - vy) ** 2 + (xx - vx) ** 2
        occupied[z0:z1, y0:y1, x0:x1] |= dist2 <= r_vox**2

    field = ndimage.distance_transform_edt(~occupied, sampling=(v_size,) * 3)
    return torch.from_numpy(field).float(), v_size


@dataclass
class SphereProteinSpec:
    """
    One placeable protein species for `SpherePackingSpecimenGenerator`.

    Attributes
    ----------
    pdb_source : str
        PDB ID (4-character code, fetched from RCSB) or a local PDB/mmCIF
        file path.
    n_copies : int, optional
        Exact instance count, used only for `target_specs` (ignored, and
        must be left unset, for `filler_specs`). Required (must be a
        positive int) for every spec passed as a `target_spec`.

        There's deliberately no per-species weight for `filler_specs`: an
        earlier `ratio` field (relative attempt-count weight) turned out to
        be a confusing lever -- `pack_hard_spheres_3d`'s largest-first
        staging already gives every species a fair shot in placement order,
        but the ACTUAL placed mix is dominated by geometry regardless of
        any requested weighting (large species hit a hard jamming ceiling
        no `ratio` can move; small species fill whatever fragments of space
        are left over and dominate the final count almost no matter what).
        Since that geometric effect swamps any deliberate weighting between
        differently-sized species anyway, all filler species are now drawn
        with equal attempt-weight -- see `SpherePackingSpecimenGenerator.
        generate`.
    """

    pdb_source: str
    n_copies: int | None = None


@dataclass
class SpherePlacement:
    """One placed instance, for ground-truth bookkeeping."""

    species_id: str  # the owning spec's pdb_source
    position_xyz: torch.Tensor  # (3,) physical Angstrom, box-centered
    rotation_matrix: torch.Tensor  # (3, 3)
    # 1-based; matches this instance's voxel value in `instance_labels`
    instance_id: int
    # "target" or "filler" -- which stage placed this instance. Tracked
    # separately from species_id since the same pdb_source can legitimately
    # appear in both target_specs and filler_specs (e.g. a target species
    # that's also generically abundant as background); export_picks uses
    # this, not species_id, to decide what counts as ground truth.
    role: Literal["target", "filler"] = "target"


class SpherePackingSpecimenGenerator:
    """
    Packs several PDB-derived protein species into a volume via hard-sphere
    packing, sized to each species' own real physical radius, then renders
    each placed instance's real scattering potential.

    Parameters
    ----------
    target_specs : list of SphereProteinSpec
        Species placed FIRST, each at its own exact `n_copies` count (must
        be set on every entry). These are the annotated ground truth --
        always exported by `export_picks`.
    filler_specs : list of SphereProteinSpec, optional
        Species placed SECOND, equal attempt-weight across species, filling
        whatever space is left around the already-placed targets (see
        `filler_occupancy_fraction`). Default empty (no filler).
    target_shape : tuple of int, optional
        Output volume shape (Z, Y, X), voxels. Default (128, 256, 256).
    v_size : float, optional
        Voxel size, Angstrom. Default 5.0.
    filler_occupancy_fraction : float, optional
        Target packing density for `filler_specs`, as a bare-sphere
        fraction of the box volume (ignored if `filler_specs` is empty).
        Both packing backends already self-limit at their own physical
        jamming ceiling rather than erroring when this is unreachable
        (RSA: ~28-41%; dense: higher but "best effort" near random-close-
        packing, ~0.64) -- so instead of hand-tuning this to a precise
        number, the default (0.5) is deliberately set high enough that
        filler simply packs until it jams, for whichever species mix and
        box you give it. Compare `len(placements)` (minus target count) to
        `n_filler_candidates` after `generate()` to see how much was
        actually placed. Lower this only if you deliberately want a
        sparser-than-maximal filler layer. Default 0.5.
    gap_angstrom : float, optional
        Minimum clearance between placed spheres' surfaces, Angstrom
        (steric/hydration-shell buffer). Default 5.0.
    packing_method : {"rsa", "dense"}, optional
        Packing backend for the FILLER stage only (targets always use RSA,
        since exact counts have no equivalent in the dense backend's
        occupancy-fraction-driven API). "rsa" (default,
        :func:`.algorithms.pack_hard_spheres_3d`) is fast (sub-second to a
        few seconds at a few thousand instances) and is the only backend
        that can honor an exclusion field, so it's the only one that can
        correctly avoid already-placed targets. "dense"
        (:func:`.algorithms.pack_hard_spheres_3d_dense`) is typically
        denser via periodic force-biased relaxation, at the cost of a few
        minutes of wall time at a few thousand instances -- but has no
        obstacle-avoidance mechanism at all, so it raises `ValueError` if
        `target_specs` is non-empty. Default "rsa".
    pad_fraction : float, optional
        Only used when `packing_method="dense"` -- passed straight through
        to `pack_hard_spheres_3d_dense`'s `pad_fraction`. Default 0.5.
    clip_axes : tuple of bool, optional
        (z, y, x), matching `target_shape`'s own axis order. True on an
        axis means placed spheres are allowed to extend past that wall --
        they're truncated naturally when rendered (`insert_particles_into_
        micrograph` already clips at volume boundaries) rather than being
        rejected/discarded outright. False (default, all axes) requires
        every instance's full extent to fit within `target_shape`. Useful
        e.g. for a tomogram whose xy field of view is a crop of a larger
        cellular region (edge particles there are fine to truncate) but
        whose z extent is a real specimen-thickness boundary particles
        should not cross: `clip_axes=(False, True, True)`. Applied to both
        the target and filler stages.
    pdb_cache_dir : str, optional
        Directory for downloaded PDB/mmCIF files. Default is
        `specter.pdb.DEFAULT_PDB_SAVEFOLDER` (the repo's own `pdb-data/`,
        anchored to the package location, not the caller's cwd).
    seed : int, optional
        Random seed.
    device : str or torch.device, optional
        Device for `PotentialBuilder`'s potential-building step ONLY --
        the FFT-based convolution that's the real compute cost for large
        species. Packing always runs on CPU regardless of this setting: the
        `vesin_torch`-based neighbor list `pack_hard_spheres_3d` uses is
        both slower on GPU at realistic particle counts (kernel-launch/
        neighbor-list-construction overhead dominates -- see that
        function's own docstring) and prone to OOM there for larger
        candidate pools, so it's hardcoded to "cpu" rather than exposed
        here. Rotation/insertion/label-stamping also always run on CPU --
        each species' built potential is moved back to CPU immediately
        after `PotentialBuilder.forward` returns.
    chunk_size : int, optional
        Number of instances to rotate per batch, per species -- caps peak
        memory when a species has many placed instances (mirrors
        `specter.crowding.CrowdWithDuplicates`'s own `chunk_size`). Default
        None (rotate all of a species' instances at once).
    progressbars : bool, optional
        Show progress bars for PDB loading, packing, and per-species
        rendering (including `PotentialBuilder`'s own build progress) while
        `generate()` runs. Default True.

    Attributes
    ----------
    placements : list of SpherePlacement
        Every successfully placed instance (targets then filler), set
        after `generate()` runs.
    n_target_requested : int
        Total requested target count (sum of `target_specs[i].n_copies`),
        set after `generate()` runs.
    n_targets_placed : int
        How many target instances were actually placed -- compare against
        `n_target_requested` to see any shortfall (the box couldn't fit
        every requested copy). Set after `generate()` runs.
    n_filler_candidates : int
        Size of the drawn filler candidate pool, set after `generate()`
        runs (0 if `filler_specs` is empty).
    n_candidates : int
        `n_target_requested + n_filler_candidates`, for convenience.
    instance_labels : torch.Tensor
        Per-instance integer label volume, shape `target_shape`, dtype
        int32. 0 is background; each placed instance's voxels (its
        rotated template, thresholded at `_INSTANCE_LABEL_REL_THRESHOLD` of
        that species' own peak) carry its `SpherePlacement.instance_id`.
        Set after `generate()` runs.
    """

    def __init__(
        self,
        target_specs: list[SphereProteinSpec],
        filler_specs: list[SphereProteinSpec] | None = None,
        target_shape: tuple[int, int, int] = (128, 256, 256),
        v_size: float = 5.0,
        filler_occupancy_fraction: float = 0.5,
        gap_angstrom: float = 5.0,
        packing_method: Literal["rsa", "dense"] = "rsa",
        pad_fraction: float = 0.5,
        clip_axes: tuple[bool, bool, bool] = (False, False, False),
        pdb_cache_dir: str = DEFAULT_PDB_SAVEFOLDER,
        seed: int | None = None,
        device: str | torch.device = "cpu",
        chunk_size: int | None = None,
        progressbars: bool = True,
    ):
        filler_specs = list(filler_specs) if filler_specs else []
        if not target_specs and not filler_specs:
            raise ValueError("target_specs and filler_specs can't both be empty")
        if packing_method not in ("rsa", "dense"):
            raise ValueError(
                f"packing_method must be 'rsa' or 'dense', got {packing_method!r}"
            )
        if packing_method == "dense" and target_specs and filler_specs:
            raise ValueError(
                "packing_method='dense' has no obstacle-avoidance mechanism, "
                "so it can't guarantee filler avoids already-placed targets "
                "-- use packing_method='rsa' whenever both target_specs and "
                "filler_specs are non-empty."
            )
        for spec in target_specs:
            if not spec.n_copies or spec.n_copies <= 0:
                raise ValueError(
                    f"target spec {spec.pdb_source!r} needs n_copies set to "
                    "a positive int (targets are placed at an exact count, "
                    "not ratio-weighted)."
                )
        self.target_specs = target_specs
        self.filler_specs = filler_specs
        self.target_shape = target_shape
        self.v_size = v_size
        self.filler_occupancy_fraction = filler_occupancy_fraction
        self.gap_angstrom = gap_angstrom
        self.packing_method = packing_method
        self.pad_fraction = pad_fraction
        self.clip_axes = clip_axes
        self.pdb_cache_dir = pdb_cache_dir
        self.seed = seed
        self.device = device
        self.chunk_size = chunk_size
        self.progressbars = progressbars

        self.placements: list[SpherePlacement] = []
        self.n_target_requested = 0
        self.n_targets_placed = 0
        self.n_filler_candidates = 0
        self.n_candidates = 0
        self.instance_labels: torch.Tensor | None = None

    def generate(self) -> torch.Tensor:
        """
        Run the full pipeline and return the assembled specimen volume.

        Returns
        -------
        torch.Tensor
            Shape `target_shape`, dtype float32.
        """
        if self.seed is not None:
            torch.manual_seed(
                self.seed
            )  # random_rotation_matrix has no generator= param

        all_specs = list(self.target_specs) + list(self.filler_specs)
        n_targets = len(self.target_specs)
        with TqdmProgress(transient=True, disable=not self.progressbars) as progress:
            pdb_task = progress.add_task("Loading PDB structures", total=len(all_specs))
            pdbs = []
            for spec in all_specs:
                progress.update(pdb_task, description=f"Loading {spec.pdb_source}")
                pdbs.append(
                    PDB(spec.pdb_source, savefolder=self.pdb_cache_dir, verbose=False)
                )
                progress.update(pdb_task, advance=1)
        species_radii = torch.tensor([float(pdb.max_diameter) / 2.0 for pdb in pdbs])

        box = (
            self.target_shape[0] * self.v_size,
            self.target_shape[1] * self.v_size,
            self.target_shape[2] * self.v_size,
        )
        box_volume = box[0] * box[1] * box[2]

        # --- Stage 1: targets, exact counts, placed first ---
        target_radii_list: list[float] = []
        target_species_list: list[int] = []
        for i, spec in enumerate(self.target_specs):
            target_radii_list += [float(species_radii[i])] * spec.n_copies  # type: ignore[operator]
            target_species_list += [i] * spec.n_copies  # type: ignore[operator]
        self.n_target_requested = len(target_radii_list)

        if target_radii_list:
            target_radii_t = torch.tensor(target_radii_list)
            with status(
                f"Packing {len(target_radii_list)} target instances",
                disable=not self.progressbars,
            ):
                target_coords, target_accepted_idx = pack_hard_spheres_3d(
                    target_radii_t,
                    box,
                    gap=self.gap_angstrom,
                    seed=self.seed,
                    device="cpu",  # see self.device's own docstring
                    clip_axes=self.clip_axes,
                )
            target_species_t = torch.tensor(target_species_list, dtype=torch.long)
            target_accepted_species = target_species_t[target_accepted_idx]
            target_accepted_radii = target_radii_t[target_accepted_idx]
        else:
            target_coords = torch.empty((0, 3))
            target_accepted_species = torch.empty((0,), dtype=torch.long)
            target_accepted_radii = torch.empty((0,))
        self.n_targets_placed = int(target_coords.shape[0])

        # --- Stage 2: filler, equal attempt-weight, avoiding placed targets ---
        if self.filler_specs:
            filler_species_radii = species_radii[n_targets:]
            # No per-species weight (see SphereProteinSpec's own docstring
            # for why) -- draw_species_pool/pack_hard_spheres_3d_dense still
            # take a ratios tensor, so pass a uniform one.
            filler_ratios = torch.ones(len(self.filler_specs))

            with status(
                f"Packing filler instances ({self.packing_method})",
                disable=not self.progressbars,
            ):
                if self.packing_method == "rsa":
                    pool_radii, pool_species_idx = draw_species_pool(
                        filler_species_radii,
                        filler_ratios,
                        self.filler_occupancy_fraction,
                        box_volume,
                        seed=self.seed,
                    )
                    self.n_filler_candidates = int(pool_radii.numel())
                    exclusion_field: torch.Tensor | None = None
                    field_v_size: float | None = None
                    if target_coords.shape[0] > 0:
                        # Pad the forbidden mask by one voxel before the
                        # distance transform (exclusion_distance_field's own
                        # docstring recommendation) so trilinear-
                        # interpolation "bleed" on a coarse grid can't let a
                        # filler candidate creep closer to a target than
                        # `gap` actually allows.
                        exclusion_field, field_v_size = _build_sphere_exclusion_field(
                            target_coords,
                            target_accepted_radii + self.v_size,
                            self.target_shape,
                            self.v_size,
                        )
                    filler_coords, filler_accepted_idx = pack_hard_spheres_3d(
                        pool_radii,
                        box,
                        gap=self.gap_angstrom,
                        seed=self.seed,
                        device="cpu",  # see self.device's own docstring
                        clip_axes=self.clip_axes,
                        exclusion_distance_field=exclusion_field,
                        field_v_size=field_v_size,
                    )
                    filler_accepted_species = (
                        pool_species_idx[filler_accepted_idx] + n_targets
                    )
                else:
                    filler_coords, _filler_radii_out, filler_species_out = (
                        pack_hard_spheres_3d_dense(
                            filler_species_radii,
                            filler_ratios,
                            self.filler_occupancy_fraction,
                            box,
                            gap=self.gap_angstrom,
                            seed=self.seed,
                            device="cpu",  # see self.device's own docstring
                            pad_fraction=self.pad_fraction,
                            clip_axes=self.clip_axes,
                        )
                    )
                    self.n_filler_candidates = int(filler_coords.shape[0])
                    filler_accepted_species = filler_species_out + n_targets
        else:
            filler_coords = torch.empty((0, 3))
            filler_accepted_species = torch.empty((0,), dtype=torch.long)
            self.n_filler_candidates = 0

        coords = torch.cat([target_coords, filler_coords], dim=0)
        accepted_species_idx = torch.cat(
            [target_accepted_species, filler_accepted_species], dim=0
        )
        self.n_candidates = self.n_target_requested + self.n_filler_candidates

        volume = torch.zeros(self.target_shape, dtype=torch.float32)
        instance_labels = torch.zeros(self.target_shape, dtype=torch.int32)
        self.placements = []
        next_instance_id = 1

        with TqdmProgress(transient=True, disable=not self.progressbars) as progress:
            render_task = progress.add_task("Rendering species", total=len(all_specs))
            for species_i, spec in enumerate(all_specs):
                mask = accepted_species_idx == species_i
                if not bool(mask.any()):
                    progress.update(render_task, advance=1)
                    continue
                n_instances_preview = int(mask.sum())
                progress.update(
                    render_task,
                    description=(
                        f"Rendering {spec.pdb_source} ({n_instances_preview} instances)"
                    ),
                )
                pdb = pdbs[species_i]
                n = estimate_protein_box_size(pdb.max_diameter, self.v_size)
                builder = PotentialBuilder(
                    n_xyz=n,
                    dx=self.v_size,
                    atomic_numbers=pdb.atomic_numbers,
                    progressbars=self.progressbars,
                ).to(self.device)
                # Brought back to CPU immediately -- rotation/insertion/label
                # stamping below stay CPU-only regardless of self.device (see
                # that parameter's own docstring); only the potential build
                # itself runs on self.device.
                template = builder.forward(pdb.coordinates, method="analytic").to("cpu")
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
                        pixel_size=self.v_size,
                        micrograph=volume,
                    )
                    binarized = (rotated > label_threshold).to(
                        torch.int32
                    ) * instance_ids[start:end].view(-1, 1, 1, 1)
                    instance_labels = _insert_instance_labels(
                        binarized,
                        species_coords[start:end],
                        pixel_size=self.v_size,
                        labels=instance_labels,
                    )

                role: Literal["target", "filler"] = (
                    "target" if species_i < n_targets else "filler"
                )
                for i in range(n_instances):
                    self.placements.append(
                        SpherePlacement(
                            species_id=spec.pdb_source,
                            position_xyz=species_coords[i],
                            rotation_matrix=R[i],
                            instance_id=int(instance_ids[i]),
                            role=role,
                        )
                    )
                progress.update(render_task, advance=1)

        self.instance_labels = instance_labels
        return volume

    def export_picks(
        self,
        output_dir: str | Path,
        annotation_version: str = "1.0",
        oriented: bool = True,
        include_filler: bool = False,
    ) -> dict[str, Path]:
        """
        Write one copick/CryoET-Data-Portal-style .ndjson pick file per
        placed TARGET species -- same schema as `specimen.cryoet.
        CryoETSpecimenGenerator.export_picks` (one JSON object per line:
        ``{"type": "point"|"orientedPoint", "location": {"x", "y", "z"}[,
        "xyz_rotation_matrix"]}``), so picks from either generator are
        interchangeable downstream.

        Filler placements are excluded by default (see `include_filler`) --
        they're crowding background, not annotated ground truth. Grouping
        is by `SpherePlacement.role` first, then `species_id`, so a
        pdb_source used as BOTH a target and filler species still keeps
        its target instances and filler instances in separate files (never
        silently merged) if `include_filler=True`.

        Coordinates are converted from this generator's box-centered
        convention (`SpherePlacement.position_xyz`, origin at the volume's
        center, matching `pack_hard_spheres_3d`) to the corner-relative
        (``0..extent``) convention copick/the portal actually use -- the
        same conversion `CryoETSpecimenGenerator.export_picks` performs,
        just starting from a different internal coordinate origin (see
        that method's own docstring for the underlying axis-convention
        note).

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
        include_filler : bool, optional
            If True, also write pick files for filler species (suffixed
            ``-filler`` when their pdb_source collides with a target's, to
            avoid overwriting the target's own file). Default False.

        Returns
        -------
        dict[str, pathlib.Path]
            Mapping of species id (`SphereProteinSpec.pdb_source`, suffixed
            ``-filler`` on a target/filler pdb_source collision) to written
            file path.
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

        target_sources = {spec.pdb_source for spec in self.target_specs}
        by_species: dict[str, list[SpherePlacement]] = {}
        for placed in self.placements:
            if placed.role == "filler" and not include_filler:
                continue
            key = placed.species_id
            if placed.role == "filler" and placed.species_id in target_sources:
                key = f"{placed.species_id}-filler"
            by_species.setdefault(key, []).append(placed)

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
