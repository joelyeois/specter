"""
Public membrane specimen generator.

Ties together the four membrane submodules built this session into one
class: organic shape (``_field``), calibrated bilayer potential
(``_profile``), anti-aliased rasterization onto an output grid
(``_raster``), and normal-aligned transmembrane protein insertion
(``_placement``).

Everything shares one centered-origin physical coordinate frame: physical
``(0, 0, 0)`` is the volume's own center, matching the convention already
used throughout ``_field``/``_raster``. This generator does not place a
membrane instance off-center within a larger tomogram itself -- that
compositing step (offsetting, collision-rejecting random placement)
belongs to ``specter.specimen.tomogram.MembraneTomogramGenerator``, the
same way the other specimen generators in this package are used standalone
before being composited.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import torch

from ...arrays import clip_insert_bounds
from ...pdb import DEFAULT_PDB_SAVEFOLDER, PDB
from ...potential import PotentialBuilder
from ...rotations import build_affine_matrix, rotate_volume
from .._parallel_render import (
    build_templates_concurrently,
    resolve_render_devices,
    resolve_render_workers,
)
from ..packing import estimate_protein_box_size
from ._field import MembraneField, _grid_points_xyz
from ._field_spherical_harmonics import generate_membrane_field_spherical_harmonics
from ._field_swept_spline import generate_membrane_field_swept_spline
from ._placement import (
    align_principal_axis_to_z,
    align_transmembrane_depth,
    orientation_for_normal,
    sample_surface_sites,
)
from ._profile import (
    BilayerProfile,
    build_analytic_bilayer_profile,
    build_reference_lipid_patch,
    estimate_bilayer_peak_amplitude,
)
from ._raster import rasterize_membrane_density

# target_shape_zyx auto-sizing/clamping (spherical_harmonics/swept_spline
# only -- see MembraneGenerator.__init__): an organelle's own half-extent
# may occupy at most this fraction of the working box's own half-extent,
# leaving margin for undulation/wander rather than letting a drawn/explicit
# size exactly touch the boundary.
_SIZE_MARGIN_FRACTION = 0.85
# Below this safe half-extent, refuse rather than silently clamp every
# instance down to a degenerate, barely-there size.
_MIN_SAFE_HALF_EXTENT_A = 50.0


def _resolve_range(
    name: str, range_ab: tuple[float, float], min_low: float
) -> tuple[float, float]:
    low, high = range_ab
    if low > high or low < min_low:
        raise ValueError(
            f"{name} must be (low, high) with {min_low} <= low <= high, "
            f"got {range_ab!r}"
        )
    return low, high


def _draw_uniform(rng: torch.Generator, low: float, high: float) -> float:
    return float(torch.empty(1).uniform_(low, high, generator=rng).item())


# Per-call element budget for _chunked_upsample_density's grid_sample calls
# -- chosen well under the ~2^31 total-element limit confirmed directly on
# both F.interpolate and F.grid_sample's CUDA kernels ("invalid
# configuration argument", regardless of available memory), with enough
# margin that the per-chunk coordinate buffer (3 floats/point) also stays
# small (~1.2 GB at this budget).
_UPSAMPLE_CHUNK_VOXELS = 100_000_000


def _chunked_upsample_density(
    coarse_volume: torch.Tensor,
    gen_v_size: float,
    origin_xyz: torch.Tensor,
    target_shape_zyx: tuple[int, int, int],
    v_size: float,
) -> torch.Tensor:
    """
    Upsample a coarse density raster onto a fine target grid, in Z-chunks.

    Reuses `MembraneField.sample()` -- the same trilinear point-sampling
    machinery `rasterize_membrane_density` already uses for the primary
    (non-decoupled) raster path -- rather than `F.interpolate`, so both
    agree on the exact same align_corners=False/pixel-center convention
    (`MembraneField._normalized_grid`). Chunked along Z so neither the
    coordinate buffer nor any one CUDA kernel launch has to handle the
    whole (potentially multi-billion-voxel) fine grid at once -- see
    `_UPSAMPLE_CHUNK_VOXELS`.

    Parameters
    ----------
    coarse_volume : torch.Tensor
        Density raster on the coarse generation grid, shape matching
        `coarse_volume.shape`, spacing `gen_v_size`.
    gen_v_size : float
        Coarse grid's voxel size, Angstrom.
    origin_xyz : torch.Tensor
        Physical `(x, y, z)` location of index `(0, 0, 0)`, shared by both
        the coarse and fine grids (same physical extent, see caller).
    target_shape_zyx : tuple of int
        Fine output grid shape.
    v_size : float
        Fine grid's voxel size, Angstrom.

    Returns
    -------
    torch.Tensor
        Density volume, shape `target_shape_zyx`, same device/dtype as
        `coarse_volume`.
    """
    coarse_field = MembraneField(
        phi=coarse_volume, spacing_a=gen_v_size, origin_xyz=origin_xyz
    )
    nz, ny, nx = target_shape_zyx
    chunk_z = max(1, _UPSAMPLE_CHUNK_VOXELS // max(1, ny * nx))
    fine = torch.empty(
        target_shape_zyx, dtype=coarse_volume.dtype, device=coarse_volume.device
    )
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        chunk_origin = origin_xyz.clone()
        chunk_origin[2] = chunk_origin[2] + z0 * v_size
        points_xyz = _grid_points_xyz(
            (z1 - z0, ny, nx), v_size, chunk_origin, coarse_volume.device
        )
        fine[z0:z1] = coarse_field.sample(points_xyz)
    return fine


@dataclass
class TransmembraneSpec:
    """
    One transmembrane protein species to attempt placing in the membrane.

    Parameters
    ----------
    pdb_source : str
        PDB ID or local PDB/mmCIF path. Used as the species id and, when
        ``template`` is not supplied, as the input to ``PotentialBuilder``.
    frequency : int, optional
        Relative placement weight among species. Default 1.
    tm_span_mask : torch.Tensor, optional
        Boolean mask, same atom order as the loaded structure's
        coordinates, selecting the transmembrane span. Default None, which
        centers the full structure's z-extent instead (see
        :func:`~specter.specimen.membrane._placement.align_transmembrane_depth`).
        Ignored when ``template`` is supplied -- depth alignment must
        already be baked into a prebuilt template.
    parameterization : str, optional
        PotentialBuilder parameterization for PDB-backed specs. Default
        "shtyrov".
    template : torch.Tensor, optional
        Prebuilt density template with shape ``(Z, Y, X)``, already
        depth-aligned (transmembrane span centered on the template's own
        z-center). If supplied, no PDB is loaded for this species.
    """

    pdb_source: str
    frequency: int = 1
    tm_span_mask: torch.Tensor | None = None
    parameterization: str = "shtyrov"
    template: torch.Tensor | None = None


def render_transmembrane_template(
    spec: TransmembraneSpec,
    v_size: float,
    pdb_cache_dir: str,
    device: str | torch.device,
) -> torch.Tensor:
    """
    Build one transmembrane species' potential template from its PDB source.

    Pure/stateless (no `MembraneGenerator` instance involved) so a caller
    driving several `MembraneGenerator`s that all share the same
    `transmembrane_specs` -- e.g. `n_instances` copies of one `[[membrane]]`
    entry in `specter build tomogram` -- can call this ONCE per species and
    pass the result via `TransmembraneSpec.template`, instead of letting
    each instance's own `MembraneGenerator._build_template` rebuild the same
    species redundantly. `MembraneGenerator._build_template` itself is a
    thin device-routing wrapper around this function.

    Parameters
    ----------
    spec : TransmembraneSpec
        Species to render (``spec.template``, if set, is ignored here --
        callers that already have a template have no reason to call this).
    v_size : float
        Output voxel size, Angstrom.
    pdb_cache_dir : str
        Passed to `PDB` for fetching/caching `spec.pdb_source`.
    device : str or torch.device
        Device to build the `PotentialBuilder` (and run its `forward`) on.

    Returns
    -------
    torch.Tensor
        Density template, shape ``(Z, Y, X)``, on `device`.
    """
    pdb = PDB(spec.pdb_source, savefolder=pdb_cache_dir, verbose=False)
    coordinates = align_principal_axis_to_z(pdb.coordinates)
    coordinates = align_transmembrane_depth(coordinates, spec.tm_span_mask)
    n = estimate_protein_box_size(pdb.max_diameter, v_size)
    builder = PotentialBuilder(
        n_xyz=n,
        dx=v_size,
        atomic_numbers=pdb.atomic_numbers,
        progressbars=False,
        parameterization=spec.parameterization,
    ).to(device)
    # "analytic" (PotentialBuilder's own documented default), not "3d" --
    # confirmed empirically the two methods integrate to the same total
    # potential (ratio ~1.001 across a 10x span of voxel sizes, on a real
    # structure) but "3d"'s convolution-based approach spreads that same
    # total over a visibly lower, smoother peak (25-45% lower max value
    # than "analytic" for identical atoms). The bilayer profile is
    # rendered with "analytic"; using "3d" here compared two differently-
    # sharpened renderings of the same underlying physics, not a
    # deliberate contrast difference between membrane and protein.
    return builder.forward(coordinates, method="analytic")


@dataclass
class TransmembranePlacement:
    """One accepted transmembrane placement."""

    species_id: str
    center_xyz: torch.Tensor
    rotation_matrix: torch.Tensor


class MembraneGenerator:
    """
    Generate an organic membrane specimen volume with transmembrane inserts.

    Parameters
    ----------
    target_shape_zyx : tuple of int, optional
        Output volume shape, ``(Z, Y, X)``. Default `None`: auto-sized from
        the (possibly randomly drawn, see `sh_axes_range_a`/`swept_*_range_a`
        below) organelle size, with margin. When given
        explicitly, the organelle size is instead CLAMPED to whatever this
        box can safely hold (with a warning if that changes anything) --
        either way, a random or explicit size can never silently produce a
        clipped shape from a mismatched box. See `MembraneField.
        clipped_at_boundary` for the last-resort check this backs up.
    v_size : float, optional
        Output voxel size, Angstrom. Also used to render transmembrane
        protein templates, so their scale matches the membrane's. Default
        5.0.
    shape_backend : {"spherical_harmonics", "swept_spline"}, optional
        Which organic-shape algorithm builds the underlying
        :class:`~specter.specimen.membrane._field.MembraneField`.
        `"spherical_harmonics"` (default) perturbs an ellipsoid with a
        random real spherical-harmonic expansion
        (:func:`~specter.specimen.membrane.
        _field_spherical_harmonics.generate_membrane_field_spherical_harmonics`)
        -- smooth by construction and given a physically-motivated (Helfrich
        bending-mode) undulation spectrum, but -- being a radius function of
        direction -- restricted to STAR-CONVEX topology: good for vesicles,
        nuclei, and roughly-spherical (or ellipsoidal, via `sh_axes_a`)
        mitochondria; cannot represent branching/self-occluding shapes.
        `"swept_spline"` instead sweeps a sphere along a smoothly wandering
        random path (:func:`~specter.specimen.membrane._field_swept_spline.
        generate_membrane_field_swept_spline`) -- an elongated, tube-shaped
        organelle (ER-tubule-like) rather than a closed blob, and unlike
        `"spherical_harmonics"` it CAN self-approach/loop back near itself (a
        real topological capability the radius-function-of-direction backend
        structurally lacks).

        Both backends are read afterward only through `MembraneField`'s
        public contract, so the calibrated bilayer profile and transmembrane
        placement below are identical either way -- this choice only affects
        the shape.
    sh_max_degree : int, optional
        Highest spherical-harmonic degree in the random surface
        perturbation -- higher values add finer surface detail. See
        `generate_membrane_field_spherical_harmonics`'s own docstring.
        `"spherical_harmonics"` backend only. Default 8.
    sh_axes_a : tuple of float, optional
        Physical semi-axes `(a_x, a_y, a_z)` of the base ellipsoid,
        Angstrom -- isotropic axes give a roughly spherical organelle,
        anisotropic axes give an elongated/flattened one. Default `None`:
        each axis is drawn independently, uniformly, from `sh_axes_range_a`
        (mild natural anisotropy for free) using a `seed`-derived generator
        independent of the shape's own randomness. `"spherical_harmonics"`
        backend only.
    sh_axes_range_a : tuple of float, optional
        `(low, high)` semi-axis draw range, Angstrom, used only when
        `sh_axes_a` is `None`. Default `(150.0, 450.0)` -- real vesicle/
        small-organelle scale (radius): synaptic vesicles run ~25-60 nm
        diameter, general/endosomal vesicles up to ~100-300 nm diameter in
        cryo-ET (PNAS 10.1073/pnas.2403136121; PMID 24455109); this range
        (30-90 nm diameter) sits inside that population without also
        reaching into mitochondria scale (~200-700 nm diameter), which is
        different enough (>10x) that a single default range can't cover
        both coherently -- pass `sh_axes_a` explicitly for that.
        `"spherical_harmonics"` backend only.
    sh_amplitude : float, optional
        RMS fractional radius perturbation (dimensionless) -- picked from a
        direct visual sweep, see `generate_membrane_field_spherical_harmonics`'s
        own docstring. `"spherical_harmonics"` backend only. Default 0.15.
    sh_spectrum_power : float, optional
        Exponent `p` in `Var(a_lm) ~ 1 / [l*(l+1)]**p`; `2.0` matches the
        Helfrich (1973) thermal bending-mode spectrum of a lipid bilayer at
        equilibrium. `"spherical_harmonics"` backend only. Default 2.0.
    swept_total_length_a : float, optional
        Approximate path CONTOUR length (not bounding-box extent), Angstrom.
        See `generate_membrane_field_swept_spline`'s own docstring. Default
        `None`: drawn uniformly from `swept_total_length_range_a`.
        `"swept_spline"` backend only.
    swept_total_length_range_a : tuple of float, optional
        `(low, high)` contour-length draw range, Angstrom, used only when
        `swept_total_length_a` is `None`. Default `(1500.0, 2500.0)` --
        sized to keep a good length:radius aspect ratio (still reads as a
        tube, not a blob) even at `swept_tube_radius_range_a`'s own upper
        bound, while keeping the auto-sized `target_shape_zyx` this range
        implies (see that parameter) in a practical, fast-to-generate
        regime: measured directly, the worst-case combination (max radius,
        max length) auto-sizes to ~540 voxels/axis at `v_size=5.0` (~0.6 GB
        float32) at this range's own default; a naively "more realistic"
        4000 A upper bound instead reaches ~1130 voxels/axis (~5.8 GB) and
        measurably slow generation, for organelle sizes well past what
        fits in a typical local tomogram crop anyway. `"swept_spline"`
        backend only.
    swept_step_length_a : float, optional
        Distance between consecutive blended sphere source centers along
        the path, Angstrom -- must stay well under `2 * swept_tube_radius_a` or
        the tube shows visible beading (warned about proactively). Default
        `None`: `0.5 * swept_tube_radius_a` (using whatever value that
        resolves to), which stays safely under the beading threshold
        regardless of which radius gets drawn, unlike a fixed absolute
        default tuned for one specific radius. `"swept_spline"` backend
        only.
    swept_tube_radius_a : float, optional
        Tube radius, Angstrom. Default `None`: drawn uniformly from
        `swept_tube_radius_range_a`. `"swept_spline"` backend only.
    swept_tube_radius_range_a : tuple of float, optional
        `(low, high)` tube-radius draw range, Angstrom, used only when
        `swept_tube_radius_a` is `None`. Default `(150.0, 400.0)` (30-80 nm
        diameter) -- real ER tubule scale: EM measurements of neuronal ER
        tubules run ~20 nm diameter (thin), general ER tubules up to ~88 nm
        diameter by STORM (PNAS 10.1073/pnas.2117559119; PMC5705721).
        `"swept_spline"` backend only.
    swept_flexibility : float, optional
        In (0, 1]; picked from a direct visual sweep (see
        `generate_membrane_field_swept_spline`'s own docstring): 0.05 is
        nearly a straight rod, 0.35+ produces sharp hooks/loops, 0.15 gives
        a gently organic, clearly non-straight tube. Default `None`: drawn
        uniformly from `swept_flexibility_range`. `"swept_spline"` backend
        only.
    swept_flexibility_range : tuple of float, optional
        `(low, high)` flexibility draw range, used only when
        `swept_flexibility` is `None`. Default `(0.08, 0.25)` -- stays
        inside the "gently organic" zone characterized by the visual sweep
        above, avoiding both the near-straight-rod and sharp-hook/loop
        extremes. `"swept_spline"` backend only.
    swept_radius_variation : float, optional
        RMS fractional variation in tube radius along the path -- see
        `generate_membrane_field_swept_spline`'s own docstring ("Radius
        variation"). Default `None`: drawn uniformly from
        `swept_radius_variation_range`. `"swept_spline"` backend only.
    swept_radius_variation_range : tuple of float, optional
        `(low, high)` radius-variation draw range, used only when
        `swept_radius_variation` is `None`. Default `(0.1, 0.3)` -- mild,
        organic caliber variation by default (real ER tubules show local
        varicosities/constrictions) rather than a perfectly uniform tube.
        `"swept_spline"` backend only.
    swept_radius_variation_sigma_points : float, optional
        Path-order smoothing for the radius noise, in PATH POINTS -- see
        `generate_membrane_field_swept_spline`'s own docstring for why 2.0
        (not a larger, intuitively "smoother" value) was picked from a
        direct visual sweep. Only affects the field when
        `swept_radius_variation > 0`. `"swept_spline"` backend only.
        Default 2.0.
    swept_blend_sharpness_a : float, optional
        Smooth-min blend radius, Angstrom. Default (`None`) is
        `0.5 * swept_tube_radius_a` -- a default tuned for a handful of
        sparse, independent blobs would under-blend a dense chain of
        sources into visible beading, so this is a separate default.
        `"swept_spline"` backend only.
    swept_path_smoothing_sigma_points : float, optional
        Path-order-aware smoothing (`scipy.ndimage.gaussian_filter1d`),
        in PATH POINTS -- see `generate_membrane_field_swept_spline`'s own
        docstring for why this must be order-aware rather than spatial.
        `"swept_spline"` backend only. Default 1.5.
    swept_curvature_iterations : int, optional
        Curvature-capping relaxation steps -- an analytic smooth-min blend
        can still have locally sharp concave curvature, so this stays
        applied here too. `"swept_spline"` backend only. Default 15.
    swept_curvature_step_fraction : float, optional
        See :func:`~specter.specimen.membrane._field.cap_curvature`.
        `"swept_spline"` backend only. Default 0.15.
    n_lipids_per_leaflet : int, optional
        Reference lipid patch size, passed to
        :func:`~specter.specimen.membrane._profile.build_reference_lipid_patch`.
        `estimate_bilayer_peak_amplitude` (the only thing this patch feeds
        into -- not the profile's shape, see `bilayer_thickness_a`/
        `bilayer_layer_sigma_a`) only reads off the patch's set of unique
        atomic species, not its size or layout, so in practice any value
        that includes at least one of every species in the lipid template
        gives the same calibrated amplitude -- this parameter has no
        material effect on `generate()`'s output at its default template.
        Kept for backward compatibility and in case a future custom lipid
        template makes species presence itself patch-size-dependent.
        Default 200.
    parameterization : str, optional
        PotentialBuilder parameterization for the lipid reference patch.
        Default "shtyrov".
    bilayer_thickness_a : float, optional
        Phosphate-to-phosphate (outer-leaflet-peak to inner-leaflet-peak)
        spacing, Angstrom, for the analytic two-Gaussian-peak bilayer
        profile (:func:`~specter.specimen.membrane._profile.
        build_analytic_bilayer_profile` -- matches real cryo-EM bilayer
        micrographs' two-line "railroad track" appearance directly, rather
        than emerging from a simulated atomic point cloud, which proved
        fragile: see that function's own docstring for the concrete bugs
        this replaced). Default 30.0 -- midpoint of `polnet`'s own
        `MB_THICK_RG` default range (25.0, 35.0) (an earlier default of
        38.0, taken from the old atomic model's own headgroup z-offsets
        rather than cross-checked against polnet specifically, sat above
        polnet's entire range and visibly read as too widely spaced).
    bilayer_layer_sigma_a : float, optional
        Gaussian width of each leaflet peak, Angstrom -- matches `polnet`'s
        own `MB_LAYER_S_RG` parameter. Default 1.25, the midpoint of
        polnet's own default range (0.5, 2.0) (an earlier default of 2.0
        sat at that range's blurriest end, not a representative value).
    membrane_scale_range : tuple of float, optional
        ``(low, high)``: a single random multiplicative scale, drawn
        uniformly from this range once per `generate()` call (seeded from
        `seed`, reproducible), applied to the calibrated bilayer peak
        amplitude BEFORE `build_analytic_bilayer_profile` -- an
        augmentation knob for varying membrane contrast/intensity across
        many generated instances. Applied to the amplitude that feeds
        `self.profile`, not as a separate post-hoc multiply on
        `self.volume` -- that keeps `_insert_blend`'s occupancy threshold
        (itself derived from `self.profile.psi.max()`) automatically
        consistent with whatever scale was drawn, with no extra
        bookkeeping. The drawn value is recorded in `self.membrane_scale`.
        Default ``(0.5, 1.0)``; pass ``(1.0, 1.0)`` for no randomization
        (always scales by exactly 1).
    transmembrane_specs : list of TransmembraneSpec, optional
        Transmembrane protein species to attempt placing. Default None (no
        transmembrane proteins).
    transmembrane_occupancy_fraction : float, optional
        Controls how transmembrane protein density is combined with the
        membrane's own density at each placement, in `_insert_blend`. A
        real transmembrane protein displaces lipid where it sits rather
        than coexisting with it, so naively adding the two (the previous
        behaviour) double-counts density in the overlap and is physically
        wrong. Instead, wherever the (already-rendered) protein template's
        own density exceeds `1.5 * transmembrane_occupancy_fraction *
        self.profile.psi.max()` (the bilayer's own calibrated peak
        density -- an already-computed, physically anchored scale, rather
        than an arbitrary constant), the membrane's density there is fully
        replaced by the protein's instead of summed with it; below `0.5 *`
        that same threshold, the membrane is left completely untouched. In
        between, a smoothstep (not a sigmoid -- a sigmoid's tail never
        truly reaches 0, which left a template's near-zero background,
        e.g. `rotate_volume`'s own zero-padding, weakly but non-negligibly
        suppressing membrane density across the whole inserted bounding
        box rather than just near the protein's actual mass) blends the
        two, avoiding a visible cutout edge at the template's own decaying
        density tail while still saturating to exactly 0/1 away from it.
        See `_insert_blend`'s own implementation for the exact band. This
        reuses the
        template density already computed for insertion as a proxy for
        occupancy rather than building a separate excluded-volume/solvent-
        accessible-surface geometry step from the atom coordinates -- a
        deliberately simpler approximation: the "hole" boundary tracks
        wherever the atomic potential happens to decay past this
        threshold, not a true van-der-Waals silhouette. Default 0.05.
    pdb_cache_dir : str, optional
        Passed to `PDB` for PDB-backed transmembrane specs. Default is
        `specter.pdb.DEFAULT_PDB_SAVEFOLDER` (`specter-data/pdb`, relative to
        the caller's cwd; see `config.default_pdb_cache_dir`).
    max_field_voxels : int, optional
        Safety cap on the working field grid's total voxel count, AND (see
        `max_output_voxels` below for the distinction) on the resolution
        `generate()` actually GENERATES the output raster at. The field's
        own spacing is derived to resolve the bilayer's physical extent
        (see `generate`'s inline comment) independent of `target_shape_zyx`
        -- fine at the small scales this was built and tested at, but that
        spacing applied UNIFORMLY across a real production-scale volume
        (hundreds of voxels per axis) produces a working grid orders of
        magnitude larger than `target_shape_zyx` itself: confirmed a (200,
        600, 600)-voxel, 10 A/voxel target implying a 500x1500x1500 (~1.1
        billion voxel) field, ballooning to 50+ GB of resident memory
        across the several such arrays field generation allocates, well
        past what's practical. If the naive spacing would exceed this
        budget, it's coarsened (increased) just enough to fit, with a
        warning -- a real, disclosed accuracy tradeoff (a coarser-than-
        ideal field resolves the bilayer's own sub-structure less
        crisply), not a silent one; this is a stopgap, not a fix for the
        underlying issue (the field is one uniform grid over the WHOLE
        domain regardless of how much of it is actually near a membrane
        surface -- a real architectural limitation, worth a proper
        adaptive/local-patch redesign later, not attempted here).

        Separately, if `target_shape_zyx` (explicit or auto-sized from
        organelle size/`v_size`) itself exceeds this many voxels,
        `generate()` GENERATES the field/raster on a coarser grid of
        exactly this budget -- covering the SAME physical extent as
        `target_shape_zyx`, just at coarser spacing -- and upsamples onto
        the full-resolution `target_shape_zyx` afterward via
        `_chunked_upsample_density` (trilinear point-sampling in Z-chunks,
        not a single `F.interpolate` call -- both `F.interpolate` and
        plain `F.grid_sample` hit a hard CUDA kernel "invalid
        configuration argument" past ~2^31 total elements, confirmed
        directly, regardless of available memory; chunking sidesteps it).
        This decouples the EXPENSIVE part (dense field generation, ~14x a
        single array's own size in peak memory, confirmed directly) from
        the requested output resolution: a large, fine-`v_size` organelle
        that would otherwise need hundreds of GB to generate directly can
        instead generate cheaply at this budget and upsample for close to
        just the final array's own size (each upsample chunk is a small,
        bounded point-sampling call -- see `_UPSAMPLE_CHUNK_VOXELS`). The
        organelle's PHYSICAL size is preserved exactly -- only its
        resolved sub-structure crispness is traded away, same as the
        field-coarsening tradeoff above. See `max_output_voxels` for the
        separate, much larger ceiling on the upsampled FINAL array itself.
        Default 200_000_000 (~800 MB at float32 per array).
    max_output_voxels : int, optional
        Hard ceiling on `target_shape_zyx`'s own total voxel count -- i.e.
        on the size of the single dense array `generate()` must ultimately
        materialize as `self.volume` (post-upsample, see `max_field_voxels`
        above), regardless of how cheaply generation itself is made via
        the coarse-then-upsample mechanism. When `target_shape_zyx` is
        auto-sized (omitted), exceeding this SHRINKS the organelle's own
        physical size to fit (same mechanism, and same warning style, as
        `max_field_voxels` used to apply directly before generation-
        resolution decoupling existed) -- this only fires for genuinely
        extreme requests, since `max_field_voxels`-based decoupling above
        already handles the common case of "fits fine once materialized,
        just too expensive to generate directly" without shrinking
        anything. When `target_shape_zyx` is given explicitly and exceeds
        this budget, raises instead of silently resizing what the caller
        explicitly asked for. Default 4_000_000_000 (~16 GB at float32 --
        comfortably fits a 24+ GB GPU; raise for a larger card, e.g. a
        40 GB budget comfortably covers several billion voxels once
        `max_field_voxels` is keeping generation itself cheap).
    device : str or torch.device, optional
        Device for generation. Default "cpu".
    seed : int, optional
        Random seed. Default None.
    render_workers : int or "auto", optional
        Number of `transmembrane_specs` species rendered concurrently (on
        background threads) when building each species' `PotentialBuilder`
        template -- see `_build_template`. Only matters when
        `transmembrane_specs` has more than one entry AND none of them
        already supply their own `TransmembraneSpec.template` (those are
        never rebuilt regardless of this setting). Default 1: fully serial,
        identical to the pre-parallel behaviour. `"auto"` resolves via
        `recommend_render_workers(len(transmembrane_specs))` -- see that
        function's own docstring.
    render_devices : list of str or torch.device, optional
        Device pool to round-robin those concurrent species across (e.g.
        multiple GPUs). Default None: every species renders on `device`
        above, still concurrently across `render_workers` threads, just not
        spread across multiple physical devices.

    References
    ----------
    Martinez-Sanchez, A., Lamm, L., Jasnin, M., & Phelippeau, H. (2024). Simulating
    the cellular context in synthetic datasets for cryo-electron tomography. IEEE
    Transactions on Medical Imaging, 43(11), 3742–3754.
    https://doi.org/10.1109/TMI.2024.3398401
    polnet source: https://github.com/anmartinezs/polnet
    """

    def __init__(
        self,
        target_shape_zyx: tuple[int, int, int] | None = None,
        v_size: float = 5.0,
        shape_backend: str = "spherical_harmonics",
        sh_max_degree: int = 8,
        sh_axes_a: tuple[float, float, float] | None = None,
        sh_axes_range_a: tuple[float, float] = (150.0, 450.0),
        sh_amplitude: float = 0.15,
        sh_spectrum_power: float = 2.0,
        swept_total_length_a: float | None = None,
        swept_total_length_range_a: tuple[float, float] = (1500.0, 2500.0),
        swept_step_length_a: float | None = None,
        swept_tube_radius_a: float | None = None,
        swept_tube_radius_range_a: tuple[float, float] = (150.0, 400.0),
        swept_flexibility: float | None = None,
        swept_flexibility_range: tuple[float, float] = (0.08, 0.25),
        swept_radius_variation: float | None = None,
        swept_radius_variation_range: tuple[float, float] = (0.1, 0.3),
        swept_radius_variation_sigma_points: float = 2.0,
        swept_blend_sharpness_a: float | None = None,
        swept_path_smoothing_sigma_points: float = 1.5,
        swept_curvature_iterations: int = 15,
        swept_curvature_step_fraction: float = 0.15,
        n_lipids_per_leaflet: int = 200,
        parameterization: str = "shtyrov",
        bilayer_thickness_a: float = 30.0,
        bilayer_layer_sigma_a: float = 1.25,
        membrane_scale_range: tuple[float, float] = (0.5, 1.0),
        transmembrane_specs: list[TransmembraneSpec] | None = None,
        transmembrane_occupancy_fraction: float = 0.05,
        pdb_cache_dir: str = DEFAULT_PDB_SAVEFOLDER,
        max_field_voxels: int = 200_000_000,
        max_output_voxels: int = 4_000_000_000,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        render_workers: int | Literal["auto"] = 1,
        render_devices: list[str | torch.device] | None = None,
    ):
        if shape_backend not in ("spherical_harmonics", "swept_spline"):
            raise ValueError(
                "shape_backend must be 'spherical_harmonics' or "
                f"'swept_spline', got {shape_backend!r}"
            )
        low, high = membrane_scale_range
        if low > high or low < 0:
            raise ValueError(
                "membrane_scale_range must be (low, high) with "
                f"0 <= low <= high, got {membrane_scale_range!r}"
            )

        v_size = float(v_size)

        # Resolve any None size parameter from its own *_range_a default,
        # via a torch.Generator seeded independently from `seed` -- same
        # pattern membrane_scale_range's own draw uses below, and a
        # separate Generator object from whatever RNG the low-level shape
        # backend function uses internally (so this draw doesn't consume
        # that stream). Resolved unconditionally, regardless of
        # shape_backend, rather than only for the active backend: cheap,
        # and keeps every one of these attributes a plain concrete value
        # afterward instead of a backend-conditional Optional leaking into
        # every later read site (including target_shape_zyx auto-sizing
        # immediately below, which needs concrete values to work with).
        meta_rng = torch.Generator(device="cpu")
        if seed is not None:
            meta_rng.manual_seed(seed)

        if sh_axes_a is None:
            lo, hi = _resolve_range("sh_axes_range_a", sh_axes_range_a, min_low=1e-6)
            sh_axes_a = (
                _draw_uniform(meta_rng, lo, hi),
                _draw_uniform(meta_rng, lo, hi),
                _draw_uniform(meta_rng, lo, hi),
            )
        else:
            # Normalize to a tuple even when given one already -- a caller
            # passing a plain list (e.g. every TOML-sourced config value,
            # since TOML arrays decode to Python lists) would otherwise make
            # the clamp check below's `clamped != sh_axes_a` compare a tuple
            # against a list and spuriously fire even when nothing needed
            # clamping (tuple != list in Python regardless of contents).
            sh_axes_a = (float(sh_axes_a[0]), float(sh_axes_a[1]), float(sh_axes_a[2]))
        if swept_tube_radius_a is None:
            lo, hi = _resolve_range(
                "swept_tube_radius_range_a", swept_tube_radius_range_a, min_low=1e-6
            )
            swept_tube_radius_a = _draw_uniform(meta_rng, lo, hi)
        if swept_total_length_a is None:
            lo, hi = _resolve_range(
                "swept_total_length_range_a", swept_total_length_range_a, min_low=1e-6
            )
            swept_total_length_a = _draw_uniform(meta_rng, lo, hi)
        if swept_flexibility is None:
            lo, hi = _resolve_range(
                "swept_flexibility_range", swept_flexibility_range, min_low=1e-6
            )
            if hi > 1.0:
                raise ValueError(
                    "swept_flexibility_range must have high <= 1, got "
                    f"{swept_flexibility_range!r}"
                )
            swept_flexibility = _draw_uniform(meta_rng, lo, hi)
        if swept_radius_variation is None:
            lo, hi = _resolve_range(
                "swept_radius_variation_range",
                swept_radius_variation_range,
                min_low=0.0,
            )
            swept_radius_variation = _draw_uniform(meta_rng, lo, hi)
        if swept_step_length_a is None:
            # Half the (now-resolved) tube radius -- always comfortably
            # under the 2*tube_radius_a beading threshold regardless of
            # which value got drawn, unlike a fixed absolute default tuned
            # for one specific radius (see generate_membrane_field_swept_
            # spline's own beading-risk docstring).
            swept_step_length_a = 0.5 * swept_tube_radius_a

        # target_shape_zyx: auto-size from the now-resolved organelle size
        # when omitted, so a casual caller never has to compute a working
        # grid by hand; clamp the organelle size to fit when an explicit
        # target_shape_zyx IS given, so a too-large drawn/explicit size can
        # never silently clip (MembraneField.clipped_at_boundary/each shape
        # backend's own boundary warning are the last-resort safety net,
        # not the primary defense).
        if target_shape_zyx is None:
            if shape_backend == "spherical_harmonics":
                safe_half_extent_a = max(sh_axes_a) / _SIZE_MARGIN_FRACTION
            else:
                safe_half_extent_a = (
                    0.5 * swept_total_length_a + swept_tube_radius_a
                ) / _SIZE_MARGIN_FRACTION
            n = max(1, math.ceil(2.0 * safe_half_extent_a / v_size))
            # This OUTPUT canvas (what becomes self.volume) is a SEPARATE
            # concern from the internal working field max_field_voxels
            # already protects (generate()'s own field_spacing_a
            # coarsening) -- and, since generation-resolution decoupling
            # was added (see max_field_voxels' own docstring), a large n
            # here is now normally absorbed by generating at a coarser
            # grid and upsampling, WITHOUT shrinking the organelle at all.
            # max_output_voxels below is the last-resort fallback for when
            # even the upsampled FINAL array (materialized via chunked
            # point-sampling, not one giant call -- see
            # _chunked_upsample_density) still wouldn't fit -- SHRINKS the
            # organelle's own physical size to fit, the same mechanism the
            # explicit-target_shape_zyx branch below already uses.
            if n**3 > max_output_voxels:
                n_capped = max(1, round(max_output_voxels ** (1.0 / 3.0)))
                scale = n_capped / n
                if shape_backend == "spherical_harmonics":
                    new_sh_axes_a = (
                        sh_axes_a[0] * scale,
                        sh_axes_a[1] * scale,
                        sh_axes_a[2] * scale,
                    )
                    warnings.warn(
                        f"MembraneGenerator: sh_axes_a {sh_axes_a} at "
                        f"v_size={v_size:.2f} A implies a {n}^3 output canvas, "
                        f"exceeding max_output_voxels ({max_output_voxels:,}) -- "
                        f"scaled down by {scale:.2f}x (to "
                        f"{tuple(round(a, 1) for a in new_sh_axes_a)}) to avoid "
                        "an OOM materializing the final array. Raise "
                        "max_output_voxels if you have the memory to spare, "
                        "increase v_size, or set sh_axes_a explicitly to get "
                        "the originally requested size.",
                        stacklevel=2,
                    )
                    sh_axes_a = new_sh_axes_a
                else:
                    new_total_length_a = swept_total_length_a * scale
                    new_tube_radius_a = swept_tube_radius_a * scale
                    warnings.warn(
                        "MembraneGenerator: swept_total_length_a/"
                        f"swept_tube_radius_a ({swept_total_length_a:.1f} A/"
                        f"{swept_tube_radius_a:.1f} A) at v_size={v_size:.2f} A "
                        f"imply a {n}^3 output canvas, exceeding "
                        f"max_output_voxels ({max_output_voxels:,}) -- scaled "
                        f"both down by {scale:.2f}x (to {new_total_length_a:.1f} "
                        f"A/{new_tube_radius_a:.1f} A) to avoid an OOM "
                        "materializing the final array. Raise max_output_voxels "
                        "if you have the memory to spare, or increase v_size, "
                        "to get the originally requested size.",
                        stacklevel=2,
                    )
                    swept_total_length_a = new_total_length_a
                    swept_tube_radius_a = new_tube_radius_a
                    if swept_step_length_a > swept_tube_radius_a:
                        swept_step_length_a = 0.5 * swept_tube_radius_a
                n = n_capped
            target_shape_zyx = (n, n, n)
        else:
            tz, ty, tx = target_shape_zyx
            box_extent_a = (tx * v_size, ty * v_size, tz * v_size)
            safe_half_extent_a = _SIZE_MARGIN_FRACTION * min(box_extent_a) / 2.0
            if safe_half_extent_a < _MIN_SAFE_HALF_EXTENT_A:
                raise ValueError(
                    f"MembraneGenerator: target_shape_zyx={target_shape_zyx!r} at "
                    f"v_size={v_size:.2f} A/voxel gives a box too small (safe "
                    f"half-extent {safe_half_extent_a:.1f} A) to hold any "
                    f"reasonably-sized {shape_backend!r} organelle -- increase "
                    "target_shape_zyx or v_size."
                )
            if shape_backend == "spherical_harmonics":
                clamped = (
                    min(sh_axes_a[0], safe_half_extent_a),
                    min(sh_axes_a[1], safe_half_extent_a),
                    min(sh_axes_a[2], safe_half_extent_a),
                )
                if clamped != sh_axes_a:
                    warnings.warn(
                        f"MembraneGenerator: sh_axes_a {sh_axes_a} exceeds what "
                        f"target_shape_zyx={target_shape_zyx!r}/v_size={v_size:.2f} "
                        f"can safely hold -- clamped to "
                        f"{tuple(round(c, 1) for c in clamped)} to avoid clipping. "
                        "Increase target_shape_zyx/v_size, or set sh_axes_a "
                        "explicitly to a smaller value, to get the originally "
                        "requested size.",
                        stacklevel=2,
                    )
                    sh_axes_a = clamped
            else:
                reach = 0.5 * swept_total_length_a + swept_tube_radius_a
                if reach > safe_half_extent_a:
                    scale = safe_half_extent_a / reach
                    new_total_length_a = swept_total_length_a * scale
                    new_tube_radius_a = swept_tube_radius_a * scale
                    warnings.warn(
                        "MembraneGenerator: swept_total_length_a/"
                        f"swept_tube_radius_a ({swept_total_length_a:.1f} A/"
                        f"{swept_tube_radius_a:.1f} A) exceed what "
                        f"target_shape_zyx={target_shape_zyx!r}/v_size={v_size:.2f} "
                        f"can safely hold -- scaled both down by {scale:.2f}x (to "
                        f"{new_total_length_a:.1f} A/{new_tube_radius_a:.1f} A) to "
                        "avoid clipping. Increase target_shape_zyx/v_size to get "
                        "the originally requested size.",
                        stacklevel=2,
                    )
                    swept_total_length_a = new_total_length_a
                    swept_tube_radius_a = new_tube_radius_a
                    if swept_step_length_a > swept_tube_radius_a:
                        swept_step_length_a = 0.5 * swept_tube_radius_a

        tz, ty, tx = target_shape_zyx
        self.target_shape_zyx: tuple[int, int, int] = (int(tz), int(ty), int(tx))
        n_out_voxels = (
            self.target_shape_zyx[0]
            * self.target_shape_zyx[1]
            * self.target_shape_zyx[2]
        )
        if n_out_voxels > max_output_voxels:
            # Only reachable for an EXPLICIT target_shape_zyx -- the auto-
            # sizing branch above already shrinks the organelle so its own
            # n**3 satisfies max_output_voxels, so n_out_voxels can't
            # exceed it there. An explicit shape is a direct caller
            # request, so raise rather than silently resize it.
            raise ValueError(
                f"MembraneGenerator: target_shape_zyx={self.target_shape_zyx!r} "
                f"({n_out_voxels:,} voxels) exceeds max_output_voxels "
                f"({max_output_voxels:,}) -- this many voxels can't be safely "
                "materialized as one dense output array regardless of "
                "generation-resolution decoupling (see max_output_voxels' own "
                "docstring). Raise max_output_voxels, or reduce "
                "target_shape_zyx/v_size."
            )
        # Decouple GENERATION resolution from the fine v_size the output
        # canvas above must end up at -- see max_field_voxels' own
        # docstring for the coarse-then-upsample mechanism and its
        # measured cost. Physical extent is preserved exactly (same
        # origin, same span); only the requested sub-structure crispness
        # is traded away when this triggers.
        if n_out_voxels > max_field_voxels:
            gen_scale = (max_field_voxels / n_out_voxels) ** (1.0 / 3.0)
            self._gen_shape_zyx: tuple[int, int, int] = (
                max(1, round(self.target_shape_zyx[0] * gen_scale)),
                max(1, round(self.target_shape_zyx[1] * gen_scale)),
                max(1, round(self.target_shape_zyx[2] * gen_scale)),
            )
            self._gen_v_size = v_size / gen_scale
            self._needs_upsample = True
            warnings.warn(
                f"MembraneGenerator: target_shape_zyx={self.target_shape_zyx!r} "
                f"({n_out_voxels:,} voxels) at v_size={v_size:.2f} A exceeds "
                f"max_field_voxels ({max_field_voxels:,}) -- generating on a "
                f"coarser {self._gen_shape_zyx!r} grid at "
                f"{self._gen_v_size:.2f} A/voxel instead, then upsampling "
                "(trilinear) to the full requested resolution. The bilayer's "
                "own sub-structure will be resolved less crisply than at full "
                "resolution; raise max_field_voxels if you have the memory to "
                "spare.",
                stacklevel=2,
            )
        else:
            self._gen_shape_zyx = self.target_shape_zyx
            self._gen_v_size = v_size
            self._needs_upsample = False
        self.v_size = v_size
        self.max_output_voxels = max_output_voxels
        self.shape_backend = shape_backend
        self.sh_max_degree = sh_max_degree
        self.sh_axes_a = sh_axes_a
        self.sh_axes_range_a = sh_axes_range_a
        self.sh_amplitude = sh_amplitude
        self.sh_spectrum_power = sh_spectrum_power
        self.swept_total_length_a = swept_total_length_a
        self.swept_total_length_range_a = swept_total_length_range_a
        self.swept_step_length_a = swept_step_length_a
        self.swept_tube_radius_a = swept_tube_radius_a
        self.swept_tube_radius_range_a = swept_tube_radius_range_a
        self.swept_flexibility = swept_flexibility
        self.swept_flexibility_range = swept_flexibility_range
        self.swept_radius_variation = swept_radius_variation
        self.swept_radius_variation_range = swept_radius_variation_range
        self.swept_radius_variation_sigma_points = swept_radius_variation_sigma_points
        self.swept_blend_sharpness_a = swept_blend_sharpness_a
        self.swept_path_smoothing_sigma_points = swept_path_smoothing_sigma_points
        self.swept_curvature_iterations = swept_curvature_iterations
        self.swept_curvature_step_fraction = swept_curvature_step_fraction
        self.n_lipids_per_leaflet = n_lipids_per_leaflet
        self.parameterization = parameterization
        self.bilayer_thickness_a = bilayer_thickness_a
        self.bilayer_layer_sigma_a = bilayer_layer_sigma_a
        self.membrane_scale_range = membrane_scale_range
        self.transmembrane_specs = transmembrane_specs or []
        self.transmembrane_occupancy_fraction = transmembrane_occupancy_fraction
        self.pdb_cache_dir = pdb_cache_dir
        self.max_field_voxels = max_field_voxels
        self.device = torch.device(device)
        self.seed = seed
        self.render_workers = resolve_render_workers(
            render_workers, len(self.transmembrane_specs)
        )
        self.render_devices = resolve_render_devices(self.device, render_devices)

        self.field: MembraneField | None = None
        self.profile: BilayerProfile | None = None
        self.volume: torch.Tensor | None = None
        self.placements: list[TransmembranePlacement] = []
        self.membrane_scale: float | None = None
        self.clipped_at_boundary: bool | None = None
        self._origin_xyz: torch.Tensor | None = None

    def generate(self) -> torch.Tensor:
        """
        Build the calibrated bilayer profile and organic field, and
        rasterize into ``self.volume``.

        Sets ``self.clipped_at_boundary`` (from the underlying
        ``MembraneField``'s own flag): ``True`` if the organelle's solid
        interior touched the working grid's edge -- an unphysical flat-cut
        truncation, not a subtle issue -- rather than only firing a
        warning. Callers compositing many instances (e.g.
        ``MembraneTomogramGenerator``) check this to drop a visibly-clipped
        instance instead of compositing it.

        Returns
        -------
        torch.Tensor
            Density volume, shape ``target_shape_zyx``.
        """
        atomic_numbers, coordinates = build_reference_lipid_patch(
            n_lipids_per_leaflet=self.n_lipids_per_leaflet,
            seed=self.seed,
            device=self.device,
        )
        peak_amplitude = estimate_bilayer_peak_amplitude(
            atomic_numbers, coordinates, parameterization=self.parameterization
        )

        # Random per-instance contrast augmentation: drawn from
        # membrane_scale_range and folded into the amplitude BEFORE
        # build_analytic_bilayer_profile, not applied as a separate
        # post-hoc multiply on self.volume -- that keeps
        # _insert_blend's occupancy threshold (itself derived from
        # self.profile.psi.max()) automatically consistent with
        # whatever scale gets drawn here, with no extra bookkeeping.
        scale_generator = torch.Generator(device="cpu")
        if self.seed is not None:
            scale_generator.manual_seed(self.seed)
        low, high = self.membrane_scale_range
        self.membrane_scale = float(
            torch.empty(1).uniform_(low, high, generator=scale_generator).item()
        )
        peak_amplitude *= self.membrane_scale

        self.profile = build_analytic_bilayer_profile(
            thickness_a=self.bilayer_thickness_a,
            layer_sigma_a=self.bilayer_layer_sigma_a,
            amplitude=peak_amplitude,
            device=self.device,
        )

        # The field's working spacing must stay fine enough to resolve the
        # bilayer's own extent (curvature capping, leaflet offset) --
        # derived from the profile's actual rendered table rather than a
        # hardcoded thickness, so this stays correct if the profile's
        # template is retuned later.
        half_extent_a = float(self.profile.distance_a.abs().max())
        # Keyed to _gen_v_size (the resolution generate() actually renders
        # at), not the fine self.v_size the output ends up at post-
        # upsample -- resolving the field finer than the raster it feeds
        # would buy nothing once generation-resolution decoupling is
        # already coarsening that raster (see max_field_voxels' own
        # docstring). Equal to self.v_size whenever decoupling didn't
        # trigger, matching the pre-decoupling behaviour exactly.
        field_spacing_a = min(self._gen_v_size, half_extent_a / 8)

        nz, ny, nx = self.target_shape_zyx
        extent_a = torch.tensor([nx, ny, nz], dtype=torch.float32) * self.v_size
        self._origin_xyz = -0.5 * extent_a

        field_shape_zyx = (
            int(torch.ceil(extent_a[2] / field_spacing_a)),
            int(torch.ceil(extent_a[1] / field_spacing_a)),
            int(torch.ceil(extent_a[0] / field_spacing_a)),
        )
        n_field_voxels = field_shape_zyx[0] * field_shape_zyx[1] * field_shape_zyx[2]
        if n_field_voxels > self.max_field_voxels:
            # coarsen just enough to fit max_field_voxels (voxel count
            # scales as 1/spacing^3) -- see max_field_voxels' own
            # docstring for why this is needed and what it trades away.
            field_spacing_a *= (n_field_voxels / self.max_field_voxels) ** (1.0 / 3.0)
            field_shape_zyx = (
                int(torch.ceil(extent_a[2] / field_spacing_a)),
                int(torch.ceil(extent_a[1] / field_spacing_a)),
                int(torch.ceil(extent_a[0] / field_spacing_a)),
            )
            warnings.warn(
                f"MembraneGenerator: the working field grid at the resolution "
                f"needed to fully resolve the bilayer ({n_field_voxels:,} voxels) "
                f"exceeds max_field_voxels ({self.max_field_voxels:,}) -- "
                f"coarsened field_spacing_a to {field_spacing_a:.2f} A "
                f"({field_shape_zyx[2]}x{field_shape_zyx[1]}x{field_shape_zyx[0]} "
                "voxels). The bilayer's own sub-structure will be resolved less "
                "crisply than at full resolution; raise max_field_voxels if you "
                "have the memory to spare.",
                stacklevel=2,
            )
        if self.shape_backend == "spherical_harmonics":
            self.field = generate_membrane_field_spherical_harmonics(
                shape_zyx=field_shape_zyx,
                spacing_a=field_spacing_a,
                sh_max_degree=self.sh_max_degree,
                sh_axes_a=self.sh_axes_a,
                sh_amplitude=self.sh_amplitude,
                sh_spectrum_power=self.sh_spectrum_power,
                device=self.device,
                seed=self.seed,
            )
        else:
            self.field = generate_membrane_field_swept_spline(
                shape_zyx=field_shape_zyx,
                spacing_a=field_spacing_a,
                total_length_a=self.swept_total_length_a,
                step_length_a=self.swept_step_length_a,
                tube_radius_a=self.swept_tube_radius_a,
                flexibility=self.swept_flexibility,
                radius_variation=self.swept_radius_variation,
                radius_variation_sigma_points=self.swept_radius_variation_sigma_points,
                blend_sharpness_a=self.swept_blend_sharpness_a,
                path_smoothing_sigma_points=self.swept_path_smoothing_sigma_points,
                curvature_iterations=self.swept_curvature_iterations,
                curvature_step_fraction=self.swept_curvature_step_fraction,
                device=self.device,
                seed=self.seed,
            )

        self.clipped_at_boundary = self.field.clipped_at_boundary

        self.volume = rasterize_membrane_density(
            self.field,
            self.profile,
            target_shape_zyx=self._gen_shape_zyx,
            target_spacing_a=self._gen_v_size,
            target_origin_xyz=self._origin_xyz,
        )
        if self._needs_upsample:
            # _gen_shape_zyx/_gen_v_size cover the SAME physical extent as
            # target_shape_zyx/v_size, anchored at the same origin (see
            # __init__) -- see _chunked_upsample_density's own docstring
            # for why this goes through MembraneField.sample() in chunks
            # rather than a single F.interpolate call.
            self.volume = _chunked_upsample_density(
                self.volume,
                self._gen_v_size,
                self._origin_xyz,
                self.target_shape_zyx,
                self.v_size,
            )
        return self.volume

    def place_transmembrane(
        self, min_spacing_a: float = 40.0, max_attempts: int | None = None
    ) -> list[TransmembranePlacement]:
        """
        Sample surface sites and insert transmembrane protein templates.

        Must be called after :meth:`generate`. Species are chosen per site
        by weighted random draw (weights = ``frequency``), the same
        weighted-selection idea `specimen.tomogram.
        MembraneTomogramGenerator`'s own ratio-mode filler species use
        (there via `draw_species_pool`).

        Parameters
        ----------
        min_spacing_a : float, optional
            Minimum center-to-center spacing between placed sites,
            Angstrom. Default 40.0.
        max_attempts : int, optional
            Passed to
            :func:`~specter.specimen.membrane._placement.sample_surface_sites`.
            Default is that function's own default.

        Returns
        -------
        list of TransmembranePlacement
        """
        if self.field is None or self.volume is None or self._origin_xyz is None:
            raise RuntimeError("call generate() before place_transmembrane()")
        if not self.transmembrane_specs:
            return []

        n_sites = sum(spec.frequency for spec in self.transmembrane_specs)
        sites_xyz, normals_xyz = sample_surface_sites(
            self.field,
            n_sites=n_sites,
            min_spacing_a=min_spacing_a,
            max_attempts=max_attempts,
            seed=self.seed,
        )
        n_found = sites_xyz.shape[0]
        if n_found < n_sites:
            severity = "zero" if n_found == 0 else "only"
            warnings.warn(
                f"MembraneGenerator.place_transmembrane: {severity} {n_found}/{n_sites} "
                "requested transmembrane sites were found -- sample_surface_sites' "
                "Newton-projection surface search exhausted max_attempts before "
                "reaching the target count. Common causes: the membrane's surface "
                "area is too small for this many well-spaced sites at this "
                "min_spacing_a (try reducing min_spacing_a or n_sites/frequency), "
                "or -- shape_backend='spherical_harmonics' specifically -- the "
                "working field grid is too coarse relative to sh_axes_a for "
                "reliable surface projection (see "
                "generate_membrane_field_spherical_harmonics's own Notes; that "
                "function also warns proactively on this). This is not "
                "raised as an error since a partial/empty result is sometimes "
                "intended (e.g. deliberately testing a too-small membrane), but "
                f"placements will be missing/absent if not: {n_found} of "
                f"{n_sites} transmembrane instances will be placed.",
                stacklevel=2,
            )

        # Keyed by index, not pdb_source, while building (a duplicate
        # pdb_source across specs is legal -- the original serial dict
        # comprehension just let the later one silently win); remapped to
        # pdb_source -> template below to preserve that exact "last spec
        # wins" behaviour for lookups in the placement loop further down.
        index_templates = build_templates_concurrently(
            keys=list(range(len(self.transmembrane_specs))),
            build_one=lambda i, device: self._build_template(
                self.transmembrane_specs[i], device
            ),
            devices=self.render_devices,
            max_workers=self.render_workers,
        )
        templates = {
            spec.pdb_source: index_templates[i]
            for i, spec in enumerate(self.transmembrane_specs)
        }
        weights = torch.tensor(
            [spec.frequency for spec in self.transmembrane_specs], dtype=torch.float32
        )
        chooser = torch.Generator(device="cpu")
        if self.seed is not None:
            chooser.manual_seed(self.seed)

        placements: list[TransmembranePlacement] = []
        for i in range(sites_xyz.shape[0]):
            chosen_idx = int(torch.multinomial(weights, 1, generator=chooser).item())
            spec = self.transmembrane_specs[chosen_idx]
            template = templates[spec.pdb_source]

            site_seed = None if self.seed is None else self.seed + i
            rotation = orientation_for_normal(normals_xyz[i], seed=site_seed)
            theta = build_affine_matrix(rotation.to(self.device))
            rotated = rotate_volume(template, theta, padding_mode="zeros")[0]

            center_zyx = self._physical_to_voxel_index(sites_xyz[i])
            self._insert_blend(rotated, center_zyx)

            placements.append(
                TransmembranePlacement(
                    species_id=spec.pdb_source,
                    center_xyz=sites_xyz[i].detach().cpu(),
                    rotation_matrix=rotation.detach().cpu(),
                )
            )

        self.placements = placements
        return placements

    def _build_template(
        self, spec: TransmembraneSpec, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Build (or reuse) one transmembrane species' potential template.

        Parameters
        ----------
        spec : TransmembraneSpec
            Species to render, or reuse via `spec.template` if supplied.
        device : torch.device, optional
            Device to build the `PotentialBuilder` (and run its `forward`)
            on -- lets concurrent callers (`build_templates_concurrently`)
            spread species across `self.render_devices` instead of
            serializing everything onto `self.device`. Default None: use
            `self.device`, matching pre-parallel behaviour exactly. Either
            way, the returned template always ends up on `self.device`
            (this method's only externally-visible contract), since that's
            what every downstream consumer -- `_insert_blend` chief among
            them -- assumes.
        """
        if spec.template is not None:
            return spec.template.detach().to(self.device, dtype=torch.float32)

        build_device = self.device if device is None else device
        return render_transmembrane_template(
            spec, self.v_size, self.pdb_cache_dir, build_device
        ).to(self.device)

    def _physical_to_voxel_index(self, site_xyz: torch.Tensor) -> torch.Tensor:
        assert self._origin_xyz is not None
        origin = self._origin_xyz.cpu()
        idx_xyz = (site_xyz.detach().cpu() - origin) / self.v_size
        idx_zyx = torch.stack([idx_xyz[2], idx_xyz[1], idx_xyz[0]])
        return idx_zyx.round().long()

    def _insert_blend(
        self, local_density: torch.Tensor, center_zyx: torch.Tensor
    ) -> None:
        """
        Insert a transmembrane protein template, replacing (not adding to)
        the membrane's own density wherever the template is occupied -- see
        `transmembrane_occupancy_fraction`'s own docstring for why plain
        addition double-counts density and how this threshold is chosen.
        """
        assert self.volume is not None
        assert self.profile is not None
        bounds = clip_insert_bounds(
            center_zyx.tolist(), local_density.shape, self.volume.shape
        )
        if bounds is None:
            return
        dst, src = bounds
        protein = local_density[src].to(self.volume.device)

        threshold = self.transmembrane_occupancy_fraction * float(
            self.profile.psi.max()
        )
        if threshold <= 0:
            self.volume[dst] += protein
            return

        # Smoothstep over a band centered on `threshold`, not an
        # unbounded-tail sigmoid: a sigmoid never truly reaches 0, so a
        # template's near-zero background (e.g. rotate_volume's own
        # zero-padding, or a real potential's slowly-decaying tail) still
        # picks up a small but nonzero weight EVERYWHERE the template was
        # inserted, incorrectly attenuating membrane density across the
        # whole (padded) bounding box rather than just near the protein's
        # actual mass (caught directly: a synthetic single-hot-voxel
        # template made this visible as a net *decrease* in total density
        # after insertion). Smoothstep instead saturates to EXACTLY 0/1
        # outside its band, so density far from the protein is left
        # untouched.
        low, high = 0.5 * threshold, 1.5 * threshold
        t = torch.clamp((protein - low) / (high - low), 0.0, 1.0)
        weight = t * t * (3.0 - 2.0 * t)
        self.volume[dst] = (1.0 - weight) * self.volume[dst] + weight * protein


__all__ = [
    "MembraneGenerator",
    "TransmembraneSpec",
    "TransmembranePlacement",
]
