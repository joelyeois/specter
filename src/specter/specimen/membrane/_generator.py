"""
Public membrane specimen generator.

Ties together the four membrane submodules built this session into one
class: organic shape (``_field``), calibrated bilayer potential
(``_profile``), anti-aliased rasterization onto an output grid
(``_raster``), and normal-aligned transmembrane protein insertion
(``_placement``).

Everything shares one centered-origin physical coordinate frame: physical
``(0, 0, 0)`` is the volume's own center, matching the convention already
used throughout ``_field``/``_raster``. This generator
does not yet support placing a membrane instance off-center within a larger
tomogram -- that compositing step belongs to a higher-level assembler, the
same way the other specimen generators in this package are used standalone
before being composited.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch

from ...arrays import clip_insert_bounds
from ...pdb import PDB
from ...potential import PotentialBuilder
from ...rotations import build_affine_matrix, rotate_volume
from ..packing import estimate_protein_box_size
from ._field import MembraneField, generate_membrane_field
from ._field_alpha import generate_membrane_field_alpha_shape
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
    target_shape_zyx : tuple of int
        Output volume shape, ``(Z, Y, X)``.
    v_size : float
        Output voxel size, Angstrom. Also used to render transmembrane
        protein templates, so their scale matches the membrane's.
    shape_backend : {"metaball", "alpha_shape", "spherical_harmonics", "swept_spline"}, optional
        Which organic-shape algorithm builds the underlying
        :class:`~specter.specimen.membrane._field.MembraneField` --
        `"metaball"` (default, unchanged from before this parameter
        existed) blends `n_sources` isotropically-scattered spheres, which
        stays a compact blob regardless of noise/curvature tuning (verified
        by direct sweep). `"alpha_shape"` instead wraps a noisy point cloud
        in an alpha shape (a from-scratch port of CTS's own `gen_mem.m`
        algorithm, :func:`~specter.specimen.membrane._field_alpha.
        generate_membrane_field_alpha_shape`), which -- via `blob_roughness`
        -- can produce genuinely non-convex/elongated, even tube-like,
        shapes (verified by direct sweep: 0.7/0.5/0.3/0.15 goes from a
        rounded bowl to a twisted ribbon to a thin tube). `"spherical_harmonics"`
        instead perturbs an ellipsoid with a random real spherical-harmonic
        expansion (:func:`~specter.specimen.membrane.
        _field_spherical_harmonics.generate_membrane_field_spherical_harmonics`)
        -- smooth by construction (no faceting to clean up, unlike
        `"alpha_shape"`) and given a physically-motivated (Helfrich
        bending-mode) undulation spectrum, but -- being a radius function of
        direction -- restricted to STAR-CONVEX topology: good for vesicles,
        nuclei, and roughly-spherical (or ellipsoidal, via `sh_axes_a`)
        mitochondria; cannot represent branching/self-occluding shapes the
        way `"alpha_shape"` at low `blob_roughness` can. `"swept_spline"`
        instead sweeps a sphere along a smoothly wandering random path
        (:func:`~specter.specimen.membrane._field_swept_spline.
        generate_membrane_field_swept_spline`) -- an elongated, tube-shaped
        organelle (ER-tubule-like) rather than a closed blob, and unlike the
        star-convex backends it CAN self-approach/loop back near itself
        (a real topological capability the radius-function-of-direction
        backends structurally lack). All four backends are read afterward
        only through `MembraneField`'s public contract, so the calibrated
        bilayer profile and transmembrane placement below are identical
        either way -- this choice only affects the shape.
    n_sources : int, optional
        Number of blended metaball sources for the organic shape.
        `"metaball"` backend only. Default 6.
    radius_range_a : tuple of float, optional
        Source radius range, Angstrom. `"metaball"` backend only. Default
        ``(150.0, 400.0)``.
    spread_a : float, optional
        Source center spread, Angstrom. `"metaball"` backend only. Default
        is :func:`~specter.specimen.membrane._field.generate_membrane_field`'s
        own default (a quarter of the working grid's smallest extent).
    noise_amplitude_a : float, optional
        Undulation noise amplitude, Angstrom. `"metaball"` backend only.
        Default 15.0.
    correlation_length_a : float, optional
        Undulation noise correlation length, Angstrom. `"metaball"` backend
        only. Default 40.0.
    curvature_iterations : int, optional
        Curvature-capping relaxation steps. `"metaball"` backend only.
        Default 30.
    blob_size_a : float, optional
        Rough target blob radius, Angstrom (CTS's own `size`). `"alpha_shape"`
        backend only. Default 300.0.
    blob_roughness : float, optional
        In (0, 1); lower is more irregular/elongated, down to tube-like at
        very low values -- see `shape_backend`'s docstring. `"alpha_shape"`
        backend only. Default 0.3.
    blob_surface_smoothing_voxels : float, optional
        Blur-then-rethreshold smoothing of the alpha-shape's solid mask
        before the surface is extracted, in working-field-grid voxels --
        removes the sharp Delaunay-facet kinks the alpha-shape wrap
        otherwise leaves visible at any realistic (100+ A) `blob_size_a`
        (verified directly, including against CTS's own native generator
        reusing the same underlying point-cloud/alpha-shape algorithm: the
        faceting is inherent to it, not something this backend introduced).
        See `generate_membrane_field_alpha_shape`'s own docstring for why
        this is expressed in voxels rather than Angstrom. `"alpha_shape"`
        backend only. Default 2.0.
    sh_max_degree : int, optional
        Highest spherical-harmonic degree in the random surface
        perturbation -- higher values add finer surface detail. See
        `generate_membrane_field_spherical_harmonics`'s own docstring.
        `"spherical_harmonics"` backend only. Default 8.
    sh_axes_a : tuple of float, optional
        Physical semi-axes `(a_x, a_y, a_z)` of the base ellipsoid,
        Angstrom -- isotropic axes give a roughly spherical organelle,
        anisotropic axes give an elongated/flattened one.
        `"spherical_harmonics"` backend only. Default `(300.0, 300.0, 300.0)`.
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
        See `generate_membrane_field_swept_spline`'s own docstring.
        `"swept_spline"` backend only. Default 500.0.
    swept_step_length_a : float, optional
        Distance between consecutive metaball source centers along the
        path, Angstrom -- must stay well under `2 * swept_tube_radius_a` or
        the tube shows visible beading (warned about proactively).
        `"swept_spline"` backend only. Default 15.0.
    swept_tube_radius_a : float, optional
        Tube radius, Angstrom. `"swept_spline"` backend only. Default 25.0.
    swept_flexibility : float, optional
        In (0, 1]; picked from a direct visual sweep (see
        `generate_membrane_field_swept_spline`'s own docstring): 0.05 is
        nearly a straight rod, 0.35+ produces sharp hooks/loops, 0.15 gives
        a gently organic, clearly non-straight tube. `"swept_spline"`
        backend only. Default 0.15.
    swept_blend_sharpness_a : float, optional
        Smooth-min blend radius, Angstrom. Default (`None`) is
        `0.5 * swept_tube_radius_a` -- deliberately NOT the `"metaball"`
        backend's own default, which under-blends a dense chain of sources
        into visible beading. `"swept_spline"` backend only.
    swept_path_smoothing_sigma_points : float, optional
        Path-order-aware smoothing (`scipy.ndimage.gaussian_filter1d`),
        in PATH POINTS -- see `generate_membrane_field_swept_spline`'s own
        docstring for why this must be order-aware rather than spatial.
        `"swept_spline"` backend only. Default 1.5.
    swept_curvature_iterations : int, optional
        Curvature-capping relaxation steps -- reused here for the same
        reason `"metaball"` applies it by default (an analytic smooth-min
        blend can still have locally sharp concave curvature). `"swept_spline"`
        backend only. Default 15.
    swept_curvature_step_fraction : float, optional
        See :func:`~specter.specimen.membrane._field.cap_curvature`.
        `"swept_spline"` backend only. Default 0.15.
    n_lipids_per_leaflet : int, optional
        Reference lipid patch size used ONLY to calibrate the bilayer
        profile's peak amplitude against real atomic scattering physics
        (see :func:`~specter.specimen.membrane._profile.
        estimate_bilayer_peak_amplitude`) -- not its shape, see
        `bilayer_thickness_a`/`bilayer_layer_sigma_a`. Default 200.
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
    transmembrane_specs : list of TransmembraneSpec, optional
        Transmembrane protein species to attempt placing. Default None (no
        transmembrane proteins).
    pdb_cache_dir : str, optional
        Passed to `PDB` for PDB-backed transmembrane specs. Default
        "../pdb-data/".
    max_field_voxels : int, optional
        Safety cap on the working field grid's total voxel count. The
        field's own spacing is derived to resolve the bilayer's physical
        extent (see `generate`'s inline comment) independent of
        `target_shape_zyx` -- fine at the small scales this was built and
        tested at, but that spacing applied UNIFORMLY across a real
        production-scale volume (hundreds of voxels per axis) produces a
        working grid orders of magnitude larger than `target_shape_zyx`
        itself: confirmed a (200, 600, 600)-voxel, 10 A/voxel target
        implying a 500x1500x1500 (~1.1 billion voxel) field, ballooning to
        50+ GB of resident memory across the several such arrays field
        generation allocates, well past what's practical. If the naive
        spacing would exceed this budget, it's coarsened (increased) just
        enough to fit, with a warning -- a real, disclosed accuracy
        tradeoff (a coarser-than-ideal field resolves the bilayer's own
        sub-structure less crisply), not a silent one; this is a stopgap,
        not a fix for the underlying issue (the field is one uniform grid
        over the WHOLE domain regardless of how much of it is actually
        near a membrane surface -- a real architectural limitation, worth
        a proper adaptive/local-patch redesign later, not attempted here).
        Default 200_000_000 (~800 MB at float32 per array).
    device : str or torch.device, optional
        Device for generation. Default "cpu".
    seed : int, optional
        Random seed. Default None.
    """

    def __init__(
        self,
        target_shape_zyx: tuple[int, int, int],
        v_size: float,
        shape_backend: str = "metaball",
        n_sources: int = 6,
        radius_range_a: tuple[float, float] = (150.0, 400.0),
        spread_a: float | None = None,
        noise_amplitude_a: float = 15.0,
        correlation_length_a: float = 40.0,
        curvature_iterations: int = 30,
        blob_size_a: float = 300.0,
        blob_roughness: float = 0.3,
        blob_surface_smoothing_voxels: float = 2.0,
        sh_max_degree: int = 8,
        sh_axes_a: tuple[float, float, float] = (300.0, 300.0, 300.0),
        sh_amplitude: float = 0.15,
        sh_spectrum_power: float = 2.0,
        swept_total_length_a: float = 500.0,
        swept_step_length_a: float = 15.0,
        swept_tube_radius_a: float = 25.0,
        swept_flexibility: float = 0.15,
        swept_blend_sharpness_a: float | None = None,
        swept_path_smoothing_sigma_points: float = 1.5,
        swept_curvature_iterations: int = 15,
        swept_curvature_step_fraction: float = 0.15,
        n_lipids_per_leaflet: int = 200,
        parameterization: str = "shtyrov",
        bilayer_thickness_a: float = 30.0,
        bilayer_layer_sigma_a: float = 1.25,
        transmembrane_specs: list[TransmembraneSpec] | None = None,
        pdb_cache_dir: str = "../pdb-data/",
        max_field_voxels: int = 200_000_000,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ):
        if shape_backend not in (
            "metaball",
            "alpha_shape",
            "spherical_harmonics",
            "swept_spline",
        ):
            raise ValueError(
                "shape_backend must be 'metaball', 'alpha_shape', "
                "'spherical_harmonics', or 'swept_spline', got "
                f"{shape_backend!r}"
            )
        tz, ty, tx = target_shape_zyx
        self.target_shape_zyx: tuple[int, int, int] = (int(tz), int(ty), int(tx))
        self.v_size = float(v_size)
        self.shape_backend = shape_backend
        self.n_sources = n_sources
        self.radius_range_a = radius_range_a
        self.spread_a = spread_a
        self.noise_amplitude_a = noise_amplitude_a
        self.correlation_length_a = correlation_length_a
        self.curvature_iterations = curvature_iterations
        self.blob_size_a = blob_size_a
        self.blob_roughness = blob_roughness
        self.blob_surface_smoothing_voxels = blob_surface_smoothing_voxels
        self.sh_max_degree = sh_max_degree
        self.sh_axes_a = sh_axes_a
        self.sh_amplitude = sh_amplitude
        self.sh_spectrum_power = sh_spectrum_power
        self.swept_total_length_a = swept_total_length_a
        self.swept_step_length_a = swept_step_length_a
        self.swept_tube_radius_a = swept_tube_radius_a
        self.swept_flexibility = swept_flexibility
        self.swept_blend_sharpness_a = swept_blend_sharpness_a
        self.swept_path_smoothing_sigma_points = swept_path_smoothing_sigma_points
        self.swept_curvature_iterations = swept_curvature_iterations
        self.swept_curvature_step_fraction = swept_curvature_step_fraction
        self.n_lipids_per_leaflet = n_lipids_per_leaflet
        self.parameterization = parameterization
        self.bilayer_thickness_a = bilayer_thickness_a
        self.bilayer_layer_sigma_a = bilayer_layer_sigma_a
        self.transmembrane_specs = transmembrane_specs or []
        self.pdb_cache_dir = pdb_cache_dir
        self.max_field_voxels = max_field_voxels
        self.device = torch.device(device)
        self.seed = seed

        self.field: MembraneField | None = None
        self.profile: BilayerProfile | None = None
        self.volume: torch.Tensor | None = None
        self.placements: list[TransmembranePlacement] = []
        self._origin_xyz: torch.Tensor | None = None

    def generate(self) -> torch.Tensor:
        """
        Build the calibrated bilayer profile and organic field, and
        rasterize into ``self.volume``.

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
        field_spacing_a = min(self.v_size, half_extent_a / 8)

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
        if self.shape_backend == "alpha_shape":
            self.field = generate_membrane_field_alpha_shape(
                shape_zyx=field_shape_zyx,
                spacing_a=field_spacing_a,
                blob_size_a=self.blob_size_a,
                blob_roughness=self.blob_roughness,
                surface_smoothing_voxels=self.blob_surface_smoothing_voxels,
                device=self.device,
                seed=self.seed,
            )
        elif self.shape_backend == "spherical_harmonics":
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
        elif self.shape_backend == "swept_spline":
            self.field = generate_membrane_field_swept_spline(
                shape_zyx=field_shape_zyx,
                spacing_a=field_spacing_a,
                total_length_a=self.swept_total_length_a,
                step_length_a=self.swept_step_length_a,
                tube_radius_a=self.swept_tube_radius_a,
                flexibility=self.swept_flexibility,
                blend_sharpness_a=self.swept_blend_sharpness_a,
                path_smoothing_sigma_points=self.swept_path_smoothing_sigma_points,
                curvature_iterations=self.swept_curvature_iterations,
                curvature_step_fraction=self.swept_curvature_step_fraction,
                device=self.device,
                seed=self.seed,
            )
        else:
            self.field = generate_membrane_field(
                shape_zyx=field_shape_zyx,
                spacing_a=field_spacing_a,
                n_sources=self.n_sources,
                radius_range_a=self.radius_range_a,
                spread_a=self.spread_a,
                noise_amplitude_a=self.noise_amplitude_a,
                correlation_length_a=self.correlation_length_a,
                curvature_iterations=self.curvature_iterations,
                device=self.device,
                seed=self.seed,
            )

        self.volume = rasterize_membrane_density(
            self.field,
            self.profile,
            target_shape_zyx=self.target_shape_zyx,
            target_spacing_a=self.v_size,
            target_origin_xyz=self._origin_xyz,
        )
        return self.volume

    def place_transmembrane(
        self, min_spacing_a: float = 40.0, max_attempts: int | None = None
    ) -> list[TransmembranePlacement]:
        """
        Sample surface sites and insert transmembrane protein templates.

        Must be called after :meth:`generate`. Species are chosen per site
        by weighted random draw (weights = ``frequency``), matching
        ``TetrisPackingSpecimenGenerator``'s species-selection convention.

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
                "or -- shape_backend='alpha_shape'/'spherical_harmonics' "
                "specifically -- the working field grid is too coarse relative to "
                "blob_size_a/sh_axes_a for reliable surface projection (see "
                "generate_membrane_field_alpha_shape's/"
                "generate_membrane_field_spherical_harmonics's own Notes; both "
                "functions also warn proactively on this). This is not "
                "raised as an error since a partial/empty result is sometimes "
                "intended (e.g. deliberately testing a too-small membrane), but "
                f"placements will be missing/absent if not: {n_found} of "
                f"{n_sites} transmembrane instances will be placed.",
                stacklevel=2,
            )

        templates = {
            spec.pdb_source: self._build_template(spec)
            for spec in self.transmembrane_specs
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
            self._insert_add(rotated, center_zyx)

            placements.append(
                TransmembranePlacement(
                    species_id=spec.pdb_source,
                    center_xyz=sites_xyz[i].detach().cpu(),
                    rotation_matrix=rotation.detach().cpu(),
                )
            )

        self.placements = placements
        return placements

    def _build_template(self, spec: TransmembraneSpec) -> torch.Tensor:
        if spec.template is not None:
            return spec.template.detach().to(self.device, dtype=torch.float32)

        pdb = PDB(spec.pdb_source, savefolder=self.pdb_cache_dir, verbose=False)
        coordinates = align_principal_axis_to_z(pdb.coordinates)
        coordinates = align_transmembrane_depth(coordinates, spec.tm_span_mask)
        n = estimate_protein_box_size(pdb.max_diameter, self.v_size)
        builder = PotentialBuilder(
            n_xyz=n,
            dx=self.v_size,
            atomic_numbers=pdb.atomic_numbers,
            progressbars=False,
            parameterization=spec.parameterization,
        )
        # "analytic" (PotentialBuilder's own documented default), not "3d" --
        # confirmed empirically the two methods integrate to the same total
        # potential (ratio ~1.001 across a 10x span of voxel sizes, on a
        # real structure) but "3d"'s convolution-based approach spreads that
        # same total over a visibly lower, smoother peak (25-45% lower
        # max value than "analytic" for identical atoms). The bilayer
        # profile is rendered with "analytic"; using "3d" here compared two
        # differently-sharpened renderings of the same underlying physics,
        # not a deliberate contrast difference between membrane and
        # protein.
        return builder.forward(coordinates, method="analytic").to(self.device)

    def _physical_to_voxel_index(self, site_xyz: torch.Tensor) -> torch.Tensor:
        assert self._origin_xyz is not None
        origin = self._origin_xyz.cpu()
        idx_xyz = (site_xyz.detach().cpu() - origin) / self.v_size
        idx_zyx = torch.stack([idx_xyz[2], idx_xyz[1], idx_xyz[0]])
        return idx_zyx.round().long()

    def _insert_add(
        self, local_density: torch.Tensor, center_zyx: torch.Tensor
    ) -> None:
        assert self.volume is not None
        bounds = clip_insert_bounds(
            center_zyx.tolist(), local_density.shape, self.volume.shape
        )
        if bounds is None:
            return
        dst, src = bounds
        self.volume[dst] += local_density[src].to(self.volume.device)


__all__ = [
    "MembraneGenerator",
    "TransmembraneSpec",
    "TransmembranePlacement",
]
