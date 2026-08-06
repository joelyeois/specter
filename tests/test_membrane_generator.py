import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import ndimage

from specter.specimen.membrane._field import _signed_distance_transform
from specter.specimen.membrane._generator import (
    MembraneGenerator,
    TransmembraneSpec,
)

_SMALL_KWARGS = dict(
    target_shape_zyx=(32, 32, 32),
    v_size=6.0,
    sh_axes_a=(50.0, 50.0, 50.0),
    sh_amplitude=0.15,
    n_lipids_per_leaflet=6,
)


def test_generate_produces_correct_shape_with_membrane_density():
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    volume = gen.generate()

    assert volume.shape == _SMALL_KWARGS["target_shape_zyx"]
    assert torch.isfinite(volume).all()
    assert volume.max() > 0
    assert gen.field is not None
    assert gen.profile is not None


def test_generate_is_seed_reproducible():
    gen_a = MembraneGenerator(seed=3, **_SMALL_KWARGS)
    gen_b = MembraneGenerator(seed=3, **_SMALL_KWARGS)
    volume_a = gen_a.generate()
    volume_b = gen_b.generate()
    assert torch.equal(volume_a, volume_b)


def test_place_transmembrane_before_generate_raises():
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    with pytest.raises(RuntimeError):
        gen.place_transmembrane()


def test_place_transmembrane_with_no_specs_returns_empty_list():
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    gen.generate()
    assert gen.place_transmembrane() == []


def test_place_transmembrane_with_prebuilt_template_inserts_and_records_placement():
    template = torch.zeros(9, 9, 9)
    template[4, 4, 4] = 50.0

    gen = MembraneGenerator(
        transmembrane_specs=[TransmembraneSpec("synthetic", template=template)],
        seed=0,
        **_SMALL_KWARGS,
    )
    baseline = gen.generate().clone()
    baseline_sum = baseline.sum()

    placements = gen.place_transmembrane(min_spacing_a=20.0)

    assert len(placements) >= 1
    for placement in placements:
        assert placement.species_id == "synthetic"
        R = placement.rotation_matrix
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-3)
        assert torch.isclose(torch.det(R), torch.tensor(1.0), atol=1e-3)

    assert gen.volume.sum() > baseline_sum


def test_place_transmembrane_species_selection_seed_reproducible():
    template_a = torch.zeros(7, 7, 7)
    template_a[3, 3, 3] = 10.0
    template_b = torch.zeros(7, 7, 7)
    template_b[3, 3, 3] = 20.0

    specs = [
        TransmembraneSpec("species_a", frequency=1, template=template_a),
        TransmembraneSpec("species_b", frequency=3, template=template_b),
    ]

    gen_a = MembraneGenerator(transmembrane_specs=specs, seed=7, **_SMALL_KWARGS)
    gen_a.generate()
    placements_a = gen_a.place_transmembrane(min_spacing_a=15.0)

    gen_b = MembraneGenerator(transmembrane_specs=specs, seed=7, **_SMALL_KWARGS)
    gen_b.generate()
    placements_b = gen_b.place_transmembrane(min_spacing_a=15.0)

    assert [p.species_id for p in placements_a] == [p.species_id for p in placements_b]
    for pa, pb in zip(placements_a, placements_b):
        assert torch.allclose(pa.center_xyz, pb.center_xyz)


def test_build_template_uses_analytic_method_matching_membrane_profile():
    # Regression test: PDB-backed templates were rendered with method="3d"
    # while the bilayer profile is rendered with method="analytic". Both
    # integrate to the same total potential (confirmed empirically, ratio
    # ~1.001 across a 10x span of voxel sizes on a real structure), but
    # "3d" spreads that total over a 25-45% lower peak for identical atoms
    # -- an unintended cross-method contrast mismatch between membrane and
    # protein, not a deliberate physical difference. Verify the template
    # builder now actually calls PotentialBuilder with method="analytic".
    from specter.pdb import PDB
    from specter.potential import PotentialBuilder
    from specter.specimen.membrane._placement import (
        align_principal_axis_to_z,
        align_transmembrane_depth,
    )
    from specter.specimen.packing import estimate_protein_box_size

    gen = MembraneGenerator(
        pdb_cache_dir="pdb-data/",
        **_SMALL_KWARGS,
    )
    spec = TransmembraneSpec("1C3W", parameterization="shtyrov")
    template = gen._build_template(spec)

    pdb = PDB("1C3W", savefolder="pdb-data/", verbose=False)
    coordinates = align_principal_axis_to_z(pdb.coordinates)
    coordinates = align_transmembrane_depth(coordinates, None)
    n = estimate_protein_box_size(pdb.max_diameter, gen.v_size)
    builder = PotentialBuilder(
        n_xyz=n,
        dx=gen.v_size,
        atomic_numbers=pdb.atomic_numbers,
        progressbars=False,
        parameterization="shtyrov",
    )
    expected = builder.forward(coordinates, method="analytic")

    assert torch.allclose(template.cpu(), expected, atol=1e-3)


def test_max_field_voxels_coarsens_and_warns_instead_of_exploding_memory():
    """Regression test for a real production-scale finding: the naive
    field spacing heuristic, applied uniformly to a large target_shape_zyx,
    produced a ~1.1 billion voxel working grid (50+ GB resident across the
    several such arrays field generation allocates) for a real
    (200, 600, 600)-voxel/10A tomogram. A tiny max_field_voxels here forces
    the same coarsening path at a scale cheap enough to test quickly."""
    gen = MembraneGenerator(
        target_shape_zyx=(32, 32, 32),
        v_size=6.0,
        sh_axes_a=(50.0, 50.0, 50.0),
        sh_amplitude=0.15,
        n_lipids_per_leaflet=6,
        max_field_voxels=1000,
        seed=0,
    )
    with pytest.warns(UserWarning, match="coarsened field_spacing_a"):
        volume = gen.generate()

    assert volume.shape == (32, 32, 32)
    assert torch.isfinite(volume).all()
    assert gen.field is not None
    n_field_voxels = (
        gen.field.phi.shape[0] * gen.field.phi.shape[1] * gen.field.phi.shape[2]
    )
    assert n_field_voxels <= 1000 * 1.5  # ceil() rounding can push slightly over


def test_max_field_voxels_default_does_not_warn_at_small_scale():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
        gen.generate()  # would raise if the coarsening warning fired


_METABALL_KWARGS = dict(
    target_shape_zyx=(32, 32, 32),
    v_size=6.0,
    shape_backend="metaball",
    n_sources=3,
    radius_range_a=(20.0, 30.0),
    spread_a=10.0,
    noise_amplitude_a=0.0,
    curvature_iterations=5,
    n_lipids_per_leaflet=6,
)


def test_metaball_backend_produces_correct_shape_with_membrane_density():
    """shape_backend="metaball" is deprecated (see
    test_deprecated_backends_warn_on_construction) but must remain fully
    functional -- explicit smoke coverage now that it's no longer the
    implicit default every other test exercised for free."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        gen = MembraneGenerator(seed=0, **_METABALL_KWARGS)
        volume = gen.generate()

    assert volume.shape == _METABALL_KWARGS["target_shape_zyx"]
    assert torch.isfinite(volume).all()
    assert volume.max() > 0


def test_metaball_backend_is_seed_reproducible():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        gen_a = MembraneGenerator(seed=3, **_METABALL_KWARGS)
        gen_b = MembraneGenerator(seed=3, **_METABALL_KWARGS)
    volume_a = gen_a.generate()
    volume_b = gen_b.generate()
    assert torch.equal(volume_a, volume_b)


@pytest.mark.parametrize("shape_backend", ["metaball", "alpha_shape"])
def test_deprecated_backends_warn_on_construction(shape_backend):
    with pytest.warns(DeprecationWarning, match=shape_backend):
        if shape_backend == "metaball":
            MembraneGenerator(seed=0, **_METABALL_KWARGS)
        else:
            MembraneGenerator(
                target_shape_zyx=(80, 80, 80),
                v_size=4.0,
                shape_backend="alpha_shape",
                blob_size_a=40.0,
                blob_roughness=0.4,
                n_lipids_per_leaflet=6,
                seed=0,
            )


def test_spherical_harmonics_and_swept_spline_do_not_warn_on_construction():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        MembraneGenerator(seed=0, **_SMALL_KWARGS)
        MembraneGenerator(
            target_shape_zyx=(32, 32, 32),
            v_size=6.0,
            shape_backend="swept_spline",
            swept_total_length_a=60.0,
            swept_tube_radius_a=8.0,
            n_lipids_per_leaflet=6,
            seed=0,
        )


_ALPHA_SHAPE_KWARGS = dict(
    target_shape_zyx=(80, 80, 80),
    v_size=4.0,
    shape_backend="alpha_shape",
    blob_size_a=40.0,
    blob_roughness=0.4,
    n_lipids_per_leaflet=6,
)


def test_alpha_shape_backend_produces_correct_shape_with_membrane_density():
    """shape_backend="alpha_shape" (CTS-derived) must be a drop-in for
    "metaball" through the rest of the pipeline -- same output contract,
    same calibrated BilayerProfile underneath, only the shape geometry
    differs."""
    gen = MembraneGenerator(seed=0, **_ALPHA_SHAPE_KWARGS)
    volume = gen.generate()

    assert volume.shape == _ALPHA_SHAPE_KWARGS["target_shape_zyx"]
    assert torch.isfinite(volume).all()
    assert volume.max() > 0
    assert gen.field is not None
    assert gen.profile is not None


def test_alpha_shape_backend_is_seed_reproducible():
    gen_a = MembraneGenerator(seed=3, **_ALPHA_SHAPE_KWARGS)
    gen_b = MembraneGenerator(seed=3, **_ALPHA_SHAPE_KWARGS)
    volume_a = gen_a.generate()
    volume_b = gen_b.generate()
    assert torch.equal(volume_a, volume_b)


def test_alpha_shape_backend_supports_transmembrane_placement():
    """The whole point of building this backend on top of MembraneGenerator
    rather than reusing CTS's own MembraneBlobGenerator directly: correct
    normal-aligned transmembrane placement, which CTS's own "location"
    mechanism does not provide for arbitrary (non-pre-oriented) PDB
    structures."""
    pdb_path = Path(__file__).parent.parent / "pdb-data" / "1mbo.cif"
    if not pdb_path.exists():
        pytest.skip("bundled PDB fixture missing")
    gen = MembraneGenerator(
        transmembrane_specs=[TransmembraneSpec(pdb_source=str(pdb_path), frequency=3)],
        seed=0,
        **_ALPHA_SHAPE_KWARGS,
    )
    gen.generate()
    placements = gen.place_transmembrane(min_spacing_a=15.0)
    assert len(placements) > 0


def test_shape_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="shape_backend"):
        MembraneGenerator(
            target_shape_zyx=(32, 32, 32), v_size=6.0, shape_backend="not_a_backend"
        )


def test_alpha_shape_surface_smoothing_reduces_facet_kinks():
    """Regression test for a real user-reported issue: the alpha-shape's
    Delaunay-triangulated boundary is faceted/kinked rather than smooth at
    any realistic blob_size_a (verified visually, including against CTS's
    own native generator reusing the same underlying point-cloud/alpha-shape
    algorithm directly -- the faceting is inherent to it, not something this
    backend introduced; raising point count alone doesn't fix it either,
    since surface area grows as blob_size_a**2 while _build_blob's point
    count grows far more slowly). surface_smoothing_voxels (blur-then-
    rethreshold the solid mask, default 2.0) fixes this. Verify
    quantitatively via the isoperimetric ratio (surface area / volume^(2/3),
    from face-adjacency transitions on the phi<0 mask) -- a faceted/kinked
    surface has excess area for its enclosed volume relative to a smooth
    one; smoothing should measurably reduce it."""
    from specter.specimen.membrane._field_alpha import (
        generate_membrane_field_alpha_shape,
    )

    kwargs = dict(
        shape_zyx=(80, 80, 80),
        spacing_a=4.0,
        blob_size_a=60.0,
        blob_roughness=0.3,
        seed=0,
    )
    field_kinked = generate_membrane_field_alpha_shape(
        surface_smoothing_voxels=0.0, **kwargs
    )
    field_smooth = generate_membrane_field_alpha_shape(
        surface_smoothing_voxels=2.0, **kwargs
    )

    def isoperimetric_ratio(phi: torch.Tensor, spacing_a: float) -> float:
        import numpy as np

        mask = phi.numpy() < 0
        volume = mask.sum() * spacing_a**3
        area_voxels = sum(np.sum(mask != np.roll(mask, 1, axis=ax)) for ax in range(3))
        area = area_voxels * spacing_a**2
        return float(area / volume ** (2.0 / 3.0))

    ratio_kinked = isoperimetric_ratio(field_kinked.phi, 4.0)
    ratio_smooth = isoperimetric_ratio(field_smooth.phi, 4.0)
    assert ratio_smooth < 0.75 * ratio_kinked


def test_alpha_shape_backend_warns_when_blob_too_small_for_reliable_resolution():
    """Regression test for a real finding: a blob_size_a too small relative
    to the working grid's spacing makes sample_surface_sites' Newton
    projection unreliable, previously silently finding zero transmembrane
    sites with no warning at all. Both this proactive check (fires during
    generate()) and the reactive one in place_transmembrane below must
    fire, not just one -- verified directly across a v_size sweep that a
    too-small blob keeps failing regardless of v_size once field spacing
    saturates at its own resolution floor (driven by the bilayer profile,
    not v_size), so this is a real, not incidental, gap to guard."""
    gen = MembraneGenerator(
        target_shape_zyx=(80, 80, 80),
        v_size=4.0,
        shape_backend="alpha_shape",
        blob_size_a=18.0,  # ~6.4 voxels/radius at this spacing -- below the
        # verified-reliable floor (see _field_alpha.py's
        # _MIN_RELIABLE_VOXELS_PER_RADIUS)
        blob_roughness=0.5,
        n_lipids_per_leaflet=6,
        seed=0,
    )
    with pytest.warns(UserWarning, match="voxels/radius"):
        gen.generate()


_SH_KWARGS = dict(
    target_shape_zyx=(80, 80, 80),
    v_size=4.0,
    shape_backend="spherical_harmonics",
    sh_axes_a=(60.0, 60.0, 60.0),
    sh_max_degree=8,
    n_lipids_per_leaflet=6,
)


def test_spherical_harmonics_backend_produces_correct_shape_with_membrane_density():
    """shape_backend="spherical_harmonics" must be a drop-in for the other
    backends through the rest of the pipeline -- same output contract, same
    calibrated BilayerProfile underneath, only the shape geometry differs."""
    gen = MembraneGenerator(seed=0, **_SH_KWARGS)
    volume = gen.generate()

    assert volume.shape == _SH_KWARGS["target_shape_zyx"]
    assert torch.isfinite(volume).all()
    assert volume.max() > 0
    assert gen.field is not None
    assert gen.profile is not None


def test_spherical_harmonics_backend_is_seed_reproducible():
    gen_a = MembraneGenerator(seed=3, **_SH_KWARGS)
    gen_b = MembraneGenerator(seed=3, **_SH_KWARGS)
    volume_a = gen_a.generate()
    volume_b = gen_b.generate()
    assert torch.equal(volume_a, volume_b)


def test_spherical_harmonics_backend_supports_transmembrane_placement():
    pdb_path = Path(__file__).parent.parent / "pdb-data" / "1mbo.cif"
    if not pdb_path.exists():
        pytest.skip("bundled PDB fixture missing")
    gen = MembraneGenerator(
        transmembrane_specs=[TransmembraneSpec(pdb_source=str(pdb_path), frequency=3)],
        seed=0,
        **_SH_KWARGS,
    )
    gen.generate()
    placements = gen.place_transmembrane(min_spacing_a=15.0)
    assert len(placements) > 0


def test_spherical_harmonics_backend_warns_when_axes_too_small_for_reliable_resolution():
    """Mirrors test_alpha_shape_backend_warns_when_blob_too_small_for_reliable_resolution:
    the same Eikonal/grid-resolution reasoning applies here since this backend
    also derives phi from distance_transform_edt on a boolean mask."""
    gen = MembraneGenerator(
        target_shape_zyx=(80, 80, 80),
        v_size=4.0,
        shape_backend="spherical_harmonics",
        sh_axes_a=(18.0, 18.0, 18.0),  # ~6.4 voxels/radius, below the
        # verified-reliable floor (_MIN_RELIABLE_VOXELS_PER_RADIUS)
        n_lipids_per_leaflet=6,
        seed=0,
    )
    with pytest.warns(UserWarning, match="voxels/radius"):
        gen.generate()


_SWEPT_SPLINE_KWARGS = dict(
    target_shape_zyx=(90, 90, 90),
    v_size=4.0,
    shape_backend="swept_spline",
    swept_total_length_a=300.0,
    swept_step_length_a=15.0,
    swept_tube_radius_a=25.0,
    n_lipids_per_leaflet=6,
)


def test_swept_spline_backend_produces_correct_shape_with_membrane_density():
    """shape_backend="swept_spline" must be a drop-in for the other backends
    through the rest of the pipeline -- same output contract, same
    calibrated BilayerProfile underneath, only the shape geometry differs."""
    gen = MembraneGenerator(seed=0, **_SWEPT_SPLINE_KWARGS)
    volume = gen.generate()

    assert volume.shape == _SWEPT_SPLINE_KWARGS["target_shape_zyx"]
    assert torch.isfinite(volume).all()
    assert volume.max() > 0
    assert gen.field is not None
    assert gen.profile is not None


def test_swept_spline_backend_is_seed_reproducible():
    gen_a = MembraneGenerator(seed=3, **_SWEPT_SPLINE_KWARGS)
    gen_b = MembraneGenerator(seed=3, **_SWEPT_SPLINE_KWARGS)
    volume_a = gen_a.generate()
    volume_b = gen_b.generate()
    assert torch.equal(volume_a, volume_b)


def test_swept_spline_backend_supports_transmembrane_placement():
    pdb_path = Path(__file__).parent.parent / "pdb-data" / "1mbo.cif"
    if not pdb_path.exists():
        pytest.skip("bundled PDB fixture missing")
    gen = MembraneGenerator(
        transmembrane_specs=[TransmembraneSpec(pdb_source=str(pdb_path), frequency=3)],
        seed=0,
        **_SWEPT_SPLINE_KWARGS,
    )
    gen.generate()
    placements = gen.place_transmembrane(min_spacing_a=15.0)
    assert len(placements) > 0


def test_spherical_harmonics_zero_amplitude_isotropic_matches_sphere_sdf():
    """sh_amplitude=0.0 with isotropic sh_axes_a collapses to a plain sphere
    -- phi should match the analytic sphere SDF to within the EDT grid's own
    discretization (one voxel), mirroring
    test_single_source_field_matches_sphere_sdf's tolerance/style for the
    metaball backend's exact analytic case."""
    from specter.specimen.membrane._field_spherical_harmonics import (
        generate_membrane_field_spherical_harmonics,
    )

    spacing_a = 4.0
    radius_a = 60.0
    field = generate_membrane_field_spherical_harmonics(
        shape_zyx=(80, 80, 80),
        spacing_a=spacing_a,
        sh_axes_a=(radius_a, radius_a, radius_a),
        sh_amplitude=0.0,
        seed=0,
    )
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [radius_a, 0.0, 0.0], [0.0, 0.0, -radius_a], [30.0, 0.0, 0.0]]
    )
    expected = torch.linalg.norm(points, dim=-1) - radius_a
    sampled = field.sample(points)
    assert torch.allclose(sampled, expected, atol=1.5 * spacing_a)


def test_spherical_harmonics_zero_amplitude_anisotropic_matches_ellipsoid_surface():
    """sh_amplitude=0.0 with anisotropic sh_axes_a collapses to a plain
    ellipsoid -- points on the implicit surface (x/ax)^2+(y/ay)^2+(z/az)^2=1
    should sample near phi=0."""
    from specter.specimen.membrane._field_spherical_harmonics import (
        generate_membrane_field_spherical_harmonics,
    )

    spacing_a = 4.0
    axes = (70.0, 60.0, 60.0)
    field = generate_membrane_field_spherical_harmonics(
        shape_zyx=(60, 40, 40),
        spacing_a=spacing_a,
        sh_axes_a=axes,
        sh_amplitude=0.0,
        seed=0,
    )
    surface_points = torch.tensor(
        [[axes[0], 0.0, 0.0], [0.0, axes[1], 0.0], [0.0, 0.0, axes[2]]]
    )
    sampled = field.sample(surface_points)
    assert torch.allclose(sampled, torch.zeros(3), atol=1.5 * spacing_a)


def test_real_spherical_harmonic_matches_known_l2_m0_and_is_orthonormal():
    """Standalone check of the real-SH basis convention itself, independent
    of the field-generation pipeline: scipy.special.sph_harm_y's argument
    order/convention (theta=polar, phi=azimuthal in this scipy version) is
    easy to get backwards against the older, deprecated scipy.special.sph_harm
    (theta=azimuthal there) -- verify against a closed form and via
    Monte-Carlo orthonormality rather than trusting the convention blindly."""
    import numpy as np

    from specter.specimen.membrane._field_spherical_harmonics import (
        _real_spherical_harmonic,
    )

    theta = torch.tensor([0.3, 0.7, 1.2, 2.0]).numpy()
    phi = torch.tensor([0.1, 1.0, 2.5, 4.0]).numpy()
    y20 = _real_spherical_harmonic(2, 0, theta, phi)
    expected = 0.25 * np.sqrt(5.0 / np.pi) * (3.0 * np.cos(theta) ** 2 - 1.0)
    assert np.allclose(y20, expected, atol=1e-10)

    rng = np.random.default_rng(0)
    n = 200_000
    mc_theta = np.arccos(rng.uniform(-1.0, 1.0, n))
    mc_phi = rng.uniform(0.0, 2.0 * np.pi, n)
    y21 = _real_spherical_harmonic(2, 1, mc_theta, mc_phi)
    y2neg1 = _real_spherical_harmonic(2, -1, mc_theta, mc_phi)
    # <Y|Y'> = (1/4pi) * integral over solid angle -- Monte Carlo estimate
    # via the mean over uniformly-sampled directions.
    assert abs(np.mean(y21 * y21) * 4.0 * np.pi - 1.0) < 0.05
    assert abs(np.mean(y21 * y2neg1) * 4.0 * np.pi) < 0.05


def test_real_spherical_harmonics_grid_matches_single_call_for_every_l_m():
    """Regression test for the batched sph_harm_y_all optimization:
    generate_membrane_field_spherical_harmonics uses
    _real_spherical_harmonics_grid (one scipy.special.sph_harm_y_all call for
    the whole degree/order grid, ~5x faster) instead of one
    _real_spherical_harmonic (scipy.special.sph_harm_y) call per (l, m) --
    sph_harm_y_all's order axis uses FFT-style wraparound indexing, not
    simple ascending -m..m, which _real_spherical_harmonics_grid's _column
    helper decodes. Verify the two paths agree exactly for every (l, m) up
    to a representative max_degree, not just spot-checked ones."""
    import numpy as np

    from specter.specimen.membrane._field_spherical_harmonics import (
        _real_spherical_harmonic,
        _real_spherical_harmonics_grid,
    )

    rng = np.random.default_rng(1)
    theta = np.arccos(rng.uniform(-1.0, 1.0, 500))
    phi = rng.uniform(0.0, 2.0 * np.pi, 500)
    max_degree = 6

    batched = _real_spherical_harmonics_grid(max_degree, theta, phi)
    for degree in range(2, max_degree + 1):
        for order in range(-degree, degree + 1):
            single = _real_spherical_harmonic(degree, order, theta, phi)
            assert np.allclose(batched[(degree, order)], single, atol=1e-10)


def test_coarse_angular_grid_interpolation_matches_direct_evaluation():
    """Regression test for the coarse-angular-grid + bilinear-interpolation
    optimization: generate_membrane_field_spherical_harmonics no longer
    evaluates the harmonic perturbation directly at every working-grid voxel
    (O(voxel count), measured to dominate generate() at ~70s for a realistic
    ~10M-voxel grid) -- it synthesizes the perturbation once on a small
    _angular_grid_resolution(sh_max_degree) angular grid and bilinearly
    interpolates (torch.nn.functional.grid_sample, with a manual phi-
    periodic pad). Verify the interpolated result stays close to direct
    per-point evaluation (the ground truth _real_spherical_harmonics_grid
    itself is separately verified above) -- measured directly at
    sh_max_degree=8's default 128x256 grid: ~0.17% max error relative to the
    perturbation's own RMS=1 scale. Assert a generous bound (1%) so this
    catches a real regression (e.g. a broken periodic wrap or a badly
    undersized grid) without being sensitive to incidental resampling noise."""
    import numpy as np

    from specter.specimen.membrane._field_spherical_harmonics import (
        _angular_grid_resolution,
        _interpolate_angular_grid,
        _real_spherical_harmonics_grid,
        _sample_sh_coefficients,
        _synthesize_angular_grid,
    )

    max_degree = 8
    rng = np.random.default_rng(2)
    coefficients = _sample_sh_coefficients(max_degree, 2.0, rng)

    n_check = 20_000
    theta = np.arccos(rng.uniform(-1.0, 1.0, n_check))
    phi = rng.uniform(0.0, 2.0 * np.pi, n_check)

    real_sh = _real_spherical_harmonics_grid(max_degree, theta, phi)
    direct = np.zeros(n_check)
    for degree, order, coeff in coefficients:
        direct += coeff * real_sh[(degree, order)]

    n_theta, n_phi = _angular_grid_resolution(max_degree)
    coarse_padded = _synthesize_angular_grid(coefficients, max_degree, n_theta, n_phi)
    interpolated = _interpolate_angular_grid(coarse_padded, theta, phi, device="cpu")

    max_err = np.abs(interpolated - direct).max()
    assert max_err < 0.01, f"interpolation max error {max_err} exceeds 1% bound"


def test_swept_spline_near_straight_path_matches_capsule_sdf():
    """A near-straight path (flexibility ~ 0) sampled well away from either
    end should match a capsule/stadium SDF: perpendicular distance from the
    (recentered) centerline minus tube_radius_a. Reconstructs the exact same
    recentered/smoothed source positions generate_membrane_field_swept_spline
    builds internally (same seed, same _sample_wandering_path call) to find
    a point unambiguously nearest a single interior source, away from the
    path ends and from any other source's competing influence, so the
    smooth-min blend closely matches a plain nearest-center distance there."""
    import numpy as np
    from scipy import ndimage as scipy_ndimage

    from specter.specimen.membrane._field_swept_spline import (
        _sample_wandering_path,
        generate_membrane_field_swept_spline,
    )

    seed = 5
    step_length_a = 15.0
    total_length_a = 300.0
    tube_radius_a = 25.0
    flexibility = 1e-4
    n_points = max(2, round(total_length_a / step_length_a) + 1)

    rng = np.random.default_rng(seed)
    positions = _sample_wandering_path(n_points, step_length_a, flexibility, rng)
    bbox_mid = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    positions = positions - bbox_mid
    positions = scipy_ndimage.gaussian_filter1d(
        positions, sigma=1.5, axis=0, mode="nearest"
    )

    mid = n_points // 2
    tangent = positions[mid + 1] - positions[mid - 1]
    tangent /= np.linalg.norm(tangent)
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, tangent)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    perp = arbitrary - np.dot(arbitrary, tangent) * tangent
    perp /= np.linalg.norm(perp)

    offset = 12.0
    query_point = positions[mid] + perp * (tube_radius_a + offset)

    field = generate_membrane_field_swept_spline(
        shape_zyx=(90, 90, 90),
        spacing_a=4.0,
        total_length_a=total_length_a,
        step_length_a=step_length_a,
        tube_radius_a=tube_radius_a,
        flexibility=flexibility,
        seed=seed,
    )
    sampled = field.sample(torch.tensor(query_point, dtype=torch.float32))
    assert torch.isclose(sampled, torch.tensor(offset), atol=3.0)


def test_swept_spline_field_is_seed_reproducible():
    from specter.specimen.membrane._field_swept_spline import (
        generate_membrane_field_swept_spline,
    )

    kwargs = dict(
        shape_zyx=(70, 70, 70),
        spacing_a=4.0,
        total_length_a=300.0,
        step_length_a=15.0,
        seed=7,
    )
    field_a = generate_membrane_field_swept_spline(**kwargs)
    field_b = generate_membrane_field_swept_spline(**kwargs)
    assert torch.equal(field_a.phi, field_b.phi)


def test_swept_spline_warns_on_beading_risk():
    from specter.specimen.membrane._field_swept_spline import (
        generate_membrane_field_swept_spline,
    )

    with pytest.warns(UserWarning, match="beading"):
        generate_membrane_field_swept_spline(
            shape_zyx=(60, 60, 60),
            spacing_a=4.0,
            total_length_a=200.0,
            step_length_a=40.0,
            tube_radius_a=20.0,
            seed=0,
        )


def test_place_transmembrane_warns_on_partial_or_zero_placement():
    """Pre-existing gap, not specific to the new backend: place_transmembrane
    silently returned fewer (or zero) placements than requested with no
    warning at all, for either shape_backend. Fixed for both."""
    pdb_path = Path(__file__).parent.parent / "pdb-data" / "1mbo.cif"
    if not pdb_path.exists():
        pytest.skip("bundled PDB fixture missing")
    gen = MembraneGenerator(
        target_shape_zyx=(80, 80, 80),
        v_size=4.0,
        shape_backend="alpha_shape",
        blob_size_a=18.0,
        blob_roughness=0.5,
        n_lipids_per_leaflet=6,
        # frequency=8 (not e.g. 3): verified directly this many requested
        # sites reliably falls short at this resolution/spacing (a lower
        # frequency, e.g. 3-5, sometimes fully succeeds by chance despite
        # the low-resolution risk generate()'s own proactive warning
        # already flags -- that warning is a risk indicator, not a
        # guarantee of shortfall, so this reactive-warning test needs its
        # own separately-verified failure case).
        transmembrane_specs=[TransmembraneSpec(pdb_source=str(pdb_path), frequency=8)],
        seed=0,
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")  # let generate()'s own warning through quietly
        gen.generate()
    with pytest.warns(UserWarning, match="requested transmembrane sites were found"):
        placements = gen.place_transmembrane(min_spacing_a=15.0)
    assert len(placements) < 8


def _synthetic_inside_mask(shape=(30, 30, 30), radius=8.0) -> np.ndarray:
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    c = np.array(shape) / 2.0
    r = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    return r < radius


def test_signed_distance_transform_cpu_matches_scipy_directly():
    inside = _synthetic_inside_mask()
    spacing_a = 3.0
    phi = _signed_distance_transform(inside, spacing_a, device="cpu")

    dist_out = ndimage.distance_transform_edt(~inside, sampling=spacing_a)
    dist_in = ndimage.distance_transform_edt(inside, sampling=spacing_a)
    expected = torch.as_tensor(dist_out - dist_in, dtype=torch.float32)

    assert phi.device.type == "cpu"
    assert torch.equal(phi, expected)


def test_signed_distance_transform_gpu_matches_cpu_when_cupy_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pytest.importorskip("cupy", reason="optional gpu-edt extra not installed")

    inside = _synthetic_inside_mask()
    spacing_a = 3.0
    phi_gpu = _signed_distance_transform(inside, spacing_a, device="cuda")
    phi_cpu = _signed_distance_transform(inside, spacing_a, device="cpu")

    assert phi_gpu.device.type == "cuda"
    assert torch.allclose(phi_gpu.cpu(), phi_cpu, atol=1e-4)


def test_signed_distance_transform_falls_back_to_cpu_without_cupy(monkeypatch):
    # Regression test: the optional GPU path (cupyx.scipy.ndimage.
    # distance_transform_edt) must degrade gracefully -- not crash -- when
    # 'cupy' isn't installed, since it's an optional extra
    # (`specter[gpu-edt]`), not a core dependency. Requires an actual CUDA
    # device since the GPU path is only ever attempted for device="cuda".
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("simulated: cupy not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    inside = _synthetic_inside_mask()
    spacing_a = 3.0
    with pytest.warns(UserWarning, match="cupy' dependency isn't installed"):
        phi = _signed_distance_transform(inside, spacing_a, device="cuda")

    dist_out = ndimage.distance_transform_edt(~inside, sampling=spacing_a)
    dist_in = ndimage.distance_transform_edt(inside, sampling=spacing_a)
    expected = torch.as_tensor(dist_out - dist_in, dtype=torch.float32)

    assert phi.device.type == "cuda"
    assert torch.equal(phi.cpu(), expected)
