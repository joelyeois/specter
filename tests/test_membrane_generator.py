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
    target_shape=(32, 32, 32),
    voxel_size=6.0,
    sh_axes=(50.0, 50.0, 50.0),
    sh_amplitude=0.15,
    n_lipids_per_leaflet=6,
)


def test_generate_produces_correct_shape_with_membrane_density():
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    volume = gen.generate()

    assert volume.shape == _SMALL_KWARGS["target_shape"]
    assert torch.isfinite(volume).all()
    assert volume.max() > 0
    assert gen.field is not None
    assert gen.profile is not None


def test_membrane_scale_range_is_reproducible_and_scales_linearly():
    gen_a = MembraneGenerator(seed=3, membrane_scale_range=(0.5, 1.0), **_SMALL_KWARGS)
    volume_a = gen_a.generate()
    gen_b = MembraneGenerator(seed=3, membrane_scale_range=(0.5, 1.0), **_SMALL_KWARGS)
    volume_b = gen_b.generate()

    assert 0.5 <= gen_a.membrane_scale <= 1.0
    assert gen_a.membrane_scale == gen_b.membrane_scale
    assert torch.equal(volume_a, volume_b)

    gen_unscaled = MembraneGenerator(
        seed=3, membrane_scale_range=(1.0, 1.0), **_SMALL_KWARGS
    )
    volume_unscaled = gen_unscaled.generate()
    assert torch.allclose(volume_a, gen_a.membrane_scale * volume_unscaled, atol=1e-4)


def test_membrane_scale_range_rejects_low_greater_than_high():
    with pytest.raises(ValueError, match="membrane_scale_range"):
        MembraneGenerator(membrane_scale_range=(1.0, 0.5), **_SMALL_KWARGS)


def test_place_transmembrane_before_generate_raises():
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    with pytest.raises(RuntimeError):
        gen.place_transmembrane()


def test_place_transmembrane_with_no_specs_returns_empty_list():
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    gen.generate()
    assert gen.place_transmembrane() == []


def test_insert_blend_replaces_occupied_region_and_preserves_untouched_background():
    # Regression test for the additive-insertion fix: a transmembrane
    # protein displaces lipid where it sits rather than coexisting with it,
    # so plain addition (the old behaviour) double-counts density in the
    # overlap. _insert_blend should instead fully REPLACE membrane density
    # where the protein template is occupied, and leave it EXACTLY
    # untouched elsewhere -- the latter also catches a real bug an earlier
    # sigmoid-based (rather than smoothstep) blend had: its tail never
    # truly reaches 0, so a template's zero background picked up a small
    # but nonzero weight too, silently attenuating membrane density across
    # the whole inserted bounding box, not just near the protein's own mass.
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    gen.generate()

    # Pick an actual on-membrane voxel (nonzero density), not an arbitrary
    # point that might land in empty space, so replace-vs-add is a
    # meaningfully different outcome.
    peak_idx = (gen.volume == gen.volume.max()).nonzero()[0]
    center_zyx = peak_idx.clone()
    baseline_center = gen.volume[tuple(peak_idx.tolist())].clone()
    assert baseline_center.item() > 0  # sanity: real density here, not empty space
    offset_idx = peak_idx + torch.tensor([2, 0, 0])
    baseline_offset = gen.volume[tuple(offset_idx.tolist())].clone()

    template = torch.zeros(5, 5, 5)
    template[2, 2, 2] = 1000.0  # far above any plausible bilayer peak amplitude
    gen._insert_blend(template, center_zyx)

    # occupied voxel: replaced by the protein value, not baseline + protein
    # (an absolute, not relative, tolerance -- the baseline density here is
    # tiny relative to 1000.0, so a relative check can't tell replace and
    # add apart).
    assert gen.volume[tuple(peak_idx.tolist())].item() == pytest.approx(
        1000.0, abs=1e-3
    )
    assert gen.volume[tuple(peak_idx.tolist())].item() != pytest.approx(
        (baseline_center + 1000.0).item(), abs=1e-3
    )
    # background voxel within the inserted bounding box but zero protein
    # density there: left exactly as it was.
    assert gen.volume[tuple(offset_idx.tolist())].item() == pytest.approx(
        baseline_offset.item()
    )


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
    #
    # The reference below also mirrors the bonded-species path: under
    # parameterization="shtyrov" the shipped builder types atoms via
    # compute_atom_species, and comparing against an untyped reference would
    # measure that (~4% relative RMS) rather than the method it is testing.
    from specter.pdb import PDB
    from specter.potential import PotentialBuilder
    from specter.specimen.membrane._placement import (
        align_principal_axis_to_z,
        align_transmembrane_depth,
    )
    from specter.specimen.packing import estimate_protein_box_size

    gen = MembraneGenerator(
        pdb_cache_dir=str(Path(__file__).parent / "test_data"),
        **_SMALL_KWARGS,
    )
    spec = TransmembraneSpec("1C3W", parameterization="shtyrov")
    template = gen._build_template(spec)

    pdb = PDB(
        "1C3W",
        pdb_cache_dir=str(Path(__file__).parent / "test_data"),
        verbose=False,
        compute_atom_species=True,
    )
    coordinates = align_principal_axis_to_z(pdb.coordinates)
    coordinates = align_transmembrane_depth(coordinates, None)
    n = estimate_protein_box_size(pdb.max_diameter, gen.voxel_size)
    builder = PotentialBuilder(
        n_xyz=n,
        dx=gen.voxel_size,
        atomic_numbers=pdb.atomic_numbers,
        progressbars=False,
        parameterization="shtyrov",
        atom_species=pdb.atom_species,
    )
    expected = builder.forward(coordinates, method="analytic")

    assert torch.allclose(template.cpu(), expected, atol=1e-3)


def test_max_field_voxels_coarsens_and_warns_instead_of_exploding_memory():
    """Regression test for a real production-scale finding: the naive
    field spacing heuristic, applied uniformly to a large target_shape,
    produced a ~1.1 billion voxel working grid (50+ GB resident across the
    several such arrays field generation allocates) for a real
    (200, 600, 600)-voxel/10A tomogram. A tiny max_field_voxels here forces
    the same coarsening path at a scale cheap enough to test quickly."""
    gen = MembraneGenerator(
        target_shape=(32, 32, 32),
        voxel_size=6.0,
        sh_axes=(50.0, 50.0, 50.0),
        sh_amplitude=0.15,
        n_lipids_per_leaflet=6,
        max_field_voxels=1000,
        seed=0,
    )
    with pytest.warns(UserWarning, match="coarsening the working field grid"):
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


def test_shape_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="shape_backend"):
        MembraneGenerator(
            target_shape=(32, 32, 32), voxel_size=6.0, shape_backend="not_a_backend"
        )


_SH_KWARGS = dict(
    target_shape=(80, 80, 80),
    voxel_size=4.0,
    shape_backend="spherical_harmonics",
    sh_axes=(60.0, 60.0, 60.0),
    sh_max_degree=8,
    n_lipids_per_leaflet=6,
)


def test_spherical_harmonics_backend_produces_correct_shape_with_membrane_density():
    """shape_backend="spherical_harmonics" must be a drop-in for the other
    backends through the rest of the pipeline -- same output contract, same
    calibrated BilayerProfile underneath, only the shape geometry differs."""
    gen = MembraneGenerator(seed=0, **_SH_KWARGS)
    volume = gen.generate()

    assert volume.shape == _SH_KWARGS["target_shape"]
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
    pdb_path = Path(__file__).parent / "test_data" / "1mbo.cif"
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
    """Regression test for a real finding: sh_axes too small relative to
    the working grid's spacing makes sample_surface_sites' Newton
    projection unreliable, previously silently finding zero transmembrane
    sites with no warning at all -- this backend derives phi from
    distance_transform_edt on a boolean mask, so the same Eikonal/grid-
    resolution reasoning applies."""
    gen = MembraneGenerator(
        target_shape=(80, 80, 80),
        voxel_size=4.0,
        shape_backend="spherical_harmonics",
        sh_axes=(18.0, 18.0, 18.0),  # ~6.4 voxels/radius, below the
        # verified-reliable floor (_MIN_RELIABLE_VOXELS_PER_RADIUS)
        n_lipids_per_leaflet=6,
        seed=0,
    )
    with pytest.warns(UserWarning, match="voxels/radius"):
        gen.generate()


_SWEPT_SPLINE_KWARGS = dict(
    target_shape=(90, 90, 90),
    voxel_size=4.0,
    shape_backend="swept_spline",
    swept_total_length=300.0,
    swept_step_length_a=15.0,
    swept_tube_radius=25.0,
    n_lipids_per_leaflet=6,
)


def test_swept_spline_backend_produces_correct_shape_with_membrane_density():
    """shape_backend="swept_spline" must be a drop-in for the other backends
    through the rest of the pipeline -- same output contract, same
    calibrated BilayerProfile underneath, only the shape geometry differs."""
    gen = MembraneGenerator(seed=0, **_SWEPT_SPLINE_KWARGS)
    volume = gen.generate()

    assert volume.shape == _SWEPT_SPLINE_KWARGS["target_shape"]
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
    pdb_path = Path(__file__).parent / "test_data" / "1mbo.cif"
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
    """sh_amplitude=0.0 with isotropic sh_axes collapses to a plain sphere
    -- phi should match the analytic sphere SDF to within the EDT grid's own
    discretization (one voxel), mirroring
    test_single_source_field_matches_sphere_sdf's tolerance/style for
    blend_field's own exact analytic case."""
    from specter.specimen.membrane._field_spherical_harmonics import (
        generate_membrane_field_spherical_harmonics,
    )

    spacing_a = 4.0
    radius_a = 60.0
    field = generate_membrane_field_spherical_harmonics(
        shape_zyx=(80, 80, 80),
        spacing_a=spacing_a,
        sh_axes=(radius_a, radius_a, radius_a),
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
    """sh_amplitude=0.0 with anisotropic sh_axes collapses to a plain
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
        sh_axes=axes,
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


def test_swept_spline_radius_variation_zero_matches_constant_radius():
    """radius_variation=0.0 (the default) must reproduce the exact
    pre-existing constant-radius field bit-for-bit -- this parameter's
    addition must not change any existing caller's output."""
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
    field_default = generate_membrane_field_swept_spline(**kwargs)
    field_explicit_zero = generate_membrane_field_swept_spline(
        radius_variation=0.0, **kwargs
    )
    assert torch.equal(field_default.phi, field_explicit_zero.phi)


def test_swept_spline_radius_variation_changes_field_and_is_reproducible():
    from specter.specimen.membrane._field_swept_spline import (
        generate_membrane_field_swept_spline,
    )

    kwargs = dict(
        shape_zyx=(70, 70, 70),
        spacing_a=4.0,
        total_length_a=300.0,
        step_length_a=15.0,
        radius_variation=0.35,
        seed=7,
    )
    field_a = generate_membrane_field_swept_spline(**kwargs)
    field_b = generate_membrane_field_swept_spline(**kwargs)
    assert torch.equal(field_a.phi, field_b.phi)

    field_constant = generate_membrane_field_swept_spline(
        **{**kwargs, "radius_variation": 0.0}
    )
    assert not torch.equal(field_a.phi, field_constant.phi)


def test_swept_spline_radius_variation_mean_preserved_at_small_n_points():
    """Regression test: at few path points, a smoothed noise sequence's own
    mean is generically nonzero (not guaranteed near zero the way i.i.d.
    noise was) -- dividing by std alone (not centering first) amplified
    that offset, collapsing every drawn radius to the _MIN_RADIUS_FRACTION
    floor for some seeds instead of the intended mild variation around
    tube_radius_a. Reproduces the exact failing case found via
    MembraneGenerator's swept-spline defaults (18 points, seed 3): mean
    drawn radius must land close to tube_radius_a, not near the floor."""
    import numpy as np
    from scipy import ndimage

    from specter.specimen.membrane._field_swept_spline import (
        _MIN_RADIUS_FRACTION,
        _sample_wandering_path,
    )

    total_length_a, step_length_a, tube_radius_a = (
        262.2857142857143,
        15.0,
        21.857142857142858,
    )
    radius_variation, sigma_points, flexibility, seed = 0.1943228840827942, 2.0, 0.15, 3

    n_points = max(2, round(total_length_a / step_length_a) + 1)
    rng = np.random.default_rng(seed)
    _sample_wandering_path(n_points, step_length_a, flexibility, rng)
    noise = rng.normal(size=n_points)
    noise = ndimage.gaussian_filter1d(noise, sigma=sigma_points, mode="nearest")
    noise = noise - noise.mean()
    noise_std = float(noise.std())
    if noise_std > 0.0:
        noise = noise / noise_std
    radii = tube_radius_a * np.clip(
        1.0 + radius_variation * noise, _MIN_RADIUS_FRACTION, None
    )

    assert abs(radii.mean() - tube_radius_a) < 0.1 * tube_radius_a
    floor = _MIN_RADIUS_FRACTION * tube_radius_a
    assert not np.allclose(radii, floor, atol=0.5)


def test_swept_spline_radius_variation_rejects_negative():
    from specter.specimen.membrane._field_swept_spline import (
        generate_membrane_field_swept_spline,
    )

    with pytest.raises(ValueError, match="radius_variation"):
        generate_membrane_field_swept_spline(
            shape_zyx=(60, 60, 60),
            spacing_a=4.0,
            total_length_a=200.0,
            step_length_a=15.0,
            radius_variation=-0.1,
            seed=0,
        )


def test_swept_spline_beading_warning_uses_mean_radius_not_local_minimum():
    """With radius_variation > 0, step_length_a should only warn when it
    exceeds the MEAN drawn radius -- occasional local narrowing below
    step_length_a is the intended source of sparse, irregular
    constrictions (see module docstring's "Radius variation" section) and
    must not itself trigger the beading warning."""
    from specter.specimen.membrane._field_swept_spline import (
        generate_membrane_field_swept_spline,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        generate_membrane_field_swept_spline(
            shape_zyx=(90, 90, 90),
            spacing_a=4.0,
            total_length_a=200.0,
            step_length_a=15.0,
            tube_radius_a=25.0,
            radius_variation=0.35,
            radius_variation_sigma_points=2.0,
            seed=0,
        )

    with pytest.warns(UserWarning, match="beading"):
        generate_membrane_field_swept_spline(
            shape_zyx=(110, 110, 110),
            spacing_a=4.0,
            total_length_a=400.0,
            step_length_a=45.0,
            tube_radius_a=20.0,
            radius_variation=0.35,
            seed=0,
        )


def test_place_transmembrane_warns_on_partial_or_zero_placement():
    """Pre-existing gap: place_transmembrane silently returned fewer (or
    zero) placements than requested with no warning at all. Fixed."""
    pdb_path = Path(__file__).parent / "test_data" / "1mbo.cif"
    if not pdb_path.exists():
        pytest.skip("bundled PDB fixture missing")
    gen = MembraneGenerator(
        target_shape=(80, 80, 80),
        voxel_size=4.0,
        shape_backend="spherical_harmonics",
        sh_axes=(18.0, 18.0, 18.0),
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
    pytest.importorskip("cupy", reason="cupy not installed (macOS, or partial env)")

    inside = _synthetic_inside_mask()
    spacing_a = 3.0
    phi_gpu = _signed_distance_transform(inside, spacing_a, device="cuda")
    phi_cpu = _signed_distance_transform(inside, spacing_a, device="cpu")

    assert phi_gpu.device.type == "cuda"
    assert torch.allclose(phi_gpu.cpu(), phi_cpu, atol=1e-4)


def test_signed_distance_transform_falls_back_to_cpu_without_cupy(monkeypatch):
    # Regression test: the GPU path (cupyx.scipy.ndimage.
    # distance_transform_edt) must degrade gracefully -- not crash -- when
    # 'cupy' isn't importable. cupy IS a core dependency now, but only off
    # macOS (no cupy-cuda12x wheels there, see pyproject.toml), so this
    # fallback is still a supported configuration rather than a legacy one.
    # Requires an actual CUDA device since the GPU path is only ever
    # attempted for device="cuda".
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
    with pytest.warns(UserWarning, match="'cupy' isn't importable"):
        phi = _signed_distance_transform(inside, spacing_a, device="cuda")

    dist_out = ndimage.distance_transform_edt(~inside, sampling=spacing_a)
    dist_in = ndimage.distance_transform_edt(inside, sampling=spacing_a)
    expected = torch.as_tensor(dist_out - dist_in, dtype=torch.float32)

    assert phi.device.type == "cuda"
    assert torch.equal(phi.cpu(), expected)


def test_field_voxel_budget_matches_its_documented_derivation():
    """`_MAX_FIELD_VOXELS` is a measured budget divided by a measured cost,
    not a hand-picked number -- so it must stay consistent with the constants
    it is derived from (see this module's own comment block for the sweep).

    The cap has to satisfy the tightest of the three measured costs: VRAM on
    the cupy path, and host RSS on either path. Only the scipy-path RSS
    figure is a module constant (the other two live in the comment table), so
    that one is checked directly and the rest are checked as headroom.
    """
    from specter.specimen.membrane._generator import (
        _FIELD_BYTES_PER_VOXEL,
        _FIELD_RAM_BUDGET_BYTES,
        _FIELD_VRAM_BUDGET_BYTES,
        _FIELD_VRAM_BYTES_PER_VOXEL,
        _MAX_FIELD_VOXELS,
    )

    # scipy path: host RSS is what binds, and the cap is set right at it
    # (rounded to a round number), so allow 10% of rounding slack.
    scipy_cap = _FIELD_RAM_BUDGET_BYTES / _FIELD_BYTES_PER_VOXEL
    assert _MAX_FIELD_VOXELS <= scipy_cap * 1.1

    # cupy path: VRAM must have real headroom at the cap, or an 8 GB card
    # (the machine this budget is written for) wouldn't hold it.
    assert _MAX_FIELD_VOXELS * _FIELD_VRAM_BYTES_PER_VOXEL <= _FIELD_VRAM_BUDGET_BYTES


def test_signed_distance_transform_falls_back_when_gpu_transform_raises(monkeypatch):
    """A CuPy failure at RUN time (out of VRAM, driver/runtime mismatch that
    only shows up on kernel launch) must fall back to scipy, not abort the
    build. Now that cupy is a core dependency rather than an opt-in extra,
    every CUDA user reaches this code path, so its "never raises" contract has
    to hold past the import/is_available guards too.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pytest.importorskip("cupy", reason="cupy not installed (macOS, or partial env)")

    from cupyx.scipy import ndimage as cundimage

    def boom(*args, **kwargs):
        raise RuntimeError("simulated: out of memory on device")

    monkeypatch.setattr(cundimage, "distance_transform_edt", boom)

    inside = _synthetic_inside_mask()
    spacing_a = 3.0
    with pytest.warns(UserWarning, match="GPU distance transform.*failed"):
        phi = _signed_distance_transform(inside, spacing_a, device="cuda")

    dist_out = ndimage.distance_transform_edt(~inside, sampling=spacing_a)
    dist_in = ndimage.distance_transform_edt(inside, sampling=spacing_a)
    expected = torch.as_tensor(dist_out - dist_in, dtype=torch.float32)

    assert phi.device.type == "cuda"
    assert torch.equal(phi.cpu(), expected)


def _sh_solid_mask_cpu_float64(
    shape_zyx,
    spacing_a,
    origin_xyz,
    sh_axes,
    sh_amplitude,
    sh_max_degree,
    sh_spectrum_power,
    seed,
):
    """
    The pre-optimisation SH solid-mask construction, kept verbatim as reference.

    It built a dense (nz*ny*nx, 3) point cloud on the CPU and did every
    per-voxel operation in float64. The shipped version derives the same mask
    from three broadcast 1-D coordinate axes on `device` in float32; this
    exists so that equivalence is asserted against the original arithmetic.
    """
    import numpy as np

    from specter.specimen.membrane._field import _grid_points_xyz
    from specter.specimen.membrane._field_spherical_harmonics import (
        _angular_grid_resolution,
        _interpolate_angular_grid,
        _sample_sh_coefficients,
        _synthesize_angular_grid,
    )

    points = _grid_points_xyz(shape_zyx, spacing_a, origin_xyz, device="cpu")
    query = points.reshape(-1, 3).numpy()
    axes = np.asarray(sh_axes, dtype=np.float64)
    p_prime = query / axes
    r_prime = np.linalg.norm(p_prime, axis=-1)
    r_safe = np.clip(r_prime, 1e-12, None)
    theta = np.arccos(np.clip(p_prime[:, 2] / r_safe, -1.0, 1.0))
    phi_ang = np.arctan2(p_prime[:, 1], p_prime[:, 0])

    rng = np.random.default_rng(seed)
    coefficients = _sample_sh_coefficients(sh_max_degree, sh_spectrum_power, rng)
    n_theta, n_phi = _angular_grid_resolution(sh_max_degree)
    coarse = _synthesize_angular_grid(coefficients, sh_max_degree, n_theta, n_phi)
    pert = _interpolate_angular_grid(coarse, theta, phi_ang, device="cpu")
    r_surface = np.clip(1.0 + sh_amplitude * pert, 0.05, None)
    return torch.as_tensor((r_prime < r_surface).reshape(shape_zyx))


@pytest.mark.parametrize("seed", [11, 12, 14])
def test_sh_field_solid_mask_matches_cpu_float64_reference(seed: int) -> None:
    """
    The on-device float32 SH geometry reproduces the CPU float64 solid mask.

    `generate_membrane_field_spherical_harmonics` used to build its per-voxel
    geometry as a dense point cloud on the CPU in float64, which was the single
    largest cost in `specter build tomogram`'s membrane phase. It now broadcasts
    three 1-D coordinate axes on `device` in float32 instead.

    Equality has to be EXACT, not approximate. The mask is a threshold,
    `r_prime < r_surface`, and in float32 about one voxel in 10^8 lands close
    enough to tie the other way. Thirteen such voxels in a membrane shell were
    measured to move the region mask enough that the cytosol filler packing --
    a random sequential addition -- took a different accept/reject path and
    re-realized entirely, so a fixed `seed` no longer reproduced a fixed
    tomogram. Building each axis in float32 before upcasting reproduces NumPy's
    dtype sequence and with it the reference mask bit for bit; this test is what
    keeps that property from being optimised away again.
    """
    from specter.specimen.membrane._field_spherical_harmonics import (
        generate_membrane_field_spherical_harmonics,
    )

    shape = (48, 48, 48)
    spacing = 8.0
    axes = (120.0, 170.0, 140.0)
    extent = torch.tensor([shape[2], shape[1], shape[0]], dtype=torch.float32) * spacing
    origin = -0.5 * extent

    expected = _sh_solid_mask_cpu_float64(
        shape, spacing, origin, axes, 0.15, 6, 2.0, seed
    )
    field = generate_membrane_field_spherical_harmonics(
        shape_zyx=shape,
        spacing_a=spacing,
        sh_axes=axes,
        sh_amplitude=0.15,
        sh_max_degree=6,
        sh_spectrum_power=2.0,
        seed=seed,
        device="cpu",
    )
    # phi is negative strictly inside the solid, so its sign recovers the mask.
    got = field.phi < 0

    n_diff = int((got != expected).sum())
    assert n_diff == 0, (
        f"{n_diff} of {expected.numel()} solid-mask voxels differ from the "
        "CPU float64 reference"
    )


def test_interpolate_angular_grid_returns_tensor_for_tensor_input() -> None:
    """
    `_interpolate_angular_grid` returns the array type it was handed.

    The SH field passes torch tensors so a 100M-voxel working grid never leaves
    the GPU; `specimen/_grid.py`'s bead roughness still passes NumPy. Both call
    sites are load-bearing, so pin the contract rather than the current caller.
    """
    import numpy as np

    from specter.specimen.membrane._field_spherical_harmonics import (
        _angular_grid_resolution,
        _interpolate_angular_grid,
        _sample_sh_coefficients,
        _synthesize_angular_grid,
    )

    rng = np.random.default_rng(0)
    coefficients = _sample_sh_coefficients(6, 2.0, rng)
    n_theta, n_phi = _angular_grid_resolution(6)
    coarse = _synthesize_angular_grid(coefficients, 6, n_theta, n_phi)

    theta = rng.uniform(0.0, np.pi, 512)
    phi = rng.uniform(-np.pi, 3.0 * np.pi, 512)  # spans the wrap on both sides

    from_numpy = _interpolate_angular_grid(coarse, theta, phi, device="cpu")
    from_torch = _interpolate_angular_grid(
        coarse, torch.as_tensor(theta), torch.as_tensor(phi), device="cpu"
    )

    assert isinstance(from_numpy, np.ndarray)
    assert torch.is_tensor(from_torch)
    assert torch.allclose(
        torch.as_tensor(from_numpy, dtype=torch.float32), from_torch, atol=1e-6
    )


# ---------------------------------------------------------------------------
# One place answers "how big is this organelle"
# ---------------------------------------------------------------------------


def test_bounding_radius_matches_each_backend_formula() -> None:
    """The measurement itself, pinned so a refactor cannot quietly change it."""
    from specter.specimen.membrane import membrane_bounding_radius

    assert membrane_bounding_radius(
        "spherical_harmonics", sh_axes=(100.0, 250.0, 180.0)
    ) == pytest.approx(250.0)
    assert membrane_bounding_radius(
        "swept_spline", swept_total_length=2000.0, swept_tube_radius=300.0
    ) == pytest.approx(1300.0)


def test_bounding_radius_rejects_an_unknown_backend() -> None:
    """A new backend must be added here, not silently take another's formula."""
    from specter.specimen.membrane import membrane_bounding_radius

    with pytest.raises(ValueError, match="Unknown shape_backend"):
        membrane_bounding_radius("helicoid", sh_axes=(1.0, 1.0, 1.0))


def test_bounding_radius_requires_the_arguments_its_backend_needs() -> None:
    """Silently defaulting a missing size would understate the reach."""
    from specter.specimen.membrane import membrane_bounding_radius

    with pytest.raises(ValueError, match="sh_axes"):
        membrane_bounding_radius("spherical_harmonics")
    with pytest.raises(ValueError, match="swept_total_length"):
        membrane_bounding_radius("swept_spline", swept_tube_radius=300.0)


def test_collision_radius_and_generator_sizing_agree() -> None:
    """
    The tomogram placer and the generator's own auto-sizing must measure the
    same organelle the same way.

    They used to hold separate copies of the branch, so a third shape backend
    would have meant finding both -- and two that disagreed would place
    instances using one reach while sizing the grid for another.
    """
    from specter.specimen.membrane import MembraneGenerator, membrane_bounding_radius
    from specter.specimen.tomogram._helpers import _instance_bounding_radius

    for kwargs in (
        {"shape_backend": "spherical_harmonics", "sh_axes": (150.0, 200.0, 175.0)},
        {
            "shape_backend": "swept_spline",
            "swept_total_length": 1800.0,
            "swept_tube_radius": 250.0,
        },
    ):
        generator = MembraneGenerator(voxel_size=10.0, **kwargs)
        assert _instance_bounding_radius(generator) == pytest.approx(
            membrane_bounding_radius(
                generator.shape_backend,
                sh_axes=generator.sh_axes,
                swept_total_length=generator.swept_total_length,
                swept_tube_radius=generator.swept_tube_radius,
            )
        )


def test_generator_and_pipeline_share_one_set_of_default_ranges() -> None:
    """
    The auto-size cap must draw from the same defaults the generator would.

    `pipelines._tomogram` caps a fully-auto membrane against the tomogram box,
    and needs a lower bound to write a complete range with. It used to retype
    `MembraneGenerator`'s, so changing a default in one place left the other
    capping against a stale value -- with nothing failing, and only for a
    membrane whose size the user had left entirely unspecified.
    """
    import inspect

    import specter.pipelines._tomogram as tomogram_pipeline
    from specter.specimen import (
        DEFAULT_SH_AXES_RANGE_A,
        DEFAULT_SWEPT_TOTAL_LENGTH_RANGE_A,
        DEFAULT_SWEPT_TUBE_RADIUS_RANGE_A,
        MembraneGenerator,
    )

    params = inspect.signature(MembraneGenerator.__init__).parameters
    for name, constant in (
        ("sh_axes_range", DEFAULT_SH_AXES_RANGE_A),
        ("swept_total_length_range", DEFAULT_SWEPT_TOTAL_LENGTH_RANGE_A),
        ("swept_tube_radius_range", DEFAULT_SWEPT_TUBE_RADIUS_RANGE_A),
    ):
        assert params[name].default is constant, (
            f"MembraneGenerator.{name}'s default is no longer the shared constant"
        )
    assert tomogram_pipeline.DEFAULT_SH_AXES_RANGE_A is DEFAULT_SH_AXES_RANGE_A
