import torch

from specter.specimen.membrane._field import MembraneField
from specter.specimen.membrane._field_swept_spline import (
    generate_membrane_field_swept_spline,
)
from specter.specimen.membrane._profile import (
    BilayerProfile,
    build_reference_lipid_patch,
    compute_bilayer_profile,
)
from specter.specimen.membrane._raster import rasterize_membrane_density


def _synthetic_bilayer_profile() -> BilayerProfile:
    d = torch.linspace(-30.0, 30.0, 121)
    psi = torch.exp(-((d.abs() - 20.0) ** 2) / (2 * 2.0**2))
    return BilayerProfile(distance_angstrom=d, psi=psi)


def test_rasterize_at_matching_resolution_reproduces_profile_of_phi():
    spacing_angstrom = 5.0
    shape = (20, 20, 20)
    origin = torch.full((3,), -0.5 * 20 * spacing_angstrom)
    phi = torch.linspace(-30.0, 30.0, 20).reshape(20, 1, 1).expand(20, 20, 20).clone()
    field = MembraneField(phi=phi, spacing_angstrom=spacing_angstrom, origin_xyz=origin)
    profile = _synthetic_bilayer_profile()

    density = rasterize_membrane_density(
        field,
        profile,
        target_shape=shape,
        target_spacing_angstrom=spacing_angstrom,
        target_origin_xyz=origin,
    )
    expected = profile(phi)
    assert torch.allclose(density, expected, atol=0.05)


def test_rasterize_coarser_output_blurs_rather_than_distorts_peak_separation():
    fine_spacing = 2.0
    fine_n = 60
    origin = torch.full((3,), -0.5 * fine_n * fine_spacing)
    z_phys = origin[2] + torch.arange(fine_n, dtype=torch.float32) * fine_spacing
    phi = z_phys.reshape(fine_n, 1, 1).expand(fine_n, fine_n, fine_n).clone()
    field = MembraneField(phi=phi, spacing_angstrom=fine_spacing, origin_xyz=origin)
    profile = _synthetic_bilayer_profile()

    coarse_spacing = 12.0
    coarse_n = int(fine_n * fine_spacing / coarse_spacing)
    coarse_shape = (coarse_n, coarse_n, coarse_n)
    density_coarse = rasterize_membrane_density(
        field,
        profile,
        target_shape=coarse_shape,
        target_spacing_angstrom=coarse_spacing,
        target_origin_xyz=origin,
    )

    mid = coarse_n // 2
    line = density_coarse[:, mid, mid]
    peak_idx = torch.topk(line, 2).indices.sort().values
    z_coords = origin[2] + torch.arange(coarse_n, dtype=torch.float32) * coarse_spacing
    peak_positions = z_coords[peak_idx]
    separation = float((peak_positions[1] - peak_positions[0]).abs())
    # True separation is 40 A; anti-aliased coarse sampling should stay
    # close to it, not blow up the way naive point-sampling of the same
    # narrow profile does (see _membrane_raster module docstring).
    assert abs(separation - 40.0) < 2 * coarse_spacing


def test_rasterize_output_shape_matches_request():
    spacing_angstrom = 5.0
    shape = (10, 10, 10)
    origin = torch.zeros(3)
    phi = torch.zeros(shape)
    field = MembraneField(phi=phi, spacing_angstrom=spacing_angstrom, origin_xyz=origin)
    profile = _synthetic_bilayer_profile()

    out_shape = (7, 9, 11)
    density = rasterize_membrane_density(
        field, profile, target_shape=out_shape, target_spacing_angstrom=6.0
    )
    assert density.shape == out_shape


def test_rasterize_end_to_end_with_real_field_and_calibrated_profile():
    field = generate_membrane_field_swept_spline(
        shape_zyx=(40, 40, 40),
        spacing_angstrom=4.0,
        total_length_angstrom=80.0,
        step_length_angstrom=6.0,
        tube_radius_angstrom=12.0,
        curvature_iterations=10,
        seed=0,
    )
    atomic_numbers, coordinates = build_reference_lipid_patch(
        n_lipids_per_leaflet=6, jitter_angstrom=2.0, seed=0
    )
    profile = compute_bilayer_profile(
        atomic_numbers, coordinates, voxel_size=2.0, parameterization="shtyrov"
    )

    density = rasterize_membrane_density(
        field, profile, target_shape=(20, 20, 20), target_spacing_angstrom=8.0
    )
    assert density.shape == (20, 20, 20)
    assert torch.isfinite(density).all()
    assert density.max() > 0
