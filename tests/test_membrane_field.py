import torch

from specter.specimen.membrane._field import (
    MembraneField,
    _centred_origin_xyz,
    _grid_points_xyz,
)


def _sphere_field(radius: float, spacing_angstrom: float, n: int) -> MembraneField:
    origin_xyz = torch.full((3,), -0.5 * n * spacing_angstrom)
    coords = origin_xyz[0] + torch.arange(n, dtype=torch.float32) * spacing_angstrom
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    phi = torch.stack([xx, yy, zz], dim=-1).norm(dim=-1) - radius
    return MembraneField(
        phi=phi, spacing_angstrom=spacing_angstrom, origin_xyz=origin_xyz
    )


def test_field_sample_and_gradient_on_sphere_surface() -> None:
    field = _sphere_field(radius=50.0, spacing_angstrom=2.0, n=80)

    surface_points = torch.tensor(
        [[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, -50.0]]
    )
    sampled = field.sample(surface_points)
    assert torch.allclose(sampled, torch.zeros(3), atol=1.0)

    gradient = field.gradient(surface_points)
    expected_normals = surface_points / 50.0
    cosine = (gradient * expected_normals).sum(dim=-1)
    assert (cosine > 0.95).all()


def test_field_samples_the_signed_distance_off_the_surface() -> None:
    """Negative inside, positive outside, and metric on a coarse grid."""
    field = _sphere_field(radius=100.0, spacing_angstrom=10.0, n=40)
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [70.0, 0.0, 0.0], [0.0, 130.0, 0.0], [60.0, 0.0, 80.0]]
    )
    expected = points.norm(dim=-1) - 100.0
    # Trilinear interpolation of a cone is exact away from its apex.
    assert torch.allclose(field.sample(points)[1:], expected[1:], atol=1.0)
    assert field.sample(points)[0] < -90.0


def test_grid_points_put_the_physical_origin_on_voxel_n_over_2() -> None:
    shape_zyx = (5, 6, 8)
    origin_xyz = _centred_origin_xyz(shape_zyx, 4.0)
    points = _grid_points_xyz(shape_zyx, 4.0, origin_xyz, "cpu")
    assert points.shape == (5, 6, 8, 3)
    assert torch.allclose(points[5 // 2, 6 // 2, 8 // 2], torch.zeros(3))
    assert torch.allclose(points[0, 0, 1] - points[0, 0, 0], torch.tensor([4.0, 0, 0]))
