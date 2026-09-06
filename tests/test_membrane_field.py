import torch

from specter.specimen.membrane._field import (
    MembraneField,
    SphereSource,
    blend_field,
    cap_curvature,
)


def test_single_source_field_matches_sphere_sdf():
    source = SphereSource(center_xyz=torch.zeros(3), radius=50.0)
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 80.0, 0.0], [30.0, 40.0, 0.0]]
    )
    values = blend_field([source], points, k=0.0)
    expected = torch.linalg.norm(points, dim=-1) - 50.0
    assert torch.allclose(values, expected, atol=1e-4)


def test_field_sample_and_gradient_on_sphere_surface():
    spacing_angstrom = 2.0
    n = 80
    origin_xyz = torch.full((3,), -0.5 * n * spacing_angstrom)
    field = MembraneField(
        phi=blend_field(
            [SphereSource(center_xyz=torch.zeros(3), radius=50.0)],
            _dense_grid_points(
                spacing_angstrom=spacing_angstrom, n=n, origin_xyz=origin_xyz
            ),
            k=0.0,
        ),
        spacing_angstrom=spacing_angstrom,
        origin_xyz=origin_xyz,
    )

    surface_points = torch.tensor(
        [[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, -50.0]]
    )
    sampled = field.sample(surface_points)
    assert torch.allclose(sampled, torch.zeros(3), atol=1.0)

    gradient = field.gradient(surface_points)
    expected_normals = surface_points / 50.0
    cosine = (gradient * expected_normals).sum(dim=-1)
    assert (cosine > 0.95).all()


def test_smooth_min_blend_is_softer_than_hard_min_at_merge_point():
    sources = [
        SphereSource(center_xyz=torch.tensor([-30.0, 0.0, 0.0]), radius=40.0),
        SphereSource(center_xyz=torch.tensor([30.0, 0.0, 0.0]), radius=40.0),
    ]
    midpoint = torch.tensor([[0.0, 0.0, 0.0]])

    hard = blend_field(sources, midpoint, k=0.0)
    soft = blend_field(sources, midpoint, k=20.0)

    hard_min = min(
        torch.linalg.norm(midpoint - s.center_xyz, dim=-1) - s.radius for s in sources
    )
    assert torch.isclose(hard[0], hard_min)
    assert soft[0] < hard[0]


def test_cap_curvature_reduces_high_frequency_variance():
    torch.manual_seed(0)
    noisy = torch.randn(20, 20, 20)
    relaxed = cap_curvature(noisy, spacing_angstrom=5.0, iterations=25)

    def _laplacian_energy(volume: torch.Tensor) -> float:
        d = volume[1:-1, 1:-1, 1:-1]
        lap = (
            -6 * d
            + volume[:-2, 1:-1, 1:-1]
            + volume[2:, 1:-1, 1:-1]
            + volume[1:-1, :-2, 1:-1]
            + volume[1:-1, 2:, 1:-1]
            + volume[1:-1, 1:-1, :-2]
            + volume[1:-1, 1:-1, 2:]
        )
        return float((lap**2).mean())

    assert _laplacian_energy(relaxed) < 0.1 * _laplacian_energy(noisy)


def test_cap_curvature_zero_iterations_is_identity():
    torch.manual_seed(0)
    phi = torch.randn(8, 8, 8)
    assert torch.equal(cap_curvature(phi, spacing_angstrom=5.0, iterations=0), phi)


def _dense_grid_points(
    spacing_angstrom: float, n: int, origin_xyz: torch.Tensor
) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.float32)
    coords = origin_xyz[0] + idx * spacing_angstrom
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1)
