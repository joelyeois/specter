import math
import warnings

import pytest
import torch

from specter.specimen.membrane._field import (
    MembraneField,
    MetaballSource,
    _value_noise_grid,
    blend_field,
    cap_curvature,
    generate_membrane_field,
)


def test_single_source_field_matches_sphere_sdf():
    source = MetaballSource(center_xyz=torch.zeros(3), radius=50.0)
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 80.0, 0.0], [30.0, 40.0, 0.0]]
    )
    values = blend_field([source], points, k=0.0)
    expected = torch.linalg.norm(points, dim=-1) - 50.0
    assert torch.allclose(values, expected, atol=1e-4)


def test_field_sample_and_gradient_on_sphere_surface():
    spacing_a = 2.0
    n = 80
    origin_xyz = torch.full((3,), -0.5 * n * spacing_a)
    field = MembraneField(
        phi=blend_field(
            [MetaballSource(center_xyz=torch.zeros(3), radius=50.0)],
            _dense_grid_points(spacing_a=spacing_a, n=n, origin_xyz=origin_xyz),
            k=0.0,
        ),
        spacing_a=spacing_a,
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
        MetaballSource(center_xyz=torch.tensor([-30.0, 0.0, 0.0]), radius=40.0),
        MetaballSource(center_xyz=torch.tensor([30.0, 0.0, 0.0]), radius=40.0),
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
    relaxed = cap_curvature(noisy, spacing_a=5.0, iterations=25)

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
    assert torch.equal(cap_curvature(phi, spacing_a=5.0, iterations=0), phi)


def test_generate_membrane_field_is_seed_reproducible_and_has_organic_shape():
    kwargs = dict(
        shape_zyx=(48, 48, 48),
        spacing_a=5.0,
        n_sources=5,
        radius_range_a=(40.0, 70.0),
        spread_a=20.0,
        noise_amplitude_a=8.0,
        correlation_length_a=30.0,
        curvature_iterations=10,
        seed=42,
    )
    field_a = generate_membrane_field(**kwargs)
    field_b = generate_membrane_field(**kwargs)

    assert field_a.phi.shape == (48, 48, 48)
    assert torch.equal(field_a.phi, field_b.phi)

    inside_fraction = (field_a.phi < 0).float().mean().item()
    assert 0.02 < inside_fraction < 0.9

    query_points = torch.tensor([[0.0, 0.0, 0.0], [20.0, -10.0, 5.0]])
    gradient = field_a.gradient(query_points)
    norms = torch.linalg.norm(gradient, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-4)


def test_generate_membrane_field_different_seeds_differ():
    kwargs = dict(
        shape_zyx=(32, 32, 32),
        spacing_a=5.0,
        n_sources=4,
        radius_range_a=(30.0, 50.0),
        spread_a=10.0,
        noise_amplitude_a=5.0,
    )
    field_a = generate_membrane_field(seed=1, **kwargs)
    field_b = generate_membrane_field(seed=2, **kwargs)
    assert not torch.equal(field_a.phi, field_b.phi)


def _dense_grid_points(
    spacing_a: float, n: int, origin_xyz: torch.Tensor
) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.float32)
    coords = origin_xyz[0] + idx * spacing_a
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1)


def test_generate_membrane_field_resolution_independent_shape():
    common = dict(
        n_sources=3,
        radius_range_a=(60.0, 60.0),
        spread_a=0.0,
        noise_amplitude_a=0.0,
        curvature_iterations=0,
        seed=7,
    )
    coarse = generate_membrane_field(shape_zyx=(32, 32, 32), spacing_a=10.0, **common)
    fine = generate_membrane_field(shape_zyx=(64, 64, 64), spacing_a=5.0, **common)

    probe_xyz = torch.tensor([[60.0, 0.0, 0.0]])
    coarse_value = coarse.sample(probe_xyz)
    fine_value = fine.sample(probe_xyz)
    assert math.isclose(float(coarse_value), float(fine_value), abs_tol=2.0)


def test_value_noise_grid_upsampling_has_no_long_linear_segments():
    # Regression test: F.interpolate's "trilinear" mode is only linear
    # interpolation between lattice points, which left visible
    # piecewise-flat (faceted/polygonal) creases in the membrane boundary
    # wherever the noise lattice was sparse relative to the domain --
    # confirmed by disabling the post-upsample blur and reproducing a
    # visibly faceted membrane. A genuinely smooth noise field's second
    # finite difference should rarely be exactly flat; a piecewise-linear
    # one is flat (near zero) almost everywhere between lattice points.
    noise = _value_noise_grid(
        shape_zyx=(96, 8, 8),
        spacing_a=2.5,
        correlation_length_a=40.0,
        octaves=1,
        seed=0,
        device=torch.device("cpu"),
    )
    line = noise[:, 4, 4]
    first_diff = line[1:] - line[:-1]
    second_diff = first_diff[1:] - first_diff[:-1]
    near_zero_fraction = (second_diff.abs() < 1e-4).float().mean()
    assert near_zero_fraction < 0.5


def test_generate_membrane_field_warns_when_shape_too_small_for_sources():
    # Regression test: a box too small for spread_a + max(radius_range_a) +
    # noise_amplitude_a silently truncated the organic shape at the grid
    # boundary -- producing an unphysical flat cut where the box face cuts
    # through the membrane -- with no signal to the caller. Chosen sizes
    # here mirror a real case that produced exactly that artifact.
    with pytest.warns(UserWarning, match="clipped"):
        generate_membrane_field(
            shape_zyx=(48, 48, 48),
            spacing_a=10.0,
            n_sources=5,
            radius_range_a=(150.0, 240.0),
            spread_a=70.0,
            noise_amplitude_a=20.0,
            curvature_iterations=10,
            seed=1004,
        )


def test_generate_membrane_field_does_not_warn_when_shape_is_large_enough():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        generate_membrane_field(
            shape_zyx=(48, 48, 48),
            spacing_a=5.0,
            n_sources=5,
            radius_range_a=(40.0, 70.0),
            spread_a=20.0,
            noise_amplitude_a=8.0,
            curvature_iterations=10,
            seed=42,
        )
