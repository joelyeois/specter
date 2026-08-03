import pytest
import torch

from specter.specimen._membrane_field import MembraneField, blend_field, MetaballSource
from specter.specimen._membrane_placement import (
    align_principal_axis_to_z,
    align_transmembrane_depth,
    orientation_for_normal,
    sample_surface_sites,
)


def _sphere_field(
    radius: float = 100.0, spacing_a: float = 4.0, device: str | torch.device = "cpu"
) -> MembraneField:
    extent = 4 * radius
    n = int(extent / spacing_a)
    origin = torch.full((3,), -0.5 * n * spacing_a, device=device)
    idx = torch.arange(n, dtype=torch.float32, device=device)
    zz, yy, xx = torch.meshgrid(
        origin[2] + idx * spacing_a,
        origin[1] + idx * spacing_a,
        origin[0] + idx * spacing_a,
        indexing="ij",
    )
    points_xyz = torch.stack([xx, yy, zz], dim=-1)
    phi = blend_field(
        [MetaballSource(center_xyz=torch.zeros(3, device=device), radius=radius)],
        points_xyz,
        k=0.0,
    )
    return MembraneField(phi=phi, spacing_a=spacing_a, origin_xyz=origin)


def test_sample_surface_sites_on_cuda_device():
    # Regression test: an earlier version built `extent` on the default
    # (CPU) device regardless of the field's own device, crashing
    # immediately with a device-mismatch error the moment a GPU field was
    # used -- invisible in the CPU-only test suite.
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    field = _sphere_field(radius=100.0, spacing_a=4.0, device="cuda")
    sites, normals = sample_surface_sites(field, n_sites=3, min_spacing_a=30.0, seed=0)
    assert sites.device.type == "cuda"
    assert normals.device.type == "cuda"
    assert sites.shape[0] >= 1


def test_sample_surface_sites_on_sphere_are_on_surface_with_min_spacing():
    field = _sphere_field(radius=100.0, spacing_a=4.0)
    min_spacing_a = 40.0

    sites, normals = sample_surface_sites(
        field, n_sites=10, min_spacing_a=min_spacing_a, seed=0
    )

    assert sites.shape[0] >= 5
    assert sites.shape == normals.shape

    phi_at_sites = field.sample(sites)
    assert (phi_at_sites.abs() < 1.0).all()

    if sites.shape[0] > 1:
        dists = torch.cdist(sites, sites)
        dists.fill_diagonal_(float("inf"))
        assert (dists.min(dim=-1).values >= min_spacing_a - 1e-3).all()

    normal_norms = torch.linalg.norm(normals, dim=-1)
    assert torch.allclose(normal_norms, torch.ones_like(normal_norms), atol=1e-3)

    radial_direction = sites / torch.linalg.norm(sites, dim=-1, keepdim=True)
    cosine = (normals * radial_direction).sum(dim=-1)
    assert (cosine > 0.9).all()


def test_sample_surface_sites_seed_reproducible():
    field = _sphere_field()
    sites_a, _ = sample_surface_sites(field, n_sites=5, min_spacing_a=30.0, seed=1)
    sites_b, _ = sample_surface_sites(field, n_sites=5, min_spacing_a=30.0, seed=1)
    assert torch.equal(sites_a, sites_b)


def test_orientation_for_normal_aligns_z_axis_and_is_a_valid_rotation():
    z_hat = torch.tensor([0.0, 0.0, 1.0])
    normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    for normal in normals:
        R = orientation_for_normal(normal, seed=0)
        expected = normal / normal.norm()
        assert torch.allclose(R @ z_hat, expected, atol=1e-4)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-4)
        assert torch.isclose(torch.det(R), torch.tensor(1.0), atol=1e-4)


def test_orientation_for_normal_seed_controls_spin_only():
    normal = torch.tensor([0.3, 0.5, 0.8])
    R_a = orientation_for_normal(normal, seed=0)
    R_b = orientation_for_normal(normal, seed=1)
    z_hat = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(R_a @ z_hat, R_b @ z_hat, atol=1e-4)
    assert not torch.allclose(R_a, R_b, atol=1e-3)

    R_a_repeat = orientation_for_normal(normal, seed=0)
    assert torch.allclose(R_a, R_a_repeat)


def test_align_transmembrane_depth_centers_span_and_preserves_shape():
    coordinates = torch.tensor(
        [
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 25.0],
            [1.0, 2.0, -30.0],
        ]
    )
    tm_span_mask = torch.tensor([True, True, False, False])

    dists_before = torch.cdist(coordinates, coordinates)
    shifted = align_transmembrane_depth(coordinates, tm_span_mask)
    dists_after = torch.cdist(shifted, shifted)

    assert torch.allclose(dists_before, dists_after, atol=1e-5)

    span_z = shifted[tm_span_mask][:, 2]
    assert torch.isclose(
        0.5 * (span_z.max() + span_z.min()), torch.tensor(0.0), atol=1e-5
    )


def test_align_transmembrane_depth_without_mask_centers_full_structure():
    coordinates = torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, 30.0]])
    shifted = align_transmembrane_depth(coordinates, tm_span_mask=None)
    z = shifted[:, 2]
    assert torch.isclose(0.5 * (z.max() + z.min()), torch.tensor(0.0), atol=1e-5)


def test_align_principal_axis_to_z_aligns_elongated_structure_and_preserves_shape():
    # Regression test: a real structure's principal (longest) axis of
    # inertia diverged substantially from its native file z-axis (measured
    # on PDB 1FE1: 201.8 A along the true principal axis vs. only 162.2 A
    # along raw z, with the axis vector nowhere near (0,0,1)) -- depth
    # alignment along the wrong axis left an asymmetric extramembrane
    # domain pointing off at an angle instead of cleanly away from the
    # membrane.
    torch.manual_seed(0)
    direction = torch.tensor([1.0, 2.0, 0.5])
    direction = direction / direction.norm()
    t = torch.linspace(-50.0, 50.0, 200).unsqueeze(-1)
    jitter = torch.randn(200, 3) * 2.0
    coords = t * direction + jitter

    dists_before = torch.cdist(coords, coords)
    aligned = align_principal_axis_to_z(coords)
    dists_after = torch.cdist(aligned, aligned)
    # A pure rotation preserves distances exactly in infinite precision;
    # this tolerance just accommodates float32 error through the
    # eigendecomposition + axis-angle rotation construction, not a
    # meaningful bound on distortion.
    assert torch.allclose(dists_before, dists_after, atol=0.1)

    z_extent = aligned[:, 2].max() - aligned[:, 2].min()
    xy_extent = torch.linalg.norm(aligned[:, :2], dim=-1).max()
    assert z_extent > 80.0
    assert xy_extent < 20.0


def test_align_principal_axis_to_z_handles_near_symmetric_structure():
    torch.manual_seed(0)
    coords = torch.randn(100, 3) * 10.0
    aligned = align_principal_axis_to_z(coords)
    assert torch.isfinite(aligned).all()
