"""Geometry tests for the filament random-walk path, tangent-aligned
orientation, and per-species placement -- pure functions, no PDB fetch, no
PotentialBuilder/rendering, so these run fast and fully offline.
"""

from __future__ import annotations

import math

import torch

from specter.specimen.filament import FilamentSpec, place_filaments
from specter.specimen.filament._path import generate_filament_path
from specter.specimen.filament._placement import (
    filament_orientations,
    filament_tangents,
)


def _seeded(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_generate_filament_path_step_spacing():
    n, step = 12, 27.3
    positions = generate_filament_path(n, step, flex_deg=12.0, generator=_seeded(0))
    assert positions.shape == (n, 3)
    dists = (positions[1:] - positions[:-1]).norm(dim=-1)
    assert torch.allclose(dists, torch.full_like(dists, step), atol=1e-3)


def test_generate_filament_path_flex_bound():
    n, step, flex_deg = 20, 85.0, 3.0
    positions = generate_filament_path(n, step, flex_deg=flex_deg, generator=_seeded(1))
    tangents = filament_tangents(positions)
    # Consecutive-segment tangents (excluding the padded last repeat) must
    # not turn by more than flex_deg per step.
    cos_turn = (tangents[:-2] * tangents[1:-1]).sum(dim=-1).clamp(-1.0, 1.0)
    turn_deg = torch.rad2deg(torch.arccos(cos_turn))
    assert bool((turn_deg <= flex_deg + 1e-3).all())


def test_generate_filament_path_zero_flex_is_straight_line():
    n, step = 8, 10.0
    positions = generate_filament_path(n, step, flex_deg=0.0, generator=_seeded(2))
    tangents = filament_tangents(positions)
    # No turning allowed at all -- every tangent must match the first.
    assert torch.allclose(tangents, tangents[0].expand_as(tangents), atol=1e-4)


def test_generate_filament_path_single_monomer():
    positions = generate_filament_path(1, step=50.0, flex_deg=10.0)
    assert positions.shape == (1, 3)
    assert torch.allclose(positions[0], torch.zeros(3))


def test_generate_filament_path_respects_origin():
    origin = torch.tensor([100.0, -50.0, 25.0])
    positions = generate_filament_path(
        5, step=20.0, flex_deg=5.0, origin_xyz=origin, generator=_seeded(3)
    )
    assert torch.allclose(positions[0], origin)


def test_filament_tangents_unit_norm_and_shape():
    positions = generate_filament_path(6, step=15.0, flex_deg=8.0, generator=_seeded(4))
    tangents = filament_tangents(positions)
    assert tangents.shape == positions.shape
    norms = tangents.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # Last tangent reuses the second-to-last (no outgoing segment at the end).
    assert torch.allclose(tangents[-1], tangents[-2])


def test_filament_tangents_single_monomer_is_z():
    tangents = filament_tangents(torch.zeros((1, 3)))
    assert torch.allclose(tangents[0], torch.tensor([0.0, 0.0, 1.0]))


def test_filament_orientations_align_local_z_to_tangent():
    positions = generate_filament_path(
        7, step=27.3, flex_deg=12.0, generator=_seeded(5)
    )
    tangents = filament_tangents(positions)
    rotations = filament_orientations(positions, twist_deg=0.0)
    assert rotations.shape == (7, 3, 3)
    z_hat = torch.tensor([0.0, 0.0, 1.0])
    for i in range(7):
        mapped = rotations[i] @ z_hat
        assert torch.allclose(mapped, tangents[i], atol=1e-4)
        # Genuine rotation matrix: orthonormal, determinant +1.
        assert torch.allclose(rotations[i] @ rotations[i].T, torch.eye(3), atol=1e-4)
        assert math.isclose(float(torch.linalg.det(rotations[i])), 1.0, abs_tol=1e-4)


def test_filament_orientations_twist_rotates_about_tangent():
    # A straight filament (flex=0) has a constant tangent, so consecutive
    # monomers' orientation differs by exactly a rotation of twist_deg about
    # that shared tangent.
    positions = generate_filament_path(5, step=27.3, flex_deg=0.0, generator=_seeded(6))
    twist_deg = 166.15
    rotations = filament_orientations(positions, twist_deg=twist_deg)
    tangent = filament_tangents(positions)[0]

    relative = rotations[1] @ rotations[0].T
    cos_angle = ((torch.trace(relative) - 1) / 2).clamp(-1.0, 1.0)
    angle_deg = float(torch.rad2deg(torch.arccos(cos_angle)))
    # Rotation angle magnitude matches |twist_deg| (mod sign/axis direction).
    assert math.isclose(angle_deg, twist_deg % 360, abs_tol=1e-2) or math.isclose(
        angle_deg, 360 - (twist_deg % 360), abs_tol=1e-2
    )
    # And the relative rotation's axis is the shared tangent (relative
    # leaves it fixed).
    assert torch.allclose(relative @ tangent, tangent, atol=1e-3)


def test_place_filaments_fixed_monomer_count():
    specs = [FilamentSpec(code="X", step=10.0, flex_deg=5.0, n_copies=3, n_monomers=6)]
    instances = place_filaments(
        specs, target_shape=(64, 64, 64), voxel_size=5.0, generator=_seeded(7)
    )
    assert len(instances) == 3 * 6
    assert {inst.filament_id for inst in instances} == {0, 1, 2}
    for filament_id in range(3):
        indices = sorted(
            inst.monomer_index for inst in instances if inst.filament_id == filament_id
        )
        assert indices == list(range(6))
    assert all(inst.code == "X" for inst in instances)


def test_place_filaments_monomer_count_range():
    specs = [
        FilamentSpec(code="X", step=10.0, flex_deg=5.0, n_copies=20, n_monomers=(2, 4))
    ]
    instances = place_filaments(
        specs, target_shape=(64, 64, 64), voxel_size=5.0, generator=_seeded(8)
    )
    counts = {}
    for inst in instances:
        counts[inst.filament_id] = counts.get(inst.filament_id, 0) + 1
    assert all(2 <= c <= 4 for c in counts.values())


def test_place_filaments_multiple_species():
    specs = [
        FilamentSpec(code="A", step=10.0, flex_deg=5.0, n_copies=1, n_monomers=3),
        FilamentSpec(code="B", step=20.0, flex_deg=5.0, n_copies=1, n_monomers=4),
    ]
    instances = place_filaments(
        specs, target_shape=(64, 64, 64), voxel_size=5.0, generator=_seeded(9)
    )
    assert sum(1 for inst in instances if inst.code == "A") == 3
    assert sum(1 for inst in instances if inst.code == "B") == 4


def test_place_filaments_reproducible_with_seeded_generator():
    specs = [
        FilamentSpec(code="X", step=27.3, flex_deg=12.0, n_copies=2, n_monomers=(5, 9))
    ]
    a = place_filaments(specs, (64, 64, 64), 5.0, generator=_seeded(11))
    b = place_filaments(specs, (64, 64, 64), 5.0, generator=_seeded(11))
    assert len(a) == len(b)
    for ia, ib in zip(a, b):
        assert torch.allclose(ia.position_xyz, ib.position_xyz)
        assert torch.allclose(ia.rotation_matrix, ib.rotation_matrix)


def test_place_filaments_start_within_volume_extent():
    target_shape = (50, 40, 30)  # (Z, Y, X)
    voxel_size = 4.0
    specs = [FilamentSpec(code="X", step=5.0, flex_deg=5.0, n_copies=25, n_monomers=1)]
    instances = place_filaments(specs, target_shape, voxel_size, generator=_seeded(12))
    extent_xyz = torch.tensor(target_shape[::-1], dtype=torch.float32) * voxel_size
    for inst in instances:
        assert bool((inst.position_xyz >= 0).all())
        assert bool((inst.position_xyz <= extent_xyz).all())
