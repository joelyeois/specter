"""Geometry tests for the filament random-walk path, parallel-transported
orientation, and per-species placement -- pure functions, no PDB fetch, no
PotentialBuilder/rendering, so these run fast and fully offline.
"""

from __future__ import annotations

import math

import pytest
import roma
import torch

from specter.specimen.filament import (
    ACTIN_SPEC,
    DIMER_REPEAT,
    PROTOFILAMENT_SPEC,
    FilamentSpec,
    place_filaments,
)
from specter.specimen.filament._path import generate_filament_path
from specter.specimen.filament._placement import (
    filament_orientations,
    filament_tangents,
)
from conftest import seeded

# F-actin's deposited helix, from the helical parameters of 6DJO
# (-166.55 deg, 27.45 A) and 7BT7 (-166.6 deg, 27.5 A), which superposing
# consecutive subunits of the deposited coordinates reproduces. A negative
# twist for a positive rise: the one-start (genetic) helix is left-handed
# (Dominguez & Holmes 2011).
_DEPOSITED_ACTIN_TWIST_DEG = -166.6
_DEPOSITED_ACTIN_RISE = 27.5


def _screw(rotation: torch.Tensor, translation: torch.Tensor) -> tuple[float, float]:
    """(twist in degrees, rise) of the rigid motion ``x -> R x + t``, about
    the screw axis oriented so that the rise is positive."""
    rotvec = roma.rotmat_to_rotvec(rotation.double())
    angle = float(rotvec.norm())
    axis = rotvec / angle
    rise = float(translation.double() @ axis)
    twist = math.degrees(angle)
    return (twist, rise) if rise >= 0 else (-twist, -rise)


def _straight_actin(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A straight ACTIN_SPEC filament along an oblique direction: monomer
    positions and their template-to-world rotations."""
    direction = torch.tensor([0.3, -0.8, 0.52])
    direction = direction / direction.norm()
    positions = torch.arange(n, dtype=torch.float32).unsqueeze(1) * (
        ACTIN_SPEC.step * direction
    )
    return positions, filament_orientations(positions, ACTIN_SPEC.twist_deg)


def test_generate_filament_path_step_spacing():
    n, step = 12, ACTIN_SPEC.step
    positions = generate_filament_path(n, step, flex_deg=12.0, generator=seeded(0))
    assert positions.shape == (n, 3)
    dists = (positions[1:] - positions[:-1]).norm(dim=-1)
    assert torch.allclose(dists, torch.full_like(dists, step), atol=1e-3)


def test_generate_filament_path_flex_bound():
    n, step, flex_deg = 20, PROTOFILAMENT_SPEC.step, 3.0
    positions = generate_filament_path(n, step, flex_deg=flex_deg, generator=seeded(1))
    tangents = filament_tangents(positions)
    # Consecutive-segment tangents (excluding the padded last repeat) must
    # not turn by more than flex_deg per step.
    cos_turn = (tangents[:-2] * tangents[1:-1]).sum(dim=-1).clamp(-1.0, 1.0)
    turn_deg = torch.rad2deg(torch.arccos(cos_turn))
    assert bool((turn_deg <= flex_deg + 1e-3).all())


def test_generate_filament_path_zero_flex_is_straight_line():
    n, step = 8, 10.0
    positions = generate_filament_path(n, step, flex_deg=0.0, generator=seeded(2))
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
        5, step=20.0, flex_deg=5.0, origin_xyz=origin, generator=seeded(3)
    )
    assert torch.allclose(positions[0], origin)


def test_filament_tangents_unit_norm_and_shape():
    positions = generate_filament_path(6, step=15.0, flex_deg=8.0, generator=seeded(4))
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
        7, step=ACTIN_SPEC.step, flex_deg=12.0, generator=seeded(5)
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
    positions = generate_filament_path(
        5, step=ACTIN_SPEC.step, flex_deg=0.0, generator=seeded(6)
    )
    twist_deg = ACTIN_SPEC.twist_deg
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


def test_off_axis_subunits_keep_their_spacing_where_the_path_runs_along_minus_z():
    """The only roll between consecutive monomers is the twist.

    Each monomer's frame is parallel-transported along the path, so a
    subunit sitting off the filament axis stays one subunit spacing from
    its neighbour however the path bends. Rotating the template's ``+Z``
    onto each tangent independently instead rolls the frame wherever the
    path runs close to ``-Z``, where that rotation is singular: on this path
    it moved subunits 15.7 A off the axis up to 12.4 A out of place.
    """
    n, radius = 41, 15.7
    sweep = torch.linspace(-0.35, 0.35, n)
    tangents = torch.stack(
        [torch.full_like(sweep, 0.003), sweep, -torch.ones_like(sweep)], dim=1
    )
    tangents = tangents / tangents.norm(dim=1, keepdim=True)
    bent = torch.cat(
        [torch.zeros(1, 3), torch.cumsum(ACTIN_SPEC.step * tangents[:-1], dim=0)]
    )
    straight = torch.arange(n, dtype=torch.float32).unsqueeze(1) * torch.tensor(
        [0.0, 0.0, ACTIN_SPEC.step]
    )
    offset = torch.tensor([radius, 0.0, 0.0])

    def spacing(positions: torch.Tensor) -> torch.Tensor:
        centres = (
            positions + filament_orientations(positions, ACTIN_SPEC.twist_deg) @ offset
        )
        return (centres[1:] - centres[:-1]).norm(dim=-1)

    # About 1 deg of bend per step: only that bend may change the spacing.
    assert (spacing(bent) - spacing(straight).mean()).abs().max() < 0.5


def test_presets_take_their_numbers_from_measured_structures():
    """F-actin's rise and twist are the deposited filaments' (not the
    idealised 13/6 twist of -166.15 deg), and a free protofilament stacks
    dimers at the microtubule lattice's own repeat."""
    assert ACTIN_SPEC.step == pytest.approx(_DEPOSITED_ACTIN_RISE, abs=0.1)
    assert ACTIN_SPEC.twist_deg == pytest.approx(_DEPOSITED_ACTIN_TWIST_DEG, abs=0.1)
    assert PROTOFILAMENT_SPEC.step == DIMER_REPEAT


def test_actin_preset_is_a_left_handed_one_start_helix():
    """Consecutive monomers are related by F-actin's deposited screw.

    The screw carrying monomer i onto monomer i+1 must turn -166.6 deg
    about the filament axis for +27.5 A of rise. +166.6 deg would be the
    mirror helix, which the old +166.15 deg preset described.
    """
    positions, rotations = _straight_actin(6)
    direction = positions[1] - positions[0]
    direction = direction / direction.norm()
    for i in range(5):
        rotation = rotations[i + 1] @ rotations[i].T
        translation = positions[i + 1] - rotation @ positions[i]
        twist, rise = _screw(rotation, translation)
        assert twist == pytest.approx(_DEPOSITED_ACTIN_TWIST_DEG, abs=0.05)
        assert rise == pytest.approx(_DEPOSITED_ACTIN_RISE, abs=0.05)
        # The screw axis is the filament axis.
        axis = roma.rotmat_to_rotvec(rotation)
        assert abs(float(axis @ direction)) == pytest.approx(
            float(axis.norm()), rel=1e-4
        )


def test_actin_long_pitch_strands_are_right_handed():
    """Every second monomer lies on one of the two long-pitch strands, which
    a left-handed -166.6 deg one-start helix makes right-handed:
    2 * -166.6 + 360 = +26.8 deg per 55 A."""
    positions, rotations = _straight_actin(5)
    rotation = rotations[2] @ rotations[0].T
    translation = positions[2] - rotation @ positions[0]
    twist, rise = _screw(rotation, translation)
    assert twist == pytest.approx(2 * _DEPOSITED_ACTIN_TWIST_DEG + 360, abs=0.1)
    assert twist > 0
    assert rise == pytest.approx(2 * _DEPOSITED_ACTIN_RISE, abs=0.1)


def test_place_filaments_fixed_monomer_count():
    specs = [FilamentSpec(code="X", step=10.0, flex_deg=5.0, n_copies=3, n_monomers=6)]
    instances = place_filaments(
        specs, target_shape=(64, 64, 64), voxel_size=5.0, generator=seeded(7)
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
        specs, target_shape=(64, 64, 64), voxel_size=5.0, generator=seeded(8)
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
        specs, target_shape=(64, 64, 64), voxel_size=5.0, generator=seeded(9)
    )
    assert sum(1 for inst in instances if inst.code == "A") == 3
    assert sum(1 for inst in instances if inst.code == "B") == 4


def test_place_filaments_reproducible_with_seeded_generator():
    specs = [
        FilamentSpec(code="X", step=27.5, flex_deg=12.0, n_copies=2, n_monomers=(5, 9))
    ]
    a = place_filaments(specs, (64, 64, 64), 5.0, generator=seeded(11))
    b = place_filaments(specs, (64, 64, 64), 5.0, generator=seeded(11))
    assert len(a) == len(b)
    for ia, ib in zip(a, b):
        assert torch.allclose(ia.position_xyz, ib.position_xyz)
        assert torch.allclose(ia.rotation_matrix, ib.rotation_matrix)


def test_place_filaments_start_within_volume_extent():
    target_shape = (50, 40, 30)  # (Z, Y, X)
    voxel_size = 4.0
    specs = [FilamentSpec(code="X", step=5.0, flex_deg=5.0, n_copies=25, n_monomers=1)]
    instances = place_filaments(specs, target_shape, voxel_size, generator=seeded(12))
    extent_xyz = torch.tensor(target_shape[::-1], dtype=torch.float32) * voxel_size
    for inst in instances:
        assert bool((inst.position_xyz >= 0).all())
        assert bool((inst.position_xyz <= extent_xyz).all())
