"""
`specter.alignment` recovers a known pose, to symmetry axes and to a reference.

The phantoms are Gaussian blobs copied by the group and rendered analytically
at a rotated, shifted pose, so the pose that aligns them is known exactly and
no resampling enters the reference.
"""

from __future__ import annotations

import math
import warnings

import pytest
import roma
import torch

from specter.alignment import (
    Alignment,
    align_to_symmetry,
    align_volumes,
    apply_alignment,
)
from specter.alignment._band import widest_band
from specter.alignment._frames import frame_distance, frame_group
from specter.alignment._ladder import _reach
from specter.symmetries import get_rotation_matrices, symmetrize_exact

N = 64


def _blobs(
    n: int,
    sym: str,
    rotation: torch.Tensor | None = None,
    centre: torch.Tensor | None = None,
    seed: int = 1,
) -> torch.Tensor:
    """
    Blobs placed in one asymmetric unit and copied by the group.

    A blob at aligned-frame position ``b`` lands at
    ``n // 2 + centre + rotation @ b``, so the alignment is ``(rotation,
    centre)``.
    """
    g = torch.Generator().manual_seed(seed)
    ops = get_rotation_matrices(sym, return_affine=False).double()
    b = (torch.rand(12, 3, generator=g, dtype=torch.float64) - 0.5) * 0.4 * n
    amp = (0.5 + torch.rand(12, generator=g, dtype=torch.float64)).repeat(len(ops))
    pos = torch.einsum("gij,bj->gbi", ops, b).reshape(-1, 3)
    if rotation is not None:
        pos = pos @ rotation.T
    if centre is not None:
        pos = pos + centre
    ax = torch.arange(n, dtype=torch.float64) - n // 2
    gauss = torch.exp(-((ax[None, None, :] - pos[:, :, None]) ** 2) / (2 * 1.5**2))
    vol = torch.einsum("b,bz,by,bx->zyx", amp, gauss[:, 2], gauss[:, 1], gauss[:, 0])
    return vol.float()


def _pose(seed: int, max_shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    quat = torch.randn(4, generator=g, dtype=torch.float64)
    rot = roma.unitquat_to_rotmat(quat / quat.norm())
    centre = (torch.rand(3, generator=g, dtype=torch.float64) - 0.5) * 2 * max_shift
    return rot, centre


def _errors(
    result, sym: str, rotation: torch.Tensor, centre: torch.Tensor
) -> tuple[float, float]:
    """Angle (degrees) and centre (voxels) error, up to the equivalent frames."""
    frames = frame_group(sym)
    angle = math.degrees(float(frame_distance(result.rotation, rotation, frames)))
    d = result.centre - centre
    if frames is None:
        d = d - (d @ rotation[:, 2]) * rotation[:, 2]
    return angle, float(d.norm())


@pytest.mark.parametrize("sym", ["C2", "C3", "D2", "D7", "T", "O", "I"])
def test_recovers_a_known_pose(sym: str) -> None:
    rotation, centre = _pose(3, N / 10)
    result = align_to_symmetry(_blobs(N, sym, rotation, centre), sym, resample=False)
    angle, shift = _errors(result, sym, rotation, centre)
    assert angle < 0.01
    assert shift < 0.01
    assert result.score > 0.99
    assert result.reliable
    assert result.volume is None


def test_recovers_a_known_pose_in_an_odd_box() -> None:
    rotation, centre = _pose(4, 6.0)
    result = align_to_symmetry(_blobs(65, "D3", rotation, centre), "D3", resample=False)
    angle, shift = _errors(result, "D3", rotation, centre)
    assert angle < 0.01
    assert shift < 0.01


@pytest.mark.parametrize("sym", ["C3", "D3", "O"])
def test_an_aligned_volume_is_left_where_it_is(sym: str) -> None:
    """Of the equivalent frames, the one nearest the identity is returned."""
    result = align_to_symmetry(_blobs(N, sym), sym, resample=False)
    assert torch.allclose(result.rotation, torch.eye(3, dtype=torch.float64), atol=1e-4)
    assert float(result.centre.norm()) < 1e-3


def test_the_cyclic_frame_keeps_the_spin_and_the_nearest_point_of_the_axis() -> None:
    rotation, centre = _pose(5, 6.0)
    result = align_to_symmetry(_blobs(N, "C5", rotation, centre), "C5", resample=False)
    axis = result.rotation[:, 2]
    assert float(axis[2]) > 0
    assert abs(float(result.centre @ axis)) < 1e-9
    # no spin about z: the rotation is the shortest one taking z to the axis
    assert abs(float(result.rotation[0, 1] - result.rotation[1, 0])) < 1e-9


def test_the_aligned_volume_has_the_symmetry() -> None:
    rotation, centre = _pose(6, 5.0)
    volume = _blobs(N, "D3", rotation, centre)
    aligned = align_to_symmetry(volume, "D3").volume
    assert aligned is not None
    ops = get_rotation_matrices("D3", return_affine=False)
    averaged = symmetrize_exact(aligned, ops)
    assert float((averaged - aligned).norm() / aligned.norm()) < 1e-3
    # the input does not
    assert float((symmetrize_exact(volume, ops) - volume).norm() / volume.norm()) > 0.5


def test_apply_alignment_reproduces_the_aligned_volume_and_carries_gradients() -> None:
    rotation, centre = _pose(7, 5.0)
    volume = _blobs(N, "C3", rotation, centre).requires_grad_(True)
    result = align_to_symmetry(volume, "C3")
    again = apply_alignment(volume, rotation=result.rotation, centre=result.centre)
    assert torch.allclose(again, result.volume, atol=1e-5)
    (grad,) = torch.autograd.grad(again.square().sum(), volume)
    assert float(grad.abs().max()) > 0


def test_noise_degrades_the_pose_gracefully() -> None:
    """White noise of ten times the volume's variance costs a fraction of a degree."""
    rotation, centre = _pose(8, 5.0)
    volume = _blobs(N, "D3", rotation, centre)
    g = torch.Generator().manual_seed(0)
    noisy = volume + torch.randn(volume.shape, generator=g) * volume.std() / 0.1**0.5
    result = align_to_symmetry(noisy, "D3", resample=False)
    angle, shift = _errors(result, "D3", rotation, centre)
    assert angle < 0.5
    assert shift < 0.2
    assert result.reliable
    assert result.score < 0.9


def test_a_volume_buried_in_noise_is_reported_as_unreliable() -> None:
    volume = _blobs(N, "D3", *_pose(9, 5.0))
    g = torch.Generator().manual_seed(0)
    noisy = volume + torch.randn(volume.shape, generator=g) * volume.std() / 1e-4**0.5
    with pytest.warns(UserWarning, match="do not agree beyond the band"):
        result = align_to_symmetry(noisy, "D3", resample=False)
    assert not result.reliable


def test_a_reliable_alignment_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        align_to_symmetry(_blobs(N, "C3", *_pose(3, 5.0)), "C3", resample=False)


def test_rejects_what_it_cannot_align() -> None:
    with pytest.raises(ValueError, match="cubic"):
        align_to_symmetry(torch.zeros(32, 32, 30), "C2")
    with pytest.raises(ValueError, match="no symmetry axis"):
        align_to_symmetry(_blobs(32, "C1"), "C1")


@pytest.mark.parametrize(
    ("sym", "size"),
    [("D2", 24), ("D3", 12), ("D7", 28), ("T", 24), ("O", 24), ("I", 60)],
)
def test_frame_group_is_the_normaliser(sym: str, size: int) -> None:
    frames = frame_group(sym)
    assert frames is not None
    assert frames.shape[0] == size
    assert torch.allclose(frames[0], torch.eye(3, dtype=torch.float64))
    group = get_rotation_matrices(sym, return_affine=False).double()
    conj = torch.einsum("nij,gjk,nlk->ngil", frames, group, frames)
    nearest = (conj[:, :, None] - group[None, None]).abs().amax((-1, -2)).amin(-1)
    assert float(nearest.max()) < 1e-5


def _noisy(volume: torch.Tensor, snr: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return volume + torch.randn(volume.shape, generator=g) * volume.std() / snr**0.5


def _pair_errors(
    result: Alignment, sym: str, rotation: torch.Tensor, centre: torch.Tensor
) -> tuple[float, float]:
    """Angle (degrees) and shift (voxels) error, up to the volume's own symmetry."""
    group = get_rotation_matrices(sym, return_affine=False).double()
    angle = math.degrees(float(frame_distance(result.rotation, rotation, group)))
    return angle, float((result.centre - centre).norm())


@pytest.mark.parametrize(("n", "sym"), [(48, "C1"), (49, "C1"), (48, "D3")])
def test_align_volumes_recovers_a_known_pose(n: int, sym: str) -> None:
    """Whatever the scale and background level of the moving volume."""
    rotation, centre = _pose(11, n / 6)
    volume = 3.0 * _blobs(n, sym, rotation, centre) + 0.7
    result = align_volumes(volume, _blobs(n, sym), resample=False)
    angle, shift = _pair_errors(result, sym, rotation, centre)
    assert angle < 0.01
    assert shift < 0.01
    assert result.score > 0.99
    assert result.reliable


def test_align_volumes_returns_the_volume_in_the_reference_frame() -> None:
    reference = _blobs(48, "C1")
    volume = _blobs(48, "C1", *_pose(12, 6.0))
    aligned = align_volumes(volume, reference).volume
    assert aligned is not None
    assert float((aligned - reference).norm() / reference.norm()) < 1e-3
    assert float((volume - reference).norm() / reference.norm()) > 0.5


def test_align_volumes_tolerates_independent_noise_in_both() -> None:
    rotation, centre = _pose(13, 6.0)
    volume = _noisy(_blobs(N, "C1", rotation, centre), 0.1, 1)
    reference = _noisy(_blobs(N, "C1"), 0.1, 2)
    result = align_volumes(volume, reference, resample=False)
    angle, shift = _pair_errors(result, "C1", rotation, centre)
    assert angle < 2.0
    assert shift < 0.2
    assert result.reliable
    assert result.score < 0.9


def test_volumes_buried_in_noise_are_reported_as_unreliable() -> None:
    volume = _noisy(_blobs(48, "C1", *_pose(14, 6.0)), 1e-4, 1)
    reference = _noisy(_blobs(48, "C1"), 1e-4, 2)
    with pytest.warns(UserWarning, match="do not correlate beyond the band"):
        result = align_volumes(volume, reference, resample=False)
    assert not result.reliable


def test_the_band_reached_follows_the_signal() -> None:
    """A clean pair is refined to the widest band, a noisy one to where it fades."""
    rotation, centre = _pose(16, 6.0)
    volume, reference = _blobs(N, "C1", rotation, centre), _blobs(N, "C1")
    clean = align_volumes(volume, reference, resample=False)
    noisy = align_volumes(
        _noisy(volume, 0.1, 1), _noisy(reference, 0.1, 2), resample=False
    )
    capped = align_volumes(volume, reference, max_radius=12.0, resample=False)
    assert clean.radius == widest_band(N)
    assert 8.0 < noisy.radius < clean.radius
    assert capped.radius == 12.0


def test_reach_is_the_unbroken_run_of_agreeing_windows() -> None:
    z = torch.zeros(60, dtype=torch.float64)
    z[:30] = 2.0  # 2 sigma a shell: 5.7 sigma over a window of eight
    z[50:] = 9.0  # agreement past a gap is not reached
    assert _reach(z, 8.0) == 32  # the window 25..32 still sums to 3.5 sigma
    assert _reach(torch.zeros(60, dtype=torch.float64), 8.0) == 8
    assert _reach(z[:20], 8.0) == 19  # the last window is cut by the band


def test_align_volumes_rejects_mismatched_grids() -> None:
    with pytest.raises(ValueError, match="cubic"):
        align_volumes(torch.zeros(32, 32, 30), torch.zeros(32, 32, 30))
    with pytest.raises(ValueError, match="grid"):
        align_volumes(torch.zeros(32, 32, 32), torch.zeros(48, 48, 48))


def test_a_background_level_does_not_move_the_symmetry_alignment() -> None:
    rotation, centre = _pose(15, 5.0)
    volume = _blobs(N, "D3", rotation, centre) + 2.0
    result = align_to_symmetry(volume, "D3", resample=False)
    angle, shift = _errors(result, "D3", rotation, centre)
    assert angle < 0.01
    assert shift < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_recovers_a_known_pose_on_the_gpu() -> None:
    rotation, centre = _pose(3, N / 10)
    volume = _blobs(N, "O", rotation, centre).cuda()
    result = align_to_symmetry(volume, "O")
    assert result.volume is not None and result.volume.is_cuda
    angle, shift = _errors(
        Alignment(None, result.rotation.cpu(), result.centre.cpu(), 0, 0, 0),
        "O",
        rotation,
        centre,
    )
    assert angle < 0.01
    assert shift < 0.01
