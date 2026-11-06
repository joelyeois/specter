"""
rotate_mode="exact": a volume rotated through its own transform (a NUFFT).

`rotate_volume_exact` must express the same pose convention as the trilinear
`VolumeRotator` (checked on a low-passed object, where trilinear interpolation
is accurate), be exact where exactness can be checked (quarter turns and
whole-voxel shifts of a band-limited object), have the adjoint as its
backward, keep every band of an atom-rendered specimen where trilinear loses
the top ones, and stay the default for simulation only.
"""

from __future__ import annotations

import math
import warnings

import pytest
import roma
import torch

from conftest import blob
from specter.imaging import Poses, Specimen
from specter.rotations import (
    VolumeRotator,
    build_affine_matrix,
    rotate_volume_exact,
    translations_angstrom_to_torch,
    wrapped_fraction,
)
from specter.rotations import _exact
from specter.settings import Propagation

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _blobs(
    shape: tuple[int, int, int], spread: float = 0.2, seed: int = 3
) -> torch.Tensor:
    """Smooth Gaussian blobs near the centre: band-limited and inside the sphere."""
    axes = [torch.arange(n, dtype=torch.float64) - n // 2 for n in shape]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    g = torch.Generator().manual_seed(seed)
    centres = (
        (torch.rand(10, 3, generator=g, dtype=torch.float64) - 0.5)
        * spread
        * min(shape)
    )
    v = torch.zeros(shape, dtype=torch.float64)
    for cx, cy, cz in centres:
        v += torch.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / 4.0)
    return v.float()


def _theta(n_poses: int, nx: int, max_shift: float, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    R = roma.random_rotmat(n_poses)
    shifts = (torch.rand(n_poses, 2) - 0.5) * 2 * max_shift  # voxels, 1 A voxels
    return build_affine_matrix(R, translations_angstrom_to_torch(shifts, nx, 1.0))


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum() / (a.norm() * b.norm()))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(32, 32, 32), (28, 32, 36), (31, 31, 31)])
def test_matches_trilinear_convention_on_a_smooth_object(
    device: str, shape: tuple[int, int, int]
) -> None:
    """Same pose convention as VolumeRotator: rotation direction, origin, shift sign."""
    V = _blobs(shape).to(device)
    theta = _theta(4, shape[2], 3.0).to(device)
    nz, ny, nx = shape
    tri = VolumeRotator(nz, ny, nx, origin="relion")(V, theta)
    ex = rotate_volume_exact(V, theta)
    assert ex.shape == tri.shape
    for a, b in zip(ex, tri):
        assert _corr(a, b) > 0.999
    # The wrong conventions are far off: the inverse rotation, the opposite shift.
    R, t = theta[:, :, :3], theta[:, :, 3:]
    inverse = torch.cat([R.transpose(1, 2), t], -1)
    flipped = torch.cat([R, -t], -1)
    for wrong in (inverse, flipped):
        bad = rotate_volume_exact(V, wrong)
        assert min(_corr(a, b) for a, b in zip(bad, tri)) < 0.99


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [32, 33])
def test_quarter_turns_and_whole_voxel_shifts_are_exact(device: str, n: int) -> None:
    V = _blobs((n, n, n), spread=0.15).to(device)
    quarter = roma.rotvec_to_unitquat(torch.tensor([[0.0, 0.0, math.pi / 2]]))
    R = roma.unitquat_to_rotmat(quarter)
    theta = build_affine_matrix(R).to(device)
    got = rotate_volume_exact(V, theta, eps=1e-6)[0]
    want = VolumeRotator(n, n, n, origin="relion")(V, theta)[0]  # exact on the grid
    assert float((got - want).norm() / want.norm()) < 1e-4
    shift = torch.tensor([[3.0, -2.0]])
    theta = build_affine_matrix(
        torch.eye(3)[None], translations_angstrom_to_torch(shift, n, 1.0)
    ).to(device)
    got = rotate_volume_exact(V, theta, eps=1e-6)[0]
    tri = VolumeRotator(n, n, n, origin="relion")(V, theta)[0]  # a whole-voxel roll
    assert float((got - tri).norm() / tri.norm()) < 1e-4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(24, 24, 24), (20, 24, 26)])
# White noise fills the box, so the wrap warning fires; the adjoint holds regardless.
@pytest.mark.filterwarnings("ignore:.*wraps it to the opposite face")
def test_backward_is_the_adjoint(device: str, shape: tuple[int, int, int]) -> None:
    """<A V, W> = <V, A^T W> with A^T the autograd backward."""
    g = torch.Generator().manual_seed(1)
    V = torch.randn(shape, generator=g).to(device).requires_grad_(True)
    W = torch.randn((3, *shape), generator=g).to(device)
    theta = _theta(3, shape[2], 2.5).to(device)
    out = rotate_volume_exact(V, theta, eps=1e-6)
    (grad,) = torch.autograd.grad((out * W).sum(), V)
    lhs = float((out.detach().double() * W.double()).sum())
    rhs = float((V.detach().double() * grad.double()).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-4


def test_poses_are_chunked_without_changing_the_result(monkeypatch) -> None:
    V = _blobs((24, 24, 24))
    theta = _theta(5, 24, 2.0)
    whole = rotate_volume_exact(V, theta, eps=1e-6)
    monkeypatch.setattr(_exact, "_POINTS_PER_CALL", 1)  # one pose per NUFFT call
    chunked = rotate_volume_exact(V, theta, eps=1e-6)
    torch.testing.assert_close(chunked, whole, atol=1e-5, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_and_gpu_agree() -> None:
    V = _blobs((32, 32, 32))
    theta = _theta(3, 32, 2.0)
    cpu = rotate_volume_exact(V, theta, eps=1e-6)
    gpu = rotate_volume_exact(V.cuda(), theta.cuda(), eps=1e-6).cpu()
    assert float((gpu - cpu).norm() / cpu.norm()) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_runs_forward_and_backward_under_autocast(dtype: torch.dtype) -> None:
    """The backward runs under the forward's autocast state; points must stay float32."""
    V = _blobs((32, 32, 32)).cuda().requires_grad_(True)
    theta = _theta(2, 32, 2.0).cuda()
    with torch.autocast("cuda", dtype=dtype):
        out = rotate_volume_exact(V * 1.0, theta)
        loss = (out**2).sum()
    loss.backward()
    assert out.dtype == torch.float32
    assert V.grad is not None and torch.isfinite(V.grad).all()
    plain = rotate_volume_exact(V.detach(), theta)
    torch.testing.assert_close(out.detach(), plain, atol=1e-5, rtol=0)


def test_warns_when_density_can_wrap() -> None:
    """Density a rotation and shift can carry past a face is reported, compact content is not."""
    compact = _blobs((32, 32, 32))
    theta = _theta(2, 32, 0.0)
    assert wrapped_fraction(compact) < 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rotate_volume_exact(compact, theta)
    corners = torch.zeros(32, 32, 32)
    corners[:4, :4, :4] = 1.0
    corners[16, 16, 16] = 1.0
    assert wrapped_fraction(corners) > 0.9
    with pytest.warns(UserWarning, match="wraps it to the opposite face"):
        rotate_volume_exact(corners, theta)
    # A shift brings density closer to the faces; the bound grows with it.
    assert wrapped_fraction(compact, shift=12.0) > wrapped_fraction(compact)


def test_wrapped_fraction_is_not_fooled_by_a_reused_address() -> None:
    """A template freed and replaced at the same address gets its own histogram."""
    for _ in range(20):
        compact = _blobs((32, 32, 32))
        assert wrapped_fraction(compact) < 1e-6
        del compact
        corners = torch.zeros(32, 32, 32)
        corners[:4, :4, :4] = 1.0
        assert wrapped_fraction(corners) > 0.9
        del corners
    V = torch.zeros(32, 32, 32)
    assert wrapped_fraction(V) == 0.0
    V[:4, :4, :4] = 1.0  # an in-place change is seen too
    assert wrapped_fraction(V) > 0.9


def test_wrapped_content_reenters_at_the_opposite_face() -> None:
    """What the warning is about: exact rotation wraps where trilinear clips."""
    n = 32
    z, y, x = torch.meshgrid(*[torch.arange(n, dtype=torch.float32)] * 3, indexing="ij")
    c = n // 2
    # A blob near the +x face, built in real space so that it has no tail
    # already wrapped round to the -x side.
    V = torch.exp(-((x - (n - 4)) ** 2 + (y - c) ** 2 + (z - c) ** 2) / 2.0)
    # Shift by 4 voxels towards +x: trilinear drops what leaves, exact wraps it.
    theta = build_affine_matrix(
        torch.eye(3)[None], torch.tensor([[-4.0 * 2 / n, 0.0, 0.0]])
    )
    with pytest.warns(UserWarning, match="wraps it to the opposite face"):
        ex = rotate_volume_exact(V, theta)[0]
    tri = VolumeRotator(n, n, n, origin="relion", padding_mode="zeros")(V, theta)[0]
    low = (slice(None), slice(None), slice(0, n // 4))
    assert float(ex[low].sum()) > 0.5 * float(V.sum())
    assert float(tri[low].abs().sum()) < 1e-3 * float(V.sum())


def _band(V: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    n = V.shape[-1]
    f = torch.fft.fftfreq(n)
    k = torch.sqrt(
        f[:, None, None] ** 2
        + f[None, :, None] ** 2
        + torch.fft.rfftfreq(n)[None, None] ** 2
    )
    sel = (k >= lo) & (k < hi)
    return torch.fft.rfftn(V, dim=(-3, -2, -1))[..., sel]


def test_keeps_the_top_band_of_an_atom_rendered_specimen_where_trilinear_loses_it() -> (
    None
):
    """Against per-pose atom rendering, the reference `Specimen.from_atoms` gives."""
    coords = blob(300, 9.0, seed=5)
    numbers = torch.full((300,), 6, dtype=torch.long)
    n, voxel = 40, 0.8
    atoms = Specimen.from_atoms(coords, numbers, voxel, n)
    template = Specimen(atoms.potential, voxel)
    torch.manual_seed(2)
    poses = Poses(
        roma.random_unitquat(4), (torch.rand(4, 2) - 0.5) * 3.0, torch.arange(4)
    )
    ref = atoms.materialize(poses, voltage=300.0)
    Fref = _band(ref, 0.3, 0.5)

    def ratio(mode: str) -> float:
        V = template.materialize(poses, voltage=300.0, rotate_mode=mode)
        F = _band(V, 0.3, 0.5)
        return float((F * Fref.conj()).real.sum() / (Fref.abs() ** 2).sum())

    exact, real = ratio("exact"), ratio("real")
    assert exact > 0.9
    assert real < 0.8


def _reconstructor(**kwargs: object):
    from specter.ghostbuster import Reconstructor

    ctf = {
        "dfu": torch.full((2,), 8000.0),
        "dfv": torch.full((2,), 8000.0),
        "dfang": torch.zeros(2),
        "cs": torch.full((2,), 2.7e7),
    }
    return Reconstructor(
        V=_blobs((24, 24, 24)),
        voxel_size=1.0,
        quaternions=roma.random_unitquat(2),
        translations=torch.zeros(2, 2),
        ctf_params=ctf,
        voltage=300.0,
        dose_per_angstrom=2.0,
        **kwargs,
    )


def test_simulation_defaults_to_exact_and_reconstruction_to_real() -> None:
    assert Propagation().rotate_mode is None
    spec = Specimen(_blobs((24, 24, 24)), 1.0)
    poses = Poses(roma.random_unitquat(2), torch.zeros(2, 2), torch.arange(2))
    assert "(exact)" in spec.plan()[0]
    default = spec.materialize(poses, voltage=300.0)
    exact = spec.materialize(poses, voltage=300.0, rotate_mode="exact")
    torch.testing.assert_close(default, exact)
    assert _reconstructor().propagation.rotate_mode == "real"
    exact_model = _reconstructor(propagation=Propagation(rotate_mode="exact"))
    assert exact_model.propagation.rotate_mode == "exact"


def test_reconstructor_refuses_pose_refinement_under_exact() -> None:
    for lr in ("lr_R", "lr_T"):
        with pytest.raises(ValueError, match="exact"):
            _reconstructor(propagation=Propagation(rotate_mode="exact"), **{lr: 1e-3})


@pytest.mark.parametrize("device", DEVICES)
def test_gradient_reaches_the_template_through_scatter(device: str) -> None:
    V = _blobs((24, 24, 24)).to(device).mul(5).requires_grad_(True)
    poses = Poses(
        roma.random_unitquat(2),
        torch.tensor([[1.3, -0.4], [0.0, 2.2]]),
        torch.arange(2),
    )
    weights = torch.rand(2, 24, 24, generator=torch.Generator().manual_seed(7))
    weights = weights.to(device, torch.float64)

    def loss(v: torch.Tensor) -> torch.Tensor:
        # Rytov's real part in double: a phase object's |wave|^2 barely moves,
        # and a float32 sum cannot resolve a finite difference of it.
        wave = Specimen(v, 1.0).scatter(
            poses, voltage=300.0, scattering_model="rytov", oversample=1
        )
        return (wave.array.real.double() * weights).sum()

    # Unset, the rotation is the simulators' default: exact.
    assert "(exact)" in Specimen(V.detach(), 1.0).plan()[0]
    (grad,) = torch.autograd.grad(loss(V), V)
    assert torch.isfinite(grad).all() and float(grad.abs().max()) > 0
    d = _blobs((24, 24, 24), seed=9).to(device)
    h = 5e-2
    with torch.no_grad():
        fd = (loss(V + h * d) - loss(V - h * d)) / (2 * h)
    assert float((grad * d).sum()) == pytest.approx(float(fd), rel=2e-2)


@pytest.mark.parametrize("device", DEVICES)
def test_backward_accepts_an_expanded_gradient(device: str) -> None:
    """U.sum() hands back a stride-0 gradient, which MKL's irfftn backward refuses."""
    V = _blobs((16, 16, 16)).to(device).requires_grad_()
    rotate_volume_exact(V, _theta(2, 16, 1.0).to(device)).sum().backward()
    assert V.grad is not None and bool(torch.isfinite(V.grad).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [24, 32])  # MKL's batched inverse varies at 32
def test_a_pose_does_not_depend_on_its_batch(device: str, n: int) -> None:
    """A seeded particle stack must be the same however it is batched, bit for bit."""
    V = _blobs((n, n, n)).to(device)
    theta = _theta(6, n, 2.0, seed=11).to(device)
    whole = rotate_volume_exact(V, theta)
    singles = torch.cat([rotate_volume_exact(V, theta[i : i + 1]) for i in range(6)])
    halves = torch.cat(
        [rotate_volume_exact(V, theta[:2]), rotate_volume_exact(V, theta[2:])]
    )
    assert torch.equal(whole, singles) and torch.equal(whole, halves)


def test_angstrom_shifts_follow_the_specimen_convention() -> None:
    """Exact and trilinear agree on where a shifted particle lands at a non-trivial pose."""
    V = _blobs((32, 32, 32))
    R = roma.unitquat_to_rotmat(
        roma.random_unitquat(1, generator=torch.Generator().manual_seed(9))
    )
    T = translations_angstrom_to_torch(torch.tensor([[4.5, -3.0]]), 32, 1.5)
    theta = build_affine_matrix(R, T)
    exact = rotate_volume_exact(V, theta)[0]
    trilinear = VolumeRotator(32, 32, 32, padding_mode="zeros")(V, theta)[0]
    grid = torch.meshgrid(*[torch.arange(32.0)] * 3, indexing="ij")
    centroid = lambda v: torch.stack([(v * c).sum() / v.sum() for c in grid])  # noqa: E731
    assert torch.allclose(centroid(exact), centroid(trilinear), atol=0.02)


def test_auto_batch_sizing_counts_the_plan() -> None:
    from specter.memory import estimate_peak_bytes
    from specter.rotations._exact import plan_bytes

    args = (2, 256, 256, 512, 2, True)
    extra = estimate_peak_bytes(*args, rotate_mode="exact") - estimate_peak_bytes(*args)
    assert extra == plan_bytes((256, 256, 256))
    assert plan_bytes((256, 256, 256)) == 512**3 * 8  # upsampfac 2, complex64


def test_scatter_never_samples_rotated_slices(monkeypatch) -> None:
    """Specimen.scatter poses the whole volume; the slice sampler is not reached."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("sample_rotated_slices reached")

    monkeypatch.setattr(VolumeRotator, "sample_rotated_slices", refuse)
    V = _blobs((24, 24, 24))
    poses = Poses(roma.random_unitquat(2), torch.ones(2, 2), torch.arange(2))
    for model in ("multislice", "rytov"):
        Specimen(V, 1.0).scatter(poses, voltage=300.0, scattering_model=model)
