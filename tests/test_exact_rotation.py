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


@pytest.mark.parametrize("device", DEVICES)
def test_poses_are_chunked_without_changing_the_result(
    monkeypatch, device: str
) -> None:
    """One pose per NUFFT call gives what all poses in one call give, both ways."""
    V = _blobs((24, 24, 24)).to(device)
    W = torch.randn(5, 24, 24, 24, generator=torch.Generator().manual_seed(2))
    W = W.to(device)
    theta = _theta(5, 24, 2.0).to(device)

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        v = V.clone().requires_grad_(True)
        out = rotate_volume_exact(v, theta, eps=1e-6)
        (grad,) = torch.autograd.grad((out * W).sum(), v)
        return out.detach(), grad

    whole, whole_grad = run()
    monkeypatch.setattr(_exact, "_points_per_call", lambda device: 1)
    assert len(_exact._chunks(5, 100, torch.device(device))) == 5
    chunked, chunked_grad = run()
    if device == "cuda":
        # cuFINUFFT's type 2 does not depend on the call's other points
        assert torch.equal(chunked, whole)
    torch.testing.assert_close(chunked, whole, atol=1e-5, rtol=0)
    # the type 1 spreads with atomic adds, so the backward is close, not equal
    torch.testing.assert_close(chunked_grad, whole_grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_gpu_call_takes_several_poses_of_a_reconstruction_box() -> None:
    """Every call repeats the oversampled grid's FFT, so poses share calls."""
    device = torch.device("cuda")
    M = _exact._grid((300, 300, 300), device, False).M
    poses = _exact._points_per_call(device) // M
    total = torch.cuda.get_device_properties(device).total_memory
    if total >= 40 * 2**30:  # an L40, A100 or H100
        assert poses >= 8


def _gaussians(
    shape: tuple[int, int, int], centres: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Gaussians at `centres` ((x, y, z) voxels) sampled on the grid, in float64."""
    axes = [torch.arange(n, dtype=torch.float64) for n in shape]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    v = torch.zeros(shape, dtype=torch.float64)
    for cx, cy, cz in centres.tolist():
        v += torch.exp(
            -((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * sigma**2)
        )
    return v


@pytest.mark.parametrize("device", DEVICES)
def test_matches_the_analytic_rotation_of_gaussians(device: str) -> None:
    """
    Against the rotated and shifted Gaussians rendered afresh, to the tolerance.

    Gaussians of sigma 2 voxels are band-limited to 3e-9 at Nyquist, so the
    rendering is the exact answer, and at eps 1e-4 the rotation reaches it to
    6e-5 at boxes 256-512 (4e-2 for trilinear interpolation).
    """
    n, sigma = 40, 2.0
    g = torch.Generator().manual_seed(4)
    c = torch.full((3,), float(n // 2), dtype=torch.float64)
    d = torch.randn(12, 3, generator=g, dtype=torch.float64)
    mu = c + d / d.norm(dim=1, keepdim=True) * 9 * torch.rand(12, 1, generator=g)
    R = roma.random_rotmat(3, generator=g).double()
    T_vox = (torch.rand(3, 3, generator=g, dtype=torch.float64) - 0.5) * 4
    theta = build_affine_matrix(R.float(), (T_vox / (n / 2)).float())
    V = _gaussians((n, n, n), mu, sigma).float().to(device)
    out = rotate_volume_exact(V, theta.to(device)).double().cpu()
    for i in range(3):
        # theta = [R | R T]: a Gaussian at mu lands at c + R^T (mu - c) - T
        expected = _gaussians((n, n, n), c + (mu - c) @ R[i] - T_vox[i], sigma)
        assert float((out[i] - expected).norm() / expected.norm()) < 2e-4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.filterwarnings("ignore:.*wraps it to the opposite face")
def test_gradient_matches_a_double_precision_reference(device: str) -> None:
    """The single-precision backward, against the eps-1e-9 float64 one (1.2e-4 at 256-300^3)."""
    shape = (32, 32, 32)
    g = torch.Generator().manual_seed(6)
    V = torch.randn(shape, generator=g)
    W = torch.randn((4, *shape), generator=g)
    theta = _theta(4, 32, 2.5)

    def grad(
        v: torch.Tensor, w: torch.Tensor, t: torch.Tensor, eps: float
    ) -> torch.Tensor:
        v = v.to(device).requires_grad_(True)
        out = rotate_volume_exact(v, t.to(device), eps=eps)
        return torch.autograd.grad((out * w.to(device)).sum(), v)[0].double().cpu()

    reference = grad(V.double(), W.double(), theta.double(), 1e-9)
    single = grad(V, W, theta, _exact.DEFAULT_EXACT_TOLERANCE)
    assert float((single - reference).norm() / reference.norm()) < 5e-4


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


@pytest.mark.parametrize(
    ("scattering_model", "paired"),
    [
        ("multislice", "exact"),
        ("kinematic", "exact"),
        ("rytov", "surface"),
        ("firstborn", "surface"),
        ("projection", "surface"),
        ("ctf", "surface"),
    ],
)
def test_an_unset_rotate_mode_pairs_with_the_scattering_model(
    scattering_model: str, paired: str
) -> None:
    """Simulation and reconstruction pair alike, with no interpolation."""
    p = Propagation(scattering_model=scattering_model)
    assert p.rotate_mode is None
    assert p.rotation() == paired
    spec = Specimen(_blobs((24, 24, 24)), 1.0)
    assert f"({paired})" in spec.plan(propagation=p)[0] or (
        paired == "surface" and "posed" in spec.plan(propagation=p)[0]
    )
    model = _reconstructor(propagation=p)
    assert model.propagation.rotate_mode == paired


def test_the_pairing_falls_back_where_it_cannot_apply() -> None:
    rytov, multislice = Propagation(scattering_model="rytov"), Propagation()
    # The surface carries no crowding, ice or damage: the posed volume is formed.
    assert rytov.rotation(surface=False) == "exact"
    damaged = Specimen(_blobs((24, 24, 24)), 1.0).damage()
    assert damaged.dressing() == ["damage"]
    assert "(exact)" in damaged.plan(propagation=rytov)[0]
    absorbing = Propagation(
        scattering_model="rytov", absorption_model="inelastic_mfp", alpha=0.0
    )
    assert absorbing.rotation() == "exact"
    # Both modes carry gradients to the pose, so refinement keeps the pairing.
    for lr in ("lr_R", "lr_T"):
        model = _reconstructor(propagation=rytov, **{lr: 1e-3})
        assert model.propagation.rotate_mode == "surface"
        model = _reconstructor(propagation=multislice, **{lr: 1e-3})
        assert model.propagation.rotate_mode == "exact"
    # A materialised volume is formed by definition.
    poses = Poses(roma.random_unitquat(2), torch.zeros(2, 2), torch.arange(2))
    spec = Specimen(_blobs((24, 24, 24)), 1.0)
    torch.testing.assert_close(
        spec.materialize(poses, voltage=300.0, propagation=rytov),
        spec.materialize(poses, voltage=300.0, rotate_mode="exact"),
    )


def test_a_set_rotate_mode_is_used_as_given() -> None:
    p = Propagation(scattering_model="rytov", rotate_mode="real")
    assert p.rotation(surface=False) == "real"
    exact_model = _reconstructor(propagation=Propagation(rotate_mode="exact"))
    assert exact_model.propagation.rotate_mode == "exact"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(13, 13, 13), (12, 14, 16)])
def test_pose_gradient_matches_finite_differences(
    device: str, shape: tuple[int, int, int]
) -> None:
    """
    The gradient to the affine, its rotation block and shift column alike.

    In float64 at eps 1e-12 a central difference and the analytic gradient
    agree to ~1e-10; a wrong sign, axis order or per-axis shift scale (the
    non-cubic box) is off by order one.
    """
    V = torch.zeros(shape, dtype=torch.float64)
    n = min(shape)
    V[:n, :n, :n] = _blobs((n, n, n)).double()
    V = V.to(device)
    generator = torch.Generator().manual_seed(3)
    R = roma.random_rotmat(3, dtype=torch.float64)
    T = 0.5 * torch.randn(3, 3, dtype=torch.float64, generator=generator)
    theta = build_affine_matrix(R, T).to(device).requires_grad_()
    weights = torch.rand((3, *shape), dtype=torch.float64, generator=generator)
    weights = weights.to(device)

    def loss(th: torch.Tensor) -> torch.Tensor:
        return (rotate_volume_exact(V, th, eps=1e-12) * weights).sum()

    loss(theta).backward()
    assert theta.grad is not None
    direction = torch.randn(3, 3, 4, dtype=torch.float64, generator=generator)
    direction = direction.to(device)
    h = 1e-6
    with torch.no_grad():
        fd = (loss(theta + h * direction) - loss(theta - h * direction)) / (2 * h)
    analytic = (theta.grad * direction).sum()
    assert float(analytic) == pytest.approx(float(fd), rel=1e-7)


def test_reconstructor_refines_poses_under_exact() -> None:
    model = _reconstructor(
        propagation=Propagation(rotate_mode="exact"), lr=None, lr_R=1e-3, lr_T=1e-3
    )
    images = model(torch.arange(2)).detach() * 1.1
    optimizers, _ = model.configure_optimizers()
    before = model.rotations.detach().clone(), model.translations.detach().clone()
    loss = ((model(torch.arange(2)) - images) ** 2).mean()
    loss.backward()
    for opt in optimizers:
        opt.step()
    assert not torch.equal(model.rotations.detach(), before[0])
    assert not torch.equal(model.translations.detach(), before[1])


def _corner_content(n: int) -> torch.Tensor:
    """Density only where the exact-rotation window is zero."""
    from specter.ghostbuster._resolution import exact_rotation_support

    noise = torch.rand(n, n, n, generator=torch.Generator().manual_seed(8))
    return noise * (exact_rotation_support(n) == 0)


@pytest.mark.parametrize("rotate_mode", ["exact", "surface"])
def test_a_reconstruction_sees_only_the_windowed_map(rotate_mode: str) -> None:
    """Corner density, which both modes would carry into the image, neither predicts nor learns."""
    model = _reconstructor(
        propagation=Propagation(scattering_model="rytov", rotate_mode=rotate_mode),
        lr=0.1,
    )
    assert model.support is not None
    idx = torch.arange(2)
    with torch.no_grad():
        clean = model.forward(idx)
        model.V.add_(_corner_content(24))
        torch.testing.assert_close(model.forward(idx), clean, rtol=0, atol=0)
    model.forward(idx).sum().backward()
    assert model.V.grad is not None
    assert float(model.V.grad[model.support == 0].abs().max()) == 0.0
    assert float(model.V.grad.abs().max()) > 0.0
    # the map written out is the one the forward saw
    torch.testing.assert_close(
        model._volume_cpu(), (model.V.detach() * model.support).cpu()
    )
    assert "support" not in model.state_dict()  # rebuilt from the box size


@pytest.mark.parametrize("rotate_mode", ["real", "fourier"])
def test_interpolating_pose_modes_are_not_windowed(rotate_mode: str) -> None:
    model = _reconstructor(
        propagation=Propagation(scattering_model="rytov", rotate_mode=rotate_mode)
    )
    assert model.support is None
    torch.testing.assert_close(model._volume_cpu(), model.V.detach().cpu())


def test_warns_when_the_fsc_mask_reaches_past_the_window() -> None:
    n = 24
    full = torch.ones(n, n, n)
    exact = Propagation(scattering_model="rytov", rotate_mode="exact")
    with pytest.warns(UserWarning, match="fsc_mask"):
        _reconstructor(propagation=exact, fsc_mask=full)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from specter.ghostbuster._resolution import exact_rotation_support

        _reconstructor(propagation=exact, fsc_mask=exact_rotation_support(n) > 0.99)
        _reconstructor(  # not windowed, so not checked
            propagation=Propagation(scattering_model="rytov", rotate_mode="real"),
            fsc_mask=full,
        )


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
    exact = estimate_peak_bytes(*args, rotate_mode="exact", device="cpu")
    extra = exact - estimate_peak_bytes(*args, device="cpu")
    assert extra == plan_bytes((256, 256, 256), "cpu")
    assert plan_bytes((256, 256, 256), "cpu") == 512**3 * 8  # upsampfac 2, complex64


@pytest.mark.parametrize("device", DEVICES)
def test_upsampling_factor_follows_the_device_memory(device: str) -> None:
    """Factor 2 while one plan's grid fits the device's budget, 1.25 beyond it."""
    d = torch.device(device)
    if d.type == "cuda":
        budget = torch.cuda.get_device_properties(d).total_memory / 16
    else:
        budget = 2 * 2**30
    largest = int((budget / 8) ** (1 / 3) / 2)  # factor-2 grid of (2n)^3 complex64
    assert _exact._plan_settings((largest,) * 3, False, d) == 2.0
    assert _exact._plan_settings((largest + 2,) * 3, False, d) == 1.25


def test_scatter_never_samples_rotated_slices(monkeypatch) -> None:
    """Specimen.scatter poses the whole volume; the slice sampler is not reached."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("sample_rotated_slices reached")

    monkeypatch.setattr(VolumeRotator, "sample_rotated_slices", refuse)
    V = _blobs((24, 24, 24))
    poses = Poses(roma.random_unitquat(2), torch.ones(2, 2), torch.arange(2))
    for model in ("multislice", "rytov"):
        Specimen(V, 1.0).scatter(poses, voltage=300.0, scattering_model=model)
