"""
Rotation of a voxel grid without interpolation, through its own transform.

A rotated and shifted copy of a band-limited volume, ``U(x) = V(c + R (x - c)
+ tau)``, has the transform ``U^(p) = V^(R p) exp(2 pi i (R p) . tau)``
(``V^`` taken about the centre voxel ``c``). `rotate_volume_exact` evaluates
``V^`` at the rotated half-spectrum frequencies inside the Nyquist sphere
with one type-2 non-uniform FFT (`specter.nufft`), applies the shift as a
phase, and inverts with ``irfftn``; frequencies outside the sphere have no
rotated counterpart inside the source's band and are set to zero. The
backward is the adjoint, a type-1 NUFFT that spreads every pose's spectrum
gradient onto one grid. Only the rotations and shifts are saved, and the
volume when the pose itself is differentiated.

The gradient to the pose follows from ``U^(p) = V^(w) exp(2 pi i w . tau)``
with ``w = R p``: its change with ``w`` is ``(grad V^(w) + 2 pi i tau V^(w))``
times the phase, and with ``tau`` it is ``2 pi i w U^(p)``. Each component
of ``grad V^`` is the transform of ``-2 pi i (x_j - c_j) V``, so one more
type-2 NUFFT, of ``V`` and its three coordinate-weighted copies at the
points the backward already sets, gives everything the pose needs.

The transform is periodic, so content farther from the centre than the
inscribed sphere (less the shift) re-enters the box at the opposite face
instead of leaving it. :func:`wrapped_fraction` bounds how much can, and
`rotate_volume_exact` warns when that bound passes 1%.
"""

from __future__ import annotations

import functools
import math
import warnings
import weakref
from typing import Any

import torch

from .. import nufft
from ._volume import _translation_per_axis

__all__ = [
    "DEFAULT_EXACT_TOLERANCE",
    "plan_bytes",
    "rotate_volume_exact",
    "wrapped_fraction",
]

#: Requested relative accuracy of the NUFFTs. On a 256^3 template 1e-4 is
#: about 8x more accurate than 1e-3 for about 16% more time.
DEFAULT_EXACT_TOLERANCE = 1e-4

#: Share of the device's memory the points of one GPU NUFFT call may take;
#: poses are concatenated up to it. Every call repeats the work on the
#: oversampled grid (its FFT, deconvolution and zeroing) whatever its number
#: of points, about 47 ms forward and backward at a 300^3 box on an L40, so
#: fewer calls are faster: 24.5 ms a pose at 5 poses a call, 19.9 at 8 and
#: 15.7 at 32 (32 poses). An eighth of an 80 GB H100 holds 26 poses of a
#: 300^3 grid, 8 of 432^3 and 5 of 512^3; of a 48 GB L40, 14, 5 and 3. The
#: CPU always takes one pose per call (see `rotate_volume_exact`).
_CALL_MEMORY_FRACTION = 1 / 8

#: Peak device memory per point of a GPU call, forward and backward: the
#: points, cuFINUFFT's sort indices, the samples and the shift phase
#: (measured 55 bytes on an L40 at 256^3 and 300^3).
_BYTES_PER_POINT = 56

#: Further device memory per point when the backward also differentiates the
#: pose: four complex samples (the transform and its gradient) and the
#: products formed one component at a time from them (measured 48 bytes in
#: PyTorch's allocator at 256^3, on top of the 59 the backward holds).
_POSE_BYTES_PER_POINT = 64

#: Share of a GPU's memory one plan's oversampled grid may take at the
#: upsampling factor 2; larger grids fall back to `nufft`'s 1.25, which holds a
#: quarter of the memory. Factor 2 at a 432^3 box on an L40 takes 61 ms a pose
#: forward and backward where 1.25 takes 69 (8 poses a call), but at 512^3 it
#: is the slower (129 against 122 ms, 4 poses), and its two plans hold 10.3 and
#: 17.2 GB there. A sixteenth of an 80 GB H100 takes factor 2 to about a 436^3 box;
#: of a 48 GB L40, to about 359^3.
_UPSAMPFAC_2_FRACTION = 1 / 16

#: The same budget on the CPU, in bytes of host memory.
_UPSAMPFAC_2_BYTES = 2 * 2**30

#: cuFINUFFT spreading methods measured fastest for these transforms (H100,
#: 256^3, 16 poses): output-driven for the type 2, block-gather for the type 1.
_GPU_METHOD = {2: 1, 1: 3}

#: `wrapped_fraction` above which `rotate_volume_exact` warns. Measured on
#: apoferritin (22FX, 0.78 A) in shrinking boxes against the rotated atoms
#: rendered per pose: exact rotation stops beating trilinear interpolation
#: between boxes 166 and 168, where the bound is 1.7e-2 and 8.7e-3.
_WRAP_WARNING_FRACTION = 1e-2

#: Radial histogram resolution behind `wrapped_fraction`.
_RADIAL_BINS_PER_VOXEL = 4


class _Grid:
    """The output half-spectrum's frequencies inside the Nyquist sphere, for one shape."""

    def __init__(
        self, shape: tuple[int, int, int], device: torch.device, double: bool
    ) -> None:
        nz, ny, nx = shape
        real = torch.float64 if double else torch.float32
        f64 = dict(device=device, dtype=torch.float64)
        fz, fy, fx = torch.meshgrid(
            torch.fft.fftfreq(nz, **f64),
            torch.fft.fftfreq(ny, **f64),
            torch.fft.rfftfreq(nx, **f64),
            indexing="ij",
        )
        p = torch.stack([fx, fy, fz], -1).reshape(-1, 3)  # (x, y, z), cycles/voxel
        inside = p.norm(dim=-1) < 0.5
        self.idx = torch.nonzero(inside).flatten()
        p = p[inside]
        self.qT = p.T.to(real).contiguous()
        self.M = len(self.idx)
        # The DFT's origin is voxel 0, the NUFFT's the centre voxel c = n // 2:
        # exp(-2 pi i p . c), from integer indices so it is exact at any size.
        n = torch.tensor([nx, ny, nz], device=device)
        c = n // 2
        k = torch.round(p * n).long()
        turns = ((k * c) % n).double() / n
        cplx = torch.complex128 if double else torch.complex64
        self.origin = torch.exp(-2j * math.pi * turns.sum(-1)).to(cplx)
        self.real, self.complex = real, cplx
        self.upsampfac = _plan_settings(shape, double, device)

    # Both geometry helpers loop over poses with fixed-shape products: a
    # batched matmul or einsum picks its blocking from the batch size, which
    # moves a pose's points in the last bit with how many poses share the
    # call, and a seeded particle stack must not depend on its batching.

    def points(self, R: torch.Tensor) -> torch.Tensor:
        """The NUFFT points ``2 pi R p`` of poses R, concatenated, as (z, y, x) rows."""
        R = R.to(self.real) * (2 * math.pi)
        rq = torch.cat([R[i].flip(0) @ self.qT for i in range(len(R))], dim=1)
        return rq.contiguous()

    def phase(self, R: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """``exp(2 pi i (R p) . tau)`` times the origin phase, (n, M)."""
        R, tau = R.to(self.real), tau.to(self.real)
        arg = torch.stack(
            [((R[i].T @ tau[i]) * (2 * math.pi)) @ self.qT for i in range(len(R))]
        )
        return self.origin * torch.polar(torch.ones_like(arg), arg)


@functools.lru_cache(maxsize=4)
def _grid(shape: tuple[int, int, int], device: torch.device, double: bool) -> _Grid:
    return _Grid(shape, device, double)


def _plan_settings(
    shape: tuple[int, int, int], double: bool, device: torch.device
) -> float:
    """The upsampling factor: 2 while its grid fits the device's budget."""
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        budget = total * _UPSAMPFAC_2_FRACTION
    else:
        budget = _UPSAMPFAC_2_BYTES
    if nufft._plan_bytes(shape, double, 2.0) <= budget:
        return 2.0
    return nufft._UPSAMPFAC


def plan_bytes(
    shape: tuple[int, int, int], device: str | torch.device | None = None
) -> int:
    """
    Device memory of one single-precision plan `rotate_volume_exact` holds.

    The plan's oversampled grid lives outside torch's allocator; a forward
    holds one (type 2), a backward a second (type 1).

    Parameters
    ----------
    shape : tuple of int
        ``(Z, Y, X)``.
    device : str or torch.device, optional
        The device the plan is made on, whose memory sets its upsampling
        factor. Default: the current CUDA device when there is one, else the
        CPU.

    Returns
    -------
    int
        Bytes.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    return nufft._plan_bytes(shape, False, _plan_settings(shape, False, device))


@functools.lru_cache(maxsize=None)
def _points_per_call(device: torch.device) -> int:
    """Points one NUFFT call on `device` may take (see `_CALL_MEMORY_FRACTION`)."""
    total = torch.cuda.get_device_properties(device).total_memory
    return int(total * _CALL_MEMORY_FRACTION / _BYTES_PER_POINT)


def _chunks(
    n: int, M: int, device: torch.device, bytes_per_point: int = _BYTES_PER_POINT
) -> list[tuple[int, int]]:
    # FINUFFT's CPU interpolation changes a point's value in the last bit with
    # how many points share the call; one pose per call keeps a pose a
    # function of its own points. cuFINUFFT's type 2 is bit-identical either
    # way; its type 1 spreads with atomic adds, so the backward varies in the
    # last bits from run to run at any chunking.
    per = 1
    if device.type == "cuda":
        # A backward that also differentiates the pose holds more per point
        # (`_POSE_BYTES_PER_POINT`), so the same budget takes fewer points.
        points = _points_per_call(device) * _BYTES_PER_POINT // bytes_per_point
        per = max(1, points // max(M, 1))
    return [(s, min(s + per, n)) for s in range(0, n, per)]


class _ExactRotate(torch.autograd.Function):
    """V (Z, Y, X) real, R (B, 3, 3), tau (B, 3) voxels -> half spectra (B, Z, Y, X//2+1)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, V: torch.Tensor, R: torch.Tensor, tau: torch.Tensor, eps: float
    ) -> torch.Tensor:
        shape = tuple(V.shape)
        double = V.dtype == torch.float64
        g = _grid(shape, V.device, double)  # type: ignore[arg-type]
        # The volume is needed again only to differentiate the pose.
        pose = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        ctx.save_for_backward(R, tau, V.detach() if pose else None)
        ctx.shape, ctx.eps, ctx.double = shape, eps, double
        nz, ny, nx = shape
        values = V.detach().to(g.complex).contiguous()
        spec = torch.zeros(
            len(R), nz * ny * (nx // 2 + 1), dtype=g.complex, device=V.device
        )
        for s, e in _chunks(len(R), g.M, V.device):
            vals = nufft._execute(
                2, values, g.points(R[s:e]), shape, eps, g.upsampfac, _GPU_METHOD[2]
            )
            spec[s:e, g.idx] = vals.reshape(e - s, g.M) * g.phase(R[s:e], tau[s:e])
        return spec.reshape(len(R), nz, ny, nx // 2 + 1)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        # Autograd runs a backward under the forward's autocast state; the
        # NUFFT needs single- or double-precision points whatever that is.
        with torch.autocast(grad.device.type, enabled=False):
            return _ExactRotate._backward(ctx, grad)

    @staticmethod
    def _backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:
        R, tau, V = ctx.saved_tensors
        shape, double = ctx.shape, ctx.double
        g = _grid(shape, grad.device, double)
        eps = (
            ctx.eps if nufft._ADJOINT_EPS is None else max(ctx.eps, nufft._ADJOINT_EPS)
        )
        grad = grad.reshape(len(R), -1).to(g.complex)
        want_volume = ctx.needs_input_grad[0]
        pose = V is not None
        out = (
            torch.zeros(shape, dtype=g.real, device=grad.device)
            if want_volume
            else None
        )
        grad_R = torch.zeros_like(R) if pose else None
        grad_tau = torch.zeros_like(tau) if pose else None
        stack = _with_coordinate_weights(V, g.complex) if pose else None
        per_point = _BYTES_PER_POINT + (_POSE_BYTES_PER_POINT if pose else 0)
        for s, e in _chunks(len(R), g.M, grad.device, per_point):
            c = grad[s:e][:, g.idx] * g.phase(R[s:e], tau[s:e]).conj()
            pts = g.points(R[s:e])
            if out is not None:
                out += nufft._execute(
                    1,
                    c.reshape(-1).contiguous(),
                    pts,
                    shape,
                    eps,
                    g.upsampfac,
                    _GPU_METHOD[1],
                ).real
            if pose:
                assert grad_R is not None and grad_tau is not None
                assert stack is not None
                grad_R[s:e], grad_tau[s:e] = _pose_gradient(
                    c, stack, pts, R[s:e], tau[s:e], g, shape, eps
                )
        # rotate_volume_exact hands V over already in its working precision.
        return out, grad_R, grad_tau, None


def _with_coordinate_weights(V: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``V`` and ``-2 pi i (x_j - c_j) V`` for j = x, y, z, as one (4, Z, Y, X) stack.

    Their transforms at a point ``w`` are ``V^(w)`` and the three components
    of its gradient, ``c = shape // 2`` being the NUFFT's centre.
    """
    axes = [torch.arange(n, device=V.device, dtype=V.dtype) - n // 2 for n in V.shape]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    stack = torch.stack([V, x * V, y * V, z * V]).to(dtype)
    stack[1:] *= -2j * math.pi
    return stack


def _pose_gradient(
    c: torch.Tensor,
    stack: torch.Tensor,
    pts: torch.Tensor,
    R: torch.Tensor,
    tau: torch.Tensor,
    g: _Grid,
    shape: tuple[int, int, int],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The loss's gradient to `R` and `tau` for a chunk of poses.

    Parameters
    ----------
    c : torch.Tensor
        ``(k, M)``: the spectrum's incoming gradient times the conjugate
        phase, as the volume's backward forms it.
    stack : torch.Tensor
        `_with_coordinate_weights` of the volume.
    pts, R, tau
        The chunk's NUFFT points, rotations ``(k, 3, 3)`` and shifts
        ``(k, 3)`` in voxels.
    g : _Grid
        The frequencies ``p`` (cycles per voxel, (x, y, z)).
    shape : tuple of int
        The volume's ``(Z, Y, X)``.
    eps : float
        NUFFT accuracy, the backward's.

    Returns
    -------
    tuple of torch.Tensor
        ``(k, 3, 3)`` and ``(k, 3)``.

    Notes
    -----
    With ``S = V^(w) phase``, ``w = R p`` and the upstream gradient ``G``,
    PyTorch's convention gives ``dL/dx = Re(conj(G) dS/dx)`` for a real
    ``x``, and ``conj(G) phase = conj(c)``. So ``dL/dw_j = Re(conj(c)
    (dV^/dw_j + 2 pi i tau_j V^))``, ``dL/dR_ij = sum_p dL/dw_i p_j`` and
    ``dL/dtau_j = Re(conj(c) 2 pi i w_j V^)`` summed over ``p``.
    """
    k = len(R)
    samples = nufft._execute(2, stack, pts, shape, eps, g.upsampfac, _GPU_METHOD[2])
    samples = samples.reshape(4, k, g.M)
    value, slope = samples[0], samples[1:]  # V^(w); dV^/dw (x, y, z)
    # conj(c) 2 pi i V^, shared by both gradients; one component at a time
    # below, so no (k, 3, M) complex temporary is formed.
    cv = c.conj() * value * (2j * math.pi)
    cc = c.conj()
    grad_R = torch.empty(k, 3, 3, dtype=g.real, device=c.device)
    grad_tau = torch.empty(k, 3, dtype=g.real, device=c.device)
    Rr, t = R.to(g.real), tau.to(g.real)
    for j in range(3):
        d_w = (cc * slope[j]).real + (cv * t[:, j, None]).real  # dL/dw_j, (k, M)
        grad_R[:, j] = d_w @ g.qT.T
        w_j = torch.stack([Rr[i, j] @ g.qT for i in range(k)])  # (k, M)
        grad_tau[:, j] = (cv * w_j).real.sum(-1)
    return grad_R.to(R.dtype), grad_tau.to(tau.dtype)


class _ContiguousGrad(torch.autograd.Function):
    """Identity whose backward hands on a contiguous gradient.

    ``U.sum().backward()`` gives an expanded, stride-0 gradient, which MKL's
    ``irfftn`` backward on the CPU rejects.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return grad.contiguous()


# The last template's histogram, held against a weak reference to the tensor
# itself: a key on its address would hand a freed template's histogram to a
# new tensor that the allocator puts at the same address.
# [weakref to the volume, its _version, histogram]
_RADIAL: list[Any] = [None, -1, None]


def _radial_histogram(V: torch.Tensor) -> torch.Tensor:
    """Absolute density binned by distance from the centre voxel, on the host."""
    ref, version, hist = _RADIAL
    if ref is None or ref() is not V or version != V._version:
        axes = [
            torch.arange(n, device=V.device, dtype=torch.float32) - n // 2
            for n in V.shape
        ]
        r = torch.sqrt(
            axes[0][:, None, None] ** 2
            + axes[1][None, :, None] ** 2
            + axes[2][None, None, :] ** 2
        )
        bins = (r * _RADIAL_BINS_PER_VOXEL).to(torch.int32).reshape(-1)
        weights = V.detach().abs().reshape(-1).float()
        hist = torch.bincount(bins, weights=weights).cpu().double()
        _RADIAL[:] = [weakref.ref(V), V._version, hist]
    return hist


def wrapped_fraction(V: torch.Tensor, shift: float = 0.0) -> float:
    """
    Upper bound on the fraction of a volume an exact rotation can wrap.

    Density within ``min(shape) / 2 - shift`` voxels of the centre voxel
    ``shape // 2`` stays inside the box under every rotation followed by a
    shift of at most `shift` voxels along each axis. What lies beyond may
    cross a face, and the periodic transform brings it back on the opposite
    side, where trilinear interpolation would have clipped it.

    Parameters
    ----------
    V : torch.Tensor
        ``(Z, Y, X)`` density.
    shift : float, optional
        Largest translation along any one axis, in voxels. Default 0.

    Returns
    -------
    float
        Integrated absolute density beyond that radius, as a fraction of the
        whole (0 for an empty volume).
    """
    hist = _radial_histogram(V)
    total = float(hist.sum())
    if total == 0.0:
        return 0.0
    radius = min(V.shape) / 2 - shift
    first = max(0, math.floor(radius * _RADIAL_BINS_PER_VOXEL))
    return float(hist[first:].sum()) / total


def rotate_volume_exact(
    V: torch.Tensor,
    theta: torch.Tensor,
    *,
    eps: float = DEFAULT_EXACT_TOLERANCE,
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Rotate and shift a volume without interpolating it.

    Takes the affine matrices `rotate_volume` and `VolumeRotator` take
    (``theta = [R | t]``, origin ``"relion"``: the centre voxel ``n // 2``)
    and returns the same posed volumes, ``V`` sampled at ``R (x - c) + c +
    t``, but computed through the volume's transform: one type-2 NUFFT at the
    rotated frequencies of the output's half spectrum inside the Nyquist
    sphere, a phase for the shift and an inverse real FFT. Nothing is
    interpolated, so every band keeps its amplitude, where trilinear
    sampling keeps 0.45 of it near Nyquist. Differentiable in `V` (the
    backward is the adjoint type-1 NUFFT) and in `theta`, through one more
    type-2 NUFFT of `V` and its coordinate-weighted copies, computed only
    when `theta` requires a gradient.

    Measured on apoferritin (22FX, 256^3 at 0.78 A) and a ribosome (6BDF,
    256^3 at 1 A) against the band-limited atomic potential with the
    renderer's voxel aperture, it keeps 1.00 of the amplitude in every band,
    and its residual equals that of rendering the rotated atoms afresh.

    Parameters
    ----------
    V : torch.Tensor
        Real volume ``(Z, Y, X)``, any shape. float32 runs in single
        precision under any autocast state, float64 in double; cuFINUFFT on
        CUDA, FINUFFT on the CPU.
    theta : torch.Tensor
        ``(B, 3, 4)`` affine matrices, as `build_affine_matrix` makes them.
    eps : float, optional
        Requested relative accuracy of the NUFFTs. Default
        `DEFAULT_EXACT_TOLERANCE` (1e-4).
    align_corners : bool, optional
        The ``grid_sample`` convention `theta`'s translation was normalised
        under. Default False.

    Returns
    -------
    torch.Tensor
        ``(B, Z, Y, X)``, float32 (float64 for a float64 `V`). On the CPU each
        pose is computed on its own, so a pose's result does not depend on
        how many share the call.

    Warns
    -----
    UserWarning
        When :func:`wrapped_fraction` of `V`, for the largest per-axis shift
        requested, exceeds 1%: there the periodic transform wraps density
        to the opposite face. Below it, exact rotation was measured closer
        to the rotated atoms than trilinear interpolation.

    Notes
    -----
    The result is the band-limited (trigonometric) interpolant of `V`
    rotated, not the rotation of whatever continuous object `V` samples: a
    potential rendered from sharp atoms has folded content from beyond
    Nyquist into its voxels, and that aliasing turns with the volume.
    """
    if V.ndim != 3:
        raise ValueError(f"expected a (Z, Y, X) volume, got shape {tuple(V.shape)}")
    if V.is_complex():
        raise ValueError("rotate_volume_exact rotates a real volume")
    if theta.ndim != 3 or theta.shape[1:] != (3, 4):
        raise ValueError(
            f"expected (B, 3, 4) affine matrices, got {tuple(theta.shape)}"
        )
    nz, ny, nx = V.shape
    with torch.autocast(V.device.type, enabled=False):
        work = torch.float64 if V.dtype == torch.float64 else torch.float32
        V = V.to(work)
        theta = theta.to(device=V.device, dtype=work)
        R = theta[:, :, :3].contiguous()
        # x-normalised units -> each axis's own normalised scale -> voxels.
        t = _translation_per_axis(theta[:, :, 3], nz, ny, nx, align_corners)
        half = torch.tensor(
            [(n - 1) / 2 if align_corners else n / 2 for n in (nx, ny, nz)],
            device=V.device,
            dtype=work,
        )
        tau = t * half
        if len(tau):
            # The lab-frame displacement of the content is R^T tau.
            shift = float(torch.einsum("bji,bj->bi", R, tau).detach().abs().max())
            fraction = wrapped_fraction(V, shift)
            if fraction > _WRAP_WARNING_FRACTION:
                warnings.warn(
                    f"{fraction:.2%} of the {tuple(V.shape)} volume's density lies "
                    "where a rotation and a shift of up to "
                    f"{shift:.1f} voxels can carry it out of the box, and "
                    "rotate_mode='exact' wraps it to the opposite face. Use a "
                    "larger box, or rotate_mode='real', which clips it instead.",
                    stacklevel=4,
                )
        spec = _ExactRotate.apply(V, R, tau, eps)
        if V.is_cuda:
            out = torch.fft.irfftn(spec, s=(nz, ny, nx), dim=(-3, -2, -1))
        else:
            # MKL's batched inverse is not batch-invariant at every size (32^3
            # and 64^3 differ in the last bit from one transform alone).
            out = torch.stack(
                [torch.fft.irfftn(h, s=(nz, ny, nx), dim=(-3, -2, -1)) for h in spec]
            )
        return _ContiguousGrad.apply(out)
