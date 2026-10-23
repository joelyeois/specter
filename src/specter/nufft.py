"""
Discrete-space Fourier transforms of a voxel grid at arbitrary frequencies.

A type-2 non-uniform FFT evaluates a grid's transform at any set of
frequencies, not only at the DFT's, to a requested accuracy and without
interpolating the grid. That is what makes rotations in Fourier space exact:
the transform of a rotated volume is the volume's own transform sampled at
rotated frequencies, which `GridTransform` evaluates directly. CUDA tensors go
through cuFINUFFT (Shih et al. 2021), everything else through FINUFFT (Barnett
et al. 2019).

References
----------
.. [1] Barnett, A. H., Magland, J., & af Klinteberg, L. (2019). A parallel
   nonuniform fast Fourier transform library based on an "exponential of
   semicircle" kernel. *SIAM Journal on Scientific Computing*, 41(5),
   C479-C504. https://doi.org/10.1137/18M120885X
.. [2] Shih, Y.-h., Wright, G., Anden, J., Blaschke, J., & Barnett, A. H. (2021).
   cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs.
   *2021 IEEE International Parallel and Distributed Processing Symposium
   Workshops (IPDPSW)*, 688-697. https://doi.org/10.1109/IPDPSW52791.2021.00105
"""

from __future__ import annotations

import contextlib
import math
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

__all__ = ["GridTransform", "adjoint_tolerance", "clear_plan_cache", "sample_transform"]

#: Oversampling of FINUFFT's internal grid. 1.25 holds a 560^3 single-precision
#: plan in 2.7 GB where the library default, 2.0, needs 11 GB, at a cost in
#: speed only: the requested accuracy is kept either way.
_UPSAMPFAC = 1.25

# Plans, cached because building one costs 70-230 ms at 512^3, longer than the
# transform itself, and holds the oversampled grid: one per (type, shape,
# precision, device), shared by `GridTransform` and `sample_transform`, with
# only the points changing between calls.
_PLANS: dict[tuple[Any, ...], Any] = {}

#: Accuracy `sample_transform`'s backward requests instead of its forward's,
#: while `adjoint_tolerance` is active; None otherwise.
_ADJOINT_EPS: float | None = None


class GridTransform:
    """
    The transform of one 3D grid, evaluated at arbitrary frequencies.

    ``GridTransform(volume)(q)`` returns
    ``sum_n volume[n] * exp(-2j * pi * q . (n - c))`` with ``c = shape // 2``
    on every axis, the volume's discrete-space Fourier transform about its
    central voxel. At the DFT frequencies that differs from
    ``torch.fft.fftn(volume)`` only by the phase ``exp(+2j * pi * q . c)``.

    The plan, which holds FINUFFT's oversampled copy of the grid, is built once
    per shape, precision and device and reused; only the frequencies change.
    `clear_plan_cache` releases it.

    Parameters
    ----------
    volume : torch.Tensor
        Real or complex grid of shape (Z, Y, X). A CUDA tensor is transformed
        on its device with cuFINUFFT; any other with FINUFFT on the CPU. When
        cuFINUFFT cannot load, a CUDA tensor is transformed on the CPU with a
        warning, and results are still returned on its device. The plan is
        cached and shared with `sample_transform`.
        Single precision (float32/complex64) or double (float64/complex128).
    eps : float, optional
        Requested relative accuracy. Default 1e-6. Measured against a direct
        sum, single precision reaches 3e-5 (CPU) to 1e-4 (GPU) of the largest
        value at worst, and double precision 1e-6.

    Notes
    -----
    Frequencies are given in cycles per voxel, ordered (x, y, z) to match the
    rotation matrices `specter.symmetries` returns; the axes of ``volume`` are
    (z, y, x).
    """

    def __init__(self, volume: torch.Tensor, eps: float = 1e-6) -> None:
        if volume.ndim != 3:
            raise ValueError(
                f"expected a (Z, Y, X) grid, got shape {tuple(volume.shape)}"
            )
        double = volume.dtype in (torch.float64, torch.complex128)
        self.complex_dtype = torch.complex128 if double else torch.complex64
        self._real = torch.float64 if double else torch.float32
        self.device = volume.device
        self.shape = tuple(volume.shape)
        self._centre = torch.tensor(
            [self.shape[2] // 2, self.shape[1] // 2, self.shape[0] // 2],
            dtype=torch.float64,
            device=self.device,
        )
        values = volume.to(self.complex_dtype).contiguous()
        # The same cached plan `sample_transform` uses: a reconstruction that
        # scatters with rotate_mode="surface" already holds it, and a second
        # oversampled grid allocated beside a training step's cached memory is
        # what ran a 560^3 symmetrization out of device memory.
        self._plan, self._gpu = _cached_plan(2, self.shape, double, self.device, eps)
        self._values: Any = values if self._gpu else values.cpu().numpy()

    def __call__(self, freqs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the transform at the given frequencies.

        Parameters
        ----------
        freqs : torch.Tensor
            Frequencies of shape (M, 3) in cycles per voxel, ordered (x, y, z).

        Returns
        -------
        torch.Tensor
            Complex values of shape (M,), on the volume's device.
        """
        q = freqs.to(device=self.device if self._gpu else "cpu", dtype=self._real)
        # (x, y, z) in cycles per voxel -> radians, one row per grid axis (z, y, x)
        pts = (q.flip(-1) * (2 * math.pi)).T.contiguous()
        if self._gpu:
            self._plan.setpts(pts[0], pts[1], pts[2])
            vals = self._plan.execute(self._values)
        else:
            p = pts.numpy()
            self._plan.setpts(p[0], p[1], p[2])
            vals = torch.from_numpy(np.asarray(self._plan.execute(self._values)))
            vals = vals.to(self.device)
        # FINUFFT's centred modes put index n at n - shape // 2, the centre convention
        # above, for odd and even sizes alike; nothing to re-reference.
        return vals.reshape(-1)

    def dft_phase(self, freqs: torch.Tensor) -> torch.Tensor:
        """
        Phase that turns a centred value from `__call__` into the DFT's.

        Parameters
        ----------
        freqs : torch.Tensor
            Frequencies of shape (M, 3) in cycles per voxel, ordered (x, y, z).

        Returns
        -------
        torch.Tensor
            ``exp(-2j * pi * q . c)`` of shape (M,): multiplying the centred
            transform by it moves the phase origin to voxel 0, as
            ``torch.fft.fftn`` has it.
        """
        q = freqs.to(device=self.device, dtype=torch.float64)
        return torch.exp(-2j * math.pi * (q @ self._centre)).to(self.complex_dtype)


def clear_plan_cache() -> None:
    """Release the NUFFT plans `sample_transform` keeps, and the device memory they hold."""
    _PLANS.clear()


def _plan_bytes(shape: tuple[int, ...], double: bool) -> int:
    """Device memory of one plan's oversampled grid."""
    n = 1
    for m in shape:
        n *= math.ceil(_UPSAMPFAC * m)
    return n * (16 if double else 8)


def _cached_plan(
    kind: int, shape: tuple[int, ...], double: bool, device: torch.device, eps: float
) -> tuple[Any, bool]:
    """
    The cached FINUFFT plan for one transform type, shape, precision and device.

    Returns the plan and whether it runs on the GPU. A CUDA device falls back to
    a CPU plan, with a warning, only when cuFINUFFT cannot be loaded at all; a
    plan that cannot be created on a working GPU is an out-of-memory error.
    """
    gpu = device.type == "cuda"
    if gpu and not _cufinufft_loads():
        gpu = False
    key = (kind, shape, double, gpu, device.index if gpu else None, eps)
    if key in _PLANS:
        return _PLANS[key], gpu
    dtype = "complex128" if double else "complex64"
    kwargs: dict[str, Any] = dict(
        eps=eps,
        isign=-1 if kind == 2 else 1,
        dtype=dtype,
        modeord=0,
        upsampfac=_UPSAMPFAC,
    )
    if not gpu:
        import finufft

        _PLANS[key] = finufft.Plan(
            kind, shape, nthreads=torch.get_num_threads(), **kwargs
        )
        return _PLANS[key], False
    import cufinufft

    for attempt in range(2):
        # cuFINUFFT allocates from the driver, not from torch's caching
        # allocator, so memory torch has cached but no longer uses is invisible
        # to it until released.
        torch.cuda.empty_cache()
        try:
            _PLANS[key] = cufinufft.Plan(
                kind, shape, gpu_device_id=device.index or 0, **kwargs
            )
            return _PLANS[key], True
        except RuntimeError as err:
            if attempt:
                raise torch.OutOfMemoryError(
                    f"cuFINUFFT could not create a type-{kind} plan for a "
                    f"{shape} grid on {device} ({err}); it needs "
                    f"{_plan_bytes(shape, double) / 2**30:.1f} GiB of free device "
                    "memory beyond what torch holds"
                ) from err
            for other in [k for k in _PLANS if k[3] and k[4] == key[4]]:
                del _PLANS[other]
    raise AssertionError("unreachable")


_CUFINUFFT_LOADS: list[bool] = []


def _cufinufft_loads() -> bool:
    """Whether cuFINUFFT imports; warns once when it does not."""
    if not _CUFINUFFT_LOADS:
        try:
            import cufinufft  # noqa: F401

            _CUFINUFFT_LOADS.append(True)
        except (ImportError, OSError) as err:
            warnings.warn(
                f"cuFINUFFT is unavailable ({err}); transforming on the CPU with "
                "FINUFFT, which is slower but gives the same result.",
                stacklevel=3,
            )
            _CUFINUFFT_LOADS.append(False)
    return _CUFINUFFT_LOADS[0]


def _execute(
    kind: int,
    values: torch.Tensor,
    pts: torch.Tensor,
    shape: tuple[int, ...],
    eps: float,
) -> torch.Tensor:
    """
    Type-1 or type-2 transforms through a cached plan, on ``values``' device.

    ``values`` may carry leading batch dimensions (before the grid for a
    type 2, before the M samples for a type 1): every item is transformed at
    the same points, which are set once, since sorting them costs more than
    a transform (2.6 against 1.8 ms for a 432^2 image).
    """
    double = values.dtype == torch.complex128
    plan, gpu = _cached_plan(kind, shape, double, values.device, eps)
    core = len(shape) if kind == 2 else 1
    batch = values.shape[: values.ndim - core]
    flat = values.reshape(-1, *values.shape[values.ndim - core :])
    if gpu:
        plan.setpts(*pts)
        out = torch.stack([plan.execute(v.contiguous()) for v in flat])
    else:
        p = pts.cpu().numpy()
        plan.setpts(*p)
        host = flat.detach().cpu().numpy()
        out = torch.stack(
            [
                torch.from_numpy(np.asarray(plan.execute(np.ascontiguousarray(v))))
                for v in host
            ]
        ).to(values.device)
    return out.reshape(*batch, *out.shape[1:])


class _SampleTransform(torch.autograd.Function):
    """Type-2 forward; its adjoint, a type-1 with the opposite sign, is the backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, volume: torch.Tensor, pts: torch.Tensor, eps: float
    ) -> torch.Tensor:
        cdtype = torch.complex128 if pts.dtype == torch.float64 else torch.complex64
        ctx.save_for_backward(pts)
        # The grid is the trailing len(pts) axes; anything before it is a batch.
        ctx.shape = tuple(volume.shape[volume.ndim - pts.shape[0] :])
        ctx.eps, ctx.real = eps, not volume.is_complex()
        return _execute(2, volume.to(cdtype), pts, ctx.shape, eps)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        (pts,) = ctx.saved_tensors
        cdtype = torch.complex128 if pts.dtype == torch.float64 else torch.complex64
        eps = ctx.eps if _ADJOINT_EPS is None else max(ctx.eps, _ADJOINT_EPS)
        g = _execute(1, grad.to(cdtype), pts, ctx.shape, eps)
        return (g.real if ctx.real else g), None, None


@contextlib.contextmanager
def adjoint_tolerance(eps: float) -> Iterator[None]:
    """
    Loosen the accuracy of `sample_transform`'s backward within a block.

    The backward is a type-1 transform, which spreads every sample onto the
    grid, and its cost grows with the spreading kernel's width, which the
    requested accuracy sets. A gradient that is summed over many images
    tolerates a looser adjoint than a single forward needs: on a 300^3
    Rytov reconstruction, 1e-4 made the forward-plus-backward 24% faster
    and moved the gradient by less than its own float32 rounding between
    two exact methods (3e-4). The forward keeps its own accuracy. Applies
    to backwards run inside the block, whichever forward produced them.

    Parameters
    ----------
    eps : float
        Relative accuracy for the backward. A tighter value than the
        forward's is ignored.

    Yields
    ------
    None
    """
    global _ADJOINT_EPS
    previous = _ADJOINT_EPS
    _ADJOINT_EPS = eps
    try:
        yield
    finally:
        _ADJOINT_EPS = previous


def sample_transform(
    volume: torch.Tensor, freqs: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    A grid's transform at arbitrary frequencies, differentiable in the grid.

    The same quantity as `GridTransform`, ``sum_n volume[n] *
    exp(-2j * pi * q . (n - shape // 2))``, for a caller that evaluates it
    repeatedly and needs gradients: the plans are cached between calls
    (`clear_plan_cache` releases them), and the backward is the transform's
    adjoint, a type-1 NUFFT that spreads the incoming gradient back onto the
    grid. No gradient flows to ``freqs``. Images are transformed the same way
    as volumes, with one axis fewer, and a stack of grids sampled at the same
    frequencies is transformed with the points set once.

    Parameters
    ----------
    volume : torch.Tensor
        Real or complex grid of shape (Z, Y, X), or (Y, X) for an image,
        optionally with one leading batch axis, (B, Z, Y, X) or (B, Y, X);
        the frequencies' length says which. CUDA tensors use cuFINUFFT,
        others FINUFFT. float64/complex128 runs in double precision.
    freqs : torch.Tensor
        Frequencies of shape (M, ndim) in cycles per voxel, ordered (x, y, z),
        or (x, y) for an image.
    eps : float, optional
        Requested relative accuracy. Default 1e-6.

    Returns
    -------
    torch.Tensor
        Complex values of shape (M,), or (B, M) for a batch.
    """
    ndim = freqs.shape[-1]
    if ndim not in (2, 3) or volume.ndim not in (ndim, ndim + 1):
        raise ValueError(
            f"expected a (Z, Y, X) or (Y, X) grid, optionally batched, with "
            f"frequencies of matching dimension; got shapes {tuple(volume.shape)} "
            f"and {tuple(freqs.shape)}"
        )
    double = volume.dtype in (torch.float64, torch.complex128)
    real = torch.float64 if double else torch.float32
    q = freqs.detach().to(device=volume.device, dtype=real)
    pts = (q.flip(-1) * (2 * math.pi)).T.contiguous()
    out = _SampleTransform.apply(volume, pts, eps)
    return out.reshape(*volume.shape[: volume.ndim - ndim], -1)
