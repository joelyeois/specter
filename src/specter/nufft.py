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

import math
import warnings
from typing import Any

import numpy as np
import torch

__all__ = ["GridTransform"]

#: Oversampling of FINUFFT's internal grid. 1.25 holds a 560^3 single-precision
#: plan in 2.7 GB where the library default, 2.0, needs 11 GB, at a cost in
#: speed only: the requested accuracy is kept either way.
_UPSAMPFAC = 1.25


class GridTransform:
    """
    The transform of one 3D grid, evaluated at arbitrary frequencies.

    ``GridTransform(volume)(q)`` returns
    ``sum_n volume[n] * exp(-2j * pi * q . (n - c))`` with ``c = shape // 2``
    on every axis, the volume's discrete-space Fourier transform about its
    central voxel. At the DFT frequencies that differs from
    ``torch.fft.fftn(volume)`` only by the phase ``exp(+2j * pi * q . c)``.

    The plan, which holds FINUFFT's oversampled copy of the grid, is built once
    and reused for every call; only the frequencies change.

    Parameters
    ----------
    volume : torch.Tensor
        Real or complex grid of shape (Z, Y, X). A CUDA tensor is transformed
        on its device with cuFINUFFT; any other with FINUFFT on the CPU. When
        cuFINUFFT cannot load, a CUDA tensor is transformed on the CPU with a
        warning, and results are still returned on its device.
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
        dtype = "complex128" if double else "complex64"
        self._gpu = False
        if self.device.type == "cuda":
            try:
                import cufinufft

                self._plan: Any = cufinufft.Plan(
                    2,
                    self.shape,
                    eps=eps,
                    isign=-1,
                    dtype=dtype,
                    modeord=0,
                    upsampfac=_UPSAMPFAC,
                    gpu_device_id=self.device.index or 0,
                )
                self._values: Any = values
                self._gpu = True
            except (ImportError, OSError, RuntimeError) as err:
                warnings.warn(
                    f"cuFINUFFT is unavailable ({err}); transforming on the CPU with "
                    "FINUFFT, which is slower but gives the same result.",
                    stacklevel=2,
                )
        if not self._gpu:
            import finufft

            self._plan = finufft.Plan(
                2,
                self.shape,
                eps=eps,
                isign=-1,
                dtype=dtype,
                modeord=0,
                upsampfac=_UPSAMPFAC,
                nthreads=torch.get_num_threads(),
            )
            self._values = values.cpu().numpy()

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
