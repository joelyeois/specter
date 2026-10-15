"""
Partial temporal coherence as an explicit incoherent sum over defocus.

The Cc envelope (:func:`~specter.aberrations.cc_envelope`) multiplies the
WAVE by the Gaussian average of the defocus phase, which is exact for the
part of the image linear in the scattered wave and wrong for the rest: the
intensity of two scattered beams ``k1`` and ``k2`` is damped, over a
Gaussian focal spread, by ``exp(-(pi lambda sigma_f)^2 (k1^2 - k2^2)^2 / 2)``
(the transmission cross-coefficient), while the envelope damps it by
``E(k1) E(k2)``. The two disagree most for pairs at the same ``|k|``, which
interfere into low-frequency intensity and which the envelope suppresses
even though the focal spread cannot. :class:`FocalSpread` forms the image as
the sum it physically is, ``sum_i w_i |psi * exp(i pi lambda k^2 d_i)|^2``,
over a Gauss-Hermite quadrature of the focal spread.

References
----------
.. [1] Ishizuka, K. (1980). Contrast transfer of crystal images in TEM.
   *Ultramicroscopy*, 5(1–3), 55–65.
   https://doi.org/10.1016/0304-3991(80)90011-X
"""

from __future__ import annotations

import functools
import math

import numpy as np
import torch

from ._envelopes import focal_spread_sigma

#: Largest quadrature the integration will build: numpy's Gauss-Hermite rule
#: overflows to NaN weights somewhere between 300 and 400 nodes. Reached only
#: far outside the cryo-EM regime (120 kV at 1 A/px, oversampled 2x, needs 91).
_MAX_NODES = 300


@functools.lru_cache(maxsize=64)
def _gauss_hermite_nodes(a_max: float, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """
    The smallest Gauss-Hermite rule that averages ``exp(i a x)`` over
    ``x ~ N(0, 1)`` to within `tol` for every ``|a| <= a_max``.

    Returns standard-normal abscissae and probability weights (summing to 1).
    The error is checked on a dense grid of ``a`` rather than assumed from
    the rule's polynomial degree, since an oscillatory integrand is what the
    rule handles worst. Memoised: the search is ~20 ms for the 16 nodes of a
    300 kV image, against ~1 ms for building any other stage, and every
    caller only reads the arrays.
    """
    a = np.linspace(0.0, max(a_max, 1e-12), 4096)
    for n in range(1, _MAX_NODES + 1):
        x, w = np.polynomial.hermite.hermgauss(n)
        if not (np.isfinite(x).all() and np.isfinite(w).all()):
            break
        x, w = math.sqrt(2.0) * x, w / math.sqrt(math.pi)
        quad = (w[None, :] * np.exp(1j * a[:, None] * x[None, :])).sum(1)
        if np.abs(quad - np.exp(-0.5 * a**2)).max() < tol:
            return x, w
    raise ValueError(
        f"a focal spread reaching {a_max:.1f} rad of defocus phase needs more "
        f"than {_MAX_NODES} quadrature nodes; the envelope approximation is "
        "the only practical option at this focal spread and sampling"
    )


class FocalSpread:
    """
    The incoherent average of an image over a Gaussian spread of defocus.

    Parameters
    ----------
    sigma_f : float
        Standard deviation of the defocus spread, in Angstrom; see
        :func:`~specter.aberrations.focal_spread_sigma`.
    wavelength : float
        Electron wavelength, in Angstrom.
    tol : float, optional
        Absolute accuracy of the quadrature on the damping factor of every
        frequency pair the grid can hold. Default 1e-6.

    Notes
    -----
    The number of defocus samples is chosen per grid, from the largest phase
    a frequency pair on it can accumulate: the pixel-grid intensity at ``q``
    collects pairs ``k1 - k2 = q``, so ``|k1^2 - k2^2| <= |q| |k1 + k2|``,
    which for a wave on a grid of spacing ``dx`` imaged onto pixels of
    ``s dx`` is at most ``1 / (s dx^2)``, and never more than the grid's
    corner ``1 / (2 dx^2)``. Measured on 6bdf in 2000 A of ice at 1 A/px, oversample 2:
    16 nodes at 300 kV (0.7 eV, Cc 2.7 mm), 90 at 120 kV. Each node costs one
    complex multiply and one inverse FFT of the wave, against two FFTs per
    slice in the multislice that made it.

    Measured effect against the envelope on the same exit wave (300 kV,
    Cc 2.7 mm, 0.7 eV): the envelope route is 0.4% low in mean intensity and
    0.7% low in low-frequency image power at 2000 A of ice, 0.04% and
    under 0.1% in thin ice; at 120 kV the envelope is 5.1% low in mean
    intensity and 4-14% off in image power. dev/tcc_experiment.py.

    References
    ----------
    .. [1] Ishizuka, K. (1980). Contrast transfer of crystal images in TEM.
       *Ultramicroscopy*, 5(1–3), 55–65.
       https://doi.org/10.1016/0304-3991(80)90011-X
    """

    def __init__(self, sigma_f: float, wavelength: float, tol: float = 1e-6):
        if sigma_f < 0:
            raise ValueError(f"sigma_f={sigma_f} must be non-negative")
        self.sigma_f = float(sigma_f)
        self.wavelength = float(wavelength)
        self.tol = tol
        self._cache: dict[tuple, tuple[torch.Tensor, list[float], list[float]]] = {}

    @classmethod
    def from_instrument(
        cls,
        cc: float,
        voltage: float,
        energy_spread: float,
        deltaV_V: float,
        deltaI_I: float,
        wavelength: float,
        tol: float = 1e-6,
    ) -> FocalSpread:
        """Build from the same instrument constants the Cc envelope takes."""
        sigma = focal_spread_sigma(cc, voltage, energy_spread, deltaV_V, deltaI_I)
        return cls(sigma, wavelength, tol=tol)

    def nodes(
        self, dx: float, image_oversample: int = 1
    ) -> tuple[list[float], list[float]]:
        """Defocus offsets (Angstrom) and weights for a wave of spacing `dx`."""
        # |k1^2 - k2^2| over the pairs that reach the pixel-grid image: at most
        # |q| |k1 + k2| <= 1 / (s dx^2) for the pixel band, and never more
        # than the grid's own corner k^2 = 1 / (2 dx^2).
        k2_bound = min(1.0 / (image_oversample * dx**2), 1.0 / (2.0 * dx**2))
        a_max = math.pi * self.wavelength * self.sigma_f * k2_bound
        x, w = _gauss_hermite_nodes(a_max, self.tol)
        return [float(self.sigma_f * xi) for xi in x], [float(wi) for wi in w]

    def _grid(
        self,
        shape: tuple[int, int],
        dx: float,
        image_oversample: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, list[float], list[float]]:
        key = (shape, dx, image_oversample, device, dtype)
        if key not in self._cache:
            ky = torch.fft.fftfreq(shape[0], dx, device=device, dtype=dtype)
            kx = torch.fft.fftfreq(shape[1], dx, device=device, dtype=dtype)
            k2 = ky[:, None] ** 2 + kx[None, :] ** 2
            offsets, weights = self.nodes(dx, image_oversample)
            self._cache[key] = (k2, offsets, weights)
        return self._cache[key]

    def intensity(
        self, wave: torch.Tensor, dx: float, image_oversample: int = 1
    ) -> torch.Tensor:
        """
        ``sum_i w_i |psi * exp(i pi lambda k^2 d_i)|^2``.

        Parameters
        ----------
        wave : torch.Tensor
            Complex ``(..., Y, X)`` wave after the transfer function at the
            nominal defocus.
        dx : float
            The wave's grid spacing, in Angstrom.
        image_oversample : int, optional
            How many times finer than the pixels the wave's grid is; sets the
            quadrature (see Notes). Default 1.

        Returns
        -------
        torch.Tensor
            Real ``(..., Y, X)`` intensity on the wave's grid.
        """
        if self.sigma_f == 0:
            return wave.abs().square()
        k2, offsets, weights = self._grid(
            (wave.shape[-2], wave.shape[-1]),
            dx,
            image_oversample,
            wave.device,
            wave.real.dtype,
        )
        spectrum = torch.fft.fft2(wave)
        out: torch.Tensor | None = None
        for d, w in zip(offsets, weights):
            # Defocus enters the transfer function as exp(+i pi lambda k^2 df)
            # (aberrations._functions.defocus); the Gaussian is symmetric, so
            # the sign only decides which node is which.
            phase = torch.polar(
                torch.ones_like(k2), (math.pi * self.wavelength * d) * k2
            ).to(spectrum.dtype)
            term = torch.fft.ifft2(spectrum * phase).abs().square() * w
            out = term if out is None else out + term
        assert out is not None
        return out
