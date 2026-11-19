"""A band-limited copy of a map whose transform can be sampled anywhere."""

from __future__ import annotations

import math
from typing import Any

import torch

from ..nufft import GridTransform

_TWO_PI = 2.0 * math.pi
#: Width of the cosine edge of the band limit, in Fourier voxels. A sharp edge
#: rings across the small box and biases the centre: 2 voxels cost 0.03 voxel.
_EDGE = 8.0
#: Lowest shell scored, in Fourier voxels. Below it there are too few
#: independent points for a shell statistic.
_MIN_RADIUS = 1.5
#: Largest box of a band-limited copy unless a wider band is asked for. Its
#: band of 71 Fourier voxels holds 750,000 independent frequencies, all of
#: which a refinement scores. On noisy maps a wider band scored at as many
#: frequencies gave no better pose, and one scored at all of its own took
#: three times as long for the same pose error.
_MAX_BOX = 160


class _Sample(torch.autograd.Function):
    """A band's transform at arbitrary frequencies, differentiable in them."""

    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, band: Band) -> torch.Tensor:
        ctx.band = band
        ctx.save_for_backward(q)
        return torch.view_as_real(band.transform(q))

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:
        # d/dq sum_n v[n] exp(-2 pi i q.n) is the transform of -2 pi i n v[n].
        (q,) = ctx.saved_tensors
        d = torch.stack([m(q) for m in ctx.band.moments()], -1) * (-1j * _TWO_PI)
        return grad[:, :1] * d.real + grad[:, 1:] * d.imag, None


class Band:
    """
    A map inside a Fourier sphere, resampled on the smallest box that holds it.

    A search at radius ``r`` needs only the frequencies below ``r``, which fit
    a box of ``2 (r + edge + 1)`` voxels whatever the size of the map. Every
    transform sample then costs the same for a 96 and a 512 voxel map.

    Frequencies are in cycles per box of the original map (Fourier voxels),
    ordered (x, y, z), and the transform is about the original map's central
    voxel ``n // 2``.

    Parameters
    ----------
    spectrum : torch.Tensor
        ``torch.fft.rfftn`` of the cubic map, shape (N, N, N // 2 + 1).
    n_full : int
        Box size ``N`` of the map.
    radius : float
        Band limit in Fourier voxels. Lowered to what the map's Nyquist
        frequency allows.
    noise_power : float, optional
        Power of the white-noise floor per Fourier coefficient
        (`noise_power`'s output). Sets `signal`. Default 0.

    Attributes
    ----------
    radius : float
        Band limit in force.
    n : int
        Box size of the band-limited copy.
    volume : torch.Tensor
        The band-limited copy, float64, shape (n, n, n).
    lattice : torch.Tensor
        Lattice points of one half of the band, (M, 3), so that no point's
        Friedel mate is present.
    shell_power, shell_std, shell_count : torch.Tensor
        Mean and standard deviation of the power in each integer shell, and
        the number of lattice points it holds.
    signal : torch.Tensor
        Fraction of each shell's power above the noise floor, in [0, 1].
    """

    def __init__(
        self,
        spectrum: torch.Tensor,
        n_full: int,
        radius: float,
        noise_power: float = 0.0,
    ) -> None:
        dev = spectrum.device
        half = n_full // 2
        m = min(
            math.ceil(radius + _EDGE) + 1, (n_full - 1) // 2 if n_full % 2 else half
        )
        self.radius = min(radius, m - 1.0)
        self.n = n = 2 * m
        self.n_full = n_full
        k = torch.arange(-m, m, device=dev)
        kx = torch.arange(0, m + 1, device=dev)
        wrapped = k % n_full
        sub = spectrum[
            wrapped[:, None, None], wrapped[None, :, None], kx[None, None, :]
        ]
        kk, kkx = k.double(), kx.double()
        rad = (
            kk[:, None, None] ** 2 + kk[None, :, None] ** 2 + kkx[None, None, :] ** 2
        ).sqrt()
        window = 0.5 * (
            1 + torch.cos(math.pi * ((rad - self.radius) / _EDGE).clamp(0, 1))
        )
        sub = sub.to(torch.complex128) * window
        if n_full % 2:
            # Centre voxel n_full // 2 of the map onto centre voxel m of the copy.
            ph = half / n_full - 0.5
            sub = sub * torch.exp(
                1j
                * _TWO_PI
                * ph
                * (kk[:, None, None] + kk[None, :, None] + kkx[None, None, :])
            )
        small = torch.zeros(n, n, m + 1, dtype=torch.complex128, device=dev)
        small[(k % n)[:, None, None], (k % n)[None, :, None], kx[None, None, :]] = sub
        self.volume = torch.fft.irfftn(small, s=(n, n, n))
        self.transform = GridTransform(self.volume)
        self._moments: list[GridTransform] | None = None

        inside = (rad >= _MIN_RADIUS) & (rad <= self.radius)
        zi, yi, xi = torch.nonzero(inside, as_tuple=True)
        keep = (kx[xi] > 0) | (
            (kx[xi] == 0) & ((k[yi] > 0) | ((k[yi] == 0) & (k[zi] > 0)))
        )
        zi, yi, xi = zi[keep], yi[keep], xi[keep]
        self.lattice = torch.stack([kx[xi], k[yi], k[zi]], -1).double()
        power = sub[zi, yi, xi].abs() ** 2
        shell = rad[zi, yi, xi].round().long()
        n_shell = int(shell.max()) + 1
        zero = torch.zeros(n_shell, dtype=torch.float64, device=dev)
        count = zero.clone().index_add_(0, shell, torch.ones_like(power))
        mean = zero.clone().index_add_(0, shell, power) / count.clamp_min(1)
        var = zero.clone().index_add_(
            0, shell, (power - mean[shell]) ** 2
        ) / count.clamp_min(1)
        self.shell_power = mean.clamp_min(1e-300)
        self.shell_std = var.sqrt().clamp_min(1e-300)
        self.shell_count = count
        self.signal = ((mean - noise_power) / self.shell_power).clamp(0, 1)

    def moments(self) -> list[GridTransform]:
        """
        Transforms of the copy weighted by each coordinate.

        Returns
        -------
        list of GridTransform
            The x, y and z first moments, which give the derivative of the
            transform with respect to frequency. Built on first use.
        """
        if self._moments is None:
            n = self.n
            ax = torch.arange(n, dtype=torch.float64, device=self.volume.device)
            ax = ax - n // 2
            self._moments = [
                GridTransform(self.volume * ax[None, None, :]),
                GridTransform(self.volume * ax[None, :, None]),
                GridTransform(self.volume * ax[:, None, None]),
            ]
        return self._moments

    def points(self, n_points: int) -> torch.Tensor:
        """
        A fixed pseudo-random subset of `lattice`.

        Parameters
        ----------
        n_points : int
            Number of points wanted.

        Returns
        -------
        torch.Tensor
            Frequencies (min(n_points, M), 3). The same subset on every call,
            drawn from a private generator, so that a score is a deterministic
            function of the pose and the global random stream is untouched.
        """
        total = self.lattice.shape[0]
        if n_points >= total:
            return self.lattice
        g = torch.Generator().manual_seed(0)
        pick = torch.randperm(total, generator=g)[:n_points].sort().values
        return self.lattice[pick.to(self.lattice.device)]

    def sample(self, freqs: torch.Tensor) -> torch.Tensor:
        """
        Transform at arbitrary frequencies.

        Parameters
        ----------
        freqs : torch.Tensor
            Frequencies (M, 3) in Fourier voxels of the original map, float64.

        Returns
        -------
        torch.Tensor
            Complex values (M,). Differentiable in `freqs`.
        """
        q = freqs / self.n
        if q.requires_grad:
            return torch.view_as_complex(_Sample.apply(q, self))
        return self.transform(q)

    def amplitude_weight(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Whitening and signal weight of each frequency.

        Parameters
        ----------
        pts : torch.Tensor
            Frequencies (P, 3) in Fourier voxels.

        Returns
        -------
        torch.Tensor
            ``sqrt(signal / shell_power)`` of each point's shell, (P,): every
            shell counts in proportion to the signal it holds.
        """
        shell = shells(pts)
        return (self.signal[shell] / self.shell_power[shell]).sqrt()


def widest_band(n_full: int) -> float:
    """
    Highest band limit a `Band` is built for unless one is asked for.

    Parameters
    ----------
    n_full : int
        Box size of the map.

    Returns
    -------
    float
        The band limit, in Fourier voxels, whose copy with its cosine edge
        fills a box of `_MAX_BOX` voxels, or the highest the map's Nyquist
        frequency allows if that is lower.
    """
    nyquist = (n_full - 1) // 2 if n_full % 2 else n_full // 2
    return min(_MAX_BOX // 2 - _EDGE, nyquist) - 1.0


def shells(pts: torch.Tensor) -> torch.Tensor:
    """
    Integer shell of each frequency.

    Parameters
    ----------
    pts : torch.Tensor
        Frequencies (..., 3) in Fourier voxels.

    Returns
    -------
    torch.Tensor
        Rounded radii, (...), int64.
    """
    return pts.norm(dim=-1).round().long()


def spectrum_of(volume: torch.Tensor) -> torch.Tensor:
    """
    Transform of a cubic map with its background level removed.

    A `Band` is sampled between lattice points, where a constant background
    is not confined to the zero frequency: its transform there is a Dirichlet
    kernel, which adds a pose-independent term to every sample.

    Parameters
    ----------
    volume : torch.Tensor
        Real cubic map (N, N, N).

    Returns
    -------
    torch.Tensor
        ``torch.fft.rfftn`` of the map, float32 precision, after subtracting
        the mean of the voxels outside the inscribed sphere.
    """
    n = volume.shape[-1]
    v = volume.detach().float()
    ax = torch.arange(n, device=v.device) - n // 2
    r2 = ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2
    return torch.fft.rfftn(v - v[r2 > (n // 2) ** 2].mean())


def noise_power(spectrum: torch.Tensor, n_full: int, n_samples: int = 100_000) -> float:
    """
    White-noise floor of a map, from its outermost shells.

    Parameters
    ----------
    spectrum : torch.Tensor
        ``torch.fft.rfftn`` of the cubic map.
    n_full : int
        Box size of the map.
    n_samples : int, optional
        Number of lattice points sampled. Default 100,000.

    Returns
    -------
    float
        Median power between 0.9 and 1.0 of the Nyquist frequency, converted
        to the mean of the exponential distribution a noise coefficient's
        power follows. The median ignores the few coefficients that still
        carry signal there.
    """
    g = torch.Generator().manual_seed(0)
    d = torch.randn(n_samples, 3, generator=g, dtype=torch.float64)
    d = d / d.norm(dim=-1, keepdim=True)
    r = 0.9 + 0.1 * torch.rand(n_samples, 1, generator=g, dtype=torch.float64)
    k = (d * r * ((n_full - 1) // 2)).round().long().to(spectrum.device)
    k = torch.where(k[:, :1] < 0, -k, k)
    vals = spectrum[k[:, 2] % n_full, k[:, 1] % n_full, k[:, 0]]
    return float((vals.abs() ** 2).double().median() / math.log(2.0))
