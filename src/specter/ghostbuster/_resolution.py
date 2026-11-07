"""
Half-map resolution for the converged solver: the masked FSC, the SSNR the
Wiener prior is built from (and the unfiltering that undoes it), and the
resolution mask -- CryoSPARC's dynamic-masking recipe as its public guide
describes it, tightened while its noise-substituted FSC (Chen et al. 2013)
agrees (see `auto_mask`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ..arrays import rfft_frequency_squared
from ..fft import fourier_shell_correlation, frequency_shells, fsc_crossing

#: FSC is clipped to this range before it becomes an SSNR, so the prior is
#: finite where the halves agree perfectly and bounded where they do not.
FSC_CLIP = (1e-3, 0.999)


def masked_fsc(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None
) -> np.ndarray:
    """
    Fourier shell correlation of two volumes, after an optional mask.

    Parameters
    ----------
    a, b : torch.Tensor
        Cubic volumes, shape ``(n, n, n)``.
    mask : torch.Tensor, optional
        Multiplied into both volumes first.

    Returns
    -------
    numpy.ndarray
        FSC per shell of one Fourier voxel, ``n // 2 + 1`` values.
    """
    return fourier_shell_correlation(a, b, mask=mask)[1].cpu().numpy()


def ssnr_from_fsc(fsc: np.ndarray, smooth: int = 5) -> np.ndarray:
    """
    The signal-to-noise ratio of one half-map, ``FSC / (1 - FSC)``.

    The FSC is first averaged over `smooth` neighbouring shells and made
    non-increasing with resolution. Where it hovers near zero, shell-to-shell
    noise would otherwise change the prior by an order of magnitude or more
    between neighbouring shells; the exact point-group average mixes
    neighbouring shells slightly, and those jumps then dominate the
    problem's conditioning (on EMPIAR-12391 every step of the top band ran
    at a quarter of its length). They carry no information about the map.

    Parameters
    ----------
    fsc : numpy.ndarray
        Half-map FSC per shell.
    smooth : int
        Width of the running mean, in shells; 1 leaves the FSC as it is.

    Returns
    -------
    numpy.ndarray
        SSNR per shell, from the smoothed FSC clipped to `FSC_CLIP`.
    """
    f = np.asarray(fsc, dtype=np.float64)
    if smooth > 1:
        pad = smooth // 2
        f = np.asarray(
            np.convolve(np.pad(f, pad, mode="edge"), np.ones(smooth) / smooth, "valid"),
            dtype=np.float64,
        )
    f = np.clip(np.minimum.accumulate(f), *FSC_CLIP)
    return f / (1.0 - f)


def unfilter(volume: torch.Tensor, ssnr: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Undo the Wiener filter the prior applies: multiply by ``1 + 1 / SSNR(k)``.

    Parameters
    ----------
    volume : torch.Tensor
        A solution of the regularised problem, shape ``(n, n, n)``.
    ssnr : array_like
        The SSNR per shell the prior was built from.

    Returns
    -------
    torch.Tensor
        The unfiltered map: what the data determine at each frequency, with
        the noise the prior suppressed put back.
    """
    _, shell, _ = frequency_shells(volume.shape[-1], volume.device)
    factor = (
        1.0
        + 1.0
        / torch.as_tensor(np.asarray(ssnr), dtype=volume.dtype, device=volume.device)[
            shell
        ]
    )
    return torch.fft.irfftn(torch.fft.rfftn(volume) * factor, s=volume.shape)


def spherical_mask(n: int, inner: float = 0.85, outer: float = 0.99) -> torch.Tensor:
    """A soft sphere: 1 inside `inner`, cosine to 0 at `outer`, as fractions of half the box."""
    r = torch.arange(n, dtype=torch.float32) - n // 2
    radius = torch.sqrt(
        r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None] ** 2
    ) / (n / 2)
    edge = ((radius - inner) / (outer - inner)).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * edge))


def _phase_randomised(
    volume: torch.Tensor, start_shell: int, generator: torch.Generator
) -> torch.Tensor:
    """`volume` with the phases beyond `start_shell` replaced by random ones."""
    _, shell, _ = frequency_shells(volume.shape[-1], volume.device)
    noise = torch.randn(volume.shape, generator=generator).to(volume.device)
    phase = torch.fft.rfftn(noise)
    phase = phase / phase.abs().clamp(min=1e-30)
    f = torch.fft.rfftn(volume)
    f = torch.where(shell > start_shell, f.abs() * phase, f)
    return torch.fft.irfftn(f, s=volume.shape)


def noise_substituted_fsc(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    start_shell: int,
    seed: int = 0,
) -> np.ndarray:
    """
    The masked FSC corrected for the correlation the mask itself introduces.

    Both half-maps have their phases randomised beyond `start_shell`, so any
    correlation the mask leaves between the randomised halves is the mask's
    own; ``(FSC_t - FSC_n) / (1 - FSC_n)`` removes it (Chen et al. 2013).

    Parameters
    ----------
    a, b : torch.Tensor
        Half-maps.
    mask : torch.Tensor
        The mask under test.
    start_shell : int
        Shells up to this one keep their phases.
    seed : int
        Seeds the random phases.

    Returns
    -------
    numpy.ndarray
        Corrected FSC per shell; equal to the masked FSC up to `start_shell`.
    """
    generator = torch.Generator().manual_seed(seed)
    fsc_t = masked_fsc(a, b, mask)
    fsc_n = masked_fsc(
        _phase_randomised(a, start_shell, generator),
        _phase_randomised(b, start_shell, generator),
        mask,
    )
    corrected = (fsc_t - fsc_n) / np.clip(1.0 - fsc_n, 1e-6, None)
    corrected[: start_shell + 1] = fsc_t[: start_shell + 1]
    return corrected


def _noise_start_shell(fsc_sphere: np.ndarray, n: int) -> int:
    """The shell beyond which phases are randomised: where the FSC inside a
    soft sphere first falls below 0.8 (Chen et al. 2013 randomise from well
    inside the region the halves agree on)."""
    below = np.nonzero(fsc_sphere[1:] < 0.8)[0]
    return int(below[0]) if len(below) else n // 4


def mask_check(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    voxel_size: float,
    max_shell_gap: float = 2.0,
) -> dict[str, Any]:
    """
    Whether `mask` is too tight for two half-maps: the noise-substituted FSC.

    CryoSPARC's "corrected" FSC: both halves have their phases randomised
    beyond the shell where their FSC inside a soft sphere falls below 0.8,
    and what the mask still correlates there is its own doing and is
    removed (Chen et al. 2013). A mask is reported too tight when the
    masked FSC crosses 0.143 more than `max_shell_gap` shells beyond the
    corrected one.

    Parameters
    ----------
    a, b : torch.Tensor
        Half-maps, ``(n, n, n)``.
    mask : torch.Tensor
        The mask, on the same grid.
    voxel_size : float
        Voxel size in Å.
    max_shell_gap : float
        Largest gap, in Fourier shells, between the two crossings.

    Returns
    -------
    dict
        ``masked_resolution`` and ``corrected_resolution`` (Å), ``shell_gap``,
        ``too_tight``, and the ``corrected_fsc`` per shell.
    """
    n = a.shape[-1]
    mask = mask.to(a.device, a.dtype)
    start = _noise_start_shell(masked_fsc(a, b, spherical_mask(n).to(a.device)), n)
    masked = masked_fsc(a, b, mask)
    corrected = noise_substituted_fsc(a, b, mask, start)
    masked_shell, corrected_shell = fsc_crossing(masked), fsc_crossing(corrected)

    def angstrom(shell: float) -> float:
        return round(n * voxel_size / max(shell, 1.0), 4)

    gap = float(masked_shell - corrected_shell)
    return dict(
        masked_resolution=angstrom(masked_shell),
        corrected_resolution=angstrom(corrected_shell),
        shell_gap=round(gap, 2),
        too_tight=bool(gap > max_shell_gap),
        corrected_fsc=corrected,
    )


def resolution_mask(
    distance: np.ndarray, near: float, far: float, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """
    A soft mask from a distance map: 1 within `near`, a cosine edge to 0 at `far`.

    Parameters
    ----------
    distance : numpy.ndarray
        Distance in Å of every voxel from the thresholded region.
    near, far : float
        Distances in Å where the mask leaves 1 and reaches 0.
    device : torch.device or str
        Device of the returned mask.

    Returns
    -------
    torch.Tensor
        Mask in [0, 1].
    """
    edge = np.clip((distance - near) / (far - near), 0.0, 1.0)
    return torch.as_tensor(
        0.5 * (1.0 + np.cos(np.pi * edge)), dtype=torch.float32, device=device
    )


def _distance_from_shared_signal(
    a: torch.Tensor, b: torch.Tensor, voxel_size: float, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Distance in Å from where two half-maps share signal near their resolution.

    Both halves are band-passed to ``2 r`` to ``r`` (``r`` their FSC = 0.143
    resolution inside a soft sphere), multiplied, and the product smoothed
    over ``3 r``: noise, independent between the halves, averages out, and
    what remains is signal both carry at high resolution. Thresholded at
    `threshold` times its maximum. Density that is there at low resolution
    but disordered at high resolution, such as a capsid's packaged genome,
    is left out, so the FSC measured inside the mask is the ordered
    structure's. Returns the distance map and the spherical-mask FSC.
    """
    from scipy.ndimage import distance_transform_edt

    n = a.shape[-1]
    fsc_sphere = masked_fsc(a, b, spherical_mask(n).to(a.device))
    r = n * voxel_size / max(fsc_crossing(fsc_sphere), 1.0)
    kk = torch.sqrt(rfft_frequency_squared(a.shape[-3:], voxel_size, a.device))
    band = ((kk >= 1 / (2 * r)) & (kk < 1 / r)).float()
    fa = torch.fft.irfftn(torch.fft.rfftn(a) * band, s=a.shape)
    fb = torch.fft.irfftn(torch.fft.rfftn(b) * band, s=a.shape)
    smooth = torch.exp(-2 * (math.pi * 3 * r * kk) ** 2)
    shared = torch.fft.irfftn(torch.fft.rfftn(fa * fb) * smooth, s=a.shape)
    inside = (shared > threshold * shared.max()).cpu().numpy()
    return distance_transform_edt(~inside) * voxel_size, fsc_sphere


def auto_mask(
    a: torch.Tensor,
    b: torch.Tensor,
    voxel_size: float,
    threshold: float = 0.5,
    near_multiplier: float = 2.0,
    far_multiplier: float = 5.0,
    step: float = 0.125,
    max_shell_gap: float = 2.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    A mask around the ordered structure in two half-maps, tightened while the FSC allows it.

    The region is where the halves share signal near their resolution (see
    `_distance_from_shared_signal`), thresholded at `threshold` times that
    map's maximum, rather than where they have density: the prior's SSNR is
    then the ordered structure's, which a density mask dilutes with anything
    dense but disordered (on EMPIAR-10083 the packaged genome took the SSNR
    to 0.54 times a protein-shell mask's; with this map it is 1.16, and
    0.89-1.10 times CryoSPARC's own masks on EMPIAR-13202, 10551 and 12391).
    The edge and its tightening follow CryoSPARC's resolution mask (its
    "dynamic masking" guide, v5): the mask is 1 within ``near_multiplier *
    r`` of the region and falls along a cosine to 0 at ``far_multiplier *
    r``. ``r`` starts at the spherically masked
    FSC = 0.143 resolution and drops by `step` Å at a time until ``near``
    reaches ``2 * voxel_size + 0.25`` Å. The mask returned is the tightest
    whose FSC crosses 0.143 within `max_shell_gap` shells of its
    noise-substituted FSC (Chen et al. 2013), that is the tightest that does
    not yet correlate the halves by itself. Every step is tried rather than
    stopping at the first that fails: an FSC that hovers near 0.143 moves its
    crossing by several shells between neighbouring masks, which one failed
    step would mistake for mask-induced correlation.

    Parameters
    ----------
    a, b : torch.Tensor
        Half-maps, ``(n, n, n)``.
    voxel_size : float
        Voxel size in Å.
    threshold : float
        Fraction of the weighted average's maximum that counts as inside.
    near_multiplier, far_multiplier : float
        The mask's flat extent and outer edge, in units of ``r``.
    step : float
        Decrease of ``r`` per tightening step, Å.
    max_shell_gap : float
        Largest allowed gap, in shells, between the masked and the
        noise-substituted FSC's 0.143 crossings.

    Returns
    -------
    mask : torch.Tensor
        Mask in [0, 1].
    info : dict
        ``near`` and ``far`` (Å), the masked and corrected crossings, and the
        number of tightening steps.

    References
    ----------
    .. [1] Chen, S., McMullan, G., Faruqi, A. R., Murshudov, G. N., Short,
       J. M., Scheres, S. H. W., & Henderson, R. (2013). High-resolution
       noise substitution to measure overfitting and validate resolution in
       3D structure determination by single particle electron
       cryomicroscopy. *Ultramicroscopy*, 135, 24–35.
       https://doi.org/10.1016/j.ultramic.2013.06.004
    """
    n = a.shape[-1]
    distance, fsc_sphere = _distance_from_shared_signal(a, b, voxel_size, threshold)
    resolution = n * voxel_size / max(fsc_crossing(fsc_sphere), 1.0)
    start_shell = _noise_start_shell(fsc_sphere, n)
    floor = 2 * voxel_size + 0.25

    best: tuple[torch.Tensor, dict[str, Any]] | None = None
    r = resolution
    steps = 0
    while near_multiplier * r >= floor:
        near, far = near_multiplier * r, far_multiplier * r
        mask = resolution_mask(distance, near, far, a.device)
        masked = fsc_crossing(masked_fsc(a, b, mask))
        corrected = fsc_crossing(noise_substituted_fsc(a, b, mask, start_shell))
        if best is None or masked - corrected <= max_shell_gap:
            best = (
                mask,
                dict(
                    near=round(float(near), 3),
                    far=round(float(far), 3),
                    masked_shell=round(float(masked), 2),
                    corrected_shell=round(float(corrected), 2),
                    steps=steps,
                ),
            )
        r -= step
        steps += 1
    if best is None:
        near, far = near_multiplier * resolution, far_multiplier * resolution
        best = (
            resolution_mask(distance, near, far, a.device),
            dict(near=float(near), far=float(far), steps=0),
        )
    return best
