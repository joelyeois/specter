"""
Converged reconstruction: the map that minimises Ghostbuster's own image
loss plus a Wiener prior, found by preconditioned nonlinear conjugate
gradients.

Adam on minibatches stops on a noise floor set by its learning rate and
batch size, and a map without a prior over-fits every frequency the data do
not determine. The solver here minimises

    1/2 sum_i || y_i - p_i(V) ||^2  +  1/2 sum_k lambda(k) |V(k)|^2

exactly, where ``p_i`` is `Reconstructor`'s prediction (forward model, then
the per-particle scale about the no-specimen level) and the prior is the
Wiener (tau^2) form of RELION (Scheres 2012): ``lambda(k) = W(k) / SSNR(k)``,
``W(k)`` the data term's curvature averaged over a shell and ``SSNR(k)`` the
half-map signal-to-noise ratio, ``FSC / (1 - FSC)``. The minimiser is the
Wiener-filtered map; multiplying it by ``1 + 1 / SSNR(k)`` undoes the
filter, giving the unfiltered half-map that FSC and post-processing expect.

Every iteration sums the gradient over every particle; the step length comes
from a Gauss-Newton estimate along the search direction, refined by a
parabola through the loss on a random subset of particles. The
preconditioner is ``1 / (W(k) + lambda(k))``, radial, so it commutes with
the point-group and band-limit projections.

References
----------
.. [1] Scheres, S. H. W. (2012). RELION: Implementation of a Bayesian
   approach to cryo-EM structure determination. *Journal of Structural
   Biology*, 180(3), 519–530. https://doi.org/10.1016/j.jsb.2012.09.006
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.func import functional_call

from specter import logger

from .. import nufft
from ..symmetries import apply_symmetry
from ._helpers import _apply_kmask_inplace

if TYPE_CHECKING:
    from ._reconstructor import Reconstructor

#: Accuracy of the NUFFT adjoint (the surface sampling's backward) during the
#: gradient pass. See `specter.nufft.adjoint_tolerance`.
ADJOINT_TOLERANCE = 1e-4

#: FSC is clipped to this range before it becomes an SSNR, so the prior is
#: finite where the halves agree perfectly and bounded where they do not.
FSC_CLIP = (1e-3, 0.999)


def frequency_shells(
    n: int, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Radius, shell index and Hermitian weight of every half-spectrum voxel.

    Parameters
    ----------
    n : int
        Box size of a cubic volume.
    device : torch.device or str
        Device of the returned grids.

    Returns
    -------
    radius : torch.Tensor
        ``|k|`` in voxels of the Fourier grid, shape ``(n, n, n // 2 + 1)``.
    shell : torch.Tensor
        ``round(radius)``, clamped to ``n // 2``, as int64.
    weight : torch.Tensor
        2 for the columns the half spectrum stands in for twice, else 1, so
        a weighted sum over it is the sum over the full spectrum.
    """
    fz = torch.fft.fftfreq(n, device=device)
    fx = torch.fft.rfftfreq(n, device=device)
    radius = (
        torch.sqrt(
            fx[None, None] ** 2 + fz[None, :, None] ** 2 + fz[:, None, None] ** 2
        )
        * n
    )
    shell = torch.round(radius).long().clamp(max=n // 2)
    weight = torch.full_like(radius, 2.0)
    weight[..., 0] = 1.0
    if n % 2 == 0:
        weight[..., -1] = 1.0
    return radius, shell, weight


def _shell_sum(
    x: torch.Tensor, shell: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    n_shells = int(shell.max()) + 1
    out = torch.zeros(n_shells, device=x.device, dtype=x.dtype)
    return out.index_add_(0, shell.flatten(), (x * weight).flatten())


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
    if mask is not None:
        a, b = a * mask, b * mask
    _, shell, weight = frequency_shells(a.shape[-1], a.device)
    fa, fb = torch.fft.rfftn(a), torch.fft.rfftn(b)
    num = _shell_sum((fa * fb.conj()).real, shell, weight)
    den = torch.sqrt(
        _shell_sum(fa.abs() ** 2, shell, weight)
        * _shell_sum(fb.abs() ** 2, shell, weight)
    )
    return (num / den.clamp(min=torch.finfo(den.dtype).tiny)).cpu().numpy()


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


def fsc_crossing(fsc: np.ndarray, threshold: float = 0.143) -> float:
    """
    The shell, interpolated, where an FSC first falls below `threshold`.

    Parameters
    ----------
    fsc : numpy.ndarray
        FSC per shell.
    threshold : float
        Crossing level.

    Returns
    -------
    float
        Shell index (Fourier voxels); the last shell when it never falls below.
    """
    below = np.nonzero(fsc[1:] < threshold)[0]
    if len(below) == 0:
        return float(len(fsc) - 1)
    i = int(below[0]) + 1
    return (i - 1) + (fsc[i - 1] - threshold) / (fsc[i - 1] - fsc[i])


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


def _distance_from_density(
    a: torch.Tensor, b: torch.Tensor, voxel_size: float, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Distance in Å from the region where two half-maps' density is high.

    The half-map mean is weighted by the halves' FSC inside a soft sphere (a
    Wiener-like low-pass that keeps what the halves agree on), and
    thresholded at `threshold` times its maximum. Returns the distance map
    and that spherical-mask FSC.
    """
    from scipy.ndimage import distance_transform_edt

    n = a.shape[-1]
    fsc_sphere = masked_fsc(a, b, spherical_mask(n).to(a.device))
    _, shell, _ = frequency_shells(n, a.device)
    weight = torch.as_tensor(
        np.clip(fsc_sphere, 0.0, 1.0), dtype=torch.float32, device=a.device
    )[shell]
    average = torch.fft.irfftn(torch.fft.rfftn(0.5 * (a + b)) * weight, s=a.shape)
    inside = (average > threshold * average.max()).cpu().numpy()
    return distance_transform_edt(~inside) * voxel_size, fsc_sphere


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
    k = torch.fft.fftfreq(n, d=voxel_size, device=a.device)
    kr = torch.fft.rfftfreq(n, d=voxel_size, device=a.device)
    kk = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kr[None, None] ** 2)
    band = ((kk >= 1 / (2 * r)) & (kk < 1 / r)).float()
    fa = torch.fft.irfftn(torch.fft.rfftn(a) * band, s=a.shape)
    fb = torch.fft.irfftn(torch.fft.rfftn(b) * band, s=a.shape)
    smooth = torch.exp(-2 * (math.pi * 3 * r * kk) ** 2)
    shared = torch.fft.irfftn(torch.fft.rfftn(fa * fb) * smooth, s=a.shape)
    inside = (shared > threshold * shared.max()).cpu().numpy()
    return distance_transform_edt(~inside) * voxel_size, fsc_sphere


def support_mask(
    a: torch.Tensor,
    b: torch.Tensor,
    voxel_size: float,
    threshold: float = 0.2,
    near_multiplier: float = 2.0,
    far_multiplier: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    The soft mask the solved map is confined to, from two half-maps.

    CryoSPARC's refinement ("dynamic") mask with its defaults: the
    FSC-weighted half-map mean thresholded at 0.2 of its maximum, 1 within
    ``2 r`` of that region and a cosine to 0 at ``5 r``, ``r`` the halves'
    FSC = 0.143 resolution inside a soft sphere. Looser than the resolution
    mask by design: it only has to keep noise from growing in the solvent.

    Parameters
    ----------
    a, b : torch.Tensor
        Half-maps, ``(n, n, n)``.
    voxel_size : float
        Voxel size in Å.
    threshold, near_multiplier, far_multiplier : float
        As in CryoSPARC's dynamic refinement mask.

    Returns
    -------
    mask : torch.Tensor
        Mask in [0, 1].
    info : dict
        ``near`` and ``far`` in Å.
    """
    n = a.shape[-1]
    distance, fsc_sphere = _distance_from_density(a, b, voxel_size, threshold)
    r = n * voxel_size / max(fsc_crossing(fsc_sphere), 1.0)
    near, far = near_multiplier * r, far_multiplier * r
    return resolution_mask(distance, near, far, a.device), dict(
        near=float(near), far=float(far)
    )


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


@dataclass
class SolveResult:
    """
    What `ConvergedSolver.solve` returns.

    Attributes
    ----------
    volume : torch.Tensor
        The minimiser: the Wiener-filtered map.
    unfiltered : torch.Tensor
        `volume` with the filter undone (see `unfilter`).
    log : list of dict
        One entry per iteration: objective, step length, seconds.
    info : dict
        Batch size, particle counts and setup timings.
    absorption : torch.Tensor or None
        The absorption map W, when the solver was asked for one.
    """

    volume: torch.Tensor
    unfiltered: torch.Tensor
    log: list[dict[str, Any]] = field(default_factory=list)
    absorption: torch.Tensor | None = None
    info: dict[str, Any] = field(default_factory=dict)


class ConvergedSolver:
    """
    Minimise `Reconstructor`'s image loss plus a Wiener prior over one half-set.

    Parameters
    ----------
    model : Reconstructor
        Supplies the forward model, poses, CTF, per-particle scale, k-mask
        and symmetry, all held fixed. Must be on the device to solve on.
    images : torch.Tensor
        The half-set's images in counts, ``(N, H, W)``, in `model`'s particle
        order. May stay on the CPU; batches are copied as they are used.
    ssnr : numpy.ndarray
        Half-map SSNR per Fourier shell, ``n // 2 + 1`` values (see
        `ssnr_from_fsc`).
    batchsize : int
        Largest number of particles per forward pass. Halved until a pass
        fits in device memory; the result does not depend on it, since
        every iteration sums over every particle.
    line_search_fraction : float
        Fraction of particles the step length is refined on.
    seed : int
        Seeds the probes and line-search subsets.
    label : str
        Prefixed to progress messages, e.g. the half-set's name.
    absorption_resolution : float, optional
        Also solve for an absorption map W, band-limited to this resolution
        in Å, added to the amplitude contrast's absorption: the model sees
        ``V (sqrt(1 - a^2) + i a) + i W``. W carries a weak ridge prior, a
        thousandth of its typical data curvature, and the point-group
        symmetry, and starts at zero, so the solve starts at the starting
        map's own model. The solver applies the amplitude contrast itself,
        so `model`'s must be 0.
    noise_weighting : bool
        Weight the data term by the inverse noise power per image frequency
        shell, ``1 / sigma^2(k)``, measured once from the residuals of the
        starting map on 2% of the particles: the likelihood of noise that is
        Gaussian but not white, as CryoSPARC and RELION fit it. Plain squared
        error assumes white noise; on EMPIAR-10083 the residual power falls
        40-fold from 20 to 3.1 Å.
    support : torch.Tensor, optional
        A soft mask, ``(n, n, n)``, the map is confined to: the model sees
        ``support * S(U)``. Without it the solvent, which no prior that
        depends on resolution alone can tell from the particle, collects
        noise as the solve converges (on EMPIAR-10083 the 3.6-2.9 Å solvent
        noise grew to 1.8 times an Adam map's, and kept growing). See
        `support_mask`.
    amplitude_contrast : float
        ``a`` above: the amplitude contrast the data were fitted with, when
        an absorption map is solved for. Kept rather than replaced by W:
        W is band-limited, and dropping ``a`` changes the transfer at every
        resolution above W's band, which cost EMPIAR-10083 its top band.

    Notes
    -----
    ``W(k)`` is measured, not modelled: the curvature of the data term along
    random unit directions confined to each of up to 45 radial bands, on a
    few hundred particles, scaled to the half-set. The lowest nine shells,
    where one band average is too coarse a preconditioner, are measured one
    shell at a time on 2% of the particles; that measurement enters the
    preconditioner only, never the prior.
    """

    def __init__(
        self,
        model: Reconstructor,
        images: torch.Tensor,
        ssnr: np.ndarray,
        batchsize: int = 128,
        line_search_fraction: float = 0.05,
        seed: int = 0,
        label: str = "",
        absorption_resolution: float | None = None,
        amplitude_contrast: float = 0.0,
        support: torch.Tensor | None = None,
        noise_weighting: bool = False,
    ) -> None:
        self.model = model.eval()
        self.images = images
        self.device = model.V.device
        self.n_particles = len(images)
        self.n = int(model.V.shape[-1])
        self.level = float(model.dose_per_angstrom * model.voxel_size**2)
        self.batchsize = batchsize
        self.line_search_fraction = line_search_fraction
        self.seed = seed
        self._tag = f"converged solver{f' ({label})' if label else ''}"
        self.radius, self.shell, _ = frequency_shells(self.n, self.device)
        self.ssnr = np.asarray(ssnr, dtype=np.float64)
        if len(self.ssnr) != self.n // 2 + 1:
            raise ValueError(
                f"ssnr has {len(self.ssnr)} shells; a {self.n}^3 volume has {self.n // 2 + 1}"
            )
        self._kmask_half = model._kmask_half()
        self.absorption_resolution = absorption_resolution
        self.amplitude_contrast = amplitude_contrast
        self.support = (
            None if support is None else support.to(self.device, torch.float32)
        )
        self.noise_weighting = noise_weighting
        #: ``1 / sigma(k)`` on the images' half spectrum, normalised to a
        #: mean square of 1; None for white noise.
        self._whitening: torch.Tensor | None = None
        self._wband: torch.Tensor | None = None
        if absorption_resolution is not None:
            if model.propagation.alpha != 0:
                raise ValueError(
                    "an absorption map replaces the amplitude contrast: build the "
                    f"model with alpha = 0, not {model.propagation.alpha}"
                )
            if model.propagation.rotate_mode != "surface":
                raise ValueError(
                    "an absorption map needs rotate_mode='surface', which samples a "
                    "complex potential's transform"
                )
            #: W's band limit on the half spectrum, in Fourier voxels.
            cutoff = self.n * float(model.voxel_size) / absorption_resolution
            self._wband = (self.radius <= cutoff).float()
        self.channels = 1 if absorption_resolution is None else 2

    # ------------------------------------------------------------------
    # The forward model and the passes over particles

    def _predict(self, idx: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """The predicted images of state `X`: channel 0 the map V, and channel
        1, when present, the absorption map W, added to the amplitude
        contrast's own: ``V (sqrt(1 - a^2) + i a) + i W``."""
        if X.ndim == 3:
            V = X
        elif self.channels == 1:
            V = X[0]
        else:
            a = self.amplitude_contrast
            V = torch.complex(math.sqrt(1 - a**2) * X[0], a * X[0] + X[1])
        f = functional_call(self.model, {"V": V}, (idx,))
        scale = self.model.scale[idx][:, None, None]
        return self.level + scale * (f - self.level)

    def _whiten(self, x: torch.Tensor) -> torch.Tensor:
        """Images (or residuals) times ``1 / sigma(k)``; unchanged for white noise."""
        if self._whitening is None:
            return x
        return torch.fft.irfft2(torch.fft.rfft2(x) * self._whitening, s=x.shape[-2:])

    def _image_rings(self) -> torch.Tensor:
        n = self.images.shape[-1]
        f = torch.fft.fftfreq(n, device=self.device)
        fr = torch.fft.rfftfreq(n, device=self.device)
        radius = torch.sqrt(f[:, None] ** 2 + fr[None] ** 2) * n
        return torch.round(radius).long().clamp(max=n // 2)

    def estimate_noise(self, V: torch.Tensor) -> np.ndarray:
        """The residual power per image ring at `V` (see `_estimate_noise_pass`)."""
        result: np.ndarray = self._retrying(lambda: self._estimate_noise_pass(V))
        return result

    @torch.no_grad()
    def _estimate_noise_pass(self, V: torch.Tensor) -> np.ndarray:
        """
        The residual power per image frequency ring at `V`, and the whitening it implies.

        Measured on 2% of the particles (256 to 4096); sets the weighting the
        data term uses from then on. Returns sigma^2 per ring.
        """
        cpu_gen = torch.Generator().manual_seed(self.seed + 4)
        idx = self._subset(0.02, 256, 4096, cpu_gen)
        ring = self._image_rings()
        n_rings = int(ring.max()) + 1
        power = torch.zeros(n_rings, device=self.device, dtype=torch.float64)
        count = torch.zeros(n_rings, device=self.device, dtype=torch.float64)
        count.index_add_(
            0, ring.flatten(), torch.ones_like(ring, dtype=torch.float64).flatten()
        )
        for b in self._batches(idx):
            r = self._predict(b, V) - self._batch_images(b)
            p = torch.fft.rfft2(r).abs().pow(2).double().sum(0)
            power.index_add_(0, ring.flatten(), p.flatten())
        sigma2 = (power / (count * len(idx))).clamp(min=1e-30)
        weight = sigma2.rsqrt()[ring].float()
        # Mean square 1 over the full spectrum, so the loss keeps its scale.
        full = torch.full_like(weight, 2.0)
        full[..., 0] = 1.0
        if self.images.shape[-1] % 2 == 0:
            full[..., -1] = 1.0
        weight = weight / torch.sqrt((weight**2 * full).sum() / full.sum())
        self._whitening = weight
        return sigma2.cpu().numpy()

    def _batch_images(self, idx: torch.Tensor) -> torch.Tensor:
        return self.images[idx.to(self.images.device)].to(
            self.device, torch.float32, non_blocking=True
        )

    def _batches(self, idx: torch.Tensor) -> list[torch.Tensor]:
        return list(idx.split(self.batchsize))

    def _fit_batchsize(self, V: torch.Tensor) -> None:
        """Halve the batch size until one gradient batch fits in memory."""
        while True:
            idx = torch.arange(
                min(self.batchsize, self.n_particles), device=self.device
            )
            X = V.detach().clone().requires_grad_(True)
            try:
                with nufft.adjoint_tolerance(ADJOINT_TOLERANCE):
                    r = self._whiten(self._predict(idx, X) - self._batch_images(idx))
                    (0.5 * (r**2).sum()).backward()
                return
            except torch.cuda.OutOfMemoryError:
                if self.batchsize == 1:
                    raise
                self.batchsize = max(1, self.batchsize // 2)
                logger.info(
                    f"{self._tag}: out of memory, batch size now {self.batchsize}"
                )
            finally:
                del X
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

    def _out_of_memory(self, error: BaseException) -> bool:
        """Whether `error` is a device allocation failure, PyTorch's or cuFINUFFT's."""
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
        message = str(error)
        return isinstance(error, RuntimeError) and (
            "out of memory" in message or "non-uniform points" in message
        )

    def _retrying(self, run: Any) -> Any:
        """
        Run a pass over particles, halving the batch size on an out-of-memory error.

        On a shared GPU another process can take memory mid-solve; the pass is
        redone from its start with smaller batches, which changes nothing but
        speed, since every pass sums over its particles regardless of batch.
        """
        while True:
            try:
                return run()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if not self._out_of_memory(error) or self.batchsize == 1:
                    raise
                self.batchsize = max(1, self.batchsize // 2)
                logger.info(
                    f"{self._tag}: out of memory, batch size now {self.batchsize}"
                )
                # The NUFFT plans hold their own device buffers; release them
                # too, or the redone pass meets the same fragmented memory.
                nufft.clear_plan_cache()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

    def _gradient(self, V: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Data-term gradient and loss over every particle."""
        result: tuple[torch.Tensor, float] = self._retrying(
            lambda: self._gradient_pass(V)
        )
        return result

    def _gradient_pass(self, V: torch.Tensor) -> tuple[torch.Tensor, float]:
        X = V.detach().clone().requires_grad_(True)
        total = torch.zeros((), device=self.device, dtype=torch.float64)
        with nufft.adjoint_tolerance(ADJOINT_TOLERANCE):
            for idx in self._batches(
                torch.arange(self.n_particles, device=self.device)
            ):
                r = self._whiten(self._predict(idx, X) - self._batch_images(idx))
                loss = 0.5 * (r**2).sum()
                loss.backward()
                total += loss.detach().double()
        assert X.grad is not None
        return X.grad, float(total)

    def _curvature(
        self, V: torch.Tensor, d: torch.Tensor, step: float, idx: torch.Tensor
    ) -> float:
        """``sum_i ||J_i d||^2`` over `idx` (see `_curvature_pass`)."""
        result: float = self._retrying(lambda: self._curvature_pass(V, d, step, idx))
        return result

    @torch.no_grad()
    def _curvature_pass(
        self, V: torch.Tensor, d: torch.Tensor, step: float, idx: torch.Tensor
    ) -> float:
        """``sum_i ||J_i d||^2`` over `idx`, by a central difference of size `step`."""
        total = torch.zeros((), device=self.device, dtype=torch.float64)
        for b in self._batches(idx):
            jd = self._whiten(
                (self._predict(b, V + step * d) - self._predict(b, V - step * d))
                / (2 * step)
            )
            total += (jd.double() ** 2).sum()
        return float(total)

    def _loss_changes(
        self, V: torch.Tensor, d: torch.Tensor, steps: list[float], idx: torch.Tensor
    ) -> list[float]:
        """Change of the data loss on `idx` for each step (see `_loss_changes_pass`)."""
        result: list[float] = self._retrying(
            lambda: self._loss_changes_pass(V, d, steps, idx)
        )
        return result

    @torch.no_grad()
    def _loss_changes_pass(
        self, V: torch.Tensor, d: torch.Tensor, steps: list[float], idx: torch.Tensor
    ) -> list[float]:
        """
        Change of the data loss on `idx` from ``V`` to ``V + a d``, for each `a`.

        Formed as ``(p_a - p_0) (p_a + p_0 - 2 y) / 2`` rather than as a
        difference of two losses: near convergence a step changes a subset's
        loss by parts in 1e8, which two float32 sums cannot resolve.
        """
        totals = [
            torch.zeros((), device=self.device, dtype=torch.float64) for _ in steps
        ]
        for b in self._batches(idx):
            y = self._batch_images(b)
            p0 = self._predict(b, V)
            for k, a in enumerate(steps):
                pa = self._predict(b, V + a * d)
                totals[k] += (
                    0.5
                    * (
                        self._whiten(pa - p0).double()
                        * self._whiten(pa + p0 - 2 * y).double()
                    ).sum()
                )
        return [float(t) for t in totals]

    def _subset(
        self, fraction: float, minimum: int, maximum: int, generator: torch.Generator
    ) -> torch.Tensor:
        k = int(
            np.clip(
                round(fraction * self.n_particles),
                min(minimum, self.n_particles),
                min(maximum, self.n_particles),
            )
        )
        return (
            torch.randperm(self.n_particles, generator=generator)[:k]
            .sort()
            .values.to(self.device)
        )

    # ------------------------------------------------------------------
    # Fourier-space operators

    def _filter(self, X: torch.Tensor, multiplier: torch.Tensor) -> torch.Tensor:
        """A Fourier multiplier on each channel (or on one volume)."""
        dims = (-3, -2, -1)
        f = torch.fft.rfftn(X, dim=dims) * multiplier
        return torch.fft.irfftn(f, s=X.shape[-3:], dim=dims)

    def _project_one(self, V: torch.Tensor, band: torch.Tensor | None) -> torch.Tensor:
        model = self.model
        if model.symmetry is not None:
            V = apply_symmetry(
                V,
                model.sym_rot_matrices,
                batchsize=model.symmetry_batchsize,
                method=model.symmetry_mode,
            )
        V = V.contiguous()
        _apply_kmask_inplace(V, band)
        return V

    def _project(self, X: torch.Tensor) -> torch.Tensor:
        """S: each channel onto the point group and its band limit, as an
        Adam step does for the map."""
        if X.ndim == 3:
            return self._project_one(X, self._kmask_half)
        bands = [self._kmask_half, self._wband]
        return torch.stack([self._project_one(X[c], bands[c]) for c in range(len(X))])

    def _to_map(self, X: torch.Tensor) -> torch.Tensor:
        """The map(s) the model sees for unknowns `X`: ``support * S(X)``."""
        X = self._project(X)
        return X if self.support is None else X * self.support

    def _to_map_adjoint(self, G: torch.Tensor) -> torch.Tensor:
        """The adjoint of `_to_map`, ``S(support * G)``: a gradient in the
        map, taken back to the unknowns."""
        return self._project(G if self.support is None else G * self.support)

    def _random_direction(
        self, selection: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        shape = selection.shape
        noise = torch.complex(
            torch.randn(shape, device=self.device, generator=generator),
            torch.randn(shape, device=self.device, generator=generator),
        )
        d = torch.fft.irfftn(noise * selection, s=(self.n,) * 3)
        return d / d.norm()

    def _step_size(self, V: torch.Tensor, d: torch.Tensor) -> float:
        return 1e-2 * max(float(V.norm()), 1e-12) / float(d.norm())

    def _in_channel(self, d: torch.Tensor, channel: int) -> torch.Tensor:
        out = torch.zeros((self.channels, *d.shape), device=d.device, dtype=d.dtype)
        out[channel] = d
        return out

    def measure_curvature(
        self,
        V: torch.Tensor,
        n_bands: int = 45,
        n_probes: int = 2,
        n_probe_particles: int = 256,
        low_shells: int = 8,
    ) -> tuple[np.ndarray, list[float]]:
        """
        The data term's curvature: band averages for the prior, low shells for the preconditioner.

        Parameters
        ----------
        V : torch.Tensor
            State the curvature is measured at, ``(channels, n, n, n)``.
        n_bands : int
            Radial bands between 0 and Nyquist (at most ``n // 2``).
        n_probes : int
            Random directions per band.
        n_probe_particles : int
            Particles the band probes are evaluated on.
        low_shells : int
            Shells 0 to `low_shells` are measured one at a time.

        Returns
        -------
        band_curvature : numpy.ndarray
            Curvature per band, per unit-norm direction, for the whole half-set.
        low_shell_curvature : list of float
            The same for each low shell.
        """
        gen = torch.Generator(device=self.device).manual_seed(self.seed + 1)
        cpu_gen = torch.Generator().manual_seed(self.seed + 1)
        nb = max(1, min(n_bands, self.n // 2))
        edges = torch.linspace(0, self.n / 2, nb + 1, device=self.device)
        particles = self._subset(0.0, n_probe_particles, n_probe_particles, cpu_gen)
        per_particle = self.n_particles / len(particles)
        bands = np.zeros(nb)
        for b in range(nb):
            selection = (self.radius >= edges[b]) & (self.radius < edges[b + 1])
            for _ in range(n_probes):
                d = self._in_channel(self._random_direction(selection, gen), 0)
                bands[b] += (
                    self._curvature(V, d, self._step_size(V, d), particles)
                    * per_particle
                    / n_probes
                )
        particles = self._subset(0.02, 256, 4096, cpu_gen)
        per_particle = self.n_particles / len(particles)
        low = []
        for s in range(min(low_shells, self.n // 2) + 1):
            selection = self.shell == s
            if s == 0:
                d = torch.fft.irfftn(selection.to(torch.complex64), s=(self.n,) * 3)
                d = d / d.norm()
            else:
                d = self._random_direction(selection, gen)
            d = self._in_channel(d, 0)
            low.append(
                self._curvature(V, d, self._step_size(V, d), particles) * per_particle
            )
        return bands, low

    def measure_absorption_curvature(
        self, X: torch.Tensor
    ) -> tuple[list[int], list[float]]:
        """
        The data term's curvature along W, at a ladder of shells inside W's band.

        Measured on 2% of the particles (256 to 4096), one random direction
        per shell, and interpolated between shells for the preconditioner.
        """
        assert self._wband is not None
        gen = torch.Generator(device=self.device).manual_seed(self.seed + 3)
        cpu_gen = torch.Generator().manual_seed(self.seed + 3)
        top = int(self.radius[self._wband > 0].max())
        ladder = sorted(
            {s for s in (0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 28) if s < top} | {top}
        )
        particles = self._subset(0.02, 256, 4096, cpu_gen)
        per_particle = self.n_particles / len(particles)
        values = []
        for s in ladder:
            selection = self.shell == s
            if s == 0:
                d = torch.fft.irfftn(selection.to(torch.complex64), s=(self.n,) * 3)
                d = d / d.norm()
            else:
                d = self._random_direction(selection, gen)
            d = self._in_channel(d, 1)
            values.append(
                self._curvature(X, d, self._step_size(X, d), particles) * per_particle
            )
        return ladder, values

    # ------------------------------------------------------------------

    def solve(self, V: torch.Tensor, n_iter: int) -> SolveResult:
        """
        Run `n_iter` iterations of preconditioned PR+ conjugate gradients from `V`.

        Parameters
        ----------
        V : torch.Tensor
            Starting map, ``(n, n, n)``; a few epochs of Adam is enough.
        n_iter : int
            Iterations. Four from a five-epoch Adam map matched a sixteen-
            iteration solve to 1 - FSC <= 3e-6 in every band.

        Returns
        -------
        SolveResult
        """
        t_start = time.perf_counter()
        # The unknown is U, and the map is V = S(U), S the point-group average
        # and band limit: the problem is then unconstrained, with gradient
        # S(dL/dV) since S is self-adjoint. Projecting a gradient and the
        # iterate instead assumes S(S(x)) = S(x), which the exact average only
        # does for content inside the inscribed sphere: an I1 rotation cannot
        # carry the cube's corners onto themselves, so the noise there (16% of
        # an Adam map's norm on EMPIAR-12391) changes 14% per application, the
        # whole map 2%, and the projected iteration walked uphill.
        V0 = V.to(self.device, torch.float32)
        if self.channels == 1:
            start = V0[None]
        else:
            # W corrects the amplitude contrast; it starts at none, which is
            # the starting map's own model.
            start = torch.stack([V0, torch.zeros_like(V0)])
        V = self._to_map(start)  # the map for U = the start, (channels, n, n, n)
        del V0, start
        self._fit_batchsize(V)

        t0 = time.perf_counter()
        noise_info: dict[str, Any] = {}
        if self.noise_weighting:
            noise_info["noise_power"] = self.estimate_noise(V).tolist()
        bands, low = self.measure_curvature(V)
        nb = len(bands)
        centres = (np.arange(nb) + 0.5) * (self.n / 2) / nb
        W = torch.as_tensor(
            np.interp(self.radius.cpu().numpy(), centres, bands),
            dtype=torch.float32,
            device=self.device,
        )
        ssnr = torch.as_tensor(self.ssnr, dtype=torch.float32, device=self.device)[
            self.shell
        ]
        lam = W / ssnr
        for s, value in enumerate(low):
            W[self.shell == s] = value
        precondition = (self.radius <= self.n / 2).float() / (W + lam)
        #: The prior's per-voxel weight on the half spectrum, ``W(k) / SSNR(k)``.
        self.prior_weight = lam
        del W, ssnr
        absorption_info: dict[str, Any] = {}
        if self.channels == 2:
            assert self._wband is not None
            ladder, values = self.measure_absorption_curvature(V)
            Hw = torch.as_tensor(
                np.interp(self.radius.cpu().numpy(), ladder, values),
                dtype=torch.float32,
                device=self.device,
            )
            # A weak ridge inside the band; W is zero outside it anyway.
            lam_w = 1e-3 * float(np.median(values)) * self._wband + (1 - self._wband)
            lam = torch.stack([lam, lam_w])
            precondition = torch.stack([precondition, self._wband / (Hw + lam_w)])
            absorption_info = dict(
                absorption_ladder=ladder, absorption_curvature=values
            )
            del Hw, lam_w
        else:
            lam, precondition = lam[None], precondition[None]
        t_curvature = time.perf_counter() - t0
        logger.info(
            f"{self._tag}: curvature measured in {t_curvature:.0f} s, batch size {self.batchsize}"
        )

        cpu_gen = torch.Generator().manual_seed(self.seed + 2)
        log: list[dict[str, Any]] = []
        d_prev: torch.Tensor | None = None
        g_prev: torch.Tensor | None = None
        z_prev: torch.Tensor | None = None
        for it in range(n_iter):
            t0 = time.perf_counter()
            grad, data_loss = self._gradient(V)
            lam_V = self._filter(V, lam)
            objective = data_loss + 0.5 * float((V * lam_V).sum())
            grad_V = grad + lam_V  # dObjective/dV
            del grad, lam_V
            g = self._to_map_adjoint(grad_V)  # dObjective/dU
            # Preconditioner S M S: M is radial, diagonal on the DFT grid, but
            # an exact rotation average is not (quarter turns of an even box
            # land off-grid), so M alone puts the search direction into what S
            # annihilates and the steps change nothing.
            z = self._project(self._filter(g, precondition))
            steepest = d_prev is None or g_prev is None or z_prev is None
            if steepest:
                d = -z
            else:
                assert d_prev is not None and g_prev is not None and z_prev is not None
                beta = max(
                    0.0,
                    float((z * (g - g_prev)).sum()) / float((z_prev * g_prev).sum()),
                )
                d = -z + beta * d_prev
            # The map moves along S(d). The slope is taken along that move,
            # from the full gradient in V, not as g.d, which equals it only
            # as far as S is self-adjoint (1e-5 on a real map, 1.6e-3 for I1
            # on an 8^3 box, enough near convergence to flip its sign).
            d_map = self._to_map(d)
            slope = float((grad_V * d_map).sum())
            if slope >= 0 and not steepest:
                d = -z
                d_map = self._to_map(d)
                slope = float((grad_V * d_map).sum())
            del grad_V
            d_lam_d = float((d_map * self._filter(d_map, lam)).sum())
            subset = self._subset(
                self.line_search_fraction, 32, self.n_particles, cpu_gen
            )
            per_particle = self.n_particles / len(subset)
            curvature = (
                self._curvature(V, d_map, self._step_size(V, d_map), subset)
                * per_particle
            )
            if slope >= 0:
                # No descent even along -S M g: converged to rounding.
                alpha = 0.0
            else:
                alpha = -slope / (curvature + d_lam_d)
                # Refine the step with the data term's actual curvature along d,
                # read off its change at alpha/2 and alpha on the subset; the full
                # Hessian, not its Gauss-Newton part, so the model's nonlinearity
                # counts. Only the curvature comes from the subset. The slope is
                # the full gradient's: near the Wiener balance the data and prior
                # slopes nearly cancel, and a subset's estimate of the data slope
                # is then mostly sampling error (on EMPIAR-12391 it put every
                # step at a quarter of its length).
                half, full = self._loss_changes(V, d_map, [0.5 * alpha, alpha], subset)
                data_curvature = 4 * (full - 2 * half) / alpha**2 * per_particle
                denominator = data_curvature + d_lam_d
                a_star = -slope / denominator if denominator > 0 else alpha
                alpha = float(np.clip(a_star, 0.05 * alpha, 2.0 * alpha))
            V = V + alpha * d_map  # the map of U + alpha d, the map being linear in U
            del d_map
            d_prev, g_prev, z_prev = d, g, z
            entry = dict(
                iteration=it + 1,
                objective=objective,
                data_loss=data_loss,
                step=alpha,
                seconds=round(time.perf_counter() - t0, 1),
            )
            log.append(entry)
            logger.info(
                f"{self._tag}: iteration {it + 1}/{n_iter}, objective {objective:.6g}, {entry['seconds']:.0f} s"
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        info = dict(
            n_particles=self.n_particles,
            batchsize=self.batchsize,
            curvature_seconds=round(t_curvature, 1),
            seconds=round(time.perf_counter() - t_start, 1),
            band_curvature=bands.tolist(),
            low_shell_curvature=low,
            ssnr=self.ssnr.tolist(),
            **absorption_info,
            **noise_info,
        )
        return SolveResult(
            volume=V[0],
            unfiltered=unfilter(V[0], self.ssnr),
            log=log,
            info=info,
            absorption=V[1] if self.channels == 2 else None,
        )
