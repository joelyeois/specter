"""
FSC readouts: the resolution at a threshold crossing, and the FSC curve of
two volumes. Plotting them is `specter.plots`'s job; this module has no
matplotlib dependency, so a reconstruction can report resolution without it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .fft import fourier_shell_correlation, fsc_crossing

__all__ = [
    "HALFMAP_FSC_THRESHOLD",
    "MAP_TO_MODEL_FSC_THRESHOLD",
    "FSCCurve",
    "fsc_curve",
    "fsc_resolution",
    "resolution_between",
]


#: Gold-standard half-map FSC threshold (two independent half-set reconstructions;
#: Rosenthal & Henderson 2003).
HALFMAP_FSC_THRESHOLD = 0.143
#: Map-to-model FSC threshold (a reconstruction against a known reference): the
#: correlation with a perfect reference that a half-map FSC of 0.143 corresponds
#: to (Rosenthal & Henderson 2003).
MAP_TO_MODEL_FSC_THRESHOLD = 0.5


def fsc_resolution(
    k: torch.Tensor,
    fsc: torch.Tensor,
    threshold: float,
    k_max: float | None = None,
) -> str:
    """
    Resolution at the first FSC threshold crossing, linearly interpolated.

    The *first* crossing, as RELION and CryoSPARC report it
    (`specter.fft.fsc_crossing`). A masked FSC rises again toward Nyquist
    when both maps share the mask, because the mask correlates their noise;
    reading the last crossing then reports that artefact as resolution
    (1.79 A against a true ~1.91 A on EMPIAR-13203).

    Parameters
    ----------
    k : torch.Tensor
        Spatial frequencies (1/Å), one per shell.
    fsc : torch.Tensor
        FSC value per shell, same length as ``k``.
    threshold : float
        Crossing to interpolate -- `HALFMAP_FSC_THRESHOLD` for two half-sets,
        `MAP_TO_MODEL_FSC_THRESHOLD` against a known reference.
    k_max : float, optional
        Ignore shells above this frequency, for a curve that runs past
        Nyquist. `specter.fft.fourier_shell_correlation`'s stops there.

    Returns
    -------
    str
        Resolution formatted as ``"1.397 Å"``, ``">Nyquist"`` when the FSC
        never falls below the threshold, or ``"worse than"`` the lowest nonzero
        shell's resolution when it never reaches it.

    References
    ----------
    .. [1] Rosenthal, P. B., & Henderson, R. (2003). Optimal determination of
       particle orientation, absolute hand, and contrast loss in
       single-particle electron cryomicroscopy. *Journal of Molecular
       Biology*, 333(4), 721–745. https://doi.org/10.1016/j.jmb.2003.07.013
    """
    kf = k.detach().cpu().double()
    ff = fsc.detach().cpu()
    if k_max is not None:
        keep = int((kf <= k_max).sum())
        kf, ff = kf[:keep], ff[:keep]
    if len(ff) < 2:
        return ">Nyquist"
    shell = fsc_crossing(ff, threshold)
    if shell == len(ff) - 1:
        return ">Nyquist"
    if not bool((ff >= threshold).any()):
        return f"worse than {1.0 / float(kf[1]):.3f} Å"
    i = min(int(shell), len(kf) - 2)
    k_cross = kf[i] + (shell - i) * (kf[i + 1] - kf[i])
    return f"{1.0 / float(k_cross):.3f} Å"


@dataclass(frozen=True)
class FSCCurve:
    """
    One FSC between two volumes and its resolution, optionally also masked.

    Attributes
    ----------
    k, fsc : torch.Tensor
        Frequencies (1/Å) and the unmasked FSC, from
        `specter.fft.fourier_shell_correlation`.
    resolution : str
        At the curve's threshold, as `fsc_resolution` formats it.
    fsc_masked : torch.Tensor or None
        The FSC inside the mask, when one was given.
    resolution_masked : str or None
        Its resolution.
    """

    k: torch.Tensor
    fsc: torch.Tensor
    resolution: str
    fsc_masked: torch.Tensor | None = None
    resolution_masked: str | None = None


def fsc_curve(
    volume_a: torch.Tensor,
    volume_b: torch.Tensor,
    voxel_size: float,
    threshold: float,
    mask: torch.Tensor | None = None,
) -> FSCCurve:
    """
    The FSC of two volumes and its resolution, plus the masked pair with a mask.

    Parameters
    ----------
    volume_a, volume_b : torch.Tensor
        Volumes to correlate, shape ``(N, N, N)``; `volume_b` is moved to
        `volume_a`'s device.
    voxel_size : float
        Voxel size in Å.
    threshold : float
        `HALFMAP_FSC_THRESHOLD` or `MAP_TO_MODEL_FSC_THRESHOLD`.
    mask : torch.Tensor, optional
        Gives the masked curve as well.

    Returns
    -------
    FSCCurve
    """
    a = volume_a.detach().float()
    b = volume_b.detach().to(a.device).float()
    k, fsc = fourier_shell_correlation(a, b, pixel_size=voxel_size)
    curve = FSCCurve(k, fsc, fsc_resolution(k, fsc, threshold))
    if mask is None:
        return curve
    _, fsc_m = fourier_shell_correlation(
        a, b, pixel_size=voxel_size, mask=mask.detach().float()
    )
    return replace(
        curve, fsc_masked=fsc_m, resolution_masked=fsc_resolution(k, fsc_m, threshold)
    )


def resolution_between(
    volume_a: torch.Tensor,
    volume_b: torch.Tensor,
    voxel_size: float,
    threshold: float,
    mask: torch.Tensor | None = None,
) -> str:
    """
    Resolution at an FSC threshold between two volumes, optionally masked.

    Parameters
    ----------
    volume_a, volume_b : torch.Tensor
        Volumes to correlate, shape ``(Z, Y, X)``. Two half-sets for a
        gold-standard resolution, or a reconstruction and its reference for
        map-to-model.
    voxel_size : float
        Voxel size in Å.
    threshold : float
        `HALFMAP_FSC_THRESHOLD` for two half-sets, `MAP_TO_MODEL_FSC_THRESHOLD`
        against a known reference.
    mask : torch.Tensor, optional
        Applied to both volumes before correlation. None correlates them as-is.

    Returns
    -------
    str
        Resolution formatted as ``"1.397 Å"``, or ``">Nyquist"``.
    """
    a = volume_a.detach().float()
    b = volume_b.detach().to(a.device).float()
    m = None if mask is None else mask.detach().float()
    k, fsc = fourier_shell_correlation(a, b, pixel_size=voxel_size, mask=m)
    return fsc_resolution(k, fsc, threshold)
