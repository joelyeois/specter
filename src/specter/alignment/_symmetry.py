"""Alignment of a volume to the axes of its point group."""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import torch

from specter import logger

from ..symmetries import get_rotation_matrices
from ._apply import apply_alignment
from ._band import Band, noise_power, shells, spectrum_of, widest_band
from ._frames import axis_grid, frame_distance, frame_group, nearest_frame, so3_grid
from ._ladder import SIGNIFICANT, Alignment, Pose, refine_while_shells_agree

_TWO_PI = 2.0 * math.pi
#: Band limit of the grid searches, in Fourier voxels. It sets the rotation
#: step (1 / radius rad) and with it the size of the search.
COARSE_RADIUS = 8.0
#: Orientations given a centre search, and the number of those refined.
N_CANDIDATES = 8
N_REFINE = 3
#: Largest number of half-band frequencies a refinement scores: every one of
#: the widest band built by default.
N_POINTS = 800_000
#: Transform samples evaluated per call in the grid searches.
POINT_BUDGET = 4_000_000


@dataclass
class _Stage:
    """A band and the frequencies scored in it."""

    band: Band
    pts: torch.Tensor

    @property
    def radius(self) -> float:
        return self.band.radius

    @property
    def n_full(self) -> int:
        return self.band.n_full


def _orbit(rot: torch.Tensor, ops: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Frequencies ``R g p`` of shape (B, G, P, 3)."""
    return torch.einsum("bij,gjk,pk->bgpi", rot, ops, pts)


def _agreement(values: torch.Tensor) -> torch.Tensor:
    """
    Mean correlation between symmetry copies.

    Parameters
    ----------
    values : torch.Tensor
        One value per rotation, group element and point, (B, G, P).

    Returns
    -------
    torch.Tensor
        (B,): 1 when every copy is equal, 0 when they are unrelated.
    """
    g = values.shape[1]
    num = (values.sum(1).abs() ** 2).sum(-1)
    den = (values.abs() ** 2).sum((1, 2))
    return (num / den - 1) / (g - 1)


def _intensity_score(
    band: Band, rot: torch.Tensor, ops: torch.Tensor, pts: torch.Tensor
) -> torch.Tensor:
    """Self-rotation score of each rotation (B,), which no shift changes."""
    shell = shells(pts)
    mu, sd, a = band.shell_power[shell], band.shell_std[shell], band.signal[shell]
    out = []
    per = max(1, POINT_BUDGET // (ops.shape[0] * pts.shape[0]))
    for chunk in rot.split(per):
        u = _orbit(chunk, ops, pts)
        f = band.sample(u.reshape(-1, 3)).view(u.shape[:-1])
        out.append(_agreement((f.abs() ** 2 - mu) / sd * a))
    return torch.cat(out)


def _symmetry_score(
    band: Band,
    rot: torch.Tensor,
    centre: torch.Tensor,
    ops: torch.Tensor,
    pts: torch.Tensor,
) -> torch.Tensor:
    """Agreement of the weighted complex transform with its copies, (B,)."""
    u = _orbit(rot, ops, pts)
    f = band.sample(u.reshape(-1, 3)).view(u.shape[:-1])
    phase = torch.exp(
        1j * (_TWO_PI / band.n_full) * (u * centre[:, None, None, :]).sum(-1)
    )
    return _agreement(f * phase * band.amplitude_weight(pts))


def _shell_agreement(
    band: Band,
    rot: torch.Tensor,
    centre: torch.Tensor,
    ops: torch.Tensor,
    pts: torch.Tensor,
) -> torch.Tensor:
    """
    Agreement between the symmetry copies in each shell, in standard deviations.

    The symmetry analogue of an FSC curve, the share of a shell's power the
    copies have in common, divided by the spread ``1 / sqrt(P (G - 1))`` it
    has over ``P`` points when the copies are unrelated.
    """
    g = ops.shape[0]
    shell = shells(pts)
    u = _orbit(rot[None], ops, pts)[0]
    f = band.sample(u.reshape(-1, 3)).view(u.shape[:-1])
    f = f * torch.exp(1j * (_TWO_PI / band.n_full) * (u @ centre))
    zero = torch.zeros(
        band.shell_power.shape[0], dtype=torch.float64, device=pts.device
    )
    num = zero.clone().index_add_(0, shell, f.sum(0).abs() ** 2)
    den = zero.clone().index_add_(0, shell, (f.abs() ** 2).sum(0))
    cnt = zero.clone().index_add_(0, shell, torch.ones_like(den[shell]))
    agree = (num / den.clamp_min(1e-300) - 1) / (g - 1)
    return torch.where(cnt > 0, agree * (cnt * (g - 1)).sqrt(), zero)


def _centre_search(
    band: Band,
    rot: torch.Tensor,
    ops: torch.Tensor,
    pts: torch.Tensor,
    max_shift: float,
) -> tuple[torch.Tensor, float]:
    """Best centre (3,) for one rotation on a grid fine enough for the band."""
    step = band.n_full / (4 * band.radius)
    m = max(1, math.ceil(max_shift / step))
    ax = torch.arange(-m, m + 1, dtype=torch.float64, device=pts.device) * step
    grid = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    u = _orbit(rot[None], ops, pts)[0]
    f = band.sample(u.reshape(-1, 3)).view(u.shape[:-1]) * band.amplitude_weight(pts)
    scores = []
    for c in grid.split(max(1, POINT_BUDGET // f.numel())):
        phase = torch.exp(
            1j * (_TWO_PI / band.n_full) * torch.einsum("gpi,ci->cgp", u, c)
        )
        scores.append(_agreement(f[None] * phase))
    s = torch.cat(scores)
    best = s.argmax()
    return grid[best], float(s[best])


def align_to_symmetry(
    volume: torch.Tensor,
    sym: str,
    *,
    max_radius: float | None = None,
    max_shift: float | None = None,
    resample: bool = True,
) -> Alignment:
    """
    Align a volume to the axes of its point-group symmetry.

    Finds the rotation and centre that put the volume's symmetry axes in the
    positions `specter.symmetries.get_rotation_matrices` assumes, so that
    ``symmetrize_exact(aligned, sym)`` leaves the aligned volume unchanged.

    The rotation is found first and without a shift search, from the
    self-rotation function of Rossmann & Blow [1]_ evaluated on the
    intensities, which a shift of the volume does not change: the agreement
    of the shell-normalised intensities with their copies under the group,
    on a super-Fibonacci grid of rotations [2]_ reduced to the frames the
    group does not make equivalent (axis directions alone for a cyclic
    group). The best orientations each get a centre from a grid search on the
    complex transform, in a band of 8 Fourier voxels. Each is refined there,
    rotation and centre together, by BFGS with analytic gradients, and then
    tested on the shells beyond that band, which the search has not used.
    The run of shells in which the symmetry copies still agree sets the band
    of the next refinement, up to `max_radius`, and the test is repeated
    until the run stops growing. Every step works on a band-limited copy of
    the volume in a small box, so the cost depends weakly on the size of the
    volume.

    Of the equivalent aligned frames, the one that turns the volume least is
    returned. For a cyclic group that fixes neither the spin about the axis
    nor the position along it: the spin is left as in the input and the
    centre is the point of the axis nearest the central voxel.

    Parameters
    ----------
    volume : torch.Tensor
        Real cubic volume (N, N, N), on the device the search should run on.
    sym : str
        Point-group symbol: ``"Cn"``, ``"Dn"``, ``"T"``, ``"O"`` or ``"I"``.
    max_radius : float, optional
        Highest band limit used, in Fourier voxels (cycles per box). Default
        71, or the Nyquist frequency of a volume too small to reach it: the
        widest band whose every independent frequency the refinement scores.
        The band reached is lower when the symmetry copies stop agreeing
        before it.
    max_shift : float, optional
        Largest distance of the symmetry centre from the central voxel along
        each axis that the centre search covers, in voxels. Default ``N / 4``.
    resample : bool, optional
        Whether to return the aligned volume. Default True.

    Returns
    -------
    Alignment
        The aligned volume, the pose that produced it and how far the
        symmetry is supported.

    Raises
    ------
    ValueError
        If the volume is not cubic, or `sym` is ``"C1"``.

    Warns
    -----
    UserWarning
        If the alignment is not `Alignment.reliable`: the symmetry
        copies do not agree beyond the band the search was run on, as when
        the volume is too noisy to show the symmetry. The returned pose is
        then the best found, and may be wrong.

    Notes
    -----
    On four rendered assemblies of exact symmetry (C3, D4, T, O; 270 to 360
    voxels at 0.75 A) at random poses the pose is recovered to 0.001 degrees
    and 0.001 voxel without noise and to 0.013 degrees and 0.006 voxel with
    white noise of ten times the volume's variance, in 1 to 3 s on one GPU.

    A volume whose symmetry is approximate has no single aligned frame: the
    pose that best superposes its copies depends on the band they are
    compared in. For a rendered D7 assembly whose copies correlate to 0.7,
    the frames found in bands of 71 and 94 Fourier voxels differ by 0.03
    degrees and 0.016 voxel.

    References
    ----------
    .. [1] Rossmann, M. G., & Blow, D. M. (1962). The detection of sub-units
       within the crystallographic asymmetric unit. *Acta Crystallographica*,
       15(1), 24–31. doi:10.1107/S0365110X62000067
    .. [2] Alexa, M. (2022). Super-Fibonacci spirals: Fast, low-discrepancy
       sampling of SO(3). *2022 IEEE/CVF Conference on Computer Vision and
       Pattern Recognition (CVPR)*, 8281–8290.
       doi:10.1109/CVPR52688.2022.00811
    """
    n_full = volume.shape[-1]
    if volume.shape != (n_full, n_full, n_full):
        raise ValueError(f"expected a cubic volume, got shape {tuple(volume.shape)}")
    dev = volume.device
    ops = get_rotation_matrices(sym, return_affine=False).double().to(dev)
    order = ops.shape[0]
    if order < 2:
        raise ValueError(f"{sym} has no symmetry axis to align to")
    frames = frame_group(sym)
    frames = None if frames is None else frames.to(dev)
    max_radius = widest_band(n_full) if max_radius is None else max_radius
    max_shift = n_full / 4 if max_shift is None else max_shift

    def tick() -> float:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        return time.perf_counter()

    def distance(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(frame_distance(a, b, frames))

    start = tick()
    spectrum = spectrum_of(volume)
    floor = noise_power(spectrum, n_full)

    # 1. Rotation, from the intensities.
    band = Band(spectrum, n_full, min(COARSE_RADIUS, max_radius), floor)
    step = 1.0 / band.radius  # half an intensity speckle at the band edge
    if frames is None:
        grid = axis_grid(step).to(dev)
    else:
        grid = so3_grid(step).to(dev)
        tr = torch.einsum("bij,nji->bn", grid, frames)
        grid = grid[tr[:, 0] >= tr.amax(-1) - 1e-9]  # frames[0] is the identity
    pts = band.points(max(64, band.lattice.shape[0] // order))
    scores = _intensity_score(band, grid, ops, pts)
    picked: list[int] = []
    for i in scores.argsort(descending=True)[: 50 * N_CANDIDATES].tolist():
        if all(distance(grid[i], grid[j]) > 1.5 * step for j in picked):
            picked.append(i)
        if len(picked) == N_CANDIDATES:
            break
    logger.info(
        "align_to_symmetry: %d rotations at %.1f degrees, %d points, box %d (%.2f s)",
        grid.shape[0],
        math.degrees(step),
        order * pts.shape[0],
        band.n,
        tick() - start,
    )

    # 2. A centre for each candidate.
    t0 = tick()
    poses = []
    for i in picked:
        c, sc = _centre_search(band, grid[i], ops, pts, max_shift)
        poses.append(Pose(sc, grid[i], c))
    poses.sort(key=lambda p: -p.score)
    logger.info(
        "align_to_symmetry: centre search scores %s (%.2f s)",
        " ".join(f"{p.score:.3f}" for p in poses),
        tick() - t0,
    )

    # 3. Refinement out to where the symmetry copies stop agreeing.
    bands: dict[float, Band] = {}

    def band_at(radius: float) -> Band:
        if radius not in bands:
            bands[radius] = Band(spectrum, n_full, radius, floor)
        return bands[radius]

    def stage(radius: float) -> _Stage:
        b = band_at(radius)
        return _Stage(b, b.points(max(64, min(b.lattice.shape[0], N_POINTS) // order)))

    best, radius = refine_while_shells_agree(
        poses[:N_REFINE],
        _Stage(band, pts),
        stage=stage,
        score=lambda s, r, c: _symmetry_score(s.band, r, c, ops, s.pts),
        shell_z=lambda s, r, c: _shell_agreement(s.band, r, c, ops, s.pts),
        distance=distance,
        max_radius=max_radius,
        name="align_to_symmetry",
    )

    rot = nearest_frame(best.rot, frames)
    centre = best.centre
    if frames is None:
        axis = rot[:, 2]
        centre = centre - (centre @ axis) * axis
    result = Alignment(
        volume=apply_alignment(volume, rotation=rot, centre=centre)
        if resample
        else None,
        rotation=rot,
        centre=centre,
        score=best.score,
        significance=best.significance,
        radius=radius,
    )
    logger.info(
        "align_to_symmetry: %s score %.4f, significance %.1f, radius %.0f (%.2f s)",
        sym,
        result.score,
        result.significance,
        radius,
        tick() - start,
    )
    if not result.reliable:
        warnings.warn(
            f"The {sym} symmetry copies of the volume do not agree beyond the "
            f"band the search was run on (significance "
            f"{result.significance:.1f}, below {SIGNIFICANT:.0f}). The volume "
            "may be too noisy to show this symmetry, or may not have it; the "
            "returned pose may be wrong.",
            stacklevel=2,
        )
    return result
