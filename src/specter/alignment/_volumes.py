"""Alignment of one volume to another."""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import torch

from specter import logger

from ._apply import apply_alignment
from ._band import Band, noise_power, shells, spectrum_of, widest_band
from ._frames import frame_distance, so3_grid
from ._ladder import SIGNIFICANT, Alignment, Pose, refine_while_shells_agree
from ._symmetry import COARSE_RADIUS, N_CANDIDATES, N_POINTS, N_REFINE, POINT_BUDGET

_TWO_PI = 2.0 * math.pi


@dataclass
class _Stage:
    """Two bands, the frequencies scored and the reference's values there."""

    band: Band
    pts: torch.Tensor
    #: Whitened transform of the reference at `pts`, (P,).
    target: torch.Tensor
    #: Whitening weight of the moving volume at `pts`, (P,).
    weight: torch.Tensor

    @property
    def radius(self) -> float:
        return self.band.radius

    @property
    def n_full(self) -> int:
        return self.band.n_full


def _stage(band: Band, reference: Band, n_points: int) -> _Stage:
    pts = band.points(n_points)
    target = reference.sample(pts) * reference.amplitude_weight(pts)
    return _Stage(band, pts, target, band.amplitude_weight(pts))


def _normalised_intensity(
    band: Band, values: torch.Tensor, shell: torch.Tensor
) -> torch.Tensor:
    """Shell-normalised intensities, weighted by the signal each shell holds."""
    mu, sd, a = band.shell_power[shell], band.shell_std[shell], band.signal[shell]
    return (values.abs() ** 2 - mu) / sd * a


def _intensity_score(
    band: Band, reference: Band, rot: torch.Tensor, pts: torch.Tensor
) -> torch.Tensor:
    """Correlation of the two intensity distributions, (B,), which no shift changes."""
    shell = shells(pts)
    ref = _normalised_intensity(reference, reference.sample(pts), shell)
    out = []
    for chunk in rot.split(max(1, POINT_BUDGET // pts.shape[0])):
        u = torch.einsum("bij,pj->bpi", chunk, pts)
        f = band.sample(u.reshape(-1, 3)).view(u.shape[:-1])
        x = _normalised_intensity(band, f, shell)
        out.append((x * ref).sum(-1) / (x.square().sum(-1) * ref.square().sum()).sqrt())
    return torch.cat(out)


def _moved(stage: _Stage, rot: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
    """Whitened transform of the moved volume at the stage's points, (B, P)."""
    u = torch.einsum("bij,pj->bpi", rot, stage.pts)
    f = stage.band.sample(u.reshape(-1, 3)).view(u.shape[:-1])
    phase = torch.exp(1j * (_TWO_PI / stage.n_full) * (u * centre[:, None, :]).sum(-1))
    return f * phase * stage.weight


def _correlation(
    stage: _Stage, rot: torch.Tensor, centre: torch.Tensor
) -> torch.Tensor:
    """Correlation of the moved volume with the reference over the band, (B,)."""
    a = _moved(stage, rot, centre)
    num = (a * stage.target.conj()).real.sum(-1)
    den = (a.abs() ** 2).sum(-1) * (stage.target.abs() ** 2).sum()
    return num / den.sqrt()


def _shell_correlation(
    stage: _Stage, rot: torch.Tensor, centre: torch.Tensor
) -> torch.Tensor:
    """
    Fourier shell correlation with the reference, in standard deviations.

    Each shell's correlation is divided by the spread ``1 / sqrt(P)`` taken
    for unrelated volumes over the shell's ``P`` points.
    """
    a = _moved(stage, rot[None], centre[None])[0]
    shell = shells(stage.pts)
    zero = torch.zeros(
        stage.band.shell_power.shape[0], dtype=torch.float64, device=a.device
    )
    num = zero.clone().index_add_(0, shell, (a * stage.target.conj()).real)
    da = zero.clone().index_add_(0, shell, a.abs() ** 2)
    db = zero.clone().index_add_(0, shell, stage.target.abs() ** 2)
    cnt = zero.clone().index_add_(0, shell, torch.ones_like(num[shell]))
    fsc = num / (da * db).sqrt().clamp_min(1e-300)
    return torch.where(cnt > 0, fsc * cnt.sqrt(), zero)


def _shift_search(
    stage: _Stage, rot: torch.Tensor, max_shift: float
) -> tuple[torch.Tensor, float]:
    """
    Best shift (3,) for one rotation, from one inverse FFT.

    The correlation as a function of the shift ``c`` is the transform of the
    cross-spectrum at ``R^T c``, so placing the cross-spectrum on a lattice
    and transforming gives it on a grid of shifts in the rotated frame, at a
    quarter of the band's finest period. That grid spans half a box either
    way along the rotated axes.
    """
    pts = stage.pts
    f = stage.band.sample(pts @ rot.T) * stage.weight
    cross = f * stage.target.conj()
    norm = ((f.abs() ** 2).sum() * (stage.target.abs() ** 2).sum()).sqrt()
    m = 2 * math.ceil(2 * stage.radius)
    idx = pts.round().long() % m
    lattice = torch.zeros(m, m, m, dtype=cross.dtype, device=cross.device)
    lattice[idx[:, 0], idx[:, 1], idx[:, 2]] = cross
    # Each point stands for its Friedel mate too, which the real part supplies.
    scores = torch.fft.fftn(lattice).real / norm
    ax = torch.fft.fftfreq(m, d=1.0 / m, device=pts.device).double()
    grid = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1)
    shifts = -(grid * (stage.n_full / m)) @ rot.T
    scores = scores.masked_fill(shifts.abs().amax(-1) > max_shift, -math.inf)
    best = scores.argmax()
    return shifts.reshape(-1, 3)[best], float(scores.reshape(-1)[best])


def align_volumes(
    volume: torch.Tensor,
    reference: torch.Tensor,
    *,
    max_radius: float | None = None,
    max_shift: float | None = None,
    resample: bool = True,
) -> Alignment:
    """
    Align a volume to a reference volume by a rotation and a shift.

    Finds the rigid pose that brings `volume` onto `reference`, as needed
    before two independently reconstructed maps of one specimen are compared
    by Fourier shell correlation. The two may differ in scale, offset and
    noise.

    The rotation is found first and without a shift search, from the
    cross-rotation function of Rossmann & Blow [1]_: the correlation of the
    two volumes' shell-normalised intensities, which no shift of either
    changes, on a super-Fibonacci grid of rotations [2]_. The best
    orientations each get a shift from a grid search on the complex
    transforms, in a band of 8 Fourier voxels. Each is refined there,
    rotation and shift together, by BFGS with analytic gradients, maximising
    the correlation of the whitened transforms, and then tested on the
    shells beyond that band, which the search has not used. The run of
    shells in which the volumes still correlate sets the band of the next
    refinement, up to `max_radius`, and the test is repeated until the run
    stops growing. Every step works on band-limited copies of the volumes in
    a small box, so the cost depends weakly on their size.

    Parameters
    ----------
    volume : torch.Tensor
        Real cubic volume (N, N, N) to be moved, on the device the search
        should run on.
    reference : torch.Tensor
        Real volume of the same shape and voxel size, which stays in place.
    max_radius : float, optional
        Highest band limit used, in Fourier voxels (cycles per box). Default
        71, or the Nyquist frequency of a volume too small to reach it: the
        widest band whose every independent frequency the refinement scores.
        The band reached is lower when the volumes stop correlating before
        it.
    max_shift : float, optional
        Largest shift along each axis that the search covers, in voxels.
        Default ``N / 2``. Shifts longer than ``N / 2`` in total are covered
        only in part.
    resample : bool, optional
        Whether to return the aligned volume. Default True.

    Returns
    -------
    Alignment
        The aligned volume, the pose that produced it and how far the two
        volumes agree. A volume with point-group symmetry has several
        equivalent poses, of which any one is returned.

    Raises
    ------
    ValueError
        If the volumes are not cubic or differ in shape.

    Warns
    -----
    UserWarning
        If the alignment is not `Alignment.reliable`: the volumes do not
        correlate beyond the band the search was run on, as when they are
        too noisy or show different objects. The returned pose is then the
        best found, and may be wrong.

    Notes
    -----
    Volumes of opposite hand are not recognised: the search covers proper
    rotations only.

    On five rendered assemblies (270 to 378 voxels at 0.75 A) moved to random
    poses, the pose onto the unmoved map is recovered to 1e-4 degrees and
    1e-4 voxel without noise, and to 0.012 degrees and 0.01 voxel with
    independent white noise of ten times the signal variance in both maps,
    in 2 to 8 s on one GPU.

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
    if reference.shape != volume.shape:
        raise ValueError(
            f"the reference has shape {tuple(reference.shape)}, the volume "
            f"{tuple(volume.shape)}; resample one onto the other's grid first"
        )
    dev = volume.device
    max_radius = widest_band(n_full) if max_radius is None else max_radius
    max_shift = n_full / 2 if max_shift is None else max_shift
    eye = torch.eye(3, dtype=torch.float64, device=dev)[None]

    def tick() -> float:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        return time.perf_counter()

    def distance(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(frame_distance(a, b, eye))

    start = tick()
    spectrum = spectrum_of(volume)
    spectrum_ref = spectrum_of(reference.to(dev))
    floor = noise_power(spectrum, n_full)
    floor_ref = noise_power(spectrum_ref, n_full)

    bands: dict[float, tuple[Band, Band]] = {}

    def bands_at(radius: float) -> tuple[Band, Band]:
        if radius not in bands:
            bands[radius] = (
                Band(spectrum, n_full, radius, floor),
                Band(spectrum_ref, n_full, radius, floor_ref),
            )
        return bands[radius]

    def stage(radius: float) -> _Stage:
        return _stage(*bands_at(radius), N_POINTS)

    # 1. Rotation, from the intensities.
    coarse = stage(min(COARSE_RADIUS, max_radius))
    step = 1.0 / coarse.radius  # half an intensity speckle at the band edge
    grid = so3_grid(step).to(dev)
    reference_band = bands_at(min(COARSE_RADIUS, max_radius))[1]
    scores = _intensity_score(coarse.band, reference_band, grid, coarse.pts)
    picked: list[int] = []
    for i in scores.argsort(descending=True)[: 50 * N_CANDIDATES].tolist():
        if all(distance(grid[i], grid[j]) > 1.5 * step for j in picked):
            picked.append(i)
        if len(picked) == N_CANDIDATES:
            break
    logger.info(
        "align_volumes: %d rotations at %.1f degrees, %d points, box %d (%.2f s)",
        grid.shape[0],
        math.degrees(step),
        coarse.pts.shape[0],
        coarse.band.n,
        tick() - start,
    )

    # 2. A shift for each candidate.
    t0 = tick()
    poses = []
    for i in picked:
        c, sc = _shift_search(coarse, grid[i], max_shift)
        poses.append(Pose(sc, grid[i], c))
    poses.sort(key=lambda p: -p.score)
    logger.info(
        "align_volumes: shift search scores %s (%.2f s)",
        " ".join(f"{p.score:.3f}" for p in poses),
        tick() - t0,
    )

    # 3. Refinement out to where the two volumes stop correlating.
    best, radius = refine_while_shells_agree(
        poses[:N_REFINE],
        coarse,
        stage=stage,
        score=_correlation,
        shell_z=_shell_correlation,
        distance=distance,
        max_radius=max_radius,
        name="align_volumes",
    )

    result = Alignment(
        volume=apply_alignment(volume, rotation=best.rot, centre=best.centre)
        if resample
        else None,
        rotation=best.rot,
        centre=best.centre,
        score=best.score,
        significance=best.significance,
        radius=radius,
    )
    logger.info(
        "align_volumes: score %.4f, significance %.1f, radius %.0f (%.2f s)",
        result.score,
        result.significance,
        radius,
        tick() - start,
    )
    if not result.reliable:
        warnings.warn(
            "The volume and the reference do not correlate beyond the band "
            f"the search was run on (significance {result.significance:.1f}, "
            f"below {SIGNIFICANT:.0f}). They may be too noisy or show "
            "different objects; the returned pose may be wrong.",
            stacklevel=2,
        )
    return result
