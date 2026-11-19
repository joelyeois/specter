"""Joint refinement of a rotation and a centre out to the band the data support."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
import torch
from scipy.optimize import minimize

from specter import logger

from ._band import widest_band
from ._frames import rotvec_to_rotmat

#: A second orientation scoring this fraction of the best stays in the race
#: for one more band.
_RIVAL = 0.8
#: Shells tested together. One shell of a noisy map seldom shows agreement
#: on its own; the sum over a window does.
_WINDOW = 8
#: Combined agreement, in standard deviations, a window of shells needs for
#: the refinement to take it in.
_SHELL_SIGMA = 3.0
#: `Alignment.significance` above which an alignment is `reliable`.
SIGNIFICANT = 4.0


@dataclass
class Alignment:
    """
    A rigid pose that brings a volume into a target frame.

    `align_to_symmetry` returns the frame in which the volume's symmetry axes
    take their standard positions, `align_volumes` the frame of the reference
    volume.

    Attributes
    ----------
    volume : torch.Tensor or None
        The aligned volume, ``input`` sampled at
        ``rotation @ (x - c0) + c0 + centre`` with ``c0`` the central voxel
        ``n // 2``. None when resampling was not requested.
    rotation : torch.Tensor
        Rotation (3, 3), float64, taking aligned-frame coordinates (x, y, z)
        to input-frame coordinates. Its columns are the aligned x, y and z
        axes in the input frame.
    centre : torch.Tensor
        The point of the input volume that becomes the central voxel of the
        aligned one, (3,), float64, in voxels (x, y, z) from the input's
        central voxel. For a symmetry alignment it is the symmetry centre,
        and for a cyclic group the point of the axis nearest the central
        voxel.
    score : float
        Correlation over the band used, between the symmetry copies of the
        whitened transform or between the two whitened volumes: 1 for exact
        agreement and 0 for none. Noise lowers it.
    significance : float
        Agreement in the first shells beyond the band the search chose the
        orientation in, in standard deviations of that agreement for noise. It
        tests the pose against noise only: a clean volume of another
        symmetry aligns to its best partial match with a high significance
        and a low `score`. NaN when the volume is too small to have such
        shells.
    radius : float
        Band limit the refinement reached, in Fourier voxels.
    """

    volume: torch.Tensor | None
    rotation: torch.Tensor
    centre: torch.Tensor
    score: float
    significance: float
    radius: float

    @property
    def reliable(self) -> bool:
        """Whether `significance` exceeds 4 standard deviations."""
        return self.significance > SIGNIFICANT


@dataclass
class Pose:
    """A candidate rotation (3, 3) and centre (3,) with its score."""

    score: float
    rot: torch.Tensor
    centre: torch.Tensor
    significance: float = float("nan")


class _Stage(Protocol):
    @property
    def radius(self) -> float: ...

    @property
    def n_full(self) -> int: ...


S = TypeVar("S", bound=_Stage)


def refine(
    score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pose: Pose,
    *,
    n_full: int,
    max_iter: int,
    gtol: float,
) -> Pose:
    """
    Maximise a score over one rotation and centre by BFGS.

    Parameters
    ----------
    score : callable
        Maps rotations (1, 3, 3) and centres (1, 3) to scores (1,),
        differentiably.
    pose : Pose
        Starting point.
    n_full : int
        Box size of the volume. Rotation steps are scaled by ``n_full / 4``,
        the voxels they move a point at a typical radius, so that the six
        parameters are commensurate.
    max_iter : int
        Largest number of BFGS iterations.
    gtol : float
        Gradient norm at which BFGS stops.

    Returns
    -------
    Pose
        The refined pose and its score.
    """
    lever = n_full / 4.0
    rot, centre = pose.rot, pose.centre

    def unpack(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return rot @ rotvec_to_rotmat(x[:3] / lever), centre + x[3:]

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        xt = torch.tensor(x, dtype=torch.float64, device=rot.device, requires_grad=True)
        r, c = unpack(xt)
        loss = -score(r[None], c[None])[0]
        (grad,) = torch.autograd.grad(loss, xt)
        return float(loss), grad.cpu().numpy()

    res = minimize(
        fun,
        np.zeros(6),
        jac=True,
        method="BFGS",
        options={"gtol": gtol, "maxiter": max_iter},
    )
    r, c = unpack(torch.tensor(res.x, dtype=torch.float64, device=rot.device))
    return Pose(float(-res.fun), r, c, pose.significance)


def _reach(z: torch.Tensor, radius: float) -> int:
    """
    Highest shell up to which a pose is supported beyond a band limit.

    Parameters
    ----------
    z : torch.Tensor
        Agreement in each integer shell, in standard deviations.
    radius : float
        Band limit the pose was refined in.

    Returns
    -------
    int
        Last shell of the unbroken run of windows of `_WINDOW` shells, from
        the first shell beyond `radius`, whose combined agreement exceeds
        `_SHELL_SIGMA`. ``int(radius)`` when the first window already fails.
    """
    reach = int(radius)
    while reach + 1 < z.numel():
        window = z[reach + 1 : reach + 1 + _WINDOW]
        if float(window.sum()) <= _SHELL_SIGMA * window.numel() ** 0.5:
            break
        reach += window.numel()
    return reach


def refine_while_shells_agree(
    poses: list[Pose],
    coarse: S,
    *,
    stage: Callable[[float], S],
    score: Callable[[S, torch.Tensor, torch.Tensor], torch.Tensor],
    shell_z: Callable[[S, torch.Tensor, torch.Tensor], torch.Tensor],
    distance: Callable[[torch.Tensor, torch.Tensor], float],
    max_radius: float,
    name: str,
) -> tuple[Pose, float]:
    """
    Refine candidate poses out to the band limit the data support.

    Every candidate is refined in the coarse band and then tested on the
    shells beyond it, none of which the search has used. A pose that is
    right to within the coarse band's precision keeps its copies in phase
    far beyond that band, so the run of shells that still agree marks both
    the band the pose can be refined in directly and the band in which
    there is anything to refine against. The candidates are refined in that
    band and tested again, until the run stops growing. A candidate whose
    first window of shells shows no agreement is dropped; candidates scoring
    within reach of the best are carried along, since a coarse band cannot
    always tell the true pose from a near miss.

    Parameters
    ----------
    poses : list of Pose
        Candidates from the grid searches, in any order.
    coarse : S
        The stage the grid searches ran on.
    stage : callable
        Builds the stage for a band limit in Fourier voxels. A stage carries
        the `radius` in force, which the volume's Nyquist frequency may have
        lowered, and the box size `n_full`.
    score : callable
        ``score(stage, rot, centre)`` for rotations (B, 3, 3) and centres
        (B, 3), differentiable, returning (B,).
    shell_z : callable
        ``shell_z(stage, rot, centre)`` for one pose: the agreement in each
        integer shell of the stage, in standard deviations.
    distance : callable
        Angle in radians between two rotations, up to equivalent poses.
    max_radius : float
        Highest band limit, in Fourier voxels.
    name : str
        Prefix of the log lines.

    Returns
    -------
    Pose
        The best pose. Its `significance` is the agreement over the first
        window of shells beyond the coarse band, measured before any
        refinement used them.
    float
        The band limit reached.
    """
    dev = poses[0].rot.device
    step = 1.0 / coarse.radius

    def tick() -> float:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        return time.perf_counter()

    def climb(entries: list[Pose], current: S, last: bool) -> list[Pose]:
        refined = [
            refine(
                lambda r, c: score(current, r, c),
                p,
                n_full=current.n_full,
                max_iter=60 if last else 30,
                gtol=1e-7 if last else 1e-4,
            )
            for p in entries
        ]
        refined.sort(key=lambda p: -p.score)
        kept: list[Pose] = []
        for p in refined:
            if all(distance(p.rot, k.rot) > 0.5 * step for k in kept):
                kept.append(p)
        return kept

    def report(radius: float, t0: float) -> None:
        logger.info(
            "%s: radius %.0f scores %s (%.2f s)",
            name,
            radius,
            " ".join(f"{p.score:.4f}" for p in poses),
            tick() - t0,
        )

    t0 = tick()
    poses = climb(poses, coarse, False)
    radius = coarse.radius
    report(radius, t0)
    # A band limit at or below the coarse one leaves nothing to climb to, but
    # the first window beyond it still serves to test the pose. The shells
    # are tested in a band scored in full: a wider one is scored at a sample
    # of its frequencies, few of which fall in the inner shells.
    full = widest_band(coarse.n_full)
    look = stage(min(max(max_radius, radius + _WINDOW), max(full, radius + _WINDOW)))
    limit = max_radius if max_radius > look.radius else min(max_radius, look.radius)
    current, polished = coarse, False
    first = True
    while True:
        t0 = tick()
        confirmed, target = [], radius
        for p in poses:
            z = shell_z(look, p.rot, p.centre)
            if first:
                held_out = z[int(radius) + 1 : int(radius) + 1 + _WINDOW]
                p.significance = float(held_out.sum() / held_out.numel() ** 0.5)
            reach = _reach(z, radius)
            if reach > radius:
                confirmed.append(p)
                target = max(target, float(reach))
        first = False
        if not confirmed:
            logger.info("%s: no agreement beyond radius %.0f", name, radius)
            break
        poses = confirmed
        target = min(target, limit)
        if target <= radius:
            break
        current = stage(target)
        if current.radius <= radius:
            break
        if current.radius >= look.radius and limit > look.radius:
            look = stage(limit)
            limit = look.radius
        polished = current.radius >= limit
        rivals = [p for p in poses if p.score >= _RIVAL * poses[0].score]
        poses = climb(rivals, current, polished)
        radius = current.radius
        report(radius, t0)
        if polished:
            break
    if not polished and current is not coarse:
        t0 = tick()
        poses = climb(poses[:1], current, True)
        report(radius, t0)
    return poses[0], radius
