"""Joint refinement of a rotation and a centre in bands of doubling radius."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
import torch
from scipy.optimize import minimize

from specter import logger

from ._frames import rotvec_to_rotmat

#: A second orientation scoring this fraction of the best stays in the race
#: for one more band.
_RIVAL = 0.8
#: Agreement, in standard deviations, a shell needs for the refinement to
#: move on to it.
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
        Agreement in shells the search had not used when it chose the
        orientation, in standard deviations of that agreement for noise. It
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


def refine_in_doubling_bands(
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
    Refine candidate poses up a ladder of bands and return the best.

    Every candidate is refined in the coarse band. The band limit then
    doubles until `max_radius`, and before each step the candidates are
    tested on the shells the step adds: one that shows no agreement there is
    dropped, and the ladder ends when none does. Candidates scoring within
    reach of the best are carried along, since a coarse band cannot always
    tell the true pose from a near miss.

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
        band the ladder added, measured before any refinement used it.
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
    first = True
    while first or radius < max_radius:
        t0 = tick()
        target = min(2 * radius, max_radius)
        climbing = target > radius
        # Without a band to climb to, the next one still serves to test the
        # pose on shells the search has not seen.
        current = stage(target if climbing else 2 * radius)
        if current.radius <= radius:
            break
        confirmed = []
        for p in poses:
            z = shell_z(current, p.rot, p.centre)[int(radius) + 1 :]
            if first:
                p.significance = float(z.sum() / z.numel() ** 0.5)
            if bool((z > _SHELL_SIGMA).any()):
                confirmed.append(p)
        first = False
        if not confirmed:
            logger.info("%s: no agreement beyond radius %.0f", name, radius)
            break
        poses = confirmed
        if not climbing:
            break
        last = target >= max_radius or current.radius < target
        rivals = [p for p in poses if p.score >= _RIVAL * poses[0].score]
        poses = climb(rivals, current, last)
        radius = current.radius
        report(radius, t0)
        if last:
            break
    return poses[0], radius
