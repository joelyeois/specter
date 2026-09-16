"""Explicit exposure schedules and replayable coordinate motion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable
import torch


@dataclass(frozen=True)
class ExposureStep:
    frame: int
    start: float
    dose: float

    @property
    def midpoint(self) -> float:
        return self.start + self.dose / 2


def exposure_steps(
    frame_doses: Iterable[float], substeps: int = 1, pre_exposure: float = 0.0
) -> list[ExposureStep]:
    """Split each physical output frame into dose-weighted frozen states."""
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    if not math.isfinite(pre_exposure) or pre_exposure < 0:
        raise ValueError("pre_exposure must be finite and nonnegative")
    steps = []
    start = pre_exposure
    for frame, dose in enumerate(frame_doses):
        if not math.isfinite(dose) or dose <= 0:
            raise ValueError("frame doses must be finite and positive")
        for _ in range(substeps):
            steps.append(ExposureStep(frame, start, dose / substeps))
            start += dose / substeps
    if not steps:
        raise ValueError("at least one frame is required")
    return steps


class BrownianCoordinates:
    """Seeded motion with E[|Δr|²] = msd_per_dose * Δdose, in Å².

    This is an explicit motion approximation, not a reconstruction of the
    historical cisTEM shake rule. ``reset`` replays exactly the same schedule.
    Changing the temporal grid changes the realization; convergence studies
    must reuse a precomputed common trajectory or compare ensemble statistics.
    """

    def __init__(
        self,
        coordinates: torch.Tensor,
        msd_per_dose: float,
        seed: int,
        box_xyz: torch.Tensor | None = None,
    ):
        if not math.isfinite(msd_per_dose) or msd_per_dose < 0:
            raise ValueError("msd_per_dose must be finite and nonnegative")
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 3
            or not torch.isfinite(coordinates).all()
        ):
            raise ValueError("coordinates must be finite (N,3)")
        self.initial = coordinates.clone()
        self.msd_per_dose = msd_per_dose
        self.seed = seed
        self.box = (
            None
            if box_xyz is None
            else torch.as_tensor(
                box_xyz, device=coordinates.device, dtype=coordinates.dtype
            )
        )
        if self.box is not None and (
            self.box.shape != (3,)
            or not torch.isfinite(self.box).all()
            or (self.box <= 0).any()
        ):
            raise ValueError("box dimensions must be three finite positive lengths")
        self.reset()

    def reset(self) -> None:
        self.coordinates = self.initial.clone()
        self.exposure = 0.0
        self.generator = torch.Generator(device=self.initial.device).manual_seed(
            self.seed
        )

    def advance_to(self, exposure: float) -> torch.Tensor:
        if not math.isfinite(exposure) or exposure < self.exposure:
            raise ValueError("exposure must be finite and monotonic")
        delta = exposure - self.exposure
        if delta and self.msd_per_dose:
            self.coordinates = self.coordinates + torch.randn(
                self.coordinates.shape,
                generator=self.generator,
                device=self.coordinates.device,
                dtype=self.coordinates.dtype,
            ) * math.sqrt(self.msd_per_dose * delta / 3)
        if self.box is not None:
            self.coordinates = (
                self.coordinates + self.box / 2
            ) % self.box - self.box / 2
        self.exposure = exposure
        return self.coordinates.clone()
