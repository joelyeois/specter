"""
Physics checks for the detector coincidence-loss model.

``Detector.apply_detector_physics`` suppresses coincident electron arrivals
using a randomized square-cell grid, which is an *approximation* to the
physically-stated rule ("an electron is lost if it lands within radius r of one
already recorded"). These tests pin down the two properties that make
``coincidence_radius`` physically meaningful:

1. its effective exclusion area is ``pi * r**2`` -- i.e. the parameter really
   is a radius, not a shape-specific grid index; and
2. it agrees with an exact pairwise implementation of the same rule.

The exact rule is reimplemented here rather than shipped in ``src/`` because it
is O(n^2) and has no production use; keeping it in the test preserves the
cross-check without carrying dead code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from specter.microscope import Detector

# Enough seeds to average out the +-5% per-call cell-size jitter in the grid
# rule; too few makes the measured area noisy by several percent.
N_SEEDS = 16
SIZE = 256

# Coincidence loss must be large compared to Poisson counting noise, or the
# measurement is dominated by it: at this density ~4-16% of electrons are lost
# (per radius below) versus ~0.5% counting noise over N_SEEDS frames. Much
# lower densities make these tests meaningless rather than merely imprecise.
N0 = 0.03


def _survival_to_area(survival: float, n0: float) -> float:
    """
    Invert ``survival = (1 - exp(-n0*A)) / (n0*A)`` for the effective exclusion
    area ``A``. Exact for a cell-occupancy rule at any density, unlike the
    ``survival ~= 1 - n0*A/2`` low-density approximation.
    """
    return brentq(lambda a: (1 - math.exp(-n0 * a)) / (n0 * a) - survival, 1e-6, 500.0)


def _exact_pairwise_survival(n0: float, r: float, size: int, seed: int) -> float:
    """
    Fraction of electrons surviving under the exact rule: an electron is
    discarded if it falls within `r` of any earlier surviving electron.
    """
    rng = np.random.default_rng(seed)
    n_e = rng.poisson(n0 * size * size)
    if n_e == 0:
        return 1.0
    coords = rng.random((n_e, 2)) * size

    keep = np.ones(n_e, dtype=bool)
    r_sq = r * r
    for i in range(n_e):
        if not keep[i]:
            continue
        d_sq = ((coords[i + 1 :] - coords[i]) ** 2).sum(axis=1)
        keep[i + 1 :][d_sq < r_sq] = False
    return int(keep.sum()) / n_e


def _grid_survival(n0: float, r: float, size: int, seeds: range) -> float:
    """Mean surviving fraction from the production grid rule."""
    detector = Detector(pixel_size=1.0)
    intensity = torch.ones(size, size) / (size * size)
    out = []
    for s in seeds:
        torch.manual_seed(s)
        img = detector.apply_detector_physics(intensity, 1.0, n0, coinc_radius_pixels=r)
        out.append(img.sum().item() / (n0 * size * size))
    return float(np.mean(out))


@pytest.mark.parametrize("radius", [1.0, 2.0])
def test_effective_exclusion_area_is_pi_r_squared(radius: float) -> None:
    """
    The grid rule's effective exclusion area must equal pi*r^2, which is what
    makes `coincidence_radius` interpretable as a physical radius (and lets it
    be converted to real units via the detector's pixel pitch).

    Regression guard for the rescaling: the pre-2026-08 convention indexed the
    grid cell's *side* as ``r / sqrt(2)``, giving area ``r**2 / 2`` -- which is
    off by ``2*pi`` in area, i.e. ~2.5x in radius, and would fail here.
    """
    survival = _grid_survival(N0, radius, SIZE, range(N_SEEDS))
    area_eff = _survival_to_area(survival, N0)
    assert area_eff == pytest.approx(math.pi * radius**2, rel=0.08)


@pytest.mark.parametrize("radius", [1.0, 2.0])
def test_grid_matches_exact_pairwise_rule(radius: float) -> None:
    """
    The grid approximation must reproduce the exact pairwise exclusion rule it
    stands in for. Compared on loss fraction (1 - survival), which is the
    quantity coincidence loss actually controls.
    """
    grid_loss = 1.0 - _grid_survival(N0, radius, SIZE, range(N_SEEDS))
    exact_loss = 1.0 - float(
        np.mean([_exact_pairwise_survival(N0, radius, SIZE, s) for s in range(N_SEEDS)])
    )
    assert grid_loss == pytest.approx(exact_loss, rel=0.10)


def test_zero_radius_disables_coincidence_loss() -> None:
    """r <= 0 must fall through to plain Poisson noise, losing no electrons."""
    detector = Detector(
        pixel_size=1.0, noise_model="poisson", num_frames=4, progressbars=False
    )
    img = torch.full((64, 64), 5.0)

    torch.manual_seed(0)
    out = detector.apply_coincidence(img.clone(), dose=1.0, coincidence_radius=0.0)

    # Poisson-distributed about the input, so no systematic loss.
    assert out.mean().item() == pytest.approx(img.mean().item(), rel=0.05)


def test_larger_radius_loses_more_electrons() -> None:
    """Monotonicity: a bigger exclusion disc must suppress strictly more."""
    n0 = N0
    survivals = [
        _grid_survival(n0, r, SIZE, range(N_SEEDS)) for r in (0.5, 1.0, 2.0, 3.0)
    ]
    assert survivals == sorted(survivals, reverse=True), survivals
