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
        pixel_size=1.0, noise_model="poisson", n_frames=4, progressbars=False
    )
    img = torch.full((64, 64), 5.0)

    torch.manual_seed(0)
    out = detector.apply_coincidence(img.clone(), dose=1.0, coincidence_radius=0.0)

    # Poisson-distributed about the input, so no systematic loss.
    assert out.mean().item() == pytest.approx(img.mean().item(), rel=0.05)


def test_positive_radius_all_zero_image_returns_blank_frame() -> None:
    """
    Regression test: an all-zero image with coincidence_radius > 0 used to
    compute ``intensity_map = img / img.sum()`` with no guard for
    ``img.sum() == 0``, i.e. ``0/0 = NaN`` -- which torch.poisson then
    rejected with "invalid Poisson rate, expected rate to be non-negative"
    (NaN fails the same >=0 check a genuine negative rate would). An
    all-zero image is a real, reachable case: e.g. scattering_model="ctf"
    has no vacuum baseline (unlike multislice's exp(i*sigma*dz*V) == 1 at
    V=0), so an entirely empty specimen volume (e.g. a rare zero-particle-
    placement draw from crowding) produces an exactly-zero image. The
    physically correct output for zero expected signal is a blank frame,
    not an exception.
    """
    detector = Detector(
        pixel_size=1.0, noise_model="poisson", n_frames=4, progressbars=False
    )
    img = torch.zeros((64, 64))

    out = detector.apply_coincidence(img.clone(), dose=1.0, coincidence_radius=0.7181)

    assert torch.isfinite(out).all()
    assert (out == 0).all()


def test_larger_radius_loses_more_electrons() -> None:
    """Monotonicity: a bigger exclusion disc must suppress strictly more."""
    n0 = N0
    survivals = [
        _grid_survival(n0, r, SIZE, range(N_SEEDS)) for r in (0.5, 1.0, 2.0, 3.0)
    ]
    assert survivals == sorted(survivals, reverse=True), survivals


# ---------------------------------------------------------------------------
# Detector response: MTF (blur) vs DQE(0) (counting efficiency)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset",
    [
        "k3_200kv",
        "k3_300kv",
        "k2_300kv",
        "falcon4i_200kv",
        "falcon4i_300kv",
        "perfect_detector",
    ],
)
def test_mtf_is_normalised_at_dc(preset: str) -> None:
    """
    Every bundled MTF must be 1 at zero frequency.

    An MTF describes blur, which redistributes signal without destroying it.
    Detection efficiency is a separate effect carried by ``dqe0``. The Falcon
    presets are derived from published DQE and previously had DC = sqrt(DQE(0))
    ~ 0.96, which silently scaled counts through what should be a pure filter.
    """
    from specter import detectors

    mtf = getattr(detectors, preset)(n=128, dx=1.0, device="cpu")
    assert float(mtf.flatten()[0]) == pytest.approx(1.0, abs=1e-5)
    assert float(mtf.max()) == pytest.approx(1.0, abs=1e-5)


def test_mtf_conserves_total_counts() -> None:
    """Applying an MTF must not change the number of electrons, only where."""
    from specter.detectors import falcon4i_300kv
    from specter.fft import fft2, ifft2

    mtf = falcon4i_300kv(n=256, dx=1.0, device="cpu")
    torch.manual_seed(0)
    img = torch.poisson(torch.full((256, 256), 20.0)).double()
    blurred = torch.real(ifft2(fft2(img) * mtf))

    assert blurred.sum().item() == pytest.approx(img.sum().item(), rel=1e-6)
    # ...but it must actually blur, or the test is vacuous.
    assert blurred.std().item() < img.std().item()


@pytest.mark.parametrize("preset", ["falcon4i_200kv", "falcon4i_300kv"])
def test_falcon4i_return1d_matches_2d_radial_profile(preset: str) -> None:
    """``return1d=True``'s k_data must run from 0 (DC) up to ~Nyquist, and
    its mtf values must match the 2D MTF sampled along the same radial cut.

    Regression test: the slice used to previously be ``k_rad[n // 2:, n //
    2]``, which -- since k is native/unshifted FFT order (DC at index 0,
    not index n // 2) -- selected frequencies >= Nyquist only, the wrong
    half of the array entirely.
    """
    from specter import detectors

    n, dx = 128, 1.0
    fn = getattr(detectors, preset)
    k_1d, mtf_1d = fn(n=n, dx=dx, device="cpu", return1d=True)
    mtf_2d = fn(n=n, dx=dx, device="cpu", return1d=False)

    k_nyquist = 1 / (2 * dx)
    assert float(k_1d[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(k_1d.max()) < k_nyquist
    assert torch.all(torch.diff(k_1d) > 0)  # monotonically increasing
    assert torch.allclose(mtf_1d, mtf_2d[: n // 2, 0], atol=1e-5)


def test_dqe0_scales_detected_counts() -> None:
    """
    dqe0 must scale recorded electrons by exactly that factor, and must apply
    with noise disabled -- it is a property of detection, not of noise.
    """
    exitwave = torch.ones(1, 64, 64, dtype=torch.complex64)
    dose = torch.tensor([10.0])
    cr = torch.zeros(1)

    ideal = Detector(pixel_size=1.0).forward(exitwave, dose, cr)
    lossy = Detector(pixel_size=1.0, dqe0=0.92).forward(exitwave, dose, cr)

    assert lossy.sum().item() == pytest.approx(0.92 * ideal.sum().item(), rel=1e-6)


def test_dqe0_rejects_out_of_range() -> None:
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="dqe0"):
            Detector(pixel_size=1.0, dqe0=bad)


def test_named_detector_carries_its_dqe0() -> None:
    """The DQE(0) and MTF halves must travel together with the detector name."""
    from specter.detectors import dqe0_for_detector

    assert dqe0_for_detector("falcon4i_300kv") == 0.92
    assert dqe0_for_detector(None) == 1.0
    assert dqe0_for_detector("not_a_detector") == 1.0
