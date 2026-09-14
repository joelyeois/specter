"""Guards on the residual-envelope fit in `specter.match._metrics`."""

import numpy as np
import torch

from specter.match._metrics import _residual_envelope


def _bins(n: int = 200, k_max: float = 0.5) -> torch.Tensor:
    return torch.linspace(1e-3, k_max, n)


def test_envelope_is_refused_when_the_slope_is_not_significant() -> None:
    """
    Noise alone must not produce a B-factor, and not merely on a lucky seed.

    The previous per-bin form fitted `log(s_es / clamp(s_ss, min=1e-9))` over
    every bin with a positive ratio. On EMPIAR-10254 that returned 275 Å² from
    data whose amplitude ratio was consistent with zero.

    What a refusal rule owes is a RATE, so this asserts one over many
    realisations rather than one seed. Its two predecessors both passed at
    seed 0 while leaking badly in aggregate: a flat `|t| > 2` on the log fit
    reported an envelope for 9.0% of pure-noise inputs (200 seeds), because
    with six degrees of freedom that threshold is p = 0.09, not 0.05. The
    direct exponential fit at Student's t leaks 0.5%. The bound here is loose
    enough not to be flaky and tight enough that either predecessor fails it.
    """
    kk = _bins()
    leaked = 0
    n_trials = 40
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        s_ss = torch.as_tensor(
            1000.0 + rng.normal(0, 300, len(kk)), dtype=torch.float32
        )
        s_es = torch.as_tensor(rng.normal(0, 300, len(kk)), dtype=torch.float32)
        bfac, _ = _residual_envelope(kk, s_es, s_ss)
        leaked += int(np.isfinite(bfac))
    assert leaked <= 2, f"{leaked}/{n_trials} pure-noise inputs returned a B-factor"


def test_a_clamped_denominator_cannot_dominate_the_fit() -> None:
    """
    One bin whose denominator crosses zero must not move the answer.

    This is the EMPIAR-10254 failure in miniature: `s_ss` goes negative inside
    the fit window, the old `clamp(min=1e-9)` turned that into a ratio of 2.7e11,
    and the fit on log(a) gave it ~50x the leverage of every other point.
    """
    kk = _bins()
    true_b = 120.0
    amp = np.exp(-true_b * (kk.numpy() ** 2) / 4.0)
    s_ss = torch.full((len(kk),), 1000.0)
    s_es = torch.as_tensor(1000.0 * amp, dtype=torch.float32)
    clean, _ = _residual_envelope(kk, s_es, s_ss)

    poisoned = s_ss.clone()
    poisoned[len(kk) // 2] = -50.0  # denominator crosses zero
    spiked, _ = _residual_envelope(kk, s_es, poisoned)

    assert abs(clean - true_b) < 15.0
    assert abs(spiked - clean) < 15.0


def test_a_real_envelope_is_still_recovered() -> None:
    """A genuine B-factor must survive the guards."""
    kk = _bins()
    true_b = 150.0
    s_ss = torch.full((len(kk),), 5000.0)
    s_es = torch.as_tensor(
        5000.0 * np.exp(-true_b * (kk.numpy() ** 2) / 4.0), dtype=torch.float32
    )
    bfac, se = _residual_envelope(kk, s_es, s_ss)
    assert abs(bfac - true_b) < 15.0
    assert np.isfinite(se)
