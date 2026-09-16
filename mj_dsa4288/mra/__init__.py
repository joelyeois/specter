"""
Multi-reference alignment (MRA) toolkit for validating Perry et al. (2019).

The package is self-contained (only ``torch``/``numpy``/``matplotlib``) apart
from :mod:`mj_dsa4288.mra.template` and :mod:`mj_dsa4288.mra.phase_b`, which
call into SPECTER to build realistic templates and physics-rich image stacks.
"""

from mj_dsa4288.mra.em import em_mra
from mj_dsa4288.mra.invariants import (
    bispectrum_1d,
    mean_invariant,
    power_spectrum,
    third_moment_tensor_1d,
)
from mj_dsa4288.mra.jennrich import homojen, jennrich
from mj_dsa4288.mra.metrics import (
    align_to,
    fit_loglog_slope,
    reconstruction_snr,
    rho,
    two_regime_slopes,
)
from mj_dsa4288.mra.model import cyclic_shift, sample_mra, sigma_for_snr, snr_of
from mj_dsa4288.mra.template import normalise_template, synthetic_template

__all__ = [
    "align_to",
    "bispectrum_1d",
    "cyclic_shift",
    "em_mra",
    "fit_loglog_slope",
    "homojen",
    "jennrich",
    "mean_invariant",
    "normalise_template",
    "power_spectrum",
    "reconstruction_snr",
    "rho",
    "sample_mra",
    "sigma_for_snr",
    "snr_of",
    "synthetic_template",
    "third_moment_tensor_1d",
    "two_regime_slopes",
]
