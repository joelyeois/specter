"""
Multi-reference alignment (MRA) image-generation toolkit.

The package is self-contained (only ``torch``) apart from
:mod:`mj_dsa4288.mra.template` and :mod:`mj_dsa4288.mra.phase_b`, which call
into SPECTER to build realistic templates and physics-rich image stacks.
"""

from mj_dsa4288.mra.model import cyclic_shift, sample_mra, sigma_for_snr, snr_of
from mj_dsa4288.mra.template import normalise_template, synthetic_template

__all__ = [
    "cyclic_shift",
    "normalise_template",
    "sample_mra",
    "sigma_for_snr",
    "snr_of",
    "synthetic_template",
]
