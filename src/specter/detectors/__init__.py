"""
Direct electron detectors: the `Detector` that records an image, and the
catalogue it draws on -- the bundled MTF curves (K2, K3, Falcon 3EC,
Falcon 4i, perfect), their DQE(0), and the coincidence-loss geometry per
camera.
"""

from ._catalogue import (
    k3_200kv,
    k3_300kv,
    perfect_detector,
    k2_300kv,
    falcon3ec_300kv,
    falcon4i_300kv,
    falcon4i_200kv,
    detector_mtf,
    dqe0_for_detector,
    coincidence_occupancy,
    coincidence_radius_for_simulation,
    MTF_PRESETS,
    DQE0,
    EXCLUSION_RADIUS_PX,
    HARDWARE_FRAME_RATE_HZ,
    TYPICAL_DOSE_RATE_E_PX_S,
)
from ._detector import Detector

__all__ = [
    "Detector",
    "k3_200kv",
    "k3_300kv",
    "perfect_detector",
    "k2_300kv",
    "falcon3ec_300kv",
    "falcon4i_300kv",
    "falcon4i_200kv",
    "detector_mtf",
    "dqe0_for_detector",
    "coincidence_occupancy",
    "coincidence_radius_for_simulation",
    "MTF_PRESETS",
    "DQE0",
    "EXCLUSION_RADIUS_PX",
    "HARDWARE_FRAME_RATE_HZ",
    "TYPICAL_DOSE_RATE_E_PX_S",
]
