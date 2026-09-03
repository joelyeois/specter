"""Derive a simulation config that matches a real particle set -- see `specter match particles`.

The pieces: `_images` loads the experimental particles a refinement file
refers to, in its order; `_metrics` compares a simulated stack with them at
matched poses; `_report` renders the comparison; `_toml` writes the derived
config. `specter.pipelines.run_match` drives them.
"""

from __future__ import annotations

from ._images import load_experimental_images
from ._metrics import (
    BAND_LABELS,
    BANDS,
    MatchedPoseSNR,
    PoseAlignmentResult,
    TwinResult,
    annulus_std_profile,
    band_power_ratio,
    edge_band_means,
    matched_index_correlation,
    matched_pose_snr,
    radial_power_spectra,
    twin_test,
    water_ring_excess,
)
from ._report import MatchReport, render_report
from ._toml import dumps as dumps_toml

__all__ = [
    "BANDS",
    "BAND_LABELS",
    "MatchReport",
    "MatchedPoseSNR",
    "PoseAlignmentResult",
    "TwinResult",
    "annulus_std_profile",
    "band_power_ratio",
    "dumps_toml",
    "edge_band_means",
    "load_experimental_images",
    "matched_index_correlation",
    "matched_pose_snr",
    "radial_power_spectra",
    "render_report",
    "twin_test",
    "water_ring_excess",
]
