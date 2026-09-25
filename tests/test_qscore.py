"""Output channels of `specter.qscore.QScore`: library code logs, never prints."""

from __future__ import annotations

import logging

import pytest
import torch

from specter.qscore import QScore


def test_qscore_reports_through_the_logger_not_stdout(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    q = torch.tensor([0.2, 0.4, 0.6, 0.8])
    with caplog.at_level(logging.INFO, logger="specter"):
        qs = QScore()
        stats = qs.summarize(q, label="test", plot=False)
    assert capsys.readouterr().out == ""
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["frac_above_0.5"] == pytest.approx(0.5)
    levels = {r.getMessage().split(":")[0]: r.levelno for r in caplog.records}
    assert levels["QScore"] == logging.WARNING  # the alignment reminder
    assert any("Q-score summary" in r.getMessage() for r in caplog.records)
