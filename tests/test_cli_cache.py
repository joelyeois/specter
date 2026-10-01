from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from specter.cli._cli import cli


def test_cache_info_counts_structures_and_parsed_entries_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`specter cache info` reports downloads and ``parsed/`` entries apart."""
    cache_dir = tmp_path / "pdb"
    (cache_dir / "parsed").mkdir(parents=True)
    (cache_dir / "1abc.cif").write_text("x")
    (cache_dir / "2xyz.cif").write_text("x")
    for i in range(3):
        (cache_dir / "parsed" / f"{i}.pt").write_bytes(b"y")
    monkeypatch.setenv("SPECTER_PDB_CACHE", str(cache_dir))

    result = CliRunner().invoke(cli, ["cache", "info"])

    assert result.exit_code == 0, result.output
    assert "Structures: 2" in result.output
    assert "Parsed entries: 3" in result.output
