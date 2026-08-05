from __future__ import annotations

import subprocess as proc
import sys
from pathlib import Path

_SMALL_FIXTURE = Path(__file__).parent.parent / "pdb-data" / "1mbo.cif"


def _write_test_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
[targets]
targets = [
    {{ pdb_source = "{_SMALL_FIXTURE}", n_copies = 2 }},
]

[specimen]
target_shape = [24, 32, 32]
v_size = 10.0
filler_occupancy_fraction = 0.0
gap_angstrom = 5.0
seed = 0

[output]
output_dir = "{output_dir}"
filename = "test_tomogram"
"""
    )


def _run_build_cli(config_path: Path, *extra_args: str) -> proc.CompletedProcess:
    args = [
        sys.executable,
        "-m",
        "specter.cli._cli",
        "build",
        "tomogram",
        "--config",
        str(config_path),
        *extra_args,
    ]
    return proc.run(args, capture_output=True, text=True)


def test_cli_build_tomogram_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "tomogram.toml"
    _write_test_config(config_path, tmp_path)

    result = _run_build_cli(config_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test_tomogram.mrc").exists()

    pick_files = list(tmp_path.glob("*.ndjson"))
    assert len(pick_files) == 1
    lines = pick_files[0].read_text().strip().splitlines()
    assert len(lines) > 0
    assert '"type": "orientedPoint"' in lines[0]
    assert '"xyz_rotation_matrix"' in lines[0]


def test_cli_build_tomogram_help_smoke() -> None:
    result = proc.run(
        [sys.executable, "-m", "specter.cli._cli", "build", "tomogram", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--filler_occupancy_fraction" in result.stdout
    assert "--gap_angstrom" in result.stdout
    assert "--n_tomograms" in result.stdout


def test_cli_build_tomogram_n_tomograms(tmp_path: Path) -> None:
    """--n_tomograms 2 writes two distinct, seed-varied tomograms into their
    own numbered subdirectories instead of overwriting a single output."""
    config_path = tmp_path / "tomogram.toml"
    _write_test_config(config_path, tmp_path)

    result = _run_build_cli(config_path, "--n_tomograms", "2")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "0001" / "test_tomogram.mrc").exists()
    assert (tmp_path / "0002" / "test_tomogram.mrc").exists()
    assert not (tmp_path / "test_tomogram.mrc").exists()

    assert list((tmp_path / "0001").glob("*.ndjson"))
    assert list((tmp_path / "0002").glob("*.ndjson"))


def test_cli_build_tomogram_write_picks_override(tmp_path: Path) -> None:
    """--write_picks False overrides the loaded TOML config's default (True)
    end to end -- no .ndjson files should be written."""
    config_path = tmp_path / "tomogram.toml"
    _write_test_config(config_path, tmp_path)

    result = _run_build_cli(config_path, "--write_picks", "False")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test_tomogram.mrc").exists()
    assert list(tmp_path.glob("*.ndjson")) == []
