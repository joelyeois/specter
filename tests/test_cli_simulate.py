from __future__ import annotations

import subprocess as proc
import sys
from pathlib import Path

_FIXTURE_PDB = str(Path(__file__).parent.parent / "pdb-data" / "1mbo.cif")


def _run_particles_cli(output_dir: Path, n_particles: int = 2) -> proc.CompletedProcess:
    args = [
        sys.executable,
        "-m",
        "specter.cli._cli",
        "simulate",
        "particles",
        "--pdb_code",
        _FIXTURE_PDB,
        "--n_particles",
        str(n_particles),
        "--num_pixels",
        "32",
        "--scattering_model",
        "ctf",
        "--ice_model",
        "none",
        "--detector_model",
        "none",
        "--device",
        "cpu",
        "--output_dir",
        str(output_dir),
        "--filename",
        "particles",
    ]
    return proc.run(args, capture_output=True, text=True)


def test_cli_particles_smoke(tmp_path: Path) -> None:
    result = _run_particles_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "particles.mrcs").exists()
    assert (tmp_path / "particles.star").exists()


def test_cli_particles_help_smoke() -> None:
    result = proc.run(
        [sys.executable, "-m", "specter.cli._cli", "simulate", "particles", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--pdb_code" in result.stdout
    assert "--scattering_model" in result.stdout


def test_cli_particles_n_particles_override(tmp_path: Path) -> None:
    """--n_particles overrides the loaded TOML config's value end to end."""
    result = _run_particles_cli(tmp_path, n_particles=3)
    assert result.returncode == 0, result.stderr
    import mrcfile

    with mrcfile.open(tmp_path / "particles.mrcs") as mrc:
        assert mrc.data.shape[0] == 3
