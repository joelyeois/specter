from __future__ import annotations

import subprocess as proc
import sys
from pathlib import Path

import pytest

_FIXTURE_PDB = str(Path(__file__).parent.parent / "specter-data" / "pdb" / "1mbo.cif")


def _run_particles_cli(output_dir: Path, n_particles: int = 2) -> proc.CompletedProcess:
    args = [
        sys.executable,
        "-m",
        "specter.cli._cli",
        "simulate",
        "particles",
        "--pdb_source",
        _FIXTURE_PDB,
        "--n_particles",
        str(n_particles),
        "--n_pixels",
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
    return proc.run(args, capture_output=True, encoding="utf-8")


def test_cli_particles_smoke(tmp_path: Path) -> None:
    result = _run_particles_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "particles.mrcs").exists()
    assert (tmp_path / "particles.star").exists()


def test_cli_particles_help_smoke() -> None:
    result = proc.run(
        [sys.executable, "-m", "specter.cli._cli", "simulate", "particles", "--help"],
        capture_output=True,
        encoding="utf-8",
    )
    assert result.returncode == 0
    assert "--pdb_source" in result.stdout
    assert "--scattering_model" in result.stdout


def test_cli_particles_n_particles_override(tmp_path: Path) -> None:
    """--n_particles overrides the loaded TOML config's value end to end."""
    result = _run_particles_cli(tmp_path, n_particles=3)
    assert result.returncode == 0, result.stderr
    import mrcfile

    with mrcfile.open(tmp_path / "particles.mrcs") as mrc:
        assert mrc.data.shape[0] == 3


def test_cli_particles_advanced_flags_reach_the_star_file(tmp_path: Path) -> None:
    """Advanced-panel flags (astigmatism, phase shift) end up in the .star file's
    per-particle CTF columns, not just accepted-and-ignored."""
    import starfile

    args = [
        sys.executable,
        "-m",
        "specter.cli._cli",
        "simulate",
        "particles",
        "--pdb_source",
        _FIXTURE_PDB,
        "--n_particles",
        "4",
        "--n_pixels",
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
        str(tmp_path),
        "--filename",
        "particles",
        "--ews_curvature_sign",
        "negative",
        "--astigmatism",
        "500,800",
        "--astigmatism_angle",
        "0,180",
        "--phaseshift",
        "0.1,0.2",
    ]
    result = proc.run(args, capture_output=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr

    df = starfile.read(tmp_path / "particles.star")
    # dfv should differ from dfu now that a nonzero astigmatism was sampled,
    # and shouldn't just be a constant 0 offset either.
    assert not (df["rlnDefocusV"] == df["rlnDefocusU"]).all()
    assert (df["rlnDefocusAngle"] != 0).any()
    assert (df["rlnPhaseShift"] != 0).any()


def test_cli_particles_falcon4i_detector_model_reachable(tmp_path: Path) -> None:
    """falcon4i_300kv should be a valid --detector_model choice, not rejected."""
    args = [
        sys.executable,
        "-m",
        "specter.cli._cli",
        "simulate",
        "particles",
        "--pdb_source",
        _FIXTURE_PDB,
        "--n_particles",
        "2",
        "--n_pixels",
        "32",
        "--scattering_model",
        "ctf",
        "--ice_model",
        "none",
        "--detector_model",
        "falcon4i_300kv",
        "--device",
        "cpu",
        "--output_dir",
        str(tmp_path),
        "--filename",
        "particles",
    ]
    result = proc.run(args, capture_output=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "particles.mrcs").exists()


def _run_micrograph_cli(
    output_dir: Path, n_micrographs: int = 1
) -> proc.CompletedProcess:
    args = [
        sys.executable,
        "-m",
        "specter.cli._cli",
        "simulate",
        "micrograph",
        "--pdb_source",
        _FIXTURE_PDB,
        "--n_micrographs",
        str(n_micrographs),
        "--n_pixels",
        "32",
        "--micrograph_size",
        "64",
        "--scattering_model",
        "ctf",
        "--ice_model",
        "none",
        "--detector_model",
        "none",
        # scattering_model="ctf" is a fast, linear (not intensity-guaranteed-
        # non-negative) approximation, unlike multislice's |exitwave|^2 --
        # combined with unseeded random defocus sampling, some draws produce
        # local negative "intensity" that crashes torch.poisson. Not a bug
        # introduced by this CLI (same MicrographGenerator/Detector path the
        # old demo-script exercised); noise isn't needed for a shape-only
        # smoke test, so disable it here rather than flaking on rare draws.
        "--noise_model",
        "none",
        "--device",
        "cpu",
        "--output_dir",
        str(output_dir),
        "--filename",
        "micrographs",
    ]
    return proc.run(args, capture_output=True, encoding="utf-8")


def test_cli_micrograph_smoke(tmp_path: Path) -> None:
    result = _run_micrograph_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "micrographs.mrcs").exists()
    assert (tmp_path / "micrographs.star").exists()

    import mrcfile

    with mrcfile.open(tmp_path / "micrographs.mrcs") as mrc:
        assert mrc.data.shape == (1, 64, 64)


def test_cli_micrograph_help_smoke() -> None:
    result = proc.run(
        [sys.executable, "-m", "specter.cli._cli", "simulate", "micrograph", "--help"],
        capture_output=True,
        encoding="utf-8",
    )
    assert result.returncode == 0
    assert "--pdb_source" in result.stdout
    assert "--micrograph_size" in result.stdout


def test_cli_micrograph_n_micrographs_override(tmp_path: Path) -> None:
    """--n_micrographs overrides the loaded TOML config's value end to end."""
    result = _run_micrograph_cli(tmp_path, n_micrographs=2)
    assert result.returncode == 0, result.stderr
    import mrcfile

    with mrcfile.open(tmp_path / "micrographs.mrcs") as mrc:
        assert mrc.data.shape[0] == 2


def test_cli_tiltseries_smoke(tmp_path: Path) -> None:
    """`specter simulate tiltseries --volume_path ...` loads a pre-built
    volume from disk and runs the imaging pipeline end to end."""
    import torch

    volume_path = tmp_path / "volume.pt"
    torch.save(torch.rand(32, 48, 48) * 0.01, volume_path)

    args = [
        sys.executable,
        "-m",
        "specter.cli._cli",
        "simulate",
        "tiltseries",
        "--volume_path",
        str(volume_path),
        "--voxel_size",
        "4.0",
        "--micrograph_size",
        "48",
        "--min_tilt_angle",
        "-5",
        "--max_tilt_angle",
        "5",
        "--n_tilts",
        "3",
        "--n_frames",
        "1",
        "--ice_model",
        "none",
        "--detector_model",
        "none",
        "--scattering_model",
        "ctf",
        "--device",
        "cpu",
        "--output_dir",
        str(tmp_path),
        "--filename",
        "tiltseries",
    ]
    result = proc.run(args, capture_output=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "tiltseries.mrcs").exists()
    assert (tmp_path / "tiltseries.star").exists()

    import mrcfile

    with mrcfile.open(tmp_path / "tiltseries.mrcs") as mrc:
        assert mrc.data.shape == (3, 48, 48)


def test_cli_tiltseries_help_smoke() -> None:
    result = proc.run(
        [sys.executable, "-m", "specter.cli._cli", "simulate", "tiltseries", "--help"],
        capture_output=True,
        encoding="utf-8",
    )
    assert result.returncode == 0
    assert "--volume_path" in result.stdout
    assert "--n_tilts" in result.stdout


def test_cli_tiltseries_requires_volume_path(tmp_path: Path) -> None:
    """Without --volume_path, run_tilt_series should fail with a clear error
    rather than silently falling back to some other specimen source."""
    args = [
        sys.executable,
        "-m",
        "specter.cli._cli",
        "simulate",
        "tiltseries",
        "--device",
        "cpu",
        "--output_dir",
        str(tmp_path),
    ]
    result = proc.run(args, capture_output=True, encoding="utf-8")
    assert result.returncode != 0
    assert "volume_path" in result.stderr


def test_cli_tiltseries_tomogram_config_chains_build_and_simulate(
    tmp_path: Path,
) -> None:
    """--tomogram_config should build the specimen volume first (`specter
    build tomogram`), then image it -- both the intermediate .mrc and the
    final .mrcs/.star should land on disk from one `simulate tiltseries`
    call."""
    if not _FIXTURE_PDB or not Path(_FIXTURE_PDB).exists():
        import pytest

        pytest.skip("bundled PDB fixture missing")

    tomo_dir = tmp_path / "tomo"
    tilt_dir = tmp_path / "tilt"
    tomogram_config_path = tmp_path / "tomogram.toml"
    tomogram_config_path.write_text(
        f"""
[[targets]]
pdb_source = "{_FIXTURE_PDB}"
n_copies = 1

[specimen]
target_shape = [24, 48, 48]
voxel_size = 12.0

[output]
output_dir = "{tomo_dir}"
filename = "chained_tomogram"

[picks]
write_picks = false
write_segmentation = false
"""
    )

    args = [
        sys.executable,
        "-m",
        "specter.cli._cli",
        "simulate",
        "tiltseries",
        "--tomogram_config",
        str(tomogram_config_path),
        "--voxel_size",
        "12.0",
        "--n_tilts",
        "3",
        "--n_frames",
        "1",
        "--ice_model",
        "none",
        "--noise_model",
        "none",
        "--detector_model",
        "none",
        "--scattering_model",
        "ctf",
        "--device",
        "cpu",
        "--output_dir",
        str(tilt_dir),
        "--filename",
        "chained_tilts",
    ]
    result = proc.run(args, capture_output=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert (tomo_dir / "chained_tomogram.mrc").exists()
    assert (tilt_dir / "chained_tilts.mrcs").exists()
    assert (tilt_dir / "chained_tilts.star").exists()


def test_cli_tiltseries_tomogram_config_and_volume_path_conflict(
    tmp_path: Path,
) -> None:
    """--tomogram_config and --volume_path are mutually exclusive specimen
    sources -- passing both should fail loudly rather than silently pick one."""
    import torch

    volume_path = tmp_path / "volume.pt"
    torch.save(torch.rand(16, 24, 24) * 0.01, volume_path)

    tomogram_config_path = tmp_path / "tomogram.toml"
    tomogram_config_path.write_text(
        """
[[targets]]
pdb_source = "1mbo"
n_copies = 1

[specimen]
target_shape = [24, 24, 24]
voxel_size = 12.0
"""
    )

    args = [
        sys.executable,
        "-m",
        "specter.cli._cli",
        "simulate",
        "tiltseries",
        "--tomogram_config",
        str(tomogram_config_path),
        "--volume_path",
        str(volume_path),
        "--device",
        "cpu",
        "--output_dir",
        str(tmp_path),
    ]
    result = proc.run(args, capture_output=True, encoding="utf-8")
    assert result.returncode != 0
    assert "tomogram_config" in result.stderr
    assert "volume_path" in result.stderr


def test_cli_particles_single_particle(tmp_path: Path) -> None:
    """
    n_particles=1 must work -- the most natural first thing a user tries.

    random_quaternion squeezes the batch axis at n == 1, so the pipeline used
    to hand roma a length-1 vector instead of a quaternion and crash with an
    IndexError before writing anything.
    """
    result = _run_particles_cli(tmp_path, n_particles=1)
    assert result.returncode == 0, result.stderr

    import mrcfile

    with mrcfile.open(str(tmp_path / "particles.mrcs"), permissive=True) as mrc:
        assert mrc.data.shape[0] == 1


# PotentialBuilder defaults to the Shtyrov parameterization, which is per
# bonded species -- passing it no atom_species makes every atom fall back to
# per-element Peng, silently. run_micrograph did exactly that, so the
# micrograph path got no benefit from Shtyrov at all while run_particle_stack
# did. This asserts the wiring rather than the pixels: it captures what
# PotentialBuilder is actually constructed with, then aborts the run.
def test_run_micrograph_types_atoms_for_shtyrov(monkeypatch, tmp_path: Path) -> None:
    import specter.pipelines._micrograph as micrograph_module
    from specter.config import MicrographConfig

    captured: dict = {}

    class _Sentinel(Exception):
        pass

    real_builder = micrograph_module.PotentialBuilder

    def _spy(*args, **kwargs):
        captured["atom_species"] = kwargs.get("atom_species")
        captured["n_atoms"] = len(args[2]) if len(args) > 2 else None
        raise _Sentinel

    monkeypatch.setattr(micrograph_module, "PotentialBuilder", _spy)

    config = MicrographConfig(
        pdb_source=_FIXTURE_PDB,
        n_pixels=32,
        micrograph_size=64,
        n_micrographs=1,
        ice_model="none",
        device="cpu",
        output_dir=str(tmp_path),
    )
    with pytest.raises(_Sentinel):
        micrograph_module.run_micrograph(config)

    assert captured["atom_species"] is not None, (
        "run_micrograph built a Shtyrov PotentialBuilder without atom_species, "
        "so every atom would fall back to per-element Peng"
    )
    assert len(captured["atom_species"]) == captured["n_atoms"]
    assert any(s is not None for s in captured["atom_species"])
    assert real_builder is not _spy
