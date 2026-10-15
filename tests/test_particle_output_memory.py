"""Particle output: bounded normalization, and a two-GPU run that keeps every share."""

import os
from pathlib import Path
import subprocess
import sys

import mrcfile

import pytest
import torch

from specter.config import ParticleStackConfig
from specter.image import normalize_particles
from specter.pipelines import _particles


@pytest.mark.parametrize("n", [1, 65, 129])
@pytest.mark.parametrize("normalize", [False, True])
def test_bounded_normalization_matches_original_stack(
    monkeypatch, tmp_path, n, normalize
):
    torch.manual_seed(3)
    images = torch.rand(n, 32, 32) + torch.arange(n)[:, None, None]
    original = images.clone()
    expected = -normalize_particles(original)[0] if normalize else original
    seen = []

    def capture(particles, **_kwargs):
        seen.append(particles.clone())

    monkeypatch.setattr(_particles, "create_particle_starfile", capture)
    _particles._save_stack(
        ParticleStackConfig(
            pdb_source="6bdf", n_particles=n, normalize_particles=normalize
        ),
        str(tmp_path),
        images,
        None,
        None,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(n, -1),
        torch.zeros(n, 2),
        {},
        1.0,
        300.0,
        0.1,
        torch.ones(n),
        torch.zeros(n),
        False,
    )
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_two_gpu_cli_preserves_uneven_diagnostic_stacks(tmp_path):
    source = Path(__file__).parent / "test_data" / "1mbo.cif"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "specter.cli._cli",
            "simulate",
            "particles",
            "--pdb_source",
            str(source),
            "--n_pixels",
            "24",
            "--n_particles",
            "5",
            "--batchsize",
            "2",
            "--device",
            "0,1",
            "--ice_model",
            "none",
            "--noise_model",
            "none",
            "--save_exitwaves",
            "true",
            "--save_clean_exitwaves",
            "true",
            "--seed",
            "3",
            "--output_dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    files = list(tmp_path.glob("*.mrcs"))
    assert len(files) == 5
    for path in files:
        with mrcfile.open(path) as mrc:
            assert mrc.data.shape == (5, 24, 24)
    assert not list(tmp_path.glob("particle_shard_*.pt"))
