"""Particle output storage preserves ordering, diagnostic dtypes, and normalization."""

import itertools
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
from specter.pipelines._common import _generate_single, _reassemble_rank_files


DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
    ),
]


class IndexedImages:
    def __init__(self, device):
        self.device = device

    def __call__(self, idx):
        # Non-contiguous images, unique particle values, complex diagnostics.
        image = idx.to(self.device)[:, None, None] + torch.rand(
            len(idx), 7, 5, device=self.device
        )
        self.exitwaves = torch.complex(image, -image)
        self.clean_exitwaves = 2 * self.exitwaves
        return image.transpose(-1, -2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("exitwaves,clean", itertools.product([False, True], repeat=2))
def test_single_generation_preserves_batches_and_diagnostic_values(
    device, exitwaves, clean
):
    torch.manual_seed(7)
    reference = IndexedImages(device)
    expected, waves, dry = [], [], []
    for idx in torch.arange(5).split(2):
        expected.append(reference(idx).cpu())
        waves.append(reference.exitwaves.cpu())
        dry.append(reference.clean_exitwaves.cpu())
    state = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    torch.manual_seed(7)
    actual = _generate_single(
        IndexedImages(device),
        5,
        2,
        lambda iterable, **_: iterable,
        collect_exitwaves=exitwaves,
        collect_clean_exitwaves=clean,
    )
    torch.testing.assert_close(actual[0], torch.cat(expected), rtol=0, atol=0)
    for result, chunks, enabled in (
        (actual[1], waves, exitwaves),
        (actual[2], dry, clean),
    ):
        if enabled:
            assert result.dtype == torch.complex64
            assert result.device.type == "cpu"
            torch.testing.assert_close(result, torch.cat(chunks), rtol=0, atol=0)
        else:
            assert result is None
    after_state = (
        torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    )
    assert torch.equal(state, after_state)


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
        torch.ones(n),
        False,
    )
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)


def test_uneven_rank_shards_preserve_order_and_complex_dtype(tmp_path):
    for rank, idx in enumerate(([0, 2, 4], [1, 3])):
        values = torch.tensor(idx, dtype=torch.float32)[:, None, None]
        torch.save(torch.complex(values, -values), tmp_path / f"predictions_{rank}.pt")
        torch.save(torch.tensor(idx), tmp_path / f"batch_indices_{rank}.pt")
    actual, _order = _reassemble_rank_files(str(tmp_path), 5, 2)
    assert actual.dtype == torch.complex64
    torch.testing.assert_close(
        actual[:, 0, 0].real, torch.arange(5, dtype=torch.float32)
    )


def test_rank_length_mismatch_is_rejected_before_cleanup(tmp_path):
    torch.save(torch.ones(2, 4, 4), tmp_path / "predictions_0.pt")
    torch.save(torch.arange(3), tmp_path / "batch_indices_0.pt")
    with pytest.raises(RuntimeError, match="different lengths"):
        _reassemble_rank_files(str(tmp_path), 3, 1)
    assert (tmp_path / "predictions_0.pt").exists()


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
    assert not list(tmp_path.glob("*_0.pt"))
    assert not list(tmp_path.glob("*_1.pt"))
