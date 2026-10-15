"""Multi-device particle generation must stitch every share back, or fail loudly.

A two-GPU `specter simulate particles` driven from a `cs_path` once came back
with half its images: the main process read the shares off disk while one was
still being written, stitched what was there, and deleted it. The metadata was
still full length, so the run died inside a CTF column of the STAR writer --
after the short `.mrcs` had already been written.

These tests cover the two halves of that: stitching rejects a missing or
incomplete set of shares instead of returning a short stack, and the STAR
writer rejects mismatched metadata before it writes anything.
"""

from __future__ import annotations

import pytest
import torch

from specter.io import create_particle_starfile
from specter.pipelines._particles import _gather_shards


def _write_shards(tmp_path, shards: list[list[int]], complex_values: bool = False):
    """One share per worker, each image tagged by its particle index."""
    paths = []
    for k, ids in enumerate(shards):
        images = torch.stack([torch.full((4, 4), float(i)) for i in ids])
        waves = torch.complex(images, -images) if complex_values else None
        path = tmp_path / f"particle_shard_{k}.pt"
        torch.save(
            {
                "ids": torch.tensor(ids),
                "images": images,
                "exitwaves": waves,
                "clean_exitwaves": None,
            },
            path,
        )
        paths.append(str(path))
    return paths


def test_stitching_restores_particle_order_and_consumes_the_shares(tmp_path) -> None:
    paths = _write_shards(tmp_path, [[3, 4, 5], [0, 1, 2]], complex_values=True)

    images, exitwaves, clean = _gather_shards(paths, 6)

    assert torch.equal(images[:, 0, 0], torch.arange(6, dtype=torch.float32))
    assert exitwaves is not None and exitwaves.dtype == torch.complex64
    assert torch.equal(exitwaves[:, 0, 0].real, torch.arange(6, dtype=torch.float32))
    assert clean is None
    assert not list(tmp_path.glob("particle_shard_*.pt"))


def test_stitching_rejects_a_share_that_was_never_written(tmp_path) -> None:
    paths = _write_shards(tmp_path, [[0, 1, 2]])
    paths.append(str(tmp_path / "particle_shard_1.pt"))

    with pytest.raises(RuntimeError, match="never written"):
        _gather_shards(paths, 6)
    # Nothing is deleted on the failure path, so the surviving share is
    # still there to diagnose.
    assert (tmp_path / "particle_shard_0.pt").exists()


def test_stitching_rejects_shares_that_do_not_cover_every_particle(tmp_path) -> None:
    paths = _write_shards(tmp_path, [[0, 1, 2], [3, 4]])

    with pytest.raises(RuntimeError, match="expected 0..5"):
        _gather_shards(paths, 6)
    assert (tmp_path / "particle_shard_0.pt").exists()


def _ctf(n: int) -> dict[str, torch.Tensor]:
    return {
        "dfu": torch.full((n,), 5000.0),
        "dfv": torch.full((n,), 5000.0),
        "dfang": torch.zeros(n),
        "cs": torch.full((n,), 2.7e7),
        "phaseshift": torch.zeros(n),
    }


@pytest.mark.parametrize(
    "metadata",
    [
        dict(
            rotations=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 8),
            translations=torch.zeros(8, 2),
            ctf_params=_ctf(8),
            dx=1.5,
            voltage=300.0,
            alpha=0.1,
        ),
        dict(ctf_params=_ctf(8)),
        dict(ctf_params={"dfu": torch.full((8,), 5000.0)}),
    ],
    ids=["full_metadata", "ctf_params_only", "dfu_only"],
)
def test_starfile_rejects_metadata_longer_than_the_image_stack(
    tmp_path, metadata: dict
) -> None:
    """Half the images with all of the metadata: the shape the stitching bug took.

    The error names the offending column, and a rejected call must not leave
    a stack on disk with no STAR beside it."""
    with pytest.raises(ValueError, match="per-particle metadata does not match") as exc:
        create_particle_starfile(
            torch.randn(4, 8, 8),
            filename="particles",
            output_dir=str(tmp_path),
            **metadata,
        )
    assert "ctf_params['dfu'] has 8" in str(exc.value)
    assert not (tmp_path / "particles.mrcs").exists()
    assert not (tmp_path / "particles.star").exists()


def test_starfile_still_accepts_scalars_for_the_constant_columns(tmp_path) -> None:
    """Scalars are broadcast, so they must not trip the length check."""
    import starfile

    create_particle_starfile(
        torch.randn(3, 8, 8),
        rotations=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3),
        translations=torch.zeros(3, 2),
        ctf_params=_ctf(3),
        dx=1.5,
        voltage=300.0,
        alpha=0.1,
        dose_per_angstrom=2.0,
        bfactor=42.0,
        filename="particles",
        output_dir=str(tmp_path),
    )
    df = starfile.read(tmp_path / "particles.star")
    df = df["particles"] if isinstance(df, dict) else df
    assert len(df) == 3
