"""Multi-GPU particle generation must reassemble every rank's share, or fail loudly.

A two-GPU `specter simulate particles` driven from a `cs_path` came back with
half its images: rank 0 globbed the run directory for `predictions_*.pt` while
rank 1 was still saving its multi-gigabyte tensor, reassembled only its own
share, and deleted it. The metadata was still full length, so the run died
inside a CTF column of the STAR writer -- after the short `.mrcs` had already
been written.

These tests cover the two halves of that: the reassembly rejects an incomplete
set of rank files instead of returning a short stack, and the STAR writer
rejects mismatched metadata before it writes anything.
"""

from __future__ import annotations

import pytest
import torch

from specter.io import create_particle_starfile
from specter.pipelines._common import _reassemble_rank_files


def _write_rank_files(tmp_path, shards: dict[int, list[int]], n_pixels: int = 4):
    """Write one predictions/batch_indices pair per rank, images tagged by index."""
    for rank, indices in shards.items():
        images = torch.stack(
            [torch.full((n_pixels, n_pixels), float(i)) for i in indices]
        )
        torch.save(images, tmp_path / f"predictions_{rank}.pt")
        torch.save(torch.tensor(indices), tmp_path / f"batch_indices_{rank}.pt")


def test_reassembly_restores_the_original_particle_order(tmp_path) -> None:
    # DDP hands rank r every world_size-th particle, so each rank's own share
    # is in order but the concatenation of the two is not.
    _write_rank_files(tmp_path, {0: [0, 2, 4, 6], 1: [1, 3, 5, 7]})

    images, sort_order = _reassemble_rank_files(str(tmp_path), n=8, world_size=2)

    assert images.shape[0] == 8
    # Each image carries its own index as its pixel value.
    assert torch.equal(images[:, 0, 0], torch.arange(8, dtype=torch.float32))
    assert sort_order.shape == (8,)
    # The rank files are consumed, not left behind for the next run to find.
    assert not list(tmp_path.glob("predictions_*.pt"))
    assert not list(tmp_path.glob("batch_indices_*.pt"))


def test_reassembly_rejects_a_rank_that_has_not_written_yet(tmp_path) -> None:
    """The actual failure: rank 1's file is not on disk when rank 0 looks."""
    _write_rank_files(tmp_path, {0: [0, 2, 4, 6]})

    with pytest.raises(RuntimeError, match="expected 2 rank file"):
        _reassemble_rank_files(str(tmp_path), n=8, world_size=2)

    # Nothing is deleted on the failure path, so the surviving rank's work
    # is still there to diagnose.
    assert list(tmp_path.glob("predictions_*.pt"))


def test_reassembly_rejects_ranks_that_do_not_cover_every_particle(tmp_path) -> None:
    _write_rank_files(tmp_path, {0: [0, 2, 4, 6], 1: [1, 3, 5]})

    with pytest.raises(RuntimeError, match="did not between them produce"):
        _reassemble_rank_files(str(tmp_path), n=8, world_size=2)


def _ctf(n: int) -> dict[str, torch.Tensor]:
    return {
        "dfu": torch.full((n,), 5000.0),
        "dfv": torch.full((n,), 5000.0),
        "dfang": torch.zeros(n),
        "cs": torch.full((n,), 2.7e7),
        "phaseshift": torch.zeros(n),
    }


def test_starfile_rejects_metadata_longer_than_the_image_stack(tmp_path) -> None:
    """Half the images with all of the metadata: the shape the DDP bug took."""
    with pytest.raises(ValueError, match="per-particle metadata does not match"):
        create_particle_starfile(
            torch.randn(4, 8, 8),
            rotations=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 8),
            translations=torch.zeros(8, 2),
            ctf_params=_ctf(8),
            dx=1.5,
            voltage=300.0,
            alpha=0.1,
            filename="particles",
            output_dir=str(tmp_path),
        )


def test_starfile_writes_nothing_when_it_rejects_the_metadata(tmp_path) -> None:
    """A rejected call must not leave a stack on disk with no STAR beside it."""
    with pytest.raises(ValueError):
        create_particle_starfile(
            torch.randn(4, 8, 8),
            ctf_params=_ctf(8),
            filename="particles",
            output_dir=str(tmp_path),
        )
    assert not (tmp_path / "particles.mrcs").exists()
    assert not (tmp_path / "particles.star").exists()


def test_starfile_names_the_offending_column(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"ctf_params\['dfu'\] has 8"):
        create_particle_starfile(
            torch.randn(4, 8, 8),
            ctf_params={"dfu": torch.full((8,), 5000.0)},
            filename="particles",
            output_dir=str(tmp_path),
        )


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
        potential_scale=1.0,
        bfactor=42.0,
        filename="particles",
        output_dir=str(tmp_path),
    )
    df = starfile.read(tmp_path / "particles.star")
    df = df["particles"] if isinstance(df, dict) else df
    assert len(df) == 3
