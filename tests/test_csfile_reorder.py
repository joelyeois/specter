"""
`write_row_ordered_csfile`: put a CryoSPARC .cs into the row order specter reads by.

A restack job's stack is not in its .cs file's row order, and there are two ways
to fix that -- reorder the rows, or reorder the images. Only the second moves
image data, so the first is the default, and these check that both land on the
same pairing.
"""

from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pytest

from specter.io import row_order_conflict, write_row_ordered_csfile

N = 6
BOX = 8
#: Row i of the source .cs points at slice ROTATION[i]: a restack job's shape.
ROTATION = np.roll(np.arange(N), -2)


def _write_source(tmp_path: Path, blob_idx: np.ndarray) -> tuple[Path, np.ndarray]:
    """A .cs whose rows point at ``blob_idx``, beside the stack it points into."""
    from cryosparc.dataset import Dataset

    rng = np.random.default_rng(0)
    images = rng.normal(size=(N, BOX, BOX)).astype(np.float32)
    job = tmp_path / "P1" / "J1"
    (job / "restack").mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(job / "restack" / "batch_0.mrc"), overwrite=True) as mrc:
        mrc.set_data(images)

    dataset = Dataset(
        allocate={
            "uid": np.arange(N, dtype=np.uint64),
            "alignments3D/pose": rng.normal(size=(N, 3)).astype(np.float32),
            "blob/path": np.array(["J1/restack/batch_0.mrc"] * N),
            "blob/idx": blob_idx.astype(np.uint32),
        }
    )
    cs_file = job / "particles.cs"
    dataset.save(str(cs_file))
    return cs_file, images


@pytest.fixture
def restacked(tmp_path: Path) -> tuple[Path, np.ndarray]:
    return _write_source(tmp_path, ROTATION)


def test_reorder_moves_rows_and_writes_no_images(
    restacked: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    """The default moves metadata, not image data: one .cs comes out, the stack
    is untouched, and row i now refers to slice i of it."""
    cs_file, _ = restacked
    out = tmp_path / "out"

    write_row_ordered_csfile(cs_file, out / "p.cs")

    assert [f.name for f in sorted(out.iterdir())] == ["p.cs"]
    written = np.load(out / "p.cs")
    assert np.array_equal(written["blob/idx"], np.arange(N))
    assert row_order_conflict(str(out / "p.cs"), list(range(N))) is None


def test_reorder_carries_each_pose_onto_its_own_particles_row(
    restacked: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    """Reordering rows must move a pose *with* its particle. Source row i held
    the pose for the particle at slice ROTATION[i]; that pose must end up on
    the row for slice ROTATION[i], which is now row ROTATION[i]."""
    cs_file, _ = restacked
    out = tmp_path / "out"

    write_row_ordered_csfile(cs_file, out / "p.cs")

    source, written = np.load(cs_file), np.load(out / "p.cs")
    for row in range(N):
        assert np.array_equal(
            written["alignments3D/pose"][ROTATION[row]],
            source["alignments3D/pose"][row],
        )
        assert written["uid"][ROTATION[row]] == source["uid"][row]


def test_stack_out_writes_a_stack_in_the_row_order_instead(
    restacked: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    """stack_out reorders the images rather than the rows, for a pair that
    reads on a machine the CryoSPARC project is not on. Poses keep their rows."""
    cs_file, images = restacked
    out = tmp_path / "out"

    write_row_ordered_csfile(cs_file, out / "p.cs", stack_out=out / "p.mrcs")

    written = np.load(out / "p.cs")
    with mrcfile.open(str(out / "p.mrcs")) as mrc:
        exported = np.asarray(mrc.data, dtype=np.float32)
    assert np.array_equal(exported, images[ROTATION])
    assert np.array_equal(written["blob/idx"], np.arange(N))
    assert set(written["blob/path"].tolist()) == {b"p.mrcs"}
    source = np.load(cs_file)
    assert np.array_equal(written["alignments3D/pose"], source["alignments3D/pose"])


def test_reorder_refuses_a_partial_stack(tmp_path: Path) -> None:
    """Row i can only *be* slice i if slice i belongs to the set at all. A .cs
    holding a subset of a larger stack has no such permutation, so say that
    writing a reordered stack is the way to make row order true."""
    cs_file, _ = _write_source(tmp_path, np.arange(N) + 10)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="stack_out"):
        write_row_ordered_csfile(cs_file, out / "p.cs")


def test_n_particles_truncates_rows_and_images_together(
    restacked: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    """n_particles takes the first N rows, images and metadata together."""
    cs_file, images = restacked
    out = tmp_path / "out"

    write_row_ordered_csfile(
        cs_file, out / "p.cs", stack_out=out / "p.mrcs", n_particles=3
    )

    with mrcfile.open(str(out / "p.mrcs")) as mrc:
        exported = np.asarray(mrc.data, dtype=np.float32)
    assert len(np.load(out / "p.cs")) == 3
    assert np.array_equal(exported, images[ROTATION[:3]])
