"""
Put a CryoSPARC ``.cs`` file into the row order specter reads particles by.

Row ``i`` of a metadata file is slice ``i`` of its stack. A CryoSPARC restack
job does not write its stack that way -- it writes in the order it read its
inputs -- so its ``.cs`` and its ``.mrcs`` disagree, and it keeps the poses and
the image addresses in two different files of the same particle group besides.

Two things bring them back into agreement, and only one moves image data:

- **Reorder the rows.** Sort the metadata into the stack's slice order. The
  images never move; a few hundred kilobytes of metadata is rewritten. This is
  what to do on the machine that already holds the stack.
- **Reorder the images.** Write a new stack in the metadata's row order, giving
  a pair that carries no dependence on a CryoSPARC project directory, a sibling
  ``.cs``, or a permutation. Costs a full copy of the image data, so it is worth
  it only when moving a dataset elsewhere, where that copy happens anyway.

Either way the result is a single ``.cs`` holding both the poses and the image
addresses, with ``blob/idx`` equal to the row number so a reader can check the
ordering rather than trust it (see `specter.io.row_order_conflict`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mrcfile
import numpy as np

from .. import logger

if TYPE_CHECKING:
    from cryosparc.dataset import Dataset

#: Particles read and written per pass, to bound memory on a large set.
_CHUNK = 256


def write_row_ordered_csfile(
    cs_file: str | Path,
    out_file: str | Path,
    *,
    stack_out: str | Path | None = None,
    n_particles: int | None = None,
) -> Path:
    """
    Write a ``.cs`` file whose row ``i`` is slice ``i`` of its particle stack.

    Parameters
    ----------
    cs_file : str or Path
        Source CryoSPARC ``.cs``. Its images are located through ``blob/path``
        and ``blob/idx``, taken from a sibling ``.cs`` of the same particle
        group when this file carries no blob columns of its own, so run this
        where the CryoSPARC project still resolves.
    out_file : str or Path
        Where to write the reordered ``.cs``.
    stack_out : str or Path, optional
        Write a new particle stack here, in the source file's row order, and
        point the output at it. Costs a full copy of the image data, and is
        what makes the result readable on a machine the CryoSPARC project is
        not on. Unset (the default) reorders the rows instead and leaves the
        images where they are, which moves no image data but requires the rows
        to account for every slice of one stack.
    n_particles : int, optional
        Use only the first ``n_particles`` rows of the source. Default all.

    Returns
    -------
    pathlib.Path
        The ``.cs`` file written.

    Raises
    ------
    ValueError
        When ``stack_out`` is unset and the rows do not account for every slice
        of a single stack, so no reordering of the rows alone makes row ``i``
        slice ``i``.

    Examples
    --------
    Reorder a restack job's metadata in place, moving no images:

    >>> write_row_ordered_csfile(  # doctest: +SKIP
    ...     "J398/J398_passthrough_particles.cs", "j398.cs"
    ... )

    Write a self-contained pair to carry to another machine:

    >>> write_row_ordered_csfile(  # doctest: +SKIP
    ...     "J398/J398_passthrough_particles.cs", "j398.cs", stack_out="j398.mrcs"
    ... )
    """
    from cryosparc.dataset import Dataset

    from ._images import particle_image_refs, read_particle_images

    out_file = Path(out_file)
    refs = particle_image_refs(cs_file)
    if n_particles is not None:
        refs = refs[:n_particles]
    n = len(refs)
    stacks = sorted({path for path, _ in refs})

    dataset = Dataset.load(str(cs_file))
    if len(dataset) != n:
        dataset = dataset.slice(0, n)

    if stack_out is not None:
        stack_out = Path(stack_out)
        _write_reordered_stack(stack_out, refs, read_particle_images)
        _set_addresses(dataset, stack_out.name, np.arange(n, dtype=np.uint32))
        logger.info(
            "%d particles copied into %s in %s's row order",
            n,
            stack_out.name,
            Path(cs_file).name,
        )
    else:
        dataset = dataset.take(_row_order_matching_stack(refs, cs_file))
        _set_addresses(
            dataset, str(Path(stacks[0]).resolve()), np.arange(n, dtype=np.uint32)
        )
        logger.info(
            "%d rows reordered to match %s; no image data copied",
            n,
            Path(stacks[0]).name,
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(str(out_file))
    return out_file


def _row_order_matching_stack(
    refs: list[tuple[str, int]], cs_file: str | Path
) -> np.ndarray:
    """
    The permutation putting the metadata's rows into the stack's slice order.

    Only possible when the rows account for every slice of one stack exactly
    once: row ``i`` can only *be* slice ``i`` if slice ``i`` belongs to the set
    at all. A ``.cs`` holding a subset of a larger stack, or particles drawn
    from several stacks, has no such permutation, and reordering the images is
    then the only way to make row order true.
    """
    indices = np.array([idx for _, idx in refs])
    stacks = {path for path, _ in refs}
    if len(stacks) > 1 or not np.array_equal(np.sort(indices), np.arange(len(indices))):
        raise ValueError(
            f"{Path(cs_file).name}'s {len(indices)} rows do not account for every "
            f"slice of a single stack ({len(stacks)} stack(s), slices "
            f"{indices.min()}-{indices.max()}), so no reordering of the rows alone "
            "makes row i slice i. Pass stack_out to write a new stack in this "
            "file's row order instead."
        )
    return np.argsort(indices)


def _write_reordered_stack(
    stack_out: Path, refs: list[tuple[str, int]], read: object
) -> None:
    """Write the images ``refs`` names into one stack, in that order."""
    box = read([refs[0]]).shape[-1]  # type: ignore[operator]
    stack_out.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new_mmap(
        str(stack_out), shape=(len(refs), box, box), mrc_mode=2, overwrite=True
    ) as mrc:
        for lo in range(0, len(refs), _CHUNK):
            chunk = refs[lo : lo + _CHUNK]
            mrc.data[lo : lo + len(chunk)] = read(chunk).numpy()  # type: ignore[operator]


def _set_addresses(dataset: "Dataset", path: str, indices: np.ndarray) -> None:
    """Point every row's ``blob`` at ``path`` and slice ``indices``.

    Writing the addresses into the same file as the poses is half the point: a
    CryoSPARC particle group keeps them in separate ``.cs`` files, so neither
    one alone can be read.
    """
    missing = [f for f in ("blob/path", "blob/idx") if f not in dataset]
    if missing:
        dataset.add_fields(
            missing, ["O" if f == "blob/path" else "u4" for f in missing]
        )
    dataset["blob/path"] = np.array([path] * len(dataset))
    dataset["blob/idx"] = indices
