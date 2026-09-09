"""
Where a particle's image lives, and how to read it.

A particle stack pairs with its metadata by row: row ``i`` of the file is
slice ``i`` of the stack. That is the contract specter reads on, because it
is the one that survives a dataset being copied between machines, where a
CryoSPARC project's own layout does not.

CryoSPARC does not write its stacks that way. It records where each row's
image really is in ``blob/path`` + ``blob/idx`` (RELION spells the same
thing ``idx@file`` in ``rlnImageName``), and a Restack Particles job writes
its output in the order it reads its inputs rather than the order of the
rows it emits. So those columns are read here for two purposes: to *check* a
stack really is in row order before trusting it (`row_order_conflict`), and
to resolve images in place when asked (`particle_image_refs`), which is what
`specter export particles` uses to write a row-ordered stack in the first
place.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import torch

#: One particle's image: the stack file, and its slice of it.
ImageRef = tuple[str, int]


def read_particle_images(refs: list[ImageRef]) -> torch.Tensor:
    """
    Read the images ``refs`` names, in the order given.

    Parameters
    ----------
    refs : list of (str, int)
        ``(stack path, index)`` per particle. Grouped internally so each
        stack is opened once, however the references interleave.

    Returns
    -------
    torch.Tensor
        Shape ``(len(refs), box, box)``, float32.
    """
    by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for out_idx, (path, idx) in enumerate(refs):
        by_file[path].append((idx, out_idx))
    images: list[torch.Tensor | None] = [None] * len(refs)
    for path, entries in by_file.items():
        with mrcfile.mmap(path, permissive=True) as m:
            data = m.data
            for idx, out_idx in entries:
                img = data[idx] if data.ndim == 3 else data
                images[out_idx] = torch.as_tensor(
                    np.asarray(img, dtype=np.float32).copy()
                )
    return torch.stack([img for img in images if img is not None])


def _resolve_relative(candidates: list[str], rel: str) -> str:
    for base in candidates:
        path = os.path.join(base, rel)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"particle image file {rel!r} not found under any of {candidates}. The "
        "metadata file's paths are relative to the CryoSPARC project directory, "
        "which a file copied off that machine no longer sits in -- name the "
        "stack explicitly instead."
    )


def _cs_image_refs(cs_path: str, stack_path: str | None) -> list[ImageRef]:
    """Image references of a ``.cs`` file, in its row order."""
    from ._cryosparc import particle_stack_references

    references = particle_stack_references(cs_path)
    if references is None:
        raise ValueError(
            f"{Path(cs_path).name} carries no blob/path + blob/idx columns, and no "
            ".cs file beside it holds the same particles, so nothing says which "
            "slice of a stack each row refers to. A passthrough file never carries "
            "them on its own: a job that rewrites its images keeps the addresses in "
            "its other .cs output (restacked_particles.cs for a restack job), which "
            "has to travel alongside. Failing that, name the stack explicitly and "
            "assert that its slices follow this file's rows -- true of a stack you "
            "repacked yourself, false of a CryoSPARC restack."
        )
    paths, indices = references

    if stack_path is not None:
        stacks = sorted(set(paths.tolist()))
        if len(stacks) > 1:
            raise ValueError(
                f"{Path(cs_path).name} draws its particles from {len(stacks)} stack "
                f"files ({', '.join(stacks[:3])}, ...), which one stack path cannot "
                "stand in for. Leave it unset to read them where the .cs file says "
                "they are, or restack them into a single file first."
            )
        return [(stack_path, int(i)) for i in indices]

    job_dir = os.path.dirname(os.path.abspath(cs_path))
    bases = [os.path.dirname(job_dir), job_dir, os.getcwd()]
    resolved = {p: _resolve_relative(bases, p) for p in set(paths.tolist())}
    return [(resolved[p], int(i)) for p, i in zip(paths, indices)]


def _star_image_refs(star_path: str, stack_path: str | None) -> list[ImageRef]:
    """Image references of a ``.star`` file, in its row order."""
    import starfile

    data = starfile.read(star_path)
    table = (
        data["particles"] if isinstance(data, dict) and "particles" in data else data
    )
    if isinstance(table, dict):
        table = list(table.values())[-1]
    bases = [os.path.dirname(os.path.abspath(star_path)), os.getcwd()]
    refs: list[ImageRef] = []
    for name in table["rlnImageName"].astype(str):
        idx_str, rel = name.split("@", 1)
        # RELION indexes from 1
        path = stack_path if stack_path is not None else _resolve_relative(bases, rel)
        refs.append((path, int(idx_str) - 1))
    return refs


def particle_image_refs(
    metadata_path: str | Path, stack_path: str | Path | None = None
) -> list[ImageRef]:
    """
    Locate every particle's image, in the metadata file's row order.

    Parameters
    ----------
    metadata_path : str or Path
        CryoSPARC ``.cs`` or RELION ``.star``.
    stack_path : str or Path, optional
        A stack to read every particle from, instead of the files the
        metadata names. This overrides *where* the images are, never *which
        slice* each row refers to: copying a stack between machines breaks
        its path but not its internal order, so the metadata stays the
        authority on ordering either way. Rejected when the metadata spans
        several stacks, which one path cannot stand in for.

    Returns
    -------
    list of (str, int)
        ``(stack path, index)`` per row.
    """
    metadata_path = str(metadata_path)
    stack = None if stack_path is None else str(stack_path)
    if metadata_path.endswith(".cs"):
        return _cs_image_refs(metadata_path, stack)
    if metadata_path.endswith(".star"):
        return _star_image_refs(metadata_path, stack)
    raise ValueError(f"{metadata_path}: expected a .cs or .star file")


def row_order_conflict(
    metadata_path: str | Path, rows: list[int]
) -> tuple[int, int, str] | None:
    """
    Check that a stack in the metadata's row order would be the right stack.

    Row ``i`` pairing with slice ``i`` is an assumption, and CryoSPARC records
    enough to test it: when ``blob/idx`` is present it says which slice each
    row's image really is, and anything other than the identity means a stack
    read by row number pairs poses with other particles' images. That failure
    is silent -- the images are real particles and the poses are real poses,
    so the loss falls and every per-image diagnostic looks healthy -- which is
    why it is worth checking rather than trusting.

    Parameters
    ----------
    metadata_path : str or Path
        CryoSPARC ``.cs`` or RELION ``.star``.
    rows : list of int
        The rows about to be read, as indices into the metadata file.

    Returns
    -------
    tuple or None
        ``(row, slice, stack name)`` of the first row whose image is not the
        slice of the same number, or None when the rows are consistent with
        row order or when the metadata carries no address to check against.
    """
    references = _stack_references(metadata_path)
    if references is None:
        # Nothing to check against: a passthrough file separated from its
        # siblings, which is the ordinary state of a dataset copied off the
        # machine that made it. Row order is still the contract; it just
        # cannot be verified here.
        return None
    paths, indices = references
    for row in rows:
        if int(indices[row]) != row:
            return row, int(indices[row]), Path(str(paths[row])).name
    return None


def _stack_references(
    metadata_path: str | Path,
) -> tuple[Any, Any] | None:
    """``(paths, indices)`` per row, without resolving any path on disk.

    Deliberately separate from `particle_image_refs`: whether a stack is in row
    order is a question about ``blob/idx`` alone, and a path that no longer
    resolves must not be read as "unable to check". That conflation would skip
    the check on exactly the copied-off-the-machine datasets it exists for.
    """
    path = str(metadata_path)
    if path.endswith(".cs"):
        from ._cryosparc import particle_stack_references

        return particle_stack_references(path)
    if path.endswith(".star"):
        try:
            refs = _star_image_refs(path, stack_path="")
        except (ValueError, KeyError, FileNotFoundError):
            return None
        return [r[0] for r in refs], [r[1] for r in refs]
    return None
