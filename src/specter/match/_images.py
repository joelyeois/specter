"""Load the experimental particle images a refinement file refers to, in its order.

`extract_parameters_from_csfile` / `extract_parameters_from_starfile` take
the first ``n`` rows of a file; this module loads the first ``n`` *images*
of the same file, so the two stay index-aligned -- which every comparison
in `specter.match._metrics` depends on.
"""

from __future__ import annotations

import os
from collections import defaultdict

import mrcfile
import numpy as np
import torch


def _read_particles(refs: list[tuple[str, int]]) -> torch.Tensor:
    """Read (path, index) references, grouped by file so each stack opens once."""
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
        f"particle image file {rel!r} not found under any of {candidates}; pass "
        "--images_path with a stack in the metadata file's order instead."
    )


def _cs_image_refs(cs_path: str, n: int | None) -> list[tuple[str, int]]:
    from cryosparc.dataset import Dataset

    d = Dataset.load(cs_path)
    if "blob/path" not in d.fields():
        raise ValueError(
            f"{cs_path} carries no blob/path column (a passthrough file without "
            "images); pass --images_path with the particle stack."
        )
    paths = np.asarray(d["blob/path"]).astype(str)
    idx = np.asarray(d["blob/idx"]).astype(int)
    if n is not None:
        paths, idx = paths[:n], idx[:n]
    job_dir = os.path.dirname(os.path.abspath(cs_path))
    bases = [os.path.dirname(job_dir), job_dir, os.getcwd()]
    return [(_resolve_relative(bases, p), int(i)) for p, i in zip(paths, idx)]


def _star_image_refs(star_path: str, n: int | None) -> list[tuple[str, int]]:
    import starfile

    data = starfile.read(star_path)
    table = (
        data["particles"] if isinstance(data, dict) and "particles" in data else data
    )
    if isinstance(table, dict):
        table = list(table.values())[-1]
    names = list(table["rlnImageName"].astype(str))
    if n is not None:
        names = names[:n]
    bases = [os.path.dirname(os.path.abspath(star_path)), os.getcwd()]
    refs = []
    for name in names:
        idx_str, rel = name.split("@", 1)
        refs.append((_resolve_relative(bases, rel), int(idx_str) - 1))
    return refs


def load_experimental_images(
    metadata_path: str, n: int | None = None, images_path: str | None = None
) -> torch.Tensor:
    """
    Load the first ``n`` experimental particle images of a refinement file.

    Parameters
    ----------
    metadata_path : str
        CryoSPARC ``.cs`` or RELION ``.star``.
    n : int, optional
        How many, from the top of the file. None loads all.
    images_path : str, optional
        An ``.mrcs`` stack already in the file's order. When given it is read
        directly and the metadata's own image references are ignored.

    Returns
    -------
    torch.Tensor
        Shape (n, box, box), float32.
    """
    if images_path is not None:
        with mrcfile.mmap(images_path, permissive=True) as m:
            data = m.data if n is None else m.data[:n]
            return torch.as_tensor(np.asarray(data, dtype=np.float32).copy())
    if metadata_path.endswith(".cs"):
        refs = _cs_image_refs(metadata_path, n)
    elif metadata_path.endswith(".star"):
        refs = _star_image_refs(metadata_path, n)
    else:
        raise ValueError(f"{metadata_path}: expected a .cs or .star file")
    return _read_particles(refs)
