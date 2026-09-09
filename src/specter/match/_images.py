"""Load the experimental particle images a refinement file refers to, in its order.

`extract_parameters_from_csfile` / `extract_parameters_from_starfile` take
the first ``n`` rows of a file; this module loads the first ``n`` *images*
of the same file, so the two stay index-aligned -- which every comparison
in `specter.match._metrics` depends on. Addressing is `specter.io`'s job
(see `io/_images.py`); this is the "first n rows" front end onto it.
"""

from __future__ import annotations

import mrcfile
import numpy as np
import torch

from ..io import particle_image_refs, read_particle_images


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
        directly and the metadata's own image references are ignored, which
        is the escape hatch for a stack repacked by hand.

    Returns
    -------
    torch.Tensor
        Shape (n, box, box), float32.
    """
    if images_path is not None:
        with mrcfile.mmap(images_path, permissive=True) as m:
            data = m.data if n is None else m.data[:n]
            return torch.as_tensor(np.asarray(data, dtype=np.float32).copy())
    refs = particle_image_refs(metadata_path)
    return read_particle_images(refs if n is None else refs[:n])
