"""
Global seed control, exported as `specter.seed`, and the per-unit seeding
(`derive_seed`, `seeded_rng`) that `specter.imaging` builds on.
"""

from __future__ import annotations

import random
from types import TracebackType
from typing import TYPE_CHECKING, ContextManager

if TYPE_CHECKING:
    import torch


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility across multiple libraries.

    Sets the random seed for Python's built-in random module, NumPy, PyTorch
    (both CPU and CUDA), ensuring deterministic behavior across all random
    number generators used in the codebase.

    Parameters
    ----------
    seed : int
        Random seed value to be set across all random number generators.

    Notes
    -----
    This function sets seeds for:
    - Python's built-in `random` module
    - NumPy's random number generator
    - PyTorch's CPU random number generator
    - PyTorch's CUDA random number generator (all GPUs)

    For complete reproducibility in PyTorch, you may also need to set
    `torch.backends.cudnn.deterministic = True` and
    `torch.backends.cudnn.benchmark = False`.
    """
    # Imported here rather than at module level: `specter/__init__.py`
    # re-exports this function, so a top-level torch import would be paid by
    # every `import specter`, including `specter --help` (~3 s).
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def derive_seed(seed: int, *keys: int) -> int:
    """
    A seed for one unit of work, from the run's seed and the unit's indices.

    Hashes ``(seed, *keys)`` through NumPy's ``SeedSequence``, so every
    distinct key tuple gives an independent, well-mixed stream and the
    result depends on nothing else: not on how the work is batched, which
    device or process runs it, or what was drawn before it. That is what
    makes a seeded particle stack identical across batch sizes and DDP
    rank counts, where one shared stream would hand the n-th draw to
    whichever particle happened to come n-th.

    Parameters
    ----------
    seed : int
        The run's seed.
    *keys : int
        The unit's indices: a particle's global index, a stage number.

    Returns
    -------
    int
        A seed in ``[0, 2**32)``.
    """
    import numpy as np

    return int(np.random.SeedSequence(seed, spawn_key=tuple(keys)).generate_state(1)[0])


class seeded_rng:
    """
    Run a block on the global torch RNG seeded to `seed`, then restore it.

    A context manager over ``torch.random.fork_rng``: inside, the CPU default
    generator and, if `device` is a CUDA device, that device's generator are
    seeded to `seed`; on exit both are restored, so the block neither reads
    nor advances the surrounding stream. For code that draws from the global
    RNG through functions that take no ``generator`` argument (roma's
    rotation sampler, the Poisson-disk neighbour sampler, torch ops called
    without one). Code that does take a generator should be handed one from
    :func:`derive_seed` instead, which costs no state save/restore.

    Parameters
    ----------
    seed : int
        The seed for the block.
    device : torch.device or str or None
        A CUDA device whose generator the block draws from, or None (or a
        CPU device) when the block draws on the CPU only; forking a CUDA
        generator's state is a device synchronisation, so it is not done
        unless asked for.
    """

    def __init__(self, seed: int, device: torch.device | str | None = None) -> None:
        self.seed = seed
        self.device = device
        self._fork: ContextManager[None] | None = None

    def __enter__(self) -> seeded_rng:
        import torch

        dev = None if self.device is None else torch.device(self.device)
        cuda_index: int | None = None
        if dev is not None and dev.type == "cuda":
            cuda_index = (
                dev.index if dev.index is not None else torch.cuda.current_device()
            )
        devices = [] if cuda_index is None else [cuda_index]
        self._fork = torch.random.fork_rng(devices=devices)
        self._fork.__enter__()
        torch.random.default_generator.manual_seed(self.seed)
        if cuda_index is not None:
            torch.cuda.default_generators[cuda_index].manual_seed(self.seed)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._fork is not None
        self._fork.__exit__(exc_type, exc, tb)
        self._fork = None
