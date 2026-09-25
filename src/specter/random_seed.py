"""
Global seed control, exported as `specter.seed`.
"""

import random


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
