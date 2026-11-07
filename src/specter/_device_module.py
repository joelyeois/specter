"""
`DeviceModule`, an `nn.Module` that knows which device it was moved to.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


class DeviceModule(nn.Module):
    """
    An ``nn.Module`` with a ``device`` property.

    The device is the one the module was last moved to (``to``, ``cuda``,
    ``cpu``, or a parent's move reaching it as a child), starting on the CPU.
    It is recorded rather than read off the buffers, since several of the
    forward model's modules hold none, and a ``.to("cuda")`` resolves to the
    current CUDA device at the time of the move.
    """

    def __init__(self) -> None:
        super().__init__()
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        """The device this module was last moved to."""
        return self._device

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> Any:
        # Every move (`to`, `cuda`, `cpu`, `half`, ...) maps its tensors
        # through `fn`; mapping an empty probe the same way says where they went.
        self._device = fn(torch.empty(0, device=self._device)).device
        return super()._apply(fn, recurse)
