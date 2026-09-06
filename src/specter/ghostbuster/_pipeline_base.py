"""
What `Ghostbuster` and `TomogramGhostbuster` share once their data is loaded:
driving a Lightning `Trainer` over a reconstructor, and the binned quick run.
"""

from __future__ import annotations

from typing import Any, Sequence, TypeVar

import lightning as L
import torch

from ._run_helpers import build_trainer, resolve_device

_M = TypeVar("_M", bound=L.LightningModule)


class _GhostbusterBase:
    """
    Trainer plumbing for the two end-to-end reconstruction pipelines.

    Subclasses set ``_images`` (the observed stack, ``(N, Y, X)``),
    ``_voxel_size``, ``epochs``, ``batchsize`` and ``precision`` in their
    constructors and build their own reconstructor and loader.
    """

    _images: torch.Tensor
    _voxel_size: float
    epochs: int
    batchsize: int
    precision: str

    @staticmethod
    def _device_label(device: int | Sequence[int] | str) -> str:
        """``"GPU 0"``, ``"GPU [0, 1]"`` or ``"CPU"``, for the run banner."""
        use_gpu, ids = resolve_device(device)
        return f"GPU {ids}" if use_gpu else "CPU"

    def _fit(
        self,
        model: _M,
        loader: torch.utils.data.DataLoader[Any],
        device: int | Sequence[int] | str,
        epochs: int,
        precision: str,
        callbacks: list[Any] | None,
    ) -> _M:
        """Train ``model`` on ``loader`` for ``epochs`` and return it."""
        use_gpu, ids = resolve_device(device)
        trainer = build_trainer(use_gpu, ids, epochs, precision, callbacks)
        trainer.fit(model, loader)
        return model

    def _bin_images(self, bin_factor: int) -> tuple[torch.Tensor, float]:
        """
        The observed images average-pooled by ``bin_factor`` in each
        spatial dimension, scaled to keep their sum, with the voxel size to
        match.
        """
        pool = torch.nn.AvgPool2d(bin_factor, stride=bin_factor)
        images = pool(self._images.unsqueeze(1)).squeeze(1) * bin_factor**2
        return images, self._voxel_size * bin_factor

    @staticmethod
    def _report_test_run(model: L.LightningModule, summary: str) -> None:
        """Print the closing line of a `test_run` with the volume's range."""
        v = model.V.detach()
        print(
            f"Test run passed — {summary}  |  "
            f"V min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}"
        )
