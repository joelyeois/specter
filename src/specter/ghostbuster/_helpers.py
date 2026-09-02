from __future__ import annotations

from typing import Any, Callable

import lightning as L
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
    LRScheduler,
    OneCycleLR,
)


# ---------------------------------------------------------------------------
# Shared helpers used by both Reconstructor and TomogramReconstructor
# ---------------------------------------------------------------------------


def _build_lr_scheduler(
    scheduler: str,
    optimizer: torch.optim.Optimizer,
    lr: float,
    lr_lambda: Callable[[int], float],
    steps_per_epoch: Callable[[], int],
    total_steps: Callable[[], int],
) -> LRScheduler:
    """
    Build the volume optimiser's LR scheduler from a scheduler name.

    Parameters
    ----------
    scheduler : str
        One of "LambdaLR", "OneCycleLR", "CosineAnnealingWarmRestarts",
        "MultiplicativeLR".
    optimizer : torch.optim.Optimizer
        The volume optimiser.
    lr : float
        Learning rate, used as ``OneCycleLR``'s peak learning rate.
    lr_lambda : Callable[[int], float]
        Multiplicative schedule function for ``LambdaLR``.
    steps_per_epoch : Callable[[], int]
        Lazily-evaluated step count per epoch; only called for
        ``CosineAnnealingWarmRestarts``.
    total_steps : Callable[[], int]
        Lazily-evaluated total step count; only called for ``OneCycleLR``.

    Returns
    -------
    LRScheduler
    """
    if scheduler == "CosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestarts(
            optimizer, steps_per_epoch(), eta_min=1e-6, T_mult=2
        )
    if scheduler == "OneCycleLR":
        return OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps(),
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1000.0,
        )
    if scheduler == "MultiplicativeLR":
        return ExponentialLR(optimizer, 0.9999)
    if scheduler == "LambdaLR":
        return LambdaLR(optimizer, lr_lambda)
    raise ValueError(
        f"Unknown scheduler '{scheduler}'. "
        "Choose 'LambdaLR', 'OneCycleLR', "
        "'CosineAnnealingWarmRestarts', or 'MultiplicativeLR'."
    )


def _preprocess_particle_images(
    images: torch.Tensor, dose_per_angstrom: float, voxel_size: float
) -> torch.Tensor:
    """
    Sign-flip and dose-scale a raw particle stack to match the forward model's
    intensity units.

    Parameters
    ----------
    images : torch.Tensor
        Raw particle stack as read from the .mrc/.mrcs file.
    dose_per_angstrom : float
        Total electron dose (fluence) per image in e⁻/Å².
    voxel_size : float
        Voxel size in Å.

    Returns
    -------
    torch.Tensor
        Preprocessed images.
    """
    images = -images
    dose_per_area = dose_per_angstrom * voxel_size**2
    return dose_per_area**0.5 * images + dose_per_area


def _kmask_half_spectrum(kmask: torch.Tensor) -> torch.Tensor:
    """
    A centred k-mask as the half spectrum ``rfftn`` produces.

    ``ifftshift`` moves the mask's zero frequency to index 0, the layout of
    an unshifted FFT, and the last axis is cut to the ``n // 2 + 1``
    non-redundant frequencies of a real volume's transform. Computed once
    per mask (see :meth:`_BaseReconstructor._kmask_half`), so the per-step
    application needs no shifts at all.

    Parameters
    ----------
    kmask : torch.Tensor
        Centred mask, shape ``(Z, Y, X)``, symmetric under ``k -> -k``.

    Returns
    -------
    torch.Tensor
        Shape ``(Z, Y, X // 2 + 1)``, same dtype and device.
    """
    n = kmask.shape[-1]
    return torch.fft.ifftshift(kmask)[..., : n // 2 + 1].contiguous()


def _apply_kmask_inplace(V: torch.Tensor, kmask_half: torch.Tensor | None) -> None:
    """
    Band-limit `V` in place with a half-spectrum k-mask.

    Applied after every optimiser step. It used to be spelled as a centred
    complex FFT round trip, ``real(ifft3(fft3(V, shift=True) * kmask,
    shift=True))``: twelve ``roll`` passes and two complex 512^3 transforms,
    137 ms and 6.5 GiB per step on a 512-box reconstruction, against a
    ~160 ms training step. The real-to-complex transform of a real volume
    against the mask's unshifted half spectrum is the same operation.

    Parameters
    ----------
    V : torch.Tensor
        Real volume, ``(Z, Y, X)``; ``V.data`` is replaced.
    kmask_half : torch.Tensor or None
        From :func:`_kmask_half_spectrum`; None leaves `V` untouched.
    """
    if kmask_half is None:
        return
    V.data = torch.fft.irfftn(torch.fft.rfftn(V.data) * kmask_half, s=V.shape)


def _log_current_lr(trainer: L.Trainer, lr: float | None, log_lrs: list[float]) -> None:
    """Append the volume optimiser's current LR to ``log_lrs``, if a scheduler is active."""
    if lr is not None and trainer.lr_scheduler_configs:
        log_lrs.append(
            trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        )


def _build_epoch_metrics(
    total_loss: list[torch.Tensor],
    norm_loss: list[torch.Tensor],
    sparsity_loss: list[torch.Tensor],
    lrs: list[float],
    current_epoch: int,
    include_loss_std: bool = False,
    include_lr_min: bool = False,
) -> dict[str, Any]:
    """
    Aggregate per-batch loss/lr logs into a per-epoch metrics dict.

    Parameters
    ----------
    total_loss, norm_loss, sparsity_loss : list of torch.Tensor
        Per-batch scalar losses logged during training. ``sparsity_loss`` may
        be empty if sparsity regularisation was disabled.
    lrs : list of float
        Per-batch learning rate of the volume optimiser.
    current_epoch : int
        Zero-indexed epoch counter (``LightningModule.current_epoch``).
    include_loss_std : bool
        If True, include the per-epoch loss standard deviation.
    include_lr_min : bool
        If True, include the per-epoch minimum learning rate.

    Returns
    -------
    dict
        Keys: ``total_batches``, ``batches_per_epoch``, ``total_epochs``,
        ``total_loss``, ``epochs``.
    """
    n_batches_per_epoch = len(total_loss) // max(current_epoch, 1)
    if n_batches_per_epoch == 0:
        n_batches_per_epoch = len(total_loss)

    epochs: dict[str, Any] = {}
    for epoch in range(current_epoch + 1):
        start = epoch * n_batches_per_epoch
        end = (epoch + 1) * n_batches_per_epoch
        ep_losses = [float(v) for v in total_loss[start:end]]
        ep_norm = [float(v) for v in norm_loss[start:end]]
        ep_lrs = lrs[start:end]
        key = f"epoch_{epoch + 1:02d}"

        loss_stats: dict[str, float | None] = {
            "mean": sum(ep_losses) / len(ep_losses) if ep_losses else None,
            "min": min(ep_losses) if ep_losses else None,
            "max": max(ep_losses) if ep_losses else None,
        }
        if include_loss_std:
            mean = sum(ep_losses) / len(ep_losses) if ep_losses else 0.0
            loss_stats["std"] = (
                (sum((x - mean) ** 2 for x in ep_losses) / len(ep_losses)) ** 0.5
                if len(ep_losses) > 1
                else 0.0
            )

        lr_stats: dict[str, float | None] = {
            "start": ep_lrs[0] if ep_lrs else None,
            "end": ep_lrs[-1] if ep_lrs else None,
        }
        if include_lr_min:
            lr_stats["min"] = min(ep_lrs) if ep_lrs else None

        epochs[key] = {
            "loss": loss_stats,
            "norm_loss": {
                "mean": sum(ep_norm) / len(ep_norm) if ep_norm else None,
            },
            "lr": lr_stats,
        }
        if sparsity_loss:
            ep_sparsity = [float(v) for v in sparsity_loss[start:end]]
            epochs[key]["sparsity_loss"] = {
                "mean": sum(ep_sparsity) / len(ep_sparsity) if ep_sparsity else None,
            }

    return {
        "total_batches": len(total_loss),
        "batches_per_epoch": n_batches_per_epoch,
        "total_epochs": current_epoch + 1,
        "total_loss": [float(v) for v in total_loss],
        "epochs": epochs,
    }
