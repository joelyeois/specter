"""Shared helpers for `specter.pipelines` modules: device parsing, single/multi-GPU
generation dispatch (Lightning DDP), and output formatting.

Extracted from `demo-scripts/generate_particle_stack.py` (and duplicated,
near-verbatim, across its now-retired `_from_csfile`/`_from_starfile` siblings) so
pipeline modules share one copy instead of re-defining these per script.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Sequence

import torch
from rich.console import Console
from rich.rule import Rule

from specter.config import parse_scalar_or_range

_console = Console()


def _uniform_sample(value: str, n: int) -> torch.Tensor:
    """Sample `n` values uniformly from a `parse_scalar_or_range` "value"/"low,high" string."""
    low, high = parse_scalar_or_range(value)
    return torch.rand(n) * (high - low) + low


def _section(msg: str) -> None:
    """Print a full-width titled rule as a section separator."""
    _console.print(Rule(f"[bold yellow]{msg}[/bold yellow]", style="yellow"))


def _crop_center(t: torch.Tensor, nxy: int) -> torch.Tensor:
    """Center-crop a (..., H, W) tensor to (..., nxy, nxy). Matches Detector.forward crop."""
    H, W = t.shape[-2], t.shape[-1]
    if H == nxy and W == nxy:
        return t
    cy, cx = H // 2, W // 2
    half = nxy // 2
    return t[..., cy - half : cy + half + (nxy % 2), cx - half : cx + half + (nxy % 2)]


def _parse_device(device_str: str) -> tuple[str, str | list[int]]:
    """
    Parse a ``--device`` string into a dispatch mode and device target.

    Parameters
    ----------
    device_str : str
        One of ``"cpu"``, ``"cuda"``, ``"cuda:N"``, a bare integer ``"N"``, or
        a comma-separated list of integers (``"0,1,2,3"``) for multi-GPU.

    Returns
    -------
    mode : str
        ``"single"`` or ``"multi"``.
    target : str or list[int]
        Device string for ``"single"``, or a list of GPU ids for ``"multi"``.

    Examples
    --------
    "cpu"     -> ("single", "cpu")
    "cuda"    -> ("single", "cuda")
    "cuda:0"  -> ("single", "cuda:0")
    "0"       -> ("single", "cuda:0")
    "0,1,2"   -> ("multi",  [0, 1, 2])
    """
    parts = device_str.split(",")
    if len(parts) > 1:
        try:
            return "multi", [int(p.strip()) for p in parts]
        except ValueError:
            pass
    if parts[0].strip().isdigit():
        return "single", f"cuda:{parts[0].strip()}"
    return "single", device_str


def _generate_single(
    model: Any,
    n: int,
    batchsize: int,
    track: Any,
    collect_exitwaves: bool = False,
    collect_clean_exitwaves: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run image generation on a single device."""
    idx = torch.arange(n)
    images: list[torch.Tensor] = []
    exitwaves: list[torch.Tensor] = []
    clean_exitwaves: list[torch.Tensor] = []
    with torch.no_grad():
        for i in track(range(0, n, batchsize), description="Generating images"):
            batch = model(idx[i : i + batchsize])
            images.append(batch.detach().cpu())
            if collect_exitwaves:
                exitwaves.append(model.exitwaves.detach().cpu())
            if collect_clean_exitwaves:
                clean_exitwaves.append(model.clean_exitwaves.detach().cpu())
    images_t = torch.concat(images, dim=0)
    exitwaves_t = torch.concat(exitwaves, dim=0) if collect_exitwaves else None
    clean_exitwaves_t = (
        torch.concat(clean_exitwaves, dim=0) if collect_clean_exitwaves else None
    )
    return images_t, exitwaves_t, clean_exitwaves_t


def _generate_multi(
    model: Any,
    n: int,
    batchsize: int,
    gpu_ids: list[int],
    output_dir: str,
    collect_exitwaves: bool = False,
    collect_clean_exitwaves: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Run image generation across multiple GPUs using Lightning DDP.

    Returns
    -------
    images, exitwaves, clean_exitwaves : torch.Tensor or None
        Populated on rank 0; ``(None, None, None)`` on worker ranks.
        ``exitwaves``/``clean_exitwaves`` are ``None`` if their collect flag
        is ``False``.
    """
    import lightning as L
    from lightning.pytorch.callbacks import BasePredictionWriter
    from torch.utils.data import DataLoader

    class _Writer(BasePredictionWriter):
        def __init__(
            self, out_dir: str, save_exitwaves: bool, save_clean_exitwaves: bool
        ) -> None:
            super().__init__("epoch")
            self.out_dir = out_dir
            self.save_exitwaves = save_exitwaves
            self.save_clean_exitwaves = save_clean_exitwaves
            self._exitwaves: list = []
            self._clean_exitwaves: list = []

        def on_predict_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
        ) -> None:
            if self.save_exitwaves and hasattr(pl_module, "exitwaves"):
                self._exitwaves.append(pl_module.exitwaves.cpu())
            if self.save_clean_exitwaves and hasattr(pl_module, "clean_exitwaves"):
                self._clean_exitwaves.append(pl_module.clean_exitwaves.cpu())

        def write_on_epoch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            predictions: Sequence[Any],
            batch_indices: Sequence[Any],
        ) -> None:
            rank = trainer.global_rank
            images = torch.concat(list(predictions), dim=0)
            torch.save(images, os.path.join(self.out_dir, f"predictions_{rank}.pt"))
            idx = torch.squeeze(torch.tensor(batch_indices)).reshape(-1)
            torch.save(idx, os.path.join(self.out_dir, f"batch_indices_{rank}.pt"))
            if self.save_exitwaves and self._exitwaves:
                torch.save(
                    torch.cat(self._exitwaves, dim=0),
                    os.path.join(self.out_dir, f"exitwaves_{rank}.pt"),
                )
            if self.save_clean_exitwaves and self._clean_exitwaves:
                torch.save(
                    torch.cat(self._clean_exitwaves, dim=0),
                    os.path.join(self.out_dir, f"clean_exitwaves_{rank}.pt"),
                )

    os.makedirs(output_dir, exist_ok=True)
    dataloader: DataLoader = DataLoader(
        torch.arange(n),  # type: ignore[arg-type]
        batch_size=batchsize,
        shuffle=False,
        num_workers=os.cpu_count() or 0,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=gpu_ids,
        strategy="ddp",
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
        callbacks=[_Writer(output_dir, collect_exitwaves, collect_clean_exitwaves)],
    )

    print(f"Running multi-GPU generation on GPUs: {gpu_ids}")
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)

    # Only rank 0 reassembles; worker ranks exit cleanly
    if trainer.global_rank != 0:
        return None, None, None

    # Reassemble images in original order
    prediction_files = sorted(glob.glob(os.path.join(output_dir, "predictions_*.pt")))
    index_files = sorted(glob.glob(os.path.join(output_dir, "batch_indices_*.pt")))

    all_preds = torch.cat([torch.load(f) for f in prediction_files], dim=0)
    all_indices = torch.cat([torch.load(f) for f in index_files], dim=0)
    sort_order = torch.argsort(all_indices)
    images = all_preds[sort_order]

    for f in prediction_files + index_files:
        os.remove(f)

    # Reassemble exit waves if collected
    exitwaves = None
    if collect_exitwaves:
        exitwave_files = sorted(glob.glob(os.path.join(output_dir, "exitwaves_*.pt")))
        if exitwave_files:
            all_exitwaves = torch.cat([torch.load(f) for f in exitwave_files], dim=0)
            exitwaves = all_exitwaves[sort_order]
            for f in exitwave_files:
                os.remove(f)

    # Reassemble clean exit waves if collected
    clean_exitwaves = None
    if collect_clean_exitwaves:
        clean_files = sorted(
            glob.glob(os.path.join(output_dir, "clean_exitwaves_*.pt"))
        )
        if clean_files:
            all_clean = torch.cat([torch.load(f) for f in clean_files], dim=0)
            clean_exitwaves = all_clean[sort_order]
            for f in clean_files:
                os.remove(f)

    return images, exitwaves, clean_exitwaves


def _save_exitwave_pair(
    ew: torch.Tensor,
    suffix: str,
    output_dir: str,
    filename: str,
    pad_fft: bool,
    num_pixels: int,
) -> None:
    """Save an exit-wave tensor's magnitude and phase as separate .mrcs files."""
    import mrcfile

    if pad_fft:
        ew = _crop_center(ew, num_pixels)
    os.makedirs(output_dir, exist_ok=True)
    mag_path = os.path.join(output_dir, f"{filename}_{suffix}_magnitude.mrcs")
    phase_path = os.path.join(output_dir, f"{filename}_{suffix}_phase.mrcs")
    with mrcfile.new(mag_path, overwrite=True) as mrc:
        mrc.set_data(ew.abs().numpy().astype("float32"))
    _console.print(f"  [green]✓[/green] {mag_path}")
    with mrcfile.new(phase_path, overwrite=True) as mrc:
        mrc.set_data(ew.angle().numpy().astype("float32"))
    _console.print(f"  [green]✓[/green] {phase_path}")


def _format_elapsed(seconds: float) -> str:
    """Format an elapsed-time duration as e.g. "1h 2m 3s", dropping empty leading units."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
