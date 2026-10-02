"""Alternate loader counts on a full-size real reconstruction training loop.

This diagnostic excludes CLI spawning, epoch figures, FSC and disk output.
Complete CLI measurements are recorded separately. Every run reconstructs
at the input's full box size with the shipped physics and optimizer settings.
"""

import argparse
import gc
import json
import multiprocessing
from pathlib import Path
import time
import tomllib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--cs-file", required=True)
    p.add_argument("--mrc-file", required=True)
    p.add_argument("--dose", type=float, required=True)
    p.add_argument("--device", type=int, default=3)
    p.add_argument("--batches", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    import lightning as L
    import numpy as np
    import torch
    from specter.config import ReconstructionConfig
    from specter.ghostbuster import Ghostbuster
    from specter.pipelines._reconstruct import _ghostbuster_kwargs

    torch.set_num_threads(4)
    torch.cuda.set_device(a.device)
    with open(a.config, "rb") as f:
        config_data = tomllib.load(f)
    kwargs = {k: v for section in config_data.values() for k, v in section.items()}
    kwargs.update(
        cs_file=a.cs_file,
        mrc_file=a.mrc_file,
        dose_per_angstrom=a.dose,
        halfset="A",
        epochs=1,
        num_workers=0,
    )
    config = ReconstructionConfig(**kwargs)
    g = Ghostbuster(**_ghostbuster_kwargs(config))
    assert a.batches <= (len(g._images) + g.batchsize - 1) // g.batchsize
    results = []

    class Timing(L.Callback):
        def on_train_start(self, trainer, model):
            self.events = []
            self.wall_start = None

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            self.current = None
            if batch_idx >= a.warmup:
                if self.wall_start is None:
                    torch.cuda.synchronize()
                    self.wall_start = time.perf_counter()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                self.current = (start, end)

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            if self.current:
                self.current[1].record()
                self.events.append(self.current)
            if batch_idx == a.batches - 1:
                torch.cuda.synchronize()
                self.wall_seconds = time.perf_counter() - self.wall_start

    for workers in [0, 8, 8, 0]:
        L.seed_everything(123, workers=False)
        g.num_workers = workers
        model, loader = g._build_reconstructor_and_loader(
            g._images, g._voxel_size, g.batchsize
        )
        metric = Timing()
        trainer = L.Trainer(
            accelerator="gpu",
            devices=[a.device],
            precision=config.precision,
            max_epochs=1,
            limit_train_batches=a.batches,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            callbacks=[metric],
        )
        torch.cuda.reset_peak_memory_stats()
        trainer.fit(model, train_dataloaders=loader)
        torch.cuda.synchronize()
        row = dict(
            workers=workers,
            particles=len(g._images),
            box=g._images.shape[-1],
            batchsize=g.batchsize,
            batches=a.batches,
            warmup=a.warmup,
            measured_batches=len(metric.events),
            wall_seconds=metric.wall_seconds,
            wall_ms_per_batch=metric.wall_seconds / len(metric.events) * 1000,
            median_gpu_training_ms=float(
                np.median([s.elapsed_time(e) for s, e in metric.events])
            ),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        results.append(row)
        print(json.dumps(row), flush=True)
        del trainer, model, loader, metric
        gc.collect()
        torch.cuda.empty_cache()
    Path(a.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
