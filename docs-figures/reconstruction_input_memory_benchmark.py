"""Measure full shipped reconstruction runs on a supplied real, row-ordered pair.

The same script is imported by gold-halfset spawn workers. Only measurement,
fixed per-halfset RNG seeds and bounded CPU thread counts are added to the CLI.
"""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time


def install_measurements():
    sys.path.insert(0, os.environ["SPECTER_RECON_CHECKOUT"] + "/src")
    import lightning as L
    import torch
    from specter.ghostbuster import Ghostbuster

    torch.set_num_threads(4)
    original_init = Ghostbuster.__init__
    original_run = Ghostbuster.run

    def measured_init(self, *args, **kwargs):
        start = time.perf_counter()
        original_init(self, *args, **kwargs)
        self._benchmark_load_seconds = time.perf_counter() - start
        self._benchmark_load_peak_rss_bytes = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )

    class Metrics(L.Callback):
        def on_train_start(self, trainer, model):
            self.start = time.perf_counter()
            self.events = []
            self.current = None

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.current = (start, end)

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            # Callback ordering precedes the module's post-update k-mask. GPU
            # event timings therefore cover training_step only, not that mask.
            self.current[1].record()
            self.events.append(self.current)

        def on_train_end(self, trainer, model):
            torch.cuda.synchronize()
            self.train_seconds = time.perf_counter() - self.start
            self.step_seconds = [s.elapsed_time(e) / 1000 for s, e in self.events]
            self.global_steps = trainer.global_step

    def measured_run(self, device=0, callbacks=None):
        label = self.halfset_label or "all"
        seed = int(os.environ["SPECTER_RECON_SEED"]) + (0 if label == "A" else 1)
        L.seed_everything(seed, workers=False)
        monitor_done = threading.Event()
        driver_peak = [0]

        def monitor():
            while not monitor_done.is_set():
                query = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                for line in query.stdout.splitlines():
                    fields = [x.strip() for x in line.split(",")]
                    if (
                        len(fields) == 2
                        and fields[0] == str(os.getpid())
                        and fields[1].isdigit()
                    ):
                        driver_peak[0] = max(driver_peak[0], int(fields[1]) * 2**20)
                monitor_done.wait(0.3)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
        metric = Metrics()
        start = time.perf_counter()
        try:
            model = original_run(self, device, [*(callbacks or []), metric])
            torch.cuda.synchronize()
        finally:
            monitor_done.set()
            thread.join(timeout=3)
        result = dict(
            halfset=label,
            seed=seed,
            particles=len(self._images),
            box=self._images.shape[-1],
            epochs=self.epochs,
            batchsize=self.batchsize,
            num_workers=self.num_workers,
            precision=self.precision,
            device=device,
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            loading_seconds=self._benchmark_load_seconds,
            loading_peak_rss_bytes=self._benchmark_load_peak_rss_bytes,
            fit_seconds=time.perf_counter() - start,
            train_seconds=metric.train_seconds,
            step_seconds=metric.step_seconds,
            global_steps=metric.global_steps,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            driver_peak_bytes=driver_peak[0],
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        path = Path(os.environ["SPECTER_RECON_OUTPUT"]) / f"measurement_{label}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(
            "BENCHMARK_HALFSET",
            json.dumps({k: v for k, v in result.items() if k != "step_seconds"}),
            flush=True,
        )
        return model

    # Job.create introspects this signature; preserve it exactly.
    import functools

    Ghostbuster.__init__ = functools.wraps(original_init)(measured_init)
    Ghostbuster.run = measured_run


# Gold workers have names like SpawnProcess-1; their nested DataLoader
# workers are SpawnProcess-1:1. Keep reconstruction/Lightning imports and
# instrumentation out of the latter, as in the shipped console entry point.
if (
    "SPECTER_RECON_CHECKOUT" in os.environ
    and ":" not in multiprocessing.current_process().name
):
    install_measurements()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cs-file", required=True)
    parser.add_argument("--mrc-file", required=True)
    parser.add_argument("--dose", required=True, type=float)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--reference")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    os.environ["SPECTER_RECON_CHECKOUT"] = args.checkout
    os.environ["SPECTER_RECON_OUTPUT"] = str(Path(args.output).resolve())
    os.environ["SPECTER_RECON_SEED"] = str(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    install_measurements()
    import psutil
    from specter.cli._cli import cli

    stop = threading.Event()
    peak = {"tree_pss_bytes": 0, "tree_rss_bytes": 0, "processes": 0}
    parent = psutil.Process()

    def sample():
        while not stop.is_set():
            pss = rss = count = 0
            for proc in [parent, *parent.children(recursive=True)]:
                try:
                    mem = proc.memory_full_info()
                    pss += mem.pss
                    rss += mem.rss
                    count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak["tree_pss_bytes"] = max(peak["tree_pss_bytes"], pss)
            peak["tree_rss_bytes"] = max(peak["tree_rss_bytes"], rss)
            peak["processes"] = max(peak["processes"], count)
            stop.wait(0.5)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    options = [
        "reconstruct",
        "particle",
        "--config",
        str(Path(args.checkout) / "configs/reconstruct.toml"),
        "--cs_file",
        args.cs_file,
        "--mrc_file",
        args.mrc_file,
        "--dose_per_angstrom",
        str(args.dose),
        "--output_dir",
        str(output),
        "--device",
        args.device,
    ]
    if args.reference:
        options += ["--fsc_ref", args.reference]
    if args.workers is not None:
        options += ["--num_workers", str(args.workers)]
    start = time.perf_counter()
    try:
        cli.main(args=options, standalone_mode=False)
    finally:
        stop.set()
        thread.join(timeout=3)
    summary = dict(
        checkout=args.checkout,
        cs_file=args.cs_file,
        mrc_file=args.mrc_file,
        dose=args.dose,
        seed=args.seed,
        device=args.device,
        reference=args.reference,
        cli_seconds=time.perf_counter() - start,
        host_peak=peak,
        measurement="Five full epochs; gold halfsets; GPU step events exclude post-update k-mask; host PSS sampled every 0.5s; driver memory per worker sampled every 0.3s.",
    )
    (output / "measurement.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("BENCHMARK_CLI", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
