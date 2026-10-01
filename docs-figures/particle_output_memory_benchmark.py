"""Measure the real particle CLI and optionally replay its generated batches.

Select the source checkout before importing SPECTER; GPU peak figures in
measurement files come from the actual simulation, not the replay.
"""

# ruff: noqa: E402
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

p = argparse.ArgumentParser()
p.add_argument("--checkout", required=True)
p.add_argument("--output", required=True)
p.add_argument("--device", default="cuda:1")
p.add_argument("--n", type=int, default=3000)
p.add_argument("--seed", type=int, default=83)
p.add_argument("--paired-baseline")
a = p.parse_args()
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, str(Path(a.checkout) / "src"))
import torch
import psutil
from specter.cli._cli import cli
from specter.pipelines import _particles as pipeline

torch.set_num_threads(4)
gpu_ids = [int(v) for v in a.device.split(",")] if "," in a.device else None
torch.cuda.set_device(
    gpu_ids[int(os.environ.get("LOCAL_RANK", 0))] if gpu_ids else a.device
)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
phase = ["setup"]
peaks = {k: 0 for k in ["setup", "generation", "saving", "all"]}
samples = []
stop = threading.Event()
process = psutil.Process()


def monitor():
    while not stop.wait(0.05):
        rss = process.memory_info().rss
        peaks[phase[0]] = max(peaks[phase[0]], rss)
        peaks["all"] = max(peaks["all"], rss)


tree_stats = {"peak_processes": 1, "peak_aggregate_pss_bytes": 0}
old_monitor = monitor


def monitor():
    if not gpu_ids or "LOCAL_RANK" in os.environ:
        return old_monitor()
    last_tree = 0.0
    while not stop.wait(0.05):
        rss = process.memory_info().rss
        peaks[phase[0]] = max(peaks[phase[0]], rss)
        peaks["all"] = max(peaks["all"], rss)
        now = time.perf_counter()
        if now - last_tree >= 1:
            last_tree = now
            children = [process] + process.children(recursive=True)
            pss = 0
            for child in children:
                try:
                    pss += child.memory_full_info().pss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            tree_stats["peak_processes"] = max(
                tree_stats["peak_processes"], len(children)
            )
            tree_stats["peak_aggregate_pss_bytes"] = max(
                tree_stats["peak_aggregate_pss_bytes"], pss
            )


thread = threading.Thread(target=monitor, daemon=True)
thread.start()
raw_gen = pipeline._generate_single
expected_source = Path(a.checkout) / "src/specter/pipelines/_common.py"
assert Path(inspect.getsourcefile(raw_gen)).resolve() == expected_source.resolve()
generation = {}


def measured(model, n, batchsize, track, **flags):
    phase[0] = "generation"
    model.progressbars = False
    for module in model.modules():
        if hasattr(module, "progressbars"):
            module.progressbars = False
    counter = [0]
    t0 = time.perf_counter()

    def progress(_module, _args, _result):
        counter[0] += 1
        if counter[0] == 1 or counter[0] % 100 == 0:
            print(
                f"batch {counter[0]}/{(n + batchsize - 1) // batchsize}, elapsed {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
        if counter[0] % 100 == 0:
            samples.append(
                {
                    "batch": counter[0],
                    "rss_bytes": process.memory_info().rss,
                    "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                }
            )

    hook = model.register_forward_hook(progress)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = raw_gen(model, n, batchsize, lambda iterable, **kwargs: iterable, **flags)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    hook.remove()
    generation.update(
        {
            "seconds": elapsed,
            "n": n,
            "batchsize": batchsize,
            "pad_nxy": model.pad_nxy,
            "nz": model.nz,
            "collect_flags": flags,
            "shapes": [None if x is None else list(x.shape) for x in result],
        }
    )
    if a.paired_baseline:
        torch.save(result[0], out / "raw-images.pt")
    phase[0] = "saving"
    return result


pipeline._generate_single = measured
raw_multi = pipeline._generate_multi


def measured_multi(*args, **kwargs):
    phase[0] = "generation"
    t0 = time.perf_counter()
    result = raw_multi(*args, **kwargs)
    torch.cuda.synchronize()
    generation.update(
        {
            "seconds": time.perf_counter() - t0,
            "n": args[1],
            "batchsize": args[2],
            "gpu_ids": args[3],
        }
    )
    phase[0] = "saving"
    return result


pipeline._generate_multi = measured_multi
if gpu_ids:
    import lightning as L

    old_init = L.Trainer.__init__

    def quiet_init(self, *args, **kwargs):
        kwargs["enable_progress_bar"] = False
        return old_init(self, *args, **kwargs)

    L.Trainer.__init__ = quiet_init
    old_model = pipeline.ImageGenerator.__init__

    def quiet_model(self, *args, **kwargs):
        kwargs["progressbars"] = False
        old_model(self, *args, **kwargs)

    pipeline.ImageGenerator.__init__ = quiet_model

start = time.perf_counter()
try:
    cli(
        prog_name="specter",
        standalone_mode=False,
        args=[
            "simulate",
            "particles",
            "--config",
            str(Path(a.checkout) / "configs/particle.toml"),
            "--n_particles",
            str(a.n),
            "--batchsize",
            "2",
            "--seed",
            str(a.seed),
            "--device",
            a.device,
            "--output_dir",
            str(out),
        ],
    )
    torch.cuda.synchronize()
finally:
    elapsed = time.perf_counter() - start
    stop.set()
    thread.join()
result = {
    "checkout": a.checkout,
    "device": a.device,
    "gpu": torch.cuda.get_device_name(),
    "torch": torch.__version__,
    "threads": 4,
    "seed": a.seed,
    "cli_seconds": elapsed,
    "generation": generation,
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    "host_rss_max_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    "host_rss_peaks_bytes": peaks,
    "lifetime_samples": samples,
    "process_tree": tree_stats,
    "generator_source": inspect.getsourcefile(raw_gen),
    "source_sha256": {
        name: hashlib.sha256(
            (Path(a.checkout) / "src/specter/pipelines" / name).read_bytes()
        ).hexdigest()
        for name in ["_common.py", "_particles.py"]
    },
}
(
    out
    / (
        "measurement-rank" + os.environ["LOCAL_RANK"] + ".json"
        if "LOCAL_RANK" in os.environ
        else "measurement.json"
    )
).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)

if a.paired_baseline:
    import mrcfile
    from specter.image import normalize_particles

    old_path = Path(a.paired_baseline) / "src/specter/pipelines/_common.py"
    spec = importlib.util.spec_from_file_location(
        "specter.pipelines._common_reference", old_path
    )
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    raw = torch.load(out / "raw-images.pt", weights_only=True)

    class Replay:
        def __call__(self, idx):
            return raw[idx].to(a.device)

    trials = []
    for pair in range(3):
        results = {}
        outputs = {}
        order = ["before", "after"] if pair != 1 else ["after", "before"]
        for variant in order:
            gc.collect()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn = old._generate_single if variant == "before" else raw_gen
            images = fn(Replay(), a.n, 2, lambda iterable, **_: iterable)[0]
            torch.cuda.synchronize()
            collection_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            if variant == "before":
                images = -normalize_particles(images)[0]
            else:
                for start in range(0, len(images), 64):
                    chunk = images[start : start + 64]
                    chunk.copy_(-normalize_particles(chunk)[0])
            results[variant] = {
                "collection_seconds": collection_seconds,
                "normalization_seconds": time.perf_counter() - t0,
            }
            outputs[variant] = images
            if pair == 0:
                with mrcfile.new(out / f"paired-{variant}.mrcs", overwrite=True) as mrc:
                    mrc.set_data(images.numpy())
                    mrc.voxel_size = 1.0
        before, after = outputs["before"], outputs["after"]
        results["accuracy"] = {
            "bitwise_equal": torch.equal(before, after),
            "max_abs": float((before - after).abs().max()),
            "pixel_count": before.numel(),
        }
        trials.append(results)
        print("paired storage check", pair + 1, results, flush=True)
        del outputs, images, before, after
    (out / "paired-storage.json").write_text(
        json.dumps(
            {
                "n": a.n,
                "shape": list(raw.shape),
                "same_generated_batches": True,
                "trials": trials,
            },
            indent=2,
        )
        + "\n"
    )
