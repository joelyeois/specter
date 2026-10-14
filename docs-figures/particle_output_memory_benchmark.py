"""Measure the real particle CLI and optionally replay its generated batches.

Select the source checkout before importing SPECTER; GPU peak figures in
measurement files come from the actual simulation, not the replay.

Generation is the object-API loop `specter simulate particles` runs:
`_ParticleRun.generate` (`Specimen.scatter`, `ExitWave.apply_ctf`,
`.intensity` and `Image.detect` per batch, each batch copied into one host
stack by `_HostStack`) on one device, and `_generate_sharded`, one spawned
worker per device, on several. These replaced the generator-era
`_generate_single` collector and the Lightning DDP ranks, and are timed and
sampled in their place, so the measured quantities are unchanged:
whole-CLI time, generation time, CUDA peaks, host RSS per phase, lifetime
samples every 100 batches and, with several devices, the process tree's PSS.
A multi-device run writes ``measurement-rank0.json`` (the name the DDP-era
runs used, which `particle_output_memory_figures.py` reads) from the
launching process, with each worker's own CUDA peaks under ``ranks``; the
hooks are the object API's, so ``--checkout`` must have `specter.imaging`.

``--paired-baseline`` replays the generated images, batch by batch, through
the baseline checkout's collector (its `_generate_single` when it has one,
else its `_HostStack`) and whole-stack normalization, and through the
candidate's `_HostStack` and chunked normalization, and compares the two.
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


def _measured_shard_worker(run, start, stop, device, batchsize, path, threads):
    """
    `_shard_worker`, with its device's CUDA peaks written beside its share.

    Module-level so a spawned worker can unpickle it; the launching process
    reads ``<share>.measurement.json`` back once every worker has exited.
    """
    import torch
    from specter.pipelines._particles import _shard_worker

    torch.set_num_threads(4)
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    _shard_worker(run, start, stop, device, batchsize, path, threads)
    if cuda:
        torch.cuda.synchronize()
    Path(path + ".measurement.json").write_text(
        json.dumps(
            {
                "device": device,
                "start": start,
                "stop": stop,
                "seconds": time.perf_counter() - t0,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if cuda
                else 0,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else 0,
            }
        )
    )


def _collect(module, raw, n, batchsize, device):
    """Replay `raw` through `module`'s batch collector, a batch at a time."""
    import torch

    if hasattr(module, "_generate_single"):
        # A generator-era checkout: its collector calls a model per batch.
        class Replay:
            def __call__(self, idx):
                return raw[idx].to(device)

        return module._generate_single(Replay(), n, batchsize, lambda it, **_: it)[0]
    stack = module._HostStack(n)
    with torch.no_grad():
        for lo in range(0, n, batchsize):
            stack.append(raw[lo : lo + batchsize].to(device))
    return stack.finish()


def main() -> None:
    # Guarded so the module imports without arguments (tests/test_docs_figures_import.py).
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
    from specter.devices import parse_device
    from specter.pipelines import _common
    from specter.pipelines import _particles as pipeline

    torch.set_num_threads(4)
    devices = parse_device(a.device).devices
    multi = len(devices) > 1
    torch.cuda.set_device(devices[0])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    phase = ["setup"]
    peaks = {k: 0 for k in ["setup", "generation", "saving", "all"]}
    samples = []
    stop = threading.Event()
    process = psutil.Process()
    tree_stats = {"peak_processes": 1, "peak_aggregate_pss_bytes": 0}

    def monitor():
        last_tree = 0.0
        while not stop.wait(0.05):
            rss = process.memory_info().rss
            peaks[phase[0]] = max(peaks[phase[0]], rss)
            peaks["all"] = max(peaks["all"], rss)
            now = time.perf_counter()
            if multi and now - last_tree >= 1:
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
    raw_generate = pipeline._ParticleRun.generate
    raw_specimen = pipeline._ParticleRun.specimen
    expected_source = Path(a.checkout) / "src/specter/pipelines/_particles.py"
    assert (
        Path(inspect.getsourcefile(raw_generate)).resolve() == expected_source.resolve()
    )
    generation = {}
    specimens = []

    def specimen(self, device):
        built = raw_specimen(self, device)
        specimens.append(built)
        return built

    def measured(self, first, last, device, batchsize, progress):
        phase[0] = "generation"
        n = last - first
        n_batches = (n + batchsize - 1) // batchsize

        def counted(iterable, **_):
            for k, lo in enumerate(iterable, 1):
                yield lo
                # Resumed once batch k has been made and collected.
                if k == 1 or k % 100 == 0:
                    print(
                        f"batch {k}/{n_batches}, elapsed {time.perf_counter() - t0:.1f}s",
                        flush=True,
                    )
                if k % 100 == 0:
                    samples.append(
                        {
                            "batch": k,
                            "rss_bytes": process.memory_info().rss,
                            "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                        }
                    )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = raw_generate(self, first, last, device, batchsize, counted)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        built = specimens[-1]
        generation.update(
            {
                "seconds": elapsed,
                "n": n,
                "batchsize": batchsize,
                "pad_nxy": built.pad_nxy(self.config.pad_fft),
                "nz": built.nz,
                "collect_flags": {
                    "save_exitwaves": self.config.save_exitwaves,
                    "save_clean_exitwaves": self.config.save_clean_exitwaves,
                },
                "shapes": [None if x is None else list(x.shape) for x in result],
            }
        )
        if a.paired_baseline:
            torch.save(result[0], out / "raw-images.pt")
        phase[0] = "saving"
        return result

    pipeline._ParticleRun.specimen = specimen
    pipeline._ParticleRun.generate = measured
    raw_sharded = pipeline._generate_sharded
    ranks = []

    def measured_sharded(run, devices, batchsize, output_dir):
        phase[0] = "generation"
        t0 = time.perf_counter()
        result = raw_sharded(run, devices, batchsize, output_dir)
        torch.cuda.synchronize()
        generation.update(
            {
                "seconds": time.perf_counter() - t0,
                "n": len(run.poses),
                "batchsize": batchsize,
                "gpu_ids": list(devices),
            }
        )
        for path in sorted(Path(output_dir).glob("particle_shard_*.measurement.json")):
            ranks.append(json.loads(path.read_text()))
            path.unlink()
        phase[0] = "saving"
        return result

    pipeline._generate_sharded = measured_sharded
    pipeline._shard_worker = _measured_shard_worker

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
        "ranks": ranks,
        "generator_source": inspect.getsourcefile(raw_generate),
        "source_sha256": {
            name: hashlib.sha256(
                (Path(a.checkout) / "src/specter/pipelines" / name).read_bytes()
            ).hexdigest()
            for name in ["_common.py", "_particles.py"]
        },
    }
    (out / ("measurement-rank0.json" if multi else "measurement.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
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

        trials = []
        for pair in range(3):
            results = {}
            outputs = {}
            order = ["before", "after"] if pair != 1 else ["after", "before"]
            for variant in order:
                gc.collect()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                module = old if variant == "before" else _common
                images = _collect(module, raw, a.n, 2, devices[0])
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
                    with mrcfile.new(
                        out / f"paired-{variant}.mrcs", overwrite=True
                    ) as mrc:
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


if __name__ == "__main__":
    main()
