"""Measure the actual CLI, its generation phase, CUDA peaks, and host RSS."""

import time

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading

started = time.perf_counter()
parser = argparse.ArgumentParser()
parser.add_argument("--checkout", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--device", default="cuda:1")
parser.add_argument("--seed", type=int, default=73)
parser.add_argument("--noise", default="poisson")
parser.add_argument("--paired-baseline")
parser.add_argument("--wait-for")
parser.add_argument("--volume-path", default="tomograms/tomogram.mrc")
args = parser.parse_args()
os.environ.setdefault("MPLCONFIGDIR", "/tmp/specter-tiltseries-matplotlib")
sys.path.insert(0, str(Path(args.checkout) / "src"))

import psutil  # noqa: E402
import torch  # noqa: E402
from specter.cli._cli import cli  # noqa: E402
from specter.imagegenerator import TiltSeriesGenerator  # noqa: E402

torch.set_num_threads(4)
torch.cuda.set_device(args.device)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
phase = ["setup"]
peaks = {"setup": 0, "generation": 0, "saving": 0, "all": 0}
driver_peak = [0]
done = threading.Event()
process = psutil.Process()


def monitor():
    last_gpu = 0.0
    while not done.is_set():
        rss = process.memory_info().rss
        peaks[phase[0]] = max(peaks[phase[0]], rss)
        peaks["all"] = max(peaks["all"], rss)
        now = time.perf_counter()
        if now - last_gpu >= 1:
            last_gpu = now
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            for line in gpu.stdout.splitlines():
                fields = [x.strip() for x in line.split(",")]
                if (
                    len(fields) == 2
                    and fields[0] == str(os.getpid())
                    and fields[1].isdigit()
                ):
                    driver_peak[0] = max(driver_peak[0], int(fields[1]) * 2**20)
        done.wait(0.2)


thread = threading.Thread(target=monitor, daemon=True)
thread.start()
gen = TiltSeriesGenerator.generate_tilt_series
source_path = Path(inspect.getsourcefile(gen)).resolve()
expected_source = (
    Path(args.checkout) / "src/specter/imagegenerator/_tiltseries.py"
).resolve()
if source_path != expected_source:
    raise RuntimeError(f"Loaded {source_path}, expected {expected_source}")
generation = {}
held_model = []


def measured(self, *pargs, **kwargs):
    if args.wait_for:
        while not Path(args.wait_for).exists():
            time.sleep(1)
    if args.paired_baseline:
        held_model.append(self)
    self.progressbars = False
    counter = [0]
    detect = self.detector.forward

    def detection(*dargs, **dkwargs):
        result = detect(*dargs, **dkwargs)
        counter[0] += 1
        if counter[0] == 1 or counter[0] % 10 == 0:
            print(
                f"tilt {counter[0]}/61, elapsed {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
        return result

    self.detector.forward = detection
    phase[0] = "generation"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = gen(self, *pargs, **kwargs)
    torch.cuda.synchronize()
    generation.update(
        {
            "seconds": time.perf_counter() - t0,
            "volume_shape": list(self.volume.shape),
            "volume_device": str(self.volume.device),
            "per_tilt_solvent": self._solvent_series is not None,
            "per_tilt_damage": self._specimen_spectrum is not None,
            "collect_flags": kwargs,
            "returned_shapes": [None if x is None else list(x.shape) for x in result],
        }
    )
    self.detector.forward = detect
    phase[0] = "saving"
    return result


TiltSeriesGenerator.generate_tilt_series = measured

torch.cuda.synchronize()
cli_start = time.perf_counter()
try:
    cli(
        prog_name="specter",
        standalone_mode=False,
        args=[
            "simulate",
            "tiltseries",
            "--config",
            str(Path(args.checkout) / "configs/tiltseries.toml"),
            "--volume_path",
            args.volume_path,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--noise_model",
            args.noise,
            "--output_dir",
            str(output),
        ],
    )
    torch.cuda.synchronize()
finally:
    elapsed = time.perf_counter() - cli_start
    done.set()
    thread.join(timeout=3)

sources = [
    Path(args.checkout) / "src/specter/imagegenerator/_tiltseries.py",
    Path(args.checkout) / "src/specter/pipelines/_tiltseries.py",
]
result = {
    "checkout": args.checkout,
    "loaded_generator_source": str(source_path),
    "volume_path": str(Path(args.volume_path).resolve()),
    "device": args.device,
    "gpu": torch.cuda.get_device_name(),
    "torch": torch.__version__,
    "threads": 4,
    "seed": args.seed,
    "noise": args.noise,
    "cli_seconds": elapsed,
    "python_process_seconds": time.perf_counter() - started,
    "generation": generation,
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    "sampled_driver_peak_bytes": driver_peak[0],
    "host_rss_peaks_bytes": peaks,
    "host_rss_max_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    "source_sha256": {
        p.name + str(i): hashlib.sha256(p.read_bytes()).hexdigest()
        for i, p in enumerate(sources)
    },
    "output_files": [
        {"path": str(p.relative_to(output)), "bytes": p.stat().st_size}
        for p in output.rglob("*")
        if p.is_file()
    ],
}
(output / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)

if args.paired_baseline:
    import importlib.util
    import gc
    import mrcfile
    import specter

    old_path = Path(args.paired_baseline) / "src/specter/imagegenerator/_tiltseries.py"
    name = "specter.imagegenerator._tiltseries_benchmark_baseline"
    spec = importlib.util.spec_from_file_location(name, old_path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[name] = baseline
    spec.loader.exec_module(baseline)
    old_gen = baseline.TiltSeriesGenerator.generate_tilt_series
    model = held_model.pop()
    trials = []
    for pair in range(3):
        pair_results = {}
        # Reverse order in the middle pair to reduce timing-order bias.
        order = ["before", "after"] if pair != 1 else ["after", "before"]
        saved = {}
        for variant in order:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            specter.seed(173)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                returned = (
                    old_gen(model, torch.tensor([0]))
                    if variant == "before"
                    else gen(
                        model,
                        torch.tensor([0]),
                        collect_exitwaves=False,
                        collect_clean_images=False,
                    )
                )
            torch.cuda.synchronize()
            pair_results[variant] = {
                "seconds": time.perf_counter() - t0,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            images = returned[0]
            saved[variant] = images
            del returned
            if pair == 0:
                with mrcfile.new(
                    output / f"paired-{variant}.mrcs", overwrite=True
                ) as mrc:
                    mrc.set_data(images[0].numpy())
                    mrc.voxel_size = 5.0
            print(f"paired {pair + 1}/3 {variant}: {pair_results[variant]}", flush=True)
        a, b = saved["before"], saved["after"]
        equal = torch.equal(a, b)
        squared_diff = 0.0
        squared_ref = 0.0
        max_abs = 0.0
        for ai, bi in zip(a[0], b[0]):
            d = bi.double() - ai.double()
            max_abs = max(max_abs, float(d.abs().max()))
            squared_diff += float(d.square().sum())
            squared_ref += float(ai.double().square().sum())
        pair_results["accuracy"] = {
            "bitwise_equal": equal,
            "relative_l2": (squared_diff / squared_ref) ** 0.5,
            "max_abs": max_abs,
        }
        trials.append(pair_results)
        del a, b, ai, bi, d, images, saved
    (output / "paired-generation.json").write_text(
        json.dumps(
            {
                "seed": 173,
                "same_model": True,
                "shape": list(model.volume.shape),
                "baseline_path": str(old_path),
                "candidate_path": str(sources[0]),
                "trials": trials,
            },
            indent=2,
        )
        + "\n"
    )
