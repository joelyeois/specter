"""Measure the actual CLI, its generation phase, CUDA peaks, and host RSS.

Runs ``specter simulate tiltseries`` on the checkout's
``configs/tiltseries.toml``. Generation is the object-API loop the pipeline
runs once its `TiltSeries` is prepared: per tilt, `TiltSeries.scatter`,
`ExitWave.apply_ctf`, `.intensity` and `Image.detect`, with the images moved
to the host. It replaced the generator-era ``generate_tilt_series`` method
and is timed in its place (the loop is bracketed through the pipeline's
`track`), so the measured quantities are unchanged: whole-CLI time,
generation time, CUDA peaks, the sampled driver peak and host RSS per phase.
Ice blending and padding happen when the `TiltSeries` is built, in setup, as
they did when the generator was constructed.

``--paired-baseline`` re-runs generation twice on the same prepared series,
so both see the exact same ice and specimen: once with the baseline
checkout's `TiltSeries.scatter` and once with the candidate's, each followed
by the candidate's lens and camera stages with the arguments the CLI used
and the detector noise drawn from seed 173. The hooks are the object API's,
so both checkouts must have `specter.imaging`.
"""

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


def main() -> None:
    # Guarded so the module imports without arguments (tests/test_docs_figures_import.py).
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
    from specter.imaging import ExitWave, Image, TiltSeries  # noqa: E402
    from specter.pipelines import _tiltseries as pipeline  # noqa: E402

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
    source_path = Path(inspect.getsourcefile(TiltSeries)).resolve()
    expected_source = (Path(args.checkout) / "src/specter/imaging/_tilt.py").resolve()
    if source_path != expected_source:
        raise RuntimeError(f"Loaded {source_path}, expected {expected_source}")
    generation = {}
    held_series = []
    # Each stage's arguments per tilt, so a paired run repeats the CLI's chain.
    calls = {"apply_ctf": {}, "intensity": {}, "detect": {}}
    shapes = {"images": [], "exitwaves": []}
    track = pipeline.track
    scatter = TiltSeries.scatter
    apply_ctf = ExitWave.apply_ctf
    intensity = ExitWave.intensity
    detect = Image.detect
    pixels = ExitWave.pixels

    def measured(iterable, **_):
        if args.wait_for:
            while not Path(args.wait_for).exists():
                time.sleep(1)
        tilts = list(iterable)
        phase[0] = "generation"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for k, i in enumerate(tilts, 1):
            yield i
            # Resumed once tilt i has been detected and moved to the host.
            if k == 1 or k % 10 == 0:
                print(
                    f"tilt {k}/{len(tilts)}, elapsed {time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
        torch.cuda.synchronize()
        series = held_series[-1]
        generation.update(
            {
                "seconds": time.perf_counter() - t0,
                "volume_shape": list(series.volume.shape),
                "volume_device": str(series.volume.device),
                "per_tilt_solvent": series._solvent_series is not None,
                "per_tilt_damage": series._spectrum is not None,
                "returned_shapes": [
                    [len(shapes["images"]), *shapes["images"][0]],
                    [len(shapes["exitwaves"]), *shapes["exitwaves"][0]]
                    if shapes["exitwaves"]
                    else None,
                ],
            }
        )
        if not args.paired_baseline:
            held_series.clear()
        phase[0] = "saving"

    def scattered(self, i, **kwargs):
        if not held_series or held_series[-1] is not self:
            held_series.append(self)
        return scatter(self, i, **kwargs)

    def recorded(name, fn):
        def stage(self, *a, **k):
            calls[name][self.tilt] = (a, k)
            result = fn(self, *a, **k)
            if name == "detect":
                shapes["images"].append(list(result.array.shape[1:]))
            return result

        return stage

    def exitwave(self):
        result = pixels(self)
        if phase[0] == "generation":
            shapes["exitwaves"].append(list(result.shape[1:]))
        return result

    pipeline.track = measured
    TiltSeries.scatter = scattered
    ExitWave.apply_ctf = recorded("apply_ctf", apply_ctf)
    ExitWave.intensity = recorded("intensity", intensity)
    Image.detect = recorded("detect", detect)
    ExitWave.pixels = exitwave

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
        pipeline.track = track
        TiltSeries.scatter = scatter
        ExitWave.apply_ctf = apply_ctf
        ExitWave.intensity = intensity
        Image.detect = detect
        ExitWave.pixels = pixels

    sources = [
        Path(args.checkout) / "src/specter/imaging/_tilt.py",
        Path(args.checkout) / "src/specter/pipelines/_tiltseries.py",
    ]
    result = {
        "checkout": args.checkout,
        "loaded_tilt_series_source": str(source_path),
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

        old_path = Path(args.paired_baseline) / "src/specter/imaging/_tilt.py"
        name = "specter.imaging._tilt_benchmark_baseline"
        spec = importlib.util.spec_from_file_location(name, old_path)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[name] = baseline
        spec.loader.exec_module(baseline)
        old_scatter = baseline.TiltSeries.scatter
        series = held_series.pop()

        def generate(scatter_tilt):
            images = []
            with torch.no_grad():
                for i in range(len(series)):
                    wave = scatter_tilt(series, i, device=args.device)
                    a, k = calls["apply_ctf"][i]
                    wave = apply_ctf(wave, *a, **k)
                    a, k = calls["intensity"][i]
                    image = intensity(wave, *a, **k)
                    a, k = calls["detect"][i]
                    images.append(detect(image, *a, **{**k, "seed": 173}).array.cpu())
            return torch.cat(images)

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
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                images = generate(old_scatter if variant == "before" else scatter)
                torch.cuda.synchronize()
                pair_results[variant] = {
                    "seconds": time.perf_counter() - t0,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
                saved[variant] = images
                if pair == 0:
                    with mrcfile.new(
                        output / f"paired-{variant}.mrcs", overwrite=True
                    ) as mrc:
                        mrc.set_data(images.numpy())
                        mrc.voxel_size = series.voxel_size
                print(
                    f"paired {pair + 1}/3 {variant}: {pair_results[variant]}",
                    flush=True,
                )
            a, b = saved["before"], saved["after"]
            equal = torch.equal(a, b)
            squared_diff = 0.0
            squared_ref = 0.0
            max_abs = 0.0
            for ai, bi in zip(a, b):
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
                    "same_series": True,
                    "shape": list(series.volume.shape),
                    "baseline_path": str(old_path),
                    "candidate_path": str(sources[0]),
                    "trials": trials,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
