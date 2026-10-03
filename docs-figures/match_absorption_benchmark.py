"""Measure the actual matching CLI, recording every probe and native battery.

Optional frozen templates control the existing GPU atomic-build roundoff.
Potential builds still execute on cache misses; replay copies into the CPU
result before it reaches simulation. Template I/O is outside the timed CLI.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkout", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--capture-potentials")
    p.add_argument("--replay-potentials")
    a = p.parse_args()
    sys.path.insert(0, a.checkout + "/src")
    import psutil
    import torch
    from specter.cli._cli import cli
    from specter.pipelines import _match, _particles

    torch.set_num_threads(4)
    torch.cuda.set_device(a.device)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    frozen = {}
    if a.replay_potentials:
        for file in Path(a.replay_potentials).glob("*.pt"):
            frozen[file.stem] = torch.load(file, weights_only=True)
        original_potential = _particles._structure_and_potential

        def replay(config, pixel_size, build):
            pdb, volume = original_potential(config, pixel_size, build)
            if build:
                key = f"{config.n_pixels}-{round(float(pixel_size), 6):.8f}"
                volume.copy_(frozen[key])
            return pdb, volume

        _particles._structure_and_potential = replay
    stages = []
    old_generate = _particles._generate_single

    def generate(model, n, batchsize, track, **flags):
        for module in model.modules():
            if hasattr(module, "progressbars"):
                module.progressbars = False
        stages[-1].update(
            particles=n,
            batchsize=batchsize,
            nxy=model.nxy,
            padded_nxy=model.pad_nxy,
            nz=model.nz,
        )
        return old_generate(model, n, batchsize, track, **flags)

    _particles._generate_single = generate
    original_simulate = _match._simulate_job

    def simulate(kwargs, *args, **kw):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        row = dict(name=kwargs["filename"], seed=kwargs["seed"])
        stages.append(row)
        t = time.perf_counter()
        result = original_simulate(kwargs, *args, **kw)
        torch.cuda.synchronize()
        row.update(
            seconds=time.perf_counter() - t,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        print("MEASURED_STAGE", json.dumps(row), flush=True)
        return result

    _match._simulate_job = simulate
    old_render = _match.render_report
    summaries = []

    def render(report, *args, **kwargs):
        summaries.append(report.summary())
        return old_render(report, *args, **kwargs)

    _match.render_report = render
    stop = threading.Event()
    peak = {"rss_bytes": 0, "driver_bytes": 0}
    process = psutil.Process()

    def monitor():
        while not stop.is_set():
            peak["rss_bytes"] = max(peak["rss_bytes"], process.memory_info().rss)
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
                fields = [s.strip() for s in line.split(",")]
                if (
                    len(fields) == 2
                    and fields[0] == str(os.getpid())
                    and fields[1].isdigit()
                ):
                    peak["driver_bytes"] = max(
                        peak["driver_bytes"], int(fields[1]) * 2**20
                    )
            stop.wait(0.3)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        cli.main(
            args=[
                "match",
                "particles",
                "--config",
                a.config,
                "--device",
                a.device,
                "--output_dir",
                str(out),
                "--seed",
                str(a.seed),
            ],
            standalone_mode=False,
        )
        torch.cuda.synchronize()
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join(timeout=3)
    result = dict(
        checkout=a.checkout,
        config=a.config,
        seed=a.seed,
        device=a.device,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cli_seconds=elapsed,
        stages=stages,
        peak_allocated_bytes=max(s["peak_allocated_bytes"] for s in stages),
        peak_reserved_bytes=max(s["peak_reserved_bytes"] for s in stages),
        driver_peak_bytes=peak["driver_bytes"],
        host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        replay_potentials=a.replay_potentials,
        report=summaries[-1],
    )
    (out / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "MEASURED_CLI",
        json.dumps({k: v for k, v in result.items() if k not in ["report", "stages"]}),
        flush=True,
    )
    if a.capture_potentials:
        capture = Path(a.capture_potentials)
        capture.mkdir(parents=True, exist_ok=False)
        for key, (_, volume) in _particles._POTENTIAL_CACHE.items():
            # Cache key contains n_pixels then rounded pixel size at slots 5/6.
            torch.save(volume, capture / f"{key[5]}-{key[6]:.8f}.pt")


if __name__ == "__main__":
    main()
