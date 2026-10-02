"""Run the shipped tomogram CLI with stage and allocator measurements."""

import argparse
import hashlib
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
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--seed", type=int, default=103)
    frozen = p.add_mutually_exclusive_group()
    frozen.add_argument("--capture-components")
    frozen.add_argument("--replay-components")
    a = p.parse_args()
    sys.path.insert(0, a.checkout + "/src")
    import torch
    from specter.cli._cli import cli
    from specter.specimen.tomogram import TomogramSpecimenGenerator

    if a.capture_components or a.replay_components:
        from specter.potential import PotentialBuilder
        from specter.specimen._carbon import CarbonFilmGenerator, CarbonFilmInstance

        cache = Path(a.capture_components or a.replay_components)
        cache.mkdir(parents=True, exist_ok=True)
        potential_forward = PotentialBuilder.forward
        carbon_generate = CarbonFilmGenerator.generate

        def fixed_potential(self, coordinates, method="analytic", conv_backend=None):
            digest = hashlib.sha256()
            for tensor in [coordinates, self.atomic_numbers, self.b_factors]:
                if tensor is not None:
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            digest.update(
                repr(
                    (
                        self.nx,
                        self.ny,
                        self.nz,
                        self.dx,
                        self.parameterization,
                        self.atom_species,
                        method,
                        conv_backend,
                    )
                ).encode()
            )
            path = cache / (digest.hexdigest() + ".pt")
            if path.exists():
                return torch.load(path, weights_only=True).to(self.device)
            if a.replay_components:
                raise RuntimeError(f"Missing frozen potential {path}")
            value = potential_forward(self, coordinates, method, conv_backend)
            torch.save(value.detach().cpu(), path)
            return value

        def fixed_carbon(self, *args, **kwargs):
            path = cache / "carbon.pt"
            if path.exists():
                return CarbonFilmInstance(
                    torch.load(path, weights_only=True).to(self.device)
                )
            if a.replay_components:
                raise RuntimeError("Missing frozen carbon")
            value = carbon_generate(self, *args, **kwargs)
            torch.save(value.density.detach().cpu(), path)
            return value

        PotentialBuilder.forward = fixed_potential
        CarbonFilmGenerator.generate = fixed_carbon

    torch.set_num_threads(4)
    torch.cuda.set_device(a.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "checkout": a.checkout,
        "device": a.device,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "seed": a.seed,
        "threads": 4,
        "frozen_components": a.capture_components or a.replay_components,
        "stages": [],
    }
    done = threading.Event()
    driver_peak = [0]

    def monitor():
        while not done.is_set():
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            for line in proc.stdout.splitlines():
                fields = [x.strip() for x in line.split(",")]
                if (
                    len(fields) == 2
                    and fields[0] == str(os.getpid())
                    and fields[1].isdigit()
                ):
                    driver_peak[0] = max(driver_peak[0], int(fields[1]) * 2**20)
            done.wait(0.3)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def wrap(name):
        original = getattr(TomogramSpecimenGenerator, name)

        def measured(self, *args, **kwargs):
            torch.cuda.synchronize()
            t = time.perf_counter()
            value = original(self, *args, **kwargs)
            torch.cuda.synchronize()
            record = {
                "stage": name,
                "seconds": time.perf_counter() - t,
                "allocated_bytes": torch.cuda.memory_allocated(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            result["stages"].append(record)
            print("BENCH_STAGE", json.dumps(record), flush=True)
            if name == "generate":
                result.update(
                    {
                        "shape": list(value.shape),
                        "accumulator_device": str(value.device),
                        "protein_instances": len(self.placements),
                        "filament_instances": len(self.filament_instances),
                        "microtubule_dimers": len(self.microtubule_dimer_instances),
                        "membrane_instances": len(self.placed_membrane_instances),
                        "transmembrane_instances": len(self.transmembrane_placements),
                    }
                )
            return value

        setattr(TomogramSpecimenGenerator, name, measured)

    for name in [
        "_stage_carbon",
        "_stage_membranes",
        "_stage_filaments",
        "_stage_beads",
        "_load_structures",
        "_stage_species",
        "generate",
    ]:
        wrap(name)
    torch.cuda.synchronize()
    t = time.perf_counter()
    try:
        cli(
            prog_name="specter",
            standalone_mode=False,
            args=[
                "build",
                "tomogram",
                "--config",
                a.checkout + "/configs/tomogram.toml",
                "--device",
                a.device,
                "--seed",
                str(a.seed),
                "--output_dir",
                str(out),
            ],
        )
        torch.cuda.synchronize()
    finally:
        done.set()
        thread.join(timeout=2)
    result.update(
        {
            "cli_seconds": time.perf_counter() - t,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "sampled_driver_peak_bytes": driver_peak[0],
            "host_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        }
    )
    try:
        import cupy

        result["cupy_reserved_bytes_end"] = cupy.get_default_memory_pool().total_bytes()
    except ImportError:
        result["cupy_reserved_bytes_end"] = 0
    (out / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
