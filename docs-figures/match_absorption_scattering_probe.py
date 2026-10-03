"""Measure complex assembly and propagation at full native matching geometry."""

import argparse
import gc
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--nz", type=int, default=1641)
    p.add_argument("--nxy", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    import torch
    from specter.scattering import Scattering

    torch.set_num_threads(4)
    torch.cuda.set_device(a.device)
    torch.manual_seed(41)
    shape = (a.batch, a.nz, a.nxy, a.nxy)
    real = torch.rand(shape, device=a.device) * 0.1
    absorption = torch.rand(shape, device=a.device) * 0.02
    model = Scattering(a.nxy, 0.731, 300.0, progressbars=False).to(a.device)
    results = []
    reference = None
    with torch.no_grad():
        # Warm the FFT plan without allocating a whole complex volume.
        model(real[:, :8], absorption_potential=absorption[:, :8])
        for variant in ["complex", "paired", "paired", "complex"]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            if variant == "complex":
                wave = model(torch.complex(real, absorption))
            else:
                wave = model(real, absorption_potential=absorption)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            cpu_wave = wave.cpu()
            if reference is None:
                reference = cpu_wave
            row = dict(
                variant=variant,
                shape=shape,
                seconds=elapsed,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                bitwise_equal=torch.equal(reference, cpu_wave),
                max_abs_difference=float((reference - cpu_wave).abs().max()),
            )
            print(json.dumps(row), flush=True)
            results.append(row)
            del wave, cpu_wave
    Path(a.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
