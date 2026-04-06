"""
Benchmark: slice_batch_size impact on IterativeScattering performance.

Tests a 500x2000x2000 volume at 30° tilt with various slice_batch_size values.
Uses cuda:1, multislice model (the most common and expensive).
"""

import gc
import sys
import time

import torch

sys.path.insert(0, "/mnt/cbis/home/e0788253/czii/specter/src")

from specter.rotations import build_affine_matrix
from specter.scattering import IterativeScattering


def main():
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)

    nz, ny, nx = 500, 2000, 2000
    nxy = 2000  # output image size
    pixel_size = 1.0
    energy = 300.0
    dose = 1.0
    angle_deg = 30.0

    print(f"Volume: {nz}x{ny}x{nx}, angle: {angle_deg}°, nxy: {nxy}")
    print(f"GPU: {torch.cuda.get_device_properties(device).name}")
    print(
        f"GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB"
    )

    # Create volume on CPU to save GPU memory (IterativeScattering moves slices)
    V = torch.randn(1, nz, ny, nx)
    print(f"Volume on CPU: {V.numel() * 4 / 1e9:.2f} GB")

    # Build affine matrix for 30° tilt around X
    angle_rad = torch.deg2rad(torch.tensor(angle_deg))
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
    R = R.unsqueeze(0)  # (1, 3, 3)
    theta = build_affine_matrix(R).to(device)  # (1, 3, 4)

    # Test different slice_batch_size values
    batch_sizes = [1, 5, 10, 25, 50, 100]

    print(f"\n{'batch_size':>12s} {'Time (s)':>10s} {'Peak GPU (GB)':>14s}")
    print("-" * 40)

    for bs in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        scat = IterativeScattering(
            nxy=nxy,
            pixel_size=pixel_size,
            energy=energy,
            dose_per_angstrom=dose,
            scattering_model="multislice",
            progressbars=False,
        ).to(device)

        # Warmup (skip for large volumes, just measure cold)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        try:
            with torch.no_grad():
                exitwave = scat.forward(V, theta, slice_batch_size=bs)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"{bs:>12d} {elapsed:>10.2f} {peak:>14.2f}")
            del exitwave
        except RuntimeError:
            elapsed = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"{bs:>12d} {'OOM':>10s} {peak:>14.2f}")

        del scat
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
