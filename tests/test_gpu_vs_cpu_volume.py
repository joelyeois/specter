"""
Test: V on GPU vs CPU for IterativeScattering grid_sample performance.

Compares:
  1. V on CPU (current behavior) with slice_batch_size=1
  2. V on GPU with slice_batch_size=1
  3. V on GPU with slice_batch_size=50

Uses cuda:1, 500x2000x2000 volume at 30° tilt.
"""

import gc
import sys
import time

import torch

sys.path.insert(0, "/mnt/cbis/home/e0788253/czii/specter/src")

from specter.rotations import build_affine_matrix
from specter.scattering import IterativeScattering


def run_test(V, theta, device, slice_batch_size, label):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    scat = IterativeScattering(
        nxy=2000,
        pixel_size=1.0,
        energy=300.0,
        dose_per_angstrom=1.0,
        scattering_model="multislice",
        progressbars=False,
    ).to(device)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    try:
        with torch.no_grad():
            exitwave = scat.forward(V, theta, slice_batch_size=slice_batch_size)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  {label:<35s} {elapsed:>8.1f}s  peak={peak:.2f} GB")
        del exitwave
    except RuntimeError as e:
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  {label:<35s} {'OOM':>8s}  peak={peak:.2f} GB  ({e})")

    del scat
    gc.collect()
    torch.cuda.empty_cache()


def main():
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)

    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")

    nz, ny, nx = 500, 2000, 2000

    # Build affine matrix for 30° tilt
    angle_rad = torch.deg2rad(torch.tensor(30.0))
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32).unsqueeze(
        0
    )
    theta = build_affine_matrix(R).to(device)

    print(f"Volume: {nz}x{ny}x{nx} = {nz * ny * nx * 4 / 1e9:.2f} GB (float32)")
    print()

    # Test 1: V on CPU, batch_size=1 (baseline - current behavior)
    V_cpu = torch.randn(1, nz, ny, nx)
    print("Test 1: V on CPU")
    run_test(V_cpu, theta, device, slice_batch_size=1, label="CPU, batch_size=1")

    # Test 2: V on GPU, batch_size=1
    print("\nTest 2: V on GPU")
    gc.collect()
    torch.cuda.empty_cache()
    V_gpu = V_cpu.to(device)
    gpu_mem_used = torch.cuda.memory_allocated(device) / 1e9
    print(f"  V allocated on GPU: {gpu_mem_used:.2f} GB")
    run_test(V_gpu, theta, device, slice_batch_size=1, label="GPU, batch_size=1")

    # Test 3: V on GPU, batch_size=50
    print("\nTest 3: V on GPU, larger batch")
    run_test(V_gpu, theta, device, slice_batch_size=50, label="GPU, batch_size=50")

    # Test 4: V on GPU, batch_size=100
    print("\nTest 4: V on GPU, even larger batch")
    run_test(V_gpu, theta, device, slice_batch_size=100, label="GPU, batch_size=100")

    del V_gpu, V_cpu
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
