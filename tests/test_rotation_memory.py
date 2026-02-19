"""
Test: Can a full 500x2000x2000 volume be rotated on GPU?

When tilting at an angle, the output Z dimension grows because the bounding
box of the tilted volume is larger. This script:
1. Computes nz_new for various tilt angles
2. Estimates memory required
3. Attempts the actual rotation on cuda:1
"""

import torch
import math
import gc


def compute_nz_new(nz, ny, nx, angle_deg):
    """Compute the new Z dimension after rotating by angle_deg around X axis."""
    angle_rad = math.radians(angle_deg)
    c, s = math.cos(angle_rad), math.sin(angle_rad)

    # Corners of the volume in pixel units relative to center
    corners = torch.tensor(
        [
            [-nx / 2, -ny / 2, -nz / 2],
            [nx / 2, -ny / 2, -nz / 2],
            [-nx / 2, ny / 2, -nz / 2],
            [nx / 2, ny / 2, -nz / 2],
            [-nx / 2, -ny / 2, nz / 2],
            [nx / 2, -ny / 2, nz / 2],
            [-nx / 2, ny / 2, nz / 2],
            [nx / 2, ny / 2, nz / 2],
        ]
    )

    # Rotation matrix around X axis
    R = torch.tensor(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ]
    )

    # Rotate corners and find Z extent
    rotated = corners @ R.T
    z_min = rotated[:, 2].min().item()
    z_max = rotated[:, 2].max().item()
    nz_new = int(math.ceil(z_max - z_min))
    return max(1, nz_new)


def main():
    device = torch.device("cuda:1")
    props = torch.cuda.get_device_properties(device)
    total_mem_gb = props.total_memory / 1e9
    print(f"GPU: {props.name}")
    print(f"Total GPU memory: {total_mem_gb:.1f} GB")

    nz, ny, nx = 500, 2000, 2000
    print(f"\nInput volume: {nz} x {ny} x {nx}")
    input_bytes = nz * ny * nx * 4  # float32
    print(f"Input volume size: {input_bytes / 1e9:.2f} GB")

    # Show nz_new and memory estimates for various tilt angles
    print(
        f"\n{'Angle':>8s} {'nz_new':>8s} {'Output (GB)':>12s} {'Grid (GB)':>12s} {'Total (GB)':>12s}"
    )
    print("-" * 56)
    for angle in [0, 10, 20, 30, 45, 60]:
        nz_new = compute_nz_new(nz, ny, nx, angle)
        output_bytes = nz_new * ny * nx * 4  # float32
        # grid_sample grid: (B, nz_new, ny, nx, 3) float32
        grid_bytes = 1 * nz_new * ny * nx * 3 * 4
        total = input_bytes + output_bytes + grid_bytes
        print(
            f"{angle:>8d} {nz_new:>8d} {output_bytes/1e9:>12.2f} {grid_bytes/1e9:>12.2f} {total/1e9:>12.2f}"
        )

    # Attempt actual rotation at 30 degrees
    test_angle = 30
    nz_new = compute_nz_new(nz, ny, nx, test_angle)
    print(
        f"\n--- Attempting full rotation at {test_angle}° (nz_new={nz_new}) on {device} ---"
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        # Create input volume
        mem_before = torch.cuda.memory_allocated(device)
        V = torch.randn(1, nz, ny, nx, device=device)
        mem_after_input = torch.cuda.memory_allocated(device)
        print(f"Input allocated: {(mem_after_input - mem_before)/1e9:.2f} GB")

        # Build affine matrix for rotation around X
        angle_rad = math.radians(test_angle)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = torch.tensor(
            [[1, 0, 0], [0, c, -s], [0, s, c]], device=device, dtype=torch.float32
        )
        theta = torch.zeros(1, 3, 4, device=device)
        theta[0, :3, :3] = R

        # Create VolumeRotator for the OUTPUT dimensions
        from cryosim.rotations import VolumeRotator

        rotator = VolumeRotator(
            nz=nz, ny=ny, nx=nx, origin="relion", padding_mode="zeros"
        ).to(device)

        # Try sample_rotated_slices for ALL slices at once
        slice_indices = torch.arange(nz_new, device=device).float() - (nz_new - 1) / 2.0

        print(f"Attempting to sample {nz_new} slices at once...")
        mem_before_rot = torch.cuda.memory_allocated(device)

        rotated_slices = rotator.sample_rotated_slices(
            V,
            theta,
            slice_indices=slice_indices,
            roi_size=(ny, nx),
            padding_mode="zeros",
        )

        mem_after_rot = torch.cuda.memory_allocated(device)
        peak = torch.cuda.max_memory_allocated(device)

        print(f"Output shape: {rotated_slices.shape}")
        print(f"Output allocated: {(mem_after_rot - mem_before_rot)/1e9:.2f} GB")
        print(f"Peak GPU memory: {peak/1e9:.2f} GB")
        print("SUCCESS!")

        del rotated_slices, V, rotator

    except RuntimeError as e:
        print(f"FAILED: {e}")
        peak = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU memory before OOM: {peak/1e9:.2f} GB")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
