import torch

from specter.rotations import VolumeRotator, build_affine_matrix, rotate_volume


def rotation_x_matrix(angle_deg: float, device: torch.device, dtype: torch.dtype):
    theta = torch.deg2rad(torch.tensor(angle_deg, device=device, dtype=dtype))
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        device=device,
        dtype=dtype,
    )


def compare_methods(
    nz: int = 33,
    ny: int = 35,
    nx: int = 37,
    angles_deg=(0.0, 15.0, -30.0, 47.5),
    align_corners: bool = True,
    atol: float = 2e-5,
    rtol: float = 1e-4,
):
    device = torch.device("cpu")
    dtype = torch.float32

    # Odd dimensions keep slice indices centered on integer locations.
    if nz % 2 == 0 or ny % 2 == 0 or nx % 2 == 0:
        raise ValueError("Use odd nz/ny/nx for this equivalence check.")

    torch.manual_seed(0)
    volume = torch.randn(nz, ny, nx, device=device, dtype=dtype)

    R = torch.stack(
        [rotation_x_matrix(a, device=device, dtype=dtype) for a in angles_deg]
    )
    theta = build_affine_matrix(R)

    vol_ref = rotate_volume(
        volume,
        theta,
        origin="relion",
        padding_mode="border",
        align_corners=align_corners,
    )

    rotator = VolumeRotator(
        nz=nz,
        ny=ny,
        nx=nx,
        origin="relion",
        padding_mode="border",
        align_corners=align_corners,
    )

    slice_indices = torch.arange(nz, device=device) - (nz // 2)
    vol_slices = rotator.sample_rotated_slices(
        volume,
        theta,
        slice_indices=slice_indices,
        roi_size=(ny, nx),
        padding_mode="border",
    )

    max_abs_diff = (vol_ref - vol_slices).abs().max().item()
    mean_abs_diff = (vol_ref - vol_slices).abs().mean().item()

    print(f"max_abs_diff={max_abs_diff:.3e}")
    print(f"mean_abs_diff={mean_abs_diff:.3e}")

    torch.testing.assert_close(vol_slices, vol_ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    compare_methods(align_corners=True)
    print("PASS: sample_rotated_slices matches rotate_volume for tested angles.")
