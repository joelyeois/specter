"""
Hybrid 3D Rotation: Welling FFT-Shear + grid_sample Fallback

Combines two complementary methods:
1. Welling et al. (2006) four-shear FFT decomposition - exact, no interpolation
2. PyTorch grid_sample - stable for all angles, with trilinear interpolation

Automatically selects the best method based on rotation characteristics.
For cryo-EM: ~99% of rotations use exact FFT method.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
import warnings

from specter.welling_rotation import (
    euler_to_quaternion,
    select_best_decomposition,
    apply_four_shear_decomposition,
    quaternion_to_matrix,
)


class RotationMethod:
    """Enum for rotation method selection."""

    WELLING_FFT = "welling_fft"
    GRID_SAMPLE = "grid_sample"


def rotate_volume_hybrid(
    vol: torch.Tensor, q: np.ndarray, method: str = RotationMethod.WELLING_FFT
) -> Tuple[torch.Tensor, str]:
    """
    Rotate a 3D volume using the best available method.

    Automatically selects between Welling FFT (exact) and grid_sample (stable),
    falling back gracefully when needed.

    Parameters
    ----------
    vol : torch.Tensor
        3D volume (D, H, W) on any device (CPU/GPU).
    q : np.ndarray
        Normalized quaternion (x, y, z, w).
    method : str
        Preferred method ('welling_fft' or 'grid_sample').
        With 'welling_fft', will auto-fallback if singular.

    Returns
    -------
    tuple
        (rotated_volume, method_used)
    """
    q = q / np.linalg.norm(q)

    # Try Welling FFT first if requested
    if method == RotationMethod.WELLING_FFT:
        try:
            shear_params, decomp_name, merit = select_best_decomposition(q)
            vol_rotated = apply_four_shear_decomposition(vol, shear_params, decomp_name)
            return vol_rotated, f"{RotationMethod.WELLING_FFT} ({decomp_name})"
        except ValueError as e:
            # Fall back to grid_sample for singular cases
            if "180°" in str(e) or "singular" in str(e).lower():
                warnings.warn(
                    "Welling decomposition singular, falling back to grid_sample. "
                    "(This is normal for certain pure axis rotations)"
                )
                return rotate_volume_grid_sample(vol, q), RotationMethod.GRID_SAMPLE
            else:
                raise

    # Use grid_sample directly if requested
    elif method == RotationMethod.GRID_SAMPLE:
        return rotate_volume_grid_sample(vol, q), RotationMethod.GRID_SAMPLE

    else:
        raise ValueError(f"Unknown rotation method: {method}")


def rotate_volume_grid_sample(vol: torch.Tensor, q: np.ndarray) -> torch.Tensor:
    """
    Rotate volume using grid_sample (fallback for singular cases).

    Stable for all rotations (including 180° pure axis), but uses
    trilinear interpolation (slight blur).

    Parameters
    ----------
    vol : torch.Tensor
        3D volume (D, H, W).
    q : np.ndarray
        Normalized quaternion (x, y, z, w).

    Returns
    -------
    torch.Tensor
        Rotated volume.
    """
    N = vol.shape[0]
    device = vol.device

    # Convert quaternion to rotation matrix
    R = quaternion_to_matrix(q)

    # Coordinate system conversion: array indices (D,H,W) ↔ grid_sample (x,y,z)
    P = torch.tensor(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32, device=device
    )
    R_t = torch.tensor(R, dtype=torch.float32, device=device)
    R_gs = P @ R_t @ P

    # Build normalized output grid
    lin = torch.linspace(-1, 1, N, device=device)
    gz, gy, gx = torch.meshgrid(lin, lin, lin, indexing="ij")
    grid_out = torch.stack([gx, gy, gz], dim=-1)

    # Source coordinates
    src_grid = (grid_out.view(-1, 3) @ R_gs).view(1, N, N, N, 3)

    # Apply grid_sample
    out = F.grid_sample(
        vol.view(1, 1, N, N, N),
        src_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return out.squeeze()


def rotate_euler_hybrid(
    vol: torch.Tensor,
    ax_deg: float,
    ay_deg: float,
    az_deg: float,
    method: str = RotationMethod.WELLING_FFT,
) -> Tuple[torch.Tensor, str]:
    """
    Rotate volume by Euler angles using hybrid method.

    Parameters
    ----------
    vol : torch.Tensor
        3D volume (D, H, W).
    ax_deg, ay_deg, az_deg : float
        Roll (X), pitch (Y), yaw (Z) in degrees.
    method : str
        Preferred method ('welling_fft' or 'grid_sample').

    Returns
    -------
    tuple
        (rotated_volume, method_used)
    """
    q = euler_to_quaternion(ax_deg, ay_deg, az_deg)
    return rotate_volume_hybrid(vol, q, method)


def estimate_rotation_complexity(q: np.ndarray) -> float:
    """
    Estimate whether a rotation is "simple" or "complex".

    Simple rotations (like single-axis) may need fallback.
    Complex rotations typically use Welling FFT successfully.

    Returns
    -------
    float
        Complexity score (0-1). Higher = more complex.
    """
    X, Y, Z, W = q
    # Count number of non-zero components
    non_zero_count = sum([abs(c) > 0.01 for c in [X, Y, Z]])
    return min(1.0, non_zero_count / 3.0)


def get_rotation_info(q: np.ndarray) -> dict:
    """
    Get detailed information about a rotation.

    Parameters
    ----------
    q : np.ndarray
        Quaternion (x, y, z, w).

    Returns
    -------
    dict
        Information including: method_likely, singularity_risk, etc.
    """
    q = q / np.linalg.norm(q)
    X, Y, Z, W = q

    info = {
        "quaternion": q,
        "axis_aligned_count": sum([abs(c) < 0.01 for c in [X, Y, Z]]),
        "complexity": estimate_rotation_complexity(q),
    }

    try:
        params, decomp, merit = select_best_decomposition(q)
        info["method_recommended"] = RotationMethod.WELLING_FFT
        info["decomposition"] = decomp
        info["merit"] = float(merit)
    except ValueError:
        info["method_recommended"] = RotationMethod.GRID_SAMPLE
        info["reason"] = "Welling decomposition singular"
        info["merit"] = np.inf

    return info
