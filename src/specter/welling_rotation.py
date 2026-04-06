"""
Four-Shear FFT Rotation based on Welling et al. (2006)
"Rotation of 3D volumes by Fourier-interpolated shears"

Provides robust 3D rotation via FFT-based shearing with automatic
singularity avoidance through intelligent decomposition selection.
"""

import numpy as np
import torch
from typing import Tuple


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to 3x3 rotation matrix.

    Parameters
    ----------
    q : np.ndarray
        Quaternion with shape (4,) in (x, y, z, w) format.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)],
        ]
    )


def euler_to_quaternion(ax_deg: float, ay_deg: float, az_deg: float) -> np.ndarray:
    """
    Convert Euler angles to quaternion (ZYX convention).

    Parameters
    ----------
    ax_deg, ay_deg, az_deg : float
        Roll (X), pitch (Y), yaw (Z) angles in degrees.

    Returns
    -------
    np.ndarray
        Quaternion in (x, y, z, w) format.
    """
    ax, ay, az = np.radians(ax_deg), np.radians(ay_deg), np.radians(az_deg)

    cy, sy = np.cos(ay / 2), np.sin(ay / 2)
    cp, sp = np.cos(ax / 2), np.sin(ax / 2)
    cr, sr = np.cos(az / 2), np.sin(az / 2)

    return np.array(
        [
            sp * cy * cr - cp * sy * sr,
            cp * sy * cr + sp * cy * sr,
            cp * cy * sr - sp * sy * cr,
            cp * cy * cr + sp * sy * sr,
        ]
    )


def extract_four_shear_params(X: float, Y: float, Z: float, W: float) -> np.ndarray:
    """
    Extract four-shear parameters from quaternion components (Equation 5b).

    This extracts parameters for the forward decomposition:
        R = S_y(a,b) S_z(c,d) S_x(e,f) S_y(g,h)

    Parameters
    ----------
    X, Y, Z, W : float
        Quaternion components (already separated for cyclic permutation).

    Returns
    -------
    np.ndarray
        Shear parameters [a, b, c, d, e, f, g, h], or NaN array if singular.
    """
    # Compute the two critical denominators from the paper
    denom1 = Y * Z - X * W
    denom2 = X * Y - Z * W

    # Check for singularities
    if np.abs(denom1) < 1e-10 or np.abs(denom2) < 1e-10:
        return np.full(8, np.nan)

    try:
        a = (X**2 + Y**2) / denom1
        b = (Y**2 - Z**2) / denom2 - 2 * Y * Z * (X**2 + Y**2) / (denom2 * denom1)
        c = 2 * Y * Z / denom2
        d = 2 * (X * W - Y * Z)
        e = 2 * (X * Y - Z * W)
        f = -2 * X * Y / denom1
        g = (X**2 - Y**2) / denom1 + 2 * X * Y * (Y**2 + Z**2) / (denom1 * denom2)
        h = (-1 + X**2 + W**2) / denom2

        params = np.array([a, b, c, d, e, f, g, h])

        # Check for NaN or inf
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            return np.full(8, np.nan)

        return params
    except Exception:
        return np.full(8, np.nan)


def compute_figure_of_merit(params: np.ndarray) -> float:
    """
    Compute figure of merit for shear parameters (Equation 11 from paper).

    Uses: fm = |a| + |b| + |c| + |d| + |e| + |f| + |g| + |h|

    Lower values are better (smaller shears = more numerically stable).

    Parameters
    ----------
    params : np.ndarray
        Shear parameters [a, b, c, d, e, f, g, h].

    Returns
    -------
    float
        Figure of merit (sum of absolute values).
    """
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        return np.inf

    return np.sum(np.abs(params))


def select_best_decomposition(q: np.ndarray) -> Tuple[np.ndarray, str, float]:
    """
    Select the best four-shear decomposition from 6 patterns.

    Algorithm from Section 8 of Welling et al. (2006):
    - Try all 3 forward decompositions (cyclic permutations of axes)
    - Try all 3 backward complements (use negated quaternion for different octant)
    - Select the one with smallest figure of merit

    Key insight from Section 6.2: For any rotation, at least one of the
    forward or backward decompositions will be non-singular.

    Parameters
    ----------
    q : np.ndarray
        Normalized quaternion (x, y, z, w).

    Returns
    -------
    tuple
        (shear_params, decomposition_name, figure_of_merit)

    Raises
    ------
    ValueError
        If all decompositions are singular (only occurs for pure 180° axis rotations).
    """
    X, Y, Z, W = q

    # Ensure we have W >= 0 (use the quaternion that represents the same rotation)
    if W < 0:
        X, Y, Z, W = -X, -Y, -Z, -W

    # Six decomposition patterns with proper cyclic permutations
    patterns = {
        "S_y S_z S_x S_y (forward 1)": (X, Y, Z, W),
        "S_z S_x S_y S_z (forward 2)": (Y, Z, X, W),  # cyclic perm: X→Y→Z→X
        "S_x S_y S_z S_x (forward 3)": (Z, X, Y, W),  # cyclic perm: X→Z→Y→X
        "S_y S_x S_z S_y (backward 1)": (X, Y, Z, -W),  # complement of forward 1
        "S_z S_y S_x S_z (backward 2)": (Y, Z, X, -W),  # complement of forward 2
        "S_x S_z S_y S_x (backward 3)": (Z, X, Y, -W),  # complement of forward 3
    }

    best_merit = np.inf
    best_name = None
    best_params = None

    for name, (x, y, z, w) in patterns.items():
        # Extract parameters for this decomposition
        params = extract_four_shear_params(x, y, z, w)
        merit = compute_figure_of_merit(params)

        if merit < best_merit:
            best_merit = merit
            best_name = name
            best_params = params

    if best_name is None or np.isinf(best_merit):
        raise ValueError(
            f"Cannot decompose quaternion {q}. "
            "This is a pure 180° rotation around a coordinate axis, "
            "which has no valid four-shear decomposition (Welling et al. 2006, Section 6.2). "
            "Fall back to grid_sample for this case."
        )

    return best_params, best_name, best_merit


def fft_shear(
    vol: torch.Tensor, shear_factor: float, shift_axis: int, coord_axis: int
) -> torch.Tensor:
    """
    Apply FFT-based shear to volume (Algorithm from Section 3 of paper).

    Shifts along shift_axis by: shear_factor * (coord - N_coord/2)

    Parameters
    ----------
    vol : torch.Tensor
        3D volume (D, H, W).
    shear_factor : float
        Shear amount (typically between -1 and 1).
    shift_axis : int
        Axis along which to shift (0=D, 1=H, 2=W).
    coord_axis : int
        Axis whose coordinate determines shift amount.

    Returns
    -------
    torch.Tensor
        Sheared volume, same shape as input.
    """
    N_shift = vol.shape[shift_axis]
    N_coord = vol.shape[coord_axis]
    device = vol.device

    # FFT along shift axis
    V = torch.fft.rfft(vol, dim=shift_axis)
    n_freq = V.shape[shift_axis]

    # Frequency grid
    k = torch.arange(n_freq, dtype=torch.float32, device=device)
    k_shape = [1, 1, 1]
    k_shape[shift_axis] = n_freq
    k = k.view(k_shape)

    # Coordinate grid (centered)
    coord = torch.arange(N_coord, dtype=torch.float32, device=device) - N_coord / 2.0
    c_shape = [1, 1, 1]
    c_shape[coord_axis] = N_coord
    coord = coord.view(c_shape)

    # Phase shift
    phase = torch.exp(
        -2j * np.pi * k * (shear_factor * coord).to(torch.complex64) / N_shift
    )

    # Apply phase shift and inverse FFT
    return torch.fft.irfft(V * phase, n=N_shift, dim=shift_axis)


def apply_four_shear_decomposition(
    vol: torch.Tensor, shear_params: np.ndarray, decomposition: str
) -> torch.Tensor:
    """
    Apply four-shear decomposition to volume via FFT.

    Parameters
    ----------
    vol : torch.Tensor
        3D volume (D, H, W).
    shear_params : np.ndarray
        Eight shear parameters [a, b, c, d, e, f, g, h].
    decomposition : str
        Name of decomposition pattern (for choosing axis order).

    Returns
    -------
    torch.Tensor
        Rotated volume.
    """
    a, b, c, d, e, f, g, h = shear_params

    # Map decomposition name to axis sequence
    # Paper uses notation S_y(a,b) means: S_y with params (a,b)
    # Axes: 0=D (depth), 1=H (height), 2=W (width)

    if "S_y S_z S_x S_y" in decomposition:
        # Forward 1: S_y(g,h) S_x(e,f) S_z(c,d) S_y(a,b)
        vol = fft_shear(vol, g, 1, 0)  # S_y, shift along H, coord along D
        vol = fft_shear(vol, h, 1, 2)  # S_y, shift along H, coord along W
        vol = fft_shear(vol, e, 2, 0)  # S_x, shift along W, coord along D
        vol = fft_shear(vol, f, 2, 1)  # S_x, shift along W, coord along H
        vol = fft_shear(vol, c, 0, 2)  # S_z, shift along D, coord along W
        vol = fft_shear(vol, d, 0, 1)  # S_z, shift along D, coord along H
        vol = fft_shear(vol, a, 1, 0)  # S_y, shift along H, coord along D
        vol = fft_shear(vol, b, 1, 2)  # S_y, shift along H, coord along W

    elif "S_z S_x S_y S_z" in decomposition:
        # Forward 2: cyclic permutation of axes
        vol = fft_shear(vol, g, 2, 1)  # S_z (cyclic shifted)
        vol = fft_shear(vol, h, 2, 0)
        vol = fft_shear(vol, e, 0, 1)  # S_x (cyclic shifted)
        vol = fft_shear(vol, f, 0, 2)
        vol = fft_shear(vol, c, 1, 2)  # S_y (cyclic shifted)
        vol = fft_shear(vol, d, 1, 0)
        vol = fft_shear(vol, a, 2, 1)
        vol = fft_shear(vol, b, 2, 0)

    elif "S_x S_y S_z S_x" in decomposition:
        # Forward 3: cyclic permutation
        vol = fft_shear(vol, g, 0, 2)  # S_x (cyclic shifted)
        vol = fft_shear(vol, h, 0, 1)
        vol = fft_shear(vol, e, 1, 2)  # S_y (cyclic shifted)
        vol = fft_shear(vol, f, 1, 0)
        vol = fft_shear(vol, c, 2, 0)  # S_z (cyclic shifted)
        vol = fft_shear(vol, d, 2, 1)
        vol = fft_shear(vol, a, 0, 2)
        vol = fft_shear(vol, b, 0, 1)

    elif "backward" in decomposition.lower():
        # Backward decompositions: reverse order
        if "S_y S_x S_z S_y" in decomposition:
            vol = fft_shear(vol, h, 1, 2)
            vol = fft_shear(vol, g, 1, 0)
            vol = fft_shear(vol, f, 2, 1)
            vol = fft_shear(vol, e, 2, 0)
            vol = fft_shear(vol, d, 0, 1)
            vol = fft_shear(vol, c, 0, 2)
            vol = fft_shear(vol, b, 1, 2)
            vol = fft_shear(vol, a, 1, 0)
        elif "S_z S_y S_x S_z" in decomposition:
            vol = fft_shear(vol, h, 2, 0)
            vol = fft_shear(vol, g, 2, 1)
            vol = fft_shear(vol, f, 0, 2)
            vol = fft_shear(vol, e, 0, 1)
            vol = fft_shear(vol, d, 1, 0)
            vol = fft_shear(vol, c, 1, 2)
            vol = fft_shear(vol, b, 2, 0)
            vol = fft_shear(vol, a, 2, 1)
        elif "S_x S_z S_y S_x" in decomposition:
            vol = fft_shear(vol, h, 0, 1)
            vol = fft_shear(vol, g, 0, 2)
            vol = fft_shear(vol, f, 1, 0)
            vol = fft_shear(vol, e, 1, 2)
            vol = fft_shear(vol, d, 2, 1)
            vol = fft_shear(vol, c, 2, 0)
            vol = fft_shear(vol, b, 0, 1)
            vol = fft_shear(vol, a, 0, 2)

    return vol


def rotate_volume(vol: torch.Tensor, q: np.ndarray) -> torch.Tensor:
    """
    Rotate a 3D volume using four-shear FFT decomposition (Welling et al. 2006).

    Automatically selects the best decomposition to avoid singularities.

    Parameters
    ----------
    vol : torch.Tensor
        3D volume (D, H, W), can be on CPU or GPU.
    q : np.ndarray
        Normalized quaternion (x, y, z, w).

    Returns
    -------
    torch.Tensor
        Rotated volume, same shape as input.

    Raises
    ------
    ValueError
        If quaternion represents a pure 180° rotation around a coordinate axis
        (the only case with no valid decomposition).
    """
    # Normalize quaternion
    q = q / np.linalg.norm(q)

    # Select best decomposition
    shear_params, decomp_name, merit = select_best_decomposition(q)

    # Apply four-shear decomposition
    vol_rotated = apply_four_shear_decomposition(vol, shear_params, decomp_name)

    return vol_rotated


def rotate_euler(
    vol: torch.Tensor, ax_deg: float, ay_deg: float, az_deg: float
) -> torch.Tensor:
    """
    Rotate volume by Euler angles using four-shear FFT decomposition.

    Parameters
    ----------
    vol : torch.Tensor
        3D volume (D, H, W).
    ax_deg, ay_deg, az_deg : float
        Roll (X), pitch (Y), yaw (Z) in degrees.

    Returns
    -------
    torch.Tensor
        Rotated volume.
    """
    q = euler_to_quaternion(ax_deg, ay_deg, az_deg)
    return rotate_volume(vol, q)
