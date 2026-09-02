from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import constants as _sc

from .. import rotations

avogadro = _sc.Avogadro
density_of_amorphous_ice = 0.94  # [g/cm3]
molar_mass_of_water = 18.01528  # [g/mol]
ndensity_of_amorphous_ice = (
    density_of_amorphous_ice * avogadro / molar_mass_of_water * 1e-24
)  # [particles / Å³]


def water_molecule_coordinates(
    bond_angle: float = 105.0, bond_length: float = 0.9572
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Return coordinates of a single H2O molecule with oxygen at the origin.

    Parameters
    ----------
    bond_angle : float, optional
        Bond angle between the O-H bonds in degrees. Default is 105.0.
    bond_length : float, optional
        O-H bond length in Å. Default is 0.9572.

    Returns
    -------
    atomic_numbers : np.ndarray
        Atomic numbers of the three atoms [8, 1, 1].
    coordinates : torch.Tensor
        xyz coordinates of O, H1, H2 with shape (3, 3).
    """
    O_xyz = torch.tensor([0.0, 0.0, 0.0])
    angle_rad = torch.tensor(bond_angle) / 180 * torch.pi
    y = bond_length * torch.cos(angle_rad / 2)
    x = bond_length * torch.sin(angle_rad / 2)
    H1_xyz = torch.tensor([x, y, 0.0])
    H2_xyz = torch.tensor([-x, y, 0.0])
    coordinates = torch.stack((O_xyz, H1_xyz, H2_xyz))
    atomic_numbers = np.array([8, 1, 1])
    return atomic_numbers, coordinates


def create_n_randomly_rotated_water_molecules(
    n: int, **kwargs: Any
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Create n randomly rotated water molecules.

    Parameters
    ----------
    n : int
        Number of molecules to create.
    **kwargs
        Passed to :func:`water_molecule_coordinates` (``bond_angle``, ``bond_length``).

    Returns
    -------
    atomic_numbers : np.ndarray
        Atomic numbers of all atoms, shape (3*n,).
    coordinates : torch.Tensor
        Atomic coordinates of all atoms, shape (3*n, 3).
    """
    quats = torch.stack([rotations.random_quaternion() for _ in range(n)])
    atomic_numbers, coords = water_molecule_coordinates(**kwargs)

    O_coordinates = coords[0].repeat(n, 1)
    H1_coordinates = rotations.rotate_coordinates(coords[1], quats)
    H2_coordinates = rotations.rotate_coordinates(coords[2], quats)

    all_coords = torch.zeros(n * 3, 3)
    all_coords[0::3] = O_coordinates
    all_coords[1::3] = H1_coordinates
    all_coords[2::3] = H2_coordinates

    all_atomic_numbers = np.array([8, 1, 1] * n)
    return all_atomic_numbers, all_coords


def volume_of_ice(
    n_xyz: Sequence[int], d_xyz: Sequence[float]
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Generate atomic coordinates for a volume of amorphous ice.

    Molecules are placed at random positions (overlaps possible) with random
    orientations. Oxygen is at the molecule centre; hydrogens are rotated randomly.

    Parameters
    ----------
    n_xyz : sequence of int
        Number of voxels (nx, ny, nz).
    d_xyz : sequence of float
        Voxel size in Å (dx, dy, dz).

    Returns
    -------
    atomic_numbers : np.ndarray
        Atomic numbers of all ice atoms.
    coordinates : torch.Tensor
        Coordinates of all ice atoms in Å.
    """
    nx, ny, nz = n_xyz
    dx, dy, dz = d_xyz
    total_volume = nx * ny * nz * dx * dy * dz
    n_molecules = int(ndensity_of_amorphous_ice * total_volume)

    x_ice = (torch.rand(n_molecules) - 0.5) * dx * nx
    y_ice = (torch.rand(n_molecules) - 0.5) * dy * ny
    z_ice = (torch.rand(n_molecules) - 0.5) * dz * nz
    centers = torch.repeat_interleave(
        torch.stack((x_ice, y_ice, z_ice), dim=1), 3, dim=0
    )

    atomic_numbers, coords = create_n_randomly_rotated_water_molecules(n_molecules)
    coords += centers
    return atomic_numbers, coords


def rfftn(array: torch.Tensor) -> torch.Tensor:
    """
    Compute N-dimensional real-input Fourier transform with centering.

    Wraps torch.fft.rfftn with FFT shifting to ensure the zero-frequency component
    is centered, handling the last dimension which is complex-valued differently.

    Parameters
    ----------
    array : torch.Tensor
        Input real-valued tensor.

    Returns
    -------
    fft : torch.Tensor
        Complex-valued tensor containing the Fourier coefficients.
        Zero frequency is centered.
    """
    return torch.fft.fftshift(
        torch.fft.rfftn(torch.fft.ifftshift(array, dim=(-3, -2, -1)), dim=(-3, -2, -1)),
        dim=(-3, -2),
    )


def torch_peak_local_max(
    image: torch.Tensor, min_distance: int = 1, num_peaks: int | None = None
) -> torch.Tensor:
    """
    Find local maxima in batched 3D images and return fixed number of peaks per batch.

    Parameters
    ----------
    image : torch.Tensor
        Input tensor of shape (B, D, H, W).
    min_distance : int, optional
        Minimum separation between peaks (voxels). Default is 1.
    num_peaks : int, optional
        Number of peaks to return per batch (must be <= total peaks in each batch).
        If None, uses the minimum number of peaks found in any batch item. Default is None.

    Returns
    -------
    peaks : torch.LongTensor
        Peak coordinates (z, y, x) for each batch. Shape (B, num_peaks, 3).
    """
    B, D, H, W = image.shape
    x = image.unsqueeze(1)  # (B, 1, D, H, W)
    k = 2 * min_distance + 1
    pooled = F.max_pool3d(x, kernel_size=k, stride=1, padding=min_distance)
    mask = (x == pooled).squeeze(1)  # (B, D, H, W)

    # Flatten spatial dims
    flat_mask = mask.view(B, -1)
    flat_image = image.view(B, -1)

    # Mask non-maxima
    flat_image_masked = flat_image.clone()
    flat_image_masked[~flat_mask] = -float("inf")

    if num_peaks is None:
        num_peaks = int(flat_mask.sum(dim=1).min().item())  # take min available peaks

    # Top-k per batch
    _, topk_idx = flat_image_masked.topk(num_peaks, dim=1)

    # Convert flat indices back to 3D coords
    z = topk_idx // (H * W)
    y = (topk_idx % (H * W)) // W
    x_ = topk_idx % W

    peaks = torch.stack([z, y, x_], dim=2)  # (B, num_peaks, 3)
    return peaks
