import torch
from . import rotations
import numpy as np

import torch

def poisson_disk_neighbors(
    min_distance,
    n_points=torch.inf,
    box=(256, 256),   # (height, width)
    k=30,
    seed='origin'
):
    """
    2D Poisson-disk sampling in a rectangular box centered at the origin.

    Parameters
    ----------
    min_distance : float
        Minimum spacing between points.
    n_points : int
        Total number of points to generate (including seed).
    box : tuple of int
        (height, width) of the bounding box in pixels. Origin is at (0,0).
        Valid coordinates are y in [-H/2, H/2), x in [-W/2, W/2).
    k : int
        Number of candidate points to try per active point.
    seed : {"origin", "random"}
        If "origin", first point is at the center (0,0).
        If "random", first point is chosen uniformly inside the box.

    Returns
    -------
    pts : (m,2) torch.Tensor
        Sampled 2D coordinates, including the seed point.
    """
    H, W = box
    y_min, y_max = -H//2, H//2
    x_min, x_max = -W//2, W//2

    # initialize first point
    if seed == 'origin':
        first_point = torch.tensor([0.0, 0.0])
        n_points += 1 #don't count origin.
    elif seed == 'random':
        y = (y_max - y_min) * torch.rand(1) + y_min
        x = (x_max - x_min) * torch.rand(1) + x_min
        first_point = torch.tensor([y.item(), x.item()])
    else:
        raise ValueError("seed must be 'origin' or 'random'")

    pts = [first_point]
    active = [0]

    while active and len(pts) < n_points:
        idx = torch.randint(len(active), (1,)).item()
        center_point = pts[active[idx]]

        # generate k candidates in annulus [min_distance, 2*min_distance]
        theta = torch.rand(k) * 2 * torch.pi
        radius = min_distance + min_distance * torch.rand(k)
        candidates = center_point.unsqueeze(0) + torch.stack(
            (radius * torch.cos(theta), radius * torch.sin(theta)), dim=1
        )

        # reject candidates outside the centered box
        mask = (candidates[:, 0] >= y_min) & (candidates[:, 0] < y_max) & \
               (candidates[:, 1] >= x_min) & (candidates[:, 1] < x_max)
        candidates = candidates[mask]

        if candidates.shape[0] == 0:
            active.pop(idx)
            continue

        # distance check against all existing points
        pts_tensor = torch.stack(pts)
        diff = candidates[:, None, :] - pts_tensor[None, :, :]
        dist2 = (diff ** 2).sum(dim=2)
        min_dist2, _ = dist2.min(dim=1)
        candidates = candidates[min_dist2 >= min_distance**2]

        if candidates.shape[0] > 0:
            pts.append(candidates[0])
            active.append(len(pts) - 1)
        else:
            active.pop(idx)

    if n_points == torch.inf:
        return torch.stack(pts)
    else:
        return torch.stack(pts[:n_points])




def crowd_with_duplicates(V, min_distance, pixel_size, max_distance=None, 
                          return_coordinates=False):
    """
    Generates a crowded volume by placing multiple rotated duplicates of a given 3D volume.

    Each duplicate is positioned using Poisson-disk sampling to maintain a minimum
    separation (`min_distance`) from other duplicates. Random rotations are applied
    to each duplicate. All duplicates are summed into a single crowded volume.

    Parameters
    ----------
    V : torch.Tensor, shape (Z, Y, X)
        The input 3D volume representing a single particle. Must be real-valued.
    min_distance : float
        Minimum center-to-center distance between duplicates (in Angstroms).
    pixel_size : float
        Size of a voxel in Angstroms.
    max_distance : float, optional
        Maximum radial distance from the origin to place duplicates (in Angstroms).
        If None, defaults to half the volume diagonal plus `min_distance/2`.

    Returns
    -------
    crowded_volume : torch.Tensor, shape (Z, Y, X)
        The 3D volume containing the original particle plus all rotated, translated
        duplicates, summed together.
    """
    device = V.device
    n = V.shape[-1]  # assume cubic volume
    if max_distance is None:
        max_distance = n * pixel_size + min_distance

    # Generate 2D positions for duplicates using Poisson-disk sampling
    translations = poisson_disk_neighbors(
        min_distance, box=(max_distance, max_distance)
    )

    num_neighbours = len(translations)

    # Generate random rotations for each duplicate
    quats = rotations.random_quaternion(num_neighbours)
    R = rotations.quaternion_to_rotation_matrix(quats)
    if len(R.shape) == 2:
        R = R.unsqueeze(0)

    # Convert translations from Angstroms to normalized Torch coordinates
    T = rotations.translations_angstrom_to_torch(translations, n, pixel_size)

    # Build affine matrices combining rotation and translation
    theta = rotations.build_affine_matrix(R, T)

    # Apply rotations and translations to the volume
    vols = rotations.rotate_volume(V.to(device), theta.to(device), padding_mode="zeros")

    # Sum all duplicates into a single crowded volume
    if return_coordinates:
        return vols.sum(0), translations
    else:
        return vols.sum(0)
