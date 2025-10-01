import torch
from . import rotations
import numpy as np

def poisson_disk_neighbors(min_distance, n_points=torch.inf,
                           max_distance=5.0, k=30):
    """
    Fast 2D Poisson-disk sampling with Bridson's algorithm.
    Guarantees origin is the first point.

    Parameters
    ----------
    min_distance : float
        Minimum center-to-center spacing.
    n_points : int
        Total points to generate (including origin).
    max_distance : float, optional
        Max distance from origin.
    k : int, optional
        Number of candidate points to try per active point.

    Returns
    -------
    pts : (m,2) torch.Tensor
        Sampled 2D coordinates. First row is [0,0].
    """

    pts = [torch.zeros(2)] # first point at origin
    active = [0]

    # we exclude the origin in our counter
    n_points += 1
    
    while active and len(pts) < n_points:
        idx = torch.randint(len(active), (1,)).item()
        center = pts[active[idx]]

        # generate k candidates in parallel
        theta = torch.rand(k) * 2 * torch.pi
        radius = min_distance + min_distance * torch.rand(k)
        candidates = center.unsqueeze(0) + torch.stack(
            (radius * torch.cos(theta), radius * torch.sin(theta)), dim=1
        )

        # reject candidates outside the circle
        mask = candidates.pow(2).sum(dim=1) <= max_distance**2
        candidates = candidates[mask]

        if candidates.shape[0] == 0:
            active.pop(idx)
            continue

        # vectorized distance check against all existing points
        if pts:
            pts_tensor = torch.stack(pts)
            diff = candidates[:, None, :] - pts_tensor[None, :, :]
            dist2 = (diff ** 2).sum(dim=2)   # shape (num_candidates, num_pts)
            min_dist2, _ = dist2.min(dim=1)
            candidates = candidates[min_dist2 >= min_distance**2]

        if candidates.shape[0] > 0:
            pts.append(candidates[0])
            active.append(len(pts) - 1)
        else:
            active.pop(idx)

    if n_points == torch.inf:
        return torch.stack(pts[1:])
    else:
        return torch.stack(pts[1:n_points])


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
        max_distance = (n * pixel_size + min_distance) / 2

    # Generate 2D positions for duplicates using Poisson-disk sampling
    translations = poisson_disk_neighbors(
        min_distance, max_distance=max_distance
    )

    num_neighbours = len(translations)

    # Generate random rotations for each duplicate
    quats = rotations.random_quaternion(num_neighbours)
    R = rotations.quaternion_to_rotation_matrix(quats)

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
