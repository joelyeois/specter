import torch
from . import rotations
import numpy as np

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


def poisson_disk_neighbors_3d(
    min_distance,
    n_points=torch.inf,
    box=(256, 256, 256),  # (D,H,W) for tensor shape
    k=30,
    seed='origin',
    device='cpu'
):
    """
    Fast 3D Poisson-disk sampling in a 3D tensor of shape (D,H,W),
    returning coordinates in (x, y, z) order.
    """
    D, H, W = box
    z_min, z_max = -D//2, D//2
    y_min, y_max = -H//2, H//2
    x_min, x_max = -W//2, W//2

    # Grid acceleration
    cell_size = min_distance / np.sqrt(3)
    grid_shape = tuple(torch.ceil(torch.tensor([D, H, W]) / cell_size).int().tolist())
    grid = -torch.ones(grid_shape, dtype=torch.long, device=device)

    def point_to_grid(p):
        # p = (x, y, z)
        zi = ((p[2] - z_min) / cell_size).long().clamp(0, grid_shape[0]-1)
        yi = ((p[1] - y_min) / cell_size).long().clamp(0, grid_shape[1]-1)
        xi = ((p[0] - x_min) / cell_size).long().clamp(0, grid_shape[2]-1)
        return zi, yi, xi

    # initialize first point
    if seed == 'origin':
        first_point = torch.tensor([0.,0.,0.], device=device)  # x,y,z
        n_points += 1
    elif seed == 'random':
        x = (x_max - x_min) * torch.rand(1, device=device) + x_min
        y = (y_max - y_min) * torch.rand(1, device=device) + y_min
        z = (z_max - z_min) * torch.rand(1, device=device) + z_min
        first_point = torch.tensor([x.item(), y.item(), z.item()], device=device)
    else:
        raise ValueError("seed must be 'origin' or 'random'")

    pts = [first_point]
    active = [0]

    zi, yi, xi = point_to_grid(first_point)
    grid[zi, yi, xi] = 0

    while active and len(pts) < n_points:
        idx = torch.randint(len(active), (1,), device=device).item()
        center_point = pts[active[idx]]

        # generate k candidates in spherical shell
        phi = torch.acos(2*torch.rand(k, device=device)-1)
        theta = 2*torch.pi*torch.rand(k, device=device)
        r = min_distance * (1 + torch.rand(k, device=device))

        dx = r * torch.sin(phi) * torch.cos(theta)
        dy = r * torch.sin(phi) * torch.sin(theta)
        dz = r * torch.cos(phi)
        candidates = center_point.unsqueeze(0) + torch.stack([dx, dy, dz], dim=1)

        # filter candidates in tensor bounds (z,y,x)
        mask = (candidates[:,0]>=x_min) & (candidates[:,0]<x_max) & \
               (candidates[:,1]>=y_min) & (candidates[:,1]<y_max) & \
               (candidates[:,2]>=z_min) & (candidates[:,2]<z_max)
        candidates = candidates[mask]

        if candidates.shape[0] == 0:
            active.pop(idx)
            continue

        # grid neighbor check
        accepted = []
        for c in candidates:
            zi, yi, xi = point_to_grid(c)
            neighbor_found = False
            for dz_i in [-1,0,1]:
                for dy_i in [-1,0,1]:
                    for dx_i in [-1,0,1]:
                        nz, ny, nx = zi+dz_i, yi+dy_i, xi+dx_i
                        if 0<=nz<grid_shape[0] and 0<=ny<grid_shape[1] and 0<=nx<grid_shape[2]:
                            pid = grid[nz, ny, nx].item()
                            if pid != -1:
                                dist = torch.norm(c - pts[pid])
                                if dist < min_distance:
                                    neighbor_found = True
                                    break
                    if neighbor_found: break
                if neighbor_found: break
            if not neighbor_found:
                accepted.append(c)

        if accepted:
            new_pt = accepted[0]
            pts.append(new_pt)
            active.append(len(pts)-1)
            zi, yi, xi = point_to_grid(new_pt)
            grid[zi, yi, xi] = len(pts)-1
        else:
            active.pop(idx)

    if seed == 'origin':
        # don't include origin
        pts = pts[1:]
        return torch.stack(pts[:n_points] if n_points!=torch.inf else pts)
    elif seed == 'random':
        return torch.stack(pts[:n_points] if n_points!=torch.inf else pts)


def crowd_with_duplicates(V, min_distance, pixel_size, return_coordinates=False,
                          max_distance_xy=None, max_distance_z=None, nxy=None, nz=None):
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
        Minimum center-to-center distance between duplicates (in Angstroms), i.e. diameter
        of particle.
    pixel_size : float
        Size of a voxel in Angstroms.
    return_coordinates : bool, optional
        If True, also returns coordinates of the duplicated particles.
    max_distance_xy : float, optional
        Maximum radial distance from the origin in xy-plane to place duplicates (in Angstroms).
        If None, defaults to width of the volume plus diameter of particle.
    max_distance_z : float, optional
        Maximum radial distance from the origin in z to place duplicates (in Angstroms).
        If None, defaults to height of the volume plus diameter of particle.
    nxy : int, optional
        Width of volume to return. Used for micrographs or padded volumes.
    nz : int, optional
        Height of volume to return. Used for micrographs or padded volumes.

    Returns
    -------
    crowded_volume : torch.Tensor, shape (Z, Y, X)
        The 3D volume containing the original particle plus all rotated, translated
        duplicates, summed together.
    """
    device = V.device
    V_nxy = V.shape[-1]
    V_nz = V.shape[0]

    if max_distance_xy is None:
        max_distance_xy = V_nxy * pixel_size + min_distance
    if max_distance_z is None:
        max_distance_z = V_nz * pixel_size + min_distance
    
    if nxy is None:
        nxy = V_nxy
    if nz is None:
        nz = V_nz

    # Generate 3D positions for duplicates using Poisson-disk sampling
    translations = poisson_disk_neighbors_3d(
        min_distance, box=(max_distance_z, max_distance_xy, max_distance_xy)
    )

    num_neighbours = len(translations)

    # Generate random rotations for each duplicate
    quats = rotations.random_quaternion(num_neighbours)
    R = rotations.quaternion_to_rotation_matrix(quats)
    # in case only one position was found, ensures R is (1,3,3)
    if len(R.shape) == 2:
        R = R.unsqueeze(0)

    theta = rotations.build_affine_matrix(R)

    # Apply rotations to the volume
    vols = rotations.rotate_volume(V.to(device), theta.to(device), padding_mode="zeros")

    # insert volumes at correct coordinates
    micro = insert_particles_into_micrograph((nz, nxy, nxy), vols, translations, pixel_size=pixel_size)

    # Sum all duplicates into a single crowded volume
    if return_coordinates:
        return micro, translations
    else:
        return micro


def insert_particles_into_micrograph(
    micro_shape, volumes, positions, pixel_size=1.0
):
    """
    Insert rotated 3D volumes into a 3D micrograph centered at the origin.

    Parameters
    ----------
    micro_shape : tuple of int
        Shape of micrograph (Z, Y, X)
    volumes : torch.Tensor
        Rotated volumes, shape (N, Zp, Yp, Xp)
    positions : torch.Tensor
        (N, 3) coordinates in physical units (x, y, z), origin at center
    pixel_size : float
        Physical size of one pixel (same units as positions)
    device : str
        'cuda' or 'cpu'

    Returns
    -------
    micrograph : torch.Tensor
        Micrograph with volumes inserted
    """
    Z, Y, X = micro_shape
    N, Zp, Yp, Xp = volumes.shape
    hz, hy, hx = Zp // 2, Yp // 2, Xp // 2
    device = volumes.device

    # Allocate micrograph
    micrograph = torch.zeros(micro_shape, device=device)
    volumes = volumes.to(device)
    positions = positions.to(device)
    if positions.shape[1] == 2:
        zeros = torch.zeros((positions.shape[0], 1), device=positions.device, dtype=positions.dtype)
        positions = torch.cat([positions, zeros], dim=1)

    # Convert from physical units to pixel indices
    positions_pixels = positions / pixel_size
    positions_int = positions_pixels.round().long()  # shape (N, 3), order (x, y, z)

    # Micrograph center indices
    cz_center = Z // 2
    cy_center = Y // 2
    cx_center = X // 2

    for i in range(N):
        # Convert centered coords to array indices
        cx_index = cx_center + positions_int[i, 0]
        cy_index = cy_center + positions_int[i, 1]
        cz_index = cz_center + positions_int[i, 2]

        # Particle slice bounds
        z0 = cz_index - hz
        z1 = cz_index + hz
        y0 = cy_index - hy
        y1 = cy_index + hy
        x0 = cx_index - hx
        x1 = cx_index + hx

        # Clip to micrograph bounds
        z0_clip = max(z0, 0)
        z1_clip = min(z1, Z)
        y0_clip = max(y0, 0)
        y1_clip = min(y1, Y)
        x0_clip = max(x0, 0)
        x1_clip = min(x1, X)

        # Corresponding subvolume slice
        pz0 = z0_clip - z0
        pz1 = pz0 + (z1_clip - z0_clip)
        py0 = y0_clip - y0
        py1 = py0 + (y1_clip - y0_clip)
        px0 = x0_clip - x0
        px1 = px0 + (x1_clip - x0_clip)

        # Skip if fully outside bounds
        if (z1_clip <= z0_clip) or (y1_clip <= y0_clip) or (x1_clip <= x0_clip):
            continue

        # Add the volume to the micrograph
        micrograph[z0_clip:z1_clip, y0_clip:y1_clip, x0_clip:x1_clip] += \
            volumes[i, pz0:pz1, py0:py1, px0:px1]

    return micrograph

