import lightning as L
import numpy as np
import torch
from rich.progress import track

from . import rotations


def poisson_disk_neighbors(
    min_distance,
    n_points=torch.inf,
    box=(256, 256),  # (height, width)
    k=30,
    seed="origin",
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
    y_min, y_max = -H // 2, H // 2
    x_min, x_max = -W // 2, W // 2

    # initialize first point
    if seed == "origin":
        first_point = torch.tensor([0.0, 0.0])
        n_points += 1  # don't count origin.
    elif seed == "random":
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
        mask = (
            (candidates[:, 0] >= y_min)
            & (candidates[:, 0] < y_max)
            & (candidates[:, 1] >= x_min)
            & (candidates[:, 1] < x_max)
        )
        candidates = candidates[mask]

        if candidates.shape[0] == 0:
            active.pop(idx)
            continue

        # distance check against all existing points
        pts_tensor = torch.stack(pts)
        diff = candidates[:, None, :] - pts_tensor[None, :, :]
        dist2 = (diff**2).sum(dim=2)
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
    box=(256.0, 256.0, 256.0),  # (D,H,W) for tensor shape
    k=30,
    seed="origin",
):
    """
    Generate 3D points using Poisson-disk sampling within a 3D tensor volume.

    This function produces points in a (D, H, W) volume such that no two points
    are closer than `min_distance` pixels, creating a uniform but spatially
    separated distribution.

    Parameters
    ----------
    min_distance : float
        Minimum allowed distance between points, in Angstroms.
    n_points : int or torch.inf, optional
        Maximum number of points to generate. Default is infinite (fill the volume).
    box : tuple of float, optional
        Dimensions of the 3D volume in Angstroms, as (D, H, W). Default is (256, 256, 256).
    k : int, optional
        Number of candidate points to generate around each active point. Higher
        values produce denser sampling. Default is 30.
    seed : {'origin', 'random'}, optional
        Determines the initial seed point:
        - 'origin': starts at (0, 0, 0)
        - 'random': starts at a random location within the box

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 3), where each row is a point coordinate (x, y, z) in Angstroms.

    Notes
    -----
    - The coordinate system is centered at (0,0,0), with ranges:
      x ∈ [-W/2, W/2], y ∈ [-H/2, H/2], z ∈ [-D/2, D/2].
    - The function uses a grid-accelerated version of Bridson's Poisson-disk sampling
      algorithm for efficient neighbor checking.
    """
    D, H, W = box
    z_min, z_max = -D / 2, D / 2
    y_min, y_max = -H / 2, H / 2
    x_min, x_max = -W / 2, W / 2

    # Grid acceleration
    cell_size = min_distance / np.sqrt(3)
    grid_shape = tuple(torch.ceil(torch.tensor([D, H, W]) / cell_size).int().tolist())
    grid = -torch.ones(grid_shape, dtype=torch.long)

    def point_to_grid(p):
        # p = (x, y, z)
        zi = ((p[2] - z_min) / cell_size).long().clamp(0, grid_shape[0] - 1)
        yi = ((p[1] - y_min) / cell_size).long().clamp(0, grid_shape[1] - 1)
        xi = ((p[0] - x_min) / cell_size).long().clamp(0, grid_shape[2] - 1)
        return zi, yi, xi

    # initialize first point
    if seed == "origin":
        first_point = torch.tensor([0.0, 0.0, 0.0])  # x,y,z
        n_points += 1
    elif seed == "random":
        x = (x_max - x_min) * torch.rand(1) + x_min
        y = (y_max - y_min) * torch.rand(1) + y_min
        z = (z_max - z_min) * torch.rand(1) + z_min
        first_point = torch.tensor([x.item(), y.item(), z.item()])
    else:
        raise ValueError("seed must be 'origin' or 'random'")

    pts = [first_point]
    active = [0]

    zi, yi, xi = point_to_grid(first_point)
    grid[zi, yi, xi] = 0

    while active and len(pts) < n_points:
        idx = torch.randint(len(active), (1,)).item()
        center_point = pts[active[idx]]

        # generate k candidates in spherical shell
        phi = torch.acos(2 * torch.rand(k) - 1)
        theta = 2 * torch.pi * torch.rand(k)
        r = min_distance * (1 + torch.rand(k))

        dx = r * torch.sin(phi) * torch.cos(theta)
        dy = r * torch.sin(phi) * torch.sin(theta)
        dz = r * torch.cos(phi)
        candidates = center_point.unsqueeze(0) + torch.stack([dx, dy, dz], dim=1)

        # filter candidates in tensor bounds (z,y,x)
        mask = (
            (candidates[:, 0] >= x_min)
            & (candidates[:, 0] < x_max)
            & (candidates[:, 1] >= y_min)
            & (candidates[:, 1] < y_max)
            & (candidates[:, 2] >= z_min)
            & (candidates[:, 2] < z_max)
        )
        candidates = candidates[mask]

        if candidates.shape[0] == 0:
            active.pop(idx)
            continue

        # grid neighbor check
        accepted = []
        for c in candidates:
            zi, yi, xi = point_to_grid(c)
            neighbor_found = False
            for dz_i in [-1, 0, 1]:
                for dy_i in [-1, 0, 1]:
                    for dx_i in [-1, 0, 1]:
                        nz, ny, nx = zi + dz_i, yi + dy_i, xi + dx_i
                        if (
                            0 <= nz < grid_shape[0]
                            and 0 <= ny < grid_shape[1]
                            and 0 <= nx < grid_shape[2]
                        ):
                            pid = grid[nz, ny, nx].item()
                            if pid != -1:
                                dist = torch.norm(c - pts[pid])
                                if dist < min_distance:
                                    neighbor_found = True
                                    break
                    if neighbor_found:
                        break
                if neighbor_found:
                    break
            if not neighbor_found:
                accepted.append(c)

        if accepted:
            new_pt = accepted[0]
            pts.append(new_pt)
            active.append(len(pts) - 1)
            zi, yi, xi = point_to_grid(new_pt)
            grid[zi, yi, xi] = len(pts) - 1
        else:
            active.pop(idx)

    if seed == "origin":
        # no candidates were found
        if len(pts) == 1:
            return torch.empty((0, 3))
        else:
            # don't include origin
            pts = pts[1:]
            return torch.stack(pts[:n_points] if n_points != torch.inf else pts)
    elif seed == "random":
        return torch.stack(pts[:n_points] if n_points != torch.inf else pts)


def crowd_with_duplicates(
    V,
    min_distance,
    pixel_size,
    return_coordinates=False,
    max_distance_xy=None,
    max_distance_z=None,
    nxy=None,
    nz=None,
):
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
    R = rotations.random_rotation_matrix(num_neighbours)
    # in case only one position was found, ensures R is (1,3,3)
    if len(R.shape) == 2:
        R = R.unsqueeze(0)

    theta = rotations.build_affine_matrix(R)

    # Apply rotations to the volume
    vols = rotations.rotate_volume(V.to(device), theta.to(device), padding_mode="zeros")

    # insert volumes at correct coordinates
    micro = insert_particles_into_micrograph(
        vols, translations, pixel_size=pixel_size, micro_shape=(nz, nxy, nxy)
    )

    # Sum all duplicates into a single crowded volume
    if return_coordinates:
        return micro, translations
    else:
        return micro


def insert_particles_into_micrograph(
    volumes,
    positions,
    pixel_size=1.0,
    micro_shape=None,
    micrograph=None,
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
    N, Zp, Yp, Xp = volumes.shape
    hz, hy, hx = Zp // 2, Yp // 2, Xp // 2
    device = volumes.device

    # Allocate micrograph
    if micrograph is not None:
        micrograph = micrograph.to(device)
        Z, Y, X = micrograph.shape
    elif micro_shape is not None:
        Z, Y, X = micro_shape
        micrograph = torch.zeros(micro_shape, device=device)
    else:
        raise ValueError(
            "Must provide either `micro_shape` or an existing `micrograph`."
        )

    volumes = volumes.to(device)
    positions = positions.to(device)
    if positions.shape[1] == 2:
        zeros = torch.zeros(
            (positions.shape[0], 1), device=positions.device, dtype=positions.dtype
        )
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
        micrograph[z0_clip:z1_clip, y0_clip:y1_clip, x0_clip:x1_clip] += volumes[
            i, pz0:pz1, py0:py1, px0:px1
        ]

    return micrograph


class CrowdWithDuplicates(L.LightningModule):
    """
    Generates multiple duplicates of a 3D volume within a micrograph using
    Poisson-disk sampling for spatial placement and random rotations for orientation.
    Useful for simulating crowded particle distributions in cryo-EM datasets.

    The class supports 2D or 3D Poisson-disk sampling, optional chunked rotation
    for memory efficiency, and allows returning a complete micrograph with inserted
    particle duplicates.
    """

    def __init__(
        self,
        V,
        dx,
        min_distance,
        nxy_out=None,
        nz_out=None,
        max_distance_z=None,
        max_distance_xy=None,
        method="3d",
        n_points=torch.inf,
        seed="origin",
        chunk_size=None,
        move_to_cpu=False,
        progressbars=True,
    ):
        """
        Parameters
        ----------
        V : torch.Tensor
            The 3D volume to be duplicated, shape (D, H, W).
        dx : float
            Pixel size of the volume in the same units as `min_distance` (pixels).
        min_distance : float
            Minimum separation between particle duplicates in pixels for Poisson-disk sampling.
        nxy_out : int, optional
            Output micrograph size in xy dimensions (pixels). Defaults to V.shape[1].
        nz_out : int, optional
            Output micrograph size in z dimension (pixels). Defaults to V.shape[0].
        max_distance_z : float, optional
            Maximum extent in z for placing duplicates (pixels). Defaults to nz_out * dx + min_distance.
        max_distance_xy : float, optional
            Maximum extent in xy for placing duplicates (pixels). Defaults to nxy_out * dx + min_distance.
        method : {'2d', '3d'}, optional
            Poisson-disk sampling method. '2d' samples in xy plane only (z=0),
            '3d' samples in full 3D volume. Default is '3d'.
        n_points : int or torch.inf, optional
            Maximum number of duplicates to generate. Default is infinity (fill volume).
        seed : {'origin', 'random'}, optional
            Initial placement seed:
                - 'origin': start at center (0,0,0)
                - 'random': start at a random location within the box
        chunk_size : int, optional
            Number of volumes to rotate per batch for memory-efficient computation.
            If None, all volumes are rotated at once.
        move_to_cpu : bool, optional
            If True, intermediate rotated volumes are moved to CPU to save GPU memory.
        progressbars : bool, optional
            If True, display progress bars for chunked operations.

        Attributes
        ----------
        coords : torch.Tensor
            Coordinates of generated duplicates after Poisson-disk sampling.
        theta : torch.Tensor
            Affine rotation matrices for each duplicate.
        vols : torch.Tensor
            Rotated duplicates ready for insertion into a micrograph.
        N : int
            Number of duplicates generated.
        """
        super().__init__()

        self.register_buffer("V", V)
        self.dx = dx
        self.n = V.shape[0]
        self.min_distance = min_distance
        self.poisson_disc_method = method
        self.n_points = n_points
        self.seed = seed
        self.chunk_size = chunk_size
        self.move_to_cpu = move_to_cpu
        self.progressbars = progressbars

        if nz_out is None:
            nz_out = self.n
        self.nz_out = nz_out
        if nxy_out is None:
            nxy_out = self.n
        self.nxy_out = nxy_out

        if max_distance_xy is None:
            max_distance_xy = self.n * dx + min_distance
        self.max_distance_xy = max_distance_xy
        if max_distance_z is None:
            max_distance_z = self.n * dx + min_distance
        self.max_distance_z = max_distance_z

    def generate_coordinates(self):
        """
        Generate coordinates of duplicates using Poisson-disk sampling.

        For '2d' sampling, points are sampled in xy plane and z is set to 0.
        For '3d' sampling, points are sampled in full 3D volume.
        Coordinates are stored in `self.coords`.
        """
        if self.poisson_disc_method == "2d":
            coords = poisson_disk_neighbors(
                self.min_distance,
                n_points=self.n_points,
                box=(self.nxy_out, self.nxy_out),
                seed=self.seed,
            )
            # add z-coordinates
            zeros = torch.zeros((coords.shape[0], 1))
            coords = torch.cat([coords, zeros], dim=1)
        elif self.poisson_disc_method == "3d":
            coords = poisson_disk_neighbors_3d(
                self.min_distance,
                n_points=self.n_points,
                box=(self.max_distance_z, self.nxy_out, self.nxy_out),
                seed=self.seed,
            )
        self.coords = coords

    def generate_affine_matrices(self):
        """
        Generate random rotation matrices and corresponding affine matrices
        for each duplicate volume.

        Rotations are stored in `self.theta`.
        """
        self.N = len(self.coords)
        R = rotations.random_rotation_matrix(self.N)
        # in case only one position was found, ensures R is (1,3,3)
        if len(R.shape) == 2:
            R = R.unsqueeze(0)
        self.theta = rotations.build_affine_matrix(R)

    def rotate_volumes(self):
        """
        Rotate the original volume according to the affine matrices `self.theta`.

        If `chunk_size` is specified, volumes are rotated in batches for memory efficiency.
        The rotated volumes are stored in `self.vols`.
        """
        if self.chunk_size is not None:
            if self.move_to_cpu:
                self.vols = torch.empty((self.N,) + self.V.shape)
            else:
                self.vols = torch.empty((self.N,) + self.V.shape, device=self.device)

            for start in track(
                range(0, self.N, self.chunk_size),
                description="Rotating duplicates",
                transient=True,
                disable=not (self.progressbars),
            ):
                end = min(start + self.chunk_size, self.N)
                if self.move_to_cpu:
                    self.vols[start:end] = rotations.rotate_volume(
                        self.V,
                        self.theta[start:end].to(self.V.device),
                        padding_mode="zeros",
                    ).cpu()
                else:
                    self.vols[start:end] = rotations.rotate_volume(
                        self.V,
                        self.theta[start:end].to(self.V.device),
                        padding_mode="zeros",
                    )
        else:
            self.vols = rotations.rotate_volume(
                self.V, self.theta.to(self.V.device), padding_mode="zeros"
            )

    def insert_volumes(self):
        """
        Insert the rotated volumes (`self.vols`) into a 3D micrograph according to `self.coords`.

        Returns
        -------
        torch.Tensor
            Micrograph of shape (nz_out, nxy_out, nxy_out) containing all duplicates.
        """
        micro = insert_particles_into_micrograph(
            self.vols,
            self.coords,
            pixel_size=self.dx,
            micro_shape=(self.nz_out, self.nxy_out, self.nxy_out),
        )
        return micro

    def forward(self):
        """
        Full pipeline: generate coordinates, random rotations, rotate volumes,
        and insert them into a micrograph.

        Returns
        -------
        torch.Tensor or float
            The final micrograph containing all duplicates. Returns 0.0 if no candidates
            were generated.
        """
        self.generate_coordinates()
        self.generate_affine_matrices()
        # if no candidates, return 0
        if len(self.coords) == 0:
            return 0.0
        else:
            if self.chunk_size is None:
                self.rotate_volumes()
                micrograph = self.insert_volumes()
                if self.move_to_cpu:
                    micrograph = micrograph.cpu()
            else:
                # create micrograph
                micrograph = torch.zeros(self.nz_out, self.nxy_out, self.nxy_out)
                # rotate and insert in batches
                for start in track(
                    range(0, self.N, self.chunk_size),
                    description="Rotating duplicates and insert into micrograph",
                    transient=True,
                    disable=not (self.progressbars),
                ):
                    # get batch indicies
                    end = min(start + self.chunk_size, self.N)
                    # rotate
                    vols = rotations.rotate_volume(
                        self.V,
                        self.theta[start:end].to(self.V.device),
                        padding_mode="zeros",
                    )
                    if self.move_to_cpu:
                        vols = vols.cpu()
                    # insert
                    micrograph = insert_particles_into_micrograph(
                        vols,
                        self.coords[start:end],
                        pixel_size=self.dx,
                        micrograph=micrograph,
                    )
            return micrograph
