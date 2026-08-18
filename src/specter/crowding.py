from __future__ import annotations

from typing import Literal

import lightning as L
import torch

from .arrays import clip_insert_bounds
from .coords import poisson_disk_neighbors, poisson_disk_neighbors_3d
from .progress import track

from . import rotations

__all__ = [
    "poisson_disk_neighbors",
    "poisson_disk_neighbors_3d",
    "insert_particles_into_micrograph",
    "filter_by_z_density",
    "CrowdWithDuplicates",
]


def insert_particles_into_micrograph(
    volumes: torch.Tensor,
    positions: torch.Tensor,
    pixel_size: float = 1.0,
    micro_shape: tuple[int, int, int] | None = None,
    micrograph: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Insert rotated 3D volumes into a 3D micrograph centered at the origin.

    Parameters
    ----------
    volumes : torch.Tensor
        Rotated volumes with shape (N, Zp, Yp, Xp) where N is the number
        of particles to insert.
    positions : torch.Tensor
        Particle center coordinates in physical units (x, y, z) with shape
        (N, 3) or (N, 2). Origin is at the center of the micrograph. If
        shape is (N, 2), z-coordinates are assumed to be zero.
    pixel_size : float, optional
        Physical size of one pixel in same units as positions. Default is 1.0.
    micro_shape : tuple of int, optional
        Shape of micrograph (Z, Y, X). Required if micrograph is None.
    micrograph : torch.Tensor, optional
        Existing micrograph to insert volumes into. If None, a new micrograph
        is created with shape micro_shape.

    Returns
    -------
    micrograph : torch.Tensor
        Micrograph with volumes inserted, shape (Z, Y, X).

    Raises
    ------
    ValueError
        If neither micro_shape nor micrograph is provided.

    Notes
    -----
    If a particle extends beyond the micrograph boundaries, only the portion
    within bounds is inserted (clipping at edges).
    """
    N, Zp, Yp, Xp = volumes.shape
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
        cx_index = cx_center + int(positions_int[i, 0].item())
        cy_index = cy_center + int(positions_int[i, 1].item())
        cz_index = cz_center + int(positions_int[i, 2].item())

        bounds = clip_insert_bounds(
            (cz_index, cy_index, cx_index), (Zp, Yp, Xp), (Z, Y, X)
        )
        if bounds is None:
            continue
        dst, src = bounds
        micrograph[dst] += volumes[i][src]

    return micrograph


def filter_by_z_density(
    pts: torch.Tensor,
    z_length: float,
    sigma_frac: float = 0.05,
    peak_amplitude: float = 1.0,
    baseline: float = 0.1,
    curve_points: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Filter points based on a two-Gaussian z-density profile with peaks at the
    top and bottom of the z-range.

    Simulates particle distribution at ice-water interface where particles
    preferentially accumulate at top and bottom surfaces.

    Parameters
    ----------
    pts : torch.Tensor
        Particle coordinates with shape (N, 3) containing (x, y, z) positions.
    z_length : float
        Thickness of the ice minus particle diameter.
    sigma_frac : float, optional
        Gaussian width as fraction of z_length. Default is 0.05.
    peak_amplitude : float, optional
        Amplitude of Gaussian peaks at surfaces. Default is 1.0.
    baseline : float, optional
        Minimum probability in the bulk region. Default is 0.1.
    curve_points : int, optional
        Number of points for returning the probability curve along z.
        Default is 200.

    Returns
    -------
    pts_filtered : torch.Tensor
        Filtered points with shape (M, 3) where M <= N.
    z_distribution : torch.Tensor
        Probability density curve along z-axis with shape (curve_points,).

    Notes
    -----
    The probability distribution is: P(z) = baseline + amplitude * (G1 + G2)
    where G1 and G2 are Gaussians centered at z_min and z_max respectively.
    """
    z_min, z_max = -z_length / 2, z_length / 2
    sigma = sigma_frac * z_length

    if len(pts) == 0:
        z_curve = torch.linspace(z_min, z_max, curve_points)
        return pts, torch.zeros(curve_points)

    # compute z for each point
    z_pts = pts[:, 2]

    # Gaussian peaks at top and bottom
    g1 = torch.exp(-0.5 * ((z_pts - z_min) / sigma) ** 2)
    g2 = torch.exp(-0.5 * ((z_pts - z_max) / sigma) ** 2)

    # acceptance probability for each point
    probs_filtered = baseline + peak_amplitude * (g1 + g2)
    probs_filtered = (probs_filtered / probs_filtered.max()).clamp(0.0, 1.0)

    # filter points randomly
    mask = torch.rand(len(z_pts)) < probs_filtered
    pts_filtered = pts[mask]
    probs_filtered = probs_filtered[mask]

    # create probability curve along z
    z_curve = torch.linspace(z_min, z_max, curve_points)
    g1_curve = torch.exp(-0.5 * ((z_curve - z_min) / sigma) ** 2)
    g2_curve = torch.exp(-0.5 * ((z_curve - z_max) / sigma) ** 2)
    p_curve = baseline + peak_amplitude * (g1_curve + g2_curve)
    z_distribution = (p_curve / p_curve.max()).clamp(0.0, 1.0)

    return pts_filtered, z_distribution


class CrowdWithDuplicates(L.LightningModule):
    """
    Generates multiple duplicates of a 3D volume within a micrograph using
    Poisson-disk sampling for spatial placement and random rotations for orientation.
    Useful for simulating crowded particle distributions in cryo-EM datasets.

    The class supports 2D or 3D Poisson-disk sampling, optional chunked rotation
    for memory efficiency, and allows returning a complete micrograph with inserted
    particle duplicates.

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
    water_air_interface : bool, optional
        If True, applies a bimodal distribution along z-coordinates. Mimics the
        particle adsorption in cryo-EM ice-water interface.

    Attributes
    ----------
    coords : torch.Tensor
        Coordinates of generated duplicates after Poisson-disk sampling.
    theta : torch.Tensor
        Affine rotation matrices for each duplicate.
    volumes : torch.Tensor
        Rotated duplicates ready for insertion into a micrograph.
    N : int
        Number of duplicates generated.
    """

    def __init__(
        self,
        V: torch.Tensor,
        dx: float,
        min_distance: float,
        nxy_out: int | None = None,
        nz_out: int | None = None,
        max_distance_z: float | None = None,
        max_distance_xy: float | None = None,
        method: Literal["2d", "3d"] = "3d",
        n_points: int | float = torch.inf,
        seed: Literal["origin", "random"] = "origin",
        chunk_size: int | None = None,
        move_to_cpu: bool = False,
        progressbars: bool = True,
        water_air_interface: bool = False,
    ):
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
        self.water_air_interface = water_air_interface

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

    def generate_coordinates(self) -> None:
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
                box=(self.max_distance_z, self.max_distance_xy, self.max_distance_xy),
                seed=self.seed,
            )
            if self.water_air_interface:
                coords, self.z_distribution = filter_by_z_density(
                    coords, self.max_distance_z
                )
        else:
            raise ValueError(
                f"Unknown method '{self.poisson_disc_method}'. Choose '2d' or '3d'."
            )
        self.coords = coords

    def generate_affine_matrices(self) -> None:
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

    def rotate_volumes(self) -> None:
        """
        Rotate the original volume according to the affine matrices `self.theta`.

        If `chunk_size` is specified, volumes are rotated in batches for memory efficiency.
        The rotated volumes are stored in `self.volumes`.
        """
        if self.chunk_size is not None:
            if self.move_to_cpu:
                self.volumes = torch.empty((self.N,) + self.V.shape)
            else:
                self.volumes = torch.empty((self.N,) + self.V.shape, device=self.device)

            for start in track(
                range(0, self.N, self.chunk_size),
                description="Rotating duplicates",
                transient=True,
                disable=not self.progressbars,
            ):
                end = min(start + self.chunk_size, self.N)
                if self.move_to_cpu:
                    self.volumes[start:end] = rotations.rotate_volume(
                        self.V,
                        self.theta[start:end].to(self.V.device),
                        padding_mode="zeros",
                    ).cpu()
                else:
                    self.volumes[start:end] = rotations.rotate_volume(
                        self.V,
                        self.theta[start:end].to(self.V.device),
                        padding_mode="zeros",
                    )
        else:
            self.volumes = rotations.rotate_volume(
                self.V, self.theta.to(self.V.device), padding_mode="zeros"
            )

    def insert_volumes(self) -> torch.Tensor:
        """
        Insert the rotated volumes (`self.volumes`) into a 3D micrograph according to `self.coords`.

        Returns
        -------
        torch.Tensor
            Micrograph of shape (nz_out, nxy_out, nxy_out) containing all duplicates.
        """
        micro = insert_particles_into_micrograph(
            self.volumes,
            self.coords,
            pixel_size=self.dx,
            micro_shape=(self.nz_out, self.nxy_out, self.nxy_out),
        )
        return micro

    def forward(self) -> torch.Tensor:
        """
        Full pipeline: generate coordinates, random rotations, rotate volumes,
        and insert them into a micrograph.

        Returns
        -------
        torch.Tensor
            The final micrograph containing all duplicates, shape (nz_out, nxy_out, nxy_out).
            All-zeros if no candidate positions were generated.
        """
        self.generate_coordinates()
        self.generate_affine_matrices()
        if len(self.coords) == 0:
            return torch.zeros(
                self.nz_out, self.nxy_out, self.nxy_out, device=self.device
            )
        if self.chunk_size is None:
            self.rotate_volumes()
            micrograph = self.insert_volumes()
            if self.move_to_cpu:
                micrograph = micrograph.cpu()
        else:
            micrograph = torch.zeros(self.nz_out, self.nxy_out, self.nxy_out)
            for start in track(
                range(0, self.N, self.chunk_size),
                description="Rotating duplicates and insert into micrograph",
                transient=True,
                disable=not self.progressbars,
            ):
                end = min(start + self.chunk_size, self.N)
                volumes = rotations.rotate_volume(
                    self.V,
                    self.theta[start:end].to(self.V.device),
                    padding_mode="zeros",
                )
                if self.move_to_cpu:
                    volumes = volumes.cpu()
                micrograph = insert_particles_into_micrograph(
                    volumes,
                    self.coords[start:end],
                    pixel_size=self.dx,
                    micrograph=micrograph,
                )
        return micrograph
