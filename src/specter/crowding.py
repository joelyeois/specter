from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import lightning as L
import torch

from .arrays import clip_insert_bounds
from .coords import poisson_disk_neighbors, poisson_disk_neighbors_3d
from .progress import track

from . import rotations

if TYPE_CHECKING:
    from .ice import IceProfile

__all__ = [
    "poisson_disk_neighbors",
    "poisson_disk_neighbors_3d",
    "insert_particles_into_micrograph",
    "filter_by_z_density",
    "filter_by_local_z_density",
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


def filter_by_local_z_density(
    pts: torch.Tensor,
    z_bot: torch.Tensor,
    z_top: torch.Tensor,
    sigma_frac: float = 0.05,
    peak_amplitude: float = 1.0,
    baseline: float = 0.1,
    curve_points: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Filter points against a two-Gaussian z-density profile whose peaks sit on
    each point's *own* ice surfaces.

    The general form of :func:`filter_by_z_density`, which places both peaks at
    a single global ``±z_length / 2``. Under an
    :class:`~specter.ice.IceProfile` the surfaces move with (x, y), so a global
    pair of peaks would put adsorbed particles at the mean surface rather than
    the local one -- in the thin regions of a wedge, outside the ice entirely.

    Parameters
    ----------
    pts : torch.Tensor
        Particle coordinates with shape (N, 3) containing (x, y, z) positions.
    z_bot : torch.Tensor
        Lower ice surface at each point's (x, y), shape (N,), in Å.
    z_top : torch.Tensor
        Upper ice surface at each point's (x, y), shape (N,), in Å.
    sigma_frac : float, optional
        Gaussian width as a fraction of the local thickness. Default is 0.05.
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
        Probability density curve along z-axis with shape (curve_points,),
        evaluated at the mean surfaces.

    Notes
    -----
    The probability distribution is: P(z) = baseline + amplitude * (G1 + G2)
    where G1 and G2 are Gaussians centered at each point's own ``z_bot`` and
    ``z_top``.
    """
    if len(pts) == 0:
        return pts, torch.zeros(curve_points)

    thickness = (z_top - z_bot).clamp(min=1e-6)
    sigma = (sigma_frac * thickness).clamp(min=1e-6)
    z_pts = pts[:, 2]

    g1 = torch.exp(-0.5 * ((z_pts - z_bot) / sigma) ** 2)
    g2 = torch.exp(-0.5 * ((z_pts - z_top) / sigma) ** 2)

    probs = baseline + peak_amplitude * (g1 + g2)
    probs = (probs / probs.max()).clamp(0.0, 1.0)

    mask = torch.rand(len(z_pts)) < probs
    pts_filtered = pts[mask]

    # Curve along z, reported at the mean surfaces: with a profile there is no
    # single z-axis every point shares.
    z_min, z_max = float(z_bot.mean()), float(z_top.mean())
    sigma_curve = sigma_frac * (z_max - z_min)
    z_curve = torch.linspace(z_min, z_max, curve_points)
    g1_curve = torch.exp(-0.5 * ((z_curve - z_min) / sigma_curve) ** 2)
    g2_curve = torch.exp(-0.5 * ((z_curve - z_max) / sigma_curve) ** 2)
    p_curve = baseline + peak_amplitude * (g1_curve + g2_curve)
    z_distribution = (p_curve / p_curve.max()).clamp(0.0, 1.0)

    return pts_filtered, z_distribution


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

    A flat-slab wrapper around :func:`filter_by_local_z_density`, which takes
    the surfaces per point rather than as a single global pair.

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
    if len(pts) == 0:
        return pts, torch.zeros(curve_points)
    ones = torch.ones(len(pts), dtype=pts.dtype, device=pts.device)
    return filter_by_local_z_density(
        pts,
        -z_length / 2 * ones,
        z_length / 2 * ones,
        sigma_frac=sigma_frac,
        peak_amplitude=peak_amplitude,
        baseline=baseline,
        curve_points=curve_points,
    )


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
        Number of duplicate volumes to rotate per batch. Default 1. Rotating
        them all at once needs ~28 bytes of transient sampling grid per template
        voxel per duplicate, which a micrograph cannot afford, and batching them
        is free in both directions: wall time measured flat from 1 to 64 at
        micrograph scale while peak memory ran 0.55 GB to 30.1 GB. Raising this
        therefore only trades memory for nothing, and changes the order
        duplicates are summed into the accumulator, perturbing the output at
        float-rounding level.
    move_to_cpu : bool, optional
        If True, intermediate rotated volumes are moved to CPU to save GPU memory.
    progressbars : bool, optional
        If True, display progress bars for chunked operations.
    water_air_interface : bool, optional
        If True, applies a bimodal distribution along z-coordinates. Mimics the
        particle adsorption in cryo-EM ice-water interface.
    ice_profile : IceProfile, optional
        Laterally varying ice geometry. When given, '3d' placement is gated on
        each candidate's own column: a placement whose particle would poke out
        of the local slab is rejected, and ``water_air_interface`` adsorbs to
        the local surfaces via
        :func:`~specter.crowding.filter_by_local_z_density` rather than to a
        single global pair. Default None (a slab filling ``max_distance_z``,
        as before).
    particle_radius : float, optional
        Half-height in Å used for that fit test. Defaults to
        ``min_distance / 2``. Ignored when ``ice_profile`` is None.
    packing_backend : {'poisson_disk', 'shape'}, optional
        Placement algorithm. ``'poisson_disk'`` (default) is the original
        Bridson-sampled, bounding-sphere-exclusion placement above.
        ``'shape'`` instead packs via
        :func:`~specter.specimen.packing.pack_shapes_3d` -- the same
        Random-Sequential-Addition packer `TomogramSpecimenGenerator` uses,
        colliding the real rotated molecular footprint against a running
        occupancy grid instead of a bounding-sphere distance. Measured on
        specter's own benchmarks, bounding-sphere exclusion cannot exceed
        the density a molecule's envelope occupies within its own bounding
        sphere (~0.178 for a typical protein), while shape-aware RSA
        reaches physiological crowding densities (~0.2-0.25 volume
        fraction) on the same box -- see `pack_shapes_3d`'s own docstring
        for the measured numbers. Requires ``atom_coordinates``.

        Not yet ``ice_profile``-aware: placement is unconstrained by the
        local slab or ``water_air_interface`` when this backend is
        selected (candidates may land outside a laterally-varying ice
        profile). Wiring that in is separate, follow-up work -- see
        `pack_shapes_3d`'s ``region_mask`` parameter, which is exactly the
        hook a per-column slab mask would use.
    atom_coordinates : torch.Tensor, optional
        Atomic coordinates of the template molecule, shape (N, 3), in the
        molecule's own frame -- `specter.pdb.PDB.coordinates` is already
        centered the way this needs. Required (and used only) when
        ``packing_backend='shape'``, to rasterize the real footprint via
        `~specter.specimen.packing.build_species_mask` rather than
        colliding `V` itself (a rendered potential, not a binary shape).
    gap : float, optional
        Extra clearance baked into the shape-backend footprint mask, Å --
        forwarded to `build_species_mask`. Default 0.0. Ignored for
        ``packing_backend='poisson_disk'``, which uses `min_distance`
        instead.
    n_orientations : int, optional
        Size of the shape backend's per-species rotation cache, forwarded
        to `pack_shapes_3d`. Default 256. Ignored for ``'poisson_disk'``.
    packing_max_retries : int, optional
        Shape backend's attempts-per-instance ceiling, forwarded to
        `pack_shapes_3d` as ``max_retries``. This is the knob that sets
        achieved density -- see that function's own docstring for the
        measured density-vs-wall-time table. Default 1500. Ignored for
        ``'poisson_disk'``.
    packing_stall_patience : int, optional
        Shape backend's early-stop threshold, forwarded to `pack_shapes_3d`
        as ``stall_patience``. Default 5000. Ignored for
        ``'poisson_disk'``.
    packing_seed : int, optional
        Shape backend's RNG seed, forwarded to `pack_shapes_3d` as
        ``seed``. Distinct from ``seed`` above, which controls the
        poisson-disk backend's *starting point* strategy instead of an RNG
        seed. Default None. Ignored for ``'poisson_disk'``.
    n_candidates : int, optional
        Shape backend's candidate pool size (how many placement attempts
        RSA gets to work with before ``max_retries``/``stall_patience``
        cut it off). Default None: estimated as ``20 *
        grid_volume / footprint_volume``, clamped to [500, 200_000] --
        a rough oversampling factor, not a target count (RSA saturates
        well before exhausting the pool at realistic density). Ignored for
        ``'poisson_disk'``, which has its own ``n_points``.

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
        chunk_size: int = 1,
        move_to_cpu: bool = False,
        progressbars: bool = True,
        water_air_interface: bool = False,
        ice_profile: "IceProfile | None" = None,
        particle_radius: float | None = None,
        packing_backend: Literal["poisson_disk", "shape"] = "poisson_disk",
        atom_coordinates: torch.Tensor | None = None,
        gap: float = 0.0,
        n_orientations: int = 256,
        packing_max_retries: int = 1500,
        packing_stall_patience: int = 5000,
        packing_seed: int | None = None,
        n_candidates: int | None = None,
    ):
        super().__init__()
        if packing_backend == "shape" and atom_coordinates is None:
            raise ValueError(
                "packing_backend='shape' requires atom_coordinates (the "
                "template's real atomic coordinates, used to rasterize its "
                "footprint) -- pass the PDB's own PDB.coordinates."
            )
        self.packing_backend = packing_backend
        self.atom_coordinates = atom_coordinates
        self.gap = gap
        self.n_orientations = n_orientations
        self.packing_max_retries = packing_max_retries
        self.packing_stall_patience = packing_stall_patience
        self.packing_seed = packing_seed
        self.n_candidates = n_candidates

        self.register_buffer("V", V)
        self.dx = dx
        self.n = V.shape[0]
        self.min_distance = min_distance
        self.poisson_disc_method = method
        self.n_points = n_points
        self.seed = seed
        # Rotating a duplicate costs ~28 bytes per template voxel (the
        # (B, Z, Y, X, 3) sampling grid plus the resampled output), so rotating
        # them all at once is what a micrograph cannot afford: 864 duplicates of
        # a 256^3 template asks for 60 GiB in one allocation.
        #
        # 1 by default because batching these is free in both directions --
        # measured at micrograph scale, wall time is flat from 1 to 64
        # (86.9-88.9 s) while peak memory runs 0.55 GB to 30.1 GB. So a larger
        # value only ever trades memory for nothing. Raising it also perturbs
        # the output at float-rounding level (~4e-6 relative, measured), since
        # it changes the order duplicates are summed into the accumulator.
        self.chunk_size = chunk_size
        self.move_to_cpu = move_to_cpu
        self.progressbars = progressbars
        self.water_air_interface = water_air_interface
        self.ice_profile = ice_profile
        # min_distance is the caller's own estimate of how much room one
        # particle needs -- `MicrographSpecimenGenerator`'s callers pass
        # `pdb.max_diameter` -- so half of it is the particle's radius.
        self.particle_radius = (
            particle_radius if particle_radius is not None else min_distance / 2
        )

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
        Generate coordinates of duplicates.

        Dispatches on `packing_backend`. For ``'shape'``, see
        `_generate_coordinates_shape`. For ``'poisson_disk'`` (default):
        '2d' sampling places points in the xy plane and sets z to 0; '3d'
        sampling places points in the full 3D volume. Coordinates are
        stored in `self.coords`.
        """
        if self.packing_backend == "shape":
            self._generate_coordinates_shape()
            return
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
            if self.ice_profile is not None:
                # Reject anything that would poke out of its OWN column's slab.
                # `max_distance_z` spans the whole profile, so without this the
                # thin regions get particles sitting in vacuum.
                z_bot, z_top = self.ice_profile.surfaces_at(
                    coords[:, :2], self.nxy_out, self.dx
                )
                fits = (coords[:, 2] - self.particle_radius >= z_bot) & (
                    coords[:, 2] + self.particle_radius <= z_top
                )
                coords, z_bot, z_top = coords[fits], z_bot[fits], z_top[fits]
                if self.water_air_interface:
                    coords, self.z_distribution = filter_by_local_z_density(
                        coords, z_bot, z_top
                    )
            elif self.water_air_interface:
                coords, self.z_distribution = filter_by_z_density(
                    coords, self.max_distance_z
                )
        else:
            raise ValueError(
                f"Unknown method '{self.poisson_disc_method}'. Choose '2d' or '3d'."
            )
        self.coords = coords

    def _generate_coordinates_shape(self) -> None:
        """
        Generate coordinates (and orientations) via shape-aware RSA.

        Colliding the real rotated molecular footprint against a running
        occupancy grid instead of `min_distance`-apart bounding spheres --
        see `packing_backend`'s own docstring for why this reaches
        substantially higher density. A single species, duplicated: the
        candidate pool is one uniform `species_idx` (this class places one
        template, unlike `TomogramSpecimenGenerator`'s multi-species case).

        The accepted rotations are stored on `self._shape_rotations` for
        `generate_affine_matrices` to reuse directly -- they are the
        orientations the packer actually tested for collisions, so
        redrawing them at render time would render geometry that does not
        match what was packed (see that method's own comment).
        """
        from .specimen.packing import build_species_mask, pack_shapes_3d

        grid_shape = (self.nz_out, self.nxy_out, self.nxy_out)
        mask = build_species_mask(self.atom_coordinates, self.dx, gap=self.gap)

        n_candidates = self.n_candidates
        if n_candidates is None:
            grid_volume = self.nz_out * self.nxy_out * self.nxy_out
            mask_volume = max(int(mask.sum().item()), 1)
            n_candidates = int(min(max(20 * grid_volume / mask_volume, 500), 200_000))
        species_idx = torch.zeros(n_candidates, dtype=torch.long)

        coords, rotation_matrices, _accepted_idx, _occupancy = pack_shapes_3d(
            [mask],
            species_idx,
            grid_shape,
            self.dx,
            n_orientations=self.n_orientations,
            max_retries=self.packing_max_retries,
            stall_patience=self.packing_stall_patience,
            seed=self.packing_seed,
            device=str(self.device),
        )
        self.coords = coords
        self._shape_rotations = rotation_matrices

    def generate_affine_matrices(self) -> None:
        """
        Generate rotation matrices and corresponding affine matrices for
        each duplicate volume.

        For ``packing_backend='shape'``, reuses the orientations
        `_generate_coordinates_shape` already committed to during
        collision testing rather than drawing fresh ones -- redrawing
        would render a volume whose geometry doesn't match what the
        packer actually tested for overlaps (see
        `TomogramSpecimenGenerator._render_species_pool`'s identical
        comment). For ``'poisson_disk'``, orientation has no bearing on
        the (spherical) collision test, so rotations are drawn randomly
        here as before.

        Rotations are stored in `self.theta`.
        """
        self.N = len(self.coords)
        if self.packing_backend == "shape":
            R = self._shape_rotations
        else:
            R = rotations.random_rotation_matrix(self.N)
        # in case only one position was found, ensures R is (1,3,3)
        if len(R.shape) == 2:
            R = R.unsqueeze(0)
        self.theta = rotations.build_affine_matrix(R)

    def rotate_volumes(self) -> None:
        """
        Rotate the original volume according to the affine matrices `self.theta`.

        Rotated in batches of `chunk_size`; the results are stored in
        `self.volumes`. Note this materializes all `N` rotated volumes at once,
        so `forward` uses its own streaming loop instead -- this stays for
        callers that want the stack itself.
        """
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
            rotated = rotations.rotate_volume(
                self.V,
                self.theta[start:end].to(self.V.device),
                padding_mode="zeros",
            )
            self.volumes[start:end] = rotated.cpu() if self.move_to_cpu else rotated

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
        # Rotate and insert a chunk at a time, so neither the sampling grid nor
        # the stack of rotated duplicates is ever held for all N at once. The
        # accumulator follows `move_to_cpu`, matching where the rotated volumes
        # are sent -- at micrograph scale it is far larger than any chunk (a
        # 500 x 4096 x 4096 canvas is 33 GB) and is what `move_to_cpu` exists
        # to keep off the device.
        accumulator_device = torch.device("cpu") if self.move_to_cpu else self.device
        micrograph = torch.zeros(
            self.nz_out, self.nxy_out, self.nxy_out, device=accumulator_device
        )
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
            micrograph = insert_particles_into_micrograph(
                volumes.to(accumulator_device),
                self.coords[start:end],
                pixel_size=self.dx,
                micrograph=micrograph,
            )
        return micrograph
