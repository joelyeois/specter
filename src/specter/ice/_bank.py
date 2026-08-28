"""
Extracts ice sub-volumes from a cache of pre-generated, pre-converged ice
coordinate configurations, instead of running :class:`GradientSKIcemaker`'s
expensive L-BFGS optimisation per request.

See :func:`build_ice_cache` for how the cache itself is produced.
"""

from __future__ import annotations

import glob
import math
import os
import time
from typing import TYPE_CHECKING, Optional

import lightning as L
import matplotlib.figure
import torch

from ..arrays import (
    soft_voxelize_coordinates,
    soft_voxelize_coordinates_into,
)
from ..ice_data import ICE_CACHE_DIRNAME, bundled_ice_data
from ..progress import track
from ._energy import MLBOP
from ..potential import potential_from_deltas
from ._kernels import build_water_kernel
from ._random import RandomIcemaker

if TYPE_CHECKING:
    from ._profile import IceProfile


#: Largest magnitude a signed 16-bit fixed-point coordinate index takes. One
#: short of `int16`'s true minimum (-32768), so the mapping stays symmetric
#: about zero and the box centre lands exactly on index 0.
_FIXED_POINT_SCALE = 32767

#: Value of a config's ``coord_encoding`` key when its positions are stored as
#: :func:`encode_positions`' fixed-point indices. Absent from a config file
#: means the older raw-float16 storage, which :meth:`IceBank._load_config`
#: still reads -- the bundled ``ice_data/ice_cache`` predates this key.
FIXED_POINT_ENCODING = "int16_fixed"


def encode_positions(positions: torch.Tensor, box_L: float) -> torch.Tensor:
    """
    Quantize coordinates to signed 16-bit fixed-point indices for storage.

    Coordinates are bounded (wrapped into ``[-box_L/2, box_L/2)``), which is
    the precondition fixed-point needs and floating point cannot exploit.
    Storing them as raw ``float16`` instead spends 5 of 16 bits on an exponent
    covering a dynamic range these values never use, and leaves *relative*
    precision on an absolute quantity: the spacing between representable
    ``float16`` values grows with distance from the origin, reaching 0.0625 A
    across the outer octave of a 256 A cell -- where, since volume grows as
    r^3, half of all coordinates sit. This maps the box onto a uniform grid
    instead, so every coordinate gets the same ``box_L / (2 * 32767)``
    resolution (0.0039 A at ``box_L=256``) for the same two bytes.

    Measured on a converged 256 A config: raw ``float16`` storage costs 0.0137
    A RMS of coordinate accuracy and inflates the config's S(k) loss ~3800x
    (1.2e-4 -> 0.456), while this encoding costs 0.0011 A and ~13x (-> 0.0016).
    The rendered consequence of that difference is small -- 3.7% vs 0.3%
    relative RMS on the potential, both far below shot noise at any realistic
    dose -- so this is about not discarding the S(k) fidelity the optimiser
    exists to produce, rather than about visible image quality.

    ``int16`` rather than ``uint16``: ``torch.save`` cannot serialize
    ``torch.uint16`` as of torch 2.5.1 (``KeyError: 'dtype torch.uint16 is
    not recognized'``). A symmetric signed map has identical resolution.

    Parameters
    ----------
    positions : torch.Tensor
        Coordinates in Å, shape (N, 3), within ``[-box_L/2, box_L/2]``.
    box_L : float
        Cubic cell side length in Å.

    Returns
    -------
    torch.Tensor
        ``int16`` grid indices, same shape. Decode with
        :func:`decode_positions`.
    """
    scaled = positions.double() / (box_L / 2) * _FIXED_POINT_SCALE
    return scaled.round().clamp(-_FIXED_POINT_SCALE, _FIXED_POINT_SCALE).to(torch.int16)


def decode_positions(indices: torch.Tensor, box_L: float) -> torch.Tensor:
    """
    Reconstruct float32 coordinates from :func:`encode_positions`' indices.

    Parameters
    ----------
    indices : torch.Tensor
        ``int16`` grid indices, shape (N, 3).
    box_L : float
        Cubic cell side length in Å, from the config's ``box_L`` key.

    Returns
    -------
    torch.Tensor
        Coordinates in Å, float32.
    """
    return (indices.double() / _FIXED_POINT_SCALE * (box_L / 2)).float()


def default_ice_cache_dir() -> str:
    """
    Path to the bundled ice cache, so a standard user never needs to run
    :class:`GradientSKIcemaker` themselves -- see :func:`build_ice_cache` for
    how it was produced.

    Resolved through :mod:`importlib.resources` (see
    :func:`specter.ice_data.bundled_ice_data`), which is why the library lives
    at ``src/specter/ice_data/ice_cache`` rather than at the repository root:
    the data ships with the package and is found the same way regardless of
    install layout.

    Returns
    -------
    str
        Absolute path to the bundled ``ice_cache`` directory.
    """
    return str(bundled_ice_data(ICE_CACHE_DIRNAME))


def random_rotation_matrix(generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Draw a uniformly random 3D rotation matrix.

    QR decomposition of a random Gaussian matrix (Mezzadri, 2007), sign- and
    determinant-corrected for a proper rotation (no reflection).

    Deliberately *not* :func:`specter.rotations.random_rotation_matrix`
    (roma-based): roma's ``random_rotmat``/``random_unitquat`` only forwards
    the given ``generator`` to one of its three underlying random draws (the
    other two always fall back to the global RNG -- confirmed empirically,
    not documented), so it cannot give the fully generator-reproducible
    output :meth:`IceBank._extract_crop` needs. This QR-based path's only
    random draw is the single ``torch.randn(..., generator=generator)``
    below, so it is.

    Parameters
    ----------
    generator : torch.Generator, optional
        RNG to draw from. Default is the global RNG.

    Returns
    -------
    torch.Tensor
        Shape (3, 3), ``det(R) == 1``.
    """
    A = torch.randn(3, 3, generator=generator)
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.diagonal(R))
    Q = Q * d
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


class IceBank(L.LightningModule):
    """
    Draws randomly rotated, randomly translated ice sub-volumes from a
    library of pre-generated ice coordinate configurations.

    Each cached config is a single :class:`GradientSKIcemaker` run's
    positions, converged under the standard MLBOP recipe (see
    :func:`build_ice_cache`) and stored as a periodic cubic cell. Since
    generation is a one-time cost (~tens of minutes for a production-scale
    config), this class turns "get a new, statistically independent ice
    volume" into a request costing single-digit milliseconds on GPU: pick a
    cached config, pick a random rotation and a random center, gather
    candidate atoms via periodic wraparound of the *source*, rotate them
    into the requested volume's own local frame, and soft-voxelize.

    Unlike :class:`GradientSKIcemaker`/:class:`RandomIcemaker`, whose
    ``(n, dx, nz)`` are fixed at construction, this class takes them
    per-call — the whole point is serving arbitrarily-sized requests from a
    fixed-size cache.

    Rotation is exact (no interpolation) since it's applied to continuous
    coordinates before voxelization, unlike rotating an already-voxelized
    grid.

    Two request sizes are handled differently:

    - **Fits within a single cached config** (``n*dx``, ``n*dx``, ``nz*dx``
      each <= some cached config's own box size): use
      :meth:`generate_ice`/:meth:`generate_ice_deltas` -- a single crop,
      no relaxation needed, cheapest path.
    - **Larger than any single cached config**: use
      :meth:`generate_big_ice`/:meth:`generate_big_ice_deltas` -- tiles
      multiple independently rotated crops together and heals the tile
      boundaries with a short local MLBOP relaxation (naive concatenation
      alone leaves severe damage at the seams -- see
      ``dev/ice/seam_relax_256_*.py`` for the validation behind this).

    Parameters
    ----------
    cache_dir : str, optional
        Directory containing cached config ``.pt`` files (see
        :func:`build_ice_cache`). Every ``*.pt`` file in this directory is
        treated as a cache entry. Defaults to the bundled
        ``ice_data/ice_cache`` shipped with the repository.
    device : str or torch.device, optional
        Computation device. Default is ``"cpu"``.
    parameterization : str, optional
        Atomic scattering-factor parameterization for the ice kernel:
        ``'kirkland'``, ``'lobato'``, or ``'shtyrov'``. Default ``'shtyrov'``,
        matching `PotentialBuilder`'s own default. Note this only
        affects the kernel used to voxelize a crop's coordinates -- it does
        not change which cached configs (already-optimized coordinate sets)
        are drawn from.
    progressbars : bool, optional
        Whether to show progress bars over batched generation. Default is
        True.
    """

    method: str = "gd"
    """str: Always ``"gd"`` -- every cached config was generated via
    :class:`GradientSKIcemaker`. Kept for compatibility with code that reads
    ``icemaker.method`` (e.g. to log/report which ice model an
    externally-constructed icemaker uses)."""

    def __init__(
        self,
        cache_dir: str | None = None,
        device: str | torch.device = "cpu",
        parameterization: str = "shtyrov",
        progressbars: bool = True,
    ) -> None:
        super().__init__()
        if cache_dir is None:
            cache_dir = default_ice_cache_dir()
        self.cache_dir = cache_dir
        self.parameterization = parameterization
        self._config_paths = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if not self._config_paths:
            raise FileNotFoundError(
                f"No cached ice configs (*.pt) found in {cache_dir!r}"
            )
        # Loaded eagerly (not lazily): cache configs are small (a few MB each,
        # even smaller on disk at float16) and every config's box_L is needed
        # up front to pick a config that actually fits a given request --
        # checking box size *after* randomly picking a config would make
        # in-range requests non-deterministically fail depending on which
        # config got drawn, for any cache mixing config sizes.
        self._configs: list[dict] = [self._load_config(p) for p in self._config_paths]
        self._kernel_cache: dict[float, torch.Tensor] = {}
        self._source_pos_cache: dict[int, tuple[torch.device, torch.Tensor]] = {}
        self.progressbars = progressbars

        self.positions: Optional[torch.Tensor] = None
        """torch.Tensor or None: The most recently extracted crop's
        coordinates (N, 3), in the crop's own local frame, centered at the
        origin. Non-periodic -- unlike the other icemaker classes' outputs, a
        crop is a finite chunk of a larger periodic source, not periodic on
        its own."""
        self.current_icedeltas: Optional[torch.Tensor] = None
        """torch.Tensor or None: The most recently generated batch's
        soft-voxelized deltas."""
        self.n: Optional[int] = None
        self.dx: Optional[float] = None
        self.nz: Optional[int] = None
        self.box_x: Optional[float] = None
        self.box_y: Optional[float] = None
        self.box_z: Optional[float] = None

        if torch.device(device).type != "cpu":
            self.to(device)

    def __len__(self) -> int:
        return len(self._config_paths)

    # ------------------------------------------------------------------
    # Cache access
    # ------------------------------------------------------------------

    def _load_config(self, path: str) -> dict:
        data = torch.load(path, weights_only=False)
        # Two on-disk coordinate formats, distinguished by an explicit key
        # rather than by dtype: fixed-point indices (see `encode_positions`)
        # for anything generated since that encoding landed, raw floats for
        # everything before it -- including the bundled `ice_data/ice_cache`,
        # which must keep loading unchanged. Sniffing `int16` instead of
        # reading the key would misread any future raw-integer format.
        if data.get("coord_encoding") == FIXED_POINT_ENCODING:
            data["positions"] = decode_positions(data["positions"], data["box_L"])
        else:
            data["positions"] = data["positions"].float()  # upcast from float16
        return data

    def _get_kernel(self, dx: float) -> torch.Tensor:
        if dx not in self._kernel_cache:
            self._kernel_cache[dx] = build_water_kernel(dx, self.parameterization)
        return self._kernel_cache[dx]

    def _get_source_pos(self, config: dict) -> torch.Tensor:
        """
        Device-resident copy of ``config["positions"]``, cached per config.

        ``config["positions"]`` itself always stays on CPU (that's the
        eagerly-loaded copy from disk, shared regardless of which device
        this instance is on). The crop-extraction math in
        :meth:`_extract_crop` is otherwise CPU-only even when
        ``self.device`` is a GPU -- moving just the candidate-gathering
        computation there is a real win when many small crops are drawn in
        a loop (as :meth:`_place_tiles` does): ~4x faster per call in
        practice, and cheap in memory (a single ~527k-atom config is a few
        MB, and the transient candidate tensor per call is ~150-350 MB,
        trivial next to a modern GPU's total memory) -- see
        ``dev/ice/`` benchmarking behind this. Invalidated and rebuilt if
        ``self.device`` changes between calls.
        """
        key = id(config)
        cached = self._source_pos_cache.get(key)
        if cached is not None and cached[0] == self.device:
            return cached[1]
        pos = config["positions"].to(self.device)
        self._source_pos_cache[key] = (self.device, pos)
        return pos

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_crop(
        self,
        crop_extent: tuple[float, float, float],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Extract one randomly rotated, randomly translated crop.

        Parameters
        ----------
        crop_extent : tuple of float
            Requested ``(box_x, box_y, box_z)`` in Å.

        Returns
        -------
        torch.Tensor
            Shape (N, 3), local frame, centered at the origin.
        """
        max_extent = max(crop_extent)
        candidates = [c for c in self._configs if c["box_L"] >= max_extent]
        if not candidates:
            largest = max(c["box_L"] for c in self._configs)
            raise ValueError(
                f"Requested crop extent {crop_extent} Å exceeds every cached "
                f"config's own box size (largest available: {largest} Å) along "
                "at least one axis. This method only crops from a single "
                "config; use generate_big_ice()/generate_big_ice_deltas() "
                "instead for volumes larger than the cache."
            )
        idx = int(torch.randint(len(candidates), (1,), generator=generator).item())
        config = candidates[idx]
        box_L = config["box_L"]

        # Random draws stay on CPU regardless of self.device -- generator
        # (if given) is a CPU torch.Generator by convention throughout this
        # module/its tests, and these are tiny tensors anyway. Only the
        # actual candidate-gathering math below (O(n_source_atoms x
        # n_periodic_images), the real cost for small crops out of a large
        # source -- see _get_source_pos) runs on self.device.
        center_cpu = (torch.rand(3, generator=generator) - 0.5) * box_L
        R_cpu = random_rotation_matrix(generator=generator)

        source_pos = self._get_source_pos(config)
        half_extent = torch.tensor(
            [e / 2 for e in crop_extent], device=source_pos.device
        )
        reach = float(
            half_extent.norm()
        )  # corner-to-center distance of the (possibly non-cubic) crop box
        center = center_cpu.to(source_pos.device)
        R = R_cpu.to(source_pos.device)

        # Gather candidates via periodic wraparound of the source, using
        # enough periodic images to safely cover `reach` regardless of how
        # it compares to box_L. For center c and atom a each confined to
        # [-box_L/2, box_L/2), the offset c - a spans (-box_L, box_L), so
        # the largest image index that can ever matter is the largest
        # integer strictly below (box_L + reach) / box_L = 1 + reach/box_L,
        # i.e. m = floor(1 + reach/box_L) -- equivalently
        # 1 + floor(reach/box_L), which is always >= 1 (never just the
        # unwrapped primary cell, even for small reach: a center and atom
        # near opposite box edges still need the neighboring image to see
        # each other). See dev/ice/rotated_tiling_composition_check.py for
        # the original bug this fixes, and the notebook run that showed a
        # sharp, reach/box_L-dependent gap in atom count is not fixed by
        # this: m = ceil(reach/box_L - 0.5) silently gives m=0 (no
        # wraparound at all) whenever reach <= box_L/2, which is wrong
        # for any center near a box edge.
        m = 1 + math.floor(reach / box_L)
        shifts = (
            torch.arange(-m, m + 1, dtype=source_pos.dtype, device=source_pos.device)
            * box_L
        )
        sx, sy, sz = torch.meshgrid(shifts, shifts, shifts, indexing="ij")
        image_offsets = torch.stack([sx.flatten(), sy.flatten(), sz.flatten()], dim=1)

        candidates = (
            source_pos.unsqueeze(0) + image_offsets.unsqueeze(1) - center
        ).reshape(-1, 3)
        candidates = candidates[candidates.norm(dim=1) <= reach]

        local = candidates @ R
        in_crop = (local.abs() <= half_extent).all(dim=1)
        return local[in_crop].cpu()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_ice_deltas(
        self,
        n: int,
        dx: float,
        nz: int | None = None,
        batchsize: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Extract ``batchsize`` independent crops and soft-voxelize them.

        Each crop draws a fresh random config choice, rotation, and
        translation. The last crop's coordinates are left in
        ``self.positions`` (for :meth:`mlbop_energy`); the full batch is
        stored in ``self.current_icedeltas``.

        Parameters
        ----------
        n : int
            Number of voxels along x and y.
        dx : float
            Voxel size in Å.
        nz : int, optional
            Number of voxels along z. Defaults to ``n``.
        batchsize : int, optional
            Number of independent crops. Default is 1.
        generator : torch.Generator, optional
            RNG for reproducibility. Default is the global RNG.

        Returns
        -------
        icedeltas : torch.Tensor
            Soft-voxelized ice position volumes, shape (batchsize, nz, n, n).
        """
        self.n, self.dx, self.nz = n, dx, nz if nz is not None else n
        self.box_x = self.n * self.dx
        self.box_y = self.n * self.dx
        self.box_z = self.nz * self.dx
        crop_extent = (self.box_x, self.box_y, self.box_z)

        results = []
        for _ in track(
            range(batchsize),
            description="Extracting ice crops",
            disable=not self.progressbars or batchsize == 1,
            transient=True,
        ):
            self.positions = self._extract_crop(crop_extent, generator=generator)
            # periodic=True here isn't claiming the crop is self-periodic -- it's a
            # fix for a splat-grid fencepost bug: origin=n//2 centers voxel 0 exactly
            # at -half_extent (no atoms exist below it) while voxel n-1 sits a full
            # voxel inside +half_extent (atoms on both sides), so voxel 0 gets only
            # ~half the local density of an interior voxel and voxel n-1 gets nearly
            # all of it. Wrapping the overflow that would otherwise drop past index n
            # back onto index 0 restores voxel 0's missing "neighbor below"
            # contribution -- measured to bring low-face density from ~0.49-0.50x
            # interior up to ~0.92-0.94x, matching the high-face level (see
            # dev/ice/ for the face-statistics comparison behind this). Cheap: same
            # index-tensor op as the mask it replaces, no extra cost.
            vox = soft_voxelize_coordinates(
                self.positions,
                grid_shape=(self.nz, self.n, self.n),
                voxel_size=self.dx,
                periodic=True,
            )
            results.append(vox)
        self.current_icedeltas = torch.stack(results)
        return self.current_icedeltas

    def generate_ice(
        self,
        n: int,
        dx: float,
        nz: int | None = None,
        batchsize: int = 1,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Extract ``batchsize`` ice sub-volumes and convolve with the atomic
        potential kernel.

        Parameters
        ----------
        n : int
            Number of voxels along x and y.
        dx : float
            Voxel size in Å.
        nz : int, optional
            Number of voxels along z. Defaults to ``n``.
        batchsize : int, optional
            Number of independent ice volumes. Default is 1.
        device : torch.device or str, optional
            Target device for the output. Default is ``self.device``.
        generator : torch.Generator, optional
            RNG for reproducibility. Default is the global RNG.

        Returns
        -------
        icecubes : torch.Tensor
            Convolved ice potential volumes, shape (batchsize, nz, n, n).
        """
        if device is None:
            device = self.device
        self.generate_ice_deltas(
            n=n, dx=dx, nz=nz, batchsize=batchsize, generator=generator
        )
        assert self.current_icedeltas is not None
        kernel = self._get_kernel(dx)
        return potential_from_deltas(
            self.current_icedeltas.to(device), kernel.to(device), backend="conv3d"
        )

    # ------------------------------------------------------------------
    # Tiling for volumes larger than a single cached config
    # ------------------------------------------------------------------

    def _place_tiles(
        self,
        tile_grid_shape: tuple[int, int, int],
        tile_extent: float,
        seam_margin: float,
        generator: torch.Generator | None = None,
        splat_volume: torch.Tensor | None = None,
        splat_voxel_size: float | None = None,
        halo_margin: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[float, float, float]]:
        """
        Draw one independently rotated/translated crop of size
        ``tile_extent`` per grid cell and place them side by side.

        Parameters
        ----------
        splat_volume : torch.Tensor, optional
            If given, each tile's placed atoms are also splatted directly
            into this preallocated ``(nz, n, n)`` grid as soon as they're
            drawn (via :func:`soft_voxelize_coordinates_into`), instead of
            only being accumulated into the returned ``positions`` tensor.
            Splatting one tile (~10^5-10^6 atoms) at a time keeps peak
            memory bounded by a single tile's atom count rather than the
            full assembled volume's -- for a request spanning many tiles
            that difference is the whole point (see ``IceBank.generate_
            big_ice``'s docstring). Only valid to use when the caller
            won't subsequently move any positions (i.e. ``relax_steps ==
            0``): seam relaxation moves atoms after placement, which would
            leave the splat stale.
        splat_voxel_size : float, optional
            Voxel size (Å) for ``splat_volume``. Required if
            ``splat_volume`` is given.
        halo_margin : float, optional
            Extra distance (Å) beyond ``seam_margin``, used to additionally
            flag frozen atoms close enough to a face that a mobile atom
            (from this tile or a neighbor sharing that face) could still be
            within interaction range of them -- see ``halo_mask`` below.
            Default 0.0 (``halo_mask`` then equals ``mobile_mask``).

        Returns
        -------
        positions : torch.Tensor
            All placed atoms, shape (N, 3), centered at the origin.
        mobile_mask : torch.Tensor
            Bool mask, True for atoms within ``seam_margin`` of their own
            tile's own face (candidates for seam relaxation).
        halo_mask : torch.Tensor
            Bool mask, True for atoms within ``seam_margin + halo_margin``
            of their own tile's own face -- a superset of ``mobile_mask``
            wide enough to cover every frozen atom that could still
            interact with some mobile atom, given the energy model's finite
            interaction range. Two tiles placed side by side share the same
            physical face location, so this per-tile, own-face-only test
            already covers atoms relevant to a neighboring tile's mobile
            atoms too, without needing an explicit cross-tile neighbor
            search. Used by ``_relax_seams`` to avoid feeding the entire
            assembled volume into the energy model on every step.
        assembled_box : tuple of float
            The full tiled box size, ``(nx*tile_extent, ny*tile_extent,
            nz*tile_extent)`` -- generally larger than the final requested
            extent; the caller trims down afterward.
        """
        nx, ny, nz_tiles = tile_grid_shape
        half = tile_extent / 2
        assert (
            self.box_x is not None and self.box_y is not None and self.box_z is not None
        ), (
            "box_x/box_y/box_z must be set (by generate_big_ice_deltas) before _place_tiles"
        )
        all_parts: list[torch.Tensor] = []
        mobile_parts: list[torch.Tensor] = []
        halo_parts: list[torch.Tensor] = []
        tile_indices = [
            (ix, iy, iz)
            for ix in range(nx)
            for iy in range(ny)
            for iz in range(nz_tiles)
        ]
        for ix, iy, iz in track(
            tile_indices,
            description="Placing ice tiles",
            disable=not self.progressbars or len(tile_indices) == 1,
            transient=True,
        ):
            offset = torch.tensor(
                [
                    (ix - (nx - 1) / 2) * tile_extent,
                    (iy - (ny - 1) / 2) * tile_extent,
                    (iz - (nz_tiles - 1) / 2) * tile_extent,
                ]
            )
            crop = self._extract_crop(
                (tile_extent, tile_extent, tile_extent), generator=generator
            )
            dist_to_face = crop.abs()
            mobile_parts.append((dist_to_face > (half - seam_margin)).any(dim=1))
            halo_parts.append(
                (dist_to_face > (half - seam_margin - halo_margin)).any(dim=1)
            )
            crop_global = crop + offset
            all_parts.append(crop_global)
            if splat_volume is not None:
                assert splat_voxel_size is not None
                # periodic=True wraps only atoms landing outside splat_volume's own
                # extent (the true outer boundary of the requested output, since
                # splat_volume is sized to the final (nz, n, n) request, not the
                # possibly-larger tile grid) -- see generate_ice_deltas for why this
                # matters (a splat-grid fencepost bug, not an actual periodicity
                # claim). Interior tile-to-tile seams are untouched: an atom near
                # one tile's own face still lands at an in-bounds index of the
                # shared buffer, so this doesn't change anything there.
                #
                # But this only holds if crop_global itself never strays far past
                # splat_volume's own extent -- and a tile's own footprint
                # (tile_extent) routinely does, by design: every edge tile's
                # assembled-box placement overhangs the true requested box (see
                # assembled_box's own docstring above), and a "big request" that
                # only needs a single tile (assembled_box == tile_extent >
                # requested box) overhangs on every axis. _scatter_splat's
                # periodic=True is an *unconditional* index % grid_shape, which
                # doesn't distinguish "just past by a fencepost voxel" from
                # "hundreds of Å outside" -- the latter previously got
                # wrapped back in from arbitrary far positions instead of being
                # dropped, silently inflating ice density by up to
                # (tile_extent / requested box)^3 (measured ~5.6x for a 256 A
                # tile filling a 160 A request -- see
                # dev/ice/analytic_tile_insertion_benchmark.py). Explicitly
                # dropping atoms beyond the true requested half-extent first --
                # the same bound generate_big_ice_deltas's own `keep` filter
                # uses after relaxation, just applied per-tile before splatting
                # instead -- leaves only genuine sub-voxel fencepost overflow for
                # periodic=True to wrap, restoring the original intent without
                # reintroducing the low-face density bug periodic=True was added
                # to fix.
                keep = (
                    (crop_global[:, 0].abs() <= self.box_x / 2)
                    & (crop_global[:, 1].abs() <= self.box_y / 2)
                    & (crop_global[:, 2].abs() <= self.box_z / 2)
                )
                soft_voxelize_coordinates_into(
                    splat_volume, crop_global[keep], splat_voxel_size, periodic=True
                )
        positions = torch.cat(all_parts, dim=0)
        mobile_mask = torch.cat(mobile_parts, dim=0)
        halo_mask = torch.cat(halo_parts, dim=0)
        assembled_box = (nx * tile_extent, ny * tile_extent, nz_tiles * tile_extent)
        return positions, mobile_mask, halo_mask, assembled_box

    def _relax_seams(
        self,
        positions: torch.Tensor,
        mobile_mask: torch.Tensor,
        halo_mask: torch.Tensor,
        box: tuple[float, float, float],
        n_steps: int,
        lr: float,
        mlbop_target: float | None,
        device: torch.device | str,
    ) -> torch.Tensor:
        """
        Short local MLBOP-only relaxation of atoms near tile seams, holding
        every other atom fixed. See ``dev/ice/seam_relax_256_assemble.py``
        for the validation behind this: naive tiling alone leaves E/atom
        flipped from favorable to unfavorable at the seams, and this
        recovers most of it within a few seconds.

        Only ``halo_mask`` atoms (mobile atoms plus the surrounding frozen
        band from ``_place_tiles``) are ever fed into the energy model.
        ML-BOP's own cutoff is ~3.55 A (``MLBOP.r_cut``), and the three-body
        term reaches one further hop, so anything outside that band cannot
        influence a mobile atom's energy or gradient -- including it would
        only add cost (this model's per-step cost scales with however many
        atoms are passed in, not just how many need a gradient), not
        accuracy. For a many-tile assembly the excluded bulk can be the
        overwhelming majority of atoms, so this matters a lot in practice.
        The untouched bulk (``~halo_mask``) is reattached unchanged at the
        end without ever being moved to ``device``.

        Explicitly wrapped in ``torch.enable_grad()`` since this is called
        from ``generate_big_ice``/``generate_big_ice_deltas``, which callers
        commonly wrap in ``torch.no_grad()`` (matching how downstream
        imaging pipelines run ice generation under inference mode) --
        without this, ``requires_grad_(True)`` on a leaf tensor is silently
        overridden by the ambient no-grad context and ``loss.backward()``
        raises (no ``grad_fn`` was ever built).
        """
        if not mobile_mask.any():
            return positions
        far_mask = ~halo_mask
        local_frozen_mask = halo_mask & ~mobile_mask
        with torch.enable_grad():
            model = MLBOP(device=device)
            frozen = positions[local_frozen_mask].to(device)
            mobile = positions[mobile_mask].to(device).clone().requires_grad_(True)
            opt = torch.optim.Adam([mobile], lr=lr)
            for _ in track(
                range(n_steps),
                description="Relaxing tile seams",
                disable=not self.progressbars,
                transient=True,
            ):
                opt.zero_grad()
                full = torch.cat([frozen, mobile], dim=0)
                result = model.compute_energy(full, box_size=box, pbc=True)
                loss = (
                    result["E_per_atom"]
                    if mlbop_target is None
                    else (result["E_per_atom"] - mlbop_target) ** 2
                )
                loss.backward()
                opt.step()
        return torch.cat(
            [positions[far_mask], frozen.cpu(), mobile.detach().cpu()], dim=0
        )

    def generate_big_ice_deltas(
        self,
        n: int,
        dx: float,
        nz: int | None = None,
        batchsize: int = 1,
        tile_extent: float | None = None,
        seam_margin: float = 6.0,
        relax_steps: int = 0,
        relax_lr: float = 0.01,
        mlbop_target: float | None = -0.413,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Build a volume larger than any single cached config by tiling
        multiple independently rotated/translated crops together, then
        healing the tile boundaries with a short local MLBOP relaxation.

        For requests that fit within a single cached config, prefer
        :meth:`generate_ice_deltas` instead -- it's cheaper (no relaxation
        needed) and this method's tiling machinery adds unnecessary
        overhead for that case.

        Parameters
        ----------
        n : int
            Number of voxels along x and y.
        dx : float
            Voxel size in Å.
        nz : int, optional
            Number of voxels along z. Defaults to ``n``.
        batchsize : int, optional
            Number of independent volumes. Each draws a fresh tiling and
            relaxation. Default is 1.
        tile_extent : float, optional
            Size (Å) of each tile before assembly. Defaults to the smallest
            cached config's own box size, so any tile position can draw
            from any library entry. Larger tiles mean fewer seams (cheaper
            relaxation, better quality) but require a large enough cached
            config to support them.
        seam_margin : float, optional
            Distance (Å) from a tile's own face within which an atom is
            treated as a seam candidate for relaxation. Default 6.0 (matches
            the validated setting from this session's 256^3/128^3 tests).
        relax_steps : int, optional
            Adam steps for the seam relaxation. Default 0 (skip relaxation
            entirely -- naive tiling only, matching every higher-level
            caller's own default). Set to e.g. 200 (the point at which the
            validated 256^3 test had already plateaued) for production-
            quality seams -- see the validation notes on ``_relax_seams``
            for why unrelaxed seams carry measurably unfavorable energy.
        relax_lr : float, optional
            Adam learning rate for the relaxation. Default 0.01.
        mlbop_target : float or None, optional
            Relaxation target energy (see :meth:`_relax_seams`). Default
            -0.413, the real LDA-80K MD reference value.
        device : torch.device or str, optional
            Device to run the relaxation on. Default is ``self.device``.
        generator : torch.Generator, optional
            RNG for reproducibility. Default is the global RNG.

        Returns
        -------
        icedeltas : torch.Tensor
            Soft-voxelized ice position volumes, shape (batchsize, nz, n, n).
        """
        if device is None:
            device = self.device
        self.n, self.dx, self.nz = n, dx, nz if nz is not None else n
        self.box_x = self.n * self.dx
        self.box_y = self.n * self.dx
        self.box_z = self.nz * self.dx

        if tile_extent is None:
            tile_extent = min(c["box_L"] for c in self._configs)
        nx_tiles = max(1, math.ceil(self.box_x / tile_extent))
        ny_tiles = max(1, math.ceil(self.box_y / tile_extent))
        nz_tiles = max(1, math.ceil(self.box_z / tile_extent))

        results = []
        for _ in track(
            range(batchsize),
            description="Generating tiled ice volumes",
            disable=not self.progressbars or batchsize == 1,
            transient=True,
        ):
            # relax_steps == 0 means positions never move after placement,
            # so each tile can be splatted into the output the moment it's
            # drawn -- peak memory then scales with one tile's atom count,
            # not the whole assembled volume's (see _place_tiles). Seam
            # relaxation moves atoms after the fact, so that path still
            # needs the full assembled `positions` before voxelizing.
            vox = (
                torch.zeros(self.nz, self.n, self.n, device=device)
                if relax_steps == 0
                else None
            )
            # Only widen mobile_mask into a halo when relaxation will actually
            # run -- otherwise halo_mask is dead weight (relax_steps == 0
            # never looks at it).
            halo_margin = 2.0 * MLBOP().r_cut if relax_steps > 0 else 0.0
            positions, mobile_mask, halo_mask, assembled_box = self._place_tiles(
                (nx_tiles, ny_tiles, nz_tiles),
                tile_extent,
                seam_margin,
                generator=generator,
                splat_volume=vox,
                splat_voxel_size=self.dx if vox is not None else None,
                halo_margin=halo_margin,
            )
            if relax_steps > 0 and mobile_mask.any():
                positions = self._relax_seams(
                    positions,
                    mobile_mask,
                    halo_mask,
                    assembled_box,
                    relax_steps,
                    relax_lr,
                    mlbop_target,
                    device,
                )
            keep = (
                (positions[:, 0].abs() <= self.box_x / 2)
                & (positions[:, 1].abs() <= self.box_y / 2)
                & (positions[:, 2].abs() <= self.box_z / 2)
            )
            self.positions = positions[keep]
            if vox is None:
                # periodic=True: see generate_ice_deltas -- same outer-boundary
                # fencepost fix, applied here to the final keep-trimmed, already-
                # relaxed atom set.
                vox = soft_voxelize_coordinates(
                    self.positions,
                    grid_shape=(self.nz, self.n, self.n),
                    voxel_size=self.dx,
                    periodic=True,
                )
            results.append(vox)
        self.current_icedeltas = torch.stack(results)
        return self.current_icedeltas

    def generate_big_ice(
        self,
        n: int,
        dx: float,
        nz: int | None = None,
        batchsize: int = 1,
        tile_extent: float | None = None,
        seam_margin: float = 6.0,
        relax_steps: int = 0,
        relax_lr: float = 0.01,
        mlbop_target: float | None = -0.413,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Build a volume larger than any single cached config (see
        :meth:`generate_big_ice_deltas`) and convolve with the atomic
        potential kernel.

        Parameters are identical to :meth:`generate_big_ice_deltas`.

        Returns
        -------
        icecubes : torch.Tensor
            Convolved ice potential volumes, shape (batchsize, nz, n, n).

        Notes
        -----
        Clears ``self.current_icedeltas`` before returning -- see the comment at
        the release. :meth:`generate_ice` keeps its own, much smaller, deltas.
        """
        if device is None:
            device = self.device
        self.generate_big_ice_deltas(
            n=n,
            dx=dx,
            nz=nz,
            batchsize=batchsize,
            tile_extent=tile_extent,
            seam_margin=seam_margin,
            relax_steps=relax_steps,
            relax_lr=relax_lr,
            mlbop_target=mlbop_target,
            device=device,
            generator=generator,
        )
        assert self.current_icedeltas is not None
        kernel = self._get_kernel(dx)
        deltas = self.current_icedeltas.to(device)
        # Released here, unlike `generate_ice`'s small-volume path, which keeps
        # it for inspection. This method exists for volumes too large to come
        # from a single cached config, so the deltas are the same size as the
        # potential being returned -- 33.6 GB at micrograph_size -- and holding
        # them on the instance afterwards keeps a whole extra canvas alive for
        # an attribute nothing reads once the convolution is done.
        self.current_icedeltas = None
        potential = potential_from_deltas(deltas, kernel.to(device), backend="conv3d")
        del deltas
        return potential

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def mlbop_energy(self, pbc: bool = False) -> dict[str, float]:
        """
        Score the last-extracted crop against the ML-BOP potential.

        See :meth:`specter.ice._energy.MLBOP.compute_energy`. Defaults to
        non-periodic (``pbc=False``), unlike the other icemaker classes --
        an extracted crop is a finite chunk of a larger periodic source, not
        periodic on its own (see :meth:`generate_ice_deltas`). Pass
        ``pbc=True`` only if you know the requested extent exactly matches
        the source config's own box size.

        Parameters
        ----------
        pbc : bool, optional
            Whether to treat the crop as periodic. Default is False.

        Returns
        -------
        dict[str, float]
            See :meth:`specter.ice._energy.MLBOP.compute_energy` for the
            fields returned.
        """
        assert self.positions is not None, (
            "No positions -- call generate_ice_deltas() first"
        )
        assert (
            self.box_x is not None and self.box_y is not None and self.box_z is not None
        )
        box = (self.box_x, self.box_y, self.box_z)
        model = MLBOP(device=self.positions.device)
        with torch.no_grad():
            result = model.compute_energy(self.positions, box_size=box, pbc=pbc)
        return {k: v.item() for k, v in result.items()}

    def plot_diagnostics(
        self, save_path: str | None = None, show: bool = True
    ) -> tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
        """
        Compute ML-BOP energy/structure stats and S(k) radial profiles for
        every config in the cache, and plot them for a quick visual
        quality check of the library (e.g. after (re)running
        :func:`build_ice_cache`).

        Each config is scored with ``pbc=True`` (unlike :meth:`mlbop_energy`'s
        own ``pbc=False`` default) since a full cached config -- unlike an
        extracted crop -- genuinely is the periodic cell it was optimised
        as, and on ``self.device`` (under ``torch.no_grad()`` -- no
        gradients are needed here): scoring a full ~527k-atom config takes
        ~0.2s on GPU, so across 20 configs this stays a quick quality check
        rather than several minutes of CPU time.

        Parameters
        ----------
        save_path : str, optional
            If given, saves the energy-summary plot to
            ``f"{save_path}_energy.png"`` and the S(k) overlay to
            ``f"{save_path}_sk.png"``.
        show : bool, optional
            Whether to display the plots. Default True.

        Returns
        -------
        energy_fig, sk_fig : matplotlib.figure.Figure
            The two diagnostic figures.
        """
        import matplotlib.pyplot as plt

        from ..arrays import radial_profile_3d
        from ..fft import fft3
        from ._kernels import compute_native_target

        model = MLBOP(device=self.device)
        labels, energies, sk_profiles, dks = [], [], [], []
        for path, config in zip(self._config_paths, self._configs):
            pos, box_L, n, dx = (
                config["positions"],
                config["box_L"],
                config["n"],
                config["dx"],
            )
            labels.append(os.path.basename(path))
            with torch.no_grad():
                result = model.compute_energy(
                    pos.to(self.device), box_size=(box_L,) * 3, pbc=True
                )
            energies.append({k: v.item() for k, v in result.items()})
            vox = soft_voxelize_coordinates(
                pos, grid_shape=(n, n, n), voxel_size=dx, periodic=True
            )
            amp = torch.abs(fft3(vox, shift=True)) / (pos.shape[0] ** 0.5)
            _, prof = radial_profile_3d(amp, return_r=True)
            sk_profiles.append(prof)
            dks.append(1 / n / dx)

        ref_n, ref_dx = self._configs[0]["n"], self._configs[0]["dx"]
        native_k, native_f = compute_native_target(n=ref_n, dx=ref_dx)

        fig1, axes = plt.subplots(1, 3, figsize=(max(6, 0.4 * len(labels) + 4), 4))
        x = list(range(len(labels)))
        for ax, key, title, target in [
            (axes[0], "E_per_atom", "E/atom (eV)", -0.413),
            (axes[1], "rij_var", "O-O distance variance", None),
            (axes[2], "theta_var", "O-O-O angle variance", None),
        ]:
            ax.bar(x, [e[key] for e in energies])
            if target is not None:
                ax.axhline(
                    target, ls="--", color="k", lw=1, label=f"MD target ({target})"
                )
                ax.legend(fontsize=8)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels([str(i) for i in x], fontsize=6, rotation=90)
        fig1.suptitle(f"IceBank cache diagnostics -- {len(labels)} configs")
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        for i, (prof, dk) in enumerate(zip(sk_profiles, dks)):
            k_axis = torch.arange(len(prof)) * dk
            ax2.plot(k_axis[1:], prof[1:], alpha=0.5, lw=1, color="C0")
        ax2.plot(
            native_k[native_k > 0],
            native_f[native_k > 0],
            "k--",
            lw=2,
            label="native MD target",
        )
        ax2.set_xlabel("k (1/Å)")
        ax2.set_ylabel("radial |F(k)|")
        ax2.set_xlim(0, 1.0)
        ax2.legend()
        ax2.set_title(f"S(k) radial profile -- {len(labels)} cached configs")
        fig2.tight_layout()

        if save_path is not None:
            fig1.savefig(f"{save_path}_energy.png", dpi=130)
            fig2.savefig(f"{save_path}_sk.png", dpi=130)
        if show:
            plt.show()
        return fig1, fig2


def ice_config_filename(seed: int) -> str:
    """
    Filename a cached ice config generated with ``seed`` is stored under.

    Named after the seed rather than a position in the generation batch so
    that extending an existing cache is safe: a second run started at a
    higher ``seed_start`` writes new files instead of overwriting the
    earlier run's, and re-running an identical request is idempotent (it
    reproduces the same seeds, hence the same filenames). For the usual
    ``seed_start=0`` this is the same ``config_000.pt``, ``config_001.pt``,
    ... sequence the bundled ``ice_data/ice_cache`` uses.

    Parameters
    ----------
    seed : int
        Seed the config was generated with.

    Returns
    -------
    str
        Bare filename, e.g. ``"config_007.pt"``.
    """
    return f"config_{seed:03d}.pt"


def build_one_ice_config(
    save_path: str,
    n: int = 256,
    dx: float = 1.0,
    n_steps: int = 600,
    seed: int = 0,
    device: str | torch.device = "cuda",
    progressbars: bool = True,
) -> dict:
    """
    Generate a single ice coordinate config and save it (fixed-point) to
    ``save_path``.

    One unit of work for :func:`build_ice_cache` and for ``specter build
    ice``, which differ only in how they schedule these calls. A fresh
    :class:`GradientSKIcemaker` run under the standard MLBOP recipe
    (``rep_strength=0, mlbop_strength=0.5, mlbop_target=-0.413``,
    ``tol=1e-4``/``patience=10``). That recipe is deliberately not
    parameterised: ``mlbop_target=-0.413`` is a measured property of real
    LDA-80K MD ice rather than a preference, and configs generated under
    different recipes are different phases of ice that
    :class:`IceBank` would then draw from interchangeably.

    Parameters
    ----------
    save_path : str
        Full path of the ``.pt`` file to write. Its parent directory must
        already exist.
    n : int, optional
        Number of voxels along each side (box size = ``n * dx``). Cubic
        only: :class:`IceBank` stores one scalar ``box_L`` per config and
        filters candidate configs against it, so a non-cubic source cell
        has no representation. Default 256.
    dx : float, optional
        Voxel size in Å. Default 1.0.
    n_steps : int, optional
        L-BFGS step ceiling (an upper bound -- ``tol``/``patience`` still
        stop a plateaued run early). Default 600.
    seed : int, optional
        Global torch seed set before initialisation, making this config
        reproducible. Default 0.
    device : str or torch.device, optional
        Computation device. Default ``"cuda"``.
    progressbars : bool, optional
        Whether to show a progress bar over optimisation steps. Default True.

    Returns
    -------
    dict
        The saved metadata, without ``positions``: ``box_L``, ``n``, ``dx``,
        ``seed``, ``coord_encoding``, ``n_steps``, ``n_steps_actual``,
        ``wall_time``, ``stopped_early``, ``energy``, ``sk_loss``, ``recipe``.
        ``sk_loss`` is measured on the coordinates as they will be read back
        (encoded then decoded), so it describes the file rather than the
        optimiser's in-memory state -- see the comment at its computation for
        why neither alternative is trustworthy.

    Notes
    -----
    Coordinates are written as fixed-point indices via
    :func:`encode_positions`, not as raw floats. Configs predating that
    encoding (the bundled ``ice_data/ice_cache``) stay readable; see
    :meth:`IceBank._load_config`.
    """
    from ._gradient import GradientSKIcemaker

    # The standard recipe, bound to names so the values actually optimised
    # with and the values recorded in the saved metadata below cannot drift
    # apart.
    rep_strength, mlbop_strength, mlbop_target = 0.0, 0.5, -0.413
    tol, patience = 1e-4, 10

    started = time.perf_counter()
    torch.manual_seed(seed)
    gd = GradientSKIcemaker(n=n, dx=dx, device=device, progressbars=progressbars)
    gd.init_random()
    history = gd.optimize(
        n_steps=n_steps,
        record_every=n_steps,
        rep_strength=rep_strength,
        mlbop_strength=mlbop_strength,
        mlbop_target=mlbop_target,
        tol=tol,
        patience=patience,
    )
    energy = gd.mlbop_energy()
    assert gd.positions is not None

    # Measure S(k) fidelity on the coordinates as they will be READ BACK --
    # encoded, then decoded -- rather than on the float32 ones held in memory,
    # and never from `history["sk_loss"]`. Both alternatives misreport:
    #
    # - `history` records at `step % record_every == 0` plus one final entry
    #   only when tol/patience triggers, so at `record_every=n_steps` a run
    #   that uses its whole budget has exactly ONE record -- step 0, the
    #   pre-optimisation value of the random initialisation. The bundled
    #   `ice_data/ice_cache` carries that artifact: its ten budget-exhausting
    #   configs all record ~5e4, which is simply what a fresh random init
    #   scores at this size, while their stored coordinates score ~0.4-2.0.
    # - Storage quantization is not negligible in this metric even at the
    #   fixed-point resolution used here (~13x on S(k) loss; the raw float16
    #   this replaced cost ~3800x -- see `encode_positions`). A number
    #   describing coordinates nobody can load back is not a quality record.
    box_L = n * dx
    positions = encode_positions(gd.positions, box_L)
    with torch.no_grad():
        sk_loss, _ = gd._sk_loss(
            decode_positions(positions, box_L).to(gd.device),
            rep_strength=0.0,
            mlbop_strength=0.0,
        )

    metadata = {
        "box_L": box_L,
        "n": n,
        "dx": dx,
        "seed": seed,
        # How `positions` is stored, read by `IceBank._load_config`. Explicit
        # rather than inferred from dtype so a future encoding can be added
        # without either format having to guess about the other.
        "coord_encoding": FIXED_POINT_ENCODING,
        "n_steps": n_steps,
        # Number of outer L-BFGS steps actually taken, and the seconds they
        # took. Recorded under the same keys the bundled ice_data/ice_cache
        # uses, so cost per step stays derivable per config -- that ratio,
        # not wall time alone, is what transfers to other hardware and cell
        # sizes. Taken from `history["step"]` only when tol/patience fired,
        # since that is the sole case where a final entry is appended; a run
        # that used its whole budget has just the step-0 record (the same
        # recording quirk that makes `history["sk_loss"]` untrustworthy
        # above), and by definition took all `n_steps` of it.
        "n_steps_actual": (
            history["step"][-1] + 1 if history["stopped_early"] else n_steps
        ),
        "wall_time": time.perf_counter() - started,
        "stopped_early": history["stopped_early"],
        "energy": energy,
        "sk_loss": sk_loss.item(),
        # Stored per config, under the same key the bundled ice_data/ice_cache
        # entries use: a cache directory can hold configs from several runs,
        # and which recipe produced one determines which phase of ice it is.
        "recipe": {
            "rep_strength": rep_strength,
            "mlbop_strength": mlbop_strength,
            "mlbop_target": mlbop_target,
            "tol": tol,
            "patience": patience,
        },
    }
    torch.save({"positions": positions, **metadata}, save_path)
    return metadata


def build_ice_cache(
    cache_dir: str,
    num_configs: int,
    n: int = 256,
    dx: float = 1.0,
    n_steps: int = 600,
    device: str | torch.device = "cuda",
    seed_start: int = 0,
    progressbars: bool = True,
) -> None:
    """
    Generate ``num_configs`` independent ice coordinate configs and save
    them (fixed-point) to ``cache_dir`` for later use with :class:`IceBank`.

    One file per config, via :func:`build_one_ice_config`. This is the
    expensive, one-time cost the whole point of :class:`IceBank` is to
    amortize -- expect on the order of tens of minutes per config at
    production scale (``n=256, dx=1.0``), so this is meant to be run once
    (or occasionally, to extend/refresh the library), not per-session.

    This function runs all ``num_configs`` sequentially, on a single
    device, in one process, and overwrites any config already present for
    the same seed. Prefer ``specter build ice`` (equivalently
    :func:`specter.pipelines.run_build_ice_cache`) for a large generation
    run: it shards configs across several GPUs, skips configs the cache
    already has so an interrupted run resumes, and records a manifest of
    the convergence quality actually achieved.

    ``n_steps=600`` (vs. the class-level default of 400): S(k) gains taper
    by ~200-300 steps at this scale, but ``E_per_atom`` keeps improving out
    past 600 for ~527k-atom systems -- since these configs are permanent,
    shared assets, the extra convergence is worth the one-time cost. ``tol``/
    ``patience`` still apply on top, so a config that plateaus early stops
    early rather than burning the full budget.

    Parameters
    ----------
    cache_dir : str
        Directory to write config files to (named by
        :func:`ice_config_filename`). Created if it doesn't exist.
    num_configs : int
        Number of independent configs to generate.
    n : int, optional
        Number of voxels along each side (box size = ``n * dx``). Default 256.
    dx : float, optional
        Voxel size in Å. Default 1.0.
    n_steps : int, optional
        L-BFGS step ceiling per config (upper bound -- see ``tol``).
        Default 600.
    device : str or torch.device, optional
        Computation device. Default ``"cuda"``.
    seed_start : int, optional
        First seed used; config ``i`` uses seed ``seed_start + i``. Default 0.
    progressbars : bool, optional
        Whether to show progress bars during generation. Default True.
    """
    os.makedirs(cache_dir, exist_ok=True)
    for i in range(num_configs):
        seed = seed_start + i
        build_one_ice_config(
            os.path.join(cache_dir, ice_config_filename(seed)),
            n=n,
            dx=dx,
            n_steps=n_steps,
            seed=seed,
            device=device,
            progressbars=progressbars,
        )


def resolve_icemaker(
    ice_model: str | None,
    pixel_size: float,
    nxy: int,
    nz: int,
    ice_cache_dir: str | None = None,
    icemaker: "IceBank | RandomIcemaker | None" = None,
    parameterization: str = "shtyrov",
    progressbars: bool = True,
) -> "IceBank | RandomIcemaker | None":
    """
    Resolve the ``ice_model``/``icemaker``/``ice_cache_dir`` kwargs shared
    across :class:`~specter.specimen.MicrographSpecimenGenerator`,
    :class:`~specter.imagegenerator.MicrographGenerator`,
    :class:`~specter.imagegenerator.TiltSeriesGenerator`,
    :class:`~specter.imagegenerator.ImageGenerator`, and
    :class:`~specter.imagegenerator.ImageGeneratorFromCoordinates` into a
    concrete icemaker instance (or ``None`` if ice is disabled).

    Parameters
    ----------
    ice_model : str or None
        ``'gd'`` (:class:`IceBank`), ``'random'`` (:class:`RandomIcemaker`),
        ``'none'`` or ``None`` (no ice). Ignored when ``icemaker`` is given.
    pixel_size : float
        Voxel size in Å, used to construct a fresh ``RandomIcemaker``.
    nxy : int
        XY size in voxels, used to construct a fresh ``RandomIcemaker``.
    nz : int
        Z size in voxels, used to construct a fresh ``RandomIcemaker``.
    ice_cache_dir : str, optional
        Cache directory forwarded to a fresh ``IceBank``.
    icemaker : IceBank or RandomIcemaker, optional
        A pre-built icemaker to reuse as-is. When given, ``ice_model`` and
        ``ice_cache_dir`` are ignored.
    parameterization : str, optional
        Atomic potential parameterization for a freshly-built icemaker's
        kernel: ``'kirkland'``, ``'lobato'``, or ``'shtyrov'``. Default
        ``'shtyrov'``. Ignored when ``icemaker`` is given.
    progressbars : bool, optional
        Forwarded to a freshly-built icemaker's own ``progressbars``.
        Default True. Ignored when ``icemaker`` is given.

    Returns
    -------
    IceBank or RandomIcemaker or None
    """
    if icemaker is not None:
        return icemaker
    if ice_model is None or ice_model == "none":
        return None
    if ice_model == "gd":
        return IceBank(
            cache_dir=ice_cache_dir,
            parameterization=parameterization,
            progressbars=progressbars,
        )
    if ice_model == "random":
        return RandomIcemaker(
            dx=pixel_size,
            n=nxy,
            nz=nz,
            parameterization=parameterization,
            progressbars=progressbars,
        )
    raise ValueError(
        f"Unknown ice_model '{ice_model}'. Choose 'gd', 'random', or 'none'."
    )


def ice_blend_mask(V: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """
    Boolean mask of voxels eligible to receive ice.

    Shared masking rule for every ice-blending entry point in specter (this
    module's :func:`blend_ice_into_volume` and
    :meth:`~specter.imagegenerator._particle_base.ParticleGeneratorBase.solvate`):
    a voxel is eligible when its value is below ``threshold * V.max()``.

    Parameters
    ----------
    V : torch.Tensor
        Scattering-potential volume.
    threshold : float, optional
        Fraction of ``V.max()`` below which a voxel is treated as ice-free.
        Default 0.05.

    Returns
    -------
    torch.Tensor
        Boolean mask, same shape as ``V``.
    """
    return V.detach() < threshold * V.max()


def blend_ice_into_volume(
    V: torch.Tensor,
    icemaker: "IceBank | RandomIcemaker",
    pixel_size: float,
    threshold: float = 0.05,
    relax_steps: int = 0,
    profile: "IceProfile | None" = None,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Add ice into a scattering-potential volume, masked to voxels with little
    existing potential.

    Same masking rule used across specter's ice integration points: a voxel
    is considered ice-free (and thus eligible to receive ice) when its value
    is below ``threshold * V.max()``.

    Parameters
    ----------
    V : torch.Tensor
        Scattering-potential volume, shape (B, Z, Y, X).
    icemaker : IceBank or RandomIcemaker
        Ice source. An :class:`IceBank` draws a tiled crop matching ``V``'s
        own size via :meth:`IceBank.generate_big_ice`; a
        :class:`RandomIcemaker` is called via
        :meth:`RandomIcemaker.generate_ice` (its own fixed ``(n, dx, nz)``
        must already match ``V``).
    pixel_size : float
        Voxel size in Å, forwarded to ``IceBank.generate_big_ice``.
    threshold : float, optional
        Fraction of ``V.max()`` below which a voxel is treated as ice-free.
        Default 0.05.
    relax_steps : int, optional
        Forwarded to :meth:`IceBank.generate_big_ice` (ignored for
        ``RandomIcemaker``, which has no tile seams to relax). Default 0,
        matching every higher-level caller (``ImageGenerator``,
        ``MicrographGenerator``, ``TiltSeriesGenerator``, ...).
    profile : IceProfile, optional
        Laterally varying thickness. Ice is still generated at ``V``'s full
        ``(nz, nxy, nxy)`` extent and then confined to the profile by
        :meth:`~specter.ice.IceProfile.window`, so the caller is responsible
        for having sized ``V`` deep enough (see
        :meth:`~specter.ice.IceProfile.required_nz`). Default None: ice fills
        the box, as it always has.
    inplace : bool, optional
        Write the result into ``V`` instead of a copy, returning ``V`` itself.
        Saves one whole canvas -- 33.6 GB at ``micrograph_size`` -- but leaves
        the caller no pre-ice volume, so only pass this when nothing else holds
        a reference to ``V`` (``MicrographSpecimenGenerator`` keeps one for
        ``save_clean_exitwaves``). Default False.

    Returns
    -------
    torch.Tensor
        ``V`` with ice added at masked voxels; same shape/dtype/device.
        ``V`` itself when ``inplace``, a new tensor otherwise.
    """
    batchsize, nz, nxy, _ = V.shape
    if isinstance(icemaker, IceBank):
        # Built on V's OWN device, not the icemaker's. The ice volume is the
        # same shape as V, so at micrograph scale it is gigabytes -- a
        # 500 x 4096 x 4096 canvas is 33.5 GB -- and generating it on the
        # icemaker's device only to copy it to V's needs both at once. When a
        # caller has deliberately assembled V off the GPU because it does not
        # fit there (`MicrographSpecimenGenerator`'s `move_to_cpu`), building
        # the ice on the GPU anyway defeats that and OOMs.
        ice = icemaker.generate_big_ice(
            n=nxy,
            dx=pixel_size,
            nz=nz,
            batchsize=batchsize,
            relax_steps=relax_steps,
            device=V.device,
        )
    else:
        ice = icemaker.generate_ice(batchsize=batchsize).to(V.device)
    if profile is not None:
        # Chunked in z so the (nz, nxy, nxy) window never exists in full
        # alongside the ice volume it multiplies -- at micrograph_size both are
        # gigabytes. The chunk holds ~64 MB of float32.
        chunk = max(1, 2**24 // (nxy * nxy))
        for start in range(0, nz, chunk):
            sl = slice(start, min(start + chunk, nz))
            ice[:, sl] *= profile.window(
                nz, nxy, pixel_size, z_slice=sl, device=ice.device
            )[None]
    # Blended a z-slab at a time, in place into `ice`, then accumulated into
    # `V`. The obvious spelling, `V + ice * ice_blend_mask(V, threshold).to(
    # V.dtype)`, holds FIVE tensors the size of the whole canvas at once -- V,
    # ice, a float32 mask, the product, and the sum -- which at
    # micrograph_size is 5 x 33.6 GB and is most of why a 4096-pixel micrograph
    # peaked at 237 GB of RSS. Slabbing bounds the mask and the product to one
    # slab each, and the two accumulations are in-place, so the peak is V plus
    # ice plus a slab.
    #
    # `V.max()` is taken once up front: it is a reduction over the whole
    # volume, so computing it per slab would both cost N passes and change the
    # threshold from a global one to a per-slab one.
    cutoff = threshold * V.max()
    out = V if inplace else V.clone()
    chunk = max(1, 2**24 // (nxy * nxy))
    for start in range(0, nz, chunk):
        sl = slice(start, min(start + chunk, nz))
        ice_slab = ice[:, sl]
        ice_slab.mul_(out[:, sl].detach() < cutoff)
        out[:, sl].add_(ice_slab)
    return out
