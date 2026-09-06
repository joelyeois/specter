"""
`_IceTiling`: the half of `IceBank` that serves a request larger than any
cached config, by tiling independently drawn crops, relaxing the seams
between them with a short local MLBOP step, and trimming to the request.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterator

import torch


from ..arrays import (
    soft_voxelize_coordinates_into,
)
from ..cpu_threads import limited_cpu_threads
from ..progress import track
from ._energy import MLBOP
from ..potential import (
    potential_from_deltas,
)

if TYPE_CHECKING:
    pass


#: Molecules splatted per call when a canvas is built a slab at a time (here
#: and in `_blend_ice_slabwise`); bounds the splat's transient index set to
#: ~130 MB.
_SLAB_SPLAT_ATOMS = 2**22


class _IceTiling:
    """
    Tiling, seam relaxation and the big-canvas entry points of `IceBank`.

    A mixin: every attribute it reads is set by `IceBank`, which is the only
    class that uses it. The methods are grouped here because they never touch
    a single crop's extraction, only how crops are assembled.
    """

    _configs: list[dict]
    progressbars: bool
    device: torch.device
    positions: torch.Tensor | None
    current_icedeltas: torch.Tensor | None
    n: int | None
    dx: float | None
    nz: int | None
    box_x: float | None
    box_y: float | None
    box_z: float | None

    def _extract_crop(
        self,
        crop_extent: tuple[float, float, float],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError  # IceBank

    def _get_kernel(self, dx: float) -> torch.Tensor:
        raise NotImplementedError  # IceBank

    def _iter_tiles(
        self,
        tile_grid_shape: tuple[int, int, int],
        tile_extent: float,
        seam_margin: float,
        halo_margin: float,
        generator: torch.Generator | None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Draw the tiles of a grid one at a time.

        Yields
        ------
        positions : torch.Tensor
            One tile's molecules in the assembled frame, ``(N, 3)`` Å, on the
            bank's device.
        mobile, halo : torch.Tensor
            Bool masks over those molecules, see :meth:`_place_tiles`.
        """
        nx, ny, nz_tiles = tile_grid_shape
        half = tile_extent / 2
        tile_indices = [
            (ix, iy, iz)
            for ix in range(nx)
            for iy in range(ny)
            for iz in range(nz_tiles)
        ]
        # A tile is a few hundred small ops (offsets, masks, the crop's own
        # bookkeeping) around one real gather; on a many-core host the small
        # ops dominated at the default thread pool -- see cpu_threads.
        with limited_cpu_threads():
            for ix, iy, iz in track(
                tile_indices,
                description="Placing ice tiles",
                disable=not self.progressbars or len(tile_indices) == 1,
                transient=True,
            ):
                crop = self._extract_crop(
                    (tile_extent, tile_extent, tile_extent), generator=generator
                )
                offset = torch.tensor(
                    [
                        (ix - (nx - 1) / 2) * tile_extent,
                        (iy - (ny - 1) / 2) * tile_extent,
                        (iz - (nz_tiles - 1) / 2) * tile_extent,
                    ],
                    device=crop.device,
                    dtype=crop.dtype,
                )
                dist_to_face = crop.abs()
                mobile = (dist_to_face > (half - seam_margin)).any(dim=1)
                halo = (dist_to_face > (half - seam_margin - halo_margin)).any(dim=1)
                yield crop + offset, mobile, halo

    def _place_tiles(
        self,
        tile_grid_shape: tuple[int, int, int],
        tile_extent: float,
        seam_margin: float,
        generator: torch.Generator | None = None,
        halo_margin: float = 0.0,
        to_host: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[float, float, float]]:
        """
        Draw one independently rotated/translated crop of size
        ``tile_extent`` per grid cell and place them side by side.

        Parameters
        ----------
        to_host : bool, optional
            Move each tile's atoms to the host as they are drawn, so the
            device never holds the whole assembled set (3 GB at a
            4096-pixel micrograph, plus as much again for the final
            concatenation). Default False.
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
        parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for pos, mobile, halo in self._iter_tiles(
            tile_grid_shape, tile_extent, seam_margin, halo_margin, generator
        ):
            if to_host:
                pos, mobile, halo = pos.cpu(), mobile.cpu(), halo.cpu()
            parts.append((pos, mobile, halo))
        positions = torch.cat([p for p, _, _ in parts], dim=0)
        mobile_mask = torch.cat([m for _, m, _ in parts], dim=0)
        halo_mask = torch.cat([h for _, _, h in parts], dim=0)
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
        every other atom fixed. Naive tiling alone leaves E/atom flipped
        from favorable to unfavorable at the seams; this recovers most of it
        within a few seconds.

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

    def _tile_grid(
        self, n: int, dx: float, nz: int | None, tile_extent: float | None
    ) -> tuple[float, tuple[int, int, int], tuple[float, float, float]]:
        """
        Set the requested box on ``self`` and size the tile grid covering it.

        Parameters
        ----------
        n, dx, nz
            As for :meth:`generate_big_ice_deltas`.
        tile_extent : float or None
            Tile side in Å; None means the smallest cached config's box.

        Returns
        -------
        tile_extent : float
        grid : tuple of int
            Tiles along x, y and z.
        box : tuple of float
            The requested box in Å, ``(x, y, z)``.
        """
        nz = nz if nz is not None else n
        self.n, self.dx, self.nz = n, dx, nz
        box = (n * dx, n * dx, nz * dx)
        self.box_x, self.box_y, self.box_z = box
        if tile_extent is None:
            tile_extent = min(c["box_L"] for c in self._configs)
        grid = (
            max(1, math.ceil(box[0] / tile_extent)),
            max(1, math.ceil(box[1] / tile_extent)),
            max(1, math.ceil(box[2] / tile_extent)),
        )
        return tile_extent, grid, box

    def big_ice_positions(
        self,
        n: int,
        dx: float,
        nz: int | None = None,
        tile_extent: float | None = None,
        seam_margin: float = 6.0,
        relax_steps: int = 0,
        relax_lr: float = 0.01,
        mlbop_target: float | None = -0.413,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        to_host: bool = False,
    ) -> torch.Tensor:
        """
        The molecules of a tiled canvas, without voxelising them.

        The same draws, tiling, seam relaxation and trimming as
        :meth:`generate_big_ice_deltas`, returning the coordinates instead
        of a canvas. What :func:`blend_ice_into_volume` builds its ice from
        a z-slab at a time when the volume lives on the host: the positions
        of a micrograph-sized canvas are under a gigabyte, the canvas itself
        tens.

        Parameters
        ----------
        n, dx, nz, tile_extent, seam_margin, relax_steps, relax_lr,
        mlbop_target, device, generator
            As for :meth:`generate_big_ice_deltas`.

        to_host : bool, optional
            Collect the tiles on the host as they are drawn and return the
            result there, so the device never holds the whole set; see
            :meth:`_place_tiles`. Default False.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 3)``, x/y/z in Å centred on the canvas, on `device`
            (the bank's own device by default), or on the host with
            `to_host`.
        """
        if device is None:
            device = self.device
        tile_extent, grid, box = self._tile_grid(n, dx, nz, tile_extent)
        halo_margin = 2.0 * MLBOP().r_cut if relax_steps > 0 else 0.0
        positions, mobile_mask, halo_mask, assembled_box = self._place_tiles(
            grid,
            tile_extent,
            seam_margin,
            generator=generator,
            halo_margin=halo_margin,
            to_host=to_host,
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
        # Trim to the requested box BEFORE anything splats these with
        # periodic=True: that flag is an unconditional `index % grid_shape`,
        # and a tile's footprint routinely overhangs the requested box (every
        # edge tile does, and a single tile larger than the request overhangs
        # on all sides), so without the trim far-outside molecules would wrap
        # back in and inflate the density by up to (tile / box)^3 -- measured
        # 5.6x for a 256 A tile filling a 160 A request. What the wrap is for
        # is the sub-voxel fencepost overflow at the true faces, which the
        # trim leaves in place.
        keep = (
            (positions[:, 0].abs() <= box[0] / 2)
            & (positions[:, 1].abs() <= box[1] / 2)
            & (positions[:, 2].abs() <= box[2] / 2)
        )
        return positions[keep] if to_host else positions[keep].to(device)

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
            treated as a seam candidate for relaxation. Default 6.0
            (validated at 256^3 and 128^3).
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
        tile_extent, _grid, _box = self._tile_grid(n, dx, nz, tile_extent)
        nz = nz if nz is not None else n

        # The whole batch is allocated up front and splatted into item by
        # item: a per-item list stacked at the end would hold the batch twice
        # at the moment of the stack. The molecules of one item are drawn in
        # full first (a tenth of the canvas's bytes) and splatted in chunks,
        # the same two steps the host-volume blend takes.
        batch_vox = torch.zeros(batchsize, nz, n, n, device=device)
        for b in track(
            range(batchsize),
            description="Generating tiled ice volumes",
            disable=not self.progressbars or batchsize == 1,
            transient=True,
        ):
            positions = self.big_ice_positions(
                n,
                dx,
                nz,
                tile_extent=tile_extent,
                seam_margin=seam_margin,
                relax_steps=relax_steps,
                relax_lr=relax_lr,
                mlbop_target=mlbop_target,
                device=device,
                generator=generator,
            )
            self.positions = positions
            # periodic=True: see generate_ice_deltas -- the outer-boundary
            # fencepost fix, applied to the keep-trimmed, already-relaxed
            # molecule set, so only genuine sub-voxel overflow wraps.
            for a0 in range(0, len(positions), _SLAB_SPLAT_ATOMS):
                soft_voxelize_coordinates_into(
                    batch_vox[b],
                    positions[a0 : a0 + _SLAB_SPLAT_ATOMS],
                    dx,
                    periodic=True,
                )
        self.current_icedeltas = batch_vox
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
        # "auto": the water kernel is 8^3 at 0.73 A/voxel, where cuDNN's direct
        # conv3d on a 512^3 box is 7x slower than the FFT (215 vs 32 ms); the
        # 3^3 kernel of a >= 2 A tomogram voxel still goes direct, as does any
        # volume too large for the FFT's complex working copies.
        # `out=deltas`: the deltas are consumed here and nothing reads them
        # afterwards, so the potential overwrites them instead of taking a
        # second canvas.
        # `boundary="periodic"`: the canvas is a piece of bulk ice, not a
        # molecule in a box, so a face molecule's kernel wraps rather than
        # being cut off -- see potential_from_deltas for what the cut cost.
        potential = potential_from_deltas(
            deltas, kernel.to(device), backend="auto", out=deltas, boundary="periodic"
        )
        del deltas
        return potential
