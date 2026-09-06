"""
Add ice into a scattering-potential volume one z-slab at a time.

The one rule every ice integration point shares: a voxel takes ice in
proportion to the fraction of it not already occupied, read off the
potential by :func:`~specter.potential.potential_occupancy`. Doing that a
slab at a time bounds memory, and it has one subtlety, which is why the
bookkeeping lives here rather than at each of its call sites
(``ParticleGeneratorBase.solvate``, :func:`~specter.ice.blend_ice_into_volume`
and its host-volume branch).

The occupancy blur reads a halo of slices beyond its slab. On the low side
that halo overlaps the previous slab, which has already had its ice added:
read directly, it would show inflated potential, infer a fuller voxel, and
starve every slab boundary of ice. So the weighted ice added to the last
``halo`` slices is kept and subtracted back, which recovers the pristine
potential for exactly the slab-sized copy the blur reads. Without the halo
at all, every slab boundary would be an edge the blur sees, printed into
the ice as a seam.
"""

from __future__ import annotations

import torch

from ..potential import (
    FULL_OCCUPANCY_POTENTIAL_V,
    WATER_COARSE_GRAIN_SIGMA_ANGSTROM,
    occupancy_blur_halo_voxels,
    potential_occupancy,
)
import math
import numpy as np
import torch.nn.functional as F
from typing import TYPE_CHECKING
from ..arrays import soft_voxelize_coordinates_into
from ..fft import spatial_convolve3d_same
from ._bank import IceBank
from ._tiling import _SLAB_SPLAT_ATOMS
from ._random import RandomIcemaker
from specter.options import IceModel, ScatteringFactors

if TYPE_CHECKING:
    from ._profile import IceProfile


class IceSlabBlender:
    """
    Blend unweighted ice slabs into a volume, in ascending z order.

    Parameters
    ----------
    voxel_size : float
        Voxel size in Å.
    full_potential : float or torch.Tensor, optional
        Potential of a fully occupied voxel, forwarded to
        :func:`~specter.potential.potential_occupancy`. A tensor
        broadcastable to ``(B, 1, 1, 1)`` carries a per-image scale.
        Default :data:`~specter.potential.FULL_OCCUPANCY_POTENTIAL_V`.
    sigma_angstrom : float, optional
        Coarse-graining length in Å of the occupancy blur. Default
        :data:`~specter.potential.WATER_COARSE_GRAIN_SIGMA_ANGSTROM`.

    Notes
    -----
    One instance per volume: it holds the tail of ice added to the last
    slab, so :meth:`add` must be called for consecutive slabs from ``z = 0``
    upwards, each ending where the next begins.
    """

    def __init__(
        self,
        voxel_size: float,
        full_potential: float | torch.Tensor = FULL_OCCUPANCY_POTENTIAL_V,
        sigma_angstrom: float = WATER_COARSE_GRAIN_SIGMA_ANGSTROM,
    ) -> None:
        self.voxel_size = voxel_size
        self.full_potential = full_potential
        self.sigma_angstrom = sigma_angstrom
        self.halo = occupancy_blur_halo_voxels(voxel_size, sigma_angstrom)
        self._tail: torch.Tensor | None = None

    def add(self, V: torch.Tensor, ice: torch.Tensor, start: int, end: int) -> None:
        """
        Weight `ice` by the free fraction of ``V[:, start:end]`` and add it.

        Parameters
        ----------
        V : torch.Tensor
            The volume, ``(B, Z, Y, X)``, on any device; slices before
            `start` have already received their ice. Modified in place.
        ice : torch.Tensor
            Unweighted ice for the slab, ``(B, end - start, Y, X)``, on the
            device the blend should run on. Weighted in place.
        start, end : int
            The slab's z range in `V`.
        """
        nz = V.shape[1]
        lo, hi = max(0, start - self.halo), min(nz, end + self.halo)
        # A copy on the compute device: the blur must read the pristine
        # potential, and the subtraction below must not touch `V`.
        src = V[:, lo:hi].to(ice.device, copy=True)
        if lo < start:
            assert self._tail is not None
            src[:, : start - lo] -= self._tail[:, -(start - lo) :]
        core = slice(start - lo, start - lo + (end - start))
        occ = potential_occupancy(
            src,
            self.voxel_size,
            sigma_angstrom=self.sigma_angstrom,
            full_potential=self.full_potential,
        )[:, core]
        # In place, and reusing `occ`: `(1 - occ).clamp(0, 1)` would
        # allocate two more slabs to produce a value consumed once.
        ice.mul_(occ.neg_().add_(1.0).clamp_(0.0, 1.0))
        if V.device == ice.device:
            V[:, start:end].add_(ice)
        else:
            V[:, start:end] = (src[:, core] + ice).to(V.device)
        del src, occ
        if self.halo > 0:
            # Running tail of the last `halo` weighted slices, across as many
            # previous slabs as the halo spans: a slab can be narrower than
            # the halo on a very large canvas.
            tail = ice if self._tail is None else torch.cat([self._tail, ice], dim=1)
            self._tail = tail[:, -self.halo :].clone()
            del tail


def resolve_icemaker(
    ice_model: IceModel | None,
    voxel_size: float,
    nxy: int,
    nz: int,
    ice_cache_dir: str | None = None,
    icemaker: "IceBank | RandomIcemaker | None" = None,
    parameterization: ScatteringFactors = "kirkland",
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
    voxel_size : float
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
        ``'kirkland'``. Ignored when ``icemaker`` is given.
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
            dx=voxel_size,
            n=nxy,
            nz=nz,
            parameterization=parameterization,
            progressbars=progressbars,
        )
    raise ValueError(
        f"Unknown ice_model '{ice_model}'. Choose 'gd', 'random', or 'none'."
    )


def blend_ice_into_volume(
    V: torch.Tensor,
    icemaker: "IceBank | RandomIcemaker",
    voxel_size: float,
    full_potential: float = FULL_OCCUPANCY_POTENTIAL_V,
    relax_steps: int = 0,
    profile: "IceProfile | None" = None,
    inplace: bool = False,
    sigma_angstrom: float = WATER_COARSE_GRAIN_SIGMA_ANGSTROM,
) -> torch.Tensor:
    """
    Add ice into a scattering-potential volume, weighted by how much room
    is left in each voxel.

    Same rule used across specter's ice integration points: a voxel takes
    ice in proportion to the fraction of it not already occupied, per
    :func:`~specter.potential.potential_occupancy`.

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
    voxel_size : float
        Voxel size in Å, forwarded to ``IceBank.generate_big_ice``.
    full_potential : float, optional
        Potential of a fully-occupied voxel, V, forwarded to
        :func:`~specter.potential.potential_occupancy`. Default
        :data:`~specter.potential.FULL_OCCUPANCY_POTENTIAL_V`.
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
    sigma_angstrom : float, optional
        Coarse-graining length in Angstrom for that fallback, forwarded to
        :func:`~specter.potential.potential_occupancy`. Ignored when
        ``occupancy`` is given. Default
        :data:`~specter.potential.WATER_COARSE_GRAIN_SIGMA_ANGSTROM`.

    Returns
    -------
    torch.Tensor
        ``V`` with ice added at masked voxels; same shape/dtype/device.
        ``V`` itself when ``inplace``, a new tensor otherwise.
    """
    batchsize, nz, nxy, _ = V.shape
    if (
        isinstance(icemaker, IceBank)
        and V.device.type == "cpu"
        and icemaker.device.type != "cpu"
    ):
        # The volume is on the host because it does not fit the device (a
        # micrograph canvas), and the bank has a device. Building the ice
        # where V lives would put tile placement, the splat, the kernel
        # convolution and the occupancy blur all on the CPU over a canvas
        # of tens of GB: 11 of the 13 minutes of a 2048-pixel micrograph,
        # and a second host canvas for the ice. Instead the molecules are
        # drawn once and every voxel step runs on the device a z-slab at a
        # time, adding each finished slab into V. Same result to float
        # noise; peak device memory is a few slabs, host memory is V.
        out = V if inplace else V.clone()
        for b in range(batchsize):
            _blend_ice_slabwise(
                out[b : b + 1],
                icemaker,
                voxel_size,
                full_potential=full_potential,
                relax_steps=relax_steps,
                profile=profile,
                sigma_angstrom=sigma_angstrom,
            )
        return out
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
            dx=voxel_size,
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
                nz, nxy, voxel_size, z_slice=sl, device=ice.device
            )[None]
    # Blended a z-slab at a time, in place into `ice`, then accumulated into
    # `V`. The obvious spelling, `V + ice * (1 - potential_occupancy(V))`, holds
    # FIVE tensors the size of the whole canvas at once -- V, ice, the
    # weight, the product, and the sum -- which at
    # micrograph_size is 5 x 33.6 GB and is most of why a 4096-pixel micrograph
    # peaked at 237 GB of RSS. Slabbing bounds the mask and the product to one
    # slab each, and the two accumulations are in-place, so the peak is V plus
    # ice plus a slab.
    #
    # No global reduction: the occupancy weight is per-voxel and absolute,
    # so a slab needs nothing but itself. (A rule relative to `V.max()`
    # would make the whole volume's contents an input to every voxel's ice.)
    out = V if inplace else V.clone()
    chunk = max(1, 2**24 // (nxy * nxy))
    blender = IceSlabBlender(
        voxel_size, full_potential=full_potential, sigma_angstrom=sigma_angstrom
    )
    for start in range(0, nz, chunk):
        end = min(start + chunk, nz)
        blender.add(out, ice[:, start:end], start, end)
    return out


def _blend_ice_slabwise(
    V: torch.Tensor,
    bank: IceBank,
    voxel_size: float,
    full_potential: float = FULL_OCCUPANCY_POTENTIAL_V,
    relax_steps: int = 0,
    profile: "IceProfile | None" = None,
    sigma_angstrom: float = WATER_COARSE_GRAIN_SIGMA_ANGSTROM,
    slab_voxels: int = 2**27,
) -> torch.Tensor:
    """
    :func:`blend_ice_into_volume` for one host-resident volume, slab by slab
    on the bank's device.

    Each slab's molecules (those within a halo of it, with the canvas ends
    wrapped so the convolution is periodic in z as in
    :meth:`IceBank.generate_big_ice`) are splatted onto a slab canvas,
    convolved with the water kernel (circular in xy, real context in z),
    weighted by the occupancy of the pristine potential and added into V.
    The halo covers both the kernel's reach and the occupancy blur's; the
    ice already added to the previous slab is subtracted back before the
    blur reads it, as the whole-canvas path does.

    Parameters
    ----------
    V : torch.Tensor
        Shape ``(1, nz, n, n)``, on the host; modified in place.
    bank : IceBank
        On the compute device.
    voxel_size : float
        Voxel size in Å.
    full_potential, relax_steps, profile, sigma_angstrom
        As for :func:`blend_ice_into_volume`.
    slab_voxels : int, optional
        Voxels per slab core; the working canvases are a few times this.
        Default ``2**27`` (0.5 GB of float32).

    Returns
    -------
    torch.Tensor
        `V`.
    """
    assert V.shape[0] == 1
    _, nz, n, _ = V.shape
    device = bank.device
    dx = voxel_size
    kernel = bank._get_kernel(dx).to(device)
    kr = kernel.shape[0] // 2
    blender = IceSlabBlender(
        dx, full_potential=full_potential, sigma_angstrom=sigma_angstrom
    )
    halo = max(kr, blender.halo) + 1
    chunk = max(1, min(slab_voxels // (n * n), nz))

    # Every molecule of the canvas, bucketed by slab and parked on the host:
    # a micrograph canvas holds ~3e8 of them (3 GB of float32), and a slab
    # needs only its own few buckets, so the device holds slab canvases and
    # nothing else. Bucketing is a radix sort on small integer keys (3 s
    # for 3e8 on the host, against 37 s for a full sort of the z values),
    # and the exact z window is applied on the device per slab.
    pos = bank.big_ice_positions(
        n, dx, nz, relax_steps=relax_steps, device=device, to_host=True
    )
    zv = pos[:, 2] / dx + nz // 2  # voxel z index, as the splat computes it
    n_buckets = -(-nz // chunk)
    key = (zv / chunk).floor().clamp_(0, n_buckets - 1).to(torch.int16).numpy()
    order = torch.from_numpy(np.argsort(key, kind="stable"))
    pos = pos.index_select(0, order)
    del zv, order
    offsets = np.concatenate([[0], np.cumsum(np.bincount(key, minlength=n_buckets))])
    del key

    def slab_atoms(lo: int, hi: int) -> torch.Tensor:
        """Molecules with voxel z in [lo - 1, hi + 1), the canvas ends
        wrapped (the whole-canvas convolution is periodic in z), on `device`
        in slab-local coordinates: the splat maps coord/dx + depth//2 to the
        index, which must come out as zv - lo."""
        depth = hi - lo
        parts = []
        for shift in (0, nz, -nz):
            # Molecules with z + shift in [lo - 1, hi + 1) have z in
            # [lo - 1 - shift, hi + 1 - shift); the buckets covering that.
            b0 = max(0, math.floor((lo - 1 - shift) / chunk))
            b1 = min(n_buckets, math.ceil((hi + 1 - shift) / chunk))
            if b1 <= b0:
                continue
            xyz = pos[offsets[b0] : offsets[b1]].to(device)
            z = xyz[:, 2] / dx + nz // 2 + shift
            m = (z >= lo - 1) & (z < hi + 1)
            if not bool(m.any()):
                continue
            z_local = (z[m] - lo - depth // 2) * dx
            parts.append(torch.cat([xyz[m, :2], z_local[:, None]], dim=1))
        return torch.cat(parts, dim=0)

    for z0 in range(0, nz, chunk):
        z1 = min(z0 + chunk, nz)
        lo, hi = z0 - halo, z1 + halo
        depth = hi - lo
        coords = slab_atoms(lo, hi)
        deltas = torch.zeros(depth, n, n, device=device)
        # In atom chunks: the splat's per-corner index set is ~30 bytes per
        # molecule, and a 4096-pixel slab holds tens of millions of them.
        for a0 in range(0, len(coords), _SLAB_SPLAT_ATOMS):
            soft_voxelize_coordinates_into(
                deltas, coords[a0 : a0 + _SLAB_SPLAT_ATOMS], dx, periodic=True
            )
        del coords
        padded = F.pad(deltas[None, None], (kr, kr, kr, kr, 0, 0), mode="circular")[
            :, 0
        ]
        del deltas
        ice = spatial_convolve3d_same(padded, kernel)[
            :, halo : halo + (z1 - z0), kr:-kr, kr:-kr
        ]
        del padded
        if profile is not None:
            ice *= profile.window(nz, n, dx, z_slice=slice(z0, z1), device=device)[None]

        blender.add(V, ice, z0, z1)
        del ice
    return V
