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


class IceSlabBlender:
    """
    Blend unweighted ice slabs into a volume, in ascending z order.

    Parameters
    ----------
    pixel_size : float
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
        pixel_size: float,
        full_potential: float | torch.Tensor = FULL_OCCUPANCY_POTENTIAL_V,
        sigma_angstrom: float = WATER_COARSE_GRAIN_SIGMA_ANGSTROM,
    ) -> None:
        self.pixel_size = pixel_size
        self.full_potential = full_potential
        self.sigma_angstrom = sigma_angstrom
        self.halo = occupancy_blur_halo_voxels(pixel_size, sigma_angstrom)
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
            self.pixel_size,
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
