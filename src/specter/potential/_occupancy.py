"""
Occupancy: the fraction of each voxel that a specimen already fills.

Water cannot occupy space something else is in, so blending amorphous ice
into a specimen needs to know how much of each voxel is still free.

It is read off the potential, but only after coarse-graining to the water
probe's own length scale. Unblurred it does not work at all at the voxel
sizes specter runs at, because a potential is cusped: it
peaks at nuclei and dips between them, *including between two bonded atoms
1.5 A apart*, where no water fits. Clamping ``1 - V/V_full`` therefore
finds the boundary of every ATOM rather than the boundary of the MOLECULE,
and admits bulk ice into every interatomic gap. Measured on 1A6M at
1 A/voxel, voxels solidly inside the molecule received 0.597 of full-
strength ice; the occupancy field here gives 0.006. How wrong it goes also
depends on the render grid, since finer voxels make sharper cusps: the
excluded volume that rule recovers runs from 0.36 of the molecule's own
volume at 0.75 A/voxel to 0.87 at 4 A.

A geometric field built from van der Waals radii is sharper -- 0.006 of
full ice inside a molecule against the blur's 0.211 -- but it changes
nothing any specter output can see, because a Gaussian blur CONSERVES the integral of
V and therefore the total displaced water. It only moves where the
displacement sits, taking too little from the interior and putting the
same amount into a halo outside. Every shipped artifact either integrates
along a ray (single-particle images, tilt series -- and the blur is
isotropic, so the conservation holds at any tilt) or carries no ice at all
(``specter build tomogram``'s volume and labels). Measured on a ribosome
column: 2170.8 V*A of ice under the blur, 2171.3 under geometry, 0.02%
apart. It becomes worth having the moment a ground-truth volume ships WITH
ice in it, and not before.
"""

from __future__ import annotations

import torch

__all__ = [
    "FULL_OCCUPANCY_POTENTIAL_V",
    "WATER_COARSE_GRAIN_SIGMA_A",
    "occupancy_blur_halo_voxels",
    "potential_occupancy",
]

#: Scattering potential of a voxel entirely filled with biological
#: material, V. Protein's mean inner potential. The reference for how much
#: of a voxel is already occupied, so it has to be an ABSOLUTE quantity:
#: the whole point is that it does not depend on what else is in the
#: volume.
#:
#: Measured as integral(V dV) / (molecular volume) over four structures
#: spanning 19 kDa to 3 MDa -- 6.81 V for 1A6M, 7.00 for 1FA2, 7.07 for
#: 7VD8, 6.84 for 6QZP -- so 7.0 is protein generally, not one structure.
#:
#: Its weakest input is the molecular volume, taken as mass x 1.2122 A^3/Da
#: from the standard protein partial specific volume vbar = 0.73 cm3/g
#: (density 1.37 g/cm3). The constant scales inversely with that: vbar 0.70
#: gives 7.30 V, 0.76 gives 6.73. Two caveats on it, neither resolved:
#:
#:   - vbar is THERMODYNAMIC, the volume a solution gains per gram of
#:     protein, which folds in effects on surrounding water. What this
#:     model wants is geometric -- space unavailable to water. They are
#:     close but not the same quantity, and it is not obvious which way a
#:     better answer moves: van der Waals volume alone is smaller, since
#:     proteins pack to roughly 75%, and would raise the constant.
#:   - There is no external anchor. The ice side has one (CLAUDE.md cites
#:     Yesibolati et al. 2020 for liquid water at 4.48 +/- 0.19 V); the
#:     protein side does not. Published holography values sit around 7-8 V,
#:     which is why 7.0 is comfortable, but that range is not cited here
#:     from a checked source. Closing that gap would firm this up.
#:
#: The error it can cause is bounded and small: the constant only infers a
#: volume FRACTION, so 4% off means a voxel read as 50% full is really 48%.
#: Against the rule this replaced, where one gold bead moved 13.87% of all
#: voxels between full ice and none, that is a good trade.
FULL_OCCUPANCY_POTENTIAL_V = 7.0

#: Coarse-graining length, Angstrom, for reading occupancy off a
#: potential. Specified in ANGSTROM rather than voxels, which is the whole
#: reason the result does not depend on the render grid.
#:
#: Water exclusion is not a question about where a nucleus sits, it is
#: "does a 2.8 A water molecule fit here", so the potential has to be
#: asked at the water's own length scale. Without it, clamping ``1 - V/V_full``
#: resolves the cusp at every atom and lets bulk ice into the gaps between
#: BONDED atoms 1.5 A apart. Measured recovery of a protein's
#: solvent-excluded volume, over voxel sizes from 1 to 12 A:
#:
#:     raw     0.385  0.582  0.874  0.956  0.969  0.974  0.973   (1FA2)
#:     blurred 0.964  0.965  0.956  0.960  0.969  0.974  0.973
#:
#: 146% spread against 1.9%. Note the two converge by ~8 A and are
#: identical beyond it: a coarse voxel's own average has already removed
#: the cusps, so this is not an extra approximation, it is what gives a
#: FINE grid the coarse-graining a coarse one gets for free.
WATER_COARSE_GRAIN_SIGMA_A = 2.0


def occupancy_blur_halo_voxels(
    pixel_size: float, sigma_a: float = WATER_COARSE_GRAIN_SIGMA_A
) -> int:
    """
    Voxels of context :func:`potential_occupancy` reads beyond its input.

    A caller that evaluates the field a z-slab at a time (to bound memory
    on a volume too large to blur whole) must extend each slab by this
    much and discard the margin, or every slab boundary becomes an edge
    the blur sees.

    Parameters
    ----------
    pixel_size : float
        Voxel size in Angstrom.
    sigma_a : float, optional
        Coarse-graining length in Angstrom. Default
        :data:`WATER_COARSE_GRAIN_SIGMA_A`.

    Returns
    -------
    int
        Halo width in voxels. Zero when the blur is skipped as sub-voxel.
    """
    sigma_vox = sigma_a / pixel_size
    if sigma_vox < 0.25:
        return 0
    return max(1, int(round(3 * sigma_vox)))


def _gaussian_blur3d(V: torch.Tensor, sigma_vox: float) -> torch.Tensor:
    """
    Separable Gaussian blur over the last three axes, edges replicated.

    Each pass is a ``conv1d`` along the tensor's LAST (contiguous) axis, the
    other two axes being brought there by a transpose and a copy. A cuDNN
    ``conv3d`` with a ``(1, 1, 2r+1)``-shaped kernel computes the same thing
    but runs ~2x slower on the slabs `blend_ice_into_volume` and
    `ParticleGeneratorBase.solvate` hand it (0.36 vs 0.64 ms per 1024^2
    slice on an L40), and the blur is the largest single cost of solvation
    at a 512-pixel box. The two agree to float rounding (~2e-6 on a 7 V
    field), since only the summation order differs.
    """
    r = max(1, int(round(3 * sigma_vox)))
    x = torch.arange(-r, r + 1, device=V.device, dtype=V.dtype)
    kernel = torch.exp(-0.5 * (x / sigma_vox) ** 2)
    weight = (kernel / kernel.sum()).view(1, 1, -1)

    lead = V.shape[:-3]
    Z, Y, X = V.shape[-3:]
    nd = V.ndim
    pad = torch.nn.functional.pad
    conv1d = torch.nn.functional.conv1d

    # x axis, already contiguous
    t = conv1d(pad(V.reshape(-1, 1, X), (r, r), mode="replicate"), weight)
    out = t.reshape(*lead, Z, Y, X)
    # y axis
    t = out.transpose(-1, -2).contiguous().reshape(-1, 1, Y)
    t = conv1d(pad(t, (r, r), mode="replicate"), weight)
    out = t.reshape(*lead, Z, X, Y).transpose(-1, -2)
    # z axis
    t = out.permute(*range(nd - 3), nd - 2, nd - 1, nd - 3).contiguous()
    t = conv1d(pad(t.reshape(-1, 1, Z), (r, r), mode="replicate"), weight)
    out = t.reshape(*lead, Y, X, Z).permute(*range(nd - 3), nd - 1, nd - 3, nd - 2)
    return out.contiguous()


def potential_occupancy(
    V: torch.Tensor,
    pixel_size: float,
    sigma_a: float = WATER_COARSE_GRAIN_SIGMA_A,
    full_potential: float | torch.Tensor = FULL_OCCUPANCY_POTENTIAL_V,
) -> torch.Tensor:
    """
    Fraction of each voxel already filled, read off the potential.

    Coarse-grains `V` to the water probe's own length scale, then reads
    the volume fraction against `full_potential`. The one estimator, used
    for every specimen: a rendered structure, a bulk material, crowding
    duplicates, or a map supplied with no provenance at all.

    Parameters
    ----------
    V : torch.Tensor
        Scattering potential in volts, shape ``(..., Z, Y, X)``.
    pixel_size : float
        Voxel size in Angstrom. Required, and not merely for units:
        `sigma_a` is physical, so this is what makes the result independent
        of the render grid.
    sigma_a : float, optional
        Coarse-graining length in Angstrom. Default
        :data:`WATER_COARSE_GRAIN_SIGMA_A`.
    full_potential : float or torch.Tensor, optional
        Potential of a fully-occupied voxel, V. Default
        :data:`FULL_OCCUPANCY_POTENTIAL_V`, protein's mean inner potential.
        A tensor broadcastable to `V` is accepted so a caller holding a
        per-image scale can fold it in here. It must be folded in BEFORE
        the clamp, which is why this is a parameter rather than something
        to divide out of the result: dividing afterwards would clamp
        against the wrong reference and cap occupancy far below 1.

    Returns
    -------
    torch.Tensor
        Occupancy in [0, 1], same shape as `V`.

    Notes
    -----
    Skips the convolution when ``sigma_a`` is under a quarter of a voxel.
    That is a cost guard, not a correctness one -- the kernel degenerates
    to the identity there anyway (side weights ``exp(-12.5) ~ 4e-6`` at
    10 A voxels) -- and it is correct to skip: a voxel average over 10 A
    has already coarse-grained past 2 A, which is why raw and blurred
    agree exactly beyond ~8 A.

    Reading occupancy off a potential cannot distinguish MATERIALS, since
    the volume it sees is a single sum and `full_potential` is protein's.
    Gold and carbon sit far above it and correctly exclude water outright;
    a bilayer's acyl core at 5.4 V reads as 23% empty and keeps that much
    of its water. That is the one error a geometric field would fix, and
    this module's docstring records why one was tried and removed anyway.
    """
    if pixel_size <= 0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size}")
    if sigma_a < 0:
        raise ValueError(f"sigma_a must be non-negative, got {sigma_a}")
    if isinstance(full_potential, torch.Tensor):
        if bool((full_potential <= 0).any()):
            raise ValueError("full_potential must be positive")
    elif full_potential <= 0:
        raise ValueError(f"full_potential must be positive, got {full_potential}")

    field = V.detach()
    sigma_vox = sigma_a / pixel_size
    if sigma_vox >= 0.25:
        field = _gaussian_blur3d(field, sigma_vox)
    return (field / full_potential).clamp_(0.0, 1.0)
