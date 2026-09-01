"""
Occupancy: the fraction of each voxel that a specimen already fills.

Water cannot occupy space something else is in, so blending amorphous ice
into a specimen needs to know how much of each voxel is still free. That
is a *geometric* quantity, and this module computes it from atoms.

It exists because the alternative -- reading occupancy back out of the
scattering potential, as :func:`potential_occupancy` does -- cannot be
as sharp, and unblurred it does not work at all at the voxel sizes
specter runs at. A potential is cusped: it
peaks at nuclei and dips between them, *including between two bonded atoms
1.5 A apart*, where no water fits. Clamping ``1 - V/V_full`` therefore
finds the boundary of every ATOM rather than the boundary of the MOLECULE,
and admits bulk ice into every interatomic gap. Measured on 1A6M at
1 A/voxel, voxels solidly inside the molecule received 0.597 of full-
strength ice; the occupancy field here gives 0.006. How wrong it goes also
depends on the render grid, since finer voxels make sharper cusps: the
excluded volume that rule recovers runs from 0.36 of the molecule's own
volume at 0.75 A/voxel to 0.87 at 4 A.

Occupancy has neither problem. It does not read the potential, so it is
also unaffected by anything scaling it -- ``potential_scale``, the
parameterization, a B-factor -- none of which change how much room a
molecule takes up.
"""

from __future__ import annotations

import torch

__all__ = [
    "VDW_RADII_A",
    "DEFAULT_VDW_RADIUS_A",
    "SOLVENT_EXCLUDED_RADIUS_SCALE",
    "FULL_OCCUPANCY_POTENTIAL_V",
    "WATER_COARSE_GRAIN_SIGMA_A",
    "atomic_occupancy",
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

#: Van der Waals radii in Angstrom, by atomic number (Bondi 1964, with the
#: later 1.20 A for hydrogen). Only the elements a biomolecular structure
#: normally carries are listed; anything else falls back to
#: :data:`DEFAULT_VDW_RADIUS_A`.
VDW_RADII_A: dict[int, float] = {
    1: 1.20,  # H
    6: 1.70,  # C
    7: 1.55,  # N
    8: 1.52,  # O
    9: 1.47,  # F
    11: 2.27,  # Na
    12: 1.73,  # Mg
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    19: 2.75,  # K
    20: 2.31,  # Ca
    25: 2.05,  # Mn
    26: 2.04,  # Fe
    27: 2.00,  # Co
    28: 1.97,  # Ni
    29: 1.96,  # Cu
    30: 2.01,  # Zn
    34: 1.90,  # Se
    35: 1.85,  # Br
    53: 1.98,  # I
}

#: Radius used for an element not in :data:`VDW_RADII_A`. Carbon's, since
#: an unlisted element in a biomolecular structure is more often a light
#: heteroatom than a metal.
DEFAULT_VDW_RADIUS_A = 1.70

#: Multiplier on every van der Waals radius, turning the union of bare vdW
#: balls into the *solvent-excluded* volume.
#:
#: Water is excluded from more than the atoms themselves: it is also kept
#: out of interstitial crevices too small to hold a molecule. The bare
#: union recovers only 0.66-0.69 of a protein's solvent-excluded volume,
#: so radii are inflated to make up the rest.
#:
#: The target is ``mass x 1.2122 A^3/Da``, deliberately the same quantity
#: :data:`~specter.ice.FULL_OCCUPANCY_POTENTIAL_V` is calibrated against,
#: so the two agree on how much volume a protein occupies and the average
#: protein-to-ice contrast is unchanged by adopting this field. What
#: changes is *where* the exclusion sits.
#:
#: MEASURED, not derived. The factor cannot be reasoned out from the bare
#: union's shortfall: a union of overlapping balls does not grow as
#: ``r**3``, because the shells added by inflation overlap each other, so
#: the ``(1/0.685)**(1/3) = 1.136`` that argument gives lands at 0.85 of
#: target rather than 1.0. Measured across three structures spanning 19
#: kDa to 3 MDa and voxel sizes from 0.75 to 2.0 A, this value gives:
#:
#:     1A6M  0.997 - 1.061      1FA2  0.979 - 1.036      7VD8  0.923 - 0.997
#:
#: Two residual errors, both roughly +/-7% and neither worth chasing
#: further. Across structures, larger assemblies read slightly low.
#: Across voxel size, coarser grids read low because the partial-volume
#: ramp spans a whole voxel and erodes the concave seams between
#: overlapping spheres, which inflation creates more of. Both sit inside
#: the calibration target's own uncertainty: that target rests on the
#: standard partial specific volume ``vbar = 0.73 cm3/g``, and 0.70 to
#: 0.76 moves it by about the same +/-8%.
#:
#: For scale, the potential-derived rule this replaces recovered 0.36 to
#: 0.87 of the same quantity over the same voxel sizes -- a 140% spread
#: against 7%.
#:
#: Inflating radii rather than applying a morphological closing is
#: deliberate. A closing is the more faithful description, filling
#: crevices without moving the outer surface, but on a voxel grid its
#: probe quantises to whole voxels: a nominal 1.4 A probe silently became
#: 1.5 and 2.0 A at those voxel sizes and swung the total by 10%.
#: Inflation slightly over-extends the outer surface instead, which is the
#: more defensible error of the two, since a protein surface does carry a
#: water depletion layer.
#:
#: Calibrated on heavy-atom models. A structure with hydrogens re-added
#: sits marginally higher, since most H fall inside a bonded heavy atom's
#: sphere but not all of them do.
SOLVENT_EXCLUDED_RADIUS_SCALE = 1.30


def atomic_occupancy(
    coordinates: torch.Tensor,
    atomic_numbers: torch.Tensor,
    n_xyz: int | tuple[int, int, int],
    dx: float,
    radius_scale: float = SOLVENT_EXCLUDED_RADIUS_SCALE,
    device: torch.device | str | None = None,
    chunk_size: int = 65536,
) -> torch.Tensor:
    """
    Fraction of each voxel filled by the specimen.

    Rasterizes the union of per-atom van der Waals balls, scaled by
    `radius_scale` to the solvent-excluded volume, with partial-volume
    coverage across the boundary rather than a binary mask.

    The grid matches :class:`~specter.potential.PotentialBuilder`'s: voxel
    centres at ``(i - n/2 + 0.5) * dx``, with coordinates centred on the
    box, so the result can be used voxel-for-voxel against a potential
    built from the same coordinates.

    Parameters
    ----------
    coordinates : torch.Tensor
        Atom positions in Angstrom, shape (N, 3) as (x, y, z).
    atomic_numbers : torch.Tensor
        Atomic numbers, shape (N,), same order as `coordinates`.
    n_xyz : int or tuple of int
        Grid size as ``(nx, ny, nz)``. An int means a cubic grid.
    dx : float
        Voxel size in Angstrom.
    radius_scale : float, optional
        Multiplier on every van der Waals radius. Default
        :data:`SOLVENT_EXCLUDED_RADIUS_SCALE`. Pass ``1.0`` for the bare
        van der Waals union, which is 0.685 of the solvent-excluded volume.
    device : torch.device or str, optional
        Device to build on. Default None, meaning `coordinates`' own.
    chunk_size : int, optional
        Atoms processed per batch. Bounds peak memory to roughly
        ``chunk_size * (2k+1)**3`` elements, where ``k`` covers the largest
        radius. Default 65536.

    Returns
    -------
    torch.Tensor
        Occupancy in [0, 1], shape ``(nz, ny, nx)`` to match a potential
        volume's ``(Z, Y, X)`` axis order. 1 where the specimen fills the
        voxel, 0 where it is empty.

    Notes
    -----
    Atoms are combined through the union's signed distance function,
    ``min_i(|r - p_i| - R_i)``, with the partial-volume ramp applied once
    to that. Ramping each atom separately and taking the maximum is the
    same function, since clamping is monotonic and the max of a monotonic
    transform is that transform of the max; this spelling is written the
    way it is only because the union is what the field means.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            f"coordinates must have shape (N, 3), got {tuple(coordinates.shape)}"
        )
    if len(coordinates) != len(atomic_numbers):
        raise ValueError(
            f"coordinates and atomic_numbers must be the same length, got "
            f"{len(coordinates)} and {len(atomic_numbers)}"
        )
    if dx <= 0:
        raise ValueError(f"dx must be positive, got {dx}")
    if radius_scale <= 0:
        raise ValueError(f"radius_scale must be positive, got {radius_scale}")

    if isinstance(n_xyz, int):
        nx = ny = nz = n_xyz
    else:
        nx, ny, nz = (int(v) for v in n_xyz)

    device = torch.device(device) if device is not None else coordinates.device
    coords = coordinates.to(device=device, dtype=torch.float32)
    radii = (
        torch.tensor(
            [
                VDW_RADII_A.get(int(z), DEFAULT_VDW_RADIUS_A)
                for z in atomic_numbers.tolist()
            ],
            dtype=torch.float32,
            device=device,
        )
        * radius_scale
    )

    # Accumulates the union's signed distance, so it starts "far outside".
    # Finite rather than inf: an untouched voxel still has to survive the
    # arithmetic below and come out as empty.
    far = 1.0e30
    sdf = torch.full((nz * ny * nx,), far, dtype=torch.float32, device=device)

    # One stencil sized for the largest radius serves every atom: a smaller
    # ball simply evaluates to zero over the outer shell of the window.
    k = int(torch.ceil(radii.max() / dx + 0.5).item())
    off = torch.arange(-k, k + 1, device=device, dtype=torch.float32)
    oz, oy, ox = torch.meshgrid(off, off, off, indexing="ij")
    oz, oy, ox = oz.reshape(-1), oy.reshape(-1), ox.reshape(-1)

    # Voxel centre i sits at (i - n/2 + 0.5) * dx, so a position p maps to
    # the fractional index p/dx + n/2 - 0.5.
    half = torch.tensor([nx, ny, nz], device=device, dtype=torch.float32) / 2 - 0.5

    for start in range(0, len(coords), chunk_size):
        p = coords[start : start + chunk_size]
        r = radii[start : start + chunk_size, None]
        fidx = p / dx + half  # (M, 3) fractional (x, y, z) index
        base = torch.round(fidx).to(torch.int64)

        ix = base[:, 0, None] + ox.to(torch.int64)
        iy = base[:, 1, None] + oy.to(torch.int64)
        iz = base[:, 2, None] + oz.to(torch.int64)

        # Distance from the atom to each candidate voxel's centre, minus the
        # atom's radius: this atom's own signed distance.
        d = (
            torch.sqrt(
                (ix.to(torch.float32) - fidx[:, 0, None]) ** 2
                + (iy.to(torch.float32) - fidx[:, 1, None]) ** 2
                + (iz.to(torch.float32) - fidx[:, 2, None]) ** 2
            )
            * dx
        )
        signed = d - r

        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        # Out-of-box candidates must not win the minimum.
        signed = torch.where(inside, signed, torch.full_like(signed, far))
        flat = ((iz * ny + iy) * nx + ix).clamp_(0, nz * ny * nx - 1)
        sdf.scatter_reduce_(0, flat.reshape(-1), signed.reshape(-1), reduce="amin")

    # One partial-volume ramp over the union's surface: full a half-voxel
    # inside, empty a half-voxel outside, linear across.
    occ = (0.5 - sdf / dx).clamp_(0.0, 1.0)
    return occ.reshape(nz, ny, nx)


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
    """Separable Gaussian blur over the last three axes, edges replicated."""
    r = max(1, int(round(3 * sigma_vox)))
    x = torch.arange(-r, r + 1, device=V.device, dtype=V.dtype)
    kernel = torch.exp(-0.5 * (x / sigma_vox) ** 2)
    kernel = kernel / kernel.sum()

    lead = V.shape[:-3]
    out = V.reshape(-1, 1, *V.shape[-3:])
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = -1
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = r
        pad[2 * (2 - axis) + 1] = r
        out = torch.nn.functional.conv3d(
            torch.nn.functional.pad(out, pad, mode="replicate"),
            kernel.view(*shape),
        )
    return out.reshape(*lead, *V.shape[-3:])


def potential_occupancy(
    V: torch.Tensor,
    pixel_size: float,
    sigma_a: float = WATER_COARSE_GRAIN_SIGMA_A,
    full_potential: float = FULL_OCCUPANCY_POTENTIAL_V,
) -> torch.Tensor:
    """
    Fraction of each voxel already filled, read off the potential.

    The counterpart to :func:`atomic_occupancy` for a volume whose geometry
    is not known -- a supplied map, a bulk material, crowding duplicates
    summed in after the fact. Coarse-grains `V` to the water probe's own
    length scale first, then reads the volume fraction against
    `full_potential`.

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
    full_potential : float, optional
        Potential of a fully-occupied voxel, V. Default
        :data:`FULL_OCCUPANCY_POTENTIAL_V`, protein's mean inner potential.

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
    of its water. Where a generator knows its own geometry, prefer
    :func:`atomic_occupancy` or an analytic field.
    """
    if pixel_size <= 0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size}")
    if sigma_a < 0:
        raise ValueError(f"sigma_a must be non-negative, got {sigma_a}")
    if full_potential <= 0:
        raise ValueError(f"full_potential must be positive, got {full_potential}")

    field = V.detach()
    sigma_vox = sigma_a / pixel_size
    if sigma_vox >= 0.25:
        field = _gaussian_blur3d(field, sigma_vox)
    return (field / full_potential).clamp_(0.0, 1.0)
