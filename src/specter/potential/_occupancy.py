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

import warnings
from collections.abc import Iterator

import torch

from ..atom import atom_mass
from ..filters import gaussian_blur3d

__all__ = [
    "FULL_OCCUPANCY_POTENTIAL_V",
    "PROTEIN_VOLUME_PER_DALTON_A3",
    "WATER_COARSE_GRAIN_SIGMA_ANGSTROM",
    "full_occupancy_potential",
    "template_occupancy_reference",
    "molecular_mass_from_atoms",
    "occupancy_blur_halo_voxels",
    "potential_occupancy",
    "potential_occupancy_slabs",
]

#: Scattering potential of a voxel entirely filled with biological
#: material, V. Protein's mean inner potential. Now the FALLBACK: a
#: generator that knows its template's mass solves its own reference
#: (:func:`template_occupancy_reference`), because the rendered potential
#: depends on the scattering table -- 5.91 V for 6BDF under Shtyrov with
#: hydrogens, 8.01 V under Kirkland -- while the water a molecule displaces
#: does not. This value is what a bare volume, or a tomogram's mixture of
#: species, is read against. The reference for how much
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

#: Molecular volume per dalton of protein, A^3/Da: the partial specific
#: volume vbar = 0.73 cm^3/g, the same basis :data:`FULL_OCCUPANCY_POTENTIAL_V`
#: was measured on.
PROTEIN_VOLUME_PER_DALTON_A3 = 1.2122

#: Heavy-atom share of a protein's mass. A model deposited without hydrogens
#: weighs this fraction of the molecule it describes (H is 49% of protein
#: atoms at 1.008 Da against a 7.19 Da mean), and vbar is per gram of the
#: whole molecule.
_PROTEIN_HEAVY_ATOM_MASS_FRACTION = 0.932

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
#:
#: WHY 2.0, AND WHY NOT LARGER. Grid-independence does not pick it. That
#: spread shrinks monotonically as sigma grows, so it only rules out SMALL
#: sigma; taken alone it would argue for 5 A. What picks 2.0 is spurious
#: water INSIDE the molecule, which is U-shaped in sigma: below it the
#: cusps admit ice into the gaps between bonded atoms, above it
#: over-smoothing bleeds the molecular boundary inward and the core stops
#: reading as occupied. Measured on 1A6M over voxel sizes 0.75-4 A:
#:
#:     sigma (A)          0    0.5    1.0    1.4    2.0    2.8    3.5    5.0
#:     volume spread   2.44   1.71   1.10  1.032  1.009  1.001  1.000  1.000
#:     ice inside     0.319  0.248  0.085  0.035  0.014  0.054  0.133  0.273
#:
#: 2.0 A minimises the second row while the first is already flat, and the
#: excluded volume there is 98% of its saturated value, so the exclusion is
#: essentially complete without the boundary having moved. It also sits
#: between water's van der Waals radius (1.4 A) and its diameter (2.8 A),
#: which is the length scale the question is about.
#:
#: Provenance, so the sweep is not mistaken for a fit: the value was chosen
#: as a water-sized length and documented, and the sweep was run afterwards
#: and vindicates it. Its "inside" mask -- voxels a sigma=2.8 A probe reads
#: above 0.9 -- is a fixed reference across the row but comes from the same
#: estimator family, so a geometric distance-to-nearest-atom mask would be
#: the cleaner test. The U-shape is not an artifact of that: were the mask
#: circular the minimum would sit at 2.8, not 2.0.
WATER_COARSE_GRAIN_SIGMA_ANGSTROM = 2.0


def occupancy_blur_halo_voxels(
    voxel_size: float, sigma_angstrom: float = WATER_COARSE_GRAIN_SIGMA_ANGSTROM
) -> int:
    """
    Voxels of context :func:`potential_occupancy` reads beyond its input.

    A caller that evaluates the field a z-slab at a time (to bound memory
    on a volume too large to blur whole) must extend each slab by this
    much and discard the margin, or every slab boundary becomes an edge
    the blur sees.

    Parameters
    ----------
    voxel_size : float
        Voxel size in Angstrom.
    sigma_angstrom : float, optional
        Coarse-graining length in Angstrom. Default
        :data:`WATER_COARSE_GRAIN_SIGMA_ANGSTROM`.

    Returns
    -------
    int
        Halo width in voxels. Zero when the blur is skipped as sub-voxel.
    """
    sigma_vox = sigma_angstrom / voxel_size
    if sigma_vox < 0.25:
        return 0
    return max(1, int(round(3 * sigma_vox)))


def potential_occupancy(
    V: torch.Tensor,
    voxel_size: float,
    sigma_angstrom: float = WATER_COARSE_GRAIN_SIGMA_ANGSTROM,
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
    voxel_size : float
        Voxel size in Angstrom. Required, and not merely for units:
        `sigma_angstrom` is physical, so this is what makes the result independent
        of the render grid.
    sigma_angstrom : float, optional
        Coarse-graining length in Angstrom. Default
        :data:`WATER_COARSE_GRAIN_SIGMA_ANGSTROM`.
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
    Skips the convolution when ``sigma_angstrom`` is under a quarter of a voxel.
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
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    if sigma_angstrom < 0:
        raise ValueError(f"sigma_angstrom must be non-negative, got {sigma_angstrom}")
    if isinstance(full_potential, torch.Tensor):
        if bool((full_potential <= 0).any()):
            raise ValueError("full_potential must be positive")
    elif full_potential <= 0:
        raise ValueError(f"full_potential must be positive, got {full_potential}")

    field = V.detach()
    sigma_vox = sigma_angstrom / voxel_size
    if sigma_vox >= 0.25:
        field = gaussian_blur3d(field, sigma_vox)
    return (field / full_potential).clamp_(0.0, 1.0)


def potential_occupancy_slabs(
    V: torch.Tensor,
    voxel_size: float,
    slab: int,
    full_potential: float | torch.Tensor = FULL_OCCUPANCY_POTENTIAL_V,
    sigma_angstrom: float = WATER_COARSE_GRAIN_SIGMA_ANGSTROM,
) -> Iterator[tuple[int, int, torch.Tensor]]:
    """
    :func:`potential_occupancy`, evaluated a z-slab at a time.

    Each slab of `slab` slices is widened by :func:`occupancy_blur_halo_voxels`
    on both sides, blurred, and the margin discarded, without which every
    slab boundary becomes an edge the blur sees. Wherever the halo fits,
    each slab is identical to the matching slices of the whole-volume field.

    Parameters
    ----------
    V : torch.Tensor
        Scattering potential in volts, shape ``(..., Z, Y, X)``.
    voxel_size : float
        Voxel size in Angstrom.
    slab : int
        Slices per slab, before the halo is added. Must be positive.
    full_potential : float or torch.Tensor, optional
        Potential of a fully occupied voxel, as :func:`potential_occupancy`'s.
    sigma_angstrom : float, optional
        Coarse-graining length in Angstrom. Default
        :data:`WATER_COARSE_GRAIN_SIGMA_ANGSTROM`.

    Yields
    ------
    tuple of (int, int, torch.Tensor)
        ``(z0, z1, occupancy)``, where `occupancy` is the field over
        ``V[..., z0:z1, :, :]``: a view into the widened slab's result, which
        the caller may modify in place. The slabs cover ``[0, Z)`` in
        ascending order.
    """
    nz = V.shape[-3]
    halo = occupancy_blur_halo_voxels(voxel_size, sigma_angstrom)
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        lo, hi = max(0, z0 - halo), min(nz, z1 + halo)
        wide = potential_occupancy(
            V[..., lo:hi, :, :],
            voxel_size,
            sigma_angstrom=sigma_angstrom,
            full_potential=full_potential,
        )
        core = wide[..., z0 - lo : z0 - lo + (z1 - z0), :, :]
        del wide
        yield z0, z1, core
        del core


def molecular_mass_from_atoms(atomic_numbers: torch.Tensor) -> float:
    """
    Mass of the molecule an atomic model describes, in daltons.

    The atoms' own mass, except that a model carrying no hydrogens has it
    divided by the heavy-atom share of protein mass, 0.932: the partial
    specific volume behind :data:`PROTEIN_VOLUME_PER_DALTON_A3` is per gram of
    the whole molecule, so a hydrogen-free deposition would otherwise
    displace 7% too little water.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        Atomic numbers of every atom in the structure, shape ``(N,)``.

    Returns
    -------
    float
        Molecular mass in daltons.
    """
    z = torch.as_tensor(atomic_numbers)
    mass = float(atom_mass(z.cpu()).double().sum())
    if not bool((z == 1).any()):
        mass /= _PROTEIN_HEAVY_ATOM_MASS_FRACTION
    return mass


def full_occupancy_potential(
    V: torch.Tensor,
    voxel_size: float,
    molecular_volume_A3: float,
    sigma_angstrom: float = WATER_COARSE_GRAIN_SIGMA_ANGSTROM,
    rtol: float = 1e-4,
) -> float:
    r"""
    The occupancy reference at which a template displaces its own volume.

    :func:`potential_occupancy` reads a voxel as full at ``full_potential``
    volts, and a fixed reference only suits renderings whose potential sits at
    that level. How much water a molecule pushes aside is geometry -- its mass
    and density -- not a property of the scattering factors it was rendered
    with, yet under the fixed 7.0 V it follows them: a Shtyrov render with
    hydrogens (5.9 V mean) displaced 0.84 of its volume, a Kirkland one
    (8.0 V) 1.09. This solves instead for the reference that makes the
    displaced volume exactly the molecule's,

    .. math::
        \sum_i \min\!\left(\frac{\bar V_i}{V_{ref}},\,1\right) \Delta^3
            = \mathcal{V}_{mol},

    with :math:`\bar V` the coarse-grained potential. Without the clamp this is
    the mean inner potential :math:`\int V / \mathcal{V}_{mol}` (the blur
    conserves the integral); the clamp discards what dense spots such as a
    heme iron hold above the reference, which the solution recovers by
    reading slightly lower.

    The left side decreases monotonically in :math:`V_{ref}`, so bisection
    converges; the mean inner potential brackets it from above. Padding the
    box with empty space changes nothing, since empty voxels contribute zero.

    Parameters
    ----------
    V : torch.Tensor
        The template's potential in volts, shape ``(Z, Y, X)``, the molecule
        alone and wholly inside the box.
    voxel_size : float
        Voxel size in Angstrom.
    molecular_volume_A3 : float
        The molecule's volume in A^3: its mass
        (:func:`molecular_mass_from_atoms`) times
        :data:`PROTEIN_VOLUME_PER_DALTON_A3`.
    sigma_angstrom : float, optional
        Coarse-graining length, as :func:`potential_occupancy`'s.
    rtol : float, optional
        Relative tolerance on the reference. Default 1e-4.

    Returns
    -------
    float
        The reference potential in volts.
    """
    if molecular_volume_A3 <= 0:
        raise ValueError(
            f"molecular_volume_A3 must be positive, got {molecular_volume_A3}"
        )
    field = V.detach().float()
    sigma_vox = sigma_angstrom / voxel_size
    if sigma_vox >= 0.25:
        field = gaussian_blur3d(field, sigma_vox)
    # Only occupied voxels matter to the sum; empty space is most of a box.
    field = field[field > 0].double()
    target = molecular_volume_A3 / voxel_size**3
    total = float(field.sum())
    if total <= 0:
        raise ValueError("V has no positive potential to read occupancy from")

    def displaced(ref: float) -> float:
        return float((field / ref).clamp_(max=1.0).sum())

    if field.numel() < target:
        raise ValueError(
            "V has fewer occupied voxels than the molecular volume requires; "
            "is the box clipping the molecule, or the mass too large?"
        )
    hi = total / target  # the mean inner potential: displaced(hi) <= target
    lo = hi / 2
    while displaced(lo) < target:
        lo /= 2
    while hi - lo > rtol * hi:
        mid = 0.5 * (lo + hi)
        if displaced(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def template_occupancy_reference(
    template: torch.Tensor | None, voxel_size: float, molecular_mass: float | None
) -> float:
    """
    The occupancy reference a generator reads its template's ice against, V.

    :func:`full_occupancy_potential` for the template when its mass is known,
    so it displaces exactly its own volume of ice; the fixed
    :data:`FULL_OCCUPANCY_POTENTIAL_V` when it is not, since a bare volume
    does not say how much molecule it holds. A box that clips the molecule
    cannot hold its volume, and falls back to the fixed reference with a
    warning rather than failing a run the old reference would have finished.

    Parameters
    ----------
    template : torch.Tensor or None
        The template's potential, shape ``(Z, Y, X)``, unscaled. None (an
        ice-only specimen) has nothing to displace ice with.
    voxel_size : float
        Voxel size in Angstrom.
    molecular_mass : float or None
        Mass of the molecule in daltons, hydrogens included
        (:func:`molecular_mass_from_atoms`), or None.

    Returns
    -------
    float
        The reference potential in volts.
    """
    if molecular_mass is None or template is None:
        return FULL_OCCUPANCY_POTENTIAL_V
    if molecular_mass <= 0:
        raise ValueError(f"molecular_mass must be positive, got {molecular_mass}")
    try:
        return full_occupancy_potential(
            template, voxel_size, molecular_mass * PROTEIN_VOLUME_PER_DALTON_A3
        )
    except ValueError as err:
        warnings.warn(
            f"Could not solve this template's occupancy reference ({err}); "
            f"falling back to the fixed {FULL_OCCUPANCY_POTENTIAL_V} V, so the ice "
            "it displaces follows its scattering factors. Enlarge the box to hold "
            "the whole molecule.",
            stacklevel=3,
        )
        return FULL_OCCUPANCY_POTENTIAL_V
