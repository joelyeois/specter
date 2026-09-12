"""
Absorption: the imaginary part of the scattering potential.

An electron leaves the elastic channel by several routes -- inelastic
scattering (plasmon, ionisation), elastic scattering to angles outside the
objective aperture, and thermal diffuse scattering. A simulator represents all
of them together as the imaginary component of a complex potential, which
damps the wave as it propagates rather than tracking where each electron went.

This module holds absorption models that **transform a potential that already
exists**. A model that instead computes the imaginary component *from atoms*
-- the Weickenmeier-Kohl style absorptive form factors, say -- does not belong
here: it needs atomic numbers, species and a Debye-Waller factor rather than a
finished potential, so it belongs beside its elastic counterparts in
`specter.atom` and is selected through `PotentialBuilder`,
exactly as `kirkland`/`lobato`/`shtyrov` are. The split is by input, not by
subject matter, which is easy to get wrong when looking for "the absorption
code".

Applied where slices materialise, not to the whole volume up front.
`IterativeScattering` never holds the volume -- `_iter_slices` rotates and
fetches on demand -- so there is no "before propagation" moment to convert in.
And `complex64` is exactly twice `float32`: converting a 300x1200x1200 tomogram
up front takes it from 1.73 GB to 3.46 GB, which is the regime that class
exists to avoid. Per slice the complex form is ~11.5 MB at a time. `Scattering`
converts the whole volume instead, because it already has it and its boxes are
small enough that doubling fits. Hoisting either call for tidiness would fail
as an out-of-memory on a production tomogram, not as a test failure.

Whatever applies absorption, the stage after it has to be told: `Aberration`
takes `specimen_absorption`, and setting it wrongly double-counts amplitude
contrast at both the specimen and the lens.
"""

from __future__ import annotations

import torch

from ..constants import interaction_parameter
from ._occupancy import (
    FULL_OCCUPANCY_POTENTIAL_V,
    occupancy_blur_halo_voxels,
    potential_occupancy,
)

INELASTIC_MFP_ICE_A = 3950.0
r"""
Apparent inelastic mean free path of amorphous ice at 300 kV, in Angstrom.

Measured: 395 +/- 11 nm on a Krios with a BioQuantum energy filter (Rice et
al., 2018, *J. Struct. Biol.* **204**, 38-44), agreeing with the ~400 nm of
Yonekura et al. (2006). Simulated independently at 3956 A by Himes &
Grigorieff (2021, *IUCrJ* **8**, 943-953).

This is the *apparent* mean free path -- what leaves the recorded zero-loss
channel through a real objective aperture and energy slit -- which is the
quantity a simulator of filtered images needs. The computed *total* inelastic
mean free path is shorter, ~350 nm (Vulović et al., 2013, Fig. 1, from the
Langmore & Smith cross section), and using it would over-absorb by 13%.

**Do not scale this to another voltage.** The Langmore-Smith energy dependence
gives Lambda(300)/Lambda(120) = 1.39 where measurement gives 1.70 (partial,
4.2 mrad) to 2.45 (total): Grimm et al. (1996) measure 232 nm and 161 nm at
120 kV, and Yesibolati et al. (2020) find the standard models underestimate
lambda for water outright. Another voltage needs its own measurement.
"""

INELASTIC_MFP_PROTEIN_A = 2460.0
r"""
Inelastic mean free path of protein at 300 kV, in Angstrom. Provisional.

Derived, not measured: ``nu = sigma_inel/sigma_el = 20/Z`` per element
(Reimer & Ross-Messemer, 1990) against isolated-atom elastic cross sections,
evaluated for protein's composition (H 0.492, C 0.313, N 0.094, O 0.101 by
atom fraction at 1.35 g/cm^3, Vulović et al. 2013) and for water, with the
ratio anchored to :data:`INELASTIC_MFP_ICE_A` so the absolute normalisation
of ``nu`` never has to be trusted. That gives 0.62 x ice.

**This number carries real uncertainty and it propagates.** Excluding hydrogen
from the ``nu`` weighting -- defensible, since ``20/Z`` was measured on
elemental specimens and Z = 1 is far outside that calibration -- gives 0.485 x
ice, or 1915 A. A Malis/Egerton-style estimate gives 317 nm, but that formula
carries no density term at all, which is why it puts protein and ice nearly
together; a mean free path is ``1/(n sigma)`` and density cannot drop out.
The three readings span 1915-3170 A.

What rides on it: the amplitude contrast a particle carries against the water
it displaces is ``(V_ab,protein - V_ab,ice) / (V_0,protein - V_0,ice)``, which
is 0.048 at 2460 A and 0.020 at 3170 A. Replace this with a measurement when
one exists.
"""

__all__ = [
    "INELASTIC_MFP_ICE_A",
    "INELASTIC_MFP_PROTEIN_A",
    "absorption_potential",
    "apply_amplitude_contrast",
    "inelastic_absorption_potential",
]


def absorption_potential(mfp_A: float, voltage_kv: float) -> float:
    r"""
    The absorption potential that reproduces an inelastic mean free path.

    Vulović et al. (2013) Eq. (3),

    .. math::
        V_{ab} = \frac{1}{2 \sigma \Lambda_{in}}

    with :math:`\sigma` the interaction parameter. A complex potential
    :math:`V + i V_{ab}` transmits amplitude :math:`e^{-\sigma V_{ab} t}`, so
    intensity follows Beer-Lambert with exactly this mean free path. At 300 kV
    and :data:`INELASTIC_MFP_ICE_A` this is 0.194 V.

    Parameters
    ----------
    mfp_A : float
        Inelastic mean free path in Angstrom. Must be positive.
    voltage_kv : float
        Accelerating voltage in kV.

    Returns
    -------
    float
        Absorption potential in volts.

    Raises
    ------
    ValueError
        If `mfp_A` is not positive.
    """
    if mfp_A <= 0.0:
        raise ValueError(f"mfp_A={mfp_A} must be positive")
    return 1.0 / (2.0 * interaction_parameter(voltage_kv) * mfp_A)


#: Voxels the occupancy blur is allowed to touch in one call. The blur is
#: separable, and its first pass reshapes to ``(-1, 1, X)`` and pads, so its
#: working set is a few multiples of the input. Whole-volume evaluation asked
#: for 26 GiB on a batch of four 512-pixel boxes with 1642 slices of ice and
#: brought a `specter match particles` run down; slabbing bounds it instead.
_OCCUPANCY_MAX_VOXELS_PER_SLAB = 2**26


def _occupancy_chunked(
    v: torch.Tensor,
    voxel_size: float,
    full_potential: float,
    max_voxels_per_slab: int,
) -> torch.Tensor:
    """
    :func:`potential_occupancy`, evaluated a z-slab at a time.

    Each slab is widened by :func:`occupancy_blur_halo_voxels` and the margin
    discarded, without which every slab boundary becomes an edge the blur
    sees. Identical to the whole-volume result wherever the halo fits, which
    is what ``tests/test_inelastic_absorption.py`` pins.

    Parameters
    ----------
    v : torch.Tensor
        Real potential, shape ``(..., Z, Y, X)``.
    voxel_size : float
        Voxel size in Angstrom.
    full_potential : float
        Potential of a fully occupied voxel, V.
    max_voxels_per_slab : int
        Upper bound on the voxels handed to the blur at once.

    Returns
    -------
    torch.Tensor
        Occupancy in [0, 1], same shape as `v`.
    """
    nz = v.shape[-3]
    per_slice = v.numel() // nz
    slab = max(1, max_voxels_per_slab // max(1, per_slice))
    if slab >= nz:
        return potential_occupancy(v, voxel_size, full_potential=full_potential)

    halo = occupancy_blur_halo_voxels(voxel_size)
    out = torch.empty_like(v)
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        lo, hi = max(0, z0 - halo), min(nz, z1 + halo)
        wide = potential_occupancy(
            v[..., lo:hi, :, :], voxel_size, full_potential=full_potential
        )
        out[..., z0:z1, :, :] = wide[..., z0 - lo : z0 - lo + (z1 - z0), :, :]
        del wide
    return out


def inelastic_absorption_potential(
    v: torch.Tensor,
    voxel_size: float,
    voltage_kv: float,
    mfp_solvent_A: float = INELASTIC_MFP_ICE_A,
    mfp_specimen_A: float | None = None,
    full_potential: float = FULL_OCCUPANCY_POTENTIAL_V,
    max_voxels_per_slab: int = _OCCUPANCY_MAX_VOXELS_PER_SLAB,
) -> torch.Tensor:
    r"""
    Absorption potential from measured mean free paths, per material.

    Vulović et al. (2013) Eq. (3) with the material fraction read off the
    potential:

    .. math::
        V_{ab}(\mathbf{r}) = \frac{o(\mathbf{r})}{2\sigma\Lambda_{spec}}
                           + \frac{1 - o(\mathbf{r})}{2\sigma\Lambda_{solv}}

    where :math:`o` is :func:`potential_occupancy`, the same coarse-grained
    material-fraction field :func:`~specter.ice.blend_ice_into_volume` uses to
    decide where water goes. Using it here rather than a segmentation is what
    makes this work on a specimen with no labels.

    Absorption is a property of the *material*, not of the local potential.
    That is the whole difference from :func:`apply_amplitude_contrast`, which
    ties it to every atomic cusp: measured on amorphous ice, the two put the
    same total attenuation in but differ by 23x in how much absorption
    contrast they inject into the 30-3 A band.

    Parameters
    ----------
    v : torch.Tensor
        Real scattering potential in volts, shape ``(..., Z, Y, X)``. Only the
        specimen belongs here; the solvent term is applied where the specimen
        is not.
    voxel_size : float
        Voxel size in Angstrom. Physical, since the occupancy coarse-graining
        length is.
    voltage_kv : float
        Accelerating voltage in kV.
    mfp_solvent_A : float, optional
        Inelastic mean free path of the embedding medium in Angstrom. Default
        :data:`INELASTIC_MFP_ICE_A`, measured for amorphous ice at 300 kV --
        so a run at another voltage must pass its own.
    mfp_specimen_A : float or None, optional
        Inelastic mean free path of the specimen material in Angstrom.
        Default None uses `mfp_solvent_A`, which makes the field uniform: the
        specimen then absorbs exactly like the solvent it displaces and
        carries no absorption *contrast* at all, only the bulk attenuation.
        Pass :data:`INELASTIC_MFP_PROTEIN_A` for protein, noting the
        uncertainty documented there.
    full_potential : float, optional
        Potential of a fully occupied voxel, V. Default
        :data:`FULL_OCCUPANCY_POTENTIAL_V`.
    max_voxels_per_slab : int, optional
        Bound on the voxels the occupancy blur touches at once. The blur is
        evaluated a z-slab at a time past this, each slab widened by
        `occupancy_blur_halo_voxels` and the margin discarded, so the result
        is unchanged. Default 2**26; whole-volume evaluation asked for 26 GiB
        on a batch of four 512-pixel boxes with 1642 slices of ice.

    Returns
    -------
    torch.Tensor
        Absorption potential in volts, same shape and dtype as `v`. Combine
        with `v` via ``torch.complex(v, v_ab)`` and hand that to
        :class:`~specter.scattering.Scattering`, which uses a complex
        potential as given and ignores its ``alpha``.

    Notes
    -----
    Absorption costs nothing in time -- the wave is complex either way, so
    only an elementwise term changes -- but materialising a complex volume
    costs 3.5x the resident memory at a 512 box, because it defeats
    :meth:`~specter.scattering.Scattering.multislice`'s per-chunk
    complexification. Building the complex volume a slice chunk at a time
    keeps it near 2x.

    The field is an isotropic pointwise function of the potential, so it
    rotates with the specimen exactly as the elastic potential does and can be
    built in the molecule frame. That is *not* true of a model carrying the
    plasmon's transverse point spread function, which is defined by the beam
    direction and must be applied after the pose rotation.

    Examples
    --------
    >>> import torch
    >>> from specter.potential import inelastic_absorption_potential
    >>> v = torch.zeros(4, 8, 8)  # pure solvent
    >>> v_ab = inelastic_absorption_potential(v, 1.0, 300.0)
    >>> round(float(v_ab.mean()), 4)
    0.194
    """
    v_solvent = absorption_potential(mfp_solvent_A, voltage_kv)
    if mfp_specimen_A is None:
        # No occupancy needed: the field is uniform, so the blur is skipped
        # entirely rather than computed and then ignored.
        return torch.full_like(v, v_solvent)
    v_specimen = absorption_potential(mfp_specimen_A, voltage_kv)
    occupancy = _occupancy_chunked(v, voxel_size, full_potential, max_voxels_per_slab)
    return occupancy * (v_specimen - v_solvent) + v_solvent


def apply_amplitude_contrast(v: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    r"""
    Make a real potential complex, using the amplitude-contrast ratio.

    The empirical model the field fits per dataset: one scalar
    :math:`\alpha`, uniform over the whole specimen, rotating the potential
    into the complex plane so that a fraction of the wave is absorbed rather
    than phase-shifted.

    .. math::
        V \to V\left(\sqrt{1 - \alpha^2} + i\alpha\right)

    ``alpha`` is the quantity every format carries:
    ``rlnAmplitudeContrast`` in RELION, ``ctf/amp_contrast`` in CryoSPARC,
    ``amplitude_contrast`` in torch-ctf, conventionally 0.07-0.10.
    :func:`inelastic_absorption_potential` is the sibling that separates the
    routes instead, and is preferred wherever a mean free path is known.

    **``alpha`` is not a measurement of absorption, and 0.07-0.10 is not a
    measured range.** CTF estimation takes amplitude contrast as an *input*
    (it sits beside voltage and Cs in ctffind's parameter list) and never fits
    it; real metadata carries one value for a whole dataset next to Cs while
    defocus varies per particle; and the fitting objective is nearly flat in
    it -- on a simulated micrograph built with ``alpha = 0.10``, ctffind still
    scores best at a supplied 0.00, the score varying by 2-4% over
    ``0 <= alpha <= 0.2``. In the CTF, ``alpha`` is a phase rotation of the
    ring pattern by ``arctan(alpha)``, 5.7 degrees at 0.1.

    Read as absorption it is also not physically available: 0.1 removes 51% of
    the beam through 1200 A of ice, where inelastic scattering accounts for
    26% and inelastic plus *all* elastic scattering caps at 39%. Measured on
    EMPIAR-11377, ``alpha = 0.1`` corresponds to an attenuation length of
    166 nm against ice's measured 395 nm.

    Parameters
    ----------
    v : torch.Tensor
        Scattering potential. A complex `v` already carries its absorption and
        is returned unchanged, since rotating it again in the complex plane is
        meaningless; the guard matters because
        :class:`~specter.scattering.IterativeScattering` calls this
        unconditionally.
    alpha : float, optional
        Amplitude contrast ratio. Default 0.1.

    Returns
    -------
    torch.Tensor
        Complex potential, real part scaled by ``sqrt(1 - alpha**2)`` and
        imaginary part by ``alpha``. Returned unchanged when ``alpha`` is
        zero, which keeps a purely phase-contrast run exactly real, and when
        `v` is already complex.
    """
    if alpha == 0.0 or v.is_complex():
        return v
    return v * ((1 - alpha**2) ** 0.5 + 1j * alpha)
