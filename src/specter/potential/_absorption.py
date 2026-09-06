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


def apply_amplitude_contrast(v: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    r"""
    Make a real potential complex, using the amplitude-contrast ratio.

    The empirical model the field fits per dataset: one scalar
    :math:`\alpha`, uniform over the whole specimen, rotating the potential
    into the complex plane so that a fraction of the wave is absorbed rather
    than phase-shifted.

    .. math::
        V \to V\left(\sqrt{1 - \alpha^2} + i\alpha\right)

    ``alpha`` is the quantity CTF estimation reports and every format carries:
    ``rlnAmplitudeContrast`` in RELION, ``ctf/amp_contrast`` in CryoSPARC,
    ``amplitude_contrast`` in torch-ctf. Typical fitted values are 0.07-0.10.
    It lumps every absorption route into one number and resolves none of them;
    a model that separates them would be a sibling of this function, or -- if
    it works from atoms rather than from a potential -- would live with the
    elastic parameterizations instead. See this module's own docstring.

    Parameters
    ----------
    v : torch.Tensor
        Real-valued scattering potential.
    alpha : float, optional
        Amplitude contrast ratio. Default 0.1.

    Returns
    -------
    torch.Tensor
        Complex potential, real part scaled by ``sqrt(1 - alpha**2)`` and
        imaginary part by ``alpha``. Returned unchanged when ``alpha`` is
        zero, which keeps a purely phase-contrast run exactly real.
    """
    if alpha == 0.0:
        return v
    return v * ((1 - alpha**2) ** 0.5 + 1j * alpha)
