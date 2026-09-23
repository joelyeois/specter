"""
How many independent ice structures an exposure contains.

A movie is not a picture of one frozen solvent. The beam melts and
re-vitrifies the ice, and the specimen moves, so successive frames carry
different arrangements of the same water. That changes how the solvent adds
up. A frozen ice adds coherently: over ``N`` frames its amplitude grows as
``N`` and its power as ``N**2``, while shot noise grows only as ``N``, so its
speckle stands out ``N`` times more strongly in the sum than in one frame. An
ice that rearranges adds incoherently, and its prominence does not grow with
dose at all.

Real ice is between the two. Measured on raw movie frames of EMPIAR-11377 and
EMPIAR-11461 by tracking the 3.7 A ring's excess over its local baseline
against summed dose, the ring stops building after roughly 2 e-/A^2: fitting
an exponential loss of coherence gives 1.9 and 2.6 e-/A^2, so a 40-50 e-/A^2
exposure holds about ten independent ice structures rather than one. A model
that keeps the exposure frozen therefore overstates the water ring, and a
model that draws a fresh realisation for every frame understates it.

Only the FLUCTUATION decorrelates. The mean potential of the column is the
same water in every frame; it sets the inelastic absorption and how much
solvent the specimen displaces, and it does not average away. Scaling the
whole ice field would thin the ice, which is a different (and separately
observable) thing -- see :func:`ice_fluctuation_scale`.
"""

from __future__ import annotations

import math

#: Dose over which vitreous ice stays structurally coherent, in e-/A^2.
#: Fitted to the ring build-up of two Falcon 4i datasets (1.9 and 2.6); it
#: folds in beam-induced motion as well as re-vitrification, both of which
#: make the solvent add incoherently, and neither of which is a universal
#: constant. Treat it as a starting value to be measured per dataset where
#: movie frames are available, not as a property of ice.
DEFAULT_DECORRELATION_DOSE = 2.0


def coherent_dose_equivalent(dose: float, decorrelation_dose: float) -> float:
    """
    Dose over which the solvent adds coherently, in e-/A^2.

    With the ice's structural correlation decaying as
    ``exp(-|N - N'| / decorrelation_dose)`` across the exposure, the coherent
    sum of its contrast over a dose ``D`` is the double integral of that
    correlation, which has the closed form below. It tends to ``D`` for an ice
    that never rearranges and to ``2 * decorrelation_dose`` for one that
    rearranges quickly.

    Parameters
    ----------
    dose : float
        Exposure accumulated in the image, in e-/A^2.
    decorrelation_dose : float
        Dose over which the ice stays coherent, in e-/A^2.

    Returns
    -------
    float
        Coherent-dose equivalent, in e-/A^2.
    """
    if dose <= 0:
        raise ValueError("dose must be positive")
    if decorrelation_dose <= 0:
        raise ValueError("decorrelation_dose must be positive")
    dc = decorrelation_dose
    return 2.0 * dc * (dose - dc * (1.0 - math.exp(-dose / dc))) / dose


def effective_ice_realisations(dose: float, decorrelation_dose: float) -> float:
    """
    Number of independent ice structures an exposure contains.

    ``dose / coherent_dose_equivalent(...)``, which is 1 for an ice frozen for
    the whole exposure and grows as the ice rearranges faster. It is NOT capped
    at the frame count: the ice does not wait for the shutter, and an ice that
    rearranges within one frame already averages inside that frame.

    Parameters
    ----------
    dose : float
        Exposure accumulated in the image, in e-/A^2.
    decorrelation_dose : float
        Dose over which the ice stays coherent, in e-/A^2.

    Returns
    -------
    float
        Effective realisation count, at least 1.
    """
    return max(1.0, dose / coherent_dose_equivalent(dose, decorrelation_dose))


def ice_fluctuation_scale(dose: float, decorrelation_dose: float) -> float:
    """
    Factor to scale the ice's fluctuation about its mean by.

    Averaging ``n`` independent realisations divides their power by ``n``, so
    a single realisation reproduces the same solvent power when its
    fluctuation is scaled by ``1 / sqrt(n)``. Validated against explicitly
    rendering and averaging ``n`` realisations through the full forward model:
    the two agree to a median 0.3% across 40-2.5 A, with 2.7% scatter, on both
    datasets measured (see the repository's validation notes).

    Parameters
    ----------
    dose : float
        Exposure accumulated in the image, in e-/A^2.
    decorrelation_dose : float
        Dose over which the ice stays coherent, in e-/A^2.

    Returns
    -------
    float
        Multiplier in (0, 1] for the fluctuation, the mean left alone.
    """
    return 1.0 / math.sqrt(effective_ice_realisations(dose, decorrelation_dose))
