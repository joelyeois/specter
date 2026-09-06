"""
Microtubule surface-lattice geometry: the helical arrangement of tubulin
dimers on a closed tube.

Unlike ``_generator.py``'s single-strand filaments, a microtubule is a tube
of ``n_protofilaments`` parallel strands whose registers are staggered so
that the lateral tubulin-tubulin bonds form an ``n_start``-start helix. The
two consequences this module exists to reproduce are the tube's radius (set
by how many protofilaments have to fit around the circumference) and the
**seam** -- the single A-lattice contact where the helix fails to close on
a whole number of dimers.

Constants below are measured off deposited microtubule reconstructions
rather than taken from a textbook -- 3JAL (13-protofilament
GMPCPP-microtubule + EB3) and 6DPU (14-protofilament kinesin-microtubule):

===================  ==============  ==============  =====================
quantity             3JAL (13 pf)    6DPU (14 pf)    behaviour across N
===================  ==============  ==============  =====================
radius of pf centres    111.1 A         118.5 A      scales with N
lateral spacing          53.8 A          53.3 A      ~constant
monomer rise             40.9 A          41.4 A      ~constant
lateral stagger           9.43 A          9.00 A     tracks n_start*rise/N
===================  ==============  ==============  =====================

The last row matters: the stagger is **not** a fixed property of the
lateral bond. A lattice with a different protofilament number accommodates
it by sliding the lateral bond, which keeps the protofilaments essentially
parallel to the tube axis at every N. Protofilament skew (the "supertwist"
of non-13-protofilament microtubules) is therefore not modelled here: the
residual implied by deposited helical parameters is ~0.1 degrees, and those
parameters are not precise enough to pin it down (two 14-protofilament
entries report 25.7 vs 25.75 degrees per subunit, which alone moves the
implied skew by ~0.15 degrees). Modelling it properly needs direct moire-
period measurements as a source, not PDB metadata.

References
----------
Zhang, R., Alushin, G. M., Brown, A., & Nogales, E. (2015). Mechanistic
origin of microtubule dynamic instability and its modulation by EB
proteins. Cell, 162(4), 849-859. (PDB 3JAL)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Axial rise per tubulin **monomer** along a protofilament, Å.
#: Measured 40.9 A (3JAL, GMPCPP) and 41.4 A (6DPU, GDP).
MONOMER_RISE = 41.0

#: Axial repeat of the alpha-beta **dimer**, i.e. the stamped unit, Å.
DIMER_REPEAT = 2 * MONOMER_RISE

#: Centre-to-centre spacing between adjacent protofilaments, Å.
#: Measured 53.8 A (3JAL) and 53.3 A (6DPU) -- near-constant across
#: protofilament number, which is why the tube radius simply follows the
#: circumference.
LATERAL_SPACING = 53.5

#: Flexural persistence length of a microtubule, Å (1 mm). Measured
#: values span roughly 1-5 mm depending on nucleotide state, taxol and the
#: measurement method; the low end is used here so that thermal bending is
#: not under-stated. Used only to derive the path's per-step flex angle --
#: see `thermal_flex_deg`.
PERSISTENCE_LENGTH = 1.0e7


@dataclass(frozen=True)
class TubeLattice:
    """Surface-lattice geometry of one microtubule.

    Attributes
    ----------
    n_protofilaments : int
        Number of protofilaments around the tube.
    n_start : int
        Number of lateral-bond helix starts (3 for a real microtubule).
    radius : float
        Radius of the protofilament centres from the tube axis, Å.
    stagger : float
        Axial offset between adjacent protofilaments' registers, Å.
    dimer_repeat : float
        Axial repeat of the stamped alpha-beta dimer, Å.
    """

    n_protofilaments: int
    n_start: int
    radius: float
    stagger: float
    dimer_repeat: float

    @property
    def seam_offset(self) -> float:
        """Register mismatch across the seam, Å.

        Walking all the way around the tube accumulates
        ``n_protofilaments * stagger = n_start * MONOMER_RISE`` of axial
        offset. Reduced modulo the dimer repeat, whatever is left over is
        the mismatch between the lattice's prediction for protofilament 0
        and where protofilament 0 actually sits -- the seam. For a real
        3-start microtubule this is exactly one monomer, i.e. an alpha
        subunit ends up against a beta subunit (an A-lattice contact)
        where every other junction is B-lattice.
        """
        return (self.n_protofilaments * self.stagger) % self.dimer_repeat


def solve_tube_lattice(n_protofilaments: int = 13, n_start: int = 3) -> TubeLattice:
    """
    Build the surface lattice for an ``n_protofilaments``-stranded tube.

    Parameters
    ----------
    n_protofilaments : int, optional
        Protofilaments around the tube. Default 13, the canonical in-cell
        microtubule.
    n_start : int, optional
        Lateral-bond helix starts. Default 3.

    Returns
    -------
    TubeLattice

    Notes
    -----
    Two relations, both consequences of closing the tube:

    .. math::
        R = \\frac{N \\, a_\\mathrm{lat}}{2\\pi}, \\qquad
        s = \\frac{n_\\mathrm{start} \\, r}{N}

    The radius follows from fitting ``N`` protofilaments at their measured
    lateral spacing around the circumference; the stagger from requiring
    that ``N`` lateral steps rise by exactly ``n_start`` monomers.

    Examples
    --------
    >>> lattice = solve_tube_lattice(13)
    >>> round(lattice.radius, 1)          # measured 111.1 A in 3JAL
    110.7
    >>> round(lattice.stagger, 2)         # measured 9.43 A in 3JAL
    9.46
    >>> round(lattice.seam_offset, 1)     # one monomer -> A-lattice seam
    41.0
    """
    if n_protofilaments < 3:
        raise ValueError(
            f"n_protofilaments must be >= 3 to close a tube, got {n_protofilaments}"
        )
    if n_start < 1:
        raise ValueError(f"n_start must be >= 1, got {n_start}")

    return TubeLattice(
        n_protofilaments=n_protofilaments,
        n_start=n_start,
        radius=n_protofilaments * LATERAL_SPACING / (2 * math.pi),
        stagger=n_start * MONOMER_RISE / n_protofilaments,
        dimer_repeat=DIMER_REPEAT,
    )


def thermal_flex_deg(
    step: float = DIMER_REPEAT, persistence_length: float = PERSISTENCE_LENGTH
) -> float:
    """
    Per-step flex angle reproducing a given persistence length.

    The filament walk (`generate_filament_path`) draws each step's turn
    from ``U(0, flex)``, so the mean square turn is ``flex**2 / 3``, while
    a worm-like chain of persistence length ``L_p`` has mean square
    tangent deviation ``2 * step / L_p`` over one step. Equating them:

    .. math::
        f = \\sqrt{6 \\, \\Delta s / L_p}

    Parameters
    ----------
    step : float, optional
        Contour length per path step, Å. Default `DIMER_REPEAT`.
    persistence_length : float, optional
        Persistence length, Å. Default `PERSISTENCE_LENGTH` (1 mm).

    Returns
    -------
    float
        Flex angle in degrees -- ~0.40 degrees for a microtubule, i.e. a
        tube that stays straight to within about half its own diameter
        across a whole tomogram field. Strongly curved microtubules in
        real cellular tomograms are mechanically buckled, not thermally
        bent; use `MicrotubuleSpec.bend_radius` for those rather than
        inflating this.
    """
    if persistence_length <= 0:
        raise ValueError(f"persistence_length must be > 0, got {persistence_length}")
    return math.degrees(math.sqrt(6 * step / persistence_length))


@dataclass
class MicrotubuleSpec:
    """One microtubule species to place.

    Parameters
    ----------
    code : str or None, optional
        PDB ID or local file path for the alpha-beta tubulin dimer repeated
        through the lattice. Default None, which resolves to a dimer
        extracted from `_tubulin.MT_DIMER_SOURCE` and re-expressed in the
        microtubule frame -- the only way the wall comes out right, since
        the dimer's radial face has to point outwards (see `_tubulin`).
    n_protofilaments : int, optional
        Default 13.
    n_start : int, optional
        Default 3.
    n_copies : int, optional
        Independent microtubules of this species. Default 1.
    length : float or None, optional
        Contour length, Å. Default None: span the volume's diagonal,
        so the microtubule crosses the field the way real ones do instead
        of appearing as a stub.
    bend_radius : float or None, optional
        Radius of curvature, Å. Default None: a thermal random walk
        at `thermal_flex_deg`, which is nearly straight. Set a value (e.g.
        3e4 for 3 um) for the smooth, strongly curved microtubules seen in
        cellular tomograms, which are mechanically constrained rather than
        thermally bent -- a bigger flex angle would give a tangled tube,
        not a curved one.
    confine_to_slab : bool, optional
        Reject initial directions that would leave the volume's thinnest
        dimension before the microtubule reaches its full length. Default
        True, which reproduces the in-plane bias thin-ice microtubules show
        without introducing a tilt-angle parameter.
    """

    code: str | None = None
    n_protofilaments: int = 13
    n_start: int = 3
    n_copies: int = 1
    length: float | None = None
    bend_radius: float | None = None
    confine_to_slab: bool = True

    def lattice(self) -> TubeLattice:
        """The surface lattice this spec describes."""
        return solve_tube_lattice(self.n_protofilaments, self.n_start)
