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
rather than taken from a textbook -- 3JAL (13-protofilament GMPCPP
microtubule co-polymerized with EB3, which compacts the lattice to the GDP
spacing [1]_), 6DPU (14-protofilament undecorated GMPCPP microtubule, an
expanded lattice [2]_) and 6DPV (14-protofilament undecorated GDP
microtubule [2]_). Radius and lateral spacing are fitted to the deposited
coordinates (`_tubulin.measure_source_lattice`); rise and stagger come from
the deposited helical parameters, whose lateral rise ``s`` gives the
monomer rise as ``s * N / 3``:

===================  ===============  ===============  ===============  ===================
quantity             3JAL (13 pf)     6DPU (14 pf)     6DPV (14 pf)     behaviour
===================  ===============  ===============  ===============  ===================
nucleotide           GMPCPP + EB3     GMPCPP           GDP
radius of pf centres    110.9 A          118.0 A          118.5 A       scales with N
lateral spacing          53.7 A           53.2 A           53.5 A       ~constant
monomer rise             41.0 A           42.0 A           40.85 A      set by nucleotide
lateral stagger           9.46 A           8.99 A           8.75 A      n_start*rise/N
===================  ===============  ===============  ===============  ===================

`MONOMER_RISE` is the compacted GDP value, 41.0 A, since a microtubule in a
cell is a GDP lattice: GTP hydrolysis compacts the dimer repeat from 83 to
81 A [5]_, and the 13-protofilament dimer rises are 81.76 A (GDP) and
83.95 A (GMPCPP) in Zhang et al.'s Table S2 [2]_.

The 3-start helix, the B-lattice and the seam follow Mandelkow et al. [3]_,
including the hand: the 3-start helix is left-handed. The deposited models
agree -- stepping laterally to the neighbouring protofilament turns
-27.7 degrees about the tube axis for +9.46 A of rise in 3JAL, and
-25.75 degrees for +8.99 A in 6DPU -- and ``_tube.build_tube_instances``
lays the protofilaments out with that hand.

The last row of the table matters: the stagger is **not** a fixed property
of the lateral bond. A lattice with a different protofilament number
accommodates it by sliding the lateral bond, which keeps the protofilaments
essentially parallel to the tube axis at every N. Protofilament skew (the
"supertwist" of non-13-protofilament microtubules [4]_) is therefore not
modelled here: the residual implied by deposited helical parameters is
~0.1 degrees, and those parameters are not precise enough to pin it down
(two 14-protofilament entries report 25.7 vs 25.75 degrees per subunit,
which alone moves the implied skew by ~0.15 degrees). Modelling it properly
needs direct moire-period measurements as a source, not PDB metadata.

The axis follows a persistent random walk whose flex angle
(`thermal_flex_deg`) reproduces `PERSISTENCE_LENGTH`, the apparent
persistence length of microtubules in cells, ~30 um [6]_. That is about 100
times shorter than the thermal value measured in vitro, 0.5-8 mm across
studies [7]_ (5.2 mm for taxol-stabilized microtubules [8]_), because the
bends a microtubule acquires while it grows are frozen in by the surrounding
cytoskeleton. A microtubule under compressive load buckles instead, into
arcs a few micrometres long [9]_; `MicrotubuleSpec.bend_radius` models
that case, as an arc in place of the walk.

References
----------
.. [1] Zhang, R., Alushin, G. M., Brown, A., & Nogales, E. (2015). Mechanistic
   origin of microtubule dynamic instability and its modulation by EB
   proteins. *Cell*, 162(4), 849–859. https://doi.org/10.1016/j.cell.2015.07.012
   (PDB 3JAL, 3JAT)
.. [2] Zhang, R., LaFrance, B., & Nogales, E. (2018). Separating the effects of
   nucleotide and EB binding on microtubule structure. *Proceedings of the
   National Academy of Sciences*, 115(27), E6191–E6200.
   https://doi.org/10.1073/pnas.1802637115 (PDB 6DPU, 6DPV)
.. [3] Mandelkow, E. M., Schultheiss, R., Rapp, R., Müller, M., & Mandelkow, E.
   (1986). On the surface lattice of microtubules: helix starts,
   protofilament number, seam, and handedness. *The Journal of Cell Biology*,
   102(3), 1067–1073. https://doi.org/10.1083/jcb.102.3.1067
.. [4] Chrétien, D., & Wade, R. H. (1991). New data on the microtubule surface
   lattice. *Biology of the Cell*, 71(1–2), 161–174.
   https://doi.org/10.1016/0248-4900(91)90062-R
.. [5] Alushin, G. M., Lander, G. C., Kellogg, E. H., Zhang, R., Baker, D., &
   Nogales, E. (2014). High-resolution microtubule structures reveal the
   structural transitions in αβ-tubulin upon GTP hydrolysis. *Cell*, 157(5),
   1117–1129. https://doi.org/10.1016/j.cell.2014.03.053
.. [6] Brangwynne, C. P., MacKintosh, F. C., & Weitz, D. A. (2007). Force
   fluctuations and polymerization dynamics of intracellular microtubules.
   *Proceedings of the National Academy of Sciences*, 104(41), 16128–16133.
   https://doi.org/10.1073/pnas.0703094104
.. [7] Pampaloni, F., Lattanzi, G., Jonáš, A., Surrey, T., Frey, E., &
   Florin, E.-L. (2006). Thermal fluctuations of grafted microtubules provide
   evidence of a length-dependent persistence length. *Proceedings of the
   National Academy of Sciences*, 103(27), 10248–10253.
   https://doi.org/10.1073/pnas.0603931103 (0.5-8 mm across the literature;
   6.3 mm in the long-filament limit.)
.. [8] Gittes, F., Mickey, B., Nettleton, J., & Howard, J. (1993). Flexural
   rigidity of microtubules and actin filaments measured from thermal
   fluctuations in shape. *The Journal of Cell Biology*, 120(4), 923–934.
   https://doi.org/10.1083/jcb.120.4.923 (5.2 mm for taxol-stabilized
   microtubules.)
.. [9] Brangwynne, C. P., MacKintosh, F. C., Kumar, S., Geisse, N. A.,
   Talbot, J., Mahadevan, L., Parker, K. K., Ingber, D. E., & Weitz, D. A.
   (2006). Microtubules can bear enhanced compressive loads in living cells
   because of lateral reinforcement. *The Journal of Cell Biology*, 173(5),
   733–741. https://doi.org/10.1083/jcb.200601060 (buckling wavelength
   ~3 um in cells.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Axial rise per tubulin **monomer** along a protofilament, Å: the
#: compacted GDP lattice of a microtubule in a cell. Deposited values are
#: 40.85 A (6DPV, GDP), 41.0 A (3JAL, GMPCPP co-polymerized with EB3, which
#: compacts the lattice) and 42.0 A for the expanded, undecorated GMPCPP
#: lattice (6DPU). See the module docstring.
MONOMER_RISE = 41.0

#: Axial repeat of the alpha-beta **dimer**, i.e. the stamped unit, Å.
DIMER_REPEAT = 2 * MONOMER_RISE

#: Centre-to-centre spacing between adjacent protofilaments, Å.
#: Measured 53.7 A (3JAL), 53.2 A (6DPU) and 53.5 A (6DPV) -- near-constant
#: across protofilament number, which is why the tube radius simply follows
#: the circumference.
LATERAL_SPACING = 53.5

#: Apparent persistence length of a microtubule in a cell, Å (30 um,
#: Brangwynne et al. 2007). Bends acquired during growth are frozen in by
#: the surrounding cytoskeleton, which makes in-cell shapes a persistent
#: random walk with a persistence length about 100 times shorter than the
#: thermal one measured in vitro (0.5-8 mm, Pampaloni et al. 2006; 5.2 mm
#: for taxol-stabilized microtubules, Gittes et al. 1993). specter builds
#: cellular specimens, so the in-cell value is the default. Used only to
#: derive the path's per-step flex angle -- see `thermal_flex_deg`.
PERSISTENCE_LENGTH = 3.0e5


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
    >>> round(lattice.radius, 1)          # measured 110.9 A in 3JAL
    110.7
    >>> round(lattice.stagger, 2)         # 9.46 A in 3JAL
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

    The name records the derivation, a worm-like chain; the default
    persistence length is the apparent, non-thermal one of microtubules in
    cells (see `PERSISTENCE_LENGTH`).

    Parameters
    ----------
    step : float, optional
        Contour length per path step, Å. Default `DIMER_REPEAT`.
    persistence_length : float, optional
        Persistence length, Å. Default `PERSISTENCE_LENGTH` (30 um, the
        in-cell value). Pass ~5e7 (5 mm) for the thermal bending of a
        microtubule in vitro.

    Returns
    -------
    float
        Flex angle in degrees -- ~2.3 degrees for a microtubule in a cell,
        whose tangent then turns by ~11 degrees RMS over a 600 nm field.
        For a microtubule buckled under compressive load, which bends as a
        smooth arc rather than wandering, use `MicrotubuleSpec.bend_radius`
        rather than inflating this.
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
        Radius of curvature, Å. Default None: a persistent random walk at
        `thermal_flex_deg`, whose bends reproduce the apparent persistence
        length of microtubules in cells (`PERSISTENCE_LENGTH`, 30 um). Set
        a value (e.g. 3e4 for 3 um) for a microtubule buckled under
        compressive load, which bends as a smooth arc in one plane rather
        than wandering -- a bigger flex angle would give a kinked tube, not
        a curved one. The arc replaces the walk, so the two never compound.
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
