"""
Filament placement: random walks of monomer positions/orientations scattered
through a specimen volume, one call per set of like filaments (a "species").

Pure geometry only -- no PDB fetching, no potential rendering. A caller
(``TomogramSpecimenGenerator._stamp_filaments``) fetches each spec's
monomer structure, builds one potential template per species, then
rotates/inserts a copy at every ``FilamentInstance`` this module returns --
the same division of labor its protein stamping already uses (geometry
decided here, rendering done by the caller, which alone knows about voxel
sizes/atomic potentials).

The presets: `ACTIN_SPEC` stacks one subunit of a deposited ADP-F-actin
filament (PDB 7BT7 [4]_), lifted out in the filament frame by
``_actin.extract_actin_subunit``, on F-actin's measured helix, a rise of
27.5 A and a twist of -166.6 degrees per subunit [6]_ [7]_ (the left-handed
genetic helix of the Holmes model [2]_, whose twist varies from subunit to
subunit [1]_); `PROTOFILAMENT_SPEC` stacks one alpha-beta tubulin dimer of
a deposited microtubule (PDB 3JAL [3]_), lifted out in the microtubule
frame by ``_tubulin.extract_mt_dimer``, at the dimer repeat of the GDP
microtubule lattice, `_lattice.DIMER_REPEAT`, with no twist. Both
`flex_deg` values are CryoTomoSim's [5]_.

References
----------
.. [1] Egelman, E. H., Francis, N., & DeRosier, D. J. (1982). F-actin is a helix
   with a random variable twist. *Nature*, 298(5870), 131–135.
   https://doi.org/10.1038/298131a0
.. [2] Holmes, K. C., Popp, D., Gebhard, W., & Kabsch, W. (1990). Atomic model
   of the actin filament. *Nature*, 347(6288), 44–49.
   https://doi.org/10.1038/347044a0
.. [3] Zhang, R., Alushin, G. M., Brown, A., & Nogales, E. (2015).
   Mechanistic origin of microtubule dynamic instability and its
   modulation by EB proteins. *Cell*, 162(4), 849–859.
   https://doi.org/10.1016/j.cell.2015.07.012 (PDB 3JAL)
.. [4] Kumari, A., Kesarwani, S., Javoor, M. G., Vinothkumar, K. R., &
   Sirajuddin, M. (2020). Structural insights into actin filament
   recognition by commonly used cellular actin markers. *The EMBO Journal*,
   39(14), e104006. https://doi.org/10.15252/embj.2019104006 (PDB 7BT7)
.. [5] Purnell, C., Heebner, J., Swulius, M. T., Hylton, R., Kabonick, S.,
   Grillo, M., Grigoryev, S., Heberle, F., Waxham, M. N., & Swulius, M. T.
   (2023). Rapid synthesis of cryo-ET data for training deep learning models.
   *bioRxiv*. https://doi.org/10.1101/2023.04.28.538636
   (CTS source: https://github.com/carsonpurnell/cryotomosim_CTS)
.. [6] Dominguez, R., & Holmes, K. C. (2011). Actin structure and function.
   *Annual Review of Biophysics*, 40, 169–186.
   https://doi.org/10.1146/annurev-biophys-042910-155359
   (-166.6 ± 0.6 degrees and 27.6 A per subunit; -166.15 degrees is the
   idealised 13/6 value, which repeats after exactly 13 subunits.)
.. [7] von der Ecken, J., Müller, M., Lehman, W., Manstein, D. J., Penczek,
   P. A., & Raunser, S. (2015). Structure of the F-actin–tropomyosin
   complex. *Nature*, 519(7541), 114–117. https://doi.org/10.1038/nature14033
   (27.5 A and 166.4 degrees per subunit.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import torch

from ._actin import ACTIN_FILAMENT_SOURCE
from ._lattice import DIMER_REPEAT
from ._path import generate_filament_path
from ._placement import filament_orientations
from ._tubulin import MT_DIMER_SOURCE


#: How the structure a `FilamentSpec` names is placed relative to the
#: filament axis; see `FilamentSpec.frame`.
FilamentFrame = Literal["principal_axis", "actin_filament", "microtubule_dimer"]


@dataclass
class FilamentSpec:
    """One filament species to place.

    Parameters
    ----------
    code : str
        PDB ID (fetched from RCSB) or local file path for the structure
        the subunit comes from: the monomer itself, or, for an extracted
        `frame`, the deposited model it is lifted out of -- an F-actin
        filament for ``"actin_filament"``, a microtubule (a PDB ID whose
        asymmetric unit holds at least three neighbouring protofilaments)
        for ``"microtubule_dimer"``.
    step : float
        Axial rise per monomer, Å.
    flex_deg : float
        Maximum per-step path curvature, degrees (see
        ``generate_filament_path``).
    twist_deg : float, optional
        Helical twist per monomer, degrees, accumulated along the filament:
        a rotation about the direction of growth by the right-hand rule, so
        a positive value builds a right-handed helix and a negative one a
        left-handed helix (the sign convention of deposited helical
        parameters, which pair a twist with a positive rise). Default 0 (no
        twist -- a single, untwisted strand, e.g. a microtubule
        protofilament).
    n_copies : int, optional
        Number of independent filament instances of this species to place.
        Default 1.
    n_monomers : int or tuple of int, optional
        Monomer count per filament instance. A fixed int, or a
        ``(min, max)`` range (inclusive) drawn uniformly per instance.
        Default ``(10, 30)``.
    frame : {"principal_axis", "actin_filament", "microtubule_dimer"}, optional
        Where each subunit sits relative to the filament axis.
        ``"principal_axis"`` (default): `code` is an isolated monomer,
        centred on the axis with its longest principal axis along it, an
        approximation that fixes neither the subunit's distance from the
        axis nor its roll about it. ``"actin_filament"``: `code` is a
        deposited model of F-actin itself; one subunit is lifted out of it
        together with the filament axis (``_actin.extract_actin_subunit``),
        and every copy sits where that subunit sits in the deposited
        filament, off the axis. `ACTIN_SPEC` uses it.
        ``"microtubule_dimer"``: `code` is a deposited microtubule; one
        alpha-beta dimer is lifted out of it in the microtubule frame
        (``_tubulin.extract_mt_dimer``), the frame `MicrotubuleSpec` builds
        its tube wall from. The filament axis is the protofilament's own
        axis, through the midpoint of the dimer's two monomers, not the
        tube axis (111 A away in 3JAL): ``+Z`` runs along the protofilament toward
        the plus end, and ``+X`` is the face that pointed out of the tube.
        With ``twist_deg=0`` and ``step=DIMER_REPEAT``, a straight filament
        is one protofilament of the source lattice. `PROTOFILAMENT_SPEC`
        uses it.
    """

    code: str
    step: float
    flex_deg: float
    twist_deg: float = 0.0
    n_copies: int = 1
    n_monomers: int | tuple[int, int] = (10, 30)
    frame: FilamentFrame = "principal_axis"

    def __post_init__(self) -> None:
        if self.frame not in get_args(FilamentFrame):
            raise ValueError(
                f"FilamentSpec.frame must be one of {get_args(FilamentFrame)}, "
                f"got {self.frame!r}"
            )


#: One alpha-beta tubulin dimer of the deposited microtubule 3JAL
#: (`_tubulin.MT_DIMER_SOURCE`, Zhang et al. 2015), lifted out in the
#: microtubule frame (`_tubulin.extract_mt_dimer`) and repeated along a
#: **single protofilament** -- not a microtubule. It is the dimer a
#: `MicrotubuleSpec` builds its wall from, placed the same way: the path
#: runs through the midpoint of the dimer's two monomers, on the
#: protofilament's own axis, and each copy's +Z (alpha to beta) points
#: along it toward the plus end. step is the dimer repeat of the GDP
#: microtubule lattice (`_lattice.DIMER_REPEAT`, 82.0 A), shared with
#: `MicrotubuleSpec` rather than written twice; twist=0, since a
#: protofilament is a straight column of dimers. A straight copy therefore
#: superposes on a protofilament of 3JAL itself (CA RMSD 0.14 A over two
#: dimers, ``tests/test_microtubule_lattice.py``). flex=3 deg is CTS's own
#: tuned value for a protofilament's gentle curvature; the outward
#: curling of a real free protofilament is not modelled.
#:
#: For an actual microtubule -- 13 protofilaments closed into a tube, with
#: the lumen and the A-lattice seam -- use `MicrotubuleSpec` and
#: `place_microtubules` (``_lattice``/``_tube``). This preset remains for
#: the free protofilaments that do occur, e.g. at depolymerising ends.
#:
#: (Named MICROTUBULE_SPEC before real microtubules existed in specter;
#: renamed rather than reused, so that an existing config asking for
#: MICROTUBULE_SPEC fails loudly instead of silently changing meaning.)
PROTOFILAMENT_SPEC = FilamentSpec(
    code=MT_DIMER_SOURCE,
    step=DIMER_REPEAT,
    flex_deg=3.0,
    twist_deg=0.0,
    n_monomers=(10, 30),
    frame="microtubule_dimer",
)

#: F-actin: one subunit of the deposited ADP-F-actin filament 7BT7
#: (Kumari et al. 2020), lifted out in the filament frame
#: (`_actin.extract_actin_subunit`), so each copy sits 15.7 A off the axis
#: with its longest principal axis 81 deg from it, as in the deposited
#: filament. step=27.5 A and twist=-166.6 deg are F-actin's measured rise
#: and twist per subunit (Dominguez & Holmes 2011; von der Ecken et al.
#: 2015), and 7BT7's own deposited helical parameters; 6DJO gives
#: -166.55/27.45. The sign makes the one-start genetic helix left-handed,
#: which in turn makes the two long-pitch strands right-handed
#: (2 * -166.6 + 360 = +26.8 deg per 55 A). The idealised 13/6 twist,
#: -166.15 deg, repeats after exactly 13 subunits and is a drawing
#: convenience, not a measurement. flex=12 deg is CTS's own tuned value for
#: actin's greater flexibility relative to microtubules.
ACTIN_SPEC = FilamentSpec(
    code=ACTIN_FILAMENT_SOURCE,
    step=27.5,
    flex_deg=12.0,
    twist_deg=-166.6,
    n_monomers=(20, 60),
    frame="actin_filament",
)


@dataclass
class FilamentInstance:
    """One placed monomer copy, for rendering and ground-truth bookkeeping.

    Microtubules (``_tube.place_microtubules``) reuse this type so that the
    tomogram generator's existing filament stamping renders them unchanged;
    for those, `filament_id` identifies the tube and `protofilament_index`
    says which of its protofilaments the copy belongs to.
    """

    code: str
    filament_id: int
    monomer_index: int
    #: (3,), physical Å: the copy's point on the filament path. For a
    #: filament that is a point on its axis, level with the subunit, not
    #: necessarily the subunit's own centre: an ``"actin_filament"`` subunit
    #: sits off the axis, at ``position_xyz + rotation_matrix @ c`` for its
    #: centre ``c`` in the filament frame. For a ``"microtubule_dimer"``
    #: filament it is the midpoint of the dimer's two monomers, on the
    #: protofilament axis; for a microtubule, the dimer's centre on the
    #: tube wall.
    position_xyz: torch.Tensor
    #: (3, 3): template-to-world rotation; see `filament_orientations`.
    rotation_matrix: torch.Tensor
    protofilament_index: int | None = None


def _monomer_count(
    n_monomers: int | tuple[int, int], generator: torch.Generator | None
) -> int:
    if isinstance(n_monomers, int):
        return n_monomers
    low, high = n_monomers
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def place_filaments(
    specs: list[FilamentSpec],
    target_shape: tuple[int, int, int],
    voxel_size: float,
    generator: torch.Generator | None = None,
) -> list[FilamentInstance]:
    """
    Place every filament instance for every spec, each starting at a
    uniformly random point within the volume's physical extent (filaments
    that wander outside the volume are simply truncated at render/insert
    time, the same edge behavior CTS/polnet-placed particles already get).

    Parameters
    ----------
    specs : list of FilamentSpec
    target_shape : tuple of int
        Specimen volume shape (Z, Y, X), voxels.
    voxel_size : float
        Voxel size, Å.
    generator : torch.Generator, optional
        Random generator for start positions, initial directions, and all
        per-step turns. Default None (uses torch's global RNG).

    Returns
    -------
    list of FilamentInstance
    """
    extent_xyz = torch.tensor(target_shape[::-1], dtype=torch.float32) * voxel_size

    instances: list[FilamentInstance] = []
    for spec in specs:
        for filament_id in range(spec.n_copies):
            n = _monomer_count(spec.n_monomers, generator)
            origin_xyz = torch.rand(3, generator=generator) * extent_xyz
            positions = generate_filament_path(
                n, spec.step, spec.flex_deg, origin_xyz, generator
            )
            rotations = filament_orientations(positions, spec.twist_deg)
            for i in range(n):
                instances.append(
                    FilamentInstance(
                        code=spec.code,
                        filament_id=filament_id,
                        monomer_index=i,
                        position_xyz=positions[i],
                        rotation_matrix=rotations[i],
                    )
                )
    return instances
