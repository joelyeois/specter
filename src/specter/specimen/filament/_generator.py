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

The presets: `ACTIN_SPEC` stacks uncomplexed G-actin (PDB 1J6Z [4]_) on
F-actin's measured helix, a rise of 27.5 A and a twist of -166.6 degrees per
subunit [6]_ [7]_ (the left-handed genetic helix of the Holmes model [2]_,
whose twist varies from subunit to subunit [1]_); `PROTOFILAMENT_SPEC`
stacks the alpha-beta tubulin dimer (PDB 1TUB [3]_) at the dimer repeat of
the GDP microtubule lattice, `_lattice.DIMER_REPEAT`. Both `flex_deg` values
are CryoTomoSim's [5]_.

References
----------
.. [1] Egelman, E. H., Francis, N., & DeRosier, D. J. (1982). F-actin is a helix
   with a random variable twist. *Nature*, 298(5870), 131–135.
   https://doi.org/10.1038/298131a0
.. [2] Holmes, K. C., Popp, D., Gebhard, W., & Kabsch, W. (1990). Atomic model
   of the actin filament. *Nature*, 347(6288), 44–49.
   https://doi.org/10.1038/347044a0
.. [3] Nogales, E., Wolf, S. G., & Downing, K. H. (1998). Structure of the αβ
   tubulin dimer by electron crystallography. *Nature*, 391(6663), 199–203.
   https://doi.org/10.1038/34465
.. [4] Otterbein, L. R., Graceffa, P., & Dominguez, R. (2001). The crystal
   structure of uncomplexed actin in the ADP state. *Science*, 293(5530),
   708–711. https://doi.org/10.1126/science.1059700
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

import torch

from ._lattice import DIMER_REPEAT
from ._path import generate_filament_path
from ._placement import filament_orientations


@dataclass
class FilamentSpec:
    """One filament species to place.

    Parameters
    ----------
    code : str
        PDB ID (fetched from RCSB) or local file path for the monomer
        structure repeated along the filament.
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
    """

    code: str
    step: float
    flex_deg: float
    twist_deg: float = 0.0
    n_copies: int = 1
    n_monomers: int | tuple[int, int] = (10, 30)


#: Tubulin alpha-beta dimer (PDB 1TUB, electron crystallography) repeated
#: along a **single protofilament** -- not a microtubule. step is the dimer
#: repeat of the GDP microtubule lattice (`_lattice.DIMER_REPEAT`, 82.0 A),
#: shared with `MicrotubuleSpec` rather than written twice; flex=3 deg is
#: CTS's own tuned value for a protofilament's gentle curvature. twist=0: a
#: single protofilament doesn't itself twist.
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
    code="1TUB", step=DIMER_REPEAT, flex_deg=3.0, twist_deg=0.0, n_monomers=(10, 30)
)

#: Uncomplexed G-actin monomer (PDB 1J6Z) repeated as F-actin. step=27.5 A
#: and twist=-166.6 deg are F-actin's measured rise and twist per subunit
#: (Dominguez & Holmes 2011; von der Ecken et al. 2015; the deposited
#: filaments 6DJO and 7BT7 give -166.55/27.45 and -166.6/27.5). The sign
#: makes the one-start genetic helix left-handed, which in turn makes the two
#: long-pitch strands right-handed (2 * -166.6 + 360 = +26.8 deg per 55 A).
#: The idealised 13/6 twist, -166.15 deg, repeats after exactly 13 subunits
#: and is a drawing convenience, not a measurement. flex=12 deg is CTS's own
#: tuned value for actin's greater flexibility relative to microtubules.
#: The helix is F-actin's; the subunit arrangement is not: the monomer sits
#: on the axis with its longest principal axis along it, where a deposited
#: filament's subunits sit 16 A off the axis with that axis ~80 deg from it.
ACTIN_SPEC = FilamentSpec(
    code="1J6Z", step=27.5, flex_deg=12.0, twist_deg=-166.6, n_monomers=(20, 60)
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
    position_xyz: torch.Tensor  # (3,), physical Å
    rotation_matrix: torch.Tensor  # (3, 3)
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
