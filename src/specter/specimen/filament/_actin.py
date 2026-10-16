"""
Extraction of a single actin subunit **in the F-actin filament frame**.

F-actin is a one-start helix of subunits that do not sit on its axis: in
deposited filaments each subunit's centre lies 15.6-15.9 A from the axis
and its longest principal axis 74-81 degrees from it. Neither number
follows from an isolated monomer. Centring a G-actin structure on the axis
with its longest principal axis along it, as
``align_principal_axis_to_z`` would, builds a helix with F-actin's rise and
twist out of subunits in the wrong place and the wrong orientation.

The subunit geometry is taken instead from a structure that already has
it: a cryo-EM model of the filament itself. The filament axis is fitted
from the screw relating consecutive subunits (one Kabsch superposition per
neighbouring pair), and one subunit is re-expressed so that

* ``+Z`` is the filament axis, pointing to the **barbed end**,
* the origin lies on the axis, level with the subunit's centre, and
* ``+X`` points from the axis out through the subunit's centre,

so the subunit's centre sits at ``(r, 0, 0)`` with ``r`` its distance from
the axis. Copying it by the screw of the genetic helix, a rotation of
`ACTIN_SPEC`'s ``twist_deg`` about ``+Z`` with a translation of its
``step`` along it, rebuilds the deposited filament.

The barbed end is read off the subunit's subdomains. Subdomains 2 and 4,
residues 33-69 and 181-269 in the numbering of Kabsch et al. [2]_, face
the pointed end, and the D-loop of subdomain 2 reaches into the next
subunit along the same long-pitch strand on that side [3]_. In the default
source the centroid of subdomains 2 and 4 lies 21.7 A from that of
subdomains 1 and 3 along the axis, which fixes the polarity without
ambiguity.

The default source, PDB 7BT7, is an ADP-F-actin filament from Kumari et
al. [1]_; the subunit is extracted with its bound ADP and Mg2+.

References
----------
.. [1] Kumari, A., Kesarwani, S., Javoor, M. G., Vinothkumar, K. R., &
   Sirajuddin, M. (2020). Structural insights into actin filament
   recognition by commonly used cellular actin markers. *The EMBO Journal*,
   39(14), e104006. https://doi.org/10.15252/embj.2019104006 (PDB 7BT7)
.. [2] Kabsch, W., Mannherz, H. G., Suck, D., Pai, E. F., & Holmes, K. C.
   (1990). Atomic structure of the actin:DNase I complex. *Nature*,
   347(6288), 37–44. https://doi.org/10.1038/347037a0
.. [3] Dominguez, R., & Holmes, K. C. (2011). Actin structure and function.
   *Annual Review of Biophysics*, 40, 169–186.
   https://doi.org/10.1146/annurev-biophys-042910-155359
"""

from __future__ import annotations

from specter import logger

import math
import os
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial.transform import Rotation

from ...pdb import PDB
from ._tubulin import _chain_entities

#: Default source structure: an ADP-F-actin filament (chicken skeletal
#: muscle alpha-actin) at 3.8 A, Kumari et al. 2020. Its deposited helical
#: parameters, -166.6 degrees and 27.5 A per subunit, are `ACTIN_SPEC`'s
#: own; superposing its consecutive subunits gives -166.6 to -166.8 degrees
#: for 27.5 A. The five subunits of the deposited model leave two
#: neighbours on each side of the central one, which is the one extracted.
ACTIN_FILAMENT_SOURCE = "7BT7"

_MIN_ACTIN_RESIDUES = 330
_MAX_ACTIN_RESIDUES = 380

#: Axial rise between consecutive subunits of the genetic helix is ~27.5 A,
#: between neighbours on one long-pitch strand ~55 A; this window keeps the
#: former only.
_GENETIC_RISE_RANGE = (20.0, 35.0)

#: Residue ranges of subdomains 2 and 4 (pointed end) and of subdomains 1
#: and 3 (barbed end), Kabsch et al. 1990.
_POINTED_SUBDOMAINS = ((33, 69), (181, 269))
_BARBED_SUBDOMAINS = ((1, 32), (70, 180), (270, 375))

#: Smallest axial separation, Å, between the two pairs of subdomains for
#: the polarity to count as determined. 21.7 A in 7BT7.
_MIN_POLARITY_OFFSET = 5.0

#: Version of the extracted subunit, stamped into the cached file's data
#: block name. A cached subunit written by another version is re-extracted
#: rather than reused, so raise this whenever the atoms or the frame
#: `extract_actin_subunit` writes change.
_ACTIN_SUBUNIT_VERSION = 1


@dataclass(frozen=True)
class _ActinFilamentFit:
    """A deposited F-actin model's filament axis and subunit order."""

    #: Actin chain names, ordered along `barbed` (pointed end first).
    chains: list[str]
    #: Unit vector along the filament axis, pointing to the barbed end.
    barbed: np.ndarray
    #: A point on the filament axis.
    axis_point: np.ndarray
    #: Twist, degrees, and rise, Å, of each consecutive pair along `barbed`.
    twists_deg: list[float]
    rises: list[float]


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation ``R`` and translation ``t`` minimising ``|R s + t - q|``."""
    source_centre, target_centre = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_centre).T @ (target - target_centre)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, target_centre - rotation @ source_centre


def _residue_ca(chain: gemmi.Chain) -> dict[int, np.ndarray]:
    """CA coordinates of a chain's polymer residues, keyed by residue number."""
    out: dict[int, np.ndarray] = {}
    for res in chain.get_polymer():
        atom = res.find_atom("CA", "*")
        if atom is not None and res.seqid.num is not None:
            out[int(res.seqid.num)] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    return out


def _screw(
    ca_a: dict[int, np.ndarray], ca_b: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The rigid motion ``x -> R x + t`` superposing chain a onto chain b."""
    common = sorted(set(ca_a) & set(ca_b))
    return _kabsch(
        np.stack([ca_a[k] for k in common]), np.stack([ca_b[k] for k in common])
    )


def _subdomain_centroid(
    ca: dict[int, np.ndarray], ranges: tuple[tuple[int, int], ...]
) -> np.ndarray:
    return np.stack(
        [xyz for k, xyz in ca.items() if any(lo <= k <= hi for lo, hi in ranges)]
    ).mean(axis=0)


def _source_cif(source: str, pdb_cache_dir: str, verbose: bool) -> str:
    """Path to the source model: fetched for a PDB ID, as given for a file."""
    if len(source) == 4 and source.isalnum():
        return PDB.fetch_pdb_file(
            source,
            ext="cif",
            pdb_cache_dir=pdb_cache_dir,
            assembly=False,
            verbose=verbose,
        )
    if os.path.isfile(source):
        return source
    raise ValueError(f"{source!r} is neither a 4-character PDB ID nor an existing file")


def _actin_chains(
    st: gemmi.Structure, cif_path: str, source: str
) -> dict[str, dict[int, np.ndarray]]:
    """CA coordinates of every actin chain of a model."""
    entities = _chain_entities(st, cif_path)
    chains = {}
    for chain in st[0]:
        if "actin" not in entities.get(chain.name, ""):
            continue  # drops any bound marker, tropomyosin, myosin, ...
        ca = _residue_ca(chain)
        if _MIN_ACTIN_RESIDUES <= len(ca) <= _MAX_ACTIN_RESIDUES:
            chains[chain.name] = ca
    if len(chains) < 3:
        raise ValueError(
            f"{source}: found {len(chains)} actin chains, need >= 3 consecutive "
            "subunits to fit the filament axis"
        )
    return chains


def _fit_actin_filament(
    chains: dict[str, dict[int, np.ndarray]], source: str
) -> _ActinFilamentFit:
    """Fit the filament axis and the barbed-end direction of a model."""
    names = list(chains)
    centroids = {n: np.stack(list(chains[n].values())).mean(axis=0) for n in names}

    # Rough axis: the screw axes of every pair of subunits one genetic-helix
    # step apart, sign-aligned and averaged.
    directions = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rotation, translation = _screw(chains[a], chains[b])
            rotvec = Rotation.from_matrix(rotation).as_rotvec()
            axis = rotvec / np.linalg.norm(rotvec)
            if (
                _GENETIC_RISE_RANGE[0]
                < abs(translation @ axis)
                < _GENETIC_RISE_RANGE[1]
            ):
                directions.append(axis)
    if not directions:
        raise ValueError(
            f"{source}: no two actin chains are one genetic-helix step apart"
        )
    directions = [d if d @ directions[0] > 0 else -d for d in directions]
    rough = np.mean(directions, axis=0)
    rough /= np.linalg.norm(rough)

    # Order along the axis, and refine it from consecutive pairs only.
    order = sorted(names, key=lambda n: float(centroids[n] @ rough))
    axes, points = [], []
    for a, b in zip(order[:-1], order[1:]):
        rotation, translation = _screw(chains[a], chains[b])
        rotvec = Rotation.from_matrix(rotation).as_rotvec()
        axis = rotvec / np.linalg.norm(rotvec)
        axis = axis if axis @ rough > 0 else -axis
        rise = float(translation @ axis)
        if not _GENETIC_RISE_RANGE[0] < rise < _GENETIC_RISE_RANGE[1]:
            raise ValueError(
                f"{source}: chains {a} and {b} are adjacent along the axis but "
                f"{rise:.1f} A apart, not one genetic-helix step"
            )
        # The screw axis passes through the solution of (I - R) p = t_perp;
        # (I - R) is singular along the axis, so take the least-squares one.
        point = np.linalg.lstsq(
            np.eye(3) - rotation, translation - rise * axis, rcond=None
        )[0]
        axes.append(axis)
        points.append(point)
    direction = np.mean(axes, axis=0)
    direction /= np.linalg.norm(direction)
    axis_point = np.mean([p - (p @ direction) * direction for p in points], axis=0)

    # Subdomains 2 and 4 face the pointed end.
    offsets = [
        float(
            (
                _subdomain_centroid(chains[n], _POINTED_SUBDOMAINS)
                - _subdomain_centroid(chains[n], _BARBED_SUBDOMAINS)
            )
            @ direction
        )
        for n in order
    ]
    offset = float(np.mean(offsets))
    if abs(offset) < _MIN_POLARITY_OFFSET:
        raise ValueError(
            f"{source}: subdomains 2 and 4 sit {offset:+.1f} A from subdomains 1 "
            "and 3 along the axis; cannot tell the barbed end"
        )
    barbed = -direction if offset > 0 else direction
    if offset > 0:
        order = order[::-1]

    twists, rises = [], []
    for a, b in zip(order[:-1], order[1:]):
        rotation, translation = _screw(chains[a], chains[b])
        twists.append(
            math.degrees(float(Rotation.from_matrix(rotation).as_rotvec() @ barbed))
        )
        rises.append(float(translation @ barbed))
    return _ActinFilamentFit(order, barbed, axis_point, twists, rises)


def _subunit_block_name(stem: str) -> str:
    """mmCIF data block name of an extracted subunit, carrying its version."""
    return f"{stem}_actinsubunit_v{_ACTIN_SUBUNIT_VERSION}"


def _is_current_subunit(path: str, stem: str, source_path: str) -> bool:
    """Whether a cached subunit was written by this version of the
    extraction, after its source was last modified."""
    if os.path.getmtime(path) < os.path.getmtime(source_path):
        return False
    with open(path) as fh:
        return fh.readline().strip() == f"data_{_subunit_block_name(stem)}"


def extract_actin_subunit(
    source: str = ACTIN_FILAMENT_SOURCE,
    pdb_cache_dir: str | os.PathLike[str] | None = None,
    verbose: bool = False,
    overwrite: bool = False,
) -> str:
    """
    Extract one actin subunit in the filament frame.

    Parameters
    ----------
    source : str, optional
        PDB ID or local mmCIF path of a deposited F-actin model holding at
        least three consecutive subunits of the genetic helix. Chains whose
        entity is not actin (a bound marker, tropomyosin, myosin) are
        ignored. Default `ACTIN_FILAMENT_SOURCE`.
    pdb_cache_dir : str or path-like, optional
        Where a fetched source and the extracted subunit are cached.
        Default None: specter's usual PDB cache directory.
    verbose : bool, optional
        Log fetch and extraction detail. Default False.
    overwrite : bool, optional
        Re-extract even if a current cached subunit exists. Default False.

    Returns
    -------
    str
        Path to an mmCIF holding the central subunit of `source`, with its
        bound nucleotide and ions and without waters, in the filament
        frame: ``+Z`` along the filament axis towards the barbed end, the
        origin on the axis level with the subunit's centre, and ``+X``
        through that centre (see the module docstring).

    Raises
    ------
    ValueError
        If `source` has fewer than three actin chains, its chains are not
        consecutive subunits of the genetic helix, or its polarity cannot
        be read off the subdomains.
    """
    if pdb_cache_dir is None:
        from ...config import default_pdb_cache_dir

        pdb_cache_dir = default_pdb_cache_dir()
    pdb_cache_dir = str(pdb_cache_dir)

    cif_path = _source_cif(source, pdb_cache_dir, verbose)
    stem = Path(source).stem if os.path.isfile(source) else source.upper()
    out_path = os.path.join(pdb_cache_dir, f"{stem}-actin-subunit.cif")
    if (
        os.path.exists(out_path)
        and not overwrite
        and _is_current_subunit(out_path, stem, cif_path)
    ):
        return out_path

    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_waters()

    chains = _actin_chains(st, cif_path, source)
    fit = _fit_actin_filament(chains, source)
    name = fit.chains[len(fit.chains) // 2]

    out = gemmi.Structure()
    out.name = _subunit_block_name(stem)
    model = gemmi.Model(1)
    chain = st[0][name].clone()
    # Zero-occupancy atoms are not part of the model; `PDB` drops them on
    # one of its parse paths and not the other, so dropping them here keeps
    # the atom set the same whichever one reads this file.
    for res in chain:
        for i in reversed(range(len(res))):
            if res[i].occ <= 0.0:
                del res[i]
    model.add_chain(chain)
    out.add_model(model)

    xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for res in out[0][0] for a in res])
    centre = xyz.mean(axis=0)
    z_hat = fit.barbed
    origin = fit.axis_point + ((centre - fit.axis_point) @ z_hat) * z_hat
    radial = centre - origin
    radius = float(np.linalg.norm(radial))
    radial /= radius
    # Rows are the new basis expressed in the deposited frame, so
    # `basis @ (x - origin)` re-expresses a point in the filament frame.
    basis = np.stack([radial, np.cross(z_hat, radial), z_hat])
    for res in out[0][0]:
        for atom in res:
            x, y, z = basis @ (np.array([atom.pos.x, atom.pos.y, atom.pos.z]) - origin)
            atom.pos = gemmi.Position(float(x), float(y), float(z))
    out.setup_entities()

    os.makedirs(pdb_cache_dir, exist_ok=True)
    out.make_mmcif_document().write_file(out_path)

    if verbose:
        logger.info(
            f"{source}: subunit {name} of {len(fit.chains)} "
            f"({' '.join(fit.chains)}, pointed -> barbed), centre {radius:.2f} A "
            f"from the axis; consecutive screws "
            f"{np.mean(fit.twists_deg):+.2f} deg / {np.mean(fit.rises):.2f} A "
            f"-> {out_path}"
        )
    return out_path


def measure_actin_filament(
    source: str = ACTIN_FILAMENT_SOURCE,
    pdb_cache_dir: str | os.PathLike[str] | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Measure the helix and subunit geometry of a deposited F-actin model.

    Used to check `ACTIN_SPEC` against the structure its subunit comes
    from -- see ``tests/test_actin_subunit.py``.

    Parameters
    ----------
    source : str, optional
        PDB ID or local mmCIF path of an F-actin model. Default
        `ACTIN_FILAMENT_SOURCE`.
    pdb_cache_dir : str or path-like, optional
        PDB cache directory. Default None: specter's usual cache.
    verbose : bool, optional
        Log fetch detail. Default False.

    Returns
    -------
    dict
        ``twist_deg`` and ``rise`` (Å), the mean screw between
        consecutive subunits towards the barbed end (a negative twist is a
        left-handed genetic helix); ``radius`` (Å), the mean distance of a
        subunit's centroid from the axis; ``principal_axis_angle_deg``, the
        mean angle between a subunit's longest principal axis and the
        filament axis; and ``n_subunits``.
    """
    if pdb_cache_dir is None:
        from ...config import default_pdb_cache_dir

        pdb_cache_dir = default_pdb_cache_dir()
    cif_path = _source_cif(source, str(pdb_cache_dir), verbose)
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_waters()
    chains = _actin_chains(st, cif_path, source)
    fit = _fit_actin_filament(chains, source)

    radii, angles = [], []
    for name in fit.chains:
        xyz = np.array(
            [[a.pos.x, a.pos.y, a.pos.z] for res in st[0][name] for a in res]
        )
        offset = xyz.mean(axis=0) - fit.axis_point
        radii.append(float(np.linalg.norm(offset - (offset @ fit.barbed) * fit.barbed)))
        centred = xyz - xyz.mean(axis=0)
        principal = np.linalg.eigh(centred.T @ centred)[1][:, -1]
        angles.append(
            math.degrees(math.acos(min(1.0, abs(float(principal @ fit.barbed)))))
        )
    return {
        "twist_deg": float(np.mean(fit.twists_deg)),
        "rise": float(np.mean(fit.rises)),
        "radius": float(np.mean(radii)),
        "principal_axis_angle_deg": float(np.mean(angles)),
        "n_subunits": float(len(fit.chains)),
    }
