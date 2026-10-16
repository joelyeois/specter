"""
Extraction of a single alpha-beta tubulin dimer **in the microtubule frame**.

Placing a dimer into a tube wall needs two of its axes pinned down, not one:
the protofilament axis (which must follow the tube axis) *and* the face that
points radially outwards. ``align_principal_axis_to_z``, used for filament
monomers and membrane proteins, fixes only the first -- the roll about it is
arbitrary, and a wrong roll builds the wall out of randomly rolled dimers,
giving the wrong wall thickness and lumen diameter.

Rather than calibrate that roll, take it from a structure that already has
it: deposited microtubule reconstructions are solved in the microtubule
frame, with protofilaments running as columns of monomers at constant
``(x, y)``. Lifting one dimer out of such a model, together with the tube
axis fitted from its neighbouring protofilaments, gives both axes directly
off the deposited geometry -- no fitting, no calibration.

Which beta belongs to which alpha is read off the nucleotides. Each
monomer binds a guanine nucleotide on the face that points to the
microtubule plus end: alpha's (the non-exchangeable N-site) sits at the
interface inside the dimer, capped by the dimer's own beta, and beta's (the
exchangeable E-site) at the interface with the next dimer, exposed at the
plus end [2]_ [3]_ [4]_. Neither the chain names nor the geometry along the
axis identify the pair -- both betas next to an alpha are one monomer rise
away -- so the partner is the beta whose atoms touch the alpha's
nucleotide, and alpha -> beta is the plus-end direction. In 3JAL that beta
comes within 3.6 A of the alpha's GTP, the beta on the alpha's other side
26 A, and the plus end points along the deposited ``-z``.

The extracted dimer is re-expressed so that

* ``+Z`` is the protofilament axis, pointing alpha -> beta, toward the
  microtubule plus end,
* ``+X`` points radially outwards, away from the tube axis, and
* the origin is the midpoint of the alpha and beta CA centroids, which
  lies on the protofilament's own axis (both centroids sit within 0.2 A of
  ``Z`` in 3JAL): a tube puts it on the wall, a free protofilament
  (``_generator.PROTOFILAMENT_SPEC``) on its path,

then cached next to the downloaded structure so the (small) extraction runs
once per source. The default source, PDB 3JAL, is from Zhang et al. [1]_.

References
----------
.. [1] Zhang, R., Alushin, G. M., Brown, A., & Nogales, E. (2015). Mechanistic
   origin of microtubule dynamic instability and its modulation by EB
   proteins. *Cell*, 162(4), 849–859. https://doi.org/10.1016/j.cell.2015.07.012
   (PDB 3JAL, 3JAT)
.. [2] Nogales, E., Wolf, S. G., & Downing, K. H. (1998). Structure of the
   αβ tubulin dimer by electron crystallography. *Nature*, 391(6663),
   199–203. https://doi.org/10.1038/34465
.. [3] Nogales, E., Whittaker, M., Milligan, R. A., & Downing, K. H. (1999).
   High-resolution model of the microtubule. *Cell*, 96(1), 79–88.
   https://doi.org/10.1016/S0092-8674(00)80961-7
.. [4] Mitchison, T. J. (1993). Localization of an exchangeable GTP binding
   site at the plus end of microtubules. *Science*, 261(5124), 1044–1047.
   https://doi.org/10.1126/science.8102497
"""

from __future__ import annotations

from specter import logger

import math
import os

import gemmi
import numpy as np

from ...pdb import PDB

#: Default source structure: a 13-protofilament GMPCPP-microtubule
#: co-polymerized with EB3 (Zhang et al. 2015). EB3 compacts the lattice,
#: so its monomer rise (41.0 A) is the GDP lattice's rather than the 42.0 A
#: of undecorated GMPCPP. Its deposited asymmetric unit is a
#: 3-protofilament x 4-monomer patch -- enough neighbouring protofilaments
#: to fit the tube axis, which a single isolated dimer would not give.
#: The two 131-residue EB3 chains are dropped during extraction.
#:
#: Chosen over the more common kinesin-decorated microtubule structures
#: (e.g. 3JAT) and the undecorated 6DPU because those are 14-protofilament
#: lattices, so their dimer sits at a different radius from the
#: 13-protofilament default.
MT_DIMER_SOURCE = "3JAL"

_MIN_TUBULIN_RESIDUES = 380
_MAX_TUBULIN_RESIDUES = 480
#: Protofilament columns are ~53 A apart, monomers within a column ~41 A;
#: this groups chains into columns without catching the neighbour.
_COLUMN_RADIUS = 25.0

#: Guanine nucleotides deposited at tubulin's N- and E-sites: GTP, GDP and
#: the slowly hydrolysable analogues GMPCPP (G2P), GMPPCP (GCP), GTPgammaS
#: (GSP) and GMPPNP (GNP).
_GUANINE_NUCLEOTIDES = frozenset({"GTP", "GDP", "G2P", "GCP", "GSP", "GNP"})

#: Largest atom-atom distance, Å, at which a beta still counts as capping an
#: alpha's N-site nucleotide. 3.6 A in 3JAL; the beta on the far side of the
#: alpha, which belongs to the previous dimer, is 26 A away.
_N_SITE_CONTACT = 6.0

#: Version of the extracted dimer, stamped into the cached file's data block
#: name. A cached dimer written by another version is re-extracted rather
#: than reused, so raise this whenever the atoms or the frame
#: `extract_mt_dimer` writes change. Unstamped files (version 1) paired each
#: alpha with the beta of the neighbouring dimer, and their +Z pointed at
#: the minus end.
_MT_DIMER_VERSION = 2


def _ca_centroid(chain: gemmi.Chain) -> np.ndarray | None:
    """Centroid of a chain's CA atoms, or None if it is not a protein chain
    of roughly tubulin size."""
    ca = [[a.pos.x, a.pos.y, a.pos.z] for res in chain for a in res if a.name == "CA"]
    if not _MIN_TUBULIN_RESIDUES <= len(ca) <= _MAX_TUBULIN_RESIDUES:
        return None
    return np.asarray(ca).mean(axis=0)


def _nucleotide_xyz(chain: gemmi.Chain) -> np.ndarray | None:
    """Coordinates of the guanine nucleotide a tubulin chain binds, or None."""
    xyz = [
        [a.pos.x, a.pos.y, a.pos.z]
        for res in chain
        if res.name in _GUANINE_NUCLEOTIDES
        for a in res
    ]
    return np.asarray(xyz) if xyz else None


def _polymer_xyz(chain: gemmi.Chain) -> np.ndarray:
    """Coordinates of a chain's polymer atoms, ligands excluded."""
    return np.asarray(
        [[a.pos.x, a.pos.y, a.pos.z] for res in chain.get_polymer() for a in res]
    )


def _central_column(
    columns: list[list[str]],
    centroids: dict[str, np.ndarray],
    direction: np.ndarray,
    axis_point: np.ndarray,
) -> list[str]:
    """The protofilament column in the middle of the patch, by azimuth."""
    radials = []
    for column in columns:
        offset = np.stack([centroids[c] for c in column]).mean(axis=0) - axis_point
        offset = offset - np.dot(offset, direction) * direction
        radials.append(offset / np.linalg.norm(offset))
    # Azimuths measured from the patch's mean direction, so a patch that
    # straddles +-180 degrees in the deposited frame still sorts in order.
    reference = np.sum(radials, axis=0)
    reference = reference / np.linalg.norm(reference)
    side = np.cross(direction, reference)
    azimuths = [
        math.atan2(float(np.dot(r, side)), float(np.dot(r, reference))) for r in radials
    ]
    return columns[int(np.argsort(azimuths)[len(columns) // 2])]


def _dimer_block_name(source: str) -> str:
    """mmCIF data block name of an extracted dimer, carrying its version."""
    return f"{source}_mtdimer_v{_MT_DIMER_VERSION}"


def _is_current_dimer(path: str, source: str) -> bool:
    """Whether a cached dimer was written by this version of the extraction."""
    with open(path) as fh:
        return fh.readline().strip() == f"data_{_dimer_block_name(source)}"


def _entity_descriptions(cif_path: str) -> dict[str, str]:
    """Map entity id -> description, for telling alpha from beta tubulin."""
    block = gemmi.cif.read(cif_path).sole_block()
    out = {}
    for row in block.find("_entity.", ["id", "pdbx_description"]):
        out[row.str(0)] = row.str(1).lower()
    return out


def _chain_entities(st: gemmi.Structure, cif_path: str) -> dict[str, str]:
    """Map auth chain name -> lowercased entity description."""
    descriptions = _entity_descriptions(cif_path)
    subchain_to_entity = {
        sub: entity.name for entity in st.entities for sub in entity.subchains
    }
    out = {}
    for chain in st[0]:
        if not len(chain):
            continue
        entity_id = subchain_to_entity.get(chain[0].subchain)
        if entity_id is not None:
            out[chain.name] = descriptions.get(entity_id, "")
    return out


def _fit_tube_axis(
    centroids: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """
    Fit the microtubule axis from a patch of protofilaments.

    Returns
    -------
    direction : np.ndarray
        Unit vector along the tube axis, shape ``(3,)``.
    point : np.ndarray
        A point on the axis, shape ``(3,)``.
    columns : list of list of str
        Chain names grouped into protofilament columns.
    """
    names = list(centroids)
    coords = np.stack([centroids[n] for n in names])

    # Group chains into protofilament columns. Do it in 3-D first with a
    # generous radius, then refine: monomers of one protofilament are
    # collinear, so a column's own spread defines the axis direction.
    columns: list[list[str]] = []
    seeds: list[np.ndarray] = []
    for name, centroid in zip(names, coords):
        for column, seed in zip(columns, seeds):
            offset = centroid - seed
            # Distance from the seed measured perpendicular to the
            # column's own extent is not yet known, so use the fact that
            # within a column consecutive monomers are ~41 A apart while
            # columns are ~53 A apart in a fixed direction: cluster on the
            # two coordinates that vary least within a column.
            if np.linalg.norm(offset) < _COLUMN_RADIUS:
                column.append(name)
                break
        else:
            columns.append([name])
            seeds.append(centroid)
            continue
        # Re-seed with the running mean so a long column keeps growing.
        seeds[columns.index(column)] = np.stack([centroids[c] for c in column]).mean(
            axis=0
        )

    # The clustering above only merges near-coincident chains, so refine:
    # direction = dominant direction of within-structure nearest-neighbour
    # offsets at the monomer rise (~41 A).
    offsets = []
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            d = b - a
            if 30.0 < np.linalg.norm(d) < 50.0:
                offsets.append(d if d[np.argmax(np.abs(d))] > 0 else -d)
    if not offsets:
        raise ValueError(
            "could not identify a protofilament direction: no chain pairs "
            "separated by one monomer rise"
        )
    direction = np.mean(offsets, axis=0)
    direction = direction / np.linalg.norm(direction)

    # Now regroup into true columns: chains whose offsets are parallel to
    # the axis belong to the same protofilament.
    columns = []
    for name, centroid in zip(names, coords):
        for column in columns:
            ref = centroids[column[0]]
            perp = (centroid - ref) - np.dot(centroid - ref, direction) * direction
            if np.linalg.norm(perp) < _COLUMN_RADIUS:
                column.append(name)
                break
        else:
            columns.append([name])

    if len(columns) < 3:
        raise ValueError(
            f"need >= 3 protofilaments to fit the tube axis, found {len(columns)}"
        )

    # Circumcircle of three column positions, in the plane perpendicular to
    # the axis, gives the axis' position.
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(e1, direction)) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 = e1 - np.dot(e1, direction) * direction
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)

    column_centres = [
        np.stack([centroids[c] for c in column]).mean(axis=0) for column in columns
    ]
    flat = np.array([[np.dot(c, e1), np.dot(c, e2)] for c in column_centres])
    (x1, y1), (x2, y2), (x3, y3) = flat[:3]
    denom = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(denom) < 1e-6:
        raise ValueError("protofilament columns are collinear; cannot fit tube axis")
    ux = (
        (x1**2 + y1**2) * (y2 - y3)
        + (x2**2 + y2**2) * (y3 - y1)
        + (x3**2 + y3**2) * (y1 - y2)
    ) / denom
    uy = (
        (x1**2 + y1**2) * (x3 - x2)
        + (x2**2 + y2**2) * (x1 - x3)
        + (x3**2 + y3**2) * (x2 - x1)
    ) / denom
    point = ux * e1 + uy * e2
    return direction, point, columns


def extract_mt_dimer(
    source: str = MT_DIMER_SOURCE,
    pdb_cache_dir: str | os.PathLike[str] | None = None,
    verbose: bool = False,
    overwrite: bool = False,
) -> str:
    """
    Extract one alpha-beta tubulin dimer in the microtubule frame.

    Parameters
    ----------
    source : str, optional
        PDB ID of a microtubule reconstruction whose asymmetric unit holds
        at least three neighbouring protofilaments. Default
        `MT_DIMER_SOURCE`.
    pdb_cache_dir : str or path-like, optional
        Where the source structure and the extracted dimer are cached.
        Default None: specter's usual PDB cache directory.
    verbose : bool, optional
        Print fetch/extraction detail. Default False.
    overwrite : bool, optional
        Re-extract even if the cached dimer exists. Default False.

    Returns
    -------
    str
        Path to an mmCIF holding the two chains of one dimer, rotated so
        ``+Z`` is the protofilament axis, pointing alpha -> beta toward the
        plus end, and ``+X`` points radially outwards from the tube axis,
        with the origin midway between the two monomers' CA centroids.

    Raises
    ------
    ValueError
        If the source has too few protofilaments to fit a tube axis, or no
        alpha in the chosen protofilament has its nucleotide capped by a
        beta one monomer rise away.

    Notes
    -----
    An alpha's partner is the beta in contact with the alpha's own
    nucleotide (the N-site, inside the dimer), not merely a beta one
    monomer rise away: the beta on the alpha's other side is equally far
    and belongs to the neighbouring dimer. See the module docstring.
    """
    if pdb_cache_dir is None:
        from ...config import default_pdb_cache_dir

        pdb_cache_dir = default_pdb_cache_dir()
    pdb_cache_dir = str(pdb_cache_dir)

    out_path = os.path.join(pdb_cache_dir, f"{source}-mtdimer.cif")
    if (
        os.path.exists(out_path)
        and not overwrite
        and _is_current_dimer(out_path, source)
    ):
        return out_path

    cif_path = PDB.fetch_pdb_file(
        source, ext="cif", pdb_cache_dir=pdb_cache_dir, assembly=False, verbose=verbose
    )
    st = gemmi.read_structure(cif_path)
    st.setup_entities()

    entities = _chain_entities(st, cif_path)
    centroids: dict[str, np.ndarray] = {}
    for chain in st[0]:
        description = entities.get(chain.name, "")
        if "tubulin" not in description:
            continue  # drops EB3 and any other bound factor
        centroid = _ca_centroid(chain)
        if centroid is not None:
            centroids[chain.name] = centroid

    if len(centroids) < 4:
        raise ValueError(
            f"{source}: found {len(centroids)} tubulin chains, too few to "
            "identify a protofilament"
        )

    direction, axis_point, columns = _fit_tube_axis(centroids)

    # Take the dimer from the column in the middle of the patch, the one
    # furthest from its edges.
    column = _central_column(columns, centroids, direction, axis_point)
    alphas = [c for c in column if "alpha" in entities[c]]
    betas = [c for c in column if "beta" in entities[c]]
    if not alphas or not betas:
        raise ValueError(
            f"{source}: column {column} has no alpha/beta pair "
            f"({[entities[c] for c in column]})"
        )

    # Pair an alpha with the beta capping its nucleotide: that interface is
    # the one inside the dimer (see the module docstring). Distance along
    # the axis cannot decide it, since the beta of the previous dimer is
    # one monomer rise away on the alpha's other side.
    pair = None
    for a in alphas:
        site = _nucleotide_xyz(st[0][a])
        if site is None:
            continue
        contact = {
            b: float(
                np.linalg.norm(
                    _polymer_xyz(st[0][b])[:, None] - site[None], axis=-1
                ).min()
            )
            for b in betas
        }
        b = min(contact, key=contact.__getitem__)
        rise = abs(float(np.dot(centroids[b] - centroids[a], direction)))
        if contact[b] <= _N_SITE_CONTACT and 30.0 < rise < 50.0:
            pair = (a, b)
            break
    if pair is None:
        raise ValueError(
            f"{source}: no alpha in column {column} has its nucleotide capped "
            "by a beta one monomer rise away"
        )
    alpha_name, beta_name = pair

    # alpha -> beta is the plus-end direction, and fixes the sign of +Z.
    plus = direction * np.sign(
        np.dot(centroids[beta_name] - centroids[alpha_name], direction)
    )

    dimer_centre = np.stack([centroids[alpha_name], centroids[beta_name]]).mean(axis=0)

    # Radial direction: from the tube axis out to the dimer, measured
    # perpendicular to the axis.
    radial = dimer_centre - axis_point
    radial = radial - np.dot(radial, direction) * direction
    radius = float(np.linalg.norm(radial))
    radial = radial / radius
    tangential = np.cross(plus, radial)

    # Rows are the new basis expressed in the deposited frame, so
    # `coords @ basis.T` re-expresses coordinates in the microtubule frame.
    basis = np.stack([radial, tangential, plus])

    out = gemmi.Structure()
    out.name = _dimer_block_name(source)
    model = gemmi.Model(1)
    for name in (alpha_name, beta_name):
        model.add_chain(st[0][name].clone())
    out.add_model(model)
    for chain in out[0]:
        for res in chain:
            for atom in res:
                local = np.array([atom.pos.x, atom.pos.y, atom.pos.z]) - dimer_centre
                x, y, z = basis @ local
                atom.pos = gemmi.Position(float(x), float(y), float(z))
    out.setup_entities()

    os.makedirs(pdb_cache_dir, exist_ok=True)
    out.make_mmcif_document().write_file(out_path)

    if verbose:
        logger.info(
            f"{source}: dimer {alpha_name}(alpha)+{beta_name}(beta) from a "
            f"{len(columns)}-protofilament patch; fitted radius "
            f"{radius:.1f} A, plus end along "
            f"[{plus[0]:+.3f}, {plus[1]:+.3f}, {plus[2]:+.3f}] "
            f"-> {out_path}"
        )
    return out_path


def measure_source_lattice(
    source: str = MT_DIMER_SOURCE,
    pdb_cache_dir: str | os.PathLike[str] | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Measure lattice parameters directly off a deposited microtubule model.

    Used to check `_lattice`'s constants against the structures they were
    derived from -- see ``tests/test_microtubule_lattice.py``.

    Parameters
    ----------
    source : str, optional
        PDB ID of a microtubule reconstruction. Default `MT_DIMER_SOURCE`.
    pdb_cache_dir : str or path-like, optional
        PDB cache directory. Default None: specter's usual cache.
    verbose : bool, optional
        Print fetch detail. Default False.

    Returns
    -------
    dict
        ``radius``, ``lateral_spacing``, ``monomer_rise`` and
        ``n_protofilaments`` (inferred from the azimuthal step between
        adjacent protofilaments), all in Å where applicable.
        ``monomer_rise`` is half the dimer repeat, measured between
        monomers two apart along a protofilament: consecutive monomers are
        not evenly spaced (in 6DPU the intra-dimer spacing is 41.4 A and the
        inter-dimer one 42.6 A), so their median spacing would report the
        shorter one rather than the lattice's rise.
    """
    if pdb_cache_dir is None:
        from ...config import default_pdb_cache_dir

        pdb_cache_dir = default_pdb_cache_dir()
    cif_path = PDB.fetch_pdb_file(
        source,
        ext="cif",
        pdb_cache_dir=str(pdb_cache_dir),
        assembly=False,
        verbose=verbose,
    )
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    entities = _chain_entities(st, cif_path)
    centroids = {}
    for chain in st[0]:
        if "tubulin" not in entities.get(chain.name, ""):
            continue
        centroid = _ca_centroid(chain)
        if centroid is not None:
            centroids[chain.name] = centroid

    direction, axis_point, columns = _fit_tube_axis(centroids)
    column_centres = [
        np.stack([centroids[c] for c in column]).mean(axis=0) for column in columns
    ]

    e1 = column_centres[0] - axis_point
    e1 = e1 - np.dot(e1, direction) * direction
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)

    radii, angles = [], []
    for centre in column_centres:
        offset = centre - axis_point
        offset = offset - np.dot(offset, direction) * direction
        radii.append(float(np.linalg.norm(offset)))
        angles.append(math.degrees(math.atan2(np.dot(offset, e2), np.dot(offset, e1))))
    radius = float(np.mean(radii))
    d_phi = float(np.median(np.diff(sorted(angles))))

    rises = []
    for column in columns:
        along = np.sort([float(np.dot(centroids[c], direction)) for c in column])
        rises += list((along[2:] - along[:-2]) / 2)
    monomer_rise = float(np.median(rises)) if rises else float("nan")

    return {
        "radius": radius,
        "lateral_spacing": radius * math.radians(d_phi),
        "monomer_rise": monomer_rise,
        "n_protofilaments": 360.0 / d_phi,
    }
