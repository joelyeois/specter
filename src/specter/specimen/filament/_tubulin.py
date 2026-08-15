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

The extracted dimer is re-expressed so that

* ``+Z`` is the protofilament axis, pointing alpha -> beta (the microtubule
  plus-end direction), and
* ``+X`` points radially outwards, away from the tube axis,

then cached next to the downloaded structure so the (small) extraction runs
once per source.
"""

from __future__ import annotations

import math
import os

import gemmi
import numpy as np

from ...pdb import PDB

#: Default source structure: a 13-protofilament GMPCPP-microtubule solved
#: with EB3 bound (Zhang et al. 2015). Its deposited asymmetric unit is a
#: 3-protofilament x 4-monomer patch -- enough neighbouring protofilaments
#: to fit the tube axis, which a single isolated dimer would not give.
#: The two 131-residue EB3 chains are dropped during extraction.
#:
#: Chosen over the more common kinesin-decorated microtubule structures
#: (6DPU, 3JAT) because those are 14-protofilament lattices, so their dimer
#: sits at a different radius from the 13-protofilament default.
MT_DIMER_SOURCE = "3JAL"

_MIN_TUBULIN_RESIDUES = 380
_MAX_TUBULIN_RESIDUES = 480
#: Protofilament columns are ~53 A apart, monomers within a column ~41 A;
#: this groups chains into columns without catching the neighbour.
_COLUMN_RADIUS = 25.0


def _ca_centroid(chain: gemmi.Chain) -> np.ndarray | None:
    """Centroid of a chain's CA atoms, or None if it is not a protein chain
    of roughly tubulin size."""
    ca = [[a.pos.x, a.pos.y, a.pos.z] for res in chain for a in res if a.name == "CA"]
    if not _MIN_TUBULIN_RESIDUES <= len(ca) <= _MAX_TUBULIN_RESIDUES:
        return None
    return np.asarray(ca).mean(axis=0)


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
    savefolder: str | os.PathLike[str] | None = None,
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
    savefolder : str or path-like, optional
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
        ``+Z`` is the protofilament axis (alpha -> beta) and ``+X`` points
        radially outwards from the tube axis.

    Raises
    ------
    ValueError
        If the source has too few protofilaments to fit a tube axis, or no
        alpha/beta pair stacked along one protofilament could be found.
    """
    if savefolder is None:
        from ...config import default_pdb_cache_dir

        savefolder = default_pdb_cache_dir()
    savefolder = str(savefolder)

    out_path = os.path.join(savefolder, f"{source}-mtdimer.cif")
    if os.path.exists(out_path) and not overwrite:
        return out_path

    cif_path = PDB.fetch_pdb_file(
        source, ext="cif", savefolder=savefolder, assembly=False, verbose=verbose
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

    # Pick a dimer from the most central column, so the fitted radial
    # direction is least affected by the patch's edges.
    column = sorted(columns, key=len, reverse=True)[len(columns) // 2]
    alphas = [c for c in column if "alpha" in entities[c]]
    betas = [c for c in column if "beta" in entities[c]]
    if not alphas or not betas:
        raise ValueError(
            f"{source}: column {column} has no alpha/beta pair "
            f"({[entities[c] for c in column]})"
        )

    # alpha -> beta must point one monomer rise along the axis; that fixes
    # the sign of +Z (the microtubule plus-end direction).
    pair = None
    for a in alphas:
        for b in betas:
            offset = np.dot(centroids[b] - centroids[a], direction)
            if 30.0 < offset < 50.0:
                pair = (a, b)
                break
        if pair:
            break
    if pair is None:
        raise ValueError(
            f"{source}: no alpha-beta pair one monomer rise apart in column {column}"
        )
    alpha_name, beta_name = pair

    dimer_centre = np.stack([centroids[alpha_name], centroids[beta_name]]).mean(axis=0)

    # Radial direction: from the tube axis out to the dimer, measured
    # perpendicular to the axis.
    radial = dimer_centre - axis_point
    radial = radial - np.dot(radial, direction) * direction
    radius = float(np.linalg.norm(radial))
    radial = radial / radius
    tangential = np.cross(direction, radial)

    # Rows are the new basis expressed in the deposited frame, so
    # `coords @ basis.T` re-expresses coordinates in the microtubule frame.
    basis = np.stack([radial, tangential, direction])

    out = gemmi.Structure()
    out.name = f"{source}_mtdimer"
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

    os.makedirs(savefolder, exist_ok=True)
    out.make_mmcif_document().write_file(out_path)

    if verbose:
        print(
            f"{source}: dimer {alpha_name}(alpha)+{beta_name}(beta) from a "
            f"{len(columns)}-protofilament patch; fitted radius "
            f"{radius:.1f} A, axis "
            f"[{direction[0]:+.3f}, {direction[1]:+.3f}, {direction[2]:+.3f}] "
            f"-> {out_path}"
        )
    return out_path


def measure_source_lattice(
    source: str = MT_DIMER_SOURCE,
    savefolder: str | os.PathLike[str] | None = None,
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
    savefolder : str or path-like, optional
        PDB cache directory. Default None: specter's usual cache.
    verbose : bool, optional
        Print fetch detail. Default False.

    Returns
    -------
    dict
        ``radius``, ``lateral_spacing``, ``monomer_rise`` and
        ``n_protofilaments`` (inferred from the azimuthal step between
        adjacent protofilaments), all in Angstrom where applicable.
    """
    if savefolder is None:
        from ...config import default_pdb_cache_dir

        savefolder = default_pdb_cache_dir()
    cif_path = PDB.fetch_pdb_file(
        source, ext="cif", savefolder=str(savefolder), assembly=False, verbose=verbose
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
        along = sorted(float(np.dot(centroids[c], direction)) for c in column)
        rises += list(np.diff(along))
    monomer_rise = float(np.median(rises)) if rises else float("nan")

    return {
        "radius": radius,
        "lateral_spacing": radius * math.radians(d_phi),
        "monomer_rise": monomer_rise,
        "n_protofilaments": 360.0 / d_phi,
    }
