import os
from gzip import GzipFile
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np


def get_atom_symbols():
    """
    Return list of atom symbols.

    Returns
    -------
    names : ndarray
        Atom symbols in the periodic table.
    """
    elem = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "./atom-data", "atom_mass.txt"),
        delimiter=" ",
        dtype="str",
        usecols=0,
    )
    return np.array([e.upper() for e in elem])


def get_atom_masses():
    """
    Return list of atom masses.

    Returns
    -------
    masses : ndarray
        Atom masses
    """
    masses = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "./atom-data", "atom_mass.txt"),
        delimiter=" ",
        dtype="float",
        usecols=1,
    )
    return masses


def atom_symbol(atomic_number):
    """
    Return atomic symbol.

    Parameters
    ----------
    atomic_number : ndarray
        List of atomic numbers. Hydrogen is 1.

    Returns
    -------
    atom_symbols : str
        Atomic symbol
    """
    atom_symbols = get_atom_symbols()
    return atom_symbols[atomic_number - 1]


def atom_number(symbol):
    """
    Return atomic number.

    Parameters
    ----------
    symbol : str
        Atom symbol, e.g. H,C,N,...

    Returns
    -------
    atom_numbers : int
        Atomic numbers, Hydrogen has number 1.
    """

    atom_symbols = get_atom_symbols()
    return np.where(atom_symbols == symbol.upper())[0][0] + 1


def center_of_particle(coords):
    """
    Return a particle's geometric center.

    Parameters
    ----------
    coords : ndarray
        Atom coordinates of molecule with N atoms, shape (N,3)

    Returns
    -------
    center : ndarray
        Atom coordinates of the molecule's geometric, shape (1,3)
    """
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    zmin, zmax = coords[:, 2].min(), coords[:, 2].max()
    center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0])
    return center


def fetch_pdb_file(pdbcode, output="./", force=False, assembly=False):
    """
    Download a PDB file and save it in a given location.

    Parameters
    ----------
    pdbcode : str
        A valid PDB code
    output : str, optional
        The destination for the PDB file to be saved
    force : bool, optional
        Download PDB file even if it already exists
    assembly : bool, optional
        Download biological assembly

    Returns
    -------
    str
        Path to the saved PDB file
    """
    pdbcode = pdbcode.upper()
    url = "https://files.rcsb.org/download/{code}.pdb{assembly}.gz".format(
        code=pdbcode, assembly="1" if assembly else ""
    )
    filename = Path(output) / "{}.pdb".format(
        pdbcode
    )  # "".join([output,'/',pdbcode, '.pdb'])
    if (os.path.isfile(filename)) and not force:
        return str(filename)
    try:
        response = urlopen(url)
    except HTTPError:
        raise IOError("Error 404: {url} not found".format(url=url))
    compressed = BytesIO()
    compressed.write(response.read())
    compressed.seek(0)
    decompressed = GzipFile(fileobj=compressed, mode="rb")
    with open(filename, "wb") as f:
        f.write(decompressed.read())
    return str(filename)


def get_atoms_and_coordinates_from_pdb(
    filename, residual=False, multimodel=False, assemble=True
):
    """
    Parse a PDB file and return atom labes and coordinates.

    Parameters
    ----------
    filename : str
        Path to PDB file
    residual : bool, optional
        Counts residual atoms, default is False
    multimodel : bool, optional
        Parse multiple models (sometimes biological assemblies are saved as models), default is False
    assemble : bool, optional
        Apply symmetry operations and return biological assembly, default is True

    Returns
    -------
    elements : ndarray
        Atom elements listed in the PDB file
    coords : ndarray
        Atom coordinates listed in the PDB file
    rescount : int
        Return count of residual atoms if residual=True
    """

    legal_atoms = get_atom_symbols()
    elements, coords = [], []
    rescount = 0
    with open(filename) as f:
        for line in f:
            if (line[:6] == "ENDMDL") and not multimodel:
                break
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_label = line[76:78].lstrip().upper()
                (occ, tag) = (float(line[56:60]), line[16])
                use_atom = (occ > 0.5) | ((occ == 0.5) & (tag.upper() == "A"))
                if use_atom and (atom_label in legal_atoms):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    elements.append(np.where(legal_atoms == atom_label)[0][0] + 1)
                    coords.append([x, y, z])
                else:
                    rescount += 1
    elements = np.array(elements)
    coords = np.array(coords)
    if assemble:
        symmetry, trans = get_symmetry_from_pdb(filename)
        elements, coords = get_biological_assembly(elements, coords, symmetry, trans)
    out = (elements, coords)
    if residual:
        out += (rescount,)
    return out


def get_symmetry_from_pdb(filename):
    """
    Parse symmetry operators from a PDB file.

    Parameters
    ----------
    filename : str
        Path to PDB file

    Returns
    -------
    symmetry : array_like
        Symmetry matrix
    trans : array_like
        Translation vector
    """
    symmetry, trans = [], []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line[13:18] == "BIOMT":
                symmetry.append(
                    [float(line[24:33]), float(line[34:43]), float(line[44:53])]
                )
                trans.append(float(line[58:68]))
    if not len(symmetry):
        symmetry.append(np.diag([1.0, 1.0, 1.0]))
        trans.append([0.0, 0.0, 0.0])
    symmetry = np.asarray(symmetry).reshape(-1, 3, 3)
    trans = np.asarray(trans).reshape(-1, 3)
    return symmetry, trans


def get_biological_assembly(elements, coords, symmetry, translation):
    """
    Apply symmetry/translation operations and return assembled protein.

    Parameters
    ----------
    elements : ndarray
         Atom elements of assymetric unit
    coords : ndarray
         Atom coordinates of assymetric unit
    symmetry : ndarray
         Symmtery matrices
    translation : ndarray
         Translation vectors

    Returns
    -------
    elements : ndarray
        Atom elements of biological assembly
    coords : ndarray
        Atom coordinates of biological assembly
    """
    elements_assembled, coords_assembled = [], []
    for i in range(symmetry.shape[0]):
        elements_assembled.append(elements)
        s = symmetry[i]
        t = translation[i]
        v = s.dot(coords.T).T + t
        coords_assembled.append(v)
    return np.hstack(elements_assembled), np.vstack(coords_assembled)