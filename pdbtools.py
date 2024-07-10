import os
from gzip import GzipFile
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
import atom

import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb
import numpy as np

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

    legal_atoms = atom.get_atom_symbols()
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

def fetch_pdbx_file(pdb_id, output="./"):
    """
    Download a PDB file and save it in a given location.

    Parameters
    ----------
    pdbcode : str
        A valid PDB ID
    output : str, optional
        The destination for the PDB file to be saved

    Returns
    -------
    str
        Path to the saved PDB file
    """

    # Fetch the PDB file and save it locally
    file_path = rcsb.fetch(pdb_id, "pdbx", output)
    print(f"PDB file saved to: {file_path}")

    return file_path

def get_atomic_number(symbol):
    """
    Retrieves the atomic number for a given element symbol.

    Parameters
    ----------
    symbol : str
        The chemical symbol of the element (e.g., 'H', 'O', 'AU').

    Returns
    -------
    str
        The atomic number of the element, or None if the symbol is not found.
    """
    periodic_table_caps = {
        'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10,
        'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18,
        'K': 19, 'CA': 20, 'SC': 21, 'TI': 22, 'V': 23, 'CR': 24, 'MN': 25, 'FE': 26, 'CO': 27,
        'NI': 28, 'CU': 29, 'ZN': 30, 'GA': 31, 'GE': 32, 'AS': 33, 'SE': 34, 'BR': 35, 'KR': 36,
        'RB': 37, 'SR': 38, 'Y': 39, 'ZR': 40, 'NB': 41, 'MO': 42, 'TC': 43, 'RU': 44, 'RH': 45,
        'PD': 46, 'AG': 47, 'CD': 48, 'IN': 49, 'SN': 50, 'SB': 51, 'TE': 52, 'I': 53, 'XE': 54,
        'CS': 55, 'BA': 56, 'LA': 57, 'CE': 58, 'PR': 59, 'ND': 60, 'PM': 61, 'SM': 62, 'EU': 63,
        'GD': 64, 'TB': 65, 'DY': 66, 'HO': 67, 'ER': 68, 'TM': 69, 'YB': 70, 'LU': 71, 'HF': 72,
        'TA': 73, 'W': 74, 'RE': 75, 'OS': 76, 'IR': 77, 'PT': 78, 'AU': 79, 'HG': 80, 'TL': 81,
        'PB': 82, 'BI': 83, 'PO': 84, 'AT': 85, 'RN': 86, 'FR': 87, 'RA': 88, 'AC': 89, 'TH': 90,
        'PA': 91, 'U': 92, 'NP': 93, 'PU': 94, 'AM': 95, 'CM': 96, 'BK': 97, 'CF': 98, 'ES': 99,
        'FM': 100, 'MD': 101, 'NO': 102, 'LR': 103, 'RF': 104, 'DB': 105, 'SG': 106, 'BH': 107,
        'HS': 108, 'MT': 109, 'DS': 110, 'RG': 111, 'CN': 112, 'NH': 113, 'FL': 114, 'MC': 115,
        'LV': 116, 'TS': 117, 'OG': 118 }

    return periodic_table_caps.get(symbol)

def write_kirkland_xyz_file(input_filename, output_filename, fov = (100,100,100), debye_factor = 0.08, comment = ''):
    """
    Writes an XYZ file formatted for Kirkland's multi-slice simulation program.

    software. 

    Parameters
    ----------
    input_filename : str 
        The name of the input file containing the atomic structure.
    output_filename : str 
        The name of the output file to write the XYZ data.
    fov : tuple
        Field of view dimensions (default is (100, 100, 100)).
    debye_factor : float
        Debye-Waller factor (default is 0.08).
    comment : str
        A comment to include in the header of the output file.

    Returns
    ----------
    None
    """

    pdb_array = strucio.load_structure(input_filename)

    xyz = np.zeros([len(pdb_array), 4]) # [Z, x, y, z]
    for ii in range(len(pdb_array)):
        
        # Read the Atomic Number
        xyz[ii,0] = get_atomic_number(pdb_array[ii].element)

        # Pull the Coordinates
        xyz[ii,1:] = pdb_array[ii].coord

    with open(output_filename, 'w') as file:
        # Write the header
        file.write(comment+'\n')
        file.write(f"{fov[0]:.2f}\t{fov[1]:.2f}\t{fov[2]:.2f}\n")
        
        # Write the coordinates and other details
        for atom in xyz:
            file.write(f"{int(atom[0])}\t{atom[1]:<8.4f}\t{atom[2]:<8.4f}\t{atom[3]:<8.4f}\t 1.0\t{debye_factor}\n")
        
        # Write the comment or end line
        file.write('-1')        

def write_xyz_file(input_filename, output_filename, comment = ''):
    """
    Writes a standard XYZ file.

    Parameters
    ----------    
    input_filename : str 
        The name of the input file containing the atomic structure.
    output_filename : str 
        The name of the output file to write the XYZ data.
    comment : str
        A comment to include in the header of the output file.


    Returns
    ----------    
    None    
    """

    inXYZ = strucio.load_structure(input_filename)
    
    with open(output_filename, 'w') as file:
        
        # Write the header
        nAtoms = inXYZ.shape[0]
        file.write(f'{int(nAtoms)}\n')
        file.write(comment+'\n')
        
        # Write the coordinates and other details
        for ii in range(nAtoms):
            if ii < nAtoms - 1: suffix = '\n'
            else:               suffix = ''
            file.write(f"{inXYZ[ii].element}\t{inXYZ[ii].coord[0]:<8.4f}\t{inXYZ[ii].coord[1]:<8.4f}\t{inXYZ[ii].coord[2]:<8.4f}{suffix}")
