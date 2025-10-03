import os
from gzip import GzipFile
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from . import atom
import torch
import requests
import gzip
import io

import biotite.structure.io as strucio
import biotite.structure.io.pdbx as pdbx
import numpy as np

from tqdm import tqdm

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning  # correct import
import warnings

# Suppress only PDBConstructionWarnings
warnings.simplefilter('ignore', PDBConstructionWarning)

def get_available_assemblies(pdb_id):
    """Return a list of available biological assembly IDs for a PDB entry."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        assemblies = data.get("rcsb_entry_container_identifiers", {}).get("assembly_ids", [])
        print("Assemblies available: " + ", ".join(assemblies))
    except Exception as e:
        print(f"Error fetching assemblies for {pdb_id}: {e}")
        return []

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
    center = coords.mean(dim=0)
    return center

def center_coordinates(coords):
    """
    Centers coordinates on its geometric center.

    Parameters
    ----------
    coords : tensor
        Atom coordinates of molecule with N atoms, shape (N,3)

    Returns
    -------
    centered_coordinates : tensor
        Centered coordinates, shape (N,3)
    """
    center = center_of_particle(coords)
    centered_coordinates = coords - center.reshape(1, -1)
    return centered_coordinates


def fetch_pdb_file(pdb_id, format='cif', output="../pdb-data/", assembly=True):
    """
    Download a PDB file and save it in a given location.

    Parameters
    ----------
    pdb_id : str
        A valid PDB ID.
    format : str
        File format ('cif' or 'pdb').
    output : str
        Destination folder.
    assembly : bool or int
        - True  → fetch default biological assembly (assembly 1 if available).
        - False → fetch asymmetric unit.
        - int   → fetch that specific assembly if available, fallback to default 
           PDBx/mmCIF file.

    Returns
    -------
    str
        Path to the saved PDB file
    """
    get_available_assemblies(pdb_id)

    # Decide what to fetch
    if assembly is True:
        print(f"{pdb_id}: Fetching default Biological Assembly 1")
        filename = f"{pdb_id}-assembly{1}.{format}"
    elif assembly is False:
        print(f"{pdb_id}: Fetching default PDBx/mmCIF file.")
        filename = f"{pdb_id}.{format}"
    elif isinstance(assembly, int):
        print(f"{pdb_id}: Fetching Biological Assembly {assembly}")
        filename = f"{pdb_id}-assembly{assembly}.{format}"
    else:
        raise ValueError("assembly must be True, False, or int")

    # Build filepath
    file_path = os.path.join(output, filename)

    # Return existing file if available
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}, skip fetching.")
        return file_path
    else:
        # Fetch
        url = "https://files.rcsb.org/download/" + filename + ".gz"
        r = requests.get(url)
        r.raise_for_status()

        # Decompress in memory
        with gzip.open(io.BytesIO(r.content), "rt") as f:
            cif_content = f.read()

        # Save to file
        with open(file_path, "w") as f:
            f.write(cif_content)

    print(f"Downloaded to: {file_path}")
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

# def get_atoms_and_coordinates_from_pdb(
#     input_filename,
#     fov=(100, 100, 100),
#     debye_factor=0.08,
#     comment="",
#     return_array=True,
#     write_file=False,
#     output_filename=None,
#     assemble=True,
#     **kwargs,
# ):
#     """
#     Writes an XYZ file formatted for Kirkland's multi-slice simulation program.

#     software.

#     Parameters
#     ----------
#     input_filename : str
#         The name of the input file containing the atomic structure.
#     fov : tuple
#         Field of view dimensions (default is (100, 100, 100)).
#     debye_factor : float
#         Debye-Waller factor (default is 0.08).
#     comment : str
#         A comment to include in the header of the output file.
#     return_array : boolean
#         If True, returns array for elements and coordinates.
#     write_file : boolean
#         If True, writes .txt file.
#     output_filename : str
#         The name of the output file to write the XYZ data.
#     assemble : boolean
#         If True, assembles biological unit using symmetries in pdb file. Else,
#         returns assymetric subunit.

#     Returns
#     ----------
#     elements : (N,)-shape array
#     coords : (N,3)-shape array
#         x,y,z coordinates of each atom
#     """

#     pdbx_file = pdbx.CIFFile.read(input_filename)
#     if assemble:
#         biological_unit = pdbx.get_assembly(pdbx_file, **kwargs)
#     else:
#         biological_unit = pdbx.get_structure(pdbx_file, **kwargs)
#     n_atoms = len(biological_unit.element)
#     coords = np.squeeze(biological_unit.coord)  # [x, y, z]
#     elements = np.zeros(n_atoms, dtype=int)
#     for i in range(n_atoms):
#         # Read the Atomic Number
#         elements[i] = get_atomic_number(biological_unit.element[i])

#     if write_file:
#         if output_filename is None:
#             output_filename = input_filename + ".txt"
#         with open(output_filename, "w") as file:
#             # Write the header
#             file.write(comment + "\n")
#             file.write(f"{fov[0]:.2f}\t{fov[1]:.2f}\t{fov[2]:.2f}\n")

#             # Write the coordinates and other details
#             for elem, coord in zip(elements, coords):
#                 file.write(
#                     f"{int(elem)}\t{coord[0]:<8.4f}\t{coord[1]:<8.4f}\t{coord[2]:<8.4f}\t 1.0\t{debye_factor}\n"
#                 )

#             # Write the comment or end line
#             file.write("-1")
#     if return_array:
#         return torch.from_numpy(elements), torch.from_numpy(coords)


def get_atoms_and_coordinates_from_pdb(input_filename, return_array=True):
    """
    Writes an XYZ file formatted for Kirkland's multi-slice simulation program.

    Parameters
    ----------
    input_filename : str
        The name of the input file containing the atomic structure.
    return_array : boolean
        If True, returns array for elements and coordinates.

    Returns
    ----------
    elements : (N,)-shape array
    coords : (N,3)-shape array
        x,y,z coordinates of each atom
    """

    if input_filename[-3:] == 'pdb':
        parser = PDBParser()
    elif input_filename[-3:] == 'cif':
        parser = MMCIFParser()
    print(f'Parsing {input_filename}.')
    structure = parser.get_structure('structure', input_filename)

    # Extract atomic data
    coords = []
    elements = []
    for atom in tqdm(structure.get_atoms(), desc="Extracting atom coordinates and element"):
        element_symbol = atom.element.strip().upper()
        atomic_number = get_atomic_number(element_symbol)
        elements.append(atomic_number)
        
        if atomic_number is not None:
            coord = atom.get_coord()
            coords.append(coord)
    n_atoms = len(coords)
    coords = np.array(coords)
    elements = np.array(elements)

    if return_array:
        return torch.from_numpy(elements), torch.from_numpy(coords)


def write_xyz_file(input_filename, output_filename, comment=""):
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

    with open(output_filename, "w") as file:
        # Write the header
        nAtoms = inXYZ.shape[0]
        file.write(f"{int(nAtoms)}\n")
        file.write(comment + "\n")

        # Write the coordinates and other details
        for ii in range(nAtoms):
            if ii < nAtoms - 1:
                suffix = "\n"
            else:
                suffix = ""
            file.write(
                f"{inXYZ[ii].element}\t{inXYZ[ii].coord[0]:<8.4f}\t{inXYZ[ii].coord[1]:<8.4f}\t{inXYZ[ii].coord[2]:<8.4f}{suffix}"
            )