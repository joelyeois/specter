import os
from .atom import atom_number
import torch
import requests
import gzip
import io
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

import biotite.structure.io as strucio
import numpy as np

from rich.progress import track

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning  # correct import
import warnings

# Suppress only PDBConstructionWarnings
warnings.simplefilter("ignore", PDBConstructionWarning)


class PDB:
    def __init__(self, pdb_source, assembly=True, savefolder="../pdb-data/"):
        """
        Create a PDB object from either a PDB ID or a local file path.

        Args:
            pdb_source (str): Either a 4-character PDB ID (e.g., "1abc")
                              or a local file path to a PDB/mmCIF structure.
            assembly (bool): Whether to fetch the biological assembly when using a PDB ID.
            savefolder (str): Folder to store downloaded PDB/mmCIF files.
        """

        # Determine whether pdb_source is a PDB ID or file path
        if (
            isinstance(pdb_source, str)
            and len(pdb_source) == 4
            and pdb_source.isalnum()
        ):
            # Treat as PDB ID
            self.pdb_id = pdb_source
            self.filepath = PDB.fetch_pdb_file(
                pdb_source, savefolder=savefolder, assembly=assembly
            )
            self.assembly = assembly
            self.savefolder = savefolder
        elif os.path.isfile(pdb_source):
            self.filepath = pdb_source
        else:
            raise ValueError(
                f"Invalid pdb_source: '{pdb_source}'. Must be a 4-character PDB ID or a valid file path."
            )

        # get pdb structure
        self.structure = PDB.get_pdb_structure(self.filepath)

        # get atomic elements and coordinates
        self.atomic_numbers, self.coordinates = PDB.get_atoms_and_coordinates(
            self.structure
        )

        # center coordinates
        self.coordinates = PDB.center_coordinates(self.coordinates)

        # estimate max diameter
        self.max_diameter = PDB.estimate_max_diameter(self.coordinates)

    @staticmethod
    def fetch_pdb_file(pdb_id, ext="cif", savefolder="../pdb-data/", assembly=True):
        """
        Download a PDB file and save it in a given location.

        Parameters
        ----------
        pdb_id : str
            A valid PDB ID.
        ext : str
            File ext ('cif' or 'pdb').
        savefolder : str
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
        PDB.get_available_assemblies(pdb_id)

        # Decide what to fetch
        if assembly is True:
            print(f"{pdb_id}: Fetching default Biological Assembly 1")
            filename = f"{pdb_id}-assembly{1}.{ext}"
        elif assembly is False:
            print(f"{pdb_id}: Fetching default PDBx/mmCIF file.")
            filename = f"{pdb_id}.{ext}"
        elif isinstance(assembly, int):
            print(f"{pdb_id}: Fetching Biological Assembly {assembly}")
            filename = f"{pdb_id}-assembly{assembly}.{ext}"
        else:
            raise ValueError("assembly must be True, False, or int")

        # Build filepath
        file_path = os.path.join(savefolder, filename)

        # Return existing file if available
        if os.path.exists(file_path):
            print(f"File already exists: {file_path}, skip fetching.")
            return file_path
        else:
            # Fetch
            print("File does not exists, fetching.")
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

    @staticmethod
    def get_available_assemblies(pdb_id):
        """Return a list of available biological assembly IDs for a PDB entry."""
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            assemblies = data.get("rcsb_entry_container_identifiers", {}).get(
                "assembly_ids", []
            )
            print("Assemblies available: " + ", ".join(assemblies))
        except Exception as e:
            print(f"Error fetching assemblies for {pdb_id}: {e}")
            return []

    @staticmethod
    def get_pdb_structure(filepath):
        ext = filepath[-3:]
        if ext == "pdb":
            parser = PDBParser()
        elif ext == "cif":
            parser = MMCIFParser()
        else:
            raise ValueError(f"Invalid file format '{ext}'. Must be 'cif' or 'pdb'.")
        structure = parser.get_structure("structure", filepath)
        return structure

    @staticmethod
    def get_atoms_and_coordinates(structure):
        """
        Extracts atomic elements and coordinates from PDB structure.

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

        # if filepath, load the structure.
        if isinstance(structure, str):
            structure = PDB.get_pdb_structure(structure)

        # Extract atomic data
        coords = []
        elements = []
        for atom in track(
            structure.get_atoms(), description="Extracting atom coordinates and element"
        ):
            element_symbol = atom.element.strip().upper()
            elements.append(element_symbol)

            coord = atom.get_coord()
            coords.append(coord)
        coords = torch.as_tensor(np.array(coords))
        elements = atom_number(elements)

        return elements, coords

    @staticmethod
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

    @staticmethod
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
        center = PDB.center_of_particle(coords)
        centered_coordinates = coords - center.reshape(1, -1)
        return centered_coordinates

    @staticmethod
    def estimate_max_diameter(coordinates):
        hull = ConvexHull(coordinates)
        hull_points = coordinates[hull.vertices]
        max_diameter = pdist(hull_points).max()
        return max_diameter


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
