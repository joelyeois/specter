from __future__ import annotations

import gzip
import io
import os
import warnings
from typing import TYPE_CHECKING

import biotite.structure.io as strucio
import numpy as np
import requests
import torch
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning  # correct import
from .progress import track
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure

from .atom import atom_number

# Suppress only PDBConstructionWarnings
warnings.simplefilter("ignore", PDBConstructionWarning)


class PDB:
    def __init__(
        self,
        pdb_source: str,
        assembly: bool = True,
        savefolder: str = "../pdb-data/",
        origin: tuple[float, float, float] | None = None,
    ) -> None:
        """
        Create a PDB object from either a PDB ID or a local file path.

        Parameters
        ----------
        pdb_source : str
            Either a 4-character PDB ID (e.g., '1abc') or a local file path
            to a PDB/mmCIF structure file.
        assembly : bool, optional
            Whether to fetch the biological assembly when using a PDB ID.
            Default is True.
        savefolder : str, optional
            Folder to store downloaded PDB/mmCIF files. Default is '../pdb-data/'.
        origin : tuple[float, float, float] or None, optional
            Custom origin to subtract from coordinates. If None, coordinates
            are centered on their geometric center. If a tuple, that point is
            subtracted directly without auto-centering.

        Attributes
        ----------
        pdb_id : str
            The PDB ID if pdb_source is a PDB ID.
        filepath : str
            Path to the PDB/mmCIF file.
        structure : Bio.PDB.Structure.Structure
            Parsed PDB structure object.
        atomic_numbers : torch.Tensor
            Atomic numbers of all atoms in the structure, shape (N,).
        coordinates : torch.Tensor
            Coordinates shifted by origin, shape (N, 3).
        max_diameter : float
            Maximum diameter of the structure based on convex hull.
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
        if origin is None:
            self.coordinates = PDB.center_coordinates(self.coordinates)
        else:
            self.coordinates = self.coordinates - torch.tensor(
                origin, dtype=self.coordinates.dtype
            )

        # estimate max diameter
        self.max_diameter = PDB.estimate_max_diameter(self.coordinates)

    @staticmethod
    def fetch_pdb_file(
        pdb_id: str,
        ext: str = "cif",
        savefolder: str = "../pdb-data/",
        assembly: bool | int = True,
    ) -> str:
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

        # Fetch
        print("File does not exist, fetching.")
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
    def get_available_assemblies(pdb_id: str) -> None:
        """
        Print the available biological assembly IDs for a PDB entry.

        Parameters
        ----------
        pdb_id : str
            4-character PDB ID.

        Notes
        -----
        Prints the available assemblies to console. If the PDB entry cannot be
        accessed or does not have assembly information, prints an error message.
        """
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
            return

    @staticmethod
    def get_pdb_structure(filepath: str) -> Structure:
        """
        Parse a PDB or mmCIF file and return the structure object.

        Parameters
        ----------
        filepath : str
            Path to PDB (.pdb) or mmCIF (.cif) file.

        Returns
        -------
        structure : Bio.PDB.Structure.Structure
            Parsed structure object from Biopython.

        Raises
        ------
        ValueError
            If the file format is not 'pdb' or 'cif'.
        """
        ext = filepath[-3:]
        parser: PDBParser | MMCIFParser
        if ext == "pdb":
            parser = PDBParser()
        elif ext == "cif":
            parser = MMCIFParser()
        else:
            raise ValueError(f"Invalid file format '{ext}'. Must be 'cif' or 'pdb'.")
        structure = parser.get_structure("structure", filepath)
        return structure

    @staticmethod
    def get_atoms_and_coordinates(
        structure: Structure | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract atomic elements and coordinates from PDB structure.

        Parameters
        ----------
        structure : Bio.PDB.Structure.Structure or str
            Either a parsed Biopython structure object or a filepath to a
            PDB/mmCIF file. If a filepath is provided, the structure will be
            loaded automatically.

        Returns
        -------
        elements : torch.Tensor
            Atomic numbers for each atom, shape (N,).
        coords : torch.Tensor
            Atomic coordinates (x, y, z) for each atom, shape (N, 3).

        Notes
        -----
        Uses Rich progress bar to display extraction progress for large structures.
        """

        # if filepath, load the structure.
        if isinstance(structure, str):
            structure = PDB.get_pdb_structure(structure)

        # Extract atomic data
        coords_list = []
        element_symbols = []
        for atom in track(
            structure.get_atoms(), description="Extracting atom coordinates and element"
        ):
            element_symbol = atom.element.strip().upper()
            element_symbols.append(element_symbol)

            coord = atom.get_coord()
            coords_list.append(coord)
        coords = torch.as_tensor(np.array(coords_list))
        elements = atom_number(element_symbols)

        return elements, coords

    @staticmethod
    def center_of_particle(coords: torch.Tensor) -> torch.Tensor:
        """
        Return a particle's geometric center.

        Parameters
        ----------
        coords : torch.Tensor
            Atom coordinates of molecule with N atoms, shape (N, 3).

        Returns
        -------
        center : torch.Tensor
            Geometric center of the molecule, shape (3,).
        """
        center = coords.mean(dim=0)
        return center

    @staticmethod
    def center_coordinates(coords: torch.Tensor) -> torch.Tensor:
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
        return coords - center

    @staticmethod
    def estimate_max_diameter(coordinates: torch.Tensor) -> float:
        """
        Estimate the maximum diameter of a structure using convex hull.

        Parameters
        ----------
        coordinates : torch.Tensor
            Atomic coordinates with shape (N, 3).

        Returns
        -------
        max_diameter : float
            Maximum pairwise distance between convex hull vertices.

        Notes
        -----
        Computes the convex hull of the coordinates and returns the maximum
        distance between any two hull vertices.
        """
        hull = ConvexHull(coordinates)
        hull_points = coordinates[hull.vertices]
        max_diameter = pdist(hull_points).max()
        return max_diameter


def write_xyz_file(
    input_filename: str, output_filename: str, comment: str = ""
) -> None:
    """
    Write atomic structure to standard XYZ file format.

    Parameters
    ----------
    input_filename : str
        Path to input structure file (PDB, mmCIF, etc.).
    output_filename : str
        Path for output XYZ file.
    comment : str, optional
        Comment line to include in XYZ file header. Default is empty string.

    Returns
    -------
    None
        Writes XYZ file to disk.

    Notes
    -----
    XYZ format:= Header commented line 1: number of atoms
    Line 2: comment
    Lines 3+: element x y z
    """

    inXYZ = strucio.load_structure(input_filename)

    with open(output_filename, "w") as file:
        # Write the header
        nAtoms = inXYZ.shape[0]
        file.write(f"{int(nAtoms)}\n")
        file.write(comment + "\n")

        # Write the coordinates and other details
        for ii in range(nAtoms):
            file.write(
                f"{inXYZ[ii].element}\t{inXYZ[ii].coord[0]:<8.4f}\t{inXYZ[ii].coord[1]:<8.4f}\t{inXYZ[ii].coord[2]:<8.4f}\n"
            )
