from __future__ import annotations

import torch
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from typing import Literal

from . import atom, pdbtools, potential


class PDB2MRC:
    """
    Converts PDB structure to MRC potential volume.

    Parameters
    ----------
    pdb_code : str, optional
        PDB code to fetch.
    pdb_path : str, optional
        Path to local PDB file.
    """

    def __init__(self, pdb_code: str | None = None, pdb_path: str | None = None):
        self.pdb_code = pdb_code
        self.pdb_path = pdb_path

    def fetch_pdb(
        self,
        pdb_folder: str = "../pdb-data/",
        assembly: bool = True,
        center: bool = True,
    ) -> None:
        """
        Fetch PDB file or load local file, and extract coordinates.

        Parameters
        ----------
        pdb_folder : str, optional
            Folder to save fetched PDB file. Default "../pdb-data/".
        assembly : bool, optional
            Whether to fetch biological assembly. Default True.
        center : bool, optional
            Whether to center the coordinates. Default True.
        """
        self.assembly = assembly
        if self.pdb_code is not None:
            # saves PDB file
            self.pdb_folder = pdb_folder
            self.pdb_path = pdbtools.fetch_pdb_file(
                self.pdb_code, output=pdb_folder, assembly=assembly
            )

        # extract atomic coordinates and symbols
        self.atomic_numbers, self.coords = pdbtools.get_atoms_and_coordinates_from_pdb(
            self.pdb_path
        )
        self.n_atoms = len(self.coords)
        self.unique_elements = atom.atom_symbol(torch.unique(self.atomic_numbers))

        # center coordinates onto origin (0,0,0)
        if center:
            center = pdbtools.center_of_particle(self.coords)
            self.centered_coords = self.coords - center.reshape(1, -1)
        else:
            self.centered_coords = self.coords

        self.estimate_max_diameter()

    def estimate_max_diameter(self, coords: torch.Tensor | None = None) -> None:
        """
        Estimate maximum diameter of the particle using Convex Hull.

        Parameters
        ----------
        coords : torch.Tensor, optional
            Coordinates. If None, uses `self.centered_coords`.
        """
        if coords is None:
            coords = self.centered_coords
        hull = ConvexHull(coords)
        hull_points = coords[hull.vertices]
        self.max_diameter = pdist(hull_points).max()

    def build_particle(
        self,
        n: int = 256,
        dx: float = 1.0,
        super_sampling_factor: int = 4,
        method: Literal["2d", "3d"] = "3d",
        atom_size_px: int = 11,
    ) -> None:
        """
        Build potential volume from atomic coordinates.

        Parameters
        ----------
        n : int, optional
            Volume size (n, n, n). Default 256.
        dx : float, optional
            Pixel size in Å. Default 1.
        super_sampling_factor : int, optional
            Super sampling factor (unused in current implementation?). Default 4.
        method : str, optional
            '2d' or '3d'. Default '3d'.
        atom_size_px : int, optional
            Atom box size in pixels for kernel. Default 11.
        """
        self.n = n
        self.dx = dx
        self.super_sampling_factor = 4
        self.atom_size_px = atom_size_px

        if method == "2d":
            self.particle, self.occupancy, self.atomic_potentials = (
                potential.build_potential_volume_fftconvolve_2d(
                    self.atomic_numbers,
                    self.centered_coords,
                    (n, n, n),
                    dx,
                )
            )
        elif method == "3d":
            self.particle, self.occupancy, self.atomic_potentials = (
                potential.build_potential_volume_fftconvolve_3d(
                    self.atomic_numbers,
                    self.centered_coords,
                    (n, n, n),
                    dx,
                )
            )
