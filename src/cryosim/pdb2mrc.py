import torch
from . import potential
from . import atom
from . import pdbtools
from .icemaker import Icemaker
from .fft_tools import fftconvolve
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

class PDB2MRC:
    """Ice-making machine"""

    def __init__(self, pdb_code=None, pdb_path=None):
        self.pdb_code = pdb_code
        self.pdb_path = pdb_path

    def fetch_pdb(self, pdb_folder='../pdb-data/', assembly=True, center=True):
        self.assembly = assembly
        if self.pdb_code is not None:
            # saves PDB file
            self.pdb_folder = pdb_folder
            self.pdb_path = pdbtools.fetch_pdb_file(self.pdb_code,
                                                    output=pdb_folder,
                                                    assembly=assembly)

        # extract atomic coordinates and symbols
        self.atomic_numbers, self.coords = pdbtools.get_atoms_and_coordinates_from_pdb(self.pdb_path)
        self.n_atoms = len(self.coords)
        self.unique_elements = atom.atom_symbol(torch.unique(self.atomic_numbers))

        # center coordinates onto origin (0,0,0)
        if center:
            center = pdbtools.center_of_particle(self.coords)
            self.centered_coords = self.coords - center.reshape(1, -1)
        else:
            self.centered_coords = self.coords

        self.estimate_max_diameter()

    def estimate_max_diameter(self, coords=None):
        if coords is None:
            coords = self.centered_coords
        hull = ConvexHull(coords)
        hull_points = coords[hull.vertices]
        self.max_diameter = pdist(hull_points).max()


    def build_particle(self, 
                       n=256, 
                       dx=1, 
                       super_sampling_factor=4, 
                       method='3d',
                       atom_size_px=11,
                      ):
        self.n = n
        self.dx = dx
        self.super_sampling_factor = 4
        self.atom_size_px = atom_size_px

        if method == '2d':
            self.particle, self.occupancy, self.atomic_potentials = potential.build_potential_volume_fftconvolve_2d(
                self.atomic_numbers,
                self.centered_coords,
                (n, n, n),
                (dx, dx, dx),
            )
        elif method == '3d':
            self.particle, self.occupancy, self.atomic_potentials = potential.build_potential_volume_fftconvolve_3d(
                self.atomic_numbers,
                self.centered_coords,
                (n, n, n),
                (dx, dx, dx),
            )