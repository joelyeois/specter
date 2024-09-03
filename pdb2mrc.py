import torch
import potential
import atom
import pdbtools
from icemaker import Icemaker
from fft_tools import fftconvolve

class PDB2MRC:
    """Ice-making machine"""
    
    def __init__(self, pdb_code):
        self.pdb_code = pdb_code

    def fetch_pdb(self, pdb_folder='./pdb-data/'):
        # saves PDB file
        self.pdb_folder = pdb_folder
        self.pdb_path = pdbtools.fetch_pdb_file(self.pdb_code, output=pdb_folder)

        # extract atomic coordinates and symbols
        self.atomic_numbers, self.coords = pdbtools.get_atoms_and_coordinates_from_pdb(self.pdb_path)
        self.n_atoms = len(self.coords)
        self.unique_elements = atom.atom_symbol(torch.unique(self.atomic_numbers))

        # center coordinates onto origin (0,0,0)
        center = pdbtools.center_of_particle(self.coords)
        self.centered_coords = self.coords - center.reshape(1, -1)

    def build_particle(self, 
                       n=256, 
                       dx=1, 
                       super_sampling_factor=4, 
                       method='fftconvolve',
                       atom_size_px=11,
                      ):
        self.n = n
        self.dx = dx
        self.super_sampling_factor = 4
        self.atom_size_px = atom_size_px
        if method == 'per_atom':
            self.particle, self.occupancy = potential.build_potential_volume(
                self.atomic_numbers,
                self.centered_coords,
                (n, n, n),
                (dx, dx, dx),
                # atom_size_px=11,
                super_sampling_factor=super_sampling_factor,
                convention="relion",
                method="snapped-3d",
            )
        elif method == 'fftconvolve':
            self.particle, self.occupancy, self.atomic_potentials = potential.build_potential_volume_fftconvolve(
                self.atomic_numbers,
                self.centered_coords,
                (n, n, n),
                (dx, dx, dx),
                # atom_size_px=11,
                super_sampling_factor=super_sampling_factor,
                convention="relion",
                method="snapped-3d",
                compute_high_res=compute_high_res,
            )

    def solvate(self,
                source='torch',
                filepath='ice-data/mdsim_f_kernel_averaged_200x200x200_0.5A.pt',
                min_distance_A=1.5,
                niter=5
               ):
        self.icemaker = Icemaker()
        self.icemaker.get_mdsim_averaged_f_kernel(filepath, source=source)
        self.icemaker.interpolate_mdsim_f_kernel(self.n, self.dx)
        self.icemaker.generate_ice(min_distance=int(min_distance_A//self.dx), niter=niter)

        # add ice to particle
        self.icecube_deltas = self.icemaker.current_ice_vol.clone()
        self.icy_particle = self.particle.clone()

        # remove ice from particle regions
        self.icecube_deltas[self.occupancy] = 0

        # check if atomic potential for O has already been calcuated
        if 'O' not in self.atomic_potentials:
            sx, sy, sz, sX, sY, sZ = coordinate_grid_3d(
                (self.atom_size_px, self.atom_size_px, self.atom_size_px), (self.dx, self.dx, self.dx), convention="torch"
            )
            sR = torch.sqrt(sX**2 + sY**2 + sZ**2)
            O_pot = potential.atomic_potential_3d(8, sR)
        else:
            # make sure to bin to correct pixel size
            super_sampling_factor = int(self.dx / self.atomic_potentials['ssdx'])
            avgpool3d = torch.nn.AvgPool3d(super_sampling_factor, stride=super_sampling_factor)
            O_pot = avgpool3d(self.atomic_potentials['O'][None, None]).squeeze() * self.dx

        # convolve
        self.icecube = fftconvolve(self.icecube_deltas, O_pot, mode='same')

        # make icy particle
        self.icy_particle += self.icecube