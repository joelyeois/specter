import torch
import rotations
import potential
import numpy as np

avogadro = 6.02214076e23
density_of_amorphous_ice = 0.94  # [g/cm3]
molar_mass_of_water = 18.01528  # [g/mol]
ndensity_of_amorphous_ice = (
    density_of_amorphous_ice * avogadro / molar_mass_of_water * 1e-24
)  # [particles / A3]

def water_molecule_coordinates(bond_angle=105, bond_length=0.9572):
    """Returns coordinates of a single H2O molecule, where oxygen is defined as the
    origin (0,0,0).

    Parameters
    ----------
    band_angle : float
        Bond angle between the O-H bonds in degrees.
    bond_length : float
        Bond length of O-H bond in angstrom.

    Returns
    -------
    atomic_numbers : tensor
        Atomic numbers of the atoms in H2O
    coordinates : tensor
        xyz coordinates of the atoms in H2O.
    """
    #set oxygen as origin
    O_xyz = torch.tensor([0,0,0])

    #assume H2O is in xy-plane, with symmetry about y-axis.
    bond_angle = torch.tensor(bond_angle) / 180 * torch.pi
    y = bond_length * torch.cos(bond_angle/2)
    x = bond_length * torch.sin(bond_angle/2)
    H1_xyz = torch.tensor([x,y,0])
    H2_xyz = torch.tensor([-x,y,0])

    coordinates = torch.stack((O_xyz,H1_xyz,H2_xyz))
    atomic_numbers = np.array([8,1,1])
    return atomic_numbers, coordinates

def create_n_randomly_rotated_water_molecules(n, **kwargs):
    quats = torch.stack([rotations.random_quaternion() for _ in range(n)])
    atomic_numbers, coordinates = water_molecule_coordinates(**kwargs)
    
    # rotate only the hydrogens since oxygen is at origin.
    O_coordinates = coordinates[0].repeat(n,1)
    H1_coordinates = rotations.rotate_coordinates(coordinates[1], quats)
    H2_coordinates = rotations.rotate_coordinates(coordinates[2], quats)

    # concat coordinates together again.
    coordinates = torch.zeros(n*3,3)
    coordinates[0::3] = O_coordinates
    coordinates[1::3] = torch.from_numpy(H1_coordinates)
    coordinates[2::3] = torch.from_numpy(H2_coordinates)

    atomic_numbers = np.array([8,1,1]*n)
    return atomic_numbers, coordinates

def volume_of_ice(n_xyz, d_xyz):
    nx, ny, nz = n_xyz
    dx, dy, dz = d_xyz
    dv = dx*dy*dz
    nv = nx * ny * nz
    total_vol = nv * dv #A^3
    n_ice_molecules = int(ndensity_of_amorphous_ice * total_vol)
    print(n_ice_molecules)

    # sample ice positions
    # this may cause overlapping ice molecules.
    x_ice = (torch.rand(n_ice_molecules) - 0.5) * dx * nx
    y_ice = (torch.rand(n_ice_molecules) - 0.5) * dy * ny
    z_ice = (torch.rand(n_ice_molecules) - 0.5) * dz * nz
    ice_centers = torch.stack((x_ice, y_ice, z_ice), dim=1)

    # repeat for easy coordinate translation later. Each set of H,O,O atoms have the
    # same ice center, hence the 3x repeat_interleaves.
    ice_centers = torch.repeat_interleave(ice_centers, 3, dim=0)

    # create ice molecules
    ice_atomic_numbers, ice_coordinates = create_n_randomly_rotated_water_molecules(n_ice_molecules)

    # translate ice coordinates
    ice_coordinates += ice_centers
    return ice_atomic_numbers, ice_coordinates