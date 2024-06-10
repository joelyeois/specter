import torch
import atom
from scipy.special import kn


def atomic_potential_2d(atomic_number, r_xy, dx):
    """Returns the 2D projected atomic potential for a specific element given a 
    2D grid of radial distances from the atom core. Kirkland C.20.

    Note: There is a singularity at r = 0 because the atomic nucleaus is essentially
    a point charge on this scale (~1e-5 Angstroms).
    
    Parameters
    ----------
    atomic_number : int
        Atomic number, Hyrdogen has number 1.
    r_xy : 2D tensor
        Distances from the atomic core in units of Ångstrom. r^2 = x^2 + y^2.
        Assume equally spaced grid along x and y, i.e. dx = dy.
        
    Returns
    -------
    potential : tensor
        Atomic potential in units of V-Ångstrom, same shape as r_xy.
    """
    a0 = 0.529 # Bohr radius, [Angstrom]
    e  = 14.4 # electron charge, [V-Angstrom]
    c1 = 4 * ( torch.pi**2 ) * a0 * e
    c2 = 2 * ( torch.pi**2 ) * a0 * e

    # get scattering factors
    atom_params_dict = atom.get_atom_params_dict()
    P = torch.from_numpy(atom_params_dict[atomic_number]['params'])
    # tile scattering factors to match r_xy grid
    P = P[:,:,None,None].expand((4, 3) + r_xy.shape)

    s1 = c1 * torch.sum( P[0] * kn(0.,2 * torch.pi * r_xy * torch.sqrt( P[1] )), 0)
    s2 = c2 * torch.sum( P[2] / P[3] * torch.exp( -(torch.pi**2) * (r_xy**2) / P[3]), 0)
    return (s1 + s2) * dx**2
    

def atomic_potential_3d(atomic_number, r_xyz, dx):
    """Returns the 3D atomic potential for a specific element and given a 3D grid 
    of radial distances from the atom core. Kirkland C.19.

    Summing along the z-axes (or any other axes due to symmetry) should yield
    approximately the same results as atomic_potential_2d.

    Note: There is a singularity at r = 0 because the atomic nucleaus is essentially
    a point charge on this scale (~1e-5 Angstroms).
    
    Parameters
    ----------
    atomic_number : int
        Atomic number, Hyrdogen has number 1.
    r_xyz : 3D tensor
        Distances from the atomic core in units of Ångstrom. r^2 = x^2 + y^2 + z^2.
        Assume equally spaced grid along x and y, i.e. dx = dy.
        
    Returns
    -------
    potential : tensor
        Atomic potential in units of V-Ångstrom, same shape as r_xyz.
    """
    a0 = 0.529 # Bohr radius, [Angstrom]
    e  = 14.4 # electron charge, [V-Angstrom]
    c1 = 2 * ( torch.pi**2 ) * a0 * e
    c2 = 2 * ( torch.pi**(5/2) ) * a0 * e
    
    # get scattering factors
    atom_params_dict = atom.get_atom_params_dict()
    P = torch.from_numpy(atom_params_dict[atomic_number]['params'])
    # tile scattering factors to match r_xy grid
    P = P[:,:, None, None, None].expand((4, 3) + r_xyz.shape)

    s1 = c1 * torch.sum( P[0] / r_xyz * torch.exp(-2 * torch.pi * r_xyz * torch.sqrt( P[1] )), 0)
    s2 = c2 * torch.sum( P[2] * P[3]**(-3/2) * torch.exp( -(torch.pi**2) * (r_xyz**2) / P[3]), 0)
    return (s1 + s2) * dx**3