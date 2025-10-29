import torch
from functools import lru_cache
from torch.special import modified_bessel_k0
from importlib import resources

@lru_cache(maxsize=1)
def load_kirkland_parameters():
    """
    Load Kirkland's atom scattering parameters from a text file.

    Returns
    -------
    out : torch.Tensor
        A tensor of shape (104, 3, 4) containing the scattering parameters
        for atomic numbers 0–103. Index 0 is all zeros.

    Notes
    -----
    Kirkland uses a parameterization of 12 parameters to model the scattering potential.
    For a single element, params[i] is a 3x4 array:
    params[i] = [
        [a1, b1, a2, b2],
        [a3, b3, c1, d1],
        [c2, d2, c3, d3]
    ]
    
    We reorder this to
    out[i] = [
        [a1, b1, c1, d1],
        [a2, b2, c2, d2],
        [a3, b3, c3, d3]
    ]
    
    References
    ----------
    Kirkland, E. J. (2010). *Advanced Computing in Electron Microscopy*, 2nd Edition,
    Appendix C.4.
    """
    params_list = []
    
     # Use importlib.resources to get the file path safely
    data_file = resources.files("cryosim.atom_data").joinpath(
        "kirkland_scattering_parameters.txt"
    )
    
    with resources.as_file(data_file) as fpath:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("Z=") or line.startswith("chisq="):
                    continue
                numbers = [float(x) for x in line.split()]
                params_list.append(numbers)

    params  = torch.as_tensor(params_list, dtype=torch.float32)
    params  = params .view(103, 3, 4)

    # Prepend a zero entry for index 0
    zeros = torch.zeros((1, 3, 4), dtype=torch.float32)
    params = torch.cat([zeros, params], dim=0)  # shape (104, 3, 4)

    # Reorder parameters
    out = torch.empty_like(params)
    out[:, 0, :] = torch.stack([params[:, 0, 0], params[:, 0, 1], params[:, 1, 2], params[:, 1, 3]], dim=-1)
    out[:, 1, :] = torch.stack([params[:, 0, 2], params[:, 0, 3], params[:, 2, 0], params[:, 2, 1]], dim=-1)
    out[:, 2, :] = torch.stack([params[:, 1, 0], params[:, 1, 1], params[:, 2, 2], params[:, 2, 3]], dim=-1)
    return out


@lru_cache(maxsize=1)
def load_lobato_parameters():
    """
    Load Lobato electron scattering parameters from a text file.

    Returns
    -------
    torch.Tensor
        A tensor of shape (104, 5, 2) containing the scattering parameters
        for atomic numbers 0–103. Index 0 is all zeros.

    Notes
    -----
    Lobato uses a parameterization of 10 parameters to model the scattering potential.
    For a single element, params[i] is a 5x2 array:
    params_tensor[i] = [
        [a1, b1],
        [a2, b2],
        [a3, b3],
        [a4, b4],
        [a5, b5]
    ]
    
    References
    ----------
    Lobato, I., & Van Dyck, D. (2014). An accurate parameterization for scattering
    factors, electron densities and electrostatic potentials for neutral atoms that obey
    all physical constraints. Acta Crystallographica. Section A, Foundations and
    Advances, 70(6), 636–649.
    """
    params_list = []

    # Use importlib.resources to get the file path safely
    data_file = resources.files("cryosim.atom_data").joinpath(
        "lobato_scattering_parameters.txt"
    )
    with resources.as_file(data_file) as fpath:
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        # Element symbol
        element = line
        i += 1

        block = []
        for _ in range(5):
            a, b = map(float, lines[i].strip().split(','))
            block.append([a, b])
            i += 1
        params_list.append(block)
    
    params = torch.as_tensor(params_list, dtype=torch.float32)

    # Prepend a zero entry for index 0
    zeros = torch.zeros((1, 5, 2), dtype=torch.float32)
    params = torch.cat([zeros, params], dim=0)  # shape (104, 5, 2)

    return params


def kirkland_atomic_potential_2d(atomic_number, r_xy):
    """
    Compute 2D projected electrostatic potential for an atom using Kirkland parameters.
    Vectorized over the 3 scattering terms to avoid explicit Python loops.

    Parameters
    ----------
    atomic_number : int
        Atomic number of the element.
    r_xy : torch.Tensor
        2D grid of radial distances, shape (Nx, Ny)

    Returns
    -------
    torch.Tensor
        2D potential, same shape as r_xy
    """
    device = r_xy.device
    a0 = 0.529  # Bohr radius, Angstrom
    e = 14.4    # electron charge, V-Angstrom

    c1 = 4 * (torch.pi**2) * a0 * e
    c2 = 2 * (torch.pi**2) * a0 * e

    # Load Kirkland scattering parameters
    kirkland_params = load_kirkland_parameters()  # shape (104, 3, 4)
    P = kirkland_params[atomic_number]        # shape (3, 4)

    # Separate columns: a_i, b_i, c_i, d_i
    a = P[:, 0].unsqueeze(-1).unsqueeze(-1)  # shape (3,1,1)
    b = P[:, 1].unsqueeze(-1).unsqueeze(-1)  # shape (3,1,1)
    c = P[:, 2].unsqueeze(-1).unsqueeze(-1)  # shape (3,1,1)
    d = P[:, 3].unsqueeze(-1).unsqueeze(-1)  # shape (3,1,1)

    r = r_xy.unsqueeze(0)  # shape (1, Nx, Ny)

    # Vectorized sum over i=1..3
    s1 = c1 * torch.sum(a * modified_bessel_k0(2 * torch.pi * r * torch.sqrt(b)), dim=0)
    s2 = c2 * torch.sum(c / d * torch.exp(-(torch.pi**2) * r**2 / d), dim=0)

    return s1 + s2

def kirkland_atomic_potential_3d(atomic_number, r_xyz):
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
    device = r_xyz.device
    a0 = 0.529  # Bohr radius, [Angstrom]
    e = 14.4  # electron charge, [V-Angstrom]
    c1 = 2 * (torch.pi**2) * a0 * e
    c2 = 2 * (torch.pi ** (5 / 2)) * a0 * e

    # get scattering factors
    kirkland_params = load_kirkland_parameters()  # shape (104, 3, 4)
    P = kirkland_params[atomic_number]        # shape (3, 4)
    P = P.to(device)
    
    # Separate columns: a_i, b_i, c_i, d_i
    a = P[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape (3,1,1,1)
    b = P[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    c = P[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    d = P[:, 3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    r = r_xyz.unsqueeze(0)  # shape (1, Nx, Ny, Nz)
    
    s1 = c1 * torch.sum(a / r * torch.exp(-2 * torch.pi * r * torch.sqrt(b)), 0)
    s2 = c2 * torch.sum(c * d ** (-3 / 2) * torch.exp(-(torch.pi**2) * (r**2) / d), 0)
    return s1 + s2

def kirkland_atomic_potential_3d_fourier(atomic_number, k_xyz):
    """Returns the Fourier transformed 3D atomic potential for a specific element and 
    given a 3D grid of radial distances from the atom core. Kirkland C.15.

    Parameters
    ----------
    atomic_number : int
        Atomic number, Hyrdogen has number 1.
    k_xyz : 3D tensor
        Distances from the atomic core in units of Ångstrom. k^2 = kx^2 + ky^2 + kz^2.
        Assume equally spaced grid along kx and ky, i.e. dkx = dky.

    Returns
    -------
    potential : tensor
        Atomic potential in Fourier space in units of 1/V-Ångstrom, same shape as r_xyz.
    """
    device = k_xyz.device
    a0 = 0.529  # Bohr radius, [Angstrom]
    e = 14.4  # electron charge, [V-Angstrom]

    # get scattering factors
    kirkland_params = load_kirkland_parameters()  # shape (104, 3, 4)
    P = kirkland_params[atomic_number]        # shape (3, 4)
    P = P.to(device)
    
    # Extract columns
    a = P[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (3,1,1,1)
    b = P[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    c = P[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    d = P[:, 3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    k2 = k_xyz**2
    k2 = k2.unsqueeze(0)  # shape (1, Nx, Ny, Nz)

    s1 = torch.sum(a / (k_xyz**2 + b), 0)
    s2 = torch.sum(c * torch.exp(-d * k2), 0)
    return s1 + s2
    
def lobato_atomic_potential_3d(atomic_number, r_xyz):
    """Returns the 3D atomic potential for a specific element and given a 3D grid
    of radial distances from the atom core. Lobato Eq.59.

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
    device = r_xyz.device
    vac_perm = 1 / 4 / torch.pi
    a0 = 0.529  # Bohr radius, [Angstrom]
    e = 14.4  # electron charge, [V-Angstrom]
    kappa = 2 * vac_perm / a0 / e
    c1 = torch.pi**2 / kappa

    # get scattering factors
    lobato_params = load_lobato_parameters()  # shape (104, 5, 2)
    P = lobato_params[atomic_number]        # shape (5, 2)
    P = P.to(device)
    
    # Separate columns: a_i, b_i
    a = P[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape (3,1,1,1)
    b = P[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    r = r_xyz.unsqueeze(0)  # shape (1, Nx, Ny, Nz)

    s1 = torch.sum( c1* a / b**1.5 * (b**0.5/torch.pi / r + 1) * torch.exp(-2 * torch.pi * r / b**0.5), 0)
    return s1

def lobato_atomic_potential_3d_fourier(atomic_number, k_xyz):
    """Returns the Fourier transformed 3D atomic potential for a specific element and 
    given a 3D grid of radial distances from the atom core. Kirkland C.15.

    Parameters
    ----------
    atomic_number : int
        Atomic number, Hyrdogen has number 1.
    k_xyz : 3D tensor
        Distances from the atomic core in units of Ångstrom. k^2 = kx^2 + ky^2 + kz^2.
        Assume equally spaced grid along kx and ky, i.e. dkx = dky.

    Returns
    -------
    potential : tensor
        Atomic potential in Fourier space in units of 1/V-Ångstrom, same shape as r_xyz.
    """
    device = k_xyz.device
    vac_perm = 1 / 4 / torch.pi
    a0 = 0.529  # Bohr radius, [Angstrom]
    e = 14.4  # electron charge, [V-Angstrom]
    kappa = 2 * vac_perm / a0 / e
    c1 = torch.pi**2 / kappa

    # get scattering factors
    lobato_params = load_lobato_parameters()  # shape (104, 3, 4)
    P = lobato_params[atomic_number]        # shape (3, 4)
    P = P.to(device)
    
    # Separate columns: a_i, b_i
    a = P[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape (3,1,1,1)
    b = P[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    k2 = k_xyz**2
    k2 = k2.unsqueeze(0)  # shape (1, Nx, Ny, Nz)

    s1 = torch.sum(a * (2 + b * k2) / (1 + b * k2)**2, 0)
    return s1 + s2