from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

import gemmi
import torch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from torch.special import modified_bessel_k0, modified_bessel_k1

# A handful of bundled Shtyrov bonded-species entries (e.g. 'O(C, amide)',
# 'H(N)') have a b_i = 0 term. In the b_i*exp(-b_i*k^2/4) Fourier-space
# parameterization that is a constant (frequency-independent) contribution,
# whose real-space inverse transform is a Dirac delta at r=0 — the closed-form
# real-space formula (4*pi/b)^1.5 * exp(-pi^2*r^2/b*4) divides by b, so it is
# NaN exactly at b=0. Flooring b at this epsilon turns the idealized delta
# into an extremely narrow but finite Gaussian (real-space width ~1e-3 Å,
# negligible at any grid spacing this codebase uses), consistent with the
# point-charge-like singularity at r=0 these potentials already have.
MIN_GAUSSIAN_B = 1e-4


@lru_cache(maxsize=1)
def load_kirkland_parameters() -> torch.Tensor:
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
    For a single element, params[i] is a 3x4 array::

        params[i] = [
            [a1, b1, a2, b2],
            [a3, b3, c1, d1],
            [c2, d2, c3, d3]
        ]

    We reorder this to::

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
    data_file = resources.files("specter.atom_data").joinpath(
        "kirkland_scattering_parameters.txt"
    )

    with resources.as_file(data_file) as fpath:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("Z=") or line.startswith("chisq="):
                    continue
                numbers = [float(x) for x in line.split()]
                params_list.append(numbers)

    params = torch.as_tensor(params_list, dtype=torch.float32)
    params = params.view(103, 3, 4)

    # Prepend a zero entry for index 0
    zeros = torch.zeros((1, 3, 4), dtype=torch.float32)
    params = torch.cat([zeros, params], dim=0)  # shape (104, 3, 4)

    # Reorder parameters
    out = torch.empty_like(params)
    out[:, 0, :] = torch.stack(
        [params[:, 0, 0], params[:, 0, 1], params[:, 1, 2], params[:, 1, 3]], dim=-1
    )
    out[:, 1, :] = torch.stack(
        [params[:, 0, 2], params[:, 0, 3], params[:, 2, 0], params[:, 2, 1]], dim=-1
    )
    out[:, 2, :] = torch.stack(
        [params[:, 1, 0], params[:, 1, 1], params[:, 2, 2], params[:, 2, 3]], dim=-1
    )
    return out


@lru_cache(maxsize=1)
def load_lobato_parameters() -> torch.Tensor:
    """
    Load Lobato electron scattering parameters from a text file.

    Returns
    -------
    params : torch.Tensor
        A tensor of shape (104, 5, 2) containing the scattering parameters
        for atomic numbers 0–103. Index 0 is all zeros.

    Notes
    -----
    Lobato uses a parameterization of 10 parameters to model the scattering potential.
    For a single element, params[i] is a 5x2 array::

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
    data_file = resources.files("specter.atom_data").joinpath(
        "lobato_scattering_parameters.txt"
    )
    with resources.as_file(data_file) as fpath:
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        # Element symbol
        i += 1

        block = []
        for _ in range(5):
            a, b = map(float, lines[i].strip().split(","))
            block.append([a, b])
            i += 1
        params_list.append(block)

    params = torch.as_tensor(params_list, dtype=torch.float32)

    # Prepend a zero entry for index 0
    zeros = torch.zeros((1, 5, 2), dtype=torch.float32)
    params = torch.cat([zeros, params], dim=0)  # shape (104, 5, 2)

    return params


@lru_cache(maxsize=1)
def load_shtyrov_parameters(filepath: str) -> torch.Tensor:
    """
    Load Shtyrov electron scattering parameters from a MMCIF file.

    Parameters
    ----------
    filepath : str
        Path to the MMCIF file.

    Returns
    -------
    params : torch.Tensor
        A tensor of shape (N, 5, 2) containing the scattering parameters
        for atomic numbers 0 – N-1. Note that these are NOT elemental numbers.

    Notes
    -----
    Shtyrov uses a parameterization of 10 parameters to model the scattering potential.
    For a single element, params[i] is a 5x2 array::

        params_tensor[i] = [
            [a1, b1],
            [a2, b2],
            [a3, b3],
            [a4, b4],
            [a5, b5]
        ]

    References
    ----------
    Shtyrov, A., Wilson, H., Slowik, D., Yamashita, K., Li, J., Wojdyr, M., Chen, S.,
    McMullan, G., Short, J. M., Russo, C. J., Henderson, R., & Murshudov, G. N. (2026).
    Measurement of atomic scattering factors by cryoelectron microscopy. Proceedings of
    the National Academy of Sciences, 123(19), e2528758123.
    https://doi.org/10.1073/pnas.2528758123
    """
    # parse .cif file
    print("Parsing MMCIF file.")
    cif_dict = MMCIF2Dict(filepath)

    # parse scat_id as integer tensor
    ids = torch.tensor(
        [int(x) for x in cif_dict["_lmb_scat_coef.scat_id"]], dtype=torch.long
    )
    max_id = int(torch.max(ids).item())

    # parse coefficients
    a_keys = [f"_lmb_scat_coef.coef_a{i}" for i in range(1, 6)]
    b_keys = [f"_lmb_scat_coef.coef_b{i}" for i in range(1, 6)]
    a = torch.stack(
        [
            torch.tensor([float(x) for x in cif_dict[k]], dtype=torch.float32)
            for k in a_keys
        ],
        dim=1,
    )
    b = torch.stack(
        [
            torch.tensor([float(x) for x in cif_dict[k]], dtype=torch.float32)
            for k in b_keys
        ],
        dim=1,
    )

    # create empty tensor (N = max_id + 1)
    params = torch.zeros((max_id + 1, 5, 2), dtype=torch.float32)

    # fill according to scat_id
    params[ids, :, 0] = a
    params[ids, :, 1] = b
    print(f"Number of unique 'atoms': {len(ids)}")
    return params


def kirkland_atomic_potential_2d(
    atomic_number: int, r_xy: torch.Tensor
) -> torch.Tensor:
    """
    Compute 2D projected electrostatic potential for an atom using Kirkland parameters.

    Vectorized over the 3 scattering terms to avoid explicit Python loops.

    Parameters
    ----------
    atomic_number : int
        Atomic number of the element.
    r_xy : torch.Tensor
        2D grid of radial distances, shape (Nx, Ny).

    Returns
    -------
    potential : torch.Tensor
        2D potential, same shape as r_xy.
    """
    device = r_xy.device
    a0 = 0.529  # Bohr radius, [Å]
    e = 14.4  # electron charge, [V·Å]

    c1 = 4 * (torch.pi**2) * a0 * e
    c2 = 2 * (torch.pi**2) * a0 * e

    # Load Kirkland scattering parameters
    kirkland_params = load_kirkland_parameters()  # shape (104, 3, 4)
    P = kirkland_params[atomic_number]  # shape (3, 4)
    P = P.to(device)

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


def kirkland_atomic_potential_3d(
    atomic_number: int, r_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Compute the 3D atomic potential for a specific element.

    Based on Kirkland C.19.
    Summing along the z-axes (or any other axes due to symmetry) should yield
    approximately the same results as atomic_potential_2d.

    Note: There is a singularity at r = 0 because the atomic nucleus is essentially
    a point charge on this scale (~1e-5 Å).

    Parameters
    ----------
    atomic_number : int
        Atomic number, Hydrogen has number 1.
    r_xyz : torch.Tensor
        Distances from the atomic core in units of Å. r^2 = x^2 + y^2 + z^2.
        Assume equally spaced grid along x and y, i.e. dx = dy.

    Returns
    -------
    potential : torch.Tensor
        Atomic potential in units of V, same shape as r_xyz.
    """
    device = r_xyz.device
    a0 = 0.529  # Bohr radius, [Å]
    e = 14.4  # electron charge, [V·Å]
    c1 = 2 * (torch.pi**2) * a0 * e
    c2 = 2 * (torch.pi ** (5 / 2)) * a0 * e

    # get scattering factors
    kirkland_params = load_kirkland_parameters()  # shape (104, 3, 4)
    P = kirkland_params[atomic_number]  # shape (3, 4)
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


def kirkland_atomic_potential_3d_fourier(
    atomic_number: int, k_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Fourier transformed 3D atomic potential for a specific element.

    Based on Kirkland C.15.

    Parameters
    ----------
    atomic_number : int
        Atomic number, Hydrogen has number 1.
    k_xyz : torch.Tensor
        Spatial frequencies in units of 1/Å. k^2 = kx^2 + ky^2 + kz^2.
        Assume equally spaced grid along kx and ky, i.e. dkx = dky.

    Returns
    -------
    potential : torch.Tensor
        Electron scattering factor in units of Å, same shape as k_xyz.
        Multiply by c1 = 2*pi*e*a0 = 47.9 V·Å² to obtain the Fourier-space
        potential in V·Å³, as `PotentialBuilder` does.
    """
    device = k_xyz.device

    # get scattering factors
    kirkland_params = load_kirkland_parameters()  # shape (104, 3, 4)
    P = kirkland_params[atomic_number]  # shape (3, 4)
    P = P.to(device)

    # Extract columns
    a = P[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (3,1,1,1)
    b = P[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    c = P[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    d = P[:, 3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    k2 = k_xyz**2
    k2 = k2.unsqueeze(0)  # shape (1, Nx, Ny, Nz)

    s1 = torch.sum(a / (k2 + b), 0)
    s2 = torch.sum(c * torch.exp(-d * k2), 0)
    return s1 + s2


def lobato_atomic_potential_2d(atomic_number: int, r_xy: torch.Tensor) -> torch.Tensor:
    """
    Compute the 3D atomic potential for a specific element using Lobato parameterization.

    Based on Lobato Eq.16.

    Parameters
    ----------
    atomic_number : int
        Atomic number, Hydrogen has number 1.
    r_xy : torch.Tensor
        Distances from the atomic core in units of Å. r^2 = x^2 + y^2.
        Assume equally spaced grid along x and y, i.e. dx = dy.

    Returns
    -------
    potential : torch.Tensor
        Atomic potential in units of V·Å, same shape as r_xy.
    """
    device = r_xy.device
    vac_perm = 1 / 4 / torch.pi
    a0 = 0.529  # Bohr radius, [Å]
    e = 14.4  # electron charge, [V·Å]
    kappa = 2 * vac_perm / a0 / e

    # get scattering factors
    lobato_params = load_lobato_parameters()  # shape (104, 5, 2)
    P = lobato_params[atomic_number]  # shape (5, 2)
    P = P.to(device)

    # Separate columns: a_i, b_i
    a = P[:, 0].unsqueeze(-1).unsqueeze(-1)  # shape (3,1,1)
    b = P[:, 1].unsqueeze(-1).unsqueeze(-1)

    r = r_xy.unsqueeze(0)  # shape (1, Nx, Ny)
    s1 = modified_bessel_k0(2 * torch.pi * r / b**0.5)
    s2 = r * modified_bessel_k1(2 * torch.pi * r / b**0.5)
    s = 1 / kappa * torch.sum(2 * torch.pi**2 * a / b**1.5 * (s1 + s2), 0)

    return s


def lobato_atomic_potential_3d(atomic_number: int, r_xyz: torch.Tensor) -> torch.Tensor:
    """
    Compute the 3D atomic potential for a specific element using Lobato parameterization.

    Based on Lobato Eq.15.

    Note: There is a singularity at r = 0 because the atomic nucleus is essentially
    a point charge on this scale (~1e-5 Å).

    Parameters
    ----------
    atomic_number : int
        Atomic number, Hydrogen has number 1.
    r_xyz : torch.Tensor
        Distances from the atomic core in units of Å. r^2 = x^2 + y^2 + z^2.
        Assume equally spaced grid along x and y, i.e. dx = dy.

    Returns
    -------
    potential : torch.Tensor
        Atomic potential in units of V, same shape as r_xyz.
    """
    device = r_xyz.device
    vac_perm = 1 / 4 / torch.pi
    a0 = 0.529  # Bohr radius, [Å]
    e = 14.4  # electron charge, [V·Å]
    kappa = 2 * vac_perm / a0 / e
    c1 = torch.pi**2 / kappa

    # get scattering factors
    lobato_params = load_lobato_parameters()  # shape (104, 5, 2)
    P = lobato_params[atomic_number]  # shape (5, 2)
    P = P.to(device)

    # Separate columns: a_i, b_i
    a = P[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape (3,1,1,1)
    b = P[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    r = r_xyz.unsqueeze(0)  # shape (1, Nx, Ny, Nz)

    s1 = torch.sum(
        c1
        * a
        / b**1.5
        * (b**0.5 / torch.pi / r + 1)
        * torch.exp(-2 * torch.pi * r / b**0.5),
        0,
    )
    return s1


def lobato_atomic_potential_3d_fourier(
    atomic_number: int, k_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Fourier transformed 3D atomic potential for a specific element using Lobato parameterization.

    Based on Lobato Eq. 56.

    Parameters
    ----------
    atomic_number : int
        Atomic number, Hydrogen has number 1.
    k_xyz : torch.Tensor
        Spatial frequencies in units of 1/Å. k^2 = kx^2 + ky^2 + kz^2.
        Assume equally spaced grid along kx and ky, i.e. dkx = dky.

    Returns
    -------
    potential : torch.Tensor
        Electron scattering factor in units of Å, same shape as k_xyz.
        Multiply by c1 = 2*pi*e*a0 = 47.9 V·Å² to obtain the Fourier-space
        potential in V·Å³, as `PotentialBuilder` does.
    """
    device = k_xyz.device

    # get scattering factors
    lobato_params = load_lobato_parameters()  # shape (104, 3, 4)
    P = lobato_params[atomic_number]  # shape (3, 4)
    P = P.to(device)

    # Separate columns: a_i, b_i
    a = P[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape (3,1,1,1)
    b = P[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    k2 = k_xyz**2
    k2 = k2.unsqueeze(0)  # shape (1, Nx, Ny, Nz)

    s1 = torch.sum(a * (2 + b * k2) / (1 + b * k2) ** 2, 0)
    return s1


def shtyrov_atomic_potential_3d_fourier(
    atomic_number: int, k_xyz: torch.Tensor, filepath: str
) -> torch.Tensor:
    """
    Compute the 3D atomic potential for a specific element using Shtyrov parameterization.

    Based on Shtyrov et al. (2026) Eq.18 -- see :func:`load_shtyrov_parameters`'s
    References section for the full citation.

    Note: There is a singularity at r = 0 because the atomic nucleus is essentially
    a point charge on this scale (~1e-5 Å).

    Parameters
    ----------
    atomic_number : int
        Atomic number, Hydrogen has number 1.
    k_xyz : torch.Tensor
        Spatial frequencies in units of 1/Å. k^2 = kx^2 + ky^2 + kz^2.
        Assume equally spaced grid along kx and ky, i.e. dkx = dky.
    filepath : str
        Path to the Shtyrov parameter file (MMCIF).

    Returns
    -------
    potential : torch.Tensor
        Electron scattering factor in units of Å, same shape as k_xyz.
        Multiply by c1 = 2*pi*e*a0 = 47.9 V·Å² to obtain the Fourier-space
        potential in V·Å³, as `PotentialBuilder` does.
    """
    device = k_xyz.device

    # get scattering factors
    shtyrov_params = load_shtyrov_parameters(filepath)  # shape (N, 5, 2)
    P = shtyrov_params[atomic_number]  # shape (5, 2)
    P = P.to(device)

    # Separate columns: a_i, b_i
    a = P[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape (3,1,1,1)
    b = P[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    k2 = k_xyz**2
    k2 = k2.unsqueeze(0)  # shape (1, Nx, Ny, Nz)

    s1 = torch.sum(a * torch.exp(-b * k2 / 4), 0)
    return s1


def shtyrov_atomic_potential_3d(
    atomic_number: int,
    r_xyz: torch.Tensor,
    filepath: str,
    voltage: float = 300,
) -> torch.Tensor:
    """
    Compute the 3D atomic potential for a specific element using Shtyrov parameterization.

    Based on Shtyrov et al. (2026) Eq.18 -- see :func:`load_shtyrov_parameters`'s
    References section for the full citation.

    Note: There is a singularity at r = 0 because the atomic nucleus is essentially
    a point charge on this scale (~1e-5 Å).

    Parameters
    ----------
    atomic_number : int
        Atomic number, Hydrogen has number 1.
    r_xyz : torch.Tensor
        3D grid of radial distances.
    filepath : str
        Path to the Shtyrov parameter file.
    voltage : float, optional
        Beam accelerating voltage in kV. Default 300.

    Returns
    -------
    potential : torch.Tensor
        Atomic potential in units of V, same shape as r_xyz.
    """
    a0 = 0.529  # Bohr radius, [Å]
    e = 14.4  # electron charge, [V·Å]
    c1 = 2 * torch.pi * e * a0

    shtyrov_params = load_shtyrov_parameters(filepath)  # shape (N, 5, 2)
    P = shtyrov_params[atomic_number].to(r_xyz.device)  # shape (5, 2)
    a = P[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape (5,1,1,1)
    b = P[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).clamp(min=MIN_GAUSSIAN_B)

    r2 = (r_xyz**2).unsqueeze(0)  # shape (1, Nx, Ny, Nz)

    # Closed-form 3D inverse Fourier transform of a sum of a_i*exp(-b_i*k^2/4)
    # Gaussians — same functional form Kirkland/Lobato use for their Gaussian
    # terms, evaluated directly with no FFT/grid dependence (unlike an FFT
    # route, this needs no periodic box large enough to resolve the
    # slowest-decaying terms without aliasing).
    s1 = c1 * torch.sum(
        a * (4 * torch.pi / b) ** 1.5 * torch.exp(-(torch.pi**2) * r2 / b * 4), 0
    )
    return s1


@lru_cache(maxsize=4)
def load_shtyrov_species_parameters(json_filepath: str) -> dict[str, torch.Tensor]:
    """
    Load Shtyrov bonded-species electron scattering parameters from a JSON file.

    Parameters
    ----------
    json_filepath : str
        Path to an sffit-format JSON parameter file (e.g. the bundled
        ``specter.atom_data/params_cat.json`` or ``params_emdb.json``).

    Returns
    -------
    params : dict of str to torch.Tensor
        Maps each bonded-species descriptor (e.g. `"O(HH)"`, `"C(HHHC)"`,
        as produced by :meth:`specter.pdb.PDB.get_atom_species`) to a
        tensor of shape (5, 2) of `(a_i, b_i)` coefficients.

    Notes
    -----
    Unlike :func:`load_shtyrov_parameters`, which reads a numeric-`scat_id`
    mmCIF table, this reads sffit's native JSON output keyed directly by
    the human-readable bonded-species string, matching what
    :meth:`specter.pdb.PDB.get_atom_species` produces.

    References
    ----------
    sffit: https://github.com/as2875/sffit
    """
    with open(json_filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    params = {}
    for species, entry in raw.items():
        coeffs = entry["coefficients"]
        params[species] = torch.tensor(
            [[c["a"], c["b"]] for c in coeffs], dtype=torch.float32
        )
    return params


def shtyrov_atomic_potential_3d_fourier_by_species(
    species: str, k_xyz: torch.Tensor, params: dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Compute the 3D Fourier-space atomic potential for a bonded species.

    Based on Shtyrov et al. (2026) Eq.18 -- see :func:`load_shtyrov_parameters`'s
    References section for the full citation. Same functional form as
    :func:`shtyrov_atomic_potential_3d_fourier`, but keyed by bonded-species
    descriptor (e.g. `"O(HH)"`) instead of atomic number.

    Parameters
    ----------
    species : str
        Bonded-species descriptor, as produced by
        :meth:`specter.pdb.PDB.get_atom_species` (e.g. `"O(HH)"`).
    k_xyz : torch.Tensor
        Distances from the atomic core in units of 1/Å. k^2 = kx^2 + ky^2 + kz^2.
    params : dict of str to torch.Tensor
        Species parameter table, as returned by
        :func:`load_shtyrov_species_parameters`.

    Returns
    -------
    potential : torch.Tensor
        Electron scattering factor in units of Å, same shape as `k_xyz`.
    """
    device = k_xyz.device
    P = params[species].to(device)  # shape (5, 2)

    a = P[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    b = P[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    k2 = k_xyz**2
    k2 = k2.unsqueeze(0)

    return torch.sum(a * torch.exp(-b * k2 / 4), 0)


def shtyrov_atomic_potential_3d_by_species(
    species: str,
    r_xyz: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute the 3D real-space atomic potential for a bonded species.

    Based on Shtyrov et al. (2026) Eq.18 -- see :func:`load_shtyrov_parameters`'s
    References section for the full citation. Same closed-form Gaussian-sum approach as
    :func:`shtyrov_atomic_potential_3d`, but keyed by bonded-species
    descriptor (e.g. `"O(HH)"`) instead of atomic number.

    Parameters
    ----------
    species : str
        Bonded-species descriptor, as produced by
        :meth:`specter.pdb.PDB.get_atom_species` (e.g. `"O(HH)"`).
    r_xyz : torch.Tensor
        3D grid of radial distances, in Å.
    params : dict of str to torch.Tensor
        Species parameter table, as returned by
        :func:`load_shtyrov_species_parameters`.

    Returns
    -------
    potential : torch.Tensor
        Atomic potential in units of V, same shape as `r_xyz`.
    """
    a0 = 0.529  # Bohr radius, [Å]
    e = 14.4  # electron charge, [V·Å]
    c1 = 2 * torch.pi * e * a0

    P = params[species].to(r_xyz.device)  # shape (5, 2)
    a = P[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    b = P[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).clamp(min=MIN_GAUSSIAN_B)

    r2 = (r_xyz**2).unsqueeze(0)

    return c1 * torch.sum(
        a * (4 * torch.pi / b) ** 1.5 * torch.exp(-(torch.pi**2) * r2 / b * 4), 0
    )


def peng_atomic_potential_3d(atomic_number: int, r_xyz: torch.Tensor) -> torch.Tensor:
    """
    Compute the 3D real-space atomic potential using gemmi's built-in
    standalone-atom electron scattering factors (Peng et al. 1996, `c4322`
    table -- see References).

    This is the same fallback sffit itself uses (`sffit fit.py::do_mmcif`)
    for any bonded species not covered by a fitted Shtyrov parameter table:
    a per-element, unbonded (independent-atom-model) scattering factor,
    evaluated with the same closed-form Gaussian-sum approach as
    :func:`shtyrov_atomic_potential_3d_by_species`, so it combines
    coherently (same units, same sign convention) with matched-species
    Shtyrov kernels in the same potential volume.

    Parameters
    ----------
    atomic_number : int
        Atomic number, Hydrogen has number 1.
    r_xyz : torch.Tensor
        3D grid of radial distances, in Å.

    Returns
    -------
    potential : torch.Tensor
        Atomic potential in units of V, same shape as `r_xyz`.

    References
    ----------
    Peng, L.-M., Ren, G., Dudarev, S. L., & Whelan, M. J. (1996). Robust
    parameterization of elastic and absorptive electron atomic scattering
    factors. Acta Crystallographica Section A, 52(2), 257–276.
    https://doi.org/10.1107/S0108767395014371
    """
    a0 = 0.529  # Bohr radius, [Å]
    e = 14.4  # electron charge, [V·Å]
    c1 = 2 * torch.pi * e * a0

    coef = gemmi.Element(atomic_number).c4322
    a = torch.tensor(coef.a, dtype=torch.float32, device=r_xyz.device).view(5, 1, 1, 1)
    b = (
        torch.tensor(coef.b, dtype=torch.float32, device=r_xyz.device)
        .view(5, 1, 1, 1)
        .clamp(min=MIN_GAUSSIAN_B)
    )

    r2 = (r_xyz**2).unsqueeze(0)

    return c1 * torch.sum(
        a * (4 * torch.pi / b) ** 1.5 * torch.exp(-(torch.pi**2) * r2 / b * 4), 0
    )


def yukawa_shell_average(
    p: torch.Tensor, R: torch.Tensor | float, lam: torch.Tensor
) -> torch.Tensor:
    """
    Exact volume-average of a Yukawa/screened-Coulomb term `exp(-lam*r)/r`
    over a sphere of radius `R`, at distance `p` from the singularity.

    Kirkland's and Lobato's real-space potentials both contain terms of this
    form (`a/r * exp(-2*pi*r*sqrt(b))` and similar), which have a genuine,
    unbounded singularity at r=0 — unlike a Gaussian's large-but-finite peak,
    this cannot be handled by the exact `erf` voxel-average approach used for
    Shtyrov/Peng (a Gaussian separates into independent x/y/z 1D integrals;
    `1/r` depends on the full radius and does not separate that way, and any
    coarse or naive point sample risks landing arbitrarily close to the true
    singularity — confirmed empirically to blow up to 10^5-10^7-times the
    correct value for some sub-voxel atom positions).

    This function instead computes the *exact* average of the term over a
    sphere approximating the voxel's volume (a standard "smeared/screened
    potential" technique from electrostatics/Ewald-summation literature):
    the spherical-shell mean of a radial function at an off-center point
    reduces, via the substitution u=|point-atom|, to a 1D integral that is
    finite everywhere (derived and verified with sympy: finite as `p->0`,
    continuous at `p=R`, reduces exactly to the point value `exp(-lam*p)/p`
    as `R->0`). Verified numerically against Monte Carlo cube-integration
    (random, non-grid-aligned sampling): ~2-3% relative error for the atom's
    own nearest voxel, growing to ~18-20% at that voxel's edge/corner (where
    approximating the cube by a sphere is least accurate) — a bounded,
    understood geometric approximation, categorically different from the
    unbounded blow-ups of point/coarse sampling.

    Parameters
    ----------
    p : torch.Tensor
        Distance from the singularity (atom) to the sphere's center, in Å.
    R : torch.Tensor or float
        Sphere radius, in Å — typically `h*(6/pi)**(1/3)` for a cubic voxel
        of half-width `h`, matching the voxel's volume.
    lam : torch.Tensor
        Decay rate of the term (`2*pi*sqrt(b)` for Kirkland's Yukawa term,
        `2*pi/sqrt(b)` for Lobato's).

    Returns
    -------
    torch.Tensor
        Volume-averaged value, same shape as the broadcast of `p` and `lam`.
    """
    # The near-field branch is mathematically finite as p->0 (verified via
    # sympy limit), but evaluating it at a *literal* tiny p in float32 hits
    # catastrophic cancellation (numerator and 1/p both blow up and only
    # cancel in exact arithmetic) — confirmed empirically: p=1e-8 gives a
    # value ~3x too large in float32, while p=1e-3 matches the float64
    # reference to <0.01%. 1e-3 Å is physically negligible (atoms essentially
    # never sit closer than this to a voxel center) but numerically safe.
    p = p.clamp(min=1e-3)
    # Two independent overflow risks, both handled below:
    # (1) torch.where evaluates *both* branches everywhere and backprops
    #     through both (grad_output is zeroed for the unselected branch, but
    #     0*inf=nan still corrupts the sum): `near`'s exp(lam*(R+p)) grows
    #     unboundedly as p->inf (window-edge points far outside its p<=R
    #     domain), and `far`'s /p term blows up as p->0 (outside its p>R
    #     domain). Fixed by clamping each branch's own p input to its safe
    #     domain so neither branch ever evaluates out of range.
    # (2) The textbook far/near forms each contain a bare exp(2*R*lam) factor
    #     that is *combined* with a decaying exp(-lam*(...)) factor down-
    #     stream — mathematically the product is always bounded, but
    #     evaluating exp(2*R*lam) as its own intermediate value overflows in
    #     float32 for fast-decaying terms (confirmed for e.g. Kirkland
    #     sulfur's b=167 Yukawa term, lam~81/A, 2*R*lam~101 -> exp(101)=inf)
    #     even though the true combined value is finite and tiny. Fixed by
    #     algebraically pre-combining exponents (verified equal to the
    #     original via sympy) so every exponent actually evaluated is <=0
    #     within that branch's clamped domain, regardless of lam.
    p_far = p.clamp(min=R)
    p_near = p.clamp(max=R)
    far = (
        3
        * (
            (R * lam - 1) * torch.exp(lam * (R - p_far))
            + (R * lam + 1) * torch.exp(-lam * (R + p_far))
        )
        / (2 * R**3 * lam**3 * p_far)
    )
    near = (
        3
        * (
            2 * lam * p_near
            + (R * lam + 1) * torch.exp(-lam * (R + p_near))
            - (R * lam + 1) * torch.exp(lam * (p_near - R))
        )
        / (2 * R**3 * lam**3 * p_near)
    )
    return torch.where(p > R, far, near)


def plain_exp_shell_average(
    p: torch.Tensor, R: torch.Tensor | float, lam: torch.Tensor
) -> torch.Tensor:
    """
    Exact volume-average of a plain exponential term `exp(-lam*r)` (no
    `1/r` factor, no singularity) over a sphere of radius `R` at distance
    `p` from the sphere's reference point.

    Companion to :func:`yukawa_shell_average` for Lobato's real-space term
    `a/b^1.5 * (sqrt(b)/(pi*r) + 1) * exp(-2*pi*r/sqrt(b))`, which splits
    exactly into `(a/(pi*b)) * yukawa_shell_average(...)` (the `1/r` part)
    plus `(a/b**1.5) * plain_exp_shell_average(...)` (the `+1` part). This
    term has no true singularity, but is still radially symmetric (not
    separable into independent x/y/z integrals), so it uses the same
    shell-averaging derivation. Derived and verified the same way (sympy:
    finite at `p=0`, continuous at `p=R`, reduces to `exp(-lam*p)` as
    `R->0`; Monte Carlo cube-integration: ~3% typical, ~10% worst-case at
    the voxel edge — better-behaved than the singular term, as expected).

    Parameters
    ----------
    p : torch.Tensor
        Distance from the origin to the sphere's center, in Å.
    R : torch.Tensor or float
        Sphere radius, in Å.
    lam : torch.Tensor
        Decay rate of the term.

    Returns
    -------
    torch.Tensor
        Volume-averaged value, same shape as the broadcast of `p` and `lam`.
    """
    # The near-field branch is mathematically finite as p->0 (verified via
    # sympy limit), but evaluating it at a *literal* tiny p in float32 hits
    # catastrophic cancellation (numerator and 1/p both blow up and only
    # cancel in exact arithmetic) — confirmed empirically: p=1e-8 gives a
    # value ~3x too large in float32, while p=1e-3 matches the float64
    # reference to <0.01%. 1e-3 Å is physically negligible (atoms essentially
    # never sit closer than this to a voxel center) but numerically safe.
    p = p.clamp(min=1e-3)
    # Same two overflow risks as yukawa_shell_average (see its comments for
    # the full explanation): (1) torch.where computing both branches
    # everywhere, with near's exp(2*lam*p)/exp(lam*(R+p)) blowing up at
    # window-edge p far outside its p<=R domain, and far's /p blowing up as
    # p->0 outside its p>R domain — fixed by clamping each branch's own p
    # input to its safe domain; (2) the textbook forms' bare exp(2*R*lam)
    # factor overflows in float32 for fast-decaying terms even though it's
    # always combined with a decaying exp(-lam*(...)) factor into a finite
    # product — fixed by algebraically pre-combining exponents (verified
    # equal to the original via sympy; the near branch's 4*lam*p*exp(lam*(R+p))
    # term combines with the outer exp(-lam*(R+p)) to cancel exactly to
    # 4*lam*p, no exponential at all) so every exponent evaluated is <=0.
    p_far = p.clamp(min=R)
    p_near = p.clamp(max=R)
    far = (
        3
        * (
            (R**2 * lam**2 + R * lam**2 * p_far + 3 * R * lam + lam * p_far + 3)
            * torch.exp(-lam * (R + p_far))
            + (-(R**2) * lam**2 + R * lam**2 * p_far + 3 * R * lam - lam * p_far - 3)
            * torch.exp(lam * (R - p_far))
        )
        / (2 * R**3 * lam**4 * p_far)
    )
    near = (
        3
        * (
            (-(R**2) * lam**2 + R * lam**2 * p_near - 3 * R * lam + lam * p_near - 3)
            * torch.exp(lam * (p_near - R))
            + (R**2 * lam**2 + R * lam**2 * p_near + 3 * R * lam + lam * p_near + 3)
            * torch.exp(-lam * (R + p_near))
            + 4 * lam * p_near
        )
        / (2 * R**3 * lam**4 * p_near)
    )
    return torch.where(p > R, far, near)
