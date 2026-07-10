"""
Shared, stateless physics-kernel construction for ice generation.

These functions used to be duplicated (with minor variations) across
``APIcemaker``, ``GradientSKIcemaker``, and ``RandomIcemaker`` — see the
individual docstrings below for what each one replaces. Keeping them here as
pure functions means every algorithm class builds these kernels the same way,
by construction, instead of by convention.
"""

from __future__ import annotations

import os

import torch
from torchinterp1d import interp1d

from .. import potential
from ..arrays import radial_grid_3d, real_to_kgrid_3d
from ..atom import kirkland_atomic_potential_3d, lobato_atomic_potential_3d
from ..fft import fft3

# Grid the bundled (and any custom) mdsim radial-average target files are
# computed on. Not user-configurable: it describes the fixed grid a target
# .pt file was generated on, not something independently choosable per call.
MDSIM_DX = 0.25
MDSIM_N = 400


def load_mdsim_f_radial_avg(
    saved_data_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load the bundled (or custom) MD-simulation radial-average |F(k)| target.

    Parameters
    ----------
    saved_data_path : str, optional
        Path to a precomputed radial-average |F(k)| target ``.pt`` file, in the
        same format as the bundled default (a 1D tensor indexed by k-bin on the
        fixed ``MDSIM_N`` x ``MDSIM_N`` x ``MDSIM_N``, ``MDSIM_DX`` grid). If
        None, uses the bundled ``ice-data/mdsim_f_radial_avg_400x400x400_0.25A.pt``.

    Returns
    -------
    mdsim_radial_k : torch.Tensor
        k-axis (1/Å) matching ``mdsim_f_radial_avg``'s bins.
    mdsim_f_radial_avg : torch.Tensor
        Radial-average |F(k)| (stores sqrt(S(k))).
    """
    if saved_data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        saved_data_path = os.path.join(
            root_dir, "ice-data", "mdsim_f_radial_avg_400x400x400_0.25A.pt"
        )
    mdsim_f_radial_avg = torch.load(saved_data_path, weights_only=True)
    mdsim_dk = 1 / MDSIM_N / MDSIM_DX
    mdsim_radial_k = torch.arange(len(mdsim_f_radial_avg)) * mdsim_dk
    return mdsim_radial_k, mdsim_f_radial_avg


def ice_kspace_radial_grid(
    n: int, nz: int, dx: float, device: str | torch.device | None = None
) -> torch.Tensor:
    """
    Build the DC-centered 3D k-space radial magnitude grid.

    Parameters
    ----------
    n : int
        Number of voxels along x and y.
    nz : int
        Number of voxels along z.
    dx : float
        Voxel size in Å (isotropic).
    device : str or torch.device, optional
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        ``|K| = sqrt(kx^2+ky^2+kz^2)``, shape ``(nz, n, n)``, fftshifted (DC at
        the center voxel).
    """
    kx = torch.fft.fftshift(torch.fft.fftfreq(n, dx, device=device))
    ky = kx
    kz = torch.fft.fftshift(torch.fft.fftfreq(nz, dx, device=device))
    KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
    return torch.sqrt(KX**2 + KY**2 + KZ**2)


def interpolate_target_kernel(
    K: torch.Tensor,
    mdsim_radial_k: torch.Tensor,
    mdsim_f_radial_avg: torch.Tensor,
    n_ice_molecules: float,
    half: bool = False,
) -> torch.Tensor:
    """
    Interpolate the 1D MD-simulation radial-average target onto a 3D k-grid.

    Parameters
    ----------
    K : torch.Tensor
        DC-centered 3D k-space radial magnitude grid, shape ``(nz, n, n)`` — see
        :func:`ice_kspace_radial_grid`. Must be radially symmetric (a function of
        ``|K|`` only) for the ``half=True`` half-kernel to be correct (see Notes).
    mdsim_radial_k, mdsim_f_radial_avg : torch.Tensor
        Target radial profile — see :func:`load_mdsim_f_radial_avg`.
    n_ice_molecules : float
        Number of ice molecules in the target volume; scales the interpolated
        amplitude and sets the DC (center) term.
    half : bool, optional
        If True, return the rfftn half-kernel view instead of the full kernel.

    Returns
    -------
    torch.Tensor
        Full kernel, shape ``K.shape``, or (if ``half``) the half-kernel, shape
        ``(nz, n, n // 2 + 1)``.

    Notes
    -----
    ``half=True`` builds the half-kernel by flipping the first half of the fully
    *centered* kernel along the x-axis, rather than building a second,
    separately x-unshifted K-grid. This is only valid because the kernel is
    radially symmetric (``f(-kx) == f(kx)``, since it depends only on ``|K|``):
    for such a kernel, the centered array's first half (most-negative-frequency
    through DC) is the mirror image of the unshifted rfftn half (DC through
    most-positive-frequency), so flipping one gives the other exactly.
    ``APIcemaker.irfftn`` expects precisely this x-axis-unshifted /
    z,y-axes-centered half-kernel convention (it only ``ifftshift``s the z, y
    axes before calling ``torch.fft.irfftn``).
    """
    nz, n, _ = K.shape
    interp = interp1d(mdsim_radial_k[1:], mdsim_f_radial_avg[1:], K.ravel())
    kernel = interp.reshape(nz, n, n) * (n_ice_molecules**0.5)
    kernel[nz // 2, n // 2, n // 2] = n_ice_molecules
    if half:
        return torch.flip(kernel[:, :, : n // 2 + 1], dims=[2])
    return kernel


def build_atomic_potential_kernel(
    dx: float, parameterization: str = "kirkland"
) -> torch.Tensor:
    """
    Build the real-space atomic potential kernel for a water molecule (oxygen).

    Parameters
    ----------
    dx : float
        Voxel size in Å.
    parameterization : str, optional
        ``'kirkland'`` (default), ``'lobato'``, or ``'shtyrov'``.

    Returns
    -------
    torch.Tensor
        Potential kernel volume, downsampled to the target grid.
    """
    ssn, ssdx, ssf = potential.compute_supersampling_parameters(dx)
    # set original convention to torch to avoid singularity at origin.
    sR = radial_grid_3d(ssn, ssdx, convention="torch")

    # for binning super-sampled grids to main volume grid.
    avgpool3d = torch.nn.AvgPool3d(ssf, stride=ssf)

    if parameterization == "kirkland":
        pot = kirkland_atomic_potential_3d(8, sR)
    elif parameterization == "lobato":
        pot = lobato_atomic_potential_3d(8, sR)
    elif parameterization == "shtyrov":
        # from params_cat.json, 'O(HH)'
        params = torch.tensor(
            [
                [0.3131, 0.8722],
                [0.8102, 4.9669],
                [0.9812, 14.1666],
                [-0.5997, 64.1638],
                [-0.1519, 121.3711],
            ]
        )
        # Separate columns: a_i, b_i
        a = params[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape (3,1,1,1)
        b = params[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        k_xyz = real_to_kgrid_3d(sR)
        k2 = k_xyz**2
        k2 = k2.unsqueeze(0)  # shape (1, Nx, Ny, Nz)

        s1_f = torch.sum(a * torch.exp(-b * k2 / 4), 0)
        dkx = k_xyz[1, 0, 0] - k_xyz[0, 0, 0]
        dky = k_xyz[0, 1, 0] - k_xyz[0, 0, 0]
        dkz = k_xyz[0, 0, 1] - k_xyz[0, 0, 0]
        pot = -torch.abs(fft3(s1_f, shift=True)) * dkx * dky * dkz  # need to negate
    else:
        raise ValueError(
            f"Unknown parameterization '{parameterization}'. "
            "Choose 'kirkland', 'lobato', or 'shtyrov'."
        )

    return avgpool3d(pot[None, None]).squeeze()
