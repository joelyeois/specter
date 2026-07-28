"""
Shared, stateless physics-kernel construction for ice generation.

These functions used to be duplicated (with minor variations) across the
individual icemaker classes — see the individual docstrings below for what
each one replaces. Keeping them here as pure functions means every algorithm
class builds these kernels the same way, by construction, instead of by
convention.
"""

from __future__ import annotations

import os
from importlib import resources

import torch
from torchinterp1d import interp1d

from .. import potential
from ..arrays import (
    radial_grid_3d,
    radial_profile_3d,
    soft_voxelize_coordinates,
)
from ..atom import (
    kirkland_atomic_potential_3d,
    load_shtyrov_species_parameters,
    lobato_atomic_potential_3d,
    shtyrov_atomic_potential_3d_by_species,
)
from ..fft import fft3

# Grid the bundled (and any custom) mdsim radial-average target files are
# computed on. Not user-configurable: it describes the fixed grid a target
# .pt file was generated on, not something independently choosable per call.
MDSIM_DX = 0.25
MDSIM_N = 400

# Real extent of the LDA-80K MD simulation's periodic cell (see the dump
# file's own BOX BOUNDS header): ~127.12 A per side. A native target can
# only be computed directly up to this size before requiring the box-size
# extrapolation handled by interpolate_target_kernel's own interp1d call
# (safe -- box size only changes k-sampling density of the same underlying
# bulk curve) -- see compute_native_target. Kept conservative relative to
# the true ~127A limit since 100A is what was empirically validated.
_REFERENCE_MAX_BOX = 100.0
_REFERENCE_COORDS_CACHE: dict[str, torch.Tensor] = {}


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


def compute_native_target(
    n: int,
    dx: float,
    nz: int | None = None,
    reference_coords_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a radial-average |F(k)| target natively at the requested grid
    resolution, instead of interpolating the bundled 400x400x400, dx=0.25 Å
    default across a mismatched ``dx``.

    Matching ``dx`` between the target and the training grid matters far
    more than matching absolute box size: soft-voxelizing coordinates at a
    coarse ``dx`` is a lossy, aliasing discretization (high-frequency
    structure folds back into the representable k-range rather than being
    cleanly discarded), so the *simulated* radial |F(k)| computed during
    training is systematically biased relative to a target that was
    computed on a much finer grid. Comparing a biased quantity against an
    unbiased one creates a mismatch no amount of atom repositioning can
    close, which is what makes coarse-``dx`` training converge poorly
    against the bundled fine-grid target. Computing the target through the
    *same* voxelize → FFT → radial-average pipeline at the *same* ``dx``
    removes that bias (validated: dx=0.5/1.0/2.0 all went from stuck losses
    of O(1)-O(1000) to O(1e-5)-O(1e-7) once matched).

    Box size, by contrast, is safe to extrapolate: for a bulk material the
    true S(k) shape barely depends on sample size once it's large enough,
    so a box bigger than the reference simulation's real periodic cell
    (~127 Å for the bundled LDA-80K trajectory) is handled by capping the
    native computation at ``_REFERENCE_MAX_BOX`` and letting
    :func:`interpolate_target_kernel`'s existing ``interp1d`` call resample
    onto the requested grid's own (denser) k-values -- Nyquist depends only
    on ``dx``, not box size, so this never needs to extrapolate beyond the
    native computation's own k-range, only resample it more finely.

    Parameters
    ----------
    n : int
        Number of voxels along x and y.
    dx : float
        Voxel size in Å.
    nz : int, optional
        Number of voxels along z. Defaults to ``n``.
    reference_coords_path : str, optional
        Path to a pre-extracted single-frame coordinate tensor (shape
        ``(N, 3)``, centered, full simulation box). Defaults to the bundled
        ``ice-data/lda_80k_frame799_full_coords.pt`` (frame 799 of the
        LDA-80K trajectory -- single-frame vs. 790-frame-averaged targets
        were validated to give statistically indistinguishable downstream
        training results, so a single frame is used for speed).

    Returns
    -------
    mdsim_radial_k : torch.Tensor
        k-axis (1/Å) matching ``mdsim_f_radial_avg``'s bins.
    mdsim_f_radial_avg : torch.Tensor
        Radial-average |F(k)| (stores sqrt(S(k))), same convention as
        :func:`load_mdsim_f_radial_avg` -- a drop-in replacement for it.
    """
    nz = n if nz is None else nz
    if reference_coords_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        reference_coords_path = os.path.join(
            root_dir, "ice-data", "lda_80k_frame799_full_coords.pt"
        )

    if reference_coords_path not in _REFERENCE_COORDS_CACHE:
        _REFERENCE_COORDS_CACHE[reference_coords_path] = torch.load(
            reference_coords_path, weights_only=True
        )
    coords = _REFERENCE_COORDS_CACHE[reference_coords_path]

    box_xy = n * dx
    box_z = nz * dx
    trim_xy = min(box_xy, _REFERENCE_MAX_BOX)
    trim_z = min(box_z, _REFERENCE_MAX_BOX)

    half_xy, half_z = trim_xy / 2.0, trim_z / 2.0
    mask = (
        (coords[:, 0].abs() <= half_xy)
        & (coords[:, 1].abs() <= half_xy)
        & (coords[:, 2].abs() <= half_z)
    )
    trimmed = coords[mask]
    if trimmed.shape[0] == 0:
        raise ValueError(
            f"No reference atoms found within the trimmed box (n={n}, nz={nz}, "
            f"dx={dx}); check reference_coords_path."
        )

    # If the requested box exceeds the reference's real extent, this scales
    # down to the equivalent voxel count at the SAME dx and capped box size
    # (n_native * dx == trim_xy exactly), so the native computation stays at
    # the correct dx -- interpolate_target_kernel resamples the rest.
    n_native = max(1, round(n * trim_xy / box_xy))
    nz_native = max(1, round(nz * trim_z / box_z))

    vox = soft_voxelize_coordinates(
        trimmed, grid_shape=(nz_native, n_native, n_native), voxel_size=dx
    )
    amplitude = torch.abs(fft3(vox, shift=True)) / (trimmed.shape[0] ** 0.5)

    mdsim_dk = 1 / n_native / dx
    if n_native == nz_native:
        # radial_profile_3d's voxel-index binning coincides exactly with
        # physical |k|-magnitude binning when the grid is cubic (see the
        # anisotropic branch below for why they diverge otherwise) -- kept
        # as the fast path since it avoids rebuilding the k-grid.
        _, mdsim_f_radial_avg = radial_profile_3d(amplitude, return_r=True)
    else:
        # n_native != nz_native happens whenever the requested (n, nz) is
        # anisotropic and at least one axis stays under _REFERENCE_MAX_BOX
        # (e.g. n=64, nz=32 -- both under the cap, no rescaling, genuinely
        # anisotropic; distinct from e.g. n=256, nz=128, which both get
        # capped to the same 100 A limit and so end up accidentally cubic).
        # radial_profile_3d bins by voxel-INDEX distance from center, which
        # only equals physical |k| distance when voxel spacing is the same
        # along every axis -- for an anisotropic grid a voxel step along the
        # shorter axis spans a coarser physical k-spacing than a step along
        # the longer axes, so voxel-index bins silently mislabel the target's
        # own k-axis (validated: peak amplitude off by ~30%, peak position
        # shifted, and a pure binning-artifact falloff near the shorter
        # axis's Nyquist edge). Bin by physical |k|/dk instead, matching
        # exactly how GradientSKIcemaker.__init__ bins the training target
        # and the simulated |F(k)| during optimisation.
        k_native = ice_kspace_radial_grid(n_native, nz_native, dx)
        r_bins = (k_native / mdsim_dk).round().long().flatten()
        n_rbins = int(r_bins.max().item()) + 1
        bin_count = torch.bincount(r_bins, minlength=n_rbins).float().clamp(min=1)
        bin_sum = torch.bincount(r_bins, weights=amplitude.flatten(), minlength=n_rbins)
        mdsim_f_radial_avg = bin_sum / bin_count

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
) -> torch.Tensor:
    """
    Interpolate the 1D MD-simulation radial-average target onto a 3D k-grid.

    Parameters
    ----------
    K : torch.Tensor
        DC-centered 3D k-space radial magnitude grid, shape ``(nz, n, n)`` — see
        :func:`ice_kspace_radial_grid`. Must be radially symmetric (a function of
        ``|K|`` only).
    mdsim_radial_k, mdsim_f_radial_avg : torch.Tensor
        Target radial profile — see :func:`load_mdsim_f_radial_avg`.
    n_ice_molecules : float
        Number of ice molecules in the target volume; scales the interpolated
        amplitude and sets the DC (center) term.

    Returns
    -------
    torch.Tensor
        Full kernel, shape ``K.shape``.
    """
    nz, n, _ = K.shape
    interp = interp1d(mdsim_radial_k[1:], mdsim_f_radial_avg[1:], K.ravel())
    kernel = interp.reshape(nz, n, n) * (n_ice_molecules**0.5)
    kernel[nz // 2, n // 2, n // 2] = n_ice_molecules
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
        # Water oxygen is always the 'O(HH)' bonded species.
        species_path = resources.files("specter.atom_data").joinpath("params_cat.json")
        with resources.as_file(species_path) as fpath:
            species_params = load_shtyrov_species_parameters(str(fpath))
        pot = shtyrov_atomic_potential_3d_by_species("O(HH)", sR, species_params)
    else:
        raise ValueError(
            f"Unknown parameterization '{parameterization}'. "
            "Choose 'kirkland', 'lobato', or 'shtyrov'."
        )

    return avgpool3d(pot[None, None]).squeeze()
