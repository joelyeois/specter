from __future__ import annotations

import functools
import multiprocessing as mp
import warnings
from importlib import resources
from typing import Any, Sequence

import gemmi
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from .progress import TqdmProgress, track

from .arrays import (
    radial_grid_2d,
    radial_grid_3d,
    soft_voxelize_coordinates,
    soft_voxelize_xy_coordinates,
)
from .atom import (
    MIN_GAUSSIAN_B,
    atom_symbol,
    kirkland_atomic_potential_2d,
    kirkland_atomic_potential_3d,
    load_kirkland_parameters,
    load_lobato_parameters,
    load_shtyrov_species_parameters,
    lobato_atomic_potential_2d,
    lobato_atomic_potential_3d,
    peng_atomic_potential_3d,
    plain_exp_shell_average,
    shtyrov_atomic_potential_3d,
    shtyrov_atomic_potential_3d_by_species,
    yukawa_shell_average,
)
from .fft import fftconvolve


def compute_supersampling_parameters(
    dx: float, width_atom: float = 5.0, dx_atom: float = 0.1
) -> tuple[int, float, int]:
    """
    Compute grid size, pixel spacing and supersampling factor for atomic potentials.

    Ensures the grid is fine enough to sample the potential (default 0.1 Å/pixel)
    and wide enough to cover the 'size' of an atom (default 5Å). Can be downsampled
    later using integer stride.

    Parameters
    ----------
    dx : float
        Desired final pixel size (Å) of the main volume.
    width_atom : float
        Size of an atom in Å.
    dx_atom : float
        Pixel size to sample the potential of an atom.

    Returns
    -------
    n_atom : int
        Number of pixels along each axis in the supersampled grid.
    ss_dx : float
        Pixel size of the supersampled grid.
    ssf : int
        Supersampling factor
    """
    if dx <= 0 or width_atom <= 0 or dx_atom <= 0:
        raise ValueError("dx, width_atom, and dx_atom must be > 0.")
    if dx <= dx_atom:
        ssf = 1
        ss_dx = dx
        n_atom = int(width_atom / dx)
        n_atom = max(n_atom, 3)
        # make even
        n_atom = n_atom + (n_atom % 2)
        return n_atom, ss_dx, ssf

    # Number of pixels at atom sampling
    n_atom = int(torch.ceil(torch.as_tensor(width_atom / dx_atom)))

    # Step 1: make divisible by ssf
    ssf = int(torch.round(torch.as_tensor(dx / dx_atom)))
    ss_dx = dx / ssf

    # Ensure pooled kernel has at least 3 pixels per axis.
    n_atom = max(n_atom, 3 * ssf)

    # Step 2: adjust n_atom to satisfy both evenness and divisibility
    # find the smallest even number divisible by ssf and >= n_atom
    while (n_atom % ssf != 0) or (n_atom % 2 != 0):
        n_atom += 1

    return n_atom, ss_dx, ssf


def build_potential_volume_fftconvolve_3d(
    atomic_numbers: torch.Tensor,
    centered_coords: torch.Tensor,
    n_xyz: int | Sequence[int],
    dx: float,
    disable_tqdm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a 3D potential volume by convolving atomic kernels with voxelized positions.

    Computes potentials for each unique element on a supersampled grid, then bins
    down to the main volume resolution via average pooling.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        1D tensor of atomic numbers. Hydrogen is 1.
    centered_coords : torch.Tensor
        Atomic xyz coordinates, shape (N, 3), centered at origin.
    n_xyz : int or sequence of int
        Grid size (nx, ny, nz). If int, assumes cubic grid.
    dx : float
        Voxel size of main volume in Å.
    disable_tqdm : bool, optional
        Disable progress bar. Default is False.

    Returns
    -------
    potential_volume : torch.Tensor
        Sampled potential volume, shape (nz, ny, nx).
    sR : torch.Tensor
        Radial coordinate grid used for atomic potential sampling.
    """
    if isinstance(n_xyz, int):
        nx = ny = nz = n_xyz
    else:
        nx, ny, nz = n_xyz

    # create super-sampled (ss) coordinate system
    ssn, ssdx, ssf = compute_supersampling_parameters(dx)
    # torch convention avoids singularity at origin
    sR = radial_grid_3d(ssn, ssdx, convention="torch")

    avgpool3d = torch.nn.AvgPool3d(ssf, stride=ssf)
    potential_volume = torch.zeros(nz, ny, nx)

    for elem in track(
        torch.unique(atomic_numbers),
        description="Building elements",
        disable=disable_tqdm,
    ):
        atomic_indices = torch.squeeze(torch.argwhere(atomic_numbers == elem))

        # soft_voxelize_coordinates is differentiable w.r.t. coordinates
        temp_vol = soft_voxelize_coordinates(
            centered_coords[atomic_indices].reshape(-1, 3),
            grid_shape=(nz, ny, nx),
            voxel_size=dx,
        )

        pot = kirkland_atomic_potential_3d(int(elem), sR)

        if ssf != 1:
            pot = avgpool3d(pot[None, None]).squeeze(0).squeeze(0)
        potential_volume += fftconvolve(temp_vol, pot, mode="same")
    return potential_volume, sR


def build_potential_volume_fftconvolve_2d(
    atomic_numbers: torch.Tensor,
    centered_coords: torch.Tensor,
    n_xyz: int | Sequence[int],
    dx: float,
    disable_tqdm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a 3D potential volume using 2D projected atomic kernels.

    Projects atoms onto XY slices and convolves with 2D atomic potential kernels.
    Faster than the 3D variant but less accurate in Z.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        1D tensor of atomic numbers. Hydrogen is 1.
    centered_coords : torch.Tensor
        Atomic xyz coordinates, shape (N, 3), centered at origin.
    n_xyz : int or sequence of int
        Grid size (nx, ny, nz). Assumes nx = ny. If int, assumes cubic grid.
    dx : float
        Pixel size of main volume in Å (assumes dx = dy).
    disable_tqdm : bool, optional
        Disable progress bar. Default is False.

    Returns
    -------
    potential_volume : torch.Tensor
        Sampled potential volume, shape (nz, ny, nx).
    sR : torch.Tensor
        Radial coordinate grid used for atomic potential sampling.
    """
    if isinstance(n_xyz, int):
        nx = ny = nz = n_xyz
    else:
        nx, ny, nz = n_xyz

    ssn, ssdx, ssf = compute_supersampling_parameters(dx)
    sR = radial_grid_2d(ssn, ssdx, convention="torch")
    avgpool2d = torch.nn.AvgPool2d(ssf, stride=ssf)
    potential_volume = torch.zeros(nz, ny, nx)

    for elem in track(
        torch.unique(atomic_numbers),
        description="Building elements",
        disable=disable_tqdm,
    ):
        atomic_indices = torch.squeeze(torch.argwhere(atomic_numbers == elem))

        # soft_voxelize_xy_coordinates is differentiable w.r.t. coordinates
        temp_vol = soft_voxelize_xy_coordinates(
            centered_coords[atomic_indices].reshape(-1, 3),
            grid_shape=(nz, ny, nx),
            voxel_size=dx,
        )

        pot = kirkland_atomic_potential_2d(int(elem), sR)

        if ssf != 1:
            pot = avgpool2d(pot[None, None]).squeeze(0).squeeze(0)

        temp_vol_b = temp_vol.unsqueeze(1)  # (nz, 1, ny, nx)
        pot_b = pot.unsqueeze(0).unsqueeze(0)  # (1, 1, ky, kx)
        convolved = F.conv2d(temp_vol_b, pot_b, padding="same")
        potential_volume += convolved.squeeze(1)  # (nz, ny, nx)
    return potential_volume, sR


def _local_window_geometry(
    coords: torch.Tensor,
    grid_shape: tuple[int, int, int],
    dx: float,
    rcut: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shared local-window geometry for the analytic scatter-add potential
    builders (Shtyrov/Peng/Kirkland/Lobato): nearest voxel per atom, the
    continuous sub-voxel offset from it to the atom's true position, and
    the shared (atom-independent) window of voxel offsets.

    Parameters
    ----------
    coords : torch.Tensor
        Atom coordinates, shape (N, 3), same centered-origin convention as
        `soft_voxelize_coordinates`.
    grid_shape : tuple of int
        Output volume shape (nz, ny, nx).
    dx : float
        Voxel size, in Å.
    rcut : float
        Radius (Å) of the local window evaluated around each atom.

    Returns
    -------
    center_idx : torch.Tensor
        Nearest voxel index per atom, shape (N, 3), long, detached.
    frac_offset : torch.Tensor
        Continuous sub-voxel offset from `center_idx` to the atom's true
        position (voxel units), shape (N, 3). Differentiable w.r.t. `coords`.
    offsets_vox : torch.Tensor
        Window offsets shared by every atom (voxel units), shape (w, w, w, 3).
    """
    device = coords.device
    dtype = coords.dtype
    nz, ny, nx = grid_shape
    rcut_vox = int(torch.ceil(torch.tensor(rcut / dx)).item())

    coords_voxel = coords / dx
    origin = torch.tensor([nx // 2, ny // 2, nz // 2], device=device, dtype=dtype)
    coords_voxel_centered = coords_voxel + origin
    coords_voxel_centered = coords_voxel_centered[..., [2, 1, 0]]  # (x,y,z)->(z,y,x)
    center_idx = torch.round(coords_voxel_centered).long().detach()
    frac_offset = coords_voxel_centered - center_idx.to(dtype)

    lin = torch.arange(-rcut_vox, rcut_vox + 1, device=device, dtype=dtype)
    offsets_vox = torch.stack(
        torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1
    )  # (w,w,w,3)

    return center_idx, frac_offset, offsets_vox


def _scatter_window_values(
    vals: torch.Tensor,
    center_idx: torch.Tensor,
    offsets_vox: torch.Tensor,
    grid_shape: tuple[int, int, int],
) -> torch.Tensor:
    """
    Scatter-add per-atom, per-window-voxel values into a full volume,
    clipping (not wrapping) windows that partially fall outside the grid.

    Parameters
    ----------
    vals : torch.Tensor
        Values to scatter, shape (N, w, w, w).
    center_idx : torch.Tensor
        Nearest voxel index per atom, shape (N, 3), long.
    offsets_vox : torch.Tensor
        Window offsets shared by every atom, shape (w, w, w, 3).
    grid_shape : tuple of int
        Output volume shape (nz, ny, nx).

    Returns
    -------
    potential_volume : torch.Tensor
        Volume, shape `grid_shape`.
    """
    nz, ny, nx = grid_shape
    device = vals.device
    dtype = vals.dtype

    abs_idx = center_idx[:, None, None, None, :] + offsets_vox[None].long()
    valid = (
        (abs_idx[..., 0] >= 0)
        & (abs_idx[..., 0] < nz)
        & (abs_idx[..., 1] >= 0)
        & (abs_idx[..., 1] < ny)
        & (abs_idx[..., 2] >= 0)
        & (abs_idx[..., 2] < nx)
    )
    flat_idx = (
        abs_idx[..., 0].clamp(0, nz - 1) * ny * nx
        + abs_idx[..., 1].clamp(0, ny - 1) * nx
        + abs_idx[..., 2].clamp(0, nx - 1)
    )

    potential_volume = torch.zeros(nz * ny * nx, device=device, dtype=dtype)
    potential_volume = potential_volume.scatter_add(0, flat_idx[valid], vals[valid])
    return potential_volume.view(nz, ny, nx)


def build_potential_volume_analytic_scatter(
    coords: torch.Tensor,
    a_coefs: torch.Tensor,
    b_coefs: torch.Tensor,
    grid_shape: tuple[int, int, int],
    dx: float,
    rcut: float = 5.0,
) -> torch.Tensor:
    """
    Build a potential volume by analytically evaluating each atom's
    closed-form Gaussian-sum potential, *exactly averaged over each voxel's
    volume*, in a small local window around the atom, then scatter-adding
    into the full volume.

    Unlike `build_potential_volume_fftconvolve_3d` (splat atom onto the grid,
    then FFT-convolve a shared, precomputed, downsampled kernel per element
    group), this needs no FFT, no kernel precomputation, no box-size/aliasing
    dependence, and works with genuinely per-atom coefficients (no need to
    group atoms sharing a kernel).

    Critically, this evaluates the *exact voxel average* of the potential,
    not a point sample at the nearest grid point. A point sample would be
    wildly sensitive to exactly where an atom sits relative to the grid —
    the underlying potential is sharply peaked (near-singular) at the atom
    center, so point-sampling can swing the reported peak value by over an
    order of magnitude depending on sub-voxel position alone, with no
    physical meaning (confirmed empirically: ~26x swing for one atom moved
    by half a voxel). Voxel-averaging is exactly what
    `compute_supersampling_parameters` + `AvgPool3d` already achieve for
    Kirkland/Lobato/the grouped Shtyrov kernels, by finely supersampling and
    averaging down — this function gets the same effect in closed form
    instead: a 3D isotropic Gaussian separates into independent x/y/z 1D
    Gaussians, each with an exact antiderivative (the error function), so
    the exact average of `sqrt(4*pi/b)*exp(-4*pi^2*x^2/b)` over
    `[x0-h, x0+h]` is `(erf(2*pi/sqrt(b)*(x0+h)) - erf(2*pi/sqrt(b)*(x0-h))) / (4h)`
    — verified against brute-force numerical quadrature to <0.001%. The full
    3D voxel average is the product of this applied independently to x, y, z.

    Differentiable w.r.t. `coords`: gradients flow through the continuous
    per-window offset used in `erf`. Only the discrete choice of *which*
    voxels a given atom's window covers is non-differentiable (via
    `round()`/integer indexing) — the same property `soft_voxelize_coordinates`
    already has for its nearest-8-voxels splat.

    Parameters
    ----------
    coords : torch.Tensor
        Atom coordinates, shape (N, 3), in the same centered-origin
        convention as `soft_voxelize_coordinates` (physical (0,0,0) maps to
        voxel index `[nz//2, ny//2, nx//2]`).
    a_coefs, b_coefs : torch.Tensor
        Per-atom Gaussian coefficients, shape (N, 5) each — from a matched
        Shtyrov species (`load_shtyrov_species_parameters`) or the Peng
        `gemmi.Element(z).c4322` fallback, looked up per atom by the caller.
    grid_shape : tuple of int
        Output volume shape (nz, ny, nx).
    dx : float
        Voxel size, in Å.
    rcut : float, optional
        Radius (Å) of the local window evaluated around each atom. Default
        5.0 Å, matching sffit's own `--rcut` convention (its default of 10 Å
        resolves to a 5 Å radius) — verified to hold every bundled Shtyrov
        species and Peng/gemmi element to well under 0.05% of peak value at
        this radius.

    Returns
    -------
    potential_volume : torch.Tensor
        Potential volume, shape `grid_shape`, in units of V·Å.
    """
    a0 = 0.529  # Bohr radius, [Å]
    e = 14.4  # electron charge, [V·Å]
    c1 = 2 * torch.pi * e * a0

    b_coefs = b_coefs.clamp(min=MIN_GAUSSIAN_B)
    h = dx / 2  # half voxel width, for the voxel-average integration bounds

    center_idx, frac_offset, offsets_vox = _local_window_geometry(
        coords, grid_shape, dx, rcut
    )

    # Continuous physical offset from each atom to each of its window
    # voxels' centers, per axis (kept separate for the erf voxel-average —
    # not squared/summed to r2 the way a point-sample formula would).
    rel_vox = offsets_vox[None] - frac_offset[:, None, None, None, :]  # (N,w,w,w,3)
    rel_ang = rel_vox * dx  # (N,w,w,w,3)

    b5 = b_coefs[:, None, None, None, None, :]  # (N,1,1,1,1,5)
    a5 = a_coefs[:, None, None, None, :]  # (N,1,1,1,5)
    k = 2 * torch.pi / torch.sqrt(b5)
    x0 = rel_ang.unsqueeze(-1)  # (N,w,w,w,3,1) to broadcast against the 5 terms
    voxel_avg_per_axis = (torch.erf(k * (x0 + h)) - torch.erf(k * (x0 - h))) / (
        4 * h
    )  # (N,w,w,w,3,5)
    voxel_avg = voxel_avg_per_axis.prod(dim=-2)  # (N,w,w,w,5): product over x,y,z
    vals = c1 * (a5 * voxel_avg).sum(-1)  # (N,w,w,w)

    return _scatter_window_values(vals, center_idx, offsets_vox, grid_shape)


def _gaussian_voxel_average_3d(
    x0: torch.Tensor, h: float, d: torch.Tensor
) -> torch.Tensor:
    """
    Exact 3D voxel average of `d**-1.5 * exp(-pi^2*r^2/d)` (Kirkland's
    Gaussian-term shape — note this is *not* the same normalization as
    Shtyrov/Peng's `(4*pi/b)**1.5*exp(-4*pi^2*r^2/b)`), via the per-axis
    `erf` antiderivative, same principle as `build_potential_volume_analytic_scatter`.

    Parameters
    ----------
    x0 : torch.Tensor
        Per-axis continuous offset from atom to window-voxel center, shape
        (..., 3, T) where the axis-3 (second-to-last) dim is the spatial
        x/y/z axis and T is broadcastable against `d`'s trailing term dim
        (T=1 if there is only one Gaussian term).
    h : float
        Half voxel width, in Å.
    d : torch.Tensor
        Kirkland `d_i` coefficient(s), shape (..., 1, T) to broadcast against
        `x0`'s spatial axis.

    Returns
    -------
    torch.Tensor
        Voxel average, same shape as `x0` minus its spatial axis-3 dim, i.e.
        (..., T).
    """
    k = torch.pi / torch.sqrt(d)
    per_axis = (torch.erf(k * (x0 + h)) - torch.erf(k * (x0 - h))) / (
        4 * h * torch.sqrt(torch.tensor(torch.pi, dtype=x0.dtype, device=x0.device))
    )
    return per_axis.prod(dim=-2)


def build_potential_volume_analytic_scatter_kirkland(
    atomic_numbers: torch.Tensor,
    coords: torch.Tensor,
    grid_shape: tuple[int, int, int],
    dx: float,
    rcut: float = 5.0,
) -> torch.Tensor:
    """
    Kirkland analogue of `build_potential_volume_analytic_scatter`.

    Kirkland's real-space potential (`kirkland_atomic_potential_3d`, App.
    C.19) is a sum of 3 Gaussian terms (`c_i*d_i**-1.5*exp(-pi^2*r^2/d_i)`)
    and 3 Yukawa/screened-Coulomb terms (`a_i/r*exp(-2*pi*r*sqrt(b_i))`).
    The Gaussian terms get the same exact `erf` voxel-average treatment as
    Shtyrov/Peng (see `_gaussian_voxel_average_3d`); the Yukawa terms have a
    genuine `1/r` singularity with no equivalent closed form, so they use
    `yukawa_shell_average`'s sphere-of-equal-volume approximation instead
    (~2-3% typical, ~18-20% worst-case relative error — see that function's
    docstring for the full derivation and validation).

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        Atomic numbers, shape (N,).
    coords : torch.Tensor
        Atom coordinates, shape (N, 3), centered-origin convention (see
        `build_potential_volume_analytic_scatter`).
    grid_shape : tuple of int
        Output volume shape (nz, ny, nx).
    dx : float
        Voxel size, in Å.
    rcut : float, optional
        Radius (Å) of the local window evaluated around each atom. Default
        5.0 Å — checked to leave <0.001% of peak value for every element in
        the Kirkland table at this radius.

    Returns
    -------
    potential_volume : torch.Tensor
        Potential volume, shape `grid_shape`, in units of V·Å.
    """
    device = coords.device
    dtype = coords.dtype
    h = dx / 2
    R = h * (6 / torch.pi) ** (1 / 3)

    a0 = 0.529  # Bohr radius, [Å]
    e = 14.4  # electron charge, [V·Å]
    c1_yukawa = 2 * (torch.pi**2) * a0 * e
    c2_gauss = 2 * (torch.pi ** (5 / 2)) * a0 * e

    kirkland_params = load_kirkland_parameters().to(device=device, dtype=dtype)
    P = kirkland_params[atomic_numbers]  # (N, 3, 4)
    a_yuk, b_yuk, c_gauss, d_gauss = (P[:, :, i] for i in range(4))  # each (N, 3)

    center_idx, frac_offset, offsets_vox = _local_window_geometry(
        coords, grid_shape, dx, rcut
    )
    rel_vox = offsets_vox[None] - frac_offset[:, None, None, None, :]  # (N,w,w,w,3)
    rel_ang = rel_vox * dx  # (N,w,w,w,3)
    p = rel_ang.norm(dim=-1)  # (N,w,w,w)

    # Yukawa terms (3 per atom): shell-average at radial distance p.
    lam_yuk = 2 * torch.pi * torch.sqrt(b_yuk)  # (N,3)
    p3 = p.unsqueeze(-1)  # (N,w,w,w,1)
    lam_yuk_b = lam_yuk[:, None, None, None, :]  # (N,1,1,1,3)
    yukawa_avg = yukawa_shell_average(p3, R, lam_yuk_b)  # (N,w,w,w,3)
    yukawa_term = c1_yukawa * (a_yuk[:, None, None, None, :] * yukawa_avg).sum(-1)

    # Gaussian terms (3 per atom): exact erf voxel average, per axis.
    d_gauss_b = d_gauss[:, None, None, None, :]  # (N,1,1,1,3)
    x0 = rel_ang.unsqueeze(-1)  # (N,w,w,w,3,1) to broadcast against the 3 terms
    gauss_avg = _gaussian_voxel_average_3d(x0, h, d_gauss_b.unsqueeze(-2))
    gauss_term = c2_gauss * (c_gauss[:, None, None, None, :] * gauss_avg).sum(-1)

    vals = yukawa_term + gauss_term
    return _scatter_window_values(vals, center_idx, offsets_vox, grid_shape)


def build_potential_volume_analytic_scatter_lobato(
    atomic_numbers: torch.Tensor,
    coords: torch.Tensor,
    grid_shape: tuple[int, int, int],
    dx: float,
    rcut: float = 5.0,
) -> torch.Tensor:
    """
    Lobato analogue of `build_potential_volume_analytic_scatter`.

    Lobato's real-space potential (`lobato_atomic_potential_3d`, Eq. 15) is
    entirely Yukawa-type: `a_i/b_i**1.5 * (sqrt(b_i)/(pi*r) + 1) *
    exp(-2*pi*r/sqrt(b_i))`, which splits exactly into
    `(a_i/(pi*b_i)) * yukawa_shell_average(...)` (the `1/r` part, genuine
    singularity) plus `(a_i/b_i**1.5) * plain_exp_shell_average(...)` (the
    `+1` part, smooth, no singularity) — see both functions' docstrings.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        Atomic numbers, shape (N,).
    coords : torch.Tensor
        Atom coordinates, shape (N, 3), centered-origin convention.
    grid_shape : tuple of int
        Output volume shape (nz, ny, nx).
    dx : float
        Voxel size, in Å.
    rcut : float, optional
        Radius (Å) of the local window evaluated around each atom. Default
        5.0 Å — checked to leave <0.001% of peak value for every element in
        the Lobato table at this radius.

    Returns
    -------
    potential_volume : torch.Tensor
        Potential volume, shape `grid_shape`, in units of V·Å.
    """
    device = coords.device
    dtype = coords.dtype
    h = dx / 2
    R = h * (6 / torch.pi) ** (1 / 3)

    vac_perm = 1 / 4 / torch.pi
    a0 = 0.529  # Bohr radius, [Å]
    e = 14.4  # electron charge, [V·Å]
    kappa = 2 * vac_perm / a0 / e
    c1 = torch.pi**2 / kappa

    lobato_params = load_lobato_parameters().to(device=device, dtype=dtype)
    P = lobato_params[atomic_numbers]  # (N, 5, 2)
    a, b = P[:, :, 0], P[:, :, 1]  # each (N, 5)

    center_idx, frac_offset, offsets_vox = _local_window_geometry(
        coords, grid_shape, dx, rcut
    )
    rel_vox = offsets_vox[None] - frac_offset[:, None, None, None, :]  # (N,w,w,w,3)
    rel_ang = rel_vox * dx
    p = rel_ang.norm(dim=-1).unsqueeze(-1)  # (N,w,w,w,1)

    lam = (2 * torch.pi / torch.sqrt(b))[:, None, None, None, :]  # (N,1,1,1,5)
    yukawa_avg = yukawa_shell_average(p, R, lam)  # (N,w,w,w,5)
    plain_avg = plain_exp_shell_average(p, R, lam)  # (N,w,w,w,5)

    a_b = a[:, None, None, None, :]
    b_b = b[:, None, None, None, :]
    term = (a_b / (torch.pi * b_b)) * yukawa_avg + (a_b / b_b**1.5) * plain_avg
    vals = c1 * term.sum(-1)

    return _scatter_window_values(vals, center_idx, offsets_vox, grid_shape)


# Per-element/species minimum `rcut` (Å) needed for `method="analytic"` to
# capture >=99.5% of that element/species' total integrated potential, for
# *every* radius beyond it too (verified via numerical radial integration --
# integral_0^r V(r)*4*pi*r^2 dr vs. the same integral to r=inf, scanning
# every r on a fine grid and taking the largest r where the deviation still
# exceeds 0.5% -- not a naive bisection assuming the captured fraction is
# monotonic in r, which it is *not* whenever a species/element has mixed-sign
# Gaussian terms: e.g. Shtyrov's O(HH) has two negative-amplitude terms
# (a=-0.60, b=64.2 and a=-0.15, b=121.4) that decay slower than its positive
# terms, so the naive cumulative-fraction curve overshoots 100% before the
# slow negative tail pulls it back down -- a plain bisection would have
# stopped at the first crossing (~0.6 Å) and given a value giving ~17.5%
# *too much* total potential in practice, caught by
# test_potential_builder_analytic_robust_to_subvoxel_position. 14/42 Shtyrov
# species and most Lobato/light-Peng elements have this mixed-sign
# structure; Kirkland's coefficients happen to have no negative terms.
# 99.5% was chosen to match (and for some elements, improve on) the accuracy
# the previous fixed default of rcut=5.0 actually achieved for realistic
# biological elements (e.g. K/Na, the common ions with the slowest-decaying
# tails among elements that actually appear in cryo-EM structures -- rcut=5.0
# gave them only ~99.4-99.8%). Exotic heavy alkali metals (Rb, Cs, Fr) need
# up to ~5.8-5.9 Å; light elements (H, C, N, O -- the bulk of any structure)
# only need ~1.9-2.5 Å. Index 0 is an unused padding row (element 0 doesn't
# exist), matching load_kirkland_parameters()'s convention. See
# `recommended_rcut` below, which `PotentialBuilder` uses by default.
_KIRKLAND_MIN_RCUT = [
    0.0, 2.4, 1.6, 4.9, 3.5, 2.9, 2.5, 2.1, 1.9, 1.7,
    1.5, 4.8, 3.9, 3.8, 3.3, 2.9, 2.5, 2.3, 2.1, 5.2,
    4.4, 4.1, 3.9, 3.7, 3.9, 3.5, 3.5, 3.4, 3.3, 3.4,
    3.2, 3.5, 3.2, 2.9, 2.6, 2.4, 2.2, 5.8, 4.8, 4.5,
    4.1, 3.8, 3.7, 3.5, 3.5, 3.4, 2.4, 3.3, 3.1, 3.5,
    3.3, 3.0, 2.8, 2.6, 2.4, 4.9, 4.7, 4.5, 4.5, 4.7,
    4.5, 4.6, 4.5, 4.5, 4.2, 4.4, 4.4, 4.3, 4.2, 4.2,
    3.8, 3.6, 3.5, 3.2, 3.1, 3.1, 3.0, 2.8, 2.7, 2.6,
    2.6, 3.5, 3.2, 2.9, 2.5, 2.7, 2.6, 5.3, 4.2, 3.9,
    4.1, 4.3, 4.2, 4.2, 4.3, 4.2, 3.9, 3.9, 4.1, 4.0,
    4.0, 3.9, 3.8, 3.7,
]  # fmt: skip

_LOBATO_MIN_RCUT = [
    0.0, 2.3, 1.6, 4.4, 3.2, 2.7, 2.4, 2.1, 1.9, 1.7,
    1.5, 4.5, 3.6, 3.5, 3.1, 2.7, 2.4, 2.2, 1.9, 5.1,
    4.1, 3.8, 3.7, 3.5, 3.9, 3.3, 3.2, 3.1, 3.1, 3.2,
    3.0, 3.3, 3.0, 2.7, 2.5, 2.3, 2.1, 5.8, 4.8, 4.3,
    4.0, 4.1, 4.0, 3.9, 3.9, 3.8, 2.5, 3.4, 2.9, 3.3,
    3.1, 2.9, 2.7, 2.4, 2.3, 5.9, 5.1, 4.6, 4.9, 4.8,
    4.8, 4.7, 4.7, 4.6, 4.2, 4.5, 4.5, 4.4, 4.4, 4.4,
    4.3, 3.9, 3.6, 3.4, 3.2, 3.1, 3.0, 2.7, 3.0, 2.7,
    2.7, 3.1, 3.2, 3.1, 2.9, 2.7, 2.5, 5.4, 4.8, 4.3,
    3.9, 4.1, 4.1, 4.0, 4.2, 4.2, 3.9, 3.8, 4.0, 4.0,
    4.0, 3.9, 3.9, 3.6,
]  # fmt: skip

# Peng (gemmi c4322) fallback table. gemmi has no c4322 entry for Z=99-103
# (the heaviest synthetic actinides/transactinides), so those rows use a
# safe fallback of 5.0 (matching the old fixed default) rather than an
# extrapolated number -- these elements essentially never appear in cryo-EM
# structures.
_PENG_MIN_RCUT = [
    0.0, 2.2, 1.6, 4.2, 3.1, 2.7, 2.4, 2.1, 1.8, 1.6,
    1.5, 4.2, 3.5, 3.4, 3.1, 2.7, 2.4, 2.1, 2.0, 4.9,
    4.2, 3.8, 3.7, 3.5, 3.6, 3.3, 3.2, 3.2, 3.0, 3.2,
    2.9, 3.3, 3.0, 2.7, 2.4, 2.3, 2.1, 4.5, 4.3, 3.9,
    3.7, 3.5, 3.4, 3.3, 3.2, 3.1, 2.2, 3.0, 2.9, 3.2,
    3.1, 2.8, 2.6, 2.5, 2.3, 4.7, 4.5, 4.1, 4.1, 4.3,
    4.3, 4.2, 4.2, 4.0, 3.9, 4.1, 4.1, 3.6, 4.0, 3.9,
    3.9, 3.6, 3.4, 3.2, 3.1, 3.0, 2.8, 2.8, 2.6, 2.5,
    2.5, 3.0, 2.9, 2.8, 2.7, 2.5, 2.4, 4.9, 4.3, 4.0,
    3.6, 3.9, 3.8, 3.3, 4.0, 3.8, 3.6, 3.5, 3.2, 5.0,
    5.0, 5.0, 5.0, 5.0,
]  # fmt: skip

_SHTYROV_MIN_RCUT = {
    "C(CCC)": 3.1, "C(CCN)": 2.7, "C(CCO)": 2.5,
    "C(CNN)": 2.5, "C(CNO)": 3.2, "C(COO)": 3.5,
    "C(HCC)": 3.4, "C(HCCC)": 3.1, "C(HCCN)": 3.5,
    "C(HCCO)": 2.9, "C(HCN)": 2.1, "C(HCNO)": 2.5,
    "C(HHC)": 2.3, "C(HHCC)": 2.7, "C(HHCN)": 3.0,
    "C(HHCO)": 2.6, "C(HHCS)": 2.9, "C(HHHC)": 3.1,
    "C(HHHS)": 2.7, "C(HNN)": 1.7, "C(NNN)": 2.3,
    "Fe(NNNN)": 2.3, "H(C)": 3.8, "H(N)": 4.1,
    "N(CC)": 2.0, "N(CCC)": 3.2, "N(CCFe)": 2.6,
    "N(HCC)": 3.4, "N(HHC)": 3.7, "N(HHHC)": 2.8,
    "O(C)": 3.3, "O(C, amide)": 3.4, "O(C, carboxyl)": 3.3,
    "O(CC)": 2.5, "O(CP)": 2.2, "O(HC)": 3.0,
    "O(HH)": 3.6, "O(P)": 1.7, "O(PP)": 2.6,
    "P(OOOO)": 2.5, "S(CC)": 1.7, "S(HC)": 2.3,
}  # fmt: skip


def recommended_rcut(
    atomic_numbers: torch.Tensor,
    parameterization: str = "kirkland",
    atom_species: Sequence[str | None] | None = None,
) -> float:
    """
    Recommend a `method="analytic"` `rcut` (Å) for the given structure.

    Looks up each present element (or, for Shtyrov, each present bonded
    species -- falling back to its element for unmatched/Peng-fallback
    atoms) in a precomputed table of the minimum radius needed to capture
    >=99.5% of that element/species' total integrated potential, and
    returns the max over everything actually present. Structures made only
    of light elements (H, C, N, O) need only ~2-2.5 Å; heavier or more
    diffuse elements (e.g. K, Na) need more (~5 Å) -- see the module-level
    `_KIRKLAND_MIN_RCUT`/etc. tables' docstring comment for the full
    derivation and validation.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        Atomic numbers present in the structure.
    parameterization : str, optional
        'kirkland', 'lobato', or 'shtyrov'. Default 'kirkland'.
    atom_species : sequence of str or None, optional
        Per-atom bonded-species descriptors, same length/order as
        `atomic_numbers`. Only used when `parameterization='shtyrov'`.
        Default is None (every atom uses its plain per-element Peng value).

    Returns
    -------
    float
        Recommended `rcut`, in Å.
    """
    unique_z = torch.unique(atomic_numbers).tolist()
    if parameterization == "kirkland":
        return max(_KIRKLAND_MIN_RCUT[z] for z in unique_z)
    if parameterization == "lobato":
        return max(_LOBATO_MIN_RCUT[z] for z in unique_z)
    if parameterization == "shtyrov":
        if atom_species is None:
            return max(_PENG_MIN_RCUT[z] for z in unique_z)
        needed = []
        for z, species in zip(atomic_numbers.tolist(), atom_species):
            if species is not None and species in _SHTYROV_MIN_RCUT:
                needed.append(_SHTYROV_MIN_RCUT[species])
            else:
                needed.append(_PENG_MIN_RCUT[z])
        return max(needed)
    raise ValueError(
        f"Unknown parameterization '{parameterization}'. "
        "Choose 'kirkland', 'lobato', or 'shtyrov'."
    )


class PotentialBuilder(L.LightningModule):
    """
    Module for building 3D electrostatic potential volumes from atomic coordinates.

    Computes potentials using supersampled atomic potential kernels and
    convolution, supporting multiple parameterizations (Kirkland, Lobato, Shtyrov).

    Parameters
    ----------
    n_xyz : int or tuple of int
        Grid size (nx, ny, nz). If int, assumes cubic grid.
    dx : float
        Pixel/voxel size in Å.
    atomic_numbers : torch.Tensor
        Atomic numbers of all atoms in structure.
    progressbars : bool, optional
        Enable progress bars during computation. Default is True.
    parameterization : str, optional
        Atomic potential parameterization: 'kirkland', 'lobato', or 'shtyrov'.
        Default is 'kirkland'.
    conv_backend : str, optional
        Convolution backend: 'fftconvolve' or 'conv3d'. Default is 'fftconvolve'.
    mmcif_filepath : str, optional
        Path to a numeric-`scat_id`-keyed mmCIF Shtyrov parameter file.
        Only used when `atom_species` is not given. Default is None.
    atom_species : sequence of str or None, optional
        Per-atom bonded-species descriptors (e.g. `"O(HH)"`, `"C(HHHC)"`,
        as produced by `specter.pdb.PDB.get_atom_species`), same length
        and atom order as `atomic_numbers`. Only used when
        `parameterization='shtyrov'`; ignored by Kirkland/Lobato. Atoms
        with no matching species (`None`, or not covered by
        `shtyrov_params_path`) fall back to gemmi's built-in per-element
        `c4322` (Peng et al.) scattering factors for that atom — the same
        fallback sffit itself uses. Default is None.
    shtyrov_params_path : str, optional
        Path to an sffit-format JSON species parameter file (see
        `specter.atom.load_shtyrov_species_parameters`). Only used when
        `atom_species` is given. Defaults to the bundled
        `specter.atom_data/params_cat.json`.
    rcut : float, optional
        Radius (Å) of the local evaluation window used by
        `forward(method="analytic")`, for every `parameterization`. Only
        used by `method="analytic"`. Default is None, which auto-selects
        the smallest radius that still captures >=99.5% of the total
        integrated potential of every element/species actually present
        (see `recommended_rcut`) — e.g. ~2-2.5 Å for a light-element-only
        (H/C/N/O) structure, up to ~5-6 Å if heavier or more diffuse
        elements (e.g. K, Na, or heavier alkali metals) are present. Pass
        an explicit value to override the auto-selection.
    periodic : bool, optional
        If True, wrap out-of-bounds voxel indices with periodic boundary
        conditions during soft voxelization. Use when coordinates were
        generated with periodic BCs (e.g. from GradientSKIcemaker). Only
        applies to the '3d' method; raises ValueError if used with '2d'.
        Default is False.

    Attributes
    ----------
    atomic_potentials_2d : torch.Tensor
        Precomputed 2D atomic potentials for each unique element.
    atomic_potentials_3d : torch.Tensor
        Precomputed 3D atomic potentials for each unique element.
    """

    def __init__(
        self,
        n_xyz: int | Sequence[int],
        dx: float,
        atomic_numbers: torch.Tensor,
        progressbars: bool = True,
        parameterization: str = "kirkland",
        conv_backend: str = "fftconvolve",
        mmcif_filepath: str | None = None,
        atom_species: Sequence[str | None] | None = None,
        shtyrov_params_path: str | None = None,
        rcut: float | None = None,
        periodic: bool = False,
    ):
        super().__init__()

        if isinstance(n_xyz, int):
            self.nx = self.ny = self.nz = n_xyz
        else:
            self.nx, self.ny, self.nz = n_xyz
        self.dx = dx
        self.progressbars = progressbars
        self.conv_backend = conv_backend
        self.mmcif_filepath = mmcif_filepath
        self.atom_species = atom_species
        self.shtyrov_params_path = shtyrov_params_path
        self.rcut = (
            rcut
            if rcut is not None
            else recommended_rcut(atomic_numbers, parameterization, atom_species)
        )
        self.periodic = periodic
        # Cache for method="analytic"'s per-atom (a, b) coefficients — built
        # lazily on first use since it needs a full atom_species lookup pass.
        self._analytic_coefs: tuple[torch.Tensor, torch.Tensor] | None = None

        self.ssn, self.ssdx, self.ssf = compute_supersampling_parameters(dx)
        sR_2d = radial_grid_2d(self.ssn, self.ssdx, convention="torch")
        sR_3d = radial_grid_3d(self.ssn, self.ssdx, convention="torch")
        self.register_buffer("sR_2d", sR_2d)
        self.register_buffer("sR_3d", sR_3d)

        self.atomic_numbers = atomic_numbers
        self.unique_elements = torch.unique(atomic_numbers)
        # Populated instead of unique_elements when parameterization is
        # 'shtyrov' and atom_species is given (see get_3d_atomic_potentials).
        self.shtyrov_groups: list[tuple[str, int | str]] | None = None
        atomic_potentials_2d = torch.empty(
            len(self.unique_elements), self.ssn // self.ssf, self.ssn // self.ssf
        )
        atomic_potentials_3d = torch.empty(
            len(self.unique_elements),
            self.ssn // self.ssf,
            self.ssn // self.ssf,
            self.ssn // self.ssf,
        )
        self.register_buffer("atomic_potentials_2d", atomic_potentials_2d)
        self.register_buffer("atomic_potentials_3d", atomic_potentials_3d)

        self.parameterization = parameterization
        self.avgpool2d = torch.nn.AvgPool2d(self.ssf, stride=self.ssf)
        self.avgpool3d = torch.nn.AvgPool3d(self.ssf, stride=self.ssf)
        if parameterization in ("kirkland", "lobato"):
            self.get_2d_atomic_potentials()
        self.get_3d_atomic_potentials()

    def get_2d_atomic_potentials(
        self, unique_elements: torch.Tensor | None = None
    ) -> None:
        """
        Compute and cache 2D atomic potential kernels for unique elements.

        Parameters
        ----------
        unique_elements : torch.Tensor, optional
            Elements to compute potentials for. If None, uses all unique
            elements from initialization. Default is None.

        Notes
        -----
        Potentials are supersampled and downsampled to main grid resolution.
        Results are stored in `self.atomic_potentials_2d`.
        """
        if unique_elements is None:
            unique_elements = self.unique_elements
        else:
            self.unique_elements = unique_elements
            self.atomic_potentials_2d = torch.empty(
                len(unique_elements), self.ssn // self.ssf, self.ssn // self.ssf
            )

        for i, elem in enumerate(self.unique_elements):
            if self.parameterization == "kirkland":
                pot = kirkland_atomic_potential_2d(int(elem), self.sR_2d)
            elif self.parameterization == "lobato":
                pot = lobato_atomic_potential_2d(int(elem), self.sR_2d)
            else:
                raise ValueError(
                    f"Unknown parameterization '{self.parameterization}'. "
                    "Choose 'kirkland' or 'lobato'."
                )

            if self.ssf != 1:
                pot = self.avgpool2d(pot[None, None]).squeeze(0).squeeze(0)

            self.atomic_potentials_2d[i] = pot

    def get_3d_atomic_potentials(
        self, unique_elements: torch.Tensor | None = None
    ) -> None:
        """
        Compute and cache 3D atomic potential kernels for unique elements.

        Parameters
        ----------
        unique_elements : torch.Tensor, optional
            Elements to compute potentials for. If None, uses all unique
            elements from initialization. Default is None.

        Notes
        -----
        Potentials are supersampled and downsampled to main grid resolution.
        Results are stored in `self.atomic_potentials_3d`.
        Supports Kirkland, Lobato, and Shtyrov parameterizations. When
        `parameterization='shtyrov'` and `self.atom_species` is given,
        kernels are grouped by bonded-species descriptor instead of by
        element (see `_get_3d_shtyrov_species_potentials`). If
        `atom_species` is None and no `mmcif_filepath` was given either (so
        the legacy numeric-`scat_id`-keyed path has nothing to use), falls
        back to treating every atom as unmatched -- i.e. plain per-element
        Peng `c4322` factors for all atoms, same as passing
        `atom_species=[None] * len(atomic_numbers)` explicitly. A given
        `mmcif_filepath` (legacy path) is left untouched.
        """
        if (
            self.parameterization == "shtyrov"
            and self.atom_species is None
            and self.mmcif_filepath is None
        ):
            self.atom_species = [None] * len(self.atomic_numbers)

        if self.parameterization == "shtyrov" and self.atom_species is not None:
            self._get_3d_shtyrov_species_potentials()
            return

        if unique_elements is None:
            unique_elements = self.unique_elements
        else:
            # update unique elements
            self.unique_elements = unique_elements
            self.atomic_potentials_3d = torch.empty(
                len(unique_elements),
                self.ssn // self.ssf,
                self.ssn // self.ssf,
                self.ssn // self.ssf,
            )

        for i, elem in enumerate(self.unique_elements):
            if self.parameterization == "kirkland":
                pot = kirkland_atomic_potential_3d(int(elem), self.sR_3d)
            elif self.parameterization == "lobato":
                pot = lobato_atomic_potential_3d(int(elem), self.sR_3d)
            elif self.parameterization == "shtyrov":
                if self.mmcif_filepath is None:
                    raise ValueError("mmcif_filepath must be specified.")
                pot = shtyrov_atomic_potential_3d(
                    int(elem), self.sR_3d, self.mmcif_filepath
                )
            else:
                raise ValueError(
                    f"Unknown parameterization '{self.parameterization}'. "
                    "Choose 'kirkland', 'lobato', or 'shtyrov'."
                )

            if self.ssf != 1:
                pot = self.avgpool3d(pot[None, None]).squeeze(0).squeeze(0)

            self.atomic_potentials_3d[i] = pot

    def _get_3d_shtyrov_species_potentials(self) -> None:
        """
        Build 3D Shtyrov kernels grouped by bonded-species descriptor.

        Atoms whose `atom_species` entry is `None` or not present in the
        species parameter table fall back to a per-element, unbonded
        scattering factor instead (gemmi's built-in `c4322` Peng et al.
        table — the same fallback sffit itself uses for atom types missing
        from a fitted table, see `sffit/fit.py::do_mmcif`), with a single
        aggregated warning (no atom is silently dropped). Populates
        `self.shtyrov_groups` (a list of `("species", name)` /
        `("element", z)` tags, one per row of `self.atomic_potentials_3d`)
        for `forward` to consume.
        """
        params_path = self.shtyrov_params_path
        if params_path is None:
            params_path = str(
                resources.files("specter.atom_data").joinpath("params_cat.json")
            )
        species_table = load_shtyrov_species_parameters(params_path)

        assert self.atom_species is not None
        matched = sorted(
            {s for s in self.atom_species if s is not None and s in species_table}
        )
        unmatched_mask = torch.tensor(
            [s is None or s not in species_table for s in self.atom_species]
        )
        fallback_elements = sorted(
            {int(z) for z in self.atomic_numbers[unmatched_mask].tolist()}
        )

        if fallback_elements:
            warnings.warn(
                f"{int(unmatched_mask.sum())} atom(s) had no matching Shtyrov "
                f"species (elements: {[str(atom_symbol(z)) for z in fallback_elements]}); "
                "falling back to gemmi's built-in Peng et al. scattering "
                "factors for those atoms.",
                stacklevel=2,
            )

        self.shtyrov_groups = [("species", s) for s in matched] + [
            ("element", z) for z in fallback_elements
        ]

        self.atomic_potentials_3d = torch.empty(
            len(self.shtyrov_groups),
            self.ssn // self.ssf,
            self.ssn // self.ssf,
            self.ssn // self.ssf,
        )
        for i, (kind, key) in enumerate(self.shtyrov_groups):
            if kind == "species":
                assert isinstance(key, str)
                pot = shtyrov_atomic_potential_3d_by_species(
                    key, self.sR_3d, species_table
                )
            else:
                assert isinstance(key, int)
                pot = peng_atomic_potential_3d(key, self.sR_3d)

            if self.ssf != 1:
                pot = self.avgpool3d(pot[None, None]).squeeze(0).squeeze(0)

            self.atomic_potentials_3d[i] = pot

    def _get_analytic_atom_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build (and cache) per-atom Gaussian coefficients for `forward(method="analytic")`.

        Unlike the grouped-kernel path (`_get_3d_shtyrov_species_potentials`),
        analytic evaluation needs no shared kernel per group — each atom's
        own `(a, b)` coefficients (matched species, or the Peng `c4322`
        fallback) are looked up directly.

        Returns
        -------
        a_coefs, b_coefs : torch.Tensor
            Per-atom Gaussian coefficients, shape (N, 5) each.
        """
        if self._analytic_coefs is not None:
            return self._analytic_coefs

        assert self.atom_species is not None
        params_path = self.shtyrov_params_path
        if params_path is None:
            params_path = str(
                resources.files("specter.atom_data").joinpath("params_cat.json")
            )
        species_table = load_shtyrov_species_parameters(params_path)

        n = len(self.atom_species)
        a_coefs = torch.empty(n, 5, device=self.device)
        b_coefs = torch.empty(n, 5, device=self.device)
        fallback_elements: set[str] = set()
        n_fallback = 0
        for i, species in enumerate(self.atom_species):
            if species is not None and species in species_table:
                P = species_table[species]
            else:
                z = int(self.atomic_numbers[i])
                coef = gemmi.Element(z).c4322
                P = torch.stack([torch.tensor(coef.a), torch.tensor(coef.b)], dim=-1)
                fallback_elements.add(str(atom_symbol(z)))
                n_fallback += 1
            a_coefs[i] = P[:, 0]
            b_coefs[i] = P[:, 1]

        if fallback_elements:
            warnings.warn(
                f"{n_fallback} atom(s) had no matching Shtyrov species "
                f"(elements: {sorted(fallback_elements)}); falling back to "
                "gemmi's built-in Peng et al. scattering factors for those atoms.",
                stacklevel=2,
            )

        self._analytic_coefs = (a_coefs, b_coefs)
        return self._analytic_coefs

    def _get_analytic_peng_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build (and cache) per-atom Peng `c4322` coefficients for
        `forward(method="analytic")` when `parameterization='shtyrov'` but no
        `atom_species` bonded typing was given — every atom uses its plain,
        unbonded per-element scattering factor (no species table lookup).

        Returns
        -------
        a_coefs, b_coefs : torch.Tensor
            Per-atom Gaussian coefficients, shape (N, 5) each.
        """
        if self._analytic_coefs is not None:
            return self._analytic_coefs

        atomic_numbers = self.atomic_numbers.to(self.device)
        n = len(atomic_numbers)
        a_coefs = torch.empty(n, 5, device=self.device)
        b_coefs = torch.empty(n, 5, device=self.device)
        for i in range(n):
            coef = gemmi.Element(int(atomic_numbers[i])).c4322
            a_coefs[i] = torch.tensor(coef.a)
            b_coefs[i] = torch.tensor(coef.b)

        self._analytic_coefs = (a_coefs, b_coefs)
        return self._analytic_coefs

    def forward(
        self,
        coordinates: torch.Tensor,
        method: str = "3d",
        conv_backend: str | None = None,
    ) -> torch.Tensor:
        """
        Build potential volume(s) from atomic coordinates.

        Parameters
        ----------
        coordinates : torch.Tensor
            Atomic coordinates. Shape (N, 3) for single volume or (B, N, 3)
            for batch of volumes.
        method : str, optional
            Voxelization method: '2d' (soft XY, hard Z), '3d' (trilinear), or
            'analytic' (per-atom closed-form evaluation in a local window,
            no splat/FFT/kernel precomputation — supported for every
            `parameterization`: Kirkland/Lobato dispatch on `atomic_numbers`
            directly via `build_potential_volume_analytic_scatter_kirkland`/
            `_lobato`; Shtyrov dispatches on per-atom bonded species via
            `build_potential_volume_analytic_scatter` when `atom_species` is
            given, falling back to plain per-element Peng `c4322` factors
            for atoms without a species, or for every atom when
            `atom_species` is None entirely). Default is '3d'.
        conv_backend : str, optional
            Convolution backend override. Default is None (uses self.conv_backend).
            Unused for `method='analytic'`.

        Returns
        -------
        potential_volume : torch.Tensor
            Electrostatic potential volume(s). Shape (nz, ny, nx) for single
            input or (B, nz, ny, nx) for batched input.

        Notes
        -----
        '2d'/'3d' use soft voxelization followed by convolution with
        precomputed atomic potential kernels. The 2d method is faster but
        less accurate. 'analytic' skips voxelization/convolution/kernel
        precomputation entirely — dramatically faster than '3d' for typical
        atom counts (benchmarked ~26-90x on CPU, ~18-28x on GPU for a
        ~1600-atom structure) and fully differentiable w.r.t. `coordinates`.
        """
        if method == "analytic":
            coordinates = coordinates.to(self.device)
            if coordinates.ndim == 2:
                coordinates = coordinates.unsqueeze(0)
            B = coordinates.shape[0]

            if self.parameterization == "kirkland":
                atomic_numbers = self.atomic_numbers.to(self.device)
                analytic_fn = functools.partial(
                    build_potential_volume_analytic_scatter_kirkland, atomic_numbers
                )
            elif self.parameterization == "lobato":
                atomic_numbers = self.atomic_numbers.to(self.device)
                analytic_fn = functools.partial(
                    build_potential_volume_analytic_scatter_lobato, atomic_numbers
                )
            elif self.parameterization == "shtyrov":
                if self.atom_species is not None:
                    a_coefs, b_coefs = self._get_analytic_atom_coefficients()
                else:
                    a_coefs, b_coefs = self._get_analytic_peng_coefficients()
                analytic_fn = functools.partial(
                    build_potential_volume_analytic_scatter,
                    a_coefs=a_coefs,
                    b_coefs=b_coefs,
                )
            else:
                raise ValueError(
                    f"Unknown parameterization '{self.parameterization}'. "
                    "Choose 'kirkland', 'lobato', or 'shtyrov'."
                )

            potential_volume = torch.stack(
                [
                    analytic_fn(
                        coordinates[b],
                        grid_shape=(self.nz, self.ny, self.nx),
                        dx=self.dx,
                        rcut=self.rcut,
                    )
                    for b in range(B)
                ]
            )
            if B == 1:
                potential_volume = potential_volume.squeeze(0)
            return potential_volume

        if conv_backend is None:
            conv_backend = self.conv_backend
        coordinates = coordinates.to(self.device)

        if coordinates.ndim == 2:  # (N,3) -> add batch dimension
            coordinates = coordinates.unsqueeze(0)

        B, N, _ = coordinates.shape

        potential_volume = torch.zeros(
            (B, self.nz, self.ny, self.nx), device=self.device
        )

        if method == "2d" and self.shtyrov_groups is not None:
            raise ValueError(
                "method='2d' is not supported with species-grouped Shtyrov "
                "potentials (atom_species given). Use method='3d'."
            )

        groups: list[tuple[str, int | str]] = self.shtyrov_groups or [
            ("element", int(z)) for z in self.unique_elements
        ]

        with TqdmProgress(transient=True, disable=not self.progressbars) as progress:
            task = progress.add_task("Building element ...", total=len(groups))
            for i, (kind, key) in enumerate(groups):
                label: str
                if kind == "species":
                    assert isinstance(key, str)
                    assert self.atom_species is not None
                    label = key
                    mask = torch.tensor(
                        [s == key for s in self.atom_species], device=self.device
                    )
                else:
                    assert isinstance(key, int)
                    label = str(atom_symbol(key))
                    mask = self.atomic_numbers == key
                progress.update(
                    task,
                    description=f"Building element {label}",
                    advance=1,
                )
                atomic_indices = torch.argwhere(mask).squeeze(-1)
                coords_elem = coordinates[:, atomic_indices, :]  # (B, Nelem, 3)

                if method == "2d":
                    if self.periodic:
                        raise ValueError(
                            "periodic=True is not supported with method='2d'. "
                            "Use method='3d'."
                        )
                    temp_vol = soft_voxelize_xy_coordinates(
                        coords_elem,
                        grid_shape=(self.nz, self.ny, self.nx),
                        voxel_size=self.dx,
                    )
                    temp_vol_flat = temp_vol.reshape(-1, 1, self.ny, self.nx)
                    pot_b = self.atomic_potentials_2d[i].unsqueeze(0).unsqueeze(0)
                    convolved_flat = F.conv2d(temp_vol_flat, pot_b, padding="same")
                    potential_volume += convolved_flat.reshape(
                        B, self.nz, self.ny, self.nx
                    )

                elif method == "3d":
                    temp_vol = soft_voxelize_coordinates(
                        coords_elem,
                        grid_shape=(self.nz, self.ny, self.nx),
                        voxel_size=self.dx,
                        periodic=self.periodic,
                    )
                    if conv_backend == "fftconvolve":
                        for b in range(B):
                            potential_volume[b] += fftconvolve(
                                temp_vol[b], self.atomic_potentials_3d[i], mode="same"
                            )
                    elif conv_backend == "conv3d":
                        vol_b = temp_vol.unsqueeze(1)  # (B,1,nz,ny,nx)
                        kernel = self.atomic_potentials_3d[i].unsqueeze(0).unsqueeze(0)
                        convolved = F.conv3d(vol_b, kernel, padding="same")
                        potential_volume += convolved.squeeze(1)
                    else:
                        raise ValueError(
                            f"Unknown conv_backend '{conv_backend}'. "
                            "Choose 'fftconvolve' or 'conv3d'."
                        )
                else:
                    raise ValueError(f"Unknown method '{method}'. Choose '2d' or '3d'.")

        if B == 1:
            potential_volume = potential_volume.squeeze(0)
        return potential_volume


class GemmiPotentialBuilder:
    """
    Build electrostatic potential volumes using Gemmi library.

    Uses Gemmi's density calculator with Gaussian atomic form factors.
    Supports custom scattering factors from mmCIF files.

    Parameters
    ----------
    n_xyz : int or tuple of int
        Grid size (nx, ny, nz). If int, assumes cubic grid.
    dx : float
        Pixel/voxel size in Å.
    atomic_numbers : torch.Tensor, optional
        Atomic numbers of all atoms. Default is None.
    b_factor : float, optional
        Isotropic B-factor. Default is 20.0.
    """

    def __init__(
        self,
        n_xyz: int | Sequence[int],
        dx: float,
        atomic_numbers: torch.Tensor | None = None,
        b_factor: float | None = None,
    ):
        if isinstance(n_xyz, (int, float)):
            self.nx = self.ny = self.nz = n_xyz
        else:
            self.nx, self.ny, self.nz = n_xyz
        self.dx = dx
        self.b_factor = b_factor
        self.translate_to_center = torch.tensor(
            [[self.nx // 2 * dx, self.ny // 2 * dx, self.nz // 2 * dx]]
        )
        """torch.Tensor: Translation vector to center atoms in grid."""
        self.atomic_numbers = atomic_numbers

        # scaling prefactor
        a0 = 0.529  # Bohr radius, [Å]
        e = 14.4  # electron charge, [V·Å]
        self.c1 = 2 * torch.pi * e * a0
        """float: Scaling factor for electrostatic potential (2π*e*a₀)."""

    def build_model(
        self,
        atom_coordinates: torch.Tensor,
        atom_elements: torch.Tensor,
    ) -> gemmi.Model:
        """
        Build Gemmi structure model from atomic coordinates and elements.

        Parameters
        ----------
        atom_coordinates : torch.Tensor
            Atomic coordinates in Å, shape (N, 3).
        atom_elements : torch.Tensor
            Atomic numbers, shape (N,).

        Returns
        -------
        model : gemmi.Model
            Gemmi model containing atoms within grid bounds.

        Notes
        -----
        Filters out atoms outside the grid boundaries.
        All atoms are assigned to chain A, residue 1.
        """
        st = gemmi.Structure()
        model = st.add_model(gemmi.Model(1))
        chain = model.add_chain("A")
        res = chain.add_residue(gemmi.Residue())

        mask = (
            (atom_coordinates[:, 0] >= 0.0)
            & (atom_coordinates[:, 0] <= self.nx * self.dx)
            & (atom_coordinates[:, 1] >= 0.0)
            & (atom_coordinates[:, 1] <= self.ny * self.dx)
            & (atom_coordinates[:, 2] >= 0.0)
            & (atom_coordinates[:, 2] <= self.nz * self.dx)
        )

        filtered_coords = atom_coordinates[mask]
        filtered_elements = atom_elements[mask]
        b_iso = self.b_factor if self.b_factor is not None else 20.0

        for i, (pos, z) in enumerate(zip(filtered_coords, filtered_elements), start=1):
            # if not (0. <= pos[0] <= self.nx * self.dx and 0. <= pos[1] <= self.ny * self.dx and 0. <= pos[2] <= self.nz * self.dx):
            #     continue
            atom = gemmi.Atom()
            atom.pos = gemmi.Position(float(pos[0]), float(pos[1]), float(pos[2]))
            atom.element = gemmi.Element(int(z))
            atom.occ = 1.0
            atom.b_iso = b_iso
            atom.serial = i
            res.add_atom(atom)
        return model

    def build_dencalc(self) -> gemmi.DensityCalculatorE:
        """
        Build a fresh Gemmi density calculator with current grid settings.

        Returns
        -------
        dencalc : gemmi.DensityCalculatorE
            Configured density calculator.

        Notes
        -----
        Creates a new calculator to avoid state contamination between
        multiple potential calculations.
        """
        dencalc = gemmi.DensityCalculatorE()
        unit_cell = gemmi.UnitCell(
            self.nx * self.dx,
            self.ny * self.dx,
            self.nz * self.dx,
            90.0,
            90.0,
            90.0,
        )
        dencalc.grid.unit_cell = unit_cell
        dencalc.grid.spacegroup = gemmi.SpaceGroup("P1")
        dencalc.grid.set_size(self.nx, self.ny, self.nz)
        return dencalc

    def build_potential_from_custom_mmcif(self, mmcif_filepath: str) -> torch.Tensor:
        """
        Build potential using custom scattering factors from mmCIF file.

        Reads scattering factor coefficients from '_lmb_scat_coef' table
        in mmCIF file for high-accuracy potential calculations.

        Parameters
        ----------
        mmcif_filepath : str
            Path to mmCIF file containing structure and scattering factors.

        Returns
        -------
        potential : torch.Tensor
            Electrostatic potential volume, shape (nz, ny, nx).

        Notes
        -----
        Uses only the first atom from the structure and applies custom
        form factors from the mmCIF file. Atoms are recentered to grid center.
        """
        st = gemmi.read_structure(mmcif_filepath)

        # --- keep only the first atom ---
        # first_atom = st[0][0][0][0]
        # new_st = gemmi.Structure()
        # model = new_st.add_model(gemmi.Model(1))
        # chain = model.add_chain("A")
        # residue = chain.add_residue(gemmi.Residue())
        # residue.add_atom(first_atom.clone())
        # st = new_st
        # --------------------------------

        block = gemmi.cif.read_file(mmcif_filepath).sole_block()
        ctable = block.find(
            "_lmb_scat_coef.",
            [
                "coef_a1",
                "coef_a2",
                "coef_a3",
                "coef_a4",
                "coef_a5",
                "coef_b1",
                "coef_b2",
                "coef_b3",
                "coef_b4",
                "coef_b5",
            ],
        )

        coefs = np.empty((len(ctable), 10))
        for ind, row in enumerate(ctable):
            coefs[ind] = [float(field) for field in row]
        max_serial = max(cra.atom.serial for cra in st[0].all())
        custom_form_factors = np.zeros((max_serial + 1, 10))
        itable = block.find("_atom_site.", ["id", "scat_id"])
        for row in itable:
            serial, scat_id = row
            custom_form_factors[int(serial)] = coefs[int(scat_id)]
            # print(scat_id)
            # break
        gemmi.set_custom_form_factors(custom_form_factors.tolist())
        dencalc = gemmi.DensityCalculatorC()

        coords = np.array([cra.atom.pos for cra in st[0].all()])  # (N, 3)
        center_geom = coords.mean(axis=0)
        for cra in st[0].all():
            translate = -np.asarray(
                center_geom.tolist()
            ) + self.translate_to_center.numpy().squeeze(0)
            cra.atom.pos += gemmi.Position(
                float(translate[0]), float(translate[1]), float(translate[2])
            )
            if self.b_factor is not None:
                cra.atom.b_iso = self.b_factor

        if self.b_factor is None:
            print(
                "Using default B-factor in mmcif file. Set b_factor to 0 if not intended."
            )

        unit_cell = gemmi.UnitCell(
            self.nx * self.dx,
            self.ny * self.dx,
            self.nz * self.dx,
            90.0,
            90.0,
            90.0,
        )
        dencalc.grid.unit_cell = unit_cell
        dencalc.grid.spacegroup = gemmi.SpaceGroup("P1")
        dencalc.grid.set_size(self.nx, self.ny, self.nz)
        dencalc.put_model_density_on_grid(st[0])
        return self.c1 * torch.as_tensor(dencalc.grid.array).transpose(0, 2)

    def _build_single_potential(
        self, coords_elements_tuple: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Build potential for a single set of coordinates (non-parallel).

        Parameters
        ----------
        coords_elements_tuple : tuple
            (coordinates, elements) where coordinates is (N, 3) and
            elements is (N,).

        Returns
        -------
        potential : torch.Tensor
            Potential volume with shape (nz, ny, nx).
        """
        coords, elements = coords_elements_tuple
        model = self.build_model(coords, elements)
        dencalc = self.build_dencalc()
        dencalc.put_model_density_on_grid(model)
        return torch.as_tensor(dencalc.grid.array).transpose(0, 2)

    @staticmethod
    def _build_parallelizable_single_potential(args: tuple[Any, ...]) -> torch.Tensor:
        """
        Build potential for parallel processing (static method).

        Parameters
        ----------
        args : tuple
            (coords, elements, nx, ny, nz, dx, b_factor) containing all
            necessary parameters for building potential.

        Returns
        -------
        potential : torch.Tensor
            Potential volume with shape (nz, ny, nx).

        Notes
        -----
        Static method to enable multiprocessing with spawn context.
        Creates fresh Gemmi objects to avoid pickling issues.
        """
        coords, elements, nx, ny, nz, dx, b_factor = args
        st = gemmi.Structure()
        model = st.add_model(gemmi.Model(1))
        chain = model.add_chain("A")
        res = chain.add_residue(gemmi.Residue())

        mask = (
            (coords[:, 0] >= 0.0)
            & (coords[:, 0] <= nx * dx)
            & (coords[:, 1] >= 0.0)
            & (coords[:, 1] <= ny * dx)
            & (coords[:, 2] >= 0.0)
            & (coords[:, 2] <= nz * dx)
        )

        filtered_coords = coords[mask]
        filtered_elements = elements[mask]

        for i, (pos, z) in enumerate(zip(filtered_coords, filtered_elements), start=1):
            atom = gemmi.Atom()
            atom.pos = gemmi.Position(float(pos[0]), float(pos[1]), float(pos[2]))
            atom.element = gemmi.Element(int(z))
            atom.occ = 1.0
            atom.b_iso = b_factor
            atom.serial = i
            res.add_atom(atom)

        dencalc = gemmi.DensityCalculatorE()
        unit_cell = gemmi.UnitCell(
            nx * dx,
            ny * dx,
            nz * dx,
            90.0,
            90.0,
            90.0,
        )
        dencalc.grid.unit_cell = unit_cell
        dencalc.grid.spacegroup = gemmi.SpaceGroup("P1")
        dencalc.grid.set_size(nx, ny, nz)
        dencalc.put_model_density_on_grid(model)
        return torch.as_tensor(dencalc.grid.array).transpose(0, 2)

    def build_potential(
        self,
        atom_coordinates: torch.Tensor,
        atomic_numbers: torch.Tensor | None = None,
        n_processes: int | None = None,
    ) -> torch.Tensor:
        """
        Build electrostatic potential volume from atomic coordinates.

        Parameters
        ----------
        atom_coordinates : torch.Tensor
            Atomic coordinates in Ų, shape (N, 3). Centered at origin.
        atomic_numbers : torch.Tensor, optional
            Atomic numbers, shape (N,). If None, uses self.atomic_numbers.
            Default is None.
        n_processes : int, optional
            Number of parallel processes. If None, runs serially.
            Default is None.

        Returns
        -------
        potential : torch.Tensor
            Electrostatic potential volume in Volt-Ångströms, shape (nz, ny, nx).

        Notes
        -----
        Coordinates are automatically translated to place origin at grid center.
        Parallel processing splits atoms across processes and sums results.
        Scaling factor c1 = 2π*e*a₀ is applied to match physical units.
        """
        if atomic_numbers is None:
            atomic_numbers = self.atomic_numbers
        else:
            self.atomic_numbers = atomic_numbers
        if atomic_numbers is None:
            raise ValueError(
                "atomic_numbers must be provided, either as an argument or set on "
                "the builder at construction time."
            )
        translated_coordinates = atom_coordinates + self.translate_to_center

        if n_processes is None:
            vol = self._build_single_potential((translated_coordinates, atomic_numbers))
            return self.c1 * vol

        chunks_coords = torch.split(translated_coordinates, n_processes)
        chunks_elements = torch.split(atomic_numbers, n_processes)
        args_list = [
            (
                chunks_coords[i],
                chunks_elements[i],
                self.nx,
                self.ny,
                self.nz,
                self.dx,
                self.b_factor,
            )
            for i in range(n_processes)
        ]

        with mp.get_context("spawn").Pool(processes=n_processes) as pool:
            results = pool.map(self._build_parallelizable_single_potential, args_list)

        total_volume = torch.stack(results).sum(0)
        return self.c1 * total_volume
