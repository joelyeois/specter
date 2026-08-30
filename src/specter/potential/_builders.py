"""Free-function scattering-potential builders: analytic and FFT-convolve."""

from __future__ import annotations

from importlib import resources
from typing import Any, Sequence

import torch

from ..arrays import (
    radial_grid_2d,
    radial_grid_3d,
    soft_voxelize_coordinates,
    soft_voxelize_xy_coordinates,
)
from ..atom import (
    MIN_GAUSSIAN_B,
    kirkland_atomic_potential_2d,
    kirkland_atomic_potential_3d,
    load_kirkland_parameters,
    load_lobato_parameters,
    load_shtyrov_species_parameters,
    lobato_atomic_potential_3d,
    peng_atomic_potential_3d,
    plain_exp_shell_average,
    shtyrov_atomic_potential_3d_by_species,
    yukawa_shell_average,
)
from ..fft import fftconvolve, spatial_convolve2d_same, spatial_convolve3d_same
from ..progress import track


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
        temp_volume = soft_voxelize_coordinates(
            centered_coords[atomic_indices].reshape(-1, 3),
            grid_shape=(nz, ny, nx),
            voxel_size=dx,
        )

        pot = kirkland_atomic_potential_3d(int(elem), sR)

        if ssf != 1:
            pot = avgpool3d(pot[None, None]).squeeze(0).squeeze(0)
        potential_volume += fftconvolve(temp_volume, pot, mode="same")
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
        temp_volume = soft_voxelize_xy_coordinates(
            centered_coords[atomic_indices].reshape(-1, 3),
            grid_shape=(nz, ny, nx),
            voxel_size=dx,
        )

        pot = kirkland_atomic_potential_2d(int(elem), sR)

        if ssf != 1:
            pot = avgpool2d(pot[None, None]).squeeze(0).squeeze(0)

        potential_volume += spatial_convolve2d_same(temp_volume, pot)  # (nz, ny, nx)
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
        Potential volume, shape `grid_shape`, in units of V.
    """
    a0 = 0.529  # Bohr radius, [Å]
    e = 14.4  # electron charge, [V·Å]
    c1 = 2 * torch.pi * e * a0

    b_coefs = b_coefs.clamp(min=MIN_GAUSSIAN_B)
    h = dx / 2  # half voxel width, for the voxel-average integration bounds

    center_idx, frac_offset, offsets_vox = _local_window_geometry(
        coords, grid_shape, dx, rcut
    )

    # Evaluated PER AXIS, not on the full window. The 3D voxel average is a
    # product of three 1D averages (see Notes), and each factor depends only
    # on its own axis's offset -- so evaluating them on the (N,w,w,w) window
    # recomputes every value w**2 times. Three (N,w,5) tables carry the same
    # information: 121x fewer erf evaluations at w=11, 9x at the w=3 of a
    # 5 A render, and it is the erf that dominates here.
    #
    # This is a rewrite of the same expression, not an approximation. It
    # agrees with the previous form to 2.4e-7 relative, against 1.9e-7 for
    # that form compared with itself across runs -- `_scatter_window_values`
    # accumulates with atomics, so neither is bitwise reproducible anyway.
    #
    # Measured on 7VD8 at dx=1.0 (36,713 atoms, w=11): 0.111 s and 9.32 GiB
    # peak before, 0.046 s and 3.76 GiB after. Do not "simplify" this back
    # into one broadcast over (N,w,w,w,3,5).
    a5 = a_coefs[:, None, :]  # (N,1,5)
    k = (2 * torch.pi / torch.sqrt(b_coefs))[:, None, :]  # (N,1,5)

    per_axis = []
    for axis in range(3):
        # The offsets along `axis`, read off the shared window rather than
        # rebuilt, so this cannot drift from _local_window_geometry.
        index: list[Any] = [0, 0, 0]
        index[axis] = slice(None)
        offsets_1d = offsets_vox[tuple(index) + (axis,)]  # (w,)
        rel_ang = (offsets_1d[None, :] - frac_offset[:, axis, None]) * dx  # (N,w)
        x0 = rel_ang[:, :, None]  # (N,w,1), to broadcast against the 5 terms
        per_axis.append(
            (torch.erf(k * (x0 + h)) - torch.erf(k * (x0 - h))) / (4 * h)
        )  # (N,w,5)

    # Fold the per-term weights into one axis's table, then contract all
    # three against the shared term index in a single pass. The output is
    # (N, w, w, w) indexed in the same axis order as `offsets_vox`.
    weighted = per_axis[0] * a5  # (N,w,5)
    vals = c1 * torch.einsum("nik,njk,nlk->nijl", weighted, per_axis[1], per_axis[2])

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
        Potential volume, shape `grid_shape`, in units of V.
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
        Potential volume, shape `grid_shape`, in units of V.
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
    parameterization: str = "shtyrov",
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


def potential_from_deltas(
    deltas: torch.Tensor,
    kernel: torch.Tensor,
    backend: str = "fftconvolve",
) -> torch.Tensor:
    """
    Convolve soft-voxelized atom deltas with an atomic potential kernel.

    The second half of the voxelize-then-convolve recipe, shared by
    :meth:`PotentialBuilder.forward` (``method='3d'``, once per element) and
    :meth:`specter.ice.IceBank.generate_big_ice` (once, for its single water
    kernel) so both assemble a potential the same way by construction.

    Parameters
    ----------
    deltas : torch.Tensor
        Soft-voxelized occupancy, shape (B, Z, Y, X) or (Z, Y, X).
    kernel : torch.Tensor
        Atomic potential kernel, shape (kz, ky, kx) -- see
        :func:`build_atomic_potential_kernel`.
    backend : str, optional
        ``'fftconvolve'`` (default) transforms the whole volume; ``'conv3d'``
        convolves directly, at a cost that scales with the kernel rather than
        the volume, which wins for the small kernels used here. The two agree
        exactly -- :func:`specter.fft.spatial_convolve3d_same` reproduces
        ``fftconvolve``'s centered-crop convention, including for
        even-sized kernels.

    Returns
    -------
    torch.Tensor
        Potential contribution, same shape as `deltas`.
    """
    squeeze_back = deltas.dim() == 3
    if squeeze_back:
        deltas = deltas[None]

    if backend == "fftconvolve":
        out = torch.stack(
            [fftconvolve(deltas[b], kernel, mode="same") for b in range(len(deltas))]
        )
    elif backend == "conv3d":
        # Deliberately not F.conv3d(padding="same"): for an even-sized kernel
        # that pads asymmetrically relative to fftconvolve's centered crop,
        # shifting the potential half a voxel off the coordinates it was built
        # from. Even kernels are not exotic -- they occur for 20 of the 36
        # pixel sizes between 0.5 and 4.0 A, including 1.0-1.7 A.
        out = spatial_convolve3d_same(deltas, kernel)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose 'fftconvolve' or 'conv3d'."
        )

    return out[0] if squeeze_back else out


def build_atomic_potential_kernel(
    dx: float,
    parameterization: str = "kirkland",
    atomic_number: int = 8,
    shtyrov_species: str | None = None,
    species_table: dict | None = None,
    *,
    sR: torch.Tensor | None = None,
    avgpool3d: torch.nn.Module | None = None,
    ssf: int | None = None,
) -> torch.Tensor:
    """
    Build the real-space potential kernel for a single element or bonded species.

    This is the one place a scattering-potential kernel gets sampled and
    binned down to the target grid. :class:`PotentialBuilder` calls it once
    per element/species when assembling a structure's kernel stack, and the
    ice and gold-bead generators call it for their single fixed species --
    so every kernel in specter comes from the same code by construction
    rather than by convention.

    Parameters
    ----------
    dx : float
        Voxel size in Å. Ignored when `sR`/`avgpool3d`/`ssf` are supplied.
    parameterization : str, optional
        ``'kirkland'`` (default), ``'lobato'``, ``'peng'``, or ``'shtyrov'``.
        ``'peng'`` is the per-element fallback used for atoms with no
        matching Shtyrov bonded species.
    atomic_number : int, optional
        Atomic number, used by every branch except ``'shtyrov'``. Default 8
        (oxygen), the ice use case this was originally written for.
    shtyrov_species : str or None, optional
        Bonded species key (Shtyrov parameterizes bonded species, not bare
        atomic numbers), e.g. ``"O(HH)"`` for water oxygen. ``None`` (the
        default) means the caller has no species to type, and under
        ``'shtyrov'`` falls back to per-element Peng at `atomic_number` --
        the same rule :class:`PotentialBuilder` applies per atom, so a bulk
        material (gold, with no elemental Shtyrov entry) and a structure
        atom with an unmatched species resolve identically. Pass the species
        explicitly whenever one exists; there is no default species, because
        a wrong one is a silently wrong kernel.
    species_table : dict, optional
        Pre-loaded Shtyrov species parameters. Loaded from the bundled
        ``params_cat.json`` if omitted -- pass one to honour a custom
        ``shtyrov_params_path``, or just to avoid re-reading the file per
        species.
    sR : torch.Tensor, optional
        Pre-computed super-sampled radial grid. Supply it (with `avgpool3d`
        and `ssf`) to reuse a caller's own grid instead of rebuilding one
        per call -- what `PotentialBuilder` does, since it holds `sR_3d` as
        a registered buffer and shares it across every element.
    avgpool3d : torch.nn.Module, optional
        Pooling layer that bins the super-sampled kernel down to `dx`.
    ssf : int, optional
        Super-sampling factor; pooling is skipped when it is 1.

    Returns
    -------
    torch.Tensor
        Potential kernel volume, downsampled to the target grid.
    """
    if sR is None or avgpool3d is None or ssf is None:
        ssn, ssdx, ssf = compute_supersampling_parameters(dx)
        # torch convention avoids the singularity at the origin
        sR = radial_grid_3d(ssn, ssdx, convention="torch")
        avgpool3d = torch.nn.AvgPool3d(ssf, stride=ssf)

    if parameterization == "kirkland":
        pot = kirkland_atomic_potential_3d(atomic_number, sR)
    elif parameterization == "lobato":
        pot = lobato_atomic_potential_3d(atomic_number, sR)
    elif parameterization == "peng":
        pot = peng_atomic_potential_3d(atomic_number, sR)
    elif parameterization == "shtyrov" and shtyrov_species is None:
        # No species to type: same fallback PotentialBuilder applies to an
        # atom whose species misses the table (see its shtyrov_groups
        # "element" branch). Safe only because it uses the caller's own
        # atomic_number -- which is exactly why this cannot be a fallback
        # for a species that was supplied but unmatched, where this
        # function has no way to know the right element.
        pot = peng_atomic_potential_3d(atomic_number, sR)
    elif parameterization == "shtyrov":
        if species_table is None:
            species_path = resources.files("specter.atom_data").joinpath(
                "params_cat.json"
            )
            with resources.as_file(species_path) as fpath:
                species_table = load_shtyrov_species_parameters(str(fpath))
        assert shtyrov_species is not None  # narrowed by the branch above
        pot = shtyrov_atomic_potential_3d_by_species(shtyrov_species, sR, species_table)
    else:
        raise ValueError(
            f"Unknown parameterization '{parameterization}'. "
            "Choose 'kirkland', 'lobato', 'peng', or 'shtyrov'."
        )

    if ssf != 1:
        pot = avgpool3d(pot[None, None]).squeeze(0).squeeze(0)
    return pot
