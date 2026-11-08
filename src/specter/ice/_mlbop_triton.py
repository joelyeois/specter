"""
Fused Triton kernels for the ML-BOP energy and its forces, on CUDA.

The torch implementation in :meth:`~specter.ice._energy.MLBOP.compute_energy`
builds the three-body sum from index lists (pairs grouped by atom, then every
pair of pairs), about sixty gather, scatter and elementwise kernels plus
their autograd backward. At 527k molecules one energy-and-gradient
evaluation took 25 ms, the cost of every step of the ice anneal. Here one
Triton program per atom loads that atom's neighbours, forms every
(j, k) angle in registers, and writes its energy and its analytic force
contributions directly: 5 ms for the same evaluation, with forces equal to
the autograd ones to float32 precision.

Three kernels:

* :func:`neighbor_table` -- a cell-list search for every neighbour within a
  list cutoff (the potential's own plus a Verlet skin), as a dense ``(N, K)``
  table padded with -1. 12 ms at 527k molecules against 48 ms for vesin.
* a compaction pass, every evaluation, that keeps the neighbours inside the
  potential's cutoff (about 4 of the 12 in the list), so the angular tile
  is ``16 x 16`` rather than ``32 x 32``.
* the energy-and-force kernel.

Pair vectors use the minimum image of an orthorhombic cell, formed in the
positions' dtype and then cast to float32 (as ``compute_dtype=float32``
does in the torch path), so positions need not be wrapped into the cell.
:func:`supported` says when this applies.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - triton ships with CUDA torch on Linux
    triton = None


def supported(
    positions: torch.Tensor, box: tuple[float, float, float], list_cutoff: float
) -> bool:
    """
    Whether these kernels can evaluate `positions` in `box`.

    They need Triton, CUDA positions and at least three cells of
    `list_cutoff` along every axis: the cell list then visits each
    neighbouring cell once, and no pair within the cutoff has a second
    periodic image within it, so the minimum image is the only one.

    Parameters
    ----------
    positions : torch.Tensor
        Positions, shape ``(N, 3)``.
    box : tuple[float, float, float]
        Orthorhombic cell lengths.
    list_cutoff : float
        Neighbour list cutoff (potential cutoff plus skin).

    Returns
    -------
    bool
    """
    return (
        triton is not None
        and positions.is_cuda
        and positions.shape[0] > 0
        and min(box) >= 3.0 * list_cutoff
    )


if triton is not None:
    _HALF_PI = tl.constexpr(1.5707963267948966)

    @triton.jit
    def _cell_kernel(
        pos_ptr,
        sorted_ptr,
        start_ptr,
        count_ptr,
        out_ptr,
        n_out_ptr,
        Lx,
        Ly,
        Lz,
        ncx,
        ncy,
        ncz,
        RC2,
        K: tl.constexpr,
        OCC: tl.constexpr,
    ):
        i = tl.program_id(0)
        px = tl.load(pos_ptr + i * 3 + 0)
        py = tl.load(pos_ptr + i * 3 + 1)
        pz = tl.load(pos_ptr + i * 3 + 2)
        cx = tl.floor((px - Lx * tl.floor(px * (1.0 / Lx))) * (ncx / Lx)).to(tl.int32)
        cy = tl.floor((py - Ly * tl.floor(py * (1.0 / Ly))) * (ncy / Ly)).to(tl.int32)
        cz = tl.floor((pz - Lz * tl.floor(pz * (1.0 / Lz))) * (ncz / Lz)).to(tl.int32)
        cx = tl.minimum(cx, ncx - 1)
        cy = tl.minimum(cy, ncy - 1)
        cz = tl.minimum(cz, ncz - 1)
        oo = tl.arange(0, OCC)
        n = 0
        for d in tl.static_range(27):
            ax = (cx + (d % 3 - 1) + ncx) % ncx
            ay = (cy + ((d // 3) % 3 - 1) + ncy) % ncy
            az = (cz + (d // 9 - 1) + ncz) % ncz
            c = (az * ncy + ay) * ncx + ax
            s = tl.load(start_ptr + c)
            m = tl.load(count_ptr + c)
            ok = oo < m
            j = tl.load(sorted_ptr + s + oo, mask=ok, other=0)
            vx = tl.load(pos_ptr + j * 3 + 0, mask=ok, other=0.0) - px
            vy = tl.load(pos_ptr + j * 3 + 1, mask=ok, other=0.0) - py
            vz = tl.load(pos_ptr + j * 3 + 2, mask=ok, other=0.0) - pz
            vx = (vx - Lx * tl.floor(vx * (1.0 / Lx) + 0.5)).to(tl.float32)
            vy = (vy - Ly * tl.floor(vy * (1.0 / Ly) + 0.5)).to(tl.float32)
            vz = (vz - Lz * tl.floor(vz * (1.0 / Lz) + 0.5)).to(tl.float32)
            hit = ok & (j != i) & (vx * vx + vy * vy + vz * vz < RC2)
            a = hit.to(tl.int32)
            slot = n + tl.cumsum(a, axis=0) - a
            tl.store(out_ptr + i * K + slot, j, mask=hit & (slot < K))
            n += tl.sum(a, axis=0)
        tl.store(n_out_ptr + i, n)

    @triton.jit
    def _compact_kernel(
        pos_ptr,
        nbr_ptr,
        out_ptr,
        n_out_ptr,
        Lx,
        Ly,
        Lz,
        RC2,
        K: tl.constexpr,
        KA: tl.constexpr,
    ):
        i = tl.program_id(0)
        kk = tl.arange(0, K)
        idx = tl.load(nbr_ptr + i * K + kk)
        has = idx >= 0
        safe = tl.where(has, idx, 0)
        vx = tl.load(pos_ptr + safe * 3 + 0, mask=has, other=0.0)
        vy = tl.load(pos_ptr + safe * 3 + 1, mask=has, other=0.0)
        vz = tl.load(pos_ptr + safe * 3 + 2, mask=has, other=0.0)
        vx -= tl.load(pos_ptr + i * 3 + 0)
        vy -= tl.load(pos_ptr + i * 3 + 1)
        vz -= tl.load(pos_ptr + i * 3 + 2)
        vx = (vx - Lx * tl.floor(vx * (1.0 / Lx) + 0.5)).to(tl.float32)
        vy = (vy - Ly * tl.floor(vy * (1.0 / Ly) + 0.5)).to(tl.float32)
        vz = (vz - Lz * tl.floor(vz * (1.0 / Lz) + 0.5)).to(tl.float32)
        act = has & (vx * vx + vy * vy + vz * vz < RC2)
        a = act.to(tl.int32)
        slot = tl.cumsum(a, axis=0) - a
        tl.store(out_ptr + i * KA + slot, idx, mask=act & (slot < KA))
        tl.store(n_out_ptr + i, tl.sum(a, axis=0))

    @triton.jit
    def _energy_force_kernel(
        pos_ptr,
        nbr_ptr,
        e_ptr,
        f_ptr,
        Lx,
        Ly,
        Lz,
        A,
        B,
        LAM1,
        LAM2,
        R,
        D,
        BETA_N,
        N_EXP,
        C2,
        D2,
        COS0,
        K: tl.constexpr,
    ):
        # Atom i's energy E_i = 1/2 sum_j f_C(r_ij) [f_R(r_ij) + b_ij f_A(r_ij)]
        # depends only on its bond vectors v_j = r_j - r_i, so its gradient
        # G_j = dE_i/dv_j goes to atom j as +G_j and to atom i as -sum_j G_j.
        i = tl.program_id(0)
        kk = tl.arange(0, K)
        idx = tl.load(nbr_ptr + i * K + kk)
        has = idx >= 0
        safe = tl.where(has, idx, 0)
        vx = tl.load(pos_ptr + safe * 3 + 0, mask=has, other=0.0)
        vy = tl.load(pos_ptr + safe * 3 + 1, mask=has, other=0.0)
        vz = tl.load(pos_ptr + safe * 3 + 2, mask=has, other=0.0)
        vx -= tl.load(pos_ptr + i * 3 + 0)
        vy -= tl.load(pos_ptr + i * 3 + 1)
        vz -= tl.load(pos_ptr + i * 3 + 2)
        vx = (vx - Lx * tl.floor(vx * (1.0 / Lx) + 0.5)).to(tl.float32)
        vy = (vy - Ly * tl.floor(vy * (1.0 / Ly) + 0.5)).to(tl.float32)
        vz = (vz - Lz * tl.floor(vz * (1.0 / Lz) + 0.5)).to(tl.float32)
        r = tl.sqrt(vx * vx + vy * vy + vz * vz)
        act = has & (r < R + D)
        r = tl.where(act, r, 1.0)
        ux = vx / r
        uy = vy / r
        uz = vz / r

        arg = _HALF_PI * (r - R) / D
        inner = r < R - D
        fc = tl.where(act, tl.where(inner, 1.0, 0.5 - 0.5 * tl.sin(arg)), 0.0)
        dfc = tl.where(act & ~inner, -0.5 * _HALF_PI / D * tl.cos(arg), 0.0)
        f_rep = A * tl.exp(-LAM1 * r)
        f_att = -B * tl.exp(-LAM2 * r)

        # Angular terms, bond j on the rows and third atom k on the columns.
        cos = (
            ux[:, None] * ux[None, :]
            + uy[:, None] * uy[None, :]
            + uz[:, None] * uz[None, :]
        )
        pair = act[:, None] & act[None, :] & (kk[:, None] != kk[None, :])
        h = cos - COS0
        den = D2 + h * h
        g = tl.where(pair, 1.0 + C2 / D2 - C2 / den, 0.0)
        dg = tl.where(pair, 2.0 * C2 * h / (den * den), 0.0)

        xi = tl.sum(g * fc[None, :], axis=1)
        pos_xi = xi > 0.0
        xi_safe = tl.where(pos_xi, xi, 1.0)
        xi_n = tl.where(pos_xi, tl.exp(N_EXP * tl.log(xi_safe)), 0.0)
        base = 1.0 + BETA_N * xi_n
        b = tl.exp((-0.5 / N_EXP) * tl.log(base))
        db = tl.where(pos_xi, -0.5 * BETA_N * xi_n / xi_safe * b / base, 0.0)

        e_i = 0.5 * tl.sum(fc * (f_rep + b * f_att), axis=0)
        radial = 0.5 * (
            dfc * (f_rep + b * f_att) + fc * (-LAM1 * f_rep - LAM2 * b * f_att)
        )
        w = 0.5 * fc * f_att * db  # dE_i / dxi_j

        # Through xi_j of the other bonds, with m as the third atom (columns).
        wg = w[:, None] * g
        wdg = w[:, None] * dg
        s_g = tl.sum(wg, axis=0)
        s_c = tl.sum(wdg * cos, axis=0)
        s_x = tl.sum(wdg * ux[:, None], axis=0)
        s_y = tl.sum(wdg * uy[:, None], axis=0)
        s_z = tl.sum(wdg * uz[:, None], axis=0)
        # Through bond m's own xi_m, with m on the rows.
        fdg = dg * fc[None, :]
        t_c = tl.sum(fdg * cos, axis=1)
        t_x = tl.sum(fdg * ux[None, :], axis=1)
        t_y = tl.sum(fdg * uy[None, :], axis=1)
        t_z = tl.sum(fdg * uz[None, :], axis=1)

        along = radial + dfc * s_g - fc / r * s_c - w / r * t_c
        gx = tl.where(act, along * ux + fc / r * s_x + w / r * t_x, 0.0)
        gy = tl.where(act, along * uy + fc / r * s_y + w / r * t_y, 0.0)
        gz = tl.where(act, along * uz + fc / r * s_z + w / r * t_z, 0.0)

        tl.store(e_ptr + i, e_i)
        tl.atomic_add(f_ptr + safe * 3 + 0, -gx, mask=act)
        tl.atomic_add(f_ptr + safe * 3 + 1, -gy, mask=act)
        tl.atomic_add(f_ptr + safe * 3 + 2, -gz, mask=act)
        tl.atomic_add(f_ptr + i * 3 + 0, tl.sum(gx, axis=0))
        tl.atomic_add(f_ptr + i * 3 + 1, tl.sum(gy, axis=0))
        tl.atomic_add(f_ptr + i * 3 + 2, tl.sum(gz, axis=0))


def neighbor_table(
    positions: torch.Tensor, box: tuple[float, float, float], cutoff: float
) -> torch.Tensor:
    """
    Every neighbour within `cutoff` of every atom, by cell list.

    Parameters
    ----------
    positions : torch.Tensor
        Positions in Angstrom, shape ``(N, 3)``, on CUDA; need not be wrapped
        into the cell.
    box : tuple[float, float, float]
        Orthorhombic cell lengths, each at least ``3 * cutoff``.
    cutoff : float
        Search radius in Angstrom.

    Returns
    -------
    torch.Tensor
        int32, shape ``(N, K)``: row i lists the neighbours of atom i (in no
        particular order) and is padded with -1. K is a power of two of at
        least 16, widened to fit the most crowded atom.
    """
    pos = positions.detach().contiguous()
    n = pos.shape[0]
    cells = [int(length // cutoff) for length in box]
    lengths = torch.tensor(box, dtype=pos.dtype, device=pos.device)
    n_cells = torch.tensor(cells, device=pos.device)
    wrapped = pos - lengths * torch.floor(pos / lengths)
    index = torch.minimum((wrapped / lengths * n_cells).long(), n_cells - 1)
    cell_id = (index[:, 2] * cells[1] + index[:, 1]) * cells[0] + index[:, 0]
    order = torch.argsort(cell_id).to(torch.int32)
    count = torch.bincount(cell_id, minlength=cells[0] * cells[1] * cells[2])
    start = (torch.cumsum(count, 0) - count).to(torch.int32)
    count = count.to(torch.int32)
    occupancy = max(4, triton.next_power_of_2(int(count.max())))
    k = 16
    while True:
        table = torch.full((n, k), -1, dtype=torch.int32, device=pos.device)
        n_found = torch.empty(n, dtype=torch.int32, device=pos.device)
        _cell_kernel[(n,)](
            pos, order, start, count, table, n_found,
            *box, *cells, cutoff * cutoff,
            K=k, OCC=occupancy, num_warps=1,
        )  # fmt: skip
        most = int(n_found.max())
        if most <= k:
            return table
        k = triton.next_power_of_2(most)


def energy_forces(
    positions: torch.Tensor,
    table: torch.Tensor,
    box: tuple[float, float, float],
    params: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ML-BOP total energy and forces from a neighbour table.

    Parameters
    ----------
    positions : torch.Tensor
        Positions in Angstrom, shape ``(N, 3)``, on CUDA.
    table : torch.Tensor
        From :func:`neighbor_table`, with a cutoff of at least the
        potential's ``R + D`` at `positions`.
    box : tuple[float, float, float]
        Orthorhombic cell lengths.
    params : dict[str, float]
        ML-BOP parameters, as :data:`~specter.ice._energy.ML_BOP_PARAMS`.

    Returns
    -------
    energy : torch.Tensor
        Total energy in eV, 0-d, float64 (a float64 sum of float32 per-atom
        energies).
    forces : torch.Tensor
        ``-dE/dpositions`` in eV/Angstrom, float32, shape ``(N, 3)``.
    """
    pos = positions.detach().contiguous()
    n, k = table.shape
    r_cut = params["R"] + params["D"]
    # Atoms with more than 16 neighbours inside the cutoff only occur in a
    # crowded random start; widen the tile for those calls.
    width = 16
    while True:
        active = torch.full((n, width), -1, dtype=torch.int32, device=pos.device)
        n_active = torch.empty(n, dtype=torch.int32, device=pos.device)
        _compact_kernel[(n,)](
            pos, table, active, n_active, *box, r_cut * r_cut,
            K=k, KA=width, num_warps=1,
        )  # fmt: skip
        most = int(n_active.max())
        if most <= width:
            break
        width = triton.next_power_of_2(most)
    e_atom = torch.empty(n, dtype=torch.float32, device=pos.device)
    forces = torch.zeros(n, 3, dtype=torch.float32, device=pos.device)
    _energy_force_kernel[(n,)](
        pos, active, e_atom, forces, *box,
        params["A"], params["B"], params["lambda1"], params["lambda2"],
        params["R"], params["D"], params["beta"] ** params["n"], params["n"],
        params["c"] ** 2, params["d"] ** 2, params["cos_theta0"],
        K=width, num_warps=max(1, width * width // 256),
    )  # fmt: skip
    return e_atom.sum(dtype=torch.float64), forces


class _EnergyWithForces(torch.autograd.Function):
    """Total energy whose backward returns the kernel's forces."""

    @staticmethod
    def forward(ctx, positions, table, box, params):  # type: ignore[override]
        energy, forces = energy_forces(positions, table, box, params)
        ctx.save_for_backward(forces)
        return energy.to(positions.dtype)

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        (forces,) = ctx.saved_tensors
        return -grad * forces.to(grad.dtype), None, None, None


def energy(
    positions: torch.Tensor,
    table: torch.Tensor,
    box: tuple[float, float, float],
    params: dict[str, float],
) -> torch.Tensor:
    """
    :func:`energy_forces`'s energy, differentiable with respect to `positions`.

    The forces are computed with the energy and kept for the backward pass,
    so a gradient costs nothing extra.

    Parameters
    ----------
    positions, table, box, params
        As :func:`energy_forces`.

    Returns
    -------
    torch.Tensor
        Total energy in eV, 0-d, in `positions`' dtype.
    """
    return _EnergyWithForces.apply(positions, table, box, params)
