"""
ML-BOP coarse-grained water potential — structural diagnostic for ice beads.

Implements the machine-learned bond-order potential (ML-BOP) for
coarse-grained water, as described in:

    Chan, H., Cherukara, M.J., Narayanan, B., Loeffler, T.D., Benmore, C.,
    Gray, S.K., & Sankaranarayanan, S.K.R.S.
    "Machine learning coarse grained models for water"
    Nature Communications 10, 379 (2019). https://doi.org/10.1038/s41467-018-08222-6

The model follows the Tersoff-Brenner bond-order formalism:

    V_pair(r_ij) = f_C(r_ij) * [f_R(r_ij) + b_ij * f_A(r_ij)]

    f_C(r): cutoff function (Eq. 2)
    f_R(r) = A * exp(-lambda1 * r)          repulsive term
    f_A(r) = -B * exp(-lambda2 * r)         attractive term
    b_ij = (1 + beta^n * xi_ij^n)^(-1/(2n)) bond-order term
    xi_ij = sum_{k != i,j} f_C(r_ik) * g(theta_ijk)
    g(theta) = 1 + c^2/d^2 - c^2 / [d^2 + (cos(theta) - cos(theta0))^2]

Total energy (standard Tersoff normalization, since b_ij != b_ji in general):

    E = 0.5 * sum_i sum_{j != i} f_C(r_ij) * [f_R(r_ij) + b_ij * f_A(r_ij)]

This is used here purely as a **diagnostic**: each icemaker bead is a
coarse-grained water molecule, so scoring a generated configuration with
ML-BOP gives a lower-is-more-ice-like structural sanity check that is
independent of whatever objective the icemaker itself optimized (RDF
matching, kernel matching, etc.). It is not wired into any generation
hot path — call it manually on coordinates you want to inspect.

UNITS
-----
A and B are in eV, lambda1/lambda2 are in Å^-1, R/D are in
Å, and c/d/cos_theta0/n/beta are dimensionless. Consequently
:meth:`MLBOP.compute_energy` returns energies in eV, provided atomic
positions are supplied in Å.

BACKEND
-------
The pairwise neighbor search runs on `vesin <https://github.com/Luthaf/vesin>`_'s
``torch`` bindings (the ``vesin-torch`` package), not ASE: it stays entirely
in ``torch`` (positions on GPU stay on GPU, no host<->device round trip per
call), supports a general (triclinic) periodic cell as well as an
orthorhombic one, and its returned distances/displacement vectors are
themselves differentiable w.r.t. the input positions, so :meth:`MLBOP.
compute_energy` doubles as a loss term (e.g. in :class:`~specter.ice.
_bank.IceBank`'s seam relaxation) as well as a QC diagnostic.

NOTE ON THE EXTRA TABLE-3 ROW (m, Gamma, lambda3)
--------------------------------------------------
The published parameter table also lists m=1.0, Gamma=1.0, lambda3=0.0.
These belong to the generalized LAMMPS-style Tersoff three-body term,
which (in full generality) multiplies g(theta) by an extra factor
Gamma * exp[lambda3^m * (r_ij - r_ik)^m] inside the xi_ij sum. With
Gamma=1 and lambda3=0, this factor is identically 1 for all distances,
so it has no effect and is correctly omitted from xi_ij below --
matching the xi_ij definition in Eq. (6) exactly.
"""

from __future__ import annotations

import math

import torch
import vesin_torch

# ML-BOP, Table 3, Chan et al., Nat. Commun. 10, 379 (2019)
ML_BOP_PARAMS: dict[str, float] = {
    "A": 1684.301476,  # eV, repulsive prefactor
    "B": 473.621419,  # eV, attractive prefactor
    "lambda1": 2.750522,  # Å^-1, repulsive decay length^-1
    "lambda2": 2.199640,  # Å^-1, attractive decay length^-1
    "R": 3.282761,  # Å, cutoff midpoint
    "D": 0.270511,  # Å, cutoff half-width
    "beta": 1e-06,  # dimensionless, bond-order prefactor
    "n": 0.770018,  # dimensionless, bond-order exponent
    "c": 77638.534354,  # dimensionless, angular function parameter
    "d": 16.148387,  # dimensionless, angular function parameter
    "cos_theta0": -0.471029,  # dimensionless, preferred bond angle cosine
}


class MLBOP:
    """
    Evaluates ML-BOP potential energy for a set of coarse-grained water bead
    positions, with optional periodic boundary conditions (orthorhombic or
    general/triclinic).

    Parameters
    ----------
    params : dict[str, float], optional
        ML-BOP parameters. Defaults to :data:`ML_BOP_PARAMS`.
    device : torch.device or str, optional
        Device the batched triple-sum in :meth:`compute_energy` runs on.
        Default is ``"cpu"``.
    """

    def __init__(
        self,
        params: dict[str, float] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        params = ML_BOP_PARAMS if params is None else params
        self.A = params["A"]
        self.B = params["B"]
        self.lambda1 = params["lambda1"]
        self.lambda2 = params["lambda2"]
        self.R = params["R"]
        self.D = params["D"]
        self.beta = params["beta"]
        self.n = params["n"]
        self.c = params["c"]
        self.d = params["d"]
        self.cos_theta0 = params["cos_theta0"]
        self.r_cut = self.R + self.D
        self.device = torch.device(device)

    def f_C(self, r: torch.Tensor) -> torch.Tensor:
        """Cutoff function, Eq. (2)."""
        Rm, Rp = self.R - self.D, self.R + self.D
        fc = torch.zeros_like(r)
        fc = torch.where(r < Rm, torch.ones_like(r), fc)
        mask = (r >= Rm) & (r <= Rp)
        ramp = 0.5 - 0.5 * torch.sin(math.pi * (r - self.R) / (2 * self.D))
        return torch.where(mask, ramp, fc)

    def f_R(self, r: torch.Tensor) -> torch.Tensor:
        return self.A * torch.exp(-self.lambda1 * r)

    def f_A(self, r: torch.Tensor) -> torch.Tensor:
        return -self.B * torch.exp(-self.lambda2 * r)

    def g(self, cos_theta: torch.Tensor) -> torch.Tensor:
        c2, d2 = self.c**2, self.d**2
        return 1.0 + c2 / d2 - c2 / (d2 + (cos_theta - self.cos_theta0) ** 2)

    def _energy_from_pairs(
        self,
        n_atoms: int,
        i_idx_t: torch.Tensor,
        j_idx_t: torch.Tensor,
        rij_t: torch.Tensor,
        vec_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Batched three-body sum given an already-resolved neighbor pair list.

        Shared core used by :meth:`compute_energy`, kept as its own method
        since it's a substantial, independently-testable piece of physics
        (the Tersoff-style three-body sum) rather than backend-specific
        plumbing.

        Parameters
        ----------
        n_atoms : int
            Number of atoms.
        i_idx_t, j_idx_t : torch.Tensor
            Long tensors of neighbor pair indices, shape ``(n_pairs,)``.
        rij_t : torch.Tensor
            Pair distances, shape ``(n_pairs,)``.
        vec_t : torch.Tensor
            Displacement vectors r_j - r_i (+ periodic shift), shape
            ``(n_pairs, 3)``.

        Returns
        -------
        dict[str, torch.Tensor]
            0-d tensors: ``E_total``, ``E_per_atom``, ``rij_mean``,
            ``rij_var``, ``theta_mean``, ``theta_var``.
        """
        device = rij_t.device
        dtype = rij_t.dtype

        # Group the flat pair list by central atom i, then scatter into a
        # dense (n_atoms, max_neighbors) padded layout. This lets the whole
        # three-body sum -- naively a Python loop over i with a (m_i x m_i)
        # matrix per atom -- become a single batched (n_atoms x
        # max_neighbors x max_neighbors) einsum with no Python-level loop
        # at all. Padding cost is linear in n_atoms for any physically
        # reasonable configuration, since the ~3.55 Å cutoff bounds how
        # many neighbors a single bead can have.
        order = torch.argsort(i_idx_t, stable=True)
        i_sorted = i_idx_t[order]
        j_sorted = j_idx_t[order]
        rij_sorted = rij_t[order]
        vec_sorted = vec_t[order]

        counts = torch.bincount(i_sorted, minlength=n_atoms)
        max_m = int(counts.max().item())
        group_start = torch.zeros(n_atoms, dtype=torch.long, device=device)
        group_start[1:] = torch.cumsum(counts, dim=0)[:-1]
        col = torch.arange(len(i_idx_t), device=device) - group_start.repeat_interleave(
            counts
        )

        pad_j = torch.full((n_atoms, max_m), -1, dtype=torch.long, device=device)
        pad_rij = torch.zeros((n_atoms, max_m), dtype=dtype, device=device)
        pad_vec = torch.zeros((n_atoms, max_m, 3), dtype=dtype, device=device)
        mask = torch.zeros((n_atoms, max_m), dtype=torch.bool, device=device)

        pad_j[i_sorted, col] = j_sorted
        pad_rij[i_sorted, col] = rij_sorted
        pad_vec[i_sorted, col] = vec_sorted
        mask[i_sorted, col] = True

        fc = torch.zeros_like(pad_rij)
        fc[mask] = self.f_C(pad_rij[mask])
        valid = mask & (fc > 0.0)

        fR = torch.zeros_like(pad_rij)
        fA = torch.zeros_like(pad_rij)
        fR[valid] = self.f_R(pad_rij[valid])
        fA[valid] = self.f_A(pad_rij[valid])

        # cos(theta_jik) and g(theta_jik) for every (j, k) neighbor pair of
        # every atom at once (axis 1 = j, axis 2 = k).
        dot = torch.einsum("imc,inc->imn", pad_vec, pad_vec)
        rr = pad_rij.unsqueeze(2) * pad_rij.unsqueeze(1)
        safe_rr = torch.where(rr != 0, rr, torch.ones_like(rr))
        cos_theta = dot / safe_rr
        g_vals = self.g(cos_theta)

        eye = torch.eye(max_m, dtype=torch.bool, device=device).unsqueeze(0)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & ~eye

        # fc broadcast over k (axis 2), matching xi_ij = sum_k f_C(r_ik) *
        # g(theta_jik) -- the weight depends only on k, not on j.
        contrib = torch.where(
            pair_valid, fc.unsqueeze(1) * g_vals, torch.zeros_like(g_vals)
        )
        xi = contrib.sum(dim=2)

        b = torch.zeros_like(pad_rij)
        b[valid] = (1.0 + (self.beta**self.n) * (xi[valid] ** self.n)) ** (
            -1.0 / (2.0 * self.n)
        )

        V = torch.where(valid, fc * (fR + b * fA), torch.zeros_like(fc))
        # Each ij bond counted from both i's and j's perspective (with
        # potentially different b_ij vs b_ji), matching the standard
        # Tersoff 1/2 normalization.
        E_total = 0.5 * V.sum()

        atom_idx = torch.arange(n_atoms, device=device).unsqueeze(1)
        unique_bond = valid & (
            pad_j > atom_idx
        )  # each unique bond counted once (i < j)
        rij_arr = pad_rij[unique_bond]
        cos_theta_arr = cos_theta[pair_valid]  # every cos(theta) used in a xi_ij sum

        nan = torch.full((), float("nan"), dtype=dtype, device=device)
        return {
            "E_total": E_total,
            "E_per_atom": E_total / n_atoms,
            "rij_mean": rij_arr.mean() if rij_arr.numel() else nan,
            "rij_var": rij_arr.var(unbiased=False) if rij_arr.numel() else nan,
            "theta_mean": cos_theta_arr.mean() if cos_theta_arr.numel() else nan,
            "theta_var": cos_theta_arr.var(unbiased=False)
            if cos_theta_arr.numel()
            else nan,
        }

    def compute_energy(
        self,
        positions: torch.Tensor,
        box_size: tuple[float, float, float] | float | torch.Tensor,
        pbc: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Compute total, per-atom ML-BOP energy and O-O structural statistics.

        Uses `vesin <https://github.com/Luthaf/vesin>`_'s ``torch`` neighbor
        list (the ``vesin-torch`` package): it stays entirely in ``torch``
        (positions on GPU stay on GPU, no host<->device round trip per
        call), and its returned distances/displacement vectors are
        themselves ``torch`` ops wired directly to ``positions``, so
        ``E_total``/``E_per_atom`` carry a ``grad_fn`` back to it -- this
        doubles as a differentiable loss term (e.g. in :class:`~specter.
        ice._bank.IceBank`'s seam relaxation) as well as a QC diagnostic.

        Parameters
        ----------
        positions : torch.Tensor
            Bead positions in Å, shape ``(N, 3)``, on ``self.device``.
            May require grad.
        box_size : float, tuple[float, float, float], or torch.Tensor
            Periodic cell. A single float is broadcast to a cubic
            orthorhombic cell; a 3-tuple gives an orthorhombic cell's own
            (x, y, z) lengths (in the same axis order as ``positions``); a
            ``(3, 3)`` tensor is used directly as a general (triclinic)
            cell matrix, rows as lattice vectors (ASE's convention).
        pbc : bool, optional
            Whether to apply periodic boundary conditions. Default True.

        Returns
        -------
        dict[str, torch.Tensor]
            0-d tensors on ``positions``'s device/dtype: ``E_total`` (eV),
            ``E_per_atom`` (eV), ``rij_mean``/``rij_var`` (Å, bead-bead
            distance mean/variance) and ``theta_mean``/``theta_var``
            (cos(theta) mean/variance over all triplets used in the
            three-body sum). ``E_total``/``E_per_atom`` retain a
            ``grad_fn``; the rest are diagnostics with no gradient use.
        """
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError(
                f"positions must have shape (N, 3), got {tuple(positions.shape)}"
            )

        n_atoms = positions.shape[0]
        device = positions.device
        dtype = positions.dtype
        nan_result = {
            "E_total": torch.zeros((), dtype=dtype, device=device),
            "E_per_atom": torch.full((), float("nan"), dtype=dtype, device=device),
            "rij_mean": torch.full((), float("nan"), dtype=dtype, device=device),
            "rij_var": torch.full((), float("nan"), dtype=dtype, device=device),
            "theta_mean": torch.full((), float("nan"), dtype=dtype, device=device),
            "theta_var": torch.full((), float("nan"), dtype=dtype, device=device),
        }
        if n_atoms == 0:
            return nan_result

        if isinstance(box_size, torch.Tensor):
            box_t = box_size.to(dtype=dtype, device=device)
        else:
            if isinstance(box_size, (int, float)):
                box_size = (float(box_size),) * 3
            box_t = torch.diag(torch.as_tensor(box_size, dtype=dtype, device=device))

        nl = vesin_torch.NeighborList(cutoff=self.r_cut, full_list=True)
        i_idx_t, j_idx_t, rij_t, vec_t = nl.compute(
            positions, box_t, periodic=pbc, quantities="ijdD"
        )
        if i_idx_t.numel() == 0:
            return nan_result

        return self._energy_from_pairs(n_atoms, i_idx_t, j_idx_t, rij_t, vec_t)
