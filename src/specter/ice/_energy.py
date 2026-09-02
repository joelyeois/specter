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


class NeighborListCache:
    """
    A Verlet-skin neighbour list, reused across nearby evaluations.

    The list is built with cutoff ``r_cut + skin`` and kept, together with
    the positions it was built from, until any atom has moved further than
    ``skin / 2`` from those positions -- the standard molecular-dynamics
    guarantee that no pair can have crossed ``r_cut`` unnoticed. Each
    evaluation recomputes the pair vectors from the CURRENT positions and the
    cached ``(i, j, cell shift)``, so distances stay differentiable with
    respect to the positions and pairs that have drifted past ``r_cut``
    simply carry ``f_C = 0``.

    Worth having wherever the same system is evaluated repeatedly at nearby
    positions: an L-BFGS line search evaluates the ice loss several times per
    step, and vesin's cell-list search was 48 ms of every one of those
    evaluations. In a full 600-step ice optimisation the list was rebuilt on
    7% of the evaluations.

    Parameters
    ----------
    skin : float, optional
        Extra cutoff radius in Angstrom. Default 1.0.
    """

    def __init__(self, skin: float = 1.0) -> None:
        self.skin = skin
        self.reference: torch.Tensor | None = None
        self.i: torch.Tensor | None = None
        self.j: torch.Tensor | None = None
        self.shift: torch.Tensor | None = None
        self.rebuilds = 0
        self.evaluations = 0

    def pairs(
        self, positions: torch.Tensor, box_t: torch.Tensor, r_cut: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ``(i, j, shift)`` for `positions`, rebuilt only when needed.

        Parameters
        ----------
        positions : torch.Tensor
            Current positions, shape ``(N, 3)``; may require grad (only the
            detached values are used to decide on a rebuild).
        box_t : torch.Tensor
            Cell matrix, shape ``(3, 3)``.
        r_cut : float
            The potential's own cutoff; the list is built at ``r_cut + skin``.

        Returns
        -------
        i, j : torch.Tensor
            Long, shape ``(n_pairs,)``.
        shift : torch.Tensor
            Periodic shift to add to ``positions[j] - positions[i]``, in the
            same units as `positions`, shape ``(n_pairs, 3)``.
        """
        pos = positions.detach()
        self.evaluations += 1
        if self.reference is not None and self.i is not None:
            moved = float((pos - self.reference).abs().max())
            if moved <= self.skin / 2:
                assert self.j is not None and self.shift is not None
                return self.i, self.j, self.shift
        nl = vesin_torch.NeighborList(cutoff=r_cut + self.skin, full_list=True)
        i, j, S = nl.compute(pos, box_t, periodic=True, quantities="ijS")
        self.i, self.j = i, j
        self.shift = S.to(pos.dtype) @ box_t
        self.reference = pos.clone()
        self.rebuilds += 1
        return i, j, self.shift


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
        Three-body sum given an already-resolved neighbor pair list.

        Shared core used by :meth:`compute_energy`, kept as its own method
        since it's a substantial, independently-testable piece of physics
        (the Tersoff-style three-body sum) rather than backend-specific
        plumbing.

        Parameters
        ----------
        n_atoms : int
            Number of atoms.
        i_idx_t, j_idx_t : torch.Tensor
            Long tensors of neighbor pair indices, shape ``(n_pairs,)``, a
            FULL list (both (i, j) and (j, i) present).
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

        Notes
        -----
        xi_ij = sum_{k != j} f_C(r_ik) g(theta_jik) runs over every ordered
        pair (i, j) and every other neighbour k of the same atom i, so it is
        evaluated over an explicit triplet list: pair p = (i, j) paired with
        every pair q = (i, k) of the same atom, q != p, which is exactly
        sum_i m_i^2 entries. An earlier form scattered the pairs into a dense
        ``(n_atoms, max_m, max_m)`` block and reduced the whole block, i.e.
        ``n_atoms * max_m**2`` entries sized by the single most-crowded atom:
        211M for 21.6M real triplets on a random 527k-atom start (53M for
        10.8M once the structure tightens), with a 9 GiB transient and a
        peak that moved with the trajectory. The triplet form gives the same
        energy and gradient (to 1e-17 in float64) at ~4x less memory.

        Gathers use ``index_select`` rather than advanced indexing on
        purpose: the backward of ``x[idx]`` is a sorted ``index_put_``, that
        of ``index_select`` an atomic ``index_add_``, and the difference was
        ~28 ms per loss evaluation in the ice optimiser.
        """
        device = rij_t.device
        dtype = rij_t.dtype

        # Group the pair list by central atom i (stable, so the order within
        # an atom's group is the input order).
        order = torch.argsort(i_idx_t, stable=True)
        i_s = i_idx_t[order]
        j_s = j_idx_t[order]
        r_s = rij_t.index_select(0, order)
        v_s = vec_t.index_select(0, order)
        n_pairs = i_s.numel()

        counts = torch.bincount(i_s, minlength=n_atoms)
        starts = torch.zeros(n_atoms, dtype=torch.long, device=device)
        starts[1:] = torch.cumsum(counts, dim=0)[:-1]

        fc = self.f_C(r_s)
        valid = fc > 0.0

        # Triplets: for pair p = (i, j), every pair q = (i, k) in i's group.
        m_p = counts[i_s]  # neighbours of p's own atom
        trip_p = torch.repeat_interleave(torch.arange(n_pairs, device=device), m_p)
        offs = torch.zeros(n_pairs, dtype=torch.long, device=device)
        offs[1:] = torch.cumsum(m_p, dim=0)[:-1]
        local = torch.arange(trip_p.numel(), device=device) - offs[trip_p]
        trip_q = starts[i_s[trip_p]] + local
        keep = (trip_q != trip_p) & valid[trip_p] & valid[trip_q]
        trip_p = trip_p[keep]
        trip_q = trip_q[keep]

        # cos(theta_jik) for every (j, k) pair of neighbours of i
        dot = (v_s.index_select(0, trip_p) * v_s.index_select(0, trip_q)).sum(-1)
        cos_theta = dot / (r_s.index_select(0, trip_p) * r_s.index_select(0, trip_q))
        # xi_ij = sum_k f_C(r_ik) g(theta_jik): the weight depends on k only
        contrib = fc.index_select(0, trip_q) * self.g(cos_theta)
        xi = torch.zeros(n_pairs, dtype=dtype, device=device).index_add_(
            0, trip_p, contrib
        )

        b = (1.0 + (self.beta**self.n) * (xi**self.n)) ** (-1.0 / (2.0 * self.n))
        V = torch.where(
            valid, fc * (self.f_R(r_s) + b * self.f_A(r_s)), torch.zeros_like(fc)
        )
        # Each ij bond counted from both i's and j's perspective (with
        # potentially different b_ij vs b_ji), matching the standard
        # Tersoff 1/2 normalization.
        E_total = 0.5 * V.sum()

        unique_bond = valid & (j_s > i_s)  # each unique bond counted once (i < j)
        rij_arr = r_s[unique_bond]
        cos_theta_arr = cos_theta  # every cos(theta) used in a xi_ij sum

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
        neighbor_cache: NeighborListCache | None = None,
        compute_dtype: torch.dtype | None = None,
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
        neighbor_cache : NeighborListCache, optional
            Reuse a Verlet-skin neighbour list across calls at nearby
            positions instead of searching afresh each time (periodic
            systems only). See :class:`NeighborListCache`. Default None.
        compute_dtype : torch.dtype, optional
            Evaluate the pair and three-body kernels in this dtype. The pair
            vectors are formed in `positions`' own dtype first, so float64
            positions keep a resolution float32 coordinates cannot express
            (an ulp at 128 A is 1.5e-5 A) while the sum itself runs at
            float32 speed. Default None: the kernels run in `positions`'
            dtype.

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

        if neighbor_cache is not None:
            if not pbc:
                raise ValueError("neighbor_cache requires pbc=True")
            i_idx_t, j_idx_t, shift = neighbor_cache.pairs(positions, box_t, self.r_cut)
            vec_t = (
                positions.index_select(0, j_idx_t)
                - positions.index_select(0, i_idx_t)
                + shift
            )
            rij_t = vec_t.norm(dim=1)
        else:
            nl = vesin_torch.NeighborList(cutoff=self.r_cut, full_list=True)
            i_idx_t, j_idx_t, rij_t, vec_t = nl.compute(
                positions, box_t, periodic=pbc, quantities="ijdD"
            )
        if compute_dtype is not None and vec_t.dtype != compute_dtype:
            vec_t = vec_t.to(compute_dtype)
            rij_t = vec_t.norm(dim=1)
        if i_idx_t.numel() == 0:
            return nan_result

        return self._energy_from_pairs(n_atoms, i_idx_t, j_idx_t, rij_t, vec_t)
