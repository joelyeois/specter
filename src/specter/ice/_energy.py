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
A and B are in eV, lambda1/lambda2 are in Angstrom^-1, R/D are in
Angstrom, and c/d/cos_theta0/n/beta are dimensionless. Consequently
:meth:`MLBOP.compute_energy` returns energies in eV, provided atomic
positions are supplied in Angstrom.

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

import numpy as np
import torch
import vesin_torch
from ase import Atoms
from ase.neighborlist import neighbor_list

# ML-BOP, Table 3, Chan et al., Nat. Commun. 10, 379 (2019)
ML_BOP_PARAMS: dict[str, float] = {
    "A": 1684.301476,  # eV, repulsive prefactor
    "B": 473.621419,  # eV, attractive prefactor
    "lambda1": 2.750522,  # Angstrom^-1, repulsive decay length^-1
    "lambda2": 2.199640,  # Angstrom^-1, attractive decay length^-1
    "R": 3.282761,  # Angstrom, cutoff midpoint
    "D": 0.270511,  # Angstrom, cutoff half-width
    "beta": 1e-06,  # dimensionless, bond-order prefactor
    "n": 0.770018,  # dimensionless, bond-order exponent
    "c": 77638.534354,  # dimensionless, angular function parameter
    "d": 16.148387,  # dimensionless, angular function parameter
    "cos_theta0": -0.471029,  # dimensionless, preferred bond angle cosine
}


def _ensure_min_cell_size(atoms: Atoms, min_length: float) -> Atoms:
    """
    Tile a periodic structure so every lattice vector clears ``min_length``.

    Guards against the same neighbor atom appearing via multiple periodic
    images within the cutoff, which the ML-BOP three-body sum does not
    disambiguate. A 2x safety margin is used because the minimum-image
    convention requires each axis to be longer than *twice* the cutoff,
    not just longer than it.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to check.
    min_length : float
        The interaction cutoff (``r_cut = R + D``) in Angstrom.

    Returns
    -------
    ase.Atoms
        Either the original atoms (if already large enough or non-periodic),
        or a repeated copy.
    """
    if not np.any(atoms.get_pbc()):
        return atoms

    lengths = atoms.cell.lengths()
    safe_length = 2.0 * min_length
    reps = [
        int(np.ceil(safe_length / length)) if length > 1e-6 else 1 for length in lengths
    ]

    if all(r <= 1 for r in reps):
        return atoms

    return atoms.repeat(tuple(reps))


class MLBOP:
    """
    Evaluates ML-BOP potential energy for a set of coarse-grained water bead
    positions, with optional orthorhombic periodic boundary conditions.

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

        Shared by :meth:`compute_energy` (pairs and distances straight from
        ASE, no grad) and :meth:`compute_energy_differentiable` (same pair
        indices from ASE, but distances/vectors recomputed from live
        positions so grad flows through) -- keeping this logic in one place
        means the two entry points are guaranteed to agree numerically.

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
        self, atoms: Atoms, progressbar: bool = True
    ) -> dict[str, float]:
        """
        Compute total, per-atom ML-BOP energy and O-O structural statistics.

        Parameters
        ----------
        atoms : ase.Atoms
            Bead positions in Angstrom. If ``atoms.pbc`` is set (with a
            non-degenerate cell), periodic boundary conditions are applied
            automatically, including cells smaller than the cutoff (ASE's
            neighbor list transparently extends to however many periodic
            images are needed).
        progressbar : bool, optional
            Unused -- kept for backwards compatibility. The computation
            below is a single batched op with no per-atom loop to report
            progress over.

        Returns
        -------
        dict[str, float]
            ``E_total`` (eV), ``E_per_atom`` (eV), ``rij_mean``/``rij_var``
            (Å, bead-bead distance mean/variance) and ``theta_mean``/
            ``theta_var`` (cos(theta) mean/variance over all triplets used
            in the three-body sum).
        """
        n_atoms = len(atoms)
        device = self.device
        nan_result = {
            "E_total": 0.0,
            "E_per_atom": float("nan"),
            "rij_mean": float("nan"),
            "rij_var": float("nan"),
            "theta_mean": float("nan"),
            "theta_var": float("nan"),
        }
        if n_atoms == 0:
            return nan_result

        # 'i', 'j': neighbor pair indices (each pair appears in both
        # directions, i.e. bothways=True convention); 'd': scalar distances;
        # 'D': Cartesian displacement vectors r_j - r_i, already correctly
        # wrapped through PBC/minimum image by ASE (this part stays on ASE's
        # cell-list neighbor search, which is what correctly handles
        # triclinic MD cells -- reimplementing general minimum-image PBC
        # is not worth it since this isn't the bottleneck).
        i_idx, j_idx, dists, vecs = neighbor_list("ijdD", atoms, cutoff=self.r_cut)
        if len(i_idx) == 0:
            return nan_result

        i_idx_t = torch.as_tensor(i_idx, dtype=torch.long, device=device)
        j_idx_t = torch.as_tensor(j_idx, dtype=torch.long, device=device)
        rij_t = torch.as_tensor(dists, dtype=torch.float64, device=device)
        vec_t = torch.as_tensor(vecs, dtype=torch.float64, device=device)

        result = self._energy_from_pairs(n_atoms, i_idx_t, j_idx_t, rij_t, vec_t)
        return {k: v.item() for k, v in result.items()}

    def compute_energy_differentiable(
        self,
        positions: torch.Tensor,
        box_size: tuple[float, float, float] | float,
        pbc: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Differentiable counterpart to :meth:`compute_energy`.

        :meth:`compute_energy` takes a plain :class:`ase.Atoms`, which means
        ``coordinates.detach().cpu().numpy()`` has already happened by the
        time it runs -- fine for the QC-diagnostic use case, but useless as
        a loss term since gradients can't flow back to the positions that
        generated it.

        Uses `vesin <https://github.com/Luthaf/vesin>`_'s ``torch`` neighbor
        list (the ``vesin-torch`` package) instead of ASE: it resolves the
        same discrete pair list (which atoms are within cutoff, and how many
        periodic images apart -- inherently non-differentiable choices in
        any neighbor-search implementation), but its returned distances and
        displacement vectors are themselves ``torch`` ops wired directly to
        the input positions, so no manual recomputation is needed and
        ``E_total``/``E_per_atom`` carry a ``grad_fn`` straight back to
        ``positions``. Unlike the ASE path, this never leaves ``torch``/the
        input device -- positions on GPU stay on GPU, with no host sync per
        call (ASE's neighbor search is numpy/CPU-only, so a differentiable
        wrapper around it -- an earlier version of this method -- pays a
        host<->device round trip every optimizer step; profiling that
        version showed it accounted for roughly a third of its added
        per-step cost relative to :meth:`compute_energy`).

        Parameters
        ----------
        positions : torch.Tensor
            Bead positions in Angstrom, shape ``(N, 3)``, on ``self.device``.
            May require grad.
        box_size : float or tuple[float, float, float]
            Periodic box lengths in Angstrom, in the same axis order as
            ``positions``. A single float is broadcast to a cubic box.
        pbc : bool, optional
            Whether to apply periodic boundary conditions. Default True.

        Returns
        -------
        dict[str, torch.Tensor]
            Same fields as :meth:`compute_energy`, as 0-d tensors on
            ``positions``'s device/dtype (``E_total``/``E_per_atom`` retain
            a ``grad_fn``; the rest are diagnostics with no gradient use).

        Raises
        ------
        ValueError
            If ``pbc`` is True and ``box_size`` is smaller than twice the
            cutoff along any axis. :meth:`compute_energy` handles this by
            repeating the ASE cell (see :func:`_ensure_min_cell_size`); this
            method instead just rejects it, matching that same constraint
            rather than relying on vesin's (correct, but separately
            unverified here) handling of that regime -- use a larger box.
        """
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError(
                f"positions must have shape (N, 3), got {tuple(positions.shape)}"
            )
        if isinstance(box_size, (int, float)):
            box_size = (float(box_size),) * 3

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

        if pbc and any(length < 2 * self.r_cut for length in box_size):
            raise ValueError(
                "box_size is smaller than 2x the ML-BOP cutoff "
                f"({2 * self.r_cut:.3f} Å) along at least one axis. "
                "compute_energy() handles this by repeating the ASE cell; "
                "compute_energy_differentiable() does not -- use a larger "
                "box_size."
            )

        box_t = torch.diag(torch.as_tensor(box_size, dtype=dtype, device=device))
        nl = vesin_torch.NeighborList(cutoff=self.r_cut, full_list=True)
        i_idx_t, j_idx_t, rij_t, vec_t = nl.compute(
            positions, box_t, periodic=pbc, quantities="ijdD"
        )
        if i_idx_t.numel() == 0:
            return nan_result

        return self._energy_from_pairs(n_atoms, i_idx_t, j_idx_t, rij_t, vec_t)


def mlbop_energy(
    coordinates: torch.Tensor,
    box_size: tuple[float, float, float] | float,
    pbc: bool = True,
    progressbar: bool = True,
) -> dict[str, float]:
    """
    Score a single generated ice bead configuration with the ML-BOP potential.

    Treats every coordinate as a coarse-grained water bead (one bead per
    water molecule) and reports the ML-BOP energy, a lower-is-more-ice-like
    structural diagnostic independent of whatever objective the icemaker
    itself optimized. Not part of any generation hot path — the three-body
    sum runs as a single batched ``torch`` computation on ``coordinates``'s
    own device (see :meth:`MLBOP.compute_energy`), but it is still intended
    for occasional QC checks, not per-batch monitoring.

    Parameters
    ----------
    coordinates : torch.Tensor
        Bead positions in Angstrom, shape ``(N, 3)``. Axis order must match
        ``box_size``, but is otherwise arbitrary (energies are translation-
        and axis-order-invariant) — for example ``GradientSKIcemaker.positions``,
        ``IceBank.positions``, or ``RandomIcemaker.positions`` (``(x, y, z)``,
        centered at the origin) all work as long as ``box_size`` uses the
        same axis order.
    box_size : float or tuple[float, float, float]
        Periodic box lengths in Angstrom, in the same axis order as
        ``coordinates``. A single float is broadcast to a cubic box.
    pbc : bool, optional
        Whether to apply periodic boundary conditions using ``box_size``.
        Default is True, matching the periodic tiling used when assembling
        ice volumes (:func:`~specter.ice._helpers.assemble_big_ice`).
    progressbar : bool, optional
        Whether to show a progress bar over the per-atom energy loop.
        Default is True.

    Returns
    -------
    dict[str, float]
        See :meth:`MLBOP.compute_energy` for the fields returned.
    """
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"coordinates must have shape (N, 3), got {tuple(coordinates.shape)}"
        )

    if isinstance(box_size, (int, float)):
        box_size = (float(box_size),) * 3

    device = coordinates.device
    positions = coordinates.detach().cpu().numpy().astype(float)
    # ASE's cell is anchored at the origin, spanning [0, box_size) along each
    # axis. Icemaker coordinate conventions vary (e.g. RandomIcemaker/
    # GradientSKIcemaker/MDSimDump centre at the origin, [-box/2, box/2)),
    # and ASE's non-periodic binning silently clips out-of-cell coordinates
    # to the boundary bin rather than rejecting them -- so centred,
    # non-periodic input would otherwise pile every atom with a negative
    # coordinate onto a single edge bin. Shift into [0, box_size) so the
    # function is translation-invariant regardless of the input convention.
    positions = positions - positions.min(axis=0, keepdims=True)
    atoms = Atoms(
        numbers=np.full(len(positions), 8),  # oxygen stand-in for a CG water bead
        positions=positions,
        cell=np.diag(box_size),
        pbc=pbc,
    )

    model = MLBOP(device=device)
    if pbc:
        atoms = _ensure_min_cell_size(atoms, model.r_cut)

    return model.compute_energy(atoms, progressbar=progressbar)
