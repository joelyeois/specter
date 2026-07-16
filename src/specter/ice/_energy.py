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

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from ..progress import track

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
    """

    def __init__(self, params: dict[str, float] | None = None) -> None:
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

    def f_C(self, r: np.ndarray) -> np.ndarray:
        """Cutoff function, Eq. (2)."""
        r = np.asarray(r, dtype=float)
        Rm, Rp = self.R - self.D, self.R + self.D
        fc = np.zeros_like(r)
        fc[r < Rm] = 1.0
        mask = (r >= Rm) & (r <= Rp)
        fc[mask] = 0.5 - 0.5 * np.sin(np.pi * (r[mask] - self.R) / (2 * self.D))
        return fc

    def f_R(self, r: float) -> float:
        return self.A * np.exp(-self.lambda1 * r)

    def f_A(self, r: float) -> float:
        return -self.B * np.exp(-self.lambda2 * r)

    def g(self, cos_theta: float) -> float:
        c2, d2 = self.c**2, self.d**2
        return 1.0 + c2 / d2 - c2 / (d2 + (cos_theta - self.cos_theta0) ** 2)

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
            Whether to show a progress bar over the per-atom energy loop.
            Default is True.

        Returns
        -------
        dict[str, float]
            ``E_total`` (eV), ``E_per_atom`` (eV), ``rij_mean``/``rij_var``
            (Å, bead-bead distance mean/variance) and ``theta_mean``/
            ``theta_var`` (cos(theta) mean/variance over all triplets used
            in the three-body sum).
        """
        n_atoms = len(atoms)
        rij_samples = []  # each unique bond counted once (i < j)
        cos_theta_samples = []  # every cos(theta) actually used in a xi_ij sum

        # 'i', 'j': neighbor pair indices (each pair appears in both
        # directions, i.e. bothways=True convention); 'd': scalar distances;
        # 'D': Cartesian displacement vectors r_j - r_i, already correctly
        # wrapped through PBC/minimum image by ASE.
        i_idx, j_idx, dists, vecs = neighbor_list("ijdD", atoms, cutoff=self.r_cut)

        neighbors: list[list[tuple[int, float, np.ndarray]]] = [
            [] for _ in range(n_atoms)
        ]
        for i, j, rij, rij_vec in zip(i_idx, j_idx, dists, vecs):
            neighbors[i].append((int(j), float(rij), rij_vec))

        E_total = 0.0

        for i in track(
            range(n_atoms),
            description="Computing ML-BOP energy",
            disable=not progressbar,
            transient=True,
        ):
            for j, rij, rij_vec in neighbors[i]:
                fc_ij = self.f_C(np.array([rij]))[0]
                if fc_ij == 0.0:
                    continue
                fR_ij = self.f_R(rij)
                fA_ij = self.f_A(rij)

                if j > i:
                    rij_samples.append(rij)

                # three-body sum xi_ij: loop over k != i, j that are
                # neighbors of i (bond-order is computed w.r.t. atom i)
                xi_ij = 0.0
                for k, rik, rik_vec in neighbors[i]:
                    if k == j:
                        continue
                    fc_ik = self.f_C(np.array([rik]))[0]
                    if fc_ik == 0.0:
                        continue
                    cos_theta = np.dot(rij_vec, rik_vec) / (rij * rik)
                    cos_theta_samples.append(cos_theta)
                    xi_ij += fc_ik * self.g(cos_theta)

                b_ij = (1.0 + (self.beta**self.n) * (xi_ij**self.n)) ** (
                    -1.0 / (2.0 * self.n)
                )
                V_ij = fc_ij * (fR_ij + b_ij * fA_ij)
                E_total += V_ij

        # Each ij bond counted from both i's and j's perspective (with
        # potentially different b_ij vs b_ji), matching the standard
        # Tersoff 1/2 normalization.
        E_total = 0.5 * E_total

        rij_arr = np.asarray(rij_samples, dtype=float)
        cos_theta_arr = np.asarray(cos_theta_samples, dtype=float)

        return {
            "E_total": float(E_total),
            "E_per_atom": float(E_total / n_atoms) if n_atoms else float("nan"),
            "rij_mean": float(np.mean(rij_arr)) if rij_arr.size else float("nan"),
            "rij_var": float(np.var(rij_arr)) if rij_arr.size else float("nan"),
            "theta_mean": float(np.mean(cos_theta_arr))
            if cos_theta_arr.size
            else float("nan"),
            "theta_var": float(np.var(cos_theta_arr))
            if cos_theta_arr.size
            else float("nan"),
        }


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
    itself optimized. Not part of any generation hot path — this loops over
    neighbor triplets in pure Python and is intended for occasional QC
    checks, not per-batch monitoring.

    Parameters
    ----------
    coordinates : torch.Tensor
        Bead positions in Angstrom, shape ``(N, 3)``. Axis order must match
        ``box_size``, but is otherwise arbitrary (energies are translation-
        and axis-order-invariant) — for example
        ``APIcemaker.ice_coordinates[b] * icemaker.dx`` (voxel-index order
        ``(z, y, x)``, uncentered) or ``GradientSKIcemaker.positions`` /
        ``RandomIcemaker.positions`` (``(x, y, z)``, centered at the origin)
        both work as long as ``box_size`` uses the same axis order.
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

    model = MLBOP()
    if pbc:
        atoms = _ensure_min_cell_size(atoms, model.r_cut)

    return model.compute_energy(atoms, progressbar=progressbar)
