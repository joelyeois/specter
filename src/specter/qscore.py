"""
Per-atom Q-score calculation using PyTorch with optional GPU support.

Implements the algorithm of Pintilie et al. (2020) Science 370 (6516).
Each atom receives a Pearson correlation score between the density sampled
on concentric shells around it and the profile of an ideal Gaussian peak.

Based on the original qscore implementation by Jamali et al.:
    https://github.com/jamaliki/qscore
"""

from __future__ import annotations

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from specter.pdb import PDB

# Standard residue and atom names — mirrors qscore's selection so per-atom
# scores are directly comparable to published Q-scores.
_STANDARD_RESIDUES: frozenset[str] = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "DA",
        "DC",
        "DG",
        "DT",
        "A",
        "C",
        "G",
        "U",
    }
)
_STANDARD_ATOMS: frozenset[str] = frozenset(
    {
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "OD1",
        "ND2",
        "OD2",
        "SG",
        "OE1",
        "NE2",
        "OE2",
        "ND1",
        "CD2",
        "CE1",
        "CG1",
        "CG2",
        "CD1",
        "CE",
        "NZ",
        "SD",
        "CE2",
        "OG",
        "OG1",
        "NE1",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "OH",
        "OP1",
        "P",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N9",
        "C4",
        "N3",
        "C2",
        "N1",
        "C6",
        "C5",
        "N7",
        "C8",
        "N6",
        "O2",
        "N4",
        "N2",
        "O6",
        "O4",
        "C7",
        "O2'",
        "OXT",
    }
)


class QScore:
    """Per-atom Q-score calculator using PyTorch.

    Parameters
    ----------
    device : str
        Torch device string: ``'cpu'``, ``'cuda'``, ``'cuda:1'``, etc.
    n_radii : int
        Number of radial shells. Default 21 → R = 0.0, 0.1, …, 2.0 Å.
    n_points : int
        Points sampled uniformly on each shell per atom. Default 8.
    ref_gaussian_width : float
        σ of the reference Gaussian in Å. Default 0.6.
    k_neighbors : int
        Number of nearest neighbours used for the sphere-point ownership
        check. Any atom within 2 × R_max = 4 Å can steal a sphere point;
        the default of 32 covers all realistic protein packing densities.
    epsilon : float
        Denominator regularisation to avoid division by zero. Default 1e-6.
    cif_path : str, optional
        If provided, call ``load_cif`` immediately so the structure is ready
        before any scoring calls.
    standard_residues_only : bool
        Passed to ``load_cif`` when ``cif_path`` is given. Default ``True``.

    Examples
    --------
    Load CIF once at construction, score against multiple maps:

    >>> qs = QScore(device="cuda:0", cif_path="structure.cif")
    >>> q1 = qs.score(mrc_path="map_A.mrc")
    >>> q2 = qs.score(mrc_path="map_B.mrc")

    One-liner:

    >>> qs = QScore(device="cuda:0")
    >>> q = qs.score("structure.cif", "map.mrc")
    >>> qs.summarize(q)
    """

    def __init__(
        self,
        device: str = "cpu",
        n_radii: int = 21,
        n_points: int = 8,
        ref_gaussian_width: float = 0.6,
        k_neighbors: int = 32,
        epsilon: float = 1e-6,
        cif_path: str | None = None,
        standard_residues_only: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.n_radii = n_radii
        self.n_points = n_points
        self.ref_gaussian_width = ref_gaussian_width
        self.k_neighbors = k_neighbors
        self.epsilon = epsilon
        self.radii = torch.linspace(
            0.0, (n_radii - 1) / 10.0, n_radii, device=self.device
        )

        # Cached state populated by load_cif / load_mrc.
        self.atoms: torch.Tensor | None = None
        """torch.Tensor or None: Centred atom coordinates cached by the last call to ``load_cif``."""
        self.grid: torch.Tensor | None = None
        """torch.Tensor or None: Density grid cached by the last call to ``load_mrc``."""
        self.voxel_size: float | None = None
        """float or None: Voxel size cached by the last call to ``load_mrc``."""
        self.residue_index: np.ndarray | None = None
        """np.ndarray or None: Integer array of shape ``(N,)`` mapping each atom to its residue."""
        self.residue_labels: list[str] | None = None
        """list[str] or None: Per-residue labels ``"chain:resnum:resname"``."""
        self._centroid: np.ndarray | None = None
        self._atom_keys: list[tuple[str, int, str, str]] | None = None

        print(
            "[QScore] Reminder: verify that the CIF/PDB structure is spatially "
            "aligned to the MRC map before scoring. Misaligned inputs will "
            "produce Q-scores near zero regardless of map quality."
        )

        if cif_path is not None:
            self.load_cif(cif_path, standard_residues_only=standard_residues_only)

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load_cif(
        self,
        cif_path: str,
        standard_residues_only: bool = True,
    ) -> torch.Tensor:
        """Load a CIF/PDB file, cache atom coordinates and residue metadata.

        Centering always uses the geometric centroid of **all** atoms via
        specter's PDB loader, matching the convention used when the map was
        built by specter.

        Parameters
        ----------
        cif_path : str
            Path to a ``.cif`` or ``.pdb`` structure file.
        standard_residues_only : bool
            If ``True`` (default), keep only standard amino-acid and
            nucleotide atoms, matching the original qscore selection so that
            scores are directly comparable to published Q-scores.
            If ``False``, keep all atoms including HETATM.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 3)``, Å, centred at the origin, on ``self.device``.
            Also stored as ``self.atoms``. Residue metadata is stored in
            ``self.residue_index``, ``self.residue_labels``, and
            ``self._atom_keys``.
        """
        specter_pdb = PDB(cif_path, origin=(0.0, 0.0, 0.0))
        centroid = PDB.center_of_particle(specter_pdb.coordinates).numpy()
        self._centroid = centroid

        if not standard_residues_only:
            atoms_raw = specter_pdb.coordinates.numpy()
            N = len(atoms_raw)
            self.residue_index = np.arange(N, dtype=np.int64)
            self.residue_labels = [f"ALL:{i}" for i in range(N)]
            self._atom_keys = [("A", i, "UNK", "X") for i in range(N)]
        else:
            from Bio.PDB import PDBParser
            from Bio.PDB.MMCIFParser import MMCIFParser

            parser: PDBParser | MMCIFParser = (
                PDBParser(QUIET=True)
                if cif_path.endswith(".pdb")
                else MMCIFParser(QUIET=True)
            )
            structure = parser.get_structure("s", cif_path)

            coords: list[np.ndarray] = []
            residue_index: list[int] = []
            residue_labels: list[str] = []
            atom_keys: list[tuple[str, int, str, str]] = []
            res_idx = -1

            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.resname.strip() not in _STANDARD_RESIDUES:
                            continue
                        res_started = False
                        for atom in residue:
                            if atom.name not in _STANDARD_ATOMS:
                                continue
                            if not res_started:
                                res_idx += 1
                                residue_labels.append(
                                    f"{chain.id}:{residue.id[1]}:{residue.resname}"
                                )
                                res_started = True
                            coords.append(atom.get_coord())
                            residue_index.append(res_idx)
                            atom_keys.append(
                                (chain.id, residue.id[1], residue.resname, atom.name)
                            )
                break  # first model only

            atoms_raw = np.array(coords, dtype=np.float32)
            self.residue_index = np.array(residue_index, dtype=np.int64)
            self.residue_labels = residue_labels
            self._atom_keys = atom_keys

        self.atoms = torch.as_tensor(
            atoms_raw - centroid, dtype=torch.float32, device=self.device
        )
        return self.atoms

    def load_mrc(self, mrc_path: str) -> tuple[torch.Tensor, float]:
        """Load an MRC map, cache the grid, and return it.

        Specter places the molecule centroid at voxel ``(N//2, N//2, N//2)``.
        The MRC origin header is ignored; the physical origin is forced to
        ``−(N//2) × voxel_size`` per axis so that physical coordinate 0 maps
        to voxel N//2.

        Parameters
        ----------
        mrc_path : str
            Path to the MRC file.

        Returns
        -------
        grid : torch.Tensor
            Shape ``(1, 1, nz, ny, nx)``, float32, on ``self.device``.
            Also stored as ``self.grid``.
        voxel_size : float
            Voxel size in Å. Also stored as ``self.voxel_size``.
        """
        with mrcfile.open(mrc_path, "r") as f:
            voxel_size = float(f.voxel_size.x)
            if voxel_size <= 0:
                raise RuntimeError(f"MRC {mrc_path!r}: missing voxel size in header.")
            c = int(f.header["mapc"])
            r = int(f.header["mapr"])
            s = int(f.header["maps"])
            raw = f.data.copy()

        if c == 1 and r == 2 and s == 3:
            grid_np = raw
        elif c == 3 and r == 2 and s == 1:
            grid_np = np.moveaxis(raw, [0, 1, 2], [2, 1, 0])
        elif c == 2 and r == 1 and s == 3:
            grid_np = np.moveaxis(raw, [1, 2, 0], [2, 1, 0])
        else:
            raise RuntimeError(
                f"Unsupported MRC axis arrangement mapc={c} mapr={r} maps={s}."
            )

        self.grid = (
            torch.as_tensor(grid_np, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.voxel_size = voxel_size
        return self.grid, self.voxel_size

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_neighbor_positions(self, atoms_np: np.ndarray) -> torch.Tensor:
        N = len(atoms_np)
        k = min(self.k_neighbors + 1, N)
        tree = cKDTree(atoms_np)
        _, indices = tree.query(atoms_np, k=k, workers=-1)
        neighbor_idx = indices[:, 1:]
        neighbor_pos = atoms_np[neighbor_idx]
        return torch.as_tensor(neighbor_pos, dtype=torch.float32, device=self.device)

    def _sample_shell(
        self,
        atoms: torch.Tensor,
        R: float,
        neighbor_positions: torch.Tensor,
        max_tries: int = 100,
    ) -> torch.Tensor:
        N = atoms.shape[0]
        P = self.n_points

        if R == 0.0:
            return atoms.unsqueeze(1).expand(N, P, 3).clone()

        result = torch.zeros(N, P, 3, device=self.device)
        filled = torch.zeros(N, P, dtype=torch.bool, device=self.device)

        for _ in range(max_tries):
            if filled.all():
                break
            not_done = ~filled.all(dim=-1)
            n_left = int(not_done.sum())
            atoms_l = atoms[not_done]
            nb_l = neighbor_positions[not_done]
            dirs = F.normalize(torch.randn(n_left, P, 3, device=self.device), dim=-1)
            cands = atoms_l.unsqueeze(1) + dirs * R
            dists = (cands.unsqueeze(2) - nb_l.unsqueeze(1)).norm(dim=-1)
            valid = dists.min(dim=-1).values >= R
            filled_l = filled[not_done]
            fill_now = ~filled_l & valid
            result_l = result[not_done]
            result_l[fill_now] = cands[fill_now]
            filled_l[fill_now] = True
            result[not_done] = result_l
            filled[not_done] = filled_l

        if not filled.all():
            dirs = F.normalize(torch.randn(N, P, 3, device=self.device), dim=-1)
            result[~filled] = (atoms.unsqueeze(1) + dirs * R)[~filled]

        return result

    def _interpolate_shell(
        self, grid: torch.Tensor, pts: torch.Tensor, voxel_size: float
    ) -> torch.Tensor:
        _, _, nz, ny, nx = grid.shape
        N, P, _ = pts.shape
        dx = voxel_size
        x_n = (2.0 * (pts[..., 0] / dx + nx // 2) / (nx - 1)) - 1.0
        y_n = (2.0 * (pts[..., 1] / dx + ny // 2) / (ny - 1)) - 1.0
        z_n = (2.0 * (pts[..., 2] / dx + nz // 2) / (nz - 1)) - 1.0
        coords = torch.stack([x_n, y_n, z_n], dim=-1)
        flat = coords.reshape(1, 1, 1, N * P, 3)
        out = F.grid_sample(
            grid,
            flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return out.reshape(N, P)

    def _reference_gaussian(self, grid: torch.Tensor) -> torch.Tensor:
        mean = grid.mean().item()
        std = grid.std().item()
        high = min(mean + 10.0 * std, grid.max().item())
        low = max(mean - std, grid.min().item())
        return (high - low) * torch.exp(
            -0.5 * (self.radii / self.ref_gaussian_width) ** 2
        ) + low

    # ── public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute(
        self,
        atoms: torch.Tensor | None = None,
        grid: torch.Tensor | None = None,
        voxel_size: float | None = None,
    ) -> torch.Tensor:
        """Compute per-atom Q-scores.

        All arguments are optional when ``load_cif`` and ``load_mrc`` have
        been called first — cached values are used as defaults.

        Parameters
        ----------
        atoms : torch.Tensor, optional
            Centred atom positions, shape ``(N, 3)``, Å.
        grid : torch.Tensor, optional
            Density map, shape ``(1, 1, nz, ny, nx)``.
        voxel_size : float, optional
            Å per voxel.

        Returns
        -------
        torch.Tensor
            Per-atom Q-scores, shape ``(N,)``, on ``self.device``.
        """
        atoms = atoms if atoms is not None else self.atoms
        grid = grid if grid is not None else self.grid
        voxel_size = voxel_size if voxel_size is not None else self.voxel_size

        if atoms is None or grid is None or voxel_size is None:
            raise ValueError(
                "atoms, grid, and voxel_size must be provided either as arguments "
                "or by calling load_cif() and load_mrc() first."
            )

        atoms = atoms.to(self.device)
        grid = grid.to(self.device)

        neighbor_positions = self._build_neighbor_positions(atoms.cpu().numpy())
        ref_vals = self._reference_gaussian(grid)

        map_vals_list: list[torch.Tensor] = []
        for r_idx in range(self.n_radii):
            R = self.radii[r_idx].item()
            pts = self._sample_shell(atoms, R, neighbor_positions)
            vals = self._interpolate_shell(grid, pts, voxel_size)
            map_vals_list.append(vals)

        map_vals = torch.stack(map_vals_list, dim=1)

        N = atoms.shape[0]
        M = self.n_radii * self.n_points
        u = map_vals.reshape(N, M)
        v = (
            ref_vals.unsqueeze(1)
            .expand(self.n_radii, self.n_points)
            .reshape(M)
            .unsqueeze(0)
            .expand(N, M)
        )
        u_norm = u - u.mean(dim=-1, keepdim=True)
        v_norm = v - v.mean(dim=-1, keepdim=True)
        return (u_norm * v_norm).sum(dim=-1) / torch.sqrt(
            (u_norm**2).sum(dim=-1) * (v_norm**2).sum(dim=-1) + self.epsilon
        )

    def score(
        self,
        cif_path: str | None = None,
        mrc_path: str | None = None,
        standard_residues_only: bool = True,
    ) -> torch.Tensor:
        """Load a structure and/or map, then return per-atom Q-scores.

        Arguments that are ``None`` fall back to previously cached data from
        ``load_cif`` / ``load_mrc`` (or from ``cif_path`` passed to
        ``__init__``), so a single CIF can be scored against many maps
        without reloading:

        >>> qs = QScore(cif_path="structure.cif")
        >>> q1 = qs.score(mrc_path="map_A.mrc")
        >>> q2 = qs.score(mrc_path="map_B.mrc")

        Parameters
        ----------
        cif_path : str, optional
            Path to a CIF or PDB structure file. If ``None``, reuse the
            cached atoms from the last ``load_cif`` call.
        mrc_path : str, optional
            Path to an MRC map. If ``None``, reuse the cached map from
            the last ``load_mrc`` call.
        standard_residues_only : bool
            If ``True`` (default), score only standard amino-acid and
            nucleotide atoms. Ignored when ``cif_path`` is ``None``.

        Returns
        -------
        torch.Tensor
            Per-atom Q-scores, shape ``(N,)``, on ``self.device``.
        """
        if cif_path is not None:
            self.load_cif(cif_path, standard_residues_only=standard_residues_only)
        if mrc_path is not None:
            self.load_mrc(mrc_path)
        return self.compute()

    # ── analysis ──────────────────────────────────────────────────────────────

    def per_residue(self, q: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        """Aggregate per-atom Q-scores to per-residue means.

        Requires ``load_cif`` to have been called first so that residue
        membership is known.

        Parameters
        ----------
        q : torch.Tensor
            Per-atom Q-scores, shape ``(N,)``.

        Returns
        -------
        q_res : torch.Tensor
            Per-residue mean Q-scores, shape ``(R,)``, on CPU.
        labels : list[str]
            Residue labels ``"chain:resnum:resname"``, length R.
        """
        if self.residue_index is None or self.residue_labels is None:
            raise RuntimeError("Call load_cif() before per_residue().")

        q_np = q.cpu().numpy().astype(np.float64)
        n_res = int(self.residue_index.max()) + 1
        sums = np.bincount(self.residue_index, weights=q_np, minlength=n_res)
        counts = np.bincount(self.residue_index, minlength=n_res).astype(np.float64)
        q_res = (sums / counts).astype(np.float32)
        return torch.as_tensor(q_res), self.residue_labels

    def summarize(
        self,
        q: torch.Tensor,
        label: str = "",
        plot: bool = True,
    ) -> dict[str, float]:
        """Print Q-score statistics and optionally show a histogram.

        Parameters
        ----------
        q : torch.Tensor
            Per-atom Q-scores, shape ``(N,)``.
        label : str
            Optional label shown in the printed header and plot title.
        plot : bool
            If ``True`` (default), display a histogram using matplotlib.
            Silently skipped if matplotlib is not available.

        Returns
        -------
        dict[str, float]
            Dictionary of summary statistics.
        """
        q_np = q.cpu().numpy()
        tag = f" [{label}]" if label else ""

        stats: dict[str, float] = {
            "n_atoms": float(len(q_np)),
            "mean": float(q_np.mean()),
            "median": float(np.median(q_np)),
            "std": float(q_np.std()),
            "p10": float(np.percentile(q_np, 10)),
            "p25": float(np.percentile(q_np, 25)),
            "p75": float(np.percentile(q_np, 75)),
            "p90": float(np.percentile(q_np, 90)),
            "frac_above_0.5": float((q_np > 0.5).mean()),
            "frac_above_0.7": float((q_np > 0.7).mean()),
        }

        print(f"Q-score summary{tag}")
        print(f"  atoms            : {int(stats['n_atoms'])}")
        print(f"  mean / median    : {stats['mean']:.4f} / {stats['median']:.4f}")
        print(f"  std              : {stats['std']:.4f}")
        print(
            f"  10/25/75/90 pct  : "
            f"{stats['p10']:.3f} / {stats['p25']:.3f} / "
            f"{stats['p75']:.3f} / {stats['p90']:.3f}"
        )
        print(f"  fraction > 0.5   : {stats['frac_above_0.5']:.1%}")
        print(f"  fraction > 0.7   : {stats['frac_above_0.7']:.1%}")

        if plot:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(
                    q_np,
                    bins=60,
                    range=(-1.0, 1.0),
                    color="steelblue",
                    alpha=0.75,
                    edgecolor="white",
                )
                ax.axvline(
                    stats["mean"],
                    color="red",
                    linestyle="--",
                    label=f"mean={stats['mean']:.3f}",
                )
                ax.axvline(
                    stats["median"],
                    color="orange",
                    linestyle="--",
                    label=f"median={stats['median']:.3f}",
                )
                ax.set_xlabel("Q-score")
                ax.set_ylabel("Atom count")
                ax.set_title(f"Q-score distribution{tag}")
                ax.legend()
                plt.tight_layout()
                plt.show()
            except ImportError:
                pass

        return stats

    def to_cif(
        self,
        q: torch.Tensor,
        cif_path: str,
        per_atom: bool = True,
    ) -> None:
        """Write a CIF file with Q-scores stored as B-factors (× 100).

        Coordinates are restored to the original (uncentered) frame so the
        output can be opened alongside the source map in ChimeraX or PyMOL.

        Parameters
        ----------
        q : torch.Tensor
            Per-atom Q-scores, shape ``(N,)``.
        cif_path : str
            Path for the output ``.cif`` file.
        per_atom : bool
            If ``True`` (default), each atom gets its own Q-score as
            B-factor.  If ``False``, atoms in the same residue share the
            residue-mean Q-score.
        """
        if self._atom_keys is None or self.atoms is None:
            raise RuntimeError("Call load_cif() before to_cif().")

        from Bio.PDB.StructureBuilder import StructureBuilder
        from Bio.PDB.mmcifio import MMCIFIO

        q_np = q.cpu().numpy()

        if per_atom:
            bfactors = q_np * 100.0
        else:
            q_res, _ = self.per_residue(q)
            bfactors = q_res.numpy()[self.residue_index] * 100.0

        # Restore original (uncentered) coordinates.
        coords_np = self.atoms.cpu().numpy()
        if self._centroid is not None:
            coords_np = coords_np + self._centroid

        builder = StructureBuilder()
        builder.init_structure("1")
        builder.init_seg(" ")
        builder.init_model(0)

        current_chain: str | None = None
        current_res: tuple[str, int] | None = None

        for serial, (chain_id, resnum, resname, atomname) in enumerate(
            self._atom_keys, start=1
        ):
            if chain_id != current_chain:
                builder.init_chain(chain_id)
                current_chain = chain_id
                current_res = None

            res_key = (chain_id, resnum)
            if res_key != current_res:
                builder.init_residue(resname, " ", resnum, " ")
                current_res = res_key

            builder.init_atom(
                name=atomname,
                coord=coords_np[serial - 1],
                b_factor=float(bfactors[serial - 1]),
                occupancy=1.0,
                altloc=" ",
                fullname=atomname,
                serial_number=serial,
                element=atomname[0],
            )

        structure = builder.get_structure()
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(cif_path)
        print(f"[QScore] Written: {cif_path}  (B-factor = Q-score × 100)")
