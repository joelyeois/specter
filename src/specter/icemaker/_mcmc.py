from __future__ import annotations

from typing import Optional

import lightning as L
import numpy as np
import torch

from ..arrays import soft_voxelize_coordinates, tile_volume_from_blocks
from ..progress import ProgressManager, track
from ._ap import APIcemaker
from ._helpers import ndensity_of_amorphous_ice


class MCMCIcemaker(L.LightningModule):
    """
    Reverse Monte Carlo ice coordinate generator.

    Generates oxygen molecule positions by matching the O-O pair correlation
    function g(r) derived from an MD simulation frame via Metropolis MCMC.

    Two execution paths
    -------------------
    CPU  (device="cpu")
        Sequential Metropolis with a cell list; O(N) per sweep.
    GPU  (device="cuda")
        Fully vectorised parallel Metropolis; ~50-200× faster for N > 2000.
        g(r) is recomputed from scratch after each sweep to prevent drift.

    Parameters
    ----------
    n : int
        Number of voxels per axis. Box size is ``n * dx`` Å (cubic).
    dx : float
        Voxel size in Å.
    nz : int, optional
        Number of voxels along z. Defaults to ``n`` (cubic box).
    min_distance : float
        Hard-core O-O exclusion radius in Å.
    r_max : float, optional
        Maximum radius for g(r). Defaults to ``min(box/2 - 0.1, 14.0)`` Å.
    dr : float
        Histogram bin width for g(r) in Å.
    device : str or torch.device
        ``"cpu"`` for cell-list sequential path; ``"cuda"`` for GPU path.
    progressbars : bool, optional
        Whether to show progress bars. Default is True.
    """

    def __init__(
        self,
        n: int = 200,
        dx: float = 0.5,
        nz: Optional[int] = None,
        min_distance: float = 2.0,
        r_max: Optional[float] = None,
        dr: float = 0.1,
        device: str | torch.device = "cpu",
        progressbars: bool = True,
    ) -> None:
        super().__init__()
        self.n = n
        self.dx = dx
        self.nz = nz if nz is not None else n
        self.box_size = n * dx  # cubic box (MCMC assumes isotropy)
        self.progressbars = progressbars
        if r_max is None:
            r_max = min(self.box_size / 2 - 0.1, 14.0)
        assert (
            r_max <= self.box_size / 2
        ), "r_max must be <= box_size/2 for unambiguous PBC"

        self.min_distance = min_distance
        self.r_max = r_max
        self.dr = dr
        self.n_bins = int(r_max / dr)

        self.n_molecules = int(ndensity_of_amorphous_ice * self.box_size**3)
        self.register_buffer("r_centers", (torch.arange(self.n_bins) + 0.5) * dr)

        self.positions: Optional[torch.Tensor] = None
        self.gr_hist: Optional[torch.Tensor] = None
        self.gr_target: Optional[torch.Tensor] = None

        self._cell_size: float = r_max / 3.0
        self._nc: int = max(3, int(np.ceil(self.box_size / self._cell_size)))
        self._cell_list: dict[tuple[int, int, int], list[int]] = {}
        self._mol_cell: list[tuple[int, int, int]] = []

        if torch.device(device).type != "cpu":
            self.to(device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gr_norm(self, N: int) -> torch.Tensor:
        rho = N / self.box_size**3
        return N * rho * 4 * torch.pi * self.r_centers**2 * self.dr

    def _hist_from_dists(self, dists: torch.Tensor) -> torch.Tensor:
        mask = dists < self.r_max
        valid = dists[mask]
        if len(valid) == 0:
            return torch.zeros(self.n_bins)
        bins = torch.clamp((valid / self.dr).long(), 0, self.n_bins - 1)
        return torch.bincount(bins, minlength=self.n_bins).float()

    # ------------------------------------------------------------------
    # Cell list (CPU path)
    # ------------------------------------------------------------------

    def _pos_to_cell(self, pos: np.ndarray) -> tuple[int, int, int]:
        nc, cs = self._nc, self._cell_size
        return (int(pos[0] / cs) % nc, int(pos[1] / cs) % nc, int(pos[2] / cs) % nc)

    def _build_cell_list(self) -> None:
        self._cell_list = {}
        self._mol_cell = []
        pos_np = self.positions.cpu().numpy()
        for i, p in enumerate(pos_np):
            c = self._pos_to_cell(p)
            self._cell_list.setdefault(c, []).append(i)
            self._mol_cell.append(c)

    def _cell_neighbors(self, pos_np: np.ndarray, exclude: int) -> torch.Tensor:
        c = self._pos_to_cell(pos_np)
        nc = self._nc
        nbrs: list[int] = []
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                for ddz in (-1, 0, 1):
                    key = ((c[0] + ddx) % nc, (c[1] + ddy) % nc, (c[2] + ddz) % nc)
                    lst = self._cell_list.get(key)
                    if lst:
                        nbrs.extend(lst)
        return torch.tensor([i for i in nbrs if i != exclude], dtype=torch.long)

    def _cell_move(self, j: int, r_new_np: np.ndarray) -> None:
        old_c = self._mol_cell[j]
        new_c = self._pos_to_cell(r_new_np)
        if old_c != new_c:
            self._cell_list[old_c].remove(j)
            self._cell_list.setdefault(new_c, []).append(j)
            self._mol_cell[j] = new_c

    # ------------------------------------------------------------------
    # g(r) computation
    # ------------------------------------------------------------------

    def compute_gr(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute g(r) via full O(N²) all-pairs matrix (CPU).

        Parameters
        ----------
        positions : torch.Tensor
            Atom positions, shape (N, 3).

        Returns
        -------
        gr : torch.Tensor
            Normalised g(r), shape (n_bins,).
        hist : torch.Tensor
            Raw pair count histogram, shape (n_bins,).
        """
        N = len(positions)
        d = positions.unsqueeze(0) - positions.unsqueeze(1)
        d -= self.box_size * torch.round(d / self.box_size)
        dists = torch.norm(d, dim=2)
        mask = (dists > 1e-6) & (dists < self.r_max)
        valid = dists[mask]
        bins = torch.clamp((valid / self.dr).long(), 0, self.n_bins - 1)
        hist = torch.bincount(bins, minlength=self.n_bins).float()
        return hist / self._gr_norm(N), hist

    def compute_gr_chunked(
        self, positions: torch.Tensor, chunk_size: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute g(r) in row-chunks.

        Parameters
        ----------
        positions : torch.Tensor
            Atom positions, shape (N, 3).
        chunk_size : int
            Number of rows per chunk.

        Returns
        -------
        gr : torch.Tensor
            Normalised g(r).
        hist : torch.Tensor
            Raw count histogram.
        """
        N = len(positions)
        dev = positions.device
        hist = torch.zeros(self.n_bins, device=dev)
        for b in range(0, N, chunk_size):
            end = min(b + chunk_size, N)
            B = end - b
            d = positions[b:end].unsqueeze(1) - positions.unsqueeze(0)
            d -= self.box_size * torch.round(d / self.box_size)
            dists = torch.norm(d, dim=2)
            dists[torch.arange(B, device=dev), torch.arange(b, end, device=dev)] = (
                float("inf")
            )
            mask = (dists > 1e-6) & (dists < self.r_max)
            valid = dists[mask]
            if len(valid):
                bins = torch.clamp((valid / self.dr).long(), 0, self.n_bins - 1)
                hist += torch.bincount(bins, minlength=self.n_bins).float()
        return hist / self._gr_norm(N).to(dev), hist

    # ------------------------------------------------------------------
    # Target g(r) from MD dump
    # ------------------------------------------------------------------

    def set_target_gr_from_md(
        self,
        filepath: str,
        frame_idx: int = 0,
        trim_size: float = 100.0,
        oxygen_type: int = 1,
        chunk_size: int = 1000,
    ) -> None:
        """
        Load one MD frame, extract oxygen positions, compute target g(r).

        Parameters
        ----------
        filepath : str
            Path to LAMMPS dump file.
        frame_idx : int
            Which timestep frame to use.
        trim_size : float
            Side length (Å) of cube trimmed around the MD box centre.
        oxygen_type : int
            Atom type index for oxygen.
        chunk_size : int
            Row block size for batched distance computation.
        """
        print(f"Parsing MD dump frame {frame_idx} for oxygen atoms …")
        with open(filepath) as f:
            lines = f.readlines()

        frame_starts = [i for i, ln in enumerate(lines) if ln == "ITEM: TIMESTEP\n"]
        fs = frame_starts[frame_idx]
        n_atoms = int(lines[fs + 3])
        coord_start = fs + 9

        oxygens = []
        for ln in lines[coord_start : coord_start + n_atoms]:
            parts = ln.split()
            if int(parts[1]) == oxygen_type:
                oxygens.append([float(parts[2]), float(parts[3]), float(parts[4])])

        coords = torch.tensor(oxygens, dtype=torch.float32)
        print(f"  {len(coords)} oxygen atoms found")
        coords -= coords.mean(0)
        half = trim_size / 2
        mask = (
            (coords[:, 0].abs() < half)
            & (coords[:, 1].abs() < half)
            & (coords[:, 2].abs() < half)
        )
        coords = coords[mask]
        N_O = len(coords)
        print(f"  {N_O} oxygens after trimming to {trim_size} Å cube")

        hist = torch.zeros(self.n_bins)
        for i in range(0, N_O, chunk_size):
            ci = coords[i : i + chunk_size]
            d = ci.unsqueeze(1) - coords.unsqueeze(0)
            dists = torch.norm(d, dim=2)
            for k in range(len(ci)):
                dists[k, i + k] = float("inf")
            mask2 = dists < self.r_max
            valid = dists[mask2]
            if len(valid):
                bins = torch.clamp((valid / self.dr).long(), 0, self.n_bins - 1)
                hist += torch.bincount(bins, minlength=self.n_bins).float()

        rho_O = N_O / trim_size**3
        gr_norm_md = N_O * rho_O * 4 * torch.pi * self.r_centers**2 * self.dr
        self.gr_target = hist / gr_norm_md
        print("  Target g(r) computed.")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_random(self) -> None:
        """
        Initialise via Random Sequential Addition with hard-core exclusion.

        Uses a cell list for O(N) placement with ``min_distance`` exclusion.
        """
        N = self.n_molecules
        L = self.box_size
        md = self.min_distance

        cs = md
        nc = max(3, int(np.ceil(L / cs)))
        cs = L / nc

        placed = np.zeros((N, 3), dtype=np.float32)
        placed_count = 0
        cell_list: dict[tuple[int, int, int], list[int]] = {}

        def _cell(p: np.ndarray) -> tuple[int, int, int]:
            return (int(p[0] / cs) % nc, int(p[1] / cs) % nc, int(p[2] / cs) % nc)

        max_attempts = N * 200
        attempts = 0
        while placed_count < N and attempts < max_attempts:
            r = np.random.random(3).astype(np.float32) * L
            c = _cell(r)
            ok = True
            for ddx in (-1, 0, 1):
                if not ok:
                    break
                for ddy in (-1, 0, 1):
                    if not ok:
                        break
                    for ddz in (-1, 0, 1):
                        key = ((c[0] + ddx) % nc, (c[1] + ddy) % nc, (c[2] + ddz) % nc)
                        nbrs = cell_list.get(key)
                        if not nbrs:
                            continue
                        for i in nbrs:
                            d = r - placed[i]
                            d -= L * np.round(d / L)
                            if float(np.dot(d, d)) < md * md:
                                ok = False
                                break
            if ok:
                placed[placed_count] = r
                cell_list.setdefault(c, []).append(placed_count)
                placed_count += 1
            attempts += 1

        if placed_count < N:
            print(
                f"  RSA warning: placed {placed_count}/{N} molecules "
                f"(saturation after {attempts} attempts)"
            )

        self.positions = torch.from_numpy(placed[:placed_count])
        self.n_molecules = placed_count
        pos_dev = self.positions.to(self.device)
        _, gr_dev = self.compute_gr_chunked(pos_dev, chunk_size=512)
        self.gr_hist = gr_dev.cpu()
        print(f"  Initialised with {self.n_molecules} molecules (RSA)")

    def init_from_icemaker(self, im: APIcemaker, batch_idx: int = 0) -> None:
        """
        Initialise from peak voxel coordinates of a completed
        :meth:`APIcemaker.generate_ice_deltas` run.

        Parameters
        ----------
        im : APIcemaker
            A completed APIcemaker instance.
        batch_idx : int
            Which batch item to use.
        """
        vox = im.ice_coordinates[batch_idx].float()
        n_half = im.n // 2
        xyz = torch.zeros_like(vox)
        xyz[:, 0] = (vox[:, 2] - n_half) * im.dx + self.box_size / 2
        xyz[:, 1] = (vox[:, 1] - n_half) * im.dx + self.box_size / 2
        xyz[:, 2] = (vox[:, 0] - im.nz // 2) * im.dx + self.box_size / 2
        xyz = xyz % self.box_size
        self.positions = xyz
        self.n_molecules = len(self.positions)
        self._remove_pbc_violations()
        _, self.gr_hist = self.compute_gr(self.positions)
        print(f"  Initialised with {self.n_molecules} molecules (from APIcemaker)")

    def _remove_pbc_violations(self) -> int:
        d = self.positions.unsqueeze(0) - self.positions.unsqueeze(1)
        d -= self.box_size * torch.round(d / self.box_size)
        dists = torch.norm(d, dim=2)
        dists.fill_diagonal_(float("inf"))
        violated = (dists < self.min_distance).any(dim=1)
        n_removed = int(violated.sum())
        if n_removed > 0:
            self.positions = self.positions[~violated]
            self.n_molecules = len(self.positions)
            print(
                f"  Removed {n_removed} PBC-violating molecules ({self.n_molecules} remain)"
            )
        return n_removed

    # ------------------------------------------------------------------
    # GPU parallel sweep
    # ------------------------------------------------------------------

    def _parallel_gpu_sweep(
        self, T: float, step_size: float, chunk_size: int = 256, n_subgroups: int = 1
    ) -> tuple[int, int, float]:
        N = self.n_molecules
        dev = self.device
        L = self.box_size
        dr = self.dr
        n_bins = self.n_bins

        pos = self.positions.to(dev)
        norm = self._gr_norm(N).to(dev)
        gr_target = self.gr_target.to(dev)
        gr_hist = self.gr_hist.to(dev)

        n_accepted_total = 0
        n_hard_ok_total = 0

        perm = torch.randperm(N, device=dev)
        group_size = (N + n_subgroups - 1) // n_subgroups

        for g in range(n_subgroups):
            idx = perm[g * group_size : (g + 1) * group_size]
            Bg = len(idx)
            if Bg == 0:
                continue

            pos_g = pos[idx]
            delta = step_size * (2 * torch.rand(Bg, 3, device=dev) - 1)
            r_new_g = (pos_g + delta) % L

            h_new_g = torch.zeros(Bg, n_bins, device=dev)
            h_old_g = torch.zeros(Bg, n_bins, device=dev)
            hard_ok_g = torch.ones(Bg, dtype=torch.bool, device=dev)

            for b in range(0, Bg, chunk_size):
                end = min(b + chunk_size, Bg)
                B = end - b
                row = torch.arange(B, device=dev)
                glob = idx[b:end]

                d = r_new_g[b:end].unsqueeze(1) - pos.unsqueeze(0)
                d -= L * torch.round(d / L)
                dn = torch.norm(d, dim=2)
                dn[row, glob] = float("inf")
                hard_ok_g[b:end] = dn.min(dim=1).values >= self.min_distance

                mask_n = dn < self.r_max
                bins_n = dn.div(dr).long().clamp(0, n_bins - 1)
                bins_n[~mask_n] = n_bins
                h_ext = torch.zeros(B, n_bins + 1, device=dev)
                h_ext.scatter_add_(1, bins_n, mask_n.float())
                h_new_g[b:end] = h_ext[:, :n_bins]

                d_o = pos_g[b:end].unsqueeze(1) - pos.unsqueeze(0)
                d_o -= L * torch.round(d_o / L)
                do = torch.norm(d_o, dim=2)
                do[row, glob] = float("inf")
                mask_o = do < self.r_max
                bins_o = do.div(dr).long().clamp(0, n_bins - 1)
                bins_o[~mask_o] = n_bins
                h_ext_o = torch.zeros(B, n_bins + 1, device=dev)
                h_ext_o.scatter_add_(1, bins_o, mask_o.float())
                h_old_g[b:end] = h_ext_o[:, :n_bins]

            delta_hist_g = 2.0 * (h_new_g - h_old_g)
            E_cur = torch.mean((gr_hist / norm - gr_target) ** 2)
            gr_prop = (gr_hist.unsqueeze(0) + delta_hist_g) / norm
            E_prop = torch.mean((gr_prop - gr_target) ** 2, dim=1)
            delta_E = E_prop - E_cur

            u = torch.rand(Bg, device=dev)
            log_acc = (-delta_E / T).clamp(min=-60.0)
            accept = hard_ok_g & ((delta_E <= 0) | (u < torch.exp(log_acc)))

            n_accepted_total += int(accept.sum())
            n_hard_ok_total += int(hard_ok_g.sum())

            acc_idx = idx[accept]
            pos[acc_idx] = r_new_g[accept]
            if accept.any():
                gr_hist = gr_hist + delta_hist_g[accept].sum(0)

        self.positions = pos
        _, self.gr_hist = self.compute_gr_chunked(pos, chunk_size)
        self.gr_hist = self.gr_hist.to(dev)
        E_new = torch.mean((self.gr_hist / norm - gr_target) ** 2).item()
        return n_accepted_total, n_hard_ok_total, E_new

    # ------------------------------------------------------------------
    # MCMC main loop
    # ------------------------------------------------------------------

    def run(
        self,
        n_sweeps: int = 200,
        step_size: float = 0.5,
        temperature: float = 1e-3,
        record_every: int = 10,
        anneal: bool = True,
        T_start: float = 1.0,
        T_end: float = 1e-4,
        chunk_size: int = 256,
        n_subgroups: int = 8,
    ) -> dict:
        """
        Run Metropolis MCMC sweeps targeting ``self.gr_target``.

        Parameters
        ----------
        n_sweeps : int
            Number of sweeps (each sweep ≈ N move attempts).
        step_size : float
            Uniform displacement half-width in Å.
        temperature : float
            Metropolis temperature when ``anneal=False``.
        record_every : int
            Diagnostic recording interval in sweeps.
        anneal : bool
            Log-linearly anneal temperature from ``T_start`` to ``T_end``.
        T_start, T_end : float
            Annealing schedule endpoints.
        chunk_size : int
            Row-block size for chunked distance computation (GPU path).
        n_subgroups : int
            Sequential sub-groups per GPU sweep. Higher values reduce the
            parallel-approximation error at the cost of speed.

        Returns
        -------
        history : dict
            Keys: ``'sweep'``, ``'energy'``, ``'acceptance'``,
            ``'acceptance_mc'``, ``'gr'``.
        """
        assert self.positions is not None, "Call init_random() first"
        assert self.gr_target is not None, "Call set_target_gr_from_md() first"

        N = self.n_molecules
        use_gpu = self.device.type != "cpu"
        self.positions = self.positions.to(self.device)
        self.gr_hist = self.gr_hist.to(self.device)

        norm = self._gr_norm(N).to(self.device)
        gr_target_dev = self.gr_target.to(self.device)
        E_current = torch.mean((self.gr_hist / norm - gr_target_dev) ** 2).item()

        if anneal:
            temps = np.exp(
                np.linspace(np.log(T_start), np.log(T_end), n_sweeps)
            ).tolist()
        else:
            temps = [temperature] * n_sweeps

        history: dict = {
            "sweep": [],
            "energy": [],
            "acceptance": [],
            "acceptance_mc": [],
            "gr": [],
        }

        if use_gpu:
            print(
                f"  GPU: chunk_size={chunk_size}, n_subgroups={n_subgroups}, device={self.device}"
            )
        else:
            self._build_cell_list()
            pos_np = self.positions.cpu().numpy()

        _manager = ProgressManager()
        _pbar, _pbar_pos = _manager.get_pbar(
            range(n_sweeps),
            desc="MCMC sweeps",
            disable=not self.progressbars,
            transient=True,
        )
        try:
            for sweep in _pbar:
                T = temps[sweep]

                if use_gpu:
                    n_accepted, n_hard_ok, E_current = self._parallel_gpu_sweep(
                        T, step_size, chunk_size, n_subgroups
                    )
                    mc_rate = n_accepted / n_hard_ok if n_hard_ok > 0 else 0.0
                    accept_rate = n_accepted / N
                else:
                    n_accepted = 0
                    n_mc_trials = 0
                    n_mc_accepted = 0
                    j_all = torch.randint(0, N, (N,)).tolist()
                    delta_all = (step_size * (2 * torch.rand(N, 3) - 1)).numpy()
                    u_all = np.random.rand(N)

                    for i in range(N):
                        j = j_all[i]
                        r_old_np = pos_np[j].copy()
                        r_new_np = (r_old_np + delta_all[i]) % self.box_size

                        nbr_idx = self._cell_neighbors(r_new_np, j)
                        if len(nbr_idx) == 0:
                            n_mc_trials += 1
                            continue

                        nbr_pos = self.positions[nbr_idx]
                        d_new_vec = nbr_pos - torch.from_numpy(r_new_np)
                        d_new_vec -= self.box_size * torch.round(
                            d_new_vec / self.box_size
                        )
                        d_new = torch.norm(d_new_vec, dim=1)
                        if d_new.min().item() < self.min_distance:
                            continue

                        n_mc_trials += 1
                        nbr_old_idx = self._cell_neighbors(r_old_np, j)
                        nbr_old_pos = self.positions[nbr_old_idx]
                        d_old_vec = nbr_old_pos - torch.from_numpy(r_old_np)
                        d_old_vec -= self.box_size * torch.round(
                            d_old_vec / self.box_size
                        )
                        d_old = torch.norm(d_old_vec, dim=1)

                        h_add = self._hist_from_dists(d_new)
                        h_rem = self._hist_from_dists(d_old)
                        delta_hist = 2.0 * (h_add - h_rem)
                        gr_proposed = (self.gr_hist + delta_hist) / norm.cpu()
                        E_proposed = float(
                            torch.mean((gr_proposed - self.gr_target) ** 2)
                        )
                        delta_E = E_proposed - E_current

                        if delta_E < 0 or (T > 0 and u_all[i] < np.exp(-delta_E / T)):
                            self.positions[j] = torch.from_numpy(r_new_np)
                            pos_np[j] = r_new_np
                            self.gr_hist = self.gr_hist + delta_hist
                            E_current = E_proposed
                            n_accepted += 1
                            n_mc_accepted += 1
                            self._cell_move(j, r_new_np)

                    mc_rate = n_mc_accepted / n_mc_trials if n_mc_trials > 0 else 0.0
                    accept_rate = n_accepted / N

                if sweep % record_every == 0:
                    gr_now = (self.gr_hist / norm).cpu().clone()
                    history["sweep"].append(sweep)
                    history["energy"].append(E_current)
                    history["acceptance"].append(accept_rate)
                    history["acceptance_mc"].append(mc_rate)
                    history["gr"].append(gr_now)
                    _pbar.set_postfix(
                        E=f"{E_current:.5f}",
                        accept=f"{accept_rate:.2f}",
                        T=f"{T:.2e}",
                    )
        finally:
            _pbar.close()
            _manager.release(_pbar_pos)

        self.positions = self.positions.cpu()
        self.gr_hist = self.gr_hist.cpu()
        return history

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def voxelize(self) -> torch.Tensor:
        """
        Voxelize positions onto the ``(nz, n, n)`` grid.

        Returns
        -------
        grid : torch.Tensor
            Soft-voxelized density, shape (nz, n, n).
        """
        assert self.positions is not None, "No positions — call init_random() first"
        centred = self.positions.cpu() - self.box_size / 2
        return soft_voxelize_coordinates(
            centred, grid_shape=(self.nz, self.n, self.n), voxel_size=self.dx
        )

    def generate_ice(
        self,
        batchsize: int = 1,
        n_sweeps: int = 200,
        step_size: float = 0.5,
    ) -> torch.Tensor:
        """
        Run ``batchsize`` independent MCMC chains and return voxelized densities.

        Requires :meth:`set_target_gr_from_md` to have been called first.

        Parameters
        ----------
        batchsize : int
            Number of independent ice volumes to generate.
        n_sweeps : int
            MCMC sweeps per volume.
        step_size : float
            Displacement half-width in Å.

        Returns
        -------
        ice : torch.Tensor
            Voxelized oxygen densities, shape (batchsize, nz, n, n).
        """
        assert (
            self.gr_target is not None
        ), "Call set_target_gr_from_md() before generate_ice()"
        results = []
        for i in track(
            range(batchsize),
            description="Generating ice volumes",
            disable=not self.progressbars or batchsize == 1,
            transient=True,
        ):
            self.init_random()
            self.run(n_sweeps=n_sweeps, step_size=step_size)
            results.append(self.voxelize())
        return torch.stack(results)

    def generate_big_ice(
        self,
        target_shape: tuple[int, int, int, int],
        num_unique: int = 8,
        n_sweeps: int = 200,
        step_size: float = 0.5,
    ) -> torch.Tensor:
        """
        Generate a large ice volume by tiling unique MCMC blocks.

        Parameters
        ----------
        target_shape : tuple of int
            Output shape ``(B, nz, ny, nx)``.
        num_unique : int, optional
            Number of unique ice blocks to generate. Default is 8.
        n_sweeps : int, optional
            MCMC sweeps per block. Default is 200.
        step_size : float, optional
            Displacement half-width in Å. Default is 0.5.

        Returns
        -------
        big_ice : torch.Tensor
            Tiled ice volume of shape ``target_shape``.
        """
        cubes = self.generate_ice(
            batchsize=num_unique, n_sweeps=n_sweeps, step_size=step_size
        )
        return tile_volume_from_blocks(cubes, target_shape)
