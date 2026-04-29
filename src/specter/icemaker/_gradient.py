from __future__ import annotations

from typing import Optional

import lightning as L
import torch
import torch.nn.functional as F
from torchinterp1d import interp1d

from ..arrays import (
    radial_profile_3d,
    soft_voxelize_coordinates,
    tile_volume_from_blocks,
)
from ..fft import fft3, fftconvolve
from ..progress import ProgressManager, track
from ._ap import APIcemaker
from ._helpers import ndensity_of_amorphous_ice
from ._mcmc import MCMCIcemaker


def _wrap_coords(x: torch.Tensor, L: float) -> torch.Tensor:
    """Wrap coordinates into [-L/2, L/2)."""
    return (x + L / 2) % L - L / 2


class GradientSKIcemaker(L.LightningModule):
    """
    Differentiable S(k)-matching ice coordinate generator.

    Generates oxygen molecule positions by matching the target 3D Fourier
    amplitude |F(k)| from MD simulations via L-BFGS gradient descent on
    continuously-valued atomic positions.

    Forward pass (fully differentiable):
        positions → soft_voxelize_coordinates → FFT → radial |F(k)| → MSE → backward()

    Cost per step: O(N log N) vs O(N²) per MCMC sweep.

    Parameters
    ----------
    n : int
        Number of voxels per axis (x, y).
    dx : float
        Voxel size in Å.
    nz : int, optional
        Number of voxels along z. Defaults to ``n``.
    min_distance : float
        Hard-core exclusion radius in Å used only during RSA initialisation.
    device : str or torch.device
        Computation device.
    progressbars : bool, optional
        Whether to show progress bars. Default is True.
    """

    def __init__(
        self,
        n: int = 200,
        dx: float = 0.5,
        nz: Optional[int] = None,
        min_distance: float = 2.0,
        device: str | torch.device = "cpu",
        progressbars: bool = True,
    ) -> None:
        super().__init__()
        self.min_distance = min_distance
        self.progressbars = progressbars

        _im = APIcemaker(n=n, dx=dx, nz=nz)
        self.n = _im.n
        self.nz = _im.nz
        self.dx = _im.dx
        self.box_x = self.n * self.dx
        self.box_y = self.n * self.dx
        self.box_z = self.nz * self.dx
        self._ice_kernel: torch.Tensor = _im.ice_kernel.cpu()

        self.n_molecules = int(
            ndensity_of_amorphous_ice * self.box_x * self.box_y * self.box_z
        )

        K_flat = _im.K.cpu().ravel()
        interp_vals = interp1d(
            _im.mdsim_radial_k[1:].cpu(),
            _im.mdsim_f_radial_avg[1:].cpu(),
            K_flat,
        )
        f_kernel = interp_vals.reshape(self.nz, self.n, self.n).float()
        f_kernel = f_kernel * (self.n_molecules**0.5)
        f_kernel[self.nz // 2, self.n // 2, self.n // 2] = float(self.n_molecules)
        self.register_buffer("f_target", f_kernel)

        self.dk: float = _im.dk
        self.f_target_radial: torch.Tensor = radial_profile_3d(f_kernel.cpu())

        r_bins = (_im.K.cpu() / self.dk).round().long().flatten()
        n_rbins = int(r_bins.max().item()) + 1
        bin_count = torch.bincount(r_bins, minlength=n_rbins).float().clamp(min=1)
        self.register_buffer("_r_bins", r_bins)
        self.register_buffer("_bin_count", bin_count)
        self._n_rbins: int = n_rbins
        self.register_buffer("_f_target_rad_1d", self.f_target_radial[:n_rbins].clone())

        # Repulsion kernel in FFT convention (r=0 at [0,0,0]).
        # rep_kernel[i,j,k] = 1 if the wrapped displacement (i,j,k) is within
        # min_distance voxels of the origin, 0 otherwise.  Self-term excluded.
        # Used for O(N log N) pair exclusion in _sk_loss.
        r_min_vox = self.min_distance / self.dx
        zz = torch.arange(self.nz, dtype=torch.float32)
        yy = torch.arange(self.n, dtype=torch.float32)
        xx = torch.arange(self.n, dtype=torch.float32)
        zz = torch.where(zz < self.nz // 2, zz, zz - self.nz)
        yy = torch.where(yy < self.n // 2, yy, yy - self.n)
        xx = torch.where(xx < self.n // 2, xx, xx - self.n)
        ZZ, YY, XX = torch.meshgrid(zz, yy, xx, indexing="ij")
        R_ker = torch.sqrt(ZZ**2 + YY**2 + XX**2)
        rep_kernel = (R_ker < r_min_vox).float()
        rep_kernel[0, 0, 0] = 0.0  # exclude self-contribution
        self.register_buffer("_rep_kernel_rfft", torch.fft.rfftn(rep_kernel))

        self.positions: Optional[torch.Tensor] = None

        if torch.device(device).type != "cpu":
            self.to(device)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_random(self) -> None:
        """
        Initialise from uniform random positions (no exclusion).
        """
        pos = torch.rand(self.n_molecules, 3)
        pos[:, 0] = (pos[:, 0] - 0.5) * self.box_x
        pos[:, 1] = (pos[:, 1] - 0.5) * self.box_y
        pos[:, 2] = (pos[:, 2] - 0.5) * self.box_z
        self.positions = pos

    def init_rsa(self) -> None:
        """
        Initialise via RSA with hard-core exclusion.

        Delegates to :class:`MCMCIcemaker` for O(N) cell-list RSA.
        """
        rmc = MCMCIcemaker(
            n=self.n,
            dx=self.dx,
            nz=self.nz,
            min_distance=self.min_distance,
            device="cpu",
        )
        rmc.n_molecules = self.n_molecules
        rmc.init_random()
        raw = rmc.positions.float()
        self.positions = raw - torch.tensor(
            [self.box_x / 2, self.box_y / 2, self.box_z / 2]
        )
        self.n_molecules = rmc.n_molecules

    # ------------------------------------------------------------------
    # Differentiable S(k) loss
    # ------------------------------------------------------------------

    def _sk_loss(
        self, pos: torch.Tensor, rep_strength: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Radial-profile MSE between |FFT(voxelized positions)| and the target,
        with an optional soft repulsion penalty.

        Differentiable w.r.t. ``pos`` through soft voxelization, FFT, and
        scatter-based radial average.

        Parameters
        ----------
        pos : torch.Tensor
            Positions (N, 3) in Å, centered at origin.
        rep_strength : float, optional
            Weight of the pair-exclusion penalty.  ``0.0`` disables repulsion.
            The penalty counts other particles within ``min_distance`` of each
            voxel via FFT convolution and applies a squared relu; set to a
            positive value (e.g. 1.0) to prevent particle overlap.

        Returns
        -------
        loss : torch.Tensor
            Scalar combined loss (S(k) MSE + repulsion penalty).
        f_amp : torch.Tensor
            |F(k)| in shifted frequency space, shape (nz, n, n).
        """
        vox = soft_voxelize_coordinates(
            pos,
            grid_shape=(self.nz, self.n, self.n),
            voxel_size=self.dx,
            device=self.device,
            periodic=True,
        )
        f_amp = torch.abs(fft3(vox, shift=True))
        bin_sum = torch.zeros(self._n_rbins, device=self.device)
        bin_sum.scatter_add_(0, self._r_bins, f_amp.flatten())
        sim_radial = bin_sum / self._bin_count
        loss = torch.mean((sim_radial - self._f_target_rad_1d) ** 2)

        if rep_strength > 0.0:
            # Count OTHER particles within min_distance of each voxel via FFT
            # convolution with the precomputed exclusion kernel.
            # convolved[i] ≈ 0 for well-spaced ice; > 0 when pairs overlap.
            vox_rfft = torch.fft.rfftn(vox)
            convolved = torch.fft.irfftn(
                vox_rfft * self._rep_kernel_rfft, s=(self.nz, self.n, self.n)
            )
            rep_loss = torch.mean(F.relu(convolved) ** 2)
            loss = loss + rep_strength * rep_loss

        return loss, f_amp

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize(
        self,
        n_steps: int = 50,
        lr: float = 1.0,
        optimizer: str = "lbfgs",
        record_every: int = 5,
        rep_strength: float = 1.0,
    ) -> dict:
        """
        Pure gradient descent on the S(k) loss — no Langevin noise.

        L-BFGS (default) converges in ~50 steps vs ~500 for Adam on this smooth
        radial-profile loss; the strong-Wolfe line search adapts step sizes
        automatically so the ``lr`` parameter rarely needs tuning.

        Parameters
        ----------
        n_steps : int
            Optimizer outer iterations. For L-BFGS each outer step may call
            the closure up to 10 times for line search.
        lr : float
            Step size. Keep at 1.0 for L-BFGS; use ~0.01 for Adam.
        optimizer : str
            ``'lbfgs'`` (default) or ``'adam'``.
        record_every : int
            Diagnostic recording interval.
        rep_strength : float, optional
            Weight of the soft pair-exclusion penalty in the loss.  Prevents
            particles from overlapping during gradient descent.  Set to 0 to
            disable.  Default is 1.0.

        Returns
        -------
        history : dict
            Keys: ``'step'``, ``'loss'``, ``'radial_profile'``.
        """
        assert self.positions is not None, "Call init_random() or init_rsa() first"

        pos = self.positions.to(self.device).clone().requires_grad_(True)
        history: dict[str, list] = {"step": [], "loss": [], "radial_profile": []}

        if optimizer == "lbfgs":
            opt = torch.optim.LBFGS(
                [pos],
                lr=lr,
                max_iter=10,
                history_size=20,
                line_search_fn="strong_wolfe",
            )
            last_f_amp: list[torch.Tensor] = [torch.empty(0)]

            def closure() -> torch.Tensor:
                opt.zero_grad()
                loss, f_amp = self._sk_loss(pos, rep_strength=rep_strength)
                loss.backward()
                last_f_amp[0] = f_amp.detach()
                return loss

            _manager = ProgressManager()
            _pbar, _pbar_pos = _manager.get_pbar(
                range(n_steps),
                desc="L-BFGS for ice coordinates",
                disable=not self.progressbars,
                transient=True,
            )
            try:
                for step in _pbar:
                    loss = opt.step(closure)
                    with torch.no_grad():
                        pos.data[:, 0] = _wrap_coords(pos.data[:, 0], self.box_x)
                        pos.data[:, 1] = _wrap_coords(pos.data[:, 1], self.box_y)
                        pos.data[:, 2] = _wrap_coords(pos.data[:, 2], self.box_z)
                    loss_val = (
                        loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                    )
                    _pbar.set_postfix(loss=f"{loss_val:.6f}")
                    if step % record_every == 0:
                        with torch.no_grad():
                            rad = radial_profile_3d(last_f_amp[0].cpu())
                        history["step"].append(step)
                        history["loss"].append(loss_val)
                        history["radial_profile"].append(rad)
            finally:
                _pbar.close()
                _manager.release(_pbar_pos)

        else:  # adam
            opt_adam = torch.optim.Adam([pos], lr=lr)
            _manager = ProgressManager()
            _pbar, _pbar_pos = _manager.get_pbar(
                range(n_steps),
                desc="Adam",
                disable=not self.progressbars,
                transient=True,
            )
            try:
                for step in _pbar:
                    opt_adam.zero_grad()
                    loss, f_amp = self._sk_loss(pos, rep_strength=rep_strength)
                    loss.backward()
                    opt_adam.step()
                    with torch.no_grad():
                        pos.data[:, 0] = _wrap_coords(pos.data[:, 0], self.box_x)
                        pos.data[:, 1] = _wrap_coords(pos.data[:, 1], self.box_y)
                        pos.data[:, 2] = _wrap_coords(pos.data[:, 2], self.box_z)
                    _pbar.set_postfix(loss=f"{loss.item():.6f}")
                    if step % record_every == 0:
                        with torch.no_grad():
                            rad = radial_profile_3d(f_amp.detach().cpu())
                        history["step"].append(step)
                        history["loss"].append(loss.item())
                        history["radial_profile"].append(rad)
            finally:
                _pbar.close()
                _manager.release(_pbar_pos)

        self.positions = pos.detach().cpu()
        return history

    def sample(
        self,
        n_steps: int = 200,
        lr: float = 0.05,
        noise: float = 0.02,
        record_every: int = 50,
        rep_strength: float = 1.0,
    ) -> dict:
        """
        Langevin sampling around a converged structure.

        Combines a gradient step (keeps structure near the S(k) target) with
        Gaussian noise injection (provides positional diversity). Call
        :meth:`optimize` first.

        Parameters
        ----------
        n_steps : int
            Number of Langevin steps.
        lr : float
            Adam learning rate.
        noise : float
            Gaussian noise std per step in Å.
        record_every : int
            Diagnostic recording interval.
        rep_strength : float, optional
            Weight of the soft pair-exclusion penalty.  Default is 1.0.

        Returns
        -------
        history : dict
            Keys: ``'step'``, ``'loss'``, ``'radial_profile'``.
        """
        assert self.positions is not None, "Call optimize() or init_* first"

        pos = self.positions.to(self.device).clone().requires_grad_(True)
        opt = torch.optim.Adam([pos], lr=lr)
        history: dict[str, list] = {"step": [], "loss": [], "radial_profile": []}

        _manager = ProgressManager()
        _pbar, _pbar_pos = _manager.get_pbar(
            range(n_steps),
            desc="Langevin",
            disable=not self.progressbars,
            transient=True,
        )
        try:
            for step in _pbar:
                opt.zero_grad()
                loss, f_amp = self._sk_loss(pos, rep_strength=rep_strength)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    pos.data.add_(noise * torch.randn_like(pos))
                    pos.data[:, 0] = _wrap_coords(pos.data[:, 0], self.box_x)
                    pos.data[:, 1] = _wrap_coords(pos.data[:, 1], self.box_y)
                    pos.data[:, 2] = _wrap_coords(pos.data[:, 2], self.box_z)
                _pbar.set_postfix(loss=f"{loss.item():.6f}", noise=f"{noise:.4f}")
                if step % record_every == 0:
                    with torch.no_grad():
                        rad = radial_profile_3d(f_amp.detach().cpu())
                    history["step"].append(step)
                    history["loss"].append(loss.item())
                    history["radial_profile"].append(rad)
        finally:
            _pbar.close()
            _manager.release(_pbar_pos)

        self.positions = pos.detach().cpu()
        return history

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def voxelize(self) -> torch.Tensor:
        """
        Voxelize the current positions onto the ``(nz, n, n)`` grid.

        Returns
        -------
        grid : torch.Tensor
            Soft-voxelized density, shape (nz, n, n).
        """
        assert self.positions is not None, "No positions — call init_* first"
        with torch.no_grad():
            return soft_voxelize_coordinates(
                self.positions.cpu(),
                grid_shape=(self.nz, self.n, self.n),
                voxel_size=self.dx,
                periodic=True,
            )

    def generate_ice_deltas(
        self,
        batchsize: int = 1,
        n_steps: int = 50,
        lr: float = 1.0,
        rep_strength: float = 1.0,
        init_positions=None,
    ) -> torch.Tensor:
        """
        Run ``batchsize`` independent L-BFGS optimisations and return soft-voxelized
        ice position volumes, without convolving with scattering potentials.

        Each volume uses a fresh random initialisation then L-BFGS optimisation.
        Result is stored as ``self.current_icedeltas``.

        Parameters
        ----------
        batchsize : int
            Number of independent ice volumes.
        n_steps : int
            L-BFGS outer iterations per volume.
        lr : float
            L-BFGS initial step size (line search adapts it automatically).
        rep_strength : float, optional
            Weight of the soft pair-exclusion penalty.  Default is 1.0.

        Returns
        -------
        icedeltas : torch.Tensor
            Soft-voxelized ice position volumes, shape (batchsize, nz, n, n).
        """
        results = []
        for i in track(
            range(batchsize),
            description="Generating ice volumes",
            disable=not self.progressbars,
            transient=True,
        ):
            if init_positions is None:
                self.init_random()
            else:
                self.positions = init_positions
            self.optimize(
                n_steps=n_steps,
                lr=lr,
                optimizer="lbfgs",
                record_every=n_steps,
                rep_strength=rep_strength,
            )
            results.append(self.voxelize())
        self.current_icedeltas = torch.stack(results)
        return self.current_icedeltas

    def generate_ice(
        self,
        batchsize: int = 1,
        n_steps: int = 50,
        lr: float = 1.0,
        rep_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Run ``batchsize`` independent L-BFGS optimisations and return convolved
        ice potential volumes.

        Each volume uses a fresh random initialisation, then L-BFGS, then
        convolution with the Kirkland O potential kernel.

        Parameters
        ----------
        batchsize : int
            Number of independent ice volumes.
        n_steps : int
            L-BFGS outer iterations per volume.
        lr : float
            L-BFGS initial step size (line search adapts it automatically).
        rep_strength : float, optional
            Weight of the soft pair-exclusion penalty.  Default is 1.0.

        Returns
        -------
        ice : torch.Tensor
            Convolved ice potential volumes, shape (batchsize, nz, n, n).
        """
        self.generate_ice_deltas(
            batchsize=batchsize, n_steps=n_steps, lr=lr, rep_strength=rep_strength
        )
        return fftconvolve(
            self.current_icedeltas,
            self._ice_kernel.unsqueeze(0),
            mode="same",
            axes=(-3, -2, -1),
        )

    def generate_big_ice(
        self,
        target_shape: tuple[int, int, int, int],
        num_unique: int = 8,
        n_steps: int = 50,
        lr: float = 1.0,
        optimizer: str = "lbfgs",
        rep_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate a large ice volume by tiling unique gradient-optimised blocks.

        Parameters
        ----------
        target_shape : tuple of int
            Output shape ``(B, nz, ny, nx)``.
        num_unique : int, optional
            Number of unique ice blocks to generate. Default is 8.
        n_steps : int, optional
            Optimiser outer iterations per block. Default is 50.
        lr : float, optional
            Initial learning rate. Default is 1.0.
        optimizer : str, optional
            ``'lbfgs'`` (default) or ``'adam'``.
        rep_strength : float, optional
            Weight of the soft pair-exclusion penalty. Default is 1.0.

        Returns
        -------
        big_ice : torch.Tensor
            Tiled ice volume of shape ``target_shape``.
        """
        cubes = self.generate_ice(
            batchsize=num_unique,
            n_steps=n_steps,
            lr=lr,
            rep_strength=rep_strength,
        )
        return tile_volume_from_blocks(cubes, target_shape)

    @staticmethod
    def assemble_tiles(
        tile_positions: list[torch.Tensor],
        tile_box: tuple[float, float, float],
        nx: int,
        ny: int,
        nz: int,
    ) -> torch.Tensor:
        """
        Assemble ``nx × ny × nz`` independently-optimised tiles into one volume.

        Each tile is a unique set of atom positions (independently optimised),
        so there is no periodicity artifact.

        Parameters
        ----------
        tile_positions : list of torch.Tensor
            Exactly ``nx * ny * nz`` position tensors, each shape ``(N, 3)``,
            centered at the origin in tile-box coordinates. Order is x-fastest
            (C order).
        tile_box : (Lx, Ly, Lz)
            Physical size of one tile in Å.
        nx, ny, nz : int
            Number of tiles along each axis.

        Returns
        -------
        positions : torch.Tensor
            All assembled positions, shape ``(nx*ny*nz*N, 3)``, in a box
            centered at the origin with extent ``(nx*Lx, ny*Ly, nz*Lz)`` Å.
        """
        n_tiles = nx * ny * nz
        assert (
            len(tile_positions) == n_tiles
        ), f"Expected {n_tiles} tiles, got {len(tile_positions)}"
        Lx, Ly, Lz = tile_box
        assembled: list[torch.Tensor] = []
        for idx, (ix, iy, iz) in enumerate(
            (ix, iy, iz) for iz in range(nz) for iy in range(ny) for ix in range(nx)
        ):
            offset = torch.tensor(
                [
                    (ix - (nx - 1) / 2.0) * Lx,
                    (iy - (ny - 1) / 2.0) * Ly,
                    (iz - (nz - 1) / 2.0) * Lz,
                ]
            )
            assembled.append(tile_positions[idx] + offset)
        return torch.cat(assembled, dim=0)
