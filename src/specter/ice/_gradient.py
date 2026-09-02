from __future__ import annotations

from typing import Any, Callable, Optional

import lightning as L
import torch
import torch.nn.functional as F

from ..arrays import radial_profile_3d, soft_voxelize_coordinates
from ..fft import fft3, fftconvolve
from ..progress import ProgressManager, track
from ._energy import MLBOP, NeighborListCache
from ._helpers import ndensity_of_amorphous_ice
from ._kernels import build_water_kernel
from ._kernels import (
    compute_native_target,
    ice_kspace_radial_grid,
    interpolate_target_kernel,
    load_mdsim_f_radial_avg,
)


class _PlateauStop:
    """
    :meth:`GradientSKIcemaker.optimize`'s early-stopping rule.

    Stop once the relative change in loss, ``abs(prev - cur) / abs(prev)``,
    has stayed below `tol` for `patience` consecutive steps. Halfway through
    such a streak `on_stall` is called once (L-BFGS clears its history
    there), so a stalled line search restarts rather than being read as
    convergence; a run that has really converged plateaus through the
    restart and still stops at `patience`.

    Parameters
    ----------
    tol : float or None
        Relative tolerance; None disables the rule.
    patience : int
        Consecutive sub-tolerance steps that stop the run.
    on_stall : callable, optional
        Called at ``patience // 2`` consecutive sub-tolerance steps.
    """

    def __init__(
        self, tol: float | None, patience: int, on_stall: Callable[[], Any] | None
    ) -> None:
        self.tol = tol
        self.patience = patience
        self.on_stall = on_stall
        self.prev: float | None = None
        self.count = 0

    def update(self, loss_val: float) -> bool:
        """Feed one step's loss; True once the streak has reached `patience`."""
        if self.tol is None:
            return False
        if self.prev is not None:
            rel_change = abs(self.prev - loss_val) / (abs(self.prev) + 1e-12)
            self.count = self.count + 1 if rel_change < self.tol else 0
            if self.count == max(1, self.patience // 2) and self.on_stall is not None:
                self.on_stall()
        self.prev = loss_val
        return self.count >= self.patience


def _wrap_coords(x: torch.Tensor, L: float) -> torch.Tensor:
    """Wrap coordinates into [-L/2, L/2)."""
    return (x + L / 2) % L - L / 2


class GradientSKIcemaker(L.LightningModule):
    """
    Differentiable S(k)-matching ice coordinate generator.

    Generates oxygen molecule positions by matching the target 3D Fourier
    amplitude ``|F(k)|`` from MD simulations via L-BFGS gradient descent on
    continuously-valued atomic positions.

    Forward pass (fully differentiable)::

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
        Hard-core exclusion radius in Å for the artificial soft
        pair-exclusion penalty (see ``rep_strength`` on :meth:`optimize`).
        Only takes effect if ``rep_strength`` is set above its default of
        0.0 -- the default optimisation path uses ``mlbop_strength`` instead,
        which does not depend on this parameter.
    device : str or torch.device
        Computation device.
    parameterization : str, optional
        Atomic scattering-factor parameterization for the ice kernel:
        ``'kirkland'``, ``'lobato'``, or ``'shtyrov'``. Default
        ``'kirkland'``: Shtyrov fits bonded species of BIOMOLECULES over
        0.011-0.62 1/A, so bulk ice is out of its domain and its k=0 limit
        (which is what a mean inner potential is) extrapolates below the
        fitted range. Kirkland, Lobato and Peng are per-element and valid at
        k=0, and agree with each other there; see `build_water_kernel`.
    progressbars : bool, optional
        Whether to show progress bars. Default is True.
    mdsim_target_path : str, optional
        Path to a precomputed radial-average ``|F(k)|`` target ``.pt`` file, in
        the fixed 400x400x400, dx=0.25 Å format :func:`~specter.ice._kernels.
        load_mdsim_f_radial_avg` expects. If None (default), the target is
        instead computed natively at this instance's own ``(n, dx, nz)`` via
        :func:`~specter.ice._kernels.compute_native_target`, from the bundled
        LDA-80K (low-density amorphous ice, 80K/0atm) reference frame -- an
        in-house MD simulation, not from a published dataset.
        Matching ``dx`` between target and training grid matters far more
        than matching absolute box size -- interpolating a fine-``dx``
        target across a coarser training grid is a lossy comparison (the
        simulated side is aliased by coarse voxelization, the target isn't),
        which was validated this way: coarse-``dx`` training against the old
        fixed fine-grid default got stuck (S(k) loss O(1)-O(1e3)); against a
        natively-computed target at the same ``dx`` it converges cleanly
        (O(1e-5)-O(1e-7)), across dx=0.5/1.0/2.0 and box sizes from 16 Å up
        to 256 Å (beyond the ~127 Å real MD cell, via safe box-size
        extrapolation -- see :func:`compute_native_target`). Pass an
        explicit path to opt back into the old fixed-target behavior.
    """

    def __init__(
        self,
        n: int = 200,
        dx: float = 0.5,
        nz: Optional[int] = None,
        min_distance: float = 2.0,
        device: str | torch.device = "cpu",
        parameterization: str = "kirkland",
        progressbars: bool = True,
        mdsim_target_path: str | None = None,
    ) -> None:
        super().__init__()
        self.min_distance = min_distance
        self.progressbars = progressbars
        self.parameterization = parameterization

        self.n = n
        self.nz = nz if nz is not None else n
        self.dx = dx
        self.box_x = self.n * self.dx
        self.box_y = self.n * self.dx
        self.box_z = self.nz * self.dx
        self._ice_kernel: torch.Tensor = build_water_kernel(
            self.dx, self.parameterization
        )

        self.n_molecules = int(
            ndensity_of_amorphous_ice * self.box_x * self.box_y * self.box_z
        )

        if mdsim_target_path is None:
            mdsim_radial_k, mdsim_f_radial_avg = compute_native_target(
                n=self.n, dx=self.dx, nz=self.nz
            )
        else:
            mdsim_radial_k, mdsim_f_radial_avg = load_mdsim_f_radial_avg(
                mdsim_target_path
            )
        K = ice_kspace_radial_grid(self.n, self.nz, self.dx)
        f_kernel = interpolate_target_kernel(
            K, mdsim_radial_k, mdsim_f_radial_avg, self.n_molecules
        ).float()
        self.register_buffer("f_target", f_kernel, persistent=False)

        self.dk: float = 1 / self.n / self.dx

        # Physical |k|-magnitude bins -- NOT radial_profile_3d's voxel-index
        # bins, which only coincide
        # with these when nz == n. For anisotropic grids (nz != n), a voxel
        # step along z spans a different physical k-spacing (1/(nz*dx)) than
        # a voxel step along x/y (1/(n*dx)=dk), so voxel-index distance and
        # physical |k| distance diverge along the shorter axis -- binning
        # the target by voxel-index distance while the simulated |F(k)|
        # below is binned by physical distance silently produces mismatched
        # bin counts (and a physically meaningless comparison even where
        # shapes happen to match). Bin the target with the same _r_bins/
        # _bin_count used for the simulated side so both stay on identical,
        # physically-correct bins regardless of aspect ratio.
        r_bins = (K / self.dk).round().long().flatten()
        n_rbins = int(r_bins.max().item()) + 1
        bin_count = torch.bincount(r_bins, minlength=n_rbins).float().clamp(min=1)
        self.register_buffer("_r_bins", r_bins, persistent=False)
        self.register_buffer("_bin_count", bin_count, persistent=False)
        self._n_rbins: int = n_rbins
        target_bin_sum = torch.bincount(
            r_bins, weights=f_kernel.flatten(), minlength=n_rbins
        )
        self.register_buffer(
            "_f_target_rad_1d", target_bin_sum / bin_count, persistent=False
        )

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
        self.register_buffer(
            "_rep_kernel_rfft", torch.fft.rfftn(rep_kernel), persistent=False
        )

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

    # ------------------------------------------------------------------
    # Differentiable S(k) loss
    # ------------------------------------------------------------------

    def _sk_loss(
        self,
        pos: torch.Tensor,
        rep_strength: float = 0.0,
        mlbop_strength: float = 0.0,
        mlbop_target: float | None = None,
        neighbor_cache: NeighborListCache | None = None,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Radial-profile MSE between |FFT(voxelized positions)| and the target,
        with an optional soft repulsion penalty and/or ML-BOP energy penalty.

        Differentiable w.r.t. ``pos`` through soft voxelization, FFT, and
        scatter-based radial average.

        Parameters
        ----------
        pos : torch.Tensor
            Positions (N, 3) in Å, centered at origin.
        rep_strength : float, optional
            Weight of the artificial pair-exclusion penalty.  ``0.0``
            disables it (default).  The penalty counts other particles
            within ``min_distance`` of each voxel via FFT convolution and
            applies a squared relu; set to a positive value (e.g. 1.0) to
            prevent particle overlap. This is a cheap, purely geometric
            stand-in for a real interatomic potential.
        mlbop_strength : float, optional
            Weight of a physically-motivated alternative to
            ``rep_strength``: the ML-BOP per-atom energy (see
            :meth:`~specter.ice._energy.MLBOP.compute_energy`),
            which penalizes both overlap (two-body term) and unrealistic
            O-O-O angles (three-body bond-order term) rather than just
            overlap. ``0.0`` disables it (default). Can be combined with
            ``rep_strength`` or used on its own.
        mlbop_target : float or None, optional
            If set, the ML-BOP penalty becomes
            ``(E_per_atom - mlbop_target) ** 2`` instead of minimizing
            ``E_per_atom`` directly. Amorphous ice phases are metastable,
            not energy-minimal -- LDA ice in particular is a *compressed*
            amorphous phase, so its ML-BOP energy sits measurably above the
            true minimum (real LDA-80K MD frames score around -0.41 eV/atom,
            not the more negative values a denser/more crystalline packing
            would reach). Minimizing ``E_per_atom`` unboundedly pulls the
            optimizer toward that lower-energy, more ordered packing instead
            of matching the target amorphous phase. Ignored (falls back to
            direct minimization) if ``None`` (default).
        neighbor_cache : NeighborListCache, optional
            Verlet-skin neighbour list shared across calls at nearby
            positions, for the ML-BOP term; see
            :class:`~specter.ice._energy.NeighborListCache`. Default None
            (search afresh).
        deterministic : bool, optional
            Whether the voxel splat accumulates in a fixed order; see
            :func:`~specter.arrays.soft_voxelize_coordinates`. The radial
            bin ``scatter_add_`` below is atomic either way, so this loss is
            never bit-reproducible on CUDA; ``False`` skips a sort the
            optimiser gains nothing from. Default True.

        Returns
        -------
        loss : torch.Tensor
            Scalar combined loss (S(k) MSE + any enabled penalty terms).
        f_amp : torch.Tensor
            ``|F(k)|`` in shifted frequency space, shape (nz, n, n).
        """
        vox = soft_voxelize_coordinates(
            pos,
            grid_shape=(self.nz, self.n, self.n),
            voxel_size=self.dx,
            device=self.device,
            periodic=True,
            deterministic=deterministic,
        )
        f_amp = torch.abs(fft3(vox, shift=True)).clamp(min=1e-8)
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

        if mlbop_strength > 0.0:
            model = MLBOP(device=pos.device)
            box = (self.box_x, self.box_y, self.box_z)
            mlbop_result = model.compute_energy(
                pos,
                box_size=box,
                pbc=True,
                neighbor_cache=neighbor_cache,
                compute_dtype=torch.float32,
            )
            e_per_atom = mlbop_result["E_per_atom"]
            if mlbop_target is None:
                mlbop_penalty = e_per_atom
            else:
                mlbop_penalty = (e_per_atom - mlbop_target) ** 2
            loss = loss + mlbop_strength * mlbop_penalty

        return loss, f_amp

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def _wrap_positions_(self, pos: torch.Tensor) -> None:
        """Wrap `pos` (N, 3) into the periodic cell, in place, without grad."""
        with torch.no_grad():
            pos.data[:, 0] = _wrap_coords(pos.data[:, 0], self.box_x)
            pos.data[:, 1] = _wrap_coords(pos.data[:, 1], self.box_y)
            pos.data[:, 2] = _wrap_coords(pos.data[:, 2], self.box_z)

    def _prerelax(
        self, pos: torch.Tensor, steps: int, neighbor_cache: NeighborListCache
    ) -> None:
        """
        Relax `pos` in place against the ML-BOP energy alone.

        The starting point for :meth:`optimize`; see its ``prerelax_steps``.
        Minimises ``E_per_atom`` directly rather than toward
        ``mlbop_target``: this is overlap removal, and thirty steps from a
        random start stop far short of the target anyway.

        Parameters
        ----------
        pos : torch.Tensor
            Positions (N, 3) with ``requires_grad=True``, updated in place.
        steps : int
            L-BFGS outer steps.
        neighbor_cache : NeighborListCache
            Shared with the main optimisation that follows.
        """
        model = MLBOP(device=pos.device)
        box = (self.box_x, self.box_y, self.box_z)
        opt = torch.optim.LBFGS(
            [pos],
            lr=1.0,
            max_iter=10,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            opt.zero_grad()
            energy = model.compute_energy(
                pos,
                box_size=box,
                pbc=True,
                neighbor_cache=neighbor_cache,
                compute_dtype=torch.float32,
            )["E_per_atom"]
            energy.backward()
            return energy

        _manager = ProgressManager()
        _pbar, _pbar_pos = _manager.get_pbar(
            range(steps),
            desc="ML-BOP pre-relaxation",
            disable=not self.progressbars,
            transient=True,
        )
        try:
            for _ in _pbar:
                energy = opt.step(closure)
                self._wrap_positions_(pos)
                e_val = energy.item() if isinstance(energy, torch.Tensor) else energy
                _pbar.set_postfix(E_per_atom=f"{float(e_val):.4f}")
        finally:
            _pbar.close()
            _manager.release(_pbar_pos)

    def _run_steps(
        self,
        pos: torch.Tensor,
        n_steps: int,
        take_step: Callable[[], float],
        desc: str,
        record_every: int,
        history: dict[str, Any],
        raw_sk_loss: Callable[[], float],
        last_f_amp: list[torch.Tensor],
        stop: "_PlateauStop",
    ) -> bool:
        """
        The outer loop shared by :meth:`optimize`'s optimisers.

        Parameters
        ----------
        pos : torch.Tensor
            The positions being optimised; wrapped into the cell after
            every step.
        n_steps : int
            Step ceiling.
        take_step : callable
            Runs one optimiser step and returns the loss it started from.
        desc : str
            Progress-bar label.
        record_every : int
            Record `history` every this many steps (and at an early stop).
        history : dict
            Lists appended to: ``step``, ``loss``, ``sk_loss``,
            ``radial_profile``.
        raw_sk_loss : callable
            The penalty-free S(k) loss at the current positions.
        last_f_amp : list of torch.Tensor
            One-element box `take_step` leaves the latest ``|F(k)|`` in.
        stop : _PlateauStop
            The early-stopping rule.

        Returns
        -------
        bool
            Whether the rule stopped the run before `n_steps`.
        """

        def record(step: int, loss_val: float) -> None:
            with torch.no_grad():
                rad = radial_profile_3d(last_f_amp[0].cpu())
            history["step"].append(step)
            history["loss"].append(loss_val)
            history["sk_loss"].append(raw_sk_loss())
            history["radial_profile"].append(rad)

        manager = ProgressManager()
        pbar, pbar_pos = manager.get_pbar(
            range(n_steps), desc=desc, disable=not self.progressbars, transient=True
        )
        try:
            for step in pbar:
                loss_val = take_step()
                self._wrap_positions_(pos)
                plateaued = stop.update(loss_val)
                pbar.set_postfix(loss=f"{loss_val:.6f}")
                if step % record_every == 0:
                    record(step, loss_val)
                if plateaued:
                    if history["step"] and history["step"][-1] != step:
                        record(step, loss_val)
                    return True
        finally:
            pbar.close()
            manager.release(pbar_pos)
        return False

    def optimize(
        self,
        n_steps: int = 250,
        lr: float = 1.0,
        optimizer: str = "lbfgs",
        record_every: int = 5,
        rep_strength: float = 0.0,
        mlbop_strength: float = 0.5,
        mlbop_target: float | None = -0.413,
        tol: Optional[float] = 1e-3,
        patience: int = 10,
        history_size: int = 100,
        max_iter: int = 25,
        prerelax_steps: int = 30,
        param_dtype: torch.dtype = torch.float64,
    ) -> dict:
        """
        Pure gradient descent on the S(k) loss — no Langevin noise.

        L-BFGS (default) converges in far fewer steps than Adam on this
        smooth radial-profile loss; the strong-Wolfe line search adapts step
        sizes automatically so the ``lr`` parameter rarely needs tuning.

        The L-BFGS settings were chosen on loss against *closure count* (one
        closure is one loss + gradient evaluation, the unit of cost) at the
        bundled library's geometry (n=256, dx=1.0, ~527k atoms), on a
        6000-closure budget from seed 0:

        ================================================  =====  =====  =====  =======
        settings                                          @1000  @1500  @6000  E/atom
        ================================================  =====  =====  =====  =======
        history 20, max_iter 10, no pre-relaxation        0.266  0.139  0.053  -0.087
        history 20, max_iter 50                           0.269  0.132  0.046  -0.111
        history 100, max_iter 25, 30-step pre-relaxation  0.047  0.040  0.022  -0.203
        history 100, max_iter 50, 30-step pre-relaxation  0.048  0.048  0.022  -0.205
        ================================================  =====  =====  =====  =======

        The first row (the settings until 2026-09-02) needed 4-6k closures to
        reach a loss of 0.04-0.06; the defaults reach it in 1000-1500 and go
        on to a loss, and an ML-BOP energy against the -0.413 target, the old
        settings never got to. Each row is the same objective, recipe and
        target: the changes are to the search, not to what is searched for.
        The table was measured with float32 positions; ``param_dtype``
        (float64 by default, see below) removes the stall that made the
        last row's tail so slow, reaching its 6000-closure loss in ~1600.

        Parameters
        ----------
        n_steps : int
            Optimizer outer iterations. For L-BFGS each outer step may call
            the closure up to ``max_iter`` times, plus line-search
            evaluations. Default is 250, about 6000 closures at
            ``max_iter=25``, the budget in the table above; the loss was
            0.024 at 3000 closures and 0.022 at 6000, so a smaller value
            gives most of the result. ``tol``/``patience`` still apply on
            top of this ceiling.
        lr : float
            Step size. Keep at 1.0 for L-BFGS; use ~0.01 for Adam.
        optimizer : str
            ``'lbfgs'`` (default) or ``'adam'``.
        record_every : int
            Diagnostic recording interval.
        rep_strength : float, optional
            Weight of the artificial pair-exclusion penalty in the loss (see
            :meth:`_sk_loss`).  A cheap, purely geometric stand-in for a real
            interatomic potential -- it prevents particle overlap but is a
            weak constraint on local structure (S(k) matching alone can hide
            badly-overlapping atoms behind a good Fourier-amplitude match).
            Set to a positive value (e.g. 1.0) to opt back into this
            geometric-only behavior.  Default is 0.0 (disabled in favor of
            ``mlbop_strength``).
        mlbop_strength : float, optional
            Weight of the ML-BOP-energy-based penalty (see :meth:`_sk_loss`),
            the default alternative to ``rep_strength``. Runs fully
            differentiably inside the optimisation loop (see
            :meth:`~specter.ice._energy.MLBOP.compute_energy`),
            penalizing both overlap (two-body term) and unrealistic O-O-O
            angles (three-body term) rather than just overlap -- validated
            across dx=0.5/1.0/2.0 and box sizes up to 256 Å as the best
            balance of energy-match quality vs. S(k) fidelity. Default is
            0.5; set to 0 to disable and fall back to ``rep_strength``.
        mlbop_target : float or None, optional
            If set, matches ``E_per_atom`` to this value instead of
            minimizing it unboundedly (see :meth:`_sk_loss`) -- use the
            ML-BOP energy of a real MD reference frame (e.g. via
            :meth:`~specter.ice._mdsim.MDSimDump.mlbop_energy`) so the
            result matches the target *phase* of ice rather than drifting
            toward an arbitrarily low-energy (more crystalline) packing.
            Only meaningful when ``mlbop_strength > 0``. Default is -0.413,
            matching real LDA-80K MD ice (stable to +/-0.0001 across widely
            separated MD frames); set to ``None`` to minimize ``E_per_atom``
            unboundedly instead.
        tol : float or None, optional
            Relative loss-improvement tolerance for early stopping: once the
            fractional change in loss, ``abs(prev - cur) / abs(prev)``, stays
            below ``tol`` for ``patience`` consecutive steps, optimisation
            stops before reaching ``n_steps``. Halfway through such a streak
            the L-BFGS history is cleared (``'lbfgs'`` only), so a stalled
            line search restarts from steepest descent rather than being
            read as convergence; a run that has really converged plateaus
            through the restart and still stops at ``patience``. This tracks the combined
            differentiable loss (S(k) MSE + whichever penalty is enabled)
            already computed every step. Set to ``None`` to disable and
            always run the full ``n_steps``. Default is 1e-3, a deliberate
            trade of the tail of the curve for time at the bundled geometry:
            once S(k) is matched the loss is the energy term alone, and
            under 1e-4 a config ran 96-250 steps (4-9 min) to reach E/atom
            -0.20 to -0.28, most of which was the last few hundredths of an
            eV. The old 1e-4 was tuned on the S(k) loss curve at n=400,
            dx=0.25, where 1e-6 never sustained for ``patience`` steps
            because of ~1e-5-scale wiggles.
        patience : int, optional
            Consecutive below-``tol`` steps required to trigger early
            stopping. Default is 10.
        history_size : int, optional
            L-BFGS curvature pairs kept (``'lbfgs'`` only). Default 100: a
            1.6 M-dimensional problem gains from more than the 20 the
            optimiser used to keep, and 100 vectors are 600 MB at the
            bundled geometry, small next to the loss's own working set.
        max_iter : int, optional
            L-BFGS inner iterations per outer step (``'lbfgs'`` only).
            Default 25; 50 measured no better.
        prerelax_steps : int, optional
            L-BFGS steps on the ML-BOP energy alone, from the initial
            positions, before the full loss is optimised (``'lbfgs'`` only,
            and only when ``mlbop_strength > 0``). A uniform-random start
            has overlapping molecules and a loss of ~5e4, and the full loss
            spends its first several hundred closures on nothing else; the
            energy term alone resolves them without the FFT, in ~300 cheap
            closures. The objective is unchanged, only the starting point.
            Default 30; 0 disables. A jittered lattice at the right density
            is not an alternative: the optimiser never leaves its
            crystalline S(k) (loss stuck at 3e4 at n=128).
        param_dtype : torch.dtype, optional
            Dtype of the positions being optimised. Default float64, and
            this is load-bearing at the bundled geometry: once S(k) is
            matched (a few hundred evaluations in, MSE ~1e-4) the whole
            remaining loss is the energy term, and the descent step it
            calls for moves each atom by ~1e-6 A, below the float32 ulp of
            a coordinate at 64-128 A (7.6e-6 to 1.5e-5 A). In float32 the
            line search then returns zero-length steps and the loss freezes
            at ~0.050 with E/atom near -0.10 (which is where every
            float32-optimised library ended up); in float64 it continues
            at 1-4% per step to 0.024 / -0.19 by step 60. The FFT, splat
            and three-body kernels still run in float32: only the voxel
            fractions and pair vectors are formed in float64 first, so the
            cost is elementwise work on N x 3 values and the L-BFGS
            history (100 x 1.6 M float64, 1.3 GB at n=256). The result is
            stored back in float32.

        Returns
        -------
        history : dict
            Keys: ``'step'``, ``'loss'`` (combined loss actually optimized),
            ``'sk_loss'`` (raw S(k) MSE alone, with no penalty term --
            comparable across different ``rep_strength``/``mlbop_strength``
            settings, unlike ``'loss'``), ``'radial_profile'``, and
            ``'stopped_early'`` (bool — whether ``tol``/``patience`` cut the
            run short of ``n_steps``).
        """
        assert self.positions is not None, "Call init_random() first"

        pos = (
            self.positions.to(self.device, dtype=param_dtype)
            .clone()
            .requires_grad_(True)
        )
        history: dict[str, Any] = {
            "step": [],
            "loss": [],
            "sk_loss": [],
            "radial_profile": [],
        }

        def _raw_sk_loss() -> float:
            with torch.no_grad():
                sk_loss, _ = self._sk_loss(pos, rep_strength=0.0, mlbop_strength=0.0)
                return sk_loss.item()

        neighbor_cache = NeighborListCache()
        last_f_amp: list[torch.Tensor] = [torch.empty(0)]

        def loss_and_backward() -> torch.Tensor:
            loss, f_amp = self._sk_loss(
                pos,
                rep_strength=rep_strength,
                mlbop_strength=mlbop_strength,
                mlbop_target=mlbop_target,
                neighbor_cache=neighbor_cache,
                deterministic=False,
            )
            loss.backward()
            last_f_amp[0] = f_amp.detach()
            return loss

        on_stall: Callable[[], Any] | None
        if optimizer == "lbfgs":
            if prerelax_steps > 0 and mlbop_strength > 0.0:
                self._prerelax(pos, prerelax_steps, neighbor_cache)
            lbfgs = torch.optim.LBFGS(
                [pos],
                lr=lr,
                max_iter=max_iter,
                history_size=history_size,
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                return loss_and_backward()

            def take_step() -> float:
                loss = lbfgs.step(closure)
                return loss.item() if isinstance(loss, torch.Tensor) else float(loss)

            desc = "L-BFGS for ice coordinates"
            # A stalled line search, not convergence, is the usual reason
            # for a run of unchanged losses: L-BFGS keeps proposing the same
            # direction, the strong-Wolfe search keeps returning t = 0.
            # Dropping the curvature history restarts from steepest descent,
            # which resolves a stall in a few steps; a run that really has
            # converged plateaus through the restart and still stops at
            # ``patience``.
            on_stall = lbfgs.state.clear
        else:  # adam
            adam = torch.optim.Adam([pos], lr=lr)

            def take_step() -> float:
                adam.zero_grad()
                loss = loss_and_backward()
                adam.step()
                return loss.item()

            desc = "Adam"
            on_stall = None

        stopped_early = self._run_steps(
            pos,
            n_steps,
            take_step,
            desc,
            record_every,
            history,
            _raw_sk_loss,
            last_f_amp,
            _PlateauStop(tol, patience, on_stall),
        )

        history["stopped_early"] = stopped_early
        self.positions = pos.detach().float().cpu()
        return history

    def sample(
        self,
        n_steps: int = 200,
        lr: float = 0.05,
        noise: float = 0.02,
        record_every: int = 50,
        rep_strength: float = 0.0,
        mlbop_strength: float = 0.5,
        mlbop_target: float | None = -0.413,
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
            Weight of the artificial pair-exclusion penalty.  Default is 0.0
            (disabled in favor of ``mlbop_strength``).
        mlbop_strength : float, optional
            Weight of the ML-BOP-energy-based penalty, the default
            alternative to ``rep_strength`` (see :meth:`_sk_loss` and
            :meth:`optimize`).  Default is 0.5.
        mlbop_target : float or None, optional
            Matches ``E_per_atom`` to this value instead of minimizing it
            unboundedly (see :meth:`_sk_loss`).  Default is -0.413, matching
            real LDA-80K MD ice.

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
                loss, f_amp = self._sk_loss(
                    pos,
                    rep_strength=rep_strength,
                    mlbop_strength=mlbop_strength,
                    mlbop_target=mlbop_target,
                )
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
        n_steps: int = 400,
        lr: float = 1.0,
        rep_strength: float = 0.0,
        mlbop_strength: float = 0.5,
        mlbop_target: float | None = -0.413,
        init_positions: torch.Tensor | None = None,
        tol: Optional[float] = 1e-3,
        patience: int = 10,
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
            L-BFGS outer iterations per volume (upper bound — see ``tol``).
            Default is 400 (see :meth:`optimize`).
        lr : float
            L-BFGS initial step size (line search adapts it automatically).
        rep_strength : float, optional
            Weight of the artificial pair-exclusion penalty.  Default is 0.0
            (disabled in favor of ``mlbop_strength``; see :meth:`optimize`).
        mlbop_strength : float, optional
            Weight of the ML-BOP-energy-based penalty, the default
            alternative to ``rep_strength`` (see :meth:`_sk_loss` and
            :meth:`optimize`).  Default is 0.5.
        mlbop_target : float or None, optional
            Matches ``E_per_atom`` to this value instead of minimizing it
            unboundedly (see :meth:`_sk_loss`).  Default is -0.413, matching
            real LDA-80K MD ice.
        tol : float or None, optional
            Early-stopping tolerance forwarded to :meth:`optimize`. Default
            is 1e-3; set to ``None`` to always run the full ``n_steps``.
        patience : int, optional
            Early-stopping patience forwarded to :meth:`optimize`. Default
            is 10.

        Returns
        -------
        icedeltas : torch.Tensor
            Soft-voxelized ice position volumes, shape (batchsize, nz, n, n).
        """
        results = []
        for i in track(
            range(batchsize),
            description="Generating ice volumes",
            disable=not self.progressbars or batchsize == 1,
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
                mlbop_strength=mlbop_strength,
                mlbop_target=mlbop_target,
                tol=tol,
                patience=patience,
            )
            results.append(self.voxelize())
        self.current_icedeltas = torch.stack(results)
        return self.current_icedeltas

    def generate_ice(
        self,
        batchsize: int = 1,
        n_steps: int = 400,
        lr: float = 1.0,
        rep_strength: float = 0.0,
        mlbop_strength: float = 0.5,
        mlbop_target: float | None = -0.413,
        tol: Optional[float] = 1e-3,
        patience: int = 10,
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
            L-BFGS outer iterations per volume (upper bound — see ``tol``).
            Default is 400 (see :meth:`optimize`).
        lr : float
            L-BFGS initial step size (line search adapts it automatically).
        rep_strength : float, optional
            Weight of the artificial pair-exclusion penalty.  Default is 0.0
            (disabled in favor of ``mlbop_strength``; see :meth:`optimize`).
        mlbop_strength : float, optional
            Weight of the ML-BOP-energy-based penalty, the default
            alternative to ``rep_strength`` (see :meth:`_sk_loss` and
            :meth:`optimize`).  Default is 0.5.
        mlbop_target : float or None, optional
            Matches ``E_per_atom`` to this value instead of minimizing it
            unboundedly (see :meth:`_sk_loss`).  Default is -0.413, matching
            real LDA-80K MD ice.
        tol : float or None, optional
            Early-stopping tolerance forwarded to :meth:`optimize`. Default
            is 1e-3; set to ``None`` to always run the full ``n_steps``.
        patience : int, optional
            Early-stopping patience forwarded to :meth:`optimize`. Default
            is 10.

        Returns
        -------
        ice : torch.Tensor
            Convolved ice potential volumes, shape (batchsize, nz, n, n).
        """
        self.generate_ice_deltas(
            batchsize=batchsize,
            n_steps=n_steps,
            lr=lr,
            rep_strength=rep_strength,
            mlbop_strength=mlbop_strength,
            mlbop_target=mlbop_target,
            tol=tol,
            patience=patience,
        )
        return fftconvolve(
            self.current_icedeltas,
            self._ice_kernel.unsqueeze(0),
            mode="same",
            axes=(-3, -2, -1),
        )

    def mlbop_energy(self, pbc: bool = True) -> dict[str, float]:
        """
        Score the last-generated positions against the ML-BOP potential.

        See :meth:`specter.ice._energy.MLBOP.compute_energy`. Defaults to
        periodic boundaries (``pbc=True``): a generated block is itself the
        full periodic cell it was optimised as, unlike e.g.
        :class:`~specter.ice.MDSimDump`'s hard-edge-trimmed MD frames.

        Parameters
        ----------
        pbc : bool, optional
            Whether to treat the block as periodic with box lengths
            ``(box_x, box_y, box_z)``. Default is True.

        Returns
        -------
        dict[str, float]
            See :meth:`specter.ice._energy.MLBOP.compute_energy` for the
            fields returned.
        """
        assert self.positions is not None, "Call optimize() or init_* first"
        box = (self.box_x, self.box_y, self.box_z)
        model = MLBOP(device=self.positions.device)
        with torch.no_grad():
            result = model.compute_energy(self.positions, box_size=box, pbc=pbc)
        return {k: v.item() for k, v in result.items()}
