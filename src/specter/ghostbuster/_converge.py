"""
Converged reconstruction: the map that minimises Ghostbuster's own image
loss plus a Wiener prior, found by preconditioned nonlinear conjugate
gradients.

Adam on minibatches stops on a noise floor set by its learning rate and
batch size, and a map without a prior over-fits every frequency the data do
not determine. The solver here minimises

    1/2 sum_i || y_i - p_i(V) ||^2  +  1/2 sum_k lambda(k) |V(k)|^2

exactly, where ``p_i`` is `Reconstructor`'s prediction (forward model, then
the per-particle scale about the no-specimen level) and the prior is the
Wiener (tau^2) form of RELION (Scheres 2012): ``lambda(k) = W(k) / SSNR(k)``,
``W(k)`` the data term's curvature averaged over a shell and ``SSNR(k)`` the
half-map signal-to-noise ratio, ``FSC / (1 - FSC)``. The minimiser is the
Wiener-filtered map; multiplying it by ``1 + 1 / SSNR(k)`` undoes the
filter, giving the unfiltered half-map that FSC and post-processing expect.

Every iteration sums the gradient over every particle; the step length comes
from a Gauss-Newton estimate along the search direction, refined by a
parabola through the loss on a random subset of particles. The
preconditioner is ``1 / (W(k) + lambda(k))``, radial, so it commutes with
the point-group and band-limit projections.

Every quantity the solve takes from the particles (the gradient, the loss,
the curvature probes and the line search's loss changes) is a sum over
them, so it shards: on several devices (`solve_on_devices`) each rank sums
its contiguous share of the particles against its own replica of the map,
the shares are all-reduced, and every rank takes the same conjugate-gradient
step from the same sums. The result is the one-device solve's to float
rounding, since only the order of the summation changes.

References
----------
.. [1] Scheres, S. H. W. (2012). RELION: Implementation of a Bayesian
   approach to cryo-EM structure determination. *Journal of Structural
   Biology*, 180(3), 519–530. https://doi.org/10.1016/j.jsb.2012.09.006
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from torch.func import functional_call

from specter import logger

from .. import nufft
from ..fft import frequency_shells
from ..symmetries import apply_symmetry
from ._helpers import _apply_kmask_inplace
from ._memory import BatchPrediction, predict_batchsize
from ._resolution import spherical_mask, unfilter
from ._training import run_on_ranks

if TYPE_CHECKING:
    from ._reconstructor import Reconstructor

#: Accuracy of the NUFFT adjoint (the surface sampling's backward) during the
#: gradient pass. See `specter.nufft.adjoint_tolerance`.
ADJOINT_TOLERANCE = 1e-4

#: Map-sized float32 tensors an iteration keeps alive through its line search
#: beyond the map itself: this and the previous iteration's ``g``, ``z`` and
#: ``d``, the move ``d_map``, and the prior weight and preconditioner (half
#: spectra, one between them). The batch-size probe holds as many, since they
#: do not exist yet when it runs.
_HELD_VOLUMES = 8

#: The ``"auto"`` batch size on the CPU, where there is no device to probe.
_CPU_BATCHSIZE = 128


def out_of_memory(error: BaseException) -> bool:
    """Whether `error` is a device allocation failure, PyTorch's or cuFINUFFT's."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error)
    return isinstance(error, RuntimeError) and (
        "out of memory" in message or "non-uniform points" in message
    )


def _log_memory(device: torch.device, stage: str) -> None:
    """
    Log device memory at debug level: the driver's view and torch's.

    ``torch.cuda.mem_get_info`` is the only one of the three that sees
    cuFINUFFT's plans, which allocate outside torch's caching allocator.
    """
    if device.type != "cuda" or not logger.isEnabledFor(logging.DEBUG):
        return
    free, total = torch.cuda.mem_get_info(device)
    logger.debug(
        f"memory {stage}: device {(total - free) / 2**30:.2f}/{total / 2**30:.2f} GiB "
        f"used, torch {torch.cuda.memory_allocated(device) / 2**30:.2f} GiB "
        f"allocated, {torch.cuda.memory_reserved(device) / 2**30:.2f} GiB reserved"
    )


@dataclass
class SolveResult:
    """
    What `ConvergedSolver.solve` returns.

    Attributes
    ----------
    volume : torch.Tensor
        The minimiser: the Wiener-filtered map.
    unfiltered : torch.Tensor
        `volume` with the filter undone (see `unfilter`).
    log : list of dict
        One entry per iteration: objective, step length, seconds.
    info : dict
        Batch size, particle counts and setup timings.
    """

    volume: torch.Tensor
    unfiltered: torch.Tensor
    log: list[dict[str, Any]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


class ConvergedSolver:
    """
    Minimise `Reconstructor`'s image loss plus a Wiener prior over one half-set.

    Parameters
    ----------
    model : Reconstructor
        Supplies the forward model, poses, CTF, per-particle scale, k-mask
        and symmetry, all held fixed. Must be on the device to solve on.
    images : torch.Tensor
        The half-set's images in counts, ``(N, H, W)``, in `model`'s particle
        order. May stay on the CPU; batches are copied as they are used.
    ssnr : numpy.ndarray
        Half-map SSNR per Fourier shell, ``n // 2 + 1`` values (see
        `ssnr_from_fsc`).
    batchsize : int or "auto"
        Particles per forward pass. ``"auto"`` (the default) predicts the
        largest batch that fits in device memory from the measured peak of
        an iteration's work at a few small batch sizes (`predict_batchsize`);
        on the CPU it is 128. The result does not depend on it, since every
        iteration sums over every particle.
    line_search_fraction : float
        Fraction of particles the step length is refined on.
    seed : int
        Seeds the probes and line-search subsets.
    label : str
        Prefixed to progress messages, e.g. the half-set's name.
    probe_support : torch.Tensor, optional
        Real-space weight ``(n, n, n)`` the curvature probes are drawn
        inside. Default: the model's own window when it applies one to the
        map (``model.support``, under ``rotate_mode="exact"`` or ``"surface"``), else the
        inscribed soft sphere (`spherical_mask`), the region every pose of a
        real-space rotation keeps in the box. Pass ones for the whole box.
    distributed : bool
        Solve as one rank of the default process group, which must be
        initialised: every rank constructs a solver with the same arguments,
        on its own device, and calls `solve` together. Each pass over
        particles covers this rank's share of them and is summed across the
        ranks; ``batchsize="auto"`` is predicted per device, for its share.
        `solve_on_devices` sets this up. Default False.

    Notes
    -----
    ``W(k)`` is measured, not modelled: the curvature of the data term along
    random unit directions confined to each of up to 45 radial bands, on a
    few hundred particles, scaled to the half-set. The lowest nine shells,
    where one band average is too coarse a preconditioner, are measured one
    shell at a time on 2% of the particles; that measurement enters the
    preconditioner only, never the prior.

    The probes are confined to `probe_support`. ``W`` is per unit norm, so a
    probe with energy where the data cannot see it reads low by that energy's
    share: a forward model that windows the map to its inscribed sphere
    before rotating sees 41-46% of a whole-box probe (boxes of 300-560), and
    a whole-box ``W`` made its prior 2.2-2.4x too weak, which overfitted
    EMPIAR-12391 at 4-1.75 A (held-out +0.15 nats/particle against Rytov;
    -0.86 with the prior corrected). Inside the support the probe sees what
    the map can hold, for Fourier-sampled and real-space models alike.
    """

    def __init__(
        self,
        model: Reconstructor,
        images: torch.Tensor,
        ssnr: np.ndarray,
        batchsize: int | Literal["auto"] = "auto",
        line_search_fraction: float = 0.05,
        seed: int = 0,
        label: str = "",
        probe_support: torch.Tensor | None = None,
        distributed: bool = False,
    ) -> None:
        self.model = model.eval()
        self.rank = dist.get_rank() if distributed else 0
        self.world_size = dist.get_world_size() if distributed else 1
        self.images = images
        self.device = model.V.device
        self.n_particles = len(images)
        self.n = int(model.V.shape[-1])
        self.level = float(model.dose_per_angstrom * model.voxel_size**2)
        if batchsize != "auto" and (
            not isinstance(batchsize, int)
            or isinstance(batchsize, bool)
            or batchsize < 1
        ):
            raise ValueError(
                f"batchsize={batchsize!r} must be a positive integer or 'auto'"
            )
        self._requested_batchsize = batchsize
        self.batchsize = self.n_particles if batchsize == "auto" else batchsize
        #: The memory probe behind an ``"auto"`` batch size, once `solve` has run.
        self.batch_prediction: BatchPrediction | None = None
        self.line_search_fraction = line_search_fraction
        self.seed = seed
        self._tag = f"converged solver{f' ({label})' if label else ''}"
        self.radius, self.shell, _ = frequency_shells(self.n, self.device)
        self.ssnr = np.asarray(ssnr, dtype=np.float64)
        if len(self.ssnr) != self.n // 2 + 1:
            raise ValueError(
                f"ssnr has {len(self.ssnr)} shells; a {self.n}^3 volume has {self.n // 2 + 1}"
            )
        self._kmask_half = model._kmask_half()
        window = getattr(model, "support", None)
        support = probe_support
        if support is None:
            support = spherical_mask(self.n) if window is None else window
        #: The window the model applies to the map, which the result is given in.
        self.window = None if window is None else window.to(self.device, torch.float32)
        if tuple(support.shape) != (self.n,) * 3:
            raise ValueError(
                f"probe_support has shape {tuple(support.shape)}; the map is {self.n}^3"
            )
        self.probe_support = support.to(self.device, torch.float32)

    # ------------------------------------------------------------------
    # Ranks: which particles are this rank's, and agreeing on the rest

    def _shard(self, idx: torch.Tensor) -> torch.Tensor:
        """This rank's contiguous share of `idx`, sizes equal to within one."""
        if self.world_size == 1:
            return idx
        return idx.tensor_split(self.world_size)[self.rank]

    def _sum(self, values: torch.Tensor) -> torch.Tensor:
        """`values` summed over the ranks, in place."""
        if self.world_size > 1:
            dist.all_reduce(values)
        return values

    def _sum_floats(self, values: list[float]) -> list[float]:
        """Scalars summed over the ranks, in float64."""
        if self.world_size == 1:
            return values
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        return [float(v) for v in self._sum(t)]

    def _sync(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rank 0's `x`, on every rank.

        The ranks compute every map-sized quantity redundantly from summed
        inputs, so they already agree to the last bit wherever the kernels
        are deterministic; the exact point-group average spreads through a
        NUFFT with atomic adds, which is not. The map and each search
        direction are taken from rank 0 before the particles see them, so
        every rank's share is a share of the same sum.
        """
        if self.world_size > 1:
            dist.broadcast(x, src=0)
        return x

    def _agree(self, value: float) -> float:
        """
        Rank 0's `value`, on every rank.

        For the scalars the solve branches on: ranks that disagreed by
        rounding on the sign of a slope would take different branches and
        then wait on different collectives.
        """
        if self.world_size == 1:
            return value
        t = torch.tensor([value], dtype=torch.float64, device=self.device)
        dist.broadcast(t, src=0)
        return float(t)

    # ------------------------------------------------------------------
    # The forward model and the passes over particles

    def _predict(self, idx: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """The predicted images of map `V`, with the per-particle scale."""
        f = functional_call(self.model, {"V": V}, (idx,))
        scale = self.model.scale[idx][:, None, None]
        return self.level + scale * (f - self.level)

    def _batch_images(self, idx: torch.Tensor) -> torch.Tensor:
        return self.images[idx.to(self.images.device)].to(
            self.device, torch.float32, non_blocking=True
        )

    def _batches(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """
        `idx` in the fewest batches of at most `batchsize`, equal to within one.

        Not full batches and a short last one: the last batch's allocations
        do not fit the blocks PyTorch's cache holds for the others, and a
        line search of 150 particles at 59 (59, 59, 32) peaked 4.5 GiB, 12%,
        above one of 59, which the batch-size probe measures. Three of 50
        stay below it.
        """
        if len(idx) == 0:
            return []
        return list(idx.tensor_split(-(-len(idx) // self.batchsize)))

    def _choose_batchsize(self, V: torch.Tensor) -> None:
        """
        Predict the batch size from the memory a few small batches need.

        The probe does a gradient pass, a curvature pass and a loss-change
        pass, the work of an iteration, while holding the map-sized tensors
        an iteration holds through its line search (`_HELD_VOLUMES`), so the
        prediction covers the line search's peak and not only the gradient's.
        """
        if self._requested_batchsize != "auto":
            return
        # This rank's share is the most a pass gives it.
        n_local = max(1, len(self._shard(torch.arange(self.n_particles))))
        if self.device.type != "cuda":
            self.batchsize = min(_CPU_BATCHSIZE, n_local)
            return
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        d = torch.randn(V.shape, device=self.device, generator=gen)
        d /= d.norm()
        step = self._step_size(V, d)

        def work(b: int) -> None:
            self.batchsize = b
            idx = torch.arange(b, device=self.device)
            held = [torch.empty_like(V) for _ in range(_HELD_VOLUMES)]
            grad, _ = self._gradient_pass(V, idx)
            del grad
            self._release()  # as `solve` does before its line search
            self._curvature_pass(V, d, step, idx)
            self._loss_changes_pass(V, d, [0.5 * step, step], idx)
            del held

        self.batch_prediction = predict_batchsize(
            work, self.device, n_local, label=self._tag
        )
        self.batchsize = self.batch_prediction.batchsize

    def _release(self) -> None:
        """
        Free the NUFFT plans and PyTorch's cached blocks.

        A plan is cached per batch size it meets (the anisomag warp pads a
        batch, or each matrix group in it, to a power of two), so line-search
        subsets that split differently every iteration kept adding plans,
        and the line search's peak crept up from one iteration to the next
        (42.0 to 42.4 GiB over four, with one anisomag matrix, 3000
        particles at a 320 box). Released at the start of an iteration and
        of its line search, the plans held are one pass's. PyTorch's cache,
        emptied too, would otherwise keep what the last pass freed from
        cuFINUFFT, which allocates from the driver: up to 40 GiB reserved
        against 2 GiB allocated, measured between passes.
        """
        nufft.clear_plan_cache()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _retrying(self, run: Any) -> Any:
        """
        Run a pass over particles, halving the batch size on an out-of-memory error.

        On a shared GPU another process can take memory mid-solve; the pass is
        redone from its start with smaller batches, which changes nothing but
        speed, since every pass sums over its particles regardless of batch.
        This is a fallback, and catches PyTorch's allocation failures only:
        when cuFINUFFT's own ``cudaMalloc`` fails, the CUDA context is left
        unusable and the redone pass fails too (``cudaErrorInvalidValue``).
        Keeping clear of that is `_choose_batchsize`'s job.
        """
        while True:
            try:
                return run()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if not out_of_memory(error) or self.batchsize == 1:
                    raise
                self.batchsize = max(1, self.batchsize // 2)
                logger.info(
                    f"{self._tag}: out of memory, batch size now {self.batchsize}"
                )
                # The NUFFT plans hold their own device buffers; release them
                # too, or the redone pass meets the same fragmented memory.
                nufft.clear_plan_cache()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

    def _gradient(self, V: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Data-term gradient and loss over every particle, summed over the ranks."""
        idx = self._shard(torch.arange(self.n_particles, device=self.device))
        grad, loss = self._retrying(lambda: self._gradient_pass(V, idx))
        (loss,) = self._sum_floats([loss])
        return self._sum(grad), loss

    def _gradient_pass(
        self, V: torch.Tensor, idx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, float]:
        """Data-term gradient and loss over `idx`, every particle by default."""
        if idx is None:
            idx = torch.arange(self.n_particles, device=self.device)
        X = V.detach().clone().requires_grad_(True)
        total = torch.zeros((), device=self.device, dtype=torch.float64)
        with nufft.adjoint_tolerance(ADJOINT_TOLERANCE):
            for b in self._batches(idx):
                r = self._predict(b, X) - self._batch_images(b)
                loss = 0.5 * (r**2).sum()
                loss.backward()
                total += loss.detach().double()
        if X.grad is None:  # no particles: this rank's share is empty
            return torch.zeros_like(X), 0.0
        return X.grad, float(total)

    def _curvature(
        self, V: torch.Tensor, d: torch.Tensor, step: float, idx: torch.Tensor
    ) -> float:
        """``sum_i ||J_i d||^2`` over `idx` (see `_curvature_pass`), over the ranks."""
        idx = self._shard(idx)
        result: float = self._retrying(lambda: self._curvature_pass(V, d, step, idx))
        return self._sum_floats([result])[0]

    @torch.no_grad()
    def _curvature_pass(
        self, V: torch.Tensor, d: torch.Tensor, step: float, idx: torch.Tensor
    ) -> float:
        """``sum_i ||J_i d||^2`` over `idx`, by a central difference of size `step`."""
        total = torch.zeros((), device=self.device, dtype=torch.float64)
        for b in self._batches(idx):
            jd = (self._predict(b, V + step * d) - self._predict(b, V - step * d)) / (
                2 * step
            )
            total += (jd.double() ** 2).sum()
        return float(total)

    def _loss_changes(
        self, V: torch.Tensor, d: torch.Tensor, steps: list[float], idx: torch.Tensor
    ) -> list[float]:
        """Change of the data loss on `idx` for each step (see `_loss_changes_pass`), over the ranks."""
        idx = self._shard(idx)
        result: list[float] = self._retrying(
            lambda: self._loss_changes_pass(V, d, steps, idx)
        )
        return self._sum_floats(result)

    @torch.no_grad()
    def _loss_changes_pass(
        self, V: torch.Tensor, d: torch.Tensor, steps: list[float], idx: torch.Tensor
    ) -> list[float]:
        """
        Change of the data loss on `idx` from ``V`` to ``V + a d``, for each `a`.

        Formed as ``(p_a - p_0) (p_a + p_0 - 2 y) / 2`` rather than as a
        difference of two losses: near convergence a step changes a subset's
        loss by parts in 1e8, which two float32 sums cannot resolve.
        """
        totals = [
            torch.zeros((), device=self.device, dtype=torch.float64) for _ in steps
        ]
        for b in self._batches(idx):
            y = self._batch_images(b)
            p0 = self._predict(b, V)
            for k, a in enumerate(steps):
                pa = self._predict(b, V + a * d)
                totals[k] += (
                    0.5 * ((pa - p0).double() * (pa + p0 - 2 * y).double()).sum()
                )
        return [float(t) for t in totals]

    def _subset(
        self, fraction: float, minimum: int, maximum: int, generator: torch.Generator
    ) -> torch.Tensor:
        k = int(
            np.clip(
                round(fraction * self.n_particles),
                min(minimum, self.n_particles),
                min(maximum, self.n_particles),
            )
        )
        return (
            torch.randperm(self.n_particles, generator=generator)[:k]
            .sort()
            .values.to(self.device)
        )

    # ------------------------------------------------------------------
    # Fourier-space operators

    def _filter(self, V: torch.Tensor, multiplier: torch.Tensor) -> torch.Tensor:
        """A Fourier multiplier on the half spectrum."""
        return torch.fft.irfftn(torch.fft.rfftn(V) * multiplier, s=V.shape)

    def _project(self, V: torch.Tensor) -> torch.Tensor:
        """S: onto the point group and the band limit, as an Adam step does."""
        model = self.model
        if model.symmetry is not None:
            V = apply_symmetry(
                V,
                model.sym_rot_matrices,
                batchsize=model.symmetry_batchsize,
                method=model.symmetry_mode,
            )
        V = V.contiguous()
        _apply_kmask_inplace(V, self._kmask_half)
        return V

    def _random_direction(
        self, selection: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        shape = selection.shape
        noise = torch.complex(
            torch.randn(shape, device=self.device, generator=generator),
            torch.randn(shape, device=self.device, generator=generator),
        )
        d = torch.fft.irfftn(noise * selection, s=(self.n,) * 3) * self.probe_support
        return d / d.norm()

    def _step_size(self, V: torch.Tensor, d: torch.Tensor) -> float:
        return 1e-2 * max(float(V.norm()), 1e-12) / float(d.norm())

    def measure_curvature(
        self,
        V: torch.Tensor,
        n_bands: int = 45,
        n_probes: int = 2,
        n_probe_particles: int = 256,
        low_shells: int = 8,
    ) -> tuple[np.ndarray, list[float]]:
        """
        The data term's curvature: band averages for the prior, low shells for the preconditioner.

        Parameters
        ----------
        V : torch.Tensor
            Map the curvature is measured at, ``(n, n, n)``.
        n_bands : int
            Radial bands between 0 and Nyquist (at most ``n // 2``).
        n_probes : int
            Random directions per band.
        n_probe_particles : int
            Particles the band probes are evaluated on.
        low_shells : int
            Shells 0 to `low_shells` are measured one at a time.

        Returns
        -------
        band_curvature : numpy.ndarray
            Curvature per band, per unit-norm direction, for the whole half-set.
        low_shell_curvature : list of float
            The same for each low shell.
        """
        gen = torch.Generator(device=self.device).manual_seed(self.seed + 1)
        cpu_gen = torch.Generator().manual_seed(self.seed + 1)
        nb = max(1, min(n_bands, self.n // 2))
        edges = torch.linspace(0, self.n / 2, nb + 1, device=self.device)
        particles = self._subset(0.0, n_probe_particles, n_probe_particles, cpu_gen)
        per_particle = self.n_particles / len(particles)
        bands = np.zeros(nb)
        for b in range(nb):
            selection = (self.radius >= edges[b]) & (self.radius < edges[b + 1])
            for _ in range(n_probes):
                d = self._sync(self._random_direction(selection, gen))
                bands[b] += (
                    self._curvature(V, d, self._step_size(V, d), particles)
                    * per_particle
                    / n_probes
                )
        particles = self._subset(0.02, 256, 4096, cpu_gen)
        per_particle = self.n_particles / len(particles)
        low = []
        for s in range(min(low_shells, self.n // 2) + 1):
            selection = self.shell == s
            if s == 0:
                d = torch.fft.irfftn(selection.to(torch.complex64), s=(self.n,) * 3)
                d = d * self.probe_support
                d = d / d.norm()
            else:
                d = self._random_direction(selection, gen)
            d = self._sync(d)
            low.append(
                self._curvature(V, d, self._step_size(V, d), particles) * per_particle
            )
        return bands, low

    # ------------------------------------------------------------------

    def solve(self, V: torch.Tensor, n_iter: int) -> SolveResult:
        """
        Run `n_iter` iterations of preconditioned PR+ conjugate gradients from `V`.

        Parameters
        ----------
        V : torch.Tensor
            Starting map, ``(n, n, n)``; a few epochs of Adam is enough.
        n_iter : int
            Iterations. Four from a five-epoch Adam map matched a sixteen-
            iteration solve to 1 - FSC <= 3e-6 in every band.

        Returns
        -------
        SolveResult
        """
        t_start = time.perf_counter()
        # The unknown is U, and the map is V = S(U), S the point-group average
        # and band limit: the problem is then unconstrained, with gradient
        # S(dL/dV) since S is self-adjoint. Projecting a gradient and the
        # iterate instead assumes S(S(x)) = S(x), which the exact average only
        # does for content inside the inscribed sphere: an I1 rotation cannot
        # carry the cube's corners onto themselves, so the noise there (16% of
        # an Adam map's norm on EMPIAR-12391) changes 14% per application, the
        # whole map 2%, and the projected iteration walked uphill.
        V = self._project(V.to(self.device, torch.float32))  # the map for U = the start
        V = self._sync(V)
        self._choose_batchsize(V)

        t0 = time.perf_counter()
        bands, low = self.measure_curvature(V)
        nb = len(bands)
        centres = (np.arange(nb) + 0.5) * (self.n / 2) / nb
        W = torch.as_tensor(
            np.interp(self.radius.cpu().numpy(), centres, bands),
            dtype=torch.float32,
            device=self.device,
        )
        ssnr = torch.as_tensor(self.ssnr, dtype=torch.float32, device=self.device)[
            self.shell
        ]
        lam = W / ssnr
        for s, value in enumerate(low):
            W[self.shell == s] = value
        precondition = (self.radius <= self.n / 2).float() / (W + lam)
        #: The prior's per-voxel weight on the half spectrum, ``W(k) / SSNR(k)``.
        self.prior_weight = lam
        del W, ssnr
        t_curvature = time.perf_counter() - t0
        logger.info(
            f"{self._tag}: curvature measured in {t_curvature:.0f} s, batch size {self.batchsize}"
        )

        cpu_gen = torch.Generator().manual_seed(self.seed + 2)
        log: list[dict[str, Any]] = []
        d_prev: torch.Tensor | None = None
        g_prev: torch.Tensor | None = None
        z_prev: torch.Tensor | None = None
        for it in range(n_iter):
            t0 = time.perf_counter()
            V = self._sync(V)
            self._release()
            _log_memory(self.device, f"iteration {it + 1} start")
            grad, data_loss = self._gradient(V)
            _log_memory(self.device, f"iteration {it + 1} after gradient")
            # The exact symmetrisation in `_project` plans NUFFTs of its own,
            # allocated by cuFINUFFT from the driver; left behind the pass's
            # plans and PyTorch's cache, that allocation failed (I1, box 448,
            # batch 638 on a 46 GiB L40), and a failed cuFINUFFT cudaMalloc
            # leaves the context unusable, past `_retrying`'s reach.
            self._release()
            lam_V = self._filter(V, lam)
            objective = data_loss + 0.5 * float((V * lam_V).sum())
            grad_V = grad + lam_V  # dObjective/dV
            del grad, lam_V
            g = self._project(grad_V)  # dObjective/dU
            # Preconditioner S M S: M is radial, diagonal on the DFT grid, but
            # an exact rotation average is not (quarter turns of an even box
            # land off-grid), so M alone puts the search direction into what S
            # annihilates and the steps change nothing.
            z = self._project(self._filter(g, precondition))
            steepest = d_prev is None or g_prev is None or z_prev is None
            if steepest:
                d = -z
            else:
                assert d_prev is not None and g_prev is not None and z_prev is not None
                beta = max(
                    0.0,
                    float((z * (g - g_prev)).sum()) / float((z_prev * g_prev).sum()),
                )
                d = -z + beta * d_prev
            # The map moves along S(d). The slope is taken along that move,
            # from the full gradient in V, not as g.d, which equals it only
            # as far as S is self-adjoint (1e-5 on a real map, 1.6e-3 for I1
            # on an 8^3 box, enough near convergence to flip its sign).
            d_map = self._sync(self._project(d))
            slope = self._agree(float((grad_V * d_map).sum()))
            if slope >= 0 and not steepest:
                d = -z
                d_map = self._sync(self._project(d))
                slope = self._agree(float((grad_V * d_map).sum()))
            del grad_V
            d_lam_d = self._agree(float((d_map * self._filter(d_map, lam)).sum()))
            subset = self._subset(
                self.line_search_fraction, 32, self.n_particles, cpu_gen
            )
            per_particle = self.n_particles / len(subset)
            self._release()
            _log_memory(self.device, f"iteration {it + 1} before line search")
            curvature = (
                self._curvature(V, d_map, self._step_size(V, d_map), subset)
                * per_particle
            )
            if slope >= 0:
                # No descent even along -S M g: converged to rounding.
                alpha = 0.0
            else:
                alpha = -slope / (curvature + d_lam_d)
                # Refine the step with the data term's actual curvature along d,
                # read off its change at alpha/2 and alpha on the subset; the full
                # Hessian, not its Gauss-Newton part, so the model's nonlinearity
                # counts. Only the curvature comes from the subset. The slope is
                # the full gradient's: near the Wiener balance the data and prior
                # slopes nearly cancel, and a subset's estimate of the data slope
                # is then mostly sampling error (on EMPIAR-12391 it put every
                # step at a quarter of its length).
                half, full = self._loss_changes(V, d_map, [0.5 * alpha, alpha], subset)
                data_curvature = 4 * (full - 2 * half) / alpha**2 * per_particle
                denominator = data_curvature + d_lam_d
                a_star = -slope / denominator if denominator > 0 else alpha
                alpha = float(np.clip(a_star, 0.05 * alpha, 2.0 * alpha))
            _log_memory(self.device, f"iteration {it + 1} after line search")
            V = V + alpha * d_map  # the map of U + alpha d, the map being linear in U
            del d_map
            d_prev, g_prev, z_prev = d, g, z
            entry = dict(
                iteration=it + 1,
                objective=objective,
                data_loss=data_loss,
                step=alpha,
                seconds=round(time.perf_counter() - t0, 1),
            )
            log.append(entry)
            logger.info(
                f"{self._tag}: iteration {it + 1}/{n_iter}, objective {objective:.6g}, {entry['seconds']:.0f} s"
            )

        prediction = self.batch_prediction
        batchsizes = [self.batchsize]
        if self.world_size > 1:
            gathered: list[Any] = [None] * self.world_size
            dist.all_gather_object(gathered, self.batchsize)
            batchsizes = [int(b) for b in gathered]
        info = dict(
            n_particles=self.n_particles,
            batchsize=self.batchsize,
            n_devices=self.world_size,
            batchsize_per_device=batchsizes,
            batch_prediction=None
            if prediction is None
            else dict(
                batchsize=prediction.batchsize,
                fixed_gib=round(prediction.fixed / 2**30, 3),
                per_particle_gib=round(prediction.per_particle / 2**30, 4),
                free_gib=round(prediction.free / 2**30, 2),
            ),
            curvature_seconds=round(t_curvature, 1),
            seconds=round(time.perf_counter() - t_start, 1),
            band_curvature=bands.tolist(),
            low_shell_curvature=low,
            ssnr=self.ssnr.tolist(),
        )
        if self.window is not None:
            V = V * self.window  # the map the forward model saw
        return SolveResult(
            volume=V, unfiltered=unfilter(V, self.ssnr), log=log, info=info
        )


def solve_on_devices(
    model: Reconstructor,
    images: torch.Tensor,
    ssnr: np.ndarray,
    n_iter: int,
    devices: list[torch.device],
    **solver_kwargs: Any,
) -> SolveResult:
    """
    `ConvergedSolver.solve` on one device or data-parallel across several.

    On several devices this process is rank 0 and every other rank is a
    spawned process (`_training.run_on_ranks`) with its own replica of
    `model`, its own NUFFT plans and its own batch-size prediction, over
    its share of the particles. `images` stays on the CPU and reaches the
    other ranks through shared memory; each rank copies only its share's
    batches to its device.

    Parameters
    ----------
    model : Reconstructor
        The forward model, holding the starting map ``V``; moved to the
        first device.
    images : torch.Tensor
        The half-set's images, ``(N, H, W)``, on the CPU.
    ssnr : numpy.ndarray
        Half-map SSNR per shell (see `ConvergedSolver`).
    n_iter : int
        Conjugate-gradient iterations.
    devices : list of torch.device
        One rank per device.
    **solver_kwargs
        Further `ConvergedSolver` arguments (``batchsize``, ``seed``,
        ``label``, ...).

    Returns
    -------
    SolveResult
        Rank 0's result; every rank holds the same map.
    """
    if len(devices) == 1:
        model = model.to(devices[0])
        solver = ConvergedSolver(model, images, ssnr, **solver_kwargs)
        return solver.solve(model.V.detach(), n_iter)
    result: SolveResult = run_on_ranks(
        _solve_rank, devices, (model.cpu(), images, ssnr, n_iter, solver_kwargs)
    )
    return result


def _solve_rank(
    rank: int,
    devices: list[torch.device],
    model: Reconstructor,
    images: torch.Tensor,
    ssnr: np.ndarray,
    n_iter: int,
    solver_kwargs: dict[str, Any],
) -> SolveResult:
    """One rank of `solve_on_devices`, inside its process group."""
    model = model.to(devices[rank])
    solver = ConvergedSolver(model, images, ssnr, distributed=True, **solver_kwargs)
    return solver.solve(model.V.detach(), n_iter)
