"""
Full-batch tomogram reconstruction: the volume that minimises the tilt
series' chi-squared plus `TomogramReconstructor`'s priors, found by
preconditioned nonlinear conjugate gradients.

A tilt series holds far fewer pixels than its tomogram holds voxels, so the
data term alone has no unique minimiser, and a fit without a prior ends in
whatever its optimiser's path through the null space happens to reach. The
solver here minimises

    1/2 sum_i sum_p (y_ip - m_ip(V))^2 / n_i  +  R(V)

over every tilt ``i`` and pixel ``p`` at once, where ``m`` is
`TomogramReconstructor`'s prediction, ``n_i`` the tilt's expected counts per
pixel (its Poisson variance) and ``R`` the reconstructor's
:meth:`~TomogramReconstructor.regularisation`. With the single-scatter
models the whole series is one non-uniform FFT of the volume
(`specter.scattering.surface_slice_sum`), so a gradient over every tilt
costs about what one tilt of the slice loop does.

The step along each search direction is the minimum of a parabola through
the objective at the current volume and at the previous step length. With a
quadratic prior alone the search directions are preconditioned by
``1 / (W(k) + lambda(k))``, ``W`` the data term's curvature per Fourier
component and ``lambda`` the prior's. Without a prior no preconditioner is
applied, since it would select a different member of the data term's null
space rather than reach the same one faster.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Callable

import torch

from specter import logger

if TYPE_CHECKING:
    from ._tomogram_reconstructor import TomogramReconstructor

__all__ = ["solve_tilt_series"]

#: Preconditioner floor, as a fraction of the largest curvature.
_CURVATURE_FLOOR = 1e-4


def _gradient_spectrum(
    shape: tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    """
    Eigenvalues of the squared forward-difference gradient, in ``rfftn`` layout.

    ``sum |grad V|^2 = sum_k L(k) |V(k)|^2 / N`` on a periodic grid, with
    ``L(k) = 4 sum_axis sin^2(pi f_axis)``.
    """
    fz = torch.fft.fftfreq(shape[0], device=device)[:, None, None]
    fy = torch.fft.fftfreq(shape[1], device=device)[None, :, None]
    fx = torch.fft.rfftfreq(shape[2], device=device)[None, None, :]
    return 4 * (
        torch.sin(math.pi * fz) ** 2
        + torch.sin(math.pi * fy) ** 2
        + torch.sin(math.pi * fx) ** 2
    )


class _Objective:
    """The solver's objective over the fitted tilts, with full-batch gradients."""

    def __init__(
        self,
        model: "TomogramReconstructor",
        images: torch.Tensor,
        tilts: torch.Tensor,
        tilt_batch: int,
    ) -> None:
        self.model = model
        self.images = images
        self.tilts = tilts
        self.tilt_batch = tilt_batch
        self.passes = 0

    def _data(self, V: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        sim = self.model._forward_tilts(self.model._taper(V), idx)
        return self.model._half_chi2(sim, self.images[idx].to(V.device), idx)

    def value(self, V: torch.Tensor) -> float:
        """The objective at ``V``."""
        total = 0.0
        with torch.no_grad():
            for idx in self.tilts.split(self.tilt_batch):
                total += float(self._data(V, idx))
                self.passes += 1
            total += float(self.model.regularisation(V))
        return total

    def value_and_gradient(self, V: torch.Tensor) -> tuple[float, float, torch.Tensor]:
        """The data term, the prior and the objective's gradient at ``V``."""
        gradient = torch.zeros_like(V)
        data = 0.0
        for idx in self.tilts.split(self.tilt_batch):
            v = V.detach().requires_grad_(True)
            loss = self._data(v, idx)
            (g,) = torch.autograd.grad(loss, v)
            gradient += g
            data += float(loss)
            self.passes += 1
        v = V.detach().requires_grad_(True)
        prior = self.model.regularisation(v)
        if prior.requires_grad:
            (g,) = torch.autograd.grad(prior, v)
            gradient += g
        return data, float(prior), gradient

    def curvature(
        self, shape: tuple[int, int, int], device: torch.device
    ) -> torch.Tensor:
        """
        The data term's curvature per Fourier component, in ``rfftn`` layout.

        The normal operator's response to a voxel at the volume's centre,
        taken as shift invariant: its transform is the curvature. The
        response is the odd part of the gradient about the empty volume,
        which removes the term the data contribute there.
        """
        centre = tuple(s // 2 for s in shape)
        step = 1e-2
        probe = torch.zeros(shape, device=device)
        responses = []
        for sign in (1.0, -1.0):
            probe[centre] = sign * step
            gradient = torch.zeros_like(probe)
            for idx in self.tilts.split(self.tilt_batch):
                v = probe.clone().requires_grad_(True)
                (g,) = torch.autograd.grad(self._data(v, idx), v)
                gradient += g
            responses.append(gradient)
        response = (responses[0] - responses[1]) / (2 * step)
        response = torch.roll(response, tuple(-c for c in centre), dims=(0, 1, 2))
        return torch.fft.rfftn(response).real.clamp_min(0)


def solve_tilt_series(
    model: "TomogramReconstructor",
    images: torch.Tensor,
    n_iter: int,
    *,
    tilts: torch.Tensor | None = None,
    tilt_batch: int = 8,
    callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """
    Fit ``model.V`` to a whole tilt series by conjugate gradients, in place.

    Parameters
    ----------
    model : TomogramReconstructor
        Supplies the forward model, the loss mask and the priors, and holds
        the volume, which is updated in place. Must be on the device to
        solve on.
    images : torch.Tensor
        The observed tilt series in electron counts, ``(N_tilts, H, W)``.
    n_iter : int
        Conjugate-gradient iterations. An edge-preserving prior
        (``tv_weight``) needs on the order of 100 to 200; a quadratic prior
        alone (``smoothness``) is preconditioned and needs about 20.
    tilts : torch.Tensor, optional
        Indices of the tilts to fit. Default all of them. Leaving tilts out
        and scoring the prediction against them is a test of the
        reconstruction that needs no ground truth.
    tilt_batch : int, optional
        Tilts predicted per forward pass. The result does not depend on it;
        device memory grows with it. Default 8.
    callback : callable, optional
        Called as ``callback(iteration, entry)`` after every iteration with
        that iteration's log entry.

    Returns
    -------
    list of dict
        One entry per iteration: ``objective``, ``data`` (the chi-squared
        term), ``prior``, ``chi2_per_pixel``, ``step`` and ``seconds``.

    Notes
    -----
    The directions are Polak-Ribiere+ conjugate gradients. Each iteration
    costs one gradient over every tilt and one or more evaluations of the
    objective for the line search.
    """
    # The volume is a parameter when a learning rate was given and a buffer
    # otherwise; the solver updates either in place.
    V = model.V.data if isinstance(model.V, torch.nn.Parameter) else model.V
    device = V.device
    shape = (int(V.shape[0]), int(V.shape[1]), int(V.shape[2]))
    tilts = torch.arange(len(model.quaternions)) if tilts is None else tilts.cpu()
    objective = _Objective(model, images, tilts, tilt_batch)
    n_pixels = sum(model._n_valid_pixels(int(i)) for i in tilts)

    preconditioner: torch.Tensor | None = None
    if model.smoothness > 0 and model.tv_weight == 0:
        W = objective.curvature(shape, device)
        prior = model.smoothness * _gradient_spectrum(shape, device)
        preconditioner = 1.0 / (W + prior + _CURVATURE_FLOOR * float(W.max()))
        del W, prior

    def precondition(g: torch.Tensor) -> torch.Tensor:
        if preconditioner is None:
            return g
        return torch.fft.irfftn(torch.fft.rfftn(g) * preconditioner, s=g.shape)

    start = time.time()
    log: list[dict[str, Any]] = []
    data, prior_value, g = objective.value_and_gradient(V)
    f = data + prior_value
    z = precondition(g)
    d = -z
    gz = float((g * z).sum())
    step = 1.0
    for iteration in range(1, n_iter + 1):
        slope = float((g * d).sum())
        if slope >= 0:  # not a descent direction: restart from steepest descent
            d, slope = -z, -gz
        # The parabola through f(0), f'(0) and f(step).
        taken = step
        for _ in range(4):
            trial = objective.value(V + step * d)
            curvature = trial - f - slope * step
            if curvature > 0:
                taken = min(-slope * step * step / (2 * curvature), 20 * step)
                break
            step *= 4  # flat or concave along d: look further
        else:
            taken = step
        V.add_(d, alpha=taken)
        data_new, prior_new, g_new = objective.value_and_gradient(V)
        if data_new + prior_new > f and taken > step:
            # The parabola overshot; fall back to the trial point.
            V.add_(d, alpha=step - taken)
            taken = step
            data_new, prior_new, g_new = objective.value_and_gradient(V)
        z_new = precondition(g_new)
        beta = max(0.0, float((z_new * (g_new - g)).sum()) / gz) if gz > 0 else 0.0
        g, z, f = g_new, z_new, data_new + prior_new
        gz = float((g * z).sum())
        d = -z + beta * d
        step = taken
        entry = {
            "objective": f,
            "data": data_new,
            "prior": prior_new,
            "chi2_per_pixel": 2 * data_new / n_pixels,
            "step": taken,
            "seconds": time.time() - start,
        }
        log.append(entry)
        logger.info(
            f"  iteration {iteration:3d}  chi2/pixel {entry['chi2_per_pixel']:.4f}  "
            f"prior {prior_new:.3e}  step {taken:.2e}"
        )
        if callback is not None:
            callback(iteration, entry)
        if gz == 0:
            break
    if model.kmask is not None:
        from ._helpers import _apply_kmask_inplace

        _apply_kmask_inplace(V, model._kmask_half())
    return log
