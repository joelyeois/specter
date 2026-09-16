"""
Expectation-maximisation for MRA with known noise level (Sigworth 1998).

The observations follow a uniform mixture of Gaussians centred on the ``d``
cyclic shifts of ``theta`` (Section 2.1 of Perry et al.). EM alternates

* E-step: posterior over the latent shift (and rotation) of every sample,
  ``w_il ~ exp(<y_i, R_l theta> / sigma^2)``, computed for all ``l`` at once
  with FFT cross-correlation;
* M-step: ``theta <- (1/n) sum_i sum_l w_il R_l^{-1} y_i``, a circular
  convolution of the weights with the sample.

This is the maximum-likelihood estimator whose empirical ``1/SNR^3`` behaviour
is shown in Fig. 2 of the paper. It works for 1-D signals and 2-D images.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class EMResult:
    """Output of :func:`em_mra`."""

    theta: torch.Tensor
    log_likelihood: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False


def _correlate_all_shifts(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """``c[i, l] = sum_j y_i[j] theta[j + l]`` for every cyclic shift ``l``."""
    dims = tuple(range(1, y.ndim))
    yf = torch.fft.fftn(y, dim=dims)
    tf = torch.fft.fftn(theta)
    return torch.fft.ifftn(yf.conj() * tf, dim=dims).real


def _shift_back_weighted(w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``sum_l w[i, l] (R_l^{-1} y_i)[a] = sum_l w[i, l] y_i[a - l]`` (circular conv)."""
    dims = tuple(range(1, y.ndim))
    wf = torch.fft.fftn(w, dim=dims)
    yf = torch.fft.fftn(y, dim=dims)
    return torch.fft.ifftn(wf * yf, dim=dims).real


def _rot(t: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate a 2-D template by ``k`` quarter turns; identity for 1-D or ``k == 0``."""
    if k % 4 == 0 or t.ndim != 2:
        return t
    return torch.rot90(t, k)


def em_mra(
    y: torch.Tensor,
    sigma: float,
    n_iter: int = 100,
    init: torch.Tensor | None = None,
    rotations: bool = False,
    batch_size: int = 2048,
    tol: float = 1e-7,
    generator: torch.Generator | None = None,
) -> EMResult:
    """
    Maximum-likelihood MRA estimation by EM with known ``sigma``.

    Parameters
    ----------
    y : torch.Tensor
        Observations ``[n, d]`` or ``[n, H, W]``.
    sigma : float
        Known noise standard deviation.
    n_iter : int, optional
        Maximum number of EM iterations. Default 100.
    init : torch.Tensor, optional
        Initial template. Default: mean of a few samples plus a small random
        perturbation (a data-driven, symmetry-breaking start).
    rotations : bool, optional
        Include latent 90-degree rotations (2-D only). Default False.
    batch_size : int, optional
        Samples per E-step chunk, bounds memory at ``batch_size * d * 4`` floats.
    tol : float, optional
        Stop when the relative change of ``theta`` drops below this.
    generator : torch.Generator, optional
        RNG for the initialisation.

    Returns
    -------
    EMResult
        Estimated template (defined up to a cyclic shift / rotation) and
        the per-iteration log-likelihood.
    """
    n = y.shape[0]
    shape = y.shape[1:]
    n_shift = int(torch.tensor(shape).prod())
    if init is None:
        theta = y[: min(64, n)].mean(0)
        theta = theta + 0.1 * theta.std() * torch.randn(
            theta.shape, generator=generator
        ).to(y)
    else:
        theta = init.clone().to(y)
    n_rot = 4 if (rotations and len(shape) == 2) else 1
    result = EMResult(theta=theta)
    inv_var = 1.0 / sigma**2

    for it in range(n_iter):
        templates = [_rot(theta, k) for k in range(n_rot)]
        acc = torch.zeros_like(theta)
        ll = 0.0
        for start in range(0, n, batch_size):
            yb = y[start : start + batch_size]
            # log-weights over (rotation, shift); ||R theta||^2 is constant in l
            logits = torch.stack(
                [_correlate_all_shifts(yb, t) * inv_var for t in templates], dim=1
            )  # [b, n_rot, *shape]
            flat = logits.reshape(yb.shape[0], -1)
            lse = torch.logsumexp(flat, dim=1)
            w = torch.exp(flat - lse[:, None]).reshape(logits.shape)
            for k in range(n_rot):
                back = _shift_back_weighted(w[:, k], yb).sum(0)  # [*shape]
                acc += _rot(back, -k)
            # marginal log-likelihood up to constants (uniform prior over latents)
            norm_sq = float(theta.pow(2).sum())
            ll += (
                float((lse - torch.log(torch.tensor(float(n_rot * n_shift)))).sum())
                - 0.5 * inv_var * norm_sq * yb.shape[0]
            )
        new_theta = acc / n
        delta = float((new_theta - theta).norm() / theta.norm().clamp_min(1e-12))
        theta = new_theta
        result.log_likelihood.append(ll)
        result.n_iter = it + 1
        if delta < tol:
            result.converged = True
            break
    result.theta = theta
    return result
