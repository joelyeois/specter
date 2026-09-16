"""
Jennrich's algorithm (Harshman 1970) and the ``homoJen`` estimator of Section 4.

``homoJen`` decomposes the empirical third-moment tensor
``T3 ~ (1/d) sum_l (R_l theta)^{(x)3}``, which is a rank-``d`` tensor whose
components are the ``d`` cyclic shifts of ``theta``. Any component is therefore
an estimate of ``theta`` up to a cyclic shift and a scalar, and the scalar is
fixed with the first moment (Section 4, just below Theorem 2).
"""

from __future__ import annotations

import torch

from mj_dsa4288.mra.invariants import third_moment_tensor_1d
from mj_dsa4288.mra.metrics import align_to


def jennrich(
    tensor: torch.Tensor, rank: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """
    Recover the components of ``T ~ sum_j u_j (x) u_j (x) v_j`` (paper, eq. (5)).

    Parameters
    ----------
    tensor : torch.Tensor
        Real tensor of shape ``[m, m, p]``.
    rank : int
        Number of components ``r`` to recover.
    generator : torch.Generator, optional
        RNG for the random contraction vectors ``a, b``.

    Returns
    -------
    torch.Tensor
        ``[m, r]`` matrix whose columns are unit-norm estimates of the ``u_j``,
        in arbitrary order and up to sign.
    """
    m, _, p = tensor.shape
    a = torch.randn(p, generator=generator).to(tensor)
    b = torch.randn(p, generator=generator).to(tensor)
    a, b = a / a.norm(), b / b.norm()
    mat_a = torch.einsum("ijk,k->ij", tensor, a)
    mat_b = torch.einsum("ijk,k->ij", tensor, b)
    left, _, _ = torch.linalg.svd(mat_a)
    w = left[:, :rank]
    m_small = (w.T @ mat_a @ w) @ torch.linalg.inv(w.T @ mat_b @ w)
    _, p_mat = torch.linalg.eig(m_small)
    u = w.to(p_mat.dtype) @ p_mat
    u = u.real  # for a well-conditioned real problem the eigenvectors are real
    return u / u.norm(dim=0, keepdim=True).clamp_min(1e-12)


def homojen(
    y: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
    average_components: bool = False,
) -> torch.Tensor:
    """
    Estimate a 1-D MRA signal from the third-moment tensor (Theorem 3's estimator).

    Parameters
    ----------
    y : torch.Tensor
        Observations ``[n, d]``.
    sigma : float
        Known noise standard deviation.
    generator : torch.Generator, optional
        RNG for Jennrich's random contractions.
    average_components : bool, optional
        The paper uses only the first recovered component. If True, all ``d``
        components are aligned to the first and averaged, which trades a small
        bias for lower variance. Default False (paper behaviour).

    Returns
    -------
    torch.Tensor
        Estimate ``theta_tilde`` of shape ``[d]``, defined up to a cyclic shift.

    Notes
    -----
    The scale fix ``beta = <u, 1> / mu`` requires ``theta`` to have a
    non-negligible mean (``mu = mean(sum_j y_ij)``). Templates should therefore
    not be mean-subtracted.
    """
    n, d = y.shape
    t3 = third_moment_tensor_1d(y, sigma)
    u = jennrich(t3, d, generator=generator)
    mu_sum = float(y.sum(1).mean())  # E[<y, 1>] = sum_j theta_j
    if abs(mu_sum) < 1e-8:
        raise ValueError("homojen needs a template with non-zero mean to fix the scale")
    if average_components:
        cols = [align_to(u[:, j], u[:, 0]) for j in range(d)]
        stacked = torch.stack(cols)
        signs = torch.sign((stacked * u[:, 0]).sum(1, keepdim=True))
        u1 = (stacked * signs).mean(0)
    else:
        u1 = u[:, 0]
    beta = float(u1.sum()) / mu_sum
    return u1 / beta
