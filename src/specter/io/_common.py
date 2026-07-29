from __future__ import annotations

import torch


def _select_particles(
    mask: torch.Tensor,
    indices: torch.Tensor,
    rotations: torch.Tensor,
    translations_A: torch.Tensor,
    ctf_params: dict[str, torch.Tensor],
    scale: torch.Tensor,
    anisomag: torch.Tensor | None,
    n_particles: int | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor | None,
]:
    """Apply a per-particle boolean mask, then truncate to the first n_particles.

    Returns
    -------
    tuple
        ``(indices, rotations, translations_A, ctf_params, scale, anisomag)``.
    """
    rotations = rotations[mask]
    translations_A = translations_A[mask]
    ctf_params = {k: v[mask] for k, v in ctf_params.items()}
    scale = scale[mask]
    anisomag = None if anisomag is None else anisomag[mask]

    if n_particles is not None:
        indices = indices[:n_particles]
        rotations = rotations[:n_particles]
        translations_A = translations_A[:n_particles]
        ctf_params = {k: v[:n_particles] for k, v in ctf_params.items()}
        scale = scale[:n_particles]
        anisomag = None if anisomag is None else anisomag[:n_particles]

    return indices, rotations, translations_A, ctf_params, scale, anisomag
