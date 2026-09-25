"""
Per-particle selection and uniformity checks shared by the CryoSPARC and
RELION readers.
"""

from __future__ import annotations

import torch


def _uniform_scalar(values: torch.Tensor, name: str, source: str) -> torch.Tensor:
    """
    Reduce a per-particle column that must be one value for the whole set.

    Voltage, pixel size and amplitude contrast are scalars of the forward
    model: every consumer of the readers images at a single value of each.
    A column that varies across particles (for example a set merged from
    several optics groups) therefore has no correct scalar reduction, and is
    refused rather than averaged or truncated to its first entry.

    Parameters
    ----------
    values : torch.Tensor
        The column, one entry per particle (or a 0-d tensor).
    name : str
        The column's name in the source file, used in the error message.
    source : str
        Path of the file the column was read from.

    Returns
    -------
    torch.Tensor
        0-d tensor holding the column's common value.

    Raises
    ------
    ValueError
        If any entry differs from the first beyond ``torch.allclose``'s
        default tolerance.
    """
    flat = values.reshape(-1)
    if flat.numel() == 0:
        raise ValueError(f"{source}: column {name} is empty.")
    if not torch.allclose(flat, flat[0].expand_as(flat)):
        raise ValueError(
            f"{source}: {name} is not the same for all particles (it ranges "
            f"from {float(flat.min()):.6g} to {float(flat.max()):.6g}). specter "
            "images a particle set at one value of it; split the set by optics "
            "group and run each part separately."
        )
    return flat[0]


def _select_particles(
    mask: torch.Tensor,
    indices: torch.Tensor,
    rotations: torch.Tensor,
    translations_angstrom: torch.Tensor,
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
        ``(indices, rotations, translations_angstrom, ctf_params, scale, anisomag)``.
    """
    rotations = rotations[mask]
    translations_angstrom = translations_angstrom[mask]
    ctf_params = {k: v[mask] for k, v in ctf_params.items()}
    scale = scale[mask]
    anisomag = None if anisomag is None else anisomag[mask]

    if n_particles is not None:
        indices = indices[:n_particles]
        rotations = rotations[:n_particles]
        translations_angstrom = translations_angstrom[:n_particles]
        ctf_params = {k: v[:n_particles] for k, v in ctf_params.items()}
        scale = scale[:n_particles]
        anisomag = None if anisomag is None else anisomag[:n_particles]

    return indices, rotations, translations_angstrom, ctf_params, scale, anisomag
