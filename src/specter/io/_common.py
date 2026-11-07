"""
The parameter tuples, per-particle selection and uniformity checks shared by
the CryoSPARC and RELION readers.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

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


class _LoadedParameters(NamedTuple):
    """Every particle's imaging parameters, as a reader loads them, unfiltered."""

    voltage_kv: torch.Tensor
    pixel_size: torch.Tensor
    alpha: torch.Tensor
    rotations: torch.Tensor
    translations_angstrom: torch.Tensor
    ctf_params: dict[str, torch.Tensor]
    scale: torch.Tensor
    anisomag: torch.Tensor | None
    split: torch.Tensor | None


class ParticleParameters(NamedTuple):
    """
    What :func:`~specter.io.extract_parameters_from_csfile` and
    :func:`~specter.io.extract_parameters_from_starfile` return; it unpacks
    and indexes as the ten-element tuple it is. See either for the fields.
    """

    voltage_kv: torch.Tensor
    pixel_size: torch.Tensor
    alpha: torch.Tensor
    rotations: torch.Tensor
    translations_angstrom: torch.Tensor
    ctf_params: dict[str, torch.Tensor]
    scale: torch.Tensor
    anisomag: torch.Tensor | None
    indices: torch.Tensor
    halfset_labels: torch.Tensor | None


def _extract(
    loaded: _LoadedParameters,
    halfset: Literal["A", "B", "all"],
    split_values: tuple[int, int],
    n_particles: int | None,
    no_split: str,
) -> ParticleParameters:
    """
    One half-set of a reader's particles, truncated to the first `n_particles`.

    Parameters
    ----------
    loaded : _LoadedParameters
        Every particle, as the reader loaded it.
    halfset : {"A", "B", "all"}
        The half-set to keep; "all" also returns the raw split labels.
    split_values : tuple of int
        The raw split values of half-sets A and B in this format.
    n_particles : int, optional
        Keep only the first this many particles, after the half-set.
    no_split : str
        The error when a half-set is asked of a file without split labels.
    """
    split = loaded.split
    n_total = len(loaded.rotations)
    if halfset == "all":
        mask = torch.ones(n_total, dtype=torch.bool)
        indices = torch.arange(n_total)
        labels = split
    else:
        if split is None:
            raise ValueError(no_split)
        mask = split == split_values[0 if halfset == "A" else 1]
        indices = torch.nonzero(mask).flatten()
        labels = None
    keep = slice(None) if n_particles is None else slice(n_particles)

    def select(v: torch.Tensor) -> torch.Tensor:
        return v[mask][keep]

    return ParticleParameters(
        loaded.voltage_kv,
        loaded.pixel_size,
        loaded.alpha,
        select(loaded.rotations),
        select(loaded.translations_angstrom),
        {k: select(v) for k, v in loaded.ctf_params.items()},
        select(loaded.scale),
        None if loaded.anisomag is None else select(loaded.anisomag),
        indices[keep],
        None if labels is None else labels[keep],
    )
