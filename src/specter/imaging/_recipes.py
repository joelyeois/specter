"""
Recipes: a whole imaging run, written as the verbs a user would call.
"""

from __future__ import annotations

from typing import Any

import torch

from ._data import CTF, Exposure, Poses
from ._specimen import Specimen
from ._waves import Image

__all__ = ["simulate_particles"]


def simulate_particles(
    specimen: Specimen,
    poses: Poses,
    ctf: CTF | None,
    exposure: Exposure,
    *,
    voltage: float,
    seed: int | None = None,
    batch_size: int = 1,
    scatter: dict[str, Any] | None = None,
    optics: dict[str, Any] | None = None,
    coherence: dict[str, Any] | None = None,
    camera: dict[str, Any] | None = None,
) -> Image:
    """
    A particle stack: the whole chain, `batch_size` images at a time.

    Batched here rather than inside the verbs: an exit wave on the
    propagation grid holds ``oversample**2`` times a pixel image in complex
    numbers, so each batch is formed and detected before the next is
    propagated, and only the counts are kept.

    Parameters
    ----------
    specimen : Specimen
        The template, with any crowding, ice and damage recorded on it.
    poses : Poses
        One pose per image.
    ctf : CTF or None
        One set of transfer-function terms per image; None images the exit
        wave with no objective lens.
    exposure : Exposure
        Dose per image.
    voltage : float
        Accelerating voltage in kV.
    seed : int, optional
        Run seed for every per-image draw.
    batch_size : int, optional
        Images per pass. Default 1.
    scatter, optics, coherence, camera : dict, optional
        Keywords for :meth:`Specimen.scatter`, :meth:`ExitWave.apply_ctf`,
        :meth:`ExitWave.intensity` and :meth:`Image.detect`. A tensor with
        one row per image (a potential scale, a coincidence radius, an
        anisomag stack) is cut to each batch.

    Returns
    -------
    Image
        Counts, ``(n, nxy, nxy)``.
    """
    n = len(poses)
    batches = []
    for start in range(0, n, batch_size):
        i = slice(start, start + batch_size)
        waves = specimen.scatter(
            poses[i],
            voltage=voltage,
            exposure=exposure[i],
            seed=seed,
            **_rows(scatter, i, n),
        )
        if ctf is not None:
            waves = waves.apply_ctf(ctf[i], **_rows(optics, i, n))
        image = waves.intensity(**(coherence or {}))
        batches.append(image.detect(exposure[i], seed=seed, **_rows(camera, i, n)))
    return Image.cat(batches)


def _rows(kwargs: dict[str, Any] | None, i: slice, n: int) -> dict[str, Any]:
    """`kwargs` with every per-image tensor (first axis `n`) cut to batch `i`."""
    return {
        k: v[i] if isinstance(v, torch.Tensor) and v.ndim and len(v) == n else v
        for k, v in (kwargs or {}).items()
    }
