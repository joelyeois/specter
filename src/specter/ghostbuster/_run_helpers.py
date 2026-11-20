"""
Device resolution for the two Ghostbuster pipelines.
"""

from __future__ import annotations

from typing import Sequence

import torch

# ---------------------------------------------------------------------------
# Shared device resolution for Ghostbuster/TomogramGhostbuster run()/test_run()
# ---------------------------------------------------------------------------


def torch_devices(
    device: int | Sequence[int] | Sequence[str] | str,
) -> list[torch.device]:
    """
    The devices a ``device`` argument names, one per data-parallel rank.

    ``"cpu"`` is the only spelling that forces the CPU; every other value
    targets the GPU when CUDA is available and falls back to a single CPU
    process when it is not. Without the explicit ``"cpu"`` case,
    ``run(device=...)`` had no way to express "run on the CPU" on a machine
    that has a GPU, since the decision was taken from
    ``torch.cuda.is_available()`` alone.

    A sequence of ``"cpu"`` entries names that many CPU ranks, which train
    and solve data-parallel over the ``gloo`` backend. It exists so the
    multi-device code paths can be tested without GPUs; on one host the
    ranks share its cores and run no faster than one.

    Parameters
    ----------
    device : int or sequence of int or sequence of "cpu" or "cpu"
        A GPU index, a sequence of GPU indices for data-parallel training,
        ``"cpu"``, or a sequence of ``"cpu"`` entries.

    Returns
    -------
    list of torch.device
        At least one device; several means one rank per device.

    Raises
    ------
    ValueError
        For a string other than ``"cpu"``, an empty sequence, or a sequence
        mixing GPU indices and ``"cpu"``.
    """
    if isinstance(device, str):
        if device != "cpu":
            raise ValueError(
                f"device={device!r} is invalid: the only string accepted is "
                "'cpu'; pass a GPU index (0) or a list of them ([0, 1]) to "
                "target GPUs."
            )
        return [torch.device("cpu")]
    entries: list[int | str] = [device] if isinstance(device, int) else [*device]
    if not entries:
        raise ValueError("device is an empty sequence; name at least one device.")
    if all(entry == "cpu" for entry in entries):
        return [torch.device("cpu")] * len(entries)
    if not all(
        isinstance(entry, int) and not isinstance(entry, bool) for entry in entries
    ):
        raise ValueError(
            f"device={device!r} mixes GPU indices with other entries; pass GPU "
            "indices only, or 'cpu' entries only."
        )
    if not torch.cuda.is_available():
        return [torch.device("cpu")]
    return [torch.device("cuda", int(entry)) for entry in entries]
