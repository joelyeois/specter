"""
One grammar for the ``device`` setting, parsed in one place.

Every command takes a ``device`` string, and each dispatch style wants a
different shape back: Lightning DDP wants a mode plus a target, a process pool
wants one device string per worker, ``Ghostbuster.run`` wants GPU indices, and
the tomogram renderer wants a primary device plus an optional pool. Parsing
is done once, here, into `DeviceSpec`, and each dispatch style is a method on
it: a conversion cannot drift on grammar because it does not parse anything,
whereas one parser per consumer lets the same word mean different things on
different commands.

Availability is a separate question from grammar. `parse_device` never
consults CUDA, so a config naming CUDA on a CPU-only machine parses fine at
load time; :func:`resolve_available_device` is the one place that degrades
a bare ``"cuda"`` to the CPU, and a pipeline applies it before parsing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

#: Spellings the ``device`` setting accepts, for error messages.
DEVICE_GRAMMAR = (
    "'cpu', 'cuda', 'cuda:N', a bare GPU index like '0', or a comma-separated "
    "list of indices like '0,1,2' for multi-GPU"
)


@dataclass(frozen=True)
class DeviceSpec:
    """
    A parsed ``device`` setting: the devices to run on, in the order given.

    A list rather than a backend plus indices, because a pool is genuinely a
    list of devices and may repeat one -- ``"cpu,cpu"`` is how the test suite
    exercises the multi-worker path on a machine without several GPUs, and a
    model keyed on a single ``kind`` cannot express it.

    Attributes
    ----------
    devices : tuple of str
        Normalised torch device strings: ``"cpu"``, ``"cuda"``, or
        ``"cuda:N"``. Always at least one. This is also the worker pool for
        a pipeline that shards whole units of work across devices itself.
    """

    devices: tuple[str, ...]

    @property
    def is_multi(self) -> bool:
        """Whether more than one device was named, i.e. whether to shard."""
        return len(self.devices) > 1

    @property
    def primary(self) -> str:
        """The single device to use when not sharding."""
        return self.devices[0]

    @property
    def indices(self) -> tuple[int, ...]:
        """
        GPU indices, for consumers that take bare ints.

        Empty unless *every* entry names an indexed GPU, since a caller
        wanting indices cannot act on ``"cpu"`` or a bare ``"cuda"``.
        """
        if not all(d.startswith("cuda:") for d in self.devices):
            return ()
        return tuple(int(d[len("cuda:") :]) for d in self.devices)

    def ddp_dispatch(self) -> tuple[Literal["single", "multi"], str | list[int]]:
        """
        The dispatch for a Lightning-driven pipeline: one device, or DDP.

        DDP needs bare GPU indices, so a pool that is not all indexed GPUs
        (``"cpu,cpu"``, a bare ``"cuda,cuda"``) has nothing to shard across
        and runs single-device.

        Returns
        -------
        mode : {"single", "multi"}
        target : str or list of int
            The device string for ``"single"``, the GPU ids for ``"multi"``.

        Examples
        --------
        "cpu"     -> ("single", "cpu")
        "cuda:0"  -> ("single", "cuda:0")
        "0,1,2"   -> ("multi",  [0, 1, 2])
        """
        if self.is_multi and self.indices:
            return "multi", list(self.indices)
        return "single", self.primary

    def primary_and_pool(self) -> tuple[str, list[str] | None]:
        """
        The tomogram renderer's shape: a primary device plus an optional pool.

        A scalar means "everything on this one device". A list pools those
        devices for concurrent per-species rendering, with the first as the
        primary device for everything else (membrane and filament
        generation, rotation, accumulator sizing).

        Examples
        --------
        "cuda:1"  -> ("cuda:1", None)
        "0,1,2"   -> ("cuda:0", ["cuda:0", "cuda:1", "cuda:2"])
        """
        if self.is_multi:
            return self.primary, list(self.devices)
        return self.primary, None

    def lightning_target(self) -> int | list[int] | str:
        """
        What ``Ghostbuster.run``/``TomogramGhostbuster.run`` take as ``device``.

        A GPU index, a list of them for multi-GPU DDP, or ``"cpu"``. A bare
        ``"cuda"`` names the backend without pinning a device, and becomes 0,
        the device torch itself defaults to.

        Examples
        --------
        "cpu"    -> "cpu"
        "cuda"   -> 0
        "cuda:1" -> 1
        "0,1"    -> [0, 1]
        """
        if self.primary == "cpu":
            return "cpu"
        if self.is_multi and self.indices:
            return list(self.indices)
        return self.indices[0] if self.indices else 0


def parse_device(device_str: str) -> DeviceSpec:
    """
    Parse a ``device`` setting into a `DeviceSpec`.

    Grammar only: this does not consult `torch.cuda.is_available`, so a config
    naming CUDA on a CPU-only machine parses fine and is degraded separately by
    :func:`resolve_available_device`. Keeping the two apart is what lets a
    config be validated at load time, before any device exists.

    There is no ``"auto"``: a caller wanting every GPU names them, so the
    word cannot come to mean different things on different commands.

    Parameters
    ----------
    device_str : str
        A raw ``device`` config value.

    Returns
    -------
    DeviceSpec

    Raises
    ------
    ValueError
        If any entry is not a device.

    Examples
    --------
    "cpu"      -> DeviceSpec(("cpu",))
    "cuda"     -> DeviceSpec(("cuda",))
    "cuda:1"   -> DeviceSpec(("cuda:1",))
    "0"        -> DeviceSpec(("cuda:0",))
    "0,1,2"    -> DeviceSpec(("cuda:0", "cuda:1", "cuda:2"))
    "cpu,cpu"  -> DeviceSpec(("cpu", "cpu"))
    """
    if not isinstance(device_str, str) or not device_str.strip():
        raise ValueError(f"device={device_str!r} is empty. Use {DEVICE_GRAMMAR}.")

    parts = [p.strip() for p in device_str.split(",")]
    if any(not p for p in parts):
        raise ValueError(
            f"device={device_str!r} has an empty entry. Use {DEVICE_GRAMMAR}."
        )

    devices: list[str] = []
    for part in parts:
        if part in ("cpu", "cuda"):
            devices.append(part)
            continue
        index = part[len("cuda:") :] if part.startswith("cuda:") else part
        if not index.isdigit():
            raise ValueError(
                f"device={device_str!r} is not a device specification "
                f"specter understands. Use {DEVICE_GRAMMAR}."
            )
        devices.append(f"cuda:{index}")
    return DeviceSpec(tuple(devices))


def resolve_available_device(device_str: str, stacklevel: int = 2) -> str:
    """
    Fall back to the CPU when a config asks for CUDA and there is none.

    The simulate/build configs default to ``"cuda"``, which is what a user with
    a GPU wants and what makes the GPU paths the tested ones. Taken literally on
    a machine without CUDA, though, torch raises ``RuntimeError: No CUDA GPUs
    are available`` from somewhere deep in the forward model -- a traceback that
    says nothing about which setting caused it. A default that cannot run
    everywhere is not a default, so a CUDA request with no CUDA present
    degrades to the CPU and says so.

    An explicit device INDEX is left alone. ``"cuda:2"`` or ``"0,1"`` names
    particular hardware, so silently running somewhere else would be answering a
    different question than the one asked; those still fail, and should.

    Parameters
    ----------
    device_str : str
        A raw ``device`` config value.
    stacklevel : int, optional
        Frames to skip when attributing the fallback warning. Default 2
        blames the caller, which is the pipeline.

    Returns
    -------
    str
        ``device_str``, or ``"cpu"``.
    """
    import torch  # deferred: this module is imported by config loading

    if device_str != "cuda" or torch.cuda.is_available():
        return device_str
    warnings.warn(
        "device='cuda' but no CUDA device is available; running on the CPU "
        "instead. Set device='cpu' to silence this, or check that torch was "
        "installed with CUDA support (torch.cuda.is_available() is False).",
        stacklevel=stacklevel,
    )
    return "cpu"
