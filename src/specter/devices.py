"""
One grammar for the ``device`` setting, parsed in one place.

Every command takes a ``device`` string, and each dispatch style wants a
different shape back: Lightning DDP wants a mode plus a target, a process pool
wants one device string per worker, ``Ghostbuster.run`` wants GPU indices, and
the tomogram renderer wants a primary device plus an optional pool. Each of
those grew its own parser, and re-parsing is where they drifted: two of the
three accepted ``"auto"`` and the third did not, so the same word meant "every
visible GPU" on ``specter build ice``, a crash on ``specter simulate
particles``, and -- through a fall-through that mapped any unrecognised string
to index 0 -- a silent ``cuda:0`` on ``specter reconstruct particle``.

So: parse once, here, into `DeviceSpec`; convert per dispatch style at the call
site. A conversion cannot drift on grammar because it does not parse anything.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        ``"cuda:N"``. Always at least one.
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


def parse_device(device_str: str) -> DeviceSpec:
    """
    Parse a ``device`` setting into a `DeviceSpec`.

    Grammar only: this does not consult `torch.cuda.is_available`, so a config
    naming CUDA on a CPU-only machine parses fine and is degraded separately by
    :func:`~specter.pipelines._common.resolve_available_device`. Keeping the two
    apart is what lets a config be validated at load time, before any device
    exists.

    There is deliberately no ``"auto"``. It used to mean "every visible GPU",
    but only on the two pool parsers, and one of those resolved it via
    `recommend_render_devices` rather than by counting GPUs -- so it was neither
    universal nor single-meaning. A caller wanting every GPU names them.

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
        If any entry is not a device. Previously an unrecognised value silently
        became GPU 0 on the reconstruction path.

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
