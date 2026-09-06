"""
One grammar for `device`, shared by every consumer.

The four dispatch styles used to parse the string themselves, and re-parsing is
where they drifted: two accepted ``"auto"`` and the third did not, and the
reconstruction path mapped anything unrecognised to GPU 0. These tests pin both
halves of the fix -- one grammar, and no silent fallback.
"""

from __future__ import annotations

import pytest

from specter.devices import DeviceSpec, parse_device

VALID = [
    "cpu",
    "cuda",
    "cuda:1",
    "0",
    "0,1,2",
    "cuda:0,cuda:1",
    # A pool is a list of devices, so it may repeat one. This is how the suite
    # exercises multi-worker sharding without needing several GPUs -- see
    # tests/test_cli_build.py::test_cli_build_ice_shards_across_devices.
    "cpu,cpu",
    "cpu,0",
]
INVALID = ["auto", "banana", "gpu", "", "   ", "cuda:", "0,,1", "cuda:x"]


@pytest.mark.parametrize("spelling", VALID)
def test_valid_spellings_parse(spelling: str) -> None:
    assert isinstance(parse_device(spelling), DeviceSpec)


@pytest.mark.parametrize("spelling", INVALID)
def test_invalid_spellings_raise_rather_than_defaulting(spelling: str) -> None:
    """Silence here is what made `--device banana` train on GPU 0."""
    with pytest.raises(ValueError):
        parse_device(spelling)


def test_parsed_shapes() -> None:
    assert parse_device("cpu") == DeviceSpec(("cpu",))
    assert parse_device("cuda") == DeviceSpec(("cuda",))
    assert parse_device("cuda:1") == DeviceSpec(("cuda:1",))
    assert parse_device("0") == DeviceSpec(("cuda:0",))
    assert parse_device("0,1,2") == DeviceSpec(("cuda:0", "cuda:1", "cuda:2"))
    assert parse_device("cpu,cpu") == DeviceSpec(("cpu", "cpu"))
    assert parse_device("0,1").is_multi
    assert not parse_device("0").is_multi
    assert parse_device("0,1").primary == "cuda:0"


def test_indices_are_empty_unless_every_entry_names_a_gpu() -> None:
    """A consumer that wants bare ints cannot act on "cpu" or a bare "cuda"."""
    assert parse_device("0,1").indices == (0, 1)
    assert parse_device("cuda:2").indices == (2,)
    assert parse_device("cpu,cpu").indices == ()
    assert parse_device("cuda,cuda").indices == ()
    assert parse_device("cpu,0").indices == ()


def test_ddp_dispatch() -> None:
    """DDP wants bare GPU ids; anything else runs single-device."""
    assert parse_device("cpu").ddp_dispatch() == ("single", "cpu")
    assert parse_device("cuda").ddp_dispatch() == ("single", "cuda")
    assert parse_device("cuda:0").ddp_dispatch() == ("single", "cuda:0")
    assert parse_device("0").ddp_dispatch() == ("single", "cuda:0")
    assert parse_device("0,1,2").ddp_dispatch() == ("multi", [0, 1, 2])
    assert parse_device("cpu,cpu").ddp_dispatch() == ("single", "cpu")


def test_primary_and_pool() -> None:
    assert parse_device("cuda:1").primary_and_pool() == ("cuda:1", None)
    assert parse_device("0,1,2").primary_and_pool() == (
        "cuda:0",
        ["cuda:0", "cuda:1", "cuda:2"],
    )
    assert parse_device("cpu,cpu").primary_and_pool() == ("cpu", ["cpu", "cpu"])


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("cpu", "cpu"),
        ("cuda", 0),
        ("cuda:2", 2),
        ("0", 0),
        ("1,2", [1, 2]),
    ],
)
def test_lightning_target(spelling, expected) -> None:
    """What Ghostbuster.run takes: an index, a list of them, or "cpu"."""
    assert parse_device(spelling).lightning_target() == expected


def test_config_rejects_a_bad_device_at_load() -> None:
    """A typo should fail before a run starts, not several stages in."""
    from specter.config import ParticleStackConfig, validate_config

    config = ParticleStackConfig(pdb_source="1abc", device="banana")
    with pytest.raises(ValueError, match="device"):
        validate_config(config)
