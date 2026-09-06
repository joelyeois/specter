"""Shared pytest configuration for the specter test suite.

Configured here: intra-op thread limiting under ``pytest-xdist`` (see
:func:`_limit_threads_under_xdist` for why it is conditional rather than
unconditional), per-test resetting of the warn-once flags (see
:func:`reset_warn_once_flags`), and the box-phantom volumes and
golden-fixture helper several test modules share.

``torch`` is imported inside the fixtures rather than at module level so the
thread cap is in place before torch is first imported.
"""

import os
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "test_data"

# Threads per xdist worker. Tuned against the ``-n 32`` in pyproject.toml's
# addopts: 32 x 4 = 128, this host's core count. Measured on the full suite,
# 2 threads gives 3:13 and 8 gives 2:58 for ~38% more CPU, so 4 is both the
# fastest and the cheapest of the three.
_THREADS_PER_WORKER = "4"


def _limit_threads_under_xdist() -> None:
    """
    Cap PyTorch/OpenMP intra-op threads when running as an xdist worker.

    torch sizes its intra-op thread pool from the core count of the whole
    machine, so on a many-core host every worker process independently tries
    to use all cores. With ``-n 32`` that is 32 pools competing for the same
    cores, and the suite spends most of its time in the kernel scheduling
    threads rather than doing work.

    Capping is deliberately conditional on being under xdist. The tests are
    genuinely helped by intra-op parallelism when they have the machine to
    themselves: a serial run of ``tests/test_tomogram_generator.py`` takes
    ~3.5 min with the default thread count and ~7.5 min capped at 2. Only
    once the parallelism has been moved up to the process level does capping
    become a win. Applying it unconditionally would make a serial run
    (``-n 0``, which unsets ``PYTEST_XDIST_WORKER``) roughly twice as slow.

    The environment variables must be set before torch is imported, which is
    why this runs at conftest import time rather than from a fixture or a
    session hook.
    """
    if "PYTEST_XDIST_WORKER" not in os.environ:
        return

    # setdefault so an explicit OMP_NUM_THREADS from the caller still wins.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, _THREADS_PER_WORKER)


_limit_threads_under_xdist()


@pytest.fixture(autouse=True)
def reset_warn_once_flags():
    """
    Clear specter's warn-once flags before each test.

    Two warnings are suppressed after their first occurrence in a process --
    the missing-monomer-library one (`specter.pdb`) and the Peng-fallback one
    (`specter.potential._potential_builder`) -- because both report an
    environment- or run-level fact that a job rendering many structures would
    otherwise repeat dozens of times.

    Process-level state is test-order-dependent state: whichever test happens
    to run first consumes the single warning, and a `pytest.warns` in any
    later test in the same xdist worker fails for a reason that has nothing
    to do with the code under test. Resetting per test makes each one see a
    fresh process, which is what it is actually asserting about.
    """
    import specter.pdb as pdb_module
    import specter.potential._potential_builder as potential_builder_module

    pdb_module._monomer_library_warned = False
    potential_builder_module._peng_fallback_warned = False
    yield


def _box_phantom(n: int, lo: int, hi: int):
    """A ``(n, n, n)`` zero volume with a ``[lo:hi]`` cube of 50 V in it."""
    import torch

    volume = torch.zeros(n, n, n)
    volume[lo:hi, lo:hi, lo:hi] = 50.0
    return volume


@pytest.fixture
def small_volume():
    """3D cubic volume (32, 32, 32) with a box phantom."""
    return _box_phantom(32, 12, 20)


@pytest.fixture
def small_volume_4d(small_volume):
    """4D cubic volume (1, 32, 32, 32) as returned by MicrographSpecimenGenerator."""
    return small_volume.unsqueeze(0)


@pytest.fixture
def tiny_volume():
    """3D cubic volume (16, 16, 16) with a box phantom, for reconstruction tests."""
    return _box_phantom(16, 4, 12)


@pytest.fixture
def save_or_compare():
    """
    Golden-output helper: ``save_or_compare(name, tensor)`` saves the tensor
    as ``tests/test_data/<name>.pt`` on first run (and skips the test), and
    asserts the tensor matches it on every later run. Delete the file to
    regenerate after an intentional output change.
    """
    import torch

    def _save_or_compare(name: str, tensor) -> None:
        path = FIXTURE_DIR / f"{name}.pt"
        if path.exists():
            expected = torch.load(path, weights_only=True)
            assert torch.allclose(tensor.float(), expected.float(), atol=1e-4), (
                f"Regression failure for '{name}'. "
                "Delete the fixture file and re-run to regenerate."
            )
        else:
            FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(tensor.cpu(), path)
            pytest.skip(f"Fixture '{name}.pt' generated — re-run to verify.")

    return _save_or_compare
