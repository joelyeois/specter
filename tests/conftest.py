"""Shared pytest configuration for the specter test suite.

Two things are configured here: intra-op thread limiting under
``pytest-xdist`` (see :func:`_limit_threads_under_xdist` for why it is
conditional rather than unconditional), and per-test resetting of the
warn-once flags (see :func:`reset_warn_once_flags`).
"""

import os

import pytest

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
