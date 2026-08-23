"""Single-particle reconstruction pipeline: `ReconstructionConfig` in, a
directory holding a reconstructed volume out.

Drives `specter.ghostbuster.Ghostbuster`, which does all of the real work --
loading the `.cs` metadata and the particle stack, preprocessing the images,
and fitting a `Reconstructor`. What this module adds is the three things that
turn that class into a command:

- **Config-to-constructor translation.** Every `ReconstructionConfig` field
  except the output/job/device ones is a `Ghostbuster` keyword argument under
  the same name, so the mapping is mechanical and stays that way.
- **Device parsing.** The rest of the CLI spells devices as strings
  (`"cuda:1"`, `"0,1"`); `Ghostbuster.run` wants a GPU index, a list of them,
  or `"cpu"`. `_parse_device` already draws exactly that distinction for the
  simulation pipelines, so this reuses it rather than inventing a second
  spelling.
- **A run directory.** Every run is numbered and tracked through
  `specter.jobs` -- there is no untracked mode, the way neither RELION nor
  CryoSPARC has one. The directory is `job_base_dir/[project/]reconstructions/J00N/`
  -- project comes right after `job_base_dir`, ahead of the
  `reconstructions` job-type subfolder, since users group their own work by
  project first. `project` is optional, not required: omitting it drops
  just the project-name segment, using `job_base_dir`'s implicit default
  project rather than skipping tracking. `job_base_dir` itself defaults to
  the project root found by walking up from cwd for an existing
  `specter-data/` (`find_specter_project_root`, the same way `git` finds
  the nearest `.git`), so running from a subdirectory of an
  already-initialised project lands in the same project rather than
  starting a second, disconnected tree. Numbering (`J001`, `J002`, ...) is
  one continuous sequence per project, shared across every job type in it,
  not restarted per type. Records a parameter snapshot with the git commit,
  and can resume into a pinned `job_id`.
- **Gold-standard dispatch.** `halfset="gold"` (the default) reconstructs
  halfsets A and B and computes the halfmap FSC between them, instead of a
  single run. The two halves run as separate worker processes -- in
  parallel across devices when at least two are given, sequentially on one
  otherwise -- spawned by `_run_gold_standard`, which resolves the run
  directory (or job) exactly *once*, before either worker starts. Letting
  each worker resolve it independently would race: `jobs.Job`'s
  auto-numbering does `_next_job_id()` then a bare `mkdir()` with no
  `exist_ok`, which two concurrent callers can't both safely do. Pinning the
  directory up front sidesteps that entirely -- workers never touch `Job`,
  they just write into a path they're handed.
"""

from __future__ import annotations

import dataclasses
import itertools
import multiprocessing
import time
from pathlib import Path
from typing import Any, Literal, cast

from specter.config import (
    SPECTER_DATA_DIR,
    ReconstructionConfig,
    cryosparc_ref_for_halfset,
    find_specter_project_root,
    validate_config,
)

from ._common import (
    _console,
    _format_elapsed,
    _parse_device,
    _parse_device_pool,
    _section,
)

#: `ReconstructionConfig` fields that configure the run rather than the
#: reconstruction itself, and so are not `Ghostbuster` constructor arguments.
_NON_GHOSTBUSTER_FIELDS = frozenset(
    {
        "test_run",
        "bin_factor",
        "device",
        "project",
        "job_id",
        "job_base_dir",
    }
)


def _ghostbuster_kwargs(config: ReconstructionConfig) -> dict[str, Any]:
    """
    Translate a config into `Ghostbuster` constructor arguments.

    Parameters
    ----------
    config : ReconstructionConfig
        The run configuration.

    Returns
    -------
    dict
        Every field that names a `Ghostbuster` argument, verbatim, except
        `cryosparc_ref`: a ``"<A>,<B>"`` pair is collapsed to the reference
        for this config's halfset, since `Ghostbuster` takes one volume.
        `run_dir` is deliberately absent: the caller supplies it, either
        directly or via `specter.jobs.Job.create`.
    """
    kwargs = {
        f.name: getattr(config, f.name)
        for f in dataclasses.fields(config)
        if f.name not in _NON_GHOSTBUSTER_FIELDS
    }
    # Idempotent for a single path, so this is also correct for the gold
    # workers, whose configs already name one halfset each.
    kwargs["cryosparc_ref"] = cryosparc_ref_for_halfset(
        config.cryosparc_ref, config.halfset
    )
    return kwargs


def _reconstruct_device(device_str: str) -> int | list[int] | str:
    """
    Translate a config `device` string into what `Ghostbuster.run` expects.

    Parameters
    ----------
    device_str : str
        ``"cpu"``, ``"cuda"``, ``"cuda:N"``, a bare GPU index, or a
        comma-separated list of GPU indices.

    Returns
    -------
    int or list of int or str
        A GPU index, a list of them for multi-GPU DDP, or ``"cpu"``.

    Examples
    --------
    "cpu"    -> "cpu"
    "cuda"   -> 0
    "cuda:1" -> 1
    "0,1"    -> [0, 1]
    """
    mode, target = _parse_device(device_str)
    if mode == "multi":
        assert isinstance(target, list)
        return target
    assert isinstance(target, str)
    if target == "cpu":
        return "cpu"
    _, _, index = target.partition(":")
    return int(index) if index else 0


def run_reconstruction(config: ReconstructionConfig) -> None:
    """
    Reconstruct a 3D volume from a CryoSPARC particle stack.

    Parameters
    ----------
    config : ReconstructionConfig
        Run configuration. Validated before any file is read.

    Notes
    -----
    Outputs (the reconstructed volume, per-epoch volumes, FSC plots and a
    parameter log) are written by the `Reconstructor` itself into the run
    directory this function chooses. Nothing is returned: the trained model
    is only meaningful alongside those files.
    """
    import specter.jobs as jobs

    validate_config(config)
    start = time.time()

    _section("Reconstruction")
    _console.print(
        f"  {Path(config.cs_file).name} + {Path(config.mrc_file).name}  |  "
        f"halfset {config.halfset}  |  {config.scattering_model}  |  "
        f"{config.epochs} epochs"
    )

    # Every run is tracked -- resolved once, here, regardless of which
    # branch below actually opens the Job(s), and passed down explicitly.
    # jobs.base_directory() would be shorter but writes a process-global that
    # nothing restores, leaking this run's root into every later bare Job()
    # in the same interpreter; it is the notebook-facing session setter, not
    # a channel for library code.
    root = config.job_base_dir or str(find_specter_project_root() / SPECTER_DATA_DIR)

    if config.halfset == "gold":
        run_dir = _run_gold_standard(config, base_dir=root)
    else:
        from specter.ghostbuster import Ghostbuster

        with jobs.Job(
            "reconstructions", config.project, job_id=config.job_id, base_dir=root
        ) as job:
            # job.create logs every Ghostbuster constructor argument into
            # job.json and injects run_dir. That introspection can't see
            # fields Ghostbuster never receives (test_run, device, ...), so
            # log those separately -- between the two, job.json ends up
            # with the full config.
            job.log(
                {f: getattr(config, f) for f in ("test_run", "bin_factor", "device")}
            )
            kwargs = _ghostbuster_kwargs(config)
            device = _reconstruct_device(config.device)
            _fit(job.create(Ghostbuster, **kwargs), config, device)
            run_dir = job.dir

    _section("Done")
    _console.print(f"  {run_dir}  |  {_format_elapsed(time.time() - start)}")


def _fit(
    ghostbuster: Any, config: ReconstructionConfig, device: int | list[int] | str
) -> None:
    """Run either the full fit or the binned single-epoch sanity check."""
    if config.test_run:
        ghostbuster.test_run(bin_factor=config.bin_factor, device=device)
    else:
        ghostbuster.run(device=device)


def _run_single_halfset(config: ReconstructionConfig, run_dir: Path) -> None:
    """
    Run one halfset reconstruction into an already-resolved run directory.

    Module-level (not a closure) so `multiprocessing`'s ``spawn`` context can
    pickle it when called from `_run_both_halfsets`. Only ever called for a
    gold-standard worker -- it doesn't touch `specter.jobs` itself, since
    `_run_gold_standard` has already opened the one shared `Job` and handed
    down its directory.

    Parameters
    ----------
    config : ReconstructionConfig
        Run configuration with ``halfset`` set to ``"A"`` or ``"B"`` --
        never ``"gold"``.
    run_dir : Path
        Directory to write into. Must already exist.
    """
    from specter.ghostbuster import Ghostbuster

    device = _reconstruct_device(config.device)
    kwargs = _ghostbuster_kwargs(config)
    _fit(Ghostbuster(run_dir=run_dir, **kwargs), config, device)


def _run_both_halfsets(config: ReconstructionConfig, run_dir: Path) -> None:
    """
    Reconstruct halfsets A and B as separate worker processes.

    Runs them in parallel, one per device, when at least two devices are
    given. With only one device, runs them sequentially instead of spawning
    both onto it at once -- concurrent training memory on a single GPU isn't
    calibrated the way `specter.memory`'s forward-pass estimate is (that
    model covers `ImageGenerator`'s batched generation, not `Reconstructor`'s
    training footprint of gradients + optimizer state), so contending for one
    GPU's memory risks an OOM crash partway through a multi-hour run. Extra
    devices beyond the first two are unused -- there are only two halfsets.

    Parameters
    ----------
    config : ReconstructionConfig
        Run configuration with ``halfset="gold"``; ``device`` names the pool
        to split halfsets A and B across.
    run_dir : Path
        Shared directory both halfsets write into. Must already exist.

    Raises
    ------
    RuntimeError
        If either halfset reconstruction fails.
    """
    devices = _parse_device_pool(config.device)
    ctx = multiprocessing.get_context("spawn")
    halfsets: tuple[Literal["A"], Literal["B"]] = ("A", "B")
    half_configs = [
        dataclasses.replace(
            config, halfset=cast(Literal["A", "B"], halfset), device=device
        )
        for halfset, device in zip(halfsets, itertools.cycle(devices))
    ]

    if len(devices) >= 2:
        procs = [
            ctx.Process(target=_run_single_halfset, args=(half_config, run_dir))
            for half_config in half_configs
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        if any(p.exitcode for p in procs):
            raise RuntimeError(
                "a halfset reconstruction failed -- see per-worker output above"
            )
    else:
        for half_config in half_configs:
            p = ctx.Process(target=_run_single_halfset, args=(half_config, run_dir))
            p.start()
            p.join()
            if p.exitcode:
                raise RuntimeError(
                    f"halfset {half_config.halfset} reconstruction failed -- "
                    "see output above"
                )


def _compute_and_save_gold_standard_fsc(run_dir: Path) -> str:
    """
    Compute the halfmap FSC between ``volume_A.mrc``/``volume_B.mrc``.

    Saves ``fsc_gold_standard.png`` into ``run_dir`` and returns the
    resolution at the FSC=0.143 crossing, for the caller to persist.

    Parameters
    ----------
    run_dir : Path
        Directory containing both halfset volumes.

    Returns
    -------
    str
        Resolution string, e.g. ``"3.201 Å"`` or ``">Nyquist"``.
    """
    import matplotlib.pyplot as plt
    import mrcfile
    import torch

    from ..plots import plot_halfmap_fsc

    with mrcfile.open(str(run_dir / "volume_A.mrc")) as mrc:
        vol_a = torch.as_tensor(mrc.data.copy())
        voxel_size = float(mrc.voxel_size.x)
    with mrcfile.open(str(run_dir / "volume_B.mrc")) as mrc:
        vol_b = torch.as_tensor(mrc.data.copy())

    fig, resolutions = plot_halfmap_fsc(
        [vol_a], [vol_b], voxel_size=voxel_size, labels=["gold-standard"], show=False
    )
    assert fig is not None
    fig.savefig(run_dir / "fsc_gold_standard.png")
    plt.close(fig)
    return resolutions[0]


def _run_gold_standard(config: ReconstructionConfig, base_dir: str) -> Path:
    """
    Reconstruct both halfsets, then compute and persist the halfmap FSC.

    The job is opened exactly once, here, before either halfset worker
    starts -- letting each worker open its own `Job` independently would
    race on `specter.jobs.Job`'s auto-numbering (see the module docstring).
    Workers never open their own `Job`; they only ever write into the path
    this function hands them.

    Parameters
    ----------
    config : ReconstructionConfig
        Run configuration with ``halfset="gold"``.

    Returns
    -------
    Path
        The run directory both halfsets and the FSC plot were written into.
    """
    import specter.jobs as jobs

    with jobs.Job(
        "reconstructions", config.project, job_id=config.job_id, base_dir=base_dir
    ) as job:
        job.log(dataclasses.asdict(config))
        run_dir = job.dir
        _run_both_halfsets(config, run_dir)
        resolution = _compute_and_save_gold_standard_fsc(run_dir)
        job.log({"resolution_gold_standard": resolution})

    return run_dir
