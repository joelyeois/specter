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
- **A run directory.** Untracked runs land in `output_dir/run_name/`,
  following the same cwd-relative rule as every other specter output. Setting
  `project` instead routes the run through `specter.jobs`, which numbers the
  directory (`J001`, `J002`, ...), records a parameter snapshot with the git
  commit, and can resume into a pinned `job_id` -- the mechanism the two
  halfset runs of a gold-standard pair use to share one directory.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from specter.config import ReconstructionConfig, validate_config

from ._common import _console, _format_elapsed, _parse_device, _section

#: `ReconstructionConfig` fields that configure the run rather than the
#: reconstruction itself, and so are not `Ghostbuster` constructor arguments.
_NON_GHOSTBUSTER_FIELDS = frozenset(
    {
        "test_run",
        "bin_factor",
        "device",
        "output_dir",
        "run_name",
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
        Every field that names a `Ghostbuster` argument, verbatim. `run_dir`
        is deliberately absent: the caller supplies it, either directly or
        via `specter.jobs.Job.create`.
    """
    return {
        f.name: getattr(config, f.name)
        for f in dataclasses.fields(config)
        if f.name not in _NON_GHOSTBUSTER_FIELDS
    }


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


def _write_resolved_config(config: ReconstructionConfig, run_dir: Path) -> None:
    """
    Record the settings a run was launched with, next to its outputs.

    The reconstructor writes its own ``params.json``, but that covers only
    what reaches the `Reconstructor` -- not the device, the run layout, or
    whether this was a binned test run. This records the whole config, so a
    finished run directory says how to reproduce itself.

    Parameters
    ----------
    config : ReconstructionConfig
        The run configuration, after CLI overrides have been applied.
    run_dir : Path
        The run directory, which must already exist. The file is suffixed by
        halfset (``_A``/``_B``) exactly as the reconstructor's own outputs
        are, so two halfset runs sharing one job directory don't overwrite
        each other's record.
    """
    suffix = {"0": "_A", "1": "_B", "all": ""}[config.return_class]
    (run_dir / f"reconstruct_config{suffix}.json").write_text(
        json.dumps(dataclasses.asdict(config), indent=2, default=str)
    )


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
    from specter.ghostbuster import Ghostbuster

    validate_config(config)
    device = _reconstruct_device(config.device)
    kwargs = _ghostbuster_kwargs(config)
    start = time.time()

    _section("Reconstruction")
    _console.print(
        f"  {Path(config.cs_file).name} + {Path(config.mrc_file).name}  |  "
        f"class {config.return_class}  |  {config.scattering_model}  |  "
        f"{config.epochs} epochs"
    )

    if config.project is None:
        run_dir = Path(config.output_dir) / config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_resolved_config(config, run_dir)
        _fit(Ghostbuster(run_dir=run_dir, **kwargs), config, device)
    else:
        import specter.jobs as jobs

        jobs.base_directory(config.job_base_dir or config.output_dir)
        with jobs.Job("ghostbuster", config.project, job_id=config.job_id) as job:
            _write_resolved_config(config, job.dir)
            # job.create logs every constructor argument into job.json and
            # injects run_dir, so the Ghostbuster is built the same way here
            # as in the untracked branch, minus the explicit run_dir.
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
