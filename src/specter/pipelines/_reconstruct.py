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
  or `"cpu"`. `DeviceSpec.lightning_target` draws exactly that distinction
  from the one shared grammar, so there is no second spelling.
- **A run directory.** Every run is numbered and tracked through
  `specter.jobs` -- there is no untracked mode, the way neither RELION nor
  CryoSPARC has one. The directory is `output_dir/[project/]reconstructions/J00N/`
  -- project comes right after `output_dir`, ahead of the
  `reconstructions` job-type subfolder, since users group their own work by
  project first. `project` is optional, not required: omitting it drops
  just the project-name segment, using `output_dir`'s implicit default
  project rather than skipping tracking. `output_dir` itself defaults to
  the project root found by walking up from cwd for an existing
  `.specter` marker (`find_specter_project_root`, the same way `git` finds
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
import queue
import time
from pathlib import Path
from typing import Any, Literal, cast

from specter.config import (
    ReconstructionConfig,
    cryosparc_ref_for_halfset,
    validate_config,
)

from ..devices import parse_device, resolve_available_device
from specter.progress import console, format_elapsed, section
from ._common import (
    resolve_output_dir,
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
        "output_dir",
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


def run_reconstruction(config: ReconstructionConfig) -> None:
    """
    Reconstruct a 3D volume from a CryoSPARC particle stack.

    Parameters
    ----------
    config : ReconstructionConfig
        Run configuration. Validated before any file is read.

    Notes
    -----
    Outputs (the reconstructed volume, per-epoch volumes, FSC plots) are
    written by the `Reconstructor` itself into the run directory this
    function chooses. Everything else -- hyperparameters not already in
    `config`, loss/lr history, and per-epoch resolutions -- comes back via
    `Reconstructor.results_summary()` and is folded into the single
    `job.json` this function's `Job` writes, rather than its own file.
    Nothing is returned: the trained model is only meaningful alongside
    those files.
    """
    import specter.jobs as jobs

    validate_config(config)
    start = time.time()

    section("Reconstruction")
    console.print(
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
    root = resolve_output_dir(config, "reconstructions", create=True)

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
            device = parse_device(
                resolve_available_device(config.device)
            ).lightning_target()
            model = _fit(job.create(Ghostbuster, **kwargs), config, device)
            if model is not None:
                summary = model.results_summary()
                if config.halfset in ("A", "B"):
                    # A manual two-pass gold-standard run shares one job_id
                    # across two separate invocations -- nest by halfset and
                    # fold in whatever the other pass already recorded,
                    # rather than overwrite it. "all" has no counterpart
                    # pass, so it keeps the flat shape.
                    summary = _merge_halfset_results(
                        job.params.get("results", {}), {config.halfset: summary}
                    )
                job.log({"results": summary})
            run_dir = job.dir

    section("Done")
    console.print(f"  {run_dir}  |  {format_elapsed(time.time() - start)}")


def _fit(
    ghostbuster: Any, config: ReconstructionConfig, device: int | list[int] | str
) -> Any:
    """Run either the full fit or the binned single-epoch sanity check.

    Returns the trained `Reconstructor` (both `Ghostbuster.run` and
    `.test_run` already return it) -- `None` only if the run itself never
    calls into Lightning, which does not currently happen but would leave a
    caller checking ``model is not None`` correctly no-op instead of crashing.
    """
    if config.test_run:
        return ghostbuster.test_run(bin_factor=config.bin_factor, device=device)
    return ghostbuster.run(device=device)


def _run_single_halfset(
    config: ReconstructionConfig,
    run_dir: Path,
    result_queue: multiprocessing.Queue[tuple[str, dict[str, Any]]],
) -> None:
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
    result_queue : multiprocessing.Queue
        This worker's ``(halfset, results_summary())`` is put here for the
        orchestrator to collect -- the only channel back to it, since a
        spawned `Process`'s return value is otherwise discarded and this run
        directory is typically on NFS, where a shared results file would need
        `flock` to write safely and `flock` hangs there instead of
        serialising writers.
    """
    from specter.ghostbuster import Ghostbuster
    from specter.jobs._job import _serialize_value

    device = parse_device(resolve_available_device(config.device)).lightning_target()
    kwargs = _ghostbuster_kwargs(config)
    model = _fit(Ghostbuster(run_dir=run_dir, **kwargs), config, device)
    assert config.halfset in ("A", "B")
    # Serialized before it crosses the queue, never after. results_summary()
    # can contain a tensor (defocus_offset is a 0-dim one), and torch reduces
    # a tensor for IPC by passing a file descriptor over a socket that the
    # *sender* must outlive. This worker exits immediately after put(), so if
    # the orchestrator has not detached the fd by then the socket closes and it
    # sees `ConnectionResetError: [Errno 104]` from recv_handle -- load-
    # dependent, which is why it surfaced as an intermittent failure in a
    # different test on each parallel run. Converting to plain Python first
    # means only ordinary picklable data is ever sent. `Job.log` applies the
    # same conversion, and it is idempotent, so both halfset paths record
    # identical job.json values.
    summary = _serialize_value(model.results_summary()) if model is not None else {}
    result_queue.put((config.halfset, summary))


def _collect_halfset_result(
    proc: Any,
    result_queue: multiprocessing.Queue[tuple[str, dict[str, Any]]],
    poll_seconds: float = 2.0,
) -> tuple[str, dict[str, Any]] | None:
    """
    Wait for ``proc`` to put its result, or for it to exit without one.

    Polls rather than a single blocking ``get()``: a worker that crashes
    before reaching ``result_queue.put`` (an exception inside `_fit`, an
    OOM kill, ...) would otherwise leave a blocking ``get()`` waiting
    forever, since nothing is ever going to arrive. This only has to not
    block past that -- the caller's ``p.join()`` + exitcode check is what
    actually reports the failure, once this returns.

    Parameters
    ----------
    proc : multiprocessing.Process
        The worker to wait on.
    result_queue : multiprocessing.Queue
        Queue the worker puts its ``(halfset, results_summary)`` on.
    poll_seconds : float
        How long each ``get()`` waits before checking whether ``proc`` has
        exited. A real run is hours long, so this only bounds how quickly a
        *crash* is noticed, not the happy path.

    Returns
    -------
    tuple or None
        ``(halfset, results_summary)``, or ``None`` if the worker exited
        without ever putting one.
    """
    while True:
        try:
            return result_queue.get(timeout=poll_seconds)
        except queue.Empty:
            if proc.is_alive():
                continue
            # The result may have landed in the gap between the liveness
            # check above and here; one last non-blocking look before giving up.
            try:
                return result_queue.get_nowait()
            except queue.Empty:
                return None


def _run_both_halfsets(
    config: ReconstructionConfig, run_dir: Path
) -> dict[str, dict[str, Any]]:
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

    Each worker's result is retrieved from the shared `Queue` (via
    `_collect_halfset_result`) before this function joins it, not after: a
    `Process` that has put a large-enough item on a `Queue` can block on exit
    until a reader drains the pipe, so joining first can deadlock. Draining
    first is also what keeps the single-device branch genuinely sequential --
    each iteration waits on its own worker before starting the next.

    Parameters
    ----------
    config : ReconstructionConfig
        Run configuration with ``halfset="gold"``; ``device`` names the pool
        to split halfsets A and B across.
    run_dir : Path
        Shared directory both halfsets write into. Must already exist.

    Returns
    -------
    dict
        ``{"A": results_summary_A, "B": results_summary_B}``.

    Raises
    ------
    RuntimeError
        If either halfset reconstruction fails.
    """
    devices = list(parse_device(resolve_available_device(config.device)).devices)
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue[tuple[str, dict[str, Any]]] = ctx.Queue()
    halfsets: tuple[Literal["A"], Literal["B"]] = ("A", "B")
    half_configs = [
        dataclasses.replace(
            config, halfset=cast(Literal["A", "B"], halfset), device=device
        )
        for halfset, device in zip(halfsets, itertools.cycle(devices))
    ]

    results: dict[str, dict[str, Any]] = {}
    if len(devices) >= 2:
        procs = [
            ctx.Process(
                target=_run_single_halfset, args=(half_config, run_dir, result_queue)
            )
            for half_config in half_configs
        ]
        for p in procs:
            p.start()
        for p in procs:
            result = _collect_halfset_result(p, result_queue)
            if result is not None:
                results[result[0]] = result[1]
        for p in procs:
            p.join()
        if any(p.exitcode for p in procs):
            raise RuntimeError(
                "a halfset reconstruction failed -- see per-worker output above"
            )
    else:
        for half_config in half_configs:
            p = ctx.Process(
                target=_run_single_halfset, args=(half_config, run_dir, result_queue)
            )
            p.start()
            result = _collect_halfset_result(p, result_queue)
            if result is not None:
                results[result[0]] = result[1]
            p.join()
            if p.exitcode:
                raise RuntimeError(
                    f"halfset {half_config.halfset} reconstruction failed -- "
                    "see output above"
                )

    return results


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


def _merge_halfset_results(
    existing: dict[str, Any], new_results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """
    Merge freshly-reported halfset summaries into an existing ``results`` dict.

    ``results`` nests one entry per halfset label (``"A"``/``"B"``) plus a
    merged, chronologically-sorted ``"epochs"`` list built from every halfset
    present -- map-to-model resolutions from both halves, half-map
    resolutions from whichever half happened to compute them (see
    `Reconstructor._record_halfmap_resolutions`).

    ``existing`` is what makes the manual two-pass workflow correct:
    ``halfset="A"`` then ``halfset="B"`` into the same ``job_id``, as two
    separate process invocations. Without folding in what the first pass
    already recorded, the second pass's ``job.log`` would silently replace
    it -- `Job.log`'s own merge is a plain ``dict.update``, not a recursive
    one, so ``job.log({"results": {"B": ...}})`` on its own would overwrite
    the whole ``"results"`` value, discarding halfset A's entry entirely
    even though its files are still on disk. `_run_gold_standard` has no
    such prior state to fold in (its `Job` is freshly opened), but calls
    this the same way for one shape shared by both paths.

    Parameters
    ----------
    existing : dict
        Whatever ``results`` already held (``job.params.get("results", {})``);
        ``{}`` for a fresh job.
    new_results : dict
        ``{halfset_label: results_summary()}`` for the halfset(s) reported in
        this call -- both at once for gold-standard, one at a time for a
        manual two-pass run.

    Returns
    -------
    dict
        Every previously known halfset entry, updated or extended with
        ``new_results``, plus a freshly rebuilt ``"epochs"`` list.
    """
    merged = {k: v for k, v in existing.items() if k != "epochs"}
    merged.update(new_results)
    epochs = sorted(
        (
            entry
            for label, summary in merged.items()
            if label in ("A", "B")
            for entry in summary.get("resolutions", [])
        ),
        key=lambda e: (e.get("epoch", 0), e.get("halfset") or ""),
    )
    merged["epochs"] = epochs
    return merged


def _run_gold_standard(config: ReconstructionConfig, base_dir: str) -> Path:
    """
    Reconstruct both halfsets, then compute and persist the halfmap FSC.

    The job is opened exactly once, here, before either halfset worker
    starts -- letting each worker open its own `Job` independently would
    race on `specter.jobs.Job`'s auto-numbering (see the module docstring).
    Workers never open their own `Job`; they only ever write into the path
    this function hands them, and report their results back through
    `_run_both_halfsets`' `Queue` rather than a file of their own, so this
    is also the only place gold-standard results ever reach `job.json`.

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
        halfset_results = _run_both_halfsets(config, run_dir)
        resolution = _compute_and_save_gold_standard_fsc(run_dir)
        results = _merge_halfset_results(job.params.get("results", {}), halfset_results)
        job.log({"resolution_gold_standard": resolution, "results": results})

    return run_dir
