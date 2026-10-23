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
import math
import itertools
import multiprocessing
import queue
import time
import warnings
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Literal, cast

from specter.config import (
    ReconstructionConfig,
    cryosparc_ref_for_halfset,
    validate_config,
)

from ..devices import parse_device, resolve_available_device
from ..settings import Propagation, bundle_from_config
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
        "solver",
        "solver_iterations",
        "project",
        "job_id",
        "output_dir",
        # Propagation fields; collapsed into one `propagation` argument below.
        "scattering_model",
        "ews_curvature_sign",
        "klim",
        "rotate_mode",
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
    # Ghostbuster sets the propagation's alpha itself, from `alpha` when the
    # config overrides it and from the .cs file otherwise, so the bundle is
    # built without it: an unset override would be None, which it rejects.
    kwargs["propagation"] = bundle_from_config(Propagation, config, alpha=0.0)
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
    root = resolve_output_dir(config, "reconstructions", create=True, tracked=True)

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
    procs: Sequence[Any],
    result_queue: multiprocessing.Queue[tuple[str, dict[str, Any]]],
    poll_seconds: float = 2.0,
) -> tuple[str, dict[str, Any]] | None:
    """
    Take one result off the queue, or give up once every worker has exited.

    Polls rather than a single blocking ``get()``: a worker that crashes
    before reaching ``result_queue.put`` (an exception inside `_fit`, an
    OOM kill, ...) would otherwise leave a blocking ``get()`` waiting
    forever, since nothing is ever going to arrive. This only has to not
    block past that -- the caller's ``p.join()`` + exitcode check is what
    actually reports the failure, once this returns.

    ``procs`` is *every* worker still owed a result, not the one whose result
    this call expects. One queue serves both halfsets and either may finish
    first, so a result is only ever missing once they have all exited. Waiting
    on a single process instead dropped a live worker's summary whenever it
    finished second: its sibling's result satisfied the first call, and the
    second call saw the sibling already dead and gave up while halfset A was
    still training.

    Parameters
    ----------
    procs : sequence of multiprocessing.Process
        The workers that may still put a result.
    result_queue : multiprocessing.Queue
        Queue each worker puts its ``(halfset, results_summary)`` on.
    poll_seconds : float
        How long each ``get()`` waits before re-checking liveness. A real run
        is hours long, so this only bounds how quickly a *crash* is noticed,
        not the happy path.

    Returns
    -------
    tuple or None
        ``(halfset, results_summary)``, or ``None`` once every worker has
        exited without one left to collect.
    """
    while True:
        try:
            return result_queue.get(timeout=poll_seconds)
        except queue.Empty:
            if any(proc.is_alive() for proc in procs):
                continue
            # The result may have landed in the gap between the liveness
            # check above and here; one last non-blocking look before giving up.
            try:
                return result_queue.get_nowait()
            except queue.Empty:
                return None


def _run_both_halfsets(
    config: ReconstructionConfig,
    run_dir: Path,
    target: Any = None,
    extra_args: tuple[Any, ...] = (),
) -> dict[str, dict[str, Any]]:
    """
    Reconstruct halfsets A and B as separate worker processes.

    Runs them in parallel, one per device, when at least two devices are
    given. With only one device, runs them sequentially instead of spawning
    both onto it at once -- concurrent training memory on a single GPU isn't
    calibrated the way `specter.memory`'s forward-pass estimate is (that
    model covers `Specimen.scatter`'s batched generation, not `Reconstructor`'s
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
    target : callable, optional
        The worker, called as ``target(half_config, run_dir, result_queue,
        *extra_args)``. Default `_run_single_halfset`, the Adam fit;
        `_run_converged_halfset` is the converged solve.
    extra_args : tuple
        Further arguments for `target`.

    Returns
    -------
    dict
        ``{"A": results_summary_A, "B": results_summary_B}``.

    Raises
    ------
    RuntimeError
        If either halfset reconstruction fails.
    """
    if target is None:
        target = _run_single_halfset
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
                target=target,
                args=(half_config, run_dir, result_queue, *extra_args),
            )
            for half_config in half_configs
        ]
        for p in procs:
            p.start()
        for _ in procs:
            result = _collect_halfset_result(procs, result_queue)
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
                target=target,
                args=(half_config, run_dir, result_queue, *extra_args),
            )
            p.start()
            result = _collect_halfset_result([p], result_queue)
            if result is not None:
                results[result[0]] = result[1]
            p.join()
            if p.exitcode:
                raise RuntimeError(
                    f"halfset {half_config.halfset} reconstruction failed -- "
                    "see output above"
                )

    return results


def _compute_and_save_gold_standard_fsc(
    run_dir: Path,
    suffix: str = "",
    label: str = "gold-standard",
    mask: Any = None,
) -> tuple[str, str | None, dict[str, Any] | None]:
    """
    Compute the halfmap FSC between ``volume_A<suffix>.mrc`` and ``volume_B<suffix>.mrc``.

    Saves ``fsc_gold_standard<suffix>.png`` into ``run_dir`` and returns the
    resolution at the FSC=0.143 crossing, for the caller to persist.

    Parameters
    ----------
    run_dir : Path
        Directory containing both halfset volumes.
    suffix : str
        ``""`` for the final half-maps, ``"_adam"`` for the Adam stage's.
    label : str
        Legend entry of the curve.
    mask : torch.Tensor, optional
        A mask on the volumes' grid. When given, the figure also draws the
        masked FSC, and its crossing is returned as well.

    Returns
    -------
    unmasked : str
        Resolution string, e.g. ``"3.201 Å"`` or ``">Nyquist"``.
    masked : str or None
        The same inside `mask`; None without one.
    check : dict or None
        `ghostbuster._converge.mask_check` of `mask` on these half-maps:
        the noise-substituted ("corrected") resolution and whether the mask
        is too tight, its FSC curve dropped; None without a mask.
    """
    import matplotlib.pyplot as plt
    import mrcfile
    import torch

    from ..plots import HALFMAP_FSC_THRESHOLD, plot_halfmap_fsc, resolution_between

    with mrcfile.open(str(run_dir / f"volume_A{suffix}.mrc")) as mrc:
        vol_a = torch.as_tensor(mrc.data.copy())
        voxel_size = float(mrc.voxel_size.x)
    with mrcfile.open(str(run_dir / f"volume_B{suffix}.mrc")) as mrc:
        vol_b = torch.as_tensor(mrc.data.copy())

    fig, resolutions = plot_halfmap_fsc(
        [vol_a],
        [vol_b],
        voxel_size=voxel_size,
        labels=[label],
        mask=mask,
        show=False,
    )
    assert fig is not None
    masked = None
    check = None
    if mask is not None:
        from ..ghostbuster._converge import mask_check

        masked = resolution_between(
            vol_a, vol_b, voxel_size, HALFMAP_FSC_THRESHOLD, mask
        )
        check = mask_check(vol_a, vol_b, mask, voxel_size)
        corrected = check.pop("corrected_fsc")
        n = vol_a.shape[-1]
        k = [i / (n * voxel_size) for i in range(len(corrected))]
        ax = fig.axes[0]
        ax.plot(
            k,
            corrected,
            color="#444444",
            ls=":",
            lw=1.6,
            label=f"masked, noise-substituted ({check['corrected_resolution']:.3f} Å)",
        )
        ax.legend(frameon=True, fontsize=8, loc="upper right", framealpha=0.8)
    fig.savefig(run_dir / f"fsc_gold_standard{suffix}.png")
    plt.close(fig)
    return resolutions[0], masked, check


def _uses_converged_solver(config: ReconstructionConfig) -> bool:
    """
    Whether a gold run follows its Adam stage with the converged solve.

    The solve holds poses, shifts and defocus fixed, so a run refining any
    of them keeps the Adam result: solving afterwards would discard the
    refinement. That changes what the user gets, so it warns.
    """
    if config.solver != "converged" or config.halfset != "gold":
        return False
    refined = [
        name for name in ("lr_R", "lr_T", "lr_D") if getattr(config, name) is not None
    ]
    if refined:
        warnings.warn(
            f"{', '.join(refined)} refine(s) parameters the converged solver holds "
            "fixed; the Adam result is kept and the converged solve skipped. Set "
            "solver = 'adam' to silence this.",
            stacklevel=3,
        )
        return False
    return True


def _load_volume(path: Path) -> tuple[Any, float]:
    import mrcfile
    import torch

    with mrcfile.open(str(path), permissive=True) as mrc:
        return torch.as_tensor(mrc.data.astype("float32")), float(mrc.voxel_size.x)


def _write_volume(path: Path, volume: Any, voxel_size: float) -> None:
    import mrcfile

    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(volume.detach().cpu().numpy().astype("float32"))
        mrc.voxel_size = voxel_size


def _user_mask(config: ReconstructionConfig, shape: tuple[int, ...]) -> Any:
    """
    `fsc_mask` as a tensor when it is a map on the reconstruction's grid.

    A mask on another grid cannot be applied, and the run falls back to an
    automatic mask (or none) in its place, which changes the prior and the
    masked resolution: that warns, except under `test_run`, whose binning
    is expected to leave a full-size mask mismatched.
    """
    if config.fsc_mask is None:
        return None
    mask, _ = _load_volume(Path(config.fsc_mask))
    if tuple(mask.shape) == tuple(shape):
        return mask
    if not config.test_run:
        warnings.warn(
            f"fsc_mask {config.fsc_mask} is {tuple(mask.shape)} but the "
            f"reconstruction is {tuple(shape)}; it is not used for the Wiener "
            "prior or the masked gold-standard resolution, which use the "
            "automatic mask (mask_auto.mrc) instead, when there is one.",
            stacklevel=2,
        )
    return None


def _wiener_prior(config: ReconstructionConfig, run_dir: Path) -> dict[str, Any]:
    """
    The SSNR the prior is built from: the Adam half-maps' masked FSC.

    The mask is `fsc_mask` when it is a map on this grid, and otherwise the
    auto-tightened mask of `ghostbuster._converge.auto_mask`, written to
    ``mask_auto.mrc``. The record goes into ``job.json``, the run's only
    JSON file.
    """
    from ..ghostbuster._converge import auto_mask, masked_fsc, ssnr_from_fsc

    vol_a, voxel_size = _load_volume(run_dir / "volume_A_adam.mrc")
    vol_b, _ = _load_volume(run_dir / "volume_B_adam.mrc")
    mask = _user_mask(config, tuple(vol_a.shape))
    source = config.fsc_mask if mask is not None else "auto"
    tightening: dict[str, Any] = {}
    if mask is None:
        mask, tightening = auto_mask(vol_a, vol_b, voxel_size)
        _write_volume(run_dir / "mask_auto.mrc", mask, voxel_size)
    fsc = masked_fsc(vol_a, vol_b, mask)
    record = {
        "mask": source,
        "mask_volume": mask,
        "mask_fraction": round(float((mask > 0.5).float().mean()), 4),
        "mask_tightening": tightening,
        "fsc": [round(float(v), 6) for v in fsc],
        "ssnr": [float(v) for v in ssnr_from_fsc(fsc)],
    }
    return record


def _write_support_mask(run_dir: Path) -> Path:
    """
    The soft mask the converged solve confines the map to, from the Adam halves.

    `ghostbuster._converge.support_mask`: CryoSPARC's refinement-mask
    defaults. Written to ``mask_support.mrc``.
    """
    from ..ghostbuster._converge import support_mask

    vol_a, voxel_size = _load_volume(run_dir / "volume_A_adam.mrc")
    vol_b, _ = _load_volume(run_dir / "volume_B_adam.mrc")
    mask, _ = support_mask(vol_a, vol_b, voxel_size)
    path = run_dir / "mask_support.mrc"
    _write_volume(path, mask, voxel_size)
    return path


def _run_converged_halfset(
    config: ReconstructionConfig,
    run_dir: Path,
    result_queue: multiprocessing.Queue[tuple[str, dict[str, Any]]],
    ssnr: list[float],
    absorption_resolution: float | None = None,
    support_path: str | None = None,
    noise_weighting: bool = False,
    amplitude_contrast_in_ctf: bool = False,
) -> None:
    """
    Solve one halfset to convergence from its Adam map; a spawn worker.

    Writes ``volume_<h>_filtered.mrc``, the minimiser, and ``volume_<h>.mrc``,
    the same map with the Wiener filter undone, and puts the solve's log on
    `result_queue`, as `_run_single_halfset` does for the Adam fit.
    """
    import logging

    import numpy as np
    import torch

    import specter
    from specter.ghostbuster import Ghostbuster
    from specter.ghostbuster._converge import ConvergedSolver
    from specter.jobs._job import _serialize_value

    assert config.halfset in ("A", "B")
    h = config.halfset
    specter.set_verbosity(logging.INFO)
    device = torch.device(parse_device(resolve_available_device(config.device)).primary)
    kwargs = _ghostbuster_kwargs(config)
    # The Adam map is the start, as it is; references are not needed.
    kwargs.update(
        init_volume=str(run_dir / f"volume_{h}_adam.mrc"),
        init_lowpass=None,
        fsc_ref=None,
        fsc_mask=None,
        cryosparc_ref=None,
    )
    gb = Ghostbuster(run_dir=None, **kwargs)
    if config.test_run:
        images, voxel_size = gb._bin_images(config.bin_factor)
    else:
        images, voxel_size = gb._images, gb._voxel_size
    model, _ = gb._build_reconstructor_and_loader(images, voxel_size, 1)
    model = model.to(device)
    amplitude_contrast = float(model.propagation.alpha)
    if amplitude_contrast_in_ctf:
        # Amplitude contrast as the transfer function's phase offset on the
        # scattered wave, arcsin(a), as a linear CTF applies it, rather than
        # as absorption in the specimen: the same image for a weak object
        # except at zero frequency, where absorption would darken an image
        # whose mean the stack's normalisation has already removed.
        model.propagation = dataclasses.replace(model.propagation, alpha=0.0)
        model.phaseshift.data += math.asin(amplitude_contrast)
        amplitude_contrast = 0.0
    if absorption_resolution is not None:
        # The absorption map takes the amplitude contrast's place; the start
        # carries the Adam fit's absorption over into it.
        model.propagation = dataclasses.replace(model.propagation, alpha=0.0)
    support = None
    if support_path is not None:
        support, _ = _load_volume(Path(support_path))
        if tuple(support.shape) != tuple(model.V.shape[-3:]):
            # test_run bins the images, and the mask was made at full size.
            support = torch.nn.functional.interpolate(
                support[None, None], size=tuple(model.V.shape[-3:]), mode="trilinear"
            )[0, 0]
    solver = ConvergedSolver(
        model,
        images,
        np.asarray(ssnr),
        support=support,
        noise_weighting=noise_weighting,
        seed=0 if h == "A" else 1,
        label=h,
        absorption_resolution=absorption_resolution,
        amplitude_contrast=amplitude_contrast,
    )
    n_iter = 1 if config.test_run else config.solver_iterations
    result = solver.solve(model.V.detach(), n_iter)
    _write_volume(run_dir / f"volume_{h}_filtered.mrc", result.volume, voxel_size)
    _write_volume(run_dir / f"volume_{h}.mrc", result.unfiltered, voxel_size)
    if result.absorption is not None:
        _write_volume(
            run_dir / f"volume_{h}_absorption.mrc", result.absorption, voxel_size
        )
    summary = {
        "iterations": result.log,
        **{k: v for k, v in result.info.items() if k != "ssnr"},
    }
    result_queue.put((h, _serialize_value(summary)))


def _converge_halfsets(config: ReconstructionConfig, run_dir: Path) -> dict[str, Any]:
    """
    Stage two of a gold run: the converged solve of both halves.

    Keeps the Adam maps as ``volume_<h>_adam.mrc``, records their FSC
    (``fsc_gold_standard_adam.png``), builds the prior from it and solves
    each half on its own worker, as the Adam fit ran.

    Returns
    -------
    dict
        ``{"resolution_adam": ..., "prior": {...}, "A": log, "B": log}``.
    """
    for h in ("A", "B"):
        (run_dir / f"volume_{h}.mrc").rename(run_dir / f"volume_{h}_adam.mrc")
    prior = _wiener_prior(config, run_dir)
    mask = prior.pop("mask_volume")
    resolution_adam, resolution_adam_masked, _ = _compute_and_save_gold_standard_fsc(
        run_dir, "_adam", "Adam", mask
    )
    console.print(
        f"Converged solve: {config.solver_iterations} iterations per half from the Adam maps "
        f"(Adam half-map FSC {resolution_adam_masked} masked, {resolution_adam} "
        f"unmasked; mask: {prior['mask']})"
    )
    # A support mask (`_write_support_mask`) is not applied: on EMPIAR-10083
    # it improved the bands below 3.6 Å and lowered the top band.
    logs = _run_both_halfsets(
        config,
        run_dir,
        target=_run_converged_halfset,
        extra_args=(prior["ssnr"],),
    )
    return {
        "resolution_adam": resolution_adam,
        "resolution_adam_masked": resolution_adam_masked,
        "prior": prior,
        **logs,
    }


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


def _report_mask(
    config: ReconstructionConfig, run_dir: Path, converged: dict[str, Any] | None
) -> tuple[Any, str | None]:
    """
    The mask the final gold-standard FSC is reported in, and where it came from.

    The one the Wiener prior used when the converged solve ran (`fsc_mask`
    on the grid, else ``mask_auto.mrc``); otherwise `fsc_mask` when it is on
    the grid, else none.
    """
    if converged is not None:
        source = converged["prior"]["mask"]
        path = run_dir / "mask_auto.mrc" if source == "auto" else Path(source)
        return _load_volume(path)[0], source
    with_shape = _load_volume(run_dir / "volume_A.mrc")[0].shape
    mask = _user_mask(config, tuple(with_shape))
    return mask, (config.fsc_mask if mask is not None else None)


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
        converged = (
            _converge_halfsets(config, run_dir)
            if _uses_converged_solver(config)
            else None
        )
        mask, mask_source = _report_mask(config, run_dir, converged)
        resolution, resolution_masked, check = _compute_and_save_gold_standard_fsc(
            run_dir, mask=mask
        )
        if check is not None and check["too_tight"]:
            warnings.warn(
                f"the FSC mask ({mask_source}) is too tight: the masked FSC crosses "
                f"0.143 {check['shell_gap']:.1f} Fourier shells beyond the "
                "noise-substituted FSC, so the masked resolution "
                f"({resolution_masked}) is inflated by the mask; the corrected "
                f"resolution is {check['corrected_resolution']:.3f} Å.",
                stacklevel=2,
            )
        results = _merge_halfset_results(job.params.get("results", {}), halfset_results)
        if converged is not None:
            results["converged"] = converged
        job.log(
            {
                "resolution_gold_standard": resolution,
                "resolution_gold_standard_masked": resolution_masked,
                "resolution_gold_standard_mask": mask_source,
                "resolution_gold_standard_corrected": (
                    None if check is None else f"{check['corrected_resolution']:.3f} Å"
                ),
                "mask_check": check,
                "results": results,
            }
        )

    return run_dir
