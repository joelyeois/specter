"""Shared helpers for `specter.pipelines` modules: output-directory and job
resolution, single/multi-GPU generation dispatch (Lightning DDP), scalar-or-range
sampling, and exit-wave output."""

from __future__ import annotations

import glob
import os
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch

from specter.progress import console
from specter.config import (
    ScalarOrRange,
    default_output_dir,
    ensure_project_root,
    find_specter_project_root,
    parse_scalar_or_range,
)


def is_tracked(config: Any) -> bool:
    """Whether a config opts into `specter.jobs` tracking (`project` or `job_id` set)."""
    return config.project is not None or config.job_id is not None


def resolve_output_dir(config: Any, job_type: str, *, create: bool = False) -> str:
    """
    Resolve ``config.output_dir`` -- the single directory a run writes under.

    There is one output path field, not one per layout, so that a user has
    exactly one folder to point specter at. What that folder *means* follows
    from whether the run is tracked, which the user has already decided by
    setting (or not setting) ``project``/``job_id``:

    ==========  =================  =================
    output_dir  untracked          tracked
    ==========  =================  =================
    unset       ``<job_type>/``    ``<project root>``
    set         used verbatim      the job tree's root
    ==========  =================  =================

    Untracked, it is the leaf directory the files land in. Tracked, it is
    the root a ``[project/]<job_type>/J00N/`` tree grows under, so turning
    on ``--project`` organises output *within* the folder the user chose
    rather than relocating it somewhere else.

    The unset defaults differ because the tracked layout supplies its own
    ``<job_type>`` segment: defaulting both to ``<job_type>/`` would yield
    ``tomograms/tomograms/J001``. This is also why ``output_dir`` defaults
    to ``None`` on every config rather than to a baked-in string -- the
    default is not knowable until tracking is.

    Parameters
    ----------
    config : Any
        A pipeline config with ``output_dir``/``project``/``job_id`` fields.
    job_type : str
        Job-type folder name, e.g. ``"tomograms"``. Doubles as the artifact
        name in the untracked default, which is why the two vocabularies
        are deliberately the same.
    create : bool, optional
        Whether a missing project marker may be created (prompting at a
        terminal). Default ``False`` keeps this function pure, which is
        what lets a non-main DDP rank call it to agree on a path without
        racing its siblings. Only the one process that owns the run passes
        ``True`` -- see `_tracked_output_dir`.

    Returns
    -------
    str
        The untracked output directory, or the root of the tracked job tree.
    """
    if config.output_dir is not None:
        return str(config.output_dir)
    if is_tracked(config):
        root = ensure_project_root() if create else find_specter_project_root()
        return str(root)
    return default_output_dir(job_type)


def _deterministic_tracked_path(config: Any, job_type: str) -> str:
    """
    Compute a tracked job's directory as a pure string join -- no
    filesystem access, no `specter.jobs.Job` involved.

    ``output_dir/[project/]job_type/job_id``, matching exactly what
    `specter.jobs.Job` itself would resolve to. Requires ``config.job_id``
    to already be set (see `_tracked_output_dir`'s docstring for why: this
    exists specifically for callers -- a non-main DDP rank, or a pipeline
    computing where a *sibling* pipeline's tracked output will land --
    that need to agree on the path without opening a real `Job`
    themselves).

    Parameters
    ----------
    config : Any
        A pipeline config with ``output_dir``/``project``/``job_id``
        fields, ``job_id`` set.
    job_type : str
        Job-type folder name, e.g. ``"tomograms"``.

    Returns
    -------
    str
        The directory a `Job` with this config would resolve to.
    """
    parts = [resolve_output_dir(config, job_type)]
    if config.project is not None:
        parts.append(config.project)
    parts.extend([job_type, config.job_id])
    return os.path.join(*parts)


def _reserve_next_job_id(project: str | None, base_dir: str) -> str:
    """
    Compute (but don't create) the next free job id for a project -- the
    same read-only scan `specter.jobs.Job` itself does when auto-numbering.

    For a caller that needs to pin an id *before* a chained sub-call opens
    the real `Job` (e.g. `run_tilt_series` cascading tracking into a
    `tomogram_config` it's about to pass to `run_build_tomogram`), so both
    the sub-call and this caller's own later `_deterministic_tracked_path`
    computation agree on the same directory without the sub-call needing
    to hand anything back. Same narrow scan-then-create race window as
    `Job`'s own auto-numbering -- already accepted there, not a new risk
    introduced here.

    Parameters
    ----------
    project : str, optional
        Project name, or ``None`` for the implicit default project.
    base_dir : str
        Root directory jobs are created under, i.e. a resolved
        ``output_dir`` (see `resolve_output_dir`).

    Returns
    -------
    str
        The job id the next `Job(...)` opened for this project would get.
    """
    from pathlib import Path

    from specter.jobs._job import _next_job_id

    project_dir = Path(base_dir) if project is None else Path(base_dir) / project
    return _next_job_id(project_dir)


@contextmanager
def _tracked_output_dir(
    config: Any, job_type: str, is_main: bool = True
) -> Iterator[str]:
    """
    Resolve where a run's output goes, opening a job.json record if the
    caller opted in via ``project`` or ``job_id``.

    Untracked (the default -- ``project`` and ``job_id`` both unset):
    yields ``config.output_dir`` itself, no `specter.jobs.Job` involved.

    Tracked: yields the job directory, ``output_dir/[project/]job_type/J0NN``
    -- the same ``output_dir`` the untracked case writes straight into,
    now read as the root of a numbered tree rather than as the leaf. See
    `resolve_output_dir` for the full table, including what each case
    falls back to when ``output_dir`` is unset.
    Only ``is_main`` actually opens the `Job` -- creating the directory,
    writing ``job.json``, recording status on exit. Multi-GPU DDP dispatch
    re-executes a pipeline's whole top-level code once per rank (see
    `run_particle_stack`'s own ``is_main`` handling), so a non-main rank
    must be able to compute the *same* path without touching the
    filesystem or racing another rank to auto-assign a job id.
    ``validate_config`` requires an explicit ``job_id`` whenever tracking
    is combined with multi-GPU for exactly this reason -- the branch below
    is then a pure, deterministic string join, safe for every rank to
    compute independently and agree on, instead of each one calling
    :class:`~specter.jobs.Job` (and racing to auto-number) itself.

    Parameters
    ----------
    config : Any
        A pipeline config with ``output_dir``/``project``/``job_id``
        fields (``ParticleStackConfig``, ``MicrographConfig``, or
        ``TiltSeriesConfig``).
    job_type : str
        Job-type folder name, e.g. ``"particles"`` -- matches the artifact
        vocabulary `default_output_dir` already uses.
    is_main : bool
        Whether this process should open the `Job` itself. Pipelines with
        no multi-process dispatch (micrograph, tilt series) never need to
        pass this; it defaults to always opening it.

    Yields
    ------
    str
        The directory to write output into.
    """
    if not is_tracked(config):
        yield resolve_output_dir(config, job_type)
        return

    if not is_main:
        yield _deterministic_tracked_path(config, job_type)
        return

    import dataclasses

    import specter.jobs as jobs

    # This branch owns the run (is_main), so it is the one place allowed
    # to create a missing project marker -- see resolve_output_dir's
    # `create` argument for why non-main ranks must not.
    root = resolve_output_dir(config, job_type, create=True)
    # Passed rather than set via jobs.base_directory(): that writes a
    # process-global which nothing here restores, so every later bare Job()
    # in the same interpreter -- a notebook, a second pipeline, the next test
    # -- would silently inherit this run's root instead of $SPECTER_JOBS_DIR.
    # base_directory() is the notebook-facing session setter, not a channel
    # for library code to move an argument a few frames down.
    with jobs.Job(job_type, config.project, job_id=config.job_id, base_dir=root) as job:
        job.log(dataclasses.asdict(config))
        yield str(job.dir)


def _uniform_sample(value: ScalarOrRange, n: int) -> torch.Tensor:
    """Sample `n` values uniformly from a `parse_scalar_or_range` scalar or [low, high] pair."""
    low, high = parse_scalar_or_range(value)
    return torch.rand(n) * (high - low) + low


def _crop_center(t: torch.Tensor, nxy: int) -> torch.Tensor:
    """Center-crop a (..., H, W) tensor to (..., nxy, nxy). Matches Detector.forward crop."""
    H, W = t.shape[-2], t.shape[-1]
    if H == nxy and W == nxy:
        return t
    cy, cx = H // 2, W // 2
    half = nxy // 2
    return t[..., cy - half : cy + half + (nxy % 2), cx - half : cx + half + (nxy % 2)]


def _generate_single(
    model: Any,
    n: int,
    batchsize: int,
    track: Any,
    collect_exitwaves: bool = False,
    collect_clean_exitwaves: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run image generation on a single device."""
    idx = torch.arange(n)
    images: list[torch.Tensor] = []
    exitwaves: list[torch.Tensor] = []
    clean_exitwaves: list[torch.Tensor] = []
    with torch.no_grad():
        for i in track(range(0, n, batchsize), description="Generating images"):
            batch = model(idx[i : i + batchsize])
            images.append(batch.detach().cpu())
            if collect_exitwaves:
                exitwaves.append(model.exitwaves.detach().cpu())
            if collect_clean_exitwaves:
                clean_exitwaves.append(model.clean_exitwaves.detach().cpu())
    images_t = torch.concat(images, dim=0)
    exitwaves_t = torch.concat(exitwaves, dim=0) if collect_exitwaves else None
    clean_exitwaves_t = (
        torch.concat(clean_exitwaves, dim=0) if collect_clean_exitwaves else None
    )
    return images_t, exitwaves_t, clean_exitwaves_t


def _reassemble_rank_files(
    output_dir: str, n: int, world_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate every DDP rank's saved predictions back into one ordered stack.

    Each rank writes ``predictions_<rank>.pt`` and ``batch_indices_<rank>.pt``;
    the indices say which of the `n` requested particles that rank produced,
    so sorting by them undoes DDP's interleaved sharding.

    A short or missing rank file is an error rather than a smaller stack.
    Returning what happened to be on disk is how a two-GPU run of 2000
    particles came back with 1000 images and metadata for 2000, which the
    STAR writer then rejected with an opaque `Tensor.expand` failure after
    the ``.mrcs`` was already written.

    Returns
    -------
    images, sort_order : torch.Tensor
        Images in the caller's original order, and the permutation that put
        them there -- the exit-wave stacks are sharded identically and are
        reordered with the same `sort_order`.
    """
    prediction_files = sorted(glob.glob(os.path.join(output_dir, "predictions_*.pt")))
    index_files = sorted(glob.glob(os.path.join(output_dir, "batch_indices_*.pt")))
    if len(prediction_files) != world_size or len(index_files) != world_size:
        raise RuntimeError(
            f"multi-GPU reassembly expected {world_size} rank file(s) in "
            f"{output_dir}, found {len(prediction_files)} prediction and "
            f"{len(index_files)} index file(s). A rank failed before saving "
            "its share of the images."
        )

    all_preds = torch.cat([torch.load(f) for f in prediction_files], dim=0)
    all_indices = torch.cat([torch.load(f) for f in index_files], dim=0)
    sort_order = torch.argsort(all_indices)
    if not torch.equal(all_indices[sort_order], torch.arange(n)):
        raise RuntimeError(
            f"multi-GPU reassembly recovered {all_indices.numel()} particle "
            f"index(es) for a run of {n}; the ranks did not between them "
            "produce each particle exactly once."
        )
    images = all_preds[sort_order]

    for f in prediction_files + index_files:
        os.remove(f)
    return images, sort_order


def _generate_multi(
    model: Any,
    n: int,
    batchsize: int,
    gpu_ids: list[int],
    output_dir: str,
    collect_exitwaves: bool = False,
    collect_clean_exitwaves: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Run image generation across multiple GPUs using Lightning DDP.

    Returns
    -------
    images, exitwaves, clean_exitwaves : torch.Tensor or None
        Populated on rank 0; ``(None, None, None)`` on worker ranks.
        ``exitwaves``/``clean_exitwaves`` are ``None`` if their collect flag
        is ``False``.
    """
    import lightning as L
    from lightning.pytorch.callbacks import BasePredictionWriter
    from torch.utils.data import DataLoader

    class _Writer(BasePredictionWriter):
        def __init__(
            self, out_dir: str, save_exitwaves: bool, save_clean_exitwaves: bool
        ) -> None:
            super().__init__("epoch")
            self.out_dir = out_dir
            self.save_exitwaves = save_exitwaves
            self.save_clean_exitwaves = save_clean_exitwaves
            self._exitwaves: list = []
            self._clean_exitwaves: list = []

        def on_predict_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
        ) -> None:
            if self.save_exitwaves and hasattr(pl_module, "exitwaves"):
                self._exitwaves.append(pl_module.exitwaves.cpu())
            if self.save_clean_exitwaves and hasattr(pl_module, "clean_exitwaves"):
                self._clean_exitwaves.append(pl_module.clean_exitwaves.cpu())

        def write_on_epoch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            predictions: Sequence[Any],
            batch_indices: Sequence[Any],
        ) -> None:
            rank = trainer.global_rank
            images = torch.concat(list(predictions), dim=0)
            torch.save(images, os.path.join(self.out_dir, f"predictions_{rank}.pt"))
            idx = torch.squeeze(torch.tensor(batch_indices)).reshape(-1)
            torch.save(idx, os.path.join(self.out_dir, f"batch_indices_{rank}.pt"))
            if self.save_exitwaves and self._exitwaves:
                torch.save(
                    torch.cat(self._exitwaves, dim=0),
                    os.path.join(self.out_dir, f"exitwaves_{rank}.pt"),
                )
            if self.save_clean_exitwaves and self._clean_exitwaves:
                torch.save(
                    torch.cat(self._clean_exitwaves, dim=0),
                    os.path.join(self.out_dir, f"clean_exitwaves_{rank}.pt"),
                )

    os.makedirs(output_dir, exist_ok=True)
    dataloader: DataLoader = DataLoader(
        torch.arange(n),  # type: ignore[arg-type]
        batch_size=batchsize,
        shuffle=False,
        num_workers=os.cpu_count() or 0,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=gpu_ids,
        strategy="ddp",
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
        callbacks=[_Writer(output_dir, collect_exitwaves, collect_clean_exitwaves)],
    )

    print(f"Running multi-GPU generation on GPUs: {gpu_ids}")
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)

    # Every rank saves its own predictions_<rank>.pt from inside
    # `write_on_epoch_end`, and nothing in Lightning synchronises the ranks
    # afterwards. Without this barrier rank 0 globs the directory while a
    # slower rank is still writing its (multi-gigabyte) tensor, reassembles
    # only the shares that happen to be on disk, and deletes them -- so a
    # two-GPU run of 2000 particles silently produced a 1000-image stack.
    trainer.strategy.barrier("predictions_written")

    # Only rank 0 reassembles; worker ranks exit cleanly
    if trainer.global_rank != 0:
        return None, None, None

    # Reassemble images in original order
    images, sort_order = _reassemble_rank_files(output_dir, n, trainer.world_size)

    # Reassemble exit waves if collected
    exitwaves = None
    if collect_exitwaves:
        exitwave_files = sorted(glob.glob(os.path.join(output_dir, "exitwaves_*.pt")))
        if exitwave_files:
            all_exitwaves = torch.cat([torch.load(f) for f in exitwave_files], dim=0)
            exitwaves = all_exitwaves[sort_order]
            for f in exitwave_files:
                os.remove(f)

    # Reassemble clean exit waves if collected
    clean_exitwaves = None
    if collect_clean_exitwaves:
        clean_files = sorted(
            glob.glob(os.path.join(output_dir, "clean_exitwaves_*.pt"))
        )
        if clean_files:
            all_clean = torch.cat([torch.load(f) for f in clean_files], dim=0)
            clean_exitwaves = all_clean[sort_order]
            for f in clean_files:
                os.remove(f)

    return images, exitwaves, clean_exitwaves


def _save_exitwave_pair(
    ew: torch.Tensor,
    suffix: str,
    output_dir: str,
    filename: str,
    pad_fft: bool,
    n_pixels: int,
) -> None:
    """Save an exit-wave tensor's magnitude and phase as separate .mrcs files."""
    import mrcfile

    if pad_fft:
        ew = _crop_center(ew, n_pixels)
    os.makedirs(output_dir, exist_ok=True)
    mag_path = os.path.join(output_dir, f"{filename}_{suffix}_magnitude.mrcs")
    phase_path = os.path.join(output_dir, f"{filename}_{suffix}_phase.mrcs")
    with mrcfile.new(mag_path, overwrite=True) as mrc:
        mrc.set_data(ew.abs().numpy().astype("float32"))
    console.print(f"  [green]✓[/green] {mag_path}")
    with mrcfile.new(phase_path, overwrite=True) as mrc:
        mrc.set_data(ew.angle().numpy().astype("float32"))
    console.print(f"  [green]✓[/green] {phase_path}")
