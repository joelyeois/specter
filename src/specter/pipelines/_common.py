"""Shared helpers for `specter.pipelines` modules: output-directory and job
resolution, the host-side image stack, scalar-or-range
sampling, and exit-wave output."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch

import specter
from specter.progress import console, format_elapsed
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


def resolve_output_dir(
    config: Any, job_type: str, *, create: bool = False, tracked: bool | None = None
) -> str:
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
    tracked : bool, optional
        Whether the run is tracked. Default `is_tracked`: whether ``project``
        or ``job_id`` is set. A command that tracks every run, as
        ``specter reconstruct particle`` does, passes ``True``; reading its
        unset ``project``/``job_id`` as untracked would hand ``Job`` the
        ``<job_type>/`` default and nest the run in
        ``<job_type>/<job_type>/J001``.

    Returns
    -------
    str
        The untracked output directory, or the root of the tracked job tree.
    """
    if config.output_dir is not None:
        return str(config.output_dir)
    if is_tracked(config) if tracked is None else tracked:
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


def _seed_or_draw(seed: int | None, *, announce: bool = True) -> int:
    """
    Seed every draw from ``seed``, or from a freshly drawn seed that is printed.

    Parameters
    ----------
    seed : int, optional
        The configured seed, or ``None`` to draw one.
    announce : bool, optional
        Whether to print a drawn seed. A non-main DDP rank passes ``False``
        so the notice appears once per run rather than once per rank.

    Returns
    -------
    int
        The seed the run was seeded with.
    """
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        specter.seed(seed)
        if announce:
            console.print(f"[dim]No seed given -- using seed={seed}[/dim]")
    else:
        specter.seed(seed)
    return seed


def _crowd_min_distance(configured: float | None, max_diameter: float) -> float | None:
    """
    Resolve a config's ``crowd_min_distance``.

    ``0`` disables the minimum distance (``None``), ``None`` defaults to the
    template's own maximum diameter, and any other value is used as given.
    """
    if configured == 0:
        return None
    if configured is None:
        return max_diameter
    return configured


def _mm_to_angstrom(value_mm: float) -> float:
    """Convert a length in millimetres (a config's ``cs``/``cc``) to Angstrom."""
    return value_mm * 1e7


def _normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Standardise each (H, W) image of a stack to zero mean and unit std."""
    mean = images.mean(dim=(-2, -1), keepdim=True)
    std = images.std(dim=(-2, -1), keepdim=True)
    return (images - mean) / std.clamp(min=1e-8)


def _print_total_time(t_start: float) -> None:
    """Print a run's closing ``Total time`` footer, measured from ``t_start``."""
    elapsed = time.perf_counter() - t_start
    console.print(f"\n[bold]Total time:[/bold] {format_elapsed(elapsed)}")


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


class _HostStack:
    """Copy batches into one CPU stack without retaining or concatenating them."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.count = 0
        self.tensor: torch.Tensor | None = None

    def append(self, batch: torch.Tensor) -> None:
        end = self.count + len(batch)
        if end > self.n:
            raise RuntimeError("generation produced more images than requested")
        if self.tensor is None:
            self.tensor = torch.empty(
                (self.n, *batch.shape[1:]), dtype=batch.dtype, device="cpu"
            )
        elif (
            batch.shape[1:] != self.tensor.shape[1:] or batch.dtype != self.tensor.dtype
        ):
            raise RuntimeError(
                "generation changed image shape or dtype between batches"
            )
        self.tensor[self.count : end].copy_(batch.detach())
        self.count = end

    def finish(self) -> torch.Tensor:
        if self.tensor is None or self.count != self.n:
            raise RuntimeError(
                f"generation produced {self.count} images, expected {self.n}"
            )
        return self.tensor


def _save_exitwave_pair(
    ew: torch.Tensor,
    suffix: str,
    output_dir: str,
    filename: str,
    pad_fft: bool,
    n_pixels: int,
    pixel_size: float,
) -> None:
    """
    Save an exit-wave tensor's magnitude and phase as separate .mrcs files,
    with `pixel_size` (Angstrom) in each header.
    """
    import mrcfile

    if pad_fft:
        ew = _crop_center(ew, n_pixels)
    os.makedirs(output_dir, exist_ok=True)
    mag_path = os.path.join(output_dir, f"{filename}_{suffix}_magnitude.mrcs")
    phase_path = os.path.join(output_dir, f"{filename}_{suffix}_phase.mrcs")
    with mrcfile.new(mag_path, overwrite=True) as mrc:
        mrc.set_data(ew.abs().numpy().astype("float32"))
        mrc.voxel_size = pixel_size
    console.print(f"  [green]✓[/green] {mag_path}")
    with mrcfile.new(phase_path, overwrite=True) as mrc:
        mrc.set_data(ew.angle().numpy().astype("float32"))
        mrc.voxel_size = pixel_size
    console.print(f"  [green]✓[/green] {phase_path}")
