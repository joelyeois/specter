from __future__ import annotations

import sys
import threading
import time
import warnings
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, TypeVar

from rich.console import Console
from rich.rule import Rule
from tqdm.auto import tqdm

T = TypeVar("T")

# Independent Console instance from pipelines/_common.py's own `_console`
# (this module sits below `pipelines/` in the dependency graph, so it
# can't import that one) -- rich.console.Console is safe to instantiate
# more than once; both write to the same stdout/terminal regardless.
_section_console = Console()


def _format_elapsed(seconds: float) -> str:
    """Format an elapsed-time duration as e.g. "1h 2m 3s", dropping empty
    leading units. Independent copy of pipelines/_common.py's own helper
    of the same name (this module sits below `pipelines/` in the
    dependency graph, and it's a two-line function -- not worth an import
    the wrong way just to avoid duplicating it)."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


class ProgressManager:
    """
    Manages progress bar positions and handles notebook-specific stacking issues.
    """

    _instance: "ProgressManager | None" = None
    _lock = threading.Lock()
    _occupied_positions: set[int]

    def __new__(cls) -> "ProgressManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._occupied_positions = set()
        return cls._instance

    def _is_notebook(self) -> bool:
        """Detect whether running inside a Jupyter notebook."""
        try:
            from IPython import get_ipython

            return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
        except (NameError, ImportError):
            return False

    def _get_next_available_position(self) -> int:
        pos = 0
        while pos in self._occupied_positions:
            pos += 1
        return pos

    def get_pbar(
        self, iterable: Iterable[T] | None, desc: str, **kwargs: Any
    ) -> tuple[tqdm[T], int]:
        with self._lock:
            pos = self._get_next_available_position()
            self._occupied_positions.add(pos)

        # Map rich-style 'transient' to tqdm 'leave'
        transient = kwargs.pop("transient", None)
        if "leave" not in kwargs:
            kwargs["leave"] = (not transient) if transient is not None else (pos == 0)

        # In notebooks, tqdm handles its own stacking — setting 'position' manually
        # causes display issues. Only set it in terminal environments.
        if not self._is_notebook() and "position" not in kwargs:
            kwargs["position"] = pos

        pbar = tqdm(iterable, desc=desc, **kwargs)
        return pbar, pos

    def release(self, pos: int) -> None:
        with self._lock:
            if pos in self._occupied_positions:
                self._occupied_positions.remove(pos)

    def reset(self) -> None:
        with self._lock:
            self._occupied_positions.clear()


def track(
    iterable: Iterable[T],
    description: str = "Working...",
    total: int | None = None,
    disable: bool = False,
    transient: bool = False,
    **kwargs: Any,
) -> Iterator[T]:
    """
    Drop-in replacement for rich.progress.track using tqdm and ProgressManager.

    Parameters
    ----------
    iterable : Iterable[T]
        The iterable to track.
    description : str, optional
        Description for the progress bar. Default is "Working...".
    total : int, optional
        Total number of items in the iterable.
    disable : bool, optional
        Whether to disable the progress bar.
    transient : bool, optional
        If True, bar is cleared after completion (rich-compatible kwarg).
    **kwargs : Any
        Additional keyword arguments passed to tqdm.
    """
    if disable:
        yield from iterable
        return

    manager = ProgressManager()
    pbar, pos = manager.get_pbar(
        iterable,
        desc=description,
        total=total,
        transient=transient,
        **kwargs,
    )
    try:
        yield from pbar
    finally:
        pbar.close()
        manager.release(pos)


@contextmanager
def status(description: str, disable: bool = False) -> Iterator[None]:
    """
    Context manager that shows a running status message with elapsed time.

    Unlike ``track``, this has no iteration count — it just shows that something
    is happening and how long it has been running, then clears on exit.

    A background thread calls ``pbar.refresh()`` every 0.3s for the
    duration of the ``with`` block -- tqdm only redraws on an explicit
    ``update()``/``refresh()`` call, so without this the elapsed timer
    (and the whole line) would print once and then sit frozen for the
    entire duration of a single long blocking call wrapped in ``status``
    (e.g. one big ``pack_hard_spheres_3d`` invocation with no natural
    iteration point of its own) -- confirmed directly: this is not
    cosmetic, callers were reporting a "stuck" run that was actually still
    working.

    Parameters
    ----------
    description : str
        Label shown next to the elapsed timer.
    disable : bool, optional
        If True, no output is shown.
    """
    if disable:
        yield
        return
    manager = ProgressManager()
    pbar, pos = manager.get_pbar(
        None,
        desc=description,
        total=None,
        bar_format="{desc}: {elapsed}",
        transient=True,
    )
    stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        while not stop_heartbeat.wait(0.3):
            pbar.refresh()

    heartbeat = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat.start()
    try:
        yield
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=1.0)
        pbar.close()
        manager.release(pos)


@contextmanager
def phase(description: str, disable: bool = False) -> Iterator[None]:
    """
    Context manager that prints a titled section-header rule for a named
    pipeline phase (e.g. "generating membranes", "fetching PDB
    structures") on entry, then a PERSISTENT one-line completion summary
    -- "{description}: {elapsed}" -- via ``tqdm.write`` (safe to
    interleave with any other active ``track``/``status``/``TqdmProgress``
    bars without corrupting their display, unlike a plain ``print``) when
    it exits. Same visual convention as `pipelines._common._section`
    (a full-width titled rule), so CLI output reads consistently across
    `specter simulate`/`specter build` regardless of which layer is
    driving a given phase.

    Meant to wrap a whole pipeline phase that itself contains one or more
    transient ``status``/``TqdmProgress`` bars for live per-item feedback
    -- those still clear themselves on completion, but the header/summary
    pair leaves a permanent record in scrollback of what ran and how long
    it took, so a caller isn't left with zero information about a
    finished phase the moment its own transient bar disappears.

    Parameters
    ----------
    description : str
        Label for the section header and the printed summary line.
    disable : bool, optional
        If True, no output is shown (the block still runs and is still
        timed internally, just not printed).
    """
    if disable:
        yield
        return
    start = phase_start(description)
    try:
        yield
    finally:
        phase_done(description, start)


def phase_start(description: str, disable: bool = False) -> float:
    """
    Print the same titled section-header rule `phase()` would on entry,
    for a block that can't easily be wrapped in a ``with phase(...):``
    (e.g. retrofitting a large pre-existing block without reindenting
    it). Pairs with `phase_done`: call this immediately before the block,
    keep the returned value, then call
    ``phase_done(description, start)`` after it.

    Parameters
    ----------
    description : str
        Label for the section header.
    disable : bool, optional
        If True, no output is shown.

    Returns
    -------
    float
        A ``time.perf_counter()`` value, timestamped right after printing
        -- pass this straight through to `phase_done`.
    """
    if not disable:
        _section_console.print(
            Rule(f"[bold yellow]{description}[/bold yellow]", style="yellow")
        )
    return time.perf_counter()


def phase_done(description: str, start: float, disable: bool = False) -> None:
    """
    Print a persistent ``"{description}: {elapsed}"`` completion summary
    for a block that already recorded its own ``time.perf_counter()``
    start time (typically from `phase_start`, though any timestamp
    works) -- use this when wrapping the block in a ``with phase(...):``
    isn't practical (e.g. retrofitting a large pre-existing block without
    reindenting it): call ``start = phase_start(description)`` before the
    block, then ``phase_done(description, start)`` after it.

    Parameters
    ----------
    description : str
        Label for the printed summary line.
    start : float
        A ``time.perf_counter()`` value captured before the timed block ran.
    disable : bool, optional
        If True, no output is shown.
    """
    if disable:
        return
    elapsed = time.perf_counter() - start
    tqdm.write(f"{description}: {_format_elapsed(elapsed)}")


@contextmanager
def tqdm_warnings(show_location: bool = True) -> Iterator[None]:
    """
    Route `warnings.warn` output through `tqdm.write` for the duration of
    the block.

    Python writes a warning straight to `sys.stderr`, which is also where
    the bars in this module draw. A warning raised while a bar is live
    therefore lands mid-line, leaving output like
    ``Generating membrane instances: 0%| | 0/3 [00:00<?, ?it/s]/path/to/
    generator.py:764: UserWarning: ...`` -- the bar and the warning both
    unreadable. `tqdm.write` takes tqdm's lock, clears every live bar,
    writes, and redraws them, so the warning gets its own line and the
    bars survive.

    Parameters
    ----------
    show_location : bool, optional
        Include the ``file:lineno`` prefix and the echoed source line, as
        Python's own formatter does. Default True. Pass False for a
        command-line front end, where the location is noise: these
        warnings are raised for the user, not for whoever is debugging
        specter, and the stack frame they name is a fixed internal call
        site (every warning out of `TomogramSpecimenGenerator.generate`
        reports the one `volume = gen.generate()` line) that tells the
        reader nothing about their own input.

    Notes
    -----
    Only the OUTPUT path is replaced. Filtering is untouched, so
    `warnings.simplefilter`/`-W` and Python's once-per-location dedup all
    behave exactly as they otherwise would.
    """
    original = warnings.showwarning

    def _showwarning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        # `file` is set when a caller explicitly redirects one warning
        # somewhere else; honour that rather than hijacking it.
        if file is not None:
            original(message, category, filename, lineno, file, line)
            return
        if show_location:
            text = warnings.formatwarning(
                message, category, filename, lineno, line
            ).rstrip()
        else:
            text = f"{category.__name__}: {message}"
        tqdm.write(text, file=sys.stderr)

    warnings.showwarning = _showwarning
    try:
        yield
    finally:
        warnings.showwarning = original


class TqdmProgress:
    """
    A context manager wrapper around ProgressManager that mimics rich.progress.Progress.

    Examples
    --------
    ::

        with TqdmProgress(transient=True) as progress:
            task = progress.add_task("Processing", total=10)
            for i in range(10):
                progress.update(task, advance=1)
    """

    def __init__(self, transient: bool = False, **kwargs: Any):
        self.transient = transient
        self.kwargs = kwargs
        self.tasks: dict[int, tuple[tqdm[Any], int]] = {}
        self.manager = ProgressManager()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for pbar, pos in self.tasks.values():
            pbar.close()
            self.manager.release(pos)
        self.tasks.clear()

    def add_task(
        self, description: str, total: int | None = None, **kwargs: Any
    ) -> int:
        combined_kwargs = self.kwargs.copy()
        combined_kwargs.update(kwargs)
        if "leave" not in combined_kwargs:
            combined_kwargs["leave"] = not self.transient

        pbar, pos = self.manager.get_pbar(
            None, description, total=total, **combined_kwargs
        )
        task_id = id(pbar)
        self.tasks[task_id] = (pbar, pos)
        return task_id

    def update(
        self,
        task_id: int,
        advance: float = 0,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        if task_id in self.tasks:
            pbar, _ = self.tasks[task_id]
            if description:
                pbar.set_description(description)
            if advance:
                pbar.update(advance)

    def reset(self) -> None:
        self.__exit__(None, None, None)
