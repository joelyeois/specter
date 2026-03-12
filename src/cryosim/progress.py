from __future__ import annotations

import threading
from typing import Any, Iterable, Iterator, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


class ProgressManager:
    """
    Manages progress bar positions and handles notebook-specific stacking issues.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._occupied_positions = set()
        return cls._instance

    def _is_notebook(self) -> bool:
        """Robust notebook detection."""
        try:
            from IPython import get_ipython

            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return True  # Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except (NameError, ImportError):
            return False  # Probably standard Python interpreter

    def _get_next_available_position(self) -> int:
        pos = 0
        while pos in self._occupied_positions:
            pos += 1
        return pos

    def get_pbar(
        self, iterable: Iterable[T], desc: str, **kwargs: Any
    ) -> tuple[tqdm[T], int]:
        with self._lock:
            pos = self._get_next_available_position()
            self._occupied_positions.add(pos)

        is_nb = self._is_notebook()

        # Handle 'transient' (rich) and 'leave' (tqdm)
        transient = kwargs.pop("transient", None)
        if "leave" not in kwargs:
            # Leave the bar only if it's the outermost bar (pos 0) and not transient
            kwargs["leave"] = (not transient) if transient is not None else (pos == 0)

        # In notebooks, manually setting 'position' can cause stacking issues
        # and prevents the bar from clearing correctly in some environments.
        # tqdm.notebook handles its own vertical stacking of widgets.
        if is_nb and "position" not in kwargs:
            pbar = tqdm(iterable, desc=desc, **kwargs)
        else:
            if "position" not in kwargs:
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
    pbar, pos = manager.get_pbar(  # ← unpack pos
        iterable,
        desc=description,
        total=total,
        transient=transient,  # ← let get_pbar handle the mapping, don't duplicate here
        **kwargs,
    )
    try:
        yield from pbar
    finally:
        pbar.close()
        manager.release(pos)  # ← pass pos


class TqdmProgress:
    """
    A context manager wrapper around ProgressManager that mimics rich.progress.Progress.

    Example
    -------
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
    ):
        if task_id in self.tasks:
            pbar, _ = self.tasks[task_id]
            if description:
                pbar.set_description(description)
            if advance:
                pbar.update(advance)

    def reset(self):
        self.__exit__(None, None, None)
