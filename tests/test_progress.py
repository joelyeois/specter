"""
Tests for `specter.progress`'s output helpers.
"""

from __future__ import annotations

import contextlib
import io
import warnings

from specter.progress import TqdmProgress, tqdm_warnings


def test_tqdm_warnings_drops_the_location_prefix_when_asked():
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        with tqdm_warnings(show_location=False):
            warnings.warn("only 3/6 instances fit", UserWarning)
    assert buf.getvalue().strip() == "UserWarning: only 3/6 instances fit"


def test_tqdm_warnings_keeps_the_location_prefix_by_default():
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        with tqdm_warnings():
            warnings.warn("something to debug", UserWarning)
    text = buf.getvalue()
    assert "something to debug" in text
    assert __file__.split("/")[-1] in text


def test_tqdm_warnings_does_not_land_on_a_live_progress_bar():
    """Regression test: a warning raised while a bar was live was written
    straight to stderr by Python's own handler, so it began mid-bar-line --
    ``...0/3 [00:00<?, ?it/s]/path/to/generator.py:764: UserWarning: ...`` --
    leaving both unreadable. Routed through `tqdm.write`, the bar is cleared
    first and the warning starts its own line.
    """
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        with tqdm_warnings(show_location=False):
            with TqdmProgress() as progress:
                task = progress.add_task("Generating", total=3)
                progress.update(task, advance=1)
                warnings.warn("an instance was dropped", UserWarning)
                progress.update(task, advance=1)

    # tqdm redraws with carriage returns; the warning must begin a line of
    # its own rather than being appended to a bar segment.
    lines = buf.getvalue().replace("\r", "\n").split("\n")
    assert any(
        line.startswith("UserWarning: an instance was dropped") for line in lines
    ), buf.getvalue()


def test_tqdm_warnings_restores_the_previous_handler():
    before = warnings.showwarning
    with tqdm_warnings():
        assert warnings.showwarning is not before
    assert warnings.showwarning is before


def test_tqdm_warnings_leaves_filtering_alone():
    """Only the output path is replaced -- `simplefilter` still applies, so
    a suppressed warning stays suppressed and an escalated one still raises.
    """
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        with tqdm_warnings(show_location=False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.warn("suppressed", UserWarning)
    assert buf.getvalue() == ""

    with tqdm_warnings(show_location=False):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                warnings.warn("escalated", UserWarning)
            except UserWarning:
                pass
            else:  # pragma: no cover - only on a regression
                raise AssertionError("simplefilter('error') was not honoured")


def test_tqdm_warnings_honours_an_explicit_file_redirect():
    """A caller passing `file=` targets one stream on purpose; hijacking it
    to stderr would silently move their output. Such a warning is handed
    back to whatever handler was in place instead."""
    elsewhere = io.StringIO()
    stderr = io.StringIO()
    seen = []

    def recording(message, category, filename, lineno, file=None, line=None):
        seen.append(file)

    original = warnings.showwarning
    warnings.showwarning = recording
    try:
        with contextlib.redirect_stderr(stderr):
            with tqdm_warnings(show_location=False):
                warnings.showwarning(
                    "explicitly redirected", UserWarning, __file__, 1, elsewhere, None
                )
    finally:
        warnings.showwarning = original

    assert seen == [elsewhere]
    assert stderr.getvalue() == ""


def test_cli_main_routes_warnings_through_tqdm():
    """The CLI is the front end whose bars the un-routed handler was tearing
    up, so it has to install this -- and without the location prefix, which
    at a command line names a fixed internal call site rather than anything
    about the user's input."""
    import specter.cli._cli as cli_module

    seen = {}

    def fake_cli(**kwargs):
        seen["handler"] = warnings.showwarning
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            warnings.warn("from inside the command", UserWarning)
        seen["text"] = buf.getvalue()

    original_cli = cli_module.cli
    cli_module.cli = fake_cli
    try:
        cli_module.main()
    finally:
        cli_module.cli = original_cli

    assert seen["handler"] is not warnings.showwarning  # restored on exit
    assert seen["text"].strip() == "UserWarning: from inside the command"
