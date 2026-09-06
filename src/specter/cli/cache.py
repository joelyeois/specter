"""`specter cache` -- inspect and clear the downloaded-structure cache.

The cache lives outside any project (``~/.cache/specter/pdb`` by default, see
`specter.config.default_pdb_cache_dir`), which makes it shared and
deduplicated but not something a user stumbles across while browsing their
files. A tool that keeps a cache out of sight owes the user a way to ask
where it is and to clear it, the way ``uv cache dir``/``uv cache clean`` and
``pip cache dir``/``pip cache purge`` do.

Safe by construction: only downloads land in this directory. A structure the
user supplies by path is read where it lies and never copied here (see
`specter.pdb.PDB`), so ``clean`` can never delete something irreplaceable.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import rich_click as click
from rich.console import Console

from specter.config import default_pdb_cache_dir

from ._click_options import CONTEXT_SETTINGS

console = Console()


def _cache_contents(cache_dir: Path) -> tuple[int, int]:
    """Return ``(file_count, total_bytes)`` for a cache directory."""
    n_files = 0
    total = 0
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
            n_files += 1
    return n_files, total


def _format_size(n_bytes: int) -> str:
    """Human-readable byte count, e.g. ``1.6 GB``."""
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def build_cache_group() -> click.RichGroup:
    """Build the `specter cache` command group."""

    @click.group(name="cache", context_settings=CONTEXT_SETTINGS)
    def cache() -> None:
        """Inspect and clear the cache of downloaded PDB/mmCIF structures.

        Holds only structures fetched by accession code. Files you supply by
        path are read in place and never cached, so clearing this is always
        safe -- everything in it can be re-downloaded.

        Override the location with $SPECTER_PDB_CACHE, or move every
        XDG-aware tool's cache at once with $XDG_CACHE_HOME.
        """

    @cache.command(name="dir")
    def dir_cmd() -> None:
        """Print the cache directory."""
        click.echo(default_pdb_cache_dir())

    @cache.command(name="info")
    def info_cmd() -> None:
        """Show the cache directory, file count and total size."""
        cache_dir = Path(default_pdb_cache_dir())
        console.print(f"[bold]Location:[/bold] {cache_dir}")
        if not cache_dir.is_dir():
            console.print("[dim]Empty -- nothing downloaded yet.[/dim]")
            return
        n_files, total = _cache_contents(cache_dir)
        console.print(f"[bold]Structures:[/bold] {n_files}")
        console.print(f"[bold]Size:[/bold] {_format_size(total)}")

    @cache.command(name="clean")
    @click.option(
        "--yes",
        "-y",
        is_flag=True,
        default=False,
        help="Delete without confirming (for scripts and CI).",
    )
    def clean_cmd(yes: bool) -> None:
        """Delete every cached structure."""
        cache_dir = Path(default_pdb_cache_dir())
        if not cache_dir.is_dir():
            console.print(f"Nothing to clean -- {cache_dir} does not exist.")
            return
        n_files, total = _cache_contents(cache_dir)
        if not yes:
            click.confirm(
                f"Delete {n_files} cached structure(s) ({_format_size(total)}) "
                f"from {cache_dir}?",
                abort=True,
            )
        shutil.rmtree(cache_dir)
        console.print(
            f"[green]✓[/green] Removed {n_files} structure(s), "
            f"{_format_size(total)} freed."
        )

    return cache  # type: ignore[return-value]
