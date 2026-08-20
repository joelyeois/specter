from __future__ import annotations

import json
import sys
from typing import Any

import rich_click as click
from rich.console import Console
from rich.table import Table

from ._database import JobDatabase

console = Console()

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

_BASE_DIR_OPTION = click.option(
    "--base-dir", default=None, help="Override SPECTER_JOBS_DIR"
)
_PROJECT_OPTION = click.option(
    "--project", default=None, help="Project name (omit for a no-project job)"
)


def _short_params(
    params: dict[str, Any], keys: list[str] | None = None, max_items: int = 4
) -> str:
    """Return a compact one-line summary of scalar params.

    Parameters
    ----------
    params : dict
        Full params dict from a job record.
    keys : list of str, optional
        Exact keys to show, in order (from ``--show``). A missing key
        prints as ``key=-`` rather than being silently dropped, so a typo'd
        or job-type-specific field name is still visible in the table.
        Overrides the default "first N scalars" behaviour below, which has
        no way to know which fields matter for a given job type -- a
        result logged after training (e.g. a computed resolution) sorts
        wherever `dict` insertion order put it, not necessarily first.
    max_items : int
        Maximum number of scalar items to include when ``keys`` isn't given.

    Returns
    -------
    str
        A space-separated string of ``key=value`` pairs.
    """
    if keys is not None:
        return "  ".join(f"{k}={params.get(k, '-')}" for k in keys)
    scalars = {
        k: v
        for k, v in params.items()
        if isinstance(v, (int, float, str, bool, type(None)))
    }
    items = list(scalars.items())[:max_items]
    return "  ".join(f"{k}={v}" for k, v in items)


def build_jobs_group(name: str = "jobs") -> click.RichGroup:
    """
    Build the `jobs` command group and its subcommands.

    Parameters
    ----------
    name : str
        Name to register the group under.

    Returns
    -------
    click.RichGroup
        The assembled `jobs` command group, ready to attach to a parent
        group via `add_command`.
    """

    @click.group(name=name, cls=click.RichGroup, context_settings=CONTEXT_SETTINGS)
    def jobs() -> None:
        """Inspect and compare tracked SPECTER jobs."""

    @jobs.command("list")
    @click.option("--project", default=None, help="Filter by project name")
    @click.option(
        "--show",
        default=None,
        help="Comma-separated param keys to display in the Params column, "
        "e.g. resolution_gold_standard,lr (default: first 4 scalar params, "
        "in whatever order they were logged)",
    )
    @_BASE_DIR_OPTION
    def list_cmd(project: str | None, show: str | None, base_dir: str | None) -> None:
        """List all jobs."""
        db = JobDatabase(base_dir=base_dir)
        entries = db.list(project=project or None)
        if not entries:
            console.print("[yellow]No jobs found.[/yellow]")
            return
        show_keys = show.split(",") if show else None
        table = Table(show_header=True, header_style="bold")
        table.add_column("Project")
        table.add_column("ID")
        table.add_column("Type")
        table.add_column("Status")
        table.add_column("Created")
        # "fold" instead of the Table default "ellipsis": a key=value pair with
        # no internal spaces (e.g. resolution_gold_standard=3.2 A) is one
        # unbreakable token to a narrow terminal, and ellipsis truncation would
        # silently cut off exactly the value someone asked --show for.
        table.add_column("Params", overflow="fold")
        for entry in entries:
            created = entry.get("created_at", "")[:16].replace("T", " ")
            status = entry.get("status", "")
            color = {"complete": "green", "running": "yellow", "failed": "red"}.get(
                status, "white"
            )
            table.add_row(
                entry.get("project", ""),
                entry.get("id", ""),
                entry.get("type", ""),
                f"[{color}]{status}[/{color}]",
                created,
                _short_params(entry.get("params", {}), keys=show_keys),
            )
        console.print(table)

    @jobs.command("show")
    @click.argument("job_id")
    @_PROJECT_OPTION
    @_BASE_DIR_OPTION
    def show_cmd(job_id: str, project: str | None, base_dir: str | None) -> None:
        """Show full details for a job."""
        db = JobDatabase(base_dir=base_dir)
        try:
            entry = db.get(project, job_id)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        console.print_json(json.dumps(entry, indent=2))

    @jobs.command("diff")
    @click.argument("job_id_a")
    @click.argument("job_id_b")
    @_PROJECT_OPTION
    @_BASE_DIR_OPTION
    def diff_cmd(
        job_id_a: str, job_id_b: str, project: str | None, base_dir: str | None
    ) -> None:
        """Diff params of two jobs."""
        db = JobDatabase(base_dir=base_dir)
        try:
            diff = db.diff(project, job_id_a, job_id_b)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        if not diff:
            console.print("[green]Jobs are identical.[/green]")
            return
        table = Table(show_header=True, header_style="bold")
        table.add_column("Key")
        table.add_column(job_id_a)
        table.add_column(job_id_b)
        for key, (val_a, val_b) in sorted(diff.items()):
            table.add_row(key, str(val_a), str(val_b))
        console.print(table)

    return jobs


def main() -> None:
    """Entry point for `python -m specter.jobs._cli`."""
    build_jobs_group()(prog_name="specter jobs")


if __name__ == "__main__":
    main()
