from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from rich.console import Console
from rich.table import Table

from ._database import JobDatabase

console = Console()


def _short_params(params: dict[str, Any], max_items: int = 4) -> str:
    """Return a compact one-line summary of scalar params only.

    Parameters
    ----------
    params : dict
        Full params dict from a job record.
    max_items : int
        Maximum number of scalar items to include.

    Returns
    -------
    str
        A space-separated string of ``key=value`` pairs.
    """
    scalars = {
        k: v
        for k, v in params.items()
        if isinstance(v, (int, float, str, bool, type(None)))
    }
    items = list(scalars.items())[:max_items]
    return "  ".join(f"{k}={v}" for k, v in items)


def cmd_list(args: argparse.Namespace) -> None:
    """Handle the ``list`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    db = JobDatabase(base_dir=args.base_dir)
    jobs = db.list(project=args.project or None)
    if not jobs:
        console.print("[yellow]No jobs found.[/yellow]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Project")
    table.add_column("ID")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Params")
    for entry in jobs:
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
            _short_params(entry.get("params", {})),
        )
    console.print(table)


def cmd_show(args: argparse.Namespace) -> None:
    """Handle the ``show`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    db = JobDatabase(base_dir=args.base_dir)
    try:
        entry = db.get(args.project, args.job_id)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print_json(json.dumps(entry, indent=2))


def cmd_diff(args: argparse.Namespace) -> None:
    """Handle the ``diff`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    db = JobDatabase(base_dir=args.base_dir)
    try:
        diff = db.diff(args.project, args.job_id_a, args.job_id_b)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    if not diff:
        console.print("[green]Jobs are identical.[/green]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Key")
    table.add_column(args.job_id_a)
    table.add_column(args.job_id_b)
    for key, (val_a, val_b) in sorted(diff.items()):
        table.add_row(key, str(val_a), str(val_b))
    console.print(table)


def _add_base_dir(p: argparse.ArgumentParser) -> None:
    """Add the ``--base-dir`` option to a (sub)parser.

    Parameters
    ----------
    p : argparse.ArgumentParser
        Parser to augment.
    """
    p.add_argument("--base-dir", default=None, help="Override SPECTER_JOBS_DIR")


def main() -> None:
    """Entry point for the specter-jobs CLI."""
    parser = argparse.ArgumentParser(
        prog="specter-jobs", description="SPECTER job manager CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List all jobs")
    p_list.add_argument("--project", default=None, help="Filter by project name")
    _add_base_dir(p_list)

    p_show = sub.add_parser("show", help="Show full details for a job")
    p_show.add_argument("job_id")
    p_show.add_argument(
        "--project", default=None, help="Project name (omit for a no-project job)"
    )
    _add_base_dir(p_show)

    p_diff = sub.add_parser("diff", help="Diff params of two jobs")
    p_diff.add_argument("job_id_a")
    p_diff.add_argument("job_id_b")
    p_diff.add_argument(
        "--project", default=None, help="Project name (omit for a no-project job)"
    )
    _add_base_dir(p_diff)

    args = parser.parse_args()
    {"list": cmd_list, "show": cmd_show, "diff": cmd_diff}[args.command](args)


if __name__ == "__main__":
    main()
