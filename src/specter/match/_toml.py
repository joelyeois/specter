"""A minimal TOML writer for the flat, scalar-valued configs specter reads back.

The standard library reads TOML (`tomllib`) but does not write it, and the
values a config holds are all scalars, ``None``, or lists of scalars, so a
few lines cover it. Tables group fields the way `configs/*.toml` does;
``None`` fields are written as commented-out keys so the file documents
them without setting them (TOML has no null).
"""

from __future__ import annotations

from typing import Any


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value)) if isinstance(value, float) else str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} to TOML: {value!r}")


def dumps(tables: dict[str, dict[str, Any]], header: str = "") -> str:
    """
    Render ``{table: {key: value}}`` as TOML text.

    Parameters
    ----------
    tables : dict
        Table name -> field name -> value. A ``None`` value is emitted as a
        commented-out key. An empty table name puts its keys at top level.
    header : str, optional
        Comment lines (without the leading ``#``) written at the top.

    Returns
    -------
    str
    """
    out: list[str] = []
    for line in header.splitlines():
        out.append(f"# {line}".rstrip())
    if header:
        out.append("")
    for table, fields in tables.items():
        if table:
            out.append(f"[{table}]")
        for key, value in fields.items():
            if value is None:
                out.append(f"# {key} = (unset)")
            else:
                out.append(f"{key} = {_fmt(value)}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
