"""
Rewrite generator/reconstructor constructor calls onto the settings bundles.

For every call to one of TARGETS, keyword arguments that now live in a bundle
are collected into `propagation=Propagation(...)`, `optics=Optics(...)`,
`envelopes=Envelopes(...)`, `camera=Camera(...)`; the needed names are
imported from specter.settings. Works on .py files and on the code cells of
.ipynb files. Prints every rewritten call site so the change can be reviewed.

Usage: bundle_rewriter.py [--dry-run] PATH [PATH ...]
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

TARGETS = {
    "ImageGenerator",
    "ImageGeneratorFromCoordinates",
    "MicrographGenerator",
    "TiltSeriesGenerator",
    "Reconstructor",
    "TomogramReconstructor",
    "Ghostbuster",
    "TomogramGhostbuster",
}
# Which bundles each target accepts.
BUNDLES_FOR = {
    "ImageGenerator": ("propagation", "optics", "envelopes", "camera"),
    "ImageGeneratorFromCoordinates": ("propagation", "optics", "envelopes", "camera"),
    "MicrographGenerator": ("propagation", "optics", "envelopes", "camera"),
    "TiltSeriesGenerator": ("propagation", "optics", "envelopes", "camera"),
    "Reconstructor": ("propagation", "optics"),
    "TomogramReconstructor": ("propagation", "optics"),
    "Ghostbuster": ("propagation", "optics"),
    "TomogramGhostbuster": ("propagation", "optics"),
}
BUNDLE_CLASS = {
    "propagation": "Propagation",
    "optics": "Optics",
    "envelopes": "Envelopes",
    "camera": "Camera",
}
FIELD_TO_BUNDLE = {
    "scattering_model": "propagation",
    "alpha": "propagation",
    "ews_curvature_sign": "propagation",
    "klim": "propagation",
    "pad_fft": "propagation",
    "rotate_mode": "propagation",
    "aberration_backend": "optics",
    "lpp_params": "optics",
    "convergence_angle": "envelopes",
    "cc": "envelopes",
    "energy_spread": "envelopes",
    "deltaV_V": "envelopes",
    "deltaI_I": "envelopes",
    "dose_envelope": "envelopes",
    "detector_model": "camera",
    "noise_model": "camera",
    "n_frames": "camera",
}


def _rewrite_source(src: str, label: str, report: list[str]) -> tuple[str, set[str]]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, set()
    lines = src.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def pos(lineno: int, col: int) -> int:
        return offsets[lineno - 1] + col

    edits: list[tuple[int, int, str]] = []
    used: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (
            fn.id
            if isinstance(fn, ast.Name)
            else fn.attr
            if isinstance(fn, ast.Attribute)
            else None
        )
        if name not in TARGETS:
            continue
        allowed = BUNDLES_FOR[name]
        groups: dict[str, list[ast.keyword]] = {}
        for kw in node.keywords:
            b = FIELD_TO_BUNDLE.get(kw.arg or "")
            if b is not None and b in allowed:
                groups.setdefault(b, []).append(kw)
        if not groups:
            continue
        # Text of each moved keyword's value.
        moved = {kw for g in groups.values() for kw in g}
        parts: list[str] = []
        for kw in node.keywords:
            if kw in moved:
                continue
            value = src[
                pos(kw.value.lineno, kw.value.col_offset) : pos(
                    kw.value.end_lineno, kw.value.end_col_offset
                )
            ]
            parts.append(f"{kw.arg}={value}" if kw.arg else f"**{value}")
        for b in allowed:
            if b not in groups:
                continue
            cls = BUNDLE_CLASS[b]
            used.add(cls)
            inner = ", ".join(
                f"{kw.arg}={src[pos(kw.value.lineno, kw.value.col_offset) : pos(kw.value.end_lineno, kw.value.end_col_offset)]}"
                for kw in groups[b]
            )
            parts.append(f"{b}={cls}({inner})")
        positional = [
            src[pos(a.lineno, a.col_offset) : pos(a.end_lineno, a.end_col_offset)]
            for a in node.args
        ]
        # Multi-line call: one argument per line.
        indent = " " * (node.col_offset + 4)
        close_indent = " " * node.col_offset
        # Find the indentation of the call's own line for nested calls.
        line_start = src.rfind("\n", 0, pos(node.lineno, node.col_offset)) + 1
        base_indent = re.match(r"\s*", src[line_start:]).group(0)
        indent = base_indent + "    "
        close_indent = base_indent
        args_text = ",\n".join(indent + a for a in positional + parts)
        new_call = f"{src[pos(fn.lineno, fn.col_offset) : pos(fn.end_lineno, fn.end_col_offset)]}(\n{args_text},\n{close_indent})"
        edits.append(
            (
                pos(node.lineno, node.col_offset),
                pos(node.end_lineno, node.end_col_offset),
                new_call,
            )
        )
        report.append(
            f"{label}:{node.lineno} {name}(...) -> {', '.join(sorted(groups))}"
        )
    if not edits:
        return src, set()
    # Apply from the end so earlier offsets stay valid; skip nested overlaps.
    edits.sort(key=lambda e: e[0], reverse=True)
    out = src
    last_start = None
    for start, end, text in edits:
        if last_start is not None and end > last_start:
            continue  # nested inside an already-rewritten call
        out = out[:start] + text + out[end:]
        last_start = start
    return out, used


def _add_import(src: str, names: set[str]) -> str:
    if not names:
        return src
    stmt = f"from specter.settings import {', '.join(sorted(names))}\n"
    m = re.search(r"^from specter\.settings import (.+)$", src, flags=re.M)
    if m:
        existing = {n.strip() for n in m.group(1).split(",")}
        merged = sorted(existing | names)
        return src.replace(
            m.group(0), f"from specter.settings import {', '.join(merged)}"
        )
    # After the last top-level import.
    lines = src.split("\n")
    last = None
    depth = 0
    for i, line in enumerate(lines):
        if depth == 0 and re.match(r"^(from|import) ", line):
            last = i
            if line.rstrip().endswith("(") and ")" not in line:
                depth = 1
        elif depth == 1:
            if ")" in line:
                depth = 0
                last = i
    if last is None:
        return stmt + src
    lines.insert(last + 1, stmt.rstrip("\n"))
    return "\n".join(lines)


def rewrite_py(path: Path, dry: bool, report: list[str]) -> None:
    src = path.read_text()
    new, used = _rewrite_source(src, str(path), report)
    if new != src:
        new = _add_import(new, used)
        if not dry:
            path.write_text(new)


def rewrite_ipynb(path: Path, dry: bool, report: list[str]) -> None:
    nb = json.loads(path.read_text())
    changed = False
    all_used: set[str] = set()
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        new, used = _rewrite_source(src, f"{path}#cell{i}", report)
        if new != src:
            cell["source"] = new.splitlines(keepends=True)
            changed = True
            all_used |= used
    if changed and all_used:
        # Put the import in the first code cell that imports specter.
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code" and "specter" in "".join(cell["source"]):
                src = "".join(cell["source"])
                cell["source"] = _add_import(src, all_used).splitlines(keepends=True)
                break
    if changed and not dry:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    args = sys.argv[1:]
    dry = "--dry-run" in args
    paths = [Path(a) for a in args if a != "--dry-run"]
    report: list[str] = []
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        for f in files:
            if ".ipynb_checkpoints" in str(f):
                continue
            if f.suffix == ".py":
                rewrite_py(f, dry, report)
            elif f.suffix == ".ipynb":
                rewrite_ipynb(f, dry, report)
    print("\n".join(report))
    print(f"{len(report)} call sites{' (dry run)' if dry else ''}")


if __name__ == "__main__":
    main()
