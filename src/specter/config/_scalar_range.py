"""The scalar-or-[low, high] field convention, re-exported for the config
dataclasses from `specter._scalar_range`, where `imaging` can import it
without importing the config layer."""

from __future__ import annotations

from .._scalar_range import ScalarOrRange, parse_scalar_or_range

__all__ = ["ScalarOrRange", "parse_scalar_or_range"]
