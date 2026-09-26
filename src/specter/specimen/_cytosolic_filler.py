"""
Reference lists of generic, non-target cytosolic macromolecules, for
filling the "everything else" background of a specimen (crowding
below/around whatever specific species you're annotating as targets).

Two independent, additive tables, both adapted by the same function --
`build_filler_pool_specs` -- to `TomogramSpecimenGenerator`'s
(`specter build tomogram`) flat ``{"pdb_source"}`` filler_specs format.
`specter build tomogram`'s `filler_from_pei2016`/`filler_from_cryoetsim`
both route through this one function regardless of which table they pull
from:

- `PEI2016_CROWDING_TABLE`, which carries the source paper's own
  relative-abundance data (`occurrence_freq`). The species are weighted
  equally by default; `build_filler_pool_specs(abundance_weighted=True)`
  (`filler_pei2016_abundance_weighting` in `specter build tomogram`)
  weights them by that column instead.
- `CRYOETSIM_PARTICLE_TABLE`, which has no abundance column and is
  always weighted equally.

The tables themselves, with their provenance, live in
`specter.specimen._filler_tables` and are re-exported here.
"""

from __future__ import annotations

from ._filler_tables import CRYOETSIM_PARTICLE_TABLE, PEI2016_CROWDING_TABLE

__all__ = [
    "CRYOETSIM_PARTICLE_TABLE",
    "PEI2016_CROWDING_TABLE",
    "build_filler_pool_specs",
]


def build_filler_pool_specs(
    table: list[dict],
    max_mw_kda: float | None = None,
    min_mw_kda: float | None = None,
    categories: list[str] | None = None,
    codes: list[str] | None = None,
    exclude_codes: list[str] | None = None,
    abundance_weighted: bool = False,
) -> list[dict]:
    """
    Filter a filler reference table down to a mass range and/or category,
    and adapt it to `TomogramSpecimenGenerator`'s (`specter build
    tomogram`) flat ``{"pdb_source": ...}`` filler_specs shape -- one
    entry per selected species, weighted implicitly equally (via
    `TomogramProteinSpec.ratio`'s own default) unless `abundance_weighted`
    is set or a caller overrides that afterward.

    Works on any ``list[dict]`` with a ``"code"``/``"mw_kda"`` key per
    entry -- both `CRYOETSIM_PARTICLE_TABLE` and `PEI2016_CROWDING_TABLE`
    (this module) qualify, so the same helper adapts either (or both,
    called twice and concatenated) into one filler_specs list.
    `categories` only has an effect on tables that carry a ``"category"``
    key (`CRYOETSIM_PARTICLE_TABLE`); entries without one are kept
    regardless of this filter.

    Parameters
    ----------
    table : list of dict
        e.g. `CRYOETSIM_PARTICLE_TABLE` or `PEI2016_CROWDING_TABLE`.
    max_mw_kda : float, optional
        Only include entries at or below this mass, kDa.
    min_mw_kda : float, optional
        Only include entries at or above this mass, kDa.
    categories : list of str, optional
        Only include entries whose ``"category"`` is in this list
        (ignored for entries with no ``"category"`` key at all).
    codes : list of str, optional
        Only include these specific codes (must exist in `table`).
        Mutually exclusive with `exclude_codes`.
    exclude_codes : list of str, optional
        Drop these codes from `table`. Mutually exclusive with `codes`.
    abundance_weighted : bool, optional
        Add a ``"ratio"`` to each entry proportional to its
        ``"occurrence_freq"``, normalised to a mean of 1 over the selected
        species. The normalisation keeps the selection's combined weight
        equal to its size, which is what the equal-ratio default gives it,
        so a table mixed with other ratio-mode filler (another table, or
        hand-listed ``[[filler]]`` entries at the default ratio 1) keeps
        the same share of the candidate draw and only the split among its
        own species changes. Requires an ``"occurrence_freq"`` on every
        selected entry (`PEI2016_CROWDING_TABLE` has one;
        `CRYOETSIM_PARTICLE_TABLE` does not). Default False.

    Returns
    -------
    list[dict]
        One ``{"pdb_source": code}`` entry per selected species (plus
        ``"ratio"`` when `abundance_weighted`), ready to concatenate onto
        your own `filler_specs`/`[[filler]]` list.

    Raises
    ------
    ValueError
        If both `codes` and `exclude_codes` are given, a requested code is
        not in `table`, or `abundance_weighted` is set for a selection
        with an entry that has no ``"occurrence_freq"``.

    Notes
    -----
    A ratio is a relative probability per candidate draw, not a placed
    count: the packer attempts candidates largest-first and drops those
    that do not fit, so in a crowded region the placed proportions can
    fall short of the ratios for the larger species.
    """
    if codes is not None and exclude_codes is not None:
        raise ValueError("pass only one of codes / exclude_codes, not both")

    selected = table
    if codes is not None:
        by_code = {e["code"]: e for e in selected}
        missing = [c for c in codes if c not in by_code]
        if missing:
            raise ValueError(f"not in table: {missing}")
        selected = [by_code[c] for c in codes]
    elif exclude_codes is not None:
        selected = [e for e in selected if e["code"] not in exclude_codes]

    if categories is not None:
        selected = [
            e for e in selected if "category" not in e or e["category"] in categories
        ]
    if max_mw_kda is not None:
        selected = [e for e in selected if e["mw_kda"] <= max_mw_kda]
    if min_mw_kda is not None:
        selected = [e for e in selected if e["mw_kda"] >= min_mw_kda]

    if not abundance_weighted:
        return [{"pdb_source": e["code"]} for e in selected]
    missing_freq = [e["code"] for e in selected if "occurrence_freq" not in e]
    if missing_freq:
        raise ValueError(
            "abundance weighting needs an 'occurrence_freq' on every entry; "
            f"missing for {missing_freq}"
        )
    if not selected:
        return []
    mean_freq = sum(float(e["occurrence_freq"]) for e in selected) / len(selected)
    return [
        {"pdb_source": e["code"], "ratio": float(e["occurrence_freq"]) / mean_freq}
        for e in selected
    ]
