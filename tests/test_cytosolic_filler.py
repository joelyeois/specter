"""
Smoke tests for specter.specimen.cytosolic_filler -- table integrity
(no duplicate/malformed codes) and build_filler_pool_specs's filtering,
including cross-compatibility between its two bundled tables
(CRYOETSIM_PARTICLE_TABLE and PEI2016_CROWDING_TABLE). No network access
needed (pure dict filtering, no PDB fetch).
"""

from __future__ import annotations

from collections import Counter

from specter.specimen.cytosolic_filler import (
    CRYOETSIM_PARTICLE_TABLE,
    PEI2016_CROWDING_TABLE,
    build_filler_pool_specs,
)


def test_cryoetsim_particle_table_entries_are_well_formed():
    assert len(CRYOETSIM_PARTICLE_TABLE) > 100

    codes = [e["code"] for e in CRYOETSIM_PARTICLE_TABLE]
    dupes = [c for c, n in Counter(codes).items() if n > 1]
    assert dupes == []
    assert all(len(c) == 4 for c in codes)

    assert all(e["mw_kda"] > 0 for e in CRYOETSIM_PARTICLE_TABLE)

    categories = {e["category"] for e in CRYOETSIM_PARTICLE_TABLE}
    assert categories == {
        "macromolecules",
        "distractors",
        "transcription_translation",
        "nucleosomes",
    }


def test_build_filler_pool_specs_filters_by_mass():
    specs = build_filler_pool_specs(CRYOETSIM_PARTICLE_TABLE, max_mw_kda=20.0)
    assert {d["pdb_source"] for d in specs} == {"7ELY", "7BLG", "1EXR", "1QTX"}

    specs = build_filler_pool_specs(CRYOETSIM_PARTICLE_TABLE, min_mw_kda=4000.0)
    returned_codes = {d["pdb_source"] for d in specs}
    expected_codes = {
        e["code"] for e in CRYOETSIM_PARTICLE_TABLE if e["mw_kda"] >= 4000.0
    }
    assert returned_codes == expected_codes
    assert len(returned_codes) == 5  # sanity check against silent table edits


def test_build_filler_pool_specs_filters_by_category():
    specs = build_filler_pool_specs(
        CRYOETSIM_PARTICLE_TABLE, categories=["distractors"]
    )
    assert len(specs) == 5
    assert {"pdb_source": "1EXR"} in specs
    assert {"pdb_source": "7ELY"} not in specs  # macromolecules, excluded


def test_build_filler_pool_specs_codes_and_exclude_codes_are_mutually_exclusive():
    import pytest

    with pytest.raises(ValueError, match="only one"):
        build_filler_pool_specs(
            CRYOETSIM_PARTICLE_TABLE, codes=["1BXN"], exclude_codes=["1BXN"]
        )


def test_build_filler_pool_specs_rejects_unknown_code():
    import pytest

    with pytest.raises(ValueError, match="not in table"):
        build_filler_pool_specs(CRYOETSIM_PARTICLE_TABLE, codes=["ZZZZ"])


def test_build_filler_pool_specs_works_on_pei2016_table_too():
    """PEI2016_CROWDING_TABLE has no "category" key -- categories filter
    must be a no-op there rather than dropping everything."""
    specs = build_filler_pool_specs(PEI2016_CROWDING_TABLE, categories=["distractors"])
    assert len(specs) == len(PEI2016_CROWDING_TABLE)

    specs = build_filler_pool_specs(PEI2016_CROWDING_TABLE, max_mw_kda=100.0)
    assert 0 < len(specs) < len(PEI2016_CROWDING_TABLE)
    assert all(
        d["pdb_source"] in {e["code"] for e in PEI2016_CROWDING_TABLE} for d in specs
    )
