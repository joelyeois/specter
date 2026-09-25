"""
Smoke tests for specter.specimen -- table integrity
(no duplicate/malformed codes) and build_filler_pool_specs's filtering,
including cross-compatibility between its two bundled tables
(CRYOETSIM_PARTICLE_TABLE and PEI2016_CROWDING_TABLE). No network access
needed (pure dict filtering, no PDB fetch).
"""

from __future__ import annotations

from collections import Counter

from specter.specimen import (
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


def test_build_filler_pool_specs_is_equal_ratio_by_default():
    """No ratio key by default, so every species takes the spec's ratio 1."""
    specs = build_filler_pool_specs(PEI2016_CROWDING_TABLE)
    assert specs == [{"pdb_source": e["code"]} for e in PEI2016_CROWDING_TABLE]


def test_build_filler_pool_specs_abundance_weighting_follows_occurrence_freq():
    """
    Weighted ratios are proportional to occurrence_freq and average 1 over
    the selection, so the table's combined weight equals its equal-ratio
    total and only the split among its own species changes.
    """
    specs = build_filler_pool_specs(
        PEI2016_CROWDING_TABLE, max_mw_kda=500.0, abundance_weighted=True
    )
    selected = [e for e in PEI2016_CROWDING_TABLE if e["mw_kda"] <= 500.0]
    assert [s["pdb_source"] for s in specs] == [e["code"] for e in selected]
    ratios = [s["ratio"] for s in specs]
    assert abs(sum(ratios) / len(ratios) - 1.0) < 1e-12
    freqs = [e["occurrence_freq"] for e in selected]
    for r, f in zip(ratios, freqs):
        assert abs(r / ratios[0] - f / freqs[0]) < 1e-12


def test_build_filler_pool_specs_abundance_weighting_needs_the_column():
    import pytest

    with pytest.raises(ValueError, match="occurrence_freq"):
        build_filler_pool_specs(CRYOETSIM_PARTICLE_TABLE, abundance_weighted=True)


def test_tomogram_config_abundance_weighting_reaches_the_protein_specs():
    """
    The config switch sets each PEI2016 spec's ratio; off, every ratio is 1,
    and a hand-listed [[filler]] entry keeps its own ratio either way.
    """
    from specter.config import TomogramConfig
    from specter.pipelines import build_tomogram_generator

    def ratios(weighted: bool) -> dict[str, float]:
        cfg = TomogramConfig(
            target_shape=[16, 16, 16],
            voxel_size=10.0,
            filler=[{"pdb_source": "1fa2"}],
            filler_from_pei2016=True,
            filler_pei2016_abundance_weighting=weighted,
            device="cpu",
        )
        return {
            s.pdb_source: s.ratio for s in build_tomogram_generator(cfg).protein_specs
        }

    plain, weighted = ratios(False), ratios(True)
    assert set(plain.values()) == {1.0}
    freq = {e["code"]: e["occurrence_freq"] for e in PEI2016_CROWDING_TABLE}
    mean = sum(freq.values()) / len(freq)
    for code, f in freq.items():
        assert abs(weighted[code] - f / mean) < 1e-9
    assert weighted["1fa2"] == 1.0
