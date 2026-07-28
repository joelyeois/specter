"""Smoke test for CryoETSpecimenGenerator's parallel low-res placement:
membrane placement (polnet) and protein SVOL building (specter's
PotentialBuilder) must overlap in wall-clock time, and protein packing must
only start once both have finished. Uses mocks throughout -- no real PDB
fetch, PotentialBuilder, or polnet placement -- so this only tests the
threading/dependency wiring in _run_low_res_placement, not the physics.
"""

from __future__ import annotations

import time

from specter.specimen import cryoet
from specter.specimen import polnet_bridge as pb

SLEEP_S = 0.2


def test_membrane_and_svol_building_run_concurrently(tmp_path, monkeypatch):
    gen = cryoet.CryoETSpecimenGenerator(
        protein_specs=[{"PDB_CODE": "1bxn"}],
        membrane_specs=[
            {
                "MB_TYPE": "sphere",
                "MB_THICK_RG": (35, 45),
                "MB_LAYER_S_RG": (8, 12),
                "MB_OCC_RG": (0.001, 0.003),
                "MB_MIN_RAD": 150.0,
                "MB_MAX_RAD": 300.0,
            }
        ],
        target_shape=(32, 32, 32),
        target_v_size=2.0,
        scratch_dir=tmp_path,
        verbose=False,
    )

    intervals: dict[str, tuple[float, float]] = {}
    dummy_sample = object()

    def fake_build_protein_svols(self, scratch):
        start = time.monotonic()
        time.sleep(SLEEP_S)
        intervals["svol"] = (start, time.monotonic())
        return [{"MMER_ID": "1bxn"}]

    def fake_build_membranes(**kwargs):
        start = time.monotonic()
        time.sleep(SLEEP_S)
        intervals["membranes"] = (start, time.monotonic())
        return dummy_sample

    packing_call = {}

    def fake_add_proteins(sample, protein_param_dicts, scratch, **kwargs):
        packing_call["start"] = time.monotonic()
        packing_call["sample"] = sample
        packing_call["protein_param_dicts"] = protein_param_dicts

    monkeypatch.setattr(
        cryoet.CryoETSpecimenGenerator, "_build_protein_svols", fake_build_protein_svols
    )
    monkeypatch.setattr(pb, "build_low_res_sample_with_membranes", fake_build_membranes)
    monkeypatch.setattr(pb, "add_proteins_to_sample", fake_add_proteins)
    monkeypatch.setattr(pb, "get_protein_instances", lambda sample: [])
    monkeypatch.setattr(pb, "fit_membrane_instances", lambda *a, **k: [])

    protein_instances, membrane_instances = gen._run_low_res_placement()

    svol_start, svol_end = intervals["svol"]
    mb_start, mb_end = intervals["membranes"]
    # Genuine concurrency: each interval starts before the other finishes.
    assert svol_start < mb_end
    assert mb_start < svol_end

    # Protein packing only starts once both finish, and gets both results.
    assert packing_call["start"] >= svol_end
    assert packing_call["start"] >= mb_end
    assert packing_call["sample"] is dummy_sample
    assert packing_call["protein_param_dicts"] == [{"MMER_ID": "1bxn"}]

    assert protein_instances == []
    assert membrane_instances == []
