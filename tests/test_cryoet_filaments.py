"""End-to-end smoke test for CryoETSpecimenGenerator's filament support:
fetch a monomer structure, build its PCA-aligned potential template, place
several filament instances, and stamp them into the specimen volume.

Uses the bundled local PDB fixture (not a network fetch of the real
tubulin/actin presets) so this runs fully offline, same convention as
test_cts_specimen.py's _PDB_FIXTURE.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from specter.specimen.cryoet import CryoETSpecimenGenerator

_PDB_FIXTURE = Path(__file__).parent.parent / "pdb-data" / "1mbo.cif"


@pytest.mark.skipif(not _PDB_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_cryoet_generate_stamps_filament_density(tmp_path):
    gen = CryoETSpecimenGenerator(
        protein_specs=[],
        membrane_specs=[],
        filament_specs=[
            {
                "code": str(_PDB_FIXTURE),
                "step": 30.0,
                "flex_deg": 8.0,
                "n_filaments": 2,
                "n_monomers": 4,
            }
        ],
        target_shape=(80, 80, 80),
        target_v_size=6.0,
        scratch_dir=tmp_path,
        verbose=False,
    )
    volume = gen.generate()

    assert volume.shape == (80, 80, 80)
    assert len(gen.filament_instances) == 2 * 4
    assert bool((volume > 0).any())
    # Every monomer instance came from the one species requested.
    assert all(inst.code == str(_PDB_FIXTURE) for inst in gen.filament_instances)


@pytest.mark.skipif(not _PDB_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_cryoet_no_filament_specs_places_nothing(tmp_path):
    gen = CryoETSpecimenGenerator(
        protein_specs=[],
        membrane_specs=[],
        target_shape=(40, 40, 40),
        target_v_size=6.0,
        scratch_dir=tmp_path,
        verbose=False,
    )
    gen.generate()
    assert gen.filament_instances == []


@pytest.mark.skipif(not _PDB_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_cryoet_filament_placement_is_seed_reproducible(tmp_path):
    kwargs = dict(
        protein_specs=[],
        membrane_specs=[],
        filament_specs=[
            {
                "code": str(_PDB_FIXTURE),
                "step": 30.0,
                "flex_deg": 8.0,
                "n_filaments": 1,
                "n_monomers": 3,
            }
        ],
        target_shape=(60, 60, 60),
        target_v_size=6.0,
        seed=123,
        verbose=False,
    )
    vol_a = CryoETSpecimenGenerator(scratch_dir=tmp_path / "a", **kwargs).generate()
    vol_b = CryoETSpecimenGenerator(scratch_dir=tmp_path / "b", **kwargs).generate()
    assert torch.allclose(vol_a, vol_b)


@pytest.mark.skipif(not _PDB_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_cryoet_export_picks_includes_filaments(tmp_path):
    gen = CryoETSpecimenGenerator(
        protein_specs=[],
        membrane_specs=[],
        filament_specs=[
            {
                "code": str(_PDB_FIXTURE),
                "step": 30.0,
                "flex_deg": 8.0,
                "n_filaments": 1,
                "n_monomers": 3,
            }
        ],
        target_shape=(60, 60, 60),
        target_v_size=6.0,
        scratch_dir=tmp_path,
        verbose=False,
    )
    gen.generate()
    written = gen.export_picks(tmp_path / "picks")
    key = str(_PDB_FIXTURE)
    assert key in written
    lines = written[key].read_text().strip().splitlines()
    assert len(lines) == 3
