"""
Smoke tests for MembraneTomogramGenerator (specter.specimen.tomogram) --
the pipeline combining MembraneGenerator with region-gated (cytosol/lumen)
dense protein packing. Uses locally-cached PDB fixtures (no network fetch).
Membrane/box parameters here are deliberately small/fast and tuned (see
comments) to produce a lumen big enough to actually hold the small test
species -- correctness of the region-gating is the point, not realism.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from specter.specimen.membrane import MembraneGenerator
from specter.specimen.tomogram import MembraneTomogramGenerator, TomogramProteinSpec

_SMALL_FIXTURE = Path(__file__).parent.parent / "pdb-data" / "1mbo.cif"
_LARGE_FIXTURE = Path(__file__).parent.parent / "pdb-data" / "1bxn-assembly1.cif"

# Two blended metaball sources at this scale enclose a lumen with equivalent
# spherical radius ~48A -- comfortably bigger than 1mbo's own ~31.4A radius
# (verified directly; smaller configs leave no room for even one instance).
_MEMBRANE_KWARGS = dict(
    target_shape_zyx=(64, 64, 64),
    v_size=8.0,
    n_sources=2,
    radius_range_a=(60.0, 80.0),
    spread_a=5.0,
    noise_amplitude_a=0.0,
    curvature_iterations=5,
    n_lipids_per_leaflet=6,
)


@pytest.mark.skipif(
    not (_SMALL_FIXTURE.exists() and _LARGE_FIXTURE.exists()),
    reason="bundled PDB fixtures missing",
)
def test_membrane_tomogram_generator_places_both_locations_correctly():
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = MembraneTomogramGenerator(
        membrane_generator=mgen,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE), location="lumen"),
            TomogramProteinSpec(pdb_source=str(_LARGE_FIXTURE), location="cytosol"),
        ],
        occupancy_fraction=0.1,
        gap_angstrom=5.0,
        pdb_cache_dir="pdb-data/",
        seed=0,
    )
    volume = gen.generate()

    assert volume.shape == _MEMBRANE_KWARGS["target_shape_zyx"]
    assert torch.isfinite(volume).all()
    assert volume.max() > 0

    assert gen.regions is not None
    by_location: dict[str, list] = {}
    for p in gen.placements:
        by_location.setdefault(p.location, []).append(p)
    assert len(by_location.get("cytosol", [])) > 0
    assert len(by_location.get("lumen", [])) > 0

    v_size = mgen.v_size
    shape_zyx = mgen.target_shape_zyx
    center_zyx = torch.tensor([shape_zyx[0] / 2, shape_zyx[1] / 2, shape_zyx[2] / 2])

    def _voxel_index(position_xyz: torch.Tensor) -> tuple[int, int, int]:
        idx_xyz = (position_xyz / v_size) + center_zyx[[2, 1, 0]]
        ix, iy, iz = idx_xyz.round().long().tolist()
        return iz, iy, ix

    for p in by_location["lumen"]:
        iz, iy, ix = _voxel_index(p.position_xyz)
        assert bool(gen.regions["lumen"][iz, iy, ix])
    for p in by_location["cytosol"]:
        iz, iy, ix = _voxel_index(p.position_xyz)
        assert bool(gen.regions["cytosol"][iz, iy, ix])


@pytest.mark.skipif(not _SMALL_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_membrane_tomogram_generator_instance_labels_match_placements():
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = MembraneTomogramGenerator(
        membrane_generator=mgen,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE), location="cytosol")
        ],
        occupancy_fraction=0.05,
        gap_angstrom=5.0,
        pdb_cache_dir="pdb-data/",
        seed=0,
    )
    gen.generate()

    assert len(gen.placements) > 0
    labels = gen.instance_labels
    assert labels is not None
    assert labels.dtype == torch.int32
    placement_ids = {p.instance_id for p in gen.placements}
    assert len(placement_ids) == len(gen.placements)
    present_ids = set(torch.unique(labels[labels > 0]).tolist())
    assert present_ids == placement_ids


@pytest.mark.skipif(not _SMALL_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_membrane_tomogram_generator_warns_when_lumen_species_has_no_region():
    # tiny/flat membrane config unlikely to enclose any lumen at all
    mgen = MembraneGenerator(
        target_shape_zyx=(24, 24, 24),
        v_size=8.0,
        n_sources=1,
        radius_range_a=(200.0, 200.0),  # much larger than the grid -> no closed shell
        spread_a=0.0,
        noise_amplitude_a=0.0,
        curvature_iterations=2,
        n_lipids_per_leaflet=6,
        seed=0,
    )
    gen = MembraneTomogramGenerator(
        membrane_generator=mgen,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE), location="lumen")
        ],
        occupancy_fraction=0.1,
        pdb_cache_dir="pdb-data/",
        seed=0,
    )
    with pytest.warns(UserWarning, match="no 'lumen' region"):
        gen.generate()
    assert len(gen.placements) == 0


def test_membrane_tomogram_generator_rejects_empty_protein_specs():
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    with pytest.raises(ValueError, match="protein_specs"):
        MembraneTomogramGenerator(membrane_generator=mgen, protein_specs=[])


@pytest.mark.skipif(
    not (_SMALL_FIXTURE.exists() and _LARGE_FIXTURE.exists()),
    reason="bundled PDB fixtures missing",
)
def test_membrane_tomogram_generator_export_picks(tmp_path):
    """export_picks: coordinate conversion (corner-relative, not centered)
    against a known placement, and transmembrane species get their own
    suffixed file distinct from cytosol/lumen files."""
    from specter.specimen.membrane import TransmembraneSpec

    mgen = MembraneGenerator(
        transmembrane_specs=[
            TransmembraneSpec(pdb_source=str(_SMALL_FIXTURE), frequency=1)
        ],
        seed=0,
        **_MEMBRANE_KWARGS,
    )
    gen = MembraneTomogramGenerator(
        membrane_generator=mgen,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_LARGE_FIXTURE), location="cytosol"),
        ],
        occupancy_fraction=0.1,
        gap_angstrom=5.0,
        pdb_cache_dir="pdb-data/",
        seed=0,
    )
    gen.generate()
    assert len(gen.placements) > 0
    assert len(gen.transmembrane_placements) > 0

    written = gen.export_picks(tmp_path, annotation_version="2.0")
    cytosol_key = next(k for k in written if k.endswith("-cytosol"))
    transmembrane_key = next(k for k in written if k.endswith("-transmembrane"))
    assert cytosol_key != transmembrane_key
    assert written[cytosol_key].name.endswith("-2.0_orientedpoint.ndjson")
    assert written[transmembrane_key].exists()

    v_size = mgen.v_size
    shape_zyx = mgen.target_shape_zyx
    extent_xyz = (
        torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=torch.float32)
        * v_size
    )

    placed = gen.placements[0]
    lines = written[cytosol_key].read_text().strip().splitlines()
    # Match this placement's own row by its corner-relative x coordinate
    # (each placement gets a distinct enough position at this scale).
    import json

    expected_corner = placed.position_xyz + extent_xyz / 2
    rows = [json.loads(line) for line in lines]
    match = next(
        r for r in rows if abs(r["location"]["x"] - float(expected_corner[0])) < 1e-3
    )
    assert abs(match["location"]["y"] - float(expected_corner[1])) < 1e-3
    assert abs(match["location"]["z"] - float(expected_corner[2])) < 1e-3
    assert "xyz_rotation_matrix" in match
