"""
Smoke tests for MembraneTomogramGenerator (specter.specimen.tomogram) --
the pipeline combining one or more MembraneGenerator instances (composited)
with region-gated (cytosol/lumen) dense protein packing. Uses locally-cached
PDB fixtures (no network fetch). Membrane/box parameters here are
deliberately small/fast and tuned (see comments) to produce a lumen big
enough to actually hold the small test species -- correctness of the
region-gating is the point, not realism.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch

from specter.specimen.filament import FilamentSpec
from specter.specimen.membrane import MembraneGenerator, TransmembraneSpec
from specter.specimen.tomogram import (
    MembraneInstance,
    MembraneTomogramGenerator,
    TomogramProteinSpec,
)

_SMALL_FIXTURE = Path(__file__).parent.parent / "pdb-data" / "1mbo.cif"
_LARGE_FIXTURE = Path(__file__).parent.parent / "pdb-data" / "1bxn-assembly1.cif"

_TARGET_SHAPE_ZYX = (64, 64, 64)
_V_SIZE = 8.0

# A spherical_harmonics ellipsoid at this scale encloses a lumen with
# equivalent spherical radius ~59A -- comfortably bigger than 1mbo's own
# ~31.4A radius (verified directly; smaller configs leave no room for even
# one instance).
_MEMBRANE_KWARGS = dict(
    target_shape_zyx=_TARGET_SHAPE_ZYX,
    v_size=_V_SIZE,
    sh_axes_a=(70.0, 70.0, 70.0),
    sh_amplitude=0.15,
    n_lipids_per_leaflet=6,
)


@pytest.mark.skipif(
    not (_SMALL_FIXTURE.exists() and _LARGE_FIXTURE.exists()),
    reason="bundled PDB fixtures missing",
)
def test_membrane_tomogram_generator_places_both_locations_correctly():
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = MembraneTomogramGenerator(
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape_zyx=_TARGET_SHAPE_ZYX,
        v_size=_V_SIZE,
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

    assert volume.shape == _TARGET_SHAPE_ZYX
    assert torch.isfinite(volume).all()
    assert volume.max() > 0

    assert gen.regions is not None
    by_location: dict[str, list] = {}
    for p in gen.placements:
        by_location.setdefault(p.location, []).append(p)
    assert len(by_location.get("cytosol", [])) > 0
    assert len(by_location.get("lumen", [])) > 0

    v_size = _V_SIZE
    shape_zyx = _TARGET_SHAPE_ZYX
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
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape_zyx=_TARGET_SHAPE_ZYX,
        v_size=_V_SIZE,
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
    # tiny/flat membrane config unlikely to enclose any lumen at all --
    # deliberately uses the deprecated "metaball" backend for its
    # oversized-radius-clips-the-grid trick (a single sphere source larger
    # than the grid never closes into a shell); the point of this test is
    # the degenerate-geometry warning path, not backend choice.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mgen = MembraneGenerator(
            target_shape_zyx=(24, 24, 24),
            v_size=8.0,
            shape_backend="metaball",
            n_sources=1,
            radius_range_a=(
                200.0,
                200.0,
            ),  # much larger than the grid -> no closed shell
            spread_a=0.0,
            noise_amplitude_a=0.0,
            curvature_iterations=2,
            n_lipids_per_leaflet=6,
            seed=0,
        )
    gen = MembraneTomogramGenerator(
        # Explicit position_xyz -- this test's own oversized-radius trick
        # (deliberately bigger than the box) would otherwise also get
        # rejected by auto-placement's collision/fit check, dropping the
        # instance and short-circuiting the actual thing under test (the
        # "no lumen region" warning path, not placement itself).
        membrane_instances=[
            MembraneInstance(generator=mgen, position_xyz=(0.0, 0.0, 0.0))
        ],
        target_shape_zyx=(24, 24, 24),
        v_size=8.0,
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
        MembraneTomogramGenerator(
            membrane_instances=[MembraneInstance(generator=mgen)],
            target_shape_zyx=_TARGET_SHAPE_ZYX,
            v_size=_V_SIZE,
            protein_specs=[],
        )


def test_membrane_tomogram_generator_rejects_mismatched_v_size():
    mgen = MembraneGenerator(target_shape_zyx=_TARGET_SHAPE_ZYX, v_size=4.0, seed=0)
    with pytest.raises(ValueError, match=r"membrane_instances\[0\]"):
        MembraneTomogramGenerator(
            membrane_instances=[MembraneInstance(generator=mgen)],
            target_shape_zyx=_TARGET_SHAPE_ZYX,
            v_size=_V_SIZE,
            protein_specs=[TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE))],
        )


@pytest.mark.skipif(not _LARGE_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_membrane_tomogram_generator_composites_two_non_overlapping_instances():
    """Two instances at distinct, well-separated position_xyz -- composited
    density nonzero near both, membrane_labels has exactly 2 distinct
    nonzero IDs, each spatially localized near its own instance. Uses a
    bigger box than _MEMBRANE_KWARGS' own (100^3 vs 64^3) so the +-200A
    offsets used here fit with margin, isolating this test from the
    unrelated boundary-clipping warning a tighter box would also trigger."""
    big_shape_zyx = (100, 100, 100)
    kwargs = dict(_MEMBRANE_KWARGS, target_shape_zyx=big_shape_zyx)
    mgen_a = MembraneGenerator(seed=0, **kwargs)
    mgen_b = MembraneGenerator(seed=1, **kwargs)
    gen = MembraneTomogramGenerator(
        membrane_instances=[
            MembraneInstance(generator=mgen_a, position_xyz=(-200.0, 0.0, 0.0)),
            MembraneInstance(generator=mgen_b, position_xyz=(200.0, 0.0, 0.0)),
        ],
        target_shape_zyx=big_shape_zyx,
        v_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_LARGE_FIXTURE), location="cytosol")
        ],
        occupancy_fraction=0.05,
        gap_angstrom=5.0,
        pdb_cache_dir="pdb-data/",
        seed=0,
    )
    volume = gen.generate()

    assert torch.isfinite(volume).all()
    labels = gen.membrane_labels
    assert labels is not None
    unique_ids = set(torch.unique(labels).tolist()) - {0}
    assert unique_ids == {1, 2}

    # Each instance's own label should be spatially concentrated near its
    # own position_xyz (left half vs right half of the X axis).
    id1_x = (labels == 1).nonzero()[:, 2]
    id2_x = (labels == 2).nonzero()[:, 2]
    assert (id1_x < big_shape_zyx[2] // 2).float().mean() > 0.8
    assert (id2_x >= big_shape_zyx[2] // 2).float().mean() > 0.8


def test_membrane_tomogram_generator_auto_places_non_colliding_instances():
    """Two instances with position_xyz left at its default (None) in a box
    generously sized for both -- both should be accepted (no "dropped"
    warning), get distinct, non-overlapping labels, and have their own
    position_xyz mutated in place to the resolved coordinates (inspectable
    after generate())."""
    big_shape_zyx = (140, 140, 140)
    kwargs = dict(_MEMBRANE_KWARGS, target_shape_zyx=big_shape_zyx)
    mgen_a = MembraneGenerator(seed=0, **kwargs)
    mgen_b = MembraneGenerator(seed=1, **kwargs)
    instance_a = MembraneInstance(generator=mgen_a)
    instance_b = MembraneInstance(generator=mgen_b)
    assert instance_a.position_xyz is None
    gen = MembraneTomogramGenerator(
        membrane_instances=[instance_a, instance_b],
        target_shape_zyx=big_shape_zyx,
        v_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_LARGE_FIXTURE), location="cytosol")
        ],
        occupancy_fraction=0.05,
        pdb_cache_dir="pdb-data/",
        seed=0,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gen.generate()
    dropped_warns = [x for x in w if "dropped" in str(x.message)]
    assert not dropped_warns

    assert instance_a.position_xyz is not None
    assert instance_b.position_xyz is not None
    labels = gen.membrane_labels
    assert labels is not None
    assert set(torch.unique(labels).tolist()) - {0} == {1, 2}


def test_membrane_tomogram_generator_drops_instances_that_dont_fit():
    """Several instances, deliberately too many/too-large for a small box
    -- some must be dropped (warned about), and a dropped instance's own
    generator is never .generate()-called (self.field stays None) since
    rejection happens before generation, not after."""
    small_shape_zyx = (30, 30, 30)  # 30*8 = 240 A per axis
    kwargs = dict(
        _MEMBRANE_KWARGS, target_shape_zyx=small_shape_zyx, sh_axes_a=(70.0, 70.0, 70.0)
    )
    instances = [
        MembraneInstance(generator=MembraneGenerator(seed=i, **kwargs))
        for i in range(4)
    ]
    gen = MembraneTomogramGenerator(
        membrane_instances=instances,
        target_shape_zyx=small_shape_zyx,
        v_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_LARGE_FIXTURE), location="cytosol")
        ],
        occupancy_fraction=0.05,
        pdb_cache_dir="pdb-data/",
        seed=0,
    )
    with pytest.warns(UserWarning, match="did not fit without colliding"):
        gen.generate()

    n_placed = sum(1 for mi in instances if mi.position_xyz is not None)
    assert 0 < n_placed < len(instances)
    for mi in instances:
        if mi.position_xyz is None:
            assert mi.generator.field is None


def test_membrane_tomogram_generator_overlapping_instances_first_write_wins():
    """Two fully-overlapping instances (both explicitly at the origin --
    position_xyz now defaults to None/auto-placed, so this test pins both
    to (0,0,0) explicitly to keep testing overlap detection specifically,
    not auto-placement) -- warns on overlap, membrane_labels shows only ID
    1 in the overlap region (first-write-wins, deterministic)."""
    mgen_a = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    mgen_b = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = MembraneTomogramGenerator(
        membrane_instances=[
            MembraneInstance(generator=mgen_a, position_xyz=(0.0, 0.0, 0.0)),
            MembraneInstance(generator=mgen_b, position_xyz=(0.0, 0.0, 0.0)),
        ],
        target_shape_zyx=_TARGET_SHAPE_ZYX,
        v_size=_V_SIZE,
        protein_specs=[TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE))],
        occupancy_fraction=0.05,
        pdb_cache_dir="pdb-data/",
        seed=0,
    )
    with pytest.warns(UserWarning, match="overlaps a voxel"):
        gen.generate()

    labels = gen.membrane_labels
    assert labels is not None
    # Identical seed/shape -> identical shell masks -> full overlap ->
    # instance 2 should contribute NOTHING (every voxel it would claim was
    # already claimed by instance 1).
    assert 2 not in set(torch.unique(labels).tolist())
    assert 1 in set(torch.unique(labels).tolist())


@pytest.mark.skipif(not _SMALL_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_membrane_tomogram_generator_transmembrane_reflects_position_offset():
    offset = (100.0, -50.0, 25.0)
    kwargs = dict(_MEMBRANE_KWARGS)
    transmembrane_specs = [
        TransmembraneSpec(pdb_source=str(_SMALL_FIXTURE), frequency=2)
    ]

    mgen_origin = MembraneGenerator(
        transmembrane_specs=transmembrane_specs, seed=0, **kwargs
    )
    gen_origin = MembraneTomogramGenerator(
        membrane_instances=[
            MembraneInstance(generator=mgen_origin, position_xyz=(0.0, 0.0, 0.0))
        ],
        target_shape_zyx=_TARGET_SHAPE_ZYX,
        v_size=_V_SIZE,
        protein_specs=[TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE))],
        occupancy_fraction=0.05,
        pdb_cache_dir="pdb-data/",
        seed=0,
    )
    gen_origin.generate()

    mgen_offset = MembraneGenerator(
        transmembrane_specs=transmembrane_specs, seed=0, **kwargs
    )
    gen_offset = MembraneTomogramGenerator(
        membrane_instances=[
            MembraneInstance(generator=mgen_offset, position_xyz=offset)
        ],
        target_shape_zyx=_TARGET_SHAPE_ZYX,
        v_size=_V_SIZE,
        protein_specs=[TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE))],
        occupancy_fraction=0.05,
        pdb_cache_dir="pdb-data/",
        seed=0,
    )
    gen_offset.generate()

    assert len(gen_origin.transmembrane_placements) > 0
    assert len(gen_offset.transmembrane_placements) == len(
        gen_origin.transmembrane_placements
    )
    offset_t = torch.tensor(offset)
    for tp_origin, tp_offset in zip(
        gen_origin.transmembrane_placements, gen_offset.transmembrane_placements
    ):
        assert torch.allclose(tp_offset.center_xyz, tp_origin.center_xyz + offset_t)


@pytest.mark.skipif(
    not (_SMALL_FIXTURE.exists() and _LARGE_FIXTURE.exists()),
    reason="bundled PDB fixtures missing",
)
def test_membrane_tomogram_generator_export_picks(tmp_path):
    """export_picks: coordinate conversion (corner-relative, not centered)
    against a known placement, and transmembrane species get their own
    suffixed file distinct from cytosol/lumen files."""
    # A bigger sh_axes_a than _MEMBRANE_KWARGS' own (100A vs 70A radius) --
    # verified directly: 70A reliably finds zero transmembrane sites for
    # 1mbo at this seed/box (Newton-projection surface search exhausts
    # max_attempts against too-tight a curvature), 100A reliably finds one.
    # This test doesn't need a lumen (cytosol-only protein_specs below), so
    # unlike _MEMBRANE_KWARGS's own tuning target there's no competing
    # constraint pulling toward a smaller radius.
    mgen = MembraneGenerator(
        target_shape_zyx=_TARGET_SHAPE_ZYX,
        v_size=_V_SIZE,
        sh_axes_a=(100.0, 100.0, 100.0),
        sh_amplitude=0.15,
        n_lipids_per_leaflet=6,
        transmembrane_specs=[
            TransmembraneSpec(pdb_source=str(_SMALL_FIXTURE), frequency=1)
        ],
        seed=0,
    )
    gen = MembraneTomogramGenerator(
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape_zyx=_TARGET_SHAPE_ZYX,
        v_size=_V_SIZE,
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

    v_size = _V_SIZE
    shape_zyx = _TARGET_SHAPE_ZYX
    extent_xyz = (
        torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=torch.float32)
        * v_size
    )

    placed = gen.placements[0]
    lines = written[cytosol_key].read_text().strip().splitlines()
    # Match this placement's own row by its corner-relative x coordinate
    # (each placement gets a distinct enough position at this scale).
    expected_corner = placed.position_xyz + extent_xyz / 2
    rows = [json.loads(line) for line in lines]
    match = next(
        r for r in rows if abs(r["location"]["x"] - float(expected_corner[0])) < 1e-3
    )
    assert abs(match["location"]["y"] - float(expected_corner[1])) < 1e-3
    assert abs(match["location"]["z"] - float(expected_corner[2])) < 1e-3
    assert "xyz_rotation_matrix" in match


@pytest.mark.skipif(not _SMALL_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_membrane_tomogram_generator_places_filaments():
    """filament_specs scatters monomer instances independently of the
    membrane/protein packing above, continuing instance_labels' own
    instance-id counter (see _stamp_filaments)."""
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = MembraneTomogramGenerator(
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape_zyx=_TARGET_SHAPE_ZYX,
        v_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE), location="cytosol")
        ],
        filament_specs=[
            FilamentSpec(
                code=str(_SMALL_FIXTURE),
                step=30.0,
                flex_deg=8.0,
                n_filaments=2,
                n_monomers=4,
            )
        ],
        occupancy_fraction=0.05,
        gap_angstrom=5.0,
        pdb_cache_dir="pdb-data/",
        seed=0,
    )
    volume = gen.generate()

    assert torch.isfinite(volume).all()
    assert len(gen.filament_instances) == 2 * 4
    assert all(inst.code == str(_SMALL_FIXTURE) for inst in gen.filament_instances)

    labels = gen.instance_labels
    assert labels is not None
    protein_ids = {p.instance_id for p in gen.placements}
    present_ids = set(torch.unique(labels[labels > 0]).tolist())
    # Filament monomers got their own ids, continuing the same counter --
    # more distinct ids present than just the cytosol/lumen placements.
    assert present_ids.issuperset(protein_ids)
    assert len(present_ids) > len(protein_ids)


def test_membrane_tomogram_generator_no_filament_specs_places_nothing():
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = MembraneTomogramGenerator(
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape_zyx=_TARGET_SHAPE_ZYX,
        v_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE), location="cytosol")
        ],
        occupancy_fraction=0.05,
        pdb_cache_dir="pdb-data/",
        seed=0,
    )
    gen.generate()
    assert gen.filament_instances == []


@pytest.mark.skipif(not _SMALL_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_membrane_tomogram_generator_export_picks_includes_filaments(tmp_path):
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = MembraneTomogramGenerator(
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape_zyx=_TARGET_SHAPE_ZYX,
        v_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE), location="cytosol")
        ],
        filament_specs=[
            FilamentSpec(
                code=str(_SMALL_FIXTURE),
                step=30.0,
                flex_deg=8.0,
                n_filaments=1,
                n_monomers=3,
            )
        ],
        occupancy_fraction=0.05,
        pdb_cache_dir="pdb-data/",
        seed=0,
    )
    gen.generate()

    written = gen.export_picks(tmp_path)
    filament_key = next(k for k in written if k.endswith("-filament"))
    lines = written[filament_key].read_text().strip().splitlines()
    assert len(lines) == 3
    rows = [json.loads(line) for line in lines]
    assert all(row["type"] == "orientedPoint" for row in rows)
    assert all("xyz_rotation_matrix" in row for row in rows)

    # Filament picks are written directly in place_filaments' own
    # corner-relative [0, extent) frame -- no `+ extent_xyz / 2` shift, see
    # export_picks' own docstring -- so every coordinate should land within
    # the volume's physical extent.
    v_size = _V_SIZE
    shape_zyx = _TARGET_SHAPE_ZYX
    extent_xyz = (
        torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=torch.float32)
        * v_size
    )
    for row in rows:
        for axis, extent in zip("xyz", extent_xyz.tolist()):
            assert 0.0 <= row["location"][axis] <= extent
