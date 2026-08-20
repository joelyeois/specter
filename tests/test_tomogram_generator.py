"""
Smoke tests for TomogramSpecimenGenerator (specter.specimen.tomogram) --
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

from specter.specimen._carbon import CarbonFilmSpec
from specter.specimen.filament import FilamentSpec
from specter.specimen.membrane import MembraneGenerator, TransmembraneSpec
from specter.specimen.tomogram import (
    MembraneInstance,
    TomogramSpecimenGenerator,
    TomogramBeadSpec,
    TomogramProteinSpec,
)

_SMALL_FIXTURE = Path(__file__).parent.parent / "specter-data" / "pdb" / "1mbo.cif"
_LARGE_FIXTURE = (
    Path(__file__).parent.parent / "specter-data" / "pdb" / "1bxn-assembly1.cif"
)

_TARGET_SHAPE_ZYX = (64, 64, 64)
_V_SIZE = 8.0

# A spherical_harmonics ellipsoid at this scale encloses a lumen with
# equivalent spherical radius ~59A -- comfortably bigger than 1mbo's own
# ~31.4A radius (verified directly; smaller configs leave no room for even
# one instance).
_MEMBRANE_KWARGS = dict(
    target_shape=_TARGET_SHAPE_ZYX,
    voxel_size=_V_SIZE,
    sh_axes=(70.0, 70.0, 70.0),
    sh_amplitude=0.15,
    n_lipids_per_leaflet=6,
)


@pytest.mark.skipif(
    not (_SMALL_FIXTURE.exists() and _LARGE_FIXTURE.exists()),
    reason="bundled PDB fixtures missing",
)
def test_tomogram_specimen_generator_places_both_locations_correctly():
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = TomogramSpecimenGenerator(
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE), location="lumen"),
            TomogramProteinSpec(pdb_source=str(_LARGE_FIXTURE), location="cytosol"),
        ],
        occupancy_fraction=0.1,
        pdb_cache_dir="specter-data/pdb/",
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

    voxel_size = _V_SIZE
    shape_zyx = _TARGET_SHAPE_ZYX
    center_zyx = torch.tensor([shape_zyx[0] / 2, shape_zyx[1] / 2, shape_zyx[2] / 2])

    def _voxel_index(position_xyz: torch.Tensor) -> tuple[int, int, int]:
        idx_xyz = (position_xyz / voxel_size) + center_zyx[[2, 1, 0]]
        ix, iy, iz = idx_xyz.round().long().tolist()
        return iz, iy, ix

    for p in by_location["lumen"]:
        iz, iy, ix = _voxel_index(p.position_xyz)
        assert bool(gen.regions["lumen"][iz, iy, ix])
    for p in by_location["cytosol"]:
        iz, iy, ix = _voxel_index(p.position_xyz)
        assert bool(gen.regions["cytosol"][iz, iy, ix])


@pytest.mark.skipif(not _SMALL_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_tomogram_specimen_generator_instance_labels_match_placements():
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = TomogramSpecimenGenerator(
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE), location="cytosol")
        ],
        occupancy_fraction=0.05,
        pdb_cache_dir="specter-data/pdb/",
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
def test_tomogram_specimen_generator_warns_when_lumen_species_has_no_region():
    # Degenerate membrane config unlikely to enclose any lumen at all --
    # bilayer_thickness far exceeds the vesicle's own radius, so the two
    # leaflet offset surfaces invert/overlap through the whole interior
    # instead of leaving a hollow enclosed cavity; the point of this test
    # is the degenerate-geometry warning path, not backend choice.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mgen = MembraneGenerator(
            target_shape=(24, 24, 24),
            voxel_size=8.0,
            shape_backend="spherical_harmonics",
            sh_axes=(20.0, 20.0, 20.0),
            bilayer_thickness=150.0,  # >> sh_axes -> no hollow lumen
            n_lipids_per_leaflet=6,
            seed=0,
        )
    gen = TomogramSpecimenGenerator(
        # Single instance -- auto-placement is deterministic under a fixed
        # seed with nothing to collide against; the actual thing under
        # test is the "no lumen region" warning path, not placement itself.
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape=(24, 24, 24),
        voxel_size=8.0,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE), location="lumen")
        ],
        occupancy_fraction=0.1,
        pdb_cache_dir="specter-data/pdb/",
        seed=0,
    )
    with pytest.warns(UserWarning, match="no 'lumen' region"):
        gen.generate()
    assert len(gen.placements) == 0


def test_tomogram_specimen_generator_allows_empty_protein_specs():
    """A membrane-only tomogram (no packed protein population) is valid --
    protein_specs may be empty as long as membrane_instances/filament_specs
    isn't."""
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = TomogramSpecimenGenerator(
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[],
    )
    volume = gen.generate()
    assert volume.shape == _TARGET_SHAPE_ZYX
    assert gen.placements == []


def test_tomogram_specimen_generator_rejects_all_empty():
    with pytest.raises(ValueError, match="at least one"):
        TomogramSpecimenGenerator(
            membrane_instances=[],
            target_shape=_TARGET_SHAPE_ZYX,
            voxel_size=_V_SIZE,
            protein_specs=[],
        )


def test_tomogram_specimen_generator_rejects_mismatched_voxel_size():
    mgen = MembraneGenerator(target_shape=_TARGET_SHAPE_ZYX, voxel_size=4.0, seed=0)
    with pytest.raises(ValueError, match=r"membrane_instances\[0\]"):
        TomogramSpecimenGenerator(
            membrane_instances=[MembraneInstance(generator=mgen)],
            target_shape=_TARGET_SHAPE_ZYX,
            voxel_size=_V_SIZE,
            protein_specs=[TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE))],
        )


@pytest.mark.skipif(not _LARGE_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_tomogram_specimen_generator_composites_two_non_overlapping_instances():
    """Two collision-checked auto-placed instances -- composited density
    nonzero near both, membrane_labels has exactly 2 distinct nonzero IDs,
    each spatially localized near its own RESOLVED position_xyz (verifies
    `_insert_shell_label` places the shell at the physically correct
    offset, not just that two disjoint ID sets exist). Uses a bigger box
    than _MEMBRANE_KWARGS' own (100^3 vs 64^3) so both instances fit with
    margin, isolating this test from the unrelated boundary-clipping
    warning a tighter box would also trigger."""
    big_shape_zyx = (100, 100, 100)
    kwargs = dict(_MEMBRANE_KWARGS, target_shape=big_shape_zyx)
    mgen_a = MembraneGenerator(seed=0, **kwargs)
    mgen_b = MembraneGenerator(seed=1, **kwargs)
    instance_a = MembraneInstance(generator=mgen_a)
    instance_b = MembraneInstance(generator=mgen_b)
    gen = TomogramSpecimenGenerator(
        membrane_instances=[instance_a, instance_b],
        target_shape=big_shape_zyx,
        voxel_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_LARGE_FIXTURE), location="cytosol")
        ],
        occupancy_fraction=0.05,
        pdb_cache_dir="specter-data/pdb/",
        seed=0,
    )
    volume = gen.generate()

    assert torch.isfinite(volume).all()
    labels = gen.membrane_labels
    assert labels is not None
    unique_ids = set(torch.unique(labels).tolist()) - {0}
    assert unique_ids == {1, 2}

    # Each instance's own label should be spatially concentrated near its
    # own resolved position_xyz -- compare voxel-index centroid (converted
    # back to a physical xyz offset from the tomogram's own center) against
    # it, loose tolerance since the rendered shape is a perturbed ellipsoid,
    # not a perfect sphere.
    center_idx_zyx = torch.tensor(big_shape_zyx, dtype=torch.float32) / 2
    for instance_id, mi in [(1, instance_a), (2, instance_b)]:
        assert mi.position_xyz is not None
        voxel_idx_zyx = (labels == instance_id).nonzero().float()
        centroid_offset_xyz = (voxel_idx_zyx.mean(dim=0) - center_idx_zyx).flip(
            0
        ) * _V_SIZE
        expected_xyz = torch.tensor(mi.position_xyz)
        assert (centroid_offset_xyz - expected_xyz).norm() < 40.0


def test_tomogram_specimen_generator_auto_places_non_colliding_instances():
    """Two instances with position_xyz left at its default (None) in a box
    generously sized for both -- both should be accepted (no "dropped"
    warning), get distinct, non-overlapping labels, and have their own
    position_xyz mutated in place to the resolved coordinates (inspectable
    after generate())."""
    big_shape_zyx = (140, 140, 140)
    kwargs = dict(_MEMBRANE_KWARGS, target_shape=big_shape_zyx)
    mgen_a = MembraneGenerator(seed=0, **kwargs)
    mgen_b = MembraneGenerator(seed=1, **kwargs)
    instance_a = MembraneInstance(generator=mgen_a)
    instance_b = MembraneInstance(generator=mgen_b)
    assert instance_a.position_xyz is None
    gen = TomogramSpecimenGenerator(
        membrane_instances=[instance_a, instance_b],
        target_shape=big_shape_zyx,
        voxel_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_LARGE_FIXTURE), location="cytosol")
        ],
        occupancy_fraction=0.05,
        pdb_cache_dir="specter-data/pdb/",
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


def test_tomogram_specimen_generator_drops_instances_that_dont_fit():
    """Several instances, deliberately too many/too-large for a small box
    -- some must be dropped (warned about), and a dropped instance's own
    generator is never .generate()-called (self.field stays None) since
    rejection happens before generation, not after."""
    small_shape_zyx = (30, 30, 30)  # 30*8 = 240 A per axis
    kwargs = dict(
        _MEMBRANE_KWARGS, target_shape=small_shape_zyx, sh_axes=(70.0, 70.0, 70.0)
    )
    instances = [
        MembraneInstance(generator=MembraneGenerator(seed=i, **kwargs))
        for i in range(4)
    ]
    gen = TomogramSpecimenGenerator(
        membrane_instances=instances,
        target_shape=small_shape_zyx,
        voxel_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_LARGE_FIXTURE), location="cytosol")
        ],
        occupancy_fraction=0.05,
        pdb_cache_dir="specter-data/pdb/",
        seed=0,
    )
    with pytest.warns(UserWarning, match="did not fit without colliding"):
        gen.generate()

    n_placed = sum(1 for mi in instances if mi.position_xyz is not None)
    assert 0 < n_placed < len(instances)
    for mi in instances:
        if mi.position_xyz is None:
            assert mi.generator.field is None


def test_insert_shell_label_overlap_first_write_wins():
    """Two shell masks stamped at the same physical position overlap fully
    -- `_insert_shell_label` reports the overlap and membrane_labels shows
    only the first instance's ID there (first-write-wins, deterministic).

    With placement now always collision-checked (see
    TomogramSpecimenGenerator.generate()), forcing two ACCEPTED instances
    into full physical overlap isn't reachable through the public API any
    more -- the underlying compositing function still needs covering,
    since an irregular shape extending past its own bounding-sphere
    estimate can still produce this, so this exercises it directly."""
    from specter.specimen.tomogram.generator import _insert_shell_label

    shape_zyx = (16, 16, 16)
    labels = torch.zeros(shape_zyx, dtype=torch.int32)
    shell_mask = torch.zeros(shape_zyx, dtype=torch.bool)
    shell_mask[6:10, 6:10, 6:10] = True

    labels, overlap1 = _insert_shell_label(labels, shell_mask, 1, (0.0, 0.0, 0.0), 8.0)
    assert not overlap1
    labels, overlap2 = _insert_shell_label(labels, shell_mask, 2, (0.0, 0.0, 0.0), 8.0)
    assert overlap2

    unique_ids = set(torch.unique(labels).tolist())
    assert 2 not in unique_ids
    assert 1 in unique_ids


@pytest.mark.skipif(not _SMALL_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_tomogram_specimen_generator_transmembrane_reflects_position_offset():
    """Composited transmembrane placements equal the same MembraneGenerator
    config's own LOCAL (un-composited) placements, shifted by the
    instance's resolved (auto-placed) position_xyz -- verifies the offset
    is applied correctly at composite time, independent of where
    auto-placement actually put the instance."""
    kwargs = dict(_MEMBRANE_KWARGS)
    transmembrane_specs = [
        TransmembraneSpec(pdb_source=str(_SMALL_FIXTURE), frequency=2)
    ]

    mgen = MembraneGenerator(transmembrane_specs=transmembrane_specs, seed=0, **kwargs)
    instance = MembraneInstance(generator=mgen)
    gen = TomogramSpecimenGenerator(
        membrane_instances=[instance],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE))],
        occupancy_fraction=0.05,
        pdb_cache_dir="specter-data/pdb/",
        seed=0,
    )
    gen.generate()
    assert instance.position_xyz is not None
    assert len(gen.transmembrane_placements) > 0

    # Independently regenerate the SAME membrane config (identical seed),
    # in isolation this time, to recover its local un-composited placements.
    mgen_local = MembraneGenerator(
        transmembrane_specs=transmembrane_specs, seed=0, **kwargs
    )
    mgen_local.generate()
    local_placements = mgen_local.place_transmembrane(
        min_spacing_a=gen.min_transmembrane_spacing
    )
    assert len(local_placements) == len(gen.transmembrane_placements)

    offset_t = torch.tensor(instance.position_xyz)
    for tp_local, tp_composited in zip(local_placements, gen.transmembrane_placements):
        assert torch.allclose(tp_composited.center_xyz, tp_local.center_xyz + offset_t)


@pytest.mark.skipif(
    not (_SMALL_FIXTURE.exists() and _LARGE_FIXTURE.exists()),
    reason="bundled PDB fixtures missing",
)
def test_tomogram_specimen_generator_export_picks(tmp_path):
    """export_picks: coordinate conversion (corner-relative, not centered)
    against a known placement, and transmembrane species get their own
    suffixed file distinct from cytosol/lumen files."""
    # A bigger sh_axes than _MEMBRANE_KWARGS' own (100A vs 70A radius) --
    # verified directly: 70A reliably finds zero transmembrane sites for
    # 1mbo at this seed/box (Newton-projection surface search exhausts
    # max_attempts against too-tight a curvature), 100A reliably finds one.
    # This test doesn't need a lumen (cytosol-only protein_specs below), so
    # unlike _MEMBRANE_KWARGS's own tuning target there's no competing
    # constraint pulling toward a smaller radius.
    mgen = MembraneGenerator(
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        sh_axes=(100.0, 100.0, 100.0),
        sh_amplitude=0.15,
        n_lipids_per_leaflet=6,
        transmembrane_specs=[
            TransmembraneSpec(pdb_source=str(_SMALL_FIXTURE), frequency=1)
        ],
        seed=0,
    )
    gen = TomogramSpecimenGenerator(
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_LARGE_FIXTURE), location="cytosol"),
        ],
        occupancy_fraction=0.1,
        pdb_cache_dir="specter-data/pdb/",
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

    voxel_size = _V_SIZE
    shape_zyx = _TARGET_SHAPE_ZYX
    extent_xyz = (
        torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=torch.float32)
        * voxel_size
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
def test_tomogram_specimen_generator_places_filaments():
    """filament_specs scatters monomer instances independently of the
    membrane/protein packing above, continuing instance_labels' own
    instance-id counter (see _stamp_filaments)."""
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = TomogramSpecimenGenerator(
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE), location="cytosol")
        ],
        filament_specs=[
            FilamentSpec(
                code=str(_SMALL_FIXTURE),
                step=30.0,
                flex_deg=8.0,
                n_copies=2,
                n_monomers=4,
            )
        ],
        occupancy_fraction=0.05,
        pdb_cache_dir="specter-data/pdb/",
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


def test_tomogram_specimen_generator_no_filament_specs_places_nothing():
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = TomogramSpecimenGenerator(
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE), location="cytosol")
        ],
        occupancy_fraction=0.05,
        pdb_cache_dir="specter-data/pdb/",
        seed=0,
    )
    gen.generate()
    assert gen.filament_instances == []


@pytest.mark.skipif(not _SMALL_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_tomogram_specimen_generator_export_picks_includes_filaments(tmp_path):
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = TomogramSpecimenGenerator(
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE), location="cytosol")
        ],
        filament_specs=[
            FilamentSpec(
                code=str(_SMALL_FIXTURE),
                step=30.0,
                flex_deg=8.0,
                n_copies=1,
                n_monomers=3,
            )
        ],
        occupancy_fraction=0.05,
        pdb_cache_dir="specter-data/pdb/",
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
    voxel_size = _V_SIZE
    shape_zyx = _TARGET_SHAPE_ZYX
    extent_xyz = (
        torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=torch.float32)
        * voxel_size
    )
    for row in rows:
        for axis, extent in zip("xyz", extent_xyz.tolist()):
            assert 0.0 <= row["location"][axis] <= extent


# ---------------------------------------------------------------------
# Carbon support film (carbon_film_spec) / gold fiducial beads (bead_specs)
# ---------------------------------------------------------------------


def test_tomogram_specimen_generator_carbon_film_spec_paints_carbon_film():
    """A carbon_film_spec-only tomogram (no membrane/protein/filament) is valid --
    the carbon film should occupy part of the volume (not all of it, since
    hole_radius/edge_fraction are chosen here to leave a real hole) at
    carbon's real mean inner potential ballpark (~9-13 V, see
    specter.specimen._grid's own module docstring), and leave the rest at
    exactly zero."""
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[],
        carbon_film_spec=CarbonFilmSpec(
            hole_radius=150.0, edge_fraction=0.5, edge_side="left"
        ),
        seed=0,
    )
    volume = gen.generate()

    assert torch.isfinite(volume).all()
    occupied = volume > 0
    assert 0.0 < occupied.float().mean().item() < 1.0
    assert 5.0 < volume.max().item() < 20.0
    assert (volume[~occupied] == 0).all()


def test_tomogram_specimen_generator_carbon_film_spec_rejects_multiple_entries():
    """TomogramSpecimenGenerator itself takes a single carbon_film_spec (not a
    list) -- the "at most one [[carbon_film]] table" constraint is enforced one
    layer up, in run_build_tomogram/config.py, not here."""
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[],
        carbon_film_spec=CarbonFilmSpec(),
        seed=0,
    )
    assert gen.carbon_film_spec is not None


@pytest.mark.skipif(not _SMALL_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_tomogram_specimen_generator_bead_specs_avoid_membrane_shell():
    mgen = MembraneGenerator(seed=0, **_MEMBRANE_KWARGS)
    gen = TomogramSpecimenGenerator(
        membrane_instances=[MembraneInstance(generator=mgen)],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[],
        bead_specs=[TomogramBeadSpec(radius=15.0, count=10)],
        seed=1,
    )
    gen.generate()

    assert len(gen.bead_instances) > 0
    shell = gen.regions["shell"]
    voxel_size = _V_SIZE
    shape_zyx = _TARGET_SHAPE_ZYX
    center_zyx = torch.tensor([shape_zyx[0] / 2, shape_zyx[1] / 2, shape_zyx[2] / 2])
    for bead in gen.bead_instances:
        idx_xyz = (bead.position_xyz / voxel_size) + center_zyx[[2, 1, 0]]
        ix, iy, iz = idx_xyz.round().long().tolist()
        assert not bool(shell[iz, iy, ix])


def test_tomogram_specimen_generator_bead_radius_range_varies_sizes():
    """A [low, high] radius gives a bead population the size dispersity
    real colloidal gold has. Radii are drawn before packing, so each
    recorded instance carries its own size and the collision test used
    it."""
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[],
        bead_specs=[TomogramBeadSpec(radius=[14.0, 26.0], count=12)],
        seed=0,
    )
    gen.generate()

    radii = torch.tensor([b.radius for b in gen.bead_instances])
    assert radii.numel() >= 5
    assert radii.std() > 0.05 * radii.mean(), "radii are still monodisperse"
    assert radii.min() >= 14.0 and radii.max() <= 26.0, "drawn outside the range"
    assert abs(radii.mean().item() / 20.0 - 1.0) < 0.25


def test_bead_spec_rejects_bad_radius():
    with pytest.raises(ValueError, match="radius must be > 0"):
        TomogramBeadSpec(radius=-1.0)
    with pytest.raises(ValueError, match=r"radius range must be \[low, high\]"):
        TomogramBeadSpec(radius=[60.0, 40.0])


def test_tomogram_specimen_generator_bead_specs_only_is_valid():
    """A bead_specs-only tomogram (no membrane/protein/filament/grid) is
    valid -- beads are unrestricted ("any" location) so no membrane is
    needed to define cytosol/lumen regions."""
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[],
        bead_specs=[TomogramBeadSpec(radius=10.0, count=5)],
        seed=0,
    )
    volume = gen.generate()
    assert volume.shape == _TARGET_SHAPE_ZYX
    assert len(gen.bead_instances) > 0
    assert gen.instance_labels is not None
    assert int(gen.instance_labels.max()) == len(gen.bead_instances)


def test_tomogram_specimen_generator_bead_specs_excluded_from_protein_packing():
    """Beads placed before cytosol/lumen protein packing should be avoided
    by it -- no placed protein's center should land inside a bead's own
    radius."""
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE))],
        bead_specs=[TomogramBeadSpec(radius=30.0, count=3)],
        occupancy_fraction=0.05,
        pdb_cache_dir="specter-data/pdb/",
        seed=0,
    )
    gen.generate()

    assert len(gen.bead_instances) > 0
    assert len(gen.placements) > 0
    for bead in gen.bead_instances:
        for placed in gen.placements:
            dist = (placed.position_xyz - bead.position_xyz).norm().item()
            assert dist > bead.radius


def test_tomogram_specimen_generator_export_picks_includes_beads(tmp_path):
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[],
        bead_specs=[TomogramBeadSpec(radius=20.0, count=2)],
        seed=0,
    )
    gen.generate()

    written = gen.export_picks(tmp_path)
    bead_key = next(k for k in written if k.endswith("-bead"))
    lines = written[bead_key].read_text().strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert all(row["type"] == "point" for row in rows)
    assert all("xyz_rotation_matrix" not in row for row in rows)


def test_diagnose_zero_placements_reports_box_constraint_not_just_clearance():
    """Regression test: exclusion_field[mask].max() alone -- clearance from
    the shell/obstacle, ignoring the box wall -- can dramatically overstate
    how much room a species actually has. Found directly on a real run: it
    reported ~166 A "available" against a 72 A requirement (reading as ample
    room) for a case with ZERO truly viable positions, because every voxel
    far enough from the shell was also too close to the box wall for that
    radius. _diagnose_zero_placements has to catch this by construction.

    Synthetic geometry, chosen so the two failure modes are unambiguous:
    a 100x100x100 A box (field_voxel_size=10, shape (10,10,10)), sampling_mask
    True everywhere, exclusion_field large (200 A) EVERYWHERE except a thin
    disc near the box center where it's also large but the region is
    entirely masked out -- simpler: split the box into two halves along x.
    """
    from specter.specimen.tomogram.generator import _diagnose_zero_placements

    field_voxel_size = 10.0
    shape = (10, 10, 10)  # 100x100x100 A box
    box = (100.0, 100.0, 100.0)

    # Case 1: clearance is huge everywhere, but the radius is big enough
    # that NO position keeps the whole sphere inside a 100 A box (needs
    # radius <= 50 A). No box-valid position exists at all, regardless of
    # clearance -- best_clearance must be exactly 0.0, viable must be 0.
    mask = torch.ones(shape, dtype=torch.bool)
    huge_clearance = torch.full(shape, 200.0)
    viable, best = _diagnose_zero_placements(
        mask,
        huge_clearance,
        field_voxel_size,
        box,
        radius=60.0,
        gap=0.0,
        clip_axes=(False, False, False),
    )
    assert viable == 0
    assert best == 0.0

    # Case 2: radius small enough to fit inside the box on every voxel (5 A
    # radius needs |center| <= 45 A; the outermost voxel center is at
    # exactly 45 A), and clearance is uniformly huge -- every voxel should
    # be viable, best_clearance == 200.0 exactly (not some smaller number
    # from an over-restrictive box check).
    viable, best = _diagnose_zero_placements(
        mask,
        huge_clearance,
        field_voxel_size,
        box,
        radius=5.0,
        gap=0.0,
        clip_axes=(False, False, False),
    )
    assert viable == shape[0] * shape[1] * shape[2]
    assert best == 200.0

    # Case 3: the real bug pattern -- clearance is huge (200 A) ONLY in the
    # single outermost voxel layer (index 0 or 9 on any axis: at radius=10,
    # box-valid requires |center| <= 40 A, which excludes exactly that
    # layer -- centers at +-45 A) and small (5 A, below what a 10 A-radius
    # sphere with no gap needs) everywhere else. The two conditions never
    # overlap, so viable must be 0 -- but best_clearance among the box-valid
    # (inner) voxels must be the SMALL number (5.0), not the misleadingly
    # large one from the box-invalid shell, proving this isn't just
    # "clearance.max() within mask" in disguise.
    idx = torch.arange(10)
    is_outer_layer = (
        (idx.view(10, 1, 1).expand(shape) == 0)
        | (idx.view(10, 1, 1).expand(shape) == 9)
        | (idx.view(1, 10, 1).expand(shape) == 0)
        | (idx.view(1, 10, 1).expand(shape) == 9)
        | (idx.view(1, 1, 10).expand(shape) == 0)
        | (idx.view(1, 1, 10).expand(shape) == 9)
    )
    clearance = torch.where(is_outer_layer, torch.tensor(200.0), torch.tensor(5.0))
    viable, best = _diagnose_zero_placements(
        mask,
        clearance,
        field_voxel_size,
        box,
        radius=10.0,
        gap=0.0,
        clip_axes=(False, False, False),
    )
    assert viable == 0
    assert best == 5.0


def test_diagnose_zero_placements_honors_clip_axes():
    """A clippable axis only needs the CENTER in-bounds, not the full
    sphere -- matching pack_hard_spheres_3d's own clip_axes semantics.
    Same box as above; a 60 A-radius sphere is box-invalid on every axis
    when clip_axes is all False (case 1 above), but valid once every axis
    is marked clippable."""
    from specter.specimen.tomogram.generator import _diagnose_zero_placements

    shape = (10, 10, 10)
    box = (100.0, 100.0, 100.0)
    mask = torch.ones(shape, dtype=torch.bool)
    huge_clearance = torch.full(shape, 200.0)

    viable, best = _diagnose_zero_placements(
        mask,
        huge_clearance,
        10.0,
        box,
        radius=60.0,
        gap=0.0,
        clip_axes=(True, True, True),
    )
    assert viable == shape[0] * shape[1] * shape[2]
    assert best == 200.0


def test_membrane_tomogram_zero_placement_warning_distinguishes_unlucky_from_impossible():
    """End-to-end regression test for the fix: a genuinely too-tight-for-
    the-box species must be reported as impossible (not "ample room
    available"), and a genuinely-possible-but-unlucky one must be reported
    as unlucky (not "impossible"). Both scenarios were verified by hand
    against the real geometry before writing this test.
    """
    from specter.specimen.membrane import MembraneGenerator
    from specter.specimen.tomogram import MembraneInstance

    # Impossible case: 4 membranes competing for a small 240 A box: only
    # some fit, and the surviving membrane's shell leaves no position that
    # is both far enough from the shell AND fully inside the box for the
    # ~67 A-radius filler species.
    small_shape_zyx = (30, 30, 30)
    kwargs = dict(
        _MEMBRANE_KWARGS, target_shape=small_shape_zyx, sh_axes=(70.0, 70.0, 70.0)
    )
    instances = [
        MembraneInstance(generator=MembraneGenerator(seed=i, **kwargs))
        for i in range(4)
    ]
    gen = TomogramSpecimenGenerator(
        membrane_instances=instances,
        target_shape=small_shape_zyx,
        voxel_size=_V_SIZE,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(_LARGE_FIXTURE), location="cytosol")
        ],
        occupancy_fraction=0.05,
        # _diagnose_zero_placements is sphere-backend machinery: it reasons
        # about a bounding radius against an exclusion distance field, which
        # the shape backend has neither of. Pin the backend rather than let
        # this follow the default.
        packing_backend="sphere",
        pdb_cache_dir="specter-data/pdb/",
        seed=0,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gen.generate()
    zero_warnings = [x for x in w if "placed 0 filler instances" in str(x.message)]
    assert len(zero_warnings) == 1
    message = str(zero_warnings[0].message)
    assert "no position exists" in message
    assert "unlucky" not in message
    # The old bug: reporting exclusion_field.max() (166 A here) as if it
    # were achievable, which it never book-keeps against the box wall.
    assert "166" not in message


def test_all_beads_go_in_one_pick_file(tmp_path):
    """Every fiducial lands in a single `gold-bead` file, whatever its
    radius or population. Grouping by radius (the earlier behaviour) wrote
    one file per bead under a [low, high] radius, since every instance
    then has a unique size."""
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=_TARGET_SHAPE_ZYX,
        voxel_size=_V_SIZE,
        protein_specs=[],
        bead_specs=[
            TomogramBeadSpec(radius=[14.0, 26.0], count=6),
            TomogramBeadSpec(radius=30.0, count=2),
        ],
        seed=0,
    )
    gen.generate()
    written = gen.export_picks(str(tmp_path))

    bead_keys = sorted(k for k in written if k.endswith("-bead"))
    assert bead_keys == ["gold-bead"], bead_keys

    # Both populations' beads are in that one file.
    with open(written["gold-bead"]) as f:
        rows = [line for line in f if line.strip()]
    assert len(rows) == len(gen.bead_instances)
    assert len({b.radius for b in gen.bead_instances}) > 1, "test needs mixed sizes"


@pytest.mark.parametrize(
    "parameterization, expect_typed",
    [("shtyrov", True), ("kirkland", False)],
)
def test_shtyrov_templates_are_typed_by_bonded_species(parameterization, expect_typed):
    """Shtyrov fits scattering factors per BONDED SPECIES, so the generator
    must hand `PotentialBuilder` the bond topology -- otherwise every atom
    silently falls back to per-element Peng factors and the parameterization
    is Shtyrov in name only.

    The other parameterizations are per-element by construction, so they
    must NOT pay for the gemmi topology pass.
    """
    import specter.potential as potential_module

    captured: list[tuple[int, int | None]] = []
    original = potential_module.PotentialBuilder.__init__

    def spy(self, *args, **kwargs):
        species = kwargs.get("atom_species")
        captured.append(
            (
                len(kwargs.get("atomic_numbers", [])),
                None if species is None else sum(s is not None for s in species),
            )
        )
        return original(self, *args, **kwargs)

    potential_module.PotentialBuilder.__init__ = spy
    try:
        gen = TomogramSpecimenGenerator(
            membrane_instances=[],
            target_shape=_TARGET_SHAPE_ZYX,
            voxel_size=_V_SIZE,
            protein_specs=[
                TomogramProteinSpec(pdb_source=str(_SMALL_FIXTURE), location="cytosol")
            ],
            parameterization=parameterization,
            seed=0,
        )
        gen.generate()
    finally:
        potential_module.PotentialBuilder.__init__ = original

    assert captured, "no template was rendered -- test would be vacuous"
    for n_atoms, n_typed in captured:
        if expect_typed:
            assert n_typed is not None, "shtyrov rendered without bond topology"
            # Not every atom types, by design: 1mbo carries 324 waters whose
            # hydrogens are not modelled (so an isolated O has no bonded
            # neighbours), plus heme and a bound O2. Those fall back to Peng
            # per-element factors individually, which is the intended
            # degradation -- the bulk of the protein is what must be typed.
            assert n_typed > 0.5 * n_atoms, f"only {n_typed}/{n_atoms} atoms typed"
        else:
            assert n_typed is None, (
                f"{parameterization} paid for a topology pass it cannot use"
            )
