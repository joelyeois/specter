"""
Tests for the CryoTomoSim (CTS) replica specimen generator
(``specter.specimen.cryotomosim`` and its ``_cts_*`` support modules).

These are a from-scratch port of CTS's own algorithms and have no
dependency on polnet/VTK -- tests here must not import anything from
``specimen.cryoet``/``specimen.polnet_bridge``.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from specter.arrays import ball3d
from specter.specimen._cts_grid import BeadGenerator, CarbonFilmGenerator
from specter.specimen._cts_membrane import MembraneBlobGenerator
from specter.specimen._cts_placement import ParticlePlacer, ParticleSpec
from specter.specimen.cryotomosim import (
    BeadSpec,
    CryoTomoSimSpecimenGenerator,
    GridSpec,
    MembraneSpec,
    ProteinSpec,
    _build_membrane_location_fields,
)

_PDB_FIXTURE = Path(__file__).parent.parent / "pdb-data" / "1mbo.cif"


# ---------------------------------------------------------------------
# MembraneBlobGenerator
# ---------------------------------------------------------------------


def test_membrane_blob_is_closed_and_nondegenerate():
    """A generated blob membrane should have a substantial, non-trivial
    footprint (not collapsed to near-empty or a single voxel)."""
    gen = MembraneBlobGenerator(v_size=10.0, seed=0)
    inst = gen.generate(size=200.0, roughness=0.6, thickness=40.0)

    assert inst.density.ndim == 3
    assert inst.density.shape[0] == inst.density.shape[1] == inst.density.shape[2]
    occupied = (inst.density > 0).float().mean().item()
    # A hollow shell in a padded box should occupy a modest but
    # non-negligible fraction of voxels -- neither ~0 (degenerate/empty)
    # nor ~1 (accidentally filled solid).
    assert 0.001 < occupied < 0.5


def test_membrane_blob_has_bilayer_double_peak_profile():
    """
    Radial density profile from the shape's own centroid should show two
    separated peaks (the two leaflets), not a single monolithic peak --
    the defining signature of a bilayer rather than a solid blob.
    """
    gen = MembraneBlobGenerator(v_size=6.0, seed=2)
    inst = gen.generate(
        size=220.0,
        roughness=0.75,
        thickness=60.0,
        n_head_points=10000,
        n_tail_points=1500,
    )
    density = inst.density.numpy()
    n = density.shape[0]

    # Radial profile from the mass centroid of the generated density
    # itself (robust to the shape not being perfectly centered in its own
    # padded grid).
    coords = np.argwhere(density > 0)
    weights = density[density > 0]
    centroid = (coords * weights[:, None]).sum(axis=0) / weights.sum()

    zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    r = np.sqrt(
        (zz - centroid[0]) ** 2 + (yy - centroid[1]) ** 2 + (xx - centroid[2]) ** 2
    )
    r_bin = np.round(r).astype(int).ravel()
    prof = np.bincount(r_bin, weights=density.ravel())
    counts = np.bincount(r_bin)
    prof = prof / np.maximum(counts, 1)

    # Find local maxima in the smoothed 1D profile.
    smoothed = np.convolve(prof, np.ones(3) / 3, mode="same")
    peaks = []
    for i in range(2, len(smoothed) - 2):
        if smoothed[i] > 0 and smoothed[i] == max(smoothed[i - 2 : i + 3]):
            peaks.append((i, smoothed[i]))
    peaks.sort(key=lambda p: -p[1])
    top_peaks = sorted(p[0] for p in peaks[:4])
    assert len(top_peaks) >= 2, (
        f"expected at least 2 distinct radial density peaks (bilayer "
        f"leaflets), found {len(top_peaks)}: {top_peaks}"
    )
    # The two most prominent peaks should be meaningfully separated (not
    # the same leaflet double-counted by binning noise).
    spread = max(top_peaks) - min(top_peaks)
    assert (
        spread >= 2
    ), f"radial peaks too close together to be distinct leaflets: {top_peaks}"


def test_membrane_blob_rejects_invalid_roughness():
    gen = MembraneBlobGenerator(v_size=10.0, seed=0)
    with pytest.raises(ValueError):
        gen.generate(roughness=1.5)
    with pytest.raises(ValueError):
        gen.generate(roughness=0.0)


# ---------------------------------------------------------------------
# ParticlePlacer
# ---------------------------------------------------------------------


def test_particle_placer_avoids_overlap_and_respects_density_cutoff():
    torch.manual_seed(0)
    vol = torch.zeros((64, 64, 64))
    density = ball3d(9, 6.0)
    spec = ParticleSpec(
        species_id="ball", density=density, max_count=100, max_attempts_per_copy=30
    )
    placer = ParticlePlacer(
        volume=vol, density_cutoff=0.15, rng=torch.Generator().manual_seed(0)
    )
    placed = placer.run([spec])

    assert len(placed) > 0
    occupied = (vol > 0).float().mean().item()
    assert occupied <= 0.15 + 1e-6

    # No two placements should be close enough to have overlapped. ball3d's
    # filled sphere has diameter 6 voxels (the 9-voxel box is just padding);
    # centers closer than that diameter would imply the overlap test failed
    # to reject them. A small margin below 6.0 absorbs rotation/interpolation
    # edge effects without weakening the actual claim being tested.
    centers = torch.stack([p.center_zyx for p in placed])
    if len(centers) > 1:
        dists = torch.cdist(centers, centers)
        dists.fill_diagonal_(float("inf"))
        assert dists.min().item() >= 5.5


def test_particle_placer_stops_when_volume_full():
    torch.manual_seed(0)
    vol = torch.zeros((16, 16, 16))
    density = ball3d(5, 4.0)
    spec = ParticleSpec(species_id="ball", density=density, max_count=1000)
    placer = ParticlePlacer(
        volume=vol, density_cutoff=0.3, rng=torch.Generator().manual_seed(1)
    )
    placed = placer.run([spec])
    # Should terminate (not hang), and stay close to the cutoff. The check
    # gates the NEXT attempted placement (matching CTS's own
    # helper_randomfill.m, which also checks after inserting and can
    # overshoot by one placement's worth before stopping) rather than
    # rolling back an insert that crosses the line, so a small overshoot
    # margin (one particle's own occupancy contribution) is expected, not
    # a bug.
    assert (vol > 0).float().mean().item() <= 0.3 + 0.02
    assert len(placed) < 1000


def test_particle_placer_location_flag_requires_mask():
    vol = torch.zeros((16, 16, 16))
    density = ball3d(3, 2.0)
    spec = ParticleSpec(
        species_id="memprot", density=density, max_count=1, location="membrane"
    )
    placer = ParticlePlacer(volume=vol)
    with pytest.raises(ValueError):
        placer.run([spec])


# ---------------------------------------------------------------------
# CarbonFilmGenerator / BeadGenerator
# ---------------------------------------------------------------------


def test_bead_generator_bulk_density_is_sane():
    """A gold bead's peak voxel value should reflect gold's real bulk
    number density (atoms/A^3, scaled by voxel volume), of the same
    order of magnitude across a couple of voxel sizes."""
    gen = BeadGenerator(v_size=5.0)
    bead = gen.generate(radius=50.0)
    assert bead.density.shape[0] > 0
    assert (bead.density > 0).any()
    # ~4.5e-2 atoms/A^3 for gold * 125 A^3/voxel (5A)^3 ~ 5.7 atoms/voxel
    assert 1.0 < bead.density.max().item() < 50.0
    assert torch.isfinite(bead.density).all()


def test_bead_generator_rejects_nonpositive_radius():
    gen = BeadGenerator(v_size=5.0)
    with pytest.raises(ValueError):
        gen.generate(radius=0.0)


def test_carbon_film_has_hole_and_sane_density():
    gen = CarbonFilmGenerator(v_size=15.0, seed=0)
    film = gen.generate(target_shape=(16, 64, 64), thickness=150.0, hole_radius=300.0)
    assert film.density.shape == (16, 64, 64)
    assert torch.isfinite(film.density).all()
    assert (film.density > 0).any()
    # Hole should leave the volume's physical center emptier than its
    # edges on average (edges are outside the cut-out hole radius).
    center_mass = film.density[:, 24:40, 24:40].mean().item()
    edge_mass = film.density[:, :8, :8].mean().item()
    assert edge_mass > center_mass


# ---------------------------------------------------------------------
# CryoTomoSimSpecimenGenerator (end-to-end)
# ---------------------------------------------------------------------


@pytest.mark.skipif(not _PDB_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_cryotomosim_generator_end_to_end():
    target_shape = (32, 48, 48)
    gen = CryoTomoSimSpecimenGenerator(
        protein_specs=[ProteinSpec(pdb_source=str(_PDB_FIXTURE), max_count=2)],
        membrane_specs=[MembraneSpec(count=1, size=100.0, thickness=30.0)],
        bead_spec=BeadSpec(radii=[40.0], count_per_radius=1),
        grid_spec=GridSpec(thickness=100.0, hole_radius=150.0),
        target_shape=target_shape,
        v_size=10.0,
        ice_opacity=1.0,
        seed=0,
    )
    volume = gen.generate()

    assert volume.shape == target_shape
    assert not torch.isnan(volume).any()
    assert not torch.isinf(volume).any()
    assert (volume >= 0).all()
    assert volume.max().item() > 0
    assert len(gen.placements) > 0


@pytest.mark.skipif(not _PDB_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_cryotomosim_generator_no_extras_still_works():
    """Proteins only, no membranes/beads/grid/ice -- the minimal path."""
    target_shape = (24, 32, 32)
    gen = CryoTomoSimSpecimenGenerator(
        protein_specs=[ProteinSpec(pdb_source=str(_PDB_FIXTURE), max_count=1)],
        target_shape=target_shape,
        v_size=10.0,
        ice_opacity=0.0,
        seed=0,
    )
    volume = gen.generate()
    assert volume.shape == target_shape
    assert not torch.isnan(volume).any()
    assert volume.max().item() > 0


# ---------------------------------------------------------------------
# Membrane-embedded placement (membrane/vesicle/cytosol location gating)
# ---------------------------------------------------------------------


def _placed_membrane(v_size=8.0, size=220.0, roughness=0.75, thickness=35.0, seed=1):
    """Generate and place one membrane into a plain volume; return
    (volume, placer, location_masks, normal_fields)."""
    torch.manual_seed(0)
    vol = torch.zeros((64, 64, 64))
    placer = ParticlePlacer(
        volume=vol, density_cutoff=0.4, rng=torch.Generator().manual_seed(0)
    )
    mem_gen = MembraneBlobGenerator(v_size=v_size, seed=seed)
    inst = mem_gen.generate(size=size, roughness=roughness, thickness=thickness)
    placer.run([ParticleSpec(species_id="mem", density=inst.density, max_count=1)])
    assert len(placer.placements) == 1, "membrane failed to place at all"
    masks, normals = _build_membrane_location_fields(
        placer.placements, {"mem": inst}, (64, 64, 64)
    )
    return vol, placer, masks, normals


def test_membrane_flagged_particle_lands_on_shell_not_free_in_cytosol():
    vol, placer, masks, normals = _placed_membrane()
    density = ball3d(5, 3.0)
    spec = ParticleSpec(
        species_id="memprot",
        density=density,
        max_count=5,
        location="membrane",
        max_attempts_per_copy=30,
    )
    n_placed = placer.place_species(spec, location_masks=masks, normal_fields=normals)
    assert n_placed > 0

    memprot_placements = [p for p in placer.placements if p.species_id == "memprot"]
    assert len(memprot_placements) == n_placed
    for p in memprot_placements:
        zi, yi, xi = (int(v) for v in p.center_zyx.tolist())
        zi = min(max(zi, 0), masks["membrane"].shape[0] - 1)
        yi = min(max(yi, 0), masks["membrane"].shape[1] - 1)
        xi = min(max(xi, 0), masks["membrane"].shape[2] - 1)
        assert masks["membrane"][
            zi, yi, xi
        ], "membrane-flagged particle center is not on the membrane shell mask"
        assert not masks["cytosol"][zi, yi, xi]


def test_vesicle_and_cytosol_flags_respect_inside_outside():
    vol, placer, masks, normals = _placed_membrane()
    small_density = ball3d(5, 3.0)

    for loc in ["vesicle", "cytosol"]:
        p = ParticlePlacer(
            volume=vol.clone(),
            density_cutoff=0.4,
            rng=torch.Generator().manual_seed(3),
        )
        spec = ParticleSpec(
            species_id=f"p_{loc}",
            density=small_density,
            max_count=5,
            location=loc,
            max_attempts_per_copy=30,
        )
        n_placed = p.place_species(spec, location_masks=masks, normal_fields=normals)
        assert n_placed > 0, f"expected at least one {loc} placement to succeed"
        for placed in p.placements:
            zi, yi, xi = (int(v) for v in placed.center_zyx.tolist())
            zi = min(max(zi, 0), masks[loc].shape[0] - 1)
            yi = min(max(yi, 0), masks[loc].shape[1] - 1)
            xi = min(max(xi, 0), masks[loc].shape[2] - 1)
            assert masks[loc][
                zi, yi, xi
            ], f"{loc}-flagged particle landed outside its mask"

    # Sanity: vesicle and cytosol masks don't overlap.
    assert not (masks["vesicle"] & masks["cytosol"]).any()


# ---------------------------------------------------------------------
# Cluster / bundle placement modes
# ---------------------------------------------------------------------


def test_cluster_mode_places_multiple_nonoverlapping_particles():
    torch.manual_seed(0)
    vol = torch.zeros((64, 64, 64))
    density = ball3d(9, 6.0)
    spec = ParticleSpec(
        species_id="clu",
        density=density,
        max_count=20,
        mode="cluster",
        cluster_size=8,
        max_attempts_per_copy=30,
    )
    placer = ParticlePlacer(
        volume=vol, density_cutoff=0.3, rng=torch.Generator().manual_seed(0)
    )
    placed = placer.run([spec])

    assert len(placed) > 1, "cluster mode should place a primary plus satellites"
    centers = torch.stack([p.center_zyx for p in placed])
    dists = torch.cdist(centers, centers)
    dists.fill_diagonal_(float("inf"))
    assert dists.min().item() >= 5.5  # no-overlap, same tolerance as single-mode test

    # Satellites should be loosely clustered around the primary, not spread
    # across the whole volume (a weak check: max pairwise distance should
    # be much smaller than the volume's own diagonal).
    assert dists[torch.isfinite(dists)].max().item() < 64.0


def test_bundle_mode_places_multiple_nonoverlapping_particles():
    torch.manual_seed(0)
    vol = torch.zeros((64, 64, 64))
    density = ball3d(9, 6.0)
    spec = ParticleSpec(
        species_id="bun",
        density=density,
        max_count=20,
        mode="bundle",
        bundle_size=8,
        max_attempts_per_copy=30,
    )
    placer = ParticlePlacer(
        volume=vol, density_cutoff=0.3, rng=torch.Generator().manual_seed(1)
    )
    placed = placer.run([spec])

    assert len(placed) > 1, "bundle mode should place a primary plus satellites"
    centers = torch.stack([p.center_zyx for p in placed])
    dists = torch.cdist(centers, centers)
    dists.fill_diagonal_(float("inf"))
    assert dists.min().item() >= 5.5


# ---------------------------------------------------------------------
# export_picks
# ---------------------------------------------------------------------


@pytest.mark.skipif(not _PDB_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_export_picks_before_generate_raises():
    gen = CryoTomoSimSpecimenGenerator(
        protein_specs=[ProteinSpec(pdb_source=str(_PDB_FIXTURE), max_count=1)],
        target_shape=(24, 32, 32),
        v_size=10.0,
        ice_opacity=0.0,
        seed=0,
    )
    with pytest.raises(RuntimeError):
        gen.export_picks("/tmp/should-not-be-created")


@pytest.mark.skipif(not _PDB_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_export_picks_writes_expected_ndjson_files(tmp_path):
    target_shape = (32, 48, 48)
    v_size = 10.0
    gen = CryoTomoSimSpecimenGenerator(
        protein_specs=[ProteinSpec(pdb_source=str(_PDB_FIXTURE), max_count=3)],
        membrane_specs=[MembraneSpec(count=1, size=100.0, thickness=30.0)],
        bead_spec=BeadSpec(radii=[40.0], count_per_radius=1),
        target_shape=target_shape,
        v_size=v_size,
        ice_opacity=0.0,
        seed=0,
    )
    gen.generate()

    written = gen.export_picks(tmp_path, oriented=True)

    protein_stem = Path(str(_PDB_FIXTURE)).stem
    assert set(written.keys()) == {protein_stem, "bead_40", "membrane"}
    for path in written.values():
        assert path.exists()
        assert path.parent == tmp_path

    # Physical extent (Angstrom) of the volume, in (x, y, z) order, per the
    # same voxel-index-(z,y,x)-equals-physical-(x,y,z)[::-1]/v_size
    # convention export_picks itself uses.
    nz, ny, nx = target_shape
    extent_xyz = (nx * v_size, ny * v_size, nz * v_size)
    margin = max(extent_xyz)  # generous, just checking "not wildly out of range"

    for name, path in written.items():
        lines = path.read_text().strip().splitlines()
        assert len(lines) > 0
        for line in lines:
            row = json.loads(line)
            assert "location" in row
            x, y, z = row["location"]["x"], row["location"]["y"], row["location"]["z"]
            assert -margin <= x <= extent_xyz[0] + margin
            assert -margin <= y <= extent_xyz[1] + margin
            assert -margin <= z <= extent_xyz[2] + margin

            if name == "membrane":
                assert row["type"] == "point"
                assert "xyz_rotation_matrix" not in row
            else:
                assert row["type"] == "orientedPoint"
                R = row["xyz_rotation_matrix"]
                assert len(R) == 3 and all(len(r) == 3 for r in R)


@pytest.mark.skipif(not _PDB_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_export_picks_unoriented_omits_rotation_matrix(tmp_path):
    gen = CryoTomoSimSpecimenGenerator(
        protein_specs=[ProteinSpec(pdb_source=str(_PDB_FIXTURE), max_count=2)],
        target_shape=(24, 32, 32),
        v_size=10.0,
        ice_opacity=0.0,
        seed=0,
    )
    gen.generate()
    written = gen.export_picks(tmp_path, oriented=False)

    protein_stem = Path(str(_PDB_FIXTURE)).stem
    path = written[protein_stem]
    assert path.name.endswith("_point.ndjson")
    for line in path.read_text().strip().splitlines():
        row = json.loads(line)
        assert row["type"] == "point"
        assert "xyz_rotation_matrix" not in row
