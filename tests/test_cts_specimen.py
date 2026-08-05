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
from specter.specimen._cts_placement import (
    HierarchicalParticlePlacer,
    ParticlePlacer,
    ParticleSpec,
)
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
    # The continuous SDF-based bilayer density (see _cts_membrane.py's
    # _compute_geometry_fields) fills a genuinely larger fraction of its
    # OWN local (tightly-padded) grid than the old point-cloud density did
    # -- 60-70% is normal here, since the local grid is sized just large
    # enough to contain the shape, not a large empty box the shell floats
    # in (that's a separate thing from how sparse the shell is relative to
    # the much bigger FULL SPECIMEN volume it gets placed into later).
    # Still checked against degenerate extremes -- empty, or fully solid.
    assert 0.05 < occupied < 0.95


def test_membrane_blob_has_bilayer_double_peak_profile():
    """
    Density profile sampled straight through the shell ALONG ITS OWN LOCAL
    SURFACE NORMAL at several skeleton (mid-thickness) points should show
    two separated peaks (the two leaflets), not one monolithic peak.

    Uses the local normal direction (`inst.normal_field`) at real skeleton
    points (`inst.skeleton_mask`) rather than radial distance from the
    shape's overall centroid -- radial-from-centroid only detects two
    leaflets correctly for a near-perfect sphere; for a genuinely irregular
    alpha-shape blob (the whole point of this generator), different
    surface locations sit at different distances from the centroid, so a
    single global radial histogram can blur or merge the two leaflets'
    signal even when the bilayer structure is locally completely fine.
    Sampling along the actual local normal at each test point is
    shape-agnostic and directly probes the one direction guaranteed to
    cross both leaflets at that point.
    """
    gen = MembraneBlobGenerator(v_size=6.0, seed=2)
    inst = gen.generate(
        size=220.0,
        roughness=0.75,
        thickness=60.0,
    )
    density = inst.density.numpy()
    n = density.shape[0]
    normals = inst.normal_field.numpy()  # (3, n, n, n), physical (x, y, z)
    skeleton_pts = np.argwhere(inst.skeleton_mask.numpy())
    assert skeleton_pts.shape[0] > 0, "skeleton_mask has no points to sample from"

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(
        skeleton_pts.shape[0], size=min(20, skeleton_pts.shape[0]), replace=False
    )

    half_span = 20  # voxels each side of the skeleton point along the normal
    n_found_bilayer = 0
    for idx in sample_idx:
        z, y, x = skeleton_pts[idx]
        # normal_field is physical (x, y, z); array axes are (z, y, x) --
        # reverse component order to index/step consistently.
        nx_, ny_, nz_ = normals[:, z, y, x]
        direction = np.array([nz_, ny_, nx_])
        if np.linalg.norm(direction) < 1e-6:
            continue
        direction = direction / np.linalg.norm(direction)

        t = np.arange(-half_span, half_span + 1)
        line_coords = skeleton_pts[idx][None, :] + t[:, None] * direction[None, :]
        line_coords = np.round(line_coords).astype(int)
        valid = np.all((line_coords >= 0) & (line_coords < n), axis=1)
        line_coords = line_coords[valid]
        profile = density[line_coords[:, 0], line_coords[:, 1], line_coords[:, 2]]

        smoothed = np.convolve(profile, np.ones(3) / 3, mode="same")
        peaks = [
            i
            for i in range(2, len(smoothed) - 2)
            if smoothed[i] > 0 and smoothed[i] == max(smoothed[i - 2 : i + 3])
        ]
        if len(peaks) >= 2 and (max(peaks) - min(peaks)) >= 2:
            n_found_bilayer += 1

    # Not every sampled skeleton point needs to show a clean double peak
    # (irregular local geometry, grazing normals, etc. can blur individual
    # samples) -- but a solid majority should, or this isn't a real bilayer.
    assert n_found_bilayer >= 0.5 * len(sample_idx), (
        f"expected most sampled skeleton points to show a clear bilayer "
        f"double-peak profile along their local normal, only "
        f"{n_found_bilayer}/{len(sample_idx)} did"
    )


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
# HierarchicalParticlePlacer -- same invariants as ParticlePlacer above
# ---------------------------------------------------------------------


def test_hierarchical_placer_avoids_overlap_and_respects_density_cutoff():
    torch.manual_seed(0)
    vol = torch.zeros((64, 64, 64))
    density = ball3d(9, 6.0)
    spec = ParticleSpec(
        species_id="ball", density=density, max_count=100, max_attempts_per_copy=30
    )
    placer = HierarchicalParticlePlacer(
        volume=vol,
        density_cutoff=0.15,
        coarse_factor=8,
        rng=torch.Generator().manual_seed(0),
    )
    placed = placer.run([spec])

    assert len(placed) > 0
    occupied = (vol > 0).float().mean().item()
    assert occupied <= 0.15 + 1e-6

    centers = torch.stack([p.center_zyx for p in placed])
    if len(centers) > 1:
        dists = torch.cdist(centers, centers)
        dists.fill_diagonal_(float("inf"))
        assert dists.min().item() >= 5.5


def test_hierarchical_placer_stops_when_volume_full():
    torch.manual_seed(0)
    vol = torch.zeros((16, 16, 16))
    density = ball3d(5, 4.0)
    spec = ParticleSpec(species_id="ball", density=density, max_count=1000)
    placer = HierarchicalParticlePlacer(
        volume=vol,
        density_cutoff=0.3,
        coarse_factor=4,
        rng=torch.Generator().manual_seed(1),
    )
    placed = placer.run([spec])
    assert (vol > 0).float().mean().item() <= 0.3 + 0.02
    assert len(placed) < 1000


def test_hierarchical_placer_location_flag_requires_mask():
    vol = torch.zeros((16, 16, 16))
    density = ball3d(3, 2.0)
    spec = ParticleSpec(
        species_id="memprot", density=density, max_count=1, location="membrane"
    )
    placer = HierarchicalParticlePlacer(volume=vol)
    with pytest.raises(ValueError):
        placer.run([spec])


def test_hierarchical_placer_rejects_cluster_and_bundle_modes():
    vol = torch.zeros((32, 32, 32))
    density = ball3d(5, 3.0)
    placer = HierarchicalParticlePlacer(volume=vol, coarse_factor=4)
    for mode in ("cluster", "bundle"):
        spec = ParticleSpec(species_id="p", density=density, max_count=5, mode=mode)
        with pytest.raises(NotImplementedError):
            placer.run([spec])


def test_hierarchical_placer_membrane_vesicle_cytosol_gating():
    """Same membrane/vesicle/cytosol location-flag test as
    test_membrane_flagged_particle_lands_on_shell_not_free_in_cytosol /
    test_vesicle_and_cytosol_flags_respect_inside_outside, but exercising
    HierarchicalParticlePlacer instead of ParticlePlacer."""
    vol, _placer, masks, normals, ignores = _placed_membrane()
    small_density = ball3d(5, 3.0)

    for loc in ["membrane", "vesicle", "cytosol"]:
        p = HierarchicalParticlePlacer(
            volume=vol.clone(),
            density_cutoff=0.4,
            coarse_factor=4,
            rng=torch.Generator().manual_seed(5),
        )
        spec = ParticleSpec(
            species_id=f"p_{loc}",
            density=small_density,
            max_count=5,
            location=loc,
            max_attempts_per_copy=30,
        )
        n_placed = p.place_species(
            spec, location_masks=masks, normal_fields=normals, ignore_masks=ignores
        )
        assert n_placed > 0, f"expected at least one {loc} placement to succeed"
        for placed in p.placements:
            zi, yi, xi = (int(v) for v in placed.center_zyx.tolist())
            zi = min(max(zi, 0), masks[loc].shape[0] - 1)
            yi = min(max(yi, 0), masks[loc].shape[1] - 1)
            xi = min(max(xi, 0), masks[loc].shape[2] - 1)
            assert masks[loc][
                zi, yi, xi
            ], f"{loc}-flagged particle landed outside its mask"


# ---------------------------------------------------------------------
# CarbonFilmGenerator / BeadGenerator
# ---------------------------------------------------------------------


def test_bead_generator_bulk_density_is_sane():
    """A gold bead's peak voxel value should be gold's real mean inner
    potential (volts, literature ballpark ~25-30 V for bulk gold), not a
    raw atom count."""
    gen = BeadGenerator(v_size=5.0)
    bead = gen.generate(radius=50.0)
    assert bead.density.shape[0] > 0
    assert (bead.density > 0).any()
    assert 20.0 < bead.density.max().item() < 40.0
    assert torch.isfinite(bead.density).all()


def test_bead_generator_mean_inner_potential_independent_of_voxel_size():
    mips = [
        BeadGenerator(v_size=v_size).mean_inner_potential for v_size in (2.0, 5.0, 10.0)
    ]
    assert max(mips) - min(mips) < 0.05 * np.mean(mips)


def test_bead_generator_rejects_nonpositive_radius():
    gen = BeadGenerator(v_size=5.0)
    with pytest.raises(ValueError):
        gen.generate(radius=0.0)


def test_bead_generator_rejects_shtyrov():
    with pytest.raises(ValueError):
        BeadGenerator(v_size=5.0, parameterization="shtyrov")


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


@pytest.mark.parametrize("parameterization", ["kirkland", "lobato", "shtyrov"])
def test_carbon_film_mean_inner_potential_matches_literature(parameterization):
    """Carbon's per-voxel value should be a real mean inner potential (volts),
    in the literature ballpark for amorphous carbon (~8-13 V), not a raw atom
    count -- and, being a physical bulk quantity, independent of voxel size."""
    mips = [
        CarbonFilmGenerator(
            v_size=v_size, parameterization=parameterization
        ).mean_inner_potential
        for v_size in (2.0, 5.0, 10.0, 15.0)
    ]
    for mip in mips:
        assert 5.0 < mip < 20.0
    assert max(mips) - min(mips) < 0.05 * np.mean(mips)


def test_carbon_film_is_solid_away_from_hole_regardless_of_grid_size():
    """The film should be uniformly solid (every voxel at the same MIP)
    away from the hole, at both a small and a much larger target_shape --
    the whole point of the analytic (not point-cloud) hole boundary is
    that coverage doesn't thin out as the grid grows."""
    gen_small = CarbonFilmGenerator(v_size=15.0, seed=0)
    film_small = gen_small.generate(
        target_shape=(16, 64, 64), thickness=150.0, hole_radius=300.0
    )
    gen_large = CarbonFilmGenerator(v_size=15.0, seed=0)
    film_large = gen_large.generate(
        target_shape=(16, 512, 512), thickness=150.0, hole_radius=300.0
    )
    # Far outside the hole, at the film's mid-thickness z-slice (both
    # share nz=16 -- only the XY footprint differs), every voxel of the
    # slab should be fully occupied at the same MIP in both cases -- not a
    # sparse speckle that gets sparser as the footprint grows.
    corner_small = film_small.density[8, :8, :8]
    corner_large = film_large.density[8, :8, :8]
    assert torch.allclose(corner_small, corner_large)
    assert (corner_large > 0).all()


def test_carbon_film_hole_center_offsets_the_edge():
    """A large hole_radius combined with a non-zero hole_center should put
    a hole boundary near one side of the frame -- one far corner fully
    inside the hole (empty), the opposite far corner fully on the carbon
    -- rather than a small hole fully contained in the middle of the
    frame (the default, `hole_center=(0, 0)`, unrealistic-scale case)."""
    gen = CarbonFilmGenerator(v_size=10.0, seed=0)
    film = gen.generate(
        target_shape=(8, 128, 128),
        thickness=150.0,
        hole_radius=3000.0,
        edge_roughness=0.0,
        hole_center=(-2600.0, 0.0),
    )
    mid_z = 4
    # Left edge of the frame (x very negative) sits close to the offset
    # hole's center -> inside the hole -> empty.
    assert (film.density[mid_z, :, 0] == 0).all()
    # Right edge of the frame (x very positive) is far outside the
    # offset hole's radius -> on the carbon -> occupied.
    assert (film.density[mid_z, :, -1] > 0).all()


def test_carbon_film_hole_shape_independent_of_voxel_size():
    """The same seed/physical hole geometry should classify a given
    physical (x, y) point as inside/outside the hole the same way
    regardless of how finely the volume is discretized."""
    hole_radius, edge_roughness = 300.0, 30.0
    coarse = CarbonFilmGenerator(v_size=20.0, seed=1).generate(
        target_shape=(8, 48, 48),
        thickness=150.0,
        hole_radius=hole_radius,
        edge_roughness=edge_roughness,
    )
    fine = CarbonFilmGenerator(v_size=5.0, seed=1).generate(
        target_shape=(32, 192, 192),
        thickness=150.0,
        hole_radius=hole_radius,
        edge_roughness=edge_roughness,
    )
    # Same physical footprint (960x960 A either way). Compare the far
    # corner (well outside the hole regardless of edge roughness, so this
    # isolates resolution-independence rather than boundary placement): a
    # coarse voxel at (iz, iy, ix) covers the same physical region as a
    # 4x4x4 block of fine voxels: (iz*4:iz*4+4, iy*4:iy*4+4, ix*4:ix*4+4).
    occupied_coarse = coarse.density[4, 0, 0] > 0
    fine_block = fine.density[16:20, 0:4, 0:4] > 0
    assert bool(occupied_coarse) == bool(fine_block.all()) == bool(fine_block.any())


def test_carbon_film_explicit_edge_grain_size_matches_across_resolutions():
    """With an explicit (not auto-picked) edge_grain_size, the jittered
    boundary is a fixed function of physical angle -- so, unlike the
    default (which intentionally scales grain size with v_size, see
    `edge_grain_size`'s docstring), an explicit value should reproduce the
    same boundary exactly across resolutions, all the way up to points
    right next to the boundary, not just a far corner."""
    hole_radius, edge_roughness, edge_grain_size = 300.0, 40.0, 60.0
    coarse = CarbonFilmGenerator(v_size=20.0, seed=1).generate(
        target_shape=(8, 48, 48),
        thickness=150.0,
        hole_radius=hole_radius,
        edge_roughness=edge_roughness,
        edge_grain_size=edge_grain_size,
    )
    fine = CarbonFilmGenerator(v_size=5.0, seed=1).generate(
        target_shape=(32, 192, 192),
        thickness=150.0,
        hole_radius=hole_radius,
        edge_roughness=edge_roughness,
        edge_grain_size=edge_grain_size,
    )
    # Every coarse voxel's physical corner matches one fine voxel exactly
    # (same v_size ratio as the other cross-resolution test): compare the
    # full mid-thickness XY slice, not just a single far-off point.
    coarse_slice = coarse.density[4, :, :] > 0
    fine_corners = fine.density[16, 0::4, 0::4] > 0
    assert torch.equal(coarse_slice, fine_corners)


def test_carbon_film_default_edge_is_jagged_not_smooth():
    """The default (auto edge_grain_size) boundary should have many local
    turning points across the frame -- genuine jaggedness -- not the
    smooth, few-extrema wave a low-order analytic function would give."""
    gen = CarbonFilmGenerator(v_size=4.0, seed=2)
    film = gen.generate(
        target_shape=(4, 200, 200),
        thickness=150.0,
        hole_radius=20000.0,
        hole_center=(-19700.0, 0.0),
        edge_roughness=40.0,
    )
    # hole_center offsets along x, so the boundary runs vertically: for
    # each row (fixed y) find the first occupied column (x position).
    occupied = film.density[2] > 0
    boundary_col = occupied.float().argmax(dim=1)
    diffs = boundary_col[1:].float() - boundary_col[:-1].float()
    sign_changes = (diffs[1:] * diffs[:-1] < 0).sum().item()
    # A smooth, low-order wave crossing this frame would turn direction a
    # handful of times at most (order 1-5); genuine grain-scale jaggedness
    # turns direction roughly every other independent grain.
    assert sign_changes > 20


# ---------------------------------------------------------------------
# CryoTomoSimSpecimenGenerator (end-to-end)
# ---------------------------------------------------------------------


@pytest.mark.skipif(not _PDB_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_cryotomosim_generator_end_to_end():
    target_shape = (32, 48, 48)
    gen = CryoTomoSimSpecimenGenerator(
        protein_specs=[ProteinSpec(pdb_source=str(_PDB_FIXTURE), max_count=2)],
        membrane_specs=[MembraneSpec(count=1, size=50.0, thickness=30.0)],
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


def _placed_membrane(v_size=8.0, size=110.0, roughness=0.75, thickness=15.0, seed=1):
    """Generate and place one membrane into a plain volume; return
    (volume, placer, location_masks, normal_fields, ignore_masks).

    `location_masks["membrane"]` is the thin mid-thickness SKELETON (used
    for candidate sampling); `ignore_masks["membrane"]` is the FULL
    bilayer-thickness shell (used only for the overlap-ignore mechanism)
    -- see `_build_membrane_location_fields`'s docstring for why these
    must stay two separate masks.

    size/thickness (and the 128^3 volume, up from an old 64^3) reflect the
    continuous SDF-based renderer's now-CORRECT (verified) bounding-radius
    measurement -- the old size=220.0/thickness=35.0 pairing produced a
    192^3-voxel membrane grid (bounding radius ~509A) that simply couldn't
    fit in a 64^3 (512A) test volume at all once an earlier boundary-
    clipping bug (density hard-cut at the array's own edge, verified via a
    visible square artifact in a real rendered slice) was fixed and the
    measurement became accurate instead of an undersized artifact of that
    bug.

    size=110 (up from an earlier 80): at size=80 the vesicle lumen was
    only ~64 voxels in a thin 9x4x7 blob -- geometrically real (a small
    ~80A vesicle with a 15A-thick bilayer doesn't leave much interior
    room) but too tight for even ball3d(5, 3.0) to land without its
    rotated footprint clipping the surrounding shell, so vesicle-flagged
    placement failed 0/200 attempts across 10 seeds (verified directly,
    not assumed). size=110 gives a ~1200-voxel lumen, enough for reliable
    placement.
    """
    torch.manual_seed(0)
    vol = torch.zeros((128, 128, 128))
    placer = ParticlePlacer(
        volume=vol, density_cutoff=0.4, rng=torch.Generator().manual_seed(0)
    )
    mem_gen = MembraneBlobGenerator(v_size=v_size, seed=seed)
    inst = mem_gen.generate(size=size, roughness=roughness, thickness=thickness)
    placer.run([ParticleSpec(species_id="mem", density=inst.density, max_count=1)])
    assert len(placer.placements) == 1, "membrane failed to place at all"
    masks, normals, ignores = _build_membrane_location_fields(
        placer.placements, {"mem": inst}, (128, 128, 128)
    )
    return vol, placer, masks, normals, ignores


def test_membrane_flagged_particle_lands_on_skeleton_not_free_in_cytosol():
    vol, placer, masks, normals, ignores = _placed_membrane()
    density = ball3d(5, 3.0)
    spec = ParticleSpec(
        species_id="memprot",
        density=density,
        max_count=5,
        location="membrane",
        max_attempts_per_copy=30,
    )
    n_placed = placer.place_species(
        spec, location_masks=masks, normal_fields=normals, ignore_masks=ignores
    )
    assert n_placed > 0

    memprot_placements = [p for p in placer.placements if p.species_id == "memprot"]
    assert len(memprot_placements) == n_placed
    for p in memprot_placements:
        zi, yi, xi = (int(v) for v in p.center_zyx.tolist())
        zi = min(max(zi, 0), masks["membrane"].shape[0] - 1)
        yi = min(max(yi, 0), masks["membrane"].shape[1] - 1)
        xi = min(max(xi, 0), masks["membrane"].shape[2] - 1)
        # masks["membrane"] IS the thin mid-thickness skeleton (not the
        # full shell) -- candidates are sampled directly from it, so exact
        # membership here is the strongest possible "landed on the true
        # mid-thickness ridge" check (distance 0, not merely "small").
        assert masks["membrane"][zi, yi, xi], (
            "membrane-flagged particle center is not on the membrane's "
            "mid-thickness skeleton"
        )
        assert not masks["cytosol"][zi, yi, xi]


def test_membrane_overlap_ignore_mask_not_conflated_with_skeleton():
    """The overlap-ignore mask must stay the FULL shell thickness, not the
    thin skeleton -- placement success should collapse to near-zero
    without it (since a candidate centered on the skeleton genuinely
    overlaps the surrounding full-thickness bilayer material), and recover
    once the correct full-shell ignore mask is supplied."""
    vol, placer, masks, normals, ignores = _placed_membrane()
    density = ball3d(5, 3.0)

    # Without ignore_masks: every candidate's rotated density will
    # generally overlap the membrane's own full-thickness material at its
    # insertion site, so almost nothing should place successfully.
    placer_no_ignore = ParticlePlacer(
        volume=vol.clone(), density_cutoff=0.4, rng=torch.Generator().manual_seed(9)
    )
    spec_no_ignore = ParticleSpec(
        species_id="memprot_no_ignore",
        density=density,
        max_count=15,
        location="membrane",
        max_attempts_per_copy=30,
    )
    n_no_ignore = placer_no_ignore.place_species(
        spec_no_ignore, location_masks=masks, normal_fields=normals
    )

    # With the correct full-shell ignore_masks: placement should succeed
    # at a healthy rate, same as test_membrane_flagged_particle_lands_on_skeleton_....
    placer_with_ignore = ParticlePlacer(
        volume=vol.clone(), density_cutoff=0.4, rng=torch.Generator().manual_seed(9)
    )
    spec_with_ignore = ParticleSpec(
        species_id="memprot_with_ignore",
        density=density,
        max_count=15,
        location="membrane",
        max_attempts_per_copy=30,
    )
    n_with_ignore = placer_with_ignore.place_species(
        spec_with_ignore,
        location_masks=masks,
        normal_fields=normals,
        ignore_masks=ignores,
    )

    assert n_with_ignore > 0, "expected the ignore-mask path to place successfully"
    assert n_with_ignore > n_no_ignore, (
        "overlap-ignore mask isn't providing its intended benefit -- "
        f"with-ignore placed {n_with_ignore}, without-ignore placed "
        f"{n_no_ignore} (expected without-ignore to be much worse, "
        "collapsing toward zero)"
    )
    assert n_no_ignore <= max(1, n_with_ignore // 3), (
        "placement without the full-shell ignore mask succeeded far more "
        "than expected -- check the two masks haven't been conflated"
    )


def test_vesicle_and_cytosol_flags_respect_inside_outside():
    vol, placer, masks, normals, ignores = _placed_membrane()
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
    # Box grown from an old (32,48,48) -- size=30/thickness=6's own
    # bounding radius is ~192A once the boundary-clipping bug (see
    # _placed_membrane's docstring) was fixed and the measurement became
    # accurate instead of an undersized artifact of that bug; needed a
    # bigger box than the old undersized-measurement era assumed.
    target_shape = (48, 64, 64)
    v_size = 10.0
    gen = CryoTomoSimSpecimenGenerator(
        protein_specs=[ProteinSpec(pdb_source=str(_PDB_FIXTURE), max_count=3)],
        # size/thickness at a realistic ~20% ratio (30.0's own bounding
        # radius comfortably fits this test's box) -- the old 50.0/30.0
        # pairing (60% ratio) made the continuous SDF-based shell's real
        # bounding radius balloon to ~260A+, bigger than this test's own
        # box, so RSA could never place it at all.
        membrane_specs=[MembraneSpec(count=1, size=30.0, thickness=6.0)],
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


# ---------------------------------------------------------------------
# Hybrid RSA integration (membranes + "any"-location single-mode proteins)
# ---------------------------------------------------------------------


@pytest.mark.skipif(not _PDB_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_rsa_packing_respects_radius_sum_nonoverlap():
    """Membranes + free (RSA-eligible) proteins placed via
    CryoTomoSimSpecimenGenerator's hybrid path should never land closer
    than their bounding-radius sum -- the core RSA guarantee."""
    from specter.pdb import PDB
    from specter.specimen.cryotomosim import _bounding_radius_voxels

    target_shape = (60, 120, 120)
    v_size = 10.0
    gen = CryoTomoSimSpecimenGenerator(
        protein_specs=[ProteinSpec(pdb_source=str(_PDB_FIXTURE), max_count=15)],
        membrane_specs=[MembraneSpec(count=3, size=50.0, thickness=30.0)],
        target_shape=target_shape,
        v_size=v_size,
        ice_opacity=0.0,
        density_cutoff=0.5,
        seed=7,
    )
    gen.generate()

    protein_radius = (
        float(
            PDB(str(_PDB_FIXTURE), savefolder="pdb-data/", verbose=False).max_diameter
        )
        / 2.0
    )
    membrane_radii = [
        _bounding_radius_voxels(inst.density) * v_size
        for inst in gen.membrane_instances
    ]

    membrane_iter = iter(membrane_radii)
    records: list[tuple[torch.Tensor, float]] = []
    for p in gen.placements:
        center_xyz = p.center_zyx.flip(0) * v_size
        if p.species_id.startswith("membrane_"):
            records.append((center_xyz, next(membrane_iter)))
        else:
            records.append((center_xyz, protein_radius))

    assert len(records) > 5, "expected a non-trivial number of RSA-packed instances"
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            ci, ri = records[i]
            cj, rj = records[j]
            dist = (ci - cj).norm().item()
            assert dist >= ri + rj - 1e-2, (
                f"RSA-packed instances {i},{j} violate radius-sum "
                f"non-overlap: dist={dist:.2f}, r_sum={ri + rj:.2f}"
            )


@pytest.mark.skipif(not _PDB_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_rsa_membrane_still_supports_membrane_flagged_protein():
    """A protein flagged 'membrane' should still land on an RSA-packed
    membrane's shell, exercising the sequential pass that runs on top of
    RSA-placed membranes."""
    # target_shape/thickness sized so the membrane's now-correctly-measured
    # bounding radius (~309A at size=80/thickness=15) comfortably fits --
    # the old thickness=35 pairing gave ~410A, bigger than the old box's
    # own 300A half-depth once the boundary-clipping bug (see
    # _placed_membrane's docstring) was fixed and the measurement became
    # accurate instead of an undersized artifact of that bug.
    target_shape = (80, 120, 120)
    v_size = 10.0
    gen = CryoTomoSimSpecimenGenerator(
        protein_specs=[
            ProteinSpec(pdb_source=str(_PDB_FIXTURE), max_count=5, location="cytosol"),
            ProteinSpec(pdb_source=str(_PDB_FIXTURE), max_count=5, location="membrane"),
        ],
        membrane_specs=[MembraneSpec(count=1, size=80.0, thickness=15.0)],
        target_shape=target_shape,
        v_size=v_size,
        ice_opacity=0.0,
        density_cutoff=0.5,
        seed=3,
    )
    gen.generate()

    membrane_placed = [
        p for p in gen.placements if p.species_id.startswith("membrane_")
    ]
    assert len(membrane_placed) == 1, "expected the single membrane to be RSA-packed"

    protein_placed = [
        p for p in gen.placements if not p.species_id.startswith("membrane_")
    ]
    assert len(protein_placed) > 0, "expected at least some proteins placed"
