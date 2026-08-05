"""
Smoke tests for SpherePackingSpecimenGenerator
(specter.specimen.packing.pdb_packing) -- the hard-sphere specimen
generator, promoted from dev/packing_algorithms.py's algorithm comparison.
Uses locally-cached PDB fixtures (no network fetch).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from specter.specimen.packing import (
    SpherePackingSpecimenGenerator,
    SphereProteinSpec,
    draw_species_pool,
    pack_hard_spheres_3d,
    pack_hard_spheres_3d_dense,
)
from specter.specimen.packing.algorithms import _fb_verify_no_overlap

_SMALL_FIXTURE = Path(__file__).parent.parent / "pdb-data" / "1mbo.cif"
_LARGE_FIXTURE = Path(__file__).parent.parent / "pdb-data" / "1bxn-assembly1.cif"


def test_pack_hard_spheres_3d_is_overlap_free():
    torch.manual_seed(0)
    radii = torch.cat(
        [torch.full((10,), 40.0), torch.full((20,), 20.0), torch.full((40,), 10.0)]
    )
    box = (300.0, 300.0, 300.0)
    gap = 5.0
    coords, idx = pack_hard_spheres_3d(radii, box, gap=gap, seed=0, device="cpu")

    assert coords.shape[0] == idx.shape[0]
    assert coords.shape[0] > 0

    r = radii[idx]
    diffs = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist = diffs.norm(dim=-1)
    contact = r.unsqueeze(0) + r.unsqueeze(1) + gap
    overlap = contact - dist
    overlap.fill_diagonal_(-float("inf"))
    assert bool((overlap <= 1e-2).all())

    half = torch.tensor([150.0, 150.0, 150.0])
    assert bool((coords.abs() + r.unsqueeze(1) <= half + 1e-3).all())


def test_pack_hard_spheres_3d_empty_input():
    coords, idx = pack_hard_spheres_3d(
        torch.empty((0,)), (100.0, 100.0, 100.0), device="cpu"
    )
    assert coords.shape == (0, 3)
    assert idx.shape == (0,)


@pytest.mark.skipif(not _SMALL_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_sphere_packing_generator_filler_only():
    target_shape = (32, 48, 48)
    gen = SpherePackingSpecimenGenerator(
        target_specs=[],
        filler_specs=[SphereProteinSpec(pdb_source=str(_SMALL_FIXTURE))],
        target_shape=target_shape,
        v_size=10.0,
        filler_occupancy_fraction=0.15,
        gap_angstrom=5.0,
        seed=0,
    )
    volume = gen.generate()

    assert volume.shape == target_shape
    assert not torch.isnan(volume).any()
    assert not torch.isinf(volume).any()
    # PotentialBuilder's FFT-based convolution leaves negligible negative
    # ringing even in a single unrotated template (~1e-7 relative to peak,
    # confirmed independently of this generator) -- summing several
    # instances (the physically correct way to combine scattering
    # potentials, matching crowding.py's crowd_with_duplicates) doesn't
    # magnify it into anything meaningful, so check magnitude, not sign.
    assert volume.max().item() > 0
    assert volume.min().item() > -1e-3 * volume.max().item()
    assert len(gen.placements) > 0
    assert gen.n_candidates >= len(gen.placements)


@pytest.mark.skipif(
    not (_SMALL_FIXTURE.exists() and _LARGE_FIXTURE.exists()),
    reason="bundled PDB fixtures missing",
)
def test_sphere_packing_generator_filler_multi_species():
    """Two differently-sized filler species, equal attempt-weight -- both
    should end up represented among the placements."""
    target_shape = (32, 64, 64)
    gen = SpherePackingSpecimenGenerator(
        target_specs=[],
        filler_specs=[
            SphereProteinSpec(pdb_source=str(_SMALL_FIXTURE)),
            SphereProteinSpec(pdb_source=str(_LARGE_FIXTURE)),
        ],
        target_shape=target_shape,
        v_size=10.0,
        filler_occupancy_fraction=0.15,
        gap_angstrom=5.0,
        seed=0,
    )
    volume = gen.generate()

    assert volume.shape == target_shape
    assert volume.max().item() > 0

    species_ids = {p.species_id for p in gen.placements}
    assert str(_SMALL_FIXTURE) in species_ids
    assert str(_LARGE_FIXTURE) in species_ids


@pytest.mark.skipif(
    not (_SMALL_FIXTURE.exists() and _LARGE_FIXTURE.exists()),
    reason="bundled PDB fixtures missing",
)
def test_sphere_packing_generator_targets_exact_count_and_filler_avoids_them():
    """Targets are placed at an exact n_copies count, and filler -- packed
    afterward via the exclusion field -- must not overlap any of them."""
    target_shape = (32, 64, 64)
    gap = 5.0
    gen = SpherePackingSpecimenGenerator(
        target_specs=[SphereProteinSpec(pdb_source=str(_LARGE_FIXTURE), n_copies=3)],
        filler_specs=[SphereProteinSpec(pdb_source=str(_SMALL_FIXTURE))],
        target_shape=target_shape,
        v_size=10.0,
        filler_occupancy_fraction=0.1,
        gap_angstrom=gap,
        seed=0,
    )
    gen.generate()

    assert gen.n_target_requested == 3
    assert gen.n_targets_placed == 3  # box is roomy enough to fit all 3
    target_placements = [
        p for p in gen.placements if p.species_id == str(_LARGE_FIXTURE)
    ]
    filler_placements = [
        p for p in gen.placements if p.species_id == str(_SMALL_FIXTURE)
    ]
    assert len(target_placements) == 3
    assert len(filler_placements) > 0

    from specter.pdb import PDB

    target_radius = PDB(str(_LARGE_FIXTURE), verbose=False).max_diameter / 2.0
    filler_radius = PDB(str(_SMALL_FIXTURE), verbose=False).max_diameter / 2.0
    for t in target_placements:
        for f in filler_placements:
            dist = float((t.position_xyz - f.position_xyz).norm())
            assert dist >= target_radius + filler_radius + gap - 1e-1


def test_draw_species_pool_respects_ratio_and_sorts_largest_first():
    species_radii = torch.tensor([5.0, 20.0])
    species_ratios = torch.tensor([3.0, 1.0])
    # small box_volume/large occupancy_fraction => many draws, so the drawn
    # mix should land close to the 3:1 ratio (weak law of large numbers).
    occupancy_fraction, box_volume = 50_000.0, 1000.0
    radii, species_idx = draw_species_pool(
        species_radii, species_ratios, occupancy_fraction, box_volume, seed=0
    )
    assert radii.numel() == species_idx.numel() > 100
    assert bool((radii[:-1] >= radii[1:]).all())  # largest-first sort

    frac_small = float((species_idx == 0).float().mean())
    assert 0.6 < frac_small < 0.9  # expected 0.75, allow sampling noise

    accumulated = float((4.0 / 3.0 * torch.pi * radii**3).sum())
    assert accumulated >= occupancy_fraction * box_volume


def test_draw_species_pool_empty_when_occupancy_fraction_zero():
    radii, species_idx = draw_species_pool(
        torch.tensor([10.0]),
        torch.tensor([1.0]),
        occupancy_fraction=0.0,
        box_volume=1000.0,
    )
    assert radii.numel() == 0
    assert species_idx.numel() == 0


def test_pack_hard_spheres_3d_dense_is_overlap_free_and_inside_box():
    box = (120.0, 120.0, 120.0)
    gap = 2.0
    coords, radii_out, species_idx_out = pack_hard_spheres_3d_dense(
        torch.tensor([10.0]),
        torch.tensor([1.0]),
        occupancy_fraction=0.15,
        box=box,
        gap=gap,
        seed=0,
        pad_fraction=1.0,
        n_stages=3,
        iterations_per_stage=15,
    )

    assert coords.shape[0] == radii_out.shape[0] == species_idx_out.shape[0]
    assert coords.shape[0] > 0
    assert _fb_verify_no_overlap(coords, radii_out, gap=gap, box=None)

    half = torch.tensor([60.0, 60.0, 60.0])
    assert bool((coords.abs() + radii_out.unsqueeze(1) <= half + 1e-3).all())


def test_pack_hard_spheres_3d_dense_empty_input():
    coords, radii_out, species_idx_out = pack_hard_spheres_3d_dense(
        torch.tensor([10.0]),
        torch.tensor([1.0]),
        occupancy_fraction=0.0,
        box=(100.0, 100.0, 100.0),
        device="cpu",
    )
    assert coords.shape == (0, 3)
    assert radii_out.shape == (0,)
    assert species_idx_out.shape == (0,)


def test_pack_hard_spheres_3d_dense_warns_when_species_radius_large_relative_to_box():
    with pytest.warns(UserWarning, match="disproportionately discard"):
        pack_hard_spheres_3d_dense(
            torch.tensor([31.4, 67.3]),
            torch.tensor([3.0, 1.0]),
            occupancy_fraction=0.15,
            box=(320.0, 640.0, 640.0),
            gap=5.0,
            seed=0,
            pad_fraction=0.5,
            n_stages=1,
            iterations_per_stage=1,  # correctness of the result doesn't matter here
        )


@pytest.mark.skipif(not _SMALL_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_sphere_packing_generator_dense_method_produces_valid_instance_labels():
    target_shape = (24, 32, 32)
    gen = SpherePackingSpecimenGenerator(
        target_specs=[],
        filler_specs=[SphereProteinSpec(pdb_source=str(_SMALL_FIXTURE))],
        target_shape=target_shape,
        v_size=10.0,
        filler_occupancy_fraction=0.1,
        gap_angstrom=5.0,
        seed=0,
        packing_method="dense",
        pad_fraction=1.0,
    )
    volume = gen.generate()

    assert volume.shape == target_shape
    assert len(gen.placements) > 0
    _assert_instance_labels_match_placements(gen, target_shape)


@pytest.mark.skipif(not _SMALL_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_sphere_packing_generator_rsa_method_produces_valid_instance_labels():
    target_shape = (32, 48, 48)
    gen = SpherePackingSpecimenGenerator(
        target_specs=[],
        filler_specs=[SphereProteinSpec(pdb_source=str(_SMALL_FIXTURE))],
        target_shape=target_shape,
        v_size=10.0,
        filler_occupancy_fraction=0.15,
        gap_angstrom=5.0,
        seed=0,
        packing_method="rsa",
    )
    gen.generate()

    assert len(gen.placements) > 0
    _assert_instance_labels_match_placements(gen, target_shape)


def _assert_instance_labels_match_placements(
    gen: SpherePackingSpecimenGenerator, target_shape: tuple[int, int, int]
) -> None:
    """Shared checks for the `instance_labels` mask: right shape/dtype, one
    label id per placement (no more, no fewer), background stays 0, and
    every placement's id actually appears somewhere in the volume."""
    labels = gen.instance_labels
    assert labels is not None
    assert labels.shape == target_shape
    assert labels.dtype == torch.int32
    assert bool((labels >= 0).all())

    placement_ids = {p.instance_id for p in gen.placements}
    assert len(placement_ids) == len(gen.placements)  # every id unique

    present_ids = set(torch.unique(labels[labels > 0]).tolist())
    assert present_ids == placement_ids


def test_sphere_packing_generator_rejects_unknown_packing_method():
    with pytest.raises(ValueError, match="packing_method"):
        SpherePackingSpecimenGenerator(
            target_specs=[],
            filler_specs=[SphereProteinSpec(pdb_source="1abc")],
            packing_method="bogus",  # type: ignore[arg-type]
        )


def test_sphere_packing_generator_rejects_target_without_n_copies():
    with pytest.raises(ValueError, match="n_copies"):
        SpherePackingSpecimenGenerator(
            target_specs=[SphereProteinSpec(pdb_source="1abc")],
        )


def test_sphere_packing_generator_rejects_dense_with_targets_and_filler():
    with pytest.raises(ValueError, match="obstacle-avoidance"):
        SpherePackingSpecimenGenerator(
            target_specs=[SphereProteinSpec(pdb_source="1abc", n_copies=1)],
            filler_specs=[SphereProteinSpec(pdb_source="1def")],
            packing_method="dense",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not _SMALL_FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_sphere_packing_generator_device_only_moves_potential_builder():
    """device='cuda' must route only to PotentialBuilder's potential build
    (the real compute cost); packing stays on CPU (vesin's neighbor list is
    slower and OOM-prone on GPU at realistic particle counts), and the
    final volume always lands back on CPU regardless of device."""
    gen = SpherePackingSpecimenGenerator(
        target_specs=[SphereProteinSpec(pdb_source=str(_SMALL_FIXTURE), n_copies=3)],
        target_shape=(24, 32, 32),
        v_size=10.0,
        device="cuda",
        seed=0,
        progressbars=False,
    )
    volume = gen.generate()

    assert volume.device.type == "cpu"
    assert len(gen.placements) == 3
    assert gen.instance_labels is not None
    assert gen.instance_labels.device.type == "cpu"


def test_pack_hard_spheres_3d_clip_axes_allows_poking_past_wall_on_clippable_axes():
    box = (60.0, 200.0, 200.0)  # thin in z, roomy in y/x
    gap = 2.0
    radii = torch.full((200,), 20.0)
    half = torch.tensor([100.0, 100.0, 30.0])  # x, y, z

    coords_default, _ = pack_hard_spheres_3d(radii, box, gap=gap, seed=0, device="cpu")
    assert bool((coords_default.abs() + 20.0 <= half + 1e-3).all())

    coords_clip, idx_clip = pack_hard_spheres_3d(
        radii, box, gap=gap, seed=0, device="cpu", clip_axes=(False, True, True)
    )
    r_clip = radii[idx_clip]
    # z (non-clippable) must still fully contain every sphere
    assert bool((coords_clip[:, 2].abs() + r_clip <= half[2] + 1e-3).all())
    # centers always stay in-bounds, even on clippable axes
    assert bool((coords_clip[:, :2].abs() <= half[:2] + 1e-3).all())
    # allowing xy clipping should place at least as many spheres as the
    # strict default (more candidates survive the relaxed xy wall check)
    assert coords_clip.shape[0] >= coords_default.shape[0]
    # and at least one of them actually needed the relaxation (extends past
    # the xy wall) -- otherwise this test isn't exercising the new behavior
    assert bool(
        ((coords_clip[:, :2].abs() + r_clip.unsqueeze(1)) > half[:2] + 1e-3).any()
    )


def test_pack_hard_spheres_3d_dense_clip_axes_allows_poking_past_wall_on_clippable_axes():
    box = (320.0, 640.0, 640.0)
    gap = 5.0
    half = torch.tensor([320.0, 320.0, 160.0])  # x, y, z

    coords, radii_out, _ = pack_hard_spheres_3d_dense(
        torch.tensor([31.4, 67.3]),
        torch.tensor([3.0, 1.0]),
        occupancy_fraction=0.15,
        box=box,
        gap=gap,
        seed=0,
        pad_fraction=0.5,
        n_stages=3,
        iterations_per_stage=15,
        clip_axes=(False, True, True),
    )
    assert coords.shape[0] > 0
    assert bool((coords[:, 2].abs() + radii_out <= half[2] + 1e-3).all())
    assert bool((coords[:, :2].abs() <= half[:2] + 1e-3).all())
    assert bool(
        ((coords[:, :2].abs() + radii_out.unsqueeze(1)) > half[:2] + 1e-3).any()
    )
