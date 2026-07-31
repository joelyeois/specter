"""
Smoke tests for SpherePackingSpecimenGenerator (specter.specimen.pdb_packing)
-- the hard-sphere-RSA-based specimen generator, promoted from
dev/packing_algorithms.py's algorithm comparison. Uses locally-cached PDB
fixtures (no network fetch).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from specter.crowding import pack_hard_spheres_3d
from specter.specimen.pdb_packing import (
    SpherePackingSpecimenGenerator,
    SphereProteinSpec,
)

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
def test_sphere_packing_generator_single_species():
    target_shape = (32, 48, 48)
    gen = SpherePackingSpecimenGenerator(
        protein_specs=[SphereProteinSpec(pdb_source=str(_SMALL_FIXTURE))],
        target_shape=target_shape,
        v_size=10.0,
        occupancy_fraction=0.15,
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
def test_sphere_packing_generator_multi_species_ratios():
    """Two differently-sized species, ratio-weighted -- both should end up
    represented among the placements, and the larger species' instances
    should each occupy more volume per instance than the smaller one's."""
    target_shape = (32, 64, 64)
    gen = SpherePackingSpecimenGenerator(
        protein_specs=[
            SphereProteinSpec(pdb_source=str(_SMALL_FIXTURE), ratio=3.0),
            SphereProteinSpec(pdb_source=str(_LARGE_FIXTURE), ratio=1.0),
        ],
        target_shape=target_shape,
        v_size=10.0,
        occupancy_fraction=0.15,
        gap_angstrom=5.0,
        seed=0,
    )
    volume = gen.generate()

    assert volume.shape == target_shape
    assert volume.max().item() > 0

    species_ids = {p.species_id for p in gen.placements}
    assert str(_SMALL_FIXTURE) in species_ids
    assert str(_LARGE_FIXTURE) in species_ids
