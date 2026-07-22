"""
Tests for the ML-BOP structural diagnostic (specter.ice._energy).
"""

import numpy as np
import pytest
import torch

import specter.ice._energy as energy_module
from specter.ice import mlbop_energy
from specter.ice._energy import ML_BOP_PARAMS


def test_mlbop_energy_isolated_dimer_matches_closed_form():
    """
    Two beads with no third neighbor have xi_ij = 0, so b_ij = 1 exactly and
    the energy reduces to the bare pair term f_C(r) * (f_R(r) + f_A(r)).
    r = 3.0 Å is chosen below R - D (~3.012 Å) so f_C(r) == 1.0 exactly,
    letting the expected value be computed independently of MLBOP.f_C.
    """
    r = 3.0
    coords = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])

    result = mlbop_energy(coords, box_size=20.0, pbc=False)

    p = ML_BOP_PARAMS
    f_R = p["A"] * np.exp(-p["lambda1"] * r)
    f_A = -p["B"] * np.exp(-p["lambda2"] * r)
    expected_pair_energy = f_R + f_A  # f_C(r) == 1, b_ij == 1

    assert result["E_total"] == pytest.approx(expected_pair_energy, rel=1e-6)
    assert result["E_per_atom"] == pytest.approx(expected_pair_energy / 2, rel=1e-6)
    assert result["rij_mean"] == pytest.approx(r)
    assert np.isnan(result["theta_mean"])  # no third neighbor -> no angle samples


def test_mlbop_energy_ice_like_spacing_scores_lower_than_overlapping():
    """
    The whole point of using ML-BOP as a diagnostic is that well-separated,
    ice-like spacing should score lower (more favorable) than beads crammed
    on top of each other. Check that ordering holds, not an exact value.
    """
    box = (9.0, 9.0, 9.0)

    coords_1d = torch.arange(3) * 3.0  # simple-cubic lattice, 3 Å spacing
    zz, yy, xx = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    lattice_coords = torch.stack(
        [xx.flatten(), yy.flatten(), zz.flatten()], dim=-1
    ).float()

    torch.manual_seed(0)
    overlapping_coords = (
        torch.rand(lattice_coords.shape[0], 3) * 0.5
    )  # crammed into 0.5 Å cube

    lattice_result = mlbop_energy(lattice_coords, box_size=box, pbc=True)
    overlapping_result = mlbop_energy(overlapping_coords, box_size=box, pbc=True)

    assert lattice_result["E_per_atom"] < overlapping_result["E_per_atom"]
    assert overlapping_result["E_per_atom"] > 0  # net repulsive when beads overlap


def test_mlbop_energy_rejects_wrong_shape():
    with pytest.raises(ValueError):
        mlbop_energy(torch.zeros(5, 2), box_size=10.0)


def test_mlbop_energy_shifts_centred_coordinates_into_non_negative_range(monkeypatch):
    """
    ASE's cell is always anchored at the origin (spans [0, box_size)).
    Centred input -- e.g. MDSimDump's, GradientSKIcemaker's, or
    RandomIcemaker's coordinates, which span [-box/2, box/2) -- must be
    shifted into that range before being handed to ASE.

    This isn't just a convention mismatch: ASE's non-periodic binning
    silently *clips* negative coordinates onto the boundary bin instead of
    rejecting them, which folds the entire negative-coordinate half-space
    into a single bin. Pair counts still come out right at small scale (the
    same-bin fallback catches same-bin pairs directly), so a "does the
    energy match" test can't detect this -- it only shows up as the
    per-bin atom count scaling with N instead of local density, which blew
    up to a 3.68 TiB allocation for a real 31k-atom MDSimDump frame. So
    check the actual mechanism directly: the positions ASE receives must
    never be negative.
    """
    captured: dict[str, np.ndarray] = {}
    real_atoms = energy_module.Atoms

    def spy_atoms(*args, **kwargs):
        captured["positions"] = kwargs["positions"]
        return real_atoms(*args, **kwargs)

    monkeypatch.setattr(energy_module, "Atoms", spy_atoms)

    coords = torch.tensor([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]])
    mlbop_energy(coords, box_size=20.0, pbc=False)

    assert (captured["positions"] >= 0).all()
