"""
Tests for the ML-BOP structural diagnostic (specter.ice._energy).
"""

import numpy as np
import pytest
import torch

from specter.ice import MLBOP
from specter.ice._energy import ML_BOP_PARAMS


def _energy(coords, box_size, pbc=True):
    result = MLBOP().compute_energy(coords, box_size=box_size, pbc=pbc)
    return {k: v.item() for k, v in result.items()}


def test_mlbop_energy_isolated_dimer_matches_closed_form():
    """
    Two beads with no third neighbor have xi_ij = 0, so b_ij = 1 exactly and
    the energy reduces to the bare pair term f_C(r) * (f_R(r) + f_A(r)).
    r = 3.0 Å is chosen below R - D (~3.012 Å) so f_C(r) == 1.0 exactly,
    letting the expected value be computed independently of MLBOP.f_C.
    """
    r = 3.0
    coords = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])

    result = _energy(coords, box_size=20.0, pbc=False)

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

    lattice_result = _energy(lattice_coords, box_size=box, pbc=True)
    overlapping_result = _energy(overlapping_coords, box_size=box, pbc=True)

    assert lattice_result["E_per_atom"] < overlapping_result["E_per_atom"]
    assert overlapping_result["E_per_atom"] > 0  # net repulsive when beads overlap


def test_mlbop_energy_rejects_wrong_shape():
    with pytest.raises(ValueError):
        MLBOP().compute_energy(torch.zeros(5, 2), box_size=10.0)


def test_mlbop_energy_translation_invariant():
    """
    The vesin-based compute_energy has no origin dependence: shifting every
    coordinate by the same constant must not change the result, with no
    special-casing needed for centred (e.g. [-box/2, box/2)) input. (The
    old ASE-based path anchored its cell at the origin and needed centred
    coordinates shifted into [0, box_size) first to avoid a binning bug --
    see git history; that workaround and its test no longer apply.)
    """
    torch.manual_seed(0)
    coords = torch.rand(20, 3) * 9.0 - 4.5  # centred in [-4.5, 4.5)
    box = (9.0, 9.0, 9.0)

    result_centred = _energy(coords, box_size=box, pbc=True)
    result_shifted = _energy(coords + 100.0, box_size=box, pbc=True)

    assert result_centred["E_per_atom"] == pytest.approx(
        result_shifted["E_per_atom"], rel=1e-5
    )


def test_mlbop_energy_accepts_full_cell_matrix():
    """
    box_size may be a full (3, 3) cell matrix (rows as lattice vectors, ASE's
    convention) instead of just a diagonal box -- needed for triclinic MD
    cells (see ExtXYZDump.mlbop_energy). A diagonal 3x3 matrix must agree
    exactly with the equivalent tuple form, and a genuinely sheared cell
    should just work rather than error.
    """
    torch.manual_seed(1)
    coords = torch.rand(20, 3) * 9.0
    box_tuple = (9.0, 9.0, 9.0)
    box_matrix = torch.diag(torch.tensor(box_tuple))

    result_tuple = _energy(coords, box_size=box_tuple, pbc=True)
    result_matrix = _energy(coords, box_size=box_matrix, pbc=True)
    assert result_tuple["E_per_atom"] == pytest.approx(result_matrix["E_per_atom"])

    box_triclinic = torch.tensor([[9.0, 0.0, 0.0], [2.0, 9.0, 0.0], [1.0, 1.0, 9.0]])
    result_triclinic = _energy(coords, box_size=box_triclinic, pbc=True)
    assert np.isfinite(result_triclinic["E_per_atom"])


def test_mlbop_energy_small_cell_matches_repeated_reference():
    """
    Regression guard for removing the old ASE-path-matching '>= 2x cutoff'
    restriction: a cell smaller than 2x the ML-BOP cutoff must give the same
    per-atom energy as the same atoms placed in a properly-repeated, much
    larger reference cell -- vesin's neighbor search already handles the
    small-cell case correctly on its own (verified empirically before
    removing the restriction; see git history), so no manual cell-repeat
    workaround is needed.
    """
    model = MLBOP()
    L = 5.0  # < 2 * r_cut (~7.1)
    torch.manual_seed(2)
    coords = torch.rand(30, 3, dtype=torch.float64) * L

    result_small = model.compute_energy(coords, box_size=L, pbc=True)

    reps = 4
    offsets = (
        torch.tensor(
            [
                [ix, iy, iz]
                for ix in range(reps)
                for iy in range(reps)
                for iz in range(reps)
            ],
            dtype=torch.float64,
        )
        * L
    )
    coords_big = (coords.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1, 3)
    result_big = model.compute_energy(coords_big, box_size=L * reps, pbc=True)

    assert result_small["E_per_atom"].item() == pytest.approx(
        result_big["E_per_atom"].item(), rel=1e-8
    )
