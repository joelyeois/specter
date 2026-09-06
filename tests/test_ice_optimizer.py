"""
Tests for the ice optimiser's energy and neighbour bookkeeping, each pinned
against the simpler formulation it is equivalent to (same physics, a
float-order reshuffle at most):

* the triplet-list three-body sum against the padded dense block it
  replaced, in float64 (energy, statistics and gradient);
* the Verlet-skin neighbour cache against a fresh search, including that it
  rebuilds once an atom has moved past ``skin / 2``;
* the voxeliser's explicit-product weights and atomic splat against the
  deterministic path;
* the energy-only pre-relaxation, which must lower the ML-BOP energy of a
  random start before the full loss sees it.
"""

from __future__ import annotations

import math

import pytest
import torch

from specter.arrays import soft_voxelize_coordinates
from specter.ice import GradientSKIcemaker
from specter.ice._energy import MLBOP, NeighborListCache


def _dense_three_body_reference(
    model: MLBOP,
    n_atoms: int,
    i_idx_t: torch.Tensor,
    j_idx_t: torch.Tensor,
    rij_t: torch.Tensor,
    vec_t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    ``MLBOP._energy_from_pairs`` as it was before the triplet list: the pair
    list scattered into a dense ``(n_atoms, max_m, max_m)`` block.
    """
    device = rij_t.device
    dtype = rij_t.dtype
    order = torch.argsort(i_idx_t, stable=True)
    i_sorted = i_idx_t[order]
    j_sorted = j_idx_t[order]
    rij_sorted = rij_t[order]
    vec_sorted = vec_t[order]

    counts = torch.bincount(i_sorted, minlength=n_atoms)
    max_m = int(counts.max().item())
    group_start = torch.zeros(n_atoms, dtype=torch.long, device=device)
    group_start[1:] = torch.cumsum(counts, dim=0)[:-1]
    col = torch.arange(len(i_idx_t), device=device) - group_start.repeat_interleave(
        counts
    )

    pad_j = torch.full((n_atoms, max_m), -1, dtype=torch.long, device=device)
    pad_rij = torch.zeros((n_atoms, max_m), dtype=dtype, device=device)
    pad_vec = torch.zeros((n_atoms, max_m, 3), dtype=dtype, device=device)
    mask = torch.zeros((n_atoms, max_m), dtype=torch.bool, device=device)
    pad_j[i_sorted, col] = j_sorted
    pad_rij[i_sorted, col] = rij_sorted
    pad_vec[i_sorted, col] = vec_sorted
    mask[i_sorted, col] = True

    fc = torch.zeros_like(pad_rij)
    fc[mask] = model.f_C(pad_rij[mask])
    valid = mask & (fc > 0.0)
    fR = torch.zeros_like(pad_rij)
    fA = torch.zeros_like(pad_rij)
    fR[valid] = model.f_R(pad_rij[valid])
    fA[valid] = model.f_A(pad_rij[valid])

    dot = torch.einsum("imc,inc->imn", pad_vec, pad_vec)
    rr = pad_rij.unsqueeze(2) * pad_rij.unsqueeze(1)
    safe_rr = torch.where(rr != 0, rr, torch.ones_like(rr))
    cos_theta = dot / safe_rr
    g_vals = model.g(cos_theta)
    eye = torch.eye(max_m, dtype=torch.bool, device=device).unsqueeze(0)
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & ~eye
    contrib = torch.where(
        pair_valid, fc.unsqueeze(1) * g_vals, torch.zeros_like(g_vals)
    )
    xi = contrib.sum(dim=2)
    b = torch.zeros_like(pad_rij)
    b[valid] = (1.0 + (model.beta**model.n) * (xi[valid] ** model.n)) ** (
        -1.0 / (2.0 * model.n)
    )
    V = torch.where(valid, fc * (fR + b * fA), torch.zeros_like(fc))
    E_total = 0.5 * V.sum()

    atom_idx = torch.arange(n_atoms, device=device).unsqueeze(1)
    rij_arr = pad_rij[valid & (pad_j > atom_idx)]
    cos_theta_arr = cos_theta[pair_valid]
    return {
        "E_total": E_total,
        "E_per_atom": E_total / n_atoms,
        "rij_mean": rij_arr.mean(),
        "rij_var": rij_arr.var(unbiased=False),
        "theta_mean": cos_theta_arr.mean(),
        "theta_var": cos_theta_arr.var(unbiased=False),
    }


def _random_water(n: int, box: float, seed: int, dtype=torch.float64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 3, generator=g, dtype=dtype) - 0.5) * box


def _pairs(model: MLBOP, pos: torch.Tensor, box: float):
    import vesin_torch

    box_t = torch.eye(3, dtype=pos.dtype) * box
    nl = vesin_torch.NeighborList(cutoff=model.r_cut, full_list=True)
    return nl.compute(pos, box_t, periodic=True, quantities="ijdD")


def test_triplet_three_body_matches_dense_reference():
    """Energy, statistics and gradient agree with the padded block in float64."""
    model = MLBOP(device="cpu")
    box = 14.0
    pos = _random_water(120, box, seed=0).requires_grad_(True)
    i, j, r, v = _pairs(model, pos, box)
    assert i.numel() > 0

    new = model._energy_from_pairs(pos.shape[0], i, j, r, v)
    ref = _dense_three_body_reference(model, pos.shape[0], i, j, r, v)
    for key in (
        "E_total",
        "E_per_atom",
        "rij_mean",
        "rij_var",
        "theta_mean",
        "theta_var",
    ):
        assert torch.allclose(new[key], ref[key], rtol=1e-12, atol=1e-12), key

    (g_new,) = torch.autograd.grad(new["E_total"], pos, retain_graph=True)
    (g_ref,) = torch.autograd.grad(ref["E_total"], pos)
    assert torch.allclose(g_new, g_ref, rtol=1e-10, atol=1e-12)


def test_compute_energy_with_cache_matches_fresh_search_and_rebuilds_on_drift():
    model = MLBOP(device="cpu")
    box = 14.0
    pos = _random_water(120, box, seed=1)
    cache = NeighborListCache(skin=1.0)

    e0 = model.compute_energy(pos, box_size=box, pbc=True, neighbor_cache=cache)
    assert cache.rebuilds == 1

    # Small moves reuse the list and still agree with a fresh search: pairs
    # that crossed r_cut are inside the skin, and carry f_C = 0 either way.
    g = torch.Generator().manual_seed(2)
    for amplitude in (0.1, 0.3):
        moved = (
            pos
            + amplitude
            * (torch.rand(pos.shape, generator=g, dtype=pos.dtype) - 0.5)
            * 2
        )
        cached = model.compute_energy(
            moved, box_size=box, pbc=True, neighbor_cache=cache
        )
        fresh = model.compute_energy(moved, box_size=box, pbc=True)
        assert cache.rebuilds == 1
        assert torch.allclose(
            cached["E_total"], fresh["E_total"], rtol=1e-12, atol=1e-12
        )

    # One atom past skin / 2 forces a rebuild.
    far = pos.clone()
    far[0, 0] += 0.6
    model.compute_energy(far, box_size=box, pbc=True, neighbor_cache=cache)
    assert cache.rebuilds == 2
    assert torch.isfinite(e0["E_total"])


def test_cache_requires_periodic_boundaries():
    model = MLBOP(device="cpu")
    pos = _random_water(30, 14.0, seed=3)
    with pytest.raises(ValueError, match="pbc"):
        model.compute_energy(
            pos, box_size=14.0, pbc=False, neighbor_cache=NeighborListCache()
        )


@pytest.mark.parametrize("periodic", [True, False])
def test_atomic_splat_matches_deterministic_splat(periodic):
    g = torch.Generator().manual_seed(4)
    coords = (torch.rand(2000, 3, generator=g) - 0.5) * 20.0
    kwargs = dict(
        grid_shape=(16, 16, 16), voxel_size=1.25, device="cpu", periodic=periodic
    )
    det = soft_voxelize_coordinates(coords, deterministic=True, **kwargs)
    atomic = soft_voxelize_coordinates(coords, deterministic=False, **kwargs)
    assert torch.allclose(det, atomic, rtol=1e-6, atol=1e-6)
    assert math.isclose(float(det.sum()), float(atomic.sum()), rel_tol=1e-6)

    # Gradients through both splats agree too (the weights are the same
    # products, only the accumulation differs).
    c = coords.clone().requires_grad_(True)
    (g_det,) = torch.autograd.grad(
        (soft_voxelize_coordinates(c, deterministic=True, **kwargs) ** 2).sum(), c
    )
    (g_atomic,) = torch.autograd.grad(
        (soft_voxelize_coordinates(c, deterministic=False, **kwargs) ** 2).sum(), c
    )
    assert torch.allclose(g_det, g_atomic, rtol=1e-5, atol=1e-6)


def test_prerelaxation_lowers_the_energy_of_a_random_start():
    torch.manual_seed(0)
    gd = GradientSKIcemaker(n=24, dx=1.0, device="cpu", progressbars=False)
    gd.init_random()
    before = gd.mlbop_energy()["E_per_atom"]

    pos = gd.positions.clone().requires_grad_(True)
    gd._prerelax(pos, steps=5, neighbor_cache=NeighborListCache())
    gd.positions = pos.detach()
    after = gd.mlbop_energy()["E_per_atom"]

    assert math.isfinite(after)
    assert after < before
    # Still inside the periodic cell.
    assert float(pos.abs().max()) <= gd.box_x / 2 + 1e-6


def test_optimize_runs_with_the_new_settings_and_records_history():
    torch.manual_seed(0)
    gd = GradientSKIcemaker(n=24, dx=1.0, device="cpu", progressbars=False)
    gd.init_random()
    history = gd.optimize(
        n_steps=3,
        record_every=1,
        tol=None,
        history_size=10,
        max_iter=5,
        prerelax_steps=2,
    )
    assert len(history["loss"]) == 3
    assert all(math.isfinite(v) for v in history["loss"])
    assert history["stopped_early"] is False


def test_plateau_streak_restarts_lbfgs_history_before_stopping(monkeypatch):
    """
    A run of unchanged losses clears the L-BFGS memory halfway to
    ``patience`` (a stall, not convergence, is the usual cause at scale) and
    still stops at ``patience`` when the plateau persists through it.
    """
    torch.manual_seed(0)
    gd = GradientSKIcemaker(n=24, dx=1.0, device="cpu", progressbars=False)
    gd.init_random()

    def constant_loss(pos, *args, **kwargs):
        # Frozen loss with a live graph, so every step is a plateau step.
        return 1.0 + 0.0 * pos.sum(), torch.ones(gd.nz, gd.n, gd.n)

    monkeypatch.setattr(gd, "_sk_loss", constant_loss)

    # Record how much optimiser state each outer step starts with: an
    # emptied dict at a step after the first is the history restart.
    state_sizes: list[int] = []
    real_step = torch.optim.LBFGS.step

    def spy_step(self, closure):
        state_sizes.append(len(self.state[self._params[0]]))
        return real_step(self, closure)

    monkeypatch.setattr(torch.optim.LBFGS, "step", spy_step)

    patience = 6
    history = gd.optimize(
        n_steps=50,
        record_every=1,
        tol=1e-4,
        patience=patience,
        prerelax_steps=0,
        max_iter=2,
    )
    assert history["stopped_early"] is True
    assert history["step"][-1] == patience
    # Step 0 starts empty; the restart empties it again once, mid-streak.
    assert state_sizes[0] == 0
    assert [k for k, size in enumerate(state_sizes) if size == 0 and k > 0] == [
        patience // 2 + 1
    ]


def test_float64_positions_keep_float32_kernels_and_agree_with_float64():
    """
    ``compute_dtype=float32`` from float64 positions matches the all-float64
    energy to float32 precision, and float64 coordinates splat into a
    float32 grid.
    """
    model = MLBOP(device="cpu")
    box = 14.0
    pos = _random_water(120, box, seed=5)
    cache = NeighborListCache()
    full = model.compute_energy(pos, box_size=box, pbc=True, neighbor_cache=cache)
    mixed = model.compute_energy(
        pos,
        box_size=box,
        pbc=True,
        neighbor_cache=cache,
        compute_dtype=torch.float32,
    )
    assert mixed["E_total"].dtype == torch.float32
    assert torch.allclose(
        mixed["E_total"].double(), full["E_total"], rtol=1e-5, atol=1e-5
    )

    coords = pos.clone().requires_grad_(True)
    vol = soft_voxelize_coordinates(
        coords, grid_shape=(14, 14, 14), voxel_size=1.0, device="cpu", periodic=True
    )
    assert vol.dtype == torch.float32
    (grad,) = torch.autograd.grad((vol**2).sum(), coords)
    assert grad.dtype == torch.float64
    assert math.isclose(float(vol.sum()), pos.shape[0], rel_tol=1e-5)


def test_optimize_returns_float32_positions_from_float64_parameters():
    torch.manual_seed(0)
    gd = GradientSKIcemaker(n=24, dx=1.0, device="cpu", progressbars=False)
    gd.init_random()
    gd.optimize(n_steps=2, record_every=1, tol=None, max_iter=3, prerelax_steps=1)
    assert gd.positions.dtype == torch.float32
    assert float(gd.positions.abs().max()) <= gd.box_x / 2 + 1e-6
