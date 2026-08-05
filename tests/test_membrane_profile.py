import pytest
import torch

from specter.specimen.membrane._profile import (
    _LEAFLET_TEMPLATE,
    BilayerProfile,
    build_analytic_bilayer_profile,
    build_reference_lipid_patch,
    compute_bilayer_profile,
    estimate_bilayer_peak_amplitude,
)


def test_build_reference_lipid_patch_is_seed_reproducible_and_leaflet_symmetric():
    atomic_numbers_a, coords_a = build_reference_lipid_patch(
        n_lipids_per_leaflet=4, seed=0
    )
    atomic_numbers_b, coords_b = build_reference_lipid_patch(
        n_lipids_per_leaflet=4, seed=0
    )
    assert torch.equal(atomic_numbers_a, atomic_numbers_b)
    assert torch.equal(coords_a, coords_b)

    n_atoms_per_lipid = sum(count for _, _, count, _ in _LEAFLET_TEMPLATE)
    assert atomic_numbers_a.shape[0] == 2 * 4 * n_atoms_per_lipid

    top_leaflet = coords_a[coords_a[:, 2] > 0]
    bottom_leaflet = coords_a[coords_a[:, 2] < 0]
    assert top_leaflet.shape[0] > 0
    assert bottom_leaflet.shape[0] > 0
    assert torch.isclose(
        top_leaflet[:, 2].mean(), -bottom_leaflet[:, 2].mean(), atol=1.0
    )


def test_build_reference_lipid_patch_different_seeds_differ():
    _, coords_a = build_reference_lipid_patch(n_lipids_per_leaflet=4, seed=1)
    _, coords_b = build_reference_lipid_patch(n_lipids_per_leaflet=4, seed=2)
    assert not torch.equal(coords_a, coords_b)


def test_bilayer_profile_interpolation_matches_table_and_extrapolates_flat():
    distance_a = torch.tensor([-10.0, 0.0, 10.0, 20.0])
    psi = torch.tensor([1.0, 5.0, 2.0, 0.5])
    profile = BilayerProfile(distance_a=distance_a, psi=psi)

    assert torch.allclose(profile(distance_a), psi, atol=1e-5)

    midpoint = profile(torch.tensor([5.0]))
    assert torch.isclose(midpoint[0], torch.tensor(3.5), atol=1e-4)

    below_range = profile(torch.tensor([-100.0]))
    above_range = profile(torch.tensor([100.0]))
    assert torch.isclose(below_range[0], torch.tensor(1.0))
    assert torch.isclose(above_range[0], torch.tensor(0.5))


def test_compute_bilayer_profile_has_headgroup_peak_and_decays_outside():
    atomic_numbers, coordinates = build_reference_lipid_patch(
        n_lipids_per_leaflet=6, area_per_lipid_a2=65.0, jitter_a=2.0, seed=0
    )
    profile = compute_bilayer_profile(
        atomic_numbers,
        coordinates,
        v_size=2.0,
        parameterization="shtyrov",
    )

    assert profile.psi.shape == profile.distance_a.shape
    assert profile.psi.numel() > 0

    baseline = profile(torch.tensor([60.0, -60.0])).mean()
    near_headgroup = profile(torch.tensor([19.5, -19.5])).mean()
    assert near_headgroup > baseline

    center = profile(torch.tensor([0.0]))
    assert near_headgroup > center.mean()

    # roughly symmetric about the mid-plane
    positive_side = profile(torch.linspace(2.0, 28.0, 20))
    negative_side = profile(torch.linspace(-28.0, -2.0, 20).flip(0))
    assert torch.corrcoef(torch.stack([positive_side, negative_side]))[0, 1] > 0.7


def test_compute_bilayer_profile_phosphate_peak_dominates_glycerol_shoulder():
    # Regression test: an earlier template weighting let the glycerol/ester/
    # upper-chain region (~+-8 A) out-peak the phosphate headgroup (~+-20 A),
    # the opposite of real bilayer electron-density profiles, where the
    # phosphate peak is the tallest, sharpest feature. Needs enough lipids
    # to be a real signal rather than per-leaflet sampling noise (a 6-lipid
    # patch does not reliably resolve this ordering).
    atomic_numbers, coordinates = build_reference_lipid_patch(
        n_lipids_per_leaflet=120, area_per_lipid_a2=65.0, jitter_a=2.5, seed=0
    )
    profile = compute_bilayer_profile(
        atomic_numbers, coordinates, v_size=1.0, parameterization="shtyrov"
    )

    phosphate_peak = profile(torch.linspace(18.0, 21.0, 10)).max()
    glycerol_shoulder_peak = profile(torch.linspace(5.0, 11.0, 10)).max()
    assert phosphate_peak > glycerol_shoulder_peak


def test_compute_bilayer_profile_no_competing_peak_in_chain_region():
    """Regression test for a real, user-reported visual defect: the previous
    template (satisfying the weaker "phosphate > glycerol shoulder" check
    above by only a hair) still rendered as FOUR visible peaks, not two --
    the acyl-chain region's own atoms (spread across 6 z-levels only 2A
    apart, each individually under-blended into its neighbors) formed a
    second, nearly phosphate-height hump around +-8A, rather than a single
    smoothly-declining shoulder. A real bilayer electron-density profile
    reads as two dominant peaks (headgroups) with clearly weaker material
    in between, not four peaks of similar height. Checked at v_size=2.0,
    compute_bilayer_profile's own default and what MembraneGenerator
    actually uses (the v_size=1.0 test above only ever exercised a finer
    resolution than production use)."""
    atomic_numbers, coordinates = build_reference_lipid_patch(
        n_lipids_per_leaflet=200, area_per_lipid_a2=65.0, jitter_a=2.5, seed=0
    )
    profile = compute_bilayer_profile(
        atomic_numbers, coordinates, v_size=2.0, parameterization="shtyrov"
    )

    phosphate_peak = profile(torch.linspace(18.0, 21.0, 10)).max()
    chain_region = profile(torch.linspace(4.0, 14.0, 20))
    center_region = profile(torch.linspace(-3.0, 3.0, 10))

    # The chain/glycerol shoulder must stay clearly below the phosphate
    # peak (not just nominally lower) -- a real "weaker in between" look,
    # not a second near-tied peak.
    assert chain_region.max() < 0.8 * phosphate_peak
    # The true center (both leaflets' disordered chain termini) must be a
    # genuine trough, below the chain shoulder's own typical level -- not
    # merely below the phosphate peak (everything is below that).
    assert center_region.mean() < 0.9 * chain_region.mean()


def test_estimate_bilayer_peak_amplitude_is_positive_and_scale_consistent():
    atomic_numbers, coordinates = build_reference_lipid_patch(
        n_lipids_per_leaflet=60, seed=0
    )
    amplitude = estimate_bilayer_peak_amplitude(
        atomic_numbers, coordinates, parameterization="shtyrov"
    )
    assert amplitude > 0
    # Must match compute_bilayer_profile's own peak (same underlying
    # render) -- this function only reduces that profile to its peak, it
    # doesn't independently recompute anything.
    profile = compute_bilayer_profile(
        atomic_numbers, coordinates, parameterization="shtyrov"
    )
    assert amplitude == pytest.approx(float(profile.psi.max()))


def test_build_analytic_bilayer_profile_is_two_clean_peaks_with_zero_between():
    """This is the actual shape MembraneGenerator renders with by default
    (see _generator.py) -- a real cryo-EM bilayer micrograph reads as two
    sharp, well-separated lines with genuinely empty space between and
    outside them (the "railroad track" look), which this analytic
    two-Gaussian-peak construction gives by construction (peaks
    geometrically separated by `thickness_a`, each individually narrow
    relative to that separation), unlike compute_bilayer_profile's
    simulated-atom-cloud approach, which needed real bug-fixing effort to
    even approximately achieve this and still doesn't reach a
    near-zero baseline (see the chain-region test above)."""
    profile = build_analytic_bilayer_profile(
        thickness_a=38.0, layer_sigma_a=2.0, amplitude=5.0
    )

    outer_peak = profile(torch.tensor([19.0])).item()
    inner_peak = profile(torch.tensor([-19.0])).item()
    center = profile(torch.tensor([0.0])).item()
    far_outside = profile(torch.tensor([100.0])).item()

    # Slightly loose tolerance: the query points fall between the lookup
    # table's discrete grid samples, and linear interpolation of a concave
    # (near-peak) function always undershoots the true continuous maximum
    # a little.
    assert outer_peak == pytest.approx(5.0, abs=0.05)
    assert inner_peak == pytest.approx(5.0, abs=0.05)
    # Genuinely near-zero in between and outside, not just "lower".
    assert center < 0.01 * 5.0
    assert far_outside < 0.01 * 5.0


def test_build_analytic_bilayer_profile_defaults_are_uncalibrated_but_valid():
    profile = build_analytic_bilayer_profile()
    assert profile.psi.max() == pytest.approx(1.0, abs=0.05)
    assert torch.isfinite(profile.psi).all()
