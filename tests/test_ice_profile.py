"""Tests for :class:`specter.ice.IceProfile` and its consumers."""

from __future__ import annotations

import math

import pytest
import torch

from specter.aberrations import defocus_midplane_shift
from specter.arrays import compute_nz
from specter.crowding import filter_by_local_z_density, filter_by_z_density
from specter.ice import IceBank, IceProfile, blend_ice_into_volume
from specter.imagegenerator import MicrographGenerator


# ----------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------


def test_flat_profile_is_uniform():
    prof = IceProfile(mode="flat", mean_thickness=400.0)
    t = prof.thickness(64, 2.0)
    assert t.shape == (64, 64)
    assert torch.allclose(t, torch.full((64, 64), 400.0))
    z_bot, z_top = prof.surfaces(64, 2.0)
    assert torch.allclose(z_top - z_bot, t)
    assert torch.allclose((z_top + z_bot) / 2, torch.zeros(64, 64))


def test_wedge_spans_the_requested_range():
    prof = IceProfile(mode="wedge", thickness_range=(200.0, 800.0), angle=0.0)
    t = prof.thickness(128, 2.0)
    # Ramp along +x, uniform along y.
    assert t[:, 0].std() == pytest.approx(0.0, abs=1e-4)
    assert float(t.min()) == pytest.approx(200.0, abs=1e-3)
    assert float(t.max()) == pytest.approx(800.0, abs=1e-3)
    assert float(t.mean()) == pytest.approx(500.0, rel=1e-3)


def test_wedge_angle_rotates_the_ramp():
    flat_ramp = IceProfile(mode="wedge", thickness_range=(200.0, 800.0), angle=0.0)
    turned = IceProfile(mode="wedge", thickness_range=(200.0, 800.0), angle=90.0)
    assert torch.allclose(flat_ramp.thickness(64, 2.0).T, turned.thickness(64, 2.0))


def test_meniscus_is_thinnest_at_the_hole_centre():
    prof = IceProfile(
        mode="meniscus",
        mean_thickness=300.0,
        rim_thickness=1200.0,
        hole_radius=2000.0,
        hole_offset=(0.0, 0.0),
    )
    t = prof.thickness(129, 4.0)
    centre = t[64, 64]
    assert float(centre) == pytest.approx(300.0, abs=1.0)
    assert float(t.min()) == pytest.approx(float(centre), abs=1.0)
    # Monotonic outward along a row from the centre.
    row = t[64, 64:]
    assert torch.all(row[1:] >= row[:-1] - 1e-4)


def test_tilt_moves_both_surfaces_together():
    prof = IceProfile(mode="flat", mean_thickness=400.0, tilt=0.1)
    z_bot, z_top = prof.surfaces(64, 2.0)
    assert torch.allclose(z_top - z_bot, torch.full((64, 64), 400.0), atol=1e-3)
    c = (z_bot + z_top) / 2
    # 0.1 Å of z per Å of x, over a 64 * 2 Å field.
    assert float(c.max() - c.min()) == pytest.approx(0.1 * 63 * 2.0, rel=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "banana"},
        {"mean_thickness": -1.0},
        {"mode": "meniscus", "thickness_range": (100.0, 200.0)},
        {"mode": "wedge", "thickness_range": (-100.0, 200.0)},
        {"mode": "meniscus", "hole_radius": 0.0},
        {"softness": -1.0},
    ],
)
def test_invalid_profiles_are_rejected(kwargs):
    with pytest.raises(ValueError):
        IceProfile(**kwargs)


# ----------------------------------------------------------------------
# Sizing and defocus
# ----------------------------------------------------------------------


def test_required_nz_matches_compute_nz_for_a_flat_profile():
    for thickness, px, base_nz in [
        (500.0, 2.0, 96),
        (500.0, 1.0, 96),
        (100.0, 2.0, 96),
    ]:
        prof = IceProfile(mode="flat", mean_thickness=thickness)
        assert prof.required_nz(64, px, base_nz) == compute_nz(base_nz, thickness, px)


def test_required_nz_is_set_by_the_thickest_column():
    prof = IceProfile(mode="wedge", thickness_range=(200.0, 800.0))
    assert prof.required_nz(128, 2.0) == 400  # 800 Å / 2 Å


def test_entry_face_shift_matches_the_box_shift_for_a_flat_profile():
    """
    A flat slab fills its box, so the specimen's entry face *is* the box's and
    the profile-aware shift must reproduce ``defocus_midplane_shift`` exactly.
    """
    for thickness, px in [(500.0, 2.0), (500.0, 1.0), (330.0, 1.5)]:
        prof = IceProfile(mode="flat", mean_thickness=thickness)
        nz = prof.required_nz(64, px)
        assert prof.entry_face_shift(64, px) == pytest.approx(
            defocus_midplane_shift(nz, px), abs=px
        )


def test_entry_face_shift_ignores_box_padding():
    """
    The whole point: a wedge's box is sized by its thickest column, so the box
    shift picks up vacuum the specimen shift does not.
    """
    prof = IceProfile(mode="wedge", thickness_range=(250.0, 900.0))
    nz = prof.required_nz(128, 2.0)
    specimen = prof.entry_face_shift(128, 2.0)
    box = defocus_midplane_shift(nz, 2.0)
    assert specimen == pytest.approx(575.0 / 2, rel=1e-3)  # mean thickness / 2
    assert box - specimen == pytest.approx(162.5, rel=1e-2)


def test_entry_face_shift_follows_a_tilted_midplane():
    tilted = IceProfile(mode="flat", mean_thickness=400.0, tilt=0.1)
    # A symmetric tilt has zero mean, so the entry face is still t / 2.
    assert tilted.entry_face_shift(64, 2.0) == pytest.approx(200.0, abs=1e-3)


# ----------------------------------------------------------------------
# Window
# ----------------------------------------------------------------------


def test_window_is_one_inside_and_zero_outside():
    prof = IceProfile(mode="flat", mean_thickness=200.0, softness=2.0)
    w = prof.window(nz=200, nxy=8, pixel_size=2.0)
    assert w.shape == (200, 8, 8)
    mid = w[100]
    assert torch.all(mid > 0.99)
    assert torch.all(w[0] < 0.6)  # at the surface, the taper is ~0.5
    assert float(w.max()) <= 1.0 + 1e-6
    assert float(w.min()) >= -1e-6


def test_window_integrates_to_the_local_thickness():
    prof = IceProfile(mode="wedge", thickness_range=(100.0, 400.0), softness=2.0)
    px = 1.0
    nz = prof.required_nz(64, px)
    w = prof.window(nz, 64, px)
    projected = w.sum(dim=0) * px
    expected = prof.thickness(64, px)
    # The thickest column's taper is clipped by the box (required_nz adds no
    # headroom), so allow a couple of softness widths of slack.
    assert torch.allclose(projected, expected, atol=3 * prof.softness)


def test_window_chunks_match_the_whole():
    prof = IceProfile(mode="wedge", thickness_range=(100.0, 400.0))
    full = prof.window(64, 16, 4.0)
    chunked = torch.cat(
        [prof.window(64, 16, 4.0, z_slice=slice(s, s + 16)) for s in range(0, 64, 16)]
    )
    assert torch.equal(full, chunked)


def test_zero_softness_gives_a_hard_window():
    prof = IceProfile(mode="flat", mean_thickness=100.0, softness=0.0)
    w = prof.window(100, 4, 2.0)
    assert set(w.unique().tolist()) <= {0.0, 1.0}


# ----------------------------------------------------------------------
# Sampling at particle coordinates
# ----------------------------------------------------------------------


def test_surfaces_at_reproduces_the_field():
    prof = IceProfile(mode="wedge", thickness_range=(200.0, 800.0))
    nxy, px = 64, 2.0
    z_bot_f, z_top_f = prof.surfaces(nxy, px)
    # Sample on the exact pixel centres of one row.
    xs = (torch.arange(nxy, dtype=torch.float32) - (nxy - 1) / 2) * px
    xy = torch.stack([xs, torch.zeros(nxy)], dim=1)
    z_bot, z_top = prof.surfaces_at(xy, nxy, px)
    row = nxy // 2 if nxy % 2 else nxy // 2  # y = 0 rounds to this row
    assert torch.allclose(z_top, z_top_f[row], atol=1e-3)
    assert torch.allclose(z_bot, z_bot_f[row], atol=1e-3)


def test_surfaces_at_clamps_outside_the_field():
    prof = IceProfile(mode="flat", mean_thickness=300.0)
    far = torch.tensor([[1e6, 1e6], [-1e6, -1e6]])
    z_bot, z_top = prof.surfaces_at(far, 32, 2.0)
    assert torch.allclose(z_top, torch.full((2,), 150.0))
    assert torch.allclose(z_bot, torch.full((2,), -150.0))


# ----------------------------------------------------------------------
# Crowding
# ----------------------------------------------------------------------


def test_local_z_density_reduces_to_the_global_one():
    torch.manual_seed(0)
    pts = (torch.rand(4000, 3) - 0.5) * torch.tensor([1000.0, 1000.0, 500.0])
    ones = torch.ones(len(pts))

    torch.manual_seed(1)
    a, curve_a = filter_by_z_density(pts.clone(), 500.0)
    torch.manual_seed(1)
    b, curve_b = filter_by_local_z_density(pts.clone(), -250.0 * ones, 250.0 * ones)
    assert torch.equal(a, b)
    assert torch.equal(curve_a, curve_b)


def test_local_z_density_adsorbs_to_the_local_surfaces():
    """
    Acceptance peaks at each point's own surfaces, not at a global mean. The
    uniform ``baseline`` still accepts plenty of bulk, so this compares
    acceptance *rates* rather than counts.
    """
    torch.manual_seed(0)
    n = 200000
    z = (torch.rand(n) - 0.5) * 1000.0
    pts = torch.stack([torch.zeros(n), torch.zeros(n), z], dim=1)
    # A thin slab offset well above the origin, so "local" and "global mean"
    # surfaces are nowhere near each other.
    z_bot = torch.full((n,), 200.0)
    z_top = torch.full((n,), 300.0)
    kept, _ = filter_by_local_z_density(pts, z_bot, z_top)

    def rate(centre: float, half_width: float = 5.0) -> float:
        in_band = (z - centre).abs() < half_width
        got = (kept[:, 2] - centre).abs() < half_width
        return float(got.sum()) / max(float(in_band.sum()), 1.0)

    surface_rate = (rate(200.0) + rate(300.0)) / 2
    bulk_rate = (rate(-400.0) + rate(0.0) + rate(450.0)) / 3
    assert surface_rate > 5 * bulk_rate
    # And nothing anomalous happens at the global mid-plane of the samples.
    assert rate(0.0) == pytest.approx(bulk_rate, rel=0.5)


# ----------------------------------------------------------------------
# Blending
# ----------------------------------------------------------------------


def _tiny_bank() -> IceBank:
    return IceBank()


def _empty_volume(nz: int, nxy: int) -> torch.Tensor:
    """
    A specimen volume with one marker voxel.

    :func:`~specter.ice.ice_blend_mask` gates on ``V < threshold * V.max()``,
    so an all-zero volume has ``V.max() == 0``, the mask is False everywhere,
    and no ice is blended at all. Every production path crowds particles in
    first, so ``V.max()`` is positive by the time ice is added; these tests
    need one non-zero voxel to stand in for that.
    """
    V = torch.zeros(1, nz, nxy, nxy)
    # Centre, so it sits inside the ice for any profile and never lands in a
    # region a test is asserting to be vacuum.
    V[0, nz // 2, nxy // 2, nxy // 2] = 1.0
    return V


def test_all_zero_volume_receives_no_ice():
    """Documents the ice_blend_mask edge case _empty_volume works around."""
    torch.manual_seed(0)
    V = torch.zeros(1, 24, 32, 32)
    out = blend_ice_into_volume(V, _tiny_bank(), 4.0)
    assert float(out.abs().max()) == 0.0


def test_blend_without_profile_fills_the_box():
    torch.manual_seed(0)
    bank = _tiny_bank()
    nz, nxy, px = 24, 32, 4.0
    out = blend_ice_into_volume(_empty_volume(nz, nxy), bank, px)
    column = out[0].sum(dim=(1, 2))
    # Every slice carries ice.
    assert float(column.min()) > 0.5 * float(column.mean())


def test_blend_with_a_wedge_profile_tracks_the_thickness_field():
    torch.manual_seed(0)
    bank = _tiny_bank()
    px, nxy = 4.0, 32
    prof = IceProfile(mode="wedge", thickness_range=(40.0, 120.0), softness=2.0)
    nz = prof.required_nz(nxy, px)
    out = blend_ice_into_volume(_empty_volume(nz, nxy), bank, px, profile=prof)

    projected = out[0].sum(dim=0)
    requested = prof.thickness(nxy, px)
    # Projected potential is proportional to thickness; compare the shape by
    # correlating the two fields along the ramp.
    a = projected.mean(dim=0)
    b = requested.mean(dim=0)
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1]
    assert float(corr) > 0.95
    # Thin end really is thinner than the thick end.
    assert float(a[:4].mean()) < 0.6 * float(a[-4:].mean())


def test_blend_with_a_profile_leaves_vacuum_above_the_thin_end():
    torch.manual_seed(0)
    bank = _tiny_bank()
    px, nxy = 4.0, 32
    prof = IceProfile(mode="wedge", thickness_range=(40.0, 160.0))
    nz = prof.required_nz(nxy, px)
    out = blend_ice_into_volume(_empty_volume(nz, nxy), bank, px, profile=prof)
    # Top slice, thin end: should be empty; thick end: should not.
    top = out[0, 0]
    assert float(top[:, :4].abs().mean()) < 1e-3
    assert float(top[:, -4:].abs().mean()) > 1e-3


def _blob_template(n: int = 16) -> torch.Tensor:
    V = torch.zeros(n, n, n)
    V[6:10, 6:10, 6:10] = 20.0
    return V


def test_micrograph_generator_wires_the_profile_through():
    torch.manual_seed(0)
    px, nxy = 4.0, 64
    prof = IceProfile(mode="wedge", thickness_range=(40.0, 160.0))
    template = _blob_template()
    model = MicrographGenerator(
        template,
        nxy,
        px,
        {"cs": torch.tensor([2.7e7]), "dfu": torch.tensor([10000.0])},
        300.0,
        torch.tensor([30.0]),
        ice_model="random",
        ice_profile=prof,
        crowd_min_distance=40.0,
        noise_model="poisson",
        verbose=False,
        progressbars=False,
    )

    # Box sized by the thickest column, not by the mean.
    assert model.nz == prof.required_nz(nxy, px, template.shape[0])
    assert model.nz == 40
    # ice_thickness reports the ice, not the box depth.
    assert model.ice_thickness == pytest.approx(100.0)
    # Defocus references the specimen's entry face, not the box's.
    assert model._defocus_shift_A == pytest.approx(prof.entry_face_shift(nxy, px))
    assert model._defocus_shift_A == pytest.approx(50.0)
    assert model._defocus_shift_A != pytest.approx(defocus_midplane_shift(model.nz, px))

    image = model(torch.tensor([0]))
    assert image.shape == (1, nxy, nxy)
    assert bool(torch.isfinite(image).all())

    # The assembled volume really is wedge-shaped.
    projected = model.volume[0].sum(dim=0)
    thin, thick = float(projected[:, :8].mean()), float(projected[:, -8:].mean())
    assert thick > 3 * thin


def test_micrograph_generator_without_profile_is_unchanged():
    """The no-profile path must still reference defocus to the box."""
    torch.manual_seed(0)
    px, nxy = 4.0, 64
    model = MicrographGenerator(
        _blob_template(),
        nxy,
        px,
        {"cs": torch.tensor([2.7e7]), "dfu": torch.tensor([10000.0])},
        300.0,
        torch.tensor([30.0]),
        ice_model="random",
        ice_thickness=160.0,
        crowd_min_distance=40.0,
        noise_model="poisson",
        verbose=False,
        progressbars=False,
    )
    assert model.nz == compute_nz(16, 160.0, px)
    assert model.ice_thickness == pytest.approx(model.nz * px)
    assert model._defocus_shift_A == pytest.approx(defocus_midplane_shift(model.nz, px))


def test_profile_window_conserves_mean_density_against_no_profile():
    """
    A flat profile is not bit-identical to no profile (the window tapers), but
    it must not change the total projected ice by more than the taper width.
    """
    torch.manual_seed(0)
    bank = _tiny_bank()
    px, nxy = 4.0, 32
    thickness = 96.0
    nz = compute_nz(0, thickness, px)
    V = _empty_volume(nz, nxy)

    torch.manual_seed(1)
    plain = blend_ice_into_volume(V.clone(), bank, px)
    torch.manual_seed(1)
    prof = IceProfile(mode="flat", mean_thickness=thickness, softness=2.0)
    windowed = blend_ice_into_volume(V.clone(), bank, px, profile=prof)

    ratio = float(windowed.sum() / plain.sum())
    expected = 1.0 - 2 * prof.softness / (thickness * math.sqrt(math.pi))
    assert ratio == pytest.approx(expected, abs=0.05)
