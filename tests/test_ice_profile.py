"""Tests for :class:`specter.ice.IceProfile` and its consumers."""

from __future__ import annotations

import math

import pytest
import torch

from specter.aberrations import defocus_midplane_shift
from specter.arrays import compute_nz
from specter.config import MicrographConfig, validate_config
from specter.crowding import filter_by_local_z_density, filter_by_z_density
from specter.ice import IceBank, IceProfile, blend_ice_into_volume
from specter.imaging import CTF, Exposure, Field
from specter.pipelines._micrograph import build_ice_profile
from specter.settings import Ice
from specter.specimen import MicrographSpecimenGenerator


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


def test_centroid_is_zero_for_a_level_midplane():
    """
    The nominal defocus refers to the ice's scattering centroid. A level
    mid-plane puts it at the box's midplane however the thickness varies, so
    the vacuum a wedge leaves in its box does not move the defocus.

    A tilted film is measured on an odd field, whose pixels are symmetric
    about the origin; an even field is centred half a pixel below it (pixel
    k at (k - n // 2) * px), so there a level tilted film's centroid is the
    mid-plane's height at that centre, -tilt * px / 2.
    """
    assert IceProfile(mode="flat", mean_thickness=500.0).centroid(64, 2.0) == 0.0
    wedge = IceProfile(mode="wedge", thickness_range=(250.0, 900.0))
    assert wedge.centroid(128, 2.0) == pytest.approx(0.0, abs=1e-6)
    level_tilt = IceProfile(mode="flat", mean_thickness=400.0, tilt=0.1)
    assert level_tilt.centroid(65, 2.0) == pytest.approx(0.0, abs=1e-3)
    assert level_tilt.centroid(64, 2.0) == pytest.approx(-0.1 * 2.0 / 2, abs=1e-3)


def test_centroid_weights_a_tilted_film_by_its_thickness():
    """
    A tilted film that is also thicker on its raised side scatters more from
    that side, so its centroid rises; thicker on the lowered side, it falls.
    An odd field, symmetric about the origin, makes the two mirror images.
    """
    nxy, px, slope = 65, 2.0, 0.1
    raised = IceProfile(
        mode="wedge", thickness_range=(200.0, 800.0), tilt=slope, angle=0.0
    )
    lowered = IceProfile(
        mode="wedge", thickness_range=(800.0, 200.0), tilt=slope, angle=0.0
    )
    t, c = raised.thickness(nxy, px), raised.center(nxy, px)
    expected = float((t * c).sum() / t.sum())
    assert raised.centroid(nxy, px) == pytest.approx(expected)
    assert raised.centroid(nxy, px) > 0
    assert lowered.centroid(nxy, px) == pytest.approx(-raised.centroid(nxy, px))


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
    # Sample at the exact pixel positions of one row, pixel k at
    # (k - nxy // 2) * px (the canvas convention particles are placed in).
    xs = (torch.arange(nxy, dtype=torch.float32) - nxy // 2) * px
    xy = torch.stack([xs, torch.zeros(nxy)], dim=1)
    z_bot, z_top = prof.surfaces_at(xy, nxy, px)
    row = nxy // 2  # y = 0 is this row
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
    """A specimen volume with nothing in it."""
    return torch.zeros(1, nz, nxy, nxy)


def test_blend_without_profile_fills_the_box():
    """An empty specimen is vitreous ice, not vacuum.

    An all-zero volume used to come back empty: the blend gated on
    ``V < 0.05 * V.max()``, so zeros put the threshold at zero and no ice
    was added anywhere. The occupancy estimator is absolute, so empty space
    takes ice at full weight in every slice."""
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


# ----------------------------------------------------------------------
# Config -> profile
# ----------------------------------------------------------------------


def _cfg(**kwargs) -> MicrographConfig:
    return MicrographConfig(pdb_source="1bxn", **kwargs)


def test_default_config_builds_no_profile():
    """The default must stay on the untouched no-profile code path."""
    assert build_ice_profile(_cfg()) is None
    assert build_ice_profile(_cfg(ice_profile="flat")) is None


@pytest.mark.parametrize(
    "config_kwargs, expected",
    [
        ({"ice_tilt": 0.2}, {"mode": "flat", "tilt": 0.2}),
        (
            {
                "ice_profile": "wedge",
                "ice_thickness_range": [250.0, 900.0],
                "ice_profile_angle": 45.0,
            },
            {"mode": "wedge", "thickness_range": (250.0, 900.0), "angle": 45.0},
        ),
        # A CLI flag can only carry a string, so 'min,max' must parse.
        (
            {"ice_profile": "wedge", "ice_thickness_range": "250,900"},
            {"thickness_range": (250.0, 900.0)},
        ),
        (
            {
                "ice_profile": "meniscus",
                "ice_thickness": 300.0,
                "ice_rim_thickness": 1600.0,
                "ice_hole_radius": 2500.0,
                "ice_hole_offset": "4500,0",
            },
            {
                "mode": "meniscus",
                "mean_thickness": 300.0,
                "rim_thickness": 1600.0,
                "hole_offset": (4500.0, 0.0),
            },
        ),
    ],
    ids=["tilt_alone", "wedge", "cli_string_spelling", "meniscus_offset_hole"],
)
def test_config_builds_a_profile(config_kwargs: dict, expected: dict) -> None:
    prof = build_ice_profile(_cfg(**config_kwargs))
    assert prof is not None
    for attr, value in expected.items():
        assert getattr(prof, attr) == value, attr


def test_reversed_thickness_range_is_rejected_by_validation():
    cfg = _cfg(ice_profile="wedge", ice_thickness_range=[900.0, 250.0])
    with pytest.raises(ValueError, match="ice_thickness_range"):
        validate_config(cfg)


def _blob_template(n: int = 16) -> torch.Tensor:
    V = torch.zeros(n, n, n)
    V[6:10, 6:10, 6:10] = 20.0
    return V


def _field(px: float, nxy: int, ice: Ice) -> Field:
    return (
        Field(_blob_template(), px, nxy)
        .pack(min_distance=40.0, water_air_interface=True)
        .solvate(ice)
    )


def test_a_field_wires_the_profile_through():
    torch.manual_seed(0)
    px, nxy = 4.0, 64
    prof = IceProfile(mode="wedge", thickness_range=(40.0, 160.0))
    ice = Ice(model="random", profile=prof)
    template = _blob_template()
    with torch.no_grad():
        sample = _field(px, nxy, ice).draw(voltage=300.0)
        waves = sample.scatter(voltage=300.0)

    # Box sized by the thickest column, not by the mean.
    assert sample.nz == prof.required_nz(nxy, px, template.shape[0])
    assert sample.nz == 40
    # ice_thickness reports the ice, not the box depth.
    engine = MicrographSpecimenGenerator(template, px, nxy, ice=ice)
    assert engine.ice_thickness == pytest.approx(100.0)
    # A level wedge's centroid is the box's midplane, whatever vacuum the
    # box holds above and below the film.
    assert waves.defocus_shift == pytest.approx(defocus_midplane_shift(sample.nz, px))

    ctf = CTF({"cs": torch.tensor([2.7e7]), "dfu": torch.tensor([10000.0])})
    with torch.no_grad():
        image = waves.apply_ctf(ctf).intensity().detect(Exposure(30.0)).array
    assert image.shape == (1, nxy, nxy)
    assert bool(torch.isfinite(image).all())

    # The assembled volume really is wedge-shaped.
    projected = sample.volume[0].sum(dim=0)
    thin, thick = float(projected[:, :8].mean()), float(projected[:, -8:].mean())
    assert thick > 3 * thin


def test_a_field_without_profile_is_unchanged():
    """The no-profile path must still reference defocus to the box."""
    torch.manual_seed(0)
    px, nxy = 4.0, 64
    ice = Ice(model="random", thickness=160.0)
    with torch.no_grad():
        sample = _field(px, nxy, ice).draw(voltage=300.0)
        waves = sample.scatter(voltage=300.0)
    assert sample.nz == compute_nz(16, 160.0, px)
    engine = MicrographSpecimenGenerator(_blob_template(), px, nxy, ice=ice)
    assert engine.ice_thickness == pytest.approx(sample.nz * px)
    assert waves.defocus_shift == pytest.approx(defocus_midplane_shift(sample.nz, px))


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


def test_an_uneven_profile_pads_by_default():
    """
    The FFT joins a field's opposite edges, which match for level, uniform ice
    and not for a wedge. Left to the default, a wedge pads and a flat slab
    does not; an explicit pad_fft still wins.
    """
    px, nxy = 4.0, 32
    for profile, padded in (
        (IceProfile(mode="wedge", thickness_range=(40.0, 160.0)), True),
        (IceProfile(mode="flat", mean_thickness=100.0), False),
    ):
        sample = _field(px, nxy, Ice(model="random", profile=profile)).draw(
            voltage=300.0
        )
        with torch.no_grad():
            waves = sample.scatter(voltage=300.0)
            explicit = sample.scatter(voltage=300.0, pad_fft=False)
        assert (waves.pad_nxy > nxy) is padded
        assert explicit.pad_nxy == nxy
