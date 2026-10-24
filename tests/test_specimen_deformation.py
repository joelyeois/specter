"""
A tilt series of a bent specimen (`settings.Deformation`).

The bend must move the specimen's material where thin-plate kinematics put
it, grown to each tilt's own exposure, without tearing it at the support
and without changing its integrated potential. The images are compared
against beads rendered where an independent evaluation of the displacement
puts them, never against the warp itself.
"""

from __future__ import annotations

import math

import pytest
import torch

from specter.imaging import CTF, Exposure, Poses, Sample
from specter.imaging._deformation import burst_fraction, specimen_warp
from specter.settings import Deformation, Propagation

NZ, NXY, PIXEL = 40, 96, 10.0
DOSE = 3.0
ANGLES = [0.0, 30.0, -45.0]


def _bead(centre: tuple[float, float, float], sigma: float = 1.5) -> torch.Tensor:
    """A Gaussian bead at `centre`, ``(x, y, z)`` in A from the rotation centre."""
    axes = [(torch.arange(n) - n // 2) * PIXEL for n in (NZ, NXY, NXY)]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    cx, cy, cz = centre
    r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return torch.exp(-r2 / (2 * (sigma * PIXEL) ** 2))[None]


def _images(
    volume: torch.Tensor, deformation: Deformation = Deformation()
) -> list[torch.Tensor]:
    """Each tilt's projected phase, which is linear in the bead's potential."""
    series = Sample.from_volume(volume, PIXEL).tilt_series(
        Poses.tilt_series(ANGLES),
        Exposure.tilt_series(DOSE, len(ANGLES)),
        voltage=300.0,
        micrograph_size=NXY,
        deformation=deformation,
        # Reflect-padding would mirror a copy of the bead in from the edge.
        pad_volume=False,
        log=False,
        propagation=Propagation(
            scattering_model="projection", alpha=0.0, oversample=1, pad_fft=False
        ),
    )
    return [torch.angle(series.scatter(i).pixels()[0]) for i in range(len(ANGLES))]


def _centroid(image: torch.Tensor, r: int = 10) -> torch.Tensor:
    """``(x, y)`` centroid in pixels, over a window around the peak."""
    image = image.double()
    iy, ix = divmod(int(image.argmax()), image.shape[1])
    window = image[iy - r : iy + r + 1, ix - r : ix + r + 1]
    yy, xx = torch.meshgrid(
        torch.arange(iy - r, iy + r + 1, dtype=torch.float64),
        torch.arange(ix - r, ix + r + 1, dtype=torch.float64),
        indexing="ij",
    )
    return torch.stack([(window * xx).sum(), (window * yy).sum()]) / window.sum()


def test_a_bead_on_the_crown_moves_by_the_deflection_times_sin_tilt():
    """
    At the crown the film only lifts, so a bead on the midplane appears
    shifted by ``w sin(theta)`` across the tilt axis, with ``w`` grown to the
    middle of that tilt's own exposure. The hole is far wider than the bead,
    so the deflection is uniform over it.
    """
    height, burst = 20.0, 3.0
    deformation = Deformation(
        model="dome", height=height, hole_radius=6000.0, burst_dose=burst
    )
    bead = _bead((0.0, 0.0, 0.0))
    rigid, bent = _images(bead), _images(bead, deformation)
    for i, angle in enumerate(ANGLES):
        lift = height * burst_fraction(DOSE * i + DOSE / 2, burst)
        expected = lift * math.sin(math.radians(angle)) / PIXEL
        shift = _centroid(bent[i]) - _centroid(rigid[i])
        assert abs(float(shift[0])) < 1e-3, f"moved along the tilt axis at {angle}"
        assert float(shift[1]) == pytest.approx(expected, abs=0.01), angle


def test_an_off_centre_bead_above_the_midplane_slides_with_the_plate():
    """
    Off the crown and above the midplane a bead also slides outward by
    ``-z dw/dx``: the face of the film it sits in is stretched by the bend.
    The warped image matches a bead rendered where that displacement puts
    it, at every tilt including the untilted one, which must therefore be
    resampled rather than read straight off the volume.
    """
    height, radius = 20.0, 600.0
    deformation = Deformation(
        model="dome", height=height, hole_radius=radius, burst_dose=1e-3
    )
    x0, z0 = 200.0, 60.0
    q = 1.0 - x0**2 / radius**2
    w = height * q * q
    w_x = -4.0 * height * q * x0 / radius**2
    moved = (x0 - z0 * w_x, 0.0, z0 + w)

    bent = _images(_bead((x0, 0.0, z0)), deformation)
    displaced = _images(_bead(moved))
    rigid = _images(_bead((x0, 0.0, z0)))
    for angle, b, d, r in zip(ANGLES, bent, displaced, rigid, strict=True):
        assert torch.allclose(_centroid(b), _centroid(d), atol=0.015), angle
        assert float(_centroid(b)[0] - _centroid(r)[0]) > 0.2, angle


@pytest.mark.parametrize(
    "point", [(20.0, 0.0, 6.0), (0.0, 0.0, -9.0), (31.0, 12.0, 8.0), (55.0, 5.0, 3.0)]
)
def test_the_weight_is_the_pull_back_jacobian(point: tuple[float, float, float]):
    """
    Each value read is weighted by ``det(dX/ds)`` of the map actually
    applied, which is what conserves the integrated potential; checked
    against the map's own finite-difference Jacobian.
    """
    deformation = Deformation(
        model="dome", height=20.0, hole_radius=600.0, burst_dose=1e-3
    )
    warp = specimen_warp(deformation, 100.0, PIXEL, 200.0)
    assert warp is not None
    s = torch.tensor([point], dtype=torch.float64)
    _, weight = warp(s)
    eps = 1e-4
    columns = []
    for k in range(3):
        step = torch.zeros(1, 3, dtype=torch.float64)
        step[0, k] = eps
        columns.append((warp(s + step)[0] - warp(s - step)[0])[0] / (2 * eps))
    jacobian = torch.stack(columns, dim=1)
    # The pull-back is solved to 1e-3 voxel, so the map's own Jacobian and
    # the closed-form weight agree to ~1e-8; a wrong term would cost ~1e-3.
    assert float(weight[0]) == pytest.approx(float(torch.det(jacobian)), abs=1e-6)
    assert abs(float(weight[0]) - 1.0) > 1e-3  # the bend really strains here


def test_the_film_does_not_tear_at_the_rim():
    """
    Clamped at the rim, the film meets the support with no slope, so the
    material either side of the rim is read from neighbouring points even in
    a thick specimen's faces.
    """
    radius = 600.0
    deformation = Deformation(
        model="dome", height=30.0, hole_radius=radius, burst_dose=1e-3
    )
    warp = specimen_warp(deformation, 100.0, PIXEL, 200.0)
    assert warp is not None
    r = radius / PIXEL
    points = torch.tensor(
        [[r - 1e-3, 0.0, 15.0], [r + 1e-3, 0.0, 15.0]], dtype=torch.float64
    )
    source, _ = warp(points)
    assert float((source[1] - source[0]).norm()) < 1e-2


def test_a_film_lifted_past_the_volume_face_is_not_lost():
    """
    The volume gains the bend's height in vacuum above and below, so a bead
    the dome lifts past the specimen's own top face keeps all of its
    potential.
    """
    deformation = Deformation(
        model="dome", height=120.0, hole_radius=6000.0, burst_dose=1e-3
    )
    top = (NZ // 2 - 4) * PIXEL  # 40 A below the top face, lifted 120 A
    rigid = _images(_bead((0.0, 0.0, top)))
    bent = _images(_bead((0.0, 0.0, top)), deformation)
    # Unpadded, the lifted bead vanishes entirely; what remains here is the
    # untilted reference skipping the interpolation the bent tilt goes through.
    for r, b in zip(rigid, bent, strict=True):
        assert float(b.sum()) == pytest.approx(float(r.sum()), rel=5e-3)


def test_vacuum_above_and_below_leaves_a_defocused_image_unchanged():
    """
    The padding the bend needs is neutral: the defocus is referred to the
    midplane, so extra vacuum slices are extra propagation the defocus
    shift takes back. Here 51 A of it each side at 1 A sampling, where a
    51 A defocus error would change the image at order one.
    """
    torch.manual_seed(0)
    volume = torch.zeros(1, 30, 64, 64)
    volume[:, 10:20, 20:44, 24:40] = 6.0 + torch.rand(1, 10, 24, 16)
    padded = torch.nn.functional.pad(volume, (0, 0, 0, 0, 51, 51))
    ctf = CTF.from_values(torch.tensor([8000.0]), cs=2.7)
    images = []
    for v in (volume, padded):
        series = Sample.from_volume(v, 1.0).tilt_series(
            Poses.tilt_series([0.0]),
            Exposure.tilt_series(3.0, 1),
            voltage=300.0,
            micrograph_size=64,
            pad_volume=False,
            log=False,
            propagation=Propagation(),
        )
        wave = series.scatter(0).apply_ctf(ctf)
        images.append(wave.intensity(cc=None).pixels()[0])
    assert float(images[0].std()) > 1e-3
    assert torch.allclose(images[0], images[1], atol=2e-5)


def test_a_rigid_specimen_samples_nothing_differently():
    assert specimen_warp(Deformation(), 10.0, PIXEL, 200.0) is None
    flat = Deformation(model="dome", height=0.0)
    assert specimen_warp(flat, 10.0, PIXEL, 200.0) is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"model": "dome"}, "height"),
        ({"model": "dome", "height": 600.0, "hole_radius": 1000.0}, "gradient"),
        ({"model": "dome", "height": 10.0, "burst_dose": 0.0}, "burst_dose"),
        ({"model": "wedge", "height": 10.0}, "model"),
    ],
)
def test_deformation_refuses_what_it_cannot_model(kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        Deformation(**kwargs)
