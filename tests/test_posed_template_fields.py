"""
`Specimen`'s template-field path against the per-image path it replaces.

A damaged particle stack used to filter every image's padded canvas: the dose
envelope over the posed volume, then the occupancy blur and its damaged copy
for the ice. `Specimen.pose_template_fields` filters the template once and
poses the results with each image and its neighbours instead. Both filters are
isotropic and linear, so the two paths differ only by the order of
interpolation and filtering, by the clamp where two molecules overlap, and at
the canvas faces, where the per-image filters wrap or extend the volume. These
tests hold the two together on a smooth, compact template in the box interior,
and check which path runs.
"""

from __future__ import annotations

import pytest
import torch

from specter import rotations
from specter.ice import RandomIcemaker
from specter.imaging import Exposure, Poses, Specimen

N = 3
NXY = 48
PIXEL = 1.5
VOLTAGE = 300.0
DOSE = 40.0
SCATTER = dict(absorption_model="inelastic_mfp", inelastic_mfp_specimen=2460.0)


def _template() -> torch.Tensor:
    """Three overlapping Gaussian blobs, well inside the box."""
    z, y, x = torch.meshgrid(
        *(torch.arange(NXY, dtype=torch.float32) - (NXY - 1) / 2,) * 3,
        indexing="ij",
    )
    V = torch.zeros(NXY, NXY, NXY)
    for c, s, a in (
        ((0, 0, 0), 4.0, 6.0),
        ((3, -2, 1), 2.5, 4.0),
        ((-2, 3, -3), 2.0, 5.0),
    ):
        V += a * torch.exp(
            -((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) / (2 * s * s)
        )
    return V


def _poses() -> Poses:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        quats = rotations.random_quaternion(N).reshape(N, 4)
    return Poses(quats, torch.zeros(N, 2), torch.arange(N))


def _specimen() -> Specimen:
    return (
        Specimen(_template(), PIXEL)
        .crowd(min_distance=24.0)
        .solvate(
            icemaker=RandomIcemaker(PIXEL, NXY, NXY),
            model="random",
            thickness=NXY * PIXEL,
        )
        .damage()
    )


def _volume(
    fast: bool, seed: int | None = 5
) -> tuple[torch.Tensor, torch.Tensor | None]:
    specimen = _specimen()
    old = Specimen.pose_template_fields
    Specimen.pose_template_fields = fast
    try:
        with torch.no_grad():
            from specter.imaging._data import settings_group
            from specter.imaging._specimen import _resolve
            from specter.settings import Propagation

            p = _resolve(settings_group(Propagation, None, SCATTER), False)
            return specimen._volume(_poses(), VOLTAGE, Exposure(DOSE), seed, None, p)
    finally:
        Specimen.pose_template_fields = old


def _interior(x: torch.Tensor) -> torch.Tensor:
    """The central half of each lateral axis, away from the canvas edges."""
    n = x.shape[-1]
    return x[..., n // 4 : 3 * n // 4, n // 4 : 3 * n // 4]


def test_template_fields_match_the_per_image_filters() -> None:
    V_new, ab_new = _volume(True)
    V_old, ab_old = _volume(False)
    assert ab_new is not None and ab_old is not None
    for new, old in ((V_new, V_old), (ab_new, ab_old)):
        a, b = _interior(new.sum(-3)), _interior(old.sum(-3))
        assert float((a - b).norm() / b.norm()) < 5e-3


def test_the_per_image_path_runs_without_a_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a seed each field's neighbours would be drawn at other places."""
    calls: list[int] = []
    original = Specimen._posed_fields

    def counting(self: Specimen, *args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Specimen, "_posed_fields", counting)
    _volume(True, seed=None)
    assert calls == []
    _volume(True, seed=5)
    assert calls == [1]
