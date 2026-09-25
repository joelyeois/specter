"""
Tilt-series geometry conventions: where a translation moves a tilted image,
and how an AreTomo3 ``.aln`` maps onto SPECTER's quaternions and shifts.

The AreTomo3 conventions below were established against AreTomo3 2.2.2
itself (a bead phantom projected here and reconstructed there with
``-Cmd 2`` from the same ``.aln``); these tests pin the reader to them
without needing the binary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import roma
import torch

from specter.imagegenerator import TiltSeriesGenerator
from specter.settings import Camera, Propagation
from specter.tilt._aretomo3 import read_aretomo3_aln, tilt_to_quaternions

NZ, NXY, PX = 16, 128, 10.0


def _bead_image(rotvec: tuple[float, float, float], t_px: tuple[float, float]):
    """Projected phase of one bead off-centre in a thin slab, and its peak."""
    z, y, x = torch.meshgrid(
        torch.arange(NZ) - NZ / 2,
        torch.arange(NXY) - NXY / 2,
        torch.arange(NXY) - NXY / 2,
        indexing="ij",
    )
    volume = torch.exp(-((x - 3) ** 2 + (y + 2) ** 2 + (z - 4) ** 2) / (2 * 1.5**2))
    generator = TiltSeriesGenerator(
        volume=(2.0 * volume).reshape(1, NZ, NXY, NXY),
        micrograph_size=NXY,
        pixel_size=PX,
        ctf_params=None,
        voltage=300.0,
        dose_per_angstrom=1.0,
        quaternions=roma.rotvec_to_unitquat(
            torch.tensor([rotvec], dtype=torch.float32)
        ),
        translations=torch.tensor([[t_px[0] * PX, t_px[1] * PX]], dtype=torch.float32),
        propagation=Propagation(scattering_model="projection"),
        optics=None,
        camera=Camera(noise_model=None),
        verbose=False,
        progressbars=False,
    )
    _, exitwaves, _ = generator.generate_tilt_series(torch.tensor([0]))
    image = torch.angle(exitwaves[0, 0]).numpy()
    # Search the centre only: at high tilt the reflect-padded periphery holds
    # mirrored copies of the bead.
    c = NXY // 2
    window = image[c - 24 : c + 24, c - 24 : c + 24]
    iy, ix = np.unravel_index(window.argmax(), window.shape)
    patch = window[iy - 3 : iy + 4, ix - 3 : ix + 4]
    yy, xx = np.mgrid[-3:4, -3:4]
    w = patch / patch.sum()
    return np.array([ix + (w * xx).sum() - 24, iy + (w * yy).sum() - 24])


@pytest.mark.parametrize("tilt_deg", [30.0, 60.0])
@pytest.mark.parametrize("axis", [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.866, 0.0)])
def test_translation_is_an_image_shift_at_any_tilt_of_a_slab(
    tilt_deg: float, axis: tuple[float, float, float]
) -> None:
    """
    ``translations`` shift the tilted image by ``-t``, whatever the tilt.

    The pre-rotated translation acquires a z component, which used to be
    normalised by the slab's x width instead of its depth, so the shift
    perpendicular to the tilt axis came out as ``t cos^2`` of the tilt.
    """
    rotvec = tuple(np.deg2rad(tilt_deg) * np.asarray(axis))
    moved = _bead_image(rotvec, (8.0, -5.0)) - _bead_image(rotvec, (0.0, 0.0))
    assert moved == pytest.approx([-8.0, 5.0], abs=0.1)


def _write_aln(path: Path, rot: float, tx: float, ty: float) -> None:
    rows = "".join(
        f"{k:5d}  {rot:9.4f}  1.00000  {tx:9.3f}  {ty:9.3f}  1.00  1.00  1.00  0.00  {a:8.2f}\n"
        for k, a in enumerate((-30.0, 0.0, 30.0))
    )
    path.write_text(
        "# AreTomo Alignment / Priims bprmMn\n# RawSize = 64 64 3\n# NumPatches = 0\n"
        "# SEC     ROT         GMAG       TX          TY      SMEAN     SFIT    SCALE     BASE     TILT\n"
        + rows
    )


@pytest.mark.parametrize(
    ("rot", "axis"),
    [(0.0, (0.0, 1.0, 0.0)), (90.0, (-1.0, 0.0, 0.0)), (30.0, (-0.5, 0.866, 0.0))],
)
def test_aretomo_rot_is_measured_from_the_y_axis(rot: float, axis: tuple) -> None:
    """AreTomo3 rotates each image by ROT to put the tilt axis on y."""
    quats = tilt_to_quaternions(torch.tensor([20.0]), rot)
    rotvec = roma.unitquat_to_rotvec(quats)[0]
    expected = np.deg2rad(20.0) * np.asarray(axis)
    assert rotvec.numpy() == pytest.approx(expected, abs=1e-3)


def test_aretomo_shift_is_the_negative_of_the_correction(tmp_path: Path) -> None:
    """AreTomo3 corrects by sampling at R u + s, so raw content sits at +s."""
    aln = tmp_path / "ts.aln"
    _write_aln(aln, rot=0.0, tx=6.0, ty=-4.0)
    _, translations = read_aretomo3_aln(str(aln), pixel_size=2.0)
    assert translations.numpy() == pytest.approx(np.tile([-12.0, 8.0], (3, 1)))
