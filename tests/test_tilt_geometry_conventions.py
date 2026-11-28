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

from specter.imaging import Exposure, Poses, Sample
from specter.settings import Propagation
from specter.tilt._aretomo3 import read_aretomo3_aln, tilt_to_quaternions

NZ, NXY, PX = 16, 128, 10.0


def _projected_phase(
    volume: torch.Tensor, quaternions: torch.Tensor, shifts: torch.Tensor
) -> np.ndarray:
    """The phase of one projected tilt, on the pixel grid."""
    series = Sample.from_volume(volume, PX).tilt_series(
        Poses(quaternions, shifts, torch.arange(1)),
        Exposure.tilt_series(1.0, 1),
        voltage=300.0,
        micrograph_size=volume.shape[-1],
        propagation=Propagation(scattering_model="projection"),
        log=False,
    )
    with torch.no_grad():
        return torch.angle(series.scatter(0).pixels()[0]).numpy()


def _bead_image(rotvec: tuple[float, float, float], t_px: tuple[float, float]):
    """Projected phase of one bead off-centre in a thin slab, and its peak."""
    z, y, x = torch.meshgrid(
        torch.arange(NZ) - NZ / 2,
        torch.arange(NXY) - NXY / 2,
        torch.arange(NXY) - NXY / 2,
        indexing="ij",
    )
    volume = torch.exp(-((x - 3) ** 2 + (y + 2) ** 2 + (z - 4) ** 2) / (2 * 1.5**2))
    image = _projected_phase(
        (2.0 * volume).reshape(1, NZ, NXY, NXY),
        roma.rotvec_to_unitquat(torch.tensor([rotvec], dtype=torch.float32)),
        torch.tensor([[t_px[0] * PX, t_px[1] * PX]], dtype=torch.float32),
    )
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


@pytest.mark.parametrize("tilt_deg", [10.0, 40.0])
def test_tilted_slices_are_not_magnified(tilt_deg: float) -> None:
    """
    A tilt about x leaves every x coordinate where it was.

    The slice sampler converts its voxel-unit query points to normalized
    coordinates, and used to divide by ``(n - 1) / 2`` where the
    ``align_corners=False`` grid spans ``n / 2`` voxels per unit, magnifying
    every slice by ``n / (n - 1)``: a bead 40 voxels out landed ~0.3 further.
    """
    n, nz = 128, 16
    z, y, x = torch.meshgrid(
        torch.arange(nz) - nz / 2,
        torch.arange(n) - n / 2,
        torch.arange(n) - n / 2,
        indexing="ij",
    )
    volume = torch.exp(-((x - 40) ** 2 + y**2 + z**2) / (2 * 1.5**2))
    rotvec = torch.tensor([[np.deg2rad(tilt_deg), 0.0, 0.0]], dtype=torch.float32)
    image = _projected_phase(
        (2.0 * volume).reshape(1, nz, n, n),
        roma.rotvec_to_unitquat(rotvec),
        torch.zeros(1, 2),
    )
    band = image[n // 2 - 6 : n // 2 + 7, n // 2 + 30 : n // 2 + 51]
    profile = band.sum(axis=0)
    xs = np.arange(30, 51)
    assert float((profile * xs).sum() / profile.sum()) == pytest.approx(40.0, abs=0.05)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("rot", [84.94, -3.0, 30.0])
def test_turn_to_tilt_axis_puts_every_ray_along_a_volume_axis(
    rot: float, axis: str
) -> None:
    """In the turned frame the beam has no component along the chosen
    axis at any tilt, the tilt angles are unchanged, and the turn is the
    in-plane rotation of at most 90 degrees."""
    from specter.tilt import turn_to_tilt_axis

    quats = tilt_to_quaternions(torch.linspace(-45.0, 45.0, 7), rot)
    turned, angle = turn_to_tilt_axis(quats, axis)
    assert abs(angle) <= 90.0 + 1e-6
    R = roma.unitquat_to_rotmat(turned.double())
    lead = 0 if axis == "x" else 1
    assert float(R[:, lead, 2].abs().max()) < 1e-6
    # The beam's angle to the volume's z is still the tilt angle.
    assert R[:, 2, 2].numpy() == pytest.approx(
        np.cos(np.deg2rad(np.linspace(-45.0, 45.0, 7))), abs=1e-6
    )
    # The turn is about the beam: M R with M a rotation about z.
    M = R @ roma.unitquat_to_rotmat(quats.double()).transpose(-1, -2)
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    expected = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert M.numpy() == pytest.approx(np.tile(expected, (7, 1, 1)), abs=1e-5)


def test_aretomo_ctf_file_gives_the_ctf_parameters_and_the_defocus_hand(
    tmp_path: Path,
) -> None:
    """dfHand = -1 is the "positive" curvature sign, the mapping measured on
    experimental data, and +1 its mirror; a file without the column gives no
    hand."""
    from specter.tilt import aretomo3_defocus_hand, read_aretomo3_ctf

    header = "# Columns: #1 micrograph number; #2 - defocus 1 [A]; #3 - defocus 2\n"
    rows = [
        "   1   19804.49  19415.53    -59.02     0.0000    0.0421   10.0000",
        "   2   19724.95  19724.95    -59.02     0.0000    0.0480   10.0000",
    ]
    path = tmp_path / "ts_CTF.txt"
    path.write_text(header + "".join(f"{row}   -1\n" for row in rows))
    ctf, hand = read_aretomo3_ctf(str(path))
    assert hand == "positive"
    assert ctf["dfu"].numpy() == pytest.approx([19804.49, 19724.95])
    assert ctf["dfv"].numpy() == pytest.approx([19415.53, 19724.95])
    assert ctf["dfang"].numpy() == pytest.approx([-59.02, -59.02])
    assert ctf["phaseshift"].numpy() == pytest.approx([0.0, 0.0])

    path.write_text(header + "".join(f"{row}\n" for row in rows))
    assert read_aretomo3_ctf(str(path))[1] is None
    path.write_text(header + f"{rows[0]}   1\n{rows[1]}   -1\n")
    with pytest.raises(ValueError, match="more than one dfHand"):
        read_aretomo3_ctf(str(path))

    assert aretomo3_defocus_hand(-1) == "positive"
    assert aretomo3_defocus_hand(1) == "negative"
    with pytest.raises(ValueError, match="dfHand"):
        aretomo3_defocus_hand(0)
