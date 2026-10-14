"""
With no absorption and no aperture, the unitary scattering models keep the
mean intensity at its vacuum value of one.

Transmission through a real potential is a pure phase, and the Fresnel
propagator and the lens are both unitary, so the mean of ``|psi|^2`` over the
whole propagation field cannot change; the focal-spread image is a weighted
mean of such images with weights summing to one. Anything that changes the
wave's amplitude instead -- a bandlimit, an envelope or B-factor on the wave,
quadrature weights that do not sum to one -- shows up as a mean away from
one. The specimen is a strong phase object, about 2 rad, so that such errors
are visible: the transfer-function envelope route for the focal spread loses
1.6e-4 of the electrons here, a 20 A^2 B-factor 4e-3 and ``klim=0.66``
2.4e-4, against a tolerance of 2e-6.

Only multislice and projection are unitary. First Born and kinematic gain
``sigma^2 V^2`` (``|1 + i sigma V|^2``), and Rytov's exponent has a real part
from diffraction inside the specimen (+1.4e-4 here).
"""

from __future__ import annotations

import pytest
import torch

from specter.imaging import CTF, Poses, Specimen


def _waves(scattering_model: str, oversample: int, pad_fft: bool):
    torch.manual_seed(0)
    n = 32
    potential = torch.rand(n, n, n) * 60.0
    potential[10:22, 8:24, 12:20] += 180.0
    return Specimen(potential, 1.0).scatter(
        Poses.random(2),
        voltage=300.0,
        scattering_model=scattering_model,
        alpha=0.0,
        oversample=oversample,
        pad_fft=pad_fft,
    )


@pytest.mark.parametrize("scattering_model", ["multislice", "projection"])
@pytest.mark.parametrize("oversample", [1, 2])
@pytest.mark.parametrize("pad_fft", [False, True])
def test_unitary_models_conserve_electrons(
    scattering_model: str, oversample: int, pad_fft: bool
) -> None:
    """
    The mean over the whole field, not over the pixel crop of a padded one,
    since electrons scatter across the crop's edge.
    """
    waves = _waves(scattering_model, oversample, pad_fft)
    ctf = CTF.from_values(15000.0, n=2)
    images = {
        "exit wave": waves.intensity(cc=None),
        "lens": waves.apply_ctf(ctf).intensity(cc=None),
        "lens and focal spread": waves.apply_ctf(ctf).intensity(cc=2.7),
    }
    for stage, image in images.items():
        mean = float(image.array.mean())
        assert mean == pytest.approx(1.0, abs=2e-6), stage


def test_the_linear_model_keeps_the_vacuum_mean() -> None:
    """The lens transfers no contrast at zero frequency."""
    waves = _waves("ctf", 1, False)
    image = waves.apply_ctf(CTF.from_values(15000.0, n=2)).intensity(cc=2.7)
    assert float(image.array.mean()) == pytest.approx(1.0, abs=2e-6)
