"""
Test that SPECTER's forward-model stages can be composed by hand, without
going through the ``ImageGenerator`` class, and that doing so reproduces
``ImageGenerator``'s own output bit-for-bit given identical parameters.

This mirrors demo-notebooks/modular_pipeline/modular-pipeline-generator.ipynb:
rotate -> pad -> scatter -> aberrate -> detect, calling
``VolumeRotator``/``Scattering``/``Aberration``/``Detector`` directly. If
``ImageGenerator``'s internal pipeline ever changes order, defaults, or the
defocus midplane shift, this test should catch the divergence between the
"documented" manual composition and what the class actually does.
"""

from __future__ import annotations

import roma
import torch

from specter import rotations
from specter.aberrations import Aberration, defocus_midplane_shift
from specter.arrays import compute_nz, pad_volume
from specter.imagegenerator import ImageGenerator
from specter.microscope import Detector
from specter.rotations import VolumeRotator
from specter.scattering import Scattering


def test_modular_pipeline_matches_image_generator():
    """Hand-composed rotate/pad/scatter/aberrate/detect == ImageGenerator output."""
    nxy = 32
    pixel_size = 1.5
    voltage = 300.0
    dose = 20.0

    V0 = torch.zeros(nxy, nxy, nxy)
    V0[12:20, 12:20, 12:20] = 50.0

    quaternions = rotations.random_quaternion(2)
    translations = torch.zeros(2, 2)
    ctf_params = {"cs": torch.full((2,), 2.0e7), "dfu": torch.full((2,), 8000.0)}

    # ---- reference: ImageGenerator ----
    model = ImageGenerator(
        V0.clone(),
        pixel_size,
        quaternions,
        translations,
        ctf_params,
        voltage,
        dose,
        ice_model=None,
        scattering_model="multislice",
        aberration_model="holography",
        noise_model=None,
        ews_curvature_sign="positive",
        pad_fft=True,
        verbose=False,
        progressbars=False,
    )
    with torch.no_grad():
        expected = model(torch.arange(2))

    # ---- manual, ImageGenerator-free composition ----
    nz = compute_nz(nxy, None, pixel_size)
    pad_nxy = nxy + (nxy // 2) * 2

    rotator = VolumeRotator(nz, nxy, nxy, origin="relion")
    R = roma.unitquat_to_rotmat(quaternions)
    T = rotations.translations_angstrom_to_torch(translations, nxy, pixel_size)
    theta = rotations.build_affine_matrix(R, T)
    V = rotator(V0.clone(), theta)
    V = pad_volume(V, nxy, nz, None, True)

    scattering = Scattering(
        pad_nxy,
        pixel_size,
        voltage,
        scattering_model="multislice",
        ews_curvature_sign="positive",
        nz=nz,
        alpha=0.0,
        progressbars=False,
    )
    with torch.no_grad():
        exitwaves = scattering(V)

    # multislice evaluates the CTF at the volume's midplane, not its entry
    # face -- ImageGenerator applies this shift internally via
    # BaseImager._apply_defocus_shift, built on the same public helper.
    defocus_shift = defocus_midplane_shift(nz, pixel_size)
    manual_ctf_params = {
        "cs": ctf_params["cs"],
        "dfu": ctf_params["dfu"] - defocus_shift,
    }

    aberration = Aberration(
        pad_nxy, pixel_size, voltage, aberration_model="holography", alpha=0.0
    )
    with torch.no_grad():
        detector_waves = aberration(exitwaves, manual_ctf_params)

    detector = Detector(pixel_size, aberration_model="holography", noise_model=None)
    dose_batch = torch.full((2,), dose)
    coincidence_radius = torch.zeros(2)
    with torch.no_grad():
        actual = detector(detector_waves, dose_batch, coincidence_radius, nxy=nxy)

    assert torch.allclose(actual, expected, atol=1e-4)
