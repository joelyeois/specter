"""
Test that SPECTER's forward-model engines can be composed by hand, without
the `specter.imaging` objects, and that doing so reproduces the objects'
output given identical parameters.

This mirrors demo-notebooks/create_particle_stack_modular/particle_stack_generator_modular.ipynb:
rotate -> pad -> scatter -> aberrate -> detect, calling
``VolumeRotator``/``Scattering``/``Aberration``/``Detector`` directly. If the
chain ever changes order, defaults, or the defocus midplane shift, this test
should catch the divergence between the "documented" manual composition and
what the objects actually do.
"""

from __future__ import annotations

import roma
import torch

from specter import rotations
from specter.aberrations import Aberration, defocus_midplane_shift
from specter.arrays import compute_nz, pad_volume
from specter.imaging import CTF, Exposure, Poses, Specimen, simulate_particles
from specter.microscope import Detector
from specter.rotations import VolumeRotator
from specter.scattering import Scattering
from specter.settings import Envelopes, Propagation


def test_modular_pipeline_matches_image_generator():
    """Hand-composed rotate/pad/scatter/aberrate/detect == the objects' output."""
    nxy = 32
    pixel_size = 1.5
    voltage = 300.0
    dose = 20.0

    V0 = torch.zeros(nxy, nxy, nxy)
    V0[12:20, 12:20, 12:20] = 50.0

    quaternions = rotations.random_quaternion(2)
    translations = torch.zeros(2, 2)
    ctf_params = {"cs": torch.full((2,), 2.0e7), "dfu": torch.full((2,), 8000.0)}

    # ---- reference: the image-formation objects ----
    with torch.no_grad():
        expected = simulate_particles(
            Specimen(V0.clone(), pixel_size),
            Poses(quaternions, translations, torch.arange(2)),
            CTF(ctf_params),
            Exposure(dose),
            voltage=voltage,
            batch_size=2,
            scatter=dict(
                scattering_model="multislice",
                ews_curvature_sign="positive",
                pad_fft=True,
            ),
            camera=dict(noise=None),
        ).array

    # ---- manual composition of the engines ----
    nz = compute_nz(nxy, None, pixel_size)
    pad_nxy = nxy + (nxy // 2) * 2

    rotator = VolumeRotator(nz, nxy, nxy, origin="relion")
    R = roma.unitquat_to_rotmat(quaternions)
    T = rotations.translations_angstrom_to_torch(translations, nxy, pixel_size)
    theta = rotations.build_affine_matrix(R, T)
    V = rotator(V0.clone(), theta)
    # xy_pad_mode must match the chain's own -- this composition is only a
    # reference for it if it pads the same way.
    V = pad_volume(V, nxy, nz, None, True, xy_pad_mode="constant")

    scattering = Scattering(
        pad_nxy,
        pixel_size,
        voltage,
        scattering_model="multislice",
        ews_curvature_sign="positive",
        nz=nz,
        alpha=0.0,
        # The Propagation() default; the hand-built stage has to make the same
        # choice to compare equal.
        oversample=Propagation().oversample,
        # ... and hand the imager the wave on the propagation grid: the
        # transfer function is applied and the intensity formed there, and
        # the detector samples the intensity to the pixel grid.
        crop=False,
        progressbars=False,
    )
    s = Propagation().oversample
    with torch.no_grad():
        exitwaves = scattering(V)
    assert exitwaves.shape[-1] == pad_nxy * s

    # multislice evaluates the CTF at the volume's midplane, not its entry
    # face -- the chain applies this shift through ExitWave.defocus_shift,
    # built on the same public helper.
    defocus_shift = defocus_midplane_shift(nz, pixel_size)
    manual_ctf_params = {
        "cs": ctf_params["cs"],
        "dfu": ctf_params["dfu"] - defocus_shift,
    }

    aberration = Aberration(
        pad_nxy * s, pixel_size / s, voltage, aberration_model="nonlinear", alpha=0.0
    )
    with torch.no_grad():
        detector_waves = aberration(exitwaves, manual_ctf_params)

    # The Cc focal spread, on by default: summed over at the detector from
    # the same instrument constants `Envelopes()` carries.
    from specter.aberrations import FocalSpread
    from specter.constants import energy_to_wavelength

    env = Envelopes()
    detector = Detector(
        pixel_size,
        aberration_model="nonlinear",
        noise_model=None,
        oversample=s,
        focal_spread=FocalSpread.from_instrument(
            env.cc,
            voltage * 1e3,
            env.energy_spread,
            env.deltaV_V,
            env.deltaI_I,
            energy_to_wavelength(voltage),
        ),
    )
    dose_batch = torch.full((2,), dose)
    coincidence_radius = torch.zeros(2)
    with torch.no_grad():
        actual = detector(detector_waves, dose_batch, coincidence_radius, nxy=nxy)

    assert torch.allclose(actual, expected, atol=1e-4)

    # Cropping the WAVE to the pixel grid before imaging is a different, and
    # wrong, composition: |psi|^2's own bandwidth and the cross terms with
    # the scattering above Nyquist fold.
    from specter.scattering._kernels import fourier_crop_xy

    coarse_aberration = Aberration(
        pad_nxy, pixel_size, voltage, aberration_model="nonlinear", alpha=0.0
    )
    coarse_detector = Detector(
        pixel_size, aberration_model="nonlinear", noise_model=None
    )
    with torch.no_grad():
        cropped = coarse_detector(
            coarse_aberration(fourier_crop_xy(exitwaves, pad_nxy), manual_ctf_params),
            dose_batch,
            coincidence_radius,
            nxy=nxy,
        )
    assert not torch.allclose(cropped, expected, atol=1e-4)
