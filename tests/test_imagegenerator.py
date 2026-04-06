import torch

from specter.imagegenerator import ImageGenerator, TiltSeriesGenerator


def test_image_generator_initialization(
    dummy_volume, ctf_params_standard, basic_setup_params
):
    # Flatten dummy_volume from (1, Z, Y, X) to (Z, Y, X) for ImageGenerator
    vol = dummy_volume.squeeze(0)

    quaternions = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    translations = torch.tensor([[0.0, 0.0]])

    gen = ImageGenerator(
        scattering_potential=vol,
        pixel_size=basic_setup_params["pixel_size"],
        quaternions=quaternions,
        translations=translations,
        ctf_params=ctf_params_standard,
        energy=basic_setup_params["energy"],
        dose_per_angstrom=basic_setup_params["dose_per_angstrom"],
        verbose=False,
    )

    assert gen.nxy == 64
    assert gen.nz == 32


def test_image_generator_forward(dummy_volume, ctf_params_standard, basic_setup_params):
    vol = dummy_volume.squeeze(0)
    quaternions = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    translations = torch.tensor([[0.0, 0.0]])

    gen = ImageGenerator(
        scattering_potential=vol,
        pixel_size=basic_setup_params["pixel_size"],
        quaternions=quaternions,
        translations=translations,
        ctf_params=ctf_params_standard,
        energy=basic_setup_params["energy"],
        dose_per_angstrom=basic_setup_params["dose_per_angstrom"],
        verbose=False,
    )

    idx = torch.tensor([0])
    images = gen.forward(idx)
    assert images.shape == (1, 64, 64)


def test_tilt_series_generator_forward(
    dummy_volume, ctf_params_standard, basic_setup_params
):
    # TiltSeriesGenerator expects 4D vol (B, Z, Y, X)
    angles = torch.tensor([0.0, 10.0])

    gen = TiltSeriesGenerator(
        vol=dummy_volume,
        micrograph_size=basic_setup_params["micrograph_size"],
        pixel_size=basic_setup_params["pixel_size"],
        ctf_params=ctf_params_standard,
        energy=basic_setup_params["energy"],
        dose_per_angstrom=basic_setup_params["dose_per_angstrom"],
        angles=angles,
        verbose=False,
    )

    idx = torch.tensor([0])
    tilt_series, exitwaves, clean_images = gen.generate_tilt_series(idx)
    assert tilt_series.shape == (1, 2, 64, 64)


def test_tilt_series_generator_batching(
    dummy_volume, ctf_params_standard, basic_setup_params
):
    # Verify that slice_batch_size is accepted and used
    angles = torch.tensor([10.0])  # Use non-zero angle to trigger iterative path

    gen = TiltSeriesGenerator(
        vol=dummy_volume,
        micrograph_size=basic_setup_params["micrograph_size"],
        pixel_size=basic_setup_params["pixel_size"],
        ctf_params=ctf_params_standard,
        energy=basic_setup_params["energy"],
        dose_per_angstrom=basic_setup_params["dose_per_angstrom"],
        angles=angles,
        slice_batch_size=4,  # Set batch size
        verbose=False,
    )

    assert gen.slice_batch_size == 4

    idx = torch.tensor([0])
    tilt_series, _, _ = gen.generate_tilt_series(idx)
    assert tilt_series.shape == (1, 1, 64, 64)
