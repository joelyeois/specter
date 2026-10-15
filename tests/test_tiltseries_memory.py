"""Tilt-series memory guards preserve images, diagnostics, and random draws."""

import itertools

import mrcfile
import pytest
import torch

import specter
from specter.cli._cli import cli
from specter.cpu_threads import limited_cpu_threads
from specter.imaging import Exposure, Poses, Sample
from specter.settings import Ice

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
    ),
]
EXPOSURE = Exposure.tilt_series(torch.tensor([2.0, 4.0, 6.0]), 3)


def make(exposure="static"):
    specter.seed(3)
    volume = torch.zeros(1, 12, 32, 32)
    volume[:, 3:9, 12:20, 10:22] = 7.0 + torch.rand(6, 8, 12)
    moving = exposure in ("motion", "both")
    with limited_cpu_threads(1):
        return Sample.from_volume(volume, 2.0).tilt_series(
            Poses.tilt_series([-40.0, 0.0, 40.0]),
            EXPOSURE,
            voltage=300.0,
            micrograph_size=24,
            ice=Ice(
                model="random" if moving else None,
                motion_variance=0.38 if moving else None,
            ),
            damaged=exposure in ("damage", "both"),
            log=False,
        )


def run(series, device, exitwaves=False, clean=False):
    """
    Image every tilt from the global stream at seed 11, collecting the exit
    waves and clean images only when asked. Returns the images, the
    diagnostics, and the stream's state afterwards.
    """
    specter.seed(11)
    images, waves_out, clean_out = [], [], []
    with torch.no_grad(), limited_cpu_threads(1):
        for i in range(len(series)):
            waves = series.scatter(i, device=device)
            if exitwaves:
                waves_out.append(waves.pixels())
            intensity = waves.intensity()
            if clean:
                clean_out.append(intensity.pixels())
            images.append(intensity.detect(EXPOSURE[i : i + 1]).array)
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device == "cuda" else None
    diagnostics = (
        torch.cat(waves_out) if exitwaves else None,
        torch.cat(clean_out) if clean else None,
    )
    return torch.cat(images), diagnostics, state, cuda_state


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("exposure", ["static", "damage", "motion", "both"])
def test_every_tilt_propagates_the_series_own_volume(device, exposure):
    """
    Each tilt hands the multislice the series' volume itself, never a copy:
    a second tomogram-sized volume is what pushes a run off the device.
    """
    model = make(exposure)
    seen = []

    def check_volume(_module, args):
        assert args[0] is model.volume
        seen.append(float(args[0].sum()))

    handle = model.scattering.register_forward_pre_hook(check_volume)
    run(model, device)
    handle.remove()
    assert len(seen) == 3


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("exitwaves,clean", itertools.product([False, True], repeat=2))
def test_optional_diagnostics_preserve_images_and_random_stream(
    device, exitwaves, clean
):
    expected, _, rng, cuda_rng = run(make("both"), device)
    actual, (waves, intensity), actual_rng, actual_cuda_rng = run(
        make("both"), device, exitwaves=exitwaves, clean=clean
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _, (all_waves, all_intensity), _, _ = run(
        make("both"), device, exitwaves=True, clean=True
    )
    for collected, got, want in (
        (exitwaves, waves, all_waves),
        (clean, intensity, all_intensity),
    ):
        if collected:
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        else:
            assert got is None
    assert torch.equal(actual_rng, rng)
    if cuda_rng is not None:
        assert torch.equal(actual_cuda_rng, cuda_rng)


@pytest.mark.parametrize("save_exitwaves", [False, True])
def test_cli_exports_requested_exitwaves(tmp_path, save_exitwaves):
    volume = tmp_path / "volume.mrc"
    with mrcfile.new(volume) as mrc:
        mrc.set_data(torch.ones(8, 32, 32).numpy())
        mrc.voxel_size = 5.0
    output = tmp_path / "out"
    with limited_cpu_threads(1):
        cli(
            prog_name="specter",
            standalone_mode=False,
            args=[
                "simulate",
                "tiltseries",
                "--volume_path",
                str(volume),
                "--device",
                "cpu",
                "--n_tilts",
                "3",
                "--ice_model",
                "none",
                "--noise_model",
                "none",
                "--save_exitwaves",
                str(save_exitwaves).lower(),
                "--output_dir",
                str(output),
                "--seed",
                "5",
            ],
        )
    with mrcfile.open(output / "tilt_series.mrcs") as mrc:
        assert mrc.data.shape == (3, 32, 32)
    pairs = list(output.glob("*exitwave*.mrcs"))
    assert len(pairs) == (2 if save_exitwaves else 0)
    for path in pairs:
        with mrcfile.open(path) as mrc:
            assert mrc.data.shape == (3, 32, 32)
