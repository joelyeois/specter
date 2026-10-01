"""Tilt-series memory guards preserve images, diagnostics, and random draws."""

import itertools

import mrcfile
import pytest
import torch

import specter
from specter.cli._cli import cli
from specter.cpu_threads import limited_cpu_threads
from specter.imagegenerator import TiltSeriesGenerator
from specter.settings import Camera, Envelopes, Ice


DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
    ),
]


def make(device, *, exposure="static", scale=1.0):
    specter.seed(3)
    volume = torch.zeros(1, 12, 32, 32)
    volume[:, 3:9, 12:20, 10:22] = 7.0 + torch.rand(6, 8, 12)
    with limited_cpu_threads(1):
        return TiltSeriesGenerator(
            volume,
            24,
            2.0,
            None,
            300.0,
            torch.tensor([2.0, 4.0, 6.0]),
            angles=[-40.0, 0.0, 40.0],
            optics=None,
            envelopes=Envelopes(
                dose_envelope=exposure in ("damage", "both"),
                dose_envelope_target="specimen",
            ),
            ice=Ice(
                model="random" if exposure in ("motion", "both") else None,
                motion_variance=0.38 if exposure in ("motion", "both") else None,
            ),
            camera=Camera(noise_model="poisson", detector_model=None),
            potential_scale=scale,
            verbose=False,
            progressbars=False,
        ).to(device)


def run(model, **kwargs):
    specter.seed(11)
    with torch.no_grad(), limited_cpu_threads(1):
        result = model.generate_tilt_series(torch.tensor([0]), **kwargs)
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if model.device.type == "cuda" else None
    return result, state, cuda_state


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("exposure", ["static", "damage", "motion", "both"])
@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_volume_guard_preserves_scaled_and_per_tilt_physics(device, exposure, scale):
    reference = make(device, exposure=exposure, scale=scale)
    # Force the original unconditional multiply to compare the actual physics.
    reference._potential_scale_is_unity = False
    expected, _, _ = run(reference)
    model = make(device, exposure=exposure, scale=scale)
    seen = []

    def check_volume(_module, args):
        source = args[0]
        if scale == 1.0:
            assert source is model.volume
        else:
            assert source is not model.volume
            torch.testing.assert_close(source, model.volume * scale, rtol=0, atol=0)
        seen.append(float(source.sum()))

    handle = model.iterative_scattering.register_forward_pre_hook(check_volume)
    actual, _, _ = run(model)
    handle.remove()
    assert len(seen) == 3
    for before, after in zip(expected, actual, strict=True):
        torch.testing.assert_close(after, before, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("exitwaves,clean", itertools.product([False, True], repeat=2))
def test_optional_diagnostics_preserve_images_and_random_stream(
    device, exitwaves, clean
):
    expected, rng, cuda_rng = run(make(device, exposure="both"))
    actual, actual_rng, actual_cuda_rng = run(
        make(device, exposure="both"),
        collect_exitwaves=exitwaves,
        collect_clean_images=clean,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    for index, collected in ((1, exitwaves), (2, clean)):
        if collected:
            torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
        else:
            assert actual[index] is None
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
