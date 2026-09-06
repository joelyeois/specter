"""
Simulator and inverse share one forward model.

SPECTER's central claim is that `specter simulate` and `specter reconstruct`
image a specimen with the same physics, so a reconstruction can be trusted
against data the simulator produced. That claim is asserted throughout the
docs and was, until this file, tested nowhere: no test imported both
`specter.imagegenerator` and `specter.ghostbuster`, and every reconstruction
test fed `torch.randn` as its observed images.

Two levels here. `test_forward_matches_*` pins the claim exactly -- the same
inputs must give the same images through either entry point. The round trip
then checks the claim end to end: images simulated from a known volume must
actually reconstruct back to it.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from specter.arrays import ball3d
from specter.ghostbuster import Reconstructor
from specter.imagegenerator import ImageGenerator

SCATTERING_MODELS = ["multislice", "projection", "firstborn", "rytov"]


def _inputs(n_particles: int = 4, box: int = 16):
    """A small, deterministic imaging problem shared by both entry points."""
    torch.manual_seed(0)
    volume = ball3d(box, box / 2).float() * 0.05
    quaternions = torch.nn.functional.normalize(
        torch.randn(n_particles, 4, generator=torch.Generator().manual_seed(1)), dim=-1
    )
    return dict(
        volume=volume,
        voxel_size=2.0,
        quaternions=quaternions,
        translations=torch.zeros(n_particles, 2),
        ctf_params={
            "dfu": torch.full((n_particles,), 12000.0),
            "dfv": torch.full((n_particles,), 12000.0),
            "dfang": torch.zeros(n_particles),
            "cs": torch.full((n_particles,), 2.7e7),
        },
        voltage=300.0,
        dose_per_angstrom=2.0,
        alpha=0.1,
    )


def _simulator(inputs: dict, scattering_model: str) -> ImageGenerator:
    """The forward model as `specter simulate` builds it."""
    return ImageGenerator(
        inputs["volume"],
        inputs["voxel_size"],
        inputs["quaternions"],
        inputs["translations"],
        inputs["ctf_params"],
        inputs["voltage"],
        inputs["dose_per_angstrom"],
        ice_model=None,
        noise_model=None,
        scattering_model=scattering_model,
        alpha=inputs["alpha"],
        verbose=False,
        progressbars=False,
    )


def _inverse(inputs: dict, scattering_model: str, **kwargs) -> Reconstructor:
    """The same forward model as `specter reconstruct` builds it."""
    return Reconstructor(
        V=inputs["volume"],
        voxel_size=inputs["voxel_size"],
        quaternions=inputs["quaternions"],
        translations=inputs["translations"],
        ctf_params=inputs["ctf_params"],
        voltage=inputs["voltage"],
        dose_per_angstrom=inputs["dose_per_angstrom"],
        alpha=inputs["alpha"],
        scattering_model=scattering_model,
        **kwargs,
    )


@pytest.mark.parametrize("scattering_model", SCATTERING_MODELS)
def test_forward_matches_between_simulator_and_inverse(scattering_model: str) -> None:
    """Identical inputs must give identical images through either entry point."""
    inputs = _inputs()
    idx = torch.arange(inputs["quaternions"].shape[0])

    torch.manual_seed(0)
    simulated = _simulator(inputs, scattering_model)(idx)
    torch.manual_seed(0)
    inverted = _inverse(inputs, scattering_model).forward(idx)

    assert simulated.shape == inverted.shape
    # Exact, not approximate: `Reconstructor` delegates to an `ImageGenerator`
    # rather than reimplementing it, so the two run the same code object on the
    # same inputs. Any difference at all means that stopped being true.
    assert torch.equal(simulated, inverted), (
        "simulate and reconstruct disagree on the forward model; "
        f"max abs diff {(simulated - inverted).abs().max().item():.3e}"
    )


def test_scattering_models_are_not_interchangeable() -> None:
    """
    The parity test above is only meaningful if the models it holds fixed differ.

    Cryo-EM images a weak phase object, and in that limit multislice reduces to
    the projection approximation, so at realistic specimen strength the models
    agree to well under a percent of the image standard deviation. They are
    still distinguishable, and this pins that: were a model silently ignored on
    one side of the comparison, the exact-equality assertion above would have
    to catch a difference that is small in absolute terms.
    """
    inputs = _inputs(n_particles=2)
    idx = torch.arange(2)
    images = {}
    for model in SCATTERING_MODELS:
        torch.manual_seed(0)
        images[model] = _simulator(inputs, model)(idx)

    for a, b in itertools.combinations(SCATTERING_MODELS, 2):
        assert not torch.equal(images[a], images[b]), (
            f"{a} and {b} produced identical images, so the parity test above "
            "would pass even if one entry point ignored scattering_model"
        )


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalised cross-correlation of two volumes, in [-1, 1]."""
    x = (a - a.mean()).flatten()
    y = (b - b.mean()).flatten()
    return float((x @ y) / (x.norm() * y.norm()))


def test_simulated_images_reconstruct_back_to_their_volume() -> None:
    """
    Images simulated from a known volume must reconstruct back to it.

    The end-to-end form of the claim above: `test_forward_matches_*` shows the
    two entry points agree, this shows agreeing is useful. Fitting is a plain
    AdamW loop rather than a Lightning `Trainer` so the test stays a statement
    about the forward model rather than about the training harness.

    Thresholds are calibrated against a deliberately wrong forward model, so
    they discriminate rather than merely clear the floor. Reconstructing with
    the CTF the images were simulated with drops the loss 24x and reaches a
    correlation of 0.94; reconstructing with the defocus wrong by 8000 A drops
    it 2x and reaches 0.37.
    """
    inputs = _inputs(n_particles=64, box=16)
    truth = inputs["volume"]
    idx = torch.arange(inputs["quaternions"].shape[0])

    torch.manual_seed(0)
    observed = _simulator(inputs, "projection")(idx).detach()

    start = torch.zeros_like(truth)
    reconstructor = _inverse({**inputs, "volume": start}, "projection", lr=1e-2)
    optimizer = torch.optim.AdamW([reconstructor.V], lr=5e-2)

    first_loss = None
    for _ in range(40):
        reconstructor._bind_refined_parameters()
        loss = ((reconstructor.forward(idx) - observed) ** 2).mean()
        if first_loss is None:
            first_loss = float(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert first_loss is not None
    assert float(loss) < first_loss / 10, (
        f"loss barely moved: {first_loss:.4e} -> {float(loss):.4e}"
    )

    recovered = _correlation(reconstructor.V.detach(), truth)
    assert recovered > 0.85, (
        f"reconstruction correlates only {recovered:.3f} with truth"
    )


# ---------------------------------------------------------------------------
# Cryo-ET
#
# `TomogramReconstructor` does not delegate to `TiltSeriesGenerator` the way
# `Reconstructor` delegates to `ImageGenerator`; it builds its own
# `IterativeScattering` and `Aberration`. Agreement is therefore a property
# that has to be maintained rather than one that follows from the structure,
# which is exactly what this pins.
# ---------------------------------------------------------------------------

TILT_BOX = (16, 32)


@pytest.mark.parametrize("scattering_model", ["projection", "multislice", "ctf"])
def test_tilt_forward_matches_between_simulator_and_inverse(
    scattering_model: str,
) -> None:
    """
    A tilt image must be the same whether simulated or produced by the inverse.

    Two divergences used to break this. `TiltSeriesGenerator` inherits
    `BaseImager._apply_defocus_shift`, which moves dfu/dfv to the volume's
    midplane for a model with a Z extent, and `BaseImager._init_optics`, which
    derives `specimen_absorption` from `scattering_model`. `TomogramReconstructor`
    inherits neither and had both wrong: multislice landed 2.1% of the image
    standard deviation away, and `scattering_model="ctf"` -- where the two ends
    disagreed on whether to apply the amplitude-contrast term at all -- 566%.

    Compared at zero tilt only. Away from it the two deliberately prepare the
    volume differently: `TiltSeriesGenerator` reflect-pads XY so a tilted ray
    never samples off-grid, while `TomogramReconstructor._prepare_volume`
    documents why it does not pad and lets the FOV mask discard the periphery
    instead. That is a design difference rather than a divergence, and at
    +/-10 degrees it accounts for 10% of the image standard deviation on its
    own -- which would swamp the physics this test is about.
    """
    from specter.ghostbuster import TomogramReconstructor
    from specter.imagegenerator import TiltSeriesGenerator

    torch.manual_seed(0)
    nz, nxy = TILT_BOX
    pixel_size = 2.0
    volume = (ball3d(nxy, nxy / 3).float()[:nz] * 0.05).reshape(1, nz, nxy, nxy)
    ctf_params = {
        "dfu": torch.full((1,), 12000.0),
        "dfv": torch.full((1,), 12000.0),
        "dfang": torch.zeros(1),
        "cs": torch.full((1,), 2.7e7),
    }

    generator = TiltSeriesGenerator(
        volume=volume.clone(),
        micrograph_size=nxy,
        pixel_size=pixel_size,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        angles=torch.tensor([0.0]),
        noise_model=None,
        scattering_model=scattering_model,
        alpha=0.1,
        tilt_axis="y",
        verbose=False,
        progressbars=False,
    )
    torch.manual_seed(0)
    _, _, clean = generator.generate_tilt_series(torch.tensor([0]))

    inverse = TomogramReconstructor(
        V=volume[0].clone(),
        voxel_size=pixel_size,
        quaternions=generator.quaternions,
        translations=torch.zeros(1, 2),
        ctf_params={k: v.clone() for k, v in ctf_params.items()},
        voltage=300.0,
        scattering_model=scattering_model,
        alpha=0.1,
    )

    assert inverse._defocus_shift_angstrom == generator._defocus_shift_angstrom, (
        "simulator and inverse disagree on the midplane defocus shift"
    )

    torch.manual_seed(0)
    simulated, inverted = clean[0, 0], inverse.forward(0)
    divergence = ((simulated - inverted).abs().max() / simulated.std()).item()
    assert divergence < 1e-3, (
        f"simulator and inverse disagree by {divergence:.2%} of the image "
        "standard deviation"
    )


def test_simulated_tilt_series_reconstructs_back_to_its_volume() -> None:
    """
    A simulated tilt series must reconstruct back to the volume it came from.

    Recovery, not pixel parity, is the right claim for the tilt path. The two
    ends deliberately prepare the volume differently at the edges --
    `TiltSeriesGenerator` reflect-pads XY so a tilted ray stays on-grid, and
    `TomogramReconstructor._prepare_volume` documents why it does not, leaving
    the FOV mask to drop the periphery from the loss. Demanding identical
    images would fail on a region neither end uses. What must agree is anything
    acting on the interior: the defocus convention, the amplitude-contrast
    term, the scattering model. A reconstruction cannot converge if those
    differ, so convergence is the test.

    Calibrated against a wrong forward model rather than against zero.
    Reconstructing with the CTF the tilt series was simulated with drops the
    loss ~49x and reaches a correlation of 0.77; getting the defocus wrong by
    8000 A drops it ~5x and reaches 0.51. Correlation is capped well below 1
    by the missing wedge, which a +/-45 degree range leaves regardless of how
    well the forward model matches.
    """
    from specter.ghostbuster import TomogramReconstructor
    from specter.imagegenerator import TiltSeriesGenerator

    torch.manual_seed(0)
    nz, nxy, pixel_size = TILT_BOX[0], TILT_BOX[1], 2.0
    truth = ball3d(nxy, nxy / 3).float()[:nz] * 0.05
    angles = torch.tensor([-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0])
    n_tilts = len(angles)
    ctf_params = {
        "dfu": torch.full((1,), 12000.0),
        "dfv": torch.full((1,), 12000.0),
        "dfang": torch.zeros(1),
        "cs": torch.full((1,), 2.7e7),
    }

    generator = TiltSeriesGenerator(
        volume=truth.reshape(1, nz, nxy, nxy).clone(),
        micrograph_size=nxy,
        pixel_size=pixel_size,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        angles=angles,
        noise_model=None,
        scattering_model="multislice",
        alpha=0.1,
        tilt_axis="y",
        verbose=False,
        progressbars=False,
    )
    torch.manual_seed(0)
    _, _, observed = generator.generate_tilt_series(torch.tensor([0]))

    reconstructor = TomogramReconstructor(
        V=torch.zeros_like(truth),
        voxel_size=pixel_size,
        quaternions=generator.quaternions,
        translations=torch.zeros(n_tilts, 2),
        ctf_params={k: v.expand(n_tilts).clone() for k, v in ctf_params.items()},
        voltage=300.0,
        scattering_model="multislice",
        alpha=0.1,
        lr=5e-2,
    )
    optimizer = torch.optim.AdamW([reconstructor.V], lr=5e-2)

    first_loss = None
    for _ in range(50):
        loss = (
            sum(
                ((reconstructor.forward(i) - observed[0, i]) ** 2).mean()
                for i in range(n_tilts)
            )
            / n_tilts
        )
        if first_loss is None:
            first_loss = float(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert first_loss is not None
    assert float(loss) < first_loss / 20, (
        f"loss barely moved: {first_loss:.4e} -> {float(loss):.4e}"
    )

    recovered = _correlation(reconstructor.V.detach(), truth)
    assert recovered > 0.65, f"tomogram correlates only {recovered:.3f} with truth"
