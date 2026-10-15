"""
Simulator and inverse share one forward model.

SPECTER's central claim is that `specter simulate` and `specter reconstruct`
image a specimen with the same physics, so a reconstruction can be trusted
against data the simulator produced. That claim is asserted throughout the
docs and was, until this file, tested nowhere: no test imported both the
simulators and `specter.ghostbuster`, and every reconstruction test fed
`torch.randn` as its observed images.

Two levels here. `test_forward_matches_*` pins the claim exactly -- the same
inputs must give the same images through either entry point. The round trip
then checks the claim end to end: images simulated from a known volume must
actually reconstruct back to it.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from specter.arrays import ball3d, center_crop
from specter.ghostbuster import Reconstructor
from specter.imaging import CTF, Exposure, Poses, Specimen, simulate_particles
from specter.settings import Propagation, TiltGeometry

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


def _simulate(inputs: dict, scattering_model: str, idx: torch.Tensor) -> torch.Tensor:
    """
    The forward model as `specter simulate` runs it, noiseless, Cc off.

    Parity is about conventions the two sides must share. The focal spread
    is the one thing they deliberately do not: the simulator applies the Cc
    focal spread by default and the inverse takes no `Envelopes` at all.
    """
    poses = Poses(inputs["quaternions"], inputs["translations"], torch.arange(len(idx)))
    return simulate_particles(
        Specimen(inputs["volume"], inputs["voxel_size"]),
        poses[idx],
        CTF(inputs["ctf_params"])[idx],
        Exposure(inputs["dose_per_angstrom"]),
        voltage=inputs["voltage"],
        batch_size=len(idx),
        scatter=dict(scattering_model=scattering_model, alpha=inputs["alpha"]),
        coherence=dict(cc=None),
        camera=dict(noise=None),
    ).array


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
        **kwargs,
        propagation=Propagation(
            alpha=inputs["alpha"], scattering_model=scattering_model
        ),
    )


@pytest.mark.parametrize("scattering_model", SCATTERING_MODELS)
def test_forward_matches_between_simulator_and_inverse(scattering_model: str) -> None:
    """Identical inputs must give identical images through either entry point."""
    inputs = _inputs()
    idx = torch.arange(inputs["quaternions"].shape[0])

    torch.manual_seed(0)
    simulated = _simulate(inputs, scattering_model, idx)
    torch.manual_seed(0)
    inverted = _inverse(inputs, scattering_model).forward(idx)

    assert simulated.shape == inverted.shape
    # Exact, not approximate: `Reconstructor` runs the simulators' chain
    # (`specter.imaging`) rather than reimplementing it, so the two run the
    # same code on the same inputs. Any difference at all means that stopped
    # being true.
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
        images[model] = _simulate(inputs, model, idx)

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
    observed = _simulate(inputs, "projection", idx).detach()

    start = torch.zeros_like(truth)
    reconstructor = _inverse({**inputs, "volume": start}, "projection", lr=1e-2)
    optimizer = torch.optim.AdamW([reconstructor.V], lr=5e-2)

    first_loss = None
    for _ in range(40):
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
# `TomogramReconstructor` shares the lens and camera stages with `TiltSeries`
# (`ExitWave`) and the tilt geometry (`imaging._tilt.tilt_theta`,
# `tilt_defocus_shift`), but prepares its own volume and propagator. Agreement
# is therefore a property that has to be maintained rather than one that
# follows from the structure, which is exactly what this pins.
# ---------------------------------------------------------------------------

TILT_BOX = (16, 32)


@pytest.mark.parametrize("scattering_model", ["projection", "multislice", "ctf"])
def test_tilt_forward_matches_between_simulator_and_inverse(
    scattering_model: str,
) -> None:
    """
    A tilt image must be the same whether simulated or produced by the inverse.

    Two divergences used to break this. The tilt-series simulator moved
    dfu/dfv to the volume's midplane for a model with a Z extent, and derived
    `specimen_absorption` from `scattering_model`. `TomogramReconstructor`
    did neither and had both wrong: multislice landed 2.1% of the image
    standard deviation away, and `scattering_model="ctf"` -- where the two ends
    disagreed on whether to apply the amplitude-contrast term at all -- 566%.

    A third was the linear model's image: the inverse predicted the squared
    contrast while the simulators detect one plus it. Both now form it with
    `ExitWave.intensity`.

    Compared at zero tilt only. Away from it the two deliberately prepare the
    volume differently: `TiltSeries` reflect-pads XY so a tilted ray
    never samples off-grid, while `TomogramReconstructor._prepare_volume`
    documents why it does not pad and lets the FOV mask discard the periphery
    instead. That is a design difference rather than a divergence, and at
    +/-10 degrees it accounts for 10% of the image standard deviation on its
    own -- which would swamp the physics this test is about.
    """
    from specter.ghostbuster import TomogramReconstructor
    from specter.imaging import CTF, Exposure, Poses, Sample

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
    propagation = Propagation(scattering_model=scattering_model, alpha=0.1)
    poses = Poses.tilt_series([0.0], axis="y")

    series = Sample.from_volume(volume.clone(), pixel_size).tilt_series(
        poses,
        Exposure.tilt_series(2.0, 1),
        voltage=300.0,
        micrograph_size=nxy,
        tilt=TiltGeometry(tilt_axis="y"),
        propagation=propagation,
        log=False,
    )
    waves = series.scatter(0)
    # Cc off: the inverse takes no envelopes by design (see _simulate). The
    # intensity per unit dose, as the simulators detect it.
    intensity = waves.apply_ctf(CTF(ctf_params)).intensity(cc=None).pixels()[0]
    clean = center_crop(intensity, nxy, dim=(-2, -1))  # the field, as detected

    inverse = TomogramReconstructor(
        V=volume[0].clone(),
        voxel_size=pixel_size,
        quaternions=poses.quaternions,
        translations=torch.zeros(1, 2),
        ctf_params={k: v.clone() for k, v in ctf_params.items()},
        voltage=300.0,
        dose_per_angstrom=2.0,
        propagation=propagation,
    )

    assert inverse._defocus_shift_angstrom == waves.defocus_shift, (
        "simulator and inverse disagree on the midplane defocus shift"
    )

    # The inverse predicts counts.
    simulated = clean * 2.0 * pixel_size**2
    inverted = inverse.forward(0)
    divergence = ((simulated - inverted).abs().max() / simulated.std()).item()
    assert divergence < 1e-3, (
        f"simulator and inverse disagree by {divergence:.2%} of the image "
        "standard deviation"
    )


def test_simulated_tilt_series_reconstructs_back_to_its_volume() -> None:
    """
    A simulated tilt series must reconstruct back to the volume it came from.

    Recovery, not pixel parity, is the right claim for the tilt path; what
    must agree is anything acting on the interior: the defocus convention,
    the amplitude-contrast term, the scattering model. A reconstruction
    cannot converge if those differ, so convergence is the test.

    Both ends take the specimen's surroundings as vacuum here. By default
    `TiltSeries` reflect-pads a volume too narrow for its tilt coverage, while
    `TomogramReconstructor._prepare_volume` documents why it does not; and a
    tilt keeps its FFT margin through the transfer function, which carries
    whatever surrounds the field into its edges. With the simulator's
    padding, that edge band differs between the two ends and holds the
    recovered correlation to 0.645; without it, 0.845. Correlation is capped
    well below 1 by the missing wedge, which a +/-45 degree range leaves
    regardless of how well the forward model matches.
    """
    from specter.ghostbuster import TomogramReconstructor
    from specter.imaging import Sample

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
    poses = Poses.tilt_series(angles, axis="y")
    series = Sample.from_volume(truth.reshape(1, nz, nxy, nxy).clone(), pixel_size)
    series = series.tilt_series(
        poses,
        Exposure.tilt_series(2.0, n_tilts),
        voltage=300.0,
        micrograph_size=nxy,
        tilt=TiltGeometry(tilt_axis="y"),
        propagation=Propagation(scattering_model="multislice", alpha=0.1),
        pad_volume=False,
        log=False,
    )
    dose_per_pixel = 2.0 * pixel_size**2
    with torch.no_grad():
        # Expected counts per pixel: the intensity per unit dose, times the dose.
        observed = (
            torch.stack(
                [
                    center_crop(
                        series.scatter(i)
                        .apply_ctf(CTF(ctf_params))
                        .intensity()
                        .pixels()[0],
                        nxy,
                        dim=(-2, -1),
                    )
                    for i in range(n_tilts)
                ]
            )[None]
            * dose_per_pixel
        )

    reconstructor = TomogramReconstructor(
        V=torch.zeros_like(truth),
        voxel_size=pixel_size,
        quaternions=poses.quaternions,
        translations=torch.zeros(n_tilts, 2),
        ctf_params={k: v.expand(n_tilts).clone() for k, v in ctf_params.items()},
        voltage=300.0,
        dose_per_angstrom=2.0,
        lr=5e-2,
        propagation=Propagation(scattering_model="multislice", alpha=0.1),
    )
    optimizer = torch.optim.AdamW([reconstructor.V], lr=5e-2)

    first_loss = None
    for _ in range(50):
        loss = (
            sum(
                ((reconstructor.forward(i) - observed[0, i]) ** 2).mean()
                / dose_per_pixel
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
