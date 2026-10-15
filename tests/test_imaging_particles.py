"""
The particle chain (`specter.imaging`) against the generator it replaced.

The references under ``tests/test_data/imaging_particles_*.pt`` are the
outputs of the retired `ImageGenerator`/`ImageGeneratorFromCoordinates` on
these exact inputs and seeds, written once while both existed. The chain was
bit-identical to them then, so the exact comparisons here still hold it to
the generator's draws and noise; the inexact ones are the two places the
chain applies the same product in a different order.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from specter import rotations
from specter.imaging import CTF, Exposure, Poses, Specimen, simulate_particles
from specter.ice import RandomIcemaker

REFERENCES = Path(__file__).parent / "test_data"

N = 4
NXY = 24
PIXEL = 1.5
VOLTAGE = 300.0
DOSE = 30.0


def _template() -> torch.Tensor:
    V = torch.zeros(NXY, NXY, NXY)
    V[8:16, 9:15, 10:14] = 4.0
    V[11:13, 11:13, 6:18] = 9.0
    return V


def _inputs() -> tuple[Poses, CTF]:
    g = torch.Generator().manual_seed(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        quats = rotations.random_quaternion(N).reshape(N, 4)
    shifts = (torch.rand(N, 2, generator=g) - 0.5) * 4.0
    ctf = CTF.from_values(
        torch.linspace(8000.0, 14000.0, N), astigmatism=300.0, astigmatism_angle=30.0
    )
    return Poses(quats, shifts, torch.arange(N)), ctf


def _reference(name: str) -> torch.Tensor:
    """The retired generator's output for this case."""
    return torch.load(REFERENCES / f"imaging_particles_{name}.pt", weights_only=True)


def test_noiseless_images_match_image_generator() -> None:
    poses, ctf = _inputs()
    want = _reference("noiseless")
    with torch.no_grad():
        got = simulate_particles(
            Specimen(_template(), PIXEL),
            poses,
            ctf,
            Exposure(DOSE),
            voltage=VOLTAGE,
            batch_size=N,
            scatter=dict(pad_fft=True),
            camera=dict(noise=None),
        )
    assert got.array.shape == want.shape
    assert torch.equal(got.array, want)


def test_crowded_solvated_noisy_images_match_image_generator() -> None:
    """Same seed, same per-image keys: the same neighbours, ice and counts."""
    poses, ctf = _inputs()
    icemaker = RandomIcemaker(PIXEL, NXY, 40)
    want = _reference("crowded_solvated_noisy")
    specimen = (
        Specimen(_template(), PIXEL)
        .crowd(min_distance=12.0)
        .solvate(icemaker=icemaker, model="random", thickness=40 * PIXEL)
    )
    with torch.no_grad():
        got = simulate_particles(
            specimen,
            poses,
            ctf,
            Exposure(DOSE),
            voltage=VOLTAGE,
            seed=7,
            batch_size=2,
            scatter=dict(pad_fft=True),
        )
    assert torch.equal(got.array, want)


def test_specimen_damage_matches_image_generator() -> None:
    poses, ctf = _inputs()
    icemaker = RandomIcemaker(PIXEL, NXY, NXY)
    want = _reference("specimen_damage")
    specimen = Specimen(_template(), PIXEL).damage().solvate(icemaker=icemaker)
    with torch.no_grad():
        got = simulate_particles(
            specimen,
            poses,
            ctf,
            Exposure(DOSE),
            voltage=VOLTAGE,
            seed=7,
            batch_size=N,
            scatter=dict(pad_fft=True),
            camera=dict(noise=None),
        )
    assert torch.equal(got.array, want)


def test_a_slice_of_the_stack_draws_what_the_stack_drew() -> None:
    poses, ctf = _inputs()
    specimen = Specimen(_template(), PIXEL).crowd(min_distance=12.0)
    kwargs = dict(voltage=VOLTAGE, seed=3, scatter=dict(pad_fft=True))
    with torch.no_grad():
        whole = simulate_particles(specimen, poses, ctf, Exposure(DOSE), **kwargs)
        part = simulate_particles(
            specimen, poses[2:4], ctf[2:4], Exposure(DOSE), batch_size=2, **kwargs
        )
    assert torch.equal(part.array, whole.array[2:4])


def test_seeded_ice_does_not_depend_on_the_batch() -> None:
    """
    A seeded specimen in ice is the same whether its images are built together
    or one at a time, bit for bit on the CPU. The ice splat used to accumulate
    across threads in scheduling order, which moved every canvas by ~1e-6 V.
    """
    torch.manual_seed(0)
    template = torch.zeros(32, 32, 32)
    template[12:20, 10:22, 12:20] = 5.0
    poses = Poses.random(4)
    specimen = Specimen(template, 2.0).solvate(model="random", thickness=200.0)
    with torch.no_grad():
        whole = specimen.materialize(poses, voltage=VOLTAGE, seed=3)
        singles = [
            specimen.materialize(poses[i : i + 1], voltage=VOLTAGE, seed=3)
            for i in range(4)
        ]
    assert torch.equal(whole, torch.cat(singles))


def test_scatter_propagates_what_materialize_returns() -> None:
    poses, ctf = _inputs()
    specimen = Specimen(_template(), PIXEL).crowd(min_distance=12.0)
    with torch.no_grad():
        volumes = specimen.materialize(poses[:2], voltage=VOLTAGE, seed=5)
        waves = specimen.scatter(poses[:2], voltage=VOLTAGE, seed=5)
    assert volumes.shape == (2, NXY, NXY, NXY)
    assert volumes.sum() > _template().sum() * 2  # the neighbours are in it
    from specter.scattering import Scattering

    direct = Scattering(NXY, PIXEL, VOLTAGE, nz=NXY, oversample=2, crop=False)
    assert torch.equal(waves.array, direct(volumes))
    assert specimen.plan()[-1].startswith("scatter: multislice")


def test_the_exposure_is_never_counted_twice() -> None:
    poses, ctf = _inputs()
    with torch.no_grad():
        waves = (
            Specimen(_template(), PIXEL)
            .damage()
            .scatter(poses[:1], voltage=VOLTAGE, exposure=Exposure(DOSE))
        )
        with pytest.raises(ValueError, match="twice"):
            waves.apply_ctf(ctf[:1], dose_envelope=Exposure(DOSE))
        aberrated = waves.apply_ctf(ctf[:1])
        with pytest.raises(ValueError, match="already been applied"):
            aberrated.apply_ctf(ctf[:1])


def test_the_cc_envelope_matches_the_generator_s() -> None:
    """The envelope on the wave at intensity() is the generator's envelope on
    the transfer function: the two are the same product in Fourier space."""
    poses, ctf = _inputs()
    want = _reference("cc_envelope")
    with torch.no_grad():
        got = simulate_particles(
            Specimen(_template(), PIXEL),
            poses,
            ctf,
            Exposure(DOSE),
            voltage=VOLTAGE,
            batch_size=N,
            scatter=dict(pad_fft=True),
            coherence=dict(temporal_coherence="envelope"),
            camera=dict(noise=None),
        )
    assert torch.allclose(got.array, want, rtol=1e-5, atol=1e-5 * float(want.max()))


def _compare(
    reference: str,
    recipe_kwargs: dict,
    *,
    exact: bool = True,
    specimen: Specimen | None = None,
    batch_size: int = 2,
) -> None:
    poses, ctf = _inputs()
    want = _reference(reference)
    with torch.no_grad():
        got = simulate_particles(
            specimen if specimen is not None else Specimen(_template(), PIXEL),
            poses,
            ctf,
            Exposure(DOSE),
            voltage=VOLTAGE,
            seed=7,
            batch_size=batch_size,
            **{"camera": dict(noise=None), **recipe_kwargs},
        )
    if exact:
        assert torch.equal(got.array, want)
    else:
        assert torch.allclose(got.array, want, rtol=1e-5, atol=1e-5 * float(want.max()))


@pytest.mark.parametrize("scattering_model", ["rytov", "firstborn", "projection"])
def test_every_wave_model_matches_the_generator(scattering_model: str) -> None:
    _compare(
        f"model_{scattering_model}",
        dict(scatter=dict(scattering_model=scattering_model, alpha=0.1)),
    )


def test_the_linear_model_matches_the_generator() -> None:
    """Exact without a focal spread; with one, the envelope is applied to the
    real contrast rather than inside the transfer function, the same product."""
    _compare(
        "linear_coherent",
        dict(scatter=dict(scattering_model="ctf", alpha=0.1), coherence=dict(cc=None)),
    )
    _compare(
        "linear_cc",
        dict(scatter=dict(scattering_model="ctf", alpha=0.1)),
        exact=False,
    )


def test_per_image_settings_reach_their_own_images() -> None:
    """
    B-factor, coincidence radius and anisomag given one value per image are
    cut to each batch along with the images: a run in batches of two images
    exactly as each image run alone with its own values.
    """
    poses, ctf = _inputs()
    bfactor = torch.tensor([10.0, 20.0, 30.0, 40.0])
    radius = torch.tensor([0.0, 0.5, 1.0, 1.5])
    anisomag = torch.tensor([[1.01, 0.002], [0.0, 0.99]]).expand(N, 2, 2).clone()
    anisomag[2] = torch.tensor([[0.99, 0.0], [0.003, 1.02]])
    specimen = Specimen(_template(), PIXEL)
    common = dict(voltage=VOLTAGE, scatter=dict(pad_fft=True))
    with torch.no_grad():
        batched = simulate_particles(
            specimen,
            poses,
            ctf,
            Exposure(DOSE),
            batch_size=2,
            optics=dict(bfactor=bfactor, convergence_angle=0.05),
            camera=dict(noise=None, coincidence_radius=radius, anisomag=anisomag),
            **common,
        )
        alone = [
            simulate_particles(
                specimen,
                poses[i : i + 1],
                ctf[i : i + 1],
                Exposure(DOSE),
                optics=dict(bfactor=float(bfactor[i]), convergence_angle=0.05),
                camera=dict(
                    noise=None,
                    coincidence_radius=float(radius[i]),
                    anisomag=anisomag[i : i + 1],
                ),
                **common,
            ).array
            for i in range(N)
        ]
    torch.testing.assert_close(batched.array, torch.cat(alone))


def test_material_absorption_and_aperture_match_the_generator() -> None:
    icemaker = RandomIcemaker(PIXEL, NXY, NXY)
    propagation = dict(absorption_model="inelastic_mfp", inelastic_mfp_specimen=2460.0)
    _compare(
        "material_absorption",
        dict(scatter=dict(objective_aperture=12.0, **propagation)),
        specimen=Specimen(_template(), PIXEL).solvate(icemaker=icemaker),
    )


def test_the_torch_ctf_backend_matches_the_generator() -> None:
    """At the generator's batch size: torch_ctf's transfer function differs
    in the last bit between a batch of two and of four, in the generator too."""
    _compare("torch_ctf", dict(optics=dict(backend="torch_ctf")), batch_size=N)


def test_the_dry_specimen_is_the_generator_s_clean_reference() -> None:
    """`dry()` keeps the box and the neighbours, drops the ice and the damage."""
    poses, ctf = _inputs()
    icemaker = RandomIcemaker(PIXEL, NXY, NXY)
    specimen = (
        Specimen(_template(), PIXEL).crowd(min_distance=12.0).solvate(icemaker=icemaker)
    )
    with torch.no_grad():
        dry = specimen.dry().scatter(poses[:2], voltage=VOLTAGE, seed=7, pad_fft=True)
    assert torch.equal(dry.pixels(), _reference("dry_clean"))


def _atoms() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(3)
    coordinates = (torch.rand(40, 3, generator=g) - 0.5) * 18.0
    atomic_numbers = torch.tensor([6, 7, 8, 16] * 10)
    return coordinates, atomic_numbers


def test_a_specimen_rendered_from_atoms_matches_the_generator() -> None:
    poses, ctf = _inputs()
    coordinates, atomic_numbers = _atoms()
    icemaker = RandomIcemaker(PIXEL, NXY, NXY)
    specimen = (
        Specimen.from_atoms(coordinates.clone(), atomic_numbers, PIXEL, NXY)
        .crowd(min_distance=14.0)
        .solvate(icemaker=icemaker)
    )
    with torch.no_grad():
        got = simulate_particles(
            specimen, poses, ctf, Exposure(DOSE), voltage=VOLTAGE, seed=7, batch_size=N
        )
    assert torch.equal(got.array, _reference("atoms"))


def test_the_image_is_differentiable_in_the_atoms() -> None:
    poses, ctf = _inputs()
    coordinates, atomic_numbers = _atoms()
    coordinates.requires_grad_(True)
    specimen = Specimen.from_atoms(coordinates, atomic_numbers, PIXEL, NXY)
    waves = specimen.scatter(poses[:1], voltage=VOLTAGE)
    image = waves.apply_ctf(ctf[:1]).intensity().detect(Exposure(DOSE), noise=None)
    image.array.square().sum().backward()
    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all() and coordinates.grad.abs().sum() > 0


def test_poses_and_ctf_draw_from_their_own_seed() -> None:
    """
    A seed makes the poses and the transfer-function terms repeat without
    touching the global generator, so a seeded stack needs no specter.seed()
    to fix what it images; different seeds draw differently.
    """
    torch.manual_seed(1)
    expected = torch.rand(1)
    torch.manual_seed(1)
    first = Poses.random(4, shift=3.0, seed=0)
    again = Poses.random(4, shift=3.0, seed=0)
    assert torch.equal(torch.rand(1), expected)
    assert torch.equal(first.quaternions, again.quaternions)
    assert torch.equal(first.shifts, again.shifts)
    other = Poses.random(4, shift=3.0, seed=1)
    assert not torch.equal(first.quaternions, other.quaternions)

    ctf = CTF.sample(4, defocus=[10000.0, 20000.0], astigmatism=[0, 500], seed=0)
    same = CTF.sample(4, defocus=[10000.0, 20000.0], astigmatism=[0, 500], seed=0)
    assert all(torch.equal(ctf.params[k], same.params[k]) for k in ctf.params)
    differs = CTF.sample(4, defocus=[10000.0, 20000.0], seed=1)
    assert not torch.equal(ctf.params["dfu"], differs.params["dfu"])
