"""
Absorption derived from a measured inelastic mean free path.

The central test is that propagation delivers the mean free path the potential
encodes -- not that the code returns the number it was given, but that a plane
wave through a slab of that potential is attenuated by ``exp(-t/Lambda)``. That
is what makes the model checkable against a measurement rather than against
itself.
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest
import torch

from specter.constants import interaction_parameter
from specter.potential import (
    FULL_OCCUPANCY_POTENTIAL_V,
    INELASTIC_MFP_ICE_A,
    INELASTIC_MFP_PROTEIN_A,
    absorption_potential,
    apply_amplitude_contrast,
    inelastic_absorption_potential,
)
from specter.scattering import Scattering

VOLTAGE = 300.0


def test_absorption_potential_matches_vulovic_equation_3() -> None:
    """``V_ab = 1/(2 sigma Lambda)``, and 0.194 V for ice at 300 kV."""
    sigma = interaction_parameter(VOLTAGE)
    v_ab = absorption_potential(INELASTIC_MFP_ICE_A, VOLTAGE)
    assert v_ab == pytest.approx(1.0 / (2.0 * sigma * INELASTIC_MFP_ICE_A))
    assert v_ab == pytest.approx(0.194, abs=5e-4)


def test_absorption_potential_rejects_nonpositive_mfp() -> None:
    """A non-positive mean free path is a caller error, not an infinity."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be positive"):
            absorption_potential(bad, VOLTAGE)


def test_uniform_field_when_no_specimen_mfp_given() -> None:
    """Without a specimen mean free path the field is the solvent's, flat."""
    v = torch.rand(3, 5, 5) * 10.0
    v_ab = inelastic_absorption_potential(v, 1.0, VOLTAGE)
    expected = absorption_potential(INELASTIC_MFP_ICE_A, VOLTAGE)
    assert torch.allclose(v_ab, torch.full_like(v, expected))


def test_two_materials_interpolate_between_their_mean_free_paths() -> None:
    """Empty space gets the solvent's V_ab, full material the specimen's."""
    v = torch.zeros(8, 16, 16)
    v[3:5, 6:10, 6:10] = 10.0 * FULL_OCCUPANCY_POTENTIAL_V  # solidly occupied
    v_ab = inelastic_absorption_potential(
        v, 1.0, VOLTAGE, mfp_specimen_A=INELASTIC_MFP_PROTEIN_A
    )
    solvent = absorption_potential(INELASTIC_MFP_ICE_A, VOLTAGE)
    specimen = absorption_potential(INELASTIC_MFP_PROTEIN_A, VOLTAGE)

    assert float(v_ab[0, 0, 0]) == pytest.approx(solvent, rel=1e-5)
    assert float(v_ab[4, 8, 8]) == pytest.approx(specimen, rel=1e-5)
    # Protein's shorter mean free path means it absorbs more than the water it
    # displaces; every voxel lies between the two.
    assert specimen > solvent
    assert float(v_ab.min()) >= solvent - 1e-6
    assert float(v_ab.max()) <= specimen + 1e-6


def test_field_is_pointwise_and_so_pose_invariant() -> None:
    """
    The field depends on the potential, not on the beam direction.

    A transposed volume must give the transposed field. This is what allows it
    to be built in the molecule frame and rotated with the specimen, unlike a
    model carrying the plasmon's transverse point spread function.
    """
    v = torch.rand(12, 12, 12) * 8.0
    kwargs = dict(
        voxel_size=1.0, voltage_kv=VOLTAGE, mfp_specimen_A=INELASTIC_MFP_PROTEIN_A
    )
    direct = inelastic_absorption_potential(v, **kwargs)
    swapped = inelastic_absorption_potential(v.permute(2, 1, 0), **kwargs)
    assert torch.allclose(direct.permute(2, 1, 0), swapped, atol=1e-6)


@pytest.mark.parametrize("mfp_A", [2000.0, INELASTIC_MFP_ICE_A])
def test_multislice_recovers_the_mean_free_path_put_in(mfp_A: float) -> None:
    """
    A plane wave through the potential is attenuated by ``exp(-t/Lambda)``.

    Fitted against a per-thickness elastic-only control, so power the
    multislice band limit removes cannot be mistaken for absorption. This is
    the test that ties the model to a measured quantity.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, dx = 32, 2.0
    thicknesses = [100, 200, 400, 800]
    scattering = Scattering(
        nxy=n,
        pixel_size=dx,
        voltage=VOLTAGE,
        scattering_model="multislice",
        progressbars=False,
    ).to(device)

    ratios = []
    for thickness in thicknesses:
        nz = int(thickness / dx)
        # Uniform solvent: the mean free path is a bulk property, so the test
        # should not depend on any particular specimen structure.
        v = torch.full((1, nz, n, n), 4.55, device=device)
        v_ab = inelastic_absorption_potential(v, dx, VOLTAGE, mfp_solvent_A=mfp_A)
        with torch.no_grad():
            i_full = float(
                (scattering.multislice(torch.complex(v, v_ab)).abs() ** 2).mean()
            )
            i_elastic = float((scattering.multislice(v).abs() ** 2).mean())
        ratios.append(i_full / i_elastic)

    # thickness = Lambda * (-ln I); fit the slope.
    x = torch.tensor([-math.log(r) for r in ratios], dtype=torch.float64)
    y = torch.tensor(thicknesses, dtype=torch.float64)
    design = torch.stack([x, torch.ones_like(x)], dim=1)
    slope = float(torch.linalg.lstsq(design, y.unsqueeze(1)).solution.flatten()[0])
    assert slope == pytest.approx(mfp_A, rel=0.01)


def test_transmitted_fraction_is_physically_available() -> None:
    """
    Ice must not remove more of the beam than its cross section allows.

    Through 1200 A of ice, inelastic scattering removes 26%. The fitted
    ``alpha = 0.1`` removes 51% -- more than inelastic plus *all* elastic
    scattering combined (39%) -- which is the error this model exists to
    avoid. A regression that reintroduces it fails here.
    """
    sigma = interaction_parameter(VOLTAGE)
    thickness = 1200.0

    v_ab = absorption_potential(INELASTIC_MFP_ICE_A, VOLTAGE)
    physical = math.exp(-2.0 * sigma * v_ab * thickness)
    assert physical == pytest.approx(math.exp(-thickness / INELASTIC_MFP_ICE_A))
    assert 1.0 - physical == pytest.approx(0.262, abs=0.005)

    # alpha applied to ice's mean inner potential, for contrast.
    fitted = math.exp(-2.0 * sigma * 0.100 * 4.55 * thickness)
    assert 1.0 - fitted > 0.39


def test_amplitude_contrast_leaves_a_complex_potential_alone() -> None:
    """
    A potential that already carries absorption is returned unchanged.

    ``IterativeScattering`` calls ``apply_amplitude_contrast`` unconditionally,
    so without this guard a complex potential would be silently rotated in the
    complex plane by ``sqrt(1 - alpha^2) + i alpha``.
    """
    v = torch.rand(4, 4, 4)
    v_ab = inelastic_absorption_potential(v, 1.0, VOLTAGE)
    complex_v = torch.complex(v, v_ab)
    assert apply_amplitude_contrast(complex_v, alpha=0.1) is complex_v
    # The real path is untouched by the guard.
    assert torch.allclose(
        apply_amplitude_contrast(v, alpha=0.1),
        v * ((1 - 0.1**2) ** 0.5 + 1j * 0.1),
    )


def test_propagation_rejects_alpha_alongside_the_mfp_model() -> None:
    """The two routes to the imaginary potential would double-count."""
    from specter.settings import Propagation

    with pytest.raises(ValueError, match="double-count"):
        Propagation(absorption_model="inelastic_mfp", alpha=0.1)


def test_propagation_rejects_the_mfp_model_with_scattering_model_ctf() -> None:
    """``"ctf"`` has a real exit wave and absorbs at the lens instead."""
    from specter.settings import Propagation

    with pytest.raises(ValueError, match="not available with"):
        Propagation(absorption_model="inelastic_mfp", scattering_model="ctf")


def test_pipeline_drops_a_dataset_alpha_under_the_mfp_model() -> None:
    """
    A ``.cs``/``.star`` amplitude contrast must not reach the mfp path.

    `_resolve_imaging_parameters` normally lets the dataset's value override
    the config's, which would trip `Propagation`'s double-count guard. It is
    dropped instead, because CTF estimation takes amplitude contrast as an
    input and never fits it.
    """
    from dataclasses import dataclass

    from specter.pipelines._particles import _resolve_imaging_parameters

    @dataclass
    class _Config:
        cs_path: None = None
        star_path: None = None
        pixel_size: float = 1.0
        voltage: float = 300.0
        alpha: float = 0.1
        absorption_model: str = "inelastic_mfp"

    _, _, _, alpha = _resolve_imaging_parameters(_Config())
    assert alpha == 0.0
    _, _, _, alpha = _resolve_imaging_parameters(_Config(absorption_model="alpha"))
    assert alpha == 0.1


@pytest.mark.parametrize(
    "specimen_mfp", [None, INELASTIC_MFP_PROTEIN_A], ids=["ice-only", "ice+protein"]
)
def test_generator_transmission_matches_the_mean_free_path(
    specimen_mfp: float | None,
) -> None:
    """
    A full generator run transmits ``exp(-t/Lambda)`` through its ice.

    The end-to-end version of the mean-free-path test: not the potential in
    isolation but the image a generator produces, ice and aberrations
    included. Pins the wiring -- that the absorption field is read off the
    specimen before solvation and reaches `Scattering` as a complex volume.
    """
    import math

    import specter
    from specter.imagegenerator import ImageGenerator
    from specter.settings import Camera, Crowding, Envelopes, Ice, Optics, Propagation

    n, dx, thickness, dose = 48, 2.0, 400.0, 40.0
    volume = torch.zeros(n, n, n)
    volume[20:28, 20:28, 20:28] = 7.0
    ctf_params = {
        "dfu": torch.tensor([10000.0]),
        "dfv": torch.tensor([10000.0]),
        "dfang": torch.tensor([0.0]),
        "cs": torch.tensor([2.7e7]),
        "phaseshift": torch.tensor([0.0]),
    }
    specter.seed(3)
    generator = ImageGenerator(
        volume,
        dx,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.zeros(1, 2),
        ctf_params,
        VOLTAGE,
        dose_per_angstrom=dose,
        propagation=Propagation(
            absorption_model="inelastic_mfp", inelastic_mfp_specimen=specimen_mfp
        ),
        ice=Ice(model="gd", thickness=thickness),
        crowding=Crowding(n_points=0),
        camera=Camera(noise_model="none", detector_model="none"),
        envelopes=Envelopes(),
        optics=Optics(),
        progressbars=False,
    )
    with torch.no_grad():
        image = generator(torch.tensor([0]))

    transmission = float(image.mean()) / (dose * dx**2)
    expected = math.exp(-thickness / INELASTIC_MFP_ICE_A)
    # The specimen is a small fraction of the box, so giving it its own
    # (shorter) mean free path moves the mean only slightly -- and downwards.
    assert transmission == pytest.approx(expected, rel=0.02)
    if specimen_mfp is not None:
        assert transmission < expected


@pytest.mark.parametrize("cap", [2**14, 2**12, 2**11])
def test_slabbed_occupancy_matches_whole_volume(cap: int) -> None:
    """
    Bounding the occupancy blur's memory must not change the field.

    Evaluating it whole asked for 26 GiB on a batch of four 512-pixel boxes
    with 1642 slices of ice, which brought a `specter match particles` run
    down. Slabbing bounds it, and is exact because each slab is widened by
    `occupancy_blur_halo_voxels` and the margin discarded -- without which
    every slab boundary would become an edge the blur sees.
    """
    torch.manual_seed(0)
    v = torch.rand(2, 40, 24, 24) * 12.0
    kwargs = dict(
        voxel_size=1.0, voltage_kv=VOLTAGE, mfp_specimen_A=INELASTIC_MFP_PROTEIN_A
    )
    whole = inelastic_absorption_potential(v, max_voxels_per_slab=10**9, **kwargs)
    slabbed = inelastic_absorption_potential(v, max_voxels_per_slab=cap, **kwargs)
    assert torch.equal(slabbed, whole)


def test_dose_weights_raise_the_noise_floor_without_touching_the_signal() -> None:
    """
    A signal-preserving exposure filter leaves a rising noise floor.

    `Envelopes.dose_envelope` cannot express this: it attenuates the signal and
    leaves the noise white, by construction. Real motion correction sums frames
    under per-frequency weights normalised to sum to one, which leaves the
    signal alone and multiplies the noise power by ``sum(w^2)/n_frames``.
    Measured on EMPIAR-11377's own RBMC weights, that gain reaches ~7x by 90%
    of Nyquist -- far larger than anything absorption does, and the term that
    dominated a sim/experiment 2D classification until it was modelled.
    """
    from specter.microscope import Detector

    n_frames, n_bins, n = 8, 64, 64
    # Damage-aware weighting: frames are weighted nearly equally at low
    # frequency, where nothing has decayed yet, and increasingly unequally
    # toward Nyquist, where only the earliest frames still carry signal. It is
    # that growing inequality that raises the noise floor.
    ramp = torch.linspace(0.0, 1.0, n_bins)
    w = torch.stack(
        [torch.ones(n_bins) + (i - n_frames / 2) * ramp * 0.3 for i in range(n_frames)]
    ).clamp(min=0.0)

    flat = torch.full((n, n), 40.0)
    plain = Detector(
        pixel_size=1.0, noise_model="poisson", n_frames=n_frames, progressbars=False
    )
    weighted = Detector(
        pixel_size=1.0,
        noise_model="poisson",
        n_frames=n_frames,
        dose_weights=w,
        progressbars=False,
    )
    torch.manual_seed(0)
    a = plain.apply_coincidence(flat.clone(), torch.tensor(40.0), 0.3)
    torch.manual_seed(0)
    b = weighted.apply_coincidence(flat.clone(), torch.tensor(40.0), 0.3)

    # Signal (the mean) is untouched: the weights sum to n_frames at every
    # frequency, so the DC term is unchanged.
    assert float(b.mean()) == pytest.approx(float(a.mean()), rel=0.02)

    def band_power(x: torch.Tensor, lo: float, hi: float) -> float:
        p = torch.fft.rfft2(x - x.mean()).abs() ** 2
        ky = torch.fft.fftfreq(x.shape[0])
        kx = torch.fft.rfftfreq(x.shape[1])
        r = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2) / 0.5
        return float(p[(r >= lo) & (r < hi)].mean())

    low = band_power(b, 0.05, 0.15) / band_power(a, 0.05, 0.15)
    high = band_power(b, 0.7, 0.9) / band_power(a, 0.7, 0.9)
    # The floor rises with frequency rather than scaling uniformly.
    assert high > low


def test_dose_weights_axis_is_keyed_on_absolute_frequency() -> None:
    """
    The weights' radial axis ends at the MOVIE's Nyquist, not the image's.

    EMPIAR-11377's weights run to 1.368 1/A, twice the Nyquist of its
    0.731 A/px particles -- not because the movie was sampled differently
    (`hyperparams.cs` records the same 0.731) but because the weights array
    carries twice the radial sampling of the FCC beside it. Reading the axis
    as the particles' own Nyquist overstates the noise gain threefold: 6.80x
    against a measured 2.22x at 0.9 Nyquist, where the right mapping gives
    3.33x. Keying on absolute frequency is what makes a later crop harmless.
    """
    from specter.microscope import Detector

    n_frames, n_bins, n = 8, 64, 64
    ramp = torch.linspace(0.0, 1.0, n_bins)
    w = torch.stack(
        [torch.ones(n_bins) + (i - n_frames / 2) * ramp * 0.3 for i in range(n_frames)]
    ).clamp(min=0.0)

    flat = torch.full((n, n), 40.0)

    def top_band_gain(max_frequency: float | None) -> float:
        det = Detector(
            pixel_size=1.0,
            noise_model="poisson",
            n_frames=n_frames,
            dose_weights=w,
            dose_weights_max_frequency=max_frequency,
            progressbars=False,
        )
        torch.manual_seed(0)
        x = det.apply_coincidence(flat.clone(), torch.tensor(40.0), 0.3)
        p = torch.fft.rfft2(x - x.mean()).abs() ** 2
        ky = torch.fft.fftfreq(n)
        kx = torch.fft.rfftfreq(n)
        r = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2) / 0.5
        return float(p[(r >= 0.7) & (r < 0.9)].mean()) / float(
            p[(r >= 0.05) & (r < 0.2)].mean()
        )

    # The image's Nyquist is 0.5 1/A at a 1 A pixel. An axis that actually
    # runs to twice that is sampled only half way up by this image, so reading
    # it as if it ended at Nyquist walks further up the curve and overstates
    # the gain.
    assumed_nyquist = top_band_gain(None)
    actually_twice = top_band_gain(1.0)
    assert assumed_nyquist > actually_twice


def test_load_dose_weights_derives_its_frequency_axis() -> None:
    """
    The axis is derived from the job's files, or refused.

    A wrong axis raises nowhere downstream -- the weights still apply, just at
    the wrong frequencies -- so an underivable one is an error rather than a
    guess. The pixel size alone is not enough and must not be treated as
    though it were: CryoSPARC records 0.731 for EMPIAR-11377's weights, the
    particles' own pixel size, while the axis runs to twice their Nyquist.
    """
    import numpy as np

    from specter.io import load_dose_weights

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "refm_empirical_dw.npy")
    np.save(path, np.ones((4, 80), dtype=np.float32))

    # Nothing beside it: refuse rather than assume.
    with pytest.raises(ValueError, match="cannot determine which frequencies"):
        load_dose_weights(path)

    # Explicit frequency is always honoured.
    _, freq = load_dose_weights(path, max_frequency=1.25)
    assert freq == pytest.approx(1.25)

    # With an FCC of half the bins beside it, the axis spans twice Nyquist.
    np.save(os.path.join(tmp, "refm_fcc.npy"), np.ones((4, 40), dtype=np.float32))
    weights, freq = load_dose_weights(path, max_frequency=2.0)
    assert weights.shape == (4, 80)
    assert freq == pytest.approx(2.0)


@pytest.mark.parametrize(
    "model", ["multislice", "rytov", "firstborn", "kinematic", "projection"]
)
@pytest.mark.parametrize("sign", ["negative", "positive"])
def test_uniform_absorption_matches_explicit_field_and_gradient(model, sign):
    """The allocation-free path preserves each model's complex-potential semantics."""
    generator = torch.Generator().manual_seed(17)
    v = (
        torch.rand(1, 5, 8, 8, generator=generator, dtype=torch.float64) * 7
    ).requires_grad_()
    vab = absorption_potential(INELASTIC_MFP_ICE_A, VOLTAGE)
    kwargs = dict(
        nxy=8,
        nz=5,
        pixel_size=2.0,
        voltage=VOLTAGE,
        scattering_model=model,
        ews_curvature_sign=sign,
        progressbars=False,
    )
    implicit = Scattering(**kwargs, uniform_absorption=vab).double()
    explicit = Scattering(**kwargs).double()
    actual = implicit(v)
    expected = explicit(torch.complex(v, torch.full_like(v, vab)))
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    actual_grad = torch.autograd.grad(actual.abs().square().sum(), v)[0]
    expected_grad = torch.autograd.grad(expected.abs().square().sum(), v)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    "model", ["multislice", "rytov", "firstborn", "kinematic", "projection"]
)
def test_uniform_absorption_slab_matches_model_approximation(model):
    nz, dx = 10, 2.0
    vab = absorption_potential(INELASTIC_MFP_ICE_A, VOLTAGE)
    scattering = Scattering(
        nxy=8,
        nz=nz,
        pixel_size=dx,
        voltage=VOLTAGE,
        scattering_model=model,
        uniform_absorption=vab,
        progressbars=False,
    )
    a = scattering.sigma * dx * vab
    amplitude = (
        1 - nz * a
        if model == "firstborn"
        else 1 + nz * math.expm1(-a)
        if model == "kinematic"
        else math.exp(-nz * a)
    )
    actual = scattering(torch.zeros(1, nz, 8, 8)).abs().square().mean().item()
    assert actual == pytest.approx(amplitude**2, rel=1e-6)


def test_ctf_rejects_uniform_absorption():
    with pytest.raises(ValueError, match="uniform_absorption"):
        Scattering(8, 2.0, VOLTAGE, scattering_model="ctf", uniform_absorption=0.2)
    scattering = Scattering(8, 2.0, VOLTAGE, scattering_model="ctf")
    scattering.uniform_absorption = 0.2
    with pytest.raises(ValueError, match="uniform_absorption"):
        scattering(torch.zeros(1, 4, 8, 8))


@pytest.mark.parametrize("kind", ["micrograph", "tiltseries", "tomogram"])
@pytest.mark.parametrize("specimen_mfp", [None, INELASTIC_MFP_PROTEIN_A])
def test_iterative_consumers_reject_unsupported_mfp(kind, specimen_mfp):
    """Never silently simulate a real volume without the requested absorption."""
    from specter.imagegenerator import MicrographGenerator, TiltSeriesGenerator
    from specter.ghostbuster import TomogramReconstructor
    from specter.settings import Propagation

    v = torch.zeros(4, 8, 8)
    propagation = Propagation(
        absorption_model="inelastic_mfp", inelastic_mfp_specimen=specimen_mfp
    )
    with pytest.raises(
        ValueError, match="does not support absorption_model='inelastic_mfp'"
    ):
        if kind == "tomogram":
            TomogramReconstructor(
                v,
                2.0,
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.zeros(1, 2),
                {},
                VOLTAGE,
                propagation=propagation,
            )
        else:
            cls = MicrographGenerator if kind == "micrograph" else TiltSeriesGenerator
            cls(v, 8, 2.0, None, VOLTAGE, 10.0, propagation=propagation, optics=None)
