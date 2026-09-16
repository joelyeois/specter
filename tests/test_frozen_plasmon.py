"""Operator, exposure and source invariants for the production plasmon path."""

import math
import pytest
import torch
import roma

from specter.inelastic import (
    AtomicPlasmonSpecimen,
    BrownianCoordinates,
    ExposureStep,
    FrozenPlasmonForward,
    PlasmonFilter,
    PotentialState,
)
from specter.microscope import Detector
from specter.potential import absorption_potential
from specter.rotations import build_affine_matrix
from specter.scattering import IterativeScattering, Scattering


def propagator(n=12, dx=2.0, **kwargs):
    return IterativeScattering(n, dx, 300.0, progressbars=False, **kwargs)


def test_uniform_slab_beer_lambert():
    filt = PlasmonFilter.approximate_drude(2.0, 300.0, padding=0)
    s = propagator()
    for nz in (1, 5, 20):
        v = torch.ones(1, nz, 12, 12) * 4.5
        a = torch.full_like(v, absorption_potential(3950.0, 300.0))
        wave = s(v, 0.0, absorption_source=a, absorption_filter=filt)
        assert float(wave.abs().square().mean()) == pytest.approx(
            math.exp(-nz * 2 / 3950), abs=3e-6
        )


@pytest.mark.parametrize("reverse", ["negative", "positive"])
def test_paired_eager_parity_and_gradients(reverse):
    torch.manual_seed(10)
    v = (torch.rand(1, 5, 12, 12) * 4).requires_grad_()
    a = (torch.rand_like(v) * 0.3).requires_grad_()
    filt = PlasmonFilter.approximate_drude(2.0, 300.0, padding=4)
    eager = Scattering(12, 2.0, 300.0, progressbars=False, ews_curvature_sign=reverse)
    expected = eager(torch.complex(v, filt(a)))
    actual = propagator(ews_curvature_sign=reverse)(
        v, 0.0, absorption_source=a, absorption_filter=filt
    )
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    weights = torch.rand(1, 12, 12)
    g1 = torch.autograd.grad(
        (actual.abs().square() * weights).sum(), (v, a), retain_graph=True
    )
    g2 = torch.autograd.grad((expected.abs().square() * weights).sum(), (v, a))
    for x, y in zip(g1, g2):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=1e-4)


def test_tilt_checkpoint_and_slice_batch_parity():
    torch.manual_seed(2)
    v = torch.rand(1, 8, 12, 12, requires_grad=True)
    a = torch.rand(1, 8, 12, 12, requires_grad=True)
    pose = build_affine_matrix(roma.rotvec_to_rotmat(torch.tensor([[0.0, 0.43, 0.0]])))
    filt = PlasmonFilter.approximate_drude(2.0, 300.0, padding=4)
    s = propagator(pad_fft=True, fft_pad_margin=3)
    plain = s(v, pose, absorption_source=a, absorption_filter=filt)
    ckpt = s(
        v,
        pose,
        absorption_source=a,
        absorption_filter=filt,
        slice_batchsize=3,
        checkpoint_chunks=2,
    )
    torch.testing.assert_close(plain, ckpt)
    weight = torch.rand_like(plain.real)
    g1 = torch.autograd.grad(
        (plain.abs().square() * weight).sum(), (v, a), retain_graph=True
    )
    g2 = torch.autograd.grad((ckpt.abs().square() * weight).sum(), (v, a))
    for x, y in zip(g1, g2):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-4)
        assert torch.isfinite(x).all() and x.abs().sum() > 0


def test_detector_intensity_matches_wave_path():
    torch.manual_seed(4)
    detector = Detector(2.0, dqe0=0.8, noise_model=None, progressbars=False)
    wave = torch.randn(2, 12, 12, dtype=torch.complex64)
    dose = torch.tensor([3.0, 7.0])
    radius = torch.zeros(2)
    torch.testing.assert_close(
        detector(wave, dose, radius),
        detector.from_intensity(wave.abs().square(), dose, radius),
    )


def test_incoherent_exposure_and_partition_invariance():
    # Optical phase reversals cancel in a coherent average but cannot erase counts.
    v = torch.zeros(1, 3, 12, 12)

    def state(step):
        return PotentialState(v, v)

    def optics(wave, step):
        return wave if step.midpoint < 2 else -wave

    model = FrozenPlasmonForward(
        propagator(), Detector(2.0, dqe0=0.8, n_frames=1), None, optics=optics
    )
    one = model(state, [4.0], substeps=1)
    many = model(state, [4.0], substeps=4)
    torch.testing.assert_close(one.images, many.images)
    torch.testing.assert_close(many.images, torch.full_like(many.images, 4 * 4 * 0.8))
    assert many.intensities.min() > 0.999


def test_unequal_doses_and_cumulative_tilt_exposure():
    starts = []
    v = torch.zeros(1, 2, 12, 12)

    def state(step):
        starts.append(step.midpoint)
        return PotentialState(v, torch.full_like(v, 0.1 * step.midpoint))

    model = FrozenPlasmonForward(propagator(), Detector(2.0, n_frames=1), None)
    result = model(state, [1.0, 3.0], substeps=2, pre_exposure=2.0, poses=[0.0, 0.0])
    assert starts == [2.25, 2.75, 3.75, 5.25]
    assert result.images.shape == (2, 1, 12, 12)
    assert result.images[1].mean() > result.images[0].mean()


def test_motion_and_detector_replay_preserve_rng():
    coords = torch.zeros(10, 3)
    motion = BrownianCoordinates(coords, 2.0, 17)
    first = motion.advance_to(2.0)
    motion.reset()
    torch.testing.assert_close(first, motion.advance_to(2.0))
    v = torch.zeros(1, 2, 12, 12)
    model = FrozenPlasmonForward(
        propagator(), Detector(2.0, noise_model="poisson", n_frames=1), None
    )
    before = torch.random.get_rng_state()
    a = model(lambda step: PotentialState(v, v), [3.0, 2.0], detector_seed=15)
    assert torch.equal(before, torch.random.get_rng_state())
    b = model(lambda step: PotentialState(v, v), [3.0, 2.0], detector_seed=15)
    torch.testing.assert_close(a.images, b.images, rtol=0, atol=0)


def test_atomic_source_conserves_calibrated_strength():
    specimen = AtomicPlasmonSpecimen(
        torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        torch.tensor([6, 8]),
        (8, 12, 12),
        2.0,
        {6: 3.0, 8: 5.0},
        provenance="test known coefficients",
    )
    state = specimen(ExposureStep(0, 0.0, 1.0))
    assert float(state.absorption_source.sum() * 8) == pytest.approx(8.0)
    assert state.elastic.shape == state.absorption_source.shape == (1, 8, 12, 12)


def test_kernel_quadrature_and_bad_spectrum():
    f = PlasmonFilter.approximate_drude(2.0, 300.0, padding=0)
    k = f.kernel((12, 12), torch.zeros(12, 12))
    q = 1 / (12 * 2)
    from specter.constants import energy_to_wavelength

    dcs = torch.trapezoid(
        f.loss_density
        / ((energy_to_wavelength(300.0) * q) ** 2 + (f.energies / 600000) ** 2),
        f.energies,
    )
    dc = torch.trapezoid(f.loss_density / (f.energies / 600000) ** 2, f.energies)
    assert float(k[0, 1]) == pytest.approx(float(torch.sqrt(dcs / dc)), rel=1e-5)
    assert k[0, 0] == 1
    with pytest.raises(ValueError, match="positive area"):
        PlasmonFilter(
            2.0, 300.0, torch.tensor([7.5, 100.0]), torch.zeros(2), provenance="bad"
        )


def test_guard_double_absorption():
    with pytest.raises(ValueError, match="alpha=0"):
        FrozenPlasmonForward(propagator(alpha=0.1), Detector(2.0), None)
    with pytest.raises(ValueError, match="single-readout"):
        FrozenPlasmonForward(propagator(), Detector(2.0, n_frames=4), None)


def test_source_halo_includes_neighbours_outside_roi():
    # The propagation ROI is only 12 px, inside a 20 px specimen. Absorbers
    # just outside the ROI still affect it through plasmon delocalization.
    v = torch.zeros(1, 1, 20, 20)
    a = torch.zeros_like(v)
    a[..., 3, 10] = 50.0
    filt = PlasmonFilter.approximate_drude(2.0, 300.0, padding=4)
    s = propagator()
    actual = s(v, 0.0, absorption_source=a, absorption_filter=filt)
    assert float(actual.abs().square().mean()) < 0.99999
    # A manually sampled 20px halo and crop is an independent reference.
    vi = filt.filter_sampled(a)
    expected = Scattering(12, 2.0, 300.0, progressbars=False)(
        torch.complex(torch.zeros_like(vi), vi)
    )
    torch.testing.assert_close(actual, expected)


def test_damage_preserves_dc_and_attenuates_structure():
    from specter.inelastic import PotentialDoseDamage

    damage = PotentialDoseDamage(2.0)
    torch.manual_seed(12)
    v = torch.rand(8, 12, 12)
    a = damage(v, ExposureStep(0, 0.0, 1.0))
    b = damage(v, ExposureStep(0, 30.0, 1.0))
    torch.testing.assert_close(a.mean(), v.mean())
    torch.testing.assert_close(b.mean(), v.mean())
    assert b.std() < a.std() < v.std()


def test_moving_water_rebuilds_both_fields_and_replays():
    torch.manual_seed(7)
    waters = (torch.rand(100, 3) - 0.5) * 12
    specimen = AtomicPlasmonSpecimen(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([6]),
        (8, 12, 12),
        2.0,
        {6: 3.0},
        provenance="test coefficients",
        water_coordinates=waters,
        water_strength=2.0,
        water_msd_per_dose=3.0,
        seed=31,
    )
    first = specimen(ExposureStep(0, 0.0, 1.0))
    second = specimen(ExposureStep(1, 1.0, 1.0))
    assert not torch.equal(first.elastic, second.elastic)
    assert not torch.equal(first.absorption_source, second.absorption_source)
    specimen.reset()
    replay = specimen(ExposureStep(0, 0.0, 1.0))
    torch.testing.assert_close(first.elastic, replay.elastic)
    torch.testing.assert_close(first.absorption_source, replay.absorption_source)


def test_existing_imager_frozen_entrypoint():
    from specter.imagegenerator import MicrographGenerator
    from specter.settings import Camera, Ice, Propagation

    v = torch.zeros(2, 12, 12)
    imager = MicrographGenerator(
        v[None],
        12,
        2.0,
        None,
        300.0,
        4.0,
        optics=None,
        ice=Ice(model=None),
        camera=Camera(noise_model=None, detector_model=None),
        propagation=Propagation(alpha=0),
        verbose=False,
        progressbars=False,
    )
    state = PotentialState(v[None], torch.zeros_like(v[None]))
    result = imager.simulate_frozen(lambda step: state, None, [1.0, 3.0], substeps=2)
    assert result.images.shape == (2, 1, 12, 12)
    torch.testing.assert_close(result.summed_image, imager.forward(torch.tensor([0])))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_paired_forward():
    v = torch.rand(1, 3, 12, 12)
    a = torch.rand_like(v)
    cpu = propagator()(v, 20.0, absorption_source=a)
    gpu = propagator().cuda()(v.cuda(), 20.0, absorption_source=a.cuda())
    torch.testing.assert_close(cpu, gpu.cpu(), atol=2e-5, rtol=2e-5)
