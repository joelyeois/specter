import numpy as np
import pytest
import roma
import torch

from specter.rotations import build_affine_matrix, rotate_volume
from specter.potential import apply_amplitude_contrast
from specter.scattering import (
    IterativeScattering,
    Scattering,
)


def test_apply_amplitude_contrast():
    v = torch.ones(4, 4)
    alpha = 0.1
    cv = apply_amplitude_contrast(v, alpha=alpha)

    expected_real = np.sqrt(0.99)
    assert torch.allclose(
        cv.real, torch.tensor(expected_real, dtype=torch.float32), atol=1e-6
    )
    assert torch.allclose(cv.imag, torch.tensor(0.1, dtype=torch.float32), atol=1e-6)


def test_iterative_scattering_batch_size(dummy_volume):
    scat_iter = IterativeScattering(
        nxy=64,
        pixel_size=1.0,
        voltage=300.0,
        scattering_model="multislice",
        progressbars=False,
    )

    R = roma.rotvec_to_rotmat(torch.tensor([[0.0, 0.1, 0.0]])).to(dummy_volume.device)
    theta_matrix = build_affine_matrix(R)

    psi1 = scat_iter.forward(dummy_volume, theta_matrix, slice_batchsize=1)
    psi4 = scat_iter.forward(dummy_volume, theta_matrix, slice_batchsize=4)

    assert torch.allclose(psi1, psi4, atol=1e-5)

    scat_iter_rytov = IterativeScattering(
        nxy=64,
        pixel_size=1.0,
        voltage=300.0,
        scattering_model="rytov",
        progressbars=False,
    )

    psi1_rytov = scat_iter_rytov.rytov(dummy_volume, theta_matrix, slice_batchsize=1)
    psi4_rytov = scat_iter_rytov.rytov(dummy_volume, theta_matrix, slice_batchsize=4)

    assert torch.allclose(psi1_rytov, psi4_rytov, atol=1e-5)


def test_multislice_checkpointing_matches_uncheckpointed(dummy_volume):
    """Gradient checkpointing must not change the multislice exit wave."""
    scat_iter = IterativeScattering(
        nxy=64, pixel_size=1.0, voltage=300.0, progressbars=False
    )
    R = roma.rotvec_to_rotmat(torch.tensor([[0.0, 0.1, 0.0]])).to(dummy_volume.device)
    theta_matrix = build_affine_matrix(R)

    psi = scat_iter.multislice(dummy_volume, theta_matrix)
    psi_ckpt3 = scat_iter.multislice(dummy_volume, theta_matrix, checkpoint_chunks=3)
    psi_ckpt8 = scat_iter.multislice(dummy_volume, theta_matrix, checkpoint_chunks=8)

    assert torch.allclose(psi, psi_ckpt3, atol=1e-5)
    assert torch.allclose(psi, psi_ckpt8, atol=1e-5)


def test_multislice_checkpointing_backprops(dummy_volume):
    """Gradients must flow through the checkpointed multislice path."""
    scat_iter = IterativeScattering(
        nxy=64, pixel_size=1.0, voltage=300.0, progressbars=False
    )
    R = roma.rotvec_to_rotmat(torch.tensor([[0.0, 0.1, 0.0]])).to(dummy_volume.device)
    theta_matrix = build_affine_matrix(R)

    V = dummy_volume.clone().requires_grad_(True)
    psi = scat_iter.multislice(V, theta_matrix, checkpoint_chunks=3)
    psi.abs().sum().backward()

    assert V.grad is not None
    assert torch.isfinite(V.grad).all()
    assert V.grad.abs().sum() > 0


def test_parallel_rytov_matches_iterative_rytov(dummy_volume):
    """The fully-parallel Rytov path should agree with the slice-by-slice one."""
    scat_iter = IterativeScattering(
        nxy=64, pixel_size=1.0, voltage=300.0, progressbars=False
    )
    R = roma.rotvec_to_rotmat(torch.tensor([[0.0, 0.1, 0.0]])).to(dummy_volume.device)
    theta_matrix = build_affine_matrix(R)

    psi_iterative = scat_iter.rytov(dummy_volume, theta_matrix)
    psi_parallel = scat_iter.parallel_rytov(dummy_volume, theta_matrix)
    psi_parallel_ckpt = scat_iter.parallel_rytov(
        dummy_volume, theta_matrix, checkpoint_chunks=5
    )

    assert torch.allclose(psi_iterative, psi_parallel, atol=1e-4)
    assert torch.allclose(psi_parallel, psi_parallel_ckpt, atol=1e-5)


def test_parallel_rytov_checkpointing_backprops(dummy_volume):
    """Gradients must flow through the checkpointed parallel_rytov path."""
    scat_iter = IterativeScattering(
        nxy=64, pixel_size=1.0, voltage=300.0, progressbars=False
    )
    R = roma.rotvec_to_rotmat(torch.tensor([[0.0, 0.1, 0.0]])).to(dummy_volume.device)
    theta_matrix = build_affine_matrix(R)

    V = dummy_volume.clone().requires_grad_(True)
    psi = scat_iter.parallel_rytov(V, theta_matrix, checkpoint_chunks=5)
    psi.abs().sum().backward()

    assert V.grad is not None
    assert torch.isfinite(V.grad).all()
    assert V.grad.abs().sum() > 0


@pytest.mark.parametrize("scattering_model", ["firstborn", "kinematic", "ctf"])
def test_iterative_models_consistent_across_batch_size(dummy_volume, scattering_model):
    """firstborn/kinematic/ctf must be invariant to the slice_batchsize chunking."""
    scat_iter = IterativeScattering(
        nxy=64,
        pixel_size=1.0,
        voltage=300.0,
        scattering_model=scattering_model,
        progressbars=False,
    )
    R = roma.rotvec_to_rotmat(torch.tensor([[0.0, 0.1, 0.0]])).to(dummy_volume.device)
    theta_matrix = build_affine_matrix(R)

    method = getattr(scat_iter, scattering_model)
    psi1 = method(dummy_volume, theta_matrix, slice_batchsize=1)
    psi4 = method(dummy_volume, theta_matrix, slice_batchsize=4)

    assert torch.allclose(psi1, psi4, atol=1e-5)


@pytest.fixture
def dummy_volume() -> torch.Tensor:
    volume = torch.zeros(1, 32, 64, 64)
    volume[0, 12:20, 24:40, 24:40] = 1.0
    return volume


@pytest.fixture
def padded_volume():
    """Single 3D volume (Z, Y, X) with non-zero content only in the central 16^3 block.

    The generous zero padding ensures rotated coordinates outside the volume
    boundary contribute nothing, making Scattering and IterativeScattering
    numerically comparable at arbitrary tilt angles.
    """
    volume = torch.zeros(64, 64, 64)
    volume[24:40, 24:40, 24:40] = 1.0
    return volume


@pytest.mark.parametrize(
    "scattering_model",
    ["projection", "multislice", "rytov", "firstborn", "kinematic", "ctf"],
)
@pytest.mark.parametrize(
    "angle_deg, atol",
    [
        (0.0, 1e-4),
        # 45° tolerance reflects interpolation noise between 3-D volume resample
        # (rotate_volume) and 2-D slice-by-slice sampling (IterativeScattering).
        (45.0, 5e-3),
    ],
)
def test_scattering_vs_iterative_consistency(
    padded_volume, angle_deg, atol, scattering_model
):
    """
    Scattering with a pre-rotated volume matches IterativeScattering for the
    same pose, for every mode both implement.

    The two are independent implementations of the same six models -- roughly
    1,360 lines with no inheritance between them -- split by caller rather than
    by physics: `Scattering` takes a whole volume, `IterativeScattering` streams
    Z-slices for volumes too large to rotate at once, which is what every tilt
    series and micrograph runs. Nothing but this test stops them drifting apart.

    It used to cover ``"projection"`` alone, which is the case least able to
    drift: a plain sum along Z with no propagation between slices, so streaming
    and whole-volume paths coincide almost trivially. The modes that can drift
    are the ones where the streamed version has to get slice ordering,
    propagator distances and tilt geometry right, and ``"multislice"`` -- the
    default, and what a tilt series actually uses -- was among the untested
    ones. Measured agreement at 0°: projection 2.0e-08, ctf 4.0e-08, firstborn
    and kinematic 6.0e-08, rytov 7.2e-07, multislice 3.1e-06.

    ``nz`` goes to `Scattering` only: `IterativeScattering` derives it from the
    volume it is handed, and rejects the argument.
    """
    nxy = 64

    theta_rad = torch.deg2rad(torch.tensor(angle_deg))
    c, s = torch.cos(theta_rad), torch.sin(theta_rad)
    R = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float32
    ).unsqueeze(0)
    theta_matrix = build_affine_matrix(R)

    kwargs = dict(
        nxy=nxy,
        pixel_size=1.0,
        voltage=300.0,
        scattering_model=scattering_model,
        alpha=0.1,
        progressbars=False,
    )

    scat = Scattering(**kwargs, nz=padded_volume.shape[0])
    scat_iter = IterativeScattering(**kwargs)

    volume_rotated = rotate_volume(padded_volume, theta_matrix, padding_mode="zeros")
    psi_scat = scat.forward(volume_rotated)

    psi_iter = scat_iter.forward(padded_volume.unsqueeze(0), theta_matrix)

    assert psi_scat.shape == psi_iter.shape
    assert torch.allclose(psi_scat, psi_iter, atol=atol), (
        f"{scattering_model} disagrees between implementations at "
        f"{angle_deg}°: max diff {(psi_scat - psi_iter).abs().max().item():.2e}"
    )


def test_klim_bandlimit_keeps_low_frequencies(dummy_volume):
    """``klim`` must zero content *above* the cutoff and leave content
    *below* it untouched -- Kirkland's antialiasing bandlimit, not its
    inverse. Regression test for a fftshift/native-FFT convention mismatch
    between ``disk2d`` (centered at N//2) and the unshifted spectrum
    ``Scattering.multislice`` actually multiplies it into."""
    nxy = 64
    pixel_size = 1.0
    klim = 0.5
    scat = Scattering(
        nxy=nxy,
        pixel_size=pixel_size,
        voltage=300.0,
        scattering_model="multislice",
        klim=klim,
        progressbars=False,
    )
    psi = scat(dummy_volume)

    k = torch.fft.fftfreq(nxy, pixel_size)
    kxx, kyy = torch.meshgrid(k, k, indexing="ij")
    k_mag = torch.sqrt(kxx**2 + kyy**2)
    k_nyquist = 1.0 / (2.0 * pixel_size)

    spectrum = torch.fft.fft2(psi[0])
    below_cutoff = spectrum[k_mag <= klim * k_nyquist]
    above_cutoff = spectrum[k_mag > klim * k_nyquist]

    # Masked every slice through 32 FFT/IFFT round trips, so the "zeroed"
    # region carries float32 round-off rather than being exactly 0 -- still
    # several orders of magnitude below the surviving low-frequency content.
    assert torch.abs(above_cutoff).max() < 1e-3 * torch.abs(below_cutoff).max()
    assert torch.abs(below_cutoff).max() > 1e-6


# ---------------------------------------------------------------------------
# Multislice chunking: the chunked/unbind form of the slice loop must stay
# bitwise identical to the plain per-slice form it replaced. Asserted with
# torch.equal rather than allclose -- the restructuring only changes kernel
# launch granularity, never per-element arithmetic, so any drift at all is a
# regression rather than tolerable round-off.
# ---------------------------------------------------------------------------


def _multislice_per_slice_reference(scat, V):
    """The pre-chunking implementation of Scattering.multislice, verbatim."""
    F = scat.F_real + 1j * scat.F_imag
    if scat.ews_curvature_sign == "negative":
        V = torch.flip(V, dims=(1,))
    exitwave = None
    for i in range(V.size(1)):
        t = torch.exp(1j * scat.sigma * scat.pixel_size * V[:, i].to(scat.device))
        wv = t if exitwave is None else t * exitwave
        exitwave = torch.fft.ifft2(
            torch.fft.fft2(wv, dim=(-1, -2)) * F * scat.kmask, dim=(-1, -2)
        )
    return exitwave


@pytest.mark.parametrize(
    "alpha, klim, sign, nz",
    [
        (0.07, None, "negative", 16),
        (0.0, None, "negative", 16),
        (0.07, 0.66, "negative", 16),
        (0.07, None, "positive", 16),
        (0.07, None, "negative", 13),  # nz not divisible by the chunk size
    ],
)
def test_multislice_matches_per_slice_reference(alpha, klim, sign, nz):
    """Chunked multislice is bitwise identical to the per-slice form."""
    n = 32
    scat = Scattering(
        nxy=n,
        pixel_size=1.0,
        voltage=300.0,
        scattering_model="multislice",
        alpha=alpha,
        klim=klim,
        ews_curvature_sign=sign,
        nz=nz,
        progressbars=False,
    )
    torch.manual_seed(0)
    V = torch.rand(2, nz, n, n) * 4.0

    got = scat.multislice(apply_amplitude_contrast(V, alpha=alpha))
    want = _multislice_per_slice_reference(
        scat, apply_amplitude_contrast(V, alpha=alpha)
    )
    assert torch.equal(got, want)


def test_multislice_gradients_match_per_slice_reference():
    """Gradients w.r.t. the potential are bitwise identical too."""
    n, nz = 32, 16
    scat = Scattering(
        nxy=n,
        pixel_size=1.0,
        voltage=300.0,
        scattering_model="multislice",
        alpha=0.07,
        nz=nz,
        progressbars=False,
    )
    torch.manual_seed(0)
    V = torch.rand(2, nz, n, n) * 4.0

    def grad_of(fn):
        v = V.clone().requires_grad_(True)
        fn(scat, apply_amplitude_contrast(v, alpha=0.07)).real.sum().backward()
        return v.grad

    assert torch.equal(
        grad_of(lambda s, x: s.multislice(x)),
        grad_of(_multislice_per_slice_reference),
    )


def test_multislice_accepts_volume_off_compute_device():
    """A volume on a different device than the module still streams through."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA to have two distinct devices")
    n, nz = 32, 16
    scat = Scattering(
        nxy=n,
        pixel_size=1.0,
        voltage=300.0,
        scattering_model="multislice",
        alpha=0.07,
        nz=nz,
        progressbars=False,
    ).to("cuda")
    torch.manual_seed(0)
    V_cpu = apply_amplitude_contrast(torch.rand(1, nz, n, n) * 4.0, alpha=0.07)

    out = scat.multislice(V_cpu)
    assert out.device.type == "cuda"
    assert torch.equal(out, scat.multislice(V_cpu.cuda()))


# ---------------------------------------------------------------------------
# Fourier-space slice summation: rytov/firstborn/kinematic sum over z BEFORE
# the inverse FFT (one inverse FFT instead of nz). The inverse FFT is linear so
# this is the same quantity, but reassociating a float sum changes rounding --
# unlike the multislice chunking above, these are allclose, not torch.equal.
# ---------------------------------------------------------------------------


def _rytov_sum_after_reference(scat, V):
    """Pre-change rytov: nz inverse FFTs, then sum."""
    F = scat.F_real + 1j * scat.F_imag
    if scat.ews_curvature_sign == "negative":
        V = torch.flip(V, dims=(1,))
    scattered = torch.fft.ifft2(
        torch.fft.fft2(1j * scat.sigma * scat.pixel_size * V, dim=(-1, -2))
        * F[None, ...],
        dim=(-1, -2),
    )
    return torch.exp(torch.sum(scattered, dim=1))


def _firstborn_sum_after_reference(scat, V):
    """Pre-change firstborn: nz inverse FFTs, then sum."""
    F = scat.F_real + 1j * scat.F_imag
    if scat.ews_curvature_sign == "negative":
        V = torch.flip(V, dims=(1,))
    ew = torch.fft.ifft2(
        scat.sigma * scat.pixel_size * torch.fft.fft2(V, dim=(-1, -2)) * F[None, ...],
        dim=(-1, -2),
    )
    return 1 + 1j * torch.sum(ew, 1)


def _kinematic_sum_after_reference(scat, V):
    """Pre-change kinematic: nz inverse FFTs, then sum."""
    F = scat.F_real + 1j * scat.F_imag
    if scat.ews_curvature_sign == "negative":
        V = torch.flip(V, dims=(1,))
    t = torch.exp(1j * scat.sigma * scat.pixel_size * V) - 1
    return 1 + torch.sum(
        torch.fft.ifft2(torch.fft.fft2(t, dim=(-1, -2)) * F[None, ...], dim=(-1, -2)),
        1,
    )


_SUM_AFTER_REFERENCES = {
    "rytov": _rytov_sum_after_reference,
    "firstborn": _firstborn_sum_after_reference,
    "kinematic": _kinematic_sum_after_reference,
}


@pytest.mark.parametrize("model", ["rytov", "firstborn", "kinematic"])
@pytest.mark.parametrize("sign", ["negative", "positive"])
def test_fourier_space_summation_matches_sum_after_ifft(model, sign):
    """Summing before the inverse FFT gives the same exit wave as summing after."""
    n, nz = 32, 16
    scat = Scattering(
        nxy=n,
        pixel_size=1.0,
        voltage=300.0,
        scattering_model=model,
        alpha=0.07,
        ews_curvature_sign=sign,
        nz=nz,
        progressbars=False,
    )
    torch.manual_seed(0)
    V = apply_amplitude_contrast(torch.rand(2, nz, n, n) * 4.0, alpha=0.07)

    got = getattr(scat, model)(V)
    want = _SUM_AFTER_REFERENCES[model](scat, V)
    # float32 reassociation only -- eps is ~1.2e-7, exp() in rytov amplifies it
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("model", ["rytov", "firstborn", "kinematic"])
def test_fourier_space_summation_gradients_match(model):
    """Gradients w.r.t. the potential are unchanged by the reassociation."""
    n, nz = 32, 16
    scat = Scattering(
        nxy=n,
        pixel_size=1.0,
        voltage=300.0,
        scattering_model=model,
        alpha=0.07,
        nz=nz,
        progressbars=False,
    )
    torch.manual_seed(0)
    V = torch.rand(2, nz, n, n) * 4.0

    def grad_of(fn):
        v = V.clone().requires_grad_(True)
        fn(apply_amplitude_contrast(v, alpha=0.07)).real.sum().backward()
        return v.grad

    got = grad_of(getattr(scat, model))
    want = grad_of(lambda x: _SUM_AFTER_REFERENCES[model](scat, x))
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-7)


def _iterative_multislice_unhoisted_reference(scat, V, theta_matrix):
    """
    `IterativeScattering.multislice` with the bandlimit applied per slice.

    The shipped loop folds the bandlimit into the propagator once
    (``Fk = F * kmask``) instead of evaluating ``psi_k * F * kmask`` on every
    slice. This is the pre-hoist arithmetic, kept verbatim as the reference.
    """
    from specter.fft import fft2, ifft2

    if scat.pad_fft:
        F = scat.F_step_padded_real + 1j * scat.F_step_padded_imag
        kmask = scat.kmask_padded
        canvas = scat.padded_nxy
        roi_size = canvas
    else:
        F = scat.F_step_real + 1j * scat.F_step_imag
        kmask = scat.kmask
        canvas = scat.nxy
        roi_size = None

    exitwave = torch.ones(
        V.shape[0], canvas, canvas, device=scat.device, dtype=torch.complex64
    )
    for _, _, slice_sample in scat._iter_slices(
        V, theta_matrix, 1, "reference", roi_size=roi_size
    ):
        slice_complex = apply_amplitude_contrast(slice_sample, alpha=scat.alpha)
        t = torch.exp(1j * scat.sigma * scat.pixel_size * slice_complex)
        exitwave = ifft2(fft2(t * exitwave) * F * kmask)
    return exitwave


@pytest.mark.parametrize("klim", [None, 0.66])
@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_iterative_multislice_bandlimit_hoist_is_exact(alpha, klim, pad_fft=False):
    """
    Folding the bandlimit into the propagator changes nothing, bit for bit.

    `psi_k * F * kmask` and `psi_k * (F * kmask)` are only interchangeable
    because `kmask` is binary (0.0/1.0), or the Python int 1 when `klim` is
    None -- in which case the shipped form also skips a full-size complex
    multiply by one, per slice. A soft or apodised mask would make the
    reassociation inexact, so this asserts equality rather than closeness.

    `klim` is parametrized over both spellings for that reason. The padded
    propagator takes the same code path with a different buffer; what the
    reassociation needs of it is covered by
    `test_iterative_scattering_bandlimit_masks_are_binary`.
    """
    torch.manual_seed(0)
    n, nz = 32, 12
    scat = IterativeScattering(
        nxy=n,
        pixel_size=2.0,
        voltage=300.0,
        scattering_model="multislice",
        klim=klim,
        alpha=alpha,
        pad_fft=pad_fft,
        progressbars=False,
    )
    V = torch.rand(2, nz, n, n) * 0.5

    angle = torch.tensor(25.0 * np.pi / 180.0)
    c, s = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(2, 3, 4)
    theta[:, :, :3] = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    got = scat.multislice(V, theta, slice_batchsize=1)
    want = _iterative_multislice_unhoisted_reference(scat, V, theta)

    assert torch.equal(got, want)


@pytest.mark.parametrize("pad_fft", [False, True])
@pytest.mark.parametrize("klim", [None, 0.66])
def test_iterative_scattering_bandlimit_masks_are_binary(klim, pad_fft):
    """
    Every bandlimit mask is binary (or the int 1), which is what makes the hoist exact.

    `multislice` folds the mask into the propagator once rather than applying it
    per slice. Reassociating `(psi_k * F) * kmask` to `psi_k * (F * kmask)` is
    bitwise-exact for a 0.0/1.0 mask and NOT exact for a soft or apodised one,
    so this pins the property the optimisation rests on -- including for the
    padded propagator, whose own buffer is a separate construction.
    """
    scat = IterativeScattering(
        nxy=32,
        pixel_size=2.0,
        voltage=300.0,
        scattering_model="multislice",
        klim=klim,
        pad_fft=pad_fft,
        progressbars=False,
    )
    mask = scat.kmask_padded if pad_fft else scat.kmask
    if not torch.is_tensor(mask):
        assert mask == 1
        return
    assert torch.equal(mask, (mask > 0).to(mask.dtype)), (
        "bandlimit mask must be binary for the propagator hoist to stay exact"
    )
