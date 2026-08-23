import numpy as np
import pytest
import roma
import torch

from specter.rotations import build_affine_matrix, rotate_volume
from specter.scattering import (
    IterativeScattering,
    Scattering,
    complex_potential,
)


def test_complex_potential():
    v = torch.ones(4, 4)
    alpha = 0.1
    cv = complex_potential(v, alpha=alpha)

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
    "angle_deg, atol",
    [
        (0.0, 1e-4),
        # 45° tolerance reflects interpolation noise between 3-D volume resample
        # (rotate_volume) and 2-D slice-by-slice sampling (IterativeScattering).
        (45.0, 5e-3),
    ],
)
def test_scattering_vs_iterative_consistency(padded_volume, angle_deg, atol):
    """Scattering with a pre-rotated volume matches IterativeScattering for the same pose."""
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
        scattering_model="projection",
        alpha=0.1,
        progressbars=False,
    )

    scat = Scattering(**kwargs)
    scat_iter = IterativeScattering(**kwargs)

    volume_rotated = rotate_volume(padded_volume, theta_matrix, padding_mode="zeros")
    psi_scat = scat.forward(volume_rotated)

    psi_iter = scat_iter.forward(padded_volume.unsqueeze(0), theta_matrix)

    assert psi_scat.shape == psi_iter.shape
    assert torch.allclose(psi_scat, psi_iter, atol=atol), (
        f"Max diff at {angle_deg}°: {(psi_scat - psi_iter).abs().max().item():.2e}"
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

    got = scat.multislice(complex_potential(V, alpha=alpha))
    want = _multislice_per_slice_reference(scat, complex_potential(V, alpha=alpha))
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
        fn(scat, complex_potential(v, alpha=0.07)).real.sum().backward()
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
    V_cpu = complex_potential(torch.rand(1, nz, n, n) * 4.0, alpha=0.07)

    out = scat.multislice(V_cpu)
    assert out.device.type == "cuda"
    assert torch.equal(out, scat.multislice(V_cpu.cuda()))
