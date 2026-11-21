"""
Single-scatter slice sums sampled from the unposed volume's transform.

The first-Born slice sum ``sum_z fft2(V_z) * F_z`` that `Scattering` forms for
the Rytov and first-Born models is, exactly, the volume's discrete-space
Fourier transform sampled on the Ewald paraboloid ``k_z = -/+ lambda |k|^2 / 2``
times a phase; projection and the CTF model sample the flat plane ``k_z = 0``
instead. A pose rotates that surface rather than the volume, and a lateral
shift multiplies it by a phase ramp, so a posed exit wave needs one non-uniform
FFT of the UNROTATED volume (`specter.nufft.sample_transform`) and no
interpolation anywhere. Multislice and the kinematic model are not single
samples of the transform and have no such form.

Only frequencies inside the Nyquist sphere, ``|k| dx < 1/2``, are sampled; the
rest of the image's spectrum (its corners, between Nyquist and sqrt(2) times
it) is zero. That is the band `specter.rotations.rotate_volume_exact` keeps,
for the same reason: beyond the sphere a rotated frequency leaves the cube the
volume's transform is known on, so what the transform returns there depends
on the orientation, and is an alias of the periodic spectrum rather than the
specimen. With the same band, a pose applied by sampling the surface and one
applied by rotating exactly give the same image to rounding.
"""

from __future__ import annotations

import math

import roma
import torch

from ..nufft import sample_transform
from ..options import EwaldSphereSign

__all__ = ["SURFACE_MODELS", "surface_slice_sum"]

#: The scattering models whose exit wave is a function of one slice sum.
SURFACE_MODELS = ("projection", "firstborn", "rytov", "ctf")


def surface_slice_sum(
    volume: torch.Tensor,
    quaternions: torch.Tensor,
    shifts: torch.Tensor,
    *,
    pad_nxy: int,
    pixel_size: float,
    wavelength: float,
    ews_curvature_sign: EwaldSphereSign,
    flat: bool,
) -> torch.Tensor:
    """
    The posed volume's slice sum, sampled from its transform.

    Equal to what `Scattering` forms on the volume rotated by
    `specter.rotations.VolumeRotator` (``origin="relion"``) to each pose and
    zero-padded to ``pad_nxy``: ``sum_z fft2(V_z) * F_z`` with the first-Born
    propagators (``flat=False``), or ``fft2(sum_z V_z)`` (``flat=True``). The
    volume is rotated about voxel ``shape // 2`` so that output voxel ``r``
    holds the template at ``R r``, then displaced by ``-shift`` in the image
    plane, as ``build_affine_matrix`` composes the two.

    Parameters
    ----------
    volume : torch.Tensor
        Real potential of shape (Z, Y, X), unposed, Y == X.
    quaternions : torch.Tensor
        Poses of shape (B, 4), roma (x, y, z, w) order.
    shifts : torch.Tensor
        In-plane shifts of shape (B, 2), (x, y) in Angstrom.
    pad_nxy : int
        Lateral size of the propagation grid, ``>= X`` (with a symmetric
        ``(pad_nxy - X) // 2`` margin, as `Specimen.pad_nxy` gives).
    pixel_size : float
        Voxel size in Angstrom; also the slice thickness.
    wavelength : float
        Electron wavelength in Angstrom.
    ews_curvature_sign : {"negative", "positive"}
        Which way the paraboloid curves, as in `Scattering`.
    flat : bool
        Sample the plane ``k_z = 0`` (projection, CTF) instead of the
        paraboloid.

    Returns
    -------
    torch.Tensor
        Complex spectra of shape (B, pad_nxy, pad_nxy) in native FFT order,
        zero outside the Nyquist sphere ``|k| dx < 1/2``. Differentiable in
        ``volume``, ``shifts`` and ``quaternions``.

    Notes
    -----
    The gradient to the pose follows from the chain rule through the sample
    points: the spectrum at frequency ``q`` is the transform ``V^(R q)``, so
    its change with the rotation is ``grad V^(R q) . (dR q)``. Each component
    of ``grad V^`` is itself a transform, of the volume weighted by one of its
    centred coordinates (``-2 pi i x V``), evaluated at the same points; the
    three are computed, in one batched NUFFT, only when ``quaternions``
    requires a gradient.
    """
    nz, ny, nx = volume.shape
    if ny != nx:
        raise ValueError(f"surface sampling needs a square volume, got {ny} x {nx}")
    if pad_nxy < nx or (pad_nxy - nx) % 2:
        raise ValueError(f"pad_nxy={pad_nxy} must be X={nx} plus an even margin")
    device, dx, lam = volume.device, pixel_size, wavelength
    f = torch.fft.fftfreq(pad_nxy, d=dx, device=device, dtype=torch.float64)
    ky, kx = torch.meshgrid(f, f, indexing="ij")
    k2 = kx**2 + ky**2
    sign = -1.0 if ews_curvature_sign == "negative" else 1.0
    kz = torch.zeros_like(k2) if flat else sign * lam * k2 / 2
    q = torch.stack([kx, ky, kz], -1).reshape(-1, 3)  # 1/A, (x, y, z)

    # The band `rotate_volume_exact` keeps; a rotation preserves |q|, so the
    # points inside are the same for every pose and only they are evaluated.
    inside = torch.nonzero((q * dx).norm(dim=-1) < 0.5).flatten()
    rot = roma.unitquat_to_rotmat(quaternions.to(device, torch.float64))
    rotated = torch.einsum("bij,mj->bmi", rot, q[inside]) * dx  # R q, cycles/voxel
    points = rotated.detach().reshape(-1, 3)
    batch = len(rot)
    sampled = sample_transform(volume, points)
    if rotated.requires_grad:
        # Zero in value, so the spectrum is unchanged; its derivative is the
        # first-order change of V^(R q) with the points, and so with the pose.
        moved = (rotated.reshape(-1, 3) - points).to(sampled.real.dtype)
        sampled = sampled + (moved * _transform_gradient(volume, points)).sum(-1)
    values = sampled.new_zeros(batch, pad_nxy * pad_nxy)
    values = values.index_copy(1, inside, sampled.reshape(batch, -1))
    values = values.reshape(batch, pad_nxy, pad_nxy)

    # sample_transform is centred on the template's voxel shape // 2; the
    # slice sum's phase origin is voxel 0 of the padded box, whose centre the
    # template's lands on.
    centre = pad_nxy // 2
    phase = -2 * math.pi * dx * (kx * centre + ky * centre + kz * (nz // 2))
    if not flat:
        # The propagator from slice z to the exit plane is
        # exp(-i pi lambda dz d k^2) over the remaining distance d: dz*(nz - z)
        # for the negative sign, whose beam enters at slice 0, and dz*(z + 1)
        # for the positive sign, whose beam enters at the last slice. Sampling
        # at k_z absorbs the z-dependent part.
        depth = nz if ews_curvature_sign == "negative" else 1
        phase = phase - math.pi * lam * dx * depth * k2
    factor = torch.polar(torch.ones_like(phase), phase)

    t = shifts.to(device=device, dtype=torch.float64)
    ramp = (
        2 * math.pi * (kx[None] * t[:, 0, None, None] + ky[None] * t[:, 1, None, None])
    )
    factor = factor[None] * torch.polar(torch.ones_like(ramp), ramp)
    return values * factor.to(values.dtype)


def _transform_gradient(volume: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    The gradient of `volume`'s transform at `points`, in their (x, y, z) order.

    ``V^(p) = sum_n V[n] exp(-2 pi i p . (n - c))`` with ``c = shape // 2``, as
    `sample_transform` defines it, so ``dV^/dp_j`` is the transform of
    ``-2 pi i (n_j - c_j) V``. Held constant by autograd: the pose gradient
    needs it only to first order.

    Parameters
    ----------
    volume : torch.Tensor
        Real grid of shape (Z, Y, X).
    points : torch.Tensor
        Frequencies of shape (M, 3) in cycles per voxel, (x, y, z).

    Returns
    -------
    torch.Tensor
        Complex gradients of shape (M, 3).
    """
    with torch.no_grad():
        v = volume.detach()
        axes = [
            torch.arange(size, device=v.device, dtype=v.dtype) - size // 2
            for size in v.shape
        ]
        z, y, x = torch.meshgrid(*axes, indexing="ij")
        weighted = torch.stack([x * v, y * v, z * v])  # (x, y, z) order
        samples = sample_transform(weighted, points)  # (3, M)
        return (-2j * math.pi) * samples.T
