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
        Complex spectra of shape (B, pad_nxy, pad_nxy) in native FFT order.
        Differentiable in ``volume`` and ``shifts``, not in ``quaternions``.
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

    rot = roma.unitquat_to_rotmat(quaternions.detach().to(device, torch.float64))
    rotated = torch.einsum("bij,mj->bmi", rot, q)  # R q: the template's frequency
    batch = len(rot)
    values = sample_transform(volume, (rotated * dx).reshape(-1, 3))
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
