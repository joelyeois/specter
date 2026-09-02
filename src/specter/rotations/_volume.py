from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from ..fft import fft3, ifft3


def affine_sampling_grid(
    theta: torch.Tensor,
    nz: int,
    ny: int,
    nx: int,
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Sampling grid for ``grid_sample``, equal to
    ``F.affine_grid(theta, [B, 1, nz, ny, nx], align_corners)``.

    Parameters
    ----------
    theta : torch.Tensor
        Affine matrices, shape (B, 3, 4), mapping output (x, y, z) in
        normalized coordinates to input coordinates.
    nz, ny, nx : int
        Output volume dimensions.
    align_corners : bool, optional
        As for ``affine_grid``. Default False.

    Returns
    -------
    torch.Tensor
        Shape (B, nz, ny, nx, 3), last axis (x, y, z).

    Notes
    -----
    ``affine_grid`` materialises the identity grid, stacks a ones column
    onto it and runs a ``bmm`` over every voxel: at 512^3 that is 43 ms and
    ~1.5 GB of temporaries for a grid whose ``grid_sample`` then costs 7 ms.
    The grid is affine in three 1-D coordinate vectors, so it is built here
    as a sum of three broadcast outer products and written exactly once
    (2.9 ms at 512^3, no temporaries). Values agree with ``affine_grid`` to
    float rounding (4e-7 in normalized units at 512^3, and 7e-16 against a
    float64 reference; ``affine_grid`` itself sits at 9e-8 from that
    reference).
    """
    B = theta.shape[0]
    device, dtype = theta.device, theta.dtype

    def axis(n: int) -> torch.Tensor:
        if align_corners:
            return torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
        return (torch.arange(n, device=device, dtype=dtype) * 2.0 + 1.0) / n - 1.0

    A = theta[:, :, :3]  # out = A @ (x, y, z) + b
    b = theta[:, :, 3]
    x, y, z = axis(nx), axis(ny), axis(nz)
    gx = x.view(1, 1, 1, nx, 1) * A[:, :, 0].view(B, 1, 1, 1, 3)
    gy = y.view(1, 1, ny, 1, 1) * A[:, :, 1].view(B, 1, 1, 1, 3)
    gz = z.view(1, nz, 1, 1, 1) * A[:, :, 2].view(B, 1, 1, 1, 3) + b.view(B, 1, 1, 1, 3)
    return (gx + gy) + gz


def _relion_rotation_grid(
    theta: torch.Tensor,
    nz: int,
    ny: int,
    nx: int,
    align_corners: bool,
) -> torch.Tensor:
    """
    Build a sampling grid for RELION-convention rotation about [nz//2, ny//2, nx//2].

    Parameters
    ----------
    theta : torch.Tensor
        Batch of affine matrices, Bx3x4.
    nz, ny, nx : int
        Volume dimensions.
    align_corners : bool
        Passed to affine_grid.

    Returns
    -------
    grid : torch.Tensor
        Sampling grid, shape (B, nz, ny, nx, 3).

    Notes
    -----
    Composed into a single ``affine_grid`` call rather than built as an identity
    grid that is then centred, scaled, rotated, unscaled, uncentred and
    translated in six separate elementwise passes. Every one of those passes
    materialises another ``(B, nz, ny, nx, 3)`` tensor -- 201 MB per batch
    element for a 256^3 volume -- which made the grid, not the ``grid_sample``
    it feeds, the dominant cost of :func:`rotate_volume`.

    The chain is affine in the identity grid ``g``, so it collapses exactly::

        ((g - c) * s) @ R.T / s + c + dx  ==  g @ A + b

    with ``A = diag(s) R.T diag(1/s)`` and ``b = c + dx - c @ A``, and
    ``affine_grid(M)`` itself evaluates ``g @ M[:, :3].T + M[:, 3]``. Measured
    1.8x faster at 1.7x lower peak memory on a 256^3 volume, and identical to
    the six-pass form to 2e-16 in float64.

    ``c`` is the identity grid's value at ``[nz//2, ny//2, nx//2]``, the RELION
    origin, written in closed form here (in ``affine_grid``'s (x, y, z) output
    order) instead of read back out of a materialised grid.
    """
    device, dtype = theta.device, theta.dtype

    # scale (to make the coordinates isotropic) and the RELION centre in
    # normalized (x, y, z) coordinates, built in ONE host-to-device transfer.
    # A small volume's rotation is launch-bound rather than bandwidth-bound --
    # a 32^3 grid_sample is ~0.35 ms of pure overhead either way -- so a second
    # `torch.tensor([...])` here, cheap as it looks, measured as a 4-5%
    # regression on the sub-64^3 volumes `packing/_shape.py` rotates in bulk.
    if align_corners:
        center_xyz = [
            2.0 * (nx // 2) / (nx - 1) - 1.0,
            2.0 * (ny // 2) / (ny - 1) - 1.0,
            2.0 * (nz // 2) / (nz - 1) - 1.0,
        ]
    else:
        center_xyz = [
            (2.0 * (nx // 2) + 1.0) / nx - 1.0,
            (2.0 * (ny // 2) + 1.0) / ny - 1.0,
            (2.0 * (nz // 2) + 1.0) / nz - 1.0,
        ]
    consts = torch.tensor(
        [(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2] + center_xyz,
        device=device,
        dtype=dtype,
    )
    scale, center = consts[:3], consts[3:]

    # A = diag(scale) @ R.T @ diag(1/scale)
    A = (
        scale.view(1, 3, 1)
        * theta[..., :-1].transpose(1, 2)
        * (1.0 / scale).view(1, 1, 3)
    )
    center = center.view(1, 1, 3).expand(theta.size(0), 1, 3)

    # translate the grid
    offset = center.squeeze(1) + theta[..., -1] - center.bmm(A).squeeze(1)
    composed = torch.cat([A.transpose(1, 2), offset.unsqueeze(-1)], dim=-1)
    return affine_sampling_grid(composed, nz, ny, nx, align_corners)


def rotate_volume(
    V: torch.Tensor,
    theta: torch.Tensor,
    origin: Literal["relion", "center"] = "relion",
    padding_mode: Literal["zeros", "border", "reflection"] = "border",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Rotates a single 3D volume based on the batch of 3x4 affine transform matrices.

    Written by Tristan Bepler.

    Parameters
    ----------
    V : torch.Tensor
        The volume to be rotated, must be real-valued. Shape (Z, Y, X).
    theta : torch.Tensor
        Batch of affine matrices, Bx3x4.
        Concatenates a 3x3 rotation matrix and a 3x1 translation vector.
    origin : str, optional
        Convention for the index of the origin of rotation. "relion" defines the
        origin to be at [nz//2, ny//2, nx//2], whereas "center" sets it to the
        grid's geometric centre, [(nz - 1) / 2, (ny - 1) / 2, (nx - 1) / 2]. The
        two differ by half a voxel per axis for even-sized volumes and coincide
        for odd-sized ones. Default "relion".
    padding_mode : str, optional
        Padding mode for grid_sample. Default "border".

    Returns
    -------
    V_rotated : torch.Tensor
         The rotated volumes. Shape (B, Z, Y, X).
    """

    B = theta.size(0)
    nz, ny, nx = V.size()

    # create the coordinate grid depending on origin convention.
    if origin == "relion":
        grid = _relion_rotation_grid(theta, nz, ny, nx, align_corners)
    elif origin == "center":
        grid = affine_sampling_grid(theta, nz, ny, nx, align_corners)

    # transform the volume
    V = V.unsqueeze(0).unsqueeze(1)  # (1 x 1 x Z x Y x X)
    V = V.expand(B, 1, nz, ny, nx)
    V_ = F.grid_sample(V, grid, align_corners=align_corners, padding_mode=padding_mode)

    return V_.squeeze(1)  # B x Z x Y x X


def split_affine_translation(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split an affine into a translation-free affine and its real-space displacement.

    A translation cannot be applied by resampling Fourier coefficients: shifting
    the samples of a spectrum modulates the density rather than moving it. Fourier
    rotation paths therefore strip the translation out, rotate, and reapply the
    displacement as a phase ramp (see :func:`apply_fourier_translation`).

    Parameters
    ----------
    theta : torch.Tensor
        Batch of affine matrices, shape (B, 3, 4), as built by
        :func:`build_affine_matrix`.

    Returns
    -------
    theta_rot : torch.Tensor
        The same affines with a zero translation column, shape (B, 3, 4).
    displacement : torch.Tensor
        Real-space displacement the translation column encodes, shape (B, 3), in
        normalized coordinates and (x, y, z) order.

    Notes
    -----
    ``rotate_volume`` samples the input at ``R x + T'``, which transforms the
    density by ``R^-1`` and then displaces it by ``-R^-1 T'``. The rotation part
    is assumed orthonormal, so ``R^-1`` is evaluated as ``R^T``.
    """
    R = theta[..., :3]
    T = theta[..., 3]
    theta_rot = torch.concat([R, torch.zeros_like(T).unsqueeze(-1)], dim=-1)
    displacement = -torch.einsum("bji,bj->bi", R, T)
    return theta_rot, displacement


def fourier_origin_displacement(
    theta: torch.Tensor, nz: int, ny: int, nx: int
) -> torch.Tensor:
    """
    Displacement converting a DC-centred rotation into one about the grid centre.

    A Fourier-space rotation is always centred on the spectrum's DC coefficient,
    which ``fft3(V, shift=True)`` places at index ``n // 2``: the ``origin="relion"``
    convention. Rotating about a different point ``c`` is the same rotation followed
    by a rigid displacement of ``(I - R^-1)(c - c0)``, so ``origin="center"`` costs
    one extra term in the phase ramp rather than a second resampling.

    Parameters
    ----------
    theta : torch.Tensor
        Batch of affine matrices, shape (B, 3, 4).
    nz, ny, nx : int
        Volume dimensions. Must be equal; see Raises.

    Returns
    -------
    displacement : torch.Tensor
        Displacement in normalized coordinates, shape (B, 3), (x, y, z) order,
        ready to be added to :func:`split_affine_translation`'s displacement.

    Raises
    ------
    ValueError
        If the volume is not cubic. The correction is computed in voxel units,
        since ``(I - R^-1)`` mixes axes and normalized coordinates scale
        differently per axis. A non-cubic box would also need `rotate_volume`'s
        anisotropic rescaling, which only its ``origin="relion"`` path performs.

    Notes
    -----
    ``origin="center"`` sits at ``(n - 1) / 2`` and ``origin="relion"`` at
    ``n // 2``, so the two differ by half a voxel per axis for even-sized volumes
    and coincide exactly for odd-sized ones, where this returns zero.
    """
    if not (nz == ny == nx):
        raise ValueError(
            "origin='center' is only supported for cubic volumes in Fourier space, "
            f"got (nz, ny, nx) = ({nz}, {ny}, {nx}). Use origin='relion', or rotate "
            "in real space."
        )

    R = theta[..., :3]
    offset = -0.5 if nz % 2 == 0 else 0.0
    delta = torch.full(
        (theta.shape[0], 3), offset, device=theta.device, dtype=theta.dtype
    )
    # (I - R^-1) delta, in voxels, then converted to normalized coordinates.
    displacement_voxels = delta - torch.einsum("bji,bj->bi", R, delta)
    return displacement_voxels * 2.0 / nz


def apply_fourier_translation(
    V_f: torch.Tensor, displacement: torch.Tensor
) -> torch.Tensor:
    """
    Displace a volume by multiplying its spectrum with a phase ramp.

    Parameters
    ----------
    V_f : torch.Tensor
        Batch of fftshifted complex spectra, shape (B, Z, Y, X).
    displacement : torch.Tensor
        Displacement in normalized coordinates, shape (B, 3), (x, y, z) order, as
        returned by :func:`split_affine_translation`.

    Returns
    -------
    V_f_shifted : torch.Tensor
        Spectra multiplied by ``exp(-2*pi*i*f.d)``, shape (B, Z, Y, X).

    Notes
    -----
    A phase ramp is exact to floating-point precision, including for sub-voxel
    displacements, where the real-space path interpolates. It is also circular:
    density leaving one face reappears at the opposite one, whereas
    :func:`rotate_volume` pads according to `padding_mode`. The two agree wherever
    the density is compact and away from the box edges.
    """
    if not torch.any(displacement != 0):
        return V_f

    _, nz, ny, nx = V_f.shape
    real_dtype = V_f.real.dtype

    def ramp(n: int, d_norm: torch.Tensor) -> torch.Tensor:
        # Normalized coordinates span [-1, 1] across n samples, so one normalized
        # unit is n/2 voxels.
        d = d_norm.to(real_dtype) * n / 2
        f = torch.fft.fftshift(
            torch.fft.fftfreq(n, device=V_f.device, dtype=real_dtype)
        )
        return torch.exp(-2j * torch.pi * f[None, :] * d[:, None])

    return (
        V_f
        * ramp(nz, displacement[:, 2])[:, :, None, None]
        * ramp(ny, displacement[:, 1])[:, None, :, None]
        * ramp(nx, displacement[:, 0])[:, None, None, :]
    )


def rotate_volume_fourier(
    V: torch.Tensor,
    theta: torch.Tensor,
    origin: Literal["relion", "center"] = "relion",
    padding_mode: Literal["zeros", "border", "reflection"] = "border",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Rotate a 3D volume by interpolating in Fourier space.

    Transforms to Fourier space, rotates real and imaginary parts separately
    using :func:`rotate_volume`, then transforms back. Any translation carried by
    `theta` is applied as a phase ramp rather than by resampling, which would
    modulate the density instead of moving it.

    Parameters
    ----------
    V : torch.Tensor
        Volume to rotate, shape (Z, Y, X).
    theta : torch.Tensor
        Batch of affine matrices, shape (B, 3, 4).
    origin : str, optional
        Rotation origin convention ('relion' or 'center'). The spectrum is
        fftshifted, placing its DC term at ``n // 2``, so the rotation itself is
        always centred there; 'center' is reached by folding the half-voxel offset
        into the phase ramp (see :func:`fourier_origin_displacement`, which
        restricts it to cubic volumes). Default is 'relion'.
    padding_mode : str, optional
        Padding mode for grid sampling. Default is 'border'.
    align_corners : bool, optional
        Passed to affine_grid and grid_sample. Default is False.

    Returns
    -------
    V_rot : torch.Tensor
        Rotated volume, shape (B, Z, Y, X).
    """
    if origin not in ("relion", "center"):
        raise ValueError(f"Unknown origin: {origin}. Must be 'relion' or 'center'.")

    # Fourier domain
    V_f = fft3(V, shift=True)  # Z x X x Y

    theta_rot, displacement = split_affine_translation(theta)
    if origin == "center":
        displacement = displacement + fourier_origin_displacement(theta, *V.shape)

    # rotate real and imag parts
    V_f_rot_real = rotate_volume(
        V_f.real,
        theta_rot,
        origin="relion",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    V_f_rot_imag = rotate_volume(
        V_f.imag,
        theta_rot,
        origin="relion",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    V_f_rot = apply_fourier_translation(
        torch.complex(V_f_rot_real, V_f_rot_imag), displacement
    )
    V_rot = ifft3(V_f_rot, shift=True)
    return V_rot.real  # B x Z x X x Y


def translations_angstrom_to_torch(
    T: torch.Tensor, n: int, voxel_size: float
) -> torch.Tensor:
    """
    Builds a batch of normalized translation vectors from rlnOriginXAngst and rlnOriginYAngst.

    Torch affine matrix uses a normalized coordinate system, where the coordinates
    of each axis ranges from [-1, 1]. Therefore, we need to do a coordinate
    transformation to match this Torch coordinate system for translations.

    Parameters
    ----------
    T : torch.Tensor
        Translation vector of shape (N, 2), built from [rlnOriginXAngst, rlnOriginYAngst]. In Å.
    n : int
        Number of pixels in x/y direction.
    voxel_size : float
        Voxel size in Å.

    Returns
    -------
    T_norm : torch.Tensor
        Batch of Torch normalized translation vectors with shape (N, 3).
    """
    num = len(T)
    if T.shape[-1] == 2:
        tz = torch.zeros(num, device=T.device)
        T = torch.concat([T, tz[..., None]], dim=-1)
    T_norm = T * 2 / n / voxel_size
    return T_norm


def build_affine_matrix(R: torch.Tensor, T: torch.Tensor | None = None) -> torch.Tensor:
    """
    Build a batch of Torch affine matrices (N, 3, 4) from rotation matrices and translations.

    The translation vector is pre-rotated, T_i' = sum_j R_ij * T_j, so that the
    shift acts in the lab frame rather than in the particle's own frame.
    ``grid_sample`` samples the input at ``R x + T'``, which transforms the density
    by ``R^-1`` followed by ``-R^-1 T'``; setting ``T' = R T`` makes the net
    displacement ``-T`` for any rotation. This matches RELION/CryoSPARC's origin
    offsets, which are image-plane shifts applied after the projection direction is
    fixed. Passing ``T`` through unrotated would instead shift the particle before
    the rotation.

    Parameters
    ----------
    R : torch.Tensor
        Batch of rotation matrices, shape (N, 3, 3).
    T : torch.Tensor, optional
        Batch of Torch-normalized translation vectors, shape (N, 3).
        Must be normalized via :func:`translations_angstrom_to_torch` first.
        If None, zero translations are used.

    Returns
    -------
    theta : torch.Tensor
        Batch of affine matrices, shape (N, 3, 4).
    """
    if R.ndim == 2:
        R = R.unsqueeze(0)
    if T is not None and T.ndim == 1:
        T = T.unsqueeze(0)
    if T is None:
        T = R.new_zeros(R.shape[0], 3)

    Tprime = torch.zeros_like(T)
    Tprime[:, 0] = R[:, 0, 0] * T[:, 0] + R[:, 0, 1] * T[:, 1] + R[:, 0, 2] * T[:, 2]
    Tprime[:, 1] = R[:, 1, 0] * T[:, 0] + R[:, 1, 1] * T[:, 1] + R[:, 1, 2] * T[:, 2]
    Tprime[:, 2] = R[:, 2, 0] * T[:, 0] + R[:, 2, 1] * T[:, 1] + R[:, 2, 2] * T[:, 2]
    theta = torch.concat([R, Tprime.unsqueeze(2)], dim=-1)
    return theta
