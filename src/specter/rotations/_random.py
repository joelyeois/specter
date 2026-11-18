"""Random rotations (quaternions and rotation matrices)."""

from __future__ import annotations

from typing import Literal

import roma
import torch


def random_quaternion(
    batchsize: int = 1,
    convention: Literal["xyzw", "wxyz"] = "xyzw",
    device: str | torch.device = "cpu",
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate uniformly random unit quaternions.

    Parameters
    ----------
    batchsize : int, optional
        Number of quaternions to generate. Default 1.
    convention : str, optional
        Quaternion convention ('xyzw' or 'wxyz'). Default 'xyzw'.
    device : str or torch.device, optional
        Device for output tensor. Default 'cpu'.
    generator : torch.Generator, optional
        Draw from this generator, on whatever device it lives, instead of the
        global one. Without it the draw is roma's, unchanged; with it, a
        normalised 4D Gaussian, which is uniform on the unit 3-sphere and so
        on rotations. roma forwards a generator to only part of its own draw.

    Returns
    -------
    quats : torch.Tensor
        Random unit quaternions. Shape (batchsize, 4) or (4,) if batchsize=1.

    References
    ----------
    .. [1] Shoemake, K. (1992). Uniform random rotations. In *Graphics Gems
       III (IBM Version)* (pp. 124–132). Elsevier.
       https://doi.org/10.1016/B978-0-08-050755-2.50036-1
       (roma's draw.)
    .. [2] Muller, M. E. (1959). A note on a method for generating points
       uniformly on *n*-dimensional spheres. *Communications of the ACM*,
       2(4), 19–20. https://doi.org/10.1145/377939.377946
       (The normalised 4D Gaussian.)
    """
    if generator is None:
        quats = roma.random_unitquat(size=(batchsize,), device=device)  # xyzw
    else:
        quats = _gaussian_unitquat(batchsize, generator).to(device)

    if convention == "wxyz":
        quats = roma.quat_xyzw_to_wxyz(quats)
    elif convention != "xyzw":
        raise ValueError("convention must be 'xyzw' or 'wxyz'")

    if batchsize == 1:
        return quats.squeeze(0)
    return quats


def random_rotation_matrix(
    batchsize: int = 1,
    device: str | torch.device = "cpu",
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate uniformly random 3x3 rotation matrices.

    Parameters
    ----------
    batchsize : int
        Number of rotation matrices to generate.
    device : str or torch.device
        Device for the output tensor.
    generator : torch.Generator, optional
        Draw from this generator, on whatever device it lives; see
        :func:`random_quaternion`. Without it the draw is roma's, unchanged.

    Returns
    -------
    rotmats : torch.Tensor
        Tensor of shape (batchsize, 3, 3) containing rotation matrices.

    References
    ----------
    .. [1] Shoemake, K. (1992). Uniform random rotations. In *Graphics Gems
       III (IBM Version)* (pp. 124–132). Elsevier.
       https://doi.org/10.1016/B978-0-08-050755-2.50036-1
       (roma's draw.)
    .. [2] Muller, M. E. (1959). A note on a method for generating points
       uniformly on *n*-dimensional spheres. *Communications of the ACM*,
       2(4), 19–20. https://doi.org/10.1145/377939.377946
       (The normalised 4D Gaussian.)
    """
    if generator is None:
        rotmats = roma.random_rotmat(size=(batchsize,), device=device)
    else:
        quats = _gaussian_unitquat(batchsize, generator)
        rotmats = roma.unitquat_to_rotmat(quats).to(device)

    if batchsize == 1:
        return rotmats.squeeze(0)
    return rotmats


def resample_viewing_directions(
    quaternions: torch.Tensor,
    n: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Draw `n` orientations with the viewing directions of a set of poses.

    A pose's viewing direction is the particle-frame axis that points along
    the beam, the last column of its rotation matrix. Each draw takes the
    viewing direction of one source pose, chosen uniformly with replacement,
    and a uniformly random rotation about the beam: a specimen's preferred
    orientations constrain which face lies along the beam and leave the
    in-plane angle free. A source set with uniform orientations therefore
    yields uniform orientations.

    Parameters
    ----------
    quaternions : torch.Tensor
        Source poses as unit quaternions, shape (m, 4), in roma's ``xyzw``
        convention. They sample the viewing-direction distribution, so more
        poses resolve it more finely.
    n : int
        Number of orientations to draw.
    generator : torch.Generator, optional
        Draw from this generator, on whatever device it lives, instead of the
        global one.

    Returns
    -------
    quats : torch.Tensor
        Unit quaternions, shape (n, 4), ``xyzw``, on the CPU.
    """
    source = torch.as_tensor(quaternions, dtype=torch.float32).cpu()
    if source.ndim != 2 or source.shape[-1] != 4 or len(source) == 0:
        raise ValueError(
            f"quaternions must have shape (m, 4) with m >= 1, not {tuple(source.shape)}"
        )
    source = source / source.norm(dim=-1, keepdim=True)
    device = "cpu" if generator is None else generator.device
    rows = torch.randint(len(source), (n,), generator=generator, device=device).cpu()
    half = torch.pi * torch.rand(n, generator=generator, device=device).cpu()
    zeros = torch.zeros(n)
    # A rotation about z applied before the pose turns the image in its plane
    # and leaves the last column of the pose's rotation matrix where it was.
    spin = torch.stack([zeros, zeros, torch.sin(half), torch.cos(half)], dim=-1)
    return roma.quat_product(source[rows], spin)


def sample_viewing_direction_histogram(
    histogram: torch.Tensor,
    n: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Draw `n` orientations from a viewing-direction histogram.

    The histogram is binned in the azimuth and elevation of the viewing
    direction ``d``, the particle-frame axis that points along the beam
    (the last column of the pose's rotation matrix): ``azimuth =
    atan2(d_y, d_x)`` and ``elevation = asin(d_z)``. Each draw picks a bin
    with probability proportional to its count, a direction uniform over
    the part of the sphere the bin covers, and a uniform in-plane angle.
    The axes are those of CryoSPARC's viewing direction distribution plot,
    with the pose convention of its ``alignments3D/pose``.

    Parameters
    ----------
    histogram : torch.Tensor
        Non-negative particle counts, shape (n_elevation, n_azimuth). Row
        ``i`` covers elevations from ``-pi/2 + i * pi / n_elevation`` and
        column ``j`` azimuths from ``-pi + j * 2 * pi / n_azimuth``, in
        equal steps. Counts are per bin, not per unit solid angle, so
        uniform orientations give rows that scale with ``cos(elevation)``.
    n : int
        Number of orientations to draw.
    generator : torch.Generator, optional
        Draw from this generator, on whatever device it lives, instead of the
        global one.

    Returns
    -------
    quats : torch.Tensor
        Unit quaternions, shape (n, 4), ``xyzw``, on the CPU.
    """
    counts = torch.as_tensor(histogram, dtype=torch.float64).cpu()
    if counts.ndim != 2 or counts.numel() == 0:
        raise ValueError(
            "histogram must have shape (n_elevation, n_azimuth), "
            f"not {tuple(counts.shape)}"
        )
    if not torch.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError(
            "histogram must hold finite, non-negative counts with a positive sum"
        )
    n_elevation, n_azimuth = counts.shape
    device = "cpu" if generator is None else generator.device
    bins = torch.multinomial(
        counts.flatten().to(device), n, replacement=True, generator=generator
    ).cpu()
    u = torch.rand(n, 3, generator=generator, device=device).cpu().double()
    row = torch.div(bins, n_azimuth, rounding_mode="floor")
    column = bins % n_azimuth
    azimuth = -torch.pi + (column + u[:, 0]) * (2 * torch.pi / n_azimuth)
    # Uniform over the bin's area is uniform in sin(elevation).
    edges = torch.sin(
        -torch.pi / 2 + torch.arange(n_elevation + 1) * (torch.pi / n_elevation)
    ).double()
    sine = edges[row] + u[:, 1] * (edges[row + 1] - edges[row])
    polar = torch.pi / 2 - torch.asin(sine.clamp(-1.0, 1.0))
    # Rz(azimuth) Ry(polar) carries z onto the viewing direction, and the
    # last angle spins about the beam.
    angles = torch.stack([azimuth, polar, 2 * torch.pi * u[:, 2]], dim=-1)
    return roma.euler_to_unitquat("ZYZ", angles.float())


def _gaussian_unitquat(batchsize: int, generator: torch.Generator) -> torch.Tensor:
    """
    ``(batchsize, 4)`` uniform unit quaternions (xyzw) from `generator` alone.

    Drawn on the generator's own device, the one place its stream can be
    read, so a CUDA generator gives CUDA draws. A 4D isotropic Gaussian
    normalised to unit length is uniform on the 3-sphere.
    """
    q = torch.randn(batchsize, 4, generator=generator, device=generator.device)
    return q / q.norm(dim=-1, keepdim=True)


def random_rotation_matrix_from_generator(
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Draw one uniformly random 3D rotation matrix from a given generator.

    QR decomposition of a random Gaussian matrix (Mezzadri, 2007), sign- and
    determinant-corrected for a proper rotation (no reflection).

    The draw :meth:`IceBank._extract_crop` has always made, kept beside
    :func:`random_rotation_matrix` so seeded ice stays what it was. Its only
    random draw is the single ``torch.randn(..., generator=generator)``
    below, so it is fully generator-reproducible; roma's ``random_rotmat``/
    ``random_unitquat`` are not, since they forward a generator to only one of
    their three underlying draws (the other two fall back to the global RNG
    -- confirmed empirically, not documented).

    Parameters
    ----------
    generator : torch.Generator, optional
        RNG to draw from, on any device: the Gaussian is drawn where the
        generator lives and decomposed on the CPU, so a CPU generator's
        rotations are what they always were. Default is the global RNG.

    Returns
    -------
    torch.Tensor
        Shape (3, 3), on the CPU, ``det(R) == 1``.

    References
    ----------
    .. [1] Mezzadri, F. (2007). How to generate random matrices from the
       classical compact groups. *Notices of the American Mathematical
       Society*, 54(5), 592–604. https://arxiv.org/abs/math-ph/0609050
    """
    device = "cpu" if generator is None else generator.device
    A = torch.randn(3, 3, generator=generator, device=device).cpu()
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.diagonal(R))
    Q = Q * d
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q
