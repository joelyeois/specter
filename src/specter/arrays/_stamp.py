"""
Stamping copies of a small volume into a large one at centred positions:
:func:`insert_particles_into_micrograph`, shared by crowding, the membrane
and tomogram generators and the packers.
"""

from __future__ import annotations

import contextlib

import torch

from ..cpu_threads import limited_cpu_threads
from ._padding import clip_insert_bounds


def insert_particles_into_micrograph(
    volumes: torch.Tensor,
    positions: torch.Tensor,
    pixel_size: float = 1.0,
    micro_shape: tuple[int, int, int] | None = None,
    micrograph: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Insert rotated 3D volumes into a 3D micrograph centered at the origin.

    Parameters
    ----------
    volumes : torch.Tensor
        Rotated volumes with shape (N, Zp, Yp, Xp) where N is the number
        of particles to insert.
    positions : torch.Tensor
        Particle center coordinates in physical units (x, y, z) with shape
        (N, 3) or (N, 2). Origin is at the center of the micrograph. If
        shape is (N, 2), z-coordinates are assumed to be zero.
    pixel_size : float, optional
        Physical size of one pixel in same units as positions. Default is 1.0.
    micro_shape : tuple of int, optional
        Shape of micrograph (Z, Y, X). Required if micrograph is None.
    micrograph : torch.Tensor, optional
        Existing micrograph to insert volumes into. If None, a new micrograph
        is created with shape micro_shape.

    Returns
    -------
    micrograph : torch.Tensor
        Micrograph with volumes inserted, shape (Z, Y, X).

    Raises
    ------
    ValueError
        If neither micro_shape nor micrograph is provided.

    Notes
    -----
    If a particle extends beyond the micrograph boundaries, only the portion
    within bounds is inserted (clipping at edges).

    Each volume is stamped at the whole voxel nearest its position, with its
    index ``n // 2`` on that voxel; the remainder, up to half a voxel per
    axis, is not applied here. A caller that resamples its copies anyway
    (every rotated copy does) carries it through that resampling instead,
    with :func:`~specter.rotations.fold_subvoxel_offsets`, and passes the whole-voxel positions
    that function returns.
    """
    N, Zp, Yp, Xp = volumes.shape
    device = volumes.device

    # Allocate micrograph
    if micrograph is not None:
        micrograph = micrograph.to(device)
        Z, Y, X = micrograph.shape
    elif micro_shape is not None:
        Z, Y, X = micro_shape
        micrograph = torch.zeros(micro_shape, device=device)
    else:
        raise ValueError(
            "Must provide either `micro_shape` or an existing `micrograph`."
        )

    volumes = volumes.to(device)
    positions = positions.to(device)
    if positions.shape[1] == 2:
        zeros = torch.zeros(
            (positions.shape[0], 1), device=positions.device, dtype=positions.dtype
        )
        positions = torch.cat([positions, zeros], dim=1)

    # Convert from physical units to pixel indices
    positions_pixels = positions / pixel_size
    positions_int = positions_pixels.round().long()  # shape (N, 3), order (x, y, z)

    # Micrograph center indices
    cz_center = Z // 2
    cy_center = Y // 2
    cx_center = X // 2

    # Per particle: a bounds clip and one slice add. Under the default
    # thread pool of a many-core host the small CPU ops dominate; see
    # cpu_threads. A CUDA canvas has no CPU tensor op in the loop at all,
    # so the cap there only pays `torch.set_num_threads`' pool-resize cost
    # (3.15 ms per down/up pair at a 64-thread pool on this host) once per
    # call -- 337 calls for one 21.5k-instance tomogram species pool.
    cap = contextlib.nullcontext() if micrograph.is_cuda else limited_cpu_threads()
    with cap:
        _insert_all(
            volumes,
            positions_int,
            (cz_center, cy_center, cx_center),
            (Zp, Yp, Xp),
            (Z, Y, X),
            micrograph,
        )
    return micrograph


def _insert_all(
    volumes: torch.Tensor,
    positions_int: torch.Tensor,
    center: tuple[int, int, int],
    part_shape: tuple[int, int, int],
    micro_shape: tuple[int, int, int],
    micrograph: torch.Tensor,
) -> None:
    """The insertion loop of :func:`insert_particles_into_micrograph`."""
    cz_center, cy_center, cx_center = center
    Zp, Yp, Xp = part_shape
    Z, Y, X = micro_shape
    # One host transfer for the whole chunk rather than three `.item()`s per
    # particle: on a CUDA tensor each of those is a separate device
    # synchronisation, and the loop does nothing else that needs one.
    positions_list = positions_int.cpu().tolist()
    for i, (px_i, py_i, pz_i) in enumerate(positions_list):
        # Convert centered coords to array indices
        cx_index = cx_center + px_i
        cy_index = cy_center + py_i
        cz_index = cz_center + pz_i

        bounds = clip_insert_bounds(
            (cz_index, cy_index, cx_index), (Zp, Yp, Xp), (Z, Y, X)
        )
        if bounds is None:
            continue
        dst, src = bounds
        micrograph[dst] += volumes[i][src]
