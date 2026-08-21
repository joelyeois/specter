"""Assemble a bank of pre-generated 3D blocks into a larger volume."""

from __future__ import annotations

import torch


def tile_volume_from_blocks(
    blocks: torch.Tensor,
    target_shape: tuple[int, int, int, int],
) -> torch.Tensor:
    """
    Tile a bank of 3-D blocks into a larger volume with random augmentation per tile.

    Each placed tile receives an independent random roll, flip, and 90°-multiple
    rotation before insertion, breaking periodicity that would otherwise create
    visible seams at block boundaries.

    Parameters
    ----------
    blocks : torch.Tensor
        Pre-generated block bank, shape ``(N_blocks, S, S, S)``. Blocks must be cubic.
    target_shape : tuple of int
        Desired output shape ``(N_batch, A, B, C)``.

    Returns
    -------
    torch.Tensor
        Assembled volume cropped to ``target_shape``, shape ``(N_batch, A, B, C)``.
    """
    N_blocks, block_size, _, _ = blocks.shape
    N_batch, A, B, C = target_shape

    batch_volumes = []
    for _ in range(N_batch):
        n_a = (A + block_size - 1) // block_size
        n_b = (B + block_size - 1) // block_size
        n_c = (C + block_size - 1) // block_size

        tile_idx = torch.randint(0, N_blocks, (n_a, n_b, n_c))

        a_slices = []
        for i in range(n_a):
            b_slices = []
            for j in range(n_b):
                c_slices = []
                for k in range(n_c):
                    blk = blocks[tile_idx[i, j, k]].clone()

                    # Random roll along all three axes
                    shifts = (
                        int(torch.randint(0, block_size, (1,)).item()),
                        int(torch.randint(0, block_size, (1,)).item()),
                        int(torch.randint(0, block_size, (1,)).item()),
                    )
                    blk = torch.roll(blk, shifts=shifts, dims=(0, 1, 2))

                    # Random flip along each axis
                    for dim in (0, 1, 2):
                        if torch.rand(1).item() < 0.5:
                            blk = torch.flip(blk, dims=(dim,))

                    # Random 90° rotations in each plane
                    for d0, d1 in ((0, 1), (0, 2), (1, 2)):
                        k_rot = int(torch.randint(0, 4, (1,)).item())
                        blk = torch.rot90(blk, k=k_rot, dims=(d0, d1))

                    c_slices.append(blk)
                b_slices.append(torch.cat(c_slices, dim=2))
            a_slices.append(torch.cat(b_slices, dim=1))

        batch_volumes.append(torch.cat(a_slices, dim=0))

    return torch.stack(batch_volumes, dim=0)[:, :A, :B, :C]


def tile_volume_from_blocks_blended(
    blocks: torch.Tensor,
    target_shape: tuple[int, int, int, int],
    overlap_frac: float = 0.5,
    conserve_sum: bool = True,
) -> torch.Tensor:
    """
    Tile a bank of 3-D blocks into a larger volume using overlap-add blending.

    Same per-tile random roll/flip/90-degree-rotation augmentation as
    ``tile_volume_from_blocks``, but adjacent blocks are cross-faded with a
    raised-cosine taper over ``overlap_frac * block_size`` instead of being
    concatenated edge-to-edge. ``tile_volume_from_blocks`` abuts blocks with a
    hard edge; each block tiles seamlessly with itself (blocks are generated
    with periodic boundary conditions), but two different blocks placed side
    by side don't share matching boundary values, and that mismatch sits on
    the regular block-size lattice as an axis-aligned artifact in Fourier
    space. Blending removes the hard discontinuity at the cost of a small,
    local reduction in high-frequency content within the overlap band.

    Blending two uncorrelated blocks with weights that sum to 1 is unbiased
    in expectation, but any single realization tends to read slightly low:
    features that land in the transition band get scaled by w<1 on both
    sides with no compensating boost. If ``conserve_sum`` is set, each
    assembled volume is rescaled so its total exactly matches
    ``blocks.mean()`` times the number of output voxels — the expected total
    for that many voxels of the source block distribution.

    Parameters
    ----------
    blocks : torch.Tensor
        Pre-generated block bank, shape ``(N_blocks, S, S, S)``. Blocks must be cubic.
    target_shape : tuple of int
        Desired output shape ``(N_batch, A, B, C)``.
    overlap_frac : float, optional
        Fraction of the block size used as the crossfade region at each
        edge. 0.5 gives a classic constant-overlap-add (Hann) taper. Default
        is 0.5.
    conserve_sum : bool, optional
        If True (default), rescale each output volume so its total sum
        matches the expected total implied by the source blocks' mean.

    Returns
    -------
    torch.Tensor
        Assembled volume, shape ``(N_batch, A, B, C)``.
    """
    N_blocks, S, _, _ = blocks.shape
    N_batch, A, B, C = target_shape
    device = blocks.device
    ov = max(1, int(S * overlap_frac))
    stride = S - ov

    # Bin-centered sample points (not linspace's inclusive endpoints): avoids a
    # true-zero weight landing on a real voxel, which would zero out both sides
    # of a seam at once (degenerates badly for small ov, e.g. ov=1).
    theta = (torch.arange(ov, device=device) + 0.5) / ov * (torch.pi / 2)
    ramp = torch.sin(theta) ** 2
    w1d = torch.ones(S, device=device)
    w1d[:ov] = ramp
    w1d[S - ov :] = ramp.flip(0)
    window3d = w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]

    def positions(total: int) -> list[int]:
        pos = [0]
        while pos[-1] + S < total:
            pos.append(pos[-1] + stride)
        return pos

    pa, pb, pc = positions(A), positions(B), positions(C)
    pad_shape = (pa[-1] + S, pb[-1] + S, pc[-1] + S)
    target_sum = blocks.mean() * (A * B * C)

    batch_volumes = []
    for _ in range(N_batch):
        acc = torch.zeros(pad_shape, device=device)
        wsum = torch.zeros(pad_shape, device=device)
        for i in pa:
            for j in pb:
                for k in pc:
                    idx = int(torch.randint(0, N_blocks, (1,)).item())
                    blk = blocks[idx].clone()

                    shifts = (
                        int(torch.randint(0, S, (1,)).item()),
                        int(torch.randint(0, S, (1,)).item()),
                        int(torch.randint(0, S, (1,)).item()),
                    )
                    blk = torch.roll(blk, shifts=shifts, dims=(0, 1, 2))

                    for dim in (0, 1, 2):
                        if torch.rand(1).item() < 0.5:
                            blk = torch.flip(blk, dims=(dim,))

                    for d0, d1 in ((0, 1), (0, 2), (1, 2)):
                        k_rot = int(torch.randint(0, 4, (1,)).item())
                        blk = torch.rot90(blk, k=k_rot, dims=(d0, d1))

                    acc[i : i + S, j : j + S, k : k + S] += blk * window3d
                    wsum[i : i + S, j : j + S, k : k + S] += window3d

        volume = acc / wsum.clamp_min(1e-8)
        volume = volume[:A, :B, :C]

        if conserve_sum:
            volume = volume * (target_sum / volume.sum().clamp_min(1e-8))

        batch_volumes.append(volume)

    return torch.stack(batch_volumes, dim=0)
