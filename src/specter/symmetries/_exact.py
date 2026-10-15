"""Point-group averaging without interpolation, through the volume's own transform."""

from __future__ import annotations

import torch

from ..nufft import GridTransform

#: Half-spectrum frequencies evaluated per NUFFT call. Bounds the working set at
#: about 0.5 GB of frequencies and sample points regardless of box size.
_CHUNK = 1 << 24


def _half_spectrum_freqs(
    shape: tuple[int, int, int], start: int, stop: int, device: torch.device
) -> torch.Tensor:
    """(x, y, z) frequencies in cycles per voxel of flat rfftn indices [start, stop)."""
    nz, ny, nx = shape
    nh = nx // 2 + 1
    i = torch.arange(start, stop, device=device)
    iz, rem = i // (ny * nh), i % (ny * nh)
    iy, ix = rem // nh, rem % nh

    def signed(j: torch.Tensor, n: int) -> torch.Tensor:
        return torch.where(j < (n + 1) // 2, j, j - n).double() / n

    return torch.stack([ix.double() / nx, signed(iy, ny), signed(iz, nz)], -1)


def symmetrize_exact(
    volume: torch.Tensor, rotations: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Average a volume over rotations about its central voxel, without interpolation.

    The transform of ``V(R r)`` is ``V^(R q)``, so the average is formed in
    Fourier space by evaluating the volume's discrete-space transform at
    rotated DFT frequencies (`specter.nufft.GridTransform`) and inverting the
    mean. No step resamples the grid, so for a map that is band-limited and
    lies inside the box's inscribed sphere, as a particle reconstruction does,
    the result is exact to the NUFFT's accuracy for any rotation, and
    averaging twice over a point group changes nothing (2e-6 relative, against
    0.15-0.21 per pass for ``apply_symmetry(method="fourier")`` and 0.015-0.02
    for ``"real"`` on a compact test object). Content reaching the box faces
    wraps, as in any Fourier-space rotation. A frequency in the corners of the
    cube, beyond the Nyquist sphere, averages only over the operators that
    keep it inside the band the grid represents.

    Parameters
    ----------
    volume : torch.Tensor
        Real volume of shape (Z, Y, X), rotated about voxel ``shape // 2``.
    rotations : torch.Tensor
        Rotation matrices of shape (G, 3, 3), or affine (G, 3, 4) as
        `get_rotation_matrices` returns, acting on (x, y, z). Output voxel
        ``r`` averages ``volume`` at ``R r``, the convention of
        `specter.rotations.rotate_volume`.
    eps : float, optional
        Requested NUFFT accuracy. Default 1e-6.

    Returns
    -------
    torch.Tensor
        The averaged volume, same shape, dtype and device as ``volume``.
    """
    shape = tuple(volume.shape)
    if len(shape) != 3:
        raise ValueError(f"expected a (Z, Y, X) volume, got shape {shape}")
    nz, ny, nx = shape
    device = volume.device
    rot = rotations[..., :3, :3].to(device=device, dtype=torch.float64)
    transform = GridTransform(volume, eps=eps)
    nh = nx // 2 + 1
    total = nz * ny * nh
    spectrum = torch.zeros(total, dtype=transform.complex_dtype, device=device)
    for start in range(0, total, _CHUNK):
        stop = min(start + _CHUNK, total)
        q = _half_spectrum_freqs(shape, start, stop, device)  # type: ignore[arg-type]
        acc = torch.zeros(stop - start, dtype=transform.complex_dtype, device=device)
        count = torch.zeros(stop - start, dtype=torch.float64, device=device)
        for r in rot:
            rq = q @ r.T
            # A frequency in the cube's corners (|q| > 1/2) can rotate out of the
            # band the grid represents, where the transform only repeats
            # periodically. Averaging such an alias in would corrupt the corners
            # and break idempotence, so each frequency averages over the
            # operators that keep it inside; within the Nyquist sphere that is
            # every operator.
            inside = rq.abs().amax(-1) <= 0.5 + 1e-9
            acc += torch.where(inside, transform(rq), 0)
            count += inside
        spectrum[start:stop] = acc * transform.dft_phase(q) / count.clamp(min=1)
    del transform
    out = torch.fft.irfftn(spectrum.reshape(nz, ny, nh), s=(nz, ny, nx))
    return out.to(volume.dtype)
