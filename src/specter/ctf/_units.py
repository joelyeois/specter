"""Unit-conversion helpers bridging specter's legacy, physical-k-based CTF
convention and torch-ctf's own units.

torch-ctf's scalar CTF terms (defocus, cs, phase_shift, ...) use
micrometers/mm/degrees -- ordinary physical units, just not Å. Its
Zernike-coefficient terms (trefoil, tetrafoil, beam-tilt-derived coma),
however, are evaluated against a *radial frequency normalized to the
grid's own maximum spatial frequency* (``rho`` in [0, 1]), not physical
1/Å -- ``torch_ctf._ctf_core._build_freq_grid`` calls
``torch_grid_utils.polar_grid.fftfreq_grid_polar`` with its default
``normalize_rho=True``. This is easy to miss: a Zernike coefficient
calibrated against a physical-k convention (``coeff * k**n``, e.g.
specter's own ``aberrations._functions.trefoil``/``beamtilt``, or a
RELION/cryoSPARC-reported value) must be rescaled by
``zernike_rho_max(image_shape, pixel_size)**n`` before handing it to
torch-ctf -- and that scale is *not* simply ``1/(2*pixel_size)`` (the
single-axis Nyquist frequency); for a square grid it's larger by a factor
of up to sqrt(2), since ``rho`` is normalized by the grid's corner
(diagonal) frequency, not the axis Nyquist.

This module is purely an opt-in migration/parity bridge (used by the
parity tests in ``tests/test_ctf_transfer.py``). Code written to target
torch-ctf directly should just use its native rho-normalized convention
for Zernike terms and skip this entirely.
"""

from __future__ import annotations

import torch


def zernike_rho_max(image_shape: tuple[int, int], pixel_size: float) -> float:
    """Maximum (corner) spatial frequency of an fftfreq grid, in 1/Å.

    torch-ctf's Zernike-coefficient terms are evaluated against
    ``rho = k / zernike_rho_max(...)``, so this is the factor needed to
    convert a physical-k-calibrated coefficient (``coeff * k**n``) into
    torch-ctf's convention: ``coeff_zernike = coeff * zernike_rho_max(...) ** n``.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Shape of the 2D image the CTF is being computed for.
    pixel_size : float
        Pixel size in Å.

    Returns
    -------
    float
        The grid's maximum radial spatial frequency, in 1/Å.
    """
    h, w = image_shape
    ky = torch.fft.fftfreq(h, d=pixel_size)
    kx = torch.fft.fftfreq(w, d=pixel_size)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    return torch.sqrt(KY**2 + KX**2).max().item()
