"""
Clean templates ``theta`` for the MRA experiments.

Two sources:

* :func:`synthetic_template` -- a smooth random signal with full Fourier
  support (a *generic* signal in the paper's sense), no SPECTER needed.
* :func:`template_from_volume` / :func:`template_from_pdb` -- a single-view,
  physics-free projection of a real macromolecule computed with SPECTER's
  ``ImageGenerator`` in ``projection`` mode. This is Phase A of the plan: the
  simulator only provides a realistic ``theta``; shifting and noising happen in
  :mod:`mj_dsa4288.mra.model`.
"""

from __future__ import annotations

from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
PDB_CACHE = _HERE.parent / "data" / "pdb"


def synthetic_template(
    shape: tuple[int, ...],
    corr_length: float = 4.0,
    mean_offset: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Smooth random template with full Fourier support (a generic signal).

    White Gaussian noise is filtered by ``1 / (1 + (k * corr_length)^2)``, which
    is strictly positive at every frequency, so every Fourier coefficient is
    non-zero almost surely (the genericity condition of Section 2.2). The
    result is normalised to unit 2-norm after adding ``mean_offset`` times the
    template's RMS as a constant, so the first moment is usable for the scale
    fix in :func:`mj_dsa4288.mra.jennrich.homojen`.

    Parameters
    ----------
    shape : tuple[int, ...]
        ``(d,)`` for a 1-D signal or ``(H, W)`` for an image.
    corr_length : float, optional
        Correlation length in pixels. Default 4.
    mean_offset : float, optional
        Constant offset relative to the RMS of the fluctuating part.
    generator : torch.Generator, optional
        RNG.

    Returns
    -------
    torch.Tensor
        Template with ``||theta||_2 = 1``.
    """
    noise = torch.randn(shape, generator=generator)
    grids = torch.meshgrid(*[torch.fft.fftfreq(s) for s in shape], indexing="ij")
    k2 = sum(g**2 for g in grids)
    filt = 1.0 / (1.0 + (2 * torch.pi * corr_length) ** 2 * k2)
    field = torch.fft.ifftn(torch.fft.fftn(noise) * filt).real
    field = field - field.mean()
    field = field + mean_offset * field.pow(2).mean().sqrt()
    return normalise_template(field)


def normalise_template(theta: torch.Tensor) -> torch.Tensor:
    """Scale ``theta`` to unit 2-norm (the paper's ``||theta||_2 = 1`` convention)."""
    return theta / theta.norm()


def template_from_volume(
    volume: torch.Tensor,
    pixel_size: float,
    quaternion: torch.Tensor | None = None,
    voltage: float = 300.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Physics-free single-view projection of a scattering potential via SPECTER.

    Runs ``ImageGenerator`` with ``scattering_model="projection"`` and no ice,
    noise, detector or envelopes, and returns the projected phase
    ``2 sigma_e dx sum_z V`` (the exit-wave phase in the weak-phase limit),
    i.e. the linear projection image the MRA model needs.

    Parameters
    ----------
    volume : torch.Tensor
        Scattering potential ``[Z, Y, X]``.
    pixel_size : float
        Pixel size in Angstrom.
    quaternion : torch.Tensor, optional
        Fixed view ``[4]``; identity if omitted.
    voltage : float, optional
        Accelerating voltage in kV. Only scales the phase. Default 300.
    device : str, optional
        Torch device.

    Returns
    -------
    torch.Tensor
        Real projection image ``[Y, X]`` on CPU, normalised to unit 2-norm.
    """
    from specter.imagegenerator import ImageGenerator

    if quaternion is None:
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])
    model = ImageGenerator(
        volume,
        pixel_size,
        quaternions=quaternion[None].float(),
        translations=torch.zeros(1, 2),
        ctf_params={"dfu": torch.zeros(1), "cs": torch.zeros(1)},
        voltage=voltage,
        dose_per_angstrom=1.0,
        ice_model=None,
        scattering_model="projection",
        aberration_model="holography",
        noise_model=None,
        detector_model=None,
        alpha=0.0,
        verbose=False,
    ).to(device)
    with torch.no_grad():
        model(torch.tensor([0], device=device))
    phi = model.exitwaves[0].detach().cpu()
    if torch.is_complex(phi):
        phi = torch.angle(phi)
    return normalise_template(phi.float())


def template_from_pdb(
    pdb_id: str,
    num_pixels: int = 48,
    pixel_size: float = 3.0,
    quaternion: torch.Tensor | None = None,
    voltage: float = 300.0,
    parameterization: str = "kirkland",
    device: str = "cpu",
    cache_dir: Path = PDB_CACHE,
) -> torch.Tensor:
    """
    Build a template from a PDB entry with SPECTER's ``PotentialBuilder``.

    Downloads go to ``mj_dsa4288/data/pdb`` (gitignored), never to the repo's
    shared ``pdb-data`` folder.

    Parameters
    ----------
    pdb_id : str
        Four-character PDB code or local mmCIF/PDB path.
    num_pixels : int, optional
        Box size in pixels. Keep small (32 to 64); the bispectrum is O(d^3).
    pixel_size : float, optional
        Pixel size in Angstrom.
    quaternion : torch.Tensor, optional
        Fixed view; identity if omitted.
    voltage : float, optional
        kV. Default 300.
    parameterization : str, optional
        Atomic potential parameterisation. Default ``"kirkland"``.
    device : str, optional
        Torch device.
    cache_dir : Path, optional
        Where PDB files are stored.

    Returns
    -------
    torch.Tensor
        Real projection image ``[num_pixels, num_pixels]`` with unit 2-norm.
    """
    from specter.pdb import PDB
    from specter.potential import PotentialBuilder

    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb = PDB(pdb_id, assembly=True, savefolder=str(cache_dir), verbose=False)
    builder = PotentialBuilder(
        num_pixels, pixel_size, pdb.atomic_numbers, parameterization=parameterization
    )
    with torch.no_grad():
        volume = builder(pdb.coordinates, method="analytic").clone()
    return template_from_volume(volume, pixel_size, quaternion, voltage, device)
