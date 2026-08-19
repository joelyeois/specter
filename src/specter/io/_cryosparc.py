from __future__ import annotations

from typing import Literal

import roma
import torch
from cryosparc.dataset import Dataset
from rich.console import Console

from ..constants import energy_to_wavelength
from ._common import _select_particles

_console = Console()


def _load_csfile_parameters(
    csfile_path: str,
    rotation_representation: Literal["quaternion", "rotvec"],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]:
    """Load and derive all per-particle imaging parameters from a .cs file, unfiltered.

    Returns
    -------
    tuple
        ``(voltage_kv, pixel_size, alpha, rotations, translations_A, ctf_params,
        scale, anisomag, split)`` for every particle in the dataset.
    """
    dataset = Dataset.load(csfile_path)

    # extract translations
    translations_px = torch.as_tensor(dataset["alignments3D/shift"])
    pixel_size = torch.as_tensor(dataset["alignments3D/psize_A"])
    translations_A = translations_px * pixel_size[..., None]
    if torch.allclose(pixel_size[0], pixel_size.mean()):
        pixel_size = pixel_size[0]
    else:
        _console.print(
            "[yellow]Warning:[/yellow] pixel size is not the same for all particles."
        )

    # extract spherical aberration
    cs_mm = torch.as_tensor(dataset["ctf/cs_mm"])
    cs_A = cs_mm * 1e7

    # extract defocus
    dfang_rad = torch.as_tensor(dataset["ctf/df_angle_rad"])
    dfang_deg = dfang_rad / torch.pi * 180
    dfu_A = torch.as_tensor(dataset["ctf/df1_A"])
    dfv_A = torch.as_tensor(dataset["ctf/df2_A"])

    # extract amplitude contrast
    alpha = torch.as_tensor(dataset["ctf/amp_contrast"])
    if torch.allclose(alpha[0], alpha.mean()):
        alpha = alpha[0]
    else:
        _console.print(
            "[yellow]Warning:[/yellow] amplitude contrast is not the same for all particles."
        )

    # extract voltage
    voltage_kv = torch.as_tensor(dataset["ctf/accel_kv"])
    if torch.allclose(voltage_kv[0], voltage_kv.mean()):
        voltage_kv = voltage_kv[0]
    else:
        _console.print(
            "[yellow]Warning:[/yellow] voltage is not the same for all particles."
        )
    wavelength_A = energy_to_wavelength(voltage_kv)

    # extract rotations
    pose = torch.as_tensor(dataset["alignments3D/pose"], dtype=torch.float32)
    if rotation_representation == "quaternion":
        rotations = roma.rotvec_to_unitquat(pose)
    elif rotation_representation == "rotvec":
        rotations = pose

    # extract split
    split = torch.as_tensor(dataset["alignments3D/split"].astype(int))

    # extract beamtilt
    beamtiltx_rad = torch.arcsin(torch.as_tensor(dataset["ctf/tilt_A"][:, 0] / cs_A))
    beamtilty_rad = torch.arcsin(torch.as_tensor(dataset["ctf/tilt_A"][:, 1] / cs_A))

    # extract phaseshift
    phaseshift_rad = torch.as_tensor(dataset["ctf/phase_shift_rad"])

    # extract ctf shift, and add to translations
    beamshift_A = torch.as_tensor(dataset["ctf/shift_A"])
    translations_A -= beamshift_A

    # extract trefoil. CryoSPARC's ctf/trefoil_A is a raw Å-scale
    # coefficient; specter's trefoil1/trefoil2 (see
    # aberrations._functions.trefoil) are the direct k^3-domain phase
    # coefficients in Å³, related by chi_trefoil =
    # (2*pi/3)*wavelength^2*trefoil_A -- see newctf.py's
    # params_to_coeffs_odd/gen_basis_odd for the CryoSPARC-side derivation.
    trefoil_A = torch.as_tensor(dataset["ctf/trefoil_A"])
    trefoil1 = (2 * torch.pi / 3) * wavelength_A**2 * trefoil_A[:, 0]
    trefoil2 = (2 * torch.pi / 3) * wavelength_A**2 * trefoil_A[:, 1]

    # extract tetrafoil. CryoSPARC's ctf/tetra_A holds 4 raw Å-scale
    # coefficients spanning secondary astigmatism (n=4, m=+-2) and true
    # tetrafoil (n=4, m=+-4); specter's tetrafoil1-4 (see
    # aberrations._functions.tetrafoil) are the direct k^4-domain phase
    # coefficients in Å⁴. See newctf.py's
    # params_to_coeffs_even/gen_basis_even for the CryoSPARC-side
    # derivation of these prefactors (including the sign/index mapping).
    tetra_A = torch.as_tensor(dataset["ctf/tetra_A"])
    tetrafoil1 = -2 * torch.pi * wavelength_A**3 * tetra_A[:, 0]
    tetrafoil2 = 2 * torch.pi * wavelength_A**3 * tetra_A[:, 1]
    tetrafoil3 = (torch.pi / 2) * wavelength_A**3 * tetra_A[:, 2]
    tetrafoil4 = -(torch.pi / 2) * wavelength_A**3 * tetra_A[:, 3]

    # extract per-particle scale factors
    scale = torch.as_tensor(dataset["alignments3D/alpha"])

    # extract anisotropic magnification
    # cryosparc defines the M matrix in Fourier space, and stores it after
    # subtracting away the identity. Ghostbuster uses the real-space M instead.
    anisomag_raw = torch.as_tensor(dataset["ctf/anisomag"]).reshape(-1, 2, 2)
    anisomag: torch.Tensor | None
    if torch.allclose(torch.tensor(0.0), torch.sum(anisomag_raw)):
        anisomag = None
    else:
        anisomag = anisomag_raw + torch.eye(2).unsqueeze(0)
        # Compute the real-space equivalent matrix
        anisomag = torch.inverse(anisomag.mT)

        # correct for anisotropic shift
        corrected_shifts = translations_A.unsqueeze(
            -1
        )  # Add a dimension to make it (B, 2, 1)

        # Perform batch matrix multiplication
        corrected_shifts = torch.bmm(anisomag, corrected_shifts)

        # Remove the last dimension to get (B, 2)
        corrected_shifts = corrected_shifts.squeeze(-1)
        translations_A = corrected_shifts

    ctf_params = {
        "cs": cs_A,
        "dfu": dfu_A,
        "dfv": dfv_A,
        "dfang": dfang_deg,
        "tiltx": beamtiltx_rad,
        "tilty": beamtilty_rad,
        "phaseshift": phaseshift_rad,
        "trefoil1": trefoil1,
        "trefoil2": trefoil2,
        "tetrafoil1": tetrafoil1,
        "tetrafoil2": tetrafoil2,
        "tetrafoil3": tetrafoil3,
        "tetrafoil4": tetrafoil4,
    }

    return (
        voltage_kv,
        pixel_size,
        alpha,
        rotations,
        translations_A,
        ctf_params,
        scale,
        anisomag,
        split,
    )


def extract_parameters_from_csfile(
    csfile_path: str,
    halfset: Literal["A", "B", "all"] = "all",
    rotation_representation: Literal["quaternion", "rotvec"] = "quaternion",
    n_particles: int | None = None,
) -> tuple:
    """
    Extract poses and CTF parameters from CryoSPARC .cs file.

    Parameters
    ----------
    csfile_path : str
        Path of the .cs file.
    halfset : str, optional
        Which gold-standard half-set to return, from ``alignments3D/split``
        (raw values 0 and 1). Options are 'A', 'B', or 'all'. Default is 'all'.
    rotation_representation : str, optional
        Representation of rotations. 'quaternion' or 'rotvec'. Default is 'quaternion'.
    n_particles : int, optional
        If given, only the first ``n_particles`` particles (after filtering by
        ``halfset``) are returned. Default is None (return all).

    Returns
    -------
    voltage_kv : torch.Tensor
        Voltage in kV.
    pixel_size : torch.Tensor
        Pixel sizes in Å.
    alpha : torch.Tensor
        Amplitude contrast ratio.
    rotations : torch.Tensor
        Quaternions with shape (N, 4) or rotation vectors.
    translations_A : torch.Tensor
        xy-translations in Å with shape (N, 2).
    ctf_params : torch.Tensor
        CTF parameters with shape (N, 7). Parameters are (Cs, dfu, dfv, dfang, tiltx, tilty, phaseshift).
    scale : torch.Tensor
        Per-particle scale factors.
    anisomag : torch.Tensor or None
        Anisotropic magnification matrices (N, 2, 2) or None if identity.
    indices : torch.Tensor
        Indices of the extracted particles from the dataset.
    halfset_labels : torch.Tensor or None
        1-D integer tensor of length ``N`` with the raw ``alignments3D/split``
        values (0 or 1) -- 0 corresponds to halfset 'A', 1 to halfset 'B'.
        Only returned when ``halfset == "all"``; ``None`` otherwise.
    """
    (
        voltage_kv,
        pixel_size,
        alpha,
        rotations,
        translations_A,
        ctf_params,
        scale,
        anisomag,
        split,
    ) = _load_csfile_parameters(csfile_path, rotation_representation)

    if halfset == "all":
        mask = torch.ones_like(split, dtype=torch.bool)
        indices = torch.arange(len(split))
        halfset_labels: torch.Tensor | None = split
    else:  # "A" or "B"
        mask = split == {"A": 0, "B": 1}[halfset]
        indices = torch.squeeze(torch.nonzero(mask))
        halfset_labels = None

    indices, rotations, translations_A, ctf_params, scale, anisomag = _select_particles(
        mask,
        indices,
        rotations,
        translations_A,
        ctf_params,
        scale,
        anisomag,
        n_particles,
    )
    if halfset_labels is not None and n_particles is not None:
        halfset_labels = halfset_labels[:n_particles]

    return (
        voltage_kv,
        pixel_size,
        alpha,
        rotations,
        translations_A,
        ctf_params,
        scale,
        anisomag,
        indices,
        halfset_labels,
    )
