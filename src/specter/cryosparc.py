from __future__ import annotations

import os
from typing import Literal

import mrcfile
import numpy as np
import pandas as pd
import starfile
import torch
from cryosparc.dataset import Dataset
from rich.console import Console
from scipy.spatial.transform import Rotation

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
        ``(energy_kev, pixel_size, alpha, rotations, translations_A, ctf_params,
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

    # extract energy
    energy_kev = torch.as_tensor(dataset["ctf/accel_kv"])
    if torch.allclose(energy_kev[0], energy_kev.mean()):
        energy_kev = energy_kev[0]
    else:
        _console.print(
            "[yellow]Warning:[/yellow] energy is not the same for all particles."
        )

    # extract rotations
    r = Rotation.from_rotvec(dataset["alignments3D/pose"])
    if rotation_representation == "quaternion":
        rotations = torch.from_numpy(r.as_quat()).type(torch.float32)
    elif rotation_representation == "rotvec":
        # rotations = torch.from_numpy(r.as_rotvec()).type(torch.float32)
        rotations = torch.as_tensor(dataset["alignments3D/pose"])

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

    # extract trefoil
    tref1 = torch.as_tensor(dataset["ctf/trefoil_A"][:, 0] / 1000)
    tref2 = torch.as_tensor(dataset["ctf/trefoil_A"][:, 1] / 1000)

    # extract per-particle scale factors
    scale = torch.as_tensor(dataset["alignments3D/alpha"])

    # extract anisotropic magnification
    # cryosparc defines the M matrix in Fourier space, and stores it after
    # subtracting away the identity. Ghostbuster uses the real-space M instead.
    anisomag = torch.as_tensor(dataset["ctf/anisomag"]).reshape(-1, 2, 2)
    if torch.allclose(torch.tensor(0.0), torch.sum(anisomag)):
        anisomag = None
    else:
        anisomag = anisomag + torch.eye(2).unsqueeze(0)
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
        "tref1": tref1,
        "tref2": tref2,
    }

    return (
        energy_kev,
        pixel_size,
        alpha,
        rotations,
        translations_A,
        ctf_params,
        scale,
        anisomag,
        split,
    )


def _select_particles(
    mask: torch.Tensor,
    indices: torch.Tensor,
    rotations: torch.Tensor,
    translations_A: torch.Tensor,
    ctf_params: dict[str, torch.Tensor],
    scale: torch.Tensor,
    anisomag: torch.Tensor | None,
    n_particles: int | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor | None,
]:
    """Apply a per-particle boolean mask, then truncate to the first n_particles.

    Returns
    -------
    tuple
        ``(indices, rotations, translations_A, ctf_params, scale, anisomag)``.
    """
    rotations = rotations[mask]
    translations_A = translations_A[mask]
    ctf_params = {k: v[mask] for k, v in ctf_params.items()}
    scale = scale[mask]
    anisomag = None if anisomag is None else anisomag[mask]

    if n_particles is not None:
        indices = indices[:n_particles]
        rotations = rotations[:n_particles]
        translations_A = translations_A[:n_particles]
        ctf_params = {k: v[:n_particles] for k, v in ctf_params.items()}
        scale = scale[:n_particles]
        anisomag = None if anisomag is None else anisomag[:n_particles]

    return indices, rotations, translations_A, ctf_params, scale, anisomag


def extract_parameters_from_csfile(
    csfile_path: str,
    return_class: Literal["0", "1", "all"] = "all",
    rotation_representation: Literal["quaternion", "rotvec"] = "quaternion",
    n_particles: int | None = None,
) -> tuple:
    """
    Extract poses and CTF parameters from CryoSPARC .cs file.

    Parameters
    ----------
    csfile_path : str
        Path of the .cs file.
    return_class : str, optional
        Specifies which particle class to return. Options are '0', '1', or 'all'.
        Default is 'all'.
    rotation_representation : str, optional
        Representation of rotations. 'quaternion' or 'rotvec'. Default is 'quaternion'.
    n_particles : int, optional
        If given, only the first ``n_particles`` particles (after filtering by
        ``return_class``) are returned. Default is None (return all).

    Returns
    -------
    energy_kev : torch.Tensor
        Energy in keV.
    pixel_size : torch.Tensor
        Pixel sizes in Ångstrom.
    alpha : torch.Tensor
        Amplitude contrast ratio.
    rotations : torch.Tensor
        Quaternions with shape (N, 4) or rotation vectors.
    translations_A : torch.Tensor
        xy-translations in Ångstrom with shape (N, 2).
    ctf_params : torch.Tensor
        CTF parameters with shape (N, 7). Parameters are (Cs, dfu, dfv, dfang, tiltx, tilty, phaseshift).
    scale : torch.Tensor
        Per-particle scale factors.
    anisomag : torch.Tensor or None
        Anisotropic magnification matrices (N, 2, 2) or None if identity.
    indices : torch.Tensor
        Indices of the extracted particles from the dataset.
    halfset_labels : torch.Tensor or None
        1-D integer tensor of length ``N`` with values ``0`` or ``1`` from
        ``alignments3D/split``, ready to pass as ``halfset_labels`` to
        :func:`~specter.ghostbuster.run_halfsets`.
        Only returned when ``return_class == "all"``; ``None`` otherwise.
    """
    (
        energy_kev,
        pixel_size,
        alpha,
        rotations,
        translations_A,
        ctf_params,
        scale,
        anisomag,
        split,
    ) = _load_csfile_parameters(csfile_path, rotation_representation)

    if return_class == "all":
        mask = torch.ones_like(split, dtype=torch.bool)
        indices = torch.arange(len(split))
        halfset_labels: torch.Tensor | None = split
    else:  # "0" or "1"
        mask = split == int(return_class)
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
        energy_kev,
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


def create_particle_starfile(
    particles: torch.Tensor,
    rotations: torch.Tensor | np.ndarray | None = None,
    translations: torch.Tensor | np.ndarray | None = None,
    alpha: float = 0.1,
    folderpath: str = "",
    energy: float | None = None,
    dx: float | None = None,
    filename: str = "particles",
    ctf_params: torch.Tensor | None = None,
    dose_per_angstrom: torch.Tensor | float | None = None,
    coincidence_radius: torch.Tensor | float | None = None,
    potential_scale: torch.Tensor | float | None = None,
) -> None:
    """
    Save particle stack as MRCS and create RELION .star file.

    Parameters
    ----------
    particles : torch.Tensor or np.ndarray
        Stack of 2D particle images, shape (N, H, W).
    rotations : torch.Tensor or np.ndarray, optional
        Pose information per particle. Can be quaternions (N,4), matrices (N,3,3),
        or rotation vectors (N,3). Default is None.
    translations : torch.Tensor or np.ndarray, optional
        Translations per particle (x, y) in Ångstrom. Default is None.
    alpha : float, optional
        Amplitude contrast ratio. Default is 0.1.
    folderpath : str, optional
        Directory to save MRCS and STAR files. Default is "" (current directory).
    energy : float, optional
        Electron beam energy in kV. Required.
    dx : float, optional
        Pixel size in Ångstrom. Required.
    filename : str, optional
        Name of the output starfile and mrcfile (without extension). Default is "particles".
    ctf_params : torch.Tensor, optional
        CTF parameters for each particle, shape (N, K).
        Expected columns: [Cs (Å), dfu (Å), dfv (Å), dfang (rad), ..., phaseshift (rad)].
    dose_per_angstrom : torch.Tensor or float, optional
        Dose per particle in e⁻/Å². Saved as ``specterDosePerAngstrom``.
    coincidence_radius : torch.Tensor or float, optional
        Coincidence radius per particle in pixels. Saved as ``specterCoincidenceRadius``.
    potential_scale : torch.Tensor or float, optional
        Per-particle potential scale factor. Saved as ``specterPotentialScale``.

    """

    if folderpath:
        os.makedirs(folderpath, exist_ok=True)

    # create projections mrcs file
    mrcs_path = os.path.join(folderpath, filename + ".mrcs")
    with mrcfile.new(mrcs_path, overwrite=True) as mrc:
        mrc.set_data(particles.numpy().astype(np.float32))

    _console.print(f"  [green]✓[/green] {mrcs_path}")

    # convert rotations to Relion euler
    euler = None
    if rotations is not None:
        if len(rotations.shape) == 3:
            # N x 3 x 3 -> rotation matrices
            R = Rotation.from_matrix(rotations)
        elif rotations.shape[-1] == 4:
            # N x 4 -> quaternions
            R = Rotation.from_quat(rotations)
        elif rotations.shape[-1] == 3:
            # N x 3 -> rotvec
            R = Rotation.from_rotvec(rotations)
        euler = R.as_euler("ZYZ", degrees=True)

    # create associated starfile
    n = len(particles)

    # support both dict and 2-D tensor for ctf_params
    if isinstance(ctf_params, dict):
        zeros = torch.zeros(n)
        cs_A = ctf_params.get("cs", zeros)
        dfu = ctf_params.get("dfu", zeros)
        dfv = ctf_params.get("dfv", dfu)
        dfang = ctf_params.get("dfang", zeros)
        pshift = ctf_params.get("phaseshift", zeros)
    else:
        cs_A = ctf_params[:, 0]
        dfu = ctf_params[:, 1]
        dfv = ctf_params[:, 2]
        dfang = ctf_params[:, 3]
        pshift = ctf_params[:, 5]

    d = {
        "rlnVoltage": energy,
        "rlnSphericalAberration": cs_A / 1e7,
        "rlnAmplitudeContrast": alpha,
        "rlnImagePixelSize": dx,
        "rlnImageName": [str(i) + "@" + filename + ".mrcs" for i in range(1, n + 1)],
        "rlnDefocusU": dfu,
        "rlnDefocusV": dfv,
        "rlnDefocusAngle": dfang,
        "rlnPhaseShift": pshift,
        "rlnOriginXAngst": translations[:, 0],
        "rlnOriginYAngst": translations[:, 1],
    }
    if euler is not None:
        d["rlnAngleRot"] = euler[:, 0]
        d["rlnAngleTilt"] = euler[:, 1]
        d["rlnAnglePsi"] = euler[:, 2]

    if dose_per_angstrom is not None:
        d["specterDosePerAngstrom"] = torch.as_tensor(dose_per_angstrom).expand(n)
    if coincidence_radius is not None:
        d["specterCoincidenceRadius"] = torch.as_tensor(coincidence_radius).expand(n)
    if potential_scale is not None:
        d["specterPotentialScale"] = torch.as_tensor(potential_scale).expand(n)

    particles_df = pd.DataFrame(data=d)

    star_path = os.path.join(folderpath, filename + ".star")
    starfile.write(particles_df, star_path, overwrite=True)
    _console.print(f"  [green]✓[/green] {star_path}")


def create_micrograph_starfile(
    n: int,
    energy: float,
    pixel_size: float,
    alpha: float,
    ctf_params: dict,
    folderpath: str = "",
    filename: str = "micrographs",
    dose_per_angstrom: torch.Tensor | float | None = None,
    coincidence_radius: torch.Tensor | float | None = None,
    potential_scale: torch.Tensor | float | None = None,
    tilt_angles: torch.Tensor | None = None,
) -> str:
    """
    Create a RELION-compatible .star file for a simulated micrograph stack.

    Parameters
    ----------
    n : int
        Number of micrographs.
    energy : float
        Electron beam energy in kV.
    pixel_size : float
        Pixel size in Ångstrom.
    alpha : float
        Amplitude contrast ratio.
    ctf_params : dict
        CTF parameters, each a 1-D tensor of length n. Expected keys:
        ``cs`` (Å), ``dfu`` (Å), and optionally ``dfv`` (Å), ``dfang`` (deg).
    folderpath : str, optional
        Directory to save the STAR file. Default is "" (current directory).
    filename : str, optional
        Base name for the output files (no extension). Default is "micrographs".
    dose_per_angstrom : torch.Tensor or float, optional
        Dose per micrograph in e⁻/Å². Saved as ``specterDosePerAngstrom``.
    coincidence_radius : torch.Tensor or float, optional
        Coincidence radius per micrograph in pixels. Saved as ``specterCoincidenceRadius``.
    potential_scale : torch.Tensor or float, optional
        Potential scale per micrograph. Saved as ``specterPotentialScale``.
    tilt_angles : torch.Tensor, optional
        Tilt angles in degrees for each micrograph. Saved as ``specterTiltAngle``.

    Returns
    -------
    star_path : str
        Path to the saved STAR file.
    """
    if folderpath:
        os.makedirs(folderpath, exist_ok=True)

    zeros = torch.zeros(n)
    cs_A = ctf_params.get("cs", zeros)
    dfu = ctf_params.get("dfu", zeros)
    dfv = ctf_params.get("dfv", dfu)
    dfang = ctf_params.get("dfang", zeros)

    d = {
        "rlnMicrographName": [str(i + 1) + "@" + filename + ".mrcs" for i in range(n)],
        "rlnVoltage": energy,
        "rlnSphericalAberration": torch.as_tensor(cs_A) / 1e7,
        "rlnAmplitudeContrast": alpha,
        "rlnImagePixelSize": pixel_size,
        "rlnDefocusU": dfu,
        "rlnDefocusV": dfv,
        "rlnDefocusAngle": dfang,
    }

    if dose_per_angstrom is not None:
        d["specterDosePerAngstrom"] = torch.as_tensor(dose_per_angstrom).expand(n)
    if coincidence_radius is not None:
        d["specterCoincidenceRadius"] = torch.as_tensor(coincidence_radius).expand(n)
    if potential_scale is not None:
        d["specterPotentialScale"] = torch.as_tensor(potential_scale).expand(n)
    if tilt_angles is not None:
        d["specterTiltAngle"] = torch.as_tensor(tilt_angles)

    df = pd.DataFrame(data=d)

    star_path = os.path.join(folderpath, filename + ".star")
    starfile.write(df, star_path, overwrite=True)
    _console.print(f"  [green]✓[/green] {star_path}")
    return star_path
