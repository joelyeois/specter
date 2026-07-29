from __future__ import annotations

import os
from typing import Literal

import mrcfile
import numpy as np
import pandas as pd
import roma
import starfile
import torch
from rich.console import Console

from ..imagegenerator import ParticleGeneratorBase
from ._common import _select_particles

_console = Console()

_REQUIRED_STAR_COLUMNS = (
    "rlnVoltage",
    "rlnSphericalAberration",
    "rlnAmplitudeContrast",
    "rlnImagePixelSize",
    "rlnDefocusU",
)


def _merge_optics(particles: pd.DataFrame, optics: pd.DataFrame) -> pd.DataFrame:
    """Merge a RELION 3.1+ ``optics`` block into the ``particles`` block."""
    particles = particles.copy()
    if "rlnOpticsGroup" in particles.columns and "rlnOpticsGroup" in optics.columns:
        shared = [
            c
            for c in optics.columns
            if c in particles.columns and c != "rlnOpticsGroup"
        ]
        return particles.merge(
            optics.drop(columns=shared), on="rlnOpticsGroup", how="left"
        )
    if len(optics) > 1:
        raise ValueError(
            "STAR file has multiple optics groups but the particles table has "
            "no 'rlnOpticsGroup' column to match them against."
        )
    for col in optics.columns:
        if col not in particles.columns:
            particles[col] = optics[col].iloc[0]
    return particles


def _load_starfile_parameters(
    starfile_path: str,
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
    torch.Tensor | None,
]:
    """Load and derive all per-particle imaging parameters from a RELION STAR file, unfiltered.

    Returns
    -------
    tuple
        ``(energy_kev, pixel_size, alpha, rotations, translations_A, ctf_params,
        scale, anisomag, split)`` for every particle in the file. ``anisomag``
        is always ``None`` (anisotropic magnification is not currently parsed
        from STAR files). ``split`` is ``None`` if the file has no
        ``rlnRandomSubset`` column.
    """
    data = starfile.read(starfile_path)
    if isinstance(data, dict):
        if "particles" not in data or "optics" not in data:
            raise KeyError(
                "Multi-block STAR file must contain 'optics' and 'particles' blocks."
            )
        particles = _merge_optics(data["particles"], data["optics"])
    else:
        particles = data

    missing = [c for c in _REQUIRED_STAR_COLUMNS if c not in particles.columns]
    if missing:
        raise KeyError(f"STAR file is missing required column(s): {', '.join(missing)}")

    n = len(particles)

    def col(name: str, default: float = 0.0) -> torch.Tensor:
        if name in particles.columns:
            return torch.as_tensor(
                particles[name].to_numpy(dtype=np.float64), dtype=torch.float32
            )
        return torch.full((n,), float(default))

    def scalar_col(name: str) -> torch.Tensor:
        values = col(name)
        if torch.allclose(values[0], values.mean()):
            return values[0]
        _console.print(
            f"[yellow]Warning:[/yellow] {name} is not the same for all particles."
        )
        return values

    energy_kev = scalar_col("rlnVoltage")
    pixel_size = scalar_col("rlnImagePixelSize")
    alpha = scalar_col("rlnAmplitudeContrast")

    cs_A = col("rlnSphericalAberration") * 1e7
    dfu_A = col("rlnDefocusU")
    dfv_A = col("rlnDefocusV") if "rlnDefocusV" in particles.columns else dfu_A
    dfang_deg = col("rlnDefocusAngle")
    # RELION stores phase shift in degrees, unlike specter's internal radians
    phaseshift_rad = col("rlnPhaseShift") * torch.pi / 180

    ctf_params = {
        "cs": cs_A,
        "dfu": dfu_A,
        "dfv": dfv_A,
        "dfang": dfang_deg,
        "phaseshift": phaseshift_rad,
    }
    if "rlnBeamTiltX" in particles.columns or "rlnBeamTiltY" in particles.columns:
        ctf_params["tiltx"] = col("rlnBeamTiltX") * 1e-3
        ctf_params["tilty"] = col("rlnBeamTiltY") * 1e-3

    # extract rotations (RELION ZYZ Euler convention: rot, tilt, psi)
    if {"rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"} <= set(particles.columns):
        euler_deg = torch.stack(
            [col("rlnAngleRot"), col("rlnAngleTilt"), col("rlnAnglePsi")], dim=-1
        )
        if rotation_representation == "quaternion":
            rotations = roma.euler_to_unitquat("ZYZ", euler_deg, degrees=True)
        else:
            rotations = roma.euler_to_rotvec("ZYZ", euler_deg, degrees=True)
    else:
        rotations = (
            torch.tensor([[0.0, 0.0, 0.0, 1.0]] * n)
            if rotation_representation == "quaternion"
            else torch.zeros(n, 3)
        )

    # extract translations
    if {"rlnOriginXAngst", "rlnOriginYAngst"} <= set(particles.columns):
        translations_A = torch.stack(
            [col("rlnOriginXAngst"), col("rlnOriginYAngst")], dim=-1
        )
    elif {"rlnOriginX", "rlnOriginY"} <= set(particles.columns):
        translations_px = torch.stack([col("rlnOriginX"), col("rlnOriginY")], dim=-1)
        translations_A = translations_px * pixel_size
    else:
        translations_A = torch.zeros(n, 2)

    # extract per-particle CTF scale factor
    scale = col("rlnCtfScalefactor", default=1.0)

    # anisotropic magnification is not currently parsed from STAR files
    anisomag: torch.Tensor | None = None

    # extract half-set label (raw RELION values: 1 or 2)
    split = (
        col("rlnRandomSubset").to(torch.int64)
        if "rlnRandomSubset" in particles.columns
        else None
    )

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


def extract_parameters_from_starfile(
    starfile_path: str,
    return_class: Literal["1", "2", "all"] = "all",
    rotation_representation: Literal["quaternion", "rotvec"] = "quaternion",
    n_particles: int | None = None,
) -> tuple:
    """
    Extract poses and CTF parameters from a RELION-format STAR file.

    Equivalent to :func:`~specter.io.extract_parameters_from_csfile`, but for
    RELION ``.star`` files instead of CryoSPARC ``.cs`` files. Supports both
    the single-block layout written by :func:`create_particle_starfile` and
    the two-block (``optics`` + ``particles``) layout used by RELION 3.1+.

    Parameters
    ----------
    starfile_path : str
        Path of the .star file.
    return_class : str, optional
        Which RELION half-set to return, matching the raw values of
        ``rlnRandomSubset`` ('1' or '2'), or 'all'. Default is 'all'.
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
        Pixel size in Ångstrom.
    alpha : torch.Tensor
        Amplitude contrast ratio.
    rotations : torch.Tensor
        Quaternions with shape (N, 4) or rotation vectors.
    translations_A : torch.Tensor
        xy-translations in Ångstrom with shape (N, 2).
    ctf_params : dict[str, torch.Tensor]
        CTF parameters keyed by name (``cs``, ``dfu``, ``dfv``, ``dfang``,
        ``phaseshift``, plus ``tiltx``/``tilty`` if present in the file).
    scale : torch.Tensor
        Per-particle CTF scale factor (``rlnCtfScalefactor``), or ones if absent.
    anisomag : None
        Always ``None`` — anisotropic magnification is not currently parsed
        from STAR files.
    indices : torch.Tensor
        Indices of the extracted particles from the file.
    halfset_labels : torch.Tensor or None
        1-D integer tensor of length ``N`` with the raw ``rlnRandomSubset``
        values (1 or 2). Only returned when ``return_class == "all"``;
        ``None`` otherwise.
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
    ) = _load_starfile_parameters(starfile_path, rotation_representation)

    n_total = len(rotations)

    if return_class == "all":
        mask = torch.ones(n_total, dtype=torch.bool)
        indices = torch.arange(n_total)
        halfset_labels: torch.Tensor | None = split
    else:
        if split is None:
            raise ValueError(
                "STAR file has no 'rlnRandomSubset' column; cannot filter by return_class."
            )
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
    ctf_params: torch.Tensor | dict[str, torch.Tensor] | None = None,
    dose_per_angstrom: torch.Tensor | float | None = None,
    coincidence_radius: torch.Tensor | float | None = None,
    potential_scale: torch.Tensor | float | None = None,
    bfactor: torch.Tensor | float | None = None,
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
    ctf_params : torch.Tensor or dict[str, torch.Tensor], optional
        CTF parameters for each particle. Either a tensor of shape (N, K)
        with columns [Cs (Å), dfu (Å), dfv (Å), dfang (rad), ..., phaseshift
        (rad)], or a dict keyed by parameter name (e.g. as returned by
        :meth:`~specter.imagegenerator.BaseImager.ctf_params_dict`).
    dose_per_angstrom : torch.Tensor or float, optional
        Dose per particle in e⁻/Å². Saved as ``specterDosePerAngstrom``.
    coincidence_radius : torch.Tensor or float, optional
        Coincidence radius per particle in pixels. Saved as ``specterCoincidenceRadius``.
    potential_scale : torch.Tensor or float, optional
        Per-particle potential scale factor. Saved as ``specterPotentialScale``.
    bfactor : torch.Tensor or float, optional
        Per-particle B-factor. Saved as ``specterBfactor``.

    Notes
    -----
    ``dose_per_angstrom``, ``coincidence_radius``, ``potential_scale``, and
    ``bfactor`` are only recorded in the STAR file if explicitly passed here
    — they are not inferred from anywhere else. To avoid having to re-enter
    values already given to an :class:`~specter.imagegenerator.ImageGenerator`
    (and risking the STAR file silently not matching what was actually
    simulated), prefer :func:`create_particle_starfile_from_model`, which
    reads all of these directly from the model that generated ``particles``.
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
        rotations_t = torch.as_tensor(rotations, dtype=torch.float32)
        if len(rotations_t.shape) == 3:
            # N x 3 x 3 -> rotation matrices (re-orthonormalized, matching
            # scipy's from_matrix Markley-SVD correction for inexact input)
            R = roma.special_procrustes(rotations_t)
        elif rotations_t.shape[-1] == 4:
            # N x 4 -> quaternions
            R = roma.unitquat_to_rotmat(rotations_t)
        elif rotations_t.shape[-1] == 3:
            # N x 3 -> rotvec
            R = roma.rotvec_to_rotmat(rotations_t)
        euler = roma.rotmat_to_euler("ZYZ", R, degrees=True).numpy()

    # create associated starfile
    n = len(particles)
    translations = (
        torch.zeros(n, 2) if translations is None else torch.as_tensor(translations)
    )

    # support both dict and 2-D tensor for ctf_params (None means "all zeros")
    if ctf_params is None or isinstance(ctf_params, dict):
        zeros = torch.zeros(n)
        ctf_params = ctf_params or {}
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
    if bfactor is not None:
        d["specterBfactor"] = torch.as_tensor(bfactor).expand(n)

    particles_df = pd.DataFrame(data=d)

    star_path = os.path.join(folderpath, filename + ".star")
    starfile.write(particles_df, star_path, overwrite=True)
    _console.print(f"  [green]✓[/green] {star_path}")


def create_particle_starfile_from_model(
    particles: torch.Tensor,
    model: ParticleGeneratorBase,
    folderpath: str = "",
    filename: str = "particles",
) -> None:
    """
    Save particle stack as MRCS and create RELION .star file, reading pose,
    CTF, and physics parameters (dose, coincidence radius, potential scale,
    bfactor) directly from the model that generated ``particles``.

    Use this instead of :func:`create_particle_starfile` to avoid re-typing
    values already given to the model — and having the STAR file silently
    fall out of sync with what was actually simulated if those values are
    ever changed without updating the save call to match.

    Parameters
    ----------
    particles : torch.Tensor
        Stack of 2D particle images generated by ``model``, shape (N, H, W).
        N must match the number of poses ``model`` was constructed with.
    model : ParticleGeneratorBase
        The (already-constructed) particle generator used to produce
        ``particles``, e.g. an :class:`~specter.imagegenerator.ImageGenerator`
        or :class:`~specter.imagegenerator.ImageGeneratorFromCoordinates`.
    folderpath : str, optional
        Directory to save MRCS and STAR files. Default is "" (current directory).
    filename : str, optional
        Name of the output starfile and mrcfile (without extension). Default is "particles".
    """
    create_particle_starfile(
        particles,
        rotations=model.quaternions,
        translations=model.translations,
        alpha=model.alpha,
        folderpath=folderpath,
        energy=model.energy,
        dx=model.pixel_size,
        filename=filename,
        ctf_params=model.ctf_params_dict(),
        dose_per_angstrom=model.dose_per_angstrom,
        coincidence_radius=model.coincidence_radius,
        potential_scale=model.potential_scale,
        bfactor=getattr(model, "bfactor", None),
    )


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
    bfactor: torch.Tensor | float | None = None,
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
    bfactor : torch.Tensor or float, optional
        B-factor per micrograph. Saved as ``specterBfactor``.
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
    if bfactor is not None:
        d["specterBfactor"] = torch.as_tensor(bfactor).expand(n)
    if tilt_angles is not None:
        d["specterTiltAngle"] = torch.as_tensor(tilt_angles)

    df = pd.DataFrame(data=d)

    star_path = os.path.join(folderpath, filename + ".star")
    starfile.write(df, star_path, overwrite=True)
    _console.print(f"  [green]✓[/green] {star_path}")
    return star_path
