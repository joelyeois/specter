"""
Reads imaging parameters, poses and CTF terms from a CryoSPARC ``.cs``
file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import roma
import torch
from cryosparc.dataset import Dataset
from rich.console import Console

from .. import logger
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
        ``(voltage_kv, pixel_size, alpha, rotations, translations_angstrom, ctf_params,
        scale, anisomag, split)`` for every particle in the dataset.
    """
    dataset = Dataset.load(csfile_path)

    # extract translations. `alignments3D/shift` is in pixels of the grid the
    # ALIGNMENT was done on, so it must be scaled by the alignment's own pixel
    # size -- which is not necessarily the images'.
    translations_px = torch.as_tensor(dataset["alignments3D/shift"])
    alignment_psize = torch.as_tensor(dataset["alignments3D/psize_A"])
    translations_angstrom = translations_px * alignment_psize[..., None]

    # The pixel size to RENDER at is the images', `blob/psize_A`. These differ
    # whenever the refinement was run binned: a 2x-binned refinement records
    # alignments3D/psize_A = 2 * blob/psize_A, and taking the alignment value
    # renders the specimen at half the sampling into the same box -- a field of
    # view twice too wide, silently. Measured on EMPIAR-11377's J205
    # (alignments3D 1.462, images 0.731): every pair-by-pair pose correlation
    # collapsed to the shuffled level and the background noise floor came out
    # at 0.62 of the experiment's instead of 2.15.
    pixel_size = alignment_psize
    try:
        # Both a real `Dataset` and a plain mapping raise KeyError for a
        # missing column, so ask rather than sniffing for a `fields` method.
        image_psize: torch.Tensor | None = torch.as_tensor(dataset["blob/psize_A"])
    except KeyError:
        image_psize = None
    if image_psize is not None:
        if not torch.allclose(image_psize, alignment_psize):
            _console.print(
                f"[yellow]Warning:[/yellow] {csfile_path}: the images are at "
                f"{float(image_psize.flatten()[0]):.4f} A/px "
                f"(blob/psize_A) but the alignment was done at "
                f"{float(alignment_psize.flatten()[0]):.4f} A/px "
                "(alignments3D/psize_A), i.e. a binned refinement. Rendering at "
                "the images' pixel size; shifts are still converted with the "
                "alignment's."
            )
        pixel_size = image_psize
    else:
        # No `blob/psize_A` means this file cannot say what its images are
        # sampled at, and the alignment value is the only thing available --
        # which is exactly the case that bites, because a passthrough with no
        # blob columns is what a binned refinement emits. Say so rather than
        # proceeding silently; the caller can check the stack's own header.
        _console.print(
            f"[yellow]Warning:[/yellow] {csfile_path}: no blob/psize_A, so the "
            f"pixel size is taken from alignments3D/psize_A "
            f"({float(alignment_psize.flatten()[0]):.4f} A/px). If the "
            "refinement was binned this is NOT the pixel size of the images; "
            "check it against the particle stack before rendering."
        )

    if torch.allclose(pixel_size[0], pixel_size.mean()):
        pixel_size = pixel_size[0]
    else:
        _console.print(
            "[yellow]Warning:[/yellow] pixel size is not the same for all particles."
        )

    # extract spherical aberration
    cs_mm = torch.as_tensor(dataset["ctf/cs_mm"])
    cs_angstrom = cs_mm * 1e7

    # extract defocus
    dfang_rad = torch.as_tensor(dataset["ctf/df_angle_rad"])
    dfang_deg = dfang_rad / torch.pi * 180
    dfu_angstrom = torch.as_tensor(dataset["ctf/df1_A"])
    dfv_angstrom = torch.as_tensor(dataset["ctf/df2_A"])

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
    wavelength_angstrom = energy_to_wavelength(voltage_kv)

    # extract rotations
    pose = torch.as_tensor(dataset["alignments3D/pose"], dtype=torch.float32)
    if rotation_representation == "quaternion":
        rotations = roma.rotvec_to_unitquat(pose)
    elif rotation_representation == "rotvec":
        rotations = pose

    # extract split
    split = torch.as_tensor(dataset["alignments3D/split"].astype(int))

    # extract beamtilt
    beamtiltx_rad = torch.arcsin(
        torch.as_tensor(dataset["ctf/tilt_A"][:, 0] / cs_angstrom)
    )
    beamtilty_rad = torch.arcsin(
        torch.as_tensor(dataset["ctf/tilt_A"][:, 1] / cs_angstrom)
    )

    # extract phaseshift
    phaseshift_rad = torch.as_tensor(dataset["ctf/phase_shift_rad"])

    # extract ctf shift, and add to translations
    beamshift_angstrom = torch.as_tensor(dataset["ctf/shift_A"])
    translations_angstrom -= beamshift_angstrom

    # extract trefoil. CryoSPARC's ctf/trefoil_A is a raw Å-scale
    # coefficient; specter's trefoil1/trefoil2 (see
    # aberrations._functions.trefoil) are the direct k^3-domain phase
    # coefficients in Å³, related by chi_trefoil =
    # (2*pi/3)*wavelength^2*trefoil_angstrom -- see newctf.py's
    # params_to_coeffs_odd/gen_basis_odd for the CryoSPARC-side derivation.
    trefoil_angstrom = torch.as_tensor(dataset["ctf/trefoil_A"])
    trefoil1 = (2 * torch.pi / 3) * wavelength_angstrom**2 * trefoil_angstrom[:, 0]
    trefoil2 = (2 * torch.pi / 3) * wavelength_angstrom**2 * trefoil_angstrom[:, 1]

    # extract tetrafoil. CryoSPARC's ctf/tetra_A holds 4 raw Å-scale
    # coefficients spanning secondary astigmatism (n=4, m=+-2) and true
    # tetrafoil (n=4, m=+-4); specter's tetrafoil1-4 (see
    # aberrations._functions.tetrafoil) are the direct k^4-domain phase
    # coefficients in Å⁴. See newctf.py's
    # params_to_coeffs_even/gen_basis_even for the CryoSPARC-side
    # derivation of these prefactors (including the sign/index mapping).
    tetra_angstrom = torch.as_tensor(dataset["ctf/tetra_A"])
    tetrafoil1 = -2 * torch.pi * wavelength_angstrom**3 * tetra_angstrom[:, 0]
    tetrafoil2 = 2 * torch.pi * wavelength_angstrom**3 * tetra_angstrom[:, 1]
    tetrafoil3 = (torch.pi / 2) * wavelength_angstrom**3 * tetra_angstrom[:, 2]
    tetrafoil4 = -(torch.pi / 2) * wavelength_angstrom**3 * tetra_angstrom[:, 3]

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
        corrected_shifts = translations_angstrom.unsqueeze(
            -1
        )  # Add a dimension to make it (B, 2, 1)

        # Perform batch matrix multiplication
        corrected_shifts = torch.bmm(anisomag, corrected_shifts)

        # Remove the last dimension to get (B, 2)
        corrected_shifts = corrected_shifts.squeeze(-1)
        translations_angstrom = corrected_shifts

    ctf_params = {
        "cs": cs_angstrom,
        "dfu": dfu_angstrom,
        "dfv": dfv_angstrom,
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
        translations_angstrom,
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
    translations_angstrom : torch.Tensor
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
        translations_angstrom,
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

    indices, rotations, translations_angstrom, ctf_params, scale, anisomag = (
        _select_particles(
            mask,
            indices,
            rotations,
            translations_angstrom,
            ctf_params,
            scale,
            anisomag,
            n_particles,
        )
    )
    if halfset_labels is not None and n_particles is not None:
        halfset_labels = halfset_labels[:n_particles]

    return (
        voltage_kv,
        pixel_size,
        alpha,
        rotations,
        translations_angstrom,
        ctf_params,
        scale,
        anisomag,
        indices,
        halfset_labels,
    )


def _blob_columns(dataset: object) -> tuple[np.ndarray, np.ndarray] | None:
    """``(blob/path, blob/idx)`` of a loaded dataset, or None if it carries no images."""
    if "blob/idx" not in dataset or "blob/path" not in dataset:  # type: ignore[operator]
        return None
    return (
        np.asarray(dataset["blob/path"]).astype(str),  # type: ignore[index]
        np.asarray(dataset["blob/idx"]).astype(np.int64),  # type: ignore[index]
    )


def _row_order(uid: np.ndarray, other_uid: np.ndarray) -> np.ndarray | None:
    """Index array reordering ``other_uid``'s rows onto ``uid``'s, or None if they differ.

    Sibling files of one particle group carry the same particles, but nothing
    guarantees the same row order, so they are matched on ``uid`` rather than
    position. Returning None on any mismatch is what keeps an unrelated ``.cs``
    sitting in the same directory from being paired in silently.
    """
    if len(uid) != len(other_uid):
        return None
    ascending = np.argsort(other_uid)
    position = np.searchsorted(other_uid[ascending], uid)
    if np.any(position >= len(ascending)):
        return None
    order = ascending[position]
    return order if np.array_equal(other_uid[order], uid) else None


def particle_stack_references(
    csfile_path: str | Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Per-row ``(blob/path, blob/idx)`` for the particles a ``.cs`` file describes.

    A CryoSPARC particle image is addressed by ``blob/path`` and ``blob/idx``,
    never by its row number. A restack job writes its output stack in the order
    it read its inputs, which is not the order of the rows it emits: on a
    1000-particle restack of the CryoSPARC tutorial set, row ``i`` refers to
    slice ``(i + 370) % 1000``. Reading the stack sequentially instead pairs
    every pose with a different particle's image, and no rotationally averaged
    statistic shows it -- only a matched-index correlation does
    (:func:`specter.match.matched_index_correlation`).

    Poses and blobs routinely live in different files of the same particle
    group: a restack job puts the alignments in ``*_passthrough_particles.cs``
    and the images in ``restacked_particles.cs``. When ``csfile_path`` carries
    no blob columns of its own, sibling ``.cs`` files in the same directory are
    searched for one holding the same particles, matched on ``uid``.

    Parameters
    ----------
    csfile_path : str or Path
        Path of the ``.cs`` file.

    Returns
    -------
    tuple of numpy.ndarray, or None
        ``(paths, indices)`` in the file's own row order, where ``paths`` are
        the project-relative stack paths and ``indices`` the slice of each
        stack. None when neither this file nor any sibling carries images.
    """
    csfile_path = Path(csfile_path)
    dataset = Dataset.load(str(csfile_path))
    blobs = _blob_columns(dataset)
    if blobs is not None:
        return blobs
    if "uid" not in dataset:
        return None

    uid = np.asarray(dataset["uid"])
    for sibling in sorted(csfile_path.parent.glob("*.cs")):
        if sibling.resolve() == csfile_path.resolve():
            continue
        try:
            other = Dataset.load(str(sibling))
        except Exception:  # noqa: BLE001 - an unreadable neighbour is not our problem
            continue
        blobs = _blob_columns(other)
        if blobs is None or "uid" not in other:
            continue
        order = _row_order(uid, np.asarray(other["uid"]))
        if order is None:
            continue
        logger.info(
            "particle images addressed by %s (blob/path, blob/idx)", sibling.name
        )
        return blobs[0][order], blobs[1][order]
    return None
