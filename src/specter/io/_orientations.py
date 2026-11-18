"""
Reads a viewing-direction distribution: the orientations alone from a
CryoSPARC ``.cs`` or RELION ``.star`` file, a binned histogram, or the
``.bild`` arrows a CryoSPARC refinement draws the distribution with.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import roma
import starfile
import torch
from cryosparc.dataset import Dataset

from ..rotations import sample_viewing_direction_histogram

_STAR_ANGLES = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
_HISTOGRAM_SUFFIXES = (".npy", ".csv", ".txt")
# A histogram or a BILD file is turned into this many poses, drawn from a
# fixed seed so that the result depends on the file alone.
_HISTOGRAM_DRAWS = 65536


def _read_bild_arrows(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The viewing directions and counts drawn by a BILD file's arrows.

    CryoSPARC draws each populated direction bin as a ``.cylinder`` that
    starts on a sphere about the map centre and points radially outward,
    with a length proportional to the bin's particle count. The direction
    is therefore the cylinder's own axis, and its length a relative count.
    Every other BILD command (``.color``, comments) is ignored.
    """
    cylinders = []
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if fields and fields[0] == ".cylinder":
                cylinders.append([float(v) for v in fields[1:7]])
    if not cylinders:
        raise ValueError(f"{path}: no .cylinder arrows to read a distribution from.")
    ends = torch.tensor(cylinders, dtype=torch.float64)
    axes = ends[:, 3:] - ends[:, :3]
    lengths = axes.norm(dim=-1)
    if (lengths <= 0).any():
        raise ValueError(f"{path}: an arrow has zero length.")
    return axes / lengths[:, None], lengths


def _sample_arrows(
    directions: torch.Tensor,
    weights: torch.Tensor,
    n: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    `n` orientations whose viewing directions are drawn from weighted arrows.

    An arrow stands for a bin, so each draw is spread uniformly over a
    spherical cap about its arrow, of radius half the median spacing between
    neighbouring arrows. The in-plane angle is uniform.
    """
    rows = torch.multinomial(weights, n, replacement=True, generator=generator)
    if len(directions) > 1:
        cosines = (directions @ directions.T).fill_diagonal_(-1.0)
        spacing = torch.arccos(cosines.max(dim=-1).values.clamp(-1.0, 1.0))
        cap = 0.5 * float(spacing.median())
    else:
        cap = 0.0
    u = torch.rand(n, 3, generator=generator, dtype=torch.float64)
    # A direction uniform over the cap, in a frame whose z is the arrow.
    cos_tilt = 1.0 - u[:, 0] * (1.0 - np.cos(cap))
    sin_tilt = torch.sqrt((1.0 - cos_tilt**2).clamp(min=0.0))
    turn = 2 * torch.pi * u[:, 1]
    centre = directions[rows]
    helper = torch.where(
        centre[:, 2:].abs() < 0.9,
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )
    e1 = torch.linalg.cross(helper, centre)
    e1 = e1 / e1.norm(dim=-1, keepdim=True)
    e2 = torch.linalg.cross(centre, e1)
    d = cos_tilt[:, None] * centre + sin_tilt[:, None] * (
        torch.cos(turn)[:, None] * e1 + torch.sin(turn)[:, None] * e2
    )
    # Rz(azimuth) Ry(polar) carries z onto d; the last angle spins about the beam.
    angles = torch.stack(
        [
            torch.atan2(d[:, 1], d[:, 0]),
            torch.arccos(d[:, 2].clamp(-1.0, 1.0)),
            2 * torch.pi * u[:, 2],
        ],
        dim=-1,
    )
    return roma.euler_to_unitquat("ZYZ", angles.float())


def read_orientations(path: str | Path) -> torch.Tensor:
    """
    Read a set of orientations that samples a viewing-direction distribution.

    For a refinement's particle file these are its particles' orientations.
    Only the pose columns are required, so a file that lacks the CTF or
    image columns :func:`extract_parameters_from_csfile` and
    :func:`extract_parameters_from_starfile` need can still be read. The
    quaternions are the ones those two readers return.

    A histogram needs no particle file. It is a table of particle counts
    binned by the viewing direction's elevation (rows, from -90 to 90
    degrees) and azimuth (columns, from -180 to 180 degrees), as
    :func:`specter.rotations.sample_viewing_direction_histogram` defines
    them. It is converted to 65536 orientations drawn from it with a fixed
    seed, so the same file always yields the same set.

    A ``.bild`` file is the 3D drawing of the distribution that a CryoSPARC
    refinement saves beside its viewing direction distribution plot, and
    needs no particle file either. Each arrow gives a populated direction
    and, through its length, the relative particle count there. It is
    converted to 65536 orientations the same way, each spread over its
    arrow's bin.

    Parameters
    ----------
    path : str or Path
        A CryoSPARC ``.cs`` file with an ``alignments3D/pose`` column, a
        RELION ``.star`` file with ``rlnAngleRot``, ``rlnAngleTilt`` and
        ``rlnAnglePsi`` columns, a histogram as a 2D ``.npy`` array or a
        ``.csv``/``.txt`` table, or a CryoSPARC viewing-direction ``.bild``
        file. The format is taken from the extension.

    Returns
    -------
    torch.Tensor
        Unit quaternions, shape (n, 4), in roma's ``xyzw`` convention.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".cs":
        dataset = Dataset.load(str(path))
        try:
            pose = dataset["alignments3D/pose"]
        except KeyError:
            raise KeyError(
                f"{path}: no alignments3D/pose column; orientations come from "
                "a 3D refinement's particle output."
            ) from None
        quaternions = roma.rotvec_to_unitquat(
            torch.as_tensor(pose, dtype=torch.float32)
        )
    elif suffix == ".star":
        data = starfile.read(str(path))
        if isinstance(data, dict):
            if "particles" not in data:
                raise KeyError(f"{path}: no 'particles' block.")
            data = data["particles"]
        missing = [c for c in _STAR_ANGLES if c not in data.columns]
        if missing:
            raise KeyError(f"{path}: missing column(s) {', '.join(missing)}.")
        euler_deg = torch.as_tensor(
            data[list(_STAR_ANGLES)].to_numpy(dtype=np.float64), dtype=torch.float32
        )
        quaternions = roma.euler_to_unitquat("ZYZ", euler_deg, degrees=True)
    elif suffix in _HISTOGRAM_SUFFIXES:
        if suffix == ".npy":
            table = np.load(str(path), allow_pickle=False)
        else:
            with open(path) as handle:
                delimiter = "," if "," in handle.readline() else None
            table = np.loadtxt(str(path), delimiter=delimiter, ndmin=2)
        try:
            quaternions = sample_viewing_direction_histogram(
                torch.as_tensor(table),
                _HISTOGRAM_DRAWS,
                generator=torch.Generator().manual_seed(0),
            )
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from None
    elif suffix == ".bild":
        directions, weights = _read_bild_arrows(path)
        quaternions = _sample_arrows(
            directions, weights, _HISTOGRAM_DRAWS, torch.Generator().manual_seed(0)
        )
    else:
        raise ValueError(
            f"{path}: orientations are read from a .cs or .star file, a "
            ".npy/.csv/.txt histogram or a .bild file, "
            f"not {suffix or 'a file with no extension'!r}."
        )
    if len(quaternions) == 0:
        raise ValueError(f"{path}: the file holds no particles.")
    return quaternions
