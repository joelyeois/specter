from __future__ import annotations

import warnings
from typing import Sequence

import ase.io as ase_io
import numpy as np
from ase import Atoms
import pandas as pd
import torch

from ..arrays import (
    radial_profile_3d,
    soft_voxelize_coordinates,
    tile_volume_from_blocks_blended,
)
from ..coords import radial_distribution_function
from ..fft import fft3


class MDSimDump:
    """
    Lazy reader and analyser for LAMMPS dump files from amorphous ice MD simulations.

    Indexes frame start positions on construction, then parses and trims
    coordinates on demand. The box geometric centre (from the dump header's
    box-bound lines) is used to centre each frame before trimming to a cubic
    region of side ``trim_size``.

    Parameters
    ----------
    filepath : str
        Path to the LAMMPS dump file.
    n : int, optional
        Number of voxels per axis for soft-voxelization. Default 200.
    dx : float, optional
        Voxel size in Å. Default 0.5.
    trim_size : float, optional
        Side length in Å of the cubic region extracted around the box centre.
        Default 100.0.

    Attributes
    ----------
    n_frames : int
        Total number of timestep frames found in the dump.
    coordinates : list of torch.Tensor or None
        Set by :meth:`get_coordinates`; one tensor per requested frame,
        each of shape ``(N_i, 3)`` in Å (N_i may vary between frames as
        atoms drift across the trim boundary).
    """

    # LAMMPS dump header is exactly 9 lines per frame:
    #   0: ITEM: TIMESTEP
    #   1: <timestep>
    #   2: ITEM: NUMBER OF ATOMS
    #   3: <n_atoms>
    #   4: ITEM: BOX BOUNDS ...
    #   5: xlo xhi
    #   6: ylo yhi
    #   7: zlo zhi
    #   8: ITEM: ATOMS id [type] x y z [...]
    #   9+: atom data
    # The column set after "id" varies by dump (some include a "type" column,
    # some carry extra per-atom values like "c_1"), so the x/y/z offset is
    # parsed from this header rather than assumed fixed.
    _HEADER_LINES = 9

    def __init__(
        self,
        filepath: str,
        n: int = 200,
        dx: float = 0.5,
        trim_size: float = 100.0,
    ) -> None:
        self.n = n
        self.dx = dx
        self.trim_size = trim_size

        fov = n * dx
        if fov > trim_size:
            warnings.warn(
                f"Field of view {fov:.1f} Å (n={n}, dx={dx} Å) exceeds "
                f"trim_size={trim_size:.1f} Å. Voxelized volumes will contain "
                "zero-padded regions outside the trimmed coordinate cube.",
                stacklevel=2,
            )

        with open(filepath) as f:
            self._lines = f.readlines()

        self._frame_starts: list[int] = [
            i for i, ln in enumerate(self._lines) if ln == "ITEM: TIMESTEP\n"
        ]
        self.n_frames: int = len(self._frame_starts)
        self._box_center: torch.Tensor = self._parse_box_center(0)
        self._coord_start: int = self._parse_coord_column_start(0)
        self.coordinates: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_coord_column_start(self, frame_idx: int) -> int:
        """Return the index of the ``x`` column within a per-atom data row.

        Parsed from the ``ITEM: ATOMS ...`` header (e.g. ``id type x y z``
        or ``id x y z c_1``), assuming ``x y z`` appear contiguously and in
        order. Assumed constant across all frames of the dump.
        """
        fs = self._frame_starts[frame_idx]
        header = self._lines[fs + 8]
        names = header.split()[2:]  # drop "ITEM:" "ATOMS"
        try:
            x_idx = names.index("x")
        except ValueError as exc:
            raise ValueError(
                f"Could not find an 'x' column in dump ATOMS header: {header!r}"
            ) from exc
        if names[x_idx : x_idx + 3] != ["x", "y", "z"]:
            raise ValueError(
                f"Expected contiguous 'x y z' columns in dump ATOMS header: {header!r}"
            )
        return x_idx

    def _parse_box_center(self, frame_idx: int) -> torch.Tensor:
        """Return (cx, cy, cz) from the box-bound lines of one frame.

        Handles both orthogonal (``lo hi``) and triclinic (``lo hi tilt``)
        LAMMPS dump formats by using only the first two tokens per line.
        """
        fs = self._frame_starts[frame_idx]
        center = []
        for offset in (5, 6, 7):
            parts = self._lines[fs + offset].split()
            lo, hi = float(parts[0]), float(parts[1])
            center.append((lo + hi) / 2.0)
        return torch.tensor(center, dtype=torch.float32)

    def _trim_frame(self, frame_idx: int) -> torch.Tensor:
        """
        Parse, centre, and trim one frame's coordinates.

        Reads the atom lines for ``frame_idx``, subtracts the box centre
        (parsed from the dump header), then keeps only atoms within
        ±trim_size/2 along every axis.

        Parameters
        ----------
        frame_idx : int
            Index into the frame list (0-based).

        Returns
        -------
        coords : torch.Tensor
            Trimmed coordinates, shape ``(N_trimmed, 3)``, in Å relative to
            the box centre.
        """
        fs = self._frame_starts[frame_idx]
        n_atoms = int(self._lines[fs + 3])
        coord_start = fs + self._HEADER_LINES

        lines_block = self._lines[coord_start : coord_start + n_atoms]
        s = self._coord_start
        coords_np = np.array(
            [ln.split()[s : s + 3] for ln in lines_block], dtype=np.float32
        )
        coords = torch.from_numpy(np.ascontiguousarray(coords_np))
        coords -= self._box_center

        half = self.trim_size / 2.0
        mask = (
            (coords[:, 0].abs() <= half)
            & (coords[:, 1].abs() <= half)
            & (coords[:, 2].abs() <= half)
        )
        return coords[mask]

    def _resolve_frames(
        self,
        frames: int | Sequence[int] | torch.Tensor | None,
    ) -> list[int]:
        """
        Resolve the ``frames`` argument to a sorted list of frame indices.

        ``None`` returns all frames from index 10 onwards (frames 0–9 are
        pre-steady-state). A warning is raised if any index < 10 is requested.
        """
        if frames is None:
            return list(range(10, self.n_frames))

        if isinstance(frames, int):
            frame_list = [frames]
        elif isinstance(frames, torch.Tensor):
            frame_list = frames.tolist()
        else:
            frame_list = list(frames)

        if any(f < 10 for f in frame_list):
            warnings.warn(
                "Frames 0–9 are typically pre-steady-state and may not represent "
                "equilibrium amorphous ice. Consider using frames ≥ 10.",
                stacklevel=3,
            )

        out_of_range = [f for f in frame_list if f < 0 or f >= self.n_frames]
        if out_of_range:
            raise IndexError(
                f"Frame indices {out_of_range} are out of range "
                f"(dump has {self.n_frames} frames, indices 0–{self.n_frames - 1})."
            )

        return frame_list

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_coordinates(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Parse and return trimmed coordinates for the specified frames.

        Results are stored as ``self.coordinates`` for subsequent calls to
        :meth:`get_voxels`, :meth:`compute_sk_3d`, and :meth:`compute_rdf`.

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            Frame indices to retrieve. ``None`` (default) uses all frames
            from index 10 onwards (frames 0–9 are pre-steady-state).

        Returns
        -------
        coordinates : list of torch.Tensor
            One tensor per frame, each of shape ``(N_i, 3)`` in Å centred
            at the box origin. ``N_i`` may differ between frames as atoms
            drift across the trim boundary.
        """
        frame_list = self._resolve_frames(frames)
        self.coordinates = [self._trim_frame(i) for i in frame_list]
        return self.coordinates

    def get_voxels(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Soft-voxelize trimmed coordinates onto a cubic ``(n, n, n)`` grid.

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.

        Returns
        -------
        voxels : torch.Tensor
            Shape ``(F, n, n, n)``, one voxelized cube per frame.
        """
        coords_list = self.get_coordinates(frames)
        voxels = [
            soft_voxelize_coordinates(
                c,
                grid_shape=(self.n, self.n, self.n),
                voxel_size=self.dx,
            )
            for c in coords_list
        ]
        return torch.stack(voxels)

    def compute_sk_3d(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the mean Fourier amplitude sqrt(S(k)) averaged over frames.

        By the definition S(k) = |FFT3(vox_f)|^2 / N_f (N_f = atom count of
        frame f, which varies slightly frame-to-frame as atoms drift across
        the trim boundary — see :meth:`get_coordinates`), so each frame's
        raw |FFT3| is divided by sqrt(N_f) *before* averaging over frames:
        mean_f( |FFT3(vox_f)| / sqrt(N_f) ). Without this, the result scales
        with the trimmed atom count instead of being an intensive (per-atom)
        quantity, which would make it meaningless as a target to rescale by a
        *different* target atom count downstream (see
        :func:`specter.ice._kernels.interpolate_target_kernel`).

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.

        Returns
        -------
        sk_3d : torch.Tensor
            Shape ``(n, n, n)``, DC component at centre (fftshift applied).
        """
        voxels = self.get_voxels(frames)
        assert self.coordinates is not None
        n_atoms = torch.tensor(
            [c.shape[0] for c in self.coordinates], dtype=voxels.dtype
        )
        amplitudes = torch.abs(fft3(voxels, shift=True)) / n_atoms.sqrt().view(
            -1, 1, 1, 1
        )
        return torch.mean(amplitudes, dim=0)

    def compute_sk_radial(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the spherically-averaged structure factor S(k).

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.

        Returns
        -------
        k : torch.Tensor
            Spatial frequency values in Å⁻¹.
        sk_radial : torch.Tensor
            Radially-averaged S(k).
        """
        sk3d = self.compute_sk_3d(frames)
        r_bins, sk_radial = radial_profile_3d(sk3d, return_r=True)
        dk = 1.0 / (self.n * self.dx)
        k = r_bins.float() * dk
        return k, sk_radial

    def compute_rdf(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
        dr: float = 0.5,
        r_max: float | None = None,
        chunk_size: int | None = None,
        approximate: bool = False,
        n_samples: int = 1_000_000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the radial distribution function g(r) averaged over frames.

        Delegates to :func:`specter.coords.radial_distribution_function` with
        ``volume = trim_size³`` and per-frame number density.  No periodic
        boundary conditions are applied (coordinates are from a trimmed,
        non-periodic cube).

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.
        dr : float, optional
            Bin width in Å. Default 0.5.
        r_max : float, optional
            Maximum radius in Å. Defaults to ``trim_size`` (the side length
            of the trimmed cube).
        chunk_size : int or None, optional
            Row block size for chunked exact mode. If ``None``, uses
            ``torch.pdist`` (full O(N²) — exact but high memory for large N).
        approximate : bool, optional
            Use random-pair-sampling approximation. Default False.
        n_samples : int, optional
            Pairs drawn in approximate mode. Default 1 000 000.

        Returns
        -------
        r : torch.Tensor
            Bin-centre radii in Å.
        g_r : torch.Tensor
            g(r) averaged across the requested frames.
        """
        if r_max is None:
            r_max = self.trim_size

        coords_list = self.get_coordinates(frames)
        volume = self.trim_size**3

        gr_sum: torch.Tensor | None = None
        r_out: torch.Tensor | None = None
        for coords in coords_list:
            r, gr = radial_distribution_function(
                coords,
                volume=volume,
                dr=dr,
                r_max=r_max,
                chunk_size=chunk_size,
                approximate=approximate,
                n_samples=n_samples,
            )
            if gr_sum is None:
                gr_sum = gr
                r_out = r
            else:
                gr_sum = gr_sum + gr

        assert gr_sum is not None and r_out is not None
        return r_out, gr_sum / len(coords_list)

    def generate_ice(
        self,
        batchsize: int = 1,
        frames: int | Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return a batch of voxelized ice volumes sampled from the MD dump.

        Parameters
        ----------
        batchsize : int, optional
            Number of ice volumes to return. Frames are sampled with replacement
            when ``batchsize`` exceeds the number of available frames.
            Default is 1.
        frames : int or sequence of int or Tensor or None, optional
            Which frames to draw from. ``None`` uses all equilibrated frames
            (index 10 and above). See :meth:`get_coordinates`.

        Returns
        -------
        ice : torch.Tensor
            Voxelized ice volumes, shape ``(batchsize, n, n, n)``.
        """
        voxels = self.get_voxels(frames)
        idx = torch.randint(0, voxels.shape[0], (batchsize,))
        return voxels[idx]

    def generate_big_ice(
        self,
        target_shape: tuple[int, int, int, int],
        num_unique: int = 8,
        frames: int | Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate a large ice volume by tiling frames sampled from the MD dump.

        Parameters
        ----------
        target_shape : tuple of int
            Output shape ``(B, nz, ny, nx)``.
        num_unique : int, optional
            Number of unique frames to sample as tile sources. Default is 8.
        frames : int or sequence of int or Tensor or None, optional
            Which frames to draw from. ``None`` uses all equilibrated frames
            (index 10 and above). See :meth:`get_coordinates`.

        Returns
        -------
        big_ice : torch.Tensor
            Tiled ice volume of shape ``target_shape``.
        """
        cubes = self.generate_ice(batchsize=num_unique, frames=frames)
        return tile_volume_from_blocks_blended(cubes, target_shape)

    def __repr__(self) -> str:
        return (
            f"MDSimDump(n_frames={self.n_frames}, n={self.n}, "
            f"dx={self.dx} Å, trim_size={self.trim_size} Å)"
        )


class ExtXYZDump:
    """
    Reader and analyser for extxyz trajectory files from amorphous ice MD simulations.

    Reads multi-frame extxyz files (e.g. cooling trajectories) using ASE.
    Coordinates are already confined to the periodic unit cell, so no spatial
    trimming is performed — atoms are only re-centred around the box midpoint
    before voxelization.  Optionally loads per-frame temperature metadata from
    a companion CSV, enabling temperature-based frame selection.

    Parameters
    ----------
    filepath : str
        Path to the extxyz trajectory file (may contain multiple frames).
    n : int, optional
        Number of voxels per axis for soft-voxelization. Default 200.
    dx : float, optional
        Voxel size in Å. Default 0.5.
    metadata_csv : str or None, optional
        Path to a CSV file whose last row contains per-frame temperatures.
        When provided, temperatures are stored in ``self.temperatures``.
        The CSV is expected to follow the format of ``gr_all_210K_280K_52frames.csv``
        where the last row (index ``-1``) is the temperature of each frame column.

    Attributes
    ----------
    n_frames : int
        Total number of frames found in the trajectory.
    temperatures : torch.Tensor or None
        Per-frame temperatures in K, shape ``(n_frames,)``, or ``None`` if no
        metadata CSV was supplied.
    coordinates : list of torch.Tensor or None
        Set by :meth:`get_coordinates`; one tensor per requested frame,
        each of shape ``(N, 3)`` in Å relative to the box centre.
    """

    def __init__(
        self,
        filepath: str,
        n: int = 200,
        dx: float = 0.5,
        metadata_csv: str | None = None,
    ) -> None:
        self.n = n
        self.dx = dx

        atoms_data = ase_io.read(filepath, format="extxyz", index=":")
        if not isinstance(atoms_data, list):
            # single-frame files are returned as a single Atoms object
            atoms_data = [atoms_data]
        self._atoms_list: list[Atoms] = atoms_data

        self.n_frames: int = len(self._atoms_list)

        fov = n * dx
        cell0 = self._atoms_list[0].cell.array
        cell_size = float(np.linalg.norm(cell0.sum(axis=0)) / np.sqrt(3))
        if fov > cell_size:
            warnings.warn(
                f"Field of view {fov:.1f} Å (n={n}, dx={dx} Å) exceeds the "
                f"estimated cell size {cell_size:.1f} Å. Voxelized volumes may "
                "contain zero-padded regions.",
                stacklevel=2,
            )

        self.temperatures: torch.Tensor | None = None
        if metadata_csv is not None:
            df = pd.read_csv(metadata_csv, index_col=0)
            temps = df.iloc[-1, :].to_numpy(dtype=np.float32)
            if len(temps) != self.n_frames:
                warnings.warn(
                    f"metadata_csv has {len(temps)} temperature entries but "
                    f"the trajectory has {self.n_frames} frames. "
                    "Temperatures will be stored as-is.",
                    stacklevel=2,
                )
            self.temperatures = torch.from_numpy(temps)

        self.coordinates: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _box_center(self, frame_idx: int) -> torch.Tensor:
        """Return the geometric centre of the simulation cell for one frame."""
        cell = self._atoms_list[frame_idx].cell.array  # (3, 3), rows = lattice vectors
        center = 0.5 * cell.sum(axis=0)
        return torch.tensor(center, dtype=torch.float32)

    def _center_frame(self, frame_idx: int) -> torch.Tensor:
        """
        Re-centre one frame's coordinates around the box midpoint.

        All atoms are already within the periodic cell, so no spatial masking
        is applied.

        Parameters
        ----------
        frame_idx : int
            Index into the frame list (0-based).

        Returns
        -------
        coords : torch.Tensor
            Coordinates, shape ``(N, 3)``, in Å relative to the box centre.
        """
        atoms = self._atoms_list[frame_idx]
        coords_np = atoms.positions.astype(np.float32)  # (N, 3)
        coords = torch.from_numpy(np.ascontiguousarray(coords_np))
        coords -= self._box_center(frame_idx)
        return coords

    def _resolve_frames(
        self,
        frames: int | Sequence[int] | torch.Tensor | None,
        t_min: float | None,
        t_max: float | None,
    ) -> list[int]:
        """
        Resolve ``frames`` / temperature bounds to a sorted list of frame indices.

        ``frames=None`` with no temperature bounds uses all frames.
        Temperature filtering requires a metadata CSV to have been supplied.
        """
        if t_min is not None or t_max is not None:
            if self.temperatures is None:
                raise ValueError(
                    "Temperature-based filtering requires a metadata_csv to be "
                    "passed to the constructor."
                )
            lo = t_min if t_min is not None else -float("inf")
            hi = t_max if t_max is not None else float("inf")
            return [
                i for i, T in enumerate(self.temperatures.tolist()) if lo <= T <= hi
            ]

        if frames is None:
            return list(range(self.n_frames))

        if isinstance(frames, int):
            frame_list = [frames]
        elif isinstance(frames, torch.Tensor):
            frame_list = frames.tolist()
        else:
            frame_list = list(frames)

        out_of_range = [f for f in frame_list if f < 0 or f >= self.n_frames]
        if out_of_range:
            raise IndexError(
                f"Frame indices {out_of_range} are out of range "
                f"(trajectory has {self.n_frames} frames, indices 0–{self.n_frames - 1})."
            )

        return frame_list

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_coordinates(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> list[torch.Tensor]:
        """
        Return re-centred coordinates for the specified frames.

        Results are stored as ``self.coordinates`` for subsequent calls to
        :meth:`get_voxels`, :meth:`compute_sk_3d`, and :meth:`compute_rdf`.

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            Frame indices to retrieve. ``None`` (default) uses all frames.
        t_min : float or None, optional
            Lower temperature bound (K) for frame selection. Requires
            ``metadata_csv`` to have been provided.
        t_max : float or None, optional
            Upper temperature bound (K) for frame selection. Requires
            ``metadata_csv`` to have been provided.

        Returns
        -------
        coordinates : list of torch.Tensor
            One tensor per frame, each of shape ``(N, 3)`` in Å centred
            at the box midpoint.
        """
        frame_list = self._resolve_frames(frames, t_min, t_max)
        self.coordinates = [self._center_frame(i) for i in frame_list]
        return self.coordinates

    def get_voxels(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> torch.Tensor:
        """
        Soft-voxelize coordinates onto a cubic ``(n, n, n)`` grid.

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.
        t_min : float or None, optional
            See :meth:`get_coordinates`.
        t_max : float or None, optional
            See :meth:`get_coordinates`.

        Returns
        -------
        voxels : torch.Tensor
            Shape ``(F, n, n, n)``, one voxelized cube per frame.
        """
        coords_list = self.get_coordinates(frames, t_min=t_min, t_max=t_max)
        voxels = [
            soft_voxelize_coordinates(
                c,
                grid_shape=(self.n, self.n, self.n),
                voxel_size=self.dx,
            )
            for c in coords_list
        ]
        return torch.stack(voxels)

    def compute_sk_3d(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> torch.Tensor:
        """
        Compute the mean Fourier amplitude sqrt(S(k)) averaged over frames.

        By the definition S(k) = |FFT3(vox_f)|^2 / N_f (N_f = atom count of
        frame f), each frame's raw |FFT3| is divided by sqrt(N_f) *before*
        averaging over frames: mean_f( |FFT3(vox_f)| / sqrt(N_f) ). Without
        this, the result scales with the frame's atom count instead of being
        an intensive (per-atom) quantity, which would make it meaningless as
        a target to rescale by a *different* target atom count downstream
        (see :func:`specter.ice._kernels.interpolate_target_kernel`).

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.
        t_min : float or None, optional
            See :meth:`get_coordinates`.
        t_max : float or None, optional
            See :meth:`get_coordinates`.

        Returns
        -------
        sk_3d : torch.Tensor
            Shape ``(n, n, n)``, DC component at centre (fftshift applied).
        """
        voxels = self.get_voxels(frames, t_min=t_min, t_max=t_max)
        assert self.coordinates is not None
        n_atoms = torch.tensor(
            [c.shape[0] for c in self.coordinates], dtype=voxels.dtype
        )
        amplitudes = torch.abs(fft3(voxels, shift=True)) / n_atoms.sqrt().view(
            -1, 1, 1, 1
        )
        return torch.mean(amplitudes, dim=0)

    def compute_sk_radial(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the spherically-averaged structure factor S(k).

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.
        t_min : float or None, optional
            See :meth:`get_coordinates`.
        t_max : float or None, optional
            See :meth:`get_coordinates`.

        Returns
        -------
        k : torch.Tensor
            Spatial frequency values in Å⁻¹.
        sk_radial : torch.Tensor
            Radially-averaged S(k).
        """
        sk3d = self.compute_sk_3d(frames, t_min=t_min, t_max=t_max)
        r_bins, sk_radial = radial_profile_3d(sk3d, return_r=True)
        dk = 1.0 / (self.n * self.dx)
        k = r_bins.float() * dk
        return k, sk_radial

    def compute_rdf(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
        dr: float = 0.5,
        r_max: float | None = None,
        chunk_size: int | None = None,
        approximate: bool = False,
        n_samples: int = 1_000_000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the radial distribution function g(r) averaged over frames.

        The cell volume is read directly from each ASE frame so that NPT
        volume fluctuations are accounted for in the number density.

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.
        t_min : float or None, optional
            See :meth:`get_coordinates`.
        t_max : float or None, optional
            See :meth:`get_coordinates`.
        dr : float, optional
            Bin width in Å. Default 0.5.
        r_max : float or None, optional
            Maximum radius in Å. Defaults to half the minimum cell length of
            the first selected frame (largest sphere fitting in the cell).
        chunk_size : int or None, optional
            Row block size for chunked exact mode.
        approximate : bool, optional
            Use random-pair-sampling approximation. Default False.
        n_samples : int, optional
            Pairs drawn in approximate mode. Default 1 000 000.

        Returns
        -------
        r : torch.Tensor
            Bin-centre radii in Å.
        g_r : torch.Tensor
            g(r) averaged across the requested frames.
        """
        frame_list = self._resolve_frames(frames, t_min, t_max)
        coords_list = self.get_coordinates(frame_list)

        if r_max is None:
            r_max = float(self._atoms_list[frame_list[0]].cell.lengths().min()) / 2.0

        gr_sum: torch.Tensor | None = None
        r_out: torch.Tensor | None = None
        for frame_idx, coords in zip(frame_list, coords_list):
            volume = float(self._atoms_list[frame_idx].get_volume())
            r, gr = radial_distribution_function(
                coords,
                volume=volume,
                dr=dr,
                r_max=r_max,
                chunk_size=chunk_size,
                approximate=approximate,
                n_samples=n_samples,
            )
            if gr_sum is None:
                gr_sum = gr
                r_out = r
            else:
                gr_sum = gr_sum + gr

        assert gr_sum is not None and r_out is not None
        return r_out, gr_sum / len(coords_list)

    def generate_ice(
        self,
        batchsize: int = 1,
        frames: int | Sequence[int] | torch.Tensor | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> torch.Tensor:
        """
        Return a batch of voxelized ice volumes sampled from the trajectory.

        Parameters
        ----------
        batchsize : int, optional
            Number of ice volumes to return. Frames are sampled with replacement
            when ``batchsize`` exceeds the number of available frames. Default 1.
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.
        t_min : float or None, optional
            See :meth:`get_coordinates`.
        t_max : float or None, optional
            See :meth:`get_coordinates`.

        Returns
        -------
        ice : torch.Tensor
            Voxelized ice volumes, shape ``(batchsize, n, n, n)``.
        """
        voxels = self.get_voxels(frames, t_min=t_min, t_max=t_max)
        idx = torch.randint(0, voxels.shape[0], (batchsize,))
        return voxels[idx]

    def generate_big_ice(
        self,
        target_shape: tuple[int, int, int, int],
        num_unique: int = 8,
        frames: int | Sequence[int] | torch.Tensor | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> torch.Tensor:
        """
        Generate a large ice volume by tiling frames sampled from the trajectory.

        Parameters
        ----------
        target_shape : tuple of int
            Output shape ``(B, nz, ny, nx)``.
        num_unique : int, optional
            Number of unique frames to sample as tile sources. Default 8.
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.
        t_min : float or None, optional
            See :meth:`get_coordinates`.
        t_max : float or None, optional
            See :meth:`get_coordinates`.

        Returns
        -------
        big_ice : torch.Tensor
            Tiled ice volume of shape ``target_shape``.
        """
        cubes = self.generate_ice(
            batchsize=num_unique, frames=frames, t_min=t_min, t_max=t_max
        )
        return tile_volume_from_blocks_blended(cubes, target_shape)

    def __repr__(self) -> str:
        temp_str = ""
        if self.temperatures is not None:
            t = self.temperatures
            temp_str = f", T={t.min().item():.0f}–{t.max().item():.0f} K"
        return (
            f"ExtXYZDump(n_frames={self.n_frames}, n={self.n}, "
            f"dx={self.dx} Å{temp_str})"
        )
