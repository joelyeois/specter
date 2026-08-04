"""
Tetris-style template packing for specimen generation.

This is a PyTorch-native port of TemplateLearning's
``functions/multitetris_python.py`` placement idea: each candidate molecule is
randomly rotated, converted to a shell template, cross-correlated against the
currently occupied specimen, then placed at the best positive-scoring position.
The shell rewards contact at the candidate's outer band and strongly penalizes
overlap with the candidate's inner band, producing compact non-overlapping
packs without drawing random positions one at a time.

The public contract matches the other specimen generators in this package:
``generate()`` returns a plain ``torch.Tensor`` with shape ``(Z, Y, X)`` and
successful placements are kept as dataclass records with ``center_zyx`` and an
``xyz`` rotation matrix.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

from ...arrays import clip_insert_bounds
from ...fft import fftconvolve
from ...pdb import PDB
from ...potential import PotentialBuilder
from ...progress import TqdmProgress
from ...rotations import build_affine_matrix, random_rotation_matrix, rotate_volume
from .pdb_packing import estimate_protein_box_size

CorrelationBackend = Literal["conv3d", "fft", "auto"]
ThresholdMode = Literal["absolute", "relative", "auto"]


@dataclass
class TetrisProteinSpec:
    """
    One species for ``TetrisPackingSpecimenGenerator``.

    Parameters
    ----------
    pdb_source : str
        PDB ID or local PDB/mmCIF path. Used as the species id and, when
        ``template`` is not supplied, as the input to ``PotentialBuilder``.
    frequency : int, optional
        Number of placement attempts per outer iteration, matching
        TemplateLearning's ``frequencies`` argument. Default 1.
    template : torch.Tensor, optional
        Prebuilt density template with shape ``(Z, Y, X)``. If supplied, no PDB
        is loaded for this species.
    """

    pdb_source: str
    frequency: int = 1
    template: torch.Tensor | None = None


@dataclass
class TetrisPlacement:
    """One accepted Tetris placement."""

    species_id: str
    center_zyx: torch.Tensor
    rotation_matrix: torch.Tensor
    score: float


def _as_3tuple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError("target_shape must have three entries")
    return (int(value[0]), int(value[1]), int(value[2]))


def _odd_kernel_size(sigma: float, truncate: float = 4.0) -> int:
    if sigma <= 0:
        return 1
    radius = max(1, int(math.ceil(truncate * sigma)))
    return 2 * radius + 1


def _gaussian_kernel1d(
    sigma: float, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    size = _odd_kernel_size(sigma)
    if size == 1:
        return torch.ones(1, dtype=dtype, device=device)
    radius = size // 2
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _gaussian_blur3d(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return volume
    kernel = _gaussian_kernel1d(sigma, volume.dtype, volume.device)
    radius = kernel.numel() // 2
    out = volume[None, None]
    for shape in ((-1, 1, 1), (1, -1, 1), (1, 1, -1)):
        k = kernel.reshape(1, 1, *shape)
        out = F.conv3d(out, k, padding=tuple(radius if s == -1 else 0 for s in shape))
    return out[0, 0]


def _ball_kernel(radius: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return torch.ones((1, 1, 1, 1, 1), dtype=dtype, device=device)
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    ball = (xx**2 + yy**2 + zz**2) <= float(radius**2)
    return ball.to(dtype).unsqueeze(0).unsqueeze(0)


def _binary_dilation(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = _ball_kernel(radius, torch.float32, mask.device)
    hit = F.conv3d(mask.float()[None, None], kernel, padding=radius)
    return hit[0, 0] > 0


def _binary_erosion(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = _ball_kernel(radius, torch.float32, mask.device)
    needed = kernel.sum()
    hit = F.conv3d(mask.float()[None, None], kernel, padding=radius)
    return hit[0, 0] >= needed


def _binary_closing(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    return _binary_erosion(_binary_dilation(mask, radius), radius)


def _layer_from_distance(mask: torch.Tensor, distance: int) -> torch.Tensor:
    if distance == 0:
        return _binary_closing(mask, 1)
    if distance < 0:
        return _binary_erosion(_binary_closing(mask, 1), -distance)
    return _binary_dilation(mask, distance)


def _insert_add(
    volume: torch.Tensor, local_density: torch.Tensor, center_zyx: torch.Tensor
) -> None:
    bounds = clip_insert_bounds(center_zyx.tolist(), local_density.shape, volume.shape)
    if bounds is None:
        return
    dst, src = bounds
    volume[dst] += local_density[src]


def _zero_invalid_boundaries(
    correlation: torch.Tensor, template_shape: Sequence[int]
) -> None:
    hz, hy, hx = (int(s) // 2 for s in template_shape)
    correlation[: hz + 1, :, :] = 0
    correlation[:, : hy + 1, :] = 0
    correlation[:, :, : hx + 1] = 0
    correlation[-hz - 1 :, :, :] = 0
    correlation[:, -hy - 1 :, :] = 0
    correlation[:, :, -hx - 1 :] = 0


def _active_crop(
    volume: torch.Tensor,
    centers_zyx: list[torch.Tensor],
    margin: int,
) -> tuple[tuple[slice, slice, slice], torch.Tensor]:
    if not centers_zyx:
        full = tuple(slice(0, s) for s in volume.shape)
        return full, volume
    centers = torch.stack([c.to(device=volume.device) for c in centers_zyx]).long()
    shape = torch.tensor(volume.shape, device=volume.device)
    start = torch.clamp(centers.min(dim=0).values - margin, min=0)
    stop = torch.minimum(centers.max(dim=0).values + margin, shape)
    slices = tuple(slice(int(start[i]), int(stop[i])) for i in range(3))
    return slices, volume[slices]


def _repad_like(
    cropped: torch.Tensor,
    slices: tuple[slice, slice, slice],
    shape: Sequence[int],
) -> torch.Tensor:
    out = cropped.new_zeros(tuple(shape))
    out[slices] = cropped
    return out


def _correlate_same_conv3d(
    volume: torch.Tensor, template: torch.Tensor
) -> torch.Tensor:
    kernel = template.to(dtype=volume.dtype).unsqueeze(0).unsqueeze(0)
    kz, ky, kx = template.shape
    pad_z = kz - 1
    pad_y = ky - 1
    pad_x = kx - 1
    padded = F.pad(
        volume[None, None],
        (
            pad_x // 2,
            pad_x - pad_x // 2,
            pad_y // 2,
            pad_y - pad_y // 2,
            pad_z // 2,
            pad_z - pad_z // 2,
        ),
    )
    return F.conv3d(padded, kernel)[0, 0]


def _correlate_same_fft(volume: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    flipped = torch.flip(template.to(dtype=volume.dtype), dims=(0, 1, 2))
    return fftconvolve(volume, flipped, mode="same").real


class TetrisPackingSpecimenGenerator:
    """
    Pack protein templates into a 3D volume with the Tetris contact algorithm.

    Parameters
    ----------
    protein_specs : list of TetrisProteinSpec
        Species to attempt.
    target_shape : tuple of int, optional
        Output volume shape in Specter order ``(Z, Y, X)``. Default
        ``(64, 128, 128)``.
    v_size : float, optional
        Voxel size in Angstrom, used when building PDB-backed templates.
        Default 5.0.
    iterations : int, optional
        Number of outer Tetris iterations. Default 1.
    insertion_distances : tuple of int, optional
        Inner/outer shell distances in voxels, matching TemplateLearning's
        ``insersion_distances`` semantics. Default ``(-2, 0)``.
    sigma : float, optional
        Gaussian blur sigma in voxels before thresholding. Default 1.65.
    gray_level_threshold : float, optional
        Threshold after Gaussian blur. In ``threshold_mode="absolute"`` this is
        an absolute density value. In ``threshold_mode="relative"`` this value
        is ignored in favor of ``relative_threshold_fraction``. In
        ``threshold_mode="auto"``, the absolute value is used when it falls
        inside the blurred density's dynamic range; otherwise a relative
        threshold is used. Default 100.0.
    threshold_mode : {"auto", "absolute", "relative"}, optional
        How to interpret ``gray_level_threshold``. ``"auto"`` keeps
        TemplateLearning-style absolute thresholds for MRC-like volumes while
        avoiding empty masks for Specter ``PotentialBuilder`` templates whose
        density scale is much smaller. Default ``"auto"``.
    relative_threshold_fraction : float, optional
        Fraction of the blurred density maximum to use for relative/auto
        thresholding. Default 0.15.
    overlap_penalty : float, optional
        Multiplier applied to the inner layer. TemplateLearning uses 100.
        Default 100.0.
    parameterization : str, optional
        PotentialBuilder parameterization for PDB-backed specs. Default
        ``"shtyrov"``.
    device : str or torch.device, optional
        Device for placement operations. Default ``"cpu"``.
    correlation_backend : {"conv3d", "fft", "auto"}, optional
        Correlation implementation. ``"conv3d"`` uses PyTorch direct
        cross-correlation, ``"fft"`` uses Specter's FFT convolution helper
        with the score template flipped, and ``"auto"`` picks between them
        from the active crop/template size. Default ``"conv3d"``.
    progressbars : bool, optional
        Show Specter progress bars while building templates and placing
        molecules. Default True.
    center_first : bool, optional
        If True, place the first species at the center before the correlation
        loop, matching the original algorithm. Default True.
    grind : bool, optional
        Continue later attempts after saturation instead of stopping. Default
        False.
    seed : int, optional
        Random seed. Default None.
    """

    def __init__(
        self,
        protein_specs: list[TetrisProteinSpec],
        target_shape: tuple[int, int, int] = (64, 128, 128),
        v_size: float = 5.0,
        iterations: int = 1,
        insertion_distances: tuple[int, int] = (-2, 0),
        sigma: float = 1.65,
        gray_level_threshold: float = 100.0,
        threshold_mode: ThresholdMode = "auto",
        relative_threshold_fraction: float = 0.15,
        overlap_penalty: float = 100.0,
        pdb_cache_dir: str = "../pdb-data/",
        parameterization: str = "shtyrov",
        device: str | torch.device = "cpu",
        correlation_backend: CorrelationBackend = "conv3d",
        progressbars: bool = True,
        center_first: bool = True,
        grind: bool = False,
        seed: int | None = None,
    ):
        if not protein_specs:
            raise ValueError("protein_specs must be non-empty")
        if insertion_distances[0] > insertion_distances[1]:
            raise ValueError("insertion_distances must be ordered as (inner, outer)")
        if correlation_backend not in ("conv3d", "fft", "auto"):
            raise ValueError(
                "correlation_backend must be one of 'conv3d', 'fft', or 'auto'"
            )
        if threshold_mode not in ("absolute", "relative", "auto"):
            raise ValueError(
                "threshold_mode must be one of 'absolute', 'relative', or 'auto'"
            )
        if relative_threshold_fraction <= 0:
            raise ValueError("relative_threshold_fraction must be positive")

        self.protein_specs = protein_specs
        self.target_shape = _as_3tuple(target_shape)
        self.v_size = float(v_size)
        self.iterations = int(iterations)
        self.insertion_distances = (
            int(insertion_distances[0]),
            int(insertion_distances[1]),
        )
        self.sigma = float(sigma)
        self.gray_level_threshold = float(gray_level_threshold)
        self.threshold_mode = threshold_mode
        self.relative_threshold_fraction = float(relative_threshold_fraction)
        self.overlap_penalty = float(overlap_penalty)
        self.pdb_cache_dir = pdb_cache_dir
        self.parameterization = parameterization
        self.device = torch.device(device)
        self.correlation_backend = correlation_backend
        self.progressbars = progressbars
        self.center_first = center_first
        self.grind = grind
        self.seed = seed

        self.placements: list[TetrisPlacement] = []
        self.saturated_species: set[str] = set()

    def _build_template(self, spec: TetrisProteinSpec) -> torch.Tensor:
        if spec.template is not None:
            return spec.template.detach().to(self.device, dtype=torch.float32)

        pdb = PDB(spec.pdb_source, savefolder=self.pdb_cache_dir, verbose=False)
        n = estimate_protein_box_size(pdb.max_diameter, self.v_size)
        builder = PotentialBuilder(
            n_xyz=n,
            dx=self.v_size,
            atomic_numbers=pdb.atomic_numbers,
            progressbars=False,
            parameterization=self.parameterization,
        )
        return builder.forward(pdb.coordinates).to(self.device)

    def _rotate_template(
        self, template: torch.Tensor, rotation_matrix: torch.Tensor
    ) -> torch.Tensor:
        theta = build_affine_matrix(rotation_matrix.to(template.device))
        return rotate_volume(template, theta, padding_mode="zeros")[0]

    def _make_binary(self, density: torch.Tensor) -> torch.Tensor:
        blurred = _gaussian_blur3d(density, self.sigma)
        threshold = self._threshold_value(blurred)
        return blurred > threshold

    def _threshold_value(self, density: torch.Tensor) -> torch.Tensor:
        max_value = density.max()
        if self.threshold_mode == "absolute":
            return max_value.new_tensor(self.gray_level_threshold)
        relative = max_value * self.relative_threshold_fraction
        if self.threshold_mode == "relative":
            return relative
        if self.gray_level_threshold < float(max_value.item()):
            return max_value.new_tensor(self.gray_level_threshold)
        return relative

    def _make_score_template(self, rotated_density: torch.Tensor) -> torch.Tensor:
        binary = self._make_binary(rotated_density)
        inner_dist, outer_dist = self.insertion_distances
        outer_bool = _layer_from_distance(binary, outer_dist)
        inner_bool = _layer_from_distance(binary, inner_dist)
        if bool(binary.any()):
            if not bool(outer_bool.any()):
                outer_bool = binary
            if not bool(inner_bool.any()):
                inner_bool = binary
        outer = outer_bool.to(torch.float32)
        inner = inner_bool.to(torch.float32)
        return outer - self.overlap_penalty * inner

    def _current_occupancy(
        self, volume: torch.Tensor, centers_zyx: list[torch.Tensor], margin: int
    ) -> tuple[tuple[slice, slice, slice], torch.Tensor]:
        slices, cropped = _active_crop(volume, centers_zyx, margin)
        binary = self._make_binary(cropped).to(torch.float32)
        return slices, binary

    def _best_position(
        self,
        volume: torch.Tensor,
        centers_zyx: list[torch.Tensor],
        score_template: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        margin = max(score_template.shape)
        slices, occupancy = self._current_occupancy(volume, centers_zyx, margin)
        cropped_corr = self._correlate_same(occupancy, score_template)
        correlation = _repad_like(cropped_corr, slices, volume.shape)
        _zero_invalid_boundaries(correlation, score_template.shape)
        flat_idx = int(torch.argmax(correlation).item())
        score = float(correlation.reshape(-1)[flat_idx].item())
        center = torch.tensor(
            torch.unravel_index(
                torch.tensor(flat_idx, device=volume.device), volume.shape
            ),
            device=volume.device,
            dtype=torch.long,
        )
        return center, score

    def _correlate_same(
        self, occupancy: torch.Tensor, score_template: torch.Tensor
    ) -> torch.Tensor:
        backend = self.correlation_backend
        if backend == "auto":
            # Direct conv is usually faster for small kernels; FFT wins once
            # the candidate occupies a substantial fraction of the active box.
            kernel_voxels = score_template.numel()
            active_voxels = occupancy.numel()
            backend = "fft" if kernel_voxels > 0.02 * active_voxels else "conv3d"
        if backend == "fft":
            return _correlate_same_fft(occupancy, score_template)
        return _correlate_same_conv3d(occupancy, score_template)

    def _record(
        self,
        spec: TetrisProteinSpec,
        center_zyx: torch.Tensor,
        rotation_matrix: torch.Tensor,
        score: float,
    ) -> None:
        self.placements.append(
            TetrisPlacement(
                species_id=spec.pdb_source,
                center_zyx=center_zyx.detach().cpu(),
                rotation_matrix=rotation_matrix.detach().cpu(),
                score=score,
            )
        )

    def generate(self) -> torch.Tensor:
        """
        Run Tetris packing and return the assembled specimen volume.

        Returns
        -------
        torch.Tensor
            Volume with shape ``target_shape``.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        volume = torch.zeros(self.target_shape, dtype=torch.float32, device=self.device)
        self.placements = []
        self.saturated_species = set()
        centers_zyx: list[torch.Tensor] = []

        total = self.iterations * sum(
            spec.frequency for spec in self.protein_specs
        ) + int(self.center_first)

        with TqdmProgress(transient=True, disable=not self.progressbars) as progress:
            template_task = progress.add_task(
                "Building Tetris templates", total=len(self.protein_specs)
            )
            templates = []
            for spec in self.protein_specs:
                progress.update(
                    template_task,
                    description=f"Building Tetris template {spec.pdb_source}",
                )
                templates.append(self._build_template(spec))
                progress.update(template_task, advance=1)

            place_task = progress.add_task("Packing Tetris templates", total=total)

            if self.center_first:
                spec = self.protein_specs[0]
                progress.update(
                    place_task,
                    description=f"Packing Tetris template {spec.pdb_source} center",
                )
                rotation = random_rotation_matrix(batchsize=1, device=self.device)
                rotated = self._rotate_template(templates[0], rotation)
                center = torch.tensor(
                    [s // 2 for s in self.target_shape],
                    device=self.device,
                    dtype=torch.long,
                )
                _insert_add(volume, rotated, center)
                centers_zyx.append(center.detach().cpu())
                self._record(spec, center, rotation, score=float("inf"))
                progress.update(place_task, advance=1)

            failed_attempts = {spec.pdb_source: 0 for spec in self.protein_specs}

            max_failed_attempts = 100

            for _ in range(self.iterations):
                active_indices = [
                    i
                    for i, spec in enumerate(self.protein_specs)
                    if spec.pdb_source not in self.saturated_species
                ]

                if not active_indices:
                    break

                for _ in range(sum(spec.frequency for spec in self.protein_specs)):
                    active_indices = [
                        i
                        for i, spec in enumerate(self.protein_specs)
                        if spec.pdb_source not in self.saturated_species
                    ]

                    if not active_indices:
                        break

                    # weighted random choice based on frequency
                    weights = torch.tensor(
                        [self.protein_specs[i].frequency for i in active_indices],
                        dtype=torch.float32,
                    )

                    chosen = active_indices[torch.multinomial(weights, 1).item()]

                    spec = self.protein_specs[chosen]
                    template = templates[chosen]

                    progress.update(
                        place_task,
                        description=f"Packing Tetris template {spec.pdb_source}",
                    )

                    rotation = random_rotation_matrix(batchsize=1, device=self.device)

                    rotated = self._rotate_template(template, rotation)

                    score_template = self._make_score_template(rotated)

                    center, score = self._best_position(
                        volume,
                        centers_zyx,
                        score_template,
                    )

                    progress.update(place_task, advance=1)

                    if score <= 0:
                        failed_attempts[spec.pdb_source] += 1

                        if failed_attempts[spec.pdb_source] >= max_failed_attempts:
                            self.saturated_species.add(spec.pdb_source)

                            progress.update(
                                place_task,
                                description=(f"{spec.pdb_source} saturated"),
                            )

                        continue

                    # successful placement
                    failed_attempts[spec.pdb_source] = 0

                    _insert_add(
                        volume,
                        rotated,
                        center,
                    )

                    centers_zyx.append(center.detach().cpu())

                    self._record(
                        spec,
                        center,
                        rotation,
                        score,
                    )

                    progress.update(
                        place_task,
                        description=(f"Packing {spec.pdb_source} score={score:.3g}"),
                    )

        return volume.detach().cpu() if self.device.type != "cpu" else volume

    def export_picks(
        self,
        output_dir: str | Path,
        annotation_version: str = "1.0",
        oriented: bool = True,
    ) -> dict[str, Path]:
        """
        Write one copick/CryoET-Data-Portal-style NDJSON pick file per species.

        Coordinates are exported as corner-relative physical ``x, y, z``
        locations in Angstrom.
        """
        if not self.placements:
            raise RuntimeError("call generate() before export_picks()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        point_type = "orientedPoint" if oriented else "point"

        by_species: dict[str, list[TetrisPlacement]] = {}
        for placed in self.placements:
            by_species.setdefault(placed.species_id, []).append(placed)

        for species_id, placed_list in by_species.items():
            name = Path(species_id).stem
            path = (
                output_dir / f"{name}-{annotation_version}_{point_type.lower()}.ndjson"
            )
            with open(path, "w") as f:
                for placed in placed_list:
                    z, y, x = placed.center_zyx.to(torch.float32) * self.v_size
                    row: dict = {
                        "type": point_type,
                        "location": {
                            "x": float(x),
                            "y": float(y),
                            "z": float(z),
                        },
                    }
                    if oriented:
                        row["xyz_rotation_matrix"] = (
                            placed.rotation_matrix.numpy().tolist()
                        )
                    f.write(json.dumps(row) + "\n")
            written[species_id] = path

        return written


__all__ = [
    "TetrisPackingSpecimenGenerator",
    "TetrisPlacement",
    "TetrisProteinSpec",
]
