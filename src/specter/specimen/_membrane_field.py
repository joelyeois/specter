"""
Signed-field geometry for organic membrane shapes.

Builds a single dense signed field ``phi`` whose zero level set is a smooth,
non-self-intersecting membrane mid-surface: a smooth-min ("metaball") blend of
randomly placed spherical sources, perturbed with multi-octave value noise for
undulation, then relaxed with a bounded number of diffusion steps to cap
curvature. The bilayer leaflets are later extracted elsewhere as the ``+-
t/2`` level sets of this same field -- because both leaflets come from one
field rather than two independently offset surfaces, they cannot
self-intersect on concave geometry as long as the local curvature radius
stays above ``t/2`` everywhere, which the curvature-capping step guarantees.

All lengths are physical (Angstrom); the working grid's voxel spacing is a
parameter independent of any downstream output voxel size, so shape fidelity
does not depend on the resolution the caller ultimately rasterizes at.

Coordinates follow the rest of Specter: physical points are ``(x, y, z)``
triples, dense grids are ``(Z, Y, X)`` tensors, and ``MembraneField.origin_xyz``
is the physical location of grid index ``(0, 0, 0)`` (a corner, not the
centered-at-atom convention used by ``soft_voxelize_coordinates``).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MetaballSource:
    """
    One spherical source contributing to a smooth-min blended field.

    Parameters
    ----------
    center_xyz : torch.Tensor
        Physical center, shape ``(3,)``, Angstrom.
    radius : float
        Sphere radius, Angstrom.
    """

    center_xyz: torch.Tensor
    radius: float


@dataclass
class MembraneField:
    """
    Dense signed field over a physical working grid.

    Parameters
    ----------
    phi : torch.Tensor
        Signed field, shape ``(Z, Y, X)``. Negative inside the membrane
        solid, positive outside, zero at the mid-surface.
    spacing_a : float
        Isotropic voxel spacing of ``phi``, Angstrom.
    origin_xyz : torch.Tensor
        Physical ``(x, y, z)`` location of grid index ``(0, 0, 0)``, Angstrom.
    """

    phi: torch.Tensor
    spacing_a: float
    origin_xyz: torch.Tensor

    def sample(self, points_xyz: torch.Tensor) -> torch.Tensor:
        """
        Trilinearly interpolate ``phi`` at arbitrary physical points.

        Parameters
        ----------
        points_xyz : torch.Tensor
            Physical ``(x, y, z)`` points, shape ``(..., 3)``, Angstrom.

        Returns
        -------
        torch.Tensor
            Interpolated field values, shape ``(...)``. Points outside the
            grid are clamped to the nearest boundary value.
        """
        device = self.phi.device
        dtype = self.phi.dtype
        flat = points_xyz.reshape(-1, 3).to(device=device, dtype=dtype)
        grid = self._normalized_grid(flat)
        volume = self.phi[None, None]
        sampled = F.grid_sample(
            volume, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return sampled.reshape(points_xyz.shape[:-1])

    def gradient(
        self, points_xyz: torch.Tensor, eps_a: float | None = None
    ) -> torch.Tensor:
        """
        Unit outward normal at arbitrary physical points via finite differences.

        Parameters
        ----------
        points_xyz : torch.Tensor
            Physical ``(x, y, z)`` points, shape ``(..., 3)``, Angstrom.
        eps_a : float, optional
            Finite-difference step, Angstrom. Default ``0.5 * spacing_a``.

        Returns
        -------
        torch.Tensor
            Unit vectors pointing from negative (inside) to positive
            (outside) ``phi``, shape ``(..., 3)``.
        """
        eps = eps_a if eps_a is not None else 0.5 * self.spacing_a
        device = points_xyz.device
        dtype = points_xyz.dtype
        offsets = torch.eye(3, device=device, dtype=dtype) * eps
        components = []
        for axis in range(3):
            plus = self.sample(points_xyz + offsets[axis])
            minus = self.sample(points_xyz - offsets[axis])
            components.append((plus - minus) / (2.0 * eps))
        grad = torch.stack(components, dim=-1)
        norm = torch.linalg.norm(grad, dim=-1, keepdim=True).clamp_min(1e-8)
        return grad / norm

    def _normalized_grid(self, points_xyz: torch.Tensor) -> torch.Tensor:
        device = self.phi.device
        origin = self.origin_xyz.to(device=device, dtype=points_xyz.dtype)
        nz, ny, nx = self.phi.shape
        idx_x = (points_xyz[:, 0] - origin[0]) / self.spacing_a
        idx_y = (points_xyz[:, 1] - origin[1]) / self.spacing_a
        idx_z = (points_xyz[:, 2] - origin[2]) / self.spacing_a
        norm_x = 2.0 * (idx_x + 0.5) / nx - 1.0
        norm_y = 2.0 * (idx_y + 0.5) / ny - 1.0
        norm_z = 2.0 * (idx_z + 0.5) / nz - 1.0
        return torch.stack([norm_x, norm_y, norm_z], dim=-1).reshape(1, -1, 1, 1, 3)


def sample_metaball_sources(
    n_sources: int,
    radius_range_a: tuple[float, float],
    spread_a: float,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> list[MetaballSource]:
    """
    Draw randomly placed, randomly sized spherical sources.

    Parameters
    ----------
    n_sources : int
        Number of sources.
    radius_range_a : tuple of float
        ``(min, max)`` radius, Angstrom.
    spread_a : float
        Sources are drawn uniformly within ``[-spread_a, spread_a]`` on each
        axis, centered on the physical origin.
    device : str or torch.device, optional
        Device for the returned source centers. Default ``"cpu"``.
    seed : int, optional
        Random seed. Default ``None``.

    Returns
    -------
    list of MetaballSource
    """
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    radii = torch.empty(n_sources).uniform_(*radius_range_a, generator=generator)
    centers = (torch.rand((n_sources, 3), generator=generator) * 2.0 - 1.0) * spread_a
    return [
        MetaballSource(center_xyz=centers[i].to(device), radius=float(radii[i]))
        for i in range(n_sources)
    ]


def _smin_pair(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
    if k <= 0:
        return torch.minimum(a, b)
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return torch.lerp(b, a, h) - k * h * (1.0 - h)


def blend_field(
    sources: list[MetaballSource], points_xyz: torch.Tensor, k: float
) -> torch.Tensor:
    """
    Evaluate the smooth-min union of ``sources`` at arbitrary points.

    Each source contributes ``|x - center| - radius`` (an exact sphere SDF);
    the sources are combined with a polynomial smooth-min so the result is a
    single continuous field whose zero level set is a smooth, organic union
    of the spheres rather than a hard-edged one.

    Parameters
    ----------
    sources : list of MetaballSource
    points_xyz : torch.Tensor
        Physical ``(x, y, z)`` points, shape ``(..., 3)``, Angstrom.
    k : float
        Smooth-min blend radius, Angstrom. ``k <= 0`` gives a hard (sharp)
        union.

    Returns
    -------
    torch.Tensor
        Signed field values, shape ``(...)``.
    """
    if not sources:
        raise ValueError("sources must be non-empty")
    acc = (
        torch.linalg.norm(points_xyz - sources[0].center_xyz, dim=-1)
        - sources[0].radius
    )
    for s in sources[1:]:
        d = torch.linalg.norm(points_xyz - s.center_xyz, dim=-1) - s.radius
        acc = _smin_pair(acc, d, k)
    return acc


def _grid_points_xyz(
    shape_zyx: tuple[int, int, int],
    spacing_a: float,
    origin_xyz: torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    nz, ny, nx = shape_zyx
    iz = torch.arange(nz, device=device, dtype=dtype)
    iy = torch.arange(ny, device=device, dtype=dtype)
    ix = torch.arange(nx, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(iz, iy, ix, indexing="ij")
    origin = origin_xyz.to(device=device, dtype=dtype)
    x = origin[0] + xx * spacing_a
    y = origin[1] + yy * spacing_a
    z = origin[2] + zz * spacing_a
    return torch.stack([x, y, z], dim=-1)


def _value_noise_grid(
    shape_zyx: tuple[int, int, int],
    spacing_a: float,
    correlation_length_a: float,
    octaves: int,
    seed: int | None,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    noise = torch.zeros(shape_zyx, device=device, dtype=dtype)
    amplitude = 1.0
    total_amplitude = 0.0
    length = correlation_length_a
    for _ in range(octaves):
        lattice_shape = tuple(
            max(2, int(math.ceil(s * spacing_a / length)) + 1) for s in shape_zyx
        )
        lattice = (torch.rand(lattice_shape, generator=generator) * 2.0 - 1.0).to(
            device=device, dtype=dtype
        )
        upsampled = F.interpolate(
            lattice[None, None], size=shape_zyx, mode="trilinear", align_corners=True
        )[0, 0]
        # F.interpolate's trilinear mode is *linear* interpolation between
        # lattice points -- only C0 continuous, so wherever the lattice is
        # sparse relative to the domain it leaves visible piecewise-flat
        # creases (confirmed: disabling this blur reproduces a visibly
        # faceted/polygonal membrane boundary). A half-lattice-cell Gaussian
        # blur rounds off those creases without touching the octave's
        # actual large-scale undulation, which lives at the full
        # lattice-cell scale.
        cell_size_vox = length / spacing_a
        upsampled = _gaussian_blur3d(upsampled, sigma_vox=0.5 * cell_size_vox)
        noise = noise + amplitude * upsampled
        total_amplitude += amplitude
        amplitude *= 0.5
        length *= 0.5
    return noise / total_amplitude


def _gaussian_blur3d(volume: torch.Tensor, sigma_vox: float) -> torch.Tensor:
    if sigma_vox <= 0:
        return volume
    radius = max(1, int(3 * sigma_vox))
    x = torch.arange(-radius, radius + 1, dtype=volume.dtype, device=volume.device)
    kernel1d = torch.exp(-(x**2) / (2 * sigma_vox**2))
    kernel1d = kernel1d / kernel1d.sum()
    out = volume[None, None]
    for shape in ((-1, 1, 1), (1, -1, 1), (1, 1, -1)):
        kernel = kernel1d.reshape(1, 1, *shape)
        pad = tuple(radius if s == -1 else 0 for s in shape)
        out = F.conv3d(out, kernel, padding=pad)
    return out[0, 0]


def _laplacian3d(volume: torch.Tensor) -> torch.Tensor:
    kernel = torch.zeros((1, 1, 3, 3, 3), dtype=volume.dtype, device=volume.device)
    kernel[0, 0, 1, 1, 1] = -6.0
    for offset in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):
        kernel[0, 0, offset[0], offset[1], offset[2]] = 1.0
    padded = F.pad(volume[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")
    return F.conv3d(padded, kernel)[0, 0]


def cap_curvature(
    phi: torch.Tensor,
    spacing_a: float,
    iterations: int,
    step_fraction: float = 0.15,
) -> torch.Tensor:
    """
    Diffusion-based relaxation that caps high-curvature features of ``phi``.

    A cheap proxy for true mean-curvature flow: repeatedly nudges each voxel
    towards its local Laplacian, which damps sharp concavities/convexities
    fastest and leaves broad, already-smooth regions largely unchanged. Used
    to guarantee the later ``+-t/2`` bilayer leaflet offset cannot
    self-intersect.

    Parameters
    ----------
    phi : torch.Tensor
        Signed field, shape ``(Z, Y, X)``.
    spacing_a : float
        Voxel spacing of ``phi``, Angstrom.
    iterations : int
        Number of relaxation steps.
    step_fraction : float, optional
        Explicit-diffusion step size as a fraction of the stability bound
        (``1/6`` for the 7-point 3D stencil used here). Default 0.15.

    Returns
    -------
    torch.Tensor
        Relaxed field, same shape as ``phi``.
    """
    if iterations <= 0:
        return phi
    step = step_fraction * spacing_a**2
    out = phi
    for _ in range(iterations):
        laplacian = _laplacian3d(out) / spacing_a**2
        out = out + step * laplacian
    return out


def generate_membrane_field(
    shape_zyx: tuple[int, int, int],
    spacing_a: float,
    n_sources: int = 6,
    radius_range_a: tuple[float, float] = (150.0, 400.0),
    spread_a: float | None = None,
    blend_sharpness_a: float | None = None,
    noise_amplitude_a: float = 15.0,
    correlation_length_a: float = 40.0,
    noise_octaves: int = 3,
    curvature_iterations: int = 30,
    curvature_step_fraction: float = 0.15,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> MembraneField:
    """
    Generate an organic membrane mid-surface field.

    Combines a smooth-min blend of randomly placed spheres (arbitrary,
    non-primitive topology), multi-octave value noise (undulation), and
    diffusion-based curvature capping (guarantees a safe later bilayer
    offset) into a single dense :class:`MembraneField`.

    Parameters
    ----------
    shape_zyx : tuple of int
        Working grid shape, ``(Z, Y, X)``. Should cover the intended membrane
        instance's bounding box, not necessarily a full tomogram -- callers
        generating within a larger volume should crop to a local region
        first to keep memory bounded.
    spacing_a : float
        Working grid voxel spacing, Angstrom. Independent of any downstream
        output voxel size; choose e.g. ``min(v_size, thickness_a / 8)`` so
        curvature capping stays well-resolved regardless of the final raster
        resolution.
    n_sources : int, optional
        Number of blended spherical sources. Default 6.
    radius_range_a : tuple of float, optional
        Source radius range, Angstrom. Default ``(150.0, 400.0)``.
    spread_a : float, optional
        Source center spread, Angstrom (see :func:`sample_metaball_sources`).
        Default ``0.25`` of the smallest grid physical extent.
    blend_sharpness_a : float, optional
        Smooth-min blend radius, Angstrom. Default ``0.3`` of the mean
        source radius.
    noise_amplitude_a : float, optional
        Undulation noise amplitude, Angstrom. Default 15.0.
    correlation_length_a : float, optional
        Coarsest noise octave's correlation length, Angstrom. Default 40.0.
    noise_octaves : int, optional
        Number of noise octaves. Default 3.
    curvature_iterations : int, optional
        Curvature-capping relaxation steps. Default 30.
    curvature_step_fraction : float, optional
        See :func:`cap_curvature`. Default 0.15.
    device : str or torch.device, optional
        Device for the returned field. Default ``"cpu"``.
    seed : int, optional
        Random seed. Default ``None``.

    Returns
    -------
    MembraneField
    """
    extent_a = (
        torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=torch.float32)
        * spacing_a
    )
    origin_xyz = -0.5 * extent_a

    if spread_a is None:
        spread_a = 0.25 * float(extent_a.min())
    if blend_sharpness_a is None:
        blend_sharpness_a = 0.3 * sum(radius_range_a) / 2.0

    points_xyz = _grid_points_xyz(shape_zyx, spacing_a, origin_xyz, device)
    sources = sample_metaball_sources(
        n_sources, radius_range_a, spread_a, device=device, seed=seed
    )
    phi = blend_field(sources, points_xyz, blend_sharpness_a)

    if noise_amplitude_a > 0:
        noise = _value_noise_grid(
            shape_zyx, spacing_a, correlation_length_a, noise_octaves, seed, device
        )
        phi = phi + noise_amplitude_a * noise

    phi = cap_curvature(phi, spacing_a, curvature_iterations, curvature_step_fraction)

    _warn_if_clipped_at_boundary(phi)

    return MembraneField(phi=phi, spacing_a=spacing_a, origin_xyz=origin_xyz)


def _warn_if_clipped_at_boundary(phi: torch.Tensor) -> None:
    boundary = torch.cat(
        [
            phi[0].reshape(-1),
            phi[-1].reshape(-1),
            phi[:, 0].reshape(-1),
            phi[:, -1].reshape(-1),
            phi[:, :, 0].reshape(-1),
            phi[:, :, -1].reshape(-1),
        ]
    )
    if bool((boundary < 0).any()):
        warnings.warn(
            "generate_membrane_field: the membrane's solid interior "
            "(phi < 0) touches shape_zyx's boundary -- the organic shape "
            "is being clipped by the working grid rather than tapering to "
            "zero inside it (a real, previously observed failure mode: it "
            "produces an unphysical flat cut where the box face truncates "
            "the shape). Increase shape_zyx, or reduce spread_a / "
            "radius_range_a / noise_amplitude_a, so the shape's full "
            "extent -- including noise excursions -- fits within the grid.",
            stacklevel=2,
        )


__all__ = [
    "MetaballSource",
    "MembraneField",
    "sample_metaball_sources",
    "blend_field",
    "cap_curvature",
    "generate_membrane_field",
]
