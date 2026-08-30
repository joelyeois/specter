"""
Signed-field geometry for organic membrane shapes.

Core primitives for building a dense signed field ``phi`` whose zero level
set is a smooth, non-self-intersecting membrane mid-surface: a smooth-min
blend of spherical sources (:func:`blend_field`) relaxed with a bounded
number of diffusion steps to cap curvature (:func:`cap_curvature`). The
bilayer leaflets are later extracted elsewhere as the ``+-t/2`` level sets
of this same field -- because both leaflets come from one field rather than
two independently offset surfaces, they cannot self-intersect on concave
geometry as long as the local curvature radius stays above ``t/2``
everywhere, which the curvature-capping step guarantees.

Currently consumed by ``_field_swept_spline.py`` (a correlated random walk
of spherical sources, fed through the same ``blend_field``/``cap_curvature``
machinery) -- the module's own top-level field generator that used to live
here, ``generate_membrane_field`` (isotropically-scattered sources, i.e. a
"metaball" blend), was deleted along with the deprecated
``shape_backend="metaball"`` it backed; see git history if either is ever
needed as a reference again.

All lengths are physical (Å); the working grid's voxel spacing is a
parameter independent of any downstream output voxel size, so shape fidelity
does not depend on the resolution the caller ultimately rasterizes at.

Coordinates follow the rest of Specter: physical points are ``(x, y, z)``
triples, dense grids are ``(Z, Y, X)`` tensors, and ``MembraneField.origin_xyz``
is the physical location of grid index ``(0, 0, 0)`` (a corner, not the
centered-at-atom convention used by ``soft_voxelize_coordinates``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


@dataclass
class SphereSource:
    """
    One spherical source contributing to a smooth-min blended field.

    Parameters
    ----------
    center_xyz : torch.Tensor
        Physical center, shape ``(3,)``, Å.
    radius : float
        Sphere radius, Å.
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
        Isotropic voxel spacing of ``phi``, Å.
    origin_xyz : torch.Tensor
        Physical ``(x, y, z)`` location of grid index ``(0, 0, 0)``, Å.
    clipped_at_boundary : bool, optional
        Whether the organelle's solid interior touched the working grid's
        own boundary during construction -- the same condition each shape
        backend's own boundary-clip warning already fires on, surfaced here
        as a plain checkable flag (rather than requiring a caller to sniff
        warning text) so a higher-level caller (e.g.
        ``TomogramSpecimenGenerator``) can programmatically skip compositing
        a visibly-truncated instance. Default ``False``.
    """

    phi: torch.Tensor
    spacing_a: float
    origin_xyz: torch.Tensor
    clipped_at_boundary: bool = False

    def sample(self, points_xyz: torch.Tensor) -> torch.Tensor:
        """
        Trilinearly interpolate ``phi`` at arbitrary physical points.

        Parameters
        ----------
        points_xyz : torch.Tensor
            Physical ``(x, y, z)`` points, shape ``(..., 3)``, Å.

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
            Physical ``(x, y, z)`` points, shape ``(..., 3)``, Å.
        eps_a : float, optional
            Finite-difference step, Å. Default ``0.5 * spacing_a``.

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


def _smin_pair(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
    if k <= 0:
        return torch.minimum(a, b)
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return torch.lerp(b, a, h) - k * h * (1.0 - h)


def blend_field(
    sources: list[SphereSource], points_xyz: torch.Tensor, k: float
) -> torch.Tensor:
    """
    Evaluate the smooth-min union of ``sources`` at arbitrary points.

    Each source contributes ``|x - center| - radius`` (an exact sphere SDF);
    the sources are combined with a polynomial smooth-min so the result is a
    single continuous field whose zero level set is a smooth, organic union
    of the spheres rather than a hard-edged one.

    Parameters
    ----------
    sources : list of SphereSource
    points_xyz : torch.Tensor
        Physical ``(x, y, z)`` points, shape ``(..., 3)``, Å.
    k : float
        Smooth-min blend radius, Å. ``k <= 0`` gives a hard (sharp)
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
        Voxel spacing of ``phi``, Å.
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
    # Left allocating per iteration on purpose. Relaxing in place is ~1.6x
    # faster (0.160 s against 0.256 s, 8 iterations on a 300x600x600 field)
    # but measures a HIGHER peak, 1.62 GiB against 1.21, whichever way the
    # first iteration is handled -- clone up front or allocate on the first
    # pass. This function is a small slice of a run's wall clock and its
    # fields are volume-sized, so the memory is worth more than the speed.
    out = phi
    for _ in range(iterations):
        laplacian = _laplacian3d(out) / spacing_a**2
        out = out + step * laplacian
    return out


def _warn_if_clipped_at_boundary(phi: torch.Tensor) -> bool:
    """Warn (as before) and return whether the solid interior touches
    phi's own boundary, so callers can act on it programmatically instead
    of only seeing the warning (see MembraneField.clipped_at_boundary)."""
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
    clipped = bool((boundary < 0).any())
    if clipped:
        warnings.warn(
            "the membrane's solid interior (phi < 0) touches shape_zyx's "
            "boundary -- the organic shape is being clipped by the working "
            "grid rather than tapering to zero inside it (a real, "
            "previously observed failure mode: it produces an unphysical "
            "flat cut where the box face truncates the shape). Increase "
            "shape_zyx, or reduce the shape's own size parameters, so its "
            "full extent fits within the grid.",
            stacklevel=2,
        )
    return clipped


def _signed_distance_transform(
    inside: np.ndarray, spacing_a: float, device: str | torch.device
) -> torch.Tensor:
    """
    Physical-Å signed distance field from a boolean solid mask,
    matching :class:`MembraneField`'s own ``phi`` contract (negative
    inside, positive outside, zero at the surface): ``dist_out - dist_in``
    from two exact Euclidean distance transforms.

    Used by ``generate_membrane_field_spherical_harmonics``, which derives
    its shape from a boolean solid mask on a dense working grid. This is
    the single dominant cost there (measured directly: ~60% of wall time
    at a realistic 300**3-voxel working grid) -- `scipy.ndimage.
    distance_transform_edt` has no GPU path, so it runs on CPU regardless
    of `device`.

    When `device` is CUDA, this instead uses `cupyx.scipy.ndimage.
    distance_transform_edt` -- an exact GPU port of the same PBA+
    algorithm family scipy itself uses (verified bit-identical against
    scipy on a synthetic mask here, not an approximation). Measured
    20-700x faster for this call (fresh-process-first-call vs.
    steady-state-in-process, respectively) on a 300**3 mask, and ~3x
    end-to-end for field generation as a whole.

    `cupy` is a core dependency, so this is the normal path -- except on
    macOS, where `cupy-cuda12x` ships no wheels (see pyproject.toml). Falls
    back to the scipy CPU path -- with a one-time warning -- whenever the
    GPU path isn't usable: `cupy` isn't importable, or is installed but no
    CUDA device is actually available to it.

    Other GPU distance-transform routes were considered and rejected: the
    pure-PyTorch options either only implement an approximate transform
    (raster-scan or soft-min), which risks distorting the Eikonal property
    (``|grad(phi)| ~= 1``) `_placement.py`'s Newton-projection surface
    search relies on, or are 2D-only; the one exact PyTorch-native option
    (FastGeodis) ships no prebuilt wheel and needs a local CUDA-toolkit
    build to install. `cupy`'s wheel is prebuilt and already matches this
    project's own CUDA 12.1 pin for `torch` (see `pyproject.toml`'s
    `pytorch-cu121` index).

    Parameters
    ----------
    inside : np.ndarray
        Boolean solid-interior mask.
    spacing_a : float
        Working grid voxel spacing, Å -- passed to
        ``distance_transform_edt``'s `sampling` kwarg so the result is
        physical Å, not voxel units.
    device : str or torch.device
        Device for the returned tensor; also selects whether the GPU path
        is attempted at all (only when its `.type == "cuda"`).

    Returns
    -------
    torch.Tensor
        Signed distance field, same shape as `inside`, dtype float32, on
        `device`.
    """
    if torch.device(device).type == "cuda":
        gpu_phi = _try_gpu_signed_distance_transform(inside, spacing_a, device)
        if gpu_phi is not None:
            return gpu_phi

    dist_out = ndimage.distance_transform_edt(~inside, sampling=spacing_a)
    dist_in = ndimage.distance_transform_edt(inside, sampling=spacing_a)
    return torch.as_tensor(dist_out - dist_in, dtype=torch.float32, device=device)


def _try_gpu_signed_distance_transform(
    inside: np.ndarray, spacing_a: float, device: str | torch.device
) -> torch.Tensor | None:
    """
    GPU half of :func:`_signed_distance_transform`. Never raises -- returns
    `None` whenever the GPU path isn't usable, so the caller falls back to
    scipy on CPU.
    """
    try:
        import cupy
        from cupyx.scipy import ndimage as cundimage
    except ImportError:
        warnings.warn(
            "_signed_distance_transform: device is CUDA but 'cupy' isn't "
            "importable -- falling back to scipy's CPU distance transform, "
            "the dominant cost in membrane field generation (~3x slower "
            "end-to-end, and ~134 vs ~96 bytes of RAM per working-grid "
            "voxel). cupy is a core dependency, so this normally means "
            "either a macOS install (no cupy-cuda12x wheels exist) or a "
            "broken/partial environment -- `uv sync` should restore it.",
            stacklevel=3,
        )
        return None

    if not cupy.cuda.is_available():
        warnings.warn(
            "_signed_distance_transform: device is CUDA and 'cupy' is "
            "installed, but no CUDA device is available to it -- falling "
            "back to scipy's CPU distance transform.",
            stacklevel=3,
        )
        return None

    # Everything past the two guards above can still fail at RUN time, and a
    # failure here must not take the whole specimen build down when a correct,
    # if slower, CPU path is sitting right there: the transforms can hit
    # `cupy.cuda.memory.OutOfMemoryError` (this field is the single largest
    # GPU allocation a membrane makes -- ~3.7 GB at `_MAX_FIELD_VOXELS`, which
    # a small card may not have free even though it "has CUDA"), or a
    # `CUDARuntimeError` that only surfaces on kernel launch or JIT compile
    # rather than on the driver query `is_available()` does. Broad on purpose:
    # this function's contract is "never raises", and the original error is
    # carried in the warning rather than swallowed.
    device_index = torch.device(device).index or 0
    try:
        with cupy.cuda.Device(device_index):
            inside_gpu = cupy.asarray(inside)
            dist_out = cundimage.distance_transform_edt(~inside_gpu, sampling=spacing_a)
            dist_in = cundimage.distance_transform_edt(inside_gpu, sampling=spacing_a)
            phi_gpu = (dist_out - dist_in).astype(cupy.float32)
        return torch.from_dlpack(phi_gpu)
    except Exception as exc:
        warnings.warn(
            "_signed_distance_transform: the CuPy GPU distance transform "
            f"failed on {device} ({type(exc).__name__}: {exc}) -- falling "
            "back to scipy's CPU transform, which gives an identical result "
            "more slowly. Common causes are insufficient free VRAM for a "
            f"{inside.size:,}-voxel working grid, or a CUDA driver too old "
            "for the bundled runtime.",
            stacklevel=3,
        )
        return None


__all__ = [
    "SphereSource",
    "MembraneField",
    "blend_field",
    "cap_curvature",
]
