"""
Gaussian Random Field (GRF) membrane shape backend.

Unlike the star-convex "radius function of direction" backends
(``_field_spherical_harmonics.py``/``_field_fourier.py``), this one is
genuinely VOLUMETRIC -- the actual differentiator this backend exists for:
it CAN spontaneously produce non-star-convex or even disconnected topology,
which a radius-function-of-direction representation structurally cannot.

Construction: a baseline anisotropic-ellipsoid level (the same
``|scaled_point| - 1`` term the other backends use) plus additive,
spatially-correlated noise, thresholded at zero:

    level = (r_prime - 1) + grf_noise_amplitude * normalized_noise
    inside = level < 0

The noise is white noise blurred with an ANISOTROPIC Gaussian kernel
(different correlation length per physical axis, ``grf_correlation_length_a``
-- longer correlation length on one axis stretches the resulting blobs along
it, independent of ``grf_axes_a``'s own base-ellipsoid elongation) via
``_gaussian_blur3d_aniso``, a local per-axis-sigma extension of
``_field._gaussian_blur3d`` (duplicated rather than editing the existing
tested isotropic version, matching ``_raster.py``'s own precedent of an
independent local copy over cross-module sharing). Real-space blur, not FFT
spectral synthesis -- equivalent by the convolution theorem, and simpler:
``_gaussian_blur3d`` is already a separable per-axis convolution, extending
it to three independent sigmas is a near-two-line change, versus real new
FFT machinery with its own pitfall (manual zero-padding to avoid circular-
wraparound bleeding the far side of the grid into the near side, which
direct convolution doesn't have since ``F.conv3d`` already pads explicitly).

Blurred white noise's std shrinks as blur sigma grows (fewer effectively-
independent samples averaged per voxel) -- so the noise is normalized to
unit std AFTER blurring, before scaling by ``grf_noise_amplitude``, or that
parameter's visual strength would silently depend on
``grf_correlation_length_a``.

Disconnected topology is a FEATURE, not a bug to avoid
--------------------------------------------------------
At high ``grf_noise_amplitude``, isolated islands can appear anywhere noise
dips negative, even well outside the base ellipsoid -- unlike the other
backends' `phi<0` region, which is always exactly one star-convex body by
construction, this one is a boolean threshold of two independently-signed
additive terms and has no such guarantee. This is the actual point of this
backend (the other two new backends structurally cannot do this at all), so
it is documented as example-able (see ``dev/grf_sweep.py`` for a stress-test
config) rather than "fixed" -- the DEFAULT ``grf_noise_amplitude`` is picked
to sit clearly below that regime instead.

Same exact EDT construction + ``_MIN_RELIABLE_VOXELS_PER_RADIUS`` reliability
warning + boundary-clip warning as the other boolean-mask-derived backends
(``_field_alpha.py``/``_field_spherical_harmonics.py``/``_field_fourier.py``)
-- correct here, since this is also a boolean-mask+EDT construction.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from scipy import ndimage

from ._field import MembraneField, _grid_points_xyz

# Same underlying cause as the other EDT-derived backends' own identically-
# named constant -- duplicated rather than shared, matching this package's
# established fully-independent-module convention.
_MIN_RELIABLE_VOXELS_PER_RADIUS = 8.0


def _gaussian_blur3d_aniso(
    volume: torch.Tensor, sigma_vox_zyx: tuple[float, float, float]
) -> torch.Tensor:
    """
    Separable 3D Gaussian blur with an INDEPENDENT sigma per axis --
    ``_field._gaussian_blur3d`` extended from one shared ``sigma_vox`` to
    three, in ``(z, y, x)`` array-axis order matching ``volume``'s own
    ``(Z, Y, X)`` shape.

    Parameters
    ----------
    volume : torch.Tensor
        Shape ``(Z, Y, X)``.
    sigma_vox_zyx : tuple of float
        ``(sigma_z, sigma_y, sigma_x)``, working-grid voxels. A
        non-positive entry skips blurring on that axis entirely.

    Returns
    -------
    torch.Tensor
        Blurred volume, same shape as ``volume``.
    """
    out = volume[None, None]
    for sigma_vox, conv_shape in zip(
        sigma_vox_zyx, ((-1, 1, 1), (1, -1, 1), (1, 1, -1))
    ):
        if sigma_vox <= 0:
            continue
        radius = max(1, int(3 * sigma_vox))
        x = torch.arange(-radius, radius + 1, dtype=volume.dtype, device=volume.device)
        kernel1d = torch.exp(-(x**2) / (2 * sigma_vox**2))
        kernel1d = kernel1d / kernel1d.sum()
        kernel = kernel1d.reshape(1, 1, *conv_shape)
        pad = tuple(radius if s == -1 else 0 for s in conv_shape)
        out = F.conv3d(out, kernel, padding=pad)
    return out[0, 0]


def generate_membrane_field_gaussian_random_field(
    shape_zyx: tuple[int, int, int],
    spacing_a: float,
    grf_axes_a: tuple[float, float, float] = (200.0, 200.0, 200.0),
    grf_correlation_length_a: tuple[float, float, float] = (40.0, 40.0, 40.0),
    grf_noise_amplitude: float = 0.15,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> MembraneField:
    """
    Generate an organic membrane mid-surface field from an anisotropic
    ellipsoid perturbed by a Gaussian random field, in the same
    :class:`MembraneField` representation the other backends produce.

    Parameters
    ----------
    shape_zyx : tuple of int
        Working grid shape, ``(Z, Y, X)``.
    spacing_a : float
        Working grid voxel spacing, Angstrom.
    grf_axes_a : tuple of float, optional
        Physical semi-axes ``(a_x, a_y, a_z)`` of the base ellipsoid,
        Angstrom. Default ``(200.0, 200.0, 200.0)``.
    grf_correlation_length_a : tuple of float, optional
        Noise correlation length per physical axis ``(x, y, z)``, Angstrom
        -- an independent knob from `grf_axes_a`'s own base-ellipsoid
        elongation; a shorter correlation length on one axis gives finer
        texture there, a longer one gives smoother, more elongated
        undulation. Default ``(40.0, 40.0, 40.0)``.
    grf_noise_amplitude : float, optional
        Noise strength relative to the (unit-std-normalized) blurred noise
        field. Default 0.15, picked from a direct visual sweep (0.15/0.3/
        0.5/0.8, see ``dev/grf_sweep.py``) at margin generous enough to
        separate genuine topology change from grid-boundary clipping:
        0.3 already showed a visible near-separation crack (too close to
        the disconnected-topology regime for a default meant to reliably
        stay one blob), 0.8 clearly fragmented into several disconnected
        islands (a good stress-test config, not a good default); 0.15 gave
        a clean, single, organically-irregular blob with no cracking.
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

    points_xyz = _grid_points_xyz(shape_zyx, spacing_a, origin_xyz, device=device)
    axes = torch.tensor(grf_axes_a, dtype=torch.float32, device=device)
    p_prime = points_xyz / axes
    r_prime = torch.linalg.norm(p_prime, dim=-1)

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    noise = torch.randn(shape_zyx, generator=generator).to(device)

    corr_x, corr_y, corr_z = grf_correlation_length_a
    sigma_vox_zyx = (corr_z / spacing_a, corr_y / spacing_a, corr_x / spacing_a)
    blurred = _gaussian_blur3d_aniso(noise, sigma_vox_zyx)
    normalized_noise = blurred / blurred.std().clamp_min(1e-8)

    level = (r_prime - 1.0) + grf_noise_amplitude * normalized_noise
    inside_t = level < 0.0

    inside = inside_t.cpu().numpy()
    dist_out = ndimage.distance_transform_edt(~inside, sampling=spacing_a)
    dist_in = ndimage.distance_transform_edt(inside, sampling=spacing_a)
    phi_np = dist_out - dist_in
    phi = torch.as_tensor(phi_np, dtype=torch.float32, device=device)

    voxels_per_radius = min(grf_axes_a) / spacing_a
    if voxels_per_radius < _MIN_RELIABLE_VOXELS_PER_RADIUS:
        warnings.warn(
            f"generate_membrane_field_gaussian_random_field: min(grf_axes_a) "
            f"({min(grf_axes_a):.1f} A) is only {voxels_per_radius:.1f} "
            f"working-grid voxels at spacing_a={spacing_a:.2f} A -- below the "
            f"{_MIN_RELIABLE_VOXELS_PER_RADIUS:.0f} voxels/radius verified reliable "
            "(see generate_membrane_field_alpha_shape's own Notes for the same "
            "underlying Eikonal/grid-resolution reasoning) for "
            "sample_surface_sites' Newton-projection surface sampling. Increase "
            "grf_axes_a, or decrease spacing_a, to raise this ratio.",
            stacklevel=2,
        )

    clipped = (
        inside[0].any()
        or inside[-1].any()
        or inside[:, 0].any()
        or inside[:, -1].any()
        or inside[:, :, 0].any()
        or inside[:, :, -1].any()
    )
    if clipped:
        warnings.warn(
            "generate_membrane_field_gaussian_random_field: the solid interior "
            "touches the working grid's boundary -- the shape is being hard-"
            "clipped by shape_zyx rather than tapering to zero inside it. "
            "Increase shape_zyx, or reduce grf_axes_a/grf_noise_amplitude.",
            stacklevel=2,
        )

    return MembraneField(phi=phi, spacing_a=spacing_a, origin_xyz=origin_xyz.to(device))


__all__ = ["generate_membrane_field_gaussian_random_field"]
