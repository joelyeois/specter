"""
Alpha-shape-based membrane shape backend.

``_field.py``'s metaball blend of a handful of isotropically-scattered
spheres is mathematically capped at "compact blob" -- swept
`curvature_iterations` from 0 to 30 and `noise_amplitude_a`/
`correlation_length_a` across a wide range, all stayed compact (verified
visually, not just reasoned about): noise there is only surface undulation
on top of an already-compact base union, it cannot change the base
topology. ``specimen._cts_membrane.MembraneBlobGenerator``'s noisy-point-
cloud-wrapped-in-an-alpha-shape approach can, at low `roughness`, fold into
genuinely elongated/tubular topology (verified visually: roughness 0.7/0.5/
0.3/0.15 goes from a rounded bowl to a twisted ribbon to a thin tube) --
because per-point noise magnitude scales as ``1/roughness**2``, letting the
alpha-shape wrap self-fold at low roughness in a way a small fixed union of
spheres cannot.

This module produces a :class:`~specter.specimen.membrane._field.MembraneField`
from that same alpha-shape geometry, so it drops into
:class:`~specter.specimen.membrane._generator.MembraneGenerator`'s existing
pipeline exactly like ``generate_membrane_field`` (metaball) does: both are
read afterward only through ``MembraneField``'s public contract (``.phi``,
``.spacing_a``, ``.origin_xyz``, ``.sample()``, ``.gradient()``) -- the
calibrated bilayer profile (``_profile.py``/``_raster.py``) and
transmembrane surface-site sampling (``_placement.py``) are completely
shape-backend-agnostic already, so neither needed any change to support
this. Only the SHAPE (a solid to compute a signed-distance field from) is
taken from CTS's algorithm here -- CTS's own point-scatter bilayer density
rendering (``BlobMembraneInstance.density``) is intentionally not reused;
the calibrated, continuous ``BilayerProfile`` this package already has is
strictly better-founded (real atomic reference patch, not raw point counts)
and works on any signed field regardless of its source.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy import ndimage

from .._cts_membrane import MembraneBlobGenerator, _point_in_alpha_solid
from ._field import MembraneField, _grid_points_xyz

# Verified directly (see this module's own docstring): a 25A-radius blob at
# ~7 voxels/radius made sample_surface_sites find zero valid transmembrane
# sites even after max_attempts; 40A radius at ~10-13 voxels/radius was
# reliable. This threshold sits just under the tested-reliable value, as a
# margin, not a hard cutoff pulled from theory -- so an explicit, actionable
# warning fires before a caller silently gets zero placements downstream.
_MIN_RELIABLE_VOXELS_PER_RADIUS = 8.0


def generate_membrane_field_alpha_shape(
    shape_zyx: tuple[int, int, int],
    spacing_a: float,
    blob_size_a: float = 300.0,
    blob_roughness: float = 0.3,
    surface_smoothing_voxels: float = 2.0,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> MembraneField:
    """
    Generate an organic membrane mid-surface field from CTS's alpha-shape
    blob algorithm, in the same :class:`MembraneField` representation
    ``generate_membrane_field`` (metaball) produces.

    Parameters
    ----------
    shape_zyx : tuple of int
        Working grid shape, ``(Z, Y, X)``. Should cover the intended
        membrane instance's bounding box with margin -- the alpha-shape's
        actual extent depends on `blob_size_a`/`blob_roughness` (low
        roughness can extend well past `blob_size_a` itself, since the
        per-point noise scales as ``1/roughness**2``); undersizing the grid
        clips the shape at the boundary the same way
        ``generate_membrane_field`` does (no separate warning is raised
        here -- inspect the returned field's own boundary values, or grow
        `shape_zyx` and retry, if a rendered volume looks hard-clipped).
    spacing_a : float
        Working grid voxel spacing, Angstrom. Independent of any downstream
        output voxel size, matching ``generate_membrane_field``'s own
        convention.
    blob_size_a : float, optional
        Rough target radius of the generated blob, Angstrom (CTS's own
        `size` parameter). Default 300.
    blob_roughness : float, optional
        In (0, 1); higher values give a rounder, less irregular/elongated
        shape, lower values give more irregular/elongated shapes, down to
        genuinely tube-like at very low values (CTS's own `roughness`/`sp`
        parameter -- see module docstring for the visual sweep this default
        is based on). Default 0.3 (visibly non-convex/elongated, short of
        the thin-tube extreme at 0.15).
    surface_smoothing_voxels : float, optional
        Blur-then-rethreshold smoothing applied to the alpha-solid mask
        before the distance transform, in working-grid voxels. `_build_blob`
        wraps a sparse point cloud in a Delaunay-triangulation-derived alpha
        shape (as few as ~20-80 points even at CTS's own default
        `blob_size_a`/`blob_roughness`, verified directly) -- the resulting
        boundary is a faceted polyhedron, visible as sharp kinks rather than
        a smooth organic surface once rendered at any reasonably zoomed-out
        scale. Raising the point count instead (to shrink facets directly)
        only pushes the problem out, not away (surface area grows with
        `blob_size_a**2` while `_build_blob`'s point count grows far more
        slowly), and gets expensive fast. Blurring the boolean mask by a few
        voxels and rethresholding at 0.5 removes this faceting cheaply and
        uniformly across scale (verified directly: rounds sharp polyhedral
        corners into organic curves on both compact and elongated shapes,
        confirmed on both CTS's own default point density and a much denser
        one -- the faceting is inherent to the triangulated boundary, not
        fixable by point count alone). Deliberately expressed in *voxels*,
        not Angstrom or a fraction of `blob_size_a`: a fixed physical
        smoothing length large enough to smooth 300+ A blobs was verified
        directly to erase genuinely thin, low-`blob_roughness` tube-like
        protrusions entirely (their cross-section can be far smaller than
        `blob_size_a` itself), whereas a small fixed voxel count scales down
        automatically with `spacing_a` and only removes sub-facet-scale
        jaggedness. Set to 0 to disable. Default 2.0.
    device : str or torch.device, optional
        Device for the returned field. Default ``"cpu"``.
    seed : int, optional
        Random seed. Default ``None``.

    Notes
    -----
    `_placement.sample_surface_sites`' Newton projection moves a candidate
    by ``-phi(x) * unit_normal(x)`` per step, which only converges reliably
    if `phi` closely satisfies the Eikonal property (``|grad(phi)| ~= 1``)
    near the surface. The metaball backend's `phi` is an analytic smooth-min
    blend of exact sphere SDFs, well-behaved by construction; this backend's
    `phi` is a discretized `scipy.ndimage.distance_transform_edt` on a
    boolean alpha-solid mask -- algorithmically exact, but only as smooth as
    the grid resolving it. Verified directly: too few voxels across
    `blob_size_a` (e.g. a 25A-radius blob at ~7 voxels/radius) makes
    `sample_surface_sites` find zero valid sites even after `max_attempts`;
    ~10+ voxels/radius (e.g. 40A radius at 3-4A/voxel) was reliable in
    testing. Prefer a larger `blob_size_a`/finer `spacing_a` over a smaller/
    coarser one if transmembrane placement unexpectedly finds no sites.

    Returns
    -------
    MembraneField
    """
    blob_gen = MembraneBlobGenerator(v_size=spacing_a, seed=seed)
    base_points = blob_gen._build_blob(blob_size_a, blob_roughness)
    alpha = blob_size_a * 0.9

    extent_a = (
        torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=torch.float32)
        * spacing_a
    )
    origin_xyz = -0.5 * extent_a

    points_xyz = _grid_points_xyz(shape_zyx, spacing_a, origin_xyz, device="cpu")
    query = points_xyz.reshape(-1, 3).numpy()
    inside = _point_in_alpha_solid(base_points, alpha, query).reshape(shape_zyx)

    # Blur-then-rethreshold: removes the alpha-shape's Delaunay-facet
    # kinks (see this function's own surface_smoothing_voxels docstring)
    # without touching _build_blob itself, so CTS's own native generator
    # (which reuses _build_blob directly) is unaffected.
    if surface_smoothing_voxels > 0:
        smoothed = ndimage.gaussian_filter(
            inside.astype(np.float32), sigma=surface_smoothing_voxels
        )
        inside = smoothed > 0.5

    # Physical-Angstrom signed distance, matching MembraneField.phi's
    # contract exactly (negative inside, positive outside, zero at the
    # mid-surface) -- distance_transform_edt's `sampling` kwarg scales its
    # output by the given voxel spacing directly, avoiding the kind of
    # unit-mixup _cts_membrane.py's own docstring documents having to fix
    # once already (its own signed_dist there is deliberately left in
    # voxel-index units for a different downstream use; MembraneField.phi
    # is physical Angstrom, a different contract, so that convention is
    # NOT reused here).
    dist_out = ndimage.distance_transform_edt(~inside, sampling=spacing_a)
    dist_in = ndimage.distance_transform_edt(inside, sampling=spacing_a)
    phi_np = dist_out - dist_in

    phi = torch.as_tensor(phi_np, dtype=torch.float32, device=device)

    voxels_per_radius = blob_size_a / spacing_a
    if voxels_per_radius < _MIN_RELIABLE_VOXELS_PER_RADIUS:
        warnings.warn(
            f"generate_membrane_field_alpha_shape: blob_size_a ({blob_size_a:.1f} A) "
            f"is only {voxels_per_radius:.1f} working-grid voxels/radius at "
            f"spacing_a={spacing_a:.2f} A -- below the "
            f"{_MIN_RELIABLE_VOXELS_PER_RADIUS:.0f} voxels/radius verified reliable "
            "for sample_surface_sites' Newton-projection surface sampling (see this "
            "function's own Notes). Transmembrane placement may silently find zero "
            "sites at this resolution. Increase blob_size_a, or decrease spacing_a "
            "(e.g. a smaller v_size on the owning MembraneGenerator), to raise this "
            "ratio.",
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
            "generate_membrane_field_alpha_shape: the alpha-shape blob's solid "
            "interior touches the working grid's boundary -- the shape is being "
            "hard-clipped by shape_zyx rather than tapering to zero inside it "
            "(same failure mode generate_membrane_field's own boundary check "
            "guards against, see its docstring). Increase shape_zyx (or the owning "
            "MembraneGenerator's target_shape_zyx/v_size), or reduce blob_size_a "
            "(and note low blob_roughness can extend the shape's true reach well "
            "past blob_size_a itself -- see this function's own blob_roughness "
            "docstring).",
            stacklevel=2,
        )

    return MembraneField(phi=phi, spacing_a=spacing_a, origin_xyz=origin_xyz.to(device))


__all__ = ["generate_membrane_field_alpha_shape"]
