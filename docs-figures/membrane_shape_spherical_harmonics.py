"""
Generate the step-by-step figures for
docs/concepts/membrane-shape/spherical-harmonics.md, which explains the
``spherical_harmonics`` membrane shape backend
(``specter.specimen.membrane._field_spherical_harmonics``).

Every figure walks through one concept/step of that module's own Algorithm
docstring, using ONE fixed reference organelle instance (same seed/params)
wherever possible so the figures visibly chain together instead of being
unrelated one-off plots. Intermediate arrays (the raw inside/outside test,
the pre-EDT boolean mask, the coarse angular grid) are not exposed by the
public function, so `_reference_instance` below duplicates the minimal
sequence of internal-module calls needed to capture them -- it deliberately
calls the SAME private helpers the real function calls
(`_sample_sh_coefficients`, `_synthesize_angular_grid`,
`_interpolate_angular_grid`), never reimplementing the math independently,
so the figures cannot silently drift from what the shipped code actually
does.

Run with: uv run python docs-figures/membrane_shape_spherical_harmonics.py
Saves PNGs directly into docs/assets/images/ (consumed by
docs/concepts/membrane-shape/spherical-harmonics.md).
"""

from __future__ import annotations

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import ndimage

from _render import TEAL, isosurface_axes
from specter.specimen.membrane._field import _grid_points_xyz
from specter.specimen.membrane._field_spherical_harmonics import (
    _angular_grid_resolution,
    _interpolate_angular_grid,
    _real_spherical_harmonic,
    _sample_sh_coefficients,
    _synthesize_angular_grid,
)
from specter.specimen.membrane._generator import MembraneGenerator

OUT_DIR = "docs/assets/images"

# Reference instance shared across most figures, matching
# generate_membrane_field_spherical_harmonics' own defaults
# (sh_max_degree=8, sh_amplitude=0.15, sh_spectrum_power=2.0) except for a
# fixed seed chosen for a visually clear (non-degenerate, non-extreme) shape.
SHAPE_ZYX = (140, 140, 140)
SPACING_A = 3.0
SH_MAX_DEGREE = 8
SH_AXES_A = (150.0, 150.0, 150.0)
SH_AMPLITUDE = 0.15
SH_SPECTRUM_POWER = 2.0
SEED = 7


def _reference_instance() -> dict:
    """Reproduce generate_membrane_field_spherical_harmonics' own steps,
    returning every intermediate array those docs figures need."""
    extent_a = (
        torch.tensor([SHAPE_ZYX[2], SHAPE_ZYX[1], SHAPE_ZYX[0]], dtype=torch.float32)
        * SPACING_A
    )
    origin_xyz = -0.5 * extent_a
    points_xyz = _grid_points_xyz(SHAPE_ZYX, SPACING_A, origin_xyz, device="cpu")
    query = points_xyz.reshape(-1, 3).numpy()

    axes = np.asarray(SH_AXES_A, dtype=np.float64)
    p_prime = query / axes
    r_prime = np.linalg.norm(p_prime, axis=-1)
    r_safe = np.clip(r_prime, 1e-12, None)
    theta = np.arccos(np.clip(p_prime[:, 2] / r_safe, -1.0, 1.0))
    phi_ang = np.arctan2(p_prime[:, 1], p_prime[:, 0])

    rng = np.random.default_rng(SEED)
    coefficients = _sample_sh_coefficients(SH_MAX_DEGREE, SH_SPECTRUM_POWER, rng)

    n_theta, n_phi = _angular_grid_resolution(SH_MAX_DEGREE)
    coarse_padded = _synthesize_angular_grid(
        coefficients, SH_MAX_DEGREE, n_theta, n_phi
    )
    perturbation = _interpolate_angular_grid(coarse_padded, theta, phi_ang, "cpu")

    r_surface = np.clip(1.0 + SH_AMPLITUDE * perturbation, 0.05, None)
    inside = (r_prime < r_surface).reshape(SHAPE_ZYX)

    dist_out = ndimage.distance_transform_edt(~inside, sampling=SPACING_A)
    dist_in = ndimage.distance_transform_edt(inside, sampling=SPACING_A)
    phi_field = dist_out - dist_in

    return dict(
        coefficients=coefficients,
        n_theta=n_theta,
        n_phi=n_phi,
        coarse_padded=coarse_padded,
        inside=inside,
        phi_field=phi_field,
    )


def _phi_field_at_resolution(
    ref: dict, shape_zyx: tuple[int, int, int], spacing_a: float
) -> np.ndarray:
    """Re-run only the grid-resolution-dependent steps (query points, inside
    test, EDT) at a different working-grid resolution, reusing the SAME
    random coefficients and SAME coarse angular grid from `ref` (both are
    grid-resolution-independent) -- so this is still the identical organelle,
    just resolved more finely, for a smoother hero isosurface render."""
    extent_a = (
        torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=torch.float32)
        * spacing_a
    )
    origin_xyz = -0.5 * extent_a
    points_xyz = _grid_points_xyz(shape_zyx, spacing_a, origin_xyz, device="cpu")
    query = points_xyz.reshape(-1, 3).numpy()

    axes = np.asarray(SH_AXES_A, dtype=np.float64)
    p_prime = query / axes
    r_prime = np.linalg.norm(p_prime, axis=-1)
    r_safe = np.clip(r_prime, 1e-12, None)
    theta = np.arccos(np.clip(p_prime[:, 2] / r_safe, -1.0, 1.0))
    phi_ang = np.arctan2(p_prime[:, 1], p_prime[:, 0])

    perturbation = _interpolate_angular_grid(
        ref["coarse_padded"], theta, phi_ang, "cpu"
    )
    r_surface = np.clip(1.0 + SH_AMPLITUDE * perturbation, 0.05, None)
    inside = (r_prime < r_surface).reshape(shape_zyx)

    dist_out = ndimage.distance_transform_edt(~inside, sampling=spacing_a)
    dist_in = ndimage.distance_transform_edt(inside, sampling=spacing_a)
    return dist_out - dist_in


def figure_hero(ref: dict) -> None:
    """Page-top hero: shaded 3D isosurface of the reference organelle, at a
    finer working-grid resolution than the other (slice-based) figures need.

    Under directional lighting, a faint concentric-ring texture remains
    visible even at this resolution -- verified (by evaluating the harmonic
    sum directly, per-voxel, with `_real_spherical_harmonics_grid` instead
    of the interpolated path below) to NOT be an artifact of the coarse-
    angular-grid interpolation trick (`_interpolate_angular_grid`): it is
    still present when the perturbation is instead evaluated directly,
    per-voxel, with no interpolation at all. It is instead the real,
    expected texture of building a signed distance field from
    `scipy.ndimage.distance_transform_edt` on a BINARY inside/outside mask
    (module docstring step 4): EDT distances are exact to the nearest
    boundary VOXEL, not to the continuous analytic surface, so the
    resulting field has genuine voxel-grid-scale ripple -- finer
    `hero_spacing_a` shrinks it (this resolution + a softened light in
    `_render.shade_mesh` were chosen empirically to make it unobtrusive) but
    never removes it outright. Harmless for the field's actual use (the
    bilayer profile only ever samples within a thin band near the zero level
    set), but worth knowing before assuming a rendering bug."""
    hero_shape_zyx = (300, 300, 300)
    hero_spacing_a = 1.4
    hero_phi = _phi_field_at_resolution(ref, hero_shape_zyx, hero_spacing_a)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    isosurface_axes(ax, hero_phi, hero_spacing_a, TEAL)
    ax.view_init(elev=18, azim=35)
    plt.tight_layout(pad=0)
    path = f"{OUT_DIR}/membrane-sh-hero.png"
    plt.savefig(path, dpi=300, transparent=True)
    plt.close(fig)
    print(f"saved {path}")


def figure_basis_gallery() -> None:
    """Individual real spherical harmonics Y_l^m rendered as colored,
    radius-perturbed unit spheres -- the literal 'building blocks' of the
    R(theta, phi) sum."""
    lm_pairs = [(2, 0), (2, 2), (3, 1), (3, 3), (4, 0), (4, 4)]
    theta = np.linspace(0.0, np.pi, 140)
    phi = np.linspace(0.0, 2.0 * np.pi, 280)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), subplot_kw={"projection": "3d"})
    cmap = matplotlib.colormaps["RdBu_r"]
    for ax, (degree, order) in zip(axes.ravel(), lm_pairs):
        y = _real_spherical_harmonic(degree, order, tt, pp)
        y_norm = y / np.abs(y).max()
        r = 1.0 + 0.35 * y_norm
        x = r * np.sin(tt) * np.cos(pp)
        yy = r * np.sin(tt) * np.sin(pp)
        z = r * np.cos(tt)
        face_val = 0.5 * (y_norm[:-1, :-1] + y_norm[1:, 1:])
        colors = cmap(mcolors.Normalize(vmin=-1, vmax=1)(face_val))
        ax.plot_surface(
            x,
            yy,
            z,
            facecolors=colors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.view_init(elev=18, azim=35)
        ax.set_title(rf"$Y_{{{degree}}}^{{{order}}}$", fontsize=13)
    fig.suptitle(
        "Real spherical harmonics -- the basis functions R(θ, φ) is built from",
        fontsize=11,
    )
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-sh-basis-gallery.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def figure_degree_buildup(ref: dict) -> None:
    """Same random coefficient draw as the reference instance, truncated at
    increasing max degree L -- shows the sum-of-harmonics surface gaining
    detail as more terms are added, in the same (theta, phi) map used
    internally (see _synthesize_angular_grid)."""
    n_theta, n_phi = ref["n_theta"], ref["n_phi"]
    degrees = [2, 3, 4, 8]
    fig, axes = plt.subplots(1, len(degrees), figsize=(3.1 * len(degrees), 3.4))
    vmax = 0.0
    fields = []
    for max_l in degrees:
        subset = [c for c in ref["coefficients"] if c[0] <= max_l]
        field = _synthesize_angular_grid(subset, SH_MAX_DEGREE, n_theta, n_phi)[:, :-1]
        fields.append(field)
        vmax = max(vmax, np.abs(field).max())
    for ax, max_l, field in zip(axes, degrees, fields):
        im = ax.imshow(
            field,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[0, 360, 180, 0],
            aspect="auto",
        )
        ax.set_title(f"L = {max_l}", fontsize=11)
        ax.set_xlabel(r"$\phi$ (deg)")
        if max_l == degrees[0]:
            ax.set_ylabel(r"$\theta$ (deg)")
        else:
            ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.75, label="perturbation (a.u.)", pad=0.02)
    fig.suptitle(
        "Same random coefficients, increasing harmonic degree cutoff L",
        fontsize=11,
    )
    path = f"{OUT_DIR}/membrane-sh-degree-buildup.png"
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def figure_angular_grid_interp(ref: dict) -> None:
    """Coarse (n_theta x n_phi) synthesis grid vs. the bilinearly
    interpolated fine field it stands in for -- the performance trick
    described in the module's Algorithm step 2."""
    coarse = ref["coarse_padded"][:, :-1]
    n_theta, n_phi = ref["n_theta"], ref["n_phi"]

    fine_theta = np.linspace(0.0, np.pi, 4 * n_theta)
    fine_phi = np.linspace(0.0, 2.0 * np.pi, 4 * n_phi, endpoint=False)
    tt, pp = np.meshgrid(fine_theta, fine_phi, indexing="ij")
    fine = _interpolate_angular_grid(
        ref["coarse_padded"], tt.ravel(), pp.ravel(), "cpu"
    ).reshape(tt.shape)

    vmax = np.abs(coarse).max()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axes[0].imshow(
        coarse,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        extent=[0, 360, 180, 0],
        aspect="auto",
    )
    axes[0].set_title(f"coarse synthesis grid ({n_theta}×{n_phi})", fontsize=10)
    axes[1].imshow(
        fine,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
        extent=[0, 360, 180, 0],
        aspect="auto",
    )
    axes[1].set_title("bilinearly interpolated (voxel resolution)", fontsize=10)
    for ax in axes:
        ax.set_xlabel(r"$\phi$ (deg)")
    axes[0].set_ylabel(r"$\theta$ (deg)")
    axes[1].set_yticks([])
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-sh-angular-grid-interp.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def figure_spectrum() -> None:
    """Var(a_lm) ~ [l(l+1)]^-p for a few spectrum_power values -- shows why
    p=2 (Helfrich) suppresses high-degree/short-wavelength modes relative to
    p=0 (flat/white spectrum)."""
    degrees = np.arange(2, 17)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for p, style in [(0.0, ":"), (1.0, "--"), (2.0, "-"), (4.0, "-.")]:
        variance = (degrees * (degrees + 1.0)) ** (-p)
        variance /= variance[0]
        label = f"p = {p:g}" + ("  (Helfrich default)" if p == 2.0 else "")
        ax.plot(
            degrees,
            variance,
            style,
            label=label,
            color=str(0.15 + 0.15 * degrees[0] / 20),
        )
    ax.set_yscale("log")
    ax.set_xlabel("harmonic degree l")
    ax.set_ylabel(r"Var($a_{lm}$), normalized to l=2")
    ax.set_title("Undulation spectrum vs. sh_spectrum_power")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-sh-spectrum.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def figure_inside_test(ref: dict) -> None:
    """Central z-slice of the raw boolean inside/outside test -- step 3 of
    the module's Algorithm, before any distance transform is applied."""
    z_mid = SHAPE_ZYX[0] // 2
    sl = ref["inside"][z_mid]
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.imshow(sl, cmap="gray_r", origin="lower")
    ax.set_title("step 3: is |p'| < R_surface(θ, φ) ?", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-sh-inside-test.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def figure_sdf_slice(ref: dict) -> None:
    """Central z-slice of the final signed distance field, with the zero
    contour (the actual membrane mid-surface) marked, plus a 1D linescan
    across the boundary showing phi crossing zero -- makes 'signed distance
    field' concrete for readers who haven't met one before."""
    z_mid = SHAPE_ZYX[0] // 2
    sl = ref["phi_field"][z_mid]
    extent_a = SHAPE_ZYX[1] * SPACING_A
    half = extent_a / 2.0

    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    vmax = np.abs(sl).max()
    im = axes[0].imshow(
        sl,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
        extent=[-half, half, -half, half],
    )
    axes[0].contour(
        sl,
        levels=[0.0],
        colors="k",
        linewidths=1.5,
        extent=[-half, half, -half, half],
        origin="lower",
    )
    axes[0].set_title(
        r"signed distance field $\phi$ (Å) -- negative inside", fontsize=10
    )
    axes[0].set_xlabel("x (Å)")
    axes[0].set_ylabel("y (Å)")
    row_idx = sl.shape[0] // 2
    fig.colorbar(im, ax=axes[0], shrink=0.8, label=r"$\phi$ (Å)")
    axes[0].axhline(0.0, color="k", linestyle=":", linewidth=1)

    x_a = np.linspace(-half, half, sl.shape[1])
    line = sl[row_idx]
    axes[1].plot(x_a, line, color=TEAL)
    axes[1].axhline(0.0, color="0.5", linewidth=1)
    sign_change = np.where(np.diff(np.sign(line)) != 0)[0]
    for idx in sign_change:
        axes[1].axvline(x_a[idx], color="k", linestyle=":", linewidth=1)
    axes[1].set_title(
        r"linescan along y=0 -- $\phi$ crosses zero at the surface", fontsize=10
    )
    axes[1].set_xlabel("x (Å)")
    axes[1].set_ylabel(r"$\phi$ (Å)")
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-sh-sdf-slice.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def figure_parameter_sweep() -> None:
    """Outline-only (phi=0 contour) parameter sweep across sh_amplitude and
    sh_max_degree -- the finished-shape counterpart to figure_degree_buildup's
    abstract (theta, phi) view."""
    configs = [
        ("amplitude=0.10", dict(sh_amplitude=0.10, sh_max_degree=8)),
        ("amplitude=0.15 (default)", dict(sh_amplitude=0.15, sh_max_degree=8)),
        ("amplitude=0.25", dict(sh_amplitude=0.25, sh_max_degree=8)),
        ("amplitude=0.40", dict(sh_amplitude=0.40, sh_max_degree=8)),
        ("degree=3", dict(sh_amplitude=0.15, sh_max_degree=3)),
        ("degree=14", dict(sh_amplitude=0.15, sh_max_degree=14)),
    ]
    fig, axes = plt.subplots(1, len(configs), figsize=(2.2 * len(configs), 2.6))
    for ax, (label, cfg) in zip(axes, configs):
        gen = MembraneGenerator(
            target_shape=SHAPE_ZYX,
            voxel_size=SPACING_A,
            shape_backend="spherical_harmonics",
            sh_axes=SH_AXES_A,
            n_lipids_per_leaflet=1,
            seed=SEED,
            **cfg,
        )
        gen.generate()
        z_mid = SHAPE_ZYX[0] // 2
        sl = gen.field.phi[z_mid].numpy()
        ax.contourf(sl < 0, levels=[0.5, 1.5], colors=[mcolors.to_hex(TEAL)])
        ax.set_title(label, fontsize=9)
        ax.axis("off")
        ax.set_aspect("equal")
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-sh-parameter-sweep.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def figure_axes_sweep() -> None:
    """sh_axes sweep: isotropic vs. elongated vs. flattened organelles.

    Uses a larger working grid than SHAPE_ZYX/SPACING_A: the elongated
    (220, 120, 120) config's long semi-axis, inflated by up to
    ~sh_amplitude's fractional perturbation, otherwise clips against
    SHAPE_ZYX's boundary (the exact failure mode
    generate_membrane_field_spherical_harmonics' own boundary check warns
    about)."""
    shape_zyx = (170, 170, 170)
    spacing_a = 4.0
    configs = [
        ("isotropic (150,150,150)", (150.0, 150.0, 150.0)),
        ("elongated (220,120,120)", (220.0, 120.0, 120.0)),
        ("flattened (90,150,150)", (90.0, 150.0, 150.0)),
    ]
    fig, axes = plt.subplots(1, len(configs), figsize=(3.2 * len(configs), 3.4))
    for ax, (label, sh_axes) in zip(axes, configs):
        gen = MembraneGenerator(
            target_shape=shape_zyx,
            voxel_size=spacing_a,
            shape_backend="spherical_harmonics",
            sh_axes=sh_axes,
            sh_amplitude=SH_AMPLITUDE,
            sh_max_degree=SH_MAX_DEGREE,
            n_lipids_per_leaflet=1,
            seed=SEED,
        )
        gen.generate()
        z_mid = shape_zyx[0] // 2
        sl = gen.field.phi[z_mid].numpy()
        ax.contourf(sl < 0, levels=[0.5, 1.5], colors=[mcolors.to_hex(TEAL)])
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        ax.set_aspect("equal")
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-sh-axes-sweep.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    ref = _reference_instance()
    figure_hero(ref)
    figure_basis_gallery()
    figure_degree_buildup(ref)
    figure_angular_grid_interp(ref)
    figure_spectrum()
    figure_inside_test(ref)
    figure_sdf_slice(ref)
    figure_parameter_sweep()
    figure_axes_sweep()
    print("done")
