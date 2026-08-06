# Membrane shape: spherical harmonics

<div class="grid" markdown>

![A random spherical-harmonic-perturbed organelle, rendered as a shaded 3D isosurface.](../assets/images/membrane-sh-hero.png){ width="420" }

<div markdown>
Every organic membrane in SPECTER -- a vesicle, a nucleus, a roughly-spherical
mitochondrion -- starts life as a solid, and this page is about how that
solid's *shape* is built. The bilayer itself (the two-leaflet density
profile that gets rasterized onto it) is a separate, shape-independent step
covered elsewhere; here we only care about the boundary surface those
leaflets follow.

`shape_backend="spherical_harmonics"` is the default in
`specter.specimen.membrane.MembraneGenerator`. It builds that boundary as
a random perturbation of an ellipsoid, expanded in **spherical harmonics** --
the same basis functions used for planetary gravity fields, atomic orbitals,
and (the actual reason they end up here) the thermal shape fluctuations of
real lipid vesicles.
</div>

</div>

!!! info "Source"
    This page walks through
    `specter.specimen.membrane._field_spherical_harmonics`. Every figure
    below is produced by
    `docs-figures/membrane_shape.py`, which calls
    the same private helper functions the real code path uses (never an
    independent reimplementation of the math), so the figures cannot
    silently drift from what's actually shipped.

## Why this backend exists

A membrane-bound organelle only needs one property to be describable this
way: **star-convexity**. If you stand at the organelle's center and look out
in any direction, the surface is crossed exactly once. That's true for
vesicles, nuclei, and most mitochondria; it is *not* true for branching
tubules, folded ER sheets, or anything that self-occludes from its own
center.

For everything that *is* star-convex, this buys something valuable: the
whole surface collapses to a single scalar function of direction,

\[
R(\theta, \phi)
\]

-- the distance from the center to the surface, as a function of the polar
angle \(\theta\) and azimuthal angle \(\phi\). No mesh, no point cloud, no
faceting to clean up afterward. SPECTER has two other shape backends
(`"metaball"`, a smooth-min blend of spheres, and `"alpha_shape"`, a
Delaunay-derived shape that *can* represent non-star-convex topology) --
both are now deprecated in favor of this one for the common case, precisely
because a radius function is simpler and smoother than either.

## The surface as a sum of basis functions

The surface radius is written as a baseline plus a random perturbation:

\[
R(\theta, \phi) = 1 + a \sum_{l=2}^{L} \sum_{m=-l}^{l} c_{lm}\, Y_l^m(\theta, \phi)
\]

- \(Y_l^m\): the **real** spherical harmonics -- degree \(l\), order \(m\).
  These are the actual "shapes" being added together; six of them are shown
  below, colored by sign (red positive, blue negative), with radius itself
  also perturbed by the harmonic's own value so the lobes are visible in 3D.
- \(a\) = `sh_amplitude`: how large the perturbation is, as a fraction of the
  base radius.
- \(L\) = `sh_max_degree`: the highest degree included -- higher \(L\) allows
  finer surface detail.
- \(l=0\) (a uniform radius offset) and \(l=1\) (a pure translation of the
  whole sphere) are both excluded from the sum: neither carries any *shape*
  information, so both would be wasted degrees of freedom.

![Six real spherical harmonics rendered as colored, radius-perturbed unit spheres.](../assets/images/membrane-sh-basis-gallery.png){ width="720" }

Each \((l, m)\) is one fixed pattern. The organelle's actual shape comes from
summing many of them with **random** coefficients \(c_{lm}\) -- the next
section is about how those coefficients are chosen.

## A physically motivated random draw

The coefficients aren't just uniform noise. They're drawn as

\[
c_{lm} \sim \mathcal{N}\!\left(0,\ \big[l(l+1)\big]^{-p}\right), \qquad p = \texttt{sh\_spectrum\_power}
\]

i.e. independent, zero-mean Gaussians whose *variance* falls off with
degree. This is not an arbitrary choice: \(p = 2\) (the default) reproduces
the **Helfrich (1973)** thermal bending-mode spectrum of a lipid bilayer at
equilibrium, where long-wavelength undulations dominate and short-wavelength
wrinkles are naturally suppressed by the bilayer's own bending rigidity.
Lower \(p\) shifts weight toward higher-degree, finer-grained roughness --
less physically "membrane-like," but available if you want it.

![Var(a_lm) vs. harmonic degree l, for several values of spectrum_power.](../assets/images/membrane-sh-spectrum.png){ width="420" }

After drawing all the coefficients, they're rescaled so
\(\sum_{l,m} c_{lm}^2 = 1\) exactly. Because the \(Y_l^m\) are orthonormal
on the sphere, Parseval's theorem then guarantees the perturbation's RMS
value over the *whole* sphere is exactly \(1\), regardless of `sh_max_degree`
or `sh_spectrum_power` -- so `sh_amplitude` alone controls the perturbation's
overall scale, cleanly separated from how it's distributed across degrees.

The figure below fixes one random draw and just changes how many terms are
kept -- the same trade-off `sh_max_degree` controls, made visible directly
in \((\theta, \phi)\) space:

![The same random coefficients, with the harmonic sum truncated at increasing degree L.](../assets/images/membrane-sh-degree-buildup.png){ width="900" }

More terms add finer wrinkles on top of the same broad shape -- they don't
replace it, because the Helfrich-weighted variance keeps the low-degree
terms dominant.

## From a radius function to a solid

\(R(\theta, \phi)\) alone isn't a "shape" a renderer can use -- it needs to
become a solid on the actual working voxel grid. For every voxel, SPECTER:

1. Transforms the voxel's physical position into a frame scaled by the
   organelle's semi-axes `sh_axes_a` (equal axes give a round organelle;
   unequal axes stretch or flatten it), giving a local radius \(|p'|\) and
   direction \((\theta, \phi)\).
2. Tests whether that voxel is inside the perturbed surface:

\[
\text{inside} \iff |p'| < R(\theta, \phi)
\]

That's it -- a plain per-voxel comparison. Applied across the whole grid, it
produces a binary solid:

![Central slice of the raw boolean inside/outside test, before any distance transform.](../assets/images/membrane-sh-inside-test.png){ width="380" }

## From a solid to a signed distance field

A binary mask isn't quite what downstream code needs either -- placing
transmembrane proteins and rasterizing the bilayer both need to know not just
*whether* a point is inside, but *how far* it is from the surface, with a
sign indicating which side it's on. That's a **signed distance field (SDF)**:
a scalar field \(\phi\) that is negative inside the solid, positive outside,
zero exactly at the surface, and (ideally) satisfies the Eikonal property
\(|\nabla\phi| \approx 1\) -- i.e. it changes at exactly the rate of physical
distance, not faster or slower.

SPECTER builds this with `scipy.ndimage.distance_transform_edt`, run twice
(once on the solid, once on its complement) and subtracted:

\[
\phi = \mathrm{EDT}(\text{outside}) - \mathrm{EDT}(\text{inside})
\]

This is deliberately **not** the cheaper option of just using the radial
residual \(|p'| - R(\theta,\phi)\) directly as a stand-in distance: that
residual's error grows with the surface's local slope, and would distort the
bilayer's calibrated thickness precisely where the surface is most
interesting (high curvature, high-frequency wrinkles). A real Euclidean
distance transform doesn't have that problem.

![Central slice of the final signed distance field, with the zero contour marking the membrane surface, and a 1D linescan showing phi crossing zero there.](../assets/images/membrane-sh-sdf-slice.png){ width="900" }

The left panel is the same slice as above, now colored by \(\phi\) (red =
inside/negative, blue = outside/positive) with the surface's own zero
contour traced in black. The right panel takes a single horizontal line
through the middle and plots \(\phi\) along it directly -- this is the
clearest way to see what "signed distance field" actually means: a value
that decreases smoothly to zero as you approach the surface from outside,
keeps decreasing (now negative) as you keep going inside, and is exactly
zero at the boundary itself.

## Making it fast: a coarse grid plus interpolation

Evaluating the harmonic sum directly at every voxel's own direction is
expensive: at a realistic ~10M-voxel working grid, that step alone measured
at ~70 seconds, dominating the whole generation cost. But the perturbation
only contains information up to degree \(L\) -- it's a band-limited
function -- so it can be fully reconstructed from far fewer samples than
"one per voxel."

SPECTER exploits this: it synthesizes the harmonic sum once, on a small,
regular \((n_\theta, n_\phi)\) grid (about \(16L\) samples in \(\theta\), a
~16x oversample over the bare Nyquist floor), then bilinearly interpolates
that small grid at every voxel's own direction.

![A small coarse synthesis grid next to the bilinearly interpolated field it stands in for -- visually near-identical.](../assets/images/membrane-sh-angular-grid-interp.png){ width="800" }

The two panels above are deliberately hard to tell apart -- that's the
point. Measured directly, the interpolation error is about 0.17% of the
perturbation's own RMS scale: an order of magnitude below the discretization
noise the distance transform step already introduces, for a 30-150x
reduction in wall time.

## Parameters at a glance

| Parameter | Meaning | Default |
|---|---|---|
| `sh_max_degree` | Highest harmonic degree \(L\) -- more fine detail at higher values | 8 |
| `sh_axes_a` | Physical semi-axes \((a_x, a_y, a_z)\), Å -- equal for round, unequal for elongated/flattened | `(300, 300, 300)` |
| `sh_amplitude` | RMS fractional radius perturbation \(a\) | 0.15 |
| `sh_spectrum_power` | Exponent \(p\) in \(\mathrm{Var}(c_{lm}) \propto [l(l+1)]^{-p}\) -- 2.0 is the Helfrich thermal spectrum | 2.0 |

![Amplitude and degree swept across six organelles, outline only.](../assets/images/membrane-sh-parameter-sweep.png){ width="900" }

`sh_amplitude=0.15` (the default) was chosen from a direct visual sweep:
0.10 was barely distinguishable from a plain sphere, 0.40 produced visible
concave dimples (still a valid surface, just an unusually irregular-looking
organelle), and 0.15 was the smallest value that read as clearly organic
without concavity artifacts.

`sh_axes_a` controls the base ellipsoid the harmonic perturbation rides on
top of:

![Isotropic, elongated, and flattened base ellipsoids, same random perturbation.](../assets/images/membrane-sh-axes-sweep.png){ width="700" }

## Limitations

- **Star-convexity is a hard requirement.** Branching tubules, self-occluding
  folds, or anything where a ray from the center could cross the surface
  more than once cannot be represented at all by this backend -- use
  `"alpha_shape"` for that topology instead.
- **Grid resolution matters for downstream surface sampling.** The
  transmembrane protein placement step (`_placement.sample_surface_sites`)
  relies on the SDF closely satisfying the Eikonal property near the
  surface, which only holds as well as the working grid resolves it. Below
  about 8 voxels per radius, `generate_membrane_field_spherical_harmonics`
  warns explicitly and placement may silently find zero sites.
- **The SDF has real voxel-scale texture**, visible as a faint ripple under
  raking light in an isosurface render (as in this page's own hero image) --
  an expected consequence of building the field from
  `distance_transform_edt` on a *binary* mask (exact distance to the nearest
  boundary *voxel*, not to the continuous analytic surface), not a bug. It's
  well below the scale that affects the calibrated bilayer profile, which
  only ever samples within a thin band near the zero level set.
