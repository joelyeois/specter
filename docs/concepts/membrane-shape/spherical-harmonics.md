# Membrane shape: spherical harmonics

![A random spherical-harmonic-perturbed organelle, rendered as a shaded 3D isosurface.](../../assets/images/membrane-sh-hero.png){ width="750" style="display:block;margin:1.2em auto;" }
///caption
A random spherical-harmonic-perturbed organelle, rendered as a shaded 3D isosurface.
///

`MembraneGenerator` builds vesicles, nuclei, and mitochondria by perturbing
an ellipsoid with a random spherical-harmonic expansion
(`shape_backend="spherical_harmonics"`, the default). This page explains
how `MembraneGenerator` builds that shape.

!!! info "Source"
    Walks through `specter.specimen.membrane._field_spherical_harmonics`.
    Figures are produced by `docs-figures/membrane_shape.py`, which calls
    the same private helpers as the real code path.

## Star-convex surfaces

The method requires [star-convexity](https://en.wikipedia.org/wiki/Star_domain):
every ray from the organelle's center crosses the surface exactly once. True
for vesicles, nuclei, and most mitochondria.

A star-convex surface reduces to one scalar function of direction:

\[
R(\theta, \phi)
\]

## Surface as a sum of harmonics

\[
R(\theta, \phi) = 1 + a \sum_{l=2}^{L} \sum_{m=-l}^{l} c_{lm}\, Y_l^m(\theta, \phi)
\]

- \(Y_l^m\): real [spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics), degree \(l\), order \(m\).
- \(a\) = `sh_amplitude`: perturbation size, as a fraction of the base radius.
- \(L\) = `sh_max_degree`: highest degree included.
- \(l=0\) (uniform offset) and \(l=1\) (pure translation) carry no shape
  information and are excluded.

![Six real spherical harmonics rendered as colored, radius-perturbed unit spheres.](../../assets/images/membrane-sh-basis-gallery.png){ width="720"  style="display:block;margin:1.2em auto;" }
///caption
Six real spherical harmonics rendered as colored, radius-perturbed unit spheres.
///

The organelle's shape is a sum of these patterns with random coefficients
\(c_{lm}\).

## Coefficient spectrum

\[
c_{lm} \sim \mathcal{N}\!\left(0,\ \big[l(l+1)\big]^{-p}\right)
\]

\(p\) is `sh_spectrum_power`. \(p = 2\) (default) reproduces the
[Helfrich](#references) (1973) thermal bending-mode spectrum of a lipid
bilayer: long-wavelength undulations dominate, short-wavelength wrinkles are
suppressed. Lower \(p\) shifts weight toward finer, less membrane-like
roughness.

![Var(a_lm) vs. harmonic degree l, for several values of spectrum_power.](../../assets/images/membrane-sh-spectrum.png){ width="420"  style="display:block;margin:1.2em auto;" }
///caption
Var(a_lm) vs. harmonic degree l, for several values of spectrum_power.
///

SPECTER rescales the coefficients so \(\sum_{l,m} c_{lm}^2 = 1\). Since the
\(Y_l^m\) are orthonormal,
[Parseval's theorem](https://en.wikipedia.org/wiki/Parseval%27s_theorem)
fixes the perturbation's RMS to exactly \(1\) regardless of
`sh_max_degree`/`sh_spectrum_power`. `sh_amplitude` alone sets the overall
scale.

![The same random coefficients, with the harmonic sum truncated at increasing degree L.](../../assets/images/membrane-sh-degree-buildup.png){ width="900"  style="display:block;margin:1.2em auto;" }
///caption
The same random coefficients, with the harmonic sum truncated at increasing degree L.
///

Same random draw, increasing \(L\): more terms add finer wrinkles on top of
the same broad shape.

## Radius function to solid

For every voxel: transform its position into a frame scaled by the
organelle's semi-axes `sh_axes`, giving a local radius \(|p'|\) and
direction \((\theta, \phi)\). The voxel is inside if

\[
|p'| < R(\theta, \phi)
\]

![Central slice of the raw boolean inside/outside test, before any distance transform.](../../assets/images/membrane-sh-inside-test.png){ width="380"  style="display:block;margin:1.2em auto;" }
///caption
Central slice of the raw boolean inside/outside test, before any distance transform.
///

## Solid to signed distance field

Protein placement and bilayer rasterization need the distance to the
surface, which an inside/outside test does not give. SPECTER builds a
[signed distance field](https://en.wikipedia.org/wiki/Signed_distance_function)
\(\phi\) (negative inside, positive outside, zero at the surface) via two
[Euclidean distance transforms](https://en.wikipedia.org/wiki/Distance_transform):

\[
\phi = \mathrm{EDT}(\text{outside}) - \mathrm{EDT}(\text{inside})
\]

SPECTER uses \(\phi\) instead of the radial residual
\(|p'| - R(\theta,\phi)\): that residual's error grows with local slope
and would distort the bilayer's calibrated thickness at high curvature.

![Central slice of the final signed distance field, with the zero contour marking the membrane surface, and a 1D linescan showing phi crossing zero there.](../../assets/images/membrane-sh-sdf-slice.png){ width="900"  style="display:block;margin:1.2em auto;" }
///caption
Central slice of the final signed distance field, with the zero contour marking the membrane surface, and a 1D linescan showing phi crossing zero there.
///

Left: \(\phi\) over one slice (red = inside, blue = outside), zero contour
in black. Right: a linescan through the center, showing \(\phi\) crossing
zero at the surface.

## Fast harmonic synthesis

Evaluating the harmonic sum at every voxel's own direction is expensive
(60 s at a 10M-voxel grid). Since the perturbation is
[band-limited](https://en.wikipedia.org/wiki/Bandlimiting) to degree
\(L\), SPECTER instead synthesizes it once on a small \((n_\theta, n_\phi)\)
grid and [bilinearly interpolates](https://en.wikipedia.org/wiki/Bilinear_interpolation)
per voxel.

![A small coarse synthesis grid next to the bilinearly interpolated field it stands in for.](../../assets/images/membrane-sh-angular-grid-interp.png){ width="800"  style="display:block;margin:1.2em auto;" }
///caption
A small coarse synthesis grid next to the bilinearly interpolated field it stands in for.
///

Interpolation error is ~0.17% of the perturbation's peak (0.07-0.11% of its
RMS, across coefficient draws), well below the distance-transform's own
discretization noise. The angular grid costs the same regardless of how many
voxels read from it, so the saving grows with the working grid: 31x at 1M
voxels, 81x at 3M, ~180x at 10M. Measured by
`docs-figures/membrane_shape_spherical_harmonics.py --timing`.

## Parameters

| Parameter | Meaning | Default |
|---|---|---|
| `sh_max_degree` | Highest harmonic degree \(L\) | 8 |
| `sh_axes` | Physical semi-axes \((a_x, a_y, a_z)\), Å | `(300, 300, 300)` |
| `sh_amplitude` | RMS fractional radius perturbation \(a\) | 0.15 |
| `sh_spectrum_power` | Exponent \(p\) in \(\mathrm{Var}(c_{lm}) \propto [l(l+1)]^{-p}\) | 2.0 |

![Amplitude and degree swept across six organelles, outline only.](../../assets/images/membrane-sh-parameter-sweep.png){ width="900"  style="display:block;margin:1.2em auto;" }
///caption
Amplitude and degree swept across six organelles, outline only.
///

`sh_amplitude=0.15`: 0.10 is close to spherical, 0.40 produces visible
concave dimples, 0.15 is the smallest value that reads as organic without
concavity artifacts.

![Isotropic, elongated, and flattened base ellipsoids, same random perturbation.](../../assets/images/membrane-sh-axes-sweep.png){ width="700"  style="display:block;margin:1.2em auto;" }
///caption
Isotropic, elongated, and flattened base ellipsoids, same random perturbation.
///

`sh_axes` sets the base ellipsoid the perturbation rides on.

## Limitations

- **Star-convex only.** The method cannot represent branching tubules,
  self-occluding folds, or anything where a ray from the center crosses
  the surface more than once.
- **Grid resolution matters.** Below about 8 voxels per radius, surface
  sampling for protein placement can find zero sites with no other
  symptom; the generator warns when this happens.
- **The SDF has voxel-scale texture**, visible as a faint ripple under
  raking light (see the hero image), an expected consequence of computing
  distance to the nearest boundary voxel rather than the continuous
  surface. It's below the scale that affects the bilayer profile.

## References

- Helfrich, W. (1973). Elastic properties of lipid bilayers: theory and
  possible experiments. *Zeitschrift für Naturforschung C*, 28(11-12),
  693-703. [doi:10.1515/znc-1973-11-1209](https://doi.org/10.1515/znc-1973-11-1209)
