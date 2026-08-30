# Bilayer profile & transmembrane proteins

![A vesicle's sampled transmembrane sites with their surface normals, and the rendered slab with bacteriorhodopsin embedded in the bilayer.](../../assets/images/cryoet-bilayer-transmembrane.png){ width="880" style="display:block;margin:1.2em auto;" }
///caption
A vesicle's sampled transmembrane sites with their surface normals, and the rendered slab with bacteriorhodopsin embedded in the bilayer.
///

[Membrane shape](../membrane-shape/index.md) produces a signed field
\(\phi\): the geometry. This page covers everything between that field
and a rendered volume: the 1D potential profile \(\psi(d)\) that turns
distance-from-the-mid-plane into volts, the anti-aliased resampling onto
the output grid, and how proteins embed in the resulting bilayer.

!!! info "Source"
    `specter.specimen.membrane._profile`, `._raster`, `._placement`.
    `docs-figures/cryoet_specimen_bilayer.py` produces the figures.

## From shape to potential

Every membrane voxel's density is \(\psi(\phi(\mathbf{x}))\): a single 1D
lookup on the signed distance field. The generator does the
atomic-resolution work once and reuses it everywhere through that
lookup.

\(\psi(d)\) is measured, not parameterised. `build_reference_lipid_patch`
builds a schematic atomic bilayer patch, `PotentialBuilder` renders it,
and `compute_bilayer_profile` averages that render laterally over the
central 60% of the patch, which is the region carrying the intended
areal density of one lipid per 65 Å². The result is a profile in volts,
on the same scale as any `PotentialBuilder`-rendered protein template, so
a membrane and an inserted transmembrane protein are directly
commensurate in the same volume.

![Left, the measured profile against the two-Gaussian form it replaced. Right, the effect of bilayer_thickness.](../../assets/images/cryoet-bilayer-profile.png){ width="900" style="display:block;margin:1.2em auto;" }
///caption
Left, the measured profile against the two-Gaussian form it replaced. Right, the effect of bilayer_thickness.
///

The profile is a slab with headgroup peaks on it, not two isolated lines.
Headgroups reach 8.5 V and the acyl core sits near 5.4 V, both above
amorphous ice at roughly 4.6 V, so a bilayer scatters more strongly than
its surroundings across its whole thickness.

That the core exceeds ice is not obvious, and reasoning from mass density
gets the sign wrong: hydrocarbon at 0.9 g/cm³ is the less dense material.
For electrons the relevant currency is not electron density. Mott-Bethe
leaves a diffuse one-electron atom screening its own proton poorly at low
\(k\), so hydrogen scatters about 2.5× what carbon does per unit mass,
and the acyl core is the most hydrogen-rich region of the molecule. The
same effect governs the [amorphous ice](../ice.md) kernel.

`bilayer_thickness` rescales the measured profile along \(z\) at fixed
amplitude, so the integrated potential scales with thickness. That is the
physical relationship rather than a convention: lipid volume is
conserved, so a bilayer of thickness \(t\) built from lipids of volume
\(V\) occupies \(2V/t\) per lipid and deposits \(t \cdot (\text{scattering
per lipid}) / V\) per unit area. Normalising the integral instead would
make a thinner membrane denser.

### The two-Gaussian profile this replaced

Until 2026-08-31 \(\psi(d)\) was two Gaussians standing on vacuum, one per
leaflet, scaled by a fitted amplitude. It is worth knowing why, because
the construction is superficially reasonable and reproduces the
"railroad track" appearance of a real bilayer micrograph directly.

That appearance is an *imaging* signature. Two dark lines arise because
the headgroups are locally denser and because CTF phase contrast enhances
the gradient, not because the space between them is empty. Building the
appearance into the specimen therefore counts it twice, since the
simulator then images that specimen, and it deletes the acyl core the
bilayer actually holds.

The cost is invisible where it was looked for and dominant where it was
not. In a single slice the headgroup peaks dominate and the two profiles
look comparable. In a projection, which integrates through the membrane,
the analytic form carries 53 V·Å against the measured profile's 254 — a
4.8× shortfall in projected membrane contrast.

## Amplitude: why there is no longer one

There is no amplitude scalar. \(\psi(d)\) is used as measured, in volts,
and nothing rescales it to a peak.

One was fitted until 2026-08-30, by rendering each unique species in the
lipid template as a **single isolated atom** and taking the tallest raw
peak, most often the phosphate phosphorus. The reasoning was that
averaging the full patch laterally diluted that peak roughly 20-fold and
so could not be the right reference.

Both halves of that were wrong. The dilution was measured over the whole
patch, including its under-populated jittered edges, which
`compute_bilayer_profile` never averages over. And an atom's own centre
is a cusp: phosphorus measures 400.5 V at 0.5 Å sampling, 26.9 V at 2.0 Å
and 4.1 V at 4.0 Å, so it has no grid-independent value and cannot
calibrate anything at any voxel size. The fitted amplitude overstated the
bilayer 5.1×.

The rule that failure illustrates still governs anything compared against
this profile: \(\psi(d)\) is a plane average, the mean potential over a
plane at height \(d\), and a plane average is commensurate only with
another plane average.

A general principle recurs here: **the smoothness real cryo-ET membranes
show comes from the microscope's resolution limits, applied to the ground
truth after the generator builds it** (CTF, multislice, detector MTF; see
[Forward simulation](../forward-simulation.md)), not from pre-averaging
the ground truth itself.

### What anchors the scale

One identity, with no free parameter. Integrated across the bilayer,
\(\psi\) is fixed by chemistry alone:

\[
\int \psi(z)\,dz = \frac{2 \times (\text{scattering per lipid})}
                        {\text{area per lipid}}
\]

The lipid template carries the exact stoichiometry of POPC,
C\(_{42}\)H\(_{82}\)NO\(_8\)P, and the scattering integral per element is
a property of the scattering tables. Together they predict 254.5 V·Å;
the rendered patch measures 254.0. The same identity applied to a protein
predicts apoferritin's mean inner potential as 7.03 V against 7.00 V
measured by rendering it, so the check is not circular.

Hydrogen is load-bearing in that census. POPC's 82 hydrogens are a
quarter of the molecule's total scattering, and a template omitting them
carries only 59% of a real lipid.

There is deliberately no knob to scale a membrane's contrast up or down.
A `membrane_scale_range` existed until 2026-08-31, drawing a per-instance
multiplier from `(0.5, 1.0)`, and it was part of how the calibration error
above stayed hidden: a bilayer 5.1x too bright, dimmed 0.75x on average,
reads as merely wrong rather than obviously wrong. Contrast that varies
between real membranes comes from defocus, ice thickness and lipid
composition, which are modelled in their own right; an arbitrary
multiplier on one species' potential is not a model of any of them.

## Anti-aliased rasterization

The field lives on its own fine working grid; the output volume is
often coarser. Resampling between them is where a real, previously
observed failure mode lives.

![The same membrane field rasterized at 4, 8 and 12 Å voxels, point-sampled on top and anti-aliased below, along a line through the vesicle wall.](../../assets/images/cryoet-bilayer-antialias.png){ width="900" style="display:block;margin:1.2em auto;" }
///caption
The same membrane field rasterized at 4, 8 and 12 Å voxels, point-sampled on top and anti-aliased below, along a line through the vesicle wall.
///

Point-sampling the fine density directly (top row) keeps the two leaflets
sharp and full height at *every* voxel size. The peaks land wherever the
sample grid happens to catch them, so the apparent leaflet separation
wanders with voxel size. The physical peak-to-peak spacing is fixed, so
that variation is pure aliasing.

Low-pass filtering first (bottom row), a Gaussian matched to the output
voxel footprint, \(\sigma = 0.5 \times\) the output spacing, gives the
physically correct behaviour instead: the leaflets lose amplitude and
merge into one broad peak as resolution drops, which is what a real
bilayer does when you stop resolving it.

## Transmembrane placement

Because \(\phi\) is a dense signed field defined everywhere rather than a
mesh, the local surface normal at any point is its gradient: exact, and
available without a mesh or a KD-tree. Placement uses that directly:

1. **Project onto the surface.** Newton iteration on the field:
   \(\mathbf{x} \leftarrow \mathbf{x} - \phi(\mathbf{x})\,\nabla\phi(\mathbf{x})\).
2. **Reject crowded sites**, enforcing `min_transmembrane_spacing`
   centre-to-centre.
3. **Orient.** Align the protein's canonical axis to the local normal,
   then apply a free random spin about that same axis (the two-step
   construction CTS and polnet both use).
4. **Set the depth.** Translate along the aligned axis so the
   transmembrane span's midpoint sits at the bilayer mid-plane. Never
   rescale: hydrophobic mismatch between a protein's real span and the
   local bilayer thickness is a genuine biophysical effect, not something
   to hide by stretching the structure.

"Canonical axis" means the structure's **longest principal axis of
inertia**, not its file z-axis. A deposited structure's native axes are an
artifact of how it was solved; for a membrane protein with a large
asymmetric extramembrane domain, that domain's real spatial extent is the
physically meaningful "sticks out of the membrane" direction. Confirmed
empirically on a real structure: its principal axis diverged far from
its native z-axis, and depth-centering along the wrong one left the
extramembrane domain pointing off at an angle. The generator doesn't
resolve the *sign* of that axis. With no topology information
(cytoplasmic vs. extracellular) to break the symmetry, which end becomes
\(+z\) is an arbitrary but deterministic PCA-sign choice.

The generator chooses species per site by weighted random draw on
`frequency`. The requested site count isn't guaranteed: if the surface is
too small for that many well-spaced sites, or the working grid is too
coarse for reliable surface projection, the generator warns and places
what it found.

## Parameters at a glance

| Parameter | Meaning | Default |
|---|---|---|
| `bilayer_thickness` | Phosphate-to-phosphate leaflet spacing \(t\), Å | 38.0 |
| `bilayer_layer_sigma_a` | Additional Gaussian broadening along \(z\), Å | 0.0 |
| `min_transmembrane_spacing` | Minimum centre-to-centre site spacing, Å | 40.0 |
| `transmembrane_occupancy_fraction` | Surface occupancy target for site sampling | 0.05 |
| `frequency` (per spec) | Relative weight among transmembrane species | 1 |
| `tm_span_mask` (per spec) | Atom mask selecting the membrane-spanning region | None (full z-extent) |

\(t\) defaults to 38.0 Å, inside the published 36–39 Å range for fluid
phosphatidylcholine. It was 30.0 until 2026-08-30, the midpoint of
polnet's `MB_THICK_RG` (25–35 Å) — a range sitting entirely below the
experimental one. The reference template's own spacing is 40 Å, so the
default rescales it by 0.95.

`bilayer_layer_sigma_a` is *additional* broadening, and defaults to none.
It set the leaflet peak width of the analytic profile; the measured
profile carries whatever width its atomic model implies, and the
rasteriser anti-aliases to the render grid separately, so there is no
width left to set here — only blur to add.

## Limitations

- **The reference lipid template is schematic**, and it now sets the
  entire profile rather than just a scale factor. Its census is exact POPC
  stoichiometry, so the integrated potential is anchored to chemistry, but
  the per-leaflet atom z-offsets are hand-picked from bilayer structural
  biology plus jitter, not a relaxed or MD-equilibrated structure. The
  distribution of that integral along \(z\) is therefore modelled, and
  the template's own 40 Å phosphate spacing sits just above the published
  36–39 Å range. Sourcing \(\psi(z)\) from an MD snapshot or from
  published component density profiles is the open item.
- **No leaflet asymmetry.** \(\psi(d)\) is symmetric about the mid-plane;
  real membranes are not.
- **No lipid composition.** One profile per membrane instance, with no
  notion of rafts, cholesterol, or local thickness variation.
- **Transmembrane instances get no voxel labels.** Their density is in the
  volume, and the generator records their placements, but they do not
  appear in `instance_labels`.
- **Depth alignment defaults to the full z-extent** when you don't give a
  `tm_span_mask`, which is wrong for a protein with a large soluble domain
  on one side only.

## References

- Martinez-Sanchez, A., Lamm, L., Jasnin, M., & Phelippeau, H. (2024).
  Simulating the cellular context in synthetic datasets for cryo-electron
  tomography. *IEEE TMI* 43(11), 3742–3754.
  [polnet source](https://github.com/anmartinezs/polnet).
- Purnell, C., et al. (2023). Rapid synthesis of cryo-ET data for training
  deep learning models. *bioRxiv* 2023.04.28.538636.
  [CTS source](https://github.com/carsonpurnell/cryotomosim_CTS).
