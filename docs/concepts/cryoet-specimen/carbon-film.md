# Carbon support film

![A carbon film summed along Z: a strip of film intruding from the bottom edge of the frame, with a rough, granular rim.](../../assets/images/cryoet-carbon-hero.png){ width="620" style="display:block;margin:1.2em auto;" }

The support film is a roughly planar slab of amorphous carbon with a
circular hole cut through it. It is painted into the canvas before
anything else, and everything placed afterwards works around it.

Its geometry is an **alpha shape** over a jittered 3D point cloud, not an
analytic boundary. This is what gives it a rim that differs from
z-slice to z-slice, with islands and overhangs, rather than a swept 2D
outline.

!!! info "Source"
    `specter.specimen._carbon` (`CarbonFilmGenerator`, `edge_hole_center`),
    a from-scratch replication of CryoTomoSim's `gen_carbon.m`/
    `carbonshape`. Figures are produced by
    `docs-figures/cryoet_specimen_carbon_film.py`.

## Why the hole is bigger than the picture

A Quantifoil R1.2/1.3 grid (the standard for high-resolution
single-particle and cryo-ET collection) has 1.2 µm holes, so a **6000 Å
radius**. That is far larger than any single tomogram's field of view. The
realistic cases are therefore: entirely inside a hole (no carbon at all),
entirely on the carbon, or catching one hole's edge near a frame border.
A small hole fully contained in the frame is not a thing you see.

![Left: three hole circles drawn to scale against the field of view. Right: the resulting film at three edge_fraction values.](../../assets/images/cryoet-carbon-hole-geometry.png){ width="900" style="display:block;margin:1.2em auto;" }

Hand-placing that means solving for where a huge circle's boundary has to
sit so that a strip of a specific width lands at a specific frame edge.
`edge_hole_center` does that solve: give it `edge_fraction` (the fraction
of the frame that ends up carbon) and `edge_side`, and it returns the
`hole_center` to pass alongside the real `hole_radius`. At real Quantifoil
scale the boundary crossing the frame then reads as close to a straight
edge, which is what it looks like in real data.

`edge_fraction` also accepts a `(low, high)` range, drawn per run: real
images don't all happen to catch the same amount of a hole's edge. The
shipped default is `(0.02, 0.05)`.

## The rim

Points are seeded uniformly across the frame's footprint (plus a 50 Å pad,
so the rough boundary doesn't clip visibly at the frame's own edge) and
through the film's thickness, at a fixed physical seed **density**:
`_SEED_VOLUME_PER_POINT` = 18000 Å³/point, about 26 Å mean spacing. Points
inside the hole radius are dropped, and every survivor is then displaced by
an isotropic 3D vector whose magnitude is drawn uniformly from
\([0, \texttt{edge\_roughness}]\). An alpha shape (α = 40 Å) over the
result gives the film's solid.

Because the jitter is large relative to the seed spacing, points shuffle
past their neighbours, and the alpha complex turns that into a boundary
that is correlated at the seed scale and topologically nontrivial:

![Rim detail at edge_roughness 0, 60 and 150 Å.](../../assets/images/cryoet-carbon-roughness.png){ width="900" style="display:block;margin:1.2em auto;" }

At the shipped default of 60 Å, the boundary's radial standard deviation
comes out to roughly a third of that, ~20 Å. Even `edge_roughness = 0` is
not a clean circle: the seeds are a Poisson cloud, so the alpha shape is
already ragged at the seed spacing.

Being a genuine 3D construction rather than a per-angle function of the
outline, the boundary is different at different heights in the slab:

![Three z-slices through the same film, showing different boundaries, detached islands and overhanging lips.](../../assets/images/cryoet-carbon-slices.png){ width="900" style="display:block;margin:1.2em auto;" }

A simpler analytic construction, jittering a circle's radius per angle and
interpolating, cannot produce this: it gives flat top and bottom faces, a
spatially constant density, and a rim that is a pure function of angle, so
every z-slice would share the exact same outline. It would also need a
fixed point count, which degrades to sparse speckle as the volume grows.
Scaling the point count with a physical seed density instead means it does
not thin out at larger volumes or finer grids.

## Density

Carbon atoms are sampled uniformly inside the alpha shape's tetrahedra and
deposited with a **trilinearly split flat weight**, not a per-atom
scattering calculation. The weight is
`atom_potential_integral / voxel_size**3`, one atom's real, physical
potential integral spread over one voxel's volume, so by the mean-field
relation \(V_0 = n \int V_{\text{atom}}\), the bulk result is correct at
any placed density and any voxel size. What it lacks is per-atom radial
structure, which nothing downstream resolves in a support film anyway. A
real per-atom physics path was prototyped separately and measured at
~40× more expensive at equal atom count for no difference in the bulk
answer.

![Left: mean inner potential at three placed-density fractions against the literature range for amorphous carbon. Right: the rendered film's own voxel histogram.](../../assets/images/cryoet-carbon-calibration.png){ width="900" style="display:block;margin:1.2em auto;" }

Atoms are placed at **0.7× bulk number density**, not bulk. That is a
calibration, not a fudge factor for appearance: the independent-atom model
carries no bonding correction, so depositing full bulk density overshoots
the 7.8–9.1 V mean inner potential reported for amorphous carbon. 0.7×
lands at ≈8.4 V, matching a real per-atom-physics measurement of 8.56 ±
1.4 V to within noise.

## Interaction with everything else

The film is painted first, into an empty canvas. Downstream:

- **Membranes and filaments** are placed carbon-aware. Membranes avoid it
  outright via collision-rejecting placement (a bounding-sphere
  approximation, so an irregular organelle's true rendered shape can
  still graze it -- whatever part of an instance's density would land on
  carbon regardless is zeroed just before merging, as a safety net, so
  the volume and that instance's own shell label exclude it
  consistently); filament monomers landing inside it are dropped after
  the fact (see [Filaments](filaments.md)).
- **Beads and protein fill** avoid it for free: the region classifier
  reads carbon's high density as `shell`, the same bucket a bilayer
  occupies, and nothing is packed into `shell`.

Hole *placement* is deterministic and controllable: `edge_hole_center`
solves directly for the required `hole_center` rather than sampling one.

## Parameters at a glance

| Parameter | Meaning | Default |
|---|---|---|
| `thickness` | Film thickness, Å | 150.0 |
| `hole_radius` | Real physical hole radius, Å | 6000.0 (Quantifoil R1.2/1.3) |
| `edge_fraction` | Fraction of the frame that ends up carbon; scalar or `[low, high]` | (0.02, 0.05) |
| `edge_side` | Which frame edge the carbon enters from: `left`/`right`/`top`/`bottom`/`random` | `random` |
| `edge_roughness` | Rim jitter magnitude, Å | 60.0 |

Fixed rather than exposed: the alpha radius (40 Å), the seed density, the
lateral pad, and the 0.7× placed-density fraction. All are either measured
calibrations or descriptions of the algorithm rather than of a specimen.

At most one `[[carbon_film]]` table per run.

## Limitations

- **No graphene or gold-foil supports**, and no fenestrated or lacey
  carbon: one circular hole in an amorphous slab.
- **The slab is flat.** Real films buckle and wrinkle; the only z
  structure here is the seed-cloud jitter.
- **No filament steering.** As above, monomers inside the film are
  dropped, not routed around.
- **Deposition has no per-atom radial structure**, by design.

## References

- Purnell, C., et al. (2023). Rapid synthesis of cryo-ET data for training
  deep learning models. *bioRxiv* 2023.04.28.538636.
  [CTS source](https://github.com/carsonpurnell/cryotomosim_CTS).
- [Quantifoil circular holes](https://www.quantifoil.com/products/quantifoil/quantifoil-circular-holes):
  the R1.2/1.3 hole spec.
