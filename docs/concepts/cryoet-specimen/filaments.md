# Filaments

![A filaments-only specimen: fourteen actin filaments scattered through the box, summed along Z.](../../assets/images/cryoet-filament-hero.png){ width="620" style="display:block;margin:1.2em auto;" }
///caption
A filaments-only specimen: fourteen actin filaments scattered through the box, summed along Z.
///

A filament species consists of a single monomer structure, replicated
along a random walk. Each monomer instance is a rigid-transformed copy
of one rendered template, so a 60-monomer F-actin filament costs one
`PotentialBuilder` render and 60 rigid-body rotations.

SPECTER ships two presets: `ACTIN_SPEC`, corresponding to F-actin, and
`PROTOFILAMENT_SPEC`, corresponding to a single microtubule
protofilament. Define custom geometries via a `FilamentSpec`.

This page covers single-strand filaments only. The complete
microtubule, with its thirteen protofilaments arranged into a closed
tube with a lumen and an A-lattice seam, gets its own page; see
[Microtubules](microtubules.md).

!!! info "Source"
    `specter.specimen.filament` (`_path`, `_placement`, `_generator`),
    stamped by `TomogramSpecimenGenerator._stamp_filaments`.
    `docs-figures/cryoet_specimen_filaments.py` produces the figures,
    calling the same functions the real code path does.

## The path: a persistent random walk

A walk generates each filament's monomer centres: it advances a fixed
distance `step` (the monomer's axial rise) along the current direction,
then rotates that direction by a random angle drawn uniformly from
\([0, \texttt{flex\_deg}]\) about a random axis perpendicular to it:

\[
p_{i+1} = p_i + s\,d_i, \qquad
d_{i+1} = R(\hat{a}_i,\ \theta_i)\,d_i, \quad
\theta_i \sim U(0, \texttt{flex\_deg}),\ \hat{a}_i \perp d_i
\]

Curvature stays bounded *per step* and unbounded *cumulatively*: the
worm-like-chain regime.

![Filament paths at four flex_deg values, all with the same monomer count and axial rise, drawn at a common scale.](../../assets/images/cryoet-filament-flex-sweep.png){ width="900" style="display:block;margin:1.2em auto;" }
///caption
Filament paths at four flex_deg values, all with the same monomer count and axial rise, drawn at a common scale.
///

All four paths shown have identical contour length; only the tangling
differs. That makes `flex_deg` the persistence knob: raise it, and the
same length of polymer occupies a smaller end-to-end span.

This construction differs from the
[swept-spline membrane backend](../membrane-shape/swept-spline.md)'s
path, which blends the previous direction with a fresh random one at a
fixed weight. Here, the turn is a bounded rotation of the previous
direction, so `flex_deg` caps how sharply a filament can kink over a
single monomer, the physically relevant quantity for a stiff polymer.

## Orientation: tangent alignment plus screw twist

Each monomer's rotation combines two operations, reflecting the screw
symmetry of helical polymers:

1. **Align to the tangent.** Rotate the monomer's canonical \(+Z\) onto
   the local path tangent (monomer \(i\) → monomer \(i+1\); the last
   monomer reuses its predecessor's tangent).
2. **Apply the twist.** Rotate about that same axis by an accumulated
   \(i \cdot \texttt{twist\_deg}\).

SPECTER fixes the canonical \(+Z\) axis during template construction,
pre-rotating the monomer's coordinates so its **longest principal
axis** points along \(+Z\). This is an approximation; see
[Limitations](#limitations) for a fuller discussion.

`twist_deg` is what separates a helical polymer from a plain stack:

![The same straight monomer chain rendered with twist_deg = 0 and with F-actin's real 166.15 degrees per monomer.](../../assets/images/cryoet-filament-twist.png){ width="900" style="display:block;margin:1.2em auto;" }
///caption
The same straight monomer chain rendered with twist_deg = 0 and with F-actin's real 166.15 degrees per monomer.
///

Without twist, every monomer presents the same face and the strand
reads as a uniform ridge. With F-actin's measured 166.15° per subunit,
the projection picks up the crossover pattern you see in a real F-actin
filament.

## Presets

![F-actin and a microtubule protofilament rendered at the same scale.](../../assets/images/cryoet-filament-presets.png){ width="900" style="display:block;margin:1.2em auto;" }
///caption
F-actin and a microtubule protofilament rendered at the same scale.
///

| Preset | `code` | `step` (Å) | `flex_deg` | `twist_deg` | `n_monomers` |
|---|---|---|---|---|---|
| `ACTIN_SPEC` | 1J6Z (G-actin) | 27.3 | 12.0 | 166.15 | (20, 60) |
| `PROTOFILAMENT_SPEC` | 1TUB (αβ-tubulin dimer) | 85.0 | 3.0 | 0.0 | (10, 30) |

`step` and `twist_deg` are measured values (F-actin's helical repeat
from Holmes/Egelman; tubulin's 85 Å dimer repeat). `flex_deg` is a
tuned value in both cases, not a persistence-length measurement.

`twist_deg = 0` is correct for the protofilament preset: a single
protofilament does not itself twist. Its 13-protofilament tube geometry
and small supertwist fall outside this preset's scope;
`PROTOFILAMENT_SPEC` gives you a single protofilament only. For the
full tube, see [`MicrotubuleSpec`](microtubules.md).

## Placement

Every instance starts at a uniformly random point in the volume and
walks in a uniformly random initial direction, with no rejection
sampling and no obstacle awareness:

- A filament that wanders **outside the volume** gets truncated at
  render time, the same edge behaviour applied to placed particles.
- SPECTER drops a monomer landing **inside the carbon film** after the
  fact, leaving a gap rather than redirecting the walk.
- Filaments do **not** avoid the membrane shell or each other, and
  aren't region-gated: a filament can cross a vesicle wall and continue
  through its lumen.

Everything placed after filaments avoids them, though: beads and the
protein-fill stage both treat placed filament voxels as obstacles.

## Parameters at a glance

| Parameter | Meaning | Default |
|---|---|---|
| `code` | PDB ID or local file for the monomer | — |
| `step` | Axial rise per monomer, Å | — |
| `flex_deg` | Maximum per-step turn, degrees | — |
| `twist_deg` | Helical twist per monomer, degrees | 0.0 |
| `n_copies` | Independent filament instances of this species | 1 |
| `n_monomers` | Monomers per instance; an int, or a `(min, max)` drawn per instance | (10, 30) |

In a TOML config these are `[[filaments]]` tables, or `actin = true` for
the `ACTIN_SPEC` preset; see
[Build a tomogram specimen](../../user-guide/build-tomogram.md).

## Limitations

- **No branching.** Each filament is a single path; you cannot
  represent Y-junctions or networks.
- **One protofilament only.** A full microtubule, with its
  13-protofilament tube geometry, gets
  [its own component](microtubules.md).
- **No collision avoidance during the walk.** SPECTER doesn't redirect
  monomers landing inside another object; it resolves overlaps only
  through the post-hoc dropping described under [Placement](#placement).
- **The stacking axis is a principal-axis approximation.** The monomer's
  canonical \(+Z\) comes from its longest principal axis, not from
  inter-subunit contact geometry. A real protofilament or actin docking
  axis follows how neighbouring subunits contact each other, and the
  inertia tensor of an isolated monomer doesn't capture that. A PDB
  structure's native coordinate axes carry no inherent relationship to
  its stacking direction either, so some convention has to apply; this
  is that convention, not a structurally validated docking.

## References

- Holmes, K. C., Popp, D., Gebhard, W., & Kabsch, W. (1990). Atomic model
  of the actin filament. *Nature*, 347(6288), 44–49.
- Egelman, E. H., Francis, N., & DeRosier, D. J. (1982). F-actin is a helix
  with a random variable twist. *Nature*, 298(5870), 131–135.
- Purnell, C., et al. (2023). Rapid synthesis of cryo-ET data for training
  deep learning models. *bioRxiv* 2023.04.28.538636.
  [CTS source](https://github.com/carsonpurnell/cryotomosim_CTS).
