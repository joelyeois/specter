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

!!! example "In code"
    This page is the theory behind [Cellular specimens for cryo-ET](../../tutorials/cellular-specimens.md), which shows it step by step with code and figures.

!!! info "Source"
    `specter.specimen.filament` (`_path`, `_placement`, `_generator`),
    stamped by `TomogramSpecimenGenerator._stamp_filaments`.
    `docs-figures/cryoet_specimen_filaments.py` produces the figures,
    calling the same functions the real code path does.

## The path: a persistent random walk

A walk generates each filament's monomer centres, following a filament
script in the CryoTomoSim source (Purnell et al. 2023): it advances a fixed
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
   \(i \cdot \texttt{twist\_deg}\), by the right-hand rule about the
   direction of growth. A positive `twist_deg` builds a right-handed helix
   and a negative one a left-handed helix, the sign convention of
   deposited helical parameters, which pair a twist with a positive rise.

SPECTER fixes the canonical \(+Z\) axis during template construction,
pre-rotating the monomer's coordinates so its **longest principal
axis** points along \(+Z\). This is an approximation; see
[Limitations](#limitations) for a fuller discussion.

`twist_deg` is what separates a helical polymer from a plain stack:

![The same straight monomer chain rendered with twist_deg = 0 and with F-actin's measured -166.6 degrees per monomer.](../../assets/images/cryoet-filament-twist.png){ width="900" style="display:block;margin:1.2em auto;" }
///caption
The same straight monomer chain rendered with twist_deg = 0 and with F-actin's measured -166.6 degrees per monomer.
///

Without twist, every monomer presents the same face and the strand
reads as a uniform ridge. With F-actin's measured −166.6° per subunit
(Dominguez & Holmes 2011), the projection picks up the crossover pattern
of a real F-actin filament.

The sign carries the hand. F-actin's one-start (genetic) helix is
left-handed: each subunit turns −166.6° about the filament axis for
+27.5 Å of rise. Superposing consecutive subunits of the deposited
filaments 6DJO (Chou & Pollard 2019) and 7BT7 (Kumari et al. 2020) gives
−166.6° to −166.8° for 27.5–27.6 Å, the same sign. Because the twist
is close to 180°, every second subunit lies on one of two long-pitch
strands, which wind the other way: \(2 \times (-166.6^\circ) + 360^\circ =
+26.8^\circ\) per 55 Å, a right-handed pair of strands crossing every
~36 nm. The idealised twist of −166.15°, which repeats after exactly 13
subunits in 6 turns, is a drawing convenience rather than a measurement
(Dominguez & Holmes 2011).

## Presets

![F-actin and a microtubule protofilament rendered at the same scale.](../../assets/images/cryoet-filament-presets.png){ width="900" style="display:block;margin:1.2em auto;" }
///caption
F-actin and a microtubule protofilament rendered at the same scale.
///

| Preset | `code` | `step` (Å) | `flex_deg` | `twist_deg` | `n_monomers` |
|---|---|---|---|---|---|
| `ACTIN_SPEC` | 1J6Z (G-actin; Otterbein et al. 2001) | 27.5 | 12.0 | −166.6 | (20, 60) |
| `PROTOFILAMENT_SPEC` | 1TUB (αβ-tubulin dimer; Nogales et al. 1998) | 82.0 | 3.0 | 0.0 | (10, 30) |

`step` and `twist_deg` are measured values. F-actin's rise and twist come
from cryo-EM reconstructions of the filament: 27.6 Å and −166.6 ± 0.6°
(Dominguez & Holmes 2011), and 27.5 Å and 166.4° (von der Ecken et al.
2015). They refine the helix of the first atomic model of the filament
(Holmes et al. 1990); the twist of individual subunits varies about this
mean (Egelman et al. 1982), and `ACTIN_SPEC` uses the mean. The
protofilament's step is the dimer repeat of the GDP microtubule lattice,
`DIMER_REPEAT` (see [Microtubules](microtubules.md#lattice-constants)),
so a free protofilament and a protofilament in a tube stack their dimers
identically. `flex_deg` is a tuned value in both cases, taken from
CryoTomoSim (Purnell et al. 2023), not a persistence-length measurement.

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
  is that convention, not a structurally validated docking. For F-actin
  the difference is large: in the deposited filaments 6DJO and 7BT7 each
  subunit's centre sits 16 Å from the filament axis and its longest
  principal axis lies 78–81° from it, while `ACTIN_SPEC` puts the
  monomer's centre on the axis and its longest principal axis along it.
  The preset therefore has F-actin's rise, twist and hand, but not its
  subunit arrangement.

## References

- Chou, S. Z., & Pollard, T. D. (2019). Mechanism of actin polymerization
  revealed by cryo-EM structures of actin filaments with three different
  bound nucleotides. *Proceedings of the National Academy of Sciences*,
  116(10), 4265–4274. (PDB 6DJO)
  [doi:10.1073/pnas.1807028115](https://doi.org/10.1073/pnas.1807028115)
- Dominguez, R., & Holmes, K. C. (2011). Actin structure and function.
  *Annual Review of Biophysics*, 40, 169–186.
  [doi:10.1146/annurev-biophys-042910-155359](https://doi.org/10.1146/annurev-biophys-042910-155359)
- Egelman, E. H., Francis, N., & DeRosier, D. J. (1982). F-actin is a helix
  with a random variable twist. *Nature*, 298(5870), 131–135.
  [doi:10.1038/298131a0](https://doi.org/10.1038/298131a0)
- Holmes, K. C., Popp, D., Gebhard, W., & Kabsch, W. (1990). Atomic model
  of the actin filament. *Nature*, 347(6288), 44–49.
  [doi:10.1038/347044a0](https://doi.org/10.1038/347044a0)
- Kumari, A., Kesarwani, S., Javoor, M. G., Vinothkumar, K. R., &
  Sirajuddin, M. (2020). Structural insights into actin filament
  recognition by commonly used cellular actin markers. *The EMBO Journal*,
  39(14), e104006. (PDB 7BT7)
  [doi:10.15252/embj.2019104006](https://doi.org/10.15252/embj.2019104006)
- Nogales, E., Wolf, S. G., & Downing, K. H. (1998). Structure of the αβ
  tubulin dimer by electron crystallography. *Nature*, 391(6663), 199–203.
  [doi:10.1038/34465](https://doi.org/10.1038/34465)
- Otterbein, L. R., Graceffa, P., & Dominguez, R. (2001). The crystal
  structure of uncomplexed actin in the ADP state. *Science*, 293(5530),
  708–711. [doi:10.1126/science.1059700](https://doi.org/10.1126/science.1059700)
- Purnell, C., Heebner, J., Swulius, M. T., Hylton, R., Kabonick, S.,
  Grillo, M., Grigoryev, S., Heberle, F., Waxham, M. N., & Swulius, M. T.
  (2023). Rapid synthesis of cryo-ET data for training deep learning
  models. *bioRxiv*.
  [doi:10.1101/2023.04.28.538636](https://doi.org/10.1101/2023.04.28.538636)
  ([CTS source](https://github.com/carsonpurnell/cryotomosim_CTS))
- von der Ecken, J., Müller, M., Lehman, W., Manstein, D. J., Penczek,
  P. A., & Raunser, S. (2015). Structure of the F-actin–tropomyosin
  complex. *Nature*, 519(7541), 114–117.
  [doi:10.1038/nature14033](https://doi.org/10.1038/nature14033)
