# Cryo-ET specimen assembly

<div class="grid" markdown>

![A simulated cryo-ET specimen, summed along Z: crowded cytosolic protein, three vesicles, actin filaments, gold fiducial beads, and a carbon film edge along the bottom.](../../assets/images/cryoet-tomogram-hero.png){ width="420" }

<div markdown>
`TomogramSpecimenGenerator` is the single specimen generator behind
[`specter build tomogram`](../../user-guide/build-tomogram.md). It
composites a cryo-ET-scale potential volume out of six independent
components, then hands back a density volume plus the ground-truth labels
that describe it.

Every component is optional. A run needs at least one of them; any
combination is valid.
</div>

</div>

!!! info "Source"
    `specter.specimen.tomogram.generator`, plus the per-component modules
    linked below. Figures are produced by
    `docs-figures/cryoet_specimen_overview.py`, which drives the real
    generator once and draws every panel from that one run's own outputs.

## The components

| Component | What it is | Module |
|---|---|---|
| [Carbon support film](carbon-film.md) | A holey-grid film with a rough, genuinely 3D rim | `specimen._carbon` |
| [Membrane shape](../membrane-shape/index.md) | Organelle geometry: vesicles or wandering tubes | `specimen.membrane._field_*` |
| [Bilayer & transmembrane proteins](bilayer.md) | Turning that geometry into calibrated potential, and embedding proteins in it | `specimen.membrane._profile`, `._raster`, `._placement` |
| [Filaments](filaments.md) | Random-walk polymers (F-actin, single protofilaments) | `specimen.filament` |
| [Microtubules](microtubules.md) | Closed 13-protofilament tubes, lumen and A-lattice seam | `specimen.filament` |
| [Gold fiducial beads](beads.md) | Real fcc gold at real bulk density | `specimen._grid` |
| [Regions & protein packing](packing.md) | Cytosol/lumen classification, RSA hard-sphere placement, crowding tables | `specimen.packing`, `specimen.tomogram._regions`, `specimen.cytosolic_filler` |

Amorphous ice is **not** one of them. A `specter build tomogram` volume is
the specimen alone; ice is added downstream, when the volume is imaged.
See [Ice structure](../ice.md) and
[Generate a tilt series](../../user-guide/tilt-series.md).

## Assembly order

Generation is strictly sequential, and each stage is only aware of the
stages before it:

```
carbon film → membranes → filaments/microtubules → gold beads → targets → filler
```

Nothing is re-placed once accepted. When a candidate position doesn't
work, it is rejected and the generator moves on rather than backtracking,
the same "reject and move on" philosophy throughout this module. That is
why the order matters: it is exactly the order of decreasing placement
freedom.

What each stage actually avoids:

| Stage | Avoids | Doesn't avoid |
|---|---|---|
| Carbon film | — (painted first, into an empty canvas) | — |
| Membranes (auto-placed) | Carbon, other auto-placed membranes | Explicitly-positioned membranes |
| Membranes (explicit `position_xyz`) | — (density overlapping carbon is clipped instead) | Everything else |
| Filaments | Carbon (monomers landing in it are dropped) | Membrane shell, other filaments |
| Microtubules | Carbon (dimers landing in it are dropped) | Membrane shell, filaments, other microtubules |
| Gold beads | Membrane shell, carbon, filaments/microtubules, other beads | Nothing (not region-gated: fiducials sit in the ice) |
| Targets | Membrane shell, carbon, filaments/microtubules, beads; restricted to their `location` region | — |
| Filler | All of the above, plus already-placed targets | — |

Two consequences worth knowing before you read a rendered volume:

- **The carbon film is not placement-aware in reverse.** It is painted
  first and everything else works around it, matching the CryoTomoSim
  algorithm this feature was ported from. A membrane you positioned by
  hand inside the film doesn't move; the part of its density that would
  land on carbon is zeroed just before compositing, so the volume and that
  instance's own shell label stay consistent with each other.
- **Filaments have no obstacle-avoiding random walk.** A walk that runs
  into the film loses the monomers that land inside it, leaving a gap in
  that filament rather than steering around the obstacle.

![The same specimen as the hero image, a thin mid-Z slab, painted by component from the run's own ground-truth label volumes.](../../assets/images/cryoet-tomogram-components.png){ width="620" style="display:block;margin:1.2em auto;" }

Every colour above comes from a label volume the generator wrote, not from
segmenting the density afterward.

## Regions

Once every membrane has been composited, the volume is classified once, on
the composite, into three disjoint regions: `shell` (bilayer material),
`lumen` (any enclosed compartment), and `cytosol` (everything reachable
from the volume's own boundary). Protein species declare which of the
latter two they belong to, and are packed only there.

This is topology, not geometry: it works for one vesicle, several disjoint
ones, or none at all without special-casing. The carbon film lands in
`shell` too, since it is dense material that nothing should be packed
into, which is exactly what `shell` means to the packing stage. See
[Regions & protein packing](packing.md).

## Ground truth

A run writes more than a density volume. Every placement is recorded, and
the label volumes below are what make a generated tomogram usable as
training data:

![Region map, per-instance membrane labels, and per-instance protein/filament/bead labels, all from the same mid-Z slice.](../../assets/images/cryoet-tomogram-ground-truth.png){ width="900" style="display:block;margin:1.2em auto;" }

- `regions`: `0` cytosol, `1` shell, `2` lumen.
- `membrane_labels`: which membrane instance a shell voxel belongs to.
  Where two instances overlap, first write wins.
- `instance_labels`: one id per placed filament monomer, bead, target and
  filler instance, handed out in that order. This is a single id space,
  not one per category.
- Picks: copick-style `.ndjson` per species, positions and orientations.
  Targets are exported by default; filler is not.

Membranes deliberately have no picks entry: a surface has no single
natural "position" the way a protein does, so `membrane_labels` and
`regions` are its ground truth instead.

## Limitations

- **Transmembrane proteins get no per-instance voxel labels.** Their
  density is correctly present in the volume, and their placements are
  recorded, but they don't appear in `instance_labels`. This is a
  documented gap, not an oversight.
- **Collision tests are bounding spheres against distance fields**, not
  exact voxel overlap. A placed protein's true, non-spherical shape can
  still graze a filament or the bilayer very close to the boundary.
- **RSA jams well below close packing.** `filler_occupancy_fraction` is a
  budget, not a promise; see [the packing page](packing.md) for where the
  ceiling actually sits.
- **An explicitly-positioned membrane is checked against nothing.** Give
  two of them the same position and they will overlap.

## Provenance

This generator is a clean-room second approach relative to
[CryoTomoSim](https://github.com/carsonpurnell/cryotomosim_CTS) (CTS), not
a port of its placement or membrane algorithms. An earlier CTS-replica
generator existed alongside it and was deleted once this one reached
feature parity. Two components *are* descended from CTS directly, both
generic bulk-material physics with no placement logic of their own: the
[gold beads](beads.md) (`gen_beads.m`) and the
[carbon film](carbon-film.md) (`gen_carbon.m`/`carbonshape`). The
[transmembrane placement](bilayer.md) construction is shared with
[polnet](https://github.com/anmartinezs/polnet), as is the bilayer's
two-Gaussian profile.

## References

- Purnell, C., et al. (2023). Rapid synthesis of cryo-ET data for training
  deep learning models. *bioRxiv* 2023.04.28.538636.
- Martinez-Sanchez, A., Lamm, L., Jasnin, M., & Phelippeau, H. (2024).
  Simulating the cellular context in synthetic datasets for cryo-electron
  tomography. *IEEE TMI* 43(11), 3742–3754.
