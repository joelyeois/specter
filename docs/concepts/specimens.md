# Specimens

Before any imaging physics runs, SPECTER needs a 3D electrostatic
potential volume \(V(x,y,z)\) to image: the "specimen." Building that
volume is where the two simulation pipelines diverge:

- **Single-particle**: SPECTER poses and places one structure (a
  PDB/mmCIF, or a pre-built volume) at random, once per image, most often
  embedded in a thin slab of amorphous ice.
- **Cryo-ET**: SPECTER populates a much larger volume with many
  instances, including membranes, filaments, and crowded proteins,
  placing and packing them against each other, then rendering at full
  resolution.

Both draw on the same two shared building blocks:

- **[Atomic potentials](atomic-potentials.md)**: `PotentialBuilder`
  renders every atom, in either pipeline, from the same
  Kirkland/Lobato/Shtyrov kernels.
- **[Ice structure](ice.md)**: both pipelines draw amorphous ice from the
  same `IceBank` cache.

Each path also adds its own pieces:

- **Single-particle**: [pose & crowding](pose-crowding.md), the
  quaternion/translation pose sampled per particle, and Poisson-disk
  crowding of many particles into a micrograph.
- **Cryo-ET**: [specimen assembly](cryoet-specimen/index.md), covering the
  carbon film, membranes ([shape](membrane-shape/index.md) and
  [bilayer](cryoet-specimen/bilayer.md)),
  [filaments](cryoet-specimen/filaments.md),
  [gold fiducials](cryoet-specimen/beads.md), and the
  [region-gated packing](cryoet-specimen/packing.md) that crowds proteins
  into what is left.

Once a specimen volume exists, everything downstream is the same
physics regardless of which path built it: propagating the electron wave
through the volume and forming an image. See [Forward
simulation](forward-simulation.md).

!!! info "Source"
    Both pipelines share `specter.potential.PotentialBuilder` and
    `specter.ice`. Single-particle specimen assembly lives in
    `specter.imagegenerator` and `specter.specimen.single_particle`;
    cryo-ET specimen assembly lives in `specter.specimen` (`tomogram/`,
    `membrane/`, `filament/`, `packing/`, `_carbon.py`, `_grid.py`).
    This package is under active development; check it for the
    current state.
