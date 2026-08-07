# Specimens

Before any imaging physics runs, SPECTER needs a 3D electrostatic
potential volume \(V(x,y,z)\) to image -- the "specimen." Building that
volume is where the two simulation pipelines genuinely diverge:

- **Single-particle**: one structure (a PDB/mmCIF, or a pre-built volume)
  is randomly posed and placed, once per image, typically embedded in a
  thin slab of amorphous ice.
- **Cryo-ET**: a much larger volume is populated with many instances --
  membranes, filaments, and crowded proteins -- placed and packed against
  each other, then rendered at full resolution.

Both draw on the same two shared building blocks:

- **[Atomic potentials](atomic-potentials.md)**: every atom, in either
  pipeline, is rendered from the same Kirkland/Lobato/Shtyrov kernels via
  `PotentialBuilder`.
- **[Ice structure](ice.md)**: both pipelines draw amorphous ice from the
  same `IceBank` cache.

What's specific to each path:

- **Single-particle**: [pose & crowding](pose-crowding.md) -- the
  quaternion/translation pose sampled per particle, and Poisson-disk
  crowding of many particles into a micrograph.
- **Cryo-ET**: [membrane shape](membrane-shape/index.md) (and, in the
  broader `specter.specimen` package, filament and protein placement) --
  the geometry needed to populate a tomogram-scale volume realistically.

Once a specimen volume exists, everything downstream -- propagating the
electron wave through it and forming an image -- is exactly the same
physics regardless of which path built it. See [Forward
simulation](forward-simulation.md).

!!! info "Source"
    `specter.potential.PotentialBuilder` and `specter.ice` are shared by
    both pipelines. Single-particle specimen assembly lives in
    `specter.imagegenerator` and `specter.specimen.single_particle`;
    cryo-ET specimen assembly lives in `specter.specimen` (`cryoet.py`,
    `cryotomosim.py`, `membrane/`, `filament/`, `packing/` -- under heavy
    active development, see that package directly for the current state).
