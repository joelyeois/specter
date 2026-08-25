# Membrane shape

`MembraneGenerator`'s `shape_backend` builds the membrane mid-surface one
of two ways, depending on organelle topology:

- [Spherical harmonics](spherical-harmonics.md): star-convex organelles
  (vesicles, nuclei, mitochondria). Default.
- [Swept spline](swept-spline.md): elongated, wandering tubes
  (ER-tubule-like topology).

Both produce a signed field rather than a rendered volume.
[Bilayer & transmembrane proteins](../cryoet-specimen/bilayer.md) covers
turning that field into calibrated scattering potential and embedding
proteins in the resulting bilayer.
[Cryo-ET specimen assembly](../cryoet-specimen/index.md) covers how the
membrane fits into a whole specimen.
