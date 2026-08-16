# Membrane shape

`MembraneGenerator`'s `shape_backend` builds the membrane mid-surface one
of two ways, depending on organelle topology:

- [Spherical harmonics](spherical-harmonics.md): star-convex organelles
  (vesicles, nuclei, mitochondria). Default.
- [Swept spline](swept-spline.md): elongated, wandering tubes
  (ER-tubule-like topology).

Both produce a signed field, not a rendered volume. Turning that field
into calibrated scattering potential, and embedding proteins in the
resulting bilayer, is covered in
[Bilayer & transmembrane proteins](../cryoet-specimen/bilayer.md); how the
membrane fits into a whole specimen is
[Cryo-ET specimen assembly](../cryoet-specimen/index.md).
