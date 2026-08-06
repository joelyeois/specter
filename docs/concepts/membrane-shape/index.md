# Membrane shape

`MembraneGenerator`'s `shape_backend` builds the membrane mid-surface one
of two ways, depending on organelle topology:

- [Spherical harmonics](spherical-harmonics.md) -- star-convex organelles
  (vesicles, nuclei, mitochondria). Default.
- [Swept spline](swept-spline.md) -- elongated, wandering tubes
  (ER-tubule-like topology).
