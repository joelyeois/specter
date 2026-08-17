# Other propagation modes

Three more propagation modes trade accuracy for speed by different means
than [Rytov](rytov.md): `firstborn` and `kinematic` linearize or
simplify the per-slice scattering, and `projection` discards
slice-to-slice propagation entirely.

!!! info "Source"
    `Scattering.firstborn` / `.kinematic` / `.projection` / `.ctf` and
    their `IterativeScattering` counterparts. Figures are produced by
    `docs-figures/scattering_accuracy.py`.

## First Born

The weak phase object approximation: each slice's transmission function
\(\exp(i\sigma\Delta z V_z)\) is linearized to its first-order Taylor term
before being propagated and summed,

\[
\psi = 1 + i\sigma\Delta z \sum_{z} \mathcal{F}^{-1}\!\big[\mathcal{F}[V_z]\cdot F_z\big]
\]

using the same per-slice Fresnel propagator \(F_z\) as
[Rytov](rytov.md#the-exponentiated-born-series). This is the single-scattering
limit: valid only while the phase accumulated by any one slice is small.

## Kinematic

Kinematic keeps each slice's transmission amplitude exact,
\(\exp(i\sigma\Delta z V_z) - 1\), rather than linearizing it, but still
combines slices by addition rather than multiplication:

\[
\psi = 1 + \sum_{z} \mathcal{F}^{-1}\!\big[\mathcal{F}\big[\exp(i\sigma\Delta z V_z) - 1\big]\cdot F_z\big]
\]

This is the first-order term of the same multislice Born series
`multislice` sums exactly, so it is a strictly better single-scattering
approximation than `firstborn` in principle. In practice, at the ice
thicknesses in the sweep below, the two track each other closely: keeping
the per-slice amplitude exact matters less than the additive (rather than
multiplicative) combination shared by both.

## Projection

The thin-specimen limit: sum the potential over Z first, then apply a
single transmission function with no propagation step at all,

\[
\psi = \exp\!\left(i\sigma\Delta z \sum_{z} V_z\right)
\]

This is also what `scattering_model="ctf"` returns as a real-valued
projected potential (`2\sigma\Delta z \sum_z V_z`, without the complex
exponential), for use with `aberration_model="ctf"`'s separate CTF-based
intensity model — see [Detector](../detector.md) and
[Aberrations](../aberrations.md).

## Accuracy vs. thickness

Using the same thickness sweep as [Rytov's accuracy
figure](rytov.md#accuracy-vs-thickness) (a `RandomIcemaker` ice slab, 300
kV, `multislice` as the reference):

![Relative error in exit-wave intensity vs. multislice, as a function of ice thickness, for first Born, kinematic, and projection (highlighted) against Rytov (faint, for context).](../../assets/images/scattering-accuracy-vs-thickness-other-modes.png){ width="600" }

Two things stand out. First, `firstborn` and `kinematic` track each other
almost exactly across the whole range — confirming that, at these
thicknesses, the additive slice combination they share dominates their
error, over the linearization difference between them. Second,
`projection` is *more* accurate than either, despite being the only mode
here that discards Fresnel propagation entirely. The likely reason is
normalization: projection's exit wave is a genuine unit-modulus phase
factor, \(|\psi| = 1\) everywhere (before absorption), at any thickness,
because it never leaves the exponential form. `firstborn` and `kinematic`
both build \(\psi\) as \(1 + (\text{something})\), which is not
unit-modulus by construction, and the resulting normalization error grows
with the total integrated potential faster than the propagation physics
projection is missing costs it, at least over the thickness range
measured here. This is specific to intensity error at these mild
thicknesses and pixel size; it is not a general claim that projection is
"more physically complete" than the other linearizations, since it is the
only one of the three with no Fresnel diffraction model at all.

## References

- Kirkland, E. J. (2010). *Advanced Computing in Electron Microscopy*, 2nd
  Edition. Springer. Ch. 5-6 (weak phase object and projection
  approximations).
