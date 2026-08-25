# Rytov

`scattering_model="rytov"` computes the exit wave as a single exponentiated
sum over all Z-slices rather than a sequential slice-by-slice recursion.
It is `Ghostbuster`'s default propagation model for single-particle
reconstruction, while `TomogramReconstructor` and every generation
pipeline default to [`multislice`](multislice.md).

!!! info "Source"
    `Scattering.rytov` / `IterativeScattering.rytov` /
    `IterativeScattering.parallel_rytov`.
    `docs-figures/scattering_accuracy.py` produces the figures.

## The exponentiated Born series

Rytov Fresnel-propagates each slice's contribution to the exit plane,
accumulates it in the phase, and exponentiates the sum once at the end:

\[
\psi = \exp\!\left(i\sigma\,\Delta z\sum_{z} \mathcal{F}^{-1}\!\big[\mathcal{F}[V_z]\cdot F_z\big]\right)
\]

where \(F_z\) is the same Fresnel propagator as
[multislice](multislice.md#the-recursion), evaluated at the distance from
slice \(z\) to the exit plane rather than a single slice thickness. This
is the Rytov approximation: the exponentiated first Born series. Expanding
the exponential to first order recovers
[`firstborn`](other-modes.md#first-born) exactly, so Rytov and first Born
agree wherever the linearization is valid and diverge only once the
per-slice phase \(\sigma\,\Delta z\,V_z\) is no longer small.

## A sum, not a recursion

Multislice's per-slice transmission depends on the *already-propagated*
wave from every previous slice (`psi_{i} = t_i * psi_{i-1}`), which is
why multislice must run sequentially. Rytov's sum over slices is
independent of any other slice's contribution before the final
exponentiation, so it parallelizes: `IterativeScattering.parallel_rytov`
computes every slice's contribution as one batched FFT pair rather than
`nz_new` sequential ones, with optional chunked gradient checkpointing to
bound memory. A fully parallel forward pass is both faster and, via
checkpointing, cheaper to hold gradients for than replaying a sequential
recursion hundreds to a thousand slices deep, which is what would make
Rytov attractive inside an iterative, tilt-aware reconstruction loop. No
pipeline wires it in yet, however: `parallel_rytov` is correctness- and
gradient-tested (`tests/test_scattering.py`) but not called from any
pipeline, config, or reconstructor class, and `TomogramReconstructor`
(the one class that uses `IterativeScattering` at all) still defaults to
`scattering_model="multislice"`.

## Accuracy vs. thickness

The tradeoff is accuracy at large thickness: Rytov has no per-slice
feedback, so it cannot represent multiple scattering the way multislice's
recursion can. The figure below sweeps specimen thickness for the same
`RandomIcemaker` ice slab used in the [multislice
page](multislice.md#watching-the-recursion-accumulate), measuring each
model's relative error in exit-wave intensity \(|\psi|^2\) against
`multislice` as the reference:

![Relative error in exit-wave intensity vs. multislice, as a function of ice thickness, for Rytov (highlighted) against the other three approximate modes (faint, for context).](../../assets/images/scattering-accuracy-vs-thickness-rytov.png){ width="600" }
///caption
Relative error in exit-wave intensity vs. multislice, as a function of ice thickness, for Rytov (highlighted) against the other three approximate modes (faint, for context).
///

Across this thickness range (up to 320 Å, spanning typical single-particle
ice thickness), Rytov's error stays one to two orders of magnitude below
`firstborn`, `kinematic`, and `projection`'s. The exponentiated form
makes the difference: even though Rytov shares first Born's linear-in-slices
structure, exponentiating the accumulated phase rather than adding \(1\)
to it (as `firstborn` does) keeps the *amplitude* of \(\psi\) correctly
normalized at every thickness, rather than only to first order. Rytov's
own mean intensity stays pinned to 1.0 (energy-conserving) at every
thickness tested, like `multislice`'s. `firstborn`/`kinematic`, by
contrast, drift substantially at large thickness. `projection`'s
placement on this plot has a separate explanation; see [Other propagation
modes](other-modes.md#accuracy-vs-thickness).

## References

- Yeo, J., & Loh, N. D. (2026). Pursuing the physics of cryo-EM image
  formation. In *Current Approaches to Cryo-Electron Microscopy*,
  *Progress in Molecular Biology and Translational Science*. Elsevier.
  [doi:10.1016/bs.pmbts.2026.05.001](https://doi.org/10.1016/bs.pmbts.2026.05.001).
  Derives the Rytov approximation as specialized to slice-wise electron
  propagation, the form implemented here.
- Rytov approximation: standard in coherent wave optics; see e.g. J. W.
  Goodman, *Introduction to Fourier Optics*, for the general derivation.
