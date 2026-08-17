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

## Where the linearization breaks down

It would be reasonable to expect `firstborn` to track `multislice`
closely at small thickness, since the weak-phase condition it needs is
easiest to satisfy there. It does not -- its *pattern* is already nearly
uncorrelated with the truth at the thinnest specimen tested, not just at
the thickest. The reason is a mismatch between what the linearization
throws away and what actually carries the physical signal.

`firstborn` and `rytov` share one underlying complex quantity, \(\Theta =
\sigma\Delta z\sum_z\mathcal{F}^{-1}[\mathcal{F}[V_z]\cdot F_z] = a+ib\)
(the same object [Rytov](rytov.md#the-exponentiated-born-series) calls
the accumulated phase): `rytov` is \(\exp(i\Theta)\), and `firstborn` is
its linearization, \(1+i\Theta\). Splitting \(\Theta\) into real and
imaginary parts exposes why linearizing first and squaring second is not
a safe order of operations:

\[
|\exp(i\Theta)|^2 = |\exp(i(a+ib))|^2 = \exp(-2b)
\qquad\text{depends only on } b=\mathrm{Im}(\Theta),
\]
\[
|1+i\Theta|^2 = 1 - 2b + a^2 + b^2
\qquad\text{picks up a spurious } a^2 \text{ term.}
\]

The true intensity is a pure rotation-then-attenuation: a real phase
shift (\(a\)) rotates the wave without changing its magnitude at all,
and only the imaginary part (\(b\), the part Fresnel propagation
converts into a genuine amplitude change) affects \(|\psi|\). `firstborn`
has no such structure -- squaring a truncated sum mixes \(a\) and \(b\)
together, and the resulting \(a^2\) term has no physical counterpart.

Whether this matters comes down to how \(a\) and \(b\) compare in
practice, and for a specimen like vitreous ice -- a predominantly
*phase* object, weakly absorptive -- \(a\) dominates \(b\) by
construction: at low spatial frequency the Fresnel propagator is close
to a real, unit-magnitude filter (\(F_z(k)\approx 1\) as \(k\to0\)), so
most of \(\Theta\)'s power lands in \(a\), and only the higher-frequency
content that genuinely diffracts leaks into \(b\).

![Standard deviation of Re(Θ) and Im(Θ) vs. thickness. Re(Θ), the ordinary phase, dominates Im(Θ), the part that actually sets true intensity, at every thickness tested -- by roughly 50x at the thinnest slab and still 2.6x at the thickest.](../../assets/images/scattering-theta-real-imag-split.png){ width="600" }

![Correlation of each mode's intensity pattern with multislice's true pattern, vs. thickness. rytov stays at 1.000 throughout; firstborn and kinematic sit near zero (firstborn briefly negative) across the entire range -- not something that only develops at large thickness.](../../assets/images/scattering-pattern-correlation-vs-thickness.png){ width="600" }

At the thinnest slab tested (8 Å), \(\mathrm{Re}(\Theta)\) already has
50x the standard deviation of \(\mathrm{Im}(\Theta)\), so `firstborn`'s
intensity fluctuation is completely dominated by the spurious \(a^2\)
term rather than the physically correct \(-2b\) term -- hence a pattern
correlation of \(-0.22\), not close to \(+1\), at the thinnest slab
where the weak-phase condition should be at its easiest. The dominance
narrows as thickness grows (to \({\approx}2.6\times\) by 320 Å, as
diffraction has more distance to convert phase into genuine amplitude
contrast) but never reverses over this range, so the correlation never
recovers either. Averaged over the whole image, the same \(a^2+b^2 =
|\Theta|^2\) excess also explains the mean-intensity bias from the
[Accuracy vs. thickness](#accuracy-vs-thickness) figures below --
\(\langle|\Theta|^2\rangle \approx 0.387\) at 320 Å, matching the
measured +38.7% bias to three significant figures -- but that aggregate
number was always the smaller half of the story: even where the *mean*
bias looks minor, the *pattern* is already wrong, because it takes only
a modest imbalance between \(a\) and \(b\), not a large accumulated
phase, to swamp the one part of \(\Theta\) that is actually informative.
[Kinematic](#kinematic) inherits the same problem for the same
structural reason (additive slice combination, not multiplicative), not
because its per-slice linearization is any less exact.

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

Read at face value, this says `projection` beats `firstborn` and
`kinematic` on intensity error, despite being the only mode here with no
Fresnel diffraction model at all -- which would be a strange result,
since `projection` has strictly less physics than either. It is not what
it looks like. The next two figures show what is actually happening.

![Mean exit-wave intensity vs. thickness, per model. A properly normalized exit wave conserves total intensity (multislice and projection stay pinned to 1.0); firstborn and kinematic drift up to +39% by 320 A.](../../assets/images/scattering-mean-intensity-vs-thickness.png){ width="600" }

![Exit-wave intensity maps at 320 A, side by side. multislice and rytov show fine, correlated speckle; firstborn is a different, coarser pattern riding on a strongly biased mean; projection is exactly flat.](../../assets/images/scattering-mode-intensity-maps.png){ width="900" style="display:block;margin:1.2em auto;" }

Two real things are happening, and they point in the same misleading
direction on the error plot:

- **`firstborn` and `kinematic` never actually track the true pattern,
  and additionally stop conserving energy as thickness grows.** A
  correctly normalized exit wave has \(\langle|\psi|^2\rangle = 1\) at
  every thickness (true for `multislice`, `rytov`, and trivially for
  `projection`). By 320 Å, `firstborn`'s mean intensity has drifted to
  \({\approx}1.39\), a \({\approx}39\%\) bias with no counterpart in the
  true physics. That bias grows with thickness, but the more fundamental
  problem doesn't wait for it: `firstborn`'s intensity *pattern* is
  already essentially uncorrelated with `multislice`'s at the thinnest
  specimen tested, not only at 320 Å -- see [Where the linearization
  breaks down](#where-the-linearization-breaks-down) for why linearizing
  before squaring is the culprit, independent of how thick the specimen
  is. This is the textbook breakdown of the first Born approximation
  once amplitude and phase get mixed by squaring: a real failure of
  these two modes, not an artifact of the comparison.
- **`projection` cannot be measured wrong by this metric, because it has
  no signal at all.** With `alpha=0` (no absorption), `projection`'s exit
  wave is \(\exp(i\sigma\Delta z \sum_z V_z)\) -- a pure phase factor, so
  \(|\psi|=1\) *exactly*, at every pixel, at every thickness. Its
  intensity map is perfectly flat (std \(\approx 10^{-7}\), floating-point
  noise). The relative-error metric above is therefore just measuring how
  far `multislice`'s own true contrast is from flat, which stays small
  (std \(0.045\) at 320 Å) over this thickness/pixel-size range. A method
  that always predicts "no contrast" scores well against a target that is
  itself close to flat -- not because it captured any thickness-dependent
  structure, but because there was not yet much structure to miss.

So the error curve's ordering is real, but the reason is not "projection
approximates thickness effects well." It is that `firstborn`/`kinematic`
have a genuine, worsening energy-conservation failure at this thickness,
while `projection`'s zero-information prediction happens to have a
smaller error than that failure, purely because the true signal is still
small. None of this makes `projection` a good general substitute for
`multislice`: it is blind to depth by construction and will read as
"accurate" on this metric right up until the specimen develops enough
real contrast to expose it, which a thicker or more strongly scattering
specimen than this one would do easily.

## Is this specific to a pure phase object (α=0)?

Every figure on this page uses `alpha=0`, chosen deliberately to isolate
the failure mode in [Where the linearization
breaks down](#where-the-linearization-breaks-down) without a second
effect layered on top. Real specimens absorb a little (`alpha` typically
0.07-0.1), so it's worth checking whether that changes the picture.

![Relative error and pattern correlation for firstborn and projection, at alpha=0 (solid) vs. a typical alpha=0.1 (dashed), across the same thickness sweep.](../../assets/images/scattering-alpha-robustness-check.png){ width="900" style="display:block;margin:1.2em auto;" }

It doesn't change the ranking, and it clarifies *why* `projection` does
as well as it does. Adding absorption barely moves either model's
relative error (left panel: the solid and dashed curve for each model
nearly overlap) -- `firstborn` is, if anything, marginally worse with
`alpha` included, since the same \(a\)-vs-\(b\) mixing problem now also
folds the modest genuine absorption signal into the same spurious
channel.

The correlation panel (right) is more interesting. At the thinnest
specimens, absorption dominates the true signal almost completely, and
*both* models track it well (correlation \({\approx}1\)) -- not because
`firstborn`'s approximation got better, but because the true pattern is
now simple enough (nearly linear in the projected potential) that even a
biased, mixed-up prediction happens to point the right way. As thickness
grows, `projection`'s correlation stays healthy (0.82-0.97 throughout
this sweep): mass-thickness/amplitude contrast is a real, well-behaved
signal that a simple projected-potential model genuinely captures, not
merely "nothing there to get wrong." `firstborn`'s correlation collapses
back toward zero well before 320 Å regardless of `alpha` -- the \(a^2\)
artifact reasserts itself once the specimen is thick enough for it to
dominate again.

## References

- Kirkland, E. J. (2010). *Advanced Computing in Electron Microscopy*, 2nd
  Edition. Springer. Ch. 5-6 (weak phase object and projection
  approximations).
