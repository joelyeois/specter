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

For any real \(\theta\), \(|1+i\theta|^2 = 1+\theta^2\) exactly -- not an
approximation, just algebra. So whenever `firstborn` truncates
\(\exp(i\theta) \approx 1+i\theta\), squaring that truncated amplitude
doesn't just lose accuracy smoothly; it manufactures a spurious extra
intensity of exactly \(\theta^2\) that the true, bounded
\(|\exp(i\theta)|^2 = 1\) never has. Averaged over an image, this predicts

\[
\langle|\psi_{\mathrm{firstborn}}|^2\rangle - 1 \;\approx\; \langle\theta^2\rangle,
\qquad \theta = \sigma\Delta z\sum_z V_z
\]

the specimen's *total* (projected) phase shift -- because Fresnel
propagation is a pure phase filter per spatial frequency and therefore
preserves total power, so propagating each slice's contribution before
summing (as `firstborn` does) redistributes contrast spatially but barely
changes \(\langle\theta^2\rangle\) itself. This is confirmed numerically
for the ice slab in the figures below: at 320 Å, \(\theta\) has mean
0.62 rad and standard deviation 0.06 rad across the image, giving
\(\langle\theta^2\rangle \approx 0.387\) -- matching the measured +38.7%
mean-intensity bias to three significant figures.

The practical takeaway is that the breakdown threshold is
\(\theta \sim O(1)\) **radian**, not \(2\pi\): nothing here is about the
phase wrapping around a full cycle. \(\langle\theta^2\rangle\) is already
a 6% bias by \(\theta \approx 0.25\) rad, and by the time \(\theta\)
approaches 1 radian -- one-sixth of a full \(2\pi\) turn -- the linear
approximation is already off by \({\approx}100\%\). In this ice slab,
\(\theta\) never exceeds 0.9 rad anywhere in the image, and `firstborn`
is already wrong by nearly 40% on average; a specimen would never need to
accumulate anywhere near a full \(2\pi\) cycle of phase to break this
approximation. [Kinematic](#kinematic) inherits essentially the same
\(\theta^2\) excess: its per-slice amplitude is exact, but slices are
still combined by addition rather than multiplication, and that additive
combination -- not the per-slice linearization -- is what this section's
argument actually depends on.

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

- **`firstborn` and `kinematic` stop conserving energy.** A correctly
  normalized exit wave has \(\langle|\psi|^2\rangle = 1\) at every
  thickness (true for `multislice`, `rytov`, and trivially for
  `projection`). By 320 Å, `firstborn`'s mean intensity has drifted to
  \({\approx}1.39\), a \({\approx}39\%\) bias with no counterpart in the
  true physics -- the direct consequence of truncating
  \(\exp(i\theta)\approx 1+i\theta\), worked out quantitatively in [Where
  the linearization breaks down](#where-the-linearization-breaks-down).
  The intensity map above confirms this isn't just a scale error either
  -- `firstborn`'s pattern is visibly different in character from
  `multislice`'s (coarser, blobbier), and numerically the two are
  essentially uncorrelated at this thickness. This is the textbook
  breakdown of the first Born approximation once the specimen is no
  longer thin: it is a real failure of these two modes, not an artifact
  of the comparison.
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

## References

- Kirkland, E. J. (2010). *Advanced Computing in Electron Microscopy*, 2nd
  Edition. Springer. Ch. 5-6 (weak phase object and projection
  approximations).
