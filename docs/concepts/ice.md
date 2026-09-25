# Ice structure

`GradientSKIcemaker` builds amorphous ice by optimising molecule
*positions* so the resulting volume reproduces a measured structure
factor and a physically motivated interatomic energy, rather than
dropping water molecules at random to the right bulk density. `IceBank`
then caches the result for near-instant reuse at simulation time. This
page works through the physics behind that optimisation; for the
practical cache/sampling side (choosing `ice_model`, building a
replacement cache at a different pixel size), see
[Using the ice cache](../user-guide/ice-cache.md).

!!! info "Source"
    `specter.ice._gradient.GradientSKIcemaker`, `specter.ice._energy.MLBOP`,
    `specter.ice._kernels`. `specter.ice._bank.IceBank` is the caching
    layer built on top; see [Using the ice cache](../user-guide/ice-cache.md).

## The optimisation target: S(k)

The structure factor \(S(k)\), the radially averaged Fourier amplitude of
the oxygen density, characterises amorphous ice's structure.
`GradientSKIcemaker` optimises a set of continuously-valued oxygen
positions so that the radially averaged \(|\mathrm{FFT}(\text{voxelised
positions})|\) matches a target \(S(k)\) curve, via mean-squared error on
the radial profile.

The target itself comes from an in-house molecular dynamics reference
frame of low-density amorphous ice at 80 K and 0 atm (LDA-80K), not from a
published dataset. By default, each `GradientSKIcemaker` instance
recomputes the target natively at its own grid spacing and box size
(`compute_native_target`), rather than interpolating from a single
fixed-resolution precomputation. Matching the voxel spacing between
target and optimisation grid matters more than matching absolute box
size: comparing against a target computed at a different, coarser
spacing aliases only one side of the comparison, and got the optimiser
stuck at an S(k) loss of \(O(1)\) to \(O(10^3)\); a natively computed
target at the same spacing converges to \(O(10^{-5})\) to
\(O(10^{-7})\) across voxel sizes from 0.5 to 2.0 Å and box sizes from
16 Å up to 256 Å, beyond the roughly 127 Å extent of the underlying MD
cell, via safe extrapolation in `compute_native_target`.

## The optimisation

Positions start out uniformly random. A short L-BFGS relaxation against
the ML-BOP energy alone (`prerelax_steps`, 30 by default) first removes
the overlapping molecules of that start, since the full loss would
otherwise spend its first several hundred evaluations on nothing else.
L-BFGS then optimises the positions (the default; Adam is also available)
against the combined loss

\[
\mathcal{L} = \mathcal{L}_{S(k)} + w_\text{MLBOP}\,\mathcal{L}_\text{MLBOP}
\]

L-BFGS's strong-Wolfe line search adapts its own step size, so the `lr`
parameter rarely needs tuning. Optimisation runs until `n_steps`, or stops
early once the fractional change in loss stays below `tol` for `patience`
consecutive steps.

The positions are optimised in float64 while the FFT, the voxel splat and
the three-body sum run in float32. The \(S(k)\) term is matched within a
few hundred evaluations, after which the remaining loss is the energy term
alone, and the descent step it calls for moves each molecule by about
\(10^{-6}\) Å: below the float32 spacing of a coordinate at 64 to 128 Å
(\(7.6 \times 10^{-6}\) to \(1.5 \times 10^{-5}\) Å). Float32 positions
therefore freeze at that point with the energy near \(-0.10\) eV/atom;
float64 positions continue to \(-0.2\) eV/atom and below. Forming the
pair vectors and voxel fractions in float64 before casting keeps that
resolution at float32 cost.

Matching \(S(k)\) alone underconstrains the local structure: two
configurations can share a radial Fourier profile while one has
physically overlapping or badly angled molecules, since a radial average
discards phase and angular information. `_sk_loss` therefore adds a
second term to penalise that, chosen by `mlbop_strength`
(`GradientSKIcemaker.optimize`)'s two options.

### ML-BOP energy penalty (default)

`mlbop_strength` weights a fully differentiable penalty computed from the
ML-BOP coarse-grained water potential (`specter.ice._energy.MLBOP`), a
Tersoff-style two-body plus three-body bond-order potential: an
exponential repulsive/attractive pair term smoothly cut off at `R ± D`
(`f_R`, `f_A`, `f_C`), combined with an angular term `g(cos θ)` that
penalises O-O-O angles away from a preferred tetrahedral-like value.
Unlike the geometric-only alternative below, this penalises both overlap
and unrealistic local geometry; testing across voxel sizes from 0.5 to
2.0 Å and box sizes up to 256 Å found it the better balance of
energy-match quality against \(S(k)\) fidelity.

`mlbop_target` (default \(-0.413\) eV/atom, the ML-BOP energy of the
LDA-80K reference frame, stable to \(\pm 0.0001\) eV/atom across widely
separated MD frames) matches the optimised configuration's per-atom
energy to that value instead of minimising it without bound. Minimising
energy without a target drifts toward a low-energy, more crystalline
packing with no natural floor, rather than the disordered target phase;
matching a measured reference value keeps the result in the right
structural regime instead. Set `mlbop_target=None` to fall back to
unbounded minimisation.

### Pair-exclusion penalty (geometric fallback)

`rep_strength` (default `0.0`, disabled) is a cheaper, purely geometric
alternative: an FFT-convolution count of other particles within
`min_distance` of each voxel, penalised by a squared ReLU. It prevents
outright overlap but has no notion of angle, so it is a weaker constraint
on local structure than the ML-BOP penalty, and S(k) matching alone can
still hide badly overlapping atoms behind a good Fourier-amplitude match
when this is the only penalty active.

## Solvent displacement

Ice is added to a specimen in proportion to the space the specimen leaves
free. The occupancy of each voxel is read off the specimen's potential after
coarse-graining it over 2 Å, the length scale of a water molecule:

\[
o(\mathbf{r}) = \min\!\left(\frac{\bar V(\mathbf{r})}{V_{ref}},\,1\right),
\qquad
V_{ice}(\mathbf{r}) \to \bigl(1 - o(\mathbf{r})\bigr)\,V_{ice}(\mathbf{r}).
\]

The Gaussian coarse-graining conserves the integral of the potential, so the
total volume of water displaced is set by \(V_{ref}\) alone. How much water a
molecule displaces is a geometric property, its mass at the partial specific
volume of protein (0.73 cm³/g), and does not depend on the scattering factors
used to render it; the rendered potential does. For 6BDF, the protein's mean
inner potential is 5.91 V under Shtyrov factors with hydrogens and 8.01 V
under Kirkland factors, so a single fixed reference of 7.0 V displaces 0.84 or
1.09 of the molecular volume respectively.

When the molecule's mass is known, `potential.full_occupancy_potential`
therefore solves for the reference at which the clamped occupancy sums to
exactly the molecular volume. The particle and micrograph pipelines supply
the mass from the atomic model (`potential.molecular_mass_from_atoms`, which
adds the hydrogens a hydrogen-free deposition omits). The reference is solved
once per template, in 0.01 to 0.04 s on a GPU. A volume supplied without a
mass, and a tomogram, which holds many species in one volume, use the fixed
`potential.FULL_OCCUPANCY_POTENTIAL_V` of 7.0 V instead.

## The solvent under exposure

A movie is not a picture of one frozen solvent. Each inelastic event deposits
tens of eV in the ice, and over a few e⁻/Å² every water molecule is hit many
times; the network is locally disrupted and re-forms. Two consequences follow.
Each frame still holds water with the equilibrium structure factor, which is
why the 3.7 Å ring of raw movie fractions does not fade across 50 e⁻/Å². But
successive frames hold different arrangements of it, so the ice's Fourier
amplitudes lose coherence with dose. The solvent is therefore not damaged in
the sense a protein is; it decorrelates.

Write the ice's fluctuation at spatial frequency \(k\) and accumulated dose
\(D\) as \(f_{\mathbf k}(D)\), with constant power \(P(k)\) and correlation
\(\rho_k(|D - D'|)\) between doses. An exposure read out as \(n\) frames of
dose \(d\), each integrating continuously over its own dose and combined with
per-frequency weights \(a_i\) (\(\sum_i a_i = 1\)), keeps the fraction

\[
\mathcal S(k) = \sum_{i,j} a_i a_j\, c_{|i-j|}(k),
\qquad
c_L(k) = \frac{1}{d^2}\int_{\text{frame } i}\int_{\text{frame } i+L}
\rho_k(|D - D'|)\,\mathrm dD\,\mathrm dD'
\]

of a frozen ice's power. With equal weights this is the continuous average
over the whole exposure, independent of \(n\):
\(\mathcal S = (2/D^2)\int_0^D (D-\tau)\,\rho_k(\tau)\,\mathrm d\tau\).
`ice.apply_solvent_exposure` scales one ice realisation's Fourier amplitudes
by \(\sqrt{\mathcal S(k)}\) before the specimen's volume is cut out of it, so
the summed image has the right second-order statistics without the frames
being simulated. \(\mathcal S(0) = 1\): the mean potential, which sets the
absorption and the volume the specimen displaces, is kept.

The correlation comes from one of two models (`ice.solvent_coherence`).
McMullan et al. (2015) assume every molecule takes an independent Gaussian
step of variance \(\sigma_0^2\) per axis per e⁻/Å², so that
\(\rho_k(\tau) = e^{-2\pi^2\sigma_0^2 k^2 \tau}\); they measured
\(\sigma_0^2 = 0.38\) Å² per e⁻/Å² from the 3.7 Å ring at 300 kV. The default
model, `"relaxed"`, is measured instead. Periodic 256 Å boxes of library ice
were evolved by such Gaussian kicks, each followed by the library's own
\(S(k)\) and ML-BOP relaxation so that every state is water, and the
coherence was measured without a grid, from the structure factor of all
527,178 molecules at reciprocal-lattice vectors of the box, over lags of 1 to
60 steps. It is compressed rather than exponential: every shell from 90 Å to
1.4 Å fits \(
ho = \exp[-(x/x_0(k))^{\beta(k)}]\) in accumulated kick
variance \(x\) to within 0.012, with \(\beta \approx 1.1\). Its time scale is
fixed so that the ring's correlation has the same area as the Gaussian
model's for the same \(\sigma_0^2\), which is the quantity a multi-frame fit
determines.

Relaxed ice departs from independent kicks in the way liquid dynamics
predicts, where collective density relaxes at a rate proportional to
\(k^2/S(k)\). Ice is nearly incompressible, so its long-wavelength density
fluctuations decorrelate far faster: the correlation area is 1/33 of the
Gaussian model's at 20 Å and 1/12 at 10 Å. Near peaks of \(S(k)\) it
decorrelates more slowly, and at short wavelengths the two models converge.
Under the Gaussian model a 141-frame summed spectrum of pure ice shows a bright
low-frequency disc with Thon rings that McMullan et al.'s Fig. 1(a) does not;
under the relaxed model it is flat there, as theirs is.

In a particle stack this is ``Ice(motion_variance=...)``
(`ice_motion_variance` in the particle config). It needs the dose envelope,
if one is applied, to act on the specimen
(``Envelopes(dose_envelope_target="specimen")``, see
[Aberrations](aberrations.md)): on the transfer function the envelope would
fade the solvent that the exposure filter already decorrelates.

## Limitations

- **The target is one phase of ice at one thermodynamic state.**
  LDA-80K's structure factor and ML-BOP reference energy describe low-
  density amorphous ice at 80 K; matching a different phase (high-density
  amorphous, crystalline) needs a different MD reference frame, and
  SPECTER does not currently bundle one.
- **`mlbop_target` assumes the reference energy is known.** The default
  value is specific to the bundled LDA-80K frame; passing a different
  target without a matching structural reference produces a configuration
  matched to an arbitrary energy rather than to a physically grounded
  phase.
- **The relaxed coherence comes from an optimiser, not molecular dynamics.**
  The relaxation step matches a structure factor and an energy; its
  long-wavelength time scales are the least certain part of the model, and
  beyond the 256 Å box (below 1/90 Å⁻¹) they are extrapolated as
  diffusive. Scaling the measured curves linearly with \(\sigma_0^2\)
  away from the kick they were measured at is untested.
- **\(\sigma_0^2\) has been checked against two motion-corrected datasets.**
  McMullan et al.'s 0.38 Å² per e⁻/Å² was measured on unaligned frames.
  Motion correction aligns patches hundreds of ångströms across and does
  not remove molecular displacements, so the same value is expected to
  hold for motion-corrected particles. With the ice thickness measured
  from the electron counts of the raw movies against a vacuum exposure,
  0.38 reproduces the water ring of EMPIAR-11461 (ring strength 1.20
  against 1.25). On EMPIAR-11377, whose thickness rests on a published
  vacuum dose rate, the simulated ring is 1.27 against 1.20. From a
  particle stack alone, \(\sigma_0^2\) trades off against ice thickness.
  The filter also acts before multislice, which is exact only for the
  projected, linear image.
- **Long-wavelength ice power differs between library and relaxed ice.**
  After many kick-and-relax steps the ice carries 0.5 to 0.9 of the
  library's density fluctuation at 10 to 90 Å. Both are outputs of the same
  structure-matching optimisation, and neither has been checked against a
  measured compressibility of amorphous ice.

## References

- McMullan, G., Vinothkumar, K. R., & Henderson, R. (2015). Thon rings from
  amorphous ice and implications of beam-induced Brownian motion in single
  particle electron cryo-microscopy. *Ultramicroscopy*, 158, 26–32.
  [doi:10.1016/j.ultramic.2015.05.017](https://doi.org/10.1016/j.ultramic.2015.05.017)
- Chan, H., Cherukara, M. J., Narayanan, B., Loeffler, T. D., Benmore, C.,
  Gray, S. K., & Sankaranarayanan, S. K. R. S. (2019). Machine learning
  coarse grained models for water. *Nature Communications*, 10, 379.
  [doi:10.1038/s41467-018-08222-6](https://doi.org/10.1038/s41467-018-08222-6)
- Tersoff, J. (1988). New empirical approach for the structure and energy
  of covalent systems. *Physical Review B*, 37(12), 6991–7000.
  [doi:10.1103/PhysRevB.37.6991](https://doi.org/10.1103/PhysRevB.37.6991)
