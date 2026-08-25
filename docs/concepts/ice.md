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

Positions start out uniformly random; L-BFGS then optimises them (the
default; Adam is also available) against the combined loss

\[
\mathcal{L} = \mathcal{L}_{S(k)} + w_\text{MLBOP}\,\mathcal{L}_\text{MLBOP}
\]

L-BFGS's strong-Wolfe line search adapts its own step size, so it
converges in roughly 50 outer iterations against several hundred for
Adam on this smooth radial-profile loss, and the `lr` parameter rarely
needs tuning as a result. Optimisation runs until `n_steps`, or stops
early once the fractional change in loss stays below `tol` for `patience`
consecutive steps.

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

## References

- Chan, H., Cherukara, M. J., Narayanan, B., Loeffler, T. D., Benmore, C.,
  Gray, S. K., & Sankaranarayanan, S. K. R. S. (2019). Machine learning
  coarse grained models for water. *Nature Communications*, 10, 379.
  [doi:10.1038/s41467-018-08222-6](https://doi.org/10.1038/s41467-018-08222-6)
- Tersoff, J. (1988). New empirical approach for the structure and energy
  of covalent systems. *Physical Review B*, 37(12), 6991–7000.
  [doi:10.1103/PhysRevB.37.6991](https://doi.org/10.1103/PhysRevB.37.6991)
