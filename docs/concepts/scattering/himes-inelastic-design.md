# Himes-style inelastic scattering: implementation sketch

The opt-in Python backend is now implemented. See [the API and its current
limits](frozen-plasmon.md). The sections below record the original design;
measured EELS calibration and an experimental/cisTEM comparison remain open.

Himes and Grigorieff (2021), [Theory and §§3.3–3.5](https://journals.iucr.org/m/issues/2021/06/00/rq5007/),
construct a composition-dependent imaginary potential with an EELS-derived
plasmon spectrum, and simulate configurations over exposure. Their angular
cross section is proportional to `1 / (theta**2 + theta_E**2)`, with
`theta_E = loss_energy / (2 * incident_energy)`. They integrate measured
low-loss spectral weights over 7.5–100 eV and take a square root to obtain
the scattering-factor shape. Per-atom inelastic/elastic probabilities set
its strength. Aperture losses remain in the optical stage.

## 1. Separate material data from elastic potential

Extend potential building to produce an elastic volume and an inelastic
source field from atomic species, number density, and calibrated cross
sections. Coarse-grained solvent needs its own scattering-center weights.
Keep explicit solvent and vacuum support. Do not infer every material from
the protein-calibrated 7 V occupancy threshold. For reconstruction from a
real potential alone, require supplied material fractions or an explicit
material prior.

Proposed interface: an absorption provider receives material slices and
beam parameters and returns an imaginary potential in volts. Both eager
and iterative propagation should use that interface. Existing alpha and
MFP behavior provide reference backends, with complex input retaining its
already-absorptive meaning.

## 2. Build and calibrate a transverse plasmon kernel

Tabulate an empirical loss spectrum with provenance, energy units, and
normalization. Evaluate the energy-integrated angular distribution on the
simulation frequency grid (`theta` approximately wavelength times spatial
frequency), then form the amplitude kernel. Cache by voltage, sampling,
canvas, spectrum, and filter acceptance.

Before implementation, inspect the paper's supplement and cisTEM source to
resolve discrete normalization, angular cutoff, DC handling, and exactly
how elastic-derived source weights are reshaped. Taking a square root of a
cross section alone does not specify an absorption potential in volts.
Use measured transmission to validate normalization independently of the
spatial spectrum. A matching ice MFP alone is not proof that the spatial
model is correct.

Apply the transverse operator after rotation into the beam frame, using
sufficient halo/padding to avoid periodic wraparound. A beam-transverse
operator cannot generally be applied once in the molecule frame and then
rotated for all poses. Check padding convergence and nonnegative absorption
without silently clipping a misnormalized kernel.

## 3. Share propagation and gradients

For every slice, construct `V_el + 1j * V_inel` and use the existing
transmission exponential. Stream real and imaginary slice chunks instead
of allocating a complex tomogram. Rotate/sample both fields with matching
geometry, preserving vacuum outside the specimen. Checkpoint both in the
iterative implementation. Define whether material fields are fixed priors
or differentiable functions of the reconstructed density; test the
corresponding gradients and eager/iterative parity.

## 4. Add exposure-dependent configurations separately

Start with a static spectral backend. A full frozen-plasmon implementation
also needs seeded configuration sampling over exposure, with appropriate
solvent motion and specimen changes. Propagate each configuration, combine
its detector-plane intensity with exposure weights, and apply detector
statistics consistently. Averaging potentials or complex waves is not
interchangeable with averaging intensities. Coordinate this with existing
dose envelopes to avoid applying the same damage twice.

## 5. Validation and scope

Validate homogeneous ice transmission over thickness and sampling; mixed
materials and vacuum interfaces; absorption spatial spectra; protein–ice
contrast; exposure-dependent solvent noise; objective-aperture effects;
and forward/gradient parity under tilt. Compare the static MFP, spectral,
and configuration-ensemble variants with identical elastic potentials and
incident dose. Use experimental zero-loss images for absolute-count and
contrast checks.

Treat this as a zero-loss-channel model. General unfiltered imaging needs
an additional model for detected inelastic intensity and its energy-dependent
optics, rather than merely changing the attenuation constant.
