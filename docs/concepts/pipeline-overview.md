# Pipeline overview

Every SPECTER forward simulator, whether it produces a single particle
image, a full micrograph, or a cryo-ET tilt series, runs the same ordered
sequence of stages on a 3D potential volume. The three differ only in
what builds that volume and how many times the chain repeats per run.
This page lays out the sequence once; each stage has its own page with
the underlying math.

!!! info "Source"
    `specter.imagegenerator._base.BaseImager` and
    `specter.imagegenerator._particle_base.ParticleGeneratorBase` define
    the shared pipeline. Concrete subclasses
    (`ImageGenerator`/`ImageGeneratorFromCoordinates`,
    `MicrographGenerator`, `TiltSeriesGenerator`) supply the volume and
    call into it.

## The chain

1. **Potential.** A 3D electrostatic potential volume \(V(x,y,z)\) either
   arrives pre-built (`ImageGenerator`, or `MicrographGenerator` when its
   `specimen` is a volume rather than a `MicrographSpecimenGenerator`), or
   `PotentialBuilder` builds it from atomic coordinates on
   the fly (`ImageGeneratorFromCoordinates`). See
   [Specimens](specimens.md) and [Atomic potentials](atomic-potentials.md).
2. **Pose.** SPECTER rotates and translates the volume (or, equivalently,
   the atomic coordinates before voxelization) per simulation instance.
   See [Pose & crowding](pose-crowding.md) for the representation and
   [Conventions](conventions.md#poses) for the sign and origin
   conventions.
3. **Crowding** *(optional)*. SPECTER Poisson-disk places
   rigid-transformed duplicates of the same volume around the primary
   instance and sums them in, before ice. See
   [Pose & crowding](pose-crowding.md#crowding).
4. **Potential scaling.** SPECTER multiplies each instance's volume by a
   per-image scalar, `potential_scale`, before propagation. Together with
   dose, `potential_scale` lets a batch mix imaging conditions in one
   forward pass.
5. **Absorption field** *(optional)*. Under
   `absorption_model="inelastic_mfp"` with a specimen mean free path,
   SPECTER reads the imaginary potential off the specimen before any ice is
   added, since the material fraction of a voxel is only defined while the
   volume holds the specimen alone. Without a specimen mean free path the
   absorption is uniform, and `Scattering` applies it as a scalar instead
   of a field. See
   [Scattering](scattering/index.md#mean-free-path-absorption-support).
6. **Specimen radiation damage** *(optional)*. Under
   `dose_envelope_target="specimen"`, SPECTER applies the dose envelope to
   the specimen's 3D potential, crowding neighbours included, before the
   ice is added. The occupancy that weights the ice is read from the
   undamaged specimen. See [Aberrations](aberrations.md#envelopes).
7. **Ice (solvation)** *(optional)*. SPECTER adds an amorphous ice
   volume, weighting it in each voxel by the fraction that the specimen
   leaves free: \(1 - \mathrm{clamp}(\bar V / 7.0\,\mathrm{V}, 0, 1)\),
   where \(\bar V\) is the specimen potential coarse-grained by a 2 Å
   Gaussian and 7.0 V is the mean inner potential of protein. Under
   `Ice(motion_variance=...)` the ice fluctuation is first filtered to what
   survives the summed exposure, with its mean kept. See
   [Ice structure](ice.md).
8. **Objective aperture** *(optional)*. When `objective_aperture` is set,
   the potential is low-passed at the aperture's cut-off, since the
   scattering beyond it is charged as absorption rather than propagated.
   This has no effect at 1 Å/px and coarser.
9. **Scattering.** `Scattering` propagates the electron wave through the
   finished volume to a complex exit wave \(\psi(x,y)\). See
   [Scattering](scattering/index.md).
10. **Aberrations.** `Aberration` applies the microscope's transfer
    function (defocus, spherical aberration, astigmatism, and
    coherence/dose envelopes) to the exit wave. With `optics=None` this
    stage is skipped. See [Aberrations](aberrations.md).
11. **Detector.** `Detector` turns the aberrated wave into expected
    electron counts per pixel, modelling MTF, DQE(0), shot noise, and (for
    direct electron detectors) coincidence loss. See
    [Detector](detector.md).

Stages 9 through 11 are [forward simulation](forward-simulation.md)
proper; stages 1 through 8 assemble the volume that forward
simulation runs on. The split also marks where the
single-particle and cryo-ET pipelines diverge: they build \(V\)
differently but hand it to the same downstream chain.

## How the generator classes use it

- **`ImageGeneratorFromCoordinates`** rebuilds \(V\) from atomic
  coordinates on every `forward()` call: it rotates the coordinates, then
  `PotentialBuilder` voxelizes them. This is the more expensive path per
  call, and the one that keeps gradients with respect to the atomic
  coordinates.
- **`ImageGenerator`** takes one pre-built volume and rotates the *volume*
  itself each call (`grid_sample`, or a Fourier-space rotation when
  `rotate_mode="fourier"`; see
  [Conventions](conventions.md#applying-a-pose-real-space-or-fourier-space)).
  Cheaper per call than the coordinate path, at the cost of losing
  per-atom gradients.
- **`MicrographGenerator`** images a whole field of view rather than one
  box. Its specimen is either a pre-built volume or a
  `MicrographSpecimenGenerator`, which places duplicates of one template
  across the micrograph with the same `Crowding` machinery (plus a
  `Packing` collision backend and its own `Ice`), and can rebuild the
  specimen for every micrograph. It then runs `IterativeScattering`
  slice-by-slice over the whole assembled volume.
- **`TiltSeriesGenerator`** runs the same scattering → aberration →
  detector chain once per tilt angle, using `IterativeScattering` to
  resample Z-slices from the volume under each tilt's affine pose rather
  than materializing a full rotated copy per angle (see
  [Scattering](scattering/index.md#scattering-vs-iterativescattering)).
  Dose accumulates across tilts; the per-tilt physics is otherwise
  unchanged from a single particle image.

## The same chain runs in reverse

`Reconstructor` and `TomogramReconstructor` optimize a volume by running
the scattering and aberration stages of this chain on a candidate volume
and comparing the result against observed images. `Reconstructor` can also
refine pose, translation and defocus; `TomogramReconstructor` refines the
volume only, since its tilt poses are fixed. Both use the same scattering
and transfer-function code as the simulators, so a change to either
stage's physics reaches both directions.

The inverse does not run the whole chain. It accepts only the `Propagation`
and `Optics` settings groups. `Envelopes` and `Camera` are absent on
purpose: an experimental dataset gives no way to determine them, and the
high-frequency loss they describe is absorbed into the reconstructed
volume, so assuming values for them would add information the data do not
contain. Both reconstructors also reject `absorption_model="inelastic_mfp"`.
See [Reconstruct a volume](../user-guide/reconstruction.md).
