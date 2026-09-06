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
   arrives pre-built (`ImageGenerator`, `MicrographGenerator` when given
   `volume`), or `PotentialBuilder` builds it from atomic coordinates on
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
5. **Ice (solvation)** *(optional)*. SPECTER adds an amorphous ice
   volume, blending it in only where the specimen's own potential is
   otherwise zero. See [Ice structure](ice.md).
6. **Scattering.** `Scattering` propagates the electron wave through the
   finished volume to a complex exit wave \(\psi(x,y)\). See
   [Scattering](scattering/index.md).
7. **Aberrations.** `Aberration` applies the microscope's transfer
   function (defocus, spherical aberration, astigmatism, and
   coherence/dose envelopes) to the exit wave. See
   [Aberrations](aberrations.md).
8. **Detector.** `Detector` turns the aberrated wave into expected
   electron counts per pixel, modelling MTF, DQE(0), shot noise, and (for
   direct electron detectors) coincidence loss. See
   [Detector](detector.md).

Stages 6 through 8 are [forward simulation](forward-simulation.md)
proper; stages 1 through 5 assemble the volume that forward
simulation runs on. The split also marks where the
single-particle and cryo-ET pipelines diverge: they build \(V\)
differently but hand it to the same downstream chain.

## How the generator classes use it

- **`ImageGeneratorFromCoordinates`** rebuilds \(V\) from atomic
  coordinates on every `forward()` call: it rotates the coordinates, then
  `PotentialBuilder` voxelizes them. This is the more expensive path per
  call, and the one that lets pose refinement backpropagate into atomic
  coordinates (used by `Ghostbuster`'s coordinate-space mode).
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

`Reconstructor` and `TomogramReconstructor` optimize a volume (and
optionally pose, translation, and defocus) by running this exact forward
chain on a candidate volume and comparing the result against observed
images. One shared forward model serves both simulation and
reconstruction: change any stage's physics, and both directions follow.
See [Reconstruct a volume](../user-guide/reconstruction.md).
