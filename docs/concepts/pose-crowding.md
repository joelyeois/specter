# Pose & crowding

Single-particle generation adds two things to the shared
[pipeline](pipeline-overview.md) that the cryo-ET path handles
differently: a pose sampled independently per image, and, optionally,
duplicates of the same template scattered around the primary instance to
mimic local particle crowding.

!!! info "Source"
    `specter.rotations` (pose representation and application),
    `specter.arrays._padding` (Z/XY padding), `specter.crowding`
    (`CrowdWithDuplicates`, `filter_by_z_density`,
    `filter_by_local_z_density`).

## Pose

Each image gets its own rotation quaternion and in-plane translation;
`ImageGenerator`/`ImageGeneratorFromCoordinates` draws each once per
instance and stores them as `(B, 4)` and `(B, 2)` tensors. Every other
pipeline in SPECTER uses the same representation, sign conventions, and
real-space/Fourier-space application choice;
[Conventions](conventions.md#poses) describes them once, and this page
does not repeat them.

The two particle-generator classes differ in *where* the pose acts.
`ImageGenerator` rotates the pre-built volume itself.
`ImageGeneratorFromCoordinates` instead rotates the atomic coordinates
before voxelization, using the inverse rotation matrix
(`roma.unitquat_to_rotmat(Q).transpose(-2, -1)`) and applying an Å-space
translation directly to the coordinates, rather than through the
`[-1, 1]`-normalized grid `grid_sample` uses. Both reach the same
convention; see
[Pipeline overview](pipeline-overview.md#how-the-generator-classes-use-it)
for why the two paths differ in cost and in whether pose gradients reach
per-atom coordinates.

## Padding

Two independent paddings feed into the volume `PotentialBuilder` produces
before the rest of the chain runs:

- **Z-padding** extends the volume to accommodate the requested ice
  thickness. `compute_nz(base_nz, ice_thickness, pixel_size)` leaves the
  depth unchanged when `ice_thickness` is `None` or already smaller than
  the particle's own extent, and otherwise sets `nz = ice_thickness //
  pixel_size`, zero-padding symmetrically to reach it.
- **XY-padding** (`pad_fft`) adds `nxy // 2` pixels on each side before
  scattering, to keep the specimen's own energy away from the boundary
  wraparound that FFT-based propagation and reflect-mode ice padding
  would otherwise introduce. It is off by default; enabling it trades
  memory and compute for reduced edge artifacts at the field of view's
  boundary.

`pad_volume` applies both after pose but before crowding, scaling, and
ice, so every later stage in the chain sees the final, padded extent.

## Crowding

`CrowdWithDuplicates`, attached to a generator when `crowd_min_distance`
is set, adds rigid-transformed copies of the *same* template volume
around the primary instance, each independently rotated and placed by
Poisson-disk sampling (2D, in the image plane only, or 3D, throughout the
volume). This models local particle crowding within one particle's box,
a different mechanism from `MicrographGenerator`'s field-of-view
assembly, which places independent particle species, usually different
from each other, across a whole micrograph; see
[Pipeline overview](pipeline-overview.md#how-the-generator-classes-use-it).

`crowd_min_distance` (the Poisson-disk minimum separation) controls
sampling density; `crowd_max_distance_z` and an implicit XY bound
derived from the volume size cap it. `n_points=inf` by default, so
sampling continues until the box is full rather than stopping at a fixed
count. `crowd_chunk_size` limits how many duplicate volumes
`CrowdWithDuplicates` rotates per batch, trading GPU memory for speed;
the default of 1 is the memory-safe choice, and `None` rotates every
duplicate at once.

### Water–air interface adsorption

Cryo-EM specimens preferentially adsorb particles at the two ice–air
interfaces rather than distributing them uniformly through the ice's
thickness. `water_air_interface=True` reproduces this with a
two-Gaussian probability profile along \(z\), peaked at the top and
bottom surfaces and falling to a configurable `baseline` in the bulk;
placement keeps or rejects each candidate position by sampling against
that profile rather than uniformly.

Without an `IceProfile`, both surfaces are a single global pair at
\(\pm z_\text{length}/2\) (`filter_by_z_density`). With one, ice
thickness varies laterally (a wedge or an otherwise non-flat specimen),
so a global pair of surfaces would place adsorbed particles at the *mean*
surface, which sits outside the ice in the thinnest regions.
`filter_by_local_z_density` instead evaluates each candidate against the
ice surfaces at its own \((x, y)\) column. The same `ice_profile` also
gates ordinary 3D placement regardless of `water_air_interface`: it
rejects outright any candidate whose particle radius would poke through
its local surface, so a placement never lands partly in vacuum.
