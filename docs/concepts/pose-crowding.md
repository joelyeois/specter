# Pose & crowding

Single-particle generation adds two things to the shared
[pipeline](pipeline-overview.md) that the cryo-ET path handles
differently: a pose sampled independently per image, and, optionally,
duplicates of the same template scattered around the primary instance to
mimic local particle crowding.

!!! example "In code"
    This page is the theory behind [Specimens: crowding and ice](../walkthrough/specimens.md#crowding) and [Scattering](../walkthrough/scattering.md#orienting-the-particle), which show it step by step with code and figures.

!!! info "Source"
    `specter.rotations` (pose representation and application),
    `specter.arrays._padding` (Z/XY padding), `specter.crowding`
    (`CrowdWithDuplicates`, `filter_by_z_density`,
    `filter_by_local_z_density`).

## Pose

Each image gets its own rotation quaternion and in-plane translation,
held by `Poses` as `(B, 4)` and `(B, 2)` tensors (`Poses.random` draws them
uniformly, the rotations by the method of Shoemake 1992). Every other
pipeline in SPECTER uses the same representation, sign conventions, and
real-space/Fourier-space application choice;
[Conventions](conventions.md#poses) describes them once, and this page
does not repeat them.

The two kinds of particle `Specimen` differ in *where* the pose acts.
A specimen with a pre-built volume rotates the volume itself. One built
with `Specimen.from_atoms` instead rotates the atomic coordinates
before voxelization, using the inverse rotation matrix
(`roma.unitquat_to_rotmat(Q).transpose(-2, -1)`, from the roma library;
Bregier 2021) and applying an Å-space
translation directly to the coordinates, rather than through the
`[-1, 1]`-normalized grid `grid_sample` uses. Both reach the same
convention; see
[Pipeline overview](pipeline-overview.md#how-the-objects-use-it)
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
  would otherwise introduce. From Python it is off for a particle box
  unless asked for (`Propagation.pad_fft` is `None`, the path's own
  default); `specter simulate particles` turns it on. It trades memory and
  compute for reduced edge artifacts at the field of view's boundary.

`pad_volume` applies both after pose but before crowding, scaling, and
ice, so every later stage in the chain sees the final, padded extent.

## Crowding

`CrowdWithDuplicates`, recorded on a specimen by
`Specimen.crowd(min_distance=...)`, adds rigid-transformed copies of the *same* template volume
around the primary instance, each independently rotated and placed by
Poisson-disk sampling (Bridson 2007), in 2D (the image plane only) or 3D
(throughout the volume). This models local particle crowding within one particle's box,
a different mechanism from a micrograph's `Field`, which packs copies of
its template across a whole field of view with a `Packing` collision
backend; see
[Pipeline overview](pipeline-overview.md#how-the-objects-use-it).

`Crowding.min_distance` (the Poisson-disk minimum separation) controls
sampling density; `max_distance_z` and `max_distance_xy` bound the region
it fills, in Å. Left unset, `max_distance_xy` is the template's edge length
plus `min_distance` (\(n\,\Delta x + d_\text{min}\)). In a TOML config the
same fields carry a `crowd_` prefix (`crowd_min_distance`,
`crowd_max_distance_z`, `crowd_max_distance_xy`). `n_points=inf` by default, so
sampling continues until the box is full rather than stopping at a fixed
count.

Left unset, `max_distance_z` is the particle box's own depth, and
it does not follow `ice_thickness`. The two are separate knobs on
purpose: raising `ice_thickness` lowers contrast and deepens the solvent
background, and letting it also deepen the neighbour slab would make a
single field change both the imaging conditions and the specimen. In a
thick-ice box the neighbours therefore occupy the middle of the column
and the remainder is water. Set `max_distance_z` explicitly to crowd a
deeper slab. `MicrographSpecimenGenerator` scales its own neighbour slab
with the volume depth instead, since a micrograph is a specimen rather
than a controlled single-particle experiment. `chunk_size` limits how many duplicate volumes
`CrowdWithDuplicates` rotates per batch, trading GPU memory for speed;
the default of 1 is the memory-safe choice, and `None` rotates every
duplicate at once.

Before rotating, `CrowdWithDuplicates` crops the template to the smallest
centred cube that no rotation about its centre can move density out of
(`crowding.rotation_safe_crop`). A template's box is sized for the image
rather than for the molecule, so most of it is empty, and rotation preserves
distance from the centre. The crop's half-width is the radius of the
furthest occupied voxel plus \(\sqrt{3}\) voxels, because trilinear
resampling can carry density from radius \(R\) to one voxel diagonal
further out, plus \(\sqrt{3}/2\) voxels for the sub-voxel offset described
below. Each axis keeps its parity, so the crop's centre stays where the
template's was. The crop changes the image only through the resampling
grid's normalisation, at the \(10^{-5}\) relative level, and draws no
random numbers, so a seeded specimen is unchanged.

Each duplicate is placed at its sampled position to sub-voxel accuracy.
The rotated copy is added into the canvas at the whole voxel nearest that
position, and the remainder, up to half a voxel on each axis, is applied
as a translation in the same affine resample that rotates the copy
(`crowding.fold_subvoxel_offsets`), so it costs no additional
interpolation pass. Trilinear resampling conserves the first moment of
the density, so the copy's centroid lands on the sampled position to
within interpolation error (about \(10^{-3}\) voxel for a smooth
template), not to within half a voxel. A tomogram's proteins, filaments,
microtubules and transmembrane proteins are placed through the same function.
[Conventions](conventions.md#canvas-coordinates) gives the frame these
positions are in, and how gold beads and membranes, which are not resampled,
reach the same accuracy.

### Water–air interface adsorption

Cryo-EM specimens preferentially adsorb particles at the two ice–air
interfaces rather than distributing them uniformly through the ice's
thickness (Noble et al. 2018). `water_air_interface=True` reproduces this with a
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

## References

- Bregier, R. (2021). Deep regression on manifolds: A 3D rotation case
  study. In *2021 International Conference on 3D Vision (3DV)*
  (pp. 166–174). IEEE.
  [doi:10.1109/3DV53792.2021.00027](https://doi.org/10.1109/3DV53792.2021.00027)
- Bridson, R. (2007). Fast Poisson disk sampling in arbitrary dimensions.
  In *ACM SIGGRAPH 2007 Sketches* (p. 22). ACM.
  [doi:10.1145/1278780.1278807](https://doi.org/10.1145/1278780.1278807)
- Noble, A. J., Dandey, V. P., Wei, H., Brasch, J., Chase, J., Acharya, P.,
  et al. (2018). Routine single particle CryoEM sample and grid
  characterization by tomography. *eLife*, 7, e34257.
  [doi:10.7554/eLife.34257](https://doi.org/10.7554/eLife.34257)
- Shoemake, K. (1992). Uniform random rotations. In *Graphics Gems III (IBM
  Version)* (pp. 124–132). Elsevier.
  [doi:10.1016/B978-0-08-050755-2.50036-1](https://doi.org/10.1016/B978-0-08-050755-2.50036-1)
