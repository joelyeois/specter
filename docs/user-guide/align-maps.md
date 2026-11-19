# Align maps

`specter align` brings a map into a chosen frame by one rotation and one
shift, and writes it resampled there. `specter align symmetry` chooses the
frame in which the map's point-group axes take their standard positions;
`specter align volumes` chooses the frame of a second map. Both read and
write cubic `.mrc` volumes, take no config, and are available from Python as
[`specter.alignment`](../api/alignment.md).

A reconstruction determines its map only up to a rigid motion, so two
independent reconstructions of one specimen rarely share a frame. Comparing
them voxel by voxel, by a Fourier shell correlation or a difference map,
requires that one be moved onto the other first, and imposing a symmetry
requires that the axes be where the symmetry operators expect them.

## Align a map to its symmetry axes

```bash
specter align symmetry map.mrc map_aligned.mrc --sym D7
```

```text
  rotation (aligned x, y, z -> input x, y, z):
     0.945475  -0.063173  -0.319510
    -0.042161   0.949015  -0.312399
     0.322955   0.308836   0.894606
  centre (x, y, z): 6.177  -2.047  -4.072 voxels = 4.633  -1.535  -3.054 Å
  score 0.2925, significance 67.2, to 94 Fourier voxels
  ✓ map_aligned.mrc
```

`--sym` names the point group: `Cn`, `Dn`, `T`, `O` or `I`. The output has
its symmetry centre on the central voxel and its axes in the positions that
`specter.symmetries` uses, so it can be symmetrised
directly. A symmetric map has several equivalent standard frames, and the
command takes the one that turns the map least. A cyclic group fixes neither
the rotation about its axis nor the position along it, and both are left as
they were in the input.

The rotation is printed as a matrix whose columns are the aligned x, y and z
axes expressed in the input frame. The centre is the point of the input map,
measured from its central voxel, that becomes the central voxel of the
output.

## Align one map to another

```bash
specter align volumes map_a.mrc map_b.mrc map_a_on_b.mrc
```

The first map is moved onto the second, which is left unchanged, and the
result is written on the second map's grid. The two maps must have the same
box and voxel size. They may differ in scale, in offset and in noise.

Two maps of a symmetric specimen can be aligned to each other directly.
Aligning each to its symmetry axes is not equivalent in general: a cyclic
map keeps its rotation about the axis, and two such maps therefore still
differ by that rotation.

## Reading the report

`score` is a correlation over the resolution band the search used, between
the symmetry copies of the map or between the two maps, after each
resolution shell has been scaled to equal weight. It is 1 for exact
agreement and falls with noise, so a low score on a noisy map is not a
failure.

`significance` is the agreement measured in resolution shells that the
search had not yet used when it chose the orientation, in standard
deviations of what noise alone would give. A value below 4 is reported with
a `!` in place of the `✓`, and the pose should not be trusted.

The last figure is the band limit the refinement reached, in Fourier
voxels. The search starts from 8 Fourier voxels and doubles the band while
the added shells still agree, up to half the Nyquist frequency.

## Method

The rotation is found before the shift. The intensities of a map's Fourier
transform do not change when the map is shifted, so comparing intensities,
between symmetry copies or between two maps, isolates the rotation, as in
the rotation function of Rossmann & Blow (1962). The comparison runs over a
uniform grid of orientations (Alexa 2022) in a band of 8 Fourier voxels,
where an exhaustive search is cheap. The shift for each of the best
orientations follows from the phases. The leading candidates are then
refined, rotation and shift together, by a gradient-based optimiser in
bands of doubling radius.

The output is resampled by an exact rotation, in which nothing is
interpolated, so the aligned map loses no resolution.

## Limitations

- Both commands require cubic volumes, and `specter align volumes` requires
  equal box and voxel sizes.
- Only proper rotations are searched. A map of the opposite hand is not
  aligned to its mirror image.
- `significance` tests the pose against noise. A clean map given the wrong
  point group, or two clean maps of different specimens, align to their best
  partial match with a high significance; the mismatch shows as a `score`
  well below 1.
- A map of a symmetric specimen aligned to another has several equivalent
  poses, and any one of them may be returned.

## References

- Alexa, M. (2022). Super-Fibonacci spirals: Fast, low-discrepancy sampling of SO(3). *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 8281–8290. [doi:10.1109/CVPR52688.2022.00811](https://doi.org/10.1109/CVPR52688.2022.00811)
- Rossmann, M. G., & Blow, D. M. (1962). The detection of sub-units within the crystallographic asymmetric unit. *Acta Crystallographica*, 15(1), 24–31. [doi:10.1107/S0365110X62000067](https://doi.org/10.1107/S0365110X62000067)
