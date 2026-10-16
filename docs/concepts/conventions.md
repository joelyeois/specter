# Conventions

This page collects the sign, unit, axis, and pose conventions that
`specter` follows. Use it in two situations: importing experimental
metadata into a simulation, and comparing `specter`'s output against
another package. Both fail silently if you mismatch a convention: a
wrong sign or a wrong axis produces a plausible image, not an error.

This page does not repeat definitions that belong to the file formats
themselves, such as the meaning of the Euler angles of RELION (Scheres
2012) or the layout of a STAR file; see
[RELION's conventions page](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html).

!!! example "In code"
    This page is the theory behind [Scattering](../walkthrough/scattering.md) and [Contrast transfer function](../walkthrough/ctf.md#the-defocus-reference-plane), which show it step by step with code and figures.

!!! info "Source"
    `specter.constants`, `specter.fft`, `specter.rotations`,
    `specter.aberrations._functions`, `specter.scattering`,
    `specter.microscope`, `specter.io`.

## Units

Units are not internally uniform. They mirror the upstream file formats
each quantity comes from, which keeps `.cs` and `.star` values usable
without conversion at the boundary.

| Quantity | Unit |
|---|---|
| Lengths, pixel size, voxel size | Å |
| Defocus (`dfu`, `dfv`) | Å, positive = underfocus |
| Spherical aberration (`cs`) | Å |
| Astigmatism angle (`dfang`) | degrees |
| Phase shift (`phaseshift`), beam tilt (`tiltx`, `tilty`) | radians |
| Accelerating voltage | kV |
| Dose | e⁻/Å² per image |
| B-factor | Å² |
| Spatial frequency | 1/Å |
| Electrostatic potential | V |
| Tilt angles | degrees |

The mixed degrees/radians row is the one to watch: within a single
`ctf_params` dict, `dfang` is in degrees while `phaseshift`, `tiltx`, and
`tilty` are in radians. This matches [CryoSPARC](https://cryosparc.com/)
(Punjani et al. 2017):
its `.cs` files store `ctf/df_angle_rad` in radians, and `io/_cryosparc.py`
converts it to degrees on the way in.

A second, larger unit distinction separates the two CTF backends. The
legacy `ctf_params` dict uses the units above. The `torch_ctf` backend's
native `CTFParameters` type instead uses micrometres for defocus,
millimetres for \(C_s\), and dimensionless Zernike coefficients.
`LegacyAberrationAdapter` converts between them, so a caller passing a
`ctf_params` dict never sees the difference, but a caller constructing
`CTFParameters` directly must use the native units.

## Array layout

Volumes are `(B, Z, Y, X)` and images are `(B, Y, X)`. \(Z\) is the beam
direction, the second-to-last axis is \(y\) (rows), and the last axis is
\(x\) (columns). This is the same order the MRC format (Cheng et al. 2015)
uses on disk, so writing
volumes and image stacks needs no transposition.

## Canvas coordinates

Positions are in Å, relative to the centre of the volume they are placed in.
Voxel \(k\) of an axis of \(n\) voxels of size \(\Delta\) sits at
\((k - \lfloor n/2 \rfloor)\,\Delta\), so the origin lies exactly on voxel
\(\lfloor n/2 \rfloor\) whatever the parity of \(n\). This is the index the
default rotation origin and a centred spectrum's DC term also use. The faces
of the volume are at \(-(\lfloor n/2 \rfloor + \tfrac{1}{2})\,\Delta\) and
\((n - \lfloor n/2 \rfloor - \tfrac{1}{2})\,\Delta\), so an even axis is
centred half a voxel below the origin rather than on it.

Everything that places density into a volume records its position in this
frame and renders at that position:

- Rotated copies (crowding duplicates, micrograph particles, tomogram
  proteins, filament subunits, tubulin dimers and transmembrane proteins) are
  added at the nearest whole voxel, and the remainder, at most half a voxel
  per axis, is applied by the affine resample that rotates them
  (`crowding.fold_subvoxel_offsets`).
- Gold beads are built with the remainder already applied to their atoms and
  boundary (`BeadGenerator.generate(offset_xyz=...)`), since resampling a
  finished bead would blur its lattice.
- A membrane instance is merged into the volume a whole voxel at a time, so
  its position is snapped to the nearest voxel before its transmembrane
  proteins, labels or region masks use it. Its own working grids put the
  organelle's centre on their voxel \(\lfloor m/2 \rfloor\).
- The shape packer records the render voxel a footprint was tested at, and
  the sphere packer's walls are the faces of the volume.
- An `IceProfile` evaluates its thickness, mid-plane and window on the same
  grid, so a particle and the ice column it is placed in agree.

## Fourier conventions

**FFTs are unshifted by default.** `specter.fft`'s `fft2`/`ifft2`/`fft3`
and their inverses return native PyTorch ordering, with the zero
frequency at index 0, unless you pass `shift=True`. Build any mask,
envelope, or propagator you construct yourself in the same ordering. A
disk centred at index \(N/2\), the `fftshift` convention, multiplied into
an unshifted spectrum removes the zero frequency and retains the Nyquist
corners.

**Frequency grids are in 1/Å.** `specter` builds them from
`torch.fft.fftfreq(n, pixel_size)`, so the Nyquist frequency is
\(1/(2\,\Delta x)\) and no cycles-per-pixel conversion is involved.

**`specter` measures azimuth from \(+y\).** On the 2D frequency grid,
`KY` varies along axis 0 and `KX` along axis 1, and the azimuthal angle is
\(\theta = \operatorname{atan2}(k_x, k_y)\). This lets `dfang` from a
`.cs` or `.star` file work directly, with no 90° offset.

## Poses

Rotations are unit quaternions in the scalar-last \((x, y, z, w)\)
ordering of `roma` (Bregier 2021). Translations are in Å.

**A quaternion \(Q\) rotates the density by \(R(Q)^{-1}\).** Rotate a
feature at \(+x\) by \(+90^\circ\) about \(z\) and it lands at \(-y\).
The volume path (`Specimen.scatter`'s `VolumeRotator`, which passes \(R\)
to `grid_sample`) and the coordinate path (`Specimen.from_atoms`, which
applies \(R^{\mathsf{T}}\) to atom positions) look like opposites in
source but produce the same
pose, because sampling a volume at \(R\mathbf{x}\) rotates its density by
\(R^{-1}\). This is the interpretation under which CryoSPARC rotation
vectors and RELION Euler angles map in without inversion.

**Translations follow RELION's origin-offset semantics.** `specter`
subtracts them rather than adding them: a translation of \((+10, 0)\) Å
moves a feature at \(x = +10\) Å to the box centre. The shift acts in the
image plane, *after* the rotation. This is the same frame RELION and
CryoSPARC define their origin offsets in: a phase ramp on an
already-extracted central slice, not a displacement of the 3D reference.

Expressing an image-plane shift as a single 3D affine requires
`build_affine_matrix` to pre-rotate the translation vector.
`grid_sample` samples the input volume at \(R\mathbf{x} + T'\), which
transforms the density by \(R^{-1}\) and then by \(-R^{-1}T'\). Setting
\(T' = RT\) makes the net displacement \(-T\) in the lab frame,
independent of the rotation. Applying \(T\) unrotated would instead shift
the particle in its own frame before the rotation. Because \(T\) always
carries a zero \(z\) component, translating the volume and translating
its projection are equivalent.

**The rotation origin is RELION's, not the geometric centre.** Volumes
rotate about index \([n_z/\!/2,\, n_y/\!/2,\, n_x/\!/2]\) by default
(`origin="relion"`). Passing `origin="center"` moves it to
\([(n_z-1)/2,\, (n_y-1)/2,\, (n_x-1)/2]\). The difference is half a voxel, which appears as a
systematic subpixel shift when comparing against a package that made the
other choice.

Conversion into `specter`'s representation:

| Source | Field | Conversion |
|---|---|---|
| CryoSPARC `.cs` | `alignments3D/pose` (rotation vector) | `roma.rotvec_to_unitquat` |
| CryoSPARC `.cs` | `alignments3D/shift` (px) | scaled to Å, `ctf/shift_A` subtracted |
| RELION `.star` | `rlnAngleRot/Tilt/Psi` (degrees) | `roma.euler_to_unitquat("ZYZ", degrees=True)` |
| RELION `.star` | `rlnOriginX/YAngst` | used as-is |

## Applying a pose: real space or Fourier space

You apply a pose to a volume by one of three methods, selected by the
`rotate_mode` field of the `Propagation` settings group, passed as
`Specimen.scatter(..., rotate_mode=...)`, or as
`propagation=Propagation(rotate_mode=...)` to `Reconstructor` or
`Ghostbuster`. In a TOML config or on the command line it is the flat field
`rotate_mode` of `ParticleStackConfig` and `ReconstructionConfig`. All three
express the same convention and start from the same affine built above.
`"real"` and `"fourier"` rotate the volume and differ only in where the
interpolation happens; `"surface"` never rotates it (see
[Sampling the Ewald surface](#sampling-the-ewald-surface)).

`"real"`, the default, resamples the density with `grid_sample`, applying the
rotation and the translation together in one trilinear pass. `"fourier"`
transforms the volume, rotates the spectrum's real and imaginary parts, and
applies the translation as a phase ramp \(\exp(-2\pi i\,\mathbf{f}\cdot
\mathbf{d})\) before transforming back.

Two consequences follow from that split. The translation is **exact** under
`"fourier"`, since a phase ramp is not an interpolation, and it is exact for
sub-voxel shifts as well. It is also **circular**: density leaving one face
reappears at the opposite one, where `"real"` pads according to
`padding_mode`.

Rotation accuracy runs the other way from what the name suggests. The
Fourier path interpolates the *spectrum* trilinearly, and an object far
from the box centre has a rapidly oscillating phase that linear
interpolation cannot follow, so its error grows with distance from the
centre while the real-space error stays flat. Measured against an
analytic ground truth, as a fraction of peak density:

| Distance from box centre | `"real"` | `"fourier"` |
|---|---|---|
| 1 voxel | 0.043 | 0.005 |
| 3 voxels | 0.035 | 0.011 |
| 6 voxels | 0.042 | 0.034 |
| 11 voxels | 0.045 | 0.096 |

`"fourier"` is therefore the more accurate choice for a compact particle
centred in a roomy box, and the less accurate one for a crowded or off-centre
specimen, or for any volume whose density reaches the box edge. It costs two
FFTs and two resampling passes rather than one.

Rotation origin follows the same `"relion"` / `"center"` choice described
above. In Fourier space, the rotation necessarily centres on the
spectrum's DC term, which `fft3(..., shift=True)` places at `n // 2`. You
reach `"center"` by folding the half-voxel offset into the same phase
ramp; it works for cubic volumes only.

### Sampling the Ewald surface

For the single-scatter models the slice sum that forms the exit wave is the
volume's transform on a surface. With the propagator \(F_z\) from slice
\(z\) to the exit plane,

\[
\sum_z \mathcal{F}_2[V_z](\mathbf{k})\,F_z(\mathbf{k})
= e^{i\phi(\mathbf{k})}\,
\hat V\!\left(k_x,\,k_y,\,\mp\tfrac{\lambda}{2}|\mathbf{k}|^2\right),
\]

where \(\hat V\) is the volume's discrete-space Fourier transform, the sign
is the Ewald curvature sign, and \(\phi\) collects the propagation to the
exit plane and the phase origin. Rytov exponentiates the same sum and first
Born linearises it; projection and the CTF model sample the plane
\(k_z = 0\). A rotation of the specimen therefore rotates the sampling
surface instead of the density, and an in-plane shift multiplies the result
by a phase ramp. `rotate_mode="surface"`
(`specter.scattering.surface_slice_sum`) evaluates the transform of the
unrotated volume at the rotated points with a type-2 non-uniform FFT
(Barnett et al. 2019; Shih et al. 2021), and its adjoint supplies the
gradient, so no step resamples the grid. On poses that `"real"` applies
without interpolating (quarter turns and whole-voxel shifts), the two agree
to 1e-5 for all four models and both curvature signs.

The method has three limits. Multislice and the kinematic model are not one
sample of the transform and are refused. Crowding, ice, dose damage,
per-image atom rendering and an objective aperture act on the posed volume,
which this method never forms, so `Specimen.scatter` refuses them as well;
it serves the bare template that `Reconstructor` scatters. No gradient flows
to the rotation, so `Reconstructor` refuses `lr_R` under it, while shifts
remain differentiable. Content shifted across the box edge wraps
periodically, where `"real"` clips it.

### Point-group averaging

Point-group averaging uses neither rotation method by default. `apply_symmetry`,
which `Reconstructor` calls once per epoch to enforce a point group, takes
`method="exact"` (`symmetrize_exact`): it evaluates the volume's
discrete-space transform at rotated frequencies with a type-2 non-uniform FFT
(Barnett et al. 2019; Shih et al. 2021) and inverts the mean. No step
resamples the grid, so averaging a map that already has the symmetry returns
it unchanged. Both interpolating methods attenuate it instead, and the loss
compounds over the epochs of a reconstruction. Measured on a 560³ capsid at
1.47 Å per voxel, as the fraction of its amplitude a symmetric map keeps after
one I1 average:

| Resolution band | `"exact"` | `"real"` | `"fourier"` |
|---|---|---|---|
| 200–20 Å | 1.000 | 0.998 | 0.718 |
| 20–8 Å | 1.000 | 0.955 | 0.717 |
| 8–4.5 Å | 1.000 | 0.837 | 0.720 |
| 4.5–3.5 Å | 1.000 | 0.719 | 0.722 |
| 3.5–2.94 Å | 0.981 | 0.598 | 0.711 |

The remainder in the last `"exact"` row is aliasing in the test map itself.
The `"fourier"` loss is flat because the capsid fills most of the box, the
regime in which the table above shows that path to be least accurate. The
exact average holds for a map that is band-limited and lies inside the box's
inscribed sphere; density reaching the box faces wraps, as in any
Fourier-space rotation.

## Optical sign conventions

The four signs that define the forward model:

\[
t(\mathbf{r}) = \exp\!\big(+i\sigma\,\Delta z\, V(\mathbf{r})\big),
\qquad
F(k) = \exp\!\big(+i\pi\lambda\,\Delta z\, k^2\big)
\]

\[
T(k) = \exp\!\big(-i\chi(k)\big),
\qquad
\chi(k) = \pi\lambda k^2\Big(\tfrac{1}{2}C_s\lambda^2k^2 - \Delta f\Big)
\]

for the slice transmission function, the Fresnel propagator, the transfer
function, and the wavefront aberration phase respectively. \(\sigma\) is
the energy-dependent interaction parameter from
`scattering.interaction_parameter`, in 1/(V·Å).

**Positive defocus is underfocus**, the standard cryo-EM convention:
RELION defines `rlnDefocusU` as positive for underfocus, and CTFFIND
(Rohou & Grigorieff 2015), Gctf (Zhang 2016), and CryoSPARC report the
same sense. In `specter` this produces the
expected phase contrast. At 300 kV with \(\Delta f = +1\) µm, a weak
positive potential appears dark, its first CTF zero falling at
\(0.070\) Å⁻¹ against the \(\sqrt{1/\lambda\Delta f} = 0.071\) Å⁻¹
predicted by the formula above. At \(\Delta f = -1\) µm the same feature
appears bright.

The **overall sign of \(\chi\) is not universal**, and this is the usual
source of confusion when porting an equation. `specter` follows
the form of Kirkland (2010), with \(\chi\) as written above and a transfer function
\(\exp(-i\chi)\). The 3DEM convention of Heymann et al. (2005), which
Gctf follows, uses \(\gamma = -\chi\). The two describe identical physics with
mirrored bookkeeping, so check an imported expression as a whole rather
than term by term.

Two further conventions belong to the propagation stage and appear where
they act: the Ewald sphere curvature sign (`ews_curvature_sign`) and
amplitude contrast (`alpha`), both under
[Scattering](scattering/index.md#two-conventions-shared-by-every-model).

## The defocus reference plane

A thick specimen has no single defocus: each plane at height \(z\) is imaged
at its own, and the defocus increases by one ångström for every ångström
towards the exit face, since an underfocused lens images a plane upstream of
the specimen. The defocus a CTF fit reports is that of the
specimen's centre of scattering mass, which Balakrishnan et al. (2023)
established and used to recover depth from a single shot. For a particle centred in its
box, or a uniform slab of ice, that is the box's midplane, and it is the value
a `.cs` or `.star` file records.

A propagating scattering model returns the wave where it leaves the volume,
at the exit face, half the box depth \(n_z\Delta x/2\) below the midplane,
where the defocus is larger by that much. `specter` therefore adds
\(n_z\Delta x/2\) to `dfu` and `dfv` before building a transfer function,
automatically (every `ExitWave` carries the shift as a `defocus_shift` of
\(-n_z\Delta x/2\), which `ExitWave.apply_ctf` subtracts), for every
propagating scattering model: `multislice`, `rytov`, `firstborn`, and
`kinematic`. The nominal defocus is then the midplane's, and is written to
a STAR or `.cs` file unchanged. The `projection` and `ctf` models have no
\(Z\) extent to offset from, so `specter` excludes them. If you drive
`Scattering` and `Aberration` directly, apply the shift yourself via
`aberrations.defocus_midplane_shift`.

A scatterer at the midplane images exactly as the same scatterer with no
depth at the nominal defocus, while one at either face images as it would
\(n_z\Delta x/2\) off it;
`tests/test_aberrations_functions.py::test_nominal_defocus_is_the_midplane_defocus`
pins this. A tilt measures the shift over its own tilted column
(`imaging._tilt.tilt_defocus_shift`), and an ice profile adds the height of the
ice's thickness-weighted centroid above the midplane
(`IceProfile.centroid`), which is zero for any film whose mid-plane is level.

The correction uses the volume's geometric midplane, which equals the
centre of scattering mass only when scattering density distributes
uniformly along \(z\). This holds for the usual arrangement of a particle
centred in an ice-filled box. It does not hold for an axially asymmetric
specimen, such as a protein positioned near one face of a thick slab,
where the true reference plane sits at the scattering centroid instead.

## Potentials and intensities

Potential volumes are in **volts**. A box of water at the density of
amorphous ice, rendered with the default `bulk_scattering_factors =
"kirkland"`, returns a mean potential of 4.55-4.58 V at 0.5-1.5 Å voxels.
That is within 1% of the 4.59 V an isolated-atom superposition of one
oxygen and two hydrogens per molecule predicts at the same density,
0.94 g/cm³. The
measured mean inner potential of liquid water, 4.48 ± 0.19 V (Yesibolati
et al. 2020), scales to about 4.21 V at the density of amorphous ice. An
isolated-atom superposition cannot represent bonding and is expected to
sit above such a measurement. The scattering factors returned
by the `*_fourier` functions in `atom/_atomic_potentials.py` are in Å;
multiplying by \(c_1 = 2\pi a_0 e = 47.9\) V·Å² converts them to the
V·Å³ Fourier-space potential (Kirkland 2010); `PotentialBuilder` applies this factor.
Projected (2D) potentials are in V·Å.

Detector output is in **expected electron counts per pixel**,

\[
I = D \cdot \Delta x^2 \cdot \mathrm{DQE}(0) \cdot |\psi|^2
\]

with \(D\) the dose in e⁻/Å². `specter` applies shot noise as a Poisson
draw on these counts. It normalises the detector MTF as
\(\sqrt{\mathrm{DQE}(k)/\mathrm{DQE}(0)}\) so it acts as a pure blur, and
applies \(\mathrm{DQE}(0)\) separately as the counting efficiency above;
see [Detector](detector.md).

## Image polarity

The raw simulated intensity has **protein dark** at underfocus, as in an
experimental micrograph.

`specter` inverts saved particle stacks. With `normalize_particles`
enabled, the default for `specter simulate particles`, `specter`
standardises each particle to zero mean and unit variance and then
negates it, so protein appears bright in the output `.mrcs`. This matches
the contrast-inverted particle stacks that RELION extraction and
CryoSPARC produce.

Reading works the other way round. A CryoSPARC stack records its polarity
in `blob/sign`: +1 for the recorded contrast, -1 for the inverted contrast
of a normalised restack. `specter match` and `Ghostbuster` bring every
image they read to the recorded polarity by multiplying it by its sign,
taken as -1 when the `.cs` file does not record one.

`specter` does not invert micrographs and tilt series.
`normalize_micrographs` and `normalize_tilt_series` both default to
disabled, and when enabled they standardise only. After any of these
normalisations the values are no longer electron counts.

## Files and output

`specter` writes results into per-artifact directories at the top of the
working directory (`particles`, `micrographs`, `tiltseries`, `tomograms`,
`ice`, `reconstructions`), resolved relative to the current working
directory. A run that opts into job tracking instead writes into a
numbered job directory, `[<project>/]<artifact>/J00N/`. The one
`output_dir` field controls both. See [Configure a run](../user-guide/configuration.md).

A `.specter` file marks the root of a project. `specter` creates it the
first time a tracked run needs one, and locates it by walking up from the
current directory the way `git` locates the nearest `.git`. Running from
a subdirectory continues that project's job numbering rather than
starting a second, disconnected sequence.

`specter` caches structures you fetch by accession code outside the
project, at `$SPECTER_PDB_CACHE`, else `$XDG_CACHE_HOME/specter/pdb`,
else `~/.cache/specter/pdb`, following the convention `torch.hub` and
HuggingFace use for the same purpose. Every project therefore shares one
download. `specter` reads a structure you supply by path where it lies
and never copies it into the cache, which is what makes `specter cache
clean` safe: the cache holds only files you can fetch again.

`specter` writes particle stacks, micrographs, and tilt series as float32
`.mrcs` paired with a `.star` file, and records the pixel size in both: in
the STAR file as `rlnImagePixelSize` and in the MRC header as `voxel_size`.
Tomogram volumes, which `specter build tomogram` writes, have no
accompanying STAR file and carry `voxel_size` in the MRC header alone.

## Reproducibility

`specter.seed(n)` seeds Python's `random`, NumPy, and PyTorch on CPU and
on all CUDA devices, and so controls every random draw `specter` makes.
It does not make a run bit-reproducible. `torch.use_deterministic_algorithms`
is not enabled, since it costs performance, and two sources of
nondeterminism remain.

- **The forward model** varies at about \(2 \times 10^{-5}\) relative from
  run to run, through the order of GPU reductions and FFTs. Poisson
  sampling then discretises that variation: most pixels round to the same
  integer count, and the few that cross a rounding boundary differ by a
  whole draw. Two identical seeded runs of `specter simulate tiltseries`
  differ on about 0.1% of pixels, by up to 74 counts. With
  `noise_model = "none"` the same runs differ on almost every pixel, but
  only at about \(2 \times 10^{-5}\) relative.
- **`specter build tomogram`** renders its species in parallel and
  accumulates them in a nondeterministic order. Two seeded runs differ on
  about 29% of voxels, by about \(6 \times 10^{-6}\) V on a volume
  peaking near 24 V.

What a seed does guarantee is the specimen: which species are placed, how
many, and where are identical from run to run, and the labels, picks and
region volumes `specter build tomogram` writes are byte-identical. When
comparing two versions of the code at a fixed seed, run the same version
twice as a control, since the differences above appear in that comparison
as well.

## References

- Balakrishnan, D., Chee, S. W., Baraissov, Z., Bosman, M., Mirsaidov, U.,
  & Loh, N. D. (2023). Single-shot, coherent, pop-out 3D metrology.
  *Communications Physics*, 6(1), 321.
  [doi:10.1038/s42005-023-01431-6](https://doi.org/10.1038/s42005-023-01431-6)
- Barnett, A. H., Magland, J., & af Klinteberg, L. (2019). A parallel
  nonuniform fast Fourier transform library based on an "exponential of
  semicircle" kernel. *SIAM Journal on Scientific Computing*, 41(5),
  C479–C504.
  [doi:10.1137/18M120885X](https://doi.org/10.1137/18M120885X)
- Bregier, R. (2021). Deep regression on manifolds: A 3D rotation case
  study. In *2021 International Conference on 3D Vision (3DV)* (pp.
  166–174). IEEE.
  [doi:10.1109/3DV53792.2021.00027](https://doi.org/10.1109/3DV53792.2021.00027)
- Cheng, A., Henderson, R., Mastronarde, D., Ludtke, S. J., Schoenmakers,
  R. H. M., Short, J., Marabini, R., Dallakyan, S., Agard, D., & Winn, M.
  (2015). MRC2014: Extensions to the MRC format header for electron
  cryo-microscopy and tomography. *Journal of Structural Biology*, 192(2),
  146–150.
  [doi:10.1016/j.jsb.2015.04.002](https://doi.org/10.1016/j.jsb.2015.04.002)
- Heymann, J. B., Chagoyen, M., & Belnap, D. M. (2005). Common conventions
  for interchange and archiving of three-dimensional electron microscopy
  information in structural biology. *Journal of Structural Biology*,
  151(2), 196–207.
  [doi:10.1016/j.jsb.2005.06.001](https://doi.org/10.1016/j.jsb.2005.06.001)
- Kirkland, E. J. (2010). *Advanced computing in electron microscopy* (2nd
  ed.). Springer.
  [doi:10.1007/978-1-4419-6533-2](https://doi.org/10.1007/978-1-4419-6533-2)
- Punjani, A., Rubinstein, J. L., Fleet, D. J., & Brubaker, M. A. (2017).
  cryoSPARC: Algorithms for rapid unsupervised cryo-EM structure
  determination. *Nature Methods*, 14(3), 290–296.
  [doi:10.1038/nmeth.4169](https://doi.org/10.1038/nmeth.4169)
- Rohou, A., & Grigorieff, N. (2015). CTFFIND4: Fast and accurate defocus
  estimation from electron micrographs. *Journal of Structural Biology*,
  192(2), 216–221.
  [doi:10.1016/j.jsb.2015.08.008](https://doi.org/10.1016/j.jsb.2015.08.008)
- Scheres, S. H. W. (2012). RELION: Implementation of a Bayesian approach to
  cryo-EM structure determination. *Journal of Structural Biology*, 180(3),
  519–530.
  [doi:10.1016/j.jsb.2012.09.006](https://doi.org/10.1016/j.jsb.2012.09.006)
- Shih, Y.-h., Wright, G., Andén, J., Blaschke, J., & Barnett, A. H. (2021).
  cuFINUFFT: A load-balanced GPU library for general-purpose nonuniform FFTs.
  In *2021 IEEE International Parallel and Distributed Processing Symposium
  Workshops (IPDPSW)* (pp. 688–697). IEEE.
  [doi:10.1109/IPDPSW52791.2021.00105](https://doi.org/10.1109/IPDPSW52791.2021.00105)
- Yesibolati, M. N., Laganà, S., Sun, H., Beleggia, M., Kathmann, S. M.,
  Kasama, T., & Mølhave, K. (2020). Mean inner potential of liquid water.
  *Physical Review Letters*, 124(6), 065502.
  [doi:10.1103/PhysRevLett.124.065502](https://doi.org/10.1103/PhysRevLett.124.065502)
- Zhang, K. (2016). Gctf: Real-time CTF determination and correction.
  *Journal of Structural Biology*, 193(1), 1–12.
  [doi:10.1016/j.jsb.2015.11.003](https://doi.org/10.1016/j.jsb.2015.11.003)
