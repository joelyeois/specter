# Conventions

This page collects the sign, unit, axis, and pose conventions that
`specter` follows. It is written for two situations: importing
experimental metadata into a simulation, and comparing `specter`'s output
against another package. Both fail silently when a convention is
mismatched, since a wrong sign or a wrong axis produces a plausible image
rather than an error.

Definitions that belong to the file formats themselves, such as the
meaning of RELION's Euler angles or the layout of a STAR file, are not
repeated here; see
[RELION's conventions page](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html).

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
`tilty` are in radians. This matches CryoSPARC, whose `.cs` files store
`ctf/df_angle_rad` in radians and are converted to degrees on the way in
by `io/_cryosparc.py`.

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
\(x\) (columns). This is the same order MRC uses on disk, so volumes and
image stacks are written without transposition.

## Fourier conventions

**FFTs are unshifted by default.** `specter.fft`'s `fft2`/`ifft2`/`fft3`
and their inverses return native PyTorch ordering, with the zero
frequency at index 0, unless `shift=True` is passed. Any mask, envelope,
or propagator that a caller builds themselves must be constructed in the
same ordering. A disk centred at index \(N/2\), the `fftshift`
convention, multiplied into an unshifted spectrum removes the zero
frequency and retains the Nyquist corners.

**Frequency grids are in 1/Å.** They are built from
`torch.fft.fftfreq(n, pixel_size)`, so the Nyquist frequency is
\(1/(2\,\Delta x)\) and no cycles-per-pixel conversion is involved.

**Azimuth is measured from \(+y\).** On the 2D frequency grid, `KY` varies
along axis 0 and `KX` along axis 1, and the azimuthal angle is
\(\theta = \operatorname{atan2}(k_x, k_y)\). This is what allows
`dfang` from a `.cs` or `.star` file to be used directly, with no 90°
offset.

## Poses

Rotations are unit quaternions in `roma`'s scalar-last \((x, y, z, w)\)
ordering. Translations are in Å.

**A quaternion \(Q\) rotates the density by \(R(Q)^{-1}\).** A feature at
\(+x\) subjected to a \(+90^\circ\) rotation about \(z\) lands at \(-y\).
The volume path (`ImageGenerator.rotate`, which passes \(R\) to
`grid_sample`) and the coordinate path
(`ImageGeneratorFromCoordinates.rotate`, which applies \(R^{\mathsf{T}}\)
to atom positions) look like opposites in source but produce the same
pose, because sampling a volume at \(R\mathbf{x}\) rotates its density by
\(R^{-1}\). This is the interpretation under which CryoSPARC rotation
vectors and RELION Euler angles map in without inversion.

**Translations follow RELION's origin-offset semantics.** They are
subtracted, not added: a translation of \((+10, 0)\) Å moves a feature at
\(x = +10\) Å to the box centre. The shift acts in the image plane,
*after* the rotation, which is the same frame RELION and CryoSPARC define
their origin offsets in, where the shift is a phase ramp applied to an
already-extracted central slice rather than a displacement of the 3D
reference.

Expressing an image-plane shift as a single 3D affine is what requires
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
\([(n+1)/2, \ldots]\). The difference is half a voxel, which appears as a
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

A pose is applied to a volume by one of two interchangeable methods, selected
by `rotate_mode` on `ImageGenerator`, `Reconstructor`, `Ghostbuster` and
`ParticleStackConfig`. Both express the same convention, and both start from
the same affine built above; they differ only in where the interpolation
happens.

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

Rotation accuracy is the reverse of what the name suggests. The Fourier path
interpolates the *spectrum* trilinearly, and an object far from the box centre
has a rapidly oscillating phase that linear interpolation cannot follow, so its
error grows with distance from the centre while the real-space error stays
flat. Measured against an analytic ground truth, as a fraction of peak density:

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

One path selects `"fourier"` by default: `apply_symmetry`, which
`Reconstructor` calls to enforce a point group on the reconstructed volume.
Symmetry operations are pure rotations about a centred volume, which is the
regime where the Fourier path is most accurate and where its circular shift
never applies.

Rotation origin follows the same `"relion"` / `"center"` choice described
above. In Fourier space the rotation is necessarily centred on the spectrum's
DC term, which `fft3(..., shift=True)` places at `n // 2`; `"center"` is
reached by folding the half-voxel offset into the same phase ramp, and is
supported for cubic volumes only.

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
RELION defines `rlnDefocusU` as positive for underfocus, and CTFFIND,
Gctf, and CryoSPARC report the same sense. In `specter` this produces the
expected phase contrast. At 300 kV with \(\Delta f = +1\) µm, a weak
positive potential appears dark, its first CTF zero falling at
\(0.070\) Å⁻¹ against the \(\sqrt{1/\lambda\Delta f} = 0.071\) Å⁻¹
predicted by the formula above. At \(\Delta f = -1\) µm the same feature
appears bright.

The **overall sign of \(\chi\) is not universal**, and this is the usual
source of confusion when porting an equation. `specter` follows
Kirkland's form, with \(\chi\) as written above and a transfer function
\(\exp(-i\chi)\). The 3DEM convention of Heymann et al., which Gctf
follows, uses \(\gamma = -\chi\). The two describe identical physics with
mirrored bookkeeping, so an imported expression must be checked as a
whole rather than term by term.

Two further conventions belong to the propagation stage and are described
where they act: the Ewald sphere curvature sign (`ews_curvature_sign`)
and amplitude contrast (`alpha`), both under
[Scattering](scattering/index.md#two-conventions-shared-by-every-model).

## The defocus reference plane

Defocus values in `.cs` and `.star` files are measured from the
specimen's entry face. A propagated exit wave does not carry that
reference. For a specimen of thickness \(T\) the effective defocus is
centred on the specimen's centre of scattering mass at \(z = T/2\), a
result established by Balakrishnan et al. and exploited there to recover
depth from a single shot.

`specter` therefore subtracts \(n_z\Delta x/2\) from `dfu` and `dfv`
before building a transfer function, automatically inside `BaseImager`,
for every propagating scattering model: `multislice`, `rytov`,
`firstborn`, and `kinematic`. The `projection` and `ctf` models have no
\(Z\) extent to offset from and are excluded. Code driving `Scattering`
and `Aberration` directly must apply the correction itself via
`aberrations.defocus_midplane_shift`, and add it back before exporting
defocus values to a STAR or `.cs` file.

The correction uses the volume's geometric midplane, which equals the
centre of scattering mass only when scattering density is distributed
uniformly along \(z\). This holds for the usual arrangement of a particle
centred in an ice-filled box. It does not hold for an axially asymmetric
specimen, such as a protein positioned near one face of a thick slab,
where the true reference plane sits at the scattering centroid instead.

## Potentials and intensities

Potential volumes are in **volts**. A box of water at the density of
amorphous ice returns a mean potential of 4.8 V against a literature mean
inner potential of approximately 4.5 V. The scattering factors returned
by the `*_fourier` functions in `atom/atomic_potentials.py` are in Å;
multiplying by \(c_1 = 2\pi a_0 e = 47.9\) V·Å² converts them to the
V·Å³ Fourier-space potential, and this factor is applied by
`PotentialBuilder`. Projected (2D) potentials are in V·Å.

Detector output is in **expected electron counts per pixel**,

\[
I = D \cdot \Delta x^2 \cdot \mathrm{DQE}(0) \cdot |\psi|^2
\]

with \(D\) the dose in e⁻/Å². Shot noise is applied as a Poisson draw on
these counts. The detector MTF is normalised as
\(\sqrt{\mathrm{DQE}(k)/\mathrm{DQE}(0)}\) so that it acts as a pure
blur, and \(\mathrm{DQE}(0)\) is applied separately as the counting
efficiency above; see [Detector](detector.md).

## Image polarity

The raw simulated intensity has **protein dark** at underfocus, as in an
experimental micrograph.

Saved particle stacks are inverted. With `normalize_particles` enabled,
which is the default for `specter simulate particles`, each particle is
standardised to zero mean and unit variance and then negated, so protein
is bright in the output `.mrcs`. This matches the contrast-inverted
particle stacks produced by RELION extraction and CryoSPARC.

Micrographs and tilt series are not inverted. `normalize_micrographs` and
`normalize_tilt_series` both default to disabled, and when enabled they
standardise only. After any of these normalisations the values are no
longer electron counts.

## Files and output

Everything `specter` writes lands under `./specter-data/`, resolved
relative to the current working directory, in per-artifact subdirectories
(`pdb`, `particles`, `micrographs`, `tiltseries`, `tomograms`, `ice`).
See [Configure a run](../user-guide/configuration.md).

Particle stacks, micrographs, and tilt series are written as float32
`.mrcs` paired with a `.star` file. **The pixel size is recorded in the
STAR file as `rlnImagePixelSize`, not in the MRC header**, so a consumer
reading the `.mrcs` alone will find no voxel size. Tomogram volumes,
written by `specter build tomogram`, do set `voxel_size` in the MRC
header, since they have no accompanying STAR file.

## Reproducibility

`specter.seed(n)` seeds Python's `random`, NumPy, and PyTorch on CPU and
on all CUDA devices. It does not set `torch.backends.cudnn.deterministic`,
so runs involving cuDNN-backed operations may still differ at the level
of floating-point reduction order.

## References

- Kirkland, E. J. (2010). *Advanced Computing in Electron Microscopy*,
  2nd ed. Springer.
- Balakrishnan, D., Chee, S. W., Baraissov, Z., Bosman, M., Mirsaidov, U.,
  & Loh, N. D. (2023). Single-shot, coherent, pop-out 3D metrology.
  *Communications Physics* **6**, 321.
  [doi:10.1038/s42005-023-01431-6](https://doi.org/10.1038/s42005-023-01431-6)
- Heymann, J. B., Chagoyen, M., & Belnap, D. M. (2005). Common conventions
  for interchange and archiving of three-dimensional electron microscopy
  information in structural biology. *Journal of Structural Biology*
  **151**(2), 196–207.
  [doi:10.1016/j.jsb.2005.06.001](https://doi.org/10.1016/j.jsb.2005.06.001)
- Rohou, A., & Grigorieff, N. (2015). CTFFIND4: Fast and accurate defocus
  estimation from electron micrographs. *Journal of Structural Biology*
  **192**(2), 216–221.
  [doi:10.1016/j.jsb.2015.08.008](https://doi.org/10.1016/j.jsb.2015.08.008)
