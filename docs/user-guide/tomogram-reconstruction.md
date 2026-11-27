# Reconstruct a tomogram

`TomogramGhostbuster` reconstructs a 3D scattering potential from a cryo-ET
tilt series with the forward model `specter simulate tiltseries` uses. It
takes the tilt images, their poses and their defocus as given, and solves for
the volume. It is available from the Python API; there is no
`specter reconstruct tomogram` command yet.

## Basic usage

```python
import torch
from specter.ghostbuster import TomogramGhostbuster
from specter.settings import Propagation
from specter.tilt import read_aretomo3_aln

quaternions, translations = read_aretomo3_aln("TS_01.aln", pixel_size=1.54)

pipeline = TomogramGhostbuster(
    "TS_01_bin6.mrc",              # tilt images in electron counts, (N, H, W)
    voxel_size=9.24,               # Å, of the images as loaded
    voltage=300.0,
    ctf_params={"dfu": dfu, "dfv": dfv, "dfang": dfang, "cs": cs},  # one entry per tilt
    quaternions=quaternions,
    translations=translations,
    nz=200,                        # slab thickness in voxels
    propagation=Propagation(scattering_model="rytov", alpha=0.07),
    tv_weight=0.25,
    iterations=200,
    run_dir="tomogram_run",
)
model = pipeline.run(device=0)
volume = model.field_volume()      # (nz, H, W), also written to volume.mrc
```

`quaternions` and `translations` are the poses `specter.imaging.Poses`
takes; `angles` with `TiltGeometry(tilt_axis=...)` replaces them for a series
whose tilt axis lies along an image axis. `ctf_params` follows the
single-particle convention: lengths in Å, `dfang` in degrees.

## Forward model

The default model is Rytov, sampled on each tilt's Ewald surface. A
single-scatter exit wave is the unrotated volume's Fourier transform sampled
on a paraboloid, so a pose turns the paraboloid rather than the volume and
the whole tilt series is one non-uniform FFT
(`specter.scattering.surface_slice_sum`). Nothing is interpolated, an
arbitrary tilt-axis direction and per-tilt shifts cost nothing, and the
defocus of every voxel follows from its position along the beam, which
covers both the defocus gradient across a tilted field and the one through
the specimen's depth.

The choice of model was measured on a simulated specimen 180 nm thick,
sampled at 2.5 Å and imaged at 300 kV and 3 µm defocus, by predicting its
tilt images under each model and comparing them with multislice images of the
same volume.

| Model | Error in the predicted images, as a fraction of their power |
| --- | --- |
| Rytov | at most 0.8% in every band from 200 Å to 5 Å, untilted; multislice itself differs from it by at most 0.2% |
| Projection, CTF (flat Ewald sphere) | below 1.5% above 20 Å; 7% (untilted) to 13% (45°) at 14 to 10 Å; 26% to 50% at 10 to 7 Å |
| First Born | below 2% from 40 Å to 5 Å, untilted; 24% to 35% at 200 to 80 Å |

Multislice therefore adds nothing that this specimen's images can show, and
it is the one model the surface cannot represent: `scattering_model=
"multislice"` walks the tilted volume slice by slice, at a cost per tilt
comparable to the whole series on the surface. The flat models fail where the
defocus spread across the specimen becomes a phase error, below about 15 Å
for this thickness. The first Born model fails at low resolution, where the
term of the intensity that is quadratic in the phase is not small for a thick
slab.

In a reconstruction of the same specimen from noiseless images, the Rytov and
flat models agreed with the true volume equally above 20 Å, and Rytov reached
a Fourier shell correlation of 0.29 against 0.22 at 10 to 7 Å and 0.21 against
0.11 at 7 to 5 Å. With the shot noise of one tilt series (3.9 e⁻/Å² per tilt)
both were below 0.08 there. A single tomogram is noise-limited well before
the scattering model matters; the difference is expected to count where many
copies are averaged.

## The volume

A tilted beam passes through specimen beside the field of view. By default
(`extend=True`) the reconstructed volume is wide enough to hold every ray
that reaches the detector (`TomogramReconstructor.covering_shape`), and
`volume.mrc` holds its central part, under the images. The alternative,
`extend=False`, reconstructs a volume as wide as the images and masks the
loss to the strip of each tilt that the volume fully explains, which
discards up to half of the highest tilts. On a simulated 100 nm specimen the
covering volume raised the Fourier shell correlation with the true volume at
400 to 100 Å from 0.53 to 0.79.

The covering volume grows across the tilt axis only when that axis lies
along a volume axis. For a series whose axis is oblique in the image, turn
the volume frame to the axis first by composing an in-plane rotation into the
quaternions.

The reconstructed potential is relative to the specimen's mean. A uniform
slab shifts the phase of the whole wave and leaves no contrast, so the mean
is not determined. With `dose_per_angstrom=None` (the default) each tilt's
dose is read off its own mean counts, which needs no dose calibration and
makes the empty space around the volume stand for more of the same specimen
rather than for vacuum. Experimental images whose illumination is uneven
should be flat-fielded first, each divided by a heavily smoothed copy of
itself and multiplied by its mean.

## Priors

A tilt series holds fewer pixels than its tomogram holds voxels (about one
in fourteen for the covering volume above), so the data do not determine the
volume. Without a prior the fit continues until it has absorbed the shot
noise. Two priors are available, both on the volume's gradient
(`TomogramReconstructor.regularisation`):

- `smoothness` weights a quadratic penalty. It damps what the data do not
  determine and prefers no structure. On the simulated specimen it performed
  as a Wiener prior built from the true power spectrum did.
- `tv_weight` weights a total variation (Rudin et al. 1992) in the smooth
  form of Charbonnier et al. (1997): quadratic for gradients below
  `tv_delta` volts per voxel and linear above. It prefers compact objects in
  a flat solvent, which lets it restore part of the missing wedge.

On the simulated 100 nm specimen (5 Å voxels, 3.9 e⁻/Å² per tilt, ±45° in 3°
steps), scored against the true volume:

| Reconstruction | FSC 400–100 Å | FSC 100–40 Å | FSC in the missing wedge, 400–100 Å | Real-space correlation below 40 Å |
| --- | --- | --- | --- | --- |
| Weighted backprojection | 0.79 | 0.64 | 0.06 | 0.35 |
| Conjugate gradients, stopped early | 0.80 | 0.64 | 0.03 | 0.50 |
| `smoothness` | 0.79 | 0.65 | 0.04 | 0.58 |
| `tv_weight` | 0.89 | 0.74 | 0.69 | 0.74 |

The first two columns are measured over the frequencies the tilt range
samples and the third over those it does not.

## Choosing the weight and validating the result

A tomogram has no independent half-set of particles, so the tests below
replace the single-particle half-map comparison. All three were run on an
experimental series, TS_69_2 of the phantom dataset of Peck et al. (2025):
31 tilts over ±45°, reconstructed at 10 Å per voxel with its deposited
alignment and defocus values.

**Half-dose cross-validation.** Sum the odd and the even movie frames of
every tilt into two series with the same signal and independent noise. Fit
one, predict the other, and score the prediction by the Fourier ring
correlation with the images it did not see. This selects the prior's weight:
on TS_69_2 the correlation at 400 to 150 Å peaked at 0.48 for `tv_weight`
between 0.2 and 0.3, against 0.44 for the best `smoothness`, and fell on
either side. The chi-squared of the same prediction rejects a weight that is
too small but is nearly flat near the optimum. The flat-CTF and Rytov models
cross-validated identically on this series (0.48 for both), as the model
comparison above predicts at 10 Å per voxel.

**Withheld tilts.** Fit all but the highest tilts and predict those. The
withheld images sample frequencies the fitted ones do not, so this tests
what the reconstruction holds in the missing wedge, which cross-validation
at the fitted tilts cannot see. On TS_69_2, fitting ±39° and predicting ±42°
and ±45°:

| Reconstruction | FRC 400–150 Å | FRC 150–80 Å | FRC 80–50 Å |
| --- | --- | --- | --- |
| Conjugate gradients, stopped early | 0.31 | 0.12 | 0.07 |
| `smoothness` | 0.39 | 0.12 | 0.09 |
| `tv_weight` | 0.54 | 0.24 | 0.15 |

**Known particles.** Where a tomogram holds annotated particles, their mean
subvolume, taken without rotating them, is elongated along the beam by the
missing wedge. For the 33 annotated ribosomes of TS_69_2 the ratio of its
width along the beam to its width across it was 1.51 in the deposited
weighted backprojection, 1.43 with `smoothness` and 1.27 with `tv_weight`.

The Fourier shell correlation between reconstructions of the two half-dose
series is also available, and on TS_69_2 it was 0.41 against 0.33 at 400 to
150 Å for the two priors. It measures reproducibility, and a prior raises it
by suppressing noise whether or not what remains is right, so it does not
rank priors on its own.

## Compute

The `"cg"` solver fits every tilt at once by preconditioned nonlinear
conjugate gradients (`solve_tilt_series`), on one device. A gradient over
the 31 tilts of TS_69_2 at 10 Å, a covering volume of 184 × 1200 × 736
voxels, takes about one second on an NVIDIA L40. With `smoothness` alone
the solve is preconditioned and 25 iterations take 36 s. With `tv_weight` it
is not, and 200 iterations take five minutes at a peak of 19 GB of device
memory.

`solver="adam"` takes `batchsize` tilts per step for `epochs`, as the
single-particle pipeline does, and is the only route for multislice.

## Limitations

The poses, shifts and defocus are held fixed; nothing refines them.

The whole volume and two non-uniform FFT plans are held on one device, which
bounds the volume at about 400 million voxels on a 46 GB card. A full
4096-pixel field at 5 Å per voxel exceeds that. The tilts decouple between
strips along the tilt axis up to the reach of the transfer function, so a
strip-wise solve would lift the bound; it is not implemented.

A volume wider than its images requires a model sampled on the surface.
Multislice reconstructs a volume as wide as the images, with the masked
loss.

The total-variation prior assumes compact objects in a flat solvent. It was
validated at 10 Å and 5 Å per voxel on a dilute specimen, where it also
leaves isolated single-voxel speckles in the solvent. Its behaviour on a
crowded cellular specimen, and on fine detail that subtomogram averaging
would rely on, is not established. Both priors leave the lowest spatial
frequencies, where the transfer function is weak, poorly constrained.

The weight of a prior depends on the voxel size and the dose, so it has to
be chosen per dataset, by the cross-validation above. The values quoted here
are for the series described.

The surface samples frequencies inside the Nyquist sphere only, so the
corners of each image's spectrum, between the Nyquist frequency and √2 times
it, are predicted as zero.

## References

- Charbonnier, P., Blanc-Feraud, L., Aubert, G., & Barlaud, M. (1997).
  Deterministic edge-preserving regularization in computed imaging. *IEEE
  Transactions on Image Processing*, 6(2), 298–311.
  [doi:10.1109/83.551699](https://doi.org/10.1109/83.551699)
- Peck, A., Yu, Y., Schwartz, J., Cheng, A., Ermel, U. H., Hutchings, J.,
  et al. (2025). A realistic phantom dataset for benchmarking cryo-ET data
  annotation. *Nature Methods*, 22(9), 1819–1823.
  [doi:10.1038/s41592-025-02800-5](https://doi.org/10.1038/s41592-025-02800-5)
- Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation
  based noise removal algorithms. *Physica D: Nonlinear Phenomena*, 60(1–4),
  259–268.
  [doi:10.1016/0167-2789(92)90242-F](https://doi.org/10.1016/0167-2789(92)90242-F)
