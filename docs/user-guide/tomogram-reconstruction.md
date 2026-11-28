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
from specter.tilt import read_aretomo3_aln, read_aretomo3_ctf, turn_to_tilt_axis

quaternions, translations = read_aretomo3_aln("TS_01.aln", pixel_size=1.54)
quaternions, turn = turn_to_tilt_axis(quaternions)   # tilt axis along volume x
ctf_params, defocus_hand = read_aretomo3_ctf("TS_01_CTF.txt")
ctf_params["cs"] = torch.full_like(ctf_params["dfu"], 2.7e7)   # Å

pipeline = TomogramGhostbuster(
    "TS_01_bin6.mrc",              # tilt images in electron counts, (N, H, W)
    voxel_size=9.24,               # Å, of the images as loaded
    voltage=300.0,
    ctf_params=ctf_params,         # one entry per tilt
    quaternions=quaternions,
    translations=translations,
    nz=200,                        # slab thickness in voxels
    propagation=Propagation(scattering_model="rytov", alpha=0.07),
    defocus_hand=defocus_hand,
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
`turn_to_tilt_axis` and `defocus_hand` are described under
[The volume](#the-volume) and [Defocus hand](#defocus-hand).

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
both were below 0.08 there.

On experimental data the models have been separated only through the
defocus gradient across a tilted field. On TS_69_2 of the phantom dataset of
Peck et al. (2025), scored by the half-dose cross-validation described
[below](#choosing-the-weight-and-validating-the-result), Rytov and the flat
CTF model were indistinguishable at 10 Å per voxel and, on a central 256 nm
sub-field, at 5 Å per voxel. Over the full 630 nm field at 5 Å, Rytov with
the correct defocus hand scored above the flat model at 20 to 10 Å away from
the tilt axis and equal to it near the axis
([Depth-dependent defocus over the full field](#depth-dependent-defocus-over-the-full-field)).
The effect of the specimen's own thickness, which the simulation places
below 10 Å, has not been resolved. At 2.5 Å per voxel, over five sub-fields
of 96 nm, Rytov scored 3.2% above the flat model at 10 to 7 Å, with a 95%
bootstrap interval over tilt images of −0.0% to +6.5%, and 3.3% above it at
7 to 5 Å, with an interval of −4.4% to +11.5%. An average of the 96
annotated ribosomes of another series of the dataset, formed directly from
the tilt images with the deposited poses, reached about 20 Å with either
model.

## Defocus hand

A model that follows the defocus through the specimen has to know which way
the defocus runs with depth. A point of the specimen at height `z` in a
tilt's beam frame (the third coordinate of `Rᵀ r` for the pose `R` and the
volume point `r`) is imaged at defocus `Δf + z` under
`Propagation(ews_curvature_sign="negative")` and at `Δf − z` under
`"positive"`. Across a tilted field `z` grows with the distance from the tilt
axis, so the same sign decides on which side of the axis the defocus
increases. The sign is a property of the dataset's geometry: SPECTER's
simulators use `"negative"`, and TS_69_2 requires `"positive"`.

`TomogramGhostbuster` takes the sign as `defocus_hand`, which overrides the
one in `propagation`. Without it, a depth-aware model (every model except
`"projection"` and `"ctf"`) is run with `propagation`'s sign and a warning.
For a series aligned by AreTomo3 (Zheng et al. 2022; Peck et al. 2026),
`read_aretomo3_ctf` returns the CTF parameters and the hand from the last
column of the CTF file, `dfHand`, for poses read by `read_aretomo3_aln`:
`dfHand = −1` is `"positive"` and `dfHand = +1` is `"negative"`. The first
mapping was measured on TS_69_2, whose file records −1; the second is its
mirror image and has not been measured on a dataset.

The wrong hand is worse than no depth correction: over the full field of
TS_69_2 at 5 Å per voxel it scored below the flat model in every band
([Depth-dependent defocus over the full field](#depth-dependent-defocus-over-the-full-field)).

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
along a volume axis. For a series whose axis is oblique in the image,
`specter.tilt.turn_to_tilt_axis` composes into the quaternions the in-plane
rotation `M` that brings the tilt axis onto a volume axis and returns its
angle. The reconstructed volume then holds the point `r` of the original
frame at `M r`.

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

The gain of the total-variation prior is confined to low resolution. On
TS_69_2 at 5 Å per voxel it predicted withheld tilts better than the
quadratic prior down to 50 Å, and it halved the half-dose cross-validated
correlation at 20 to 10 Å (0.016 and 0.010 at 20 to 14 Å and 14 to 10 Å,
against 0.030 and 0.020). The quadratic prior is the one to use where the
tomogram's high-resolution content matters.

## Choosing the weight and validating the result

A tomogram has no independent half-set of particles, so the tests below
replace the single-particle half-map comparison. All were run on an
experimental series, TS_69_2 of the phantom dataset of Peck et al. (2025):
31 tilts over ±45°, reconstructed at 10 Å per voxel with its deposited
alignment and defocus values. The same tests were applied to AreTomo3
(Zheng et al. 2022; Peck et al. 2026), run on the same binned images with
the same alignment and without its denoiser.

**Half-dose cross-validation.** Sum the odd and the even movie frames of
every tilt into two series with the same signal and independent noise. Fit
one, predict the other (`predict_tilt_series`), and score the prediction by
the Fourier ring correlation with the images it did not see. This selects
the prior's weight: on TS_69_2 the correlation at 400 to 150 Å peaked at
0.48 for `tv_weight` between 0.2 and 0.3, against 0.44 for the best
`smoothness`, and fell on either side. The chi-squared of the same
prediction rejects a weight that is too small but is nearly flat near the
optimum.

**Withheld tilts.** Fit all but the highest tilts and predict those. The
withheld images sample frequencies the fitted ones do not, so this tests
what the reconstruction holds in the missing wedge, which cross-validation
at the fitted tilts cannot see. On TS_69_2, fitting ±39° and predicting ±42°
and ±45°:

| Reconstruction | FRC 400–150 Å | FRC 150–80 Å | FRC 80–50 Å |
| --- | --- | --- | --- |
| AreTomo3, weighted backprojection | 0.36 | 0.08 | 0.04 |
| AreTomo3, SART | 0.36 | 0.12 | 0.07 |
| `smoothness` | 0.39 | 0.12 | 0.09 |
| AreTomo3 weighted backprojection, total-variation denoised afterwards | 0.48 | 0.22 | 0.12 |
| `tv_weight` | 0.54 | 0.24 | 0.15 |

The fourth row is the AreTomo3 volume reweighted radially to the amplitude
spectrum of the `smoothness` reconstruction and then denoised with the same
total-variation penalty, at the best of four weights.

**Known particles.** Where a tomogram holds annotated particles, their mean
subvolume, taken without rotating them, is elongated along the beam by the
missing wedge, and its peak height over the background measures how well
the particles stand out. For the 33 annotated ribosomes of TS_69_2:

| Reconstruction | Peak z-score | Height along the beam |
| --- | --- | --- |
| AreTomo3 weighted backprojection, reweighted | 6.1 | 300 Å |
| AreTomo3, SART | 6.2 | 300 Å |
| `smoothness` | 5.2 | not measured |
| AreTomo3 weighted backprojection, total-variation denoised afterwards | 6.4 | 300 Å |
| `tv_weight` | 10.6 | 260 Å |

The ratio of the mean subvolume's width along the beam to its width across
it was 1.51 in the deposited weighted backprojection, 1.43 with `smoothness`
and 1.27 with `tv_weight`.

These tests place the two priors differently against the conventional
reconstructions. The `smoothness` reconstruction is equivalent to a weighted
backprojection with a different radial amplitude filter. Its amplitude
spectrum is 8.4, 2.6, 1.4, 1.0 and 0.7 times that of the AreTomo3 weighted
backprojection at 400 to 150 Å, 150 to 80 Å, 80 to 50 Å, 50 to 30 Å and 30 to
20 Å, because the fit undoes the suppression of low frequencies by the ramp
filter and the transfer function, and a backprojection reweighted to that
spectrum looks the same and separates the particles at least as well. On
withheld
tilts it is level with AreTomo3's SART. Its higher visual contrast is
therefore a filter and not a gain in information. The `tv_weight`
reconstruction differs in substance: total-variation denoising applied to a
finished backprojection recovers most of its gain on withheld tilts, but
neither its particle contrast nor its reduced elongation.

The Fourier shell correlation between reconstructions of the two half-dose
series is also available, and on TS_69_2 it was 0.41 against 0.33 at 400 to
150 Å for the two priors. It measures reproducibility, and a prior raises it
by suppressing noise whether or not what remains is right, so it does not
rank priors on its own. See Limitations for what cross-validation at the
fitted tilts does not test.

### Depth-dependent defocus over the full field

A model sampled on the surface gives every voxel the defocus of its own
position, across a tilted field as well as through the depth, without
dividing the images into patches. Whether the data show this was tested on
the full 630 nm field of TS_69_2 at 5 Å per voxel: a covering volume of
368 × 2360 × 1416 voxels solved in seven strips
([Solving in strips](#solving-in-strips)) with `smoothness=3` for 20
iterations. Each half-dose series was fitted and scored against the other,
under Rytov with the correct defocus hand, Rytov with the wrong hand, and
the flat CTF model with one defocus per image. The table gives the
cross-validated Fourier ring correlation over the whole field and over the
third of the field farthest from the tilt axis (210 to 315 nm from it).

| Field | Model | 150–80 Å | 80–50 Å | 50–30 Å | 30–20 Å | 20–14 Å | 14–10 Å |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Whole | Rytov, correct hand | 0.1860 | 0.1674 | 0.1515 | 0.0692 | 0.0287 | 0.0208 |
| Whole | Flat CTF | 0.1903 | 0.1686 | 0.1517 | 0.0694 | 0.0285 | 0.0205 |
| Whole | Rytov, wrong hand | 0.1846 | 0.1667 | 0.1512 | 0.0691 | 0.0276 | 0.0193 |
| Farthest third | Rytov, correct hand | 0.1776 | 0.1623 | 0.1412 | 0.0644 | 0.0283 | 0.0201 |
| Farthest third | Flat CTF | 0.1835 | 0.1626 | 0.1410 | 0.0639 | 0.0268 | 0.0164 |
| Farthest third | Rytov, wrong hand | 0.1732 | 0.1590 | 0.1399 | 0.0624 | 0.0230 | 0.0140 |

The differences from the flat model, as a percentage of its correlation,
with 95% intervals from a bootstrap over the 31 tilts:

| Field | Model | 150–80 Å | 30–20 Å | 20–14 Å | 14–10 Å |
| --- | --- | --- | --- | --- | --- |
| Whole | Rytov, correct hand | −2.2 (−2.7, −1.8) | −0.3 (−0.4, −0.1) | +0.9 (+0.3, +1.5) | +1.6 (+0.5, +2.8) |
| Whole | Rytov, wrong hand | −3.0 (−3.3, −2.6) | −0.4 (−0.6, −0.3) | −2.9 (−3.9, −2.0) | −5.8 (−7.7, −4.0) |
| Farthest third | Rytov, correct hand | −3.2 (−4.6, −2.0) | +0.8 (+0.0, +1.6) | +5.9 (+2.4, +9.9) | +22.6 (+12.8, +34.7) |
| Farthest third | Rytov, wrong hand | −5.6 (−7.6, −3.8) | −2.4 (−3.7, −1.4) | −14.1 (−19.9, −9.0) | −14.7 (−22.2, −7.5) |

With the correct hand the position-dependent defocus is measurable at 5 Å
per voxel. It raises the correlation at 14 to 10 Å by 23% in the third of
the field farthest from the tilt axis and by 10% in the middle third, and
changes nothing in the third nearest the axis (−0.4%, with an interval of
−1.4% to +0.5%). Over the whole field the gain is 1.6%. With the wrong hand
Rytov is below the flat model in every band. At 150 to 80 Å Rytov is 2% to
3% below the flat model over the whole field with either hand, for a reason
that has not been identified.

AreTomo3's weighted backprojection of the same half-dose series, with its
tilt-strip CTF correction, was scored by the same judge: registered to the
SPECTER volume, re-projected through the flat forward model, and compared
with the other half's images. It reached 0.165, 0.140 and 0.128 at 150 to
80 Å, 80 to 50 Å and 50 to 30 Å over the whole field, against 0.190, 0.169
and 0.152 for the flat SPECTER fit. Below 30 Å the comparison is not valid.
The re-projected AreTomo3 volume correlates with the very images it was
built from at only 0.53, 0.11 and 0.02 at 30 to 20 Å, 20 to 14 Å and 14 to
10 Å, where the SPECTER prediction correlates with its own input at 0.72 to
0.75. The two volumes are therefore not on a common footing there, and the
position-dependent CTF handling of the two programs cannot be ranked by
this test.

## Compute

The `"cg"` solver fits every tilt at once by preconditioned nonlinear
conjugate gradients (`solve_tilt_series`), on one device. A gradient over
the 31 tilts of TS_69_2 at 10 Å, a covering volume of 184 × 1200 × 736
voxels, takes about one second on an NVIDIA L40. With `smoothness` alone
the solve is preconditioned by the curvature of the data term and of the
prior per Fourier component, and 20 iterations take 50 s. After the first
iteration the objective is within 0.3% of its value after 20, and the solve
ends early once an iteration no longer lowers the objective by more than
its rounding error. With `tv_weight` it is not preconditioned, and
200 iterations take seven minutes at a peak of 19 GB of device memory.

`solver="adam"` takes `batchsize` tilts per step for `epochs`, as the
single-particle pipeline does, and is the only route for multislice.

### Solving in strips

One device holds the whole volume and two non-uniform FFT plans, which
bounds a single solve at about 250 million voxels on a 46 GB card. A
4096-pixel field at 5 Å per voxel is five times that. `strips=n`, or
`max_voxels=...`, divides the volume into overlapping strips along the tilt
axis and solves each as a problem of its own
(`solve_tilt_series_in_strips`); the volume itself stays on the CPU, and
with several devices (`run(device=[0, 1, 2, 3])`) the strips are solved one
per device at a time.

The division relies on the tilt axis lying along volume x or y
(`turn_to_tilt_axis`). A ray then keeps its coordinate along the axis at
every tilt, so a strip of the volume projects onto the same band of every
image, and neighbouring strips are coupled only through the distance the
transfer function displaces the image and through the prior. Each strip
takes the constant image shift that its centre implies and compares only the
pixels whose rays stay inside it, which form a slanted band when the tilt
axis is not along an image axis. The images are not resampled. Neighbours
overlap by twice the sum of three margins: a taper of 8 voxels at the
strip's face, twice the reach of the transfer function at the Nyquist
frequency (`λ k Δf`, 3 voxels at 10 Å and 10 voxels at 5 Å for 2 µm
defocus), and half of a cross-fade of at least 16 voxels. Strips are
cross-faded in the middle of the overlap, after each has been shifted by the
constant that makes its mean over the cross-fade equal its neighbour's,
since the data do not determine a strip's mean potential.
`predict_tilt_series` predicts images from a volume in the same strips, for
cross-validation.

Agreement with a single solve was measured on TS_69_2 at 10 Å, a volume
small enough to solve whole, in three strips. With `tv_weight=0.25` (200
iterations), where two runs of the single solve agree to 0.1%, the stitched
volume differed from the single solve by 1% to 3% of its root-mean-square
value in the strips' interiors and by 1% to 7% inside the cross-fades,
depending on the resolution band; the images predicted from the two volumes
differed by at most 0.5% of their contrast away from the seams and by at
most 1.7% on them. With `smoothness=3` (20 iterations), where two runs
agree to 0.3% of the volume and 0.03% of the predicted contrast, the
stitched volume differed from the single solve by 5% to 10% in the
interiors and by 2% to 11% inside the cross-fades, and the predicted images
by 1.5% to 2.3% away from the seams and by 0.6% to 3.3% on them, the largest
value at 400 to 150 Å. The cross-validated correlations of the stitched and
the single solve agreed to four decimal places in every band. A stronger
prior couples the strips over a longer distance: with `smoothness=30` the
predicted images differed by 22% on the seams at 400 to 150 Å and by at most
4% elsewhere.

Three strips cost about three times one solve on one device, since each
strip is predicted on a grid that holds the whole image: 125 s against 45 s
for 20 `smoothness` iterations at 10 Å, and 45 s on three devices. The full
field of TS_69_2 at 5 Å, a covering volume of 368 × 2360 × 1416 voxels
(1.23 × 10⁹), is solved in seven strips of 270 voxels. Twenty `smoothness`
iterations take 174 s per strip on an L40, 22 minutes for the volume on one
device, with 43 to 45 GB of the device in use.

## Limitations

The poses, shifts and defocus are held fixed; nothing refines them.

The defocus hand is not determined by the reconstruction and has to be
supplied per dataset ([Defocus hand](#defocus-hand)).

A strip-wise solve requires the tilt axis along a volume axis and a model
sampled on the surface, and supports neither a `kmask` nor a lateral taper.
It does not minimise the objective of the whole volume: each strip's prior
ends at its faces, and each strip settles the weakly constrained low
frequencies on its own.

Two runs of the same solve agree to rounding and not bit for bit, because
the forward model's rounding differs from run to run. Two identical
20-iteration `smoothness` solves of TS_69_2 at 10 Å differed by 0.3% of the
volume's root-mean-square value and by 0.03% of the contrast of the images
they predict.

Half-dose cross-validation at the fitted tilts does not test the
three-dimensional consistency of fine detail. Beyond the Crowther limit
(Crowther et al. 1970), about 60 Å for TS_69_2, neighbouring tilts no longer
overlap in Fourier space, so each image is fitted almost independently and
the score measures per-image denoising. A fit of TS_69_2 at 2.5 Å per voxel
with the sign of every shift reversed, which misaligns the tilts, scored
0.0151 against 0.0164 at 10 to 7 Å. Claims about detail finer than the
Crowther limit need withheld tilts or subtomogram averaging.

A volume wider than its images requires a model sampled on the surface.
Multislice reconstructs a volume as wide as the images, with the masked
loss.

The total-variation prior assumes compact objects in a flat solvent. It was
validated at 10 Å and 5 Å per voxel on a dilute specimen, where it also
leaves isolated single-voxel speckles in the solvent. Its behaviour on a
crowded cellular specimen is not established. Both priors leave the lowest
spatial frequencies, where the transfer function is weak, poorly
constrained.

The comparisons with AreTomo3 were made on one series, at 10 Å and 5 Å per
voxel. AreTomo3's own half-set correlation could not serve as a judge of
its CTF-corrected reconstruction below 20 Å: its two half-dose volumes
correlated more strongly at 14 to 10 Å (0.13) than at 30 to 20 Å (0.10),
which a specimen signal does not do.

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
- Crowther, R. A., DeRosier, D. J., & Klug, A. (1970). The reconstruction
  of a three-dimensional structure from projections and its application to
  electron microscopy. *Proceedings of the Royal Society of London. A.
  Mathematical and Physical Sciences*, 317(1530), 319–340.
  [doi:10.1098/rspa.1970.0119](https://doi.org/10.1098/rspa.1970.0119)
- Peck, A., Yu, Y., Paraan, M., Kimanius, D., Ermel, U. H., Hutchings, J.,
  et al. (2026). AreTomoLive: automated reconstruction of comprehensively
  corrected and denoised cryo-electron tomograms in real time and at high
  throughput. *Nature Methods*, 23(6), 1121–1125.
  [doi:10.1038/s41592-026-03093-y](https://doi.org/10.1038/s41592-026-03093-y)
  (Describes AreTomo3.)
- Peck, A., Yu, Y., Schwartz, J., Cheng, A., Ermel, U. H., Hutchings, J.,
  et al. (2025). A realistic phantom dataset for benchmarking cryo-ET data
  annotation. *Nature Methods*, 22(9), 1819–1823.
  [doi:10.1038/s41592-025-02800-5](https://doi.org/10.1038/s41592-025-02800-5)
- Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation
  based noise removal algorithms. *Physica D: Nonlinear Phenomena*, 60(1–4),
  259–268.
  [doi:10.1016/0167-2789(92)90242-F](https://doi.org/10.1016/0167-2789(92)90242-F)
- Zheng, S., Wolff, G., Greenan, G., Chen, Z., Faas, F. G. A., Bárcena, M.,
  Koster, A. J., Cheng, Y., & Agard, D. A. (2022). AreTomo: An integrated
  software package for automated marker-free, motion-corrected
  cryo-electron tomographic alignment and reconstruction. *Journal of
  Structural Biology: X*, 6, 100068.
  [doi:10.1016/j.yjsbx.2022.100068](https://doi.org/10.1016/j.yjsbx.2022.100068)
