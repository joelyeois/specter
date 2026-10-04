# Bound the shared dose-damage and solvent-exposure FFT working set

Baseline: `f7859aa`. The shared radial filter in `src/specter/fft.py` now
uses smaller batches of independent FFTs on large fields and reuses its
private interpolation scratch. This affects specimen dose damage and
solvent exposure in matching, particle simulation, micrographs and tilt
series. No utilities are changed.

## What changes

The filter already computes its 3D transform separably: 2D transforms
over complete xy planes, then 1D transforms over complete z-lines. The
large-field complex-element budget per batch falls from 2²⁶ to 2²².
A batch always holds at least one complete xy plane or one y-row of
z-lines, even if that exceeds the element budget. Every plane, line and
frequency is still processed; this is a smaller
batch of complete transforms, not spatial cropping or a local convolution.

The weighting pass also uses a 2²²-element slab budget. Frequency
coordinates become interpolation positions in the same storage, positions
become fractional offsets, and the lower table lookup becomes the weight.
The table still has 4096 entries; its frequency axis, interpolation,
operation order, physical envelope, dose weights and frame count are
unchanged. The existing complex64 spectrum and float32 production
potentials are unchanged. There is no reduced-resolution or reduced-frame
mode, and no host offload is introduced.

Fields whose complete half-spectrum fits the previous 2²⁶-element budget
retain their previous single FFT batches. An initial candidate changed the
noise in three unselected probes: changing a cuFFT batch can change its
last-bit rounding, and a changed electron count can change subsequent
random draws in the coincidence detector. Keeping the original small-field
plans restores those seeded outputs while retaining the measured gains on
large fields. The complete CLI comparisons below use this final policy.

A full half-spectrum still resides alongside the input volume. The
existing host fallback remains available when the entire working set does
not fit on the GPU; this change does not replace that half-spectrum with a
spatial approximation.

## Production validation

These are fresh baseline/candidate measurements on NVIDIA L40 GPUs,
PyTorch 2.5.1+cu121, four CPU threads per process. Matching has no shipped
canonical TOML, so it uses its built-in production defaults: all four ice
candidates (0/400/800/1200 Å), all three crowd candidates (0/1/1.3), 64
particles per pose check/probe, `probe_bin=2`, and two 200-particle batteries
at the native input box. Each command renders all ten stacks, derives the
config, computes all metrics and writes the report and figures.

The native 512-pixel case uses the existing real EMPIAR-11377 J446 export,
aligned poses, 8B0X assembly 1, 300 kV, Falcon 4i, 40 e⁻/Å², 40 frames and
the dataset's empirical exposure-filter weights. Mean free paths are
3950 Å in the solvent and 2460 Å in the specimen. Batch size remains 1;
the native padded potential is `[1, 1641, 1024, 1024]` at 0.731 Å/voxel.

The default-absorption control uses the same frozen 256-pixel Fourier crop
of the first 200 real experimental images as the earlier matching report,
supplied identically to both revisions. Its specimen mean free path is
unset, batch size stays 4 and the native potential is
`[4, 820, 512, 512]` at 1.462 Å/voxel. That crop is a separate input
control; the optimization does not bin the 512-pixel case.

`material.toml`, `uniform-256.toml` and `inputs.json` record the exact
configurations, file provenance and image-data hashes. Raw experimental
images, potential templates and simulated MRC stacks are not committed.

## Whole-command results

| Case | Whole CLI before → after | Speedup | Allocated GPU peak | Reserved GPU peak | Sampled driver peak |
|---|---|---|---|---|---|
| Native 512 px, material absorption | 1234.4 → 1051.0 s (20m34s → 17m31s) | 1.174× | 26.235 → 23.598 GiB | 39.012 → 37.207 GiB | 39.514 → 37.709 GiB |
| Native 256 px, default absorption | 241.6 → 220.4 s (4m02s → 3m40s) | 1.096× | 9.293 → 9.293 GiB | 12.535 → 12.535 GiB | 13.035 → 13.035 GiB |

The 512-pixel case is **14.86% shorter** and saves **2.637 GiB (10.05%)
allocated**, **1.805 GiB (4.63%) reserved** and **1.805 GiB (4.57%) sampled
driver usage**. The default control is **8.78% shorter**; another simulation
stage still sets its allocated GPU peak, so its whole-CLI memory is
unchanged. The isolated filter's >2× speedup is not the whole-command
speedup: detector simulation, crowding, solvation and report work remain.

[Whole-CLI timing and memory figure](timing-memory.png),
[PDF](timing-memory.pdf), [numeric summary](summary.json). The four
`before/after-*.json` files record all stages and all match-report metrics.

Before and after use the same GPU within each case. Whole-CLI timing
includes potential builds, all simulations, input loading, CPU metrics,
report rendering and output writing. Imports and frozen-template I/O are
excluded. Both revisions replay the same baseline CPU potential templates
after cache-miss builds still execute, eliminating pre-existing atomic-build
roundoff and giving both revisions the same replay/copy overhead.

CUDA allocated and reserved peaks are the maxima over all ten jobs;
sampled driver usage is per benchmark process every 0.3 seconds and can
miss short transients. The host and other GPUs are shared. These are one
complete baseline/final pair per case, not statistical throughput estimates.
Stage records, native battery timings and separate repeated filter trials
are retained so whole-CLI variation is not mistaken for filter speed.

## Isolated filter results

The warmed dense-volume diagnostic alternates before/after/after/before.
It holds one extra float32 canvas and one float16 canvas, representing
already-existing material absorption and free-water storage. That half
precision field is existing physics, not a new dtype change. Input
generation, comparison transfers and comparisons are outside timed
filtering; all filtering and both resident fields remain on the GPU.
Both revisions' production FFT plans are warmed before timing, and scratch
is released before each trial.

| Dense-field geometry | Median filter before → after | Speedup | Allocated GPU peak before → after |
|---|---|---|---|
| 1641 × 1024 × 1024, native material case | 0.5816 → 0.2548 s | 2.28× | 25.184 → 22.546 GiB |
| 820 × 512 × 512, native 256 px per-volume case | 0.09466 → 0.04411 s | 2.15× | 5.557 → 2.917 GiB |
| 300 × 2014 × 2014, tilt padded geometry | 0.5134 → 0.2218 s | 2.31× | 18.628 → 15.984 GiB |

Each saves approximately **2.64 GiB** of working allocation. All filtered
values are bitwise identical across **3,152,530,096 voxels** in the three
geometries. These are stage-level results; the CLI's gain is measured
separately above.

[Filter timing and memory figure](filter-timing-memory.png),
[PDF](filter-timing-memory.pdf), [numeric summary](filter-summary.json).
Raw alternating trials are in `fft-512-final.json`, `fft-256-final.json`
and `fft-tilt-geometry.json`.

These diagnostic figures include those live fields but exclude the rest
of the CLI model. The native-256 and tilt-geometry diagnostics are
per-volume measurements, not whole-command memory or timing claims. The
tilt padded geometry matches the prior shipped-config validation's
`[300, 2014, 2014]` at 5 Å/voxel; its diagnostic exposure is the same
40-frame weighted 40 e⁻/Å² filter used in this controlled comparison.

The budget trials record why 2²² was chosen. Reusing scratch with the old
2²⁶ weighting budget saves memory but little time; 2²⁴ and 2²³ weighting
slabs remain slower, while 2²⁰ adds enough loop overhead to be slower again.
Smaller FFT batches also improve the full filter while lowering scratch.
The final complete CLI runs use 2²² for large-field FFT and weighting
batches and preserve the original small-volume single batches.

## Precision and Fourier comparison

All **173,015,040 detected output pixels** across all twenty stacks are
bitwise identical, with maximum absolute difference zero. All report
metrics (including every probe score), selected settings and derived
acquisition values are identical. Output-directory paths are normalized
when comparing `matched.toml`. Undefined finest-band SNR entries in the
default control remain undefined on both sides.

No measured precision loss remains in the final policy. The filtered
fields in all three dense production-sized diagnostics are also bitwise
identical. The earlier noise-changing small-batch candidate was not
accepted as the final implementation.

The figures show a seed-0 particle before/after, its zero difference,
mean two-dimensional Fourier power over all 200 native battery particles,
and radial power overlays. Image and Fourier color limits are shared
within each before/after pair. Both 2D and radial powers are identical.

- [Native 512 px Fourier comparison](fourier-material-0.png),
  [PDF](fourier-material-0.pdf), [all-stack hashes](precision-material-0.json).
- [Native 256 px Fourier comparison](fourier-uniform-256.png),
  [PDF](fourier-uniform-256.pdf), [all-stack hashes](precision-uniform-256.json).

The optimization preserves the match's existing scientific limitations.
The 512-pixel material case retains its 10× residual warning, selected
1200 Å ice and 1× particle-diameter crowd spacing. The uniform control
retains its 9× residual warning, selected 1200 Å ice and 1.3× spacing.
This optimization does not fit away the simulation-versus-experiment
residual or change any acquisition parameters.

## Checks and reproduction

CPU validation: **97 passed, 34 CUDA cases skipped** across the focused
regressions, FFTs, dose damage, specimen/solvent exposure and forward-model
parity. The final numerical regression module passed all 62 CPU/CUDA
cases before the production small-plan regression was added; all four
added small-plan CPU/CUDA cases also passed. That includes 33 CUDA cases.
Checks cover odd/rectangular and non-contiguous inputs, partial chunks,
float32/float64 and complex64/complex128 values, aliased and separate
outputs, repeated cached spectra, envelope gradients and exact retained
small-field FFT plans. Ruff, formatting and source mypy pass.

`small-plan-roundoff.json` quantifies the reason for retaining small plans:
forcing smaller batches on the 128/136 × 256² probe fields changed values
by at most 2.98e-8, relative L2 below 1e-7; the final preserved-plan policy
restores exact values. A changed Poisson electron count can alter the size
of subsequent subpixel-jitter draws in the existing coincidence detector,
so tiny field roundoff can change a full noisy realization. The final
complete comparisons check detector outputs as well as the fields.

Use `docs-figures/match_absorption_prepare.py` to reproduce the 256-pixel
control and configurations from the real input data. It verifies a native
200 × 512 × 512 input and records data hashes. For both baseline and final,
run the unchanged `docs-figures/match_absorption_benchmark.py` with the
same `--replay-potentials` directory (capture it once beforehand). Directory
names are `before-material-0`, `after-material-0`, `before-uniform-256`
and `after-uniform-256` under the comparison root.

```bash
uv run --no-sync python docs-figures/match_absorption_benchmark.py \
  --checkout /tmp/baseline --config /tmp/inputs/material.toml \
  --device cuda:1 --seed 0 --output /tmp/results/before-material-0 \
  --replay-potentials /tmp/templates/material
uv run --no-sync python docs-figures/match_absorption_benchmark.py \
  --checkout /tmp/candidate --config /tmp/inputs/material.toml \
  --device cuda:1 --seed 0 --output /tmp/results/after-material-0 \
  --replay-potentials /tmp/templates/material
uv run python docs-figures/match_absorption_compare.py \
  --root /tmp/results --output /tmp/comparison
uv run python docs-figures/dose_damage_fft_probe.py \
  --baseline /tmp/baseline --candidate /tmp/candidate \
  --weights refm_empirical_dw.npy --device cuda:3 \
  --output /tmp/results/fft-512-final.json
```

Repeat the CLI pair with `uniform-256.toml` and its own frozen template
directory. For filter comparisons, repeat the diagnostic with
`--shape 820 512 512 --pixel-size 1.462` and
`--shape 300 2014 2014 --pixel-size 5.0`, writing `fft-256-final.json`
and `fft-tilt-geometry.json`. Generate the diagnostic figure with:

```bash
uv run python docs-figures/dose_damage_fft_figures.py \
  --root /tmp/results --output /tmp/comparison
```

`fft-512-warm-sweep.json`, `fft-512-budget16m.json` and
`fft-512-budget4m.json` retain the large-field budget-selection trials.
The final warmed diagnostics and CLI runs are reported separately.
