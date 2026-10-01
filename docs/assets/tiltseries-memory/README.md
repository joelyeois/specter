# Production tilt-series memory comparison

This compares the generator and CLI at baseline commit `bfdf06c` with the
unity-scale guard and optional diagnostic collection. The imaging settings
come from `configs/tiltseries.toml`, including all 61 tilts from −45° to +45°,
multislice scattering, GD ice, Poisson noise, 10 frames, and 5 Å sampling.
The input is the existing `tomograms/tomogram.mrc`: 300 × 1200 × 1200
float32 voxels, 1,728,001,024 bytes. Its SHA-256 is
`f3c3c2bb443c4301f6a9902eb9c096deeaaf3c00d5a709b92625d082b5d7a105`.
It matches the shipped specimen geometry; it was not rebuilt from the
current tomogram recipe for this experiment. The tilt geometry pads it to
300 × 2014 × 2014 without reducing resolution or slab count.

Hardware: NVIDIA L40 GPUs, PyTorch 2.5.1+cu121, four CPU threads per
benchmark process. Independent CLI runs use seed 73. The additional
same-model comparisons use Poisson seed 173, with three before/after
pairs on GPU 1 and the order reversed in the middle pair. They reuse the
exact constructed ice and specimen, avoiding variation from another ice
construction. Both implementations are loaded from their own source files.

## What changed

At a potential scale of exactly 1, scattering reads the current padded
volume directly. Other scales still multiply it. The current volume is
resolved after each solvent or damage render, including host fallback.
The CLI does not accumulate exit waves when `save_exitwaves=false`, and
does not accumulate the unused noiseless stack. Python callers still get
all three stacks by default; the collection flags explicitly opt out.
The scattering model, precision, geometry, doses, and noise draws are
unchanged.

## Results

| Metric | Before | After |
| --- | ---: | ---: |
| Full CLI wall time, one run each | 934.61 s | 938.79 s |
| Generation, median of three same-GPU pairs | 32.96 s | 32.73 s |
| Peak CUDA allocated | 9.295 GiB | 4.762 GiB |
| Peak CUDA reserved, full CLI | 9.375 GiB | 4.842 GiB |
| Sampled per-process driver GPU peak | 9.869 GiB | 5.336 GiB |
| Peak host RSS, full process | 83.940 GiB | 83.964 GiB |
| Same-volume image/Fourier error | reference | exactly zero |

The allocation reduction is **4.533 GiB (48.77%)**, exactly one padded
float32 volume. Generation's median difference is 0.23 s (0.70%), with
individual paired improvements of 0.05, 0.23, and 0.004 s. These timings
do not establish a reliable speedup. The extra candidate CLI run on GPU 1
took 902.84 s, further illustrating setup/timing variation. Ice
construction dominates both whole-command time and host RAM; neither is
materially reduced by this change.

All three same-volume pairs produced bitwise-identical detected images:
87,840,000 pixels per pair, maximum absolute difference 0, relative L2
error 0. Their independently computed radial power profiles are exactly
equal. No numerical precision or physical approximation changed.
The independently constructed CLI runs differed in 19,829 pixels
(0.0226%), with relative L2 error 0.000594 and maximum difference 2.
That comparison includes separate ice construction and Poisson sampling;
the identical-state paired experiment isolates the implementation change
and has zero error. Both measurements are retained in the JSON.

## Measurement conventions

`measurements.json` contains the complete CLI measurements and all three
paired generation trials. CLI wall time includes ice construction,
simulation, and output writing, and excludes Python imports. Generation
time is synchronized at both ends and includes transfer/stacking of any
requested diagnostics. CUDA peak allocation and reservation are measured
by PyTorch; per-process driver memory is sampled once per second.
Host RSS is sampled every 0.2 seconds, with `getrusage` providing the exact
whole-process high-water mark. GiB means 2³⁰ bytes.

The one-off full CLI runs use GPUs 1 and 3; the extra candidate run and
the paired timings use GPU 1. Setup runs overlap on the host, so isolated
generation timing is the stronger evidence for a speed change. Whole-CLI
differences include setup variation and should not be extrapolated from
one baseline run. The peak host allocation belongs to ice construction;
removing diagnostic stacks does not remove that peak.

## Figures

- `images_and_fourier.png` / `.pdf`: the central 0° image before/after,
  difference, matching-scale 2D Fourier power, and mean radial power from
  all 61 tilts of the same-volume comparison.
- `timing_and_memory.png` / `.pdf`: actual CLI time, median paired
  generation time, GPU allocation, and whole-command peak host RSS.
- `spectra.csv`: radial power profiles, computed independently from each
  saved stack. Each image is centered before its 2D FFT; power is
  `|F|² / N_pixels`, averaged over Fourier pixels and then tilts. The
  radial plot omits DC and stops at Nyquist (0.1 Å⁻¹).

Large MRC stacks remain in `/tmp/specter-tiltseries-validation`, outside
Git. The committed figures and JSON contain the measured results.

## Validation

The focused CPU tests passed (14 passed, 12 CUDA cases skipped). The
broader CPU geometry, absorption, exposure, forward/backend parity, and
CLI tests passed (72 passed, 1 skipped). GPU validation passed all 54
tests across the new regression module, explicit absorption, and backend
parity. The new tests check exact image/diagnostic equality, preservation
of random state, non-unit scaling, moving ice, per-tilt specimen damage,
and both CLI exit-wave export settings. Ruff and source type checks pass.
After rebasing onto the concurrent main-branch test-suite update
(`a8d8e7e`), all 26 focused CPU/GPU tests passed again. The measured
generator and pipeline source hashes are unchanged by that integration.

## Reproduce

Create a baseline checkout at `bfdf06c` and a candidate checkout with
this change. From an environment with SPECTER's dependencies:

```bash
python docs-figures/tiltseries_memory_benchmark.py \
  --checkout /path/to/baseline --volume-path /path/to/tomogram.mrc \
  --device cuda:1 --output /tmp/comparison/before-1
python docs-figures/tiltseries_memory_benchmark.py \
  --checkout /path/to/candidate --volume-path /path/to/tomogram.mrc \
  --device cuda:3 --output /tmp/comparison/after-1
python docs-figures/tiltseries_memory_benchmark.py \
  --checkout /path/to/candidate --volume-path /path/to/tomogram.mrc \
  --device cuda:1 --output /tmp/comparison/after-2-paired \
  --paired-baseline /path/to/baseline
python docs-figures/tiltseries_memory_figures.py \
  /tmp/comparison docs/assets/tiltseries-memory
```
