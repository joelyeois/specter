# Saved production ice comparison

These figures use the six saved full-size cells from the paired GPU validation
of commit `f2df195`, before and after filtering inactive ML-BOP neighbour pairs.
They do not use regenerated coordinates or synthetic example images.

Geometry: 256³ voxels, 1 Å/voxel, 256 Å periodic box, 527,178 water molecules.
All builds use the standard 250-step ceiling and normal plateau stopping rule.
The comparison contains seeds 1000, 1001 and 1002. NVIDIA L40 GPUs;
PyTorch 2.5.1+cu121; four CPU threads per benchmark process.

## Figures

- `fourier_comparison.png` / `.pdf`: radial amplitude and magnified residuals
  against the production MD target. Grey envelope is the min–max across all
  20 shipped ice cells, recomputed from their saved coordinates.
- `fourier_profiles.png`: full-frequency amplitude, residual and power plots.
- `ice_and_fourier_seed_1000.png`, `...1001.png`, `...1002.png`: same central
  4 Å mean-density slab (80 Å square) and central plane of the **3D** Fourier
  transform. Matching display scales across all seeds. The Fourier map is not
  a transform of the displayed slab. These are water number-density images,
  before any water scattering kernel, CTF or detector processing.
- `timing_and_quality.png`: actual build times, step counts, same-input kernel
  times, and saved-coordinate quality relative to the shipped library.
- `ice_comparison.pdf`: six pages containing the magnified comparison,
  full spectra, all three image comparisons, and the timing/quality figure.

## Measurement conventions

Coordinates are decoded from each saved file before analysis. The voxelizer,
FFT, target and frequency bins match `GradientSKIcemaker._sk_loss`.
Amplitude is the shell mean of `|F| / sqrt(N)`. Power is the shell mean of
`|F|² / N`; it is independently averaged, not squared mean amplitude.
DC is excluded only from plots. The production loss includes all bins.
The loss is MSE in unnormalized amplitude units, as reported by the CLI.

The frequency range extends to the corners of the Fourier cube
(approximately sqrt(3) * 0.5 Å⁻¹). Above 0.5 Å⁻¹, shells are incomplete and
the corner bins contain few modes. The primary residual panel zooms to
0 < k <= 0.5 Å⁻¹, while the full figure retains the corners. Maximum
after-change amplitude deviation below 0.5 Å⁻¹ is 0.02241%. Some
98–99.6% of the after-change squared radial residual is above that boundary.
This is a decomposition of the same production loss, not a replacement metric.

Same-seed cells are not identical because CUDA floating-point accumulations
change the optimizer trajectory. The saved-coordinate loss is higher after
the change for seeds 1001 and 1002, although all three remain within the
shipped library's measured loss range. Visual similarity does not remove
that quality difference.

Whole-build timing includes optimizer initialization, optimization, quality
diagnostics, and coordinate storage. Completed steps differ. The kernel
comparison evaluates the exact same coordinates and cached neighbor lists,
with five warmed trials per variant, CUDA synchronization, forward energy,
backward gradients, and CPU readback. It excludes neighbor-list construction.
Its speedups are 1.51–1.80x; whole-build speedups are 1.22–2.32x.

## Data and reproduction

- `timings.csv`: complete build wall times, steps and quality for each pair.
- `measurements.json`: original benchmark data and recomputed spectral losses.
- `spectra.csv`: target, shipped envelope and all before/after radial curves.
- `spectra.npz`: radial curves, shipped curves and the displayed image arrays.
- `spectral_summary.csv`: complete-shell peak amplitude deviations and the
  fraction of squared residual above 0.5 Å⁻¹.

From the repository root, render all six PDF pages and the figures from
committed measurements on CPU:

```bash
uv run python docs-figures/ice_active_pairs.py
```

To recompute the spectra on a GPU while the original validation cells are
available:

```bash
uv run python docs-figures/ice_active_pairs.py --recompute --input /tmp/specter-ice-validation --device cuda:1
```

Use `--output` to select an output directory containing the saved measurement
files when rendering, or an empty destination when recomputing.
