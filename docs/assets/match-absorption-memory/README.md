# Material absorption memory in `specter match particles`

Baseline: `af847ed`. Candidate: separate elastic/absorptive fields during
multislice, and reuse the privately allocated occupancy field for its two
absorption arithmetic operations. No change to physics, dtype, sampling,
random seeds, batch size, candidate grids, or model selection. The same
particle forward model is used by particle simulation and matching.

The saving applies when `inelastic_mfp_specimen` is supplied. Uniform
absorption already follows a bounded multislice path and is unaffected.
Other scattering models retain their previous complex-volume arithmetic.
Both full input fields remain on the GPU; this change introduces no host
offload. The old complex-volume API remains available.

## Production workload

There is no shipped canonical matching TOML. These runs use `MatchConfig`'s
production defaults: 64 particles per pose check/probe, ice candidates
0/400/800/1200 Å, crowd candidates 0/1/1.3 particle diameters, `probe_bin=2`,
and two 200-particle batteries at the input stack's native box. All ten
stacks, CPU metrics, report figures and `matched.toml` are generated through
the actual CLI. No `--test-run` or lowered particle count is used.

The real EMPIAR-11377 input is the existing aligned J446 512-pixel particle
export, with 8B0X assembly 1, 300 kV, Falcon 4i, 40 e⁻/Å², 40 frames and
the dataset's empirical dose weights. The 512 export already existed before
this benchmark; the optimized run does not bin it further. Solvent mean
free path is 3950 Å; the material case uses specimen mean free path 2460 Å.
The existing CLI selects batch size 1 for this case. Its native padded
potential is `[1, 1641, 1024, 1024]` at 0.731 Å/voxel.

The separate uniform-absorption control uses a single, frozen Fourier crop
of the first 200 experimental images to 256 pixels, identically supplied
to both revisions. That case retains the default batch size 4, all probe
counts and both batteries. Its native potential is
`[4, 820, 512, 512]` at 1.462 Å/voxel. This control verifies the unchanged
default path; it is not a resolution reduction applied by the optimization.

`material.toml` and `uniform-256.toml` record the actual configurations;
`inputs.json` identifies the source files and hashes the image arrays used.
Experimental data, atomic templates and raw output stacks are not bundled.

## Measurements

| Case | CLI before → after | Allocated GPU peak | Reserved GPU peak | Driver GPU peak |
|---|---|---|---|---|
| Native 512 px, material absorption | 1391.3 → 1328.5 s (23m11s → 22m08s) | 26.693 → 26.235 GiB | 42.229 → 39.012 GiB | 42.729 → 39.514 GiB |
| 256 px, uniform control | 368.1 → 377.3 s (6m08s → 6m17s) | 9.293 → 9.293 GiB | 12.535 → 12.535 GiB | 13.035 → 13.035 GiB |

Native material absorption saved **0.458 GiB (1.72%) allocated** and
**3.217 GiB (7.62%) reserved** over the whole command; sampled driver usage
fell by 3.215 GiB (7.52%). Elapsed time was 4.52% shorter (1.047×).
The uniform control used exactly the same GPU memory and was 2.51% slower.
The small timing changes should not be treated as a guaranteed speedup.

At the exact native material propagation geometry, the alternating
whole-complex/paired diagnostic reduced allocated peak from
25.930 to 13.109 GiB (**49.4%**), with identical exit waves. Mean propagation
time was 0.2493 versus 0.2459 seconds, effectively unchanged across these
few samples. An earlier simulation stage still sets the full CLI peak,
so the propagation saving must not be advertised as a 49% CLI saving.

[Timing and memory figure](timing-memory.png),
[PDF](timing-memory.pdf), [numeric summary](summary.json), and
[propagation measurements](scattering-1641x1024.json). Each full-run JSON
also records all ten stages and every match-report metric.

Hardware: NVIDIA L40 (46,068 MiB reported capacity), PyTorch 2.5.1+cu121;
four CPU threads. Each before/after pair runs sequentially on the same GPU.
The host and other GPUs are shared. Timing is a single-pair observation,
not a statistical speed claim. Allocated and reserved peaks are CUDA
allocator counters across all ten simulation jobs; driver usage samples
the benchmark process every 0.3 seconds and can miss brief transients.

CLI elapsed time includes potential builds, all simulation jobs, input
loading, CPU comparison and report/output writing. Imports and frozen
template I/O are excluded. To isolate pre-existing GPU atomic-build
roundoff, the baseline CPU templates are captured after its timed run and
replayed into the candidate's potential results. Cache-miss GPU builds
still execute inside both timed runs. The candidate also performs the
extra CPU template copy, so these are conservative candidate timings.
Its loaded frozen templates affect host RSS; no host-memory saving or
tradeoff is claimed from these runs.

## Scientific comparison

All **173,015,040 output pixels** across the twenty stacks from the two
cases are bitwise identical; maximum absolute difference is zero.
Both cases also have exactly identical report metrics and selected
`matched.toml` settings, after normalizing their output-directory paths.
Undefined SNR entries are compared as undefined on both sides.
Float32/float64 and gradient regression checks also preserve exact values.
There is no measured precision loss.

The figures show the same seed-0 particle before/after, its zero difference,
mean two-dimensional Fourier power over all 200 native battery particles,
and overlaid radial spectra. Image limits and Fourier color limits are
shared within each pair. Two-dimensional and radial powers are identical.

- [Native 512 px Fourier comparison](fourier-material-0.png),
  [PDF](fourier-material-0.pdf), [all-stack hashes](precision-material-0.json).
- [Default 256 px Fourier comparison](fourier-uniform-256.png),
  [PDF](fourier-uniform-256.pdf), [all-stack hashes](precision-uniform-256.json).

The optimization preserves the existing match's quality, including its
limitations. Both cases pass the pose-alignment check and select 1200 Å
ice. The material case selects 1× particle-diameter crowd spacing and
retains matched-pose SNR ratios approximately
3.26/4.64/10.14/23.87/11.51 and twin Cohen's d 0.97487. The uniform control
selects 1.3× spacing, retains its 9× residual warning, and has undefined
finest-band SNR ratio because the experimental estimator is negative in
that band. These are existing simulation-versus-experiment residuals;
this memory change neither improves nor conceals them.

## Validation and reproduction

CPU checks: 158 passed, 10 CUDA skips across scattering, particles,
absorption, forward-model parity, matching and match metrics before the
private occupancy reuse; then all 50 absorption tests passed, and all four
new float32/float64, small/large slab value/gradient cases passed after
that addition. Shared micrograph/tilt absorption checks: 29 passed, two
CUDA skips. The targeted scattering run passed 23 checks, including eight
CUDA paired-field cases; both existing CUDA host-streaming and
projection-memory regressions also passed. CUDA paired
tests cover both Ewald signs, both precisions, partial chunks, batch >1,
nonzero amplitude contrast, uniform attenuation, and exact gradients.
Ruff and mypy checks pass for the modified source.

Prepare the inputs once, replacing the paths with the real dataset paths:

```bash
uv run python docs-figures/match_absorption_prepare.py \
  --metadata J446.cs --images native-512.mrcs \
  --pdb 8b0x-assembly1.cif --weights refm_empirical_dw.npy \
  --output /tmp/match-inputs
```

Use the baseline's environment for both runs. For each case, run the
benchmark at the baseline checkout with `--capture-potentials`; then run
the candidate with `--replay-potentials` pointing to that same directory:

```bash
uv run --no-sync python docs-figures/match_absorption_benchmark.py \
  --checkout /tmp/baseline --config /tmp/match-inputs/material.toml \
  --device cuda:1 --seed 0 --output /tmp/results/before-material-0 \
  --capture-potentials /tmp/results/potentials
uv run --no-sync python docs-figures/match_absorption_benchmark.py \
  --checkout /tmp/candidate --config /tmp/match-inputs/material.toml \
  --device cuda:1 --seed 0 --output /tmp/results/after-material-0 \
  --replay-potentials /tmp/results/potentials
```

Repeat with `uniform-256.toml`, directory suffix `uniform-256`, and a
separate potential directory. Generate comparisons and figures:

```bash
uv run python docs-figures/match_absorption_compare.py \
  --root /tmp/results --output /tmp/comparison
uv run python docs-figures/match_absorption_scattering_probe.py \
  --device cuda:1 --output /tmp/scattering-1641x1024.json
```

The propagation diagnostic alternates old whole-volume assembly and
paired-field assembly on the same native-sized tensors. It includes both
real input fields and complex assembly, and checks every exit-wave value.
It measures propagation alone and must not be used as a whole-CLI memory
or speed figure.
