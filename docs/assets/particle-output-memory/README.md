# Production particle output comparison

Baseline: `050cab8`. Settings: `configs/particle.toml`, extended to the
documented 3,000-particle example, with batch size pinned to 2 and seed 83.
The 6BDF biological assembly, 256-pixel box at 1 Å/pixel, Shtyrov analytic
potential, multislice scattering, GD ice, crowding, Poisson noise,
20 e⁻/Å², FFT padding, and normalization are retained. Scattering volumes
have 256 × 512 × 512 voxels. No resolution, dose frames, or physics
approximation is reduced. Default diagnostic collection remains disabled.

NVIDIA L40 GPUs; PyTorch 2.5.1+cu121; four CPU threads per process.
The single-GPU before/after runs use GPUs 1 and 3. The final two-GPU
before/after runs use idle GPUs 0 and 3. Earlier two-GPU runs on GPUs 1
and 3 are excluded from the final timing comparison because another job
started using GPU 1 during the candidate repeat. Large outputs are kept
outside Git under `/tmp/specter-particle-validation`.

## Results

| Metric | Before | After |
| --- | ---: | ---: |
| Single-GPU full CLI | 143.70 s | 144.60 s |
| Single-GPU peak host RSS | 5.304 GiB | 3.876 GiB |
| Single-GPU peak CUDA allocated | 1.326 GiB | 1.326 GiB |
| Two-GPU full CLI, idle GPUs 0 and 3 | 87.26 s | 82.89 s |
| Two-GPU peak process-tree PSS | 6.127 GiB | 4.137 GiB |
| Two-GPU peak root-process RSS | 5.379 GiB | 4.334 GiB |
| Two-GPU peak CUDA allocation per rank | 1.299 GiB | 1.299 GiB |
| Peak process count | 258 | 2 |

The single-GPU run saved 1.427 GiB of host RAM
(26.9%) with essentially unchanged runtime.
The idle two-GPU run used 32.5% less process-tree PSS and finished
5.0% sooner (1.053×). These are one-off complete CLI
measurements. The benefit is host memory and scheduling; GPU memory is
unchanged. The prior 198.70 s/248.96 s two-GPU runs on GPUs 1 and 3 are
excluded from the speed comparison after concurrent GPU work was observed.

In all three matched-batch checks, the complete normalized output is
bitwise identical: 196,608,000 pixels, maximum absolute error 0, with
exactly equal radial Fourier power. The normalization stage's median
time was 0.682 → 0.279 s; collection itself was 0.466 → 0.736 s. These
storage-stage timings do not substitute for full CLI timings above.
The independent single-GPU runs have identical STAR metadata but relative
image L2 difference 0.001540 (0.154%). That comparison includes fresh
potential/ice rendering and Poisson counts; the matched-batch result
isolates the operations changed by this patch and has zero error.

## Change and precision

The single-GPU collector copies each batch into one host stack instead
of retaining and concatenating all batches. The CLI normalizes in chunks
of 64 particles in the owned stack, rather than retaining several
whole-stack intermediates. It uses the existing background mask, mean,
standard deviation, epsilon, and contrast sign.

The multi-GPU integer-index loader uses zero workers instead of 128 per
rank. The batch prediction writer preallocates each rank's stack, records
the actual particle indices, and releases its buffers after writing.
Rank 0 validates complete coverage and loads/scatters one shard at a time
to preserve image/metadata order. Requested complex exit-wave stacks keep
their original dtype; the rank barrier and cardinality checks remain.
The existing DDP `16-mixed` setting is unchanged, as is single-GPU
precision. GPU physics kernels and batch size are unchanged.

Independent simulations can vary because their potential/ice rendering
and Poisson sampling are not bitwise deterministic. The precision check
therefore captures raw images from an actual 3,000-particle GPU run and
replays **identical generated batches** through both collectors and both
normalizers. This directly tests the changed operations without conflating
them with fresh stochastic specimen rendering. Three repetitions reverse
the collection order in the middle pair. Replay timing is a storage-stage
measurement, not a simulation speedup or simulation GPU memory measurement.

## Measurement conventions

`measurements.json` retains the real CLI measurements, periodic lifetime
samples, matched-batch results, and independent-run image comparison.
CUDA peaks come from actual GPU simulation and include setup. Generation
is synchronized at both ends. CLI time includes potential construction,
sampling, generation, normalization, and MRC/STAR writing, excluding imports.
Host RSS is sampled at 50 ms and reported using `getrusage`'s high-water
mark. Multi-GPU aggregate PSS is sampled once per second over the benchmark
process and descendants; it accounts for shared pages proportionally
instead of adding every forked worker's entire RSS. GiB means 2³⁰ bytes.

Timings are full-size one-off runs. They do not establish a statistically
reliable throughput improvement. Batch size is held fixed to preserve
random-draw assignment. VRAM reduction should not be inferred from the
host-output memory reduction.

## Figures

- `images_and_fourier.png` / `.pdf`: matched normalized particle 0,
  difference, matching-scale Fourier power, and mean radial power from
  all 3,000 particles. Each image is centered before its 2D FFT; power is
  `|F|² / N_pixels`, averaged over shell pixels then particles. The radial
  plot excludes DC and stops at Nyquist, 0.5 Å⁻¹.
- `timing_and_memory.png` / `.pdf`: measured CLI times and host-memory
  peaks for single and two GPUs.
- `spectra.csv`: independently computed matched-stack radial profiles.

## Validation

CLI and pipeline regression checks passed, including metadata ordering
and missing-rank failures. Focused tests cover non-contiguous batches,
uneven final batches, complex diagnostic values/dtypes, unchanged random
state, exact normalization across chunk boundaries, and rank-length errors.
A real two-GPU CLI test uses five particles (uneven shards), exports both
exit-wave pairs, checks all five stack shapes, and verifies rank cleanup.
Ruff and changed-source type checks pass.
The final CPU run passed 54 tests (5 CUDA-dependent cases skipped);
the focused CPU/GPU run passed all 16 applicable cases, and the separate
real two-GPU uneven-shard/export regression passed.

Buffer capacity reads Lightning's nested batch sampler, which describes
the actual rank share even when the loader's top-level sampler reports the
whole dataset. The uneven-shard regression and full-size two-GPU run
exercise that path.

## Reproduce

Create separate baseline and candidate checkouts. With project dependencies
installed, run:

```bash
python docs-figures/particle_output_memory_benchmark.py \
  --checkout /path/to/baseline --device cuda:1 --n 3000 \
  --output /tmp/comparison/before-3000
python docs-figures/particle_output_memory_benchmark.py \
  --checkout /path/to/candidate --device cuda:3 --n 3000 \
  --output /tmp/comparison/after-3000
python docs-figures/particle_output_memory_benchmark.py \
  --checkout /path/to/baseline --device 0,3 --n 3000 \
  --output /tmp/comparison/ddp-before-3000-idle
python docs-figures/particle_output_memory_benchmark.py \
  --checkout /path/to/candidate --device 0,3 --n 3000 \
  --output /tmp/comparison/ddp-after-3000-idle
python docs-figures/particle_output_memory_benchmark.py \
  --checkout /path/to/candidate --device cuda:1 --n 3000 \
  --output /tmp/comparison/same-batches-3000 \
  --paired-baseline /path/to/baseline
python docs-figures/particle_output_memory_figures.py \
  /tmp/comparison docs/assets/particle-output-memory
```
