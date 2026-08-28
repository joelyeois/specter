# Build a tomogram specimen

`specter build tomogram` composites a cryo-ET specimen volume from any
combination of: one or more organic membranes, filament species (e.g.
F-actin), gold fiducial beads, a carbon support film, and densely packed
protein species (region-gated to cytosol/lumen when a membrane is present).
It writes a `.mrc` density volume plus
copick-style `.ndjson` ground-truth picks and, by default, segmentation
label volumes. The `.mrc` is directly usable as `specter simulate
tiltseries`'s `--volume_path`. This is the specimen-building half of the
cryo-ET pipeline; see [Generate a tilt series](tilt-series.md) for the
imaging half.

## Basic usage

Every run loads a TOML config first, then applies whatever flags you pass
on the command line as overrides:

```bash
specter build tomogram --config configs/tomogram.toml
```

```bash
specter build tomogram \
    --config configs/tomogram.toml \
    --voxel_size 5.0 \
    --device cuda:0 \
    --output_dir tomograms
```

`configs/tomogram.toml` is the canonical starting point. Copy it and edit
for your own runs. It demonstrates every feature at once (composited
membranes, filaments, exact-count targets, ratio-based filler); the
[command reference](../api/cli/build.md#specter-build-tomogram) documents every
field.

## What you'll usually tune

- **Box size**: `target_shape` (Z, Y, X voxels) and `voxel_size` (Å/voxel).
- **Protein species**: `[[targets]]` (exact `n_copies` each, always
  exported to picks) and `[[filler]]`/`filler_from_pei2016`/
  `filler_from_cryoetsim` (packed around targets up to
  `filler_occupancy_fraction`, excluded from picks by default). See
  [Placement order & regions](#placement-order-regions) below.
- **Membranes**: `[[membrane]]` entries: `shape_backend`
  (`spherical_harmonics` or `swept_spline`), size ranges
  (`sh_axes_range`/`swept_total_length_range`/etc., omit for realistic
  auto-sizing), and `n_copies` for multiple independently-placed copies
  of one template.
- **Filaments**: `actin` (quick built-in F-actin preset) or hand-written
  `[[filaments]]` entries for other single-strand species.
- **Microtubules**: `[[microtubules]]` entries: real 13-protofilament tubes
  with a lumen and an A-lattice seam (`n_protofilaments`, `n_copies`,
  `length`, `bend_radius`). These aren't filaments: a `[[filaments]]` entry
  with a tubulin dimer gives a single protofilament, not a full tube.
- **Gold fiducial beads**: `[[beads]]` entries (`radius`, `n_copies`), one per
  population; `radius` takes a single number or a `[low, high]` pair drawn per
  bead. Beads avoid the membrane shell and already-placed
  filaments/microtubules, but aren't
  region-gated to cytosol/lumen. All beads go into one `gold-bead` pick
  file regardless of size.
- **Carbon support film**: at most one `[[carbon_film]]` table
  (`hole_radius`/`edge_fraction`/`edge_side`/etc.), painted into the volume
  before anything else.

Everything else (picks/segmentation toggles, compute/scaling knobs, output
naming) lives under its own panel in the
[command reference](../api/cli/build.md#specter-build-tomogram).

## Placement order & regions

Every run places components in this order: carbon film → membranes →
filaments → gold fiducial beads → protein fill. Each stage avoids what
came before it, except the carbon film: its placement isn't carbon-aware
(see `TomogramSpecimenGenerator`'s own docstring). Within the
protein-fill stage, species land in two priority tiers, independently
per region:

1. **`[[targets]]`**: placed first, each at an exact `n_copies` instance
   count. This is the annotated ground truth, always exported to picks.
2. **`[[filler]]`** (plus `filler_from_pei2016`/`filler_from_cryoetsim`):
   placed second, packed around the already-placed targets until it
   reaches `filler_occupancy_fraction` (a bare-sphere volume fraction, per
   region) or the packing jams, whichever comes first, so you rarely need
   to hand-tune it. Excluded from picks by default (`write_picks` still
   controls this; see the CLI help for the exact rule).

`location = "cytosol"` (default) or `"lumen"` on a `targets`/`filler` entry
only matters when a `[[membrane]]` is present. Without one, the whole box
is "cytosol." `filler_from_pei2016`/`filler_from_cryoetsim` pull from
bundled reference tables (Pei et al. 2016 generic crowding, and the
CryoETSim dataset table respectively; see
[Regions & protein packing](../concepts/cryoet-specimen/packing.md#references)
for both citations) instead of hand-listing PDB codes; both are additive
and can be combined with each other and with hand-written `[[filler]]`
entries.

A `"lumen"` entry needs an organelle that actually encloses one. Membrane
sizes are constrained by the thinnest box axis, not the widest: an
organelle wider than the slab is cut open where it meets the z walls, and
an open shape has no interior for the region classifier to find. A
`[[membrane]]` population sized against the field of view rather than
against `target_shape[0] * voxel_size` therefore leaves the lumen region
empty or near-empty, and every species declared for it is dropped with a
warning naming the region's free voxel count. Keeping the widest
organelle to roughly two thirds of the slab thickness makes an enclosed
interior the normal outcome. It is not a guarantee: placement remains
random and clipping remains permitted, so an occasional draw still
encloses little or nothing. Omitting the size ranges does not solve this
on its own, since the automatic cap measures the widest axis whenever
`clip_axes` permits clipping on all three, clipping being a legitimate
outcome for a lamella. Set `seed` when a run has to reproduce a specimen
whose lumen you have already checked.

## Compute & scaling flags

Rendering dozens of species and packing hundreds of filler instances can
be slow or run out of memory past a small smoke-test box, so
`configs/tomogram.toml` already sets `render_workers = "auto"`,
`accumulator_device = "auto"`, and `render_chunk_size = 64`. Keep those
`"auto"` defaults for most runs rather than hand-tuning them:

- **`device`**: `cpu | cuda | cuda:0 | 0,1,2 | auto`. A comma-separated
  list of GPU indices (or `"auto"`, every visible GPU) pools those GPUs for
  concurrent per-species rendering instead of a single device; the first
  entry becomes the primary device for everything else (packing itself
  always runs on CPU regardless).
- **`render_workers`**: how many PDB species render/fetch concurrently.
  `"auto"` picks `min(n_species, 8)`, the measured sweet spot from a
  production-scale sweep (TOML/Python config only; the `--render_workers`
  CLI flag stays integer-only). The sweep found device choice barely
  matters at this worker count.
- **`accumulator_device`**: device for the shared canvas tensors,
  decoupled from `device` (which stays the compute device regardless).
  `"auto"` estimates the canvas' memory footprint and falls back to CPU
  RAM if it wouldn't fit half of `device`'s free VRAM, useful once a
  large field of view at fine `voxel_size` needs tens of GB.
- **`render_chunk_size`**: instances rotated per GPU batch for a single species.
  Only matters once a filler species' instance count reaches the hundreds
  (a single unchunked batch can reach 8+ GB); leave unset for
  small runs.

## Benchmarks: resolution vs. time and memory

To see how `voxel_size` trades off against runtime and memory, this
benchmark builds a specimen at `configs/tomogram.toml`'s own
**production-scale field of view** (1500 × 6000 × 6000 Å) at three voxel
sizes, so only the voxel grid resolution changes between runs, not the
physical amount of specimen content. The specimen itself: one
`spherical_harmonics` membrane, target species `1bxn` × 20, and
`filler_from_pei2016` (20 species sharing a 0.5 occupancy-fraction budget,
same filler approach as the canonical config). A single hand-picked
filler species came first, then got rejected: at this box size it
packed ~109,000 instances of that one small species to hit occupancy,
unrepresentative of real configs and, at voxel_size=2, dominated by
per-instance rendering cost:

![Same field of view rendered at voxel_size = 10, 5, and 2 Å/voxel: a sum Z projection of each output volume, showing the same membrane and densely crowded protein layout at increasing voxel resolution.](../assets/images/tomogram-benchmark-projections.png){ width="900" style="display:block;margin:1.2em auto;" }
///caption
Same field of view rendered at voxel_size = 10, 5, and 2 Å/voxel: a sum Z projection of each output volume, showing the same membrane and densely crowded protein layout at increasing voxel resolution.
///

| `voxel_size` (Å/voxel) | Shape (Z, Y, X voxels) | Wall time | GPU peak | RAM peak |
|---|---|---|---|---|
| 10 | 150 × 600 × 600 | 218 s | 3.70 GB | 12.61 GB |
| 5  | 300 × 1200 × 1200 | 124 s | 11.62 GB | 6.13 GB |
| 2  | 750 × 3000 × 3000 (~6.75B voxels) | 502 s (8m 22s) | 9.29 GB | 154.07 GB |

The numbers don't move in a clean monotonic line at 10→5 Å: wall time and
RAM both bounce around within roughly a factor of 2, since at this scale
species fetch/render/packing overhead (fixed per-run, not per-voxel)
dominates over the canvas itself. The step change at 2 Å is unambiguous:
wall time quadruples versus 5 Å, and RAM jumps to **154 GB**.
That RAM spike is `accumulator_device="auto"` doing exactly what [Compute &
scaling flags](#compute-scaling-flags) above says it does: at ~6.75
billion voxels the density volume alone is ~27 GB, comfortably past half
of this GPU's free VRAM, so the shared canvas tensors get pushed to system
RAM instead of failing with a CUDA OOM. That's also why **GPU peak barely
grows** from 5 Å to 2 Å despite a 15× bigger canvas: the single biggest
consumer (the canvas) has moved off the GPU entirely, leaving only
per-instance rendering buffers and the membrane distance transform behind.
Take the exact numbers with a grain
of salt (one run on one machine, not a statistically averaged sweep), but
the *shape* (noisy at coarse resolution, a sharp RAM/time step once the
canvas stops fitting in VRAM) is the useful, likely-to-generalize part.

Shape-based collision is close to free at this scale. Running the 5 Å
configuration back to back on both backends, wall time is 117 s under
`"shape"` against 116 s under `"sphere"`, for 19,426 placed instances
against 8,128. Filler packing accounts for 28 s and 25 s of those totals
respectively, and rendering for ~27 s in both, since rendering is
dominated by per-species template construction and barely moves with
instance count.

It was not always close. The shape backend's collision loop first cost
104 s on this configuration, four times the sphere backend's. Nearly all
of that was the reject path: at realistic density RSA rejects ~99.8% of
attempts, and each rejection tested a candidate's entire footprint. The
loop now rejects on a sparse sample of footprint voxels first, which is
exact rather than approximate, and only a candidate that survives it pays
for the full comparison.

At 2 Å the collision grid would reach ~6.75 billion voxels, past the budget
`packing_voxel_size` enforces, so packing runs on a 4 Å grid while
rendering stays at 2 Å. This is automatic and needs no configuration. It
costs some packing density, since a coarser grid represents each molecule
slightly more crudely, and it is what allows a fine `voxel_size` to remain
tractable at all: colliding natively at 1 Å over this field of view would
require a 36 GB occupancy grid.

At 2 Å you will also see a warning that the membrane is being generated on
a coarser grid and upsampled. That is deliberate: resolving the bilayer
directly at this voxel size would need ~125M working-grid voxels, past the
memory one field generation is allowed to cost, so it builds at ~100M and
upsamples instead. The membrane's physical size and position are preserved
exactly; only its bilayer sub-structure is resolved less crisply. The
budget behind that switch is fixed (identical on every machine, so the same
config produces the same specimen anywhere) and sized so one field
generation fits an 8 GB card and a 16 GB host; it is not something you
configure. If you want a crisper membrane, use a coarser `voxel_size` or a
smaller field of view.

**Hardware**: single NVIDIA L40 (46 GB VRAM, one idle card selected via
`SPECTER_BENCHMARK_DEVICE` (`cuda:1` for this run), with other cards on the
host under unrelated load, `accumulator_device="auto"`,
`render_workers="auto"`, `render_chunk_size=64`), with CuPy 14.1 (a core
dependency, see [Installation](../installation.md)), on a host with an AMD EPYC 7763 64-Core Processor (128
threads) and 503 GB system RAM. Wall time is the `run_build_tomogram()`
call only (a one-time CUDA context init immediately before it is excluded,
since it's a constant ~1-2s regardless of resolution); GPU peak is
`torch.cuda.max_memory_allocated()` **plus** CuPy's own pool (the membrane
distance transform allocates outside torch's allocator, so torch's counter
alone understates it by 1.7-3.6 GB here); RAM peak is `/usr/bin/time -v`'s
"Maximum resident set size" for the whole process, each resolution run in
its own fresh subprocess so peaks don't carry over between runs. These
benchmark runs turn off picks and segmentation output (at voxel_size=2,
those label volumes alone would add ~40 GB of writes on top of
the ~27 GB density volume; real runs at this scale should budget disk
space accordingly). Reproduce with
[`docs-figures/build_tomogram_benchmark.py`](https://github.com/joelyeois/specter/blob/main/docs-figures/build_tomogram_benchmark.py)
(writes its scratch output outside the repo; the 2 Å volume alone is
larger than many systems' per-user home-directory quota).

## Output: picks & segmentation

Alongside `{filename}.mrc`, by default you get:

- **Picks** (`write_picks`): one copick-style
  `{species}-{annotation_version}_orientedpoint.ndjson` file per species.
- **Segmentation** (`write_segmentation`):
  `{filename}_protein_labels.mrc` (always), plus
  `{filename}_membrane_labels.mrc` and `{filename}_regions.mrc`
  (`0=cytosol`/`1=shell`/`2=lumen`) when a `[[membrane]]` is set. This is
  the intended ground truth for membrane geometry: a
  membrane surface has no single natural "position" the way a protein
  does, so it isn't represented in the picks files.

## Multiple tomograms

`--n_tomograms` generates several independent tomograms in one run. Beyond
the first, each lands in its own numbered subdirectory of
`output_dir` (`0001/`, `0002/`, ...) and, if you set `seed`/`--seed`, gets
its own incrementing seed, so runs don't collide but stay reproducible.

## Using it from Python instead of the CLI

`run_build_tomogram(config)` (`specter.pipelines`) is the same function the
CLI calls. Build a `TomogramConfig` directly in Python (or load one with
`specter.config.load_config`) instead of going through the command line.

## See also

- [`configs/tomogram.toml`](https://github.com/joelyeois/specter/tree/main/configs):
  the canonical, heavily-commented example config.
- [Cryo-ET specimen assembly](../concepts/cryoet-specimen/index.md): what
  each component is and how the pieces are composited, with a page per
  component ([membranes](../concepts/cryoet-specimen/bilayer.md),
  [filaments](../concepts/cryoet-specimen/filaments.md),
  [beads](../concepts/cryoet-specimen/beads.md),
  [carbon film](../concepts/cryoet-specimen/carbon-film.md),
  [regions & packing](../concepts/cryoet-specimen/packing.md)).
- [Membrane shape](../concepts/membrane-shape/index.md): the
  `spherical_harmonics`/`swept_spline` geometry backends.
- [Generate a tilt series](tilt-series.md): imaging the volume this page
  builds.
- [Configure a run](configuration.md): general TOML/CLI field reference
  conventions shared across all `specter` commands.
