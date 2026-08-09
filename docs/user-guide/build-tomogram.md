# Build a tomogram specimen

`specter build tomogram` composites a cryo-ET specimen volume from any
combination of: one or more organic membranes, filament species (e.g.
F-actin), and densely packed protein species (region-gated to cytosol/lumen
when a membrane is present). It writes a `.mrc` density volume plus
copick-style `.ndjson` ground-truth picks and, by default, segmentation
label volumes. The `.mrc` is directly usable as `specter simulate
tiltseries`'s `--volume_path` — this is the specimen-building half of the
cryo-ET pipeline; see [Generate a tilt series](tilt-series.md) for the
imaging half.

## Basic usage

Every run loads a TOML config first, then applies any flags given on the
command line as overrides:

```bash
specter build tomogram --config configs/tomogram.toml
```

```bash
specter build tomogram \
    --config configs/tomogram.toml \
    --v_size 5.0 \
    --device cuda:0 \
    --output_dir ./output/
```

`configs/tomogram.toml` is the canonical starting point — copy it and edit
for your own runs. It demonstrates every feature at once (composited
membranes, filaments, exact-count targets, ratio-based filler); run
`specter build tomogram --help` for the full field-by-field reference.

## What you'll usually tune

- **Box size** — `target_shape` (Z, Y, X voxels) and `v_size` (Å/voxel).
- **Protein species** — `[[targets]]` (exact `n_copies` each, always
  exported to picks) and `[[filler]]`/`filler_from_pei2016`/
  `filler_from_cryoetsim` (packed around targets up to
  `filler_occupancy_fraction`, excluded from picks by default). See
  [Placement order & regions](#placement-order-regions) below.
- **Membranes** — `[[membrane]]` entries: `shape_backend`
  (`spherical_harmonics` or `swept_spline`), size ranges
  (`sh_axes_range_a`/`swept_total_length_range_a`/etc., omit for realistic
  auto-sizing), and `n_instances` for multiple independently-placed copies
  of one template.
- **Filaments** — `actin` (quick built-in F-actin preset) or hand-written
  `[[filaments]]` entries for other species (e.g. microtubules).

Everything else — picks/segmentation toggles, compute/scaling knobs, output
naming — lives under its own panel in `specter build tomogram --help`.

## Placement order & regions

Generation always proceeds membranes → filaments → protein fill; each stage
avoids the placements of the ones before it. Within the protein-fill stage,
species are placed in two priority tiers, independently per region:

1. **`[[targets]]`** — placed first, each at an exact `n_copies` instance
   count. This is the annotated ground truth, always exported to picks.
2. **`[[filler]]`** (plus `filler_from_pei2016`/`filler_from_cryoetsim`) —
   placed second, packed around the already-placed targets until
   `filler_occupancy_fraction` (a bare-sphere volume fraction, per region)
   is reached or the packing jams — whichever comes first, so this rarely
   needs hand-tuning. Excluded from picks by default (`write_picks` still
   controls this; see the CLI help for the exact rule).

`location = "cytosol"` (default) or `"lumen"` on a `targets`/`filler` entry
only matters when a `[[membrane]]` is present — without one, the whole box
is "cytosol." `filler_from_pei2016`/`filler_from_cryoetsim` pull from
bundled reference tables (Pei et al. 2016 generic crowding, and the
CryoETSim dataset table respectively) instead of hand-listing PDB codes;
both are additive and can be combined with each other and with hand-written
`[[filler]]` entries.

## Compute & scaling flags

For anything past a small smoke-test box, rendering dozens of species and
packing hundreds of filler instances can be slow or OOM — which is why
`configs/tomogram.toml` already sets `render_workers = "auto"`,
`accumulator_device = "auto"`, and `chunk_size = 64`. Most runs can just
keep those `"auto"` defaults rather than hand-tuning:

- **`render_workers`** — how many PDB species render/fetch concurrently.
  `"auto"` picks `min(n_species, 8)`, the measured sweet spot from a
  production-scale sweep (TOML/Python config only — the `--render_workers`
  CLI flag stays integer-only).
- **`render_devices`** — an optional device pool (e.g. `["cuda:0",
  "cuda:1"]`) to round-robin those concurrent species across on a
  multi-GPU machine; `"auto"` uses every visible GPU.
- **`accumulator_device`** — device for the shared canvas tensors,
  decoupled from `device` (which stays the compute device regardless).
  `"auto"` estimates the canvas' memory footprint and falls back to CPU
  RAM if it wouldn't fit half of `device`'s free VRAM — useful once a
  large field of view at fine `v_size` needs tens of GB.
- **`chunk_size`** — instances rotated per GPU batch for a single species.
  Only matters once a filler species' instance count reaches the hundreds
  (a single unchunked batch has been measured at 8+ GB); leave unset for
  small runs.

If you do need to reason about these directly, `configs/tomogram.toml`'s
comments walk through the concrete OOM numbers that motivated each default.

## Output: picks & segmentation

Alongside `{filename}.mrc`, by default you get:

- **Picks** (`write_picks`) — one copick-style
  `{species}-{annotation_version}_orientedpoint.ndjson` file per species.
- **Segmentation** (`write_segmentation`) —
  `{filename}_protein_labels.mrc` (always), plus
  `{filename}_membrane_labels.mrc` and `{filename}_regions.mrc`
  (`0=cytosol`/`1=shell`/`2=lumen`) when a `[[membrane]]` is set. This is
  the intended ground truth for membrane geometry specifically — a
  membrane surface has no single natural "position" the way a protein
  does, so it isn't represented in the picks files.

## Multiple tomograms

`--n_tomograms` generates several independent tomograms in one run. Beyond
the first, each is written into its own numbered subdirectory of
`output_dir` (`0001/`, `0002/`, ...) and, if `seed`/`--seed` is set, gets
its own incrementing seed — so runs don't collide but stay reproducible.

## Using it from Python instead of the CLI

`run_build_tomogram(config)` (`specter.pipelines`) is the same function the
CLI calls — build a `TomogramConfig` directly in Python (or load one with
`specter.config.load_config`) instead of going through the command line.

## See also

- [`configs/tomogram.toml`](https://github.com/joelyeois/specter/tree/main/configs) —
  the canonical, heavily-commented example config.
- [Specimens](../concepts/specimens.md) — how cryo-ET specimen assembly
  differs from single-particle.
- [Membrane shape](../concepts/membrane-shape/index.md) — the
  `spherical_harmonics`/`swept_spline` geometry backends.
- [Generate a tilt series](tilt-series.md) — imaging the volume this page
  builds.
- [Configure a run](configuration.md) — general TOML/CLI field reference
  conventions shared across all `specter` commands.
