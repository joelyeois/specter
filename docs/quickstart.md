# Quickstart

## Simulate a particle stack

The `specter simulate particles` CLI is the quickest way to produce a
simulated cryo-EM particle stack, driven by a TOML config file:

```bash
specter simulate particles --config configs/particle.toml
```

This downloads the structure, builds the scattering potential, applies CTF
and detector effects, and writes a `.mrcs` / `.star` file pair.

An example config lives in
[`configs/`](https://github.com/joelyeois/specter/tree/main/configs) —
copy `configs/particle.toml` and edit it for your own runs. It looks like
this:

```toml
# Canonical default config for `specter simulate particles`.
# Any field can be overridden on the command line, e.g.:
#   specter simulate particles --config configs/particle.toml --n_particles 3000

[potential]
pdb_code = "6bdf"
assembly = true
num_pixels = 256
pixel_size = 1.0              # Å

[microscope]
voltage = 300.0                # kV
dose = "20"                    # e⁻/Å²; single value, or "low,high" to sample per particle
cs = 2.0                       # mm
alpha = 0.1                    # unitless, amplitude contrast ratio

# ... see the full file: configs/particle.toml
```

Any field in the config can be overridden on the command line without
editing the file:

```bash
specter simulate particles \
    --config configs/particle.toml \
    --pdb_code 6bdf \
    --n_particles 200 \
    --device cuda:0 \
    --output_dir ./output/
```

See [Configure a run](user-guide/configuration.md) for the complete field
reference. For the other three workflows (a CryoSPARC dataset twin, full
micrographs, cryo-ET tilt series), see
[Generate a particle stack](user-guide/particle-stack.md),
[Generate a micrograph](user-guide/micrograph.md), and
[Generate a tilt series](user-guide/tilt-series.md).

## Build a tomogram specimen

The `specter build tomogram` CLI packs PDB structures into a crowded 3-D
specimen volume via hard-sphere placement (no membranes yet — that's coming
in a future release), saving a `.mrc` volume plus copick-style `.ndjson`
ground-truth picks:

```bash
specter build tomogram --config configs/tomogram.toml
```

Species are placed in two priority stages: `[[targets]]` first, each at an
exact instance count (the annotated ground truth), then `[[filler]]`
second, ratio-weighted, packed around the already-placed targets to crowd
out the rest of the volume. An example config lives in
[`configs/tomogram.toml`](https://github.com/joelyeois/specter/tree/main/configs) —
copy it and edit for your own runs. It looks like this:

```toml
# Canonical default config for `specter build tomogram`.

[[targets]]
pdb_source = "1bxn"   # cytosolic RNA polymerase II complex (large)
n_copies = 20

[[filler]]
pdb_source = "1mbo"   # myoglobin (small)
ratio = 4.0

[specimen]
target_shape = [128, 256, 256]    # (Z, Y, X) voxels
v_size = 5.0                       # Å/voxel
filler_occupancy_fraction = 0.5    # bare-sphere volume fraction budget for filler

# ... see the full file: configs/tomogram.toml
```

The resulting `.mrc` is directly usable as `specter simulate tiltseries`'s
`--volume_path`. See the field docstrings in
`specter.config.TomogramConfig` for the complete reference.

## Job management

Generation runs can be recorded under a project name in a local job
database. Inspect past runs with the `specter-jobs` CLI (installed
automatically with the package) — see [Manage jobs](user-guide/jobs.md):

```bash
specter-jobs list --project my-project
specter-jobs show <job_id>
specter-jobs diff <job_id_1> <job_id_2>
```

See `demo-notebooks/` for interactive worked examples, including
micrograph and tilt-series generation (`create_micrograph/`,
`tilt-series-generator.ipynb`). For an example of composing the forward
model's individual modules by hand (e.g. to swap in a custom aberration
model) instead of going through `ImageGenerator`, see
`modular_pipeline/`.
