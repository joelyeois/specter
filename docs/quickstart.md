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
dose = 20.0                    # e⁻/Å²; single value, or [low, high] to sample per particle
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
    --output_dir specter-data/particles
```

See [Configure a run](user-guide/configuration.md) for the complete field
reference. For the other three workflows (a CryoSPARC dataset twin, full
micrographs, cryo-ET tilt series), see
[Generate a particle stack](user-guide/particle-stack.md),
[Generate a micrograph](user-guide/micrograph.md), and
[Generate a tilt series](user-guide/tilt-series.md).

## Build a tomogram specimen

The `specter build tomogram` CLI composites a specimen volume from any
combination of: one or more organic membranes, filament species (e.g.
F-actin), and densely packed protein species (region-gated to
cytosol/lumen when a membrane is present) — saving a `.mrc` volume plus
copick-style `.ndjson` ground-truth picks and, by default, segmentation
label volumes:

```bash
specter build tomogram --config configs/tomogram.toml
```

Protein species are placed in two priority stages within their region:
`[targets]` first, each at an exact instance count (the annotated ground
truth, always exported to picks), then `[filler]` second, packed around
the already-placed targets to crowd out the rest of that region (excluded
from picks by default). Generation order overall is membranes, then
filaments, then this protein fill — each stage avoids the previous ones'
placements. An example config lives in
[`configs/tomogram.toml`](https://github.com/joelyeois/specter/tree/main/configs) —
copy it and edit for your own runs. It looks like this:

```toml
# Canonical default config for `specter build tomogram`.

[targets]
targets = [
    { pdb_source = "1bxn", n_copies = 20 },  # cytosolic RNA polymerase II complex (large)
]

[filler]
filler = [
    { pdb_source = "1mbo" },  # myoglobin (small)
]

[[membrane]]
shape_backend = "spherical_harmonics"   # omit [[membrane]] entirely for no membranes

[specimen]
target_shape = [128, 256, 256]    # (Z, Y, X) voxels
v_size = 5.0                       # Å/voxel
filler_occupancy_fraction = 0.5    # bare-sphere volume fraction budget for filler, per region

# ... see the full file: configs/tomogram.toml
```

The resulting `.mrc` is directly usable as `specter simulate tiltseries`'s
`--volume_path`. See [Build a tomogram specimen](user-guide/build-tomogram.md)
for the full flag reference, including placement priority, region gating,
and the compute/scaling flags for larger runs.

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
