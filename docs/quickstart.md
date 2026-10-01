# Quickstart

## Simulate a particle stack

The `specter simulate particles` CLI is the quickest way to produce a
simulated cryo-EM particle stack. Every setting carries a built-in default,
so the only thing you must supply is the structure to simulate:

```bash
specter simulate particles --pdb_source 6bdf
```

This downloads the structure, builds the scattering potential, applies CTF
and detector effects, and writes a `.mrcs` / `.star` file pair.

### Where the example configs live

Once a run has more than a few settings, a TOML config is easier to keep
than a line of flags:

```bash
specter simulate particles --config particle.toml
```

`--config` is optional. Given one, SPECTER loads its values first, and any
flag passed alongside it overrides a single field. Given none, every field
takes its built-in default, and only a field that has no default (here,
`--pdb_source`) has to be supplied.

**The example configs are not installed with the package.** They live in the
repository, one per subcommand, at
[`configs/`](https://github.com/joelyeois/specter/tree/main/configs).
Download the one you want:

```bash
curl -O https://raw.githubusercontent.com/joelyeois/specter/main/configs/particle.toml
```

If you cloned the repository rather than installing from a wheel, they are
already in `configs/`.

Keep the config and the installed version in step. Field names change
between releases, and a config naming a field that no longer exists is
rejected by name rather than ignored.

`configs/particle.toml` looks like this:

```toml
# Canonical default config for `specter simulate particles`.
# Any field can be overridden on the command line, e.g.:
#   specter simulate particles --config configs/particle.toml --n_particles 3000

# Where the particles come from. Left unset, the run synthesizes them and
# everything below applies as written; given a file, the poses and
# per-particle CTF come from it instead.
[data]
# cs_path = "path/to/particles.cs"      # a CryoSPARC .cs file
# star_path = "path/to/particles.star"  # ... or a RELION .star file (mutually exclusive)

[specimen]
pdb_source = "6bdf"
assembly = true
n_pixels = 256
pixel_size = 1.0              # Å
ice_thickness = 0.0           # Å, 0 = minimum (particle box size)

[microscope]
voltage = 300.0                # kV
dose = 20.0                    # e⁻/Å²; single value, or [low, high] to sample per particle
cs = 2.0                       # mm
alpha = 0.1                    # unitless, amplitude contrast ratio

# ... see the full file: configs/particle.toml
```

You can override any field in the config from the command line without
editing the file:

```bash
specter simulate particles \
    --config particle.toml \
    --pdb_source 6bdf \
    --n_particles 200 \
    --device cuda:0 \
    --output_dir particles
```

See [Configure a run](user-guide/configuration.md) for the complete field
reference. For the other three workflows (a
[CryoSPARC](https://cryosparc.com/) dataset twin, full micrographs,
cryo-ET tilt series), see
[Generate a particle stack](user-guide/particle-stack.md),
[Generate a micrograph](user-guide/micrograph.md), and
[Generate a tilt series](user-guide/tilt-series.md).

## Build a tomogram specimen

The `specter build tomogram` CLI composites a specimen volume from any
combination of: one or more organic membranes, filament species (e.g.
F-actin), and densely packed protein species (region-gated to
cytosol/lumen when a membrane is present). It saves a `.mrc` volume plus
copick-style `.ndjson` ground-truth picks and, by default, segmentation
label volumes:

```bash
specter build tomogram --config tomogram.toml
```

Unlike the particle stack above, this one needs its config. The
specimen contents (`[targets]`, `[filler]`, `[[membrane]]`, filaments,
microtubules, beads) default to empty, so `specter build tomogram` with no
config renders an empty box. Download
[`configs/tomogram.toml`](https://github.com/joelyeois/specter/tree/main/configs)
for a worked scene to edit:

```bash
curl -O https://raw.githubusercontent.com/joelyeois/specter/main/configs/tomogram.toml
```

SPECTER places protein species in two priority stages within their
region: `[targets]` first, each at an exact instance count (the annotated
ground truth, always exported to picks), then `[filler]` second, packed
around the already-placed targets to crowd out the rest of that region
(excluded from picks by default). Filler draws from the species listed by
hand plus any bundled reference table switched on, such as
`filler_from_pei2016`, and `filler_occupancy_fraction` caps how much of
each region it may fill; packing stops earlier if it jams. Generation
order runs membranes, then filaments, then this protein fill; each stage
avoids the previous ones' placements. `configs/tomogram.toml` looks like this:

```toml
# Canonical default config for `specter build tomogram`.
# Any field can be overridden on the command line, e.g.:
#   specter build tomogram --config configs/tomogram.toml --n_tomograms 4

[specimen]
target_shape = [300, 1200, 1200]    # (Z, Y, X) voxels
voxel_size = 5.0                    # A/voxel
clip_axes = [true, true, true]      # (z, y, x) -- True lets instances poke past that wall
filler_occupancy_fraction = 1.0     # a ceiling, not a target: packing stops when it jams

# Target species: exact-count, placed first, written to picks as ground truth.
[targets]
targets = [
    { pdb_source = "7VD8", n_copies = 15 },  # apoferritin
    { pdb_source = "1FA2", n_copies = 15 },  # beta-amylase
    { pdb_source = "6DRV", n_copies = 15 },  # beta-galactosidase
    { pdb_source = "6QZP", n_copies = 15 },  # ribosome
    { pdb_source = "7B75", n_copies = 15 },  # thyroglobulin
    { pdb_source = "6N4V", n_copies = 15 },  # virus-like particle
]

# Filler species: ratio-weighted background crowding, placed after targets,
# around them. Hand-listed species and the bundled reference tables are
# additive.
[filler]
filler = [
    { pdb_source = "1fa2", location = "lumen" },  # beta amylase, as an example lumen filler
]

# 20 generic cytosolic proteins, Pei et al. 2016, BMC Bioinformatics 17:405,
# at equal ratios.
filler_from_pei2016 = true

[[membrane]]
shape_backend = "spherical_harmonics"
bilayer_thickness = 38.0
sh_axes_range = [150.0, 500.0]      # SEMI-axes, A
n_copies = 3

# ... see the full file: configs/tomogram.toml
```

The resulting `.mrc` is directly usable as `specter simulate tiltseries`'s
`--volume_path`. See [Build a tomogram specimen](user-guide/build-tomogram.md)
for the full flag reference, including placement priority, region gating,
and the compute/scaling flags for larger runs.

## Reconstruct a volume

`specter reconstruct particle` runs the inverse problem: it fits the same
forward model used above to a real CryoSPARC particle stack and recovers
a 3D volume. Unlike the config above, `configs/reconstruct.toml` has no
runnable default; point it at a real `.cs` file and particle stack first:

```bash
specter reconstruct particle --config reconstruct.toml --test_run
```

`--test_run` fits one epoch on binned images so a config mistake surfaces
in seconds rather than after a multi-hour run. See
[Reconstruct a volume](user-guide/reconstruction.md) for the gold-standard
workflow and the full flag reference.

## Job management

You can record generation and reconstruction runs under a project name in
a local job database. Inspect past runs with `specter jobs`; see
[Manage jobs](user-guide/jobs.md):

```bash
specter jobs list --project my-project
specter jobs show <job_id>
specter jobs diff <job_id_1> <job_id_2>
```

SPECTER caches structures fetched by accession code
(`pdb_source = "6bdf"`, above) at `~/.cache/specter/pdb` and shares them
across every project on the machine. `specter cache info` reports what is
cached, and `specter cache clean` clears it; see
[Manage the PDB cache](user-guide/cache.md).

See `demo-notebooks/` for interactive worked examples, including
micrograph and tilt-series generation (`create_micrograph/`,
`create_tilt_series/`). For an example of composing the forward model's
individual modules by hand (e.g. to swap in a custom aberration model)
instead of going through `ImageGenerator`, see
`create_particle_stack_modular/` and `create_tilt_series_modular/`.
