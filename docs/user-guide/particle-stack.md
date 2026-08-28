# Generate a particle stack

`specter simulate particles` builds a physics-based cryo-EM particle stack
from a single PDB/mmCIF structure: it samples a scattering potential,
applies pose/CTF/dose sampling, propagates through the forward model
(multislice by default), and writes a `.mrcs` +
[RELION](https://relion.readthedocs.io/) `.star` file pair.
It's the fastest way to get simulated data out of SPECTER.

## Basic usage

Every run loads a TOML config first, then applies any flags given on the
command line as overrides:

```bash
specter simulate particles --config configs/particle.toml
```

```bash
specter simulate particles \
    --config configs/particle.toml \
    --pdb_source 6bdf \
    --n_particles 200 \
    --device cuda:0 \
    --output_dir particles
```

`configs/particle.toml` is the canonical starting point. Copy it and edit
for your own runs. See [Configure a run](configuration.md) for the full
TOML/CLI field reference.

## What you'll usually tune

- **Data source**: `cs_path` or `star_path`, to take poses and per-particle
  CTF from an existing CryoSPARC or RELION dataset instead of synthesizing
  them. Leave both unset for a synthetic stack.
- **Specimen**: `pdb_source` (fetched and cached under
  `~/.cache/specter/pdb/`), `assembly` (biological assembly vs. asymmetric unit),
  `n_pixels`/`pixel_size` for the simulation box, `ice_thickness`.
- **Microscope**: `voltage`, `dose`, `cs`, `alpha` (amplitude contrast).
- **Sampling**: `defocus`, `shift` (max in-plane shift), `n_particles`.
- **Models**: `scattering_model` (`multislice` is the accurate default;
  `firstborn`/`projection` trade accuracy for speed), `noise_model`,
  `detector_model` (`none` skips detector effects; `k3_300kv`/`k3_200kv`/
  `falcon4i_*` apply a real detector's MTF/noise).
- **Post-processing**: `normalize_particles`, and `save_exitwaves` /
  `save_clean_exitwaves` if you want the pre-detector signal too (see
  below).

Everything else (ice model, crowding, aberration richness, potential
building method, and so on) lives under "Advanced" in both the TOML and
`specter simulate particles --help`; [Configure a run](configuration.md)
documents it field-by-field.

## Per-particle sampling ranges

`dose`, `defocus`, `coincidence_radius`, `potential_scale`, `astigmatism`,
and `astigmatism_angle` each take either a single number (constant for every
particle) or a `[low, high]` pair, sampled uniformly per particle:

```toml
[sampling]
defocus = [5000.0, 15000.0]   # Å, sampled per particle
```

```toml
[microscope]
dose = 20.0               # e⁻/Å², constant for every particle
```

On the command line the same fields take a comma-separated string, since a
flag can only carry one token: `--defocus 5000,15000`. That spelling still
works in a TOML file too, so configs written before the numeric form keep
working.

## Output: .mrcs + .star

Results land in `output_dir` as `<filename>.mrcs` (the image stack) and
`<filename>.star` (a RELION-format star file with one row per particle):
pose (`rlnAngleRot`/`rlnAngleTilt`/`rlnAnglePsi`, `rlnOrigin{X,Y}Angst`),
CTF (`rlnVoltage`, `rlnSphericalAberration`, `rlnAmplitudeContrast`,
`rlnDefocus{U,V}`, `rlnDefocusAngle`, `rlnPhaseShift`), plus SPECTER-specific
physics parameters not covered by the RELION spec
(`specterDosePerAngstrom`, `specterCoincidenceRadius`,
`specterPotentialScale`). Import the `.star` file directly into RELION, or
read it with [`starfile`](https://github.com/teamtomo/starfile) /
[`mrcfile`](https://github.com/ccpem/mrcfile) from Python.

## Example: matching EMPIAR-11377

As a check against real data, the config below drives
`specter simulate particles` from a real [CryoSPARC](https://cryosparc.com/)
passthrough `.cs` file
instead of randomly-sampled poses. `cs_path` reads pixel size, voltage,
alpha, pose, and CTF straight from the `.cs` file, matching a real
experimental dataset particle-for-particle:

```bash
specter simulate particles \
    --pdb_source 8b0x \
    --n_pixels 512 \
    --cs_path docs-figures/data/empiar-11377-passthrough-5particles.cs \
    --n_particles 5 \
    --dose 40 \
    --ice_model gd \
    --coincidence_radius 0.8 \
    --potential_scale 0.5 \
    --detector_model none \
    --device cuda:0
```

Pass `--star_path particles.star` instead of `--cs_path` to drive the same
code path from a RELION `.star` file: SPECTER reads both the single-block
layout it writes itself and the RELION 3.1+ two-block (`optics` +
`particles`) layout. The two flags are mutually exclusive.

`docs-figures/data/empiar-11377-passthrough-5particles.cs` is a 5-row slice
of a real passthrough `.cs` file from
[EMPIAR-11377](https://www.ebi.ac.uk/empiar/EMPIAR-11377/), a translating
70S ribosome dataset (PDB [8b0x](https://www.rcsb.org/structure/8B0X)),
committed to the repo so this exact command is reproducible without access
to the original dataset. Running it and comparing the output against the 5
matching real particle images
(`docs-figures/data/empiar-11377-real-5particles.mrcs`) gives:

![Five SPECTER-simulated particles (top row) next to the five real EMPIAR-11377 particles their poses/CTF were taken from (bottom row).](../assets/images/particle-stack-empiar-11377-comparison.png){ width="900" style="display:block;margin:1.2em auto;" }
///caption
Five SPECTER-simulated particles (top row) next to the five real EMPIAR-11377 particles their poses/CTF were taken from (bottom row).
///

That's a qualitative check on 5 particles. The stronger, quantitative check
comes from CryoSPARC itself. Pool 2000 real EMPIAR-11377 particles with 2000
SPECTER particles (same `pdb_source`/physics parameters, generated in bulk
offline) into one stack, then run that stack through a single CryoSPARC 2D
Classification job. If the simulated particles are physically realistic,
CryoSPARC should sort them into the same classes as the real particles
they're mixed with, in roughly the same proportion per class. That's what
happens across nearly all 50 classes:

![Per-class particle counts from a CryoSPARC 2D Classification job run on a pooled real+simulated EMPIAR-11377 stack, split by source.](../assets/images/particle-stack-empiar-11377-2d-classes.png){ width="900" style="display:block;margin:1.2em auto;" }
///caption
Per-class particle counts from a CryoSPARC 2D Classification job run on a pooled real+simulated EMPIAR-11377 stack, split by source.
///

This second figure isn't reproducible from the repo alone: it requires
running CryoSPARC. But `docs-figures/data/empiar-11377-mixed-2dclasses.csv`
holds the per-particle class assignments behind it, committed to the repo.
[`docs-figures/particle_stack_empiar_11377.py`](https://github.com/joelyeois/specter/blob/main/docs-figures/particle_stack_empiar_11377.py)
regenerates both figures. For the full 2000-particle `.cs`-driven workflow
(not just the 5-row demo slice), see [Generate a CryoSPARC dataset
twin](dataset-twin.md).

## Inspecting the forward model (exit waves)

Set `save_exitwaves = true` (post-detector-free signal) or
`save_clean_exitwaves = true` (noiseless) to also write
`<filename>_exitwave.mrcs` / `<filename>_clean_exitwave.mrcs` alongside the
usual output. This is useful for debugging the forward model or comparing
signal before `Detector` applies its effects, without re-running the whole
pipeline.

## Batch size

`batchsize` is how many particles go through the forward model at once. It
affects speed and peak memory only, never the images. The default is:

```toml
[compute]
batchsize = "auto"
```

which takes the smaller of two bounds. The first is what fits: the memory
free on `device` at run time, against the FFT-padded volume the box size
implies (a 256-pixel box with `pad_fft = true` holds 512×512×256 volumes, not
256³). The second is what is worth batching. Batching amortises the fixed cost
of a forward pass, but only until a single pass already saturates the device;
beyond that point a larger batch consumes memory in proportion to its size
without reducing the time per particle.

Which bound binds depends on the box. At a 256-pixel box one particle covers
67 million padded voxels and saturates a current GPU on its own, so `"auto"`
resolves to 2. At a 64-pixel box a particle covers 1 million, and batching is
worth roughly a factor of two. The CLI prints the chosen value at the start of
generation:

```
batchsize='auto' -> 2 particle(s) per pass (~3.2 GiB estimated peak, 42.5 GiB free on cuda:0)
```

The estimated peak there is a small fraction of what is free, which is the
saturation bound binding rather than the memory one: the run is not short of
memory, a larger batch simply would not go faster.

Set an integer instead to pin it, worth doing when benchmarking, when
sharing a GPU with a job that will grow after SPECTER has taken its reading,
or on CPU under a Slurm `--mem` limit (the CPU reading is the host's, not the
cgroup's). The estimate stays deliberately conservative; `specter.memory`
documents both bounds along with their measured basis.

## Multi-GPU

The default is `device = "cuda"`, which uses the GPU when there is one and
falls back to the CPU, with a warning, when there is not. An explicit index is
taken literally: `"cuda:0"` or `"0,1"` names particular hardware and fails
rather than running somewhere else.

`device` accepts a comma-separated list of GPU indices (e.g. `"0,1,2,3"`) to
split particle generation across multiple GPUs via Lightning's DDP
launcher; a single index (`"cuda:0"`) or `"cpu"` runs on one device. With
`batchsize = "auto"`, every rank builds the same-sized batch, sized to
whichever GPU in the pool has the least free memory.

## Using it from Python instead of the CLI

`run_particle_stack(config)` (`specter.pipelines`) is the same function the
CLI calls. Build a `ParticleStackConfig` directly in Python (or load one
with `specter.config.load_config`) instead of going through the command
line. For composing the forward model's individual modules by hand (e.g. to
swap in a custom aberration model), see
[`demo-notebooks/create_particle_stack_modular/`](https://github.com/joelyeois/specter/tree/main/demo-notebooks/create_particle_stack_modular).
For the plain end-to-end notebook version of this page, see
[`demo-notebooks/create_particle_stack/`](https://github.com/joelyeois/specter/tree/main/demo-notebooks/create_particle_stack).

## See also

- [Generate a CryoSPARC dataset twin](dataset-twin.md): the full
  `.cs`-driven workflow this page's example is a slice of.
- [Using the ice cache](ice-cache.md): `ice_model` options and `IceBank`.
- [Configure a run](configuration.md): complete TOML/CLI field reference.
- [Manage jobs](jobs.md): recording runs under a project name.
- [Pipeline overview](../concepts/pipeline-overview.md): how this fits
  into the rest of the forward model.
