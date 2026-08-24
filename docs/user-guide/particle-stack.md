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

- **Structure & Potential**: `pdb_source` (fetched and cached under
  `~/.cache/specter/pdb/`), `assembly` (biological assembly vs. asymmetric unit),
  `n_pixels`/`pixel_size` for the simulation box.
- **Microscope**: `voltage`, `dose`, `cs`, `alpha` (amplitude contrast).
- **Sampling**: `defocus`, `shift` (max in-plane shift), `n_particles`.
- **Models**: `scattering_model` (`multislice` is the accurate default;
  `firstborn`/`projection` trade accuracy for speed), `detector_model`
  (`none` skips detector effects entirely; `k3_300kv`/`k3_200kv`/
  `falcon4i_*` apply a real detector's MTF/noise).
- **Post-processing**: `normalize_particles`, and `save_exitwaves` /
  `save_clean_exitwaves` if you want the pre-detector signal too (see
  below).

Everything else (ice model, crowding, aberration richness, potential
building method, and so on) lives under "Advanced" in both the TOML and
`specter simulate particles --help`, and is documented field-by-field on
[Configure a run](configuration.md).

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
flag can only carry one token: `--defocus 5000,15000`. That spelling is also
still accepted in a TOML file, so configs written before the numeric form
keep working.

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
instead of randomly-sampled poses. Pixel size, voltage, alpha, pose, and
CTF are read straight from the `.cs` file via `cs_path`, matching a real
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
code path from a RELION `.star` file: both the single-block layout specter
itself writes and the RELION 3.1+ two-block (`optics` + `particles`) layout
are read. The two flags are mutually exclusive.

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
comes from CryoSPARC itself: 2000 real EMPIAR-11377 particles and 2000
SPECTER particles (same `pdb_source`/physics parameters, generated in bulk
offline) were pooled into one stack and run through a single CryoSPARC 2D
Classification job. If the simulated particles are physically realistic,
CryoSPARC should sort them into the same classes as the real particles
they're mixed with, in roughly the same proportion per class. That is
what happens across nearly all 50 classes:

![Per-class particle counts from a CryoSPARC 2D Classification job run on a pooled real+simulated EMPIAR-11377 stack, split by source.](../assets/images/particle-stack-empiar-11377-2d-classes.png){ width="900" style="display:block;margin:1.2em auto;" }
///caption
Per-class particle counts from a CryoSPARC 2D Classification job run on a pooled real+simulated EMPIAR-11377 stack, split by source.
///

This second figure isn't reproducible from the repo alone: it requires
actually running CryoSPARC. But the per-particle class assignments behind
it are committed as
`docs-figures/data/empiar-11377-mixed-2dclasses.csv`. Both figures are
regenerated by
[`docs-figures/particle_stack_empiar_11377.py`](https://github.com/joelyeois/specter/blob/main/docs-figures/particle_stack_empiar_11377.py).
For the full 2000-particle `.cs`-driven workflow (not just the 5-row
demo slice), see [Generate a CryoSPARC dataset twin](dataset-twin.md).

## Inspecting the forward model (exit waves)

Set `save_exitwaves = true` (post-detector-free signal) or
`save_clean_exitwaves = true` (noiseless) to also write
`<filename>_exitwave.mrcs` / `<filename>_clean_exitwave.mrcs` alongside the
usual output. This is useful for debugging the forward model or comparing
signal before detector effects are applied, without re-running the whole
pipeline.

## Batch size

`batchsize` is how many particles go through the forward model at once. It
affects speed and peak memory only, never the images. The default is:

```toml
[compute]
batchsize = "auto"
```

which measures the memory actually free on `device` at run time and picks the
largest batch predicted to fit, given the FFT-padded volume the box size
implies (a 256-pixel box with `pad_fft = true` holds 512×512×256 volumes, not
256³). The chosen value is printed at the start of generation:

```
batchsize='auto' -> 3 particle(s) per pass (~31.7 GiB estimated peak, 43.1 GiB free on cuda:1)
```

Set an integer instead to pin it, worth doing when benchmarking, when
sharing a GPU with a job that will grow after specter has taken its reading,
or on CPU under a Slurm `--mem` limit (the CPU reading is the host's, not the
cgroup's). The estimate is deliberately conservative and is documented, with
its measured basis, in `specter.memory`.

## Multi-GPU

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
