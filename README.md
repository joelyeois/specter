<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.png">
    <img src="docs/assets/logo-light.png" alt="SPECTER logo" width="220">
  </picture>
</p>

<h1 align="center">SPECTER</h1>
<p align="center"><strong>S</strong>cattering &amp; <strong>P</strong>ropagation of <strong>E</strong>lectrons in Cryo-EM: <strong>T</strong>win <strong>E</strong>mulator &amp; <strong>R</strong>econstruction</p>

<p align="center">
  <a href="https://joelyeois.github.io/specter/"><img src="https://github.com/joelyeois/specter/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-All%20Rights%20Reserved-lightgrey.svg" alt="License">
</p>

<p align="center">
A physics-based simulator for cryo-EM and cryo-ET, built to produce images
that match experimental data as closely as possible — and, through
Ghostbuster, the same forward model run in reverse to reconstruct a 3D map
from images.
</p>

<p align="center">
  <img src="docs/assets/images/cryoet-tomogram-hero.png" alt="A simulated cryo-ET specimen" width="560">
</p>

> **Early development notice**  
> SPECTER is under active development. APIs and behaviour may change without notice between releases — including breaking changes. It is not yet recommended for production workflows. Feedback and bug reports are welcome via [GitHub Issues](https://github.com/joelyeois/specter/issues).

---

## Table of contents

- [Why SPECTER?](#why-specter)
- [Installation](#installation)
- [Usage](#usage)
- [CLI at a glance](#cli-at-a-glance)
- [Documentation](#documentation)
- [Repository layout](#repository-layout)
- [FAQ](#faq)
- [License](#license)
- [Getting help](#getting-help)

---

## Why SPECTER?

- **Structured ice.** Ice comes from configurations optimised against a measured structure factor and a coarse-grained water potential, not from water molecules placed at random to the right bulk density.
- **Per-electron detector.** Individual electrons are placed and merged when they land too close together, reproducing the low-frequency suppression that real counting detectors show.
- **Validated against experiment.** Simulated particles pooled with real EMPIAR-11377 particles and run through a single CryoSPARC 2D classification job sort into the same classes, in roughly the same proportion, across nearly all 50 classes — see the [particle-stack guide](https://joelyeois.github.io/specter/user-guide/particle-stack/#example-matching-empiar-11377).
- **One model, both directions.** The forward model that generates images (`ImageGenerator`/`MicrographGenerator`/`TiltSeriesGenerator`) is the same model that drives reconstruction (`Ghostbuster`/`TomogramGhostbuster`), so a change to the physics applies in both directions rather than to a simulator alone.
- **GPU-accelerated.** Volume rotation, potential calculation, and wave propagation all run on PyTorch, with multi-GPU dispatch for particle stacks, tilt series, and ice cache generation.
- **Tracked runs.** Every simulation and reconstruction is recorded as a numbered job (`specter jobs list/show/diff`), so parameters and provenance are never lost between experiments.

---

## Installation

```bash
git clone https://github.com/joelyeois/specter.git
cd specter
uv sync
source .venv/bin/activate
uv pip install -e .          # the package itself, not just its dependencies
uv run --with jupyter jupyter lab
```

The default `shtyrov` scattering-factor parameterization types atoms by their
bonded neighbours, so it needs hydrogens that deposited structures do not
carry. Install the
[Monomer Library](https://github.com/MonomerLibrary/monomers) and point
`$CLIBD_MON` at it to supply them — without it roughly 44% of a protein's
atoms fall back to per-element factors:

```bash
git clone https://github.com/MonomerLibrary/monomers.git
export CLIBD_MON=/path/to/monomers
```

See the [installation guide](https://joelyeois.github.io/specter/installation/)
for the conda/pip alternative, GPU notes, the Monomer Library in full, and
troubleshooting an install outside a git checkout.

---

## Usage

The `specter simulate particles` CLI is the quickest way to produce a
simulated cryo-EM particle stack. Every setting carries a built-in default,
so the only thing you must supply is the structure to simulate:

```bash
specter simulate particles --pdb_source 6bdf
```

This downloads the structure, builds the scattering potential, applies CTF
and detector effects, and writes a `.mrcs` / `.star` file pair. A TOML
config carries the settings once there are more of them than fit on a line,
and any field in it can still be overridden on the command line:

```bash
specter simulate particles \
    --config particle.toml \
    --pdb_source 6bdf \
    --n_particles 200 \
    --device cuda:0 \
    --output_dir particles
```

**The example configs are not installed with the package.** They live in this
repository, one per subcommand, under
[`configs/`](https://github.com/joelyeois/specter/tree/main/configs). Clone
the repository, or download the one you want:

```bash
curl -O https://raw.githubusercontent.com/joelyeois/specter/main/configs/particle.toml
```

Keep the config and the installed version in step: field names change between
releases, and a config naming a field that no longer exists is rejected by
name rather than ignored.

<details>
<summary>What <code>configs/particle.toml</code> looks like</summary>

```toml
# Canonical default config for `specter simulate particles`.
# Any field can be overridden on the command line, e.g.:
#   specter simulate particles --config particle.toml --n_particles 3000

[potential]
pdb_source = "6bdf"
assembly = true
n_pixels = 256
pixel_size = 1.0              # Å

[microscope]
voltage = 300.0                # kV
dose = 20.0                    # e⁻/Å²; single value, or [low, high] to sample per particle
cs = 2.0                       # mm
alpha = 0.1                    # unitless, amplitude contrast ratio

# ... see the full file: configs/particle.toml
```

</details>

Full usage documentation — the `specter` CLI reference, the TOML
config system, the physics pipeline, ice generation, Ghostbuster
reconstruction, and job management — lives in the docs:

**[joelyeois.github.io/specter](https://joelyeois.github.io/specter/)**

Simulation is driven by the `specter` command (`specter simulate particles`,
`micrograph`, `tiltseries`, `specter build tomogram`, `specter build ice`),
and reconstruction by `specter reconstruct particle`, also spelled
`specter ghostbuster particle`. Interactive notebooks are in
`demo-notebooks/`.

---

## CLI at a glance

| Command | What it does |
|---|---|
| `specter simulate particles` | Single-particle image stack from a PDB code, or a "twin" of a real CryoSPARC/RELION dataset via `--cs_path`/`--star_path`. |
| `specter simulate micrograph` | A full field of view: many particles, crowding, and ice. |
| `specter build tomogram` | Composite a specimen volume from membranes, filaments, microtubules, beads, and packed protein species. |
| `specter simulate tiltseries` | A cryo-ET tilt series through a tomogram specimen volume, with dose accumulation across tilts. |
| `specter build ice` | A replacement `IceBank` library at a pixel size the bundled cache doesn't cover. |
| `specter reconstruct particle` (alias: `specter ghostbuster particle`) | Reconstruct a 3D map from a particle stack, jointly refining pose, translation, and defocus. |
| `specter jobs list/show/diff` | Inspect and compare parameters and provenance across past runs. |

Every subcommand takes an optional `--config path/to.toml`; any field in that
TOML can also be set directly on the command line, and any field left unset
takes its built-in default. See
[Configure a run](https://joelyeois.github.io/specter/user-guide/configuration/).

---

## Documentation

| | |
|---|---|
| [Installation](https://joelyeois.github.io/specter/installation/) | Set up SPECTER with `uv` and confirm it works with a small CPU run. |
| [Quickstart](https://joelyeois.github.io/specter/quickstart/) | Simulate a particle stack from a PDB code in one command. |
| [User guide](https://joelyeois.github.io/specter/user-guide/particle-stack/) | Particle stacks, micrographs, tilt series, tomogram specimens, ice caches, configuration, and job management. |
| [Concepts](https://joelyeois.github.io/specter/concepts/pipeline-overview/) | How potential, specimen, scattering, aberration and detector compose into one forward model. |
| [Python API](https://joelyeois.github.io/specter/api/) | Generated from docstrings, one page per subpackage. |

---

## Repository layout

```
src/specter/        # main package: physics simulator + Ghostbuster reconstruction
  imagegenerator/    # ImageGenerator, MicrographGenerator, TiltSeriesGenerator
  ghostbuster/       # Reconstructor, Ghostbuster, tomogram reconstruction
  specimen/          # membranes, filaments, microtubules, crowding, ice
  cli/               # the `specter` command
  pipelines/         # end-to-end functions behind each CLI subcommand
demo-notebooks/      # interactive, always-working usage examples
configs/             # worked example TOML configs (not shipped in the wheel)
docs/                # Concepts, user guide, and API reference (Read the Docs)
tests/               # pytest suite
```

---

## FAQ

**Do I need a GPU?**
No. Every simulator and Ghostbuster component runs on CPU as well as GPU
via PyTorch/Lightning; a CPU-only install path is documented in the
[installation guide](https://joelyeois.github.io/specter/installation/).
A GPU accelerates volume rotation, potential calculation, and wave
propagation, and multiple GPUs can be used at once for particle stacks,
tilt series, and ice cache generation.

**Does this replace RELION or CryoSPARC?**
No. SPECTER can read a real CryoSPARC `.cs` or RELION `.star` file and use
its pixel size, poses, and CTF to simulate a physically matched "twin" of
that dataset (`--cs_path`/`--star_path`), rather than sample those
parameters synthetically. Ghostbuster, its reconstruction side, is a
research-stage forward-model-based reconstructor, not a production
alternative to those packages — see [Why SPECTER?](#why-specter) above.

**How do I know the simulated images are realistic?**
Simulated particles pooled with real EMPIAR-11377 particles and passed
through one CryoSPARC 2D classification job sort into the same classes, in
roughly the same proportion, across nearly all 50 classes; see the
[worked example](https://joelyeois.github.io/specter/user-guide/particle-stack/#example-matching-empiar-11377).

**Which structures can I simulate?**
Any PDB/mmCIF entry, referenced by its 4-character PDB code — SPECTER
fetches and caches it automatically. Scattering-factor typing is most
accurate on structures with resolvable hydrogens; see the
[Monomer Library note](#installation) above.

---

## License

This source is made publicly viewable for reference purposes only; see
[`LICENSE`](LICENSE) for the exact terms. It is expected to move to an
open-source license (anticipated: BSD-3-Clause) once the associated paper
is published.

---

## Getting help

Questions, bug reports, and feature requests are welcome on
[GitHub Issues](https://github.com/joelyeois/specter/issues).
