# Overview

SPECTER simulates cryo-EM and cryo-ET images from first principles, and
reconstructs 3D maps from images by inverting the same simulation. It starts
from an atomic model, builds the specimen's electrostatic potential, embeds it
in amorphous ice, propagates an electron wave through it, applies the
microscope's optics, and records the result on a model of a real detector.
Each of those stages is a physical model with its own parameters, and each is
exposed as an object you can inspect.

![One particle followed through SPECTER's forward model.](assets/images/walkthrough/walkthrough-chain.png){ width="900" style="display:block;margin:1.2em auto;" }
///caption
One particle followed through the forward model: its rendered potential, the
crowded and solvated specimen, the phase of the exit wave, the wave after the
objective lens, and the detected image. The last two panels are low-pass
filtered to 5 Å for display.
///

## What SPECTER simulates

| Data | Specimen | Python | Command line |
|---|---|---|---|
| Single-particle stacks | One structure per image, with its own pose, neighbours and ice | [`simulate_particles`](walkthrough/particle-stacks.md) | `specter simulate particles` |
| Micrographs | A field of particles through a slab of ice, imaged as one exposure | [`Field`, `Sample`](walkthrough/micrographs.md) | `specter simulate micrograph` |
| Tilt series | A specimen volume tilted through a dose-accumulating series | [`TiltSeries`](walkthrough/tilt-series.md) | `specter simulate tiltseries` |
| Cellular specimens | Membranes, filaments, microtubules, packed proteins, gold beads, carbon film | [`specter.specimen`](api/specimen.md) | `specter build tomogram` |

In the inverse direction, `Ghostbuster` reconstructs a map from a particle
stack with the forward model above, and `specter match particles` derives a
simulation config whose images match a real dataset. See
[Reconstruction](walkthrough/reconstruction.md) and
[Match an experimental dataset](user-guide/dataset-twin.md).

## What the forward model includes

| Stage | Physics | Theory |
|---|---|---|
| Potential | Independent-atom potentials with bonded-species (Shtyrov) or per-element (Kirkland, Lobato) scattering factors, rendered with exact voxel averaging | [Atomic potentials](concepts/atomic-potentials.md) |
| Specimen | Crowding by neighbouring particles; amorphous ice optimised against a measured structure factor, displaced by the particle; radiation damage and solvent decorrelation under exposure | [Pose & crowding](concepts/pose-crowding.md), [Ice structure](concepts/ice.md) |
| Scattering | Multislice on an oversampled grid (default), Rytov, first Born and projection approximations; inelastic absorption by mean free path | [Scattering](concepts/scattering/index.md) |
| Optics | Defocus, astigmatism, spherical aberration, higher-order aberrations, phase plates; chromatic focal spread summed incoherently; spatial coherence | [Aberrations](concepts/aberrations.md) |
| Detector | Measured MTF and DQE(0) of real cameras, per-electron coincidence loss, shot noise | [Detector](concepts/detector.md) |

The same stages run in both directions. A reconstruction fits a map through
the simulator's own scattering and optics code, so a change to the physics
reaches both at once.

## Two ways to use it

**From Python**, the `specter.imaging` objects expose every stage. A short
particle stack:

```python
from specter.imaging import CTF, Exposure, Poses, Structure, simulate_particles

specimen = Structure.from_pdb("6bdf").render(voxel_size=1.0, box=256, device="cuda")
specimen = specimen.crowd(min_distance=190.0).solvate(model="gd", thickness=500.0)

images = simulate_particles(
    specimen,
    Poses.random(16, shift=2.0),
    CTF.sample(16, defocus=[5000.0, 15000.0]),
    Exposure(dose=40.0),
    voltage=300.0,
    seed=0,
)
images.save("particles.mrcs")
```

**From the command line**, the `specter` command runs the same simulations
from a TOML config, at scale and across GPUs, and writes RELION-compatible
output:

```bash
specter simulate particles --pdb_source 6bdf --n_particles 1000
```

The [walkthrough](walkthrough/index.md) teaches the Python objects one stage at
a time and ends with [the command line](walkthrough/command-line.md).

## How these docs are organised

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **Basic examples**

    Short, complete templates for the common simulations, to copy and
    modify.

    [:octicons-arrow-right-24: Basic examples](basic-examples.md)

-   :material-map-marker-path:{ .lg .middle } **Walkthrough**

    The forward model one stage per page, from an atomic model to a detected
    image and back to a map, with the physics each step needs.

    [:octicons-arrow-right-24: Start the walkthrough](walkthrough/index.md)

-   :material-school:{ .lg .middle } **Tutorials**

    Advanced topics once the walkthrough is done: real datasets, cellular
    specimens, radiation damage, inelastic scattering, gradients, custom
    forward models and ice libraries.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-console:{ .lg .middle } **Command-line guides**

    Every `specter` workflow from a TOML config: particle stacks,
    micrographs, tilt series, tomograms, matching, reconstruction and jobs.

    [:octicons-arrow-right-24: Command-line guides](user-guide/particle-stack.md)

-   :material-view-grid:{ .lg .middle } **Example gallery**

    Every example in these docs at a glance, by its figure.

    [:octicons-arrow-right-24: Gallery](gallery.md)

-   :material-function-variant:{ .lg .middle } **Theory**

    The derivations, conventions and validation behind each physical model.

    [:octicons-arrow-right-24: Theory](concepts/index.md)

-   :octicons-code-24:{ .lg .middle } **Reference**

    Every `specter` command and flag, and the Python API rendered from its
    docstrings.

    [:octicons-arrow-right-24: CLI](api/cli/index.md) ·
    [API](api/index.md)

</div>
