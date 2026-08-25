# SPECTER

<p align="center" markdown>
  ![SPECTER logo](assets/logo-light.png#only-light){ width="220" }
  ![SPECTER logo](assets/logo-dark.png#only-dark){ width="220" }
</p>

<strong>S</strong>cattering & <strong>P</strong>ropagation of <strong>E</strong>lectrons in
<strong>C</strong>ryo-EM: <strong>T</strong>win <strong>E</strong>mulator &
<strong>R</strong>econstruction. A differentiable digital twin for cryo-EM and
cryo-ET.

![A simulated cryo-ET specimen](assets/images/cryoet-tomogram-hero.png){ width="560" style="display:block;margin:1.2em auto;" }
///caption
A simulated cryo-ET specimen
///

---

## Why SPECTER?

<div class="grid cards" markdown>

-   :material-snowflake:{ .lg .middle } **Structured ice**

    Ice comes from configurations optimised against a measured structure
    factor and a coarse-grained water potential, so the pair correlations
    match vitreous ice instead of a random fill at the right density.

    [:octicons-arrow-right-24: Ice structure](concepts/ice.md)

-   :material-camera-iris:{ .lg .middle } **Per-electron detector**

    SPECTER places individual electrons and merges the ones that land too
    close together, reproducing the low-frequency suppression you see in
    real counting detectors.

    [:octicons-arrow-right-24: Detector](concepts/detector.md)

-   :material-check-decagram:{ .lg .middle } **Validated against experiment**

    Pool simulated particles with real EMPIAR-11377 particles and run one
    [CryoSPARC](https://cryosparc.com/) 2D classification job. They sort
    into the same classes, in similar proportion, across nearly all 50
    classes.

    [:octicons-arrow-right-24: See the comparison](user-guide/particle-stack.md#example-matching-empiar-11377)

-   :material-swap-horizontal:{ .lg .middle } **One model, both directions**

    The forward model that generates images also drives reconstruction.
    Change the physics once and both directions follow.

</div>

---

## Get started

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    Set up SPECTER with `uv` and confirm it works with a small CPU run.

    [:octicons-arrow-right-24: Install](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    Simulate a particle stack from a PDB code in one command.

    [:octicons-arrow-right-24: Get started](quickstart.md)

</div>

---

## Guides

### CLI

Every workflow below is a `specter` subcommand driven by a TOML config.

<div class="grid cards" markdown>

-   :material-grid:{ .lg .middle } **Particle stack**

    `specter simulate particles` builds a stack of single-particle images
    from a PDB code, sampling pose, defocus and dose per particle.

    [:octicons-arrow-right-24: Generate a particle stack](user-guide/particle-stack.md)

-   :material-image-filter-hdr:{ .lg .middle } **Micrograph**

    `specter simulate micrograph` fills a full field of view with
    particles, crowding and ice.

    [:octicons-arrow-right-24: Generate a micrograph](user-guide/micrograph.md)

-   :material-angle-acute:{ .lg .middle } **Tilt series**

    `specter simulate tiltseries` tilts through a specimen volume and
    accumulates dose across the series.

    [:octicons-arrow-right-24: Generate a tilt series](user-guide/tilt-series.md)

-   :material-cube-outline:{ .lg .middle } **Tomogram specimen**

    `specter build tomogram` packs membranes, filaments, microtubules,
    fiducial beads and a carbon film into one volume.

    [:octicons-arrow-right-24: Build a tomogram specimen](user-guide/build-tomogram.md)

-   :material-snowflake-variant:{ .lg .middle } **Ice cache**

    `specter build ice` builds a replacement `IceBank` library at a pixel
    size the bundled cache does not cover.

    [:octicons-arrow-right-24: Using the ice cache](user-guide/ice-cache.md)

-   :material-cube-send:{ .lg .middle } **Reconstruction**

    `specter reconstruct particle` fits the same forward model to a
    CryoSPARC particle stack and recovers a 3D volume.

    [:octicons-arrow-right-24: Reconstruct a volume](user-guide/reconstruction.md)

</div>

Override any config field on the command line; see
[Configure a run](user-guide/configuration.md). Have a run record itself as
a tracked job; see [Manage jobs](user-guide/jobs.md). SPECTER caches
structures you fetch by accession code and shares them across projects; see
[Manage the PDB cache](user-guide/cache.md).

### Concepts

<div class="grid cards" markdown>

-   :material-sitemap:{ .lg .middle } **Pipeline overview**

    Potential, specimen, scattering, aberration and detector compose into
    one forward model.

    [:octicons-arrow-right-24: Read more](concepts/pipeline-overview.md)

-   :material-molecule:{ .lg .middle } **Specimens**

    Atomic potentials, ice, crowding, and the cryo-ET specimen
    components.

    [:octicons-arrow-right-24: Read more](concepts/specimens.md)

-   :material-waves:{ .lg .middle } **Forward simulation**

    Propagation modes, aberrations and envelopes, and detector physics.

    [:octicons-arrow-right-24: Read more](concepts/forward-simulation.md)

-   :material-compass-outline:{ .lg .middle } **Conventions**

    Axis order, rotation sense, units and CTF sign conventions.

    [:octicons-arrow-right-24: Read more](concepts/conventions.md)

</div>

### Python API

<div class="grid cards" markdown>

-   :octicons-code-24:{ .lg .middle } **API overview**

    One page per subpackage, rendered from the docstrings.

    [:octicons-arrow-right-24: Read more](api/index.md)

-   :material-camera:{ .lg .middle } **Image generation**

    `ImageGenerator`, `MicrographGenerator`, `TiltSeriesGenerator`.

    [:octicons-arrow-right-24: specter.imagegenerator](api/imagegenerator.md)

-   :material-cube-scan:{ .lg .middle } **Specimens**

    Membranes, filaments, microtubules, beads and packing.

    [:octicons-arrow-right-24: specter.specimen](api/specimen.md)

-   :material-pipe:{ .lg .middle } **Pipelines**

    The end-to-end functions the CLI calls. Call them from Python too.

    [:octicons-arrow-right-24: specter.pipelines](api/pipelines.md)

</div>

---

## Getting help

Open an issue on the [GitHub repository](https://github.com/joelyeois/specter).
