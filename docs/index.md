# SPECTER

<p align="center" markdown>
  ![SPECTER logo](assets/logo-light.png#only-light){ width="220" }
  ![SPECTER logo](assets/logo-dark.png#only-dark){ width="220" }
</p>

**Scattering & Propagation of Electrons in Cryo-EM: Twin Emulator & Reconstruction**

Physics-based cryo-EM simulation, built to match experimental data as
closely as possible.

## Get started

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Set up SPECTER with `uv` and confirm it works with a small CPU run.

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    Simulate a particle stack from a PDB code in one command.

    [:octicons-arrow-right-24: Quickstart](quickstart.md)

</div>

## What makes this package different

These are the properties that follow from reading the source, not a
competitive comparison. Each is unusual on its own; together they define
what SPECTER is for.

<div class="grid cards" markdown>

-   :material-snowflake:{ .lg .middle } **Ice**

    ---

    Water is structured, not scattered. Ice comes from configurations
    optimised against a measured structure factor and a coarse-grained
    water potential, not randomly placed molecules at the right bulk
    density. See [Ice structure](concepts/ice.md).

-   :material-camera-iris:{ .lg .middle } **Detector**

    ---

    Coincidence loss is simulated per electron. Individual electrons are
    placed and then merged when they land too close together,
    reproducing the low-frequency suppression that real counting
    detectors show.

-   :material-content-duplicate:{ .lg .middle } **Dataset twins**

    ---

    Poses can come from a real experiment. Reading a CryoSPARC `.cs`
    file gives simulated data with the same poses, defoci and optics as
    a real dataset, with ground truth attached throughout.

-   :material-wave:{ .lg .middle } **Propagation**

    ---

    Multislice, not just a CTF multiply. The default propagates the
    wave slice by slice through the specimen, so thickness and multiple
    scattering are represented rather than assumed away.

-   :material-history:{ .lg .middle } **Provenance**

    ---

    Runs record themselves. Every job stores its complete effective
    configuration, the package version and the git commit, and refuses
    to resume under changed settings. See [Manage jobs](user-guide/jobs.md).

</div>
