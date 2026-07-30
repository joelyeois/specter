# SPECTER

<p align="center" markdown>
  ![SPECTER logo](assets/logo-light.png#only-light){ width="220" }
  ![SPECTER logo](assets/logo-dark.png#only-dark){ width="220" }
</p>

**Scattering & Propagation of Electrons in Cryo-EM: Twin Emulator & Reconstruction**

A microscope you can run backwards. SPECTER simulates what a cryo-electron
microscope would record from a known molecule, and then uses that same
simulation in reverse to recover a molecule from real images. Both directions
run on one shared piece of physics — the forward model is differentiable, so
the same code that generates an image can have gradients pushed back through
it to recover the structure that produced one. See [Pipeline](pipeline.md)
for how that works end to end, and [Physics](physics/index.md) for the
underlying equations.

## If you are new to cryo-EM

A cryo-electron microscope fires electrons through a thin film of frozen
water holding copies of a protein, each frozen in a random orientation. The
resulting images are extremely noisy — the dose has to stay low or the
electrons destroy the very thing you are imaging. Recovering a 3D structure
means combining hundreds of thousands of these faint, randomly-oriented
shadows. SPECTER can both **fake** that data convincingly and **solve** it.

## If you work in the field

Kirkland-parameterised atomic potentials on a soft-voxelised grid; full
multislice propagation with relativistic wavelength and energy-dependent
interaction parameter; a complex transfer function with spatial- and
temporal-coherence envelopes; per-electron detector modelling including
coincidence loss; and structurally optimised amorphous ice matched to a
target S(k) and an MLBOP water potential.

## Get started

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Set up SPECTER with `uv` and confirm it works with a small CPU run.

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    Simulate a particle stack from a PDB code, or reconstruct a volume
    with Ghostbuster, in one command.

    [:octicons-arrow-right-24: Quickstart](quickstart.md)

</div>

## What makes this package different

These are the properties that follow from reading the source, not a
competitive comparison. Each is unusual on its own; together they define
what SPECTER is for.

<div class="grid cards" markdown>

-   :material-sync:{ .lg .middle } **Shared model**

    ---

    Simulation and reconstruction are the same code. Ghostbuster
    instantiates the ordinary `ImageGenerator` and optimises through it —
    there is no separate, simplified reconstruction physics to drift out
    of sync.

-   :material-snowflake:{ .lg .middle } **Ice**

    ---

    Water is structured, not scattered. Ice comes from configurations
    optimised against a measured structure factor and a coarse-grained
    water potential, not randomly placed molecules at the right bulk
    density. See [Ice](ice.md).

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
    a real dataset — with ground truth attached.

-   :material-wave:{ .lg .middle } **Propagation**

    ---

    Multislice, not just a CTF multiply. The default propagates the
    wave slice by slice through the specimen, so thickness and multiple
    scattering are represented rather than assumed away.

-   :material-history:{ .lg .middle } **Provenance**

    ---

    Runs record themselves. Every job stores its complete effective
    configuration, the package version and the git commit, and refuses
    to resume under changed settings. See [Job management](jobs.md).

</div>
