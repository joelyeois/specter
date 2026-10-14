# Theory

This section is the appendix to the user guide: the physics and mathematics
behind SPECTER, one topic per page, with the derivations, conventions and
validation behind each model. The [walkthrough](../walkthrough/index.md) and
the [tutorials](../tutorials/index.md) introduce the same models through code
and link here for the detail, and every page here links back to where its
topic appears in code. For complete command-line workflows, see the
[Command-line guides](../user-guide/particle-stack.md).

- [Pipeline overview](pipeline-overview.md)
- [Conventions](conventions.md)
- Specimens
    - [Overview](specimens.md)
    - [Atomic potentials](atomic-potentials.md)
    - [Ice structure](ice.md)
    - Single-particle
        - [Pose & crowding](pose-crowding.md)
    - Cryo-ET
        - [Overview](cryoet-specimen/index.md)
        - Membrane shape
            - [Overview](membrane-shape/index.md)
            - [Spherical harmonics](membrane-shape/spherical-harmonics.md)
            - [Swept spline](membrane-shape/swept-spline.md)
        - [Bilayer & transmembrane proteins](cryoet-specimen/bilayer.md)
        - [Filaments](cryoet-specimen/filaments.md)
        - [Microtubules](cryoet-specimen/microtubules.md)
        - [Gold fiducial beads](cryoet-specimen/beads.md)
        - [Carbon support film](cryoet-specimen/carbon-film.md)
        - [Regions & protein packing](cryoet-specimen/packing.md)
- Forward simulation
    - [Overview](forward-simulation.md)
    - Scattering
        - [Overview](scattering/index.md)
        - [Multislice](scattering/multislice.md)
        - [Rytov](scattering/rytov.md)
        - [Other propagation modes](scattering/other-modes.md)
        - [Frozen-plasmon forward model](scattering/frozen-plasmon.md)
    - [Aberrations](aberrations.md)
    - [Detector](detector.md)
