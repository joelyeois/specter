# Concepts

This section lays out the physics and math behind SPECTER, one topic per
page: the same role PyTorch's "Developer Notes" or NumPy's conceptual
guide play for those projects. For task-oriented, CLI-driven how-tos
instead, see the [User Guide](../user-guide/particle-stack.md).

- [Pipeline overview](pipeline-overview.md)
- [Conventions](conventions.md)
- Specimens
    - [Overview](specimens.md)
    - [Atomic potentials](atomic-potentials.md)
    - [Ice structure](ice.md) *(work in progress)*
    - Single-particle
        - [Pose & crowding](pose-crowding.md) *(work in progress)*
    - Cryo-ET
        - [Overview](cryoet-specimen/index.md)
        - Membrane shape
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
    - [Aberrations](aberrations.md)
    - [Detector](detector.md)
- [Reconstruction math](reconstruction-math.md) *(work in progress)*
