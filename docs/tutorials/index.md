# Tutorials

These tutorials cover topics beyond the [walkthrough](../walkthrough/index.md),
which they assume you have read. Each is a self-contained page of code and
figures, built from a script in
[`docs-figures/tutorials/`](https://github.com/joelyeois/specter/tree/main/docs-figures/tutorials)
that you can run yourself with `uv run python docs-figures/tutorials/<name>.py`.
Read them in any order.

## In depth

<div class="grid cards" markdown>

-   [![Simulating a real dataset](../assets/images/tutorials/real_dataset-thumbnail.png)](real-dataset.md)

    **[Simulating a real dataset](real-dataset.md)**

    Image a structure at the poses and transfer functions of a CryoSPARC or
    RELION refinement, and compare it with the experimental particles.

-   [![Cellular specimens for cryo-ET](../assets/images/tutorials/cellular_specimens-thumbnail.png)](cellular-specimens.md)

    **[Cellular specimens for cryo-ET](cellular-specimens.md)**

    Build a crowded cellular volume from membranes, filaments, packed proteins,
    gold beads and carbon, with its ground truth, and tilt it.

-   [![Radiation damage and the solvent](../assets/images/tutorials/dose_damage-thumbnail.png)](dose-damage.md)

    **[Radiation damage and the solvent](dose-damage.md)**

    Fade the specimen with the electron dose while the ice keeps its
    structure, and let the solvent decorrelate under the exposure.

-   [![Inelastic scattering and energy filtering](../assets/images/tutorials/inelastic-thumbnail.png)](inelastic.md)

    **[Inelastic scattering and energy filtering](inelastic.md)**

    Remove inelastically scattered electrons as an energy filter does, and
    model what an unfiltered image keeps.

-   [![Gradients through the forward model](../assets/images/tutorials/differentiable-thumbnail.png)](differentiable.md)

    **[Gradients through the forward model](differentiable.md)**

    Differentiate an image with respect to its imaging parameters and atomic
    coordinates, and recover them by gradient descent.

-   [![Composing a forward model by hand](../assets/images/tutorials/custom_forward_model-thumbnail.png)](custom-forward-model.md)

    **[Composing a forward model by hand](custom-forward-model.md)**

    Rebuild the image from SPECTER's engines one stage at a time, then
    replace a stage with your own.

-   [![A tilt series from the engines](../assets/images/tutorials/tilt_series_engines-thumbnail.png)](tilt-series-engines.md)

    **[A tilt series from the engines](tilt-series-engines.md)**

    Blend the ice, size the volume for the tilt range, and propagate, focus
    and detect each tilt by hand, identically to the objects.

-   [![Building an ice library](../assets/images/tutorials/ice_library-thumbnail.png)](ice-library.md)

    **[Building an ice library](ice-library.md)**

    Optimise amorphous ice against its structure factor, and draw and tile
    crops of it at the voxel size you need.

</div>

## Short recipes

A recipe is one task and one figure, in a single block of code to copy and
adapt, plotting included. One tagged `basic` uses the high-level objects and
their defaults. One tagged `specialized` reaches below them, into the engines
or the geometry.

<div class="grid cards" markdown>

-   [![A potential from a PDB entry](../assets/images/tutorials/pdb_potential-thumbnail.png)](pdb-potential.md)

    **[A potential from a PDB entry](pdb-potential.md)**

    `basic` · An atomic model rendered into a potential volume, plotted
    as a slice and two projections.

-   [![Single atoms in bright field](../assets/images/tutorials/single_atoms-thumbnail.png)](single-atoms.md)

    **[Single atoms in bright field](single-atoms.md)**

    `specialized` · The projected potentials of five single atoms, and their
    coherent bright-field image.

-   [![Inspecting an exit wave](../assets/images/tutorials/exit_wave-thumbnail.png)](exit-wave.md)

    **[Inspecting an exit wave](exit-wave.md)**

    `basic` · The phase and amplitude of the wave leaving a particle, and
    the grid it is held on.

-   [![The coincidence radius of a camera](../assets/images/tutorials/coincidence_radius-thumbnail.png)](coincidence-radius.md)

    **[The coincidence radius of a camera](coincidence-radius.md)**

    `basic` · A camera's calibrated coincidence loss carried over to a
    simulation's pixel size and frames.

-   [![Saving micrographs with a STAR file](../assets/images/tutorials/save_micrographs-thumbnail.png)](save-micrographs.md)

    **[Saving micrographs with a STAR file](save-micrographs.md)**

    `basic` · A stack of micrographs written as an MRC file, with the STAR
    file that records each image's transfer function.

-   [![A transfer function of your own](../assets/images/tutorials/custom_transfer-thumbnail.png)](custom-transfer-function.md)

    **[A transfer function of your own](custom-transfer-function.md)**

    `specialized` · A knife edge in the back focal plane, applied to the
    wave's array between two verbs.

-   [![The four organelle kinds](../assets/images/tutorials/membrane_organelles-thumbnail.png)](membrane-organelles.md)

    **[The four organelle kinds](membrane-organelles.md)**

    `basic` · A vesicle, a branched tubule, a cisterna and a mitochondrion
    from `MembraneGenerator`.

-   [![Composing a membrane from primitives](../assets/images/tutorials/membrane_primitives-thumbnail.png)](membrane-primitives.md)

    **[Composing a membrane from primitives](membrane-primitives.md)**

    `specialized` · A budding vesicle from two balls and a tube, regularised
    and rendered as a bilayer.

-   [![A gallery of composed membranes](../assets/images/tutorials/membrane_gallery-thumbnail.png)](membrane-gallery.md)

    **[A gallery of composed membranes](membrane-gallery.md)**

    `specialized` · Eight shapes from a discocyte to a plasma membrane, each
    a few lines of primitives.

</div>

For the same workflows run from a TOML config with the `specter` command, see
the [Command-line guides](../user-guide/particle-stack.md).
