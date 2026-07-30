# Atomic potentials

!!! info "Work in progress"
    Will cover the Kirkland, Lobato and Shtyrov parameterisations, soft
    voxelisation, and the supersample-then-pool kernel construction.

## Example

The atomic potential and imaging code are validated against the worked
examples in Kirkland's *Advanced Computing in Electron Microscopy*. The
figure below is SPECTER's own output for the standard five-element test
row, reproducing the textbook's coherent bright-field line scan.

![Coherent bright-field line scan through C, Si, Cu, Au and U.](../assets/images/coherent-bright-field-linescan-kirkland.png){ width="700" }

Produced by `compare-atomic-potentials-with-kirkland.ipynb`, which places
the corresponding textbook figure alongside it for direct comparison.

<div class="grid" markdown>

![3D atomic potential against radius, per element.](../assets/images/atomic-potential-3d-kirkland.png){ width="340" }

![The same potentials projected to 2D, as used by the faster projection path.](../assets/images/projected-atomic-potential-2d-kirkland.png){ width="340" }

</div>
