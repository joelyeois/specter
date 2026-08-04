"""
Classify a rasterized membrane density volume into disjoint spatial regions
-- shell (the bilayer material itself), lumen (enclosed interior, e.g. a
vesicle's inside), and cytosol (exterior, reachable from the volume
boundary) -- so that downstream placement can restrict a protein species to
the region its biology dictates (see module docstring of
:mod:`.generator`).

`MembraneField.phi`'s sign only distinguishes lipid-solid from not-lipid-
solid; it says nothing about which side of a closed shell a given "not
lipid solid" point is on. Recovering that requires real topology: this
module labels the shell's connected complement via
:func:`scipy.ndimage.label` (full 26-connectivity, so voxelization/
thresholding noise doesn't fracture a single open region into spurious
pieces) and calls whichever labeled component(s) touch the volume's own
boundary faces "cytosol" -- by construction, that's the one region that
must be reachable from outside the volume. Everything else non-shell is
enclosed, and thus "lumen". This is deliberately membrane-shape-agnostic:
it works for a single vesicle, several disjoint vesicles, a sheet, or no
membrane at all (in which case shell/lumen are empty and cytosol is
everything) without any special-casing, unlike an approach that keyed off
the membrane generator's own internal metaball source positions.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def classify_membrane_regions(
    density: torch.Tensor,
    density_threshold: float | None = None,
) -> dict[str, torch.Tensor]:
    """
    Split a rasterized membrane density volume into shell/lumen/cytosol
    boolean masks.

    Parameters
    ----------
    density : torch.Tensor
        Rasterized membrane density, shape ``(Z, Y, X)`` -- e.g.
        ``MembraneGenerator.volume`` after ``generate()``. Must be on the
        same voxel grid as whatever will be packed against these masks.
    density_threshold : float, optional
        Density value above which a voxel counts as membrane shell
        material. Default ``0.05 * density.max()`` -- a small fraction of
        the bilayer's own peak potential, matching the relative-threshold
        convention used elsewhere for binarizing a rendered density (see
        ``specimen/packing/pdb_packing.py``'s
        ``_INSTANCE_LABEL_REL_THRESHOLD``). If ``density`` is all zero
        (no membrane present), every voxel is classified as cytosol.

    Returns
    -------
    dict of str to torch.Tensor
        ``{"shell": ..., "lumen": ..., "cytosol": ...}``, each a boolean
        tensor shaped like `density`. The three masks partition the full
        volume (every voxel belongs to exactly one).
    """
    if density_threshold is None:
        peak = float(density.max())
        density_threshold = 0.05 * peak if peak > 0 else 0.0

    shell_mask = density > density_threshold
    non_shell = ~shell_mask

    structure = np.ones((3, 3, 3), dtype=np.int8)  # full 26-connectivity
    labeled, num_features = ndimage.label(non_shell.cpu().numpy(), structure=structure)

    boundary_labels: set[int] = set()
    if num_features > 0:
        boundary_labels.update(int(v) for v in labeled[0, :, :].ravel() if v > 0)
        boundary_labels.update(int(v) for v in labeled[-1, :, :].ravel() if v > 0)
        boundary_labels.update(int(v) for v in labeled[:, 0, :].ravel() if v > 0)
        boundary_labels.update(int(v) for v in labeled[:, -1, :].ravel() if v > 0)
        boundary_labels.update(int(v) for v in labeled[:, :, 0].ravel() if v > 0)
        boundary_labels.update(int(v) for v in labeled[:, :, -1].ravel() if v > 0)

    labeled_t = torch.from_numpy(labeled).to(density.device)
    if boundary_labels:
        boundary_label_t = torch.tensor(sorted(boundary_labels), device=density.device)
        cytosol_mask = non_shell & torch.isin(labeled_t, boundary_label_t)
    else:
        cytosol_mask = torch.zeros_like(non_shell)
    lumen_mask = non_shell & ~cytosol_mask

    return {"shell": shell_mask, "lumen": lumen_mask, "cytosol": cytosol_mask}
