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
the membrane generator's own internal shape-construction state.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def _shell_bounding_box(
    shell_mask: torch.Tensor,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Half-open ``(z0, z1, y0, y1, x0, x1)`` bounds of the True voxels, or
    None when there are none.

    Reduced in two full passes rather than six, and never via
    ``nonzero()``, which would materialize one int64 index per shell voxel.

    Parameters
    ----------
    shell_mask : torch.Tensor
        Boolean volume, shape ``(Z, Y, X)``.

    Returns
    -------
    tuple of int or None
        Bounds suitable for slicing, or None if `shell_mask` is all False.
    """
    zy = shell_mask.any(dim=2)
    z_any = zy.any(dim=1)
    y_any = zy.any(dim=0)
    x_any = shell_mask.any(dim=1).any(dim=0)
    if not bool(z_any.any()):
        return None

    def _span(flags: torch.Tensor) -> tuple[int, int]:
        idx = torch.nonzero(flags, as_tuple=False).flatten()
        return int(idx[0]), int(idx[-1]) + 1

    z0, z1 = _span(z_any)
    y0, y1 = _span(y_any)
    x0, x1 = _span(x_any)
    return z0, z1, y0, y1, x0, x1


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
        ``tomogram/generator.py``'s ``_INSTANCE_LABEL_REL_THRESHOLD``). If
        ``density`` is all zero (no membrane present), every voxel is
        classified as cytosol.

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

    # Label the shell's own BOUNDING BOX, not the whole volume. Outside
    # that box there is no shell by construction, so the space out there is
    # a single connected shell-free region that reaches the volume's own
    # faces -- cytosol, decided by geometry with no labelling at all.
    # Inside the box, a component touching the box surface joins that
    # exterior (or, where the box sits flush against the volume, touches
    # the volume boundary itself) and is cytosol; one that does not is
    # sealed off by shell and is lumen. Identical answers, over as few
    # voxels as the shell actually spans.
    #
    # The box comes from the SHELL, never from a membrane's own geometry,
    # which is what keeps this exact for the density this is really handed:
    # `generator.py` stamps the carbon film in before the membrane phase,
    # so `density` is carbon plus membranes and `shell_mask` is not
    # membrane-only. Anything else above the threshold simply grows the
    # box, trading the speedup away rather than the correctness -- a full-
    # volume shell degenerates to exactly the previous behaviour.
    #
    # Worth it because `ndimage.label` is single-threaded union-find over
    # 26-connectivity and its int32 output is 4 bytes/voxel: at the 2 A
    # production grid (750x3000x3000) it was 12.8 min and a 27 GB label
    # array, against a 500-cube vesicle occupying 1.9% of that box.
    bbox = _shell_bounding_box(shell_mask)
    if bbox is None:
        # No shell anywhere: every voxel is reachable from outside.
        return {
            "shell": shell_mask,
            "lumen": torch.zeros_like(non_shell),
            "cytosol": non_shell,
        }
    z0, z1, y0, y1, x0, x1 = bbox
    non_shell_sub = non_shell[z0:z1, y0:y1, x0:x1]

    structure = np.ones((3, 3, 3), dtype=np.int8)  # full 26-connectivity
    labeled, num_features = ndimage.label(
        non_shell_sub.cpu().numpy(), structure=structure
    )

    boundary_labels: set[int] = set()
    if num_features > 0:
        boundary_labels.update(int(v) for v in labeled[0, :, :].ravel() if v > 0)
        boundary_labels.update(int(v) for v in labeled[-1, :, :].ravel() if v > 0)
        boundary_labels.update(int(v) for v in labeled[:, 0, :].ravel() if v > 0)
        boundary_labels.update(int(v) for v in labeled[:, -1, :].ravel() if v > 0)
        boundary_labels.update(int(v) for v in labeled[:, :, 0].ravel() if v > 0)
        boundary_labels.update(int(v) for v in labeled[:, :, -1].ravel() if v > 0)

    # Resolved as a lookup on the CPU-side label array, NOT as
    # `torch.isin(labeled_t, boundary_label_t)` on the device.
    #
    # `torch.isin` has no sorted fast path at these sizes and materializes
    # `elements.unsqueeze(-1) == test_elements` -- a bool tensor of shape
    # (n_voxels, n_boundary_labels). Measured at 65.1 bytes per voxel for
    # 63 boundary labels, so a 300x1200x1200 tomogram asked CUDA for
    # 25.35 GiB to answer a per-voxel membership test, and OOM'd. The
    # factor is the label COUNT, which the membrane geometry decides, so
    # this fired on some random draws and not others -- `configs/
    # tomogram.toml` unseeded reproduced it, seed 11 did not. (Above
    # roughly 128 test values torch.isin switches to a sort-based path and
    # the blowup disappears, which is why only moderate label counts hurt.)
    #
    # Labels from `ndimage.label` are dense integers 0..num_features, so a
    # boolean lookup indexed by label answers it in one gather. Doing that
    # in numpy also drops the int32 label volume's device transfer, which
    # existed only to feed the isin: 0.40 GiB of device memory total here,
    # against 25.35 + 1.73 GiB before.
    # Everything outside the shell's bounding box is cytosol (see above),
    # so start from that and overwrite only the box the labelling covered.
    cytosol_mask = torch.ones_like(non_shell)
    if boundary_labels:
        is_boundary = np.zeros(num_features + 1, dtype=bool)
        is_boundary[np.fromiter(boundary_labels, dtype=np.int64)] = True
        cytosol_sub = non_shell_sub & torch.from_numpy(is_boundary[labeled]).to(
            density.device
        )
    else:
        cytosol_sub = torch.zeros_like(non_shell_sub)
    cytosol_mask[z0:z1, y0:y1, x0:x1] = cytosol_sub
    lumen_mask = non_shell & ~cytosol_mask

    return {"shell": shell_mask, "lumen": lumen_mask, "cytosol": cytosol_mask}
