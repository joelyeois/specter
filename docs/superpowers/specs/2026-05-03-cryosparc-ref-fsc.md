# CryoSPARC Reference FSC Plotting — Design Spec

**Date:** 2026-05-03  
**Status:** Design phase  
**Objective:** Add optional CryoSPARC reference volume comparison to FSC plots during Ghostbuster reconstruction.

## Overview

When reconstructing with Ghostbuster, users may want to compare their evolving reconstruction against a reference CryoSPARC map side-by-side in FSC plots. This feature adds a `cryosparc_ref` parameter that, when provided alongside `fsc_ref`, plots both:
1. Current reconstructed volume vs. `fsc_ref` (existing behavior)
2. CryoSPARC reference vs. `fsc_ref` (new comparison curve)

Both curves appear on the same FSC plot for visual comparison across epochs.

## Parameters & Data Flow

### Add `cryosparc_ref` to Ghostbuster

- **Type:** `torch.Tensor | str | Path | None` (same as `fsc_ref`)
- **Default:** `None` (optional, no breaking changes)
- **Loading:** On Ghostbuster `__init__`, convert filepath → tensor using `mrcfile.read()`, same as `fsc_ref`
- **Stored as:** Instance attribute `self.cryosparc_ref`

### Pass through to Reconstructor

In `Ghostbuster._build_reconstructor_and_loader()`, pass `self.cryosparc_ref` to Reconstructor:
```python
model = Reconstructor(
    ...
    fsc_ref=self.fsc_ref,
    cryosparc_ref=self.cryosparc_ref,  # ← new
    fsc_mask=self.fsc_mask,
    ...
)
```

Add `cryosparc_ref` parameter to Reconstructor `__init__`:
- Type: `torch.Tensor | str | Path | None`
- Load from file if path-like (same pattern as `fsc_ref`)
- Store as `self.cryosparc_ref`
- Include in `save_hyperparameters(ignore=[...])` to skip saving the tensor

## Plotting Logic

### Condition for plotting
- FSC plots are only generated if `self.fsc_ref` is set (existing behavior)
- If `fsc_ref` is set and `cryosparc_ref` is also set, add cryosparc curve to the plot
- If `cryosparc_ref` is set but `fsc_ref` is not, no FSC plots are generated (silently skip — no error)

### Implementation in `_save_fsc_figure()`

In `Reconstructor._save_fsc_figure()`, dynamically build volume and label lists:

```python
def _save_fsc_figure(self, v: torch.Tensor, suffix: str, path: Path, label: str) -> None:
    """Compute and save FSC figure with optional CryoSPARC reference."""
    try:
        import matplotlib.pyplot as plt
        from .plots import plot_map_to_model_fsc

        fsc_ref = (
            self.fsc_ref.detach().cpu().float()
            if isinstance(self.fsc_ref, torch.Tensor)
            else self.fsc_ref
        )
        fsc_mask = (
            self.fsc_mask.detach().cpu().float()
            if isinstance(self.fsc_mask, torch.Tensor)
            else None
        )

        # Build list of volumes and labels
        vols = [v]
        labels = [label]

        # Add CryoSPARC reference if both fsc_ref and cryosparc_ref are set
        if self.cryosparc_ref is not None and self.fsc_ref is not None:
            cs_ref = (
                self.cryosparc_ref.detach().cpu().float()
                if isinstance(self.cryosparc_ref, torch.Tensor)
                else self.cryosparc_ref
            )
            vols.append(cs_ref)
            labels.append("CryoSPARC")

        fig = plot_map_to_model_fsc(
            vols,
            fsc_ref,
            voxel_size=self.voxel_size,
            mask=fsc_mask,
            labels=labels,
            show=False,
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[Reconstructor] FSC plot skipped: {exc}")
```

## Files Modified

1. **`src/specter/ghostbuster.py`**
   - `Ghostbuster.__init__()`: Add `cryosparc_ref` parameter and attribute
   - `Ghostbuster._build_reconstructor_and_loader()`: Pass `cryosparc_ref` to Reconstructor
   - `Reconstructor.__init__()`: Add `cryosparc_ref` parameter, load from file if needed, store attribute
   - `Reconstructor._save_fsc_figure()`: Build dynamic volumes/labels list, conditionally add cryosparc

2. **`src/specter/plots.py`**
   - No changes (existing `plot_map_to_model_fsc()` already handles multiple volumes and labels)

## Docstrings & Logging

- Update `Ghostbuster` docstring to document `cryosparc_ref` parameter (parallel to `fsc_ref` docs)
- Update `Reconstructor` docstring similarly
- No job logging changes needed (cryosparc_ref tensor is in `save_hyperparameters(ignore=[...])`)

## Testing

- Add minimal smoke test: pass `cryosparc_ref` to a small Ghostbuster run with `fsc_ref` set; verify FSC plot is generated and contains 2 curves
- Verify silent skip when `cryosparc_ref` is provided but `fsc_ref` is None
- Verify backward compatibility: existing code without `cryosparc_ref` works unchanged

## Edge Cases & Error Handling

| Scenario | Behavior |
|---|---|
| `cryosparc_ref` provided, `fsc_ref=None` | No FSC plots generated (fsc_ref is required for any FSC plot) |
| `cryosparc_ref=None`, `fsc_ref` provided | Plot current volume vs fsc_ref only (current behavior) |
| Both `cryosparc_ref` and `fsc_ref` provided | Plot both current volume and cryosparc reference on same FSC plot |
| Invalid filepath for `cryosparc_ref` | Caught by `mrcfile.read()`, error message printed, no FSC plot generated |

## Backward Compatibility

✅ Fully backward compatible — `cryosparc_ref` defaults to `None`, no changes to existing API.

## Success Criteria

- FSC plots during training and at fit end show both reconstruction and CryoSPARC curves when both refs are provided
- No plot generated if `cryosparc_ref` alone (without `fsc_ref`)
- Existing workflows unaffected
- Clear labels ("CryoSPARC" vs. epoch label) in legend
