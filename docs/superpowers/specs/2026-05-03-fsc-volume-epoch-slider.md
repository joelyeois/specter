# FSC & Volume Epoch Slider Visualization

**Date:** 2026-05-03  
**Component:** Interactive Jupyter notebook visualization in `plots.py`  
**Purpose:** Display FSC and volume PNG images from a Ghostbuster job with synchronized epoch slider.

## Overview

Add a `visualize_job_epochs()` function to `plots.py` that creates an interactive side-by-side visualization of FSC and volume images across reconstruction epochs. Users pass a job folder path and optionally filter by suffix (for halfset reconstructions).

## Functional Requirements

### Function Signature
```python
def visualize_job_epochs(
    job_folder: str | Path,
    suffix: str | None = None,
    image_width: int = 8,
) -> None:
```

### Parameters
- **job_folder** (`str | Path`): Path to the job folder, e.g., `"~/specter-data/empiar-10202/J002"`
- **suffix** (`str | None`): Optional filter for FSC images by suffix. Examples:
  - `None` (default): shows all FSC images (fsc_001.png, fsc_002A.png, etc.)
  - `"A"`: shows only FSC images with "A" suffix (fsc_001A.png, fsc_002A.png, etc.)
  - `"B"`: shows only FSC images with "B" suffix (fsc_001B.png, fsc_002B.png, etc.)
- **image_width** (`int`): Display width of each image in inches (default 8)

### Image Discovery
- **FSC images**: Located in `job_folder/epochs/`, match pattern `fsc_NNN{suffix}.png`
  - If suffix is specified, only include files matching that suffix
  - If suffix is None, include all FSC files
- **Volume images**: Located in `job_folder/epochs/`, match pattern `vol_NNN.png`
- Extract epoch numbers from filenames and sort numerically
- Error handling: Display clear message if no matching files found

### UI Layout
Built with `ipywidgets` (provides play/pause controls, smooth interaction):

1. **Header**: Job folder path
2. **Image Area**: Two matplotlib figures side-by-side
   - Left: FSC image with caption "FSC Epoch XXX"
   - Right: Volume image with caption "Volume Epoch XXX"
3. **Synchronized Slider**: Single slider spanning full width
   - Synced to both images
   - Shows epoch number as you drag
   - Displays epoch range (e.g., "Epoch 1/50")

### Behavior
- Slider updates both images simultaneously
- If FSC and volume have different numbers of epochs, slider uses the maximum range and shows "N/A" for missing images
- Suffix filtering applies only to FSC; volume images are always included if present

## Error Handling
- If no FSC images found with the specified suffix: show message, don't crash
- If no volume images found: show message, don't crash
- If job folder doesn't exist: raise clear `FileNotFoundError`
- If epochs folder is missing: show message with helpful path info

## Return Value
Returns `None` — displays output directly in Jupyter.

## Implementation Details
- Use `pathlib.Path` for cross-platform file handling
- Use `PIL.Image` to load and display PNG files
- Use `ipywidgets.IntSlider` for the control
- Use `ipywidgets.HBox` for side-by-side layout
- Extract epoch numbers using regex: `r'(?:fsc|vol)_(\d+)'`

## Scope
This is a standalone visualization utility. It does not modify files, save outputs, or interact with Ghostbuster training logic.
