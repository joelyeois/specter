# FSC & Volume Epoch Slider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `visualize_job_epochs()` function in `plots.py` to display FSC and volume PNG images from a Ghostbuster job with a synchronized slider in Jupyter notebooks.

**Architecture:** A single function that discovers epoch images in `job_folder/epochs/`, filters by optional suffix, and displays them side-by-side with ipywidgets synchronization. Uses PIL to load images, pathlib for cross-platform file handling, and regex to extract epoch numbers.

**Tech Stack:** `pathlib`, `re`, `PIL`, `ipywidgets`, `matplotlib`

---

## File Structure

**Modified:**
- `src/specter/plots.py` — Add `visualize_job_epochs()` function with helper function `_extract_epoch_number()`

**New (Test):**
- `tests/test_plots.py` — Test epoch discovery, suffix filtering, and error handling

---

## Task 1: Image Discovery and Epoch Extraction

**Files:**
- Modify: `src/specter/plots.py`
- Test: `tests/test_plots.py`

### Step 1: Write failing tests for epoch discovery

Create `tests/test_plots.py`:

```python
import tempfile
from pathlib import Path
import pytest
from specter.plots import _extract_epoch_number


def test_extract_epoch_number_from_fsc():
    """Extract epoch number from FSC filename."""
    assert _extract_epoch_number("fsc_001.png") == 1
    assert _extract_epoch_number("fsc_042.png") == 42
    assert _extract_epoch_number("fsc_001A.png") == 1
    assert _extract_epoch_number("fsc_123B.png") == 123


def test_extract_epoch_number_from_vol():
    """Extract epoch number from volume filename."""
    assert _extract_epoch_number("vol_001.png") == 1
    assert _extract_epoch_number("vol_050.png") == 50


def test_extract_epoch_number_invalid():
    """Invalid filenames return None."""
    assert _extract_epoch_number("random_file.png") is None
    assert _extract_epoch_number("fsc.png") is None
    assert _extract_epoch_number("vol_abc.png") is None
```

Run: `pytest tests/test_plots.py::test_extract_epoch_number_from_fsc -v`

Expected: FAIL with "cannot import name '_extract_epoch_number'"

### Step 2: Implement `_extract_epoch_number()` helper

Add to `src/specter/plots.py` after imports:

```python
import re
from pathlib import Path


def _extract_epoch_number(filename: str) -> int | None:
    """
    Extract epoch number from FSC or volume PNG filename.

    Parameters
    ----------
    filename : str
        Filename like "fsc_001A.png" or "vol_042.png"

    Returns
    -------
    int or None
        Epoch number if matched, None otherwise.
    """
    match = re.search(r'(?:fsc|vol)_(\d+)', filename)
    return int(match.group(1)) if match else None
```

Run: `pytest tests/test_plots.py::test_extract_epoch_number_from_fsc -v`

Expected: PASS

Run: `pytest tests/test_plots.py::test_extract_epoch_number_from_vol -v`

Expected: PASS

Run: `pytest tests/test_plots.py::test_extract_epoch_number_invalid -v`

Expected: PASS

### Step 3: Write tests for image discovery

Add to `tests/test_plots.py`:

```python
def test_discover_fsc_images_no_suffix():
    """Find all FSC images without suffix filter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        job_folder = Path(tmpdir)
        epochs_dir = job_folder / "epochs"
        epochs_dir.mkdir()
        
        # Create test files
        (epochs_dir / "fsc_001.png").touch()
        (epochs_dir / "fsc_002A.png").touch()
        (epochs_dir / "fsc_003B.png").touch()
        (epochs_dir / "vol_001.png").touch()
        
        # Should return all FSC files sorted by epoch
        from specter.plots import _discover_fsc_images
        fsc_files = _discover_fsc_images(job_folder, suffix=None)
        assert len(fsc_files) == 3
        assert fsc_files[0][0] == 1  # epoch number
        assert fsc_files[1][0] == 2
        assert fsc_files[2][0] == 3


def test_discover_fsc_images_with_suffix():
    """Find only FSC images matching suffix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        job_folder = Path(tmpdir)
        epochs_dir = job_folder / "epochs"
        epochs_dir.mkdir()
        
        # Create test files
        (epochs_dir / "fsc_001A.png").touch()
        (epochs_dir / "fsc_002A.png").touch()
        (epochs_dir / "fsc_003B.png").touch()
        
        from specter.plots import _discover_fsc_images
        fsc_files = _discover_fsc_images(job_folder, suffix="A")
        assert len(fsc_files) == 2
        assert all(f[0] in [1, 2] for f in fsc_files)


def test_discover_vol_images():
    """Find all volume images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        job_folder = Path(tmpdir)
        epochs_dir = job_folder / "epochs"
        epochs_dir.mkdir()
        
        # Create test files
        (epochs_dir / "vol_001.png").touch()
        (epochs_dir / "vol_002.png").touch()
        (epochs_dir / "vol_003.png").touch()
        
        from specter.plots import _discover_vol_images
        vol_files = _discover_vol_images(job_folder)
        assert len(vol_files) == 3
        assert vol_files[0][0] == 1
        assert vol_files[1][0] == 2
        assert vol_files[2][0] == 3
```

Run: `pytest tests/test_plots.py::test_discover_fsc_images_no_suffix -v`

Expected: FAIL with "cannot import name '_discover_fsc_images'"

### Step 4: Implement image discovery helpers

Add to `src/specter/plots.py`:

```python
def _discover_fsc_images(
    job_folder: str | Path, suffix: str | None = None
) -> list[tuple[int, Path]]:
    """
    Find FSC PNG files in job_folder/epochs/, sorted by epoch number.

    Parameters
    ----------
    job_folder : str or Path
        Job folder path.
    suffix : str or None
        If specified, only return files matching this suffix (e.g., "A" for fsc_001A.png).

    Returns
    -------
    list of (epoch_number, Path) tuples
        Sorted by epoch number ascending.

    Raises
    ------
    FileNotFoundError
        If job_folder/epochs/ does not exist.
    """
    job_folder = Path(job_folder)
    epochs_dir = job_folder / "epochs"
    
    if not epochs_dir.exists():
        raise FileNotFoundError(f"Epochs directory not found: {epochs_dir}")
    
    # Find matching FSC files
    pattern = f"fsc_*{suffix}.png" if suffix else "fsc_*.png"
    fsc_files = []
    for fsc_file in sorted(epochs_dir.glob(pattern)):
        epoch = _extract_epoch_number(fsc_file.name)
        if epoch is not None:
            fsc_files.append((epoch, fsc_file))
    
    # Sort by epoch number
    fsc_files.sort(key=lambda x: x[0])
    return fsc_files


def _discover_vol_images(job_folder: str | Path) -> list[tuple[int, Path]]:
    """
    Find volume PNG files in job_folder/epochs/, sorted by epoch number.

    Parameters
    ----------
    job_folder : str or Path
        Job folder path.

    Returns
    -------
    list of (epoch_number, Path) tuples
        Sorted by epoch number ascending.

    Raises
    ------
    FileNotFoundError
        If job_folder/epochs/ does not exist.
    """
    job_folder = Path(job_folder)
    epochs_dir = job_folder / "epochs"
    
    if not epochs_dir.exists():
        raise FileNotFoundError(f"Epochs directory not found: {epochs_dir}")
    
    # Find volume files
    vol_files = []
    for vol_file in sorted(epochs_dir.glob("vol_*.png")):
        epoch = _extract_epoch_number(vol_file.name)
        if epoch is not None:
            vol_files.append((epoch, vol_file))
    
    # Sort by epoch number
    vol_files.sort(key=lambda x: x[0])
    return vol_files
```

Run: `pytest tests/test_plots.py::test_discover_fsc_images_no_suffix -v`

Expected: PASS

Run: `pytest tests/test_plots.py::test_discover_fsc_images_with_suffix -v`

Expected: PASS

Run: `pytest tests/test_plots.py::test_discover_vol_images -v`

Expected: PASS

### Step 5: Commit discovery functions

```bash
git add tests/test_plots.py src/specter/plots.py
git commit -m "feat: add image discovery helpers for FSC and volume epochs"
```

---

## Task 2: Main Visualization Function

**Files:**
- Modify: `src/specter/plots.py`
- Test: `tests/test_plots.py`

### Step 1: Write tests for main function

Add to `tests/test_plots.py`:

```python
def test_visualize_job_epochs_creates_widget():
    """Function returns without error and creates interactive display."""
    with tempfile.TemporaryDirectory() as tmpdir:
        job_folder = Path(tmpdir)
        epochs_dir = job_folder / "epochs"
        epochs_dir.mkdir()
        
        # Create minimal test images (1x1 PNG)
        from PIL import Image
        img = Image.new('L', (1, 1), color=0)
        img.save(epochs_dir / "fsc_001.png")
        img.save(epochs_dir / "vol_001.png")
        
        # Should not raise
        from specter.plots import visualize_job_epochs
        try:
            import ipywidgets
            visualize_job_epochs(job_folder)
        except ImportError:
            pytest.skip("ipywidgets not installed")


def test_visualize_job_epochs_no_fsc_raises():
    """Raises clear error if no FSC images found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        job_folder = Path(tmpdir)
        epochs_dir = job_folder / "epochs"
        epochs_dir.mkdir()
        
        from PIL import Image
        img = Image.new('L', (1, 1), color=0)
        img.save(epochs_dir / "vol_001.png")
        
        from specter.plots import visualize_job_epochs
        with pytest.raises(ValueError, match="No FSC images found"):
            visualize_job_epochs(job_folder)


def test_visualize_job_epochs_no_vol_raises():
    """Raises clear error if no volume images found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        job_folder = Path(tmpdir)
        epochs_dir = job_folder / "epochs"
        epochs_dir.mkdir()
        
        from PIL import Image
        img = Image.new('L', (1, 1), color=0)
        img.save(epochs_dir / "fsc_001.png")
        
        from specter.plots import visualize_job_epochs
        with pytest.raises(ValueError, match="No volume images found"):
            visualize_job_epochs(job_folder)


def test_visualize_job_epochs_no_epochs_dir_raises():
    """Raises clear error if epochs directory missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from specter.plots import visualize_job_epochs
        with pytest.raises(FileNotFoundError, match="epochs"):
            visualize_job_epochs(tmpdir)
```

Run: `pytest tests/test_plots.py::test_visualize_job_epochs_creates_widget -v`

Expected: FAIL with "cannot import name 'visualize_job_epochs'"

### Step 2: Implement main function with ipywidgets

Add to `src/specter/plots.py` in the try/except block at the end:

```python
try:
    import ipywidgets as widgets
    from IPython.display import display, Image as IPImage
    
    def visualize_job_epochs(
        job_folder: str | Path,
        suffix: str | None = None,
        image_width: int = 8,
    ) -> None:
        """
        Display FSC and volume images side-by-side with synchronized epoch slider.

        Interactive Jupyter visualization of reconstruction progress. FSC and volume
        PNG images are displayed for each epoch with a synchronized slider control.
        Requires ipywidgets to be installed.

        Parameters
        ----------
        job_folder : str or Path
            Path to the job folder (e.g., "~/specter-data/empiar-10202/J002").
        suffix : str, optional
            Filter FSC images by suffix (e.g., "A" shows only fsc_001A.png).
            If None, all FSC images are shown.
        image_width : int, optional
            Display width of each image in inches. Default is 8.

        Raises
        ------
        FileNotFoundError
            If job_folder/epochs/ does not exist.
        ValueError
            If no FSC or volume images are found.

        Examples
        --------
        >>> visualize_job_epochs("~/specter-data/empiar-10202/J002")
        >>> visualize_job_epochs("~/specter-data/empiar-10202/J002", suffix="A")
        """
        from pathlib import Path
        from PIL import Image as PILImage
        
        job_folder = Path(job_folder).expanduser()
        
        # Discover images
        try:
            fsc_files = _discover_fsc_images(job_folder, suffix=suffix)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e)) from e
        
        try:
            vol_files = _discover_vol_images(job_folder)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e)) from e
        
        if not fsc_files:
            suffix_str = f" with suffix '{suffix}'" if suffix else ""
            raise ValueError(f"No FSC images found{suffix_str} in {job_folder / 'epochs'}")
        if not vol_files:
            raise ValueError(f"No volume images found in {job_folder / 'epochs'}")
        
        # Create epoch lists (use max range, show N/A for missing)
        fsc_epochs = {epoch: path for epoch, path in fsc_files}
        vol_epochs = {epoch: path for epoch, path in vol_files}
        all_epochs = sorted(set(fsc_epochs.keys()) | set(vol_epochs.keys()))
        
        # Create matplotlib figures for display
        fig_fsc, ax_fsc = plt.subplots(figsize=(image_width, image_width * 0.75), dpi=100)
        fig_vol, ax_vol = plt.subplots(figsize=(image_width, image_width * 0.75), dpi=100)
        plt.close(fig_fsc)
        plt.close(fig_vol)
        
        # Create output widgets
        output_fsc = widgets.Output()
        output_vol = widgets.Output()
        
        def update_display(epoch_idx):
            """Update both images for the selected epoch index."""
            epoch = all_epochs[epoch_idx]
            
            output_fsc.clear_output(wait=True)
            output_vol.clear_output(wait=True)
            
            with output_fsc:
                if epoch in fsc_epochs:
                    img_fsc = PILImage.open(fsc_epochs[epoch])
                    ax_fsc.clear()
                    ax_fsc.imshow(img_fsc)
                    ax_fsc.set_title(f"FSC Epoch {epoch}")
                    ax_fsc.axis('off')
                    plt.suptitle("")
                    fig_fsc.tight_layout()
                    display(fig_fsc)
                else:
                    from IPython.display import Markdown
                    display(Markdown(f"**FSC**: No image for epoch {epoch}"))
            
            with output_vol:
                if epoch in vol_epochs:
                    img_vol = PILImage.open(vol_epochs[epoch])
                    ax_vol.clear()
                    ax_vol.imshow(img_vol)
                    ax_vol.set_title(f"Volume Epoch {epoch}")
                    ax_vol.axis('off')
                    plt.suptitle("")
                    fig_vol.tight_layout()
                    display(fig_vol)
                else:
                    from IPython.display import Markdown
                    display(Markdown(f"**Volume**: No image for epoch {epoch}"))
        
        # Create slider
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(all_epochs) - 1,
            step=1,
            description="Epoch:",
            continuous_update=False,
        )
        
        # Update on slider change
        def on_slider_change(change):
            update_display(change['new'])
        
        slider.observe(on_slider_change, names='value')
        
        # Layout: title, images side-by-side, slider
        header = widgets.HTML(f"<h3>Job: {job_folder.name}</h3>")
        images_hbox = widgets.HBox([output_fsc, output_vol])
        
        vbox = widgets.VBox([header, images_hbox, slider])
        
        # Initial display
        update_display(0)
        display(vbox)

except ImportError:
    def visualize_job_epochs(
        job_folder: str | Path,
        suffix: str | None = None,
        image_width: int = 8,
    ) -> None:
        raise ImportError(
            "visualize_job_epochs() requires ipywidgets. "
            "Install with: pip install ipywidgets"
        )
```

Run: `pytest tests/test_plots.py::test_visualize_job_epochs_creates_widget -v`

Expected: PASS (or SKIP if ipywidgets not available)

Run: `pytest tests/test_plots.py::test_visualize_job_epochs_no_fsc_raises -v`

Expected: PASS

Run: `pytest tests/test_plots.py::test_visualize_job_epochs_no_vol_raises -v`

Expected: PASS

Run: `pytest tests/test_plots.py::test_visualize_job_epochs_no_epochs_dir_raises -v`

Expected: PASS

### Step 3: Commit main function

```bash
git add tests/test_plots.py src/specter/plots.py
git commit -m "feat: implement visualize_job_epochs() with ipywidgets slider"
```

---

## Task 3: Code Quality & Documentation

**Files:**
- Modify: `src/specter/plots.py`
- Modify: `tests/test_plots.py`

### Step 1: Run linting and type checking

```bash
ruff check src/specter/plots.py tests/test_plots.py
ruff format src/specter/plots.py tests/test_plots.py
```

Expected: No errors after formatting.

```bash
mypy src/specter/plots.py
```

Expected: PASS with no type errors.

### Step 2: Run full test suite

```bash
pytest tests/test_plots.py -v
```

Expected: All tests PASS.

### Step 3: Commit quality improvements

```bash
git add src/specter/plots.py tests/test_plots.py
git commit -m "style: format plots.py and test_plots.py to ruff/mypy standards"
```

---

## Self-Review Against Spec

✓ **Function signature** — `visualize_job_epochs(job_folder, suffix=None, image_width=8)` matches spec  
✓ **Image discovery** — `_discover_fsc_images()` and `_discover_vol_images()` find epochs correctly  
✓ **Suffix filtering** — Optional `suffix` parameter filters FSC images only  
✓ **UI layout** — ipywidgets with side-by-side images + synchronized slider  
✓ **Epoch display** — "FSC Epoch XXX" and "Volume Epoch XXX" captions shown  
✓ **Error handling** — FileNotFoundError for missing epochs dir, ValueError for no matching images  
✓ **Return value** — Returns None, displays output directly in Jupyter  
✓ **Cross-platform** — Uses `pathlib.Path` and `.expanduser()` for home directory expansion  

No gaps found. All spec requirements covered in tasks.

---

Plan complete and saved to `docs/superpowers/plans/2026-05-03-fsc-volume-epoch-slider.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?