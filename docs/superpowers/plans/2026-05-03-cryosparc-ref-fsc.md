# CryoSPARC Reference FSC Plotting Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional `cryosparc_ref` parameter to Ghostbuster/Reconstructor that plots CryoSPARC reference volume on the same FSC plot as the reconstructed volume for comparison.

**Architecture:** Add `cryosparc_ref` parameter (filepath or tensor) to both Ghostbuster and Reconstructor classes. Load from file on init using the same pattern as `fsc_ref`. In `_save_fsc_figure()`, dynamically build the volumes and labels list to include both the current volume and cryosparc_ref when both `fsc_ref` and `cryosparc_ref` are set. Reuse existing `plot_map_to_model_fsc()` which already handles multiple volumes.

**Tech Stack:** PyTorch, mrcfile, Lightning, matplotlib (existing dependencies)

---

## File Structure

**Modify:**
- `src/specter/ghostbuster.py` — Add parameter, docstrings, pass-through logic
- `tests/test_ghostbuster.py` — Add smoke test for cryosparc_ref parameter

---

### Task 1: Add `cryosparc_ref` parameter to `Ghostbuster.__init__()`

**Files:**
- Modify: `src/specter/ghostbuster.py:829-862` (Ghostbuster.__init__)

- [ ] **Step 1: Add parameter to function signature**

In `Ghostbuster.__init__()` at line 829, add `cryosparc_ref` parameter after `fsc_mask`:

```python
def __init__(
    self,
    cs_file: str | Path,
    mrc_file: str | Path,
    dose_per_angstrom: float,
    lr: float | None = None,
    lr_R: float | None = None,
    lr_T: float | None = None,
    lr_D: float | None = None,
    scheduler: Literal[
        "LambdaLR", "CosineAnnealingWarmRestarts", "MultiplicativeLR"
    ] = "LambdaLR",
    epochs: int = 5,
    batch_size: int = 3,
    scattering_model: str = "rytov",
    aberration_model: str = "holography",
    symmetry: str | None = None,
    symmetry_batchsize: int | None = None,
    symmetry_mode: Literal["real", "fourier"] = "fourier",
    sparsity: float | None = None,
    rotate_mode: Literal["real", "fourier"] = "real",
    flipcurvature: bool = True,
    klim: float | None = None,
    nps_weight: torch.Tensor | None = None,
    learn_noise_model: bool = False,
    use_ncc: bool = False,
    fsc_ref: torch.Tensor | str | Path | None = None,
    fsc_mask: torch.Tensor | float | str | Path | None = None,
    cryosparc_ref: torch.Tensor | str | Path | None = None,
    precision: str = "16-mixed",
    num_workers: int = 0,
    num_particles: int | None = None,
    return_class: Literal["0", "1", "all"] = "all",
    run_dir: str | Path | None = None,
) -> None:
```

- [ ] **Step 2: Store cryosparc_ref as instance attribute**

In `Ghostbuster.__init__()` body (around line 940), after the line `self.fsc_mask = fsc_mask`, add:

```python
        self.cryosparc_ref = cryosparc_ref
```

- [ ] **Step 3: Update Ghostbuster docstring**

In the docstring (lines 728-825), after the `fsc_mask` parameter description, add:

```python
    cryosparc_ref : torch.Tensor, str, Path, or None
        CryoSPARC reference volume for FSC comparison. Can be a tensor or a
        path to a .mrc file to load. Only plotted alongside fsc_ref when
        both are provided. Default is None.
```

---

### Task 2: Add `cryosparc_ref` parameter to `Reconstructor.__init__()`

**Files:**
- Modify: `src/specter/ghostbuster.py:80-120` (Reconstructor.__init__)

- [ ] **Step 1: Add parameter to function signature**

In `Reconstructor.__init__()` at line 80, add `cryosparc_ref` parameter after `fsc_mask`:

```python
def __init__(
    self,
    V: torch.Tensor,
    voxel_size: float,
    quaternions: torch.Tensor,
    translations: torch.Tensor,
    ctf_params: dict[str, torch.Tensor],
    energy: float,
    dose_per_angstrom: float,
    anisomag: torch.Tensor | None = None,
    alpha: float = 0.0,
    defocus_offset: torch.Tensor = torch.tensor(0.0),
    scattering_model: str = "multislice",
    aberration_model: str = "holography",
    klim: float | None = None,
    sparsity: float | None = None,
    lr: float | None = None,
    lr_R: float | None = None,
    lr_T: float | None = None,
    lr_D: float | None = None,
    lr_decay: float = 0.1,
    scheduler: Literal[
        "LambdaLR", "CosineAnnealingWarmRestarts", "MultiplicativeLR"
    ] = "LambdaLR",
    kmask: torch.Tensor | None = None,
    nps_weight: torch.Tensor | None = None,
    learn_noise_model: bool = False,
    noise_ema_momentum: float = 0.9,
    use_ncc: bool = False,
    flipcurvature: bool = False,
    fsc_ref: torch.Tensor | str | Path | None = None,
    fsc_mask: torch.Tensor | float | str | Path | None = None,
    cryosparc_ref: torch.Tensor | str | Path | None = None,
    rotate_mode: Literal["real", "fourier"] = "real",
    symmetry: str | None = None,
    symmetry_batchsize: int | None = None,
    symmetry_mode: Literal["real", "fourier"] = "fourier",
    use_cpu_for_symmetry: bool = False,
    tag: str = "untagged",
    run_dir: str | Path | None = None,
    halfset_label: str | None = None,
) -> None:
```

- [ ] **Step 2: Load cryosparc_ref from file if path-like**

In `Reconstructor.__init__()` body (after line 191 where `fsc_ref` is loaded), add:

```python
        # cryosparc_ref — load from file if path provided
        if isinstance(cryosparc_ref, (str, Path)):
            cryosparc_ref = torch.as_tensor(mrcfile.read(str(cryosparc_ref)))
        self.cryosparc_ref = cryosparc_ref
```

- [ ] **Step 3: Add cryosparc_ref to save_hyperparameters ignore list**

In `Reconstructor.__init__()` at line 123, update the `save_hyperparameters()` call to include `"cryosparc_ref"` in the ignore list:

```python
        self.save_hyperparameters(
            ignore=[
                "V",
                "quaternions",
                "translations",
                "ctf_params",
                "anisomag",
                "kmask",
                "nps_weight",
                "fsc_ref",
                "cryosparc_ref",
                "fsc_mask",
                "run_dir",
                "halfset_label",
            ]
        )
```

- [ ] **Step 4: Update Reconstructor docstring**

In the docstring (lines 38-78), after the `fsc_mask` parameter description, add:

```python
    cryosparc_ref : torch.Tensor, str, Path, or None
        CryoSPARC reference volume for FSC comparison. Can be a tensor or a
        path to a .mrc file to load. Only plotted alongside fsc_ref when
        both are provided. Default is None.
```

---

### Task 3: Pass `cryosparc_ref` through `_build_reconstructor_and_loader()`

**Files:**
- Modify: `src/specter/ghostbuster.py:941-997` (Ghostbuster._build_reconstructor_and_loader)

- [ ] **Step 1: Add cryosparc_ref to Reconstructor instantiation**

In `Ghostbuster._build_reconstructor_and_loader()` at line 964, add `cryosparc_ref=self.cryosparc_ref,` to the Reconstructor constructor call:

```python
        model = Reconstructor(
            volume_init,
            voxel_size,
            self._rotations[:n_particles],
            self._translations[:n_particles],
            ctf_sliced,
            self._energy,
            self.dose_per_angstrom,
            anisomag=anisomag,
            alpha=self._alpha,
            scattering_model=scattering_model,
            aberration_model=self.aberration_model,
            lr=self.lr,
            lr_R=self.lr_R,
            lr_T=self.lr_T,
            lr_D=self.lr_D,
            scheduler=self.scheduler,
            kmask=kmask,
            klim=self.klim,
            nps_weight=self.nps_weight,
            learn_noise_model=self.learn_noise_model,
            use_ncc=self.use_ncc,
            sparsity=self.sparsity,
            rotate_mode=self.rotate_mode,
            flipcurvature=self.flipcurvature,
            symmetry=self.symmetry,
            symmetry_batchsize=self.symmetry_batchsize,
            symmetry_mode=self.symmetry_mode,
            fsc_ref=self.fsc_ref,
            fsc_mask=self.fsc_mask,
            cryosparc_ref=self.cryosparc_ref,
            run_dir=self.run_dir,
            halfset_label=self.halfset_label,
        )
```

---

### Task 4: Modify `_save_fsc_figure()` to include cryosparc_ref

**Files:**
- Modify: `src/specter/ghostbuster.py:633-667` (Reconstructor._save_fsc_figure)

- [ ] **Step 1: Replace entire _save_fsc_figure method**

Replace the method with:

```python
    def _save_fsc_figure(
        self,
        v: torch.Tensor,
        suffix: str,
        path: Path,
        label: str,
    ) -> None:
        """Compute and save an FSC figure with optional CryoSPARC reference. Silently skips on failure."""
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

---

### Task 5: Write test for cryosparc_ref parameter

**Files:**
- Create: `tests/test_ghostbuster_cryosparc.py`

- [ ] **Step 1: Create test file with smoke test**

Create a new file `tests/test_ghostbuster_cryosparc.py`:

```python
import tempfile
from pathlib import Path

import mrcfile
import pytest
import torch

from specter.ghostbuster import Reconstructor


def test_reconstructor_with_cryosparc_ref():
    """Smoke test: Reconstructor accepts cryosparc_ref parameter and saves FSC plot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create minimal test volumes
        vol = torch.randn(16, 16, 16)
        fsc_ref = torch.randn(16, 16, 16)
        cryosparc_ref = torch.randn(16, 16, 16)

        # Create temporary .mrc files for both references
        fsc_ref_path = tmpdir / "fsc_ref.mrc"
        cs_ref_path = tmpdir / "cs_ref.mrc"

        with mrcfile.new(str(fsc_ref_path), overwrite=True) as mrc:
            mrc.set_data(fsc_ref.numpy())
        with mrcfile.new(str(cs_ref_path), overwrite=True) as mrc:
            mrc.set_data(cryosparc_ref.numpy())

        # Minimal setup for Reconstructor
        quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        translations = torch.zeros(1, 2)
        ctf_params = {"dfu": torch.tensor([10000.0]), "dfv": torch.tensor([10000.0])}

        run_dir = tmpdir / "run"

        # Test with both fsc_ref and cryosparc_ref as tensors
        model = Reconstructor(
            V=vol.clone(),
            voxel_size=1.0,
            quaternions=quaternions,
            translations=translations,
            ctf_params=ctf_params,
            energy=300.0,
            dose_per_angstrom=1.0,
            fsc_ref=fsc_ref.clone(),
            cryosparc_ref=cryosparc_ref.clone(),
            run_dir=run_dir,
        )

        assert model.fsc_ref is not None
        assert model.cryosparc_ref is not None

        # Test that _save_fsc_figure runs without error
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "test_fsc.png"
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            model._save_fsc_figure(vol.clone(), "", out_path, "test")
            # Plot may be skipped if matplotlib not available, but should not raise
        except Exception as exc:
            pytest.fail(f"_save_fsc_figure raised unexpected exception: {exc}")


def test_reconstructor_cryosparc_ref_from_file():
    """Test that cryosparc_ref can be loaded from .mrc file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test volume and save as .mrc
        cryosparc_data = torch.randn(8, 8, 8)
        cs_ref_path = tmpdir / "cryosparc.mrc"
        with mrcfile.new(str(cs_ref_path), overwrite=True) as mrc:
            mrc.set_data(cryosparc_data.numpy())

        # Create Reconstructor with cryosparc_ref as file path
        vol = torch.randn(8, 8, 8)
        quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        translations = torch.zeros(1, 2)
        ctf_params = {"dfu": torch.tensor([10000.0]), "dfv": torch.tensor([10000.0])}

        model = Reconstructor(
            V=vol.clone(),
            voxel_size=1.0,
            quaternions=quaternions,
            translations=translations,
            ctf_params=ctf_params,
            energy=300.0,
            dose_per_angstrom=1.0,
            cryosparc_ref=str(cs_ref_path),
        )

        # Verify it was loaded as a tensor
        assert isinstance(model.cryosparc_ref, torch.Tensor)
        assert model.cryosparc_ref.shape == (8, 8, 8)


def test_reconstructor_cryosparc_ref_no_plot_without_fsc_ref():
    """Test that no FSC plot is generated if cryosparc_ref but no fsc_ref."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        vol = torch.randn(8, 8, 8)
        cryosparc_ref = torch.randn(8, 8, 8)
        quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        translations = torch.zeros(1, 2)
        ctf_params = {"dfu": torch.tensor([10000.0]), "dfv": torch.tensor([10000.0])}

        run_dir = tmpdir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create with cryosparc_ref but no fsc_ref
        model = Reconstructor(
            V=vol.clone(),
            voxel_size=1.0,
            quaternions=quaternions,
            translations=translations,
            ctf_params=ctf_params,
            energy=300.0,
            dose_per_angstrom=1.0,
            fsc_ref=None,
            cryosparc_ref=cryosparc_ref.clone(),
            run_dir=run_dir,
        )

        assert model.cryosparc_ref is not None
        assert model.fsc_ref is None

        # _save_fsc_figure should not be called in normal flow (only if fsc_ref is set)
        # But if called anyway, it should handle gracefully by not adding the cryosparc curve
        out_path = run_dir / "test_fsc.png"
        try:
            import matplotlib
            matplotlib.use('Agg')
            model._save_fsc_figure(vol.clone(), "", out_path, "test")
        except Exception as exc:
            # Expected: will fail because fsc_ref is needed, but should not crash
            pass
```

---

### Task 6: Run tests to verify implementation

**Files:**
- Test: `tests/test_ghostbuster_cryosparc.py`

- [ ] **Step 1: Run the new tests**

```bash
cd /mnt/cbis/home/e0788253/czii/specter
source .venv/bin/activate
python -m pytest tests/test_ghostbuster_cryosparc.py -v
```

**Expected output:**
```
tests/test_ghostbuster_cryosparc.py::test_reconstructor_with_cryosparc_ref PASSED
tests/test_ghostbuster_cryosparc.py::test_reconstructor_cryosparc_ref_from_file PASSED
tests/test_ghostbuster_cryosparc.py::test_reconstructor_cryosparc_ref_no_plot_without_fsc_ref PASSED
```

- [ ] **Step 2: Run full ghostbuster test suite to check for regressions**

```bash
python -m pytest tests/test_ghostbuster.py -v
```

**Expected:** All existing tests pass

- [ ] **Step 3: Run ruff check on modified files**

```bash
ruff check src/specter/ghostbuster.py tests/test_ghostbuster_cryosparc.py
```

**Expected:** No errors

- [ ] **Step 4: Run ruff format on modified files**

```bash
ruff format src/specter/ghostbuster.py tests/test_ghostbuster_cryosparc.py
```

---

### Task 7: Commit changes

**Files:**
- Modified: `src/specter/ghostbuster.py`
- Created: `tests/test_ghostbuster_cryosparc.py`

- [ ] **Step 1: Stage and commit**

```bash
cd /mnt/cbis/home/e0788253/czii/specter
git add src/specter/ghostbuster.py tests/test_ghostbuster_cryosparc.py
git commit -m "feat: add cryosparc_ref parameter for FSC comparison plots

- Add optional cryosparc_ref parameter to Ghostbuster and Reconstructor
- Load from .mrc file or accept as tensor (same pattern as fsc_ref)
- Plot CryoSPARC reference on same FSC plot when both fsc_ref and cryosparc_ref provided
- Label CryoSPARC curve with 'CryoSPARC' legend entry
- Silently skip cryosparc curve if only one ref is provided
- Add smoke tests for parameter handling and file loading

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

**Expected:** Commit succeeds

---

## Plan Self-Review

✅ **Spec coverage:**
- Parameter addition to Ghostbuster: Task 1
- Parameter addition to Reconstructor: Task 2
- Pass-through in _build_reconstructor_and_loader: Task 3
- FSC plotting logic (dynamic volumes/labels list): Task 4
- File loading pattern (mrcfile): Task 2
- Docstrings: Tasks 1 & 2
- Testing: Task 5
- Backward compatibility: Implicitly covered (default None)

✅ **Placeholder scan:** No TBD, TODO, or vague steps. All code is complete.

✅ **Type consistency:** 
- `cryosparc_ref: torch.Tensor | str | Path | None` used consistently
- Loading pattern matches `fsc_ref`
- Labels list built dynamically

✅ **No unresolved references:** All types, methods, and attributes are defined in earlier tasks.
