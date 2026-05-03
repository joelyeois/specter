# Ghostbuster–Job Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the `Job` context manager to `Ghostbuster` so that every run gets a versioned output folder with per-epoch volume previews, FSC plots, and cross-validated halfset param checking.

**Architecture:** Five incremental changes: (1) make `plot3d` return a saveable Figure; (2–3) extend `Job` with a resume mode and param-exclusion/validation in `create()`; (4) teach `Ghostbuster` to derive its halfset label and exclude `return_class` from logged params; (5) add per-epoch figure-saving callbacks to `Reconstructor`.

**Tech Stack:** PyTorch, Lightning, matplotlib, seaborn, pytest, mrcfile

---

## File Map

| File | Change |
|------|--------|
| `src/specter/plots.py` | Add `show: bool = True` param and `-> matplotlib.figure.Figure` return to `plot3d` |
| `src/specter/jobs/_job.py` | Add `job_id` resume param; exclusion + validation in `create()` |
| `src/specter/ghostbuster.py` | `Ghostbuster`: `_job_log_exclude`, auto `halfset_label`; `Reconstructor`: per-epoch PNG callbacks |
| `tests/test_jobs.py` | Tests for resume, param exclusion, param validation |

---

## Task 1: `plot3d` returns a Figure

**Files:** Modify `src/specter/plots.py`

No automated test — visual function verified by the user in a notebook.

- [ ] **Step 1: Update `plot3d` signature and body**

In `src/specter/plots.py`, replace the existing `plot3d` function:

```python
def plot3d(
    vol: torch.Tensor,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    show: bool = True,
) -> matplotlib.figure.Figure:
    """
    Plot 3 orthogonal projections of a 3D volume.

    Parameters
    ----------
    vol : torch.Tensor
        3D volume tensor.
    title : str or None, optional
        Super title for the plot.
    vmin : float or None, optional
        Minimum value for colormap scaling.
    vmax : float or None, optional
        Maximum value for colormap scaling.
    cmap : str or None, optional
        Matplotlib colormap name. Default is None (uses matplotlib default).
    show : bool, optional
        Whether to call ``plt.show()``. Set ``False`` when saving to disk
        programmatically. Default is True.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object. Caller is responsible for closing it when
        ``show=False``.
    """
    fig, axes = plt.subplots(1, 3, dpi=200, constrained_layout=True, figsize=(8, 3.6))
    for i, ax in enumerate(axes.ravel()):
        im = ax.imshow(vol.sum(i), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set(xticks=[], yticks=[], title=f"projection along axis {i}")
        fig.colorbar(im, ax=ax, location="bottom")
    if title is not None:
        fig.suptitle(title, fontsize=15)
    if show:
        plt.show()
    return fig
```

- [ ] **Step 2: Lint**

```bash
ruff check src/specter/plots.py
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/specter/plots.py
git commit -m "feat: plot3d returns Figure, add show=False param for headless saving"
```

---

## Task 2: Job resume via `job_id`

**Files:** Modify `src/specter/jobs/_job.py`, `tests/test_jobs.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_jobs.py`:

```python
# ---------------------------------------------------------------------------
# Task 7: Job resume via job_id
# ---------------------------------------------------------------------------


def test_job_resume_opens_existing_folder(tmp_path: Path) -> None:
    with Job("ghostbuster", project="p", base_dir=tmp_path) as job:
        first_dir = job.dir

    with Job("ghostbuster", project="p", base_dir=tmp_path, job_id="J001") as job:
        assert job.dir == first_dir


def test_job_resume_does_not_allocate_new_id(tmp_path: Path) -> None:
    with Job("ghostbuster", project="p", base_dir=tmp_path) as _:
        pass  # J001

    with Job("ghostbuster", project="p", base_dir=tmp_path, job_id="J001") as job:
        pass

    # Next new job should still be J002, not J003
    with Job("ghostbuster", project="p", base_dir=tmp_path) as job3:
        assert job3.dir.name == "J002"


def test_job_resume_missing_id_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="J099"):
        with Job("ghostbuster", project="p", base_dir=tmp_path, job_id="J099"):
            pass


def test_job_resume_loads_existing_params(tmp_path: Path) -> None:
    with Job("ghostbuster", project="p", base_dir=tmp_path) as job:
        job.log({"lr": 0.01})

    with Job("ghostbuster", project="p", base_dir=tmp_path, job_id="J001") as job:
        data = json.loads((job.dir / "job.json").read_text())
        assert data["params"]["lr"] == 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_jobs.py::test_job_resume_opens_existing_folder tests/test_jobs.py::test_job_resume_missing_id_raises -v
```
Expected: `FAILED` with `TypeError` (unexpected keyword `job_id`).

- [ ] **Step 3: Implement resume in `Job`**

In `src/specter/jobs/_job.py`, add `job_id` parameter and update `__init__` and `__enter__`:

```python
class Job:
    def __init__(
        self,
        job_type: str,
        project: str,
        base_dir: str | Path | None = None,
        job_id: str | None = None,
    ) -> None:
        self._job_type = job_type
        self._project = project
        self._base_dir = _resolve_base_dir(base_dir)
        self._resume_job_id = job_id
        self._dir: Path | None = None
        self._job_id: str | None = None
        self._created_at: str | None = None
        self._params: dict[str, Any] = {}

    # ... (dir property and __exit__ unchanged)

    def __enter__(self) -> Job:
        project_dir = self._base_dir / self._project
        project_dir.mkdir(parents=True, exist_ok=True)

        if self._resume_job_id is not None:
            self._job_id = self._resume_job_id
            self._dir = project_dir / self._job_id
            if not self._dir.exists():
                raise FileNotFoundError(
                    f"Job {self._resume_job_id!r} not found in project "
                    f"{self._project!r} (looked in {self._dir})"
                )
            existing = json.loads((self._dir / "job.json").read_text())
            self._params = existing.get("params", {})
            self._created_at = existing.get("created_at")
        else:
            self._job_id = _next_job_id(project_dir)
            self._dir = project_dir / self._job_id
            self._dir.mkdir()
            self._created_at = datetime.now(timezone.utc).isoformat()

        self._write_json("running")
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_jobs.py::test_job_resume_opens_existing_folder tests/test_jobs.py::test_job_resume_does_not_allocate_new_id tests/test_jobs.py::test_job_resume_missing_id_raises tests/test_jobs.py::test_job_resume_loads_existing_params -v
```
Expected: all `PASSED`.

- [ ] **Step 5: Run the full test suite to catch regressions**

```bash
python -m pytest tests/test_jobs.py -v
```
Expected: all existing tests still `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add src/specter/jobs/_job.py tests/test_jobs.py
git commit -m "feat: Job accepts job_id to resume an existing job folder"
```

---

## Task 3: Param exclusion and validation in `job.create()`

**Files:** Modify `src/specter/jobs/_job.py`, `tests/test_jobs.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_jobs.py`:

```python
# ---------------------------------------------------------------------------
# Task 8: job.create() param exclusion and resume validation
# ---------------------------------------------------------------------------


class _ExcludeClass:
    _job_log_exclude: tuple[str, ...] = ("secret",)

    def __init__(self, name: str, secret: str = "hidden") -> None:
        self.name = name
        self.secret = secret


def test_job_create_excludes_marked_params(tmp_path: Path) -> None:
    with Job("dummy", project="p", base_dir=tmp_path) as job:
        job.create(_ExcludeClass, "hello", secret="password")
    data = json.loads((job.dir / "job.json").read_text())
    assert "secret" not in data["params"]
    assert data["params"]["name"] == "hello"


def test_job_resume_create_matching_params_ok(tmp_path: Path) -> None:
    with Job("dummy", project="p", base_dir=tmp_path) as job:
        job.create(_DummyClass, "hello", value=3.14)

    # Same params — should not raise
    with Job("dummy", project="p", base_dir=tmp_path, job_id="J001") as job:
        job.create(_DummyClass, "hello", value=3.14)


def test_job_resume_create_mismatched_params_raises(tmp_path: Path) -> None:
    with Job("dummy", project="p", base_dir=tmp_path) as job:
        job.create(_DummyClass, "hello", value=3.14)

    with pytest.raises(ValueError, match="value"):
        with Job("dummy", project="p", base_dir=tmp_path, job_id="J001") as job:
            job.create(_DummyClass, "hello", value=99.0)


def test_job_resume_excluded_param_difference_not_checked(tmp_path: Path) -> None:
    with Job("dummy", project="p", base_dir=tmp_path) as job:
        job.create(_ExcludeClass, "hello", secret="password1")

    # Different secret — should not raise because secret is excluded
    with Job("dummy", project="p", base_dir=tmp_path, job_id="J001") as job:
        job.create(_ExcludeClass, "hello", secret="password2")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_jobs.py::test_job_create_excludes_marked_params tests/test_jobs.py::test_job_resume_create_mismatched_params_raises -v
```
Expected: `FAILED` — exclusion and validation logic not yet implemented.

- [ ] **Step 3: Update `job.create()` to support exclusion and validation**

Replace the `create` method in `src/specter/jobs/_job.py`:

```python
def create(self, cls: type, *args: Any, **kwargs: Any) -> Any:
    """
    Capture all constructor parameters via ``inspect`` and instantiate the class.

    If ``cls.__init__`` accepts a ``run_dir`` parameter, it is automatically
    set to ``self.dir`` — the user does not need to pass it explicitly.

    If ``cls`` defines ``_job_log_exclude`` (a tuple of parameter names), those
    keys are omitted from the recorded params in ``job.json``.

    In resume mode (``job_id`` was passed at construction), the incoming params
    (after exclusion) are validated against the stored params. A ``ValueError``
    is raised if they differ, preventing accidental halfset mismatches.

    Parameters
    ----------
    cls : type
        The SPECTER class to instantiate (e.g. ``Ghostbuster``).
    *args, **kwargs
        Forwarded to ``cls.__init__``.

    Returns
    -------
    instance
        The newly created object.
    """
    import inspect

    sig = inspect.signature(cls.__init__)  # type: ignore[misc]
    bound = sig.bind(None, *args, **kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    arguments.pop("self", None)

    if "run_dir" in sig.parameters:
        arguments["run_dir"] = self._dir

    exclude: frozenset[str] = frozenset(getattr(cls, "_job_log_exclude", ()))
    serialized = {
        k: _serialize_value(v)
        for k, v in arguments.items()
        if k not in exclude and k != "run_dir"
    }

    if self._resume_job_id is not None:
        mismatches = {
            k: (self._params.get(k), serialized.get(k))
            for k in set(self._params) | set(serialized)
            if self._params.get(k) != serialized.get(k)
        }
        if mismatches:
            diff_lines = "\n".join(
                f"  {k}: stored={a!r}  incoming={b!r}"
                for k, (a, b) in sorted(mismatches.items())
            )
            raise ValueError(
                f"Params mismatch for job {self._job_id!r}. "
                f"Both halfsets must use identical settings:\n{diff_lines}"
            )
    else:
        self._params.update(serialized)
        self._write_json("running")

    return cls(**arguments)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_jobs.py::test_job_create_excludes_marked_params tests/test_jobs.py::test_job_resume_create_matching_params_ok tests/test_jobs.py::test_job_resume_create_mismatched_params_raises tests/test_jobs.py::test_job_resume_excluded_param_difference_not_checked -v
```
Expected: all `PASSED`.

- [ ] **Step 5: Full test suite**

```bash
python -m pytest tests/test_jobs.py -v
```
Expected: all `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add src/specter/jobs/_job.py tests/test_jobs.py
git commit -m "feat: job.create() respects _job_log_exclude and validates params on resume"
```

---

## Task 4: Ghostbuster `_job_log_exclude` and auto `halfset_label`

**Files:** Modify `src/specter/ghostbuster.py`

No automated test — change is verified through the job integration test at the end.

- [ ] **Step 1: Add `_job_log_exclude` and derive `halfset_label` in `Ghostbuster.__init__`**

In `src/specter/ghostbuster.py`, add the class attribute immediately before `__init__` and derive `halfset_label` at the top of the constructor:

```python
class Ghostbuster:
    """...(existing docstring unchanged)..."""

    _job_log_exclude: tuple[str, ...] = ("return_class",)

    def __init__(
        self,
        # ... all existing params unchanged ...
        return_class: Literal["0", "1", "all"] = "all",
        run_dir: str | Path | None = None,
    ) -> None:
        from .cryosparc import extract_parameters_from_csfile

        _halfset_map: dict[str, str | None] = {"0": "A", "1": "B", "all": None}
        self.halfset_label: str | None = _halfset_map[return_class]

        # ... rest of __init__ unchanged ...
        self.run_dir = Path(run_dir) if run_dir is not None else None
```

- [ ] **Step 2: Pass `halfset_label` to `Reconstructor` in `_build_reconstructor_and_loader`**

In `_build_reconstructor_and_loader`, add `halfset_label=self.halfset_label` to the `Reconstructor(...)` call:

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
            run_dir=self.run_dir,
            halfset_label=self.halfset_label,  # NEW
        )
```

- [ ] **Step 3: Lint and type-check**

```bash
ruff check src/specter/ghostbuster.py
mypy src/specter/ghostbuster.py --ignore-missing-imports
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/specter/ghostbuster.py
git commit -m "feat: Ghostbuster auto-derives halfset_label from return_class, excludes it from job params"
```

---

## Task 5: Reconstructor per-epoch figure callbacks

**Files:** Modify `src/specter/ghostbuster.py`

No automated test — callbacks verified by running a training loop in a notebook.

- [ ] **Step 1: Update `on_train_epoch_end` to save plot3d and FSC figures**

Replace the existing `on_train_epoch_end` method in `Reconstructor`:

```python
    def on_train_epoch_end(self) -> None:
        """Enforce symmetry, save per-epoch volume, plot3d preview, and FSC."""
        if self.symmetry is not None:
            self.V.data = apply_symmetry(
                self.V.data,
                self.sym_rot_matrices,
                batchsize=self.symmetry_batchsize,
                method=self.symmetry_mode,
            )

        if self._run_dir is None:
            return

        epoch = self.current_epoch + 1
        suffix = f"_{self._halfset_label}" if self._halfset_label is not None else ""
        v = self.V.detach().cpu().float()

        # --- volume MRC (existing behaviour) ---
        mrc_path = self._run_dir / "epochs" / f"{epoch:03d}{suffix}.mrc"
        with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
            mrc.set_data(v.numpy())

        # --- plot3d preview ---
        try:
            import matplotlib.pyplot as plt
            from .plots import plot3d

            fig = plot3d(v, title=f"Epoch {epoch}{suffix}", show=False)
            fig.savefig(
                self._run_dir / "epochs" / f"{epoch:03d}{suffix}.png",
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as exc:
            print(f"[Reconstructor] plot3d preview skipped: {exc}")

        # --- FSC plot ---
        if self.fsc_ref is not None:
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
                fig = plot_map_to_model_fsc(
                    [v],
                    fsc_ref,
                    voxel_size=self.voxel_size,
                    mask=fsc_mask,
                    labels=[f"epoch {epoch}{suffix}"],
                )
                fig.savefig(
                    self._run_dir / "epochs" / f"fsc_{epoch:03d}{suffix}.png",
                    bbox_inches="tight",
                )
                plt.close(fig)
            except Exception as exc:
                print(f"[Reconstructor] FSC plot skipped: {exc}")
```

- [ ] **Step 2: Update `on_fit_end` to save a final FSC figure**

Replace the existing `on_fit_end` method:

```python
    def on_fit_end(self) -> None:
        """Save the final reconstructed volume and FSC figure."""
        if self._run_dir is None:
            return

        suffix = f"_{self._halfset_label}" if self._halfset_label is not None else ""
        v = self.V.detach().cpu().float()

        # --- final volume MRC (existing behaviour) ---
        vol_path = self._run_dir / f"vol{suffix}.mrc"
        with mrcfile.new(str(vol_path), overwrite=True) as mrc:
            mrc.set_data(v.numpy())
        print(f"Saved final volume → {vol_path}")

        # --- final FSC figure ---
        if self.fsc_ref is not None:
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
                fig = plot_map_to_model_fsc(
                    [v],
                    fsc_ref,
                    voxel_size=self.voxel_size,
                    mask=fsc_mask,
                    labels=[f"final{suffix}"],
                )
                fsc_path = self._run_dir / f"fsc{suffix}.png"
                fig.savefig(fsc_path, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved final FSC → {fsc_path}")
            except Exception as exc:
                print(f"[Reconstructor] Final FSC plot skipped: {exc}")
```

- [ ] **Step 3: Lint and type-check**

```bash
ruff check src/specter/ghostbuster.py
mypy src/specter/ghostbuster.py --ignore-missing-imports
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/specter/ghostbuster.py
git commit -m "feat: Reconstructor saves plot3d preview and FSC figure after each epoch"
```

---

## Full Usage Pattern (for reference)

After all tasks are complete, the intended usage is:

```python
from specter.jobs import Job
from specter.ghostbuster import Ghostbuster

# PBS script A — runs halfset A, prints "Job folder: .../J001"
with Job("ghostbuster", "my-project") as job:
    gb = job.create(
        Ghostbuster,
        cs_file="particles.cs",
        mrc_file="stack.mrcs",
        dose_per_angstrom=40.0,
        lr=1e-3,
        symmetry="C1",
        epochs=10,
        return_class="0",          # → halfset_label "A", excluded from job.json
        fsc_ref=atomic_model_vol,  # optional: enables per-epoch FSC plots
    )
    print(f"Job folder: {job.dir}")
    gb.run(device=0)

# PBS script B — reopens the same folder, validates params match
with Job("ghostbuster", "my-project", job_id="J001") as job:
    gb = job.create(
        Ghostbuster,
        cs_file="particles.cs",
        mrc_file="stack.mrcs",
        dose_per_angstrom=40.0,
        lr=1e-3,
        symmetry="C1",
        epochs=10,
        return_class="1",   # different — but excluded, so no validation error
        fsc_ref=atomic_model_vol,
    )
    gb.run(device=0)

# J001/ folder ends up with:
#   job.json           — shared params (return_class excluded)
#   vol_A.mrc, vol_B.mrc
#   fsc_A.png, fsc_B.png
#   epochs/001_A.mrc, 001_A.png, fsc_001_A.png, ...
#   epochs/001_B.mrc, 001_B.png, fsc_001_B.png, ...
```
