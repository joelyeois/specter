# Icemaker Refactor Design

**Date:** 2026-04-28
**Status:** Approved

## Motivation

`src/specter/icemaker.py` is ~3050 lines containing 6 classes with three problems:

1. **Code duplication** — three `generate_big_ice*` variants in `Icemaker`, and near-identical L-BFGS/Adam optimizer loops in `GradientSKIcemaker.optimize`.
2. **Awkward coupling** — `GradientSKIcemaker.__init__` instantiates a throwaway `Icemaker` just to extract grid params, the MD simulation radial average, and the ice kernel.
3. **Poor organisation** — a 3050-line file covering four distinct algorithms is hard to navigate.

## Package Layout

`src/specter/icemaker.py` becomes `src/specter/icemaker/`. All public names are re-exported from `__init__.py` — the import path `from specter.icemaker import ...` continues to work. However, two class names change (`Icemaker` → `APicemaker`, `GradientSKIcemaker` → `GradientDescentIcemaker`), so any code importing those by the old name will need updating.

```
src/specter/icemaker/
    __init__.py                  # re-exports all public names
    _constants.py                # ndensity_of_amorphous_ice, avogadro, molar_mass_of_water
    _kernel.py                   # IceKernelData dataclass
    _utils.py                    # rfftn, irfftn, torch_peak_local_max, water utils,
                                 #   replace_outer_faces, _wrap_coords
    alternating_projections.py   # APicemaker (was Icemaker)
    gradient_sk.py               # GradientDescentIcemaker (was GradientSKIcemaker)
    mcmc.py                      # MCMCIcemaker
    naive.py                     # NaiveIcemaker
    mdsim.py                     # MDSimDump
    bank.py                      # IceBank
```

## Class Renames

| Old name | New name | File |
|---|---|---|
| `Icemaker` | `APicemaker` | `alternating_projections.py` |
| `GradientSKIcemaker` | `GradientDescentIcemaker` | `gradient_sk.py` |
| `MCMCIcemaker` | `MCMCIcemaker` | `mcmc.py` |
| `NaiveIcemaker` | `NaiveIcemaker` | `naive.py` |
| `MDSimDump` | `MDSimDump` | `mdsim.py` |
| `IceBank` | `IceBank` | `bank.py` |

`IceBank._METHOD_MAP` key changes from `'gs'` to `'ap'`.

`MCMCIcemaker.init_from_icemaker` renamed to `init_from_apicemaker`; type hint updated to `APicemaker`.

## Section 1 — `IceKernelData` (`_kernel.py`)

Fixes the `GradientDescentIcemaker` coupling. Holds everything both algorithm classes need:

```python
@dataclass
class IceKernelData:
    n: int
    nz: int
    dx: float
    dk: float
    n_ice_molecules: int
    K: torch.Tensor                   # (nz, n, n) radial freq grid
    mdsim_radial_k: torch.Tensor      # 1D — from ice-data/
    mdsim_f_radial_avg: torch.Tensor  # 1D — from ice-data/
    ice_kernel: torch.Tensor          # (nz, n, n) atomic potential kernel

    @classmethod
    def build(
        cls,
        n: int,
        dx: float,
        nz: int | None = None,
        parameterization: str = "kirkland",
        min_distance: float = 1.9,
        correction_factor: float | None = None,
    ) -> "IceKernelData": ...
```

`APicemaker.__init__` and `GradientDescentIcemaker.__init__` both call `IceKernelData.build(...)` and store the result. `GradientDescentIcemaker` no longer instantiates `APicemaker`. Each class derives its own algorithm-specific quantities from the shared `IceKernelData`.

## Section 2 — `APicemaker` (`alternating_projections.py`)

### 2a — `register_buffer` fix in `generate_ice_deltas`

`ice_vol` and `batch_idx` are currently registered as Lightning buffers inside the iteration loop. They are ephemeral locals and must not be buffers. Changed to plain tensor locals. `current_icedeltas` remains a buffer as it is the meaningful output read by callers after the method returns.

### 2b — `generate_big_ice*` deprecated

The three `generate_big_ice` variants are superseded by `IceBank.generate_big_ice`. They become deprecated stubs:

```python
def generate_big_ice(self, shape, num_unique=8, bin_factor=1):
    warnings.warn(
        "generate_big_ice is deprecated. Use IceBank.generate_big_ice() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

def generate_big_ice_fast(self, shape, num_unique=8, bin_factor=1):
    warnings.warn(
        "generate_big_ice_fast is deprecated. Use IceBank.generate_big_ice() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

def generate_big_ice_interpolate(self, shape, n_blocks=8, algorithm_dx=0.5):
    warnings.warn(
        "generate_big_ice_interpolate is deprecated. Use IceBank.generate_big_ice() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
```

Old implementation bodies are commented out.

## Section 3 — `GradientDescentIcemaker` (`gradient_sk.py`)

### 3a — Adam removed

`optimize()` drops the `optimizer` parameter. L-BFGS is the only supported optimizer (found to be superior in practice).

```python
def optimize(
    self,
    n_steps: int = 50,
    lr: float = 1.0,
    record_every: int = 5,
    rep_strength: float = 1.0,
) -> dict: ...
```

Adam code is removed.

### 3b — Shared optimizer loop

`optimize()` and `sample()` share an identical loop structure (progress bar, coord wrapping, recording). Extracted to a private helper:

```python
def _run_opt_loop(
    self,
    pos: torch.Tensor,
    opt: torch.optim.Optimizer,
    step_fn: Callable[[], torch.Tensor],
    n_steps: int,
    record_every: int,
    rep_strength: float,
    desc: str,
) -> dict: ...
```

`optimize()` and `sample()` both delegate to `_run_opt_loop`.

## Usage Pattern in `imagegenerator.py`

`imagegenerator.py` accepts any object with `generate_ice(batchsize)` / `generate_big_ice(shape)` — duck-typed against a Protocol. `IceBank` is the recommended entry point:

```
imagegenerator.py  ←  IceBank(method='ap'|'gd'|'mcmc')
                            ↑
                   APicemaker / GradientDescentIcemaker / MCMCIcemaker
```

Direct class usage remains available for cases requiring algorithm-specific control (e.g. `MCMCIcemaker` requires `set_target_gr_from_md` before generating).

## What Does Not Change

- Import path `from specter.icemaker import ...` — package structure preserved by `__init__.py`
- `generate_ice` signatures on all classes
- `IceBank.build` / `IceBank.generate_ice` / `IceBank.generate_big_ice`
- Physics — no algorithm logic is modified
- `NaiveIcemaker`, `MCMCIcemaker`, `MDSimDump` — moved to their own files, no logic changes
