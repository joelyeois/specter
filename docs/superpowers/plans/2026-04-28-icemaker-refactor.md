# Icemaker Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `src/specter/icemaker.py` (~3050 lines, 6 classes) into a `specter/icemaker/` package, fix the `GradientDescentIcemaker` coupling via a shared `IceKernelData` dataclass, deprecate `generate_big_ice*` on `APicemaker`, and remove Adam from `GradientDescentIcemaker`.

**Architecture:** Each class gets its own module. Shared grid/kernel state is extracted into `IceKernelData.build()` in `_kernel.py`, eliminating the `GradientDescentIcemaker` → `Icemaker` instantiation hack. `__init__.py` re-exports all public names so callers see no change to import paths.

**Tech Stack:** Python 3.11+, PyTorch, Lightning, `torchinterp1d`, `ruff`, `mypy`, `pytest`

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/specter/icemaker/__init__.py` | Re-exports all public names |
| Create | `src/specter/icemaker/_constants.py` | Physical constants |
| Create | `src/specter/icemaker/_kernel.py` | `IceKernelData`, `_load_mdsim_data`, `_build_ice_kernel` |
| Create | `src/specter/icemaker/_utils.py` | `rfftn`, `irfftn`, `torch_peak_local_max`, water utils, `replace_outer_faces`, `_wrap_coords` |
| Create | `src/specter/icemaker/alternating_projections.py` | `APicemaker` (was `Icemaker`) |
| Create | `src/specter/icemaker/naive.py` | `NaiveIcemaker` |
| Create | `src/specter/icemaker/mcmc.py` | `MCMCIcemaker` |
| Create | `src/specter/icemaker/gradient_sk.py` | `GradientDescentIcemaker` (was `GradientSKIcemaker`) |
| Create | `src/specter/icemaker/mdsim.py` | `MDSimDump` |
| Create | `src/specter/icemaker/bank.py` | `IceBank` |
| Delete | `src/specter/icemaker.py` | Replaced by package |
| Modify | `src/specter/imagegenerator.py` | Update class name imports |
| Modify | `src/specter/specimen.py` | Update class name imports |
| Create | `tests/test_icemaker_refactor.py` | Smoke tests for new package |

---

### Task 1: Package scaffold + `_constants.py`

**Files:**
- Create: `src/specter/icemaker/_constants.py`
- Create: `src/specter/icemaker/__init__.py` (stub)
- Create: `tests/test_icemaker_refactor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_icemaker_refactor.py
from specter.icemaker._constants import ndensity_of_amorphous_ice


def test_ndensity_value():
    # 0.94 g/cm³ × 6.022e23/mol ÷ 18.015 g/mol × 1e-24 cm³/Å³ ≈ 0.0314 /Å³
    assert 0.030 < ndensity_of_amorphous_ice < 0.035
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
python -m pytest tests/test_icemaker_refactor.py::test_ndensity_value -v
```

Expected: `FAILED` (ModuleNotFoundError — package doesn't exist yet)

- [ ] **Step 3: Create `src/specter/icemaker/_constants.py`**

```python
from __future__ import annotations

avogadro = 6.02214076e23
density_of_amorphous_ice = 0.94  # g/cm³
molar_mass_of_water = 18.01528   # g/mol
ndensity_of_amorphous_ice = (
    density_of_amorphous_ice * avogadro / molar_mass_of_water * 1e-24
)  # particles / Å³
```

- [ ] **Step 4: Create stub `src/specter/icemaker/__init__.py`**

```python
# populated in Task 10
```

- [ ] **Step 5: Run test — expect PASS**

```bash
python -m pytest tests/test_icemaker_refactor.py::test_ndensity_value -v
```

- [ ] **Step 6: Commit**

```bash
git add src/specter/icemaker/ tests/test_icemaker_refactor.py
git commit -m "refactor: scaffold icemaker package, add _constants.py"
```

---

### Task 2: `_utils.py`

**Files:**
- Create: `src/specter/icemaker/_utils.py`
- Modify: `tests/test_icemaker_refactor.py`

Move from `src/specter/icemaker.py`: `rfftn` (line 143), `torch_peak_local_max` (line 167), `water_molecule_coordinates` (line 36), `create_n_randomly_rotated_water_molecules` (line 67), `volume_of_ice` (line 103), `replace_outer_faces` (line 3006), `_wrap_coords` (line 991).

Add `irfftn` as a new standalone function (was `Icemaker.irfftn` method at line 658 — converted to take `n` and `nz` as explicit parameters instead of reading from `self`).

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/test_icemaker_refactor.py
import torch
from specter.icemaker._utils import (
    rfftn,
    irfftn,
    torch_peak_local_max,
    replace_outer_faces,
    _wrap_coords,
)


def test_rfftn_shape():
    x = torch.zeros(8, 16, 16)
    out = rfftn(x)
    assert out.shape == (8, 16, 9)  # last dim: 16//2 + 1 = 9
    assert out.is_complex()


def test_rfftn_irfftn_roundtrip():
    x = torch.randn(8, 16, 16)
    reconstructed = irfftn(rfftn(x), n=16, nz=8)
    assert torch.allclose(x, reconstructed, atol=1e-5)


def test_torch_peak_local_max_shape():
    x = torch.zeros(2, 8, 8, 8)
    x[0, 4, 4, 4] = 1.0
    x[1, 2, 2, 2] = 1.0
    peaks = torch_peak_local_max(x, min_distance=1, num_peaks=1)
    assert peaks.shape == (2, 1, 3)


def test_replace_outer_faces_modifies_boundaries():
    t = torch.ones(2, 10, 10, 10)
    t[:, 0, :, :] = 0.0  # zero out a face
    result = replace_outer_faces(t)
    # outer face should no longer be zero
    assert result[:, 0, :, :].sum() > 0


def test_wrap_coords():
    from specter.icemaker._utils import _wrap_coords
    x = torch.tensor([6.0, -6.0, 0.0])
    wrapped = _wrap_coords(x, L=10.0)
    assert torch.allclose(wrapped, torch.tensor([-4.0, 4.0, 0.0]))
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "rfftn or irfftn or peak_local or outer_faces or wrap_coords" -v
```

Expected: `FAILED` (ImportError)

- [ ] **Step 3: Create `src/specter/icemaker/_utils.py`**

```python
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any

from .. import rotations
from ._constants import ndensity_of_amorphous_ice


def rfftn(array: torch.Tensor) -> torch.Tensor:
    """Compute N-dimensional real-input Fourier transform with centering."""
    return torch.fft.fftshift(
        torch.fft.rfftn(
            torch.fft.ifftshift(array, dim=(-3, -2, -1)), dim=(-3, -2, -1)
        ),
        dim=(-3, -2),
    )


def irfftn(array: torch.Tensor, n: int, nz: int) -> torch.Tensor:
    """Compute inverse N-dimensional real Fourier transform with centering."""
    return torch.fft.fftshift(
        torch.fft.irfftn(
            torch.fft.ifftshift(array, dim=(-3, -2)),
            s=(nz, n, n),
            dim=(-3, -2, -1),
        ),
        dim=(-3, -2, -1),
    )


def torch_peak_local_max(
    image: torch.Tensor, min_distance: int = 1, num_peaks: int | None = None
) -> torch.LongTensor:
    """Find local maxima in batched 3D images and return fixed number of peaks per batch."""
    B, D, H, W = image.shape
    x = image.unsqueeze(1)
    k = 2 * min_distance + 1
    pooled = F.max_pool3d(x, kernel_size=k, stride=1, padding=min_distance)
    mask = (x == pooled).squeeze(1)

    flat_mask = mask.view(B, -1)
    flat_image = image.view(B, -1)
    flat_image_masked = flat_image.clone()
    flat_image_masked[~flat_mask] = -float("inf")

    if num_peaks is None:
        num_peaks = int(flat_mask.sum(dim=1).min().item())

    _, topk_idx = flat_image_masked.topk(num_peaks, dim=1)
    z = topk_idx // (H * W)
    y = (topk_idx % (H * W)) // W
    x_ = topk_idx % W
    return torch.stack([z, y, x_], dim=2)


def water_molecule_coordinates(
    bond_angle: float = 105.0, bond_length: float = 0.9572
) -> tuple[np.ndarray, torch.Tensor]:
    """Return coordinates of a single H2O molecule with oxygen at the origin."""
    O_xyz = torch.tensor([0.0, 0.0, 0.0])
    angle_rad = torch.tensor(bond_angle) / 180 * torch.pi
    y = bond_length * torch.cos(angle_rad / 2)
    x = bond_length * torch.sin(angle_rad / 2)
    H1_xyz = torch.tensor([x, y, 0.0])
    H2_xyz = torch.tensor([-x, y, 0.0])
    coordinates = torch.stack((O_xyz, H1_xyz, H2_xyz))
    return np.array([8, 1, 1]), coordinates


def create_n_randomly_rotated_water_molecules(
    n: int, **kwargs: Any
) -> tuple[np.ndarray, torch.Tensor]:
    """Create n randomly rotated water molecules."""
    quats = torch.stack([rotations.random_quaternion() for _ in range(n)])
    atomic_numbers, coords = water_molecule_coordinates(**kwargs)
    O_coordinates = coords[0].repeat(n, 1)
    H1_coordinates = rotations.rotate_coordinates(coords[1], quats)
    H2_coordinates = rotations.rotate_coordinates(coords[2], quats)
    all_coords = torch.zeros(n * 3, 3)
    all_coords[0::3] = O_coordinates
    all_coords[1::3] = H1_coordinates
    all_coords[2::3] = H2_coordinates
    return np.array([8, 1, 1] * n), all_coords


def volume_of_ice(
    n_xyz: tuple[int, int, int], d_xyz: tuple[float, float, float]
) -> tuple[np.ndarray, torch.Tensor]:
    """Generate atomic coordinates for a volume of amorphous ice."""
    nx, ny, nz = n_xyz
    dx, dy, dz = d_xyz
    total_vol = nx * ny * nz * dx * dy * dz
    n_molecules = int(ndensity_of_amorphous_ice * total_vol)
    x_ice = (torch.rand(n_molecules) - 0.5) * dx * nx
    y_ice = (torch.rand(n_molecules) - 0.5) * dy * ny
    z_ice = (torch.rand(n_molecules) - 0.5) * dz * nz
    centers = torch.repeat_interleave(
        torch.stack((x_ice, y_ice, z_ice), dim=1), 3, dim=0
    )
    atomic_numbers, coords = create_n_randomly_rotated_water_molecules(n_molecules)
    coords += centers
    return atomic_numbers, coords


def replace_outer_faces(tensors: torch.Tensor) -> torch.Tensor:
    """Replace the 6 outer faces of a batch of 3D tensors with random inner slices."""
    N, D, H, W = tensors.shape
    if D <= 2 or H <= 2 or W <= 2:
        raise ValueError("Tensor too small to have inner slices")
    z_idx = torch.randint(1, D - 1, (N,))
    y_idx = torch.randint(1, H - 1, (N,))
    x_idx = torch.randint(1, W - 1, (N,))
    batch_idx = torch.arange(N)
    tensors[batch_idx, 0, :, :] = tensors[batch_idx, z_idx, :, :]
    tensors[batch_idx, -1, :, :] = tensors[batch_idx, z_idx, :, :]
    tensors[batch_idx, :, 0, :] = tensors[batch_idx, :, y_idx, :]
    tensors[batch_idx, :, -1, :] = tensors[batch_idx, :, y_idx, :]
    tensors[batch_idx, :, :, 0] = tensors[batch_idx, :, :, x_idx]
    tensors[batch_idx, :, :, -1] = tensors[batch_idx, :, :, x_idx]
    return tensors


def _wrap_coords(x: torch.Tensor, L: float) -> torch.Tensor:
    """Wrap coordinates into [-L/2, L/2)."""
    return (x + L / 2) % L - L / 2
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "rfftn or irfftn or peak_local or outer_faces or wrap_coords" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/specter/icemaker/_utils.py tests/test_icemaker_refactor.py
git commit -m "refactor: add icemaker/_utils.py with shared utility functions"
```

---

### Task 3: `_kernel.py` — `IceKernelData`

**Files:**
- Create: `src/specter/icemaker/_kernel.py`
- Modify: `tests/test_icemaker_refactor.py`

This is the core of the coupling fix. `IceKernelData.build()` replaces the logic currently spread across `Icemaker.__init__` and the throwaway `_im = Icemaker(...)` in `GradientSKIcemaker.__init__`.

**Important:** The ice-data path changes because `_kernel.py` is now one level deeper than the original `icemaker.py`. The path needs 3 `dirname` calls instead of 2 to reach the project root.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_icemaker_refactor.py
from specter.icemaker._kernel import IceKernelData


def test_ice_kernel_data_build_shapes():
    kd = IceKernelData.build(n=16, dx=1.0, nz=8)
    assert kd.n == 16
    assert kd.nz == 8
    assert kd.K.shape == (8, 16, 16)
    assert kd.mdsim_radial_k.ndim == 1
    assert kd.mdsim_f_radial_avg.ndim == 1
    assert kd.ice_kernel.ndim == 3
    assert kd.n_ice_molecules > 0


def test_ice_kernel_data_nz_defaults_to_n():
    kd = IceKernelData.build(n=16, dx=1.0)
    assert kd.nz == 16
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "kernel_data" -v
```

Expected: `FAILED` (ImportError)

- [ ] **Step 3: Create `src/specter/icemaker/_kernel.py`**

```python
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from ._constants import ndensity_of_amorphous_ice
from ..arrays import radial_grid_3d, real_to_kgrid_3d
from ..atom import kirkland_atomic_potential_3d, lobato_atomic_potential_3d
from ..fft import fft3
from .. import potential as _potential_mod


@dataclass
class IceKernelData:
    """
    Shared grid and kernel state for ice generation algorithms.

    Parameters
    ----------
    n : int
        Number of voxels along x and y.
    nz : int
        Number of voxels along z.
    dx : float
        Voxel size in Å.
    dk : float
        Frequency step in Å⁻¹.
    n_ice_molecules : int
        Target molecule count after density and exclusion-radius correction.
    K : torch.Tensor
        Radial spatial-frequency grid, shape (nz, n, n).
    mdsim_radial_k : torch.Tensor
        1-D radial frequency values from the MD simulation reference.
    mdsim_f_radial_avg : torch.Tensor
        1-D Fourier amplitude radial average from the MD simulation reference.
    ice_kernel : torch.Tensor
        Atomic potential kernel for oxygen, shape determined by dx.
    """

    n: int
    nz: int
    dx: float
    dk: float
    n_ice_molecules: int
    K: torch.Tensor
    mdsim_radial_k: torch.Tensor
    mdsim_f_radial_avg: torch.Tensor
    ice_kernel: torch.Tensor

    @classmethod
    def build(
        cls,
        n: int,
        dx: float,
        nz: int | None = None,
        parameterization: str = "kirkland",
        min_distance: float = 1.9,
        correction_factor: float | None = None,
    ) -> "IceKernelData":
        """
        Construct IceKernelData for the given grid and physical parameters.

        Parameters
        ----------
        n : int
            Number of voxels along x and y.
        dx : float
            Voxel size in Å.
        nz : int, optional
            Number of voxels along z. Defaults to n.
        parameterization : str, optional
            Atomic potential model: 'kirkland', 'lobato', or 'shtyrov'. Default 'kirkland'.
        min_distance : float, optional
            Minimum O-O distance in Å used for molecule-count correction. Default 1.9.
        correction_factor : float, optional
            Override the computed correction factor. Computed from min_distance if None.

        Returns
        -------
        IceKernelData
        """
        nz = nz if nz is not None else n
        dk = 1.0 / n / dx
        dv = dx**3
        nv = n**2 * nz
        v = dv * nv

        min_distance_vox = int(min_distance / dx)
        min_distance_actual = min_distance_vox * dx
        if correction_factor is None:
            if min_distance_actual == 0.0:
                correction_factor = 1.0
            else:
                correction_factor = (min_distance / min_distance_actual) ** 3
        n_ice_molecules = int(ndensity_of_amorphous_ice * v / correction_factor)

        kx = torch.fft.fftshift(torch.fft.fftfreq(n, dx))
        kz = torch.fft.fftshift(torch.fft.fftfreq(nz, dx))
        KZ, KY, KX = torch.meshgrid(kz, kx, kx, indexing="ij")
        K = torch.sqrt(KX**2 + KY**2 + KZ**2)

        mdsim_radial_k, mdsim_f_radial_avg = _load_mdsim_data()
        ice_kernel = _build_ice_kernel(dx, parameterization)

        return cls(
            n=n,
            nz=nz,
            dx=dx,
            dk=dk,
            n_ice_molecules=n_ice_molecules,
            K=K,
            mdsim_radial_k=mdsim_radial_k,
            mdsim_f_radial_avg=mdsim_f_radial_avg,
            ice_kernel=ice_kernel,
        )


def _load_mdsim_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Load the precomputed MD simulation Fourier amplitude radial average from ice-data/."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # _kernel.py is at src/specter/icemaker/_kernel.py — 3 dirname calls reach project root
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    saved_data_path = os.path.join(
        root_dir, "ice-data", "mdsim_f_radial_avg_400x400x400_0.25A.pt"
    )
    mdsim_f_radial_avg = torch.load(saved_data_path)
    mdsim_dk = 1.0 / 400 / 0.25  # mdsim_n=400, mdsim_dx=0.25 Å
    mdsim_radial_k = torch.arange(len(mdsim_f_radial_avg)) * mdsim_dk
    return mdsim_radial_k, mdsim_f_radial_avg


def _build_ice_kernel(dx: float, parameterization: str = "kirkland") -> torch.Tensor:
    """Build the atomic scattering potential kernel for oxygen at the given voxel size."""
    ssn, ssdx, ssf = _potential_mod.compute_supersampling_parameters(dx)
    sR = radial_grid_3d(ssn, ssdx, convention="torch")
    avgpool3d = torch.nn.AvgPool3d(ssf, stride=ssf)

    if parameterization == "kirkland":
        pot = kirkland_atomic_potential_3d(8, sR)
    elif parameterization == "lobato":
        pot = lobato_atomic_potential_3d(8, sR)
    elif parameterization == "shtyrov":
        params = torch.tensor([
            [0.3131, 0.8722],
            [0.8102, 4.9669],
            [0.9812, 14.1666],
            [-0.5997, 64.1638],
            [-0.1519, 121.3711],
        ])
        a = params[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        b = params[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        k_xyz = real_to_kgrid_3d(sR)
        k2 = k_xyz**2
        k2 = k2.unsqueeze(0)
        s1_f = torch.sum(a * torch.exp(-b * k2 / 4), 0)
        dkx = k_xyz[1, 0, 0] - k_xyz[0, 0, 0]
        dky = k_xyz[0, 1, 0] - k_xyz[0, 0, 0]
        dkz = k_xyz[0, 0, 1] - k_xyz[0, 0, 0]
        pot = -torch.abs(fft3(s1_f, shift=True)) * dkx * dky * dkz
    else:
        raise ValueError(
            f"Unknown parameterization '{parameterization}'. "
            "Choose 'kirkland', 'lobato', or 'shtyrov'."
        )

    return avgpool3d(pot[None, None]).squeeze()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "kernel_data" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/specter/icemaker/_kernel.py tests/test_icemaker_refactor.py
git commit -m "refactor: add IceKernelData dataclass to icemaker/_kernel.py"
```

---

### Task 4: `naive.py` — `NaiveIcemaker`

**Files:**
- Create: `src/specter/icemaker/naive.py`
- Modify: `tests/test_icemaker_refactor.py`

Copy `NaiveIcemaker` verbatim from `src/specter/icemaker.py` lines 2718–2846. Update imports to use the new package-relative paths.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_icemaker_refactor.py
from specter.icemaker.naive import NaiveIcemaker


def test_naive_icemaker_generate_ice_shape():
    im = NaiveIcemaker(dx=1.0, n=16, nz=8)
    ice = im.generate_ice(batchsize=2)
    assert ice.shape == (2, 8, 16, 16)
    assert ice.dtype == torch.float32
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
python -m pytest tests/test_icemaker_refactor.py::test_naive_icemaker_generate_ice_shape -v
```

- [ ] **Step 3: Create `src/specter/icemaker/naive.py`**

Copy the `NaiveIcemaker` class from `src/specter/icemaker.py` (lines 2718–2846). Replace the top-level imports with:

```python
from __future__ import annotations

import torch
import lightning as L

from ..fft import fftconvolve
from ._constants import ndensity_of_amorphous_ice
```

`NaiveIcemaker.create_ice_kernel` uses its own inline Kirkland implementation (hardcoded `sn=28`, `AvgPool3d(4, stride=4)`) — it does not call `_build_ice_kernel` from `_kernel.py`. Copy the method body verbatim.

- [ ] **Step 4: Run test — expect PASS**

```bash
python -m pytest tests/test_icemaker_refactor.py::test_naive_icemaker_generate_ice_shape -v
```

- [ ] **Step 5: Commit**

```bash
git add src/specter/icemaker/naive.py tests/test_icemaker_refactor.py
git commit -m "refactor: move NaiveIcemaker to icemaker/naive.py"
```

---

### Task 5: `mdsim.py` — `MDSimDump`

**Files:**
- Create: `src/specter/icemaker/mdsim.py`
- Modify: `tests/test_icemaker_refactor.py`

Copy `MDSimDump` verbatim from `src/specter/icemaker.py` lines 2318–2715. No logic changes.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_icemaker_refactor.py
from specter.icemaker.mdsim import MDSimDump


def test_mdsim_dump_importable():
    # Just verifies the class can be imported and has expected attributes
    assert hasattr(MDSimDump, "_HEADER_LINES")
    assert MDSimDump._HEADER_LINES == 9
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
python -m pytest tests/test_icemaker_refactor.py::test_mdsim_dump_importable -v
```

- [ ] **Step 3: Create `src/specter/icemaker/mdsim.py`**

Copy `MDSimDump` from `src/specter/icemaker.py` (lines 2318–2715). Replace top-level imports with:

```python
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import torch

from ..arrays import soft_voxelize_coordinates
from ..fft import fft3
from ..arrays import radial_profile_3d
from ..coords import radial_distribution_function
```

- [ ] **Step 4: Run test — expect PASS**

```bash
python -m pytest tests/test_icemaker_refactor.py::test_mdsim_dump_importable -v
```

- [ ] **Step 5: Commit**

```bash
git add src/specter/icemaker/mdsim.py tests/test_icemaker_refactor.py
git commit -m "refactor: move MDSimDump to icemaker/mdsim.py"
```

---

### Task 6: `alternating_projections.py` — `APicemaker`

**Files:**
- Create: `src/specter/icemaker/alternating_projections.py`
- Modify: `tests/test_icemaker_refactor.py`

This is the largest change. Key differences from the original `Icemaker`:

1. **Class renamed** `Icemaker` → `APicemaker`
2. **`__init__`** calls `IceKernelData.build()` instead of loading MD data and building the kernel inline
3. **`generate_ice_deltas`**: `ice_vol` and `batch_idx` become plain locals (not `register_buffer`); `self.irfftn(...)` → `irfftn(..., self.n, self.nz)` from `_utils`
4. **`create_ice_kernel` method removed** (functionality now in `_kernel._build_ice_kernel`)
5. **`irfftn` method removed** (now standalone `irfftn` in `_utils`)
6. **`generate_big_ice*` methods** become deprecation stubs

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_icemaker_refactor.py
from specter.icemaker.alternating_projections import APicemaker


def test_apicemaker_generate_ice_shape():
    im = APicemaker(dx=1.0, n=16, nz=8)
    ice = im.generate_ice(batchsize=2)
    assert ice.shape == (2, 8, 16, 16)


def test_apicemaker_big_ice_deprecated():
    import warnings
    im = APicemaker(dx=1.0, n=16, nz=8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        im.generate_big_ice(shape=(1, 8, 16, 16))
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "IceBank" in str(w[0].message)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "apicemaker" -v
```

- [ ] **Step 3: Create `src/specter/icemaker/alternating_projections.py`**

Start by copying `Icemaker` from `src/specter/icemaker.py` (lines 217–988). Then apply the following changes:

**Imports (replace top of file):**

```python
from __future__ import annotations

import warnings
from typing import Sequence, Any

import lightning as L
import torch
import torch.nn.functional as F
from torchinterp1d import interp1d

from ..progress import ProgressManager, track
from ..arrays import radial_profile_3d, soft_voxelize_coordinates, tile_volume_from_blocks
from ..fft import fftconvolve
from ._constants import ndensity_of_amorphous_ice
from ._kernel import IceKernelData
from ._utils import rfftn, irfftn, torch_peak_local_max, replace_outer_faces
```

**`__init__` — replace the body with:**

```python
def __init__(
    self,
    dx: float = 0.5,
    n: int = 200,
    nz: int | None = None,
    chunk_size: int | None = None,
    progressbars: bool = True,
    parameterization: str = "kirkland",
    min_distance: float = 1.9,
    correction_factor: float | None = None,
):
    super().__init__()

    kd = IceKernelData.build(
        n=n,
        dx=dx,
        nz=nz,
        parameterization=parameterization,
        min_distance=min_distance,
        correction_factor=correction_factor,
    )

    self.chunk_size = chunk_size
    self.progressbars = progressbars
    self.parameterization = parameterization
    self.min_distance = min_distance

    self.nz = kd.nz
    self.dx = kd.dx
    self.dk = kd.dk
    self.n = kd.n
    self.dv = dx**3
    self.nv = n**2 * kd.nz
    self.v = self.dv * self.nv
    self.n_ice_molecules_theory = int(ndensity_of_amorphous_ice * self.v)
    self.n_ice_molecules = kd.n_ice_molecules

    # MD simulation reference data (stored as buffers for Lightning device dispatch)
    self.mdsim_dx = 0.25
    self.mdsim_n = 400
    self.mdsim_dk = 1.0 / 400 / 0.25
    self.register_buffer("mdsim_f_radial_avg", kd.mdsim_f_radial_avg)
    self.register_buffer("mdsim_radial_k", kd.mdsim_radial_k)

    self.register_buffer("K", kd.K)
    self.register_buffer("ice_kernel", kd.ice_kernel)

    # Algorithm-specific interpolated kernels
    self.interpolate_mdsim_f_kernel()
```

**`generate_ice_deltas` — fix `register_buffer` abuse:**

Find these two lines inside the `for i in range(niter)` loop:

```python
# REMOVE these two register_buffer calls:
self.register_buffer(
    "ice_vol",
    torch.zeros(batchsize, self.nz, self.n, self.n, device=self.device),
)
self.register_buffer(
    "batch_idx", torch.arange(batchsize).view(-1, 1).expand(-1, num_peaks)
)
```

Replace with plain locals:

```python
ice_vol = torch.zeros(batchsize, self.nz, self.n, self.n, device=self.device)
batch_idx = torch.arange(batchsize).view(-1, 1).expand(-1, num_peaks)
```

Also update all subsequent references to `self.ice_vol` and `self.batch_idx` in the loop body to use the local variables `ice_vol` and `batch_idx`.

Replace `new_ice = torch.abs(self.irfftn(ice_vol_f))` with:

```python
new_ice = torch.abs(irfftn(ice_vol_f, self.n, self.nz))
```

**Remove these two methods entirely** (both are now in `_utils.py` / `_kernel.py`):
- `create_ice_kernel` (lines 560–618 in original)
- `irfftn` (lines 658–681 in original)

**Replace the three `generate_big_ice*` methods with deprecation stubs:**

```python
def generate_big_ice(
    self,
    shape: Sequence[int],
    num_unique: int = 8,
    bin_factor: int = 1,
) -> None:
    warnings.warn(
        "generate_big_ice is deprecated. Use IceBank.generate_big_ice() instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def generate_big_ice_fast(
    self,
    shape: Sequence[int],
    num_unique: int = 8,
    bin_factor: int = 1,
) -> None:
    warnings.warn(
        "generate_big_ice_fast is deprecated. Use IceBank.generate_big_ice() instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def generate_big_ice_interpolate(
    self,
    shape: Sequence[int],
    n_blocks: int = 8,
    algorithm_dx: float = 0.5,
) -> None:
    warnings.warn(
        "generate_big_ice_interpolate is deprecated. Use IceBank.generate_big_ice() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "apicemaker" -v
```

- [ ] **Step 5: Lint**

```bash
ruff check src/specter/icemaker/alternating_projections.py
```

Fix any errors before committing.

- [ ] **Step 6: Commit**

```bash
git add src/specter/icemaker/alternating_projections.py tests/test_icemaker_refactor.py
git commit -m "refactor: add APicemaker to icemaker/alternating_projections.py"
```

---

### Task 7: `mcmc.py` — `MCMCIcemaker`

**Files:**
- Create: `src/specter/icemaker/mcmc.py`
- Modify: `tests/test_icemaker_refactor.py`

Copy `MCMCIcemaker` verbatim from `src/specter/icemaker.py` lines 996–1723. One change: rename `init_from_icemaker` → `init_from_apicemaker` and update its type hint.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_icemaker_refactor.py
from specter.icemaker.mcmc import MCMCIcemaker


def test_mcmc_icemaker_init():
    mc = MCMCIcemaker(n=16, dx=1.0, device="cpu")
    assert mc.n == 16
    assert mc.dx == 1.0


def test_mcmc_init_from_apicemaker_method_exists():
    mc = MCMCIcemaker(n=16, dx=1.0)
    assert hasattr(mc, "init_from_apicemaker")
    assert not hasattr(mc, "init_from_icemaker")
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "mcmc" -v
```

- [ ] **Step 3: Create `src/specter/icemaker/mcmc.py`**

Copy `MCMCIcemaker` from `src/specter/icemaker.py` (lines 996–1723). Replace top-level imports:

```python
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..arrays import soft_voxelize_coordinates, tile_volume_from_blocks
from ..progress import ProgressManager, track
from ._constants import ndensity_of_amorphous_ice
```

Rename method `init_from_icemaker` → `init_from_apicemaker` and update its signature:

```python
def init_from_apicemaker(self, im: "APicemaker", batch_idx: int = 0) -> None:
    """
    Initialise from peak voxel coordinates of a completed APicemaker run.

    Parameters
    ----------
    im : APicemaker
        A completed APicemaker instance with ice_coordinates set.
    batch_idx : int
        Which batch item to use.
    """
    # body is identical to the original init_from_icemaker — copy verbatim
```

Add the forward reference import at the top of the method body or use `TYPE_CHECKING`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .alternating_projections import APicemaker
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "mcmc" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/specter/icemaker/mcmc.py tests/test_icemaker_refactor.py
git commit -m "refactor: move MCMCIcemaker to icemaker/mcmc.py, rename init_from_apicemaker"
```

---

### Task 8: `gradient_sk.py` — `GradientDescentIcemaker`

**Files:**
- Create: `src/specter/icemaker/gradient_sk.py`
- Modify: `tests/test_icemaker_refactor.py`

Key changes from `GradientSKIcemaker`:

1. **Class renamed** to `GradientDescentIcemaker`
2. **`__init__`** calls `IceKernelData.build()` — the `_im = Icemaker(...)` hack is removed
3. **Adam removed** from `optimize()` — L-BFGS only, `optimizer` parameter dropped
4. **`_run_opt_loop`** extracted to share loop logic between `optimize()` and `sample()`

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/test_icemaker_refactor.py
from specter.icemaker.gradient_sk import GradientDescentIcemaker


def test_gradient_descent_icemaker_init():
    gd = GradientDescentIcemaker(n=16, dx=1.0, device="cpu")
    assert gd.n == 16
    assert gd.dx == 1.0
    assert gd.f_target.shape == (16, 16, 16)


def test_gradient_descent_optimize_no_adam_param():
    import inspect
    sig = inspect.signature(GradientDescentIcemaker.optimize)
    assert "optimizer" not in sig.parameters
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "gradient_descent" -v
```

- [ ] **Step 3: Create `src/specter/icemaker/gradient_sk.py`**

Start by copying `GradientSKIcemaker` from `src/specter/icemaker.py` (lines 1726–2315). Then apply:

**Imports:**

```python
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from ..arrays import soft_voxelize_coordinates, radial_profile_3d, tile_volume_from_blocks
from ..fft import fft3, fftconvolve
from ..progress import ProgressManager, track
from torchinterp1d import interp1d
from ._kernel import IceKernelData
from ._utils import _wrap_coords
from ._constants import ndensity_of_amorphous_ice
```

**`__init__` — replace the body:** (remove `_im = Icemaker(...)`)

```python
def __init__(
    self,
    n: int = 200,
    dx: float = 0.5,
    nz: Optional[int] = None,
    min_distance: float = 2.0,
    device: str | torch.device = "cpu",
    progressbars: bool = True,
) -> None:
    self.device = torch.device(device)
    self.min_distance = min_distance
    self.progressbars = progressbars

    kd = IceKernelData.build(n=n, dx=dx, nz=nz, min_distance=min_distance)

    self.n = kd.n
    self.nz = kd.nz
    self.dx = kd.dx
    self.box_x = kd.n * kd.dx
    self.box_y = kd.n * kd.dx
    self.box_z = kd.nz * kd.dx
    self._ice_kernel: torch.Tensor = kd.ice_kernel.cpu()
    self.n_molecules = kd.n_ice_molecules

    K_flat = kd.K.cpu().ravel()
    interp_vals = interp1d(
        kd.mdsim_radial_k[1:].cpu(),
        kd.mdsim_f_radial_avg[1:].cpu(),
        K_flat,
    )
    f_kernel = interp_vals.reshape(self.nz, self.n, self.n).float()
    f_kernel = f_kernel * (self.n_molecules**0.5)
    f_kernel[self.nz // 2, self.n // 2, self.n // 2] = float(self.n_molecules)
    self.f_target: torch.Tensor = f_kernel.to(self.device)

    self.dk: float = kd.dk
    self.f_target_radial: torch.Tensor = radial_profile_3d(f_kernel.cpu())

    r_bins = (kd.K.cpu() / kd.dk).round().long().flatten()
    n_rbins = int(r_bins.max().item()) + 1
    bin_count = torch.bincount(r_bins, minlength=n_rbins).float().clamp(min=1)
    self._r_bins: torch.Tensor = r_bins.to(self.device)
    self._bin_count: torch.Tensor = bin_count.to(self.device)
    self._n_rbins: int = n_rbins
    self._f_target_rad_1d: torch.Tensor = self.f_target_radial[:n_rbins].to(self.device)

    r_min_vox = self.min_distance / self.dx
    zz = torch.arange(self.nz, dtype=torch.float32)
    yy = torch.arange(self.n, dtype=torch.float32)
    xx = torch.arange(self.n, dtype=torch.float32)
    zz = torch.where(zz < self.nz // 2, zz, zz - self.nz)
    yy = torch.where(yy < self.n // 2, yy, yy - self.n)
    xx = torch.where(xx < self.n // 2, xx, xx - self.n)
    ZZ, YY, XX = torch.meshgrid(zz, yy, xx, indexing="ij")
    R_ker = torch.sqrt(ZZ**2 + YY**2 + XX**2)
    rep_kernel = (R_ker < r_min_vox).float()
    rep_kernel[0, 0, 0] = 0.0
    self._rep_kernel_rfft: torch.Tensor = torch.fft.rfftn(rep_kernel).to(self.device)

    self.positions: Optional[torch.Tensor] = None
```

**Add `_run_opt_loop`:**

```python
def _run_opt_loop(
    self,
    pos: torch.Tensor,
    step_fn: Callable[[], tuple[float, torch.Tensor]],
    n_steps: int,
    record_every: int,
    desc: str,
) -> dict:
    """Shared optimization loop. step_fn() returns (loss_val, f_amp) and handles opt.step()."""
    history: dict[str, list] = {"step": [], "loss": [], "radial_profile": []}
    _manager = ProgressManager()
    _pbar, _pbar_pos = _manager.get_pbar(
        range(n_steps), desc=desc, disable=not self.progressbars, transient=True
    )
    try:
        for step in _pbar:
            loss_val, f_amp = step_fn()
            _pbar.set_postfix(loss=f"{loss_val:.6f}")
            if step % record_every == 0:
                with torch.no_grad():
                    rad = radial_profile_3d(f_amp.cpu())
                history["step"].append(step)
                history["loss"].append(loss_val)
                history["radial_profile"].append(rad)
    finally:
        _pbar.close()
        _manager.release(_pbar_pos)

    self.positions = pos.detach().cpu()
    return history
```

**`optimize()` — L-BFGS only, no `optimizer` parameter:**

```python
def optimize(
    self,
    n_steps: int = 50,
    lr: float = 1.0,
    record_every: int = 5,
    rep_strength: float = 1.0,
) -> dict:
    """
    Gradient descent on the S(k) loss using L-BFGS.

    Parameters
    ----------
    n_steps : int
        Optimizer outer iterations.
    lr : float
        Initial step size (line search adapts automatically). Default 1.0.
    record_every : int
        Diagnostic recording interval.
    rep_strength : float
        Weight of the soft pair-exclusion penalty.

    Returns
    -------
    history : dict
        Keys: 'step', 'loss', 'radial_profile'.
    """
    assert self.positions is not None, "Call init_random() or init_rsa() first"

    pos = self.positions.to(self.device).clone().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [pos], lr=lr, max_iter=10, history_size=20, line_search_fn="strong_wolfe"
    )
    last_f_amp: list[torch.Tensor] = [torch.empty(0)]

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss, f_amp = self._sk_loss(pos, rep_strength=rep_strength)
        loss.backward()
        last_f_amp[0] = f_amp.detach()
        return loss

    def step_fn() -> tuple[float, torch.Tensor]:
        loss = opt.step(closure)
        with torch.no_grad():
            pos.data[:, 0] = _wrap_coords(pos.data[:, 0], self.box_x)
            pos.data[:, 1] = _wrap_coords(pos.data[:, 1], self.box_y)
            pos.data[:, 2] = _wrap_coords(pos.data[:, 2], self.box_z)
        loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        return loss_val, last_f_amp[0]

    return self._run_opt_loop(pos, step_fn, n_steps, record_every, "L-BFGS")
```

**`sample()` — refactored to use `_run_opt_loop`:**

```python
def sample(
    self,
    n_steps: int = 200,
    lr: float = 0.05,
    noise: float = 0.02,
    record_every: int = 50,
    rep_strength: float = 1.0,
) -> dict:
    """Langevin sampling around a converged structure. Call optimize() first."""
    assert self.positions is not None, "Call optimize() or init_* first"

    pos = self.positions.to(self.device).clone().requires_grad_(True)
    opt = torch.optim.Adam([pos], lr=lr)

    def step_fn() -> tuple[float, torch.Tensor]:
        opt.zero_grad()
        loss, f_amp = self._sk_loss(pos, rep_strength=rep_strength)
        loss.backward()
        opt.step()
        with torch.no_grad():
            pos.data.add_(noise * torch.randn_like(pos))
            pos.data[:, 0] = _wrap_coords(pos.data[:, 0], self.box_x)
            pos.data[:, 1] = _wrap_coords(pos.data[:, 1], self.box_y)
            pos.data[:, 2] = _wrap_coords(pos.data[:, 2], self.box_z)
        return loss.item(), f_amp.detach()

    return self._run_opt_loop(pos, step_fn, n_steps, record_every, "Langevin")
```

Copy remaining methods (`_sk_loss`, `init_random`, `init_rsa`, `voxelize`, `generate_ice_deltas`, `generate_ice`, `generate_big_ice`, `assemble_tiles`) verbatim from the original.

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "gradient_descent" -v
```

- [ ] **Step 5: Lint**

```bash
ruff check src/specter/icemaker/gradient_sk.py
```

- [ ] **Step 6: Commit**

```bash
git add src/specter/icemaker/gradient_sk.py tests/test_icemaker_refactor.py
git commit -m "refactor: add GradientDescentIcemaker to icemaker/gradient_sk.py"
```

---

### Task 9: `bank.py` — `IceBank`

**Files:**
- Create: `src/specter/icemaker/bank.py`
- Modify: `tests/test_icemaker_refactor.py`

Copy `IceBank` verbatim from `src/specter/icemaker.py` lines 2849–3004. One change: update `_METHOD_MAP` key from `'gs'` to `'ap'`.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_icemaker_refactor.py
from specter.icemaker.bank import IceBank


def test_icebank_method_key_ap():
    assert "ap" in IceBank._METHOD_MAP
    assert "gs" not in IceBank._METHOD_MAP


def test_icebank_build_and_generate():
    bank = IceBank(dx=1.0, n=16, method="ap")
    bank.build(num_unique=2)
    ice = bank.generate_ice(batchsize=2)
    assert ice.shape[0] == 2
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "icebank" -v
```

- [ ] **Step 3: Create `src/specter/icemaker/bank.py`**

Copy `IceBank` from `src/specter/icemaker.py` (lines 2849–3004). Replace imports:

```python
from __future__ import annotations

from typing import Any, Optional

import torch

from ..arrays import tile_volume_from_blocks
from ._utils import replace_outer_faces
from .alternating_projections import APicemaker
from .gradient_sk import GradientDescentIcemaker
from .mcmc import MCMCIcemaker
```

Update `_METHOD_MAP`:

```python
_METHOD_MAP: dict[str, type] = {
    "ap": APicemaker,
    "gd": GradientDescentIcemaker,
    "mcmc": MCMCIcemaker,
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "icebank" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/specter/icemaker/bank.py tests/test_icemaker_refactor.py
git commit -m "refactor: move IceBank to icemaker/bank.py, update method key to 'ap'"
```

---

### Task 10: `__init__.py` + delete old file

**Files:**
- Modify: `src/specter/icemaker/__init__.py`
- Delete: `src/specter/icemaker.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_icemaker_refactor.py
def test_top_level_imports():
    from specter.icemaker import (
        APicemaker,
        NaiveIcemaker,
        MCMCIcemaker,
        GradientDescentIcemaker,
        MDSimDump,
        IceBank,
    )
    assert APicemaker is not None
    assert IceBank is not None
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
python -m pytest tests/test_icemaker_refactor.py::test_top_level_imports -v
```

Expected: `FAILED` (ImportError — `__init__.py` is still a stub)

- [ ] **Step 3: Populate `src/specter/icemaker/__init__.py`**

```python
from __future__ import annotations

from .alternating_projections import APicemaker
from .naive import NaiveIcemaker
from .mcmc import MCMCIcemaker
from .gradient_sk import GradientDescentIcemaker
from .mdsim import MDSimDump
from .bank import IceBank
from ._utils import (
    water_molecule_coordinates,
    create_n_randomly_rotated_water_molecules,
    volume_of_ice,
    replace_outer_faces,
    rfftn,
)
from ._constants import ndensity_of_amorphous_ice

__all__ = [
    "APicemaker",
    "NaiveIcemaker",
    "MCMCIcemaker",
    "GradientDescentIcemaker",
    "MDSimDump",
    "IceBank",
    "water_molecule_coordinates",
    "create_n_randomly_rotated_water_molecules",
    "volume_of_ice",
    "replace_outer_faces",
    "rfftn",
    "ndensity_of_amorphous_ice",
]
```

- [ ] **Step 4: Run test — expect PASS**

```bash
python -m pytest tests/test_icemaker_refactor.py::test_top_level_imports -v
```

- [ ] **Step 5: Delete old file**

```bash
git rm src/specter/icemaker.py
```

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -v
```

Fix any import errors that surface. Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/specter/icemaker/__init__.py
git commit -m "refactor: finalize icemaker package __init__.py, remove old icemaker.py"
```

---

### Task 11: Update downstream imports

**Files:**
- Modify: `src/specter/imagegenerator.py`
- Modify: `src/specter/specimen.py`

Both files import `Icemaker` (now `APicemaker`) and `NaiveIcemaker`.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_icemaker_refactor.py
def test_imagegenerator_imports_cleanly():
    import specter.imagegenerator  # will fail if it still imports old name


def test_specimen_imports_cleanly():
    import specter.specimen
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "imports_cleanly" -v
```

Expected: `FAILED` (ImportError — `Icemaker` no longer exists)

- [ ] **Step 3: Update `src/specter/imagegenerator.py`**

Find line 14:

```python
from .icemaker import Icemaker, NaiveIcemaker
```

Replace with:

```python
from .icemaker import APicemaker, NaiveIcemaker
```

Then find all uses of `Icemaker` in the file (lines 323 and 550) and replace `Icemaker(` with `APicemaker(`.

- [ ] **Step 4: Update `src/specter/specimen.py`**

Find line 7:

```python
from .icemaker import Icemaker, NaiveIcemaker
```

Replace with:

```python
from .icemaker import APicemaker, NaiveIcemaker
```

Replace all uses of `Icemaker(` with `APicemaker(` in the file.

- [ ] **Step 5: Run tests — expect PASS**

```bash
python -m pytest tests/test_icemaker_refactor.py -k "imports_cleanly" -v
```

- [ ] **Step 6: Run full suite + lint**

```bash
python -m pytest tests/ -v
ruff check src/specter/
mypy src/
```

Fix any remaining issues.

- [ ] **Step 7: Commit**

```bash
git add src/specter/imagegenerator.py src/specter/specimen.py
git commit -m "refactor: update imagegenerator and specimen to use APicemaker"
```
