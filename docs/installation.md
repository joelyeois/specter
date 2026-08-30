# Installation

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or conda/pip

## Install

```bash
git clone https://github.com/joelyeois/specter.git
cd specter

uv sync
source .venv/bin/activate
```

`uv sync` installs SPECTER itself alongside its dependencies, in editable
mode, so no separate `uv pip install -e .` step is needed.

!!! success "Verify installation"
    After installation, verify that the CLI is available:

        specter --help

    You should see output similar to the following:

         Usage: specter [OPTIONS] COMMAND [ARGS]...
        
         SPECTER command-line interface.
        
        ╭─ Options ────────────────────────────────────────────────────────────────────╮
        │ --help  -h  Show this message and exit.                                      │
        ╰──────────────────────────────────────────────────────────────────────────────╯
        ╭─ Commands ───────────────────────────────────────────────────────────────────╮
        │ build        Build specimen volumes and reusable assets                      │
        │ cache        Inspect and clear the cache of downloaded PDB/mmCIF structures. │
        │ ghostbuster  Reconstruct 3D volumes from experimental images                 │
        │ jobs         Inspect and compare tracked SPECTER jobs.                       │
        │ reconstruct  Reconstruct 3D volumes from experimental images                 │
        │ simulate     Simulate cryo-EM/cryo-ET data                                   │
        ╰──────────────────────────────────────────────────────────────────────────────╯

## Choosing a CUDA version

The two CUDA-flavoured pins in `pyproject.toml` are not fixed requirements.
Edit both to match your driver:

```toml
# 1. which PyTorch build uv resolves
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }

# 2. which CuPy wheel provides the GPU distance transform
dependencies = [
    "cupy-cuda12x>=13.0; sys_platform != 'darwin'",
]
```

Change both together, to the row matching the CUDA version your driver
supports (`nvidia-smi` reports it top-right):

| Your CUDA | PyTorch index URL | CuPy package |
|---|---|---|
| 11.8 | `https://download.pytorch.org/whl/cu118` | `cupy-cuda11x` |
| 12.1 (default) | `https://download.pytorch.org/whl/cu121` | `cupy-cuda12x` |
| 12.4 | `https://download.pytorch.org/whl/cu124` | `cupy-cuda12x` |
| 12.6+ | `https://download.pytorch.org/whl/cu126` | `cupy-cuda12x` |
| none (CPU only) | `https://download.pytorch.org/whl/cpu` | remove the line |

Renaming the index in both places is fine: `name` is arbitrary, it has
to match what `[tool.uv.sources]` points at. Newer CUDA builds may also imply a
newer minimum `torch`: `cu126` wheels start at torch 2.6, for instance.

!!! tip "You may not need to change the PyTorch pin at all"

    CUDA has minor-version compatibility: any 12.x build runs on a driver that
    supports 12.0 (≥ 525.60.13 on Linux), so a 12.4 or 12.8 driver runs the
    default cu121 wheels fine. The pin only needs changing for CUDA 11, for
    ROCm/CPU-only, or when you want a newer runtime.

    `uv pip install --torch-backend auto` detects your driver and picks the
    matching PyTorch build. It works on uv's `pip` interface
    only, not `uv sync`, so treat it as an alternative to editing the pin
    rather than a replacement for it.

A mismatched CuPy wheel fails safely: the `spherical_harmonics` membrane
backend warns once and falls back to `scipy`'s CPU distance transform. A
mismatched **PyTorch** build fails at CUDA init instead of degrading.

Installing with pip instead of uv bypasses `[tool.uv.index]` (a
uv-specific setting) and gives you PyPI's default PyTorch build. Use
`pip install torch --index-url <url from the table>` to pick the build
yourself.

## Optional extras

- `docs`: build these docs locally (`uv sync --extra docs`).

!!! note "GPU distance transforms"

    CuPy is a core dependency, so a plain `uv sync` installs GPU distance
    transforms for the `spherical_harmonics` membrane backend too.

    On macOS no `cupy-cuda12x` wheels exist, so `uv sync` skips CuPy and
    the backend falls back to `scipy`'s CPU distance transform (~3x slower
    field generation, plus a one-time warning). The same fallback covers a
    machine with no CUDA device available at runtime.

## Monomer Library (for Shtyrov scattering factors)

The default `shtyrov` parameterization fits scattering factors per *bonded
species* rather than per element: SPECTER types an atom by what it is
bonded to, so a methyl carbon reads `C(HHHC)` and a water oxygen reads
`O(HH)`. Twenty of the forty-two tabulated species contain hydrogen.

Deposited structures almost never include hydrogens, since typical
resolution does not resolve them. The
[Monomer Library](https://github.com/MonomerLibrary/monomers) supplies
them instead, the same chemical-component dictionary
[REFMAC](https://www.ccp4.ac.uk/html/refmac5.html) and
[Coot](https://www2.mrc-lmb.cam.ac.uk/personal/pemsley/coot/) use. SPECTER
does not bundle it: it runs roughly 1.5 GB, and carries its own license.

Without it, every H-containing species fails to match and those atoms fall
back to per-element Peng factors. Measured on myoglobin, that is about
**44% of a hydrogen-free protein**. SPECTER still runs, and warns once
per process.

```bash
git clone https://github.com/MonomerLibrary/monomers.git
export CLIBD_MON=/path/to/monomers      # add to ~/.bashrc to persist
```

A partial clone is enough if disk is tight:

```bash
git clone --filter=blob:none --sparse https://github.com/MonomerLibrary/monomers.git
cd monomers && git sparse-checkout set a c d g h i l m n p s t v w y
```

`$CLIBD_MON` is the variable [CCP4](https://www.ccp4.ac.uk/) already uses,
so an existing CCP4 install needs no extra setup. SPECTER expands `~` and
`$VAR` in the path, and a path that does not exist fails immediately rather
than silently degrading to Peng.

The variable does not have to be exported globally. It can be set for a
single command:

```bash
CLIBD_MON=~/monomers specter build tomogram --config configs/tomogram.toml
```

or, in a notebook, from any cell before the structures are read (the path is
resolved per call, not at import):

```python
import os
os.environ["CLIBD_MON"] = os.path.expanduser("~/monomers")
```

A run can also name the library in its own config, which is what to prefer
when the result needs to be reproducible from the config alone: unlike
`pdb_cache_dir`, this choice changes the rendered potential, so a config that
omits it does not fully describe what it produced.

```toml
monomer_library_path = "~/monomers"
```

```bash
specter build tomogram --config configs/tomogram.toml --monomer_library_path ~/monomers
```

`monomer_library_path` is available on `specter simulate particles`,
`specter simulate micrograph` and `specter build tomogram`. It takes
precedence over `$CLIBD_MON`; left unset, the variable still applies, so
nothing that already relies on it needs changing.

Python callers can still override it per structure:

```python
from specter.pdb import PDB

pdb = PDB("1a6m", compute_atom_species=True,
          monomer_library_path="/path/to/monomers")
```

!!! note "What changes when a library is present"

    SPECTER adds hydrogens from the library's ideal geometry, then takes
    coordinates, elements, and species from that completed model. A
    hydrogen-free deposition therefore roughly **doubles in atom count**
    (myoglobin: 1,445 → 2,668), typing coverage rises from ~56% to ~99%, and
    the rendered potential changes by 20-30% relative RMS. SPECTER excludes
    hydrogens whose position is chemically ambiguous (a rotatable hydroxyl,
    or both tautomer hydrogens of a histidine) rather than rendering them.

    Structures that already carry hydrogens keep them where they were
    deposited; only a structure with none gets hydrogens added. See
    `readd_hydrogens` to override, below.

For what `readd_hydrogens` controls and how it interacts with typing coverage,
see [Hydrogen coordinates](concepts/atomic-potentials.md#hydrogen-coordinates)
in Concepts.

## Installing with conda/pip instead

```bash
conda create -n specter python=3.11
conda activate specter
pip install -e .
```

`pip install -e .` reads the full dependency list from `pyproject.toml`, so
you do not need a separate requirements file.

To install the optional extras (see above) via pip instead of `uv sync`:

```bash
pip install -e ".[docs]"      # build docs locally
```

## Next steps

See [Quickstart](quickstart.md) for a complete example.
