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
uv pip install -e .
```

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
        │ simulate           Simulate cryo-EM/cryo-ET data                             │
        ╰──────────────────────────────────────────────────────────────────────────────╯

## Choosing a CUDA version

The two CUDA-flavoured pins in `pyproject.toml` aren't fixed requirements.
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

Renaming the index in both places is fine; `name` is arbitrary, it just has
to match what `[tool.uv.sources]` points at. Newer CUDA builds may also imply a
newer minimum `torch`; `cu126` wheels start at torch 2.6, for instance.

!!! tip "You may not need to change the PyTorch pin at all"

    CUDA has minor-version compatibility: any 12.x build runs on a driver that
    supports 12.0 (≥ 525.60.13 on Linux). So a 12.4 or 12.8 driver runs the
    default cu121 wheels fine. The pin only needs changing for CUDA 11, for
    ROCm/CPU-only, or when you specifically want a newer runtime.

    `uv pip install --torch-backend auto` detects your driver and picks the
    matching PyTorch build automatically. It works on uv's `pip` interface
    only, not `uv sync`, so it's an alternative to editing the pin rather than
    a replacement for it.

A mismatched CuPy wheel fails safely: the `spherical_harmonics` membrane
backend warns once and falls back to `scipy`'s CPU distance transform. A
mismatched **PyTorch** build fails at CUDA init instead of degrading.

Installing with pip instead of uv bypasses `[tool.uv.index]` entirely (it is a
uv-specific setting) and gives you PyPI's default PyTorch build; use
`pip install torch --index-url <url from the table>` to choose explicitly.

## Optional extras

- `docs`: build these docs locally (`uv sync --extra docs`).

!!! note "GPU distance transforms"

    CuPy is a core dependency, so `spherical_harmonics` membrane backend GPU
    distance transforms are installed by a plain `uv sync`.

    On macOS there are no `cupy-cuda12x` wheels, so CuPy isn't installed and
    the backend falls back to `scipy`'s CPU distance transform (~3x slower
    field generation, plus a one-time warning). The same fallback covers a
    machine with no CUDA device available at runtime.

## Monomer Library (for Shtyrov scattering factors)

The default `shtyrov` parameterization fits scattering factors per *bonded
species* — `C(HHHC)` for a methyl carbon, `O(HH)` for a water oxygen — rather
than per element, so an atom is typed by what it is bonded to. Twenty of the
forty-two tabulated species contain hydrogen.

Deposited structures almost never include hydrogens, since they are not
resolved at typical resolution. Supplying them is the job of the
[Monomer Library](https://github.com/MonomerLibrary/monomers), the same
chemical-component dictionary REFMAC and Coot use. It is not bundled with
SPECTER: it is roughly 1.5 GB, and separately licensed.

Without it, every H-containing species fails to match and those atoms fall
back to per-element Peng factors — about **44% of a hydrogen-free protein**,
measured on myoglobin. SPECTER still runs, and warns once per structure.

```bash
git clone https://github.com/MonomerLibrary/monomers.git
export CLIBD_MON=/path/to/monomers      # add to ~/.bashrc to persist
```

A partial clone is enough if disk is tight:

```bash
git clone --filter=blob:none --sparse https://github.com/MonomerLibrary/monomers.git
cd monomers && git sparse-checkout set a c d g h i l m n p s t v w y
```

`$CLIBD_MON` is the variable CCP4 already uses, so an existing CCP4 install
needs no extra setup. It is the only place SPECTER looks: there is no config or
CLI equivalent, since the library is an installation detail rather than a
per-simulation choice, and the Monomer Library documents this variable as the
way to point at it. `~` and `$VAR` in the path are expanded, and a path that
does not exist fails immediately rather than silently degrading to Peng.

Python callers can still override it per structure:

```python
from specter.pdb import PDB

pdb = PDB("1a6m", compute_atom_species=True,
          monomer_library_path="/path/to/monomers")
```

!!! note "What changes when a library is present"

    Hydrogens are added from the library's ideal geometry, and coordinates,
    elements and species are then all taken from that completed model. A
    hydrogen-free deposition therefore roughly **doubles in atom count**
    (myoglobin: 1,445 → 2,668), typing coverage rises from ~56% to ~99%, and
    the rendered potential changes by 20-30% relative RMS. Hydrogens whose
    position is chemically ambiguous — a rotatable hydroxyl, or both tautomer
    hydrogens of a histidine — are excluded rather than rendered.

    Structures that already carry hydrogens keep them where they were
    deposited; only a structure with none has them added. See
    `readd_hydrogens` below to override.

### Hydrogen coordinates

A species descriptor is built from the bond graph, not from coordinates, so
the fitted factors apply whichever positions a hydrogen occupies. What
`readd_hydrogens` controls is only whether hydrogen *density* is added, and
from ideal or deposited geometry.

The default, `"auto"`, follows the file: hydrogens a structure already carries
are left exactly where they are, and hydrogens are added only to a structure
that has none. Deposited positions are information the file provides, so
nothing is gained by moving them.

| | atoms | H | typed |
|---|---:|---:|---:|
| **1A6M** — no deposited H, no library | 1,445 | 0 | 56% |
| 1A6M — `"auto"` (adds them) | 2,668 | 1,223 | 99.6% |
| **7a4m** — deposited H, `"auto"` (keeps them) | 2,862 | 1,341 | 99.4% |
| 7a4m — `readd_hydrogens=True` (replaces them) | 2,848 | 1,327 | 99.9% |

The two explicit settings remain available:

```python
PDB("7a4m", compute_atom_species=True, readd_hydrogens=True)   # ideal geometry
PDB("1a6m", compute_atom_species=True, readd_hydrogens=False)  # add none
```

`True` always re-adds from ideal geometry, matching `sffit`'s default and so
the configuration the factors were fitted in. `False` never re-adds: hydrogens
the file lacks become zero-occupancy atoms that inform their neighbours'
species without being rendered, which on 1A6M lifts typing from 56% to 99.2%
while leaving the atom set untouched at 1,445.

A partially hydrogenated structure counts as carrying them, so `"auto"` keeps
what is there rather than replacing the lot; pass `True` to complete it.

## Installing with conda/pip instead

```bash
conda create -n specter python=3.11
conda activate specter
pip install -e .
```

`pip install -e .` reads the full dependency list from `pyproject.toml`, so no
separate requirements file is needed.

To install the optional extras (see above) via pip instead of `uv sync`:

```bash
pip install -e ".[docs]"      # build docs locally
```

## Next steps

See [Quickstart](quickstart.md) for a complete example.
