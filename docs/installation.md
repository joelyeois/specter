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
