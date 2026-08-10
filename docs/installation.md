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

## Optional extras

- `docs` -- build these docs locally (`uv sync --extra docs`).
- `gpu-edt` -- GPU-accelerated (CuPy-backed) distance transforms for the
  `spherical_harmonics` membrane shape backend
  (`specter.specimen.membrane`). Purely a speed optimization: falls back
  automatically to CPU (`scipy`) when not installed, or when no CUDA
  device is available at runtime, so it's safe to skip. Requires a CUDA
  12.x GPU (matching the CUDA 12.1 build `torch` itself is pinned to).

    ```bash
    uv sync --extra gpu-edt
    ```

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
pip install -e ".[gpu-edt]"   # GPU-accelerated distance transforms
```

## Next steps

See [Quickstart](quickstart.md) for a complete example.
