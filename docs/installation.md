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

## Launch the notebooks

```bash
uv run --with jupyter jupyter lab
```

## Verify installation

```bash
specter --help
```

You should see the SPECTER CLI's usage and command list.

## Installing with conda/pip instead

```bash
conda create -n specter python=3.11
conda activate specter
pip install -r requirements.txt
pip install -e .
```

## Next steps

See [Quickstart](quickstart.md) for a complete example.
