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

## Installing with conda/pip instead

```bash
conda create -n specter python=3.11
conda activate specter
pip install -r requirements.txt
pip install -e .
```

## Next steps

See [Quickstart](quickstart.md) for a complete example.
