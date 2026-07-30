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
python demo-scripts/generate_particle_stack.py \
    --config configs/particle.toml \
    --n_particles 4 --num_pixels 128 \
    --device cpu --output_dir ./output/
```

You should get `output/particles.mrcs` and `output/particles.star`.

## Installing with conda/pip instead

```bash
conda create -n specter python=3.11
conda activate specter
pip install -r requirements.txt
pip install -e .
```

## Next steps

See [Quickstart](quickstart.md) for a complete example.
