# Cryosim

**cryosim** is a Python package for simulating cryo-electron microscopy (cryo-EM) images with physics-based models.  
It supports aberrations, scattering, detector effects, and integrates with PyTorch for GPU acceleration.

---

## Installation and running the notebooks
Clone the repository:
```bash
git clone https://github.com/joelyeois/cryosim.git
cd cryosim
```
You must install dependencies **and** the `cryosim` package itself.

### Option 1. Using [uv](https://github.com/astral-sh/uv) (recommended)
This will create a virtual environment and install all dependencies listed in `pyproject.toml.`
```bash
uv sync
```
Activate the environment with:
```bash
source .venv/bin/activate
```
Install `cryosim`:
```bash
uv pip install -e .
```
Run the notebooks:
```bash
uv run --with jupyter jupyter lab
```

### Option 2. Conda and pip
Create an environment and install the dependencies.
```bash
conda create -n cryosim python=3.11
conda activate cryosim
pip install -r requirements.txt
```
Install `cryosim`:
```bash
pip install -e .
```
Run the notebooks (you must install jupyter yourself):
```bash
jupyter lab
```

