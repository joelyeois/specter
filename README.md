<p align="center">
  <img src="images/SPECTER_Logo_White.png" alt="SPECTER logo" width="300">
</p>

# SPECTER
### Scattering & Propagation of Electrons in Cryo-EM: Twin Emulator & Reconstruction

[![Docs](https://github.com/joelyeois/specter/actions/workflows/docs.yml/badge.svg)](https://joelyeois.github.io/specter/)

**SPECTER** is a Python package for simulating cryo-electron microscopy (cryo-EM) images with physics-based models.  
It supports aberrations, scattering, detector effects, and integrates with PyTorch for GPU acceleration.

> **Early development notice**  
> SPECTER is under active development. APIs and behaviour may change without notice between releases — including breaking changes. It is not yet recommended for production workflows. Feedback and bug reports are welcome via [GitHub Issues](https://github.com/joelyeois/specter/issues).

---

## Installation

```bash
git clone https://github.com/joelyeois/specter.git
cd specter
uv sync
source .venv/bin/activate
uv pip install -e .          # the package itself, not just its dependencies
uv run --with jupyter jupyter lab
```

See the [installation guide](https://joelyeois.github.io/specter/installation.html)
for the conda/pip alternative, GPU notes, and troubleshooting an install
outside a git checkout.

---

## Usage

Full usage documentation — every demo script's CLI reference, the TOML
config system, the physics pipeline, ice generation, Ghostbuster
reconstruction, and job management — lives in the docs:

**[joelyeois.github.io/specter](https://joelyeois.github.io/specter/)**

Ready-to-run CLI scripts are in `demo-scripts/`; interactive notebooks are
in `demo-notebooks/`.

