from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import specter

# specter/__init__.py -> specter/ -> src/ -> repo root. Anchoring here (rather
# than cwd or a caller's __file__) makes path resolution work identically from
# the script (cwd = repo root, per README.md) and the notebook (cwd =
# demo-notebooks/particle_stack/, and Jupyter cells have no __file__ at all).
REPO_ROOT = Path(specter.__file__).resolve().parents[2]


@dataclass
class ParticleStackConfig:
    """Parameters for particle-stack generation, loaded from a TOML config file."""

    # --- PDB / potential ---
    pdb_code: str
    assembly: bool = True
    pdb_savefolder: str = "pdb-data"  # resolved against REPO_ROOT if relative
    num_pixels: int = 256
    pixel_size: float = 1.0  # Å

    # --- Microscope / physics ---
    energy: float = 300.0  # keV
    dose_min: float = 20.0  # e⁻/Å²
    dose_max: float | None = None  # e⁻/Å²
    num_frames: int | None = None
    cs: float = 2.0  # mm
    alpha: float = 0.1  # unitless, amplitude contrast ratio

    # --- Envelopes ---
    convergence_angle: float | None = None  # mrad
    cc: float | None = None  # mm
    energy_spread: float = 0.7  # eV (FWHM)
    deltaV_V: float = 0.06e-6  # unitless (ΔV/V)
    deltaI_I: float = 0.01e-6  # unitless (ΔI/I)
    dose_envelope: bool = False
    bfactor: float | None = None  # Å²

    # --- Defocus ---
    defocus_min: float = 5000.0  # Å
    defocus_max: float = 15000.0  # Å

    # --- Translations / shifts ---
    shift: float = 2.0  # Å, max in-plane shift (uniform ±shift)

    # --- Dataset size ---
    n_particles: int = 20

    # --- Models ---
    scattering_model: Literal["multislice", "firstborn", "projection", "ctf"] = (
        "multislice"
    )
    aberration_model: Literal["holography", "ctf"] = "holography"
    noise_model: Literal["poisson", "none"] = "poisson"
    coincidence_radius_min: float = 0.0  # pixels
    coincidence_radius_max: float | None = None  # pixels
    ice_model: Literal["gd", "ap", "mcmc", "random", "none"] = "gd"
    ice_thickness: float = 0.0  # Å, 0 = minimum (particle box size)
    num_unique_icecubes: int = 8
    ice_build_batch_size: int = 1
    icecube_size: int | None = None  # voxels
    crowd_min_distance: float | None = None  # Å
    crowd_max_distance_z: float | None = None  # Å
    potential_scale_min: float = 1.0  # unitless
    potential_scale_max: float | None = None  # unitless
    pad_fft: bool = True
    detector_model: Literal["none", "perfect", "k3_300kv", "k3_200kv"] = "none"

    # --- Post-processing ---
    normalize_particles: bool = True
    save_exitwaves: bool = False
    save_clean_exitwaves: bool = False

    # --- Compute ---
    device: str = "cpu"
    batchsize: int = 5

    # --- Output ---
    output_dir: str = "./output/"
    filename: str = "particles"


def load_config(path: str) -> ParticleStackConfig:
    """
    Load a `ParticleStackConfig` from a TOML file.

    Parameters
    ----------
    path : str
        Path to a TOML config file. May use tables (e.g. `[potential]`) to
        group fields for readability; all tables are flattened into one
        namespace before validation.

    Returns
    -------
    ParticleStackConfig
        Config with unset fields filled from `ParticleStackConfig` defaults.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    flat: dict = {}
    for value in raw.values():
        if isinstance(value, dict):
            flat.update(value)
    flat.update({k: v for k, v in raw.items() if not isinstance(v, dict)})
    config = ParticleStackConfig(**flat)
    if not os.path.isabs(config.pdb_savefolder):
        config.pdb_savefolder = str(REPO_ROOT / config.pdb_savefolder)
    return config


def apply_overrides(
    config: ParticleStackConfig, overrides: dict
) -> ParticleStackConfig:
    """
    Set fields on a `ParticleStackConfig` in place from a dict of overrides.

    Parameters
    ----------
    config : ParticleStackConfig
        Config to mutate.
    overrides : dict
        Field name -> value pairs, e.g. from parsed CLI arguments.

    Returns
    -------
    ParticleStackConfig
        The same `config` instance, mutated.
    """
    for key, value in overrides.items():
        setattr(config, key, value)
    return config
