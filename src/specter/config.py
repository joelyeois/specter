from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

import specter

# specter/__init__.py -> specter/ -> src/ -> repo root. Anchoring here (rather
# than cwd or a caller's __file__) makes path resolution work identically from
# the script (cwd = repo root, per README.md) and the notebook (cwd =
# demo-notebooks/particle_stack/, and Jupyter cells have no __file__ at all).
REPO_ROOT = Path(specter.__file__).resolve().parents[2]


@dataclass
class ParticleStackConfig:
    """Parameters for particle-stack generation, loaded from a TOML config file.

    Set `cs_path` to drive generation from a CryoSPARC .cs file instead of
    randomly-sampled poses/CTF: `pixel_size`, `energy`, `alpha`, defocus, and
    shifts are then read from the .cs file at run time via
    `extract_parameters_from_csfile` and take precedence over the
    corresponding fields below, which are unused in that mode.
    """

    # --- PDB / potential ---
    pdb_code: str
    assembly: bool = True
    pdb_savefolder: str = "pdb-data"  # resolved against REPO_ROOT if relative
    num_pixels: int = 256
    pixel_size: float = 1.0  # Å

    # --- CryoSPARC input ---
    # if set, poses/CTF/pixel_size/energy/alpha come from here
    cs_path: str | None = None

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
    ice_model: Literal["gd", "random", "none"] = "gd"
    ice_thickness: float = 0.0  # Å, 0 = minimum (particle box size)
    ice_cache_dir: str | None = None  # defaults to the bundled ice-data/ice_cache
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


@dataclass
class MicrographConfig:
    """Parameters for micrograph generation, loaded from a TOML config file."""

    # --- PDB / potential ---
    pdb_code: str
    assembly: bool = True
    pdb_savefolder: str = "pdb-data"  # resolved against REPO_ROOT if relative
    num_pixels: int = 256
    pixel_size: float = 1.0  # Å
    micrograph_size: int = 4096

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

    # --- Defocus ---
    defocus_min: float = 5000.0  # Å
    defocus_max: float = 15000.0  # Å

    # --- Dataset size ---
    n_micrographs: int = 1

    # --- Models ---
    scattering_model: Literal["multislice", "firstborn", "projection", "ctf"] = (
        "multislice"
    )
    aberration_model: Literal["holography", "ctf"] = "holography"
    noise_model: Literal["poisson", "none"] = "poisson"
    coincidence_radius_min: float = 1.8  # pixels
    coincidence_radius_max: float | None = None  # pixels
    ice_model: Literal["gd", "random", "none"] = "gd"
    ice_thickness: float = 500.0  # Å, 0 = minimum (particle box size)
    ice_cache_dir: str | None = None  # defaults to the bundled ice-data/ice_cache
    crowd_min_distance: float | None = None  # Å
    crowd_max_distance_z: float | None = None  # Å
    water_air_interface: bool = True
    potential_scale_min: float = 1.0  # unitless
    potential_scale_max: float | None = None  # unitless
    pad_fft: bool = False
    chunk_size: int | None = None
    detector_model: Literal["none", "perfect", "k3_300kv", "k3_200kv"] = "none"

    # --- Post-processing ---
    normalize_micrographs: bool = False
    save_exitwaves: bool = False
    save_clean_exitwaves: bool = False

    # --- Compute ---
    device: str = "cpu"

    # --- Output ---
    output_dir: str = "./output/"
    filename: str = "micrographs"


@dataclass
class TiltSeriesConfig:
    """Parameters for cryoET tilt-series generation, loaded from a TOML config file.

    Combines the two-stage pipeline from `dev/cryoet-specimen-generation/10440.ipynb`:
    `specter.specimen.CryoETSpecimenGenerator` places proteins/membranes into a
    specimen volume, then `TiltSeriesGenerator` blends in ice and simulates the
    tilted acquisition (multislice scattering, CTF, dose, detector, noise).
    """

    # --- Specimen placement (CryoETSpecimenGenerator) ---
    # One dict per protein species, e.g. {"PDB_CODE": "6qzp", "PMER_OCC": 0.6}.
    # Only "PDB_CODE" is required; see CryoETSpecimenGenerator's docstring for
    # the other optional polnet-style keys (PMER_OCC, PMER_L, PMER_L_MAX,
    # PMER_OVER_TOL, MMER_ISO). In TOML, provide as [[protein_specs]] tables.
    protein_specs: list[dict[str, Any]]
    # One dict per membrane species/population, polnet .mbs-equivalent keys
    # (MB_TYPE, MB_THICK_RG, MB_LAYER_S_RG, MB_OCC_RG, MB_OVER_TOL,
    # MB_DEN_CF_RG, MB_MIN_RAD/MB_MAX_RAD or MB_MIN_AXIS/MB_MAX_AXIS/
    # MB_MAX_ECC). In TOML, provide as [[membrane_specs]] tables.
    membrane_specs: list[dict[str, Any]] = field(default_factory=list)
    # Generic cytosolic filler (build_filler_protein_specs) total occupancy,
    # added on top of protein_specs to avoid an empty/flat background. None
    # disables it.
    filler_occupancy: float | None = None
    pdb_savefolder: str = "pdb-data"  # resolved against REPO_ROOT if relative
    target_shape: list[int] = field(
        default_factory=lambda: [184, 630, 630]
    )  # (Z, Y, X) voxels
    target_v_size: float = 5.0  # Å/voxel
    low_res_v_size: float = 10.0  # Å/voxel, polnet's placement-pass resolution
    membrane_potential_scale: float = (
        1.0  # unitless, on top of the auto-calibrated reference
    )
    seed: int | None = None

    # --- Microscope / physics ---
    micrograph_size: int | None = (
        None  # pixels, square; defaults to target_shape's XY extent
    )
    energy: float = 300.0  # keV
    dose_per_tilt: float = 3.0  # e⁻/Å², per tilt angle
    num_frames: int = 10  # movie frames per tilt
    cs: float = 2.0  # mm
    alpha: float = 0.1  # unitless, amplitude contrast ratio

    # --- Envelopes ---
    convergence_angle: float | None = None  # mrad
    cc: float | None = None  # mm
    energy_spread: float = 0.7  # eV (FWHM)
    deltaV_V: float = 0.06e-6  # unitless (ΔV/V)
    deltaI_I: float = 0.01e-6  # unitless (ΔI/I)
    dose_envelope: bool = False

    # --- Defocus ---
    defocus: float = 22000.0  # Å (positive = underfocus)

    # --- Tilt geometry ---
    min_tilt_angle: float = -45.0  # degrees
    max_tilt_angle: float = 45.0  # degrees
    n_tilts: int = 61
    tilt_axis: Literal["x", "y"] = "y"

    # --- Models ---
    scattering_model: Literal["multislice", "firstborn", "projection", "ctf"] = (
        "multislice"
    )
    aberration_model: Literal["holography", "ctf"] = "holography"
    noise_model: Literal["poisson", "none"] = "poisson"
    coincidence_radius: float = 1.5  # pixels
    ice_model: Literal["gd", "random", "none"] = "gd"
    ice_cache_dir: str | None = None  # defaults to the bundled ice-data/ice_cache
    ice_relax_steps: int = 0  # local MLBOP seam-relaxation steps for ice_model="gd"
    pad_fft: bool = False
    detector_model: Literal["none", "perfect", "k3_300kv", "k3_200kv"] = "none"

    # --- Post-processing ---
    normalize_tilt_series: bool = False
    save_exitwaves: bool = False

    # --- Compute ---
    device: str = "cpu"

    # --- Output ---
    output_dir: str = "./output/"
    filename: str = "tilt_series"


ConfigT = TypeVar("ConfigT", ParticleStackConfig, MicrographConfig, TiltSeriesConfig)


@overload
def load_config(
    path: str, config_cls: type[ParticleStackConfig] = ...
) -> ParticleStackConfig: ...
@overload
def load_config(path: str, config_cls: type[MicrographConfig]) -> MicrographConfig: ...
@overload
def load_config(path: str, config_cls: type[TiltSeriesConfig]) -> TiltSeriesConfig: ...
def load_config(
    path: str,
    config_cls: type[ParticleStackConfig]
    | type[MicrographConfig]
    | type[TiltSeriesConfig] = ParticleStackConfig,
) -> ParticleStackConfig | MicrographConfig | TiltSeriesConfig:
    """
    Load a config dataclass from a TOML file.

    Parameters
    ----------
    path : str
        Path to a TOML config file. May use tables (e.g. `[potential]`) to
        group fields for readability; all tables are flattened into one
        namespace before validation. List-of-tables fields (e.g.
        `[[protein_specs]]`) are passed through as `list[dict]`.
    config_cls : type[ParticleStackConfig] | type[MicrographConfig] | type[TiltSeriesConfig]
        Dataclass to populate from the TOML fields. Defaults to
        `ParticleStackConfig` for backward compatibility.

    Returns
    -------
    ParticleStackConfig | MicrographConfig | TiltSeriesConfig
        Config with unset fields filled from `config_cls` defaults.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    flat: dict = {}
    for value in raw.values():
        if isinstance(value, dict):
            flat.update(value)
    flat.update({k: v for k, v in raw.items() if not isinstance(v, dict)})
    config = config_cls(**flat)
    if not os.path.isabs(config.pdb_savefolder):
        config.pdb_savefolder = str(REPO_ROOT / config.pdb_savefolder)
    return config


def apply_overrides(config: ConfigT, overrides: dict) -> ConfigT:
    """
    Set fields on a config dataclass in place from a dict of overrides.

    Parameters
    ----------
    config : ParticleStackConfig | MicrographConfig
        Config to mutate.
    overrides : dict
        Field name -> value pairs, e.g. from parsed CLI arguments.

    Returns
    -------
    ParticleStackConfig | MicrographConfig
        The same `config` instance, mutated.
    """
    for key, value in overrides.items():
        setattr(config, key, value)
    return config
