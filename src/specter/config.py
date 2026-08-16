from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

import specter

# specter/__init__.py -> specter/ -> src/ -> repo root. Anchoring here (rather
# than cwd or a caller's __file__) makes path resolution work identically from
# the script (cwd = repo root, per README.md) and the notebook (cwd =
# demo-notebooks/particle_stack/, and Jupyter cells have no __file__ at all).
REPO_ROOT = Path(specter.__file__).resolve().parents[2]

#: Environment variable overriding where downloaded PDB/mmCIF files are cached,
#: mirroring how HF_HOME/TORCH_HOME work for those libraries.
PDB_CACHE_ENV_VAR = "SPECTER_PDB_CACHE"


#: Everything specter writes into a working directory lives under this one
#: folder, so a run leaves a single recognisable directory behind rather than
#: scattering caches and results at top level.
SPECTER_DATA_DIR = "specter-data"


def default_output_dir(artifact: str) -> str:
    """
    Default output location for one kind of simulated data.

    Each config class writes to a folder named for what it produces
    (`particles`, `micrographs`, `tiltseries`, `tomograms`) rather than a
    shared `output/`, so two different commands run in the same working
    directory don't pile their results into one folder -- and so the folder
    name alone says what is inside, without having to remember which command
    made it.

    Parameters
    ----------
    artifact : str
        Plural name of the artifact produced, e.g. ``"tomograms"``.

    Returns
    -------
    str
        ``specter-data/<artifact>``, relative to the current working
        directory like every other path in a specter config.
    """
    return os.path.join(SPECTER_DATA_DIR, artifact)


def default_pdb_cache_dir() -> str:
    """
    Default location for the downloaded-structure cache.

    Deliberately NOT anchored to `REPO_ROOT`: that only resolves to the repo
    for an editable install from a checkout. Installed as a wheel,
    `specter/__init__.py` lives in `site-packages/specter/`, so `parents[2]`
    would be the virtualenv's `lib/` directory and the cache would be written
    inside the venv.

    Relative, and therefore resolved against the current working directory --
    the same rule every other path in a specter config follows, so there is
    exactly one thing to remember: specter writes into `./specter-data/`. The
    tradeoff is that running from two different directories gives two caches;
    set `$SPECTER_PDB_CACHE` to an absolute path to share one between them.

    Returns
    -------
    str
        `$SPECTER_PDB_CACHE` when set, else `specter-data/pdb`.
    """
    override = os.environ.get(PDB_CACHE_ENV_VAR)
    if override:
        return override
    return os.path.join(SPECTER_DATA_DIR, "pdb")


#: A field that is either one number (constant for every particle) or a
#: two-element range sampled uniformly per particle. In TOML, write these as
#: plain numbers -- ``dose = 20`` or ``defocus = [5000, 15000]`` -- so config
#: files never mix quoted and unquoted numbers. The string forms (``"20"``,
#: ``"5000,15000"``) remain valid: CLI flags can only ever carry strings, and
#: older configs still parse. See :func:`parse_scalar_or_range`.
#:
#: ``str`` is deliberately first in the union: `cli/_click_options.py` types a
#: union-valued flag from its first member, and only ``str`` accepts both the
#: ``8000`` and ``5000,15000`` spellings on a command line.
ScalarOrRange = str | float | int | list[float]

#: TOML keys that have been renamed, mapped to their current spelling. Used
#: only to turn `load_config`'s "unknown field" error into one that says what
#: to write instead.
RENAMED_CONFIG_KEYS = {"grid": "carbon_film"}


def parse_scalar_or_range(value: ScalarOrRange) -> tuple[float, float]:
    """
    Parse a scalar or a two-element range into a (low, high) range.

    A bare scalar becomes a zero-width range (``low == high``), so callers
    can always uniformly sample between the two bounds without special-
    casing the constant case.

    Parameters
    ----------
    value : str or float or int or list of float
        A single number (e.g. ``20`` or ``"20"``), or a low/high pair as
        either a two-element sequence (e.g. ``[5000, 15000]``, the form a
        TOML config should use) or a comma-separated string
        (e.g. ``"5000,15000"``, the form a CLI flag has to use).

    Returns
    -------
    tuple[float, float]
        ``(low, high)``.

    Raises
    ------
    ValueError
        If ``value`` isn't one or two numbers.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), float(value)
    if isinstance(value, (list, tuple)):
        parts: list[Any] = list(value)
    else:
        parts = [p.strip() for p in str(value).split(",")]
    if len(parts) == 1:
        v = float(parts[0])
        return v, v
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    raise ValueError(f"Expected a scalar or a [low, high] pair, got {value!r}")


@dataclass
class ParticleStackConfig:
    """Parameters for particle-stack generation, loaded from a TOML config file.

    Fields are ordered basic-first, advanced-last (mirrored by
    `specter.cli.simulate`'s panel layout): the first block is what most runs
    actually tune; everything under "Advanced" exists but is rarely touched.

    Set `cs_path` (CryoSPARC .cs) or `star_path` (RELION .star) to drive
    generation from a real dataset instead of randomly-sampled poses/CTF:
    `pixel_size`, `voltage`, `alpha`, defocus, and shifts are then read from
    that file at run time via `extract_parameters_from_csfile` /
    `extract_parameters_from_starfile` and take precedence over the
    corresponding fields below, which are unused in that mode. The two are
    mutually exclusive.

    `dose`, `defocus`, `coincidence_radius`, `potential_scale` and the
    aberration-richness fields each take either a single number (e.g. ``20``,
    constant for every particle) or a ``[low, high]`` pair (e.g.
    ``[5000, 15000]``, sampled uniformly per particle). Comma-separated
    strings (``"5000,15000"``) are still accepted, since that's the only
    spelling a CLI flag can carry -- see :func:`parse_scalar_or_range`.
    """

    # --- Structure & potential (basic) ---
    pdb_code: str
    assembly: bool = True
    n_pixels: int = 256
    pixel_size: float = 1.0  # Å

    # --- Microscope (basic) ---
    voltage: float = 300.0  # kV
    dose: ScalarOrRange = 20.0  # e⁻/Å²
    cs: float = 2.0  # mm
    alpha: float = 0.1  # unitless, amplitude contrast ratio

    # --- Sampling (basic) ---
    defocus: ScalarOrRange = field(default_factory=lambda: [5000.0, 15000.0])  # Å
    shift: float = 2.0  # Å, max in-plane shift (uniform ±shift)
    n_particles: int = 20

    # --- Models (basic) ---
    scattering_model: Literal["multislice", "firstborn", "projection", "ctf"] = (
        "multislice"
    )
    detector_model: Literal[
        "none", "perfect", "k3_300kv", "k3_200kv", "falcon4i_300kv", "falcon4i_200kv"
    ] = "none"

    # --- Post-processing (basic) ---
    normalize_particles: bool = True
    save_exitwaves: bool = False
    save_clean_exitwaves: bool = False

    # --- Compute (basic) ---
    device: str = "cpu"
    # "auto" (the default) sizes the batch to the memory free on `device` at
    # run time, from the box geometry -- see specter.memory. An int pins it,
    # which is what a benchmark or a shared-GPU run wants; nothing about the
    # physics or the output depends on this, only speed and peak memory.
    batchsize: int | Literal["auto"] = "auto"

    # --- Output (basic) ---
    output_dir: str = field(default_factory=lambda: default_output_dir("particles"))
    filename: str = "particles"

    # --- Advanced ---
    # Relative to the current working directory, like any other CLI path
    # argument -- see default_pdb_cache_dir for the unset case.
    pdb_savefolder: str = field(default_factory=default_pdb_cache_dir)
    # if set, poses/CTF/pixel_size/voltage/alpha come from here (pick one)
    cs_path: str | None = None
    star_path: str | None = None
    n_frames: int | None = None
    convergence_angle: float | None = None  # mrad
    cc: float | None = None  # mm
    energy_spread: float = 0.7  # eV (FWHM)
    deltaV_V: float = 0.06e-6  # unitless (ΔV/V)
    deltaI_I: float = 0.01e-6  # unitless (ΔI/I)
    dose_envelope: bool = False
    bfactor: float | None = None  # Å²
    aberration_model: Literal["holography", "ctf"] = "holography"
    noise_model: Literal["poisson", "none"] = "poisson"
    coincidence_radius: ScalarOrRange = 0.0  # pixels
    ice_model: Literal["gd", "random", "none"] = "gd"
    ice_thickness: float = 0.0  # Å, 0 = minimum (particle box size)
    ice_cache_dir: str | None = None  # defaults to the bundled ice-data/ice_cache
    crowd_min_distance: float | None = None  # Å
    crowd_max_distance_z: float | None = None  # Å
    potential_scale: ScalarOrRange = 1.0  # unitless
    pad_fft: bool = True

    # --- Advanced: potential building ---
    potential_parameterization: Literal["shtyrov", "kirkland", "lobato"] = "shtyrov"
    potential_method: Literal["analytic", "2d", "3d"] = "analytic"
    rcut: float | None = None  # Å, auto-detected per-structure if unset
    conv_backend: str = "fftconvolve"
    periodic: bool = False
    # per-atom bonded species for Shtyrov typing; auto-detected from PDB bonds if
    # unset. Sized to the structure's atom count -- config-only, not a CLI flag.
    atom_species: list[str] | None = None
    shtyrov_params_path: str | None = None
    mmcif_filepath: str | None = None

    # --- Advanced: scattering ---
    ews_curvature_sign: Literal["negative", "positive"] = "positive"
    klim: float | None = None  # 1/Å
    rotate_mode: Literal["real", "fourier"] = "real"

    # --- Advanced: ice ---
    ice_parameterization: Literal["kirkland", "lobato", "shtyrov"] = "shtyrov"
    ice_relax_steps: int = 0

    # --- Advanced: crowding ---
    crowd_chunk_size: int | None = 1
    crowd_max_distance_xy: float | None = None  # Å
    crowd_method: Literal["2d", "3d"] = "3d"
    crowd_n_points: int | None = None
    crowd_seed: Literal["origin", "random"] = "origin"
    crowd_move_to_cpu: bool = False
    water_air_interface: bool = False

    # --- Advanced: reproducibility ---
    seed: int | None = None

    # --- Advanced: aberration richness for synthetic (non-.cs-driven) generation ---
    astigmatism: ScalarOrRange = 0.0  # Å, magnitude of dfu - dfv
    astigmatism_angle: ScalarOrRange = field(
        default_factory=lambda: [0.0, 180.0]
    )  # degrees, dfang
    phaseshift: ScalarOrRange = 0.0  # radians
    tiltx: ScalarOrRange = 0.0  # radians
    tilty: ScalarOrRange = 0.0  # radians
    trefoil1: ScalarOrRange = 0.0  # Å^3, coefficient of k^3 sin(3θ)
    trefoil2: ScalarOrRange = 0.0  # Å^3, coefficient of k^3 cos(3θ)
    # 4th-order, non-rotationally-symmetric terms -- 1/2 are secondary
    # astigmatism (n=4, m=±2), 3/4 are true 4-fold tetrafoil (n=4, m=±4);
    # see aberrations._functions.tetrafoil.
    tetrafoil1: ScalarOrRange = 0.0  # Å^4, coefficient of k^4 cos(2θ)
    tetrafoil2: ScalarOrRange = 0.0  # Å^4, coefficient of k^4 sin(2θ)
    tetrafoil3: ScalarOrRange = 0.0  # Å^4, coefficient of k^4 cos(4θ)
    tetrafoil4: ScalarOrRange = 0.0  # Å^4, coefficient of k^4 sin(4θ)

    # --- Advanced: anisotropic magnification ---
    # [[m00, m01], [m10, m11]], identity (no correction) by default. One fixed
    # matrix applied to every particle in a run (a microscope/session-level
    # calibration constant, not something that varies per particle).
    anisomag_m00: float = 1.0
    anisomag_m01: float = 0.0
    anisomag_m10: float = 0.0
    anisomag_m11: float = 1.0


# Human-readable per-field descriptions for ParticleStackConfig, used to build
# `specter simulate particles --help` (see specter/cli/_click_options.py). Kept
# here, next to the dataclass, so adding/renaming a field and its help text happen
# in the same place.
PARTICLE_STACK_HELP: dict[str, str] = {
    "pdb_code": "PDB accession code or path to a local .cif/.pdb file.",
    "assembly": "Fetch the biological assembly.",
    "n_pixels": "Number of pixels per axis for the 3-D potential box.",
    "pixel_size": "Pixel size in Angstrom.",
    "voltage": "Electron beam accelerating voltage in kV.",
    "dose": "Dose in e-/A^2: a single value (e.g. 20) for constant dose per "
    "particle, or 'low,high' (e.g. 20,60) to sample uniformly per particle "
    "(in a TOML config, write the range as [20, 60]).",
    "cs": "Spherical aberration in mm (1-3 mm typical).",
    "alpha": "Amplitude contrast ratio.",
    "defocus": "Defocus in Angstrom: a single value (e.g. 8000) for constant "
    "defocus, or 'low,high' (e.g. 5000,15000) to sample uniformly per particle "
    "(in a TOML config, write the range as [5000, 15000]).",
    "shift": "Max in-plane shift in Angstrom (uniform +/-shift).",
    "n_particles": "Number of particles to simulate.",
    "scattering_model": "Scattering model.",
    "detector_model": "Detector model.",
    "normalize_particles": "Normalize particles to zero mean and unit std.",
    "save_exitwaves": "Save exit wave magnitude and phase as separate .mrcs files.",
    "save_clean_exitwaves": "Save clean (particle-only, no ice) exit wave "
    "magnitude and phase.",
    "device": "Device to use: cpu | cuda | cuda:0 | 0,1,2,3. "
    "Comma-separated integers trigger multi-GPU Lightning DDP.",
    "batchsize": "Number of particles per forward pass. Unset (or 'auto' in "
    "a TOML config, which is the default) sizes the batch to the memory free "
    "on --device at run time; see specter.memory.recommend_batchsize.",
    "output_dir": "Directory to save .mrcs and .star files.",
    "filename": "Base name for output files (no extension).",
    "pdb_savefolder": "Folder to cache downloaded PDB files.",
    "cs_path": "Path to a CryoSPARC .cs file to drive generation from real "
    "pixel_size/voltage/alpha/poses/CTF instead of random sampling.",
    "star_path": "Path to a RELION .star file to drive generation from real "
    "pixel_size/voltage/alpha/poses/CTF instead of random sampling. Mutually "
    "exclusive with --cs_path.",
    "n_frames": "Number of movie frames. Defaults to int(dose) if not set.",
    "convergence_angle": "Beam convergence semi-angle in mrad, for the Cs "
    "(spatial coherence) envelope.",
    "cc": "Chromatic aberration coefficient in mm, for the Cc (temporal "
    "coherence) envelope.",
    "energy_spread": "FWHM of the beam energy spread in eV, used by the Cc envelope.",
    "deltaV_V": "Relative high-voltage instability, used by the Cc envelope.",
    "deltaI_I": "Relative objective-lens current instability, used by the Cc envelope.",
    "dose_envelope": "Apply the Grant & Grigorieff (2015) cumulative-dose envelope.",
    "bfactor": "Isotropic B-factor envelope in Angstrom^2.",
    "aberration_model": "Aberration model.",
    "noise_model": "Noise model. Use 'none' for no noise.",
    "coincidence_radius": "Effective coincidence exclusion radius in pixels "
    "(exclusion area = pi*r^2): a single value for constant radius, or "
    "'low,high' ([low, high] in TOML) to sample uniformly per particle.",
    "ice_model": "Ice model: 'gd' (samples the pre-generated IceBank "
    "cache), 'random' (cheap, low-realism), or 'none'.",
    "ice_thickness": "Ice thickness in Angstrom. 0 = minimum (particle box size).",
    "ice_cache_dir": "Directory of cached ice configs for ice_model='gd'. "
    "Defaults to the bundled ice-data/ice_cache.",
    "crowd_min_distance": "Minimum distance between crowded particles in "
    "Angstrom. Unset disables crowding.",
    "crowd_max_distance_z": "Maximum z-distance between crowded particles in Angstrom.",
    "potential_scale": "Potential scale factor (unitless, values < 1 "
    "approximate thicker ice): a single value for constant scale, or "
    "'low,high' ([low, high] in TOML) to sample uniformly per particle.",
    "pad_fft": "Pad the volume for FFT to avoid edge artifacts.",
    "potential_parameterization": "Atomic potential model used to build the "
    "structure's scattering potential.",
    "potential_method": "Voxelization method: 'analytic' (per-atom closed-form, "
    "no splat/FFT), '2d' (soft XY, hard Z), or '3d' (trilinear).",
    "rcut": "Cutoff radius in Angstrom for the atomic potential kernel. "
    "Auto-detected per-structure if unset.",
    "conv_backend": "Convolution backend for potential building. Unused for "
    "potential_method='analytic'.",
    "periodic": "Use periodic boundary conditions when voxelizing coordinates "
    "into the potential. Forces potential_method='3d'.",
    "shtyrov_params_path": "Override the bundled Shtyrov parameter table.",
    "mmcif_filepath": "Explicit mmCIF source for bond-typing, if pdb_code alone "
    "is ambiguous.",
    "ews_curvature_sign": "Ewald sphere curvature sign, matching CryoSPARC's "
    "convention.",
    "klim": "Reciprocal-space cutoff in 1/Angstrom. Unset uses the full Nyquist range.",
    "rotate_mode": "Volume rotation method: 'real' (trilinear interpolation) or "
    "'fourier' (no boundary artifacts).",
    "ice_parameterization": "Atomic potential model used for the ice volume "
    "specifically (independent of potential_parameterization).",
    "ice_relax_steps": "Local MLBOP seam-relaxation steps, only used when "
    "ice_model='gd' tiles multiple cached blocks.",
    "crowd_chunk_size": "Crowding volumes rotated per GPU batch. Raise for "
    "speed if you have RAM to spare; None rotates all at once.",
    "crowd_max_distance_xy": "Maximum xy-distance between crowded particles in "
    "Angstrom.",
    "crowd_method": "Poisson-disk sampling dimensionality for crowding particle "
    "placement.",
    "crowd_n_points": "Cap on the number of crowding duplicates. Unset means no cap.",
    "crowd_seed": "Crowding placement seed strategy: 'origin' (first point at "
    "the structure's center) or 'random'.",
    "crowd_move_to_cpu": "Move crowding intermediates to CPU between steps, to "
    "trade speed for lower GPU memory.",
    "water_air_interface": "Model a water-air interface when placing ice/"
    "crowding (bimodal density along z instead of uniform).",
    "seed": "RNG seed for pose/CTF/dose sampling. Auto-generated and logged if unset.",
    "astigmatism": "Astigmatism magnitude (dfu - dfv) in Angstrom: a single "
    "value for constant, or 'low,high' ([low, high] in TOML) to sample "
    "uniformly per particle.",
    "astigmatism_angle": "Astigmatism angle in degrees: a single value or "
    "'low,high' ([low, high] in TOML) range. Irrelevant when astigmatism is 0.",
    "phaseshift": "Phase shift in radians (e.g. from a Volta phase plate): a "
    "single value or 'low,high' ([low, high] in TOML) range.",
    "tiltx": "Beam tilt (x) in radians: a single value or 'low,high' "
    "([low, high] in TOML) range.",
    "tilty": "Beam tilt (y) in radians: a single value or 'low,high' "
    "([low, high] in TOML) range.",
    "trefoil1": "First trefoil (3-fold astigmatism) component in Angstrom^3: a "
    "single value or 'low,high' ([low, high] in TOML) range.",
    "trefoil2": "Second trefoil component in Angstrom^3: a single value or "
    "'low,high' ([low, high] in TOML) range.",
    "tetrafoil1": "Secondary astigmatism, k^4 cos(2*theta) coefficient in "
    "Angstrom^4: a single value or 'low,high' ([low, high] in TOML) range.",
    "tetrafoil2": "Secondary astigmatism, k^4 sin(2*theta) coefficient in "
    "Angstrom^4: a single value or 'low,high' ([low, high] in TOML) range.",
    "tetrafoil3": "Tetrafoil (4-fold astigmatism), k^4 cos(4*theta) coefficient "
    "in Angstrom^4: a single value or 'low,high' ([low, high] in TOML) range.",
    "tetrafoil4": "Tetrafoil (4-fold astigmatism), k^4 sin(4*theta) coefficient "
    "in Angstrom^4: a single value or 'low,high' ([low, high] in TOML) range.",
    "anisomag_m00": "Anisotropic magnification matrix element [0,0]. Identity "
    "(1.0) means no correction.",
    "anisomag_m01": "Anisotropic magnification matrix element [0,1]. Identity "
    "(0.0) means no correction.",
    "anisomag_m10": "Anisotropic magnification matrix element [1,0]. Identity "
    "(0.0) means no correction.",
    "anisomag_m11": "Anisotropic magnification matrix element [1,1]. Identity "
    "(1.0) means no correction.",
}


@dataclass
class MicrographConfig:
    """Parameters for micrograph generation, loaded from a TOML config file.

    `dose`, `defocus`, `coincidence_radius`, and `potential_scale` each take
    either a single number (e.g. ``20``, constant for every micrograph) or a
    ``[low, high]`` pair (e.g. ``[5000, 15000]``, sampled uniformly per
    micrograph) -- the same scalar-or-range convention as
    `ParticleStackConfig`/`TiltSeriesConfig`. Comma-separated strings
    (``"5000,15000"``) are still accepted, since that's the only spelling a
    CLI flag can carry -- see :func:`parse_scalar_or_range`.
    """

    # --- PDB / potential ---
    pdb_code: str
    assembly: bool = True
    # Relative to the current working directory, like any other CLI path
    # argument -- see default_pdb_cache_dir for the unset case.
    pdb_savefolder: str = field(default_factory=default_pdb_cache_dir)
    n_pixels: int = 256
    pixel_size: float = 1.0  # Å
    micrograph_size: int = 4096

    # --- Microscope / physics ---
    voltage: float = 300.0  # kV
    dose: ScalarOrRange = 20.0  # e⁻/Å²
    n_frames: int | None = None
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
    defocus: ScalarOrRange = field(default_factory=lambda: [5000.0, 15000.0])  # Å

    # --- Dataset size ---
    n_micrographs: int = 1

    # --- Models ---
    scattering_model: Literal["multislice", "firstborn", "projection", "ctf"] = (
        "multislice"
    )
    aberration_model: Literal["holography", "ctf"] = "holography"
    noise_model: Literal["poisson", "none"] = "poisson"
    coincidence_radius: ScalarOrRange = 0.7181  # pixels (effective exclusion radius)
    ice_model: Literal["gd", "random", "none"] = "gd"
    ice_thickness: float = 500.0  # Å, 0 = minimum (particle box size)
    ice_cache_dir: str | None = None  # defaults to the bundled ice-data/ice_cache
    crowd_min_distance: float | None = None  # Å
    crowd_max_distance_z: float | None = None  # Å
    water_air_interface: bool = True
    potential_scale: ScalarOrRange = 1.0  # unitless
    pad_fft: bool = False
    specimen_chunk_size: int | None = None
    detector_model: Literal["none", "perfect", "k3_300kv", "k3_200kv"] = "none"

    # --- Post-processing ---
    normalize_micrographs: bool = False
    save_exitwaves: bool = False
    save_clean_exitwaves: bool = False

    # --- Compute ---
    device: str = "cpu"

    # --- Output ---
    output_dir: str = field(default_factory=lambda: default_output_dir("micrographs"))
    filename: str = "micrographs"


# Human-readable per-field descriptions for MicrographConfig, used to build
# `specter simulate micrograph --help` (see specter/cli/_click_options.py). Kept
# here, next to the dataclass, so adding/renaming a field and its help text happen
# in the same place.
MICROGRAPH_HELP: dict[str, str] = {
    "pdb_code": "PDB accession code or path to a local .cif/.pdb file.",
    "assembly": "Fetch the biological assembly.",
    "pdb_savefolder": "Folder to cache downloaded PDB files.",
    "n_pixels": "Number of pixels per axis for the 3-D particle potential box.",
    "pixel_size": "Pixel size in Angstrom.",
    "micrograph_size": "Micrograph size in pixels (square).",
    "voltage": "Electron beam accelerating voltage in kV.",
    "dose": "Dose in e-/A^2: a single value (e.g. 20) for constant dose per "
    "micrograph, or 'low,high' (e.g. 20,60) to sample uniformly per micrograph "
    "(in a TOML config, write the range as [20, 60]).",
    "n_frames": "Number of movie frames. Defaults to int(dose) if not set.",
    "cs": "Spherical aberration in mm (1-3 mm typical).",
    "alpha": "Amplitude contrast ratio.",
    "convergence_angle": "Beam convergence semi-angle in mrad, for the Cs "
    "(spatial coherence) envelope.",
    "cc": "Chromatic aberration coefficient in mm, for the Cc (temporal "
    "coherence) envelope.",
    "energy_spread": "FWHM of the beam energy spread in eV, used by the Cc envelope.",
    "deltaV_V": "Relative high-voltage instability, used by the Cc envelope.",
    "deltaI_I": "Relative objective-lens current instability, used by the Cc envelope.",
    "dose_envelope": "Apply the Grant & Grigorieff (2015) cumulative-dose envelope.",
    "defocus": "Defocus in Angstrom: a single value (e.g. 8000) for constant "
    "defocus, or 'low,high' (e.g. 5000,15000) to sample uniformly per micrograph "
    "(in a TOML config, write the range as [5000, 15000]).",
    "n_micrographs": "Number of micrographs to simulate.",
    "scattering_model": "Scattering model.",
    "aberration_model": "Aberration model.",
    "noise_model": "Noise model. Use 'none' for no noise.",
    "coincidence_radius": "Effective coincidence exclusion radius in pixels "
    "(exclusion area = pi*r^2): a single value for constant radius, or "
    "'low,high' ([low, high] in TOML) to sample uniformly per micrograph.",
    "ice_model": "Ice model: 'gd' (samples the pre-generated IceBank cache), "
    "'random' (cheap, low-realism), or 'none'.",
    "ice_thickness": "Ice thickness in Angstrom. 0 = minimum (particle box size).",
    "ice_cache_dir": "Directory of cached ice configs for ice_model='gd'. "
    "Defaults to the bundled ice-data/ice_cache.",
    "crowd_min_distance": "Minimum distance between crowded particles in "
    "Angstrom. Defaults to the structure's max diameter; set to 0 to disable "
    "crowding.",
    "crowd_max_distance_z": "Maximum z-distance between crowded particles in Angstrom.",
    "water_air_interface": "Model a water-air interface when placing ice/"
    "crowding (bimodal density along z instead of uniform).",
    "potential_scale": "Potential scale factor (unitless, values < 1 "
    "approximate thicker ice): a single value for constant scale, or "
    "'low,high' ([low, high] in TOML) to sample uniformly per micrograph.",
    "pad_fft": "Pad the volume for FFT to avoid edge artifacts.",
    "specimen_chunk_size": "Slice chunk size for specimen generation. Lower "
    "if GPU memory is limited.",
    "detector_model": "Detector model.",
    "normalize_micrographs": "Normalize micrographs to zero mean and unit std.",
    "save_exitwaves": "Save exit wave magnitude and phase as separate .mrcs files.",
    "save_clean_exitwaves": "Save clean (particle-only, no ice) exit wave "
    "magnitude and phase.",
    "device": "Device to use: cpu | cuda | cuda:0.",
    "output_dir": "Directory to save .mrcs and .star files.",
    "filename": "Base name for output files (no extension).",
}


@dataclass
class TiltSeriesConfig:
    """Parameters for cryoET tilt-series generation, loaded from a TOML config file.

    Loads a pre-built specimen volume and simulates the tilted acquisition
    (multislice scattering, CTF, dose, detector, noise) -- this is the
    imaging half of the cryo-ET pipeline only. For the specimen-building
    half, see `specter build tomogram` (`TomogramConfig`/
    `specter.specimen.tomogram.TomogramSpecimenGenerator`), which writes a
    `.mrc` volume that `volume_path` below then loads directly.
    """

    # --- Specimen source ---
    # Path to a pre-built (Z, Y, X) scattering-potential volume (.mrc/.mrcs/.pt),
    # e.g. `specter build tomogram`'s own output. Required.
    volume_path: str = ""

    # --- Microscope / physics ---
    # Å/voxel of the loaded volume -- must match whatever produced it (e.g.
    # `TomogramConfig.voxel_size`), not auto-detected from the file itself.
    voxel_size: float = 5.0
    micrograph_size: int | None = (
        None  # pixels, square; defaults to the loaded volume's own XY extent
    )
    voltage: float = 300.0  # kV
    dose_per_tilt: float = 3.0  # e⁻/Å², per tilt angle
    n_frames: int = 10  # movie frames per tilt
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
    coincidence_radius: float = 0.5984  # pixels (effective exclusion radius)
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
    output_dir: str = field(default_factory=lambda: default_output_dir("tiltseries"))
    filename: str = "tilt_series"


# Human-readable per-field descriptions for TiltSeriesConfig, used to build
# `specter simulate tiltseries --help` (see specter/cli/_click_options.py).
# Kept here, next to the dataclass, so adding/renaming a field and its help
# text happen in the same place.
TILT_SERIES_HELP: dict[str, str] = {
    "volume_path": "Path to a pre-built (Z, Y, X) scattering-potential volume "
    "(.mrc/.mrcs/.pt), already in scattering-potential units -- e.g. `specter "
    "build tomogram`'s own output. Required.",
    "voxel_size": "Voxel size of the loaded volume, in Angstrom -- must "
    "match whatever produced it.",
    "micrograph_size": "Output tilt-image size in pixels (square). Defaults "
    "to the XY dimension of the specimen volume.",
    "voltage": "Electron beam accelerating voltage in kV.",
    "dose_per_tilt": "Dose per tilt angle in e-/A^2.",
    "n_frames": "Number of movie frames per tilt.",
    "cs": "Spherical aberration in mm (1-3 mm typical).",
    "alpha": "Amplitude contrast ratio.",
    "convergence_angle": "Beam convergence semi-angle in mrad, for the Cs "
    "(spatial coherence) envelope.",
    "cc": "Chromatic aberration coefficient in mm, for the Cc (temporal "
    "coherence) envelope.",
    "energy_spread": "FWHM of the beam energy spread in eV, used by the Cc envelope.",
    "deltaV_V": "Relative high-voltage instability, used by the Cc envelope.",
    "deltaI_I": "Relative objective-lens current instability, used by the Cc envelope.",
    "dose_envelope": "Apply the Grant & Grigorieff (2015) cumulative-dose envelope.",
    "defocus": "Defocus in Angstrom (positive = underfocus).",
    "min_tilt_angle": "Minimum tilt angle in degrees.",
    "max_tilt_angle": "Maximum tilt angle in degrees.",
    "n_tilts": "Number of tilt angles (evenly spaced from min to max).",
    "tilt_axis": "Tilt axis.",
    "scattering_model": "Scattering model.",
    "aberration_model": "Aberration model.",
    "noise_model": "Noise model. Use 'none' for no noise.",
    "coincidence_radius": "Effective coincidence exclusion radius in pixels "
    "(exclusion area = pi*r^2) for direct-detector modelling.",
    "ice_model": "Ice generation algorithm: 'gd' (IceBank cache), 'random' "
    "(cheap RandomIcemaker), or 'none'.",
    "ice_cache_dir": "Directory of cached ice configs for ice_model='gd'. "
    "Defaults to the bundled ice-data/ice_cache.",
    "ice_relax_steps": "Local MLBOP relaxation steps used to heal ice tile "
    "seams (ice_model='gd' only).",
    "pad_fft": "Pad volume for FFT to avoid multislice edge-wraparound "
    "artifacts under tilt.",
    "detector_model": "Detector model.",
    "normalize_tilt_series": "Normalize each tilt image to zero mean and unit std.",
    "save_exitwaves": "Save exit wave magnitude and phase as separate .mrcs files.",
    "device": "Device to use: cpu | cuda | cuda:0.",
    "output_dir": "Directory to save output files.",
    "filename": "Base name for output files (no extension).",
}


@dataclass
class TomogramConfig:
    """Parameters for tomogram specimen generation, loaded from a TOML
    config file.

    Drives `specter.specimen.tomogram.TomogramSpecimenGenerator`, the ONE
    generator behind `specter build tomogram` -- an optional composited
    organic membrane (`membrane`), optional scattered filament species
    (`filaments`/`actin`), and densely packed protein species
    (`targets`/`filler`, region-gated to `location: "cytosol"|"lumen"` when
    a membrane is present, otherwise everywhere is "cytosol" -- see
    `TomogramSpecimenGenerator`'s own module docstring). Generation order
    is membranes, then filaments, then protein fill; each stage avoids the
    previous ones' placements. Renders every placed instance's real
    scattering potential (always Shtyrov-parameterized, `PotentialBuilder`'s
    own default) and saves the assembled volume as .mrc (directly usable as
    `TiltSeriesConfig.volume_path`) plus one copick-style .ndjson pick file
    per species.

    Any combination of membrane/filaments/targets/filler is valid as long
    as at least one is non-empty -- there is no longer a separate non-
    membrane "sphere-packing" mode/generator to choose between.
    """

    # --- Specimen ---
    # One dict per target species, e.g. {"pdb_source": "6qzp", "n_copies":
    # 15}. "pdb_source" and "n_copies" required; "location" optional
    # ("cytosol"|"lumen", default "cytosol" -- only meaningful when
    # `membrane` is set, otherwise there's no "lumen" to place into). Placed
    # FIRST within its location, at this exact count, always exported to
    # picks. In TOML, provide as [[targets]] tables.
    targets: list[dict[str, Any]] = field(default_factory=list)
    # One dict per filler species, e.g. {"pdb_source": "1mbo"}. Only
    # "pdb_source" is required; optional "location" (as `targets` above,
    # default "cytosol") and "ratio" (relative abundance weight among OTHER
    # filler species sharing the same location, default 1.0 -- equal
    # attempt-weight across species if left at the default for all of
    # them). Placed SECOND within its location, around any already-placed
    # targets there, budgeted by filler_occupancy_fraction. Excluded from
    # picks by default (see write_picks/TomogramProteinSpec.role). In TOML,
    # provide as [[filler]] tables.
    filler: list[dict[str, Any]] = field(default_factory=list)
    # Additive to filler above: pull extra filler species from the
    # bundled reference tables (specter.specimen.cytosolic_filler.
    # PEI2016_CROWDING_TABLE and/or specter.specimen.cytosolic_filler.
    # CRYOETSIM_PARTICLE_TABLE), rather than hand-listing every species.
    # Both can be enabled at once -- their results are concatenated. Always
    # placed at location="cytosol" (these tables have no lumen/cytosol
    # distinction of their own).
    filler_from_pei2016: bool = False
    filler_from_cryoetsim: bool = False
    # Only affects filler_from_cryoetsim (CRYOETSIM_PARTICLE_TABLE has a
    # "category" column; PEI2016_CROWDING_TABLE doesn't, so this filter
    # has no effect there). None = all 4 usable categories (macromolecules,
    # distractors, transcription_translation, nucleosomes).
    filler_table_categories: list[str] | None = None
    # Mass range applied to whichever table(s) above are enabled.
    filler_table_max_mw_kda: float | None = None
    filler_table_min_mw_kda: float | None = None
    target_shape: list[int] = field(
        default_factory=lambda: [128, 256, 256]
    )  # (Z, Y, X) voxels
    voxel_size: float = 5.0  # Å/voxel
    # Target packing density for `ratio`-mode filler species, as a bare-
    # sphere fraction of EACH REGION's own volume it's placed in (the whole
    # box when `membrane` is empty, since then "cytosol" IS the whole box --
    # see TomogramSpecimenGenerator's own occupancy_fraction docstring).
    # Deliberately high by default -- RSA self-limits at its own physical
    # jamming ceiling rather than erroring, so filler simply packs until it
    # jams rather than needing this hand-tuned. Lower it for a deliberately
    # sparser filler layer, or if a small region (e.g. a tight vesicle
    # lumen) makes the candidate pool this implies impractically large.
    filler_occupancy_fraction: float = 0.5
    gap: float = 5.0  # minimum clearance between placed spheres
    # (z, y, x), matching target_shape's axis order. True on an axis lets a
    # placed instance's center stay in-bounds while its body pokes past
    # that wall (truncated naturally at render time) instead of being
    # rejected outright -- e.g. for a tomogram whose xy field of view is a
    # crop of a larger cellular region.
    clip_axes: list[bool] = field(default_factory=lambda: [False, False, False])
    # Relative to the current working directory, like any other CLI path
    # argument -- see default_pdb_cache_dir for the unset case.
    pdb_savefolder: str = field(default_factory=default_pdb_cache_dir)
    seed: int | None = None

    # --- Organic membrane (optional) ---
    # One or more dicts, [[membrane]] tables -- one membrane TEMPLATE each,
    # composited into the shared tomogram (see specter.specimen.tomogram.
    # MembraneInstance). Empty (default): no membrane at all -- the whole
    # tomogram is then one cytosol region. Keys are passed as **kwargs
    # straight into specter.specimen.membrane.MembraneGenerator -- e.g.
    # {"shape_backend": "spherical_harmonics", "sh_axes": [300.0, 300.0,
    # 300.0], "sh_amplitude": 0.15, "bilayer_thickness": 30.0} -- PLUS
    # three keys not real MembraneGenerator kwargs, popped before that call:
    #   - "n_copies" (int, default 1): expands this one entry into that
    #     many independent instances sharing the same template, each its
    #     own seed (config.seed + i, restarting at i=0 per entry -- editing/
    #     adding another [[membrane]] entry never perturbs an earlier
    #     entry's own instances). Can't be combined with an explicit
    #     "position_xyz" in the same entry (every copy would want the same
    #     spot) -- raises if both are given.
    #   - "position_xyz" = [x, y, z] (physical Angstrom offset from the
    #     tomogram's own center). Default omitted (None): resolved via
    #     collision-rejecting random placement against every other
    #     omitted-position instance (see TomogramSpecimenGenerator's own
    #     docstring) -- an instance that doesn't fit is dropped, not
    #     retried. Give it explicitly for manual placement instead (then
    #     n_copies must be 1).
    #   - "target_shape" = [Z, Y, X] voxels. Default omitted (None):
    #     MembraneGenerator auto-sizes a small local working grid from the
    #     organelle's own size (see its own docstring) instead of every
    #     instance rendering on a grid the size of the WHOLE tomogram
    #     canvas. Give it explicitly only if this instance genuinely needs
    #     a specific working-grid size.
    # voxel_size/seed/device/pdb_cache_dir still come from this config's own
    # voxel_size/seed/device/pdb_savefolder fields for every instance, not from
    # this dict (shape_backend one of "spherical_harmonics" (default) or
    # "swept_spline").
    membrane: list[dict[str, Any]] = field(default_factory=list)
    # Each {"pdb_source": <code or path>, "n_copies": 1, "parameterization":
    # "shtyrov"}. In TOML, provide as [[membrane_transmembrane_specs]] tables.
    # "n_copies" is per membrane instance, and is a request rather than a
    # guarantee -- MembraneGenerator.place_transmembrane warns and places
    # fewer if the surface can't fit that many at
    # membrane_min_transmembrane_spacing.
    # Applies across ALL membrane instances (not per-instance in v1). Only
    # meaningful when `membrane` is set (no bilayer to embed into otherwise).
    membrane_transmembrane_specs: list[dict[str, Any]] = field(default_factory=list)
    membrane_region_density_threshold: float | None = None
    membrane_region_max_passes: int = 300
    membrane_min_transmembrane_spacing: float = 40.0
    # Atomic scattering-factor parameterization for the targets/filler
    # protein-fill step (TomogramSpecimenGenerator's own `parameterization`
    # constructor kwarg, distinct from each MembraneGenerator instance's own
    # "parameterization" key inside its [[membrane]] dict, if set there --
    # that one's for the bilayer/transmembrane step specifically). A
    # top-level field rather than read from membrane[0] since it applies
    # regardless of whether membrane is even set. Named target_parameterization
    # (not bare parameterization) to keep it distinct from
    # potential_parameterization/ice_parameterization on the particle-stack
    # side of the codebase.
    target_parameterization: str = "shtyrov"

    # --- Filaments (optional, additive on top of membranes if present) ---
    # One dict per filament species, mapping straight onto
    # specter.specimen.filament.FilamentSpec's own kwargs, e.g.
    # {"code": "1TUB", "step": 85.0, "flex_deg": 3.0, "n_copies": 4}.
    # Placed via specter.specimen.filament.place_filaments -- specter-native
    # random-walk placement, with no region-gating and no collision
    # avoidance against the membrane shell or each other, but DOES get
    # avoided by targets/filler packing (placed right after membranes,
    # before protein fill -- see TomogramSpecimenGenerator's own
    # docstring). In TOML, provide as [[filaments]] tables.
    filaments: list[dict[str, Any]] = field(default_factory=list)
    # Convenience toggle: also place the bundled ACTIN_SPEC preset (real
    # F-actin helical repeat -- step/twist from Holmes/Egelman) without
    # hand-writing a [[filaments]] entry. Additive to filaments above (both
    # may be set at once). ACTIN_SPEC's own n_copies default (1) applies
    # here too -- for more instances or other single-strand species, use
    # [[filaments]] instead. Microtubules are NOT a filament species: they
    # are whole tubes, see [[microtubules]] below.
    actin: bool = False

    # --- Microtubules (optional, additive on top of filaments/membranes) ---
    # One dict per microtubule species, mapping straight onto
    # specter.specimen.filament.MicrotubuleSpec's own kwargs, e.g.
    # {"n_copies": 2, "n_protofilaments": 13, "bend_radius": 3e4}. Real
    # 13-protofilament tubes (lumen, A-lattice seam, tubulin dimer wall),
    # not the single protofilament PROTOFILAMENT_SPEC gives. Placed in the
    # same phase as filaments -- no region-gating, no collision avoidance,
    # but avoided by targets/filler afterwards. `code` defaults to a dimer
    # extracted from a deposited microtubule reconstruction (fetched and
    # cached on first use); `length` defaults to the volume diagonal so a
    # microtubule crosses the field. In TOML, provide as [[microtubules]]
    # tables.
    microtubules: list[dict[str, Any]] = field(default_factory=list)

    # --- Carbon support film (optional, single film) ---
    # Zero or one [[carbon_film]] table, mapping onto
    # specter.specimen.CarbonFilmSpec's own kwargs (thickness, hole_radius,
    # edge_fraction, edge_side, edge_roughness) -- e.g.
    # {"hole_radius": 6000.0, "edge_fraction": [0.02, 0.05]}. Painted
    # directly into the volume before anything else is placed; placement
    # (membranes/targets/filler) is NOT carbon-aware (a documented,
    # CTS-parity limitation -- see TomogramSpecimenGenerator's own
    # docstring). More than one entry raises. Empty (default): no carbon
    # film, pure ice.
    carbon_film: list[dict[str, Any]] = field(default_factory=list)

    # --- Gold fiducial beads (optional) ---
    # One dict per bead population, {"radius": <Angstrom>, "n_copies": 1,
    # "radius_cv": 0.0}. "radius" is required; "radius_cv" gives the
    # population size dispersity real colloidal gold has (see
    # specter.specimen.TomogramBeadSpec). Placed via the same RSA packing
    # used for membranes/targets/filler, avoiding the membrane shell and
    # any already-placed filaments -- NOT region-gated to cytosol/lumen
    # (see TomogramBeadSpec's own docstring). In TOML, provide as [[beads]]
    # tables.
    beads: list[dict[str, Any]] = field(default_factory=list)

    # How irregular each fiducial's boundary is, as an RMS fraction of its
    # radius -- see specter.specimen._grid's BeadGenerator. Either one
    # number or a [low, high] pair drawn per bead. 0.0 gives clean
    # spheres.
    bead_roughness: ScalarOrRange = 0.12

    # --- Ground-truth picks & segmentation ---
    write_picks: bool = True
    annotation_version: str = "1.0"
    # Save per-instance integer label volumes as .mrc alongside the density
    # volume ({filename}_protein_labels.mrc always; membrane mode also gets
    # {filename}_membrane_labels.mrc and {filename}_regions.mrc (0=cytosol/
    # 1=shell/2=lumen)) -- cast to uint16 (MRC has no int32 mode). The
    # segmentation mask, not a coordinate file, is the intended ground truth
    # for membrane geometry (a membrane surface has no single natural
    # "position" the way a protein does).
    write_segmentation: bool = True

    # --- Compute ---
    # "cpu" | "cuda" | "cuda:0" | a bare GPU index ("0") | a comma-separated
    # list of GPU indices ("0,1,2") to pool multiple GPUs for concurrent
    # per-species rendering (see render_workers below) | "auto" to pool
    # every visible CUDA GPU. A pooled value's first entry becomes the
    # primary device for everything else (see
    # specter.specimen._parallel_render.parse_device_pool).
    device: str = "cpu"
    # Device for the shared canvas tensors (volume/instance_labels/
    # membrane_labels), decoupled from `device` above (which stays the
    # compute device for rendering/rotation/field-generation regardless).
    # None (default): same as `device` -- one device for everything, the
    # original behaviour. "auto": estimate the canvas' own memory
    # footprint from target_shape/voxel_size and fall back to "cpu" if it
    # would exceed half of `device`'s currently free memory (see
    # TomogramSpecimenGenerator.recommend_accumulator_device's own
    # docstring). Explicit "cpu" always works regardless of that
    # estimate, keeping `device`="cuda" (fast rendering/rotation) while
    # letting the canvas itself be sized by system RAM instead of GPU
    # VRAM -- e.g. a large field of view at fine voxel_size can need tens of
    # GB, past any single GPU's VRAM but often fine in system RAM on a
    # workstation/cluster node. Rotation/rendering speed is unaffected
    # either way; the only added cost is moving each already-small
    # rotated chunk across devices once, not the canvas itself.
    accumulator_device: str | None = None
    # How many PDB species render/fetch concurrently within a single
    # tomogram: membrane_transmembrane_specs (rendered once, shared across
    # every [[membrane]] entry/n_copies copy, membrane mode only) and
    # targets/filler (cytosol/lumen protein-fill, rendered + PDB-fetched
    # once per tomogram, always) -- see MembraneGenerator/
    # TomogramSpecimenGenerator's own render_workers docstrings. Default 1:
    # fully serial, identical to the original behaviour. "auto" resolves
    # per-pool via specter.specimen._parallel_render.
    # recommend_render_workers -- min(n_species, 8), the measured sweet spot
    # from a full production-scale sweep (see that function's own
    # docstring); recommended over hand-picking a number.
    render_workers: int | Literal["auto"] = 1
    # Instances rotated per GPU batch, per species, in the targets/filler
    # protein-fill stage (TomogramSpecimenGenerator's own chunk_size
    # constructor kwarg -- rotate_volume batches ALL of a species' accepted
    # instances into one call when this is None, the original behaviour).
    # Fine at small scale, but a species with hundreds of instances (a real
    # filler species count at production target_shape/occupancy_fraction) can
    # then need many GB for one rotation call alone -- confirmed directly:
    # an 8.7 GB single allocation from one such batch, on top of whatever
    # else was already resident, was what actually tipped a production-
    # scale run into a CUDA OOM. None (default) preserves the original,
    # small-scale-safe behaviour; set e.g. 32-64 once a config's species
    # counts get into the hundreds. Named render_chunk_size (not bare
    # chunk_size) to keep it distinct from MicrographConfig's own
    # specimen_chunk_size, a different chunking knob entirely.
    render_chunk_size: int | None = None

    # --- Output ---
    output_dir: str = field(default_factory=lambda: default_output_dir("tomograms"))
    filename: str = "tomogram"


TOMOGRAM_HELP: dict[str, str] = {
    "targets": "Target protein species to pack (TOML-only, [[targets]] "
    "tables), each {'pdb_source': <code or path>, 'n_copies': <exact "
    "instance count>, 'location': 'cytosol'|'lumen' (optional, default "
    "'cytosol' -- only meaningful when [[membrane]] is set)}. Placed FIRST "
    "within its location, at this exact count, always exported to picks.",
    "filler": "Filler protein species to pack (TOML-only, [[filler]] "
    "tables), each {'pdb_source': <code or path>, 'location': "
    "'cytosol'|'lumen' (optional, default 'cytosol'), 'ratio': 1.0 "
    "(optional, relative attempt-weight among other filler species sharing "
    "the same location)}. Placed SECOND within its location, around any "
    "already-placed targets there. Excluded from picks by default (see "
    "write_picks).",
    "filler_from_pei2016": "Additive to filler: also pull filler species "
    "(location='cytosol') from the bundled PEI2016_CROWDING_TABLE (Pei et "
    "al. 2016 generic cytosolic crowding reference).",
    "filler_from_cryoetsim": "Additive to filler: also pull filler species "
    "(location='cytosol') from the bundled CRYOETSIM_PARTICLE_TABLE "
    "(CryoETSim dataset reference, Stojanovska et al. 2025).",
    "filler_table_categories": "Only used with filler_from_cryoetsim: "
    "restrict to these CRYOETSIM_PARTICLE_TABLE categories (macromolecules, "
    "distractors, transcription_translation, nucleosomes). None = all.",
    "filler_table_max_mw_kda": "Only used with filler_from_pei2016/"
    "filler_from_cryoetsim: exclude species above this mass, kDa.",
    "filler_table_min_mw_kda": "Only used with filler_from_pei2016/"
    "filler_from_cryoetsim: exclude species below this mass, kDa.",
    "target_shape": "Output specimen volume shape in voxels (Z, Y, X).",
    "voxel_size": "Voxel size in Angstrom.",
    "filler_occupancy_fraction": "Target packing density for filler "
    "species, as a bare-sphere fraction of EACH REGION's own volume it's "
    "placed in (the whole box when [[membrane]] is empty -- 'cytosol' is "
    "then the whole box). Deliberately high by default -- RSA self-limits "
    "at its own physical jamming ceiling rather than erroring, so filler "
    "simply packs until it jams rather than needing this hand-tuned. Lower "
    "it for a sparser filler layer, or if a small region (e.g. a tight "
    "vesicle lumen) makes the implied candidate pool impractically large.",
    "gap": "Minimum clearance between placed spheres' surfaces, in Angstrom.",
    "clip_axes": "(z, y, x) -- True on an axis lets a placed instance's "
    "body extend past that wall (truncated at render time) instead of "
    "being rejected outright. TOML-only (list[bool]).",
    "pdb_savefolder": "Folder to cache downloaded PDB files.",
    "seed": "Random seed.",
    "membrane": "One or more MembraneGenerator kwargs dicts (TOML-only, "
    "[[membrane]] tables, one per composited TEMPLATE) -- optional, empty "
    "by default (no membrane at all; the whole tomogram is then one "
    "cytosol region). e.g. {'shape_backend': 'spherical_harmonics', "
    "'n_copies': 3}. See MembraneGenerator's own docstring for the full "
    "per-backend parameter set; plus 'n_copies' (int, default 1, "
    "expands one entry into that many independently-seeded instances), "
    "'position_xyz' (physical Angstrom offset from the tomogram center, "
    "default omitted = collision-rejecting random placement), and "
    "'target_shape' (default omitted = auto-sized per instance).",
    "membrane_transmembrane_specs": "Transmembrane protein species (TOML-"
    "only, [[membrane_transmembrane_specs]] tables), each {'pdb_source': "
    "<code or path>, 'n_copies': 1, 'parameterization': 'shtyrov'}. Only "
    "meaningful when [[membrane]] is set, applies across all instances.",
    "membrane_region_density_threshold": "Passed through to "
    "TomogramSpecimenGenerator's own region_density_threshold.",
    "membrane_region_max_passes": "Passed through to "
    "TomogramSpecimenGenerator's own region_max_passes.",
    "membrane_min_transmembrane_spacing": "Minimum center-to-center "
    "spacing between placed transmembrane proteins, Angstrom. Only "
    "meaningful when [[membrane]] is set.",
    "target_parameterization": "Atomic scattering-factor parameterization "
    "for the targets/filler protein-fill step.",
    "filaments": "Filament species to scatter through the tomogram (TOML-"
    "only, [[filaments]] tables), each mapping onto "
    "specter.specimen.filament.FilamentSpec kwargs, e.g. {'code': '1TUB', "
    "'step': 85.0, 'flex_deg': 3.0, 'n_copies': 4}. Placed right after "
    "membranes, before targets/filler -- no region-gating, no collision "
    "avoidance against the membrane shell or each other, but targets/"
    "filler DO avoid already-placed filaments.",
    "actin": "Convenience toggle: also place the bundled ACTIN_SPEC preset "
    "(real F-actin helical repeat) without writing a [[filaments]] entry. "
    "Additive to filaments above. For more instances or other single-strand "
    "species, use [[filaments]]; for microtubules use [[microtubules]].",
    "microtubules": "Microtubule species to scatter through the tomogram "
    "(TOML-only, [[microtubules]] tables), each mapping onto "
    "specter.specimen.filament.MicrotubuleSpec kwargs, e.g. {'n_copies': 2, "
    "'n_protofilaments': 13, 'bend_radius': 30000.0}. Real 13-protofilament "
    "tubes with a lumen and an A-lattice seam -- not the single protofilament "
    "[[filaments]] with a tubulin dimer would give. Placed alongside "
    "filaments: no region-gating, no collision avoidance, but targets/filler "
    "DO avoid them.",
    "carbon_film": "Zero or one [[carbon_film]] table (TOML-only) describing a carbon "
    "support film, mapping onto specter.specimen.CarbonFilmSpec kwargs "
    "(thickness, hole_radius, edge_fraction, edge_side, edge_roughness). "
    "Painted into the volume before anything else is "
    "placed; not carbon-aware for placement (see TomogramSpecimenGenerator's "
    "own docstring). Empty (default): no carbon film.",
    "beads": "Gold fiducial bead populations to pack (TOML-only, [[beads]] "
    "tables), each {'radius': <Angstrom or [low, high]>, 'n_copies': 1}. "
    "A [low, high] radius draws a fresh size per bead, giving the population "
    "the dispersity real colloidal gold has. Placed via the same "
    "RSA packing as membranes/targets/filler, avoiding the membrane shell "
    "and already-placed filaments -- not region-gated to cytosol/lumen.",
    "bead_roughness": "How irregular each gold fiducial's boundary is, as an "
    "RMS fraction of its radius. One number, or a [low, high] pair drawn per "
    "bead so a population mixes near-round and misshapen particles. 0.0 gives "
    "clean spheres; 0.12-0.20 reads as genuinely irregular.",
    "write_picks": "Write one copick-style .ndjson pick file per species "
    "alongside the volume. Filler species (declared via 'ratio', not "
    "'n_copies') are included by default -- see TomogramProteinSpec's own "
    "role/export_picks docstrings for how to exclude them instead.",
    "annotation_version": "Version string used in pick filenames "
    "('{species}-{version}_orientedpoint.ndjson').",
    "write_segmentation": "Save per-instance integer label volumes as "
    ".mrc alongside the density volume -- the intended ground truth for "
    "membrane geometry specifically (not a coordinate file, since a "
    "membrane surface has no single natural 'position' the way a protein "
    "does). {filename}_protein_labels.mrc is always written; "
    "{filename}_membrane_labels.mrc and {filename}_regions.mrc "
    "(0=cytosol/1=shell/2=lumen) are added when [[membrane]] is set.",
    "device": "cpu | cuda | cuda:0 | 0,1,2 | auto. Drives the whole "
    "MembraneGenerator/TomogramSpecimenGenerator pipeline (shape field, "
    "bilayer profile, rasterization, transmembrane/targets/filler "
    "PotentialBuilder rendering) -- packing itself always runs on CPU "
    "regardless (vesin's neighbor list is both slower and OOM-prone on "
    "GPU at realistic particle counts). A comma-separated list of GPU "
    "indices (or 'auto', every visible GPU) pools those GPUs for "
    "concurrent per-species rendering (see render_workers); the first "
    "entry becomes the primary device for everything else.",
    "accumulator_device": "Device for the shared canvas tensors "
    "(volume/instance_labels/membrane_labels), decoupled from 'device' "
    "above (which stays the compute device regardless). None (default): "
    "same as 'device'. 'auto': estimate the canvas' own memory footprint "
    "and fall back to 'cpu' if it would exceed half of 'device''s "
    "currently free memory. Explicit 'cpu' always keeps 'device'='cuda' "
    "for fast rendering/rotation while letting the canvas itself be sized "
    "by system RAM instead of GPU VRAM -- useful for a large field of "
    "view at fine voxel_size whose canvas alone would exceed any single "
    "GPU's VRAM. Rotation/rendering speed is unaffected either way; only "
    "each already-small rotated chunk crosses devices, never the canvas "
    "itself.",
    "render_workers": "Number of PDB species rendered concurrently within "
    "one tomogram (membrane_transmembrane_specs and targets/filler each "
    "get their own concurrent build pass). Default 1 (serial, original "
    "behaviour) -- raise for tomograms with several species, especially "
    "with n_copies>1 [[membrane]] entries (all instances of one entry "
    "share one render pass). Set to 'auto' (TOML/Python config only -- the "
    "--render_workers CLI flag stays integer-only) to pick min(n_species, "
    "8) per pool automatically, the measured sweet spot from a full "
    "production-scale sweep -- see "
    "specter.specimen._parallel_render.recommend_render_workers. Round-"
    "robins across device's GPU pool when device is set to a "
    "comma-separated list or 'auto' (see device above); device choice was "
    "measured to barely matter at the recommended worker count.",
    "render_chunk_size": "Instances rotated per GPU batch, per species, when "
    "rendering targets/filler. None (default) rotates all of a species' "
    "accepted instances in one batched call -- fine at small scale, but a "
    "species with hundreds of instances can then need many GB for that "
    "one call. Set e.g. 32-64 once species counts get into the hundreds.",
    "output_dir": "Directory to save output files.",
    "filename": "Base name for the output volume (no extension).",
}


ConfigT = TypeVar(
    "ConfigT", ParticleStackConfig, MicrographConfig, TiltSeriesConfig, TomogramConfig
)


@overload
def load_config(
    path: str, config_cls: type[ParticleStackConfig] = ...
) -> ParticleStackConfig: ...
@overload
def load_config(path: str, config_cls: type[MicrographConfig]) -> MicrographConfig: ...
@overload
def load_config(path: str, config_cls: type[TiltSeriesConfig]) -> TiltSeriesConfig: ...
@overload
def load_config(path: str, config_cls: type[TomogramConfig]) -> TomogramConfig: ...
def load_config(
    path: str,
    config_cls: type[ParticleStackConfig]
    | type[MicrographConfig]
    | type[TiltSeriesConfig]
    | type[TomogramConfig] = ParticleStackConfig,
) -> ParticleStackConfig | MicrographConfig | TiltSeriesConfig | TomogramConfig:
    """
    Load a config dataclass from a TOML file.

    Parameters
    ----------
    path : str
        Path to a TOML config file. May use tables (e.g. `[potential]`) to
        group fields for readability; all tables are flattened into one
        namespace before validation. List-of-tables fields (e.g.
        `[[protein_specs]]`) are passed through as `list[dict]`.
    config_cls : type[ParticleStackConfig] | type[MicrographConfig] | type[TiltSeriesConfig] | type[TomogramConfig]
        Dataclass to populate from the TOML fields. Defaults to
        `ParticleStackConfig` for backward compatibility.

    Returns
    -------
    ParticleStackConfig | MicrographConfig | TiltSeriesConfig | TomogramConfig
        Config with unset fields filled from `config_cls` defaults.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    flat: dict = {}
    for value in raw.values():
        if isinstance(value, dict):
            flat.update(value)
    flat.update({k: v for k, v in raw.items() if not isinstance(v, dict)})
    unknown = sorted(set(flat) - {f.name for f in fields(config_cls)})
    if unknown:
        # Without this, a stale or mistyped table surfaces as a bare
        # "__init__() got an unexpected keyword argument", which names the
        # key but says nothing about what to write instead.
        detail = ", ".join(
            f"{key!r} (renamed to {RENAMED_CONFIG_KEYS[key]!r})"
            if key in RENAMED_CONFIG_KEYS
            else repr(key)
            for key in unknown
        )
        raise ValueError(
            f"{path}: unknown {config_cls.__name__} field(s): {detail}. "
            "Run the matching `specter` command with --help for the "
            "supported fields."
        )
    # Paths are deliberately NOT rewritten here: a relative path a user wrote
    # in a TOML (or passed on the CLI) is resolved against the current working
    # directory, like every other CLI tool's path argument. Only an omitted
    # pdb_savefolder gets a computed default -- see default_pdb_cache_dir.
    return config_cls(**flat)


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
