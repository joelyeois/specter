from __future__ import annotations

import os
import tomllib
import types
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import (
    Any,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

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
    ice_cache_dir: str | None = None  # defaults to the bundled ice_data/ice_cache
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

    # --- Advanced: scattering ---
    ews_curvature_sign: Literal["negative", "positive"] = "positive"
    klim: float | None = None  # 1/Å
    rotate_mode: Literal["real", "fourier"] = "real"

    # --- Advanced: ice ---
    # None follows potential_parameterization -- see run_particle_stack. Set it
    # only to deliberately parameterize the ice differently from the structure.
    ice_parameterization: Literal["kirkland", "lobato", "shtyrov"] | None = None
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
    "dose": "Total dose per particle in e-/Angstrom^2: a single value (e.g. 20) for a "
    "constant dose per particle, or 'low,high' (e.g. 20,60) to sample uniformly per particle "
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
    "n_frames": "Number of movie frames. Defaults to int(dose) if not set. "
    "Only affects the image when coincidence_radius > 0, which is what "
    "splits the dose into frames; ignored otherwise.",
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
    "Defaults to the bundled ice_data/ice_cache.",
    "crowd_min_distance": "Minimum distance between crowded particles in "
    "Angstrom. Unset disables crowding.",
    "crowd_max_distance_z": "Maximum z-distance between crowded particles in Angstrom.",
    "potential_scale": "Potential scale factor (unitless, values < 1 "
    "approximate thicker ice): a single value for constant scale, or "
    "'low,high' ([low, high] in TOML) to sample uniformly per particle.",
    "pad_fft": "Pad the volume for FFT to avoid edge artifacts.",
    "potential_parameterization": "Atomic potential model used to build the "
    "structure's scattering potential.",
    "potential_method": "Voxelization method for the structure's own "
    "potential: 'analytic' (per-atom closed-form, no splat/FFT), '2d' "
    "(soft XY, hard Z), or '3d' (trilinear). Ice is built by its own "
    "path and is unaffected by this.",
    "rcut": "Cutoff radius in Angstrom for the atomic potential kernel. "
    "Auto-detected per-structure if unset.",
    "conv_backend": "Convolution backend for potential building. Unused for "
    "potential_method='analytic'.",
    "periodic": "Wrap atom density across the box faces when voxelizing. "
    "Keep false for a particle -- a protein is a finite object, so wrapping "
    "smears its edge density onto the opposite face. Requires "
    "potential_method='3d'; 'analytic' and '2d' raise.",
    "shtyrov_params_path": "Override the bundled Shtyrov parameter table.",
    "ews_curvature_sign": "Ewald sphere curvature sign, matching CryoSPARC's "
    "convention.",
    "klim": "Reciprocal-space cutoff in 1/Angstrom. Unset uses the full Nyquist range.",
    "rotate_mode": "Volume rotation method: 'real' (trilinear interpolation) or "
    "'fourier' (no boundary artifacts).",
    "ice_parameterization": "Atomic potential model for the ice "
    "specifically. Unset, it follows potential_parameterization, so the "
    "ice and the structure it surrounds are modelled the same way; set it "
    "only to deliberately differ.",
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
    coincidence_radius: ScalarOrRange = 0.0  # pixels; 0 = plain Poisson
    ice_model: Literal["gd", "random", "none"] = "gd"
    ice_thickness: float = 500.0  # Å, 0 = minimum (particle box size)
    ice_cache_dir: str | None = None  # defaults to the bundled ice_data/ice_cache
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

    # --- Reproducibility ---
    seed: int | None = None

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
    "dose": "Total dose per micrograph in e-/Angstrom^2: a single value (e.g. 20) for a "
    "constant dose per micrograph, or 'low,high' (e.g. 20,60) to sample uniformly per micrograph "
    "(in a TOML config, write the range as [20, 60]).",
    "n_frames": "Number of movie frames. Defaults to int(dose) if not set. "
    "Only affects the image when coincidence_radius > 0, which is what "
    "splits the dose into frames; ignored otherwise.",
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
    "Defaults to the bundled ice_data/ice_cache.",
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
    "seed": "RNG seed for ice, crowding, pose and noise sampling. Auto-generated and logged if unset.",
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
    coincidence_radius: float = 0.0  # pixels; 0 = plain Poisson
    ice_model: Literal["gd", "random", "none"] = "gd"
    ice_cache_dir: str | None = None  # defaults to the bundled ice_data/ice_cache
    ice_relax_steps: int = 0  # local MLBOP seam-relaxation steps for ice_model="gd"
    pad_fft: bool = False
    detector_model: Literal["none", "perfect", "k3_300kv", "k3_200kv"] = "none"

    # --- Post-processing ---
    normalize_tilt_series: bool = False
    save_exitwaves: bool = False

    # --- Compute ---
    device: str = "cpu"

    # --- Reproducibility ---
    seed: int | None = None

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
    "dose_per_tilt": "Total dose for each tilt image in e-/Angstrom^2.",
    "n_frames": "Number of movie frames per tilt. Only affects the image "
    "when coincidence_radius > 0, which is what splits the dose into "
    "frames; ignored otherwise.",
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
    "Defaults to the bundled ice_data/ice_cache.",
    "ice_relax_steps": "Local MLBOP relaxation steps used to heal ice tile "
    "seams (ice_model='gd' only).",
    "pad_fft": "Pad volume for FFT to avoid multislice edge-wraparound "
    "artifacts under tilt.",
    "detector_model": "Detector model.",
    "normalize_tilt_series": "Normalize each tilt image to zero mean and unit std.",
    "save_exitwaves": "Save exit wave magnitude and phase as separate .mrcs files.",
    "device": "Device to use: cpu | cuda | cuda:0.",
    "seed": "RNG seed for ice, crowding, pose and noise sampling. Auto-generated and logged if unset.",
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
    #     entry's own instances). Every instance's position is resolved via
    #     collision-rejecting random placement against every other instance
    #     (see TomogramSpecimenGenerator's own docstring) -- an instance
    #     that doesn't fit is dropped, not retried. There is no manual-
    #     placement override.
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
    # One dict per bead population, {"radius": <Å>, "n_copies": 1,
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
    "per-backend parameter set; plus 'n_copies' (int, default 1, expands "
    "one entry into that many independently-seeded instances, each "
    "collision-rejecting-random-placed) and 'target_shape' (default "
    "omitted = auto-sized per instance).",
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


@dataclass
class IceCacheConfig:
    """Parameters for generating a library of amorphous-ice configurations,
    loaded from a TOML config file.

    Drives `specter.pipelines.run_build_ice_cache` (the `specter build ice`
    command), which runs `specter.ice.GradientSKIcemaker` once per
    configuration and saves the converged coordinates. The result is a
    directory of `config_NNN.pt` files an `IceBank` can draw crops from --
    point any simulation config's `ice_cache_dir` at it to use these instead
    of the bundled `ice_data/ice_cache`.

    Only the sampling geometry, the convergence budget, and the scheduling
    are configurable. The optimisation recipe itself (S(k) target, MLBOP
    weight, and the `mlbop_target = -0.413` eV/atom energy of real LDA-80K
    ice) is fixed: those are measured properties of the phase of ice being
    reproduced, and a cache mixing recipes would have `IceBank` drawing
    from several different phases interchangeably.
    """

    # --- Library ---
    # Number of independent configurations to generate. Each is a separate
    # optimisation run, so cost scales linearly -- and IceBank draws a random
    # rotation and translation from whichever config it picks, so a handful of
    # configs already gives a large space of distinct crops.
    num_configs: int = 8
    # Voxels along each side of the (cubic) periodic cell; the cell measures
    # n * dx Å. Cubic only -- IceBank stores one scalar box_L per config
    # and filters candidates against it, so a non-cubic cell is unrepresentable.
    n: int = 256
    dx: float = 1.0  # A/voxel
    # First seed; the i-th config uses seed_start + i and is saved under that
    # seed's name. Raise it past an existing library's highest seed to extend
    # that library rather than regenerate it.
    seed_start: int = 0

    # --- Optimisation ---
    # L-BFGS step ceiling per config. An upper bound only: the run stops early
    # once the loss plateaus (GradientSKIcemaker.optimize's tol/patience).
    n_steps: int = 600

    # --- Compute ---
    # "cpu" | "cuda" | "cuda:0" | a bare GPU index | a comma-separated list of
    # GPU indices ("0,1,2,3") | "auto" (every visible GPU). Multiple devices
    # shard whole configs across one worker process per device.
    device: str = "auto"

    # --- Output ---
    output_dir: str = field(default_factory=lambda: default_output_dir("ice"))
    # Regenerate configs whose file is already present, instead of skipping
    # them. Skipping is what lets an interrupted multi-hour run resume.
    overwrite: bool = False
    # Save IceBank.plot_diagnostics' energy/S(k) figures for the finished
    # library.
    diagnostics: bool = False


ICE_CACHE_HELP: dict[str, str] = {
    "num_configs": "Number of independent ice configurations to generate. "
    "Each costs a full optimisation run -- tens of minutes at the default "
    "n=256, dx=1.0.",
    "n": "Voxels along each side of the cubic periodic cell (the cell "
    "measures n*dx Angstrom). Must be at least as large as the biggest ice "
    "volume a simulation will request from it in any one dimension, or "
    "IceBank has to tile several crops together to serve the request.",
    "dx": "Voxel size in Angstrom used when optimising and when voxelizing "
    "crops. Set this to the pixel size of the simulations the cache is for.",
    "seed_start": "Seed of the first configuration; the i-th uses "
    "seed_start+i, and each is saved under its own seed's filename. Set "
    "this past an existing library's highest seed to extend it rather than "
    "regenerate it.",
    "n_steps": "L-BFGS step ceiling per configuration. An upper bound only "
    "-- a run whose loss plateaus stops early.",
    "device": "cpu | cuda | cuda:0 | a bare GPU index | a comma-separated "
    "list of GPU indices (0,1,2,3) | auto (every visible GPU, falling back "
    "to cpu). Several devices shard whole configurations across one worker "
    "process per device, so N GPUs generate a library roughly N times "
    "faster.",
    "output_dir": "Directory to write config_NNN.pt files and manifest.json "
    "to. Point a simulation config's ice_cache_dir at it to use the result. "
    "Never the bundled ice_data/ice_cache, which ships with the repository.",
    "overwrite": "Regenerate configurations already present in output_dir "
    "instead of skipping them. Skipping is what lets an interrupted run "
    "resume where it left off.",
    "diagnostics": "Save energy and S(k) diagnostic figures for the "
    "finished library to output_dir.",
}


@dataclass
class ReconstructionConfig:
    """Parameters for single-particle reconstruction, loaded from a TOML config file.

    Drives `specter.pipelines.run_reconstruction` (the `specter reconstruct
    particles` command), which builds a `specter.ghostbuster.Ghostbuster` from
    these fields and fits it. Every field maps one-to-one onto a `Ghostbuster`
    constructor argument, except the output/job fields, which decide where the
    run writes, and `test_run`/`bin_factor`, which pick `test_run()` over
    `run()`.

    Two of `Ghostbuster`'s arguments have no field here. `nps_weight` is a
    per-frequency tensor and `lpp_params` a nested table of laser-phase-plate
    settings; neither has a spelling a config field or a CLI flag can carry,
    so both remain Python-API-only.
    """

    # --- Data ---
    # The CryoSPARC .cs file supplying poses, CTF parameters and per-particle
    # scale, and the particle stack it indexes into. No .star equivalent yet:
    # Ghostbuster reads .cs only (io/_relion.py's reader is not wired in).
    cs_file: str
    mrc_file: str
    # Total fluence per image, e-/A^2. Sets the Poisson statistics the loss is
    # weighted by, so it has to be the real value for the dataset.
    dose_per_angstrom: float
    # Which gold-standard half-set to reconstruct. "gold" (the default)
    # reconstructs A and B and computes the halfmap FSC between them; "A"/"B"
    # select a single halfset (and name the outputs vol_A.mrc / vol_B.mrc),
    # useful for a quick test of one half; "all" uses every particle in one
    # single-volume run, ignoring the split entirely.
    halfset: Literal["A", "B", "all", "gold"] = "gold"
    # Reconstruct only the first N particles. Useful for a quick run before
    # committing to the full stack; None uses all of them.
    num_particles: int | None = None

    # --- Optimisation ---
    epochs: int = 5
    batchsize: int = 3
    # Volume learning rate. None freezes the volume, which is what refining
    # poses alone means.
    lr: float | None = 0.1
    scheduler: Literal[
        "LambdaLR",
        "OneCycleLR",
        "CosineAnnealingWarmRestarts",
        "MultiplicativeLR",
    ] = "LambdaLR"
    lr_decay: float = 0.1

    # --- Symmetry ---
    symmetry: str | None = None
    symmetry_mode: Literal["real", "fourier"] = "fourier"
    symmetry_batchsize: int | None = None

    # --- Sanity check ---
    # Run one epoch on bin_factor-binned images and stop, instead of the full
    # fit. Every file path, .cs field and physics setting is exercised, at a
    # fraction of the cost -- run this before a multi-hour job.
    test_run: bool = False
    bin_factor: int = 8

    # --- Compute ---
    # "cpu" | "cuda" | "cuda:N" | a bare GPU index | a comma-separated list of
    # GPU indices ("0,1"), which trains across them via Lightning DDP.
    device: str = "cuda"
    precision: str = "16-mixed"
    num_workers: int = 8

    # --- Output ---
    # Outputs land in output_dir/run_name/: vol{_A,_B}.mrc, params.json, an
    # epochs/ subdirectory of per-epoch volumes and FSC plots. Their names are
    # fixed by Reconstructor, so run_name names the directory, not the files.
    output_dir: str = field(
        default_factory=lambda: default_output_dir("reconstructions")
    )
    run_name: str = "reconstruction"

    # --- Job tracking (opt-in) ---
    # Setting `project` routes output through `specter.jobs` instead: the run
    # directory becomes job_base_dir/project/<job_id>/, numbered J001, J002,
    # ... and a job.json records the full parameter set, git commit and
    # status. That is what makes two halfset runs (halfset "A" then "B")
    # shareable into one job directory, and what lets a batch script pin a
    # job_id up front and resume into it.
    project: str | None = None
    job_id: str | None = None
    # Defaults to output_dir, so `--project foo` alone works and everything
    # still lands under specter-data/reconstructions/.
    job_base_dir: str | None = None

    # --- Reference maps (FSC logging only, never optimised against) ---
    fsc_ref: str | None = None
    fsc_mask: str | None = None
    cryosparc_ref: str | None = None
    # Rotate fsc_mask per particle, project it to 2D, and weight the
    # image-domain loss by it.
    use_2d_mask: bool = False

    # --- Refinement (unverified) ---
    # Pose and defocus refinement. Wired in but unverified: no test checks
    # recovered rotations, translations or defocus against ground truth, so
    # treat a run with these set as an experiment, not a result.
    lr_R: float | None = None
    lr_T: float | None = None
    lr_D: float | None = None
    # Constant offset in Angstrom added to every particle's dfu/dfv before the
    # transfer function is built. The starting point for lr_D refinement.
    defocus_offset: float = 0.0

    # --- Forward model ---
    # Defaults to rytov rather than the simulator's multislice: reconstruction
    # runs the forward model once per particle per step, and rytov is the
    # cheapest model that still carries curvature of the Ewald sphere.
    scattering_model: Literal["multislice", "rytov", "firstborn", "projection"] = (
        "rytov"
    )
    aberration_model: Literal["holography", "ctf"] = "holography"
    aberration_backend: Literal["legacy", "torch_ctf"] = "legacy"
    ews_curvature_sign: Literal["negative", "positive"] = "negative"
    bfactor: float | None = None
    # Hard frequency cutoff (1/A) applied to the simulated images.
    klim: float | None = None
    sparsity: float | None = None
    rotate_mode: Literal["real", "fourier"] = "real"
    learn_noise_model: bool = False
    use_ncc: bool = False


RECONSTRUCTION_HELP: dict[str, str] = {
    "cs_file": "CryoSPARC .cs file holding the particle poses, CTF "
    "parameters and per-particle scale.",
    "mrc_file": "Particle stack (.mrc/.mrcs) the .cs file indexes into.",
    "dose_per_angstrom": "Total fluence per image in e-/Angstrom^2. Sets the "
    "Poisson statistics the loss is weighted by, so it must be the dataset's "
    "real value.",
    "halfset": "Which gold-standard half-set to reconstruct. gold (the "
    "default) reconstructs A and B and computes the halfmap FSC between "
    "them; A or B alone reconstructs just that half, e.g. for a quick test; "
    "all uses every particle in one single-volume run, ignoring the split.",
    "num_particles": "Reconstruct only the first N particles instead of the "
    "whole stack.",
    "fsc_ref": "Reference volume (.mrc) for map-to-model FSC logging. Never "
    "optimised against -- reporting only.",
    "fsc_mask": "Mask volume (.mrc) applied before the FSC is computed.",
    "cryosparc_ref": "CryoSPARC's own reconstruction (.mrc), plotted "
    "alongside --fsc_ref for comparison. Only used when --fsc_ref is given "
    "too.",
    "use_2d_mask": "Rotate --fsc_mask per particle, project it to 2D, and "
    "weight the image-domain loss by it.",
    "scattering_model": "Wave propagation model used by the forward pass. "
    "rytov is the default here rather than multislice: the model runs once "
    "per particle per step, and rytov is the cheapest one that still carries "
    "Ewald-sphere curvature.",
    "aberration_model": "Aberration model applied to the exit wave.",
    "aberration_backend": "Which engine computes the transfer function. "
    "legacy uses aberrations.Aberration; torch_ctf uses the verified "
    "ctf.LegacyAberrationAdapter port.",
    "ews_curvature_sign": "Sign convention for Ewald-sphere curvature. "
    "negative matches CryoSPARC.",
    "bfactor": "Isotropic B-factor envelope in Angstrom^2 damping "
    "high-resolution signal in the forward model.",
    "defocus_offset": "Constant offset in Angstrom added to every particle's "
    "dfu/dfv. The starting point when --lr_D refinement is enabled.",
    "klim": "Hard frequency cutoff (1/Angstrom) applied to the simulated images.",
    "symmetry": "Point-group symmetry enforced on the volume, e.g. C3, D7, "
    "I1. Omit for C1.",
    "symmetry_batchsize": "Batch size used when applying symmetry operators "
    "to the volume. Lower it if symmetry expansion runs out of memory.",
    "symmetry_mode": "Domain in which symmetry is applied.",
    "epochs": "Number of passes over the particle stack.",
    "batchsize": "Particles per optimisation step.",
    "lr": "Learning rate for the volume. Unset (None) freezes the volume, "
    "which is what refining poses alone means.",
    "lr_R": "Learning rate for per-particle rotations. UNVERIFIED: pose "
    "refinement is wired in but no test checks recovered rotations against "
    "ground truth.",
    "lr_T": "Learning rate for per-particle translations. UNVERIFIED, see --lr_R.",
    "lr_D": "Learning rate for the defocus offset. UNVERIFIED, see --lr_R.",
    "scheduler": "Learning-rate schedule for the volume optimiser.",
    "lr_decay": "Decay rate for the LambdaLR schedule: multiplier = "
    "1 / (1 + lr_decay * sqrt(global_step)). Unused by other schedulers.",
    "sparsity": "L1 regularisation weight on the volume.",
    "rotate_mode": "Domain in which per-particle rotations are applied.",
    "learn_noise_model": "Estimate sigma^2(k) from the residuals during "
    "training (RELION-style) instead of assuming it.",
    "use_ncc": "Use a normalised cross-correlation loss instead of MSE.",
    "test_run": "Run one epoch on binned images and stop, instead of the "
    "full fit. Exercises every path, .cs field and physics setting at a "
    "fraction of the cost -- do this before a multi-hour job.",
    "bin_factor": "Spatial binning factor used by --test_run.",
    "device": "cpu | cuda | cuda:N | a bare GPU index | a comma-separated "
    "list of GPU indices (0,1), which trains across them via Lightning DDP "
    "with gradients all-reduced every step.",
    "precision": "Lightning trainer precision, e.g. 16-mixed or 32. Forced "
    "to 32 on CPU.",
    "num_workers": "Dataloader worker processes.",
    "output_dir": "Directory the run directory is created in.",
    "run_name": "Name of the run's own directory inside --output_dir. Names "
    "the directory, not the files: volume.mrc, params.json and epochs/ are "
    "named by the reconstructor.",
    "project": "Route output through specter.jobs instead of "
    "--output_dir/--run_name: the run lands in "
    "<job_base_dir>/<project>/J00N/ with a job.json recording every "
    "parameter, the git commit and the run's status. Omit for an untracked "
    "run.",
    "job_id": "Pin the job directory (e.g. J001) rather than auto-assigning "
    "the next one: resumes into it if it exists, creates it otherwise. "
    "Requires --project. This is how two halfset runs share one job.",
    "job_base_dir": "Root directory for job folders. Defaults to "
    "--output_dir, so --project alone keeps everything under "
    "specter-data/reconstructions/.",
}


ConfigT = TypeVar(
    "ConfigT",
    ParticleStackConfig,
    MicrographConfig,
    TiltSeriesConfig,
    TomogramConfig,
    IceCacheConfig,
    ReconstructionConfig,
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
@overload
def load_config(path: str, config_cls: type[IceCacheConfig]) -> IceCacheConfig: ...
@overload
def load_config(
    path: str, config_cls: type[ReconstructionConfig]
) -> ReconstructionConfig: ...
def load_config(
    path: str,
    config_cls: type[ParticleStackConfig]
    | type[MicrographConfig]
    | type[TiltSeriesConfig]
    | type[TomogramConfig]
    | type[IceCacheConfig]
    | type[ReconstructionConfig] = ParticleStackConfig,
) -> (
    ParticleStackConfig
    | MicrographConfig
    | TiltSeriesConfig
    | TomogramConfig
    | IceCacheConfig
    | ReconstructionConfig
):
    """
    Load a config dataclass from a TOML file.

    Parameters
    ----------
    path : str
        Path to a TOML config file. May use tables (e.g. `[potential]`) to
        group fields for readability; all tables are flattened into one
        namespace before validation. List-of-tables fields (e.g.
        `[[protein_specs]]`) are passed through as `list[dict]`.
    config_cls : type[ParticleStackConfig] | type[MicrographConfig] | type[TiltSeriesConfig] | type[TomogramConfig] | type[IceCacheConfig] | type[ReconstructionConfig]
        Dataclass to populate from the TOML fields. Defaults to
        `ParticleStackConfig` for backward compatibility.

    Returns
    -------
    ParticleStackConfig | MicrographConfig | TiltSeriesConfig | TomogramConfig | IceCacheConfig | ReconstructionConfig
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
    valid = {f.name for f in fields(config)}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        # A blind setattr would happily attach an override under a name no
        # field reads, so the flag would parse, be accepted, and do nothing.
        raise ValueError(
            f"apply_overrides: no such field on {type(config).__name__}: "
            f"{', '.join(repr(k) for k in unknown)}. Valid fields: "
            f"{', '.join(sorted(valid))}."
        )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# ---------------------------------------------------------------------------
# Validation
#
# Physically impossible values used to travel all the way into the simulation
# before failing -- a zero pixel_size surfaced as "ZeroDivisionError: float
# division by zero", a negative n_pixels as "Trying to create tensor with
# negative dimension", each ~10 s and six frames deep, naming nothing the user
# typed. Worse, a negative dose or an amplitude contrast ratio of 1.5 ran to
# completion and produced a plausible-looking, meaningless stack.
#
# These checks run off the config alone, before a structure is fetched or a
# voxel is written.
# ---------------------------------------------------------------------------


def _fail(field: str, value: Any, requirement: str) -> None:
    raise ValueError(f"{field}={value!r} is invalid: {requirement}.")


def _is_scalar(value: Any) -> bool:
    """
    A plain number this module can compare against a bound.

    Skips None (unset), strings (sentinels like ``batchsize="auto"``, and
    ranges like ``"20,60"`` which `_require_ordered` handles), bools, and
    anything list-like -- several fields hold a ``[low, high]`` pair or a whole
    list of species specs, and ``[0.0, 0.2] < 0`` is a TypeError, not a check.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_positive(config: Any, *fields: str) -> None:
    """Fields that are meaningless at zero (counts, sizes, voltages)."""
    for name in fields:
        value = getattr(config, name, None)
        if not _is_scalar(value):
            continue
        if value <= 0:
            _fail(name, value, "must be greater than 0")


def _require_non_negative(config: Any, *fields: str) -> None:
    """Fields where zero is meaningful ("off") but negative is not."""
    for name in fields:
        value = getattr(config, name, None)
        if not _is_scalar(value):
            continue
        if value < 0:
            _fail(name, value, "must be 0 or greater")


def _require_range(config: Any, name: str, low: float, high: float) -> None:
    value = getattr(config, name, None)
    if not _is_scalar(value):
        return
    if not low <= value <= high:
        _fail(name, value, f"must be between {low} and {high}")


def _require_ordered(config: Any, *fields: str) -> None:
    """
    Scalar-or-[low, high] fields: reject a reversed pair.

    `--dose 60,20` sampled uniformly between 60 and 20, which torch is happy
    to do and which silently yields nothing at all like the intended range.
    """
    for name in fields:
        value = getattr(config, name, None)
        if value is None:
            continue
        try:
            low, high = parse_scalar_or_range(value)
        except ValueError as exc:
            _fail(name, value, str(exc))
        if low > high:
            _fail(
                name, value, f"range is reversed -- low ({low}) exceeds high ({high})"
            )


def _require_positive_ordered(config: Any, *fields: str) -> None:
    """`_require_ordered`, plus both bounds strictly positive."""
    _require_ordered(config, *fields)
    for name in fields:
        value = getattr(config, name, None)
        if value is None:
            continue
        low, _ = parse_scalar_or_range(value)
        if low <= 0:
            _fail(name, value, "must be greater than 0")


def _require_non_negative_ordered(config: Any, *fields: str) -> None:
    """`_require_ordered`, plus both bounds at 0 or above."""
    _require_ordered(config, *fields)
    for name in fields:
        value = getattr(config, name, None)
        if value is None:
            continue
        low, _ = parse_scalar_or_range(value)
        if low < 0:
            _fail(name, value, "must be 0 or greater")


def _require_existing_file(config: Any, *fields: str) -> None:
    for name in fields:
        value = getattr(config, name, None)
        if value is None:
            continue
        if not Path(str(value)).is_file():
            _fail(name, value, "no such file")


def _require_valid_literals(config: Any) -> None:
    """
    Every ``Literal``-typed field actually holds one of its allowed values.

    Click validates the CLI flags, but a TOML file bypasses that entirely --
    nothing enforces a `Literal` at runtime, so `scattering_model = "banana"`
    in a config file used to sail through to the simulator.
    """
    hints = get_type_hints(type(config))
    for f in fields(config):
        hint = hints.get(f.name)
        if get_origin(hint) in (Union, types.UnionType):
            args = [a for a in get_args(hint) if a is not type(None)]
            hint = args[0] if args else None
        if get_origin(hint) is not Literal:
            continue
        value = getattr(config, f.name, None)
        if value is None:
            continue
        allowed = get_args(hint)
        if value not in allowed:
            _fail(f.name, value, f"must be one of {', '.join(map(repr, allowed))}")


def validate_config(config: Any) -> None:
    """
    Reject physically impossible values before any work is done.

    Called at the top of every pipeline, so it covers both the CLI (which
    reaches a pipeline after `load_config` + `apply_overrides`) and direct
    Python callers constructing a config themselves.

    Parameters
    ----------
    config : ParticleStackConfig | MicrographConfig | TiltSeriesConfig | TomogramConfig

    Raises
    ------
    ValueError
        Naming the offending field, its value, and what was required.
    """
    _require_valid_literals(config)

    # Shared across the imaging configs.
    _require_positive(
        config,
        "n_pixels",
        "pixel_size",
        "voltage",
        "n_particles",
        "n_micrographs",
        "micrograph_size",
        "n_frames",
        "batchsize",
        "n_tilts",
        "voxel_size",
        "specimen_chunk_size",
        "crowd_chunk_size",
        "crowd_n_points",
        "render_chunk_size",
        "membrane_region_max_passes",
        # IceCacheConfig. "n"/"dx" are the ice cell's own geometry -- no other
        # config class has a field by either name.
        "num_configs",
        "n",
        "dx",
        "n_steps",
        # ReconstructionConfig.
        "dose_per_angstrom",
        "num_particles",
        "symmetry_batchsize",
        "epochs",
        "bin_factor",
    )
    _require_non_negative(
        config,
        "seed_start",
        "cs",
        "ice_thickness",
        "shift",
        "bfactor",
        "ice_relax_steps",
        "energy_spread",
        "convergence_angle",
        "cc",
        "deltaV_V",
        "deltaI_I",
        "crowd_min_distance",
        "crowd_max_distance_z",
        "crowd_max_distance_xy",
        "gap",
        "rcut",
        "klim",
        "membrane_min_transmembrane_spacing",
        "filler_table_min_mw_kda",
        "filler_table_max_mw_kda",
        "dose_per_tilt",
        # ReconstructionConfig. A learning rate of exactly 0 is a legitimate
        # way to freeze one parameter group while refining another.
        "lr",
        "lr_R",
        "lr_T",
        "lr_D",
        "lr_decay",
        "sparsity",
        "num_workers",
    )
    _require_range(config, "alpha", 0.0, 1.0)
    _require_range(config, "filler_occupancy_fraction", 0.0, 1.0)
    _require_range(config, "membrane_region_density_threshold", 0.0, 1.0)

    _require_positive_ordered(config, "dose", "potential_scale")
    _require_non_negative_ordered(
        config, "defocus", "coincidence_radius", "astigmatism", "bead_roughness"
    )
    _require_ordered(config, "astigmatism_angle", "phaseshift")

    _require_existing_file(
        config,
        "cs_path",
        "star_path",
        "shtyrov_params_path",
        # ReconstructionConfig. Every one of these is read at Ghostbuster
        # construction time, so a typo'd path otherwise surfaces minutes in.
        "cs_file",
        "mrc_file",
        "fsc_ref",
        "fsc_mask",
        "cryosparc_ref",
    )

    # A job id names a directory inside a project, so on its own it has
    # nothing to identify -- and the run would silently fall back to the
    # untracked output_dir/run_name layout instead.
    if getattr(config, "job_id", None) is not None and (
        getattr(config, "project", None) is None
    ):
        _fail(
            "job_id",
            config.job_id,
            "names a directory inside a project, so it needs project set "
            "too; drop it to write to output_dir/run_name instead",
        )

    min_tilt = getattr(config, "min_tilt_angle", None)
    max_tilt = getattr(config, "max_tilt_angle", None)
    if min_tilt is not None and max_tilt is not None and min_tilt >= max_tilt:
        _fail(
            "min_tilt_angle",
            min_tilt,
            f"must be less than max_tilt_angle ({max_tilt})",
        )

    # A structure given as a path has to exist; a 4-character accession is
    # fetched, so it can only be checked by trying.
    pdb_code = getattr(config, "pdb_code", None)
    if pdb_code and (
        os.sep in str(pdb_code) or str(pdb_code).endswith((".cif", ".pdb"))
    ):
        if not Path(str(pdb_code)).is_file():
            _fail("pdb_code", pdb_code, "looks like a path, but no such file")
