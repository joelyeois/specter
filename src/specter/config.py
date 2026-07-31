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


def parse_scalar_or_range(value: str) -> tuple[float, float]:
    """
    Parse a "value" or "low,high" string into a (low, high) range.

    A bare scalar becomes a zero-width range (``low == high``), so callers
    can always uniformly sample between the two bounds without special-
    casing the constant case.

    Parameters
    ----------
    value : str
        Either a single number (e.g. ``"20"``) or two comma-separated
        numbers (e.g. ``"5000,15000"``).

    Returns
    -------
    tuple[float, float]
        ``(low, high)``.

    Raises
    ------
    ValueError
        If ``value`` isn't one or two comma-separated numbers.
    """
    parts = [p.strip() for p in value.split(",")]
    if len(parts) == 1:
        v = float(parts[0])
        return v, v
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    raise ValueError(f"Expected 'value' or 'low,high', got {value!r}")


@dataclass
class ParticleStackConfig:
    """Parameters for particle-stack generation, loaded from a TOML config file.

    Fields are ordered basic-first, advanced-last (mirrored by
    `specter.cli.simulate`'s panel layout): the first block is what most runs
    actually tune; everything under "Advanced" exists but is rarely touched.

    Set `cs_path` to drive generation from a CryoSPARC .cs file instead of
    randomly-sampled poses/CTF: `pixel_size`, `voltage`, `alpha`, defocus, and
    shifts are then read from the .cs file at run time via
    `extract_parameters_from_csfile` and take precedence over the
    corresponding fields below, which are unused in that mode.

    `dose`, `defocus`, `coincidence_radius`, and `potential_scale` each take
    either a single value (e.g. ``"20"``, constant for every particle) or a
    ``"low,high"`` range (e.g. ``"5000,15000"``, sampled uniformly per
    particle) -- see :func:`parse_scalar_or_range`.
    """

    # --- Structure & potential (basic) ---
    pdb_code: str
    assembly: bool = True
    num_pixels: int = 256
    pixel_size: float = 1.0  # Å

    # --- Microscope (basic) ---
    voltage: float = 300.0  # kV
    dose: str = "20"  # e⁻/Å²
    cs: float = 2.0  # mm
    alpha: float = 0.1  # unitless, amplitude contrast ratio

    # --- Sampling (basic) ---
    defocus: str = "5000,15000"  # Å
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
    batchsize: int = 5

    # --- Output (basic) ---
    output_dir: str = "./output/"
    filename: str = "particles"

    # --- Advanced ---
    pdb_savefolder: str = "pdb-data"  # resolved against REPO_ROOT if relative
    # if set, poses/CTF/pixel_size/voltage/alpha come from here
    cs_path: str | None = None
    num_frames: int | None = None
    convergence_angle: float | None = None  # mrad
    cc: float | None = None  # mm
    energy_spread: float = 0.7  # eV (FWHM)
    deltaV_V: float = 0.06e-6  # unitless (ΔV/V)
    deltaI_I: float = 0.01e-6  # unitless (ΔI/I)
    dose_envelope: bool = False
    bfactor: float | None = None  # Å²
    aberration_model: Literal["holography", "ctf"] = "holography"
    noise_model: Literal["poisson", "none"] = "poisson"
    coincidence_radius: str = "0"  # pixels
    ice_model: Literal["gd", "random", "none"] = "gd"
    ice_thickness: float = 0.0  # Å, 0 = minimum (particle box size)
    ice_cache_dir: str | None = None  # defaults to the bundled ice-data/ice_cache
    crowd_min_distance: float | None = None  # Å
    crowd_max_distance_z: float | None = None  # Å
    potential_scale: str = "1"  # unitless
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
    astigmatism: str = "0"  # Å, magnitude of dfu - dfv
    astigmatism_angle: str = "0,180"  # degrees, dfang
    phaseshift: str = "0"  # radians
    tiltx: str = "0"  # radians
    tilty: str = "0"  # radians
    trefoil1: str = "0"  # Å^3
    trefoil2: str = "0"  # Å^3

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
    "num_pixels": "Number of pixels per axis for the 3-D potential box.",
    "pixel_size": "Pixel size in Angstrom.",
    "voltage": "Electron beam accelerating voltage in kV.",
    "dose": "Dose in e-/A^2: a single value (e.g. 20) for constant dose per "
    "particle, or 'low,high' (e.g. 20,60) to sample uniformly per particle.",
    "cs": "Spherical aberration in mm (1-3 mm typical).",
    "alpha": "Amplitude contrast ratio.",
    "defocus": "Defocus in Angstrom: a single value (e.g. 8000) for constant "
    "defocus, or 'low,high' (e.g. 5000,15000) to sample uniformly per particle.",
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
    "batchsize": "Number of particles per forward pass.",
    "output_dir": "Directory to save .mrcs and .star files.",
    "filename": "Base name for output files (no extension).",
    "pdb_savefolder": "Folder to cache downloaded PDB files.",
    "cs_path": "Path to a CryoSPARC .cs file to drive generation from real "
    "poses/CTF instead of random sampling. Not yet used by this command "
    "(reserved for a future release).",
    "num_frames": "Number of movie frames. Defaults to int(dose) if not set.",
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
    "coincidence_radius": "Coincidence radius in pixels: a single value for "
    "constant radius, or 'low,high' to sample uniformly per particle.",
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
    "'low,high' to sample uniformly per particle.",
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
    "value for constant, or 'low,high' to sample uniformly per particle.",
    "astigmatism_angle": "Astigmatism angle in degrees: a single value or "
    "'low,high' range. Irrelevant when astigmatism is 0.",
    "phaseshift": "Phase shift in radians (e.g. from a Volta phase plate): a "
    "single value or 'low,high' range.",
    "tiltx": "Beam tilt (x) in radians: a single value or 'low,high' range.",
    "tilty": "Beam tilt (y) in radians: a single value or 'low,high' range.",
    "trefoil1": "First trefoil (3-fold astigmatism) component in Angstrom^3: a "
    "single value or 'low,high' range.",
    "trefoil2": "Second trefoil component in Angstrom^3: a single value or "
    "'low,high' range.",
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
    """Parameters for micrograph generation, loaded from a TOML config file."""

    # --- PDB / potential ---
    pdb_code: str
    assembly: bool = True
    pdb_savefolder: str = "pdb-data"  # resolved against REPO_ROOT if relative
    num_pixels: int = 256
    pixel_size: float = 1.0  # Å
    micrograph_size: int = 4096

    # --- Microscope / physics ---
    voltage: float = 300.0  # kV
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

    # --- Specimen source ---
    # Path to a pre-built (Z, Y, X) scattering-potential volume (.mrc/.mrcs/.pt).
    # When set, generation loads this volume directly instead of running the
    # polnet-based placement below -- the protein_specs/membrane_specs/
    # filler_occupancy/target_shape/target_v_size/low_res_v_size/
    # membrane_potential_scale/seed fields are then unused.
    volume_path: str = ""

    # --- Specimen placement (CryoETSpecimenGenerator; unused if volume_path is set) ---
    # One dict per protein species, e.g. {"PDB_CODE": "6qzp", "PMER_OCC": 0.6}.
    # Only "PDB_CODE" is required; see CryoETSpecimenGenerator's docstring for
    # the other optional polnet-style keys (PMER_OCC, PMER_L, PMER_L_MAX,
    # PMER_OVER_TOL, MMER_ISO). In TOML, provide as [[protein_specs]] tables.
    protein_specs: list[dict[str, Any]] = field(default_factory=list)
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
    voltage: float = 300.0  # kV
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


# Human-readable per-field descriptions for TiltSeriesConfig, used to build
# `specter simulate tiltseries --help` (see specter/cli/_click_options.py).
# Kept here, next to the dataclass, so adding/renaming a field and its help
# text happen in the same place.
TILT_SERIES_HELP: dict[str, str] = {
    "volume_path": "Path to a pre-built (Z, Y, X) scattering-potential volume "
    "(.mrc/.mrcs/.pt), already in scattering-potential units. When set, this "
    "is loaded directly instead of running polnet placement.",
    "protein_specs": "polnet protein species (TOML-only, [[protein_specs]] "
    "tables). Unused when volume_path is set.",
    "membrane_specs": "polnet membrane species (TOML-only, [[membrane_specs]] "
    "tables). Unused when volume_path is set.",
    "filler_occupancy": "Total occupancy of generic cytosolic filler added on "
    "top of protein_specs. None disables it. Unused when volume_path is set.",
    "pdb_savefolder": "Folder to cache downloaded PDB files.",
    "target_shape": "Output specimen volume shape in voxels (Z, Y, X). "
    "Unused when volume_path is set.",
    "target_v_size": "Target voxel size in Angstrom. Unused when volume_path is set.",
    "low_res_v_size": "Voxel size used for polnet's low-resolution placement "
    "pass, in Angstrom. Unused when volume_path is set.",
    "membrane_potential_scale": "Membrane potential scale factor, on top of "
    "the auto-calibrated reference. Unused when volume_path is set.",
    "seed": "Random seed for polnet's placement. Unused when volume_path is set.",
    "micrograph_size": "Output tilt-image size in pixels (square). Defaults "
    "to the XY dimension of the specimen volume.",
    "voltage": "Electron beam accelerating voltage in kV.",
    "dose_per_tilt": "Dose per tilt angle in e-/A^2.",
    "num_frames": "Number of movie frames per tilt.",
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
    "coincidence_radius": "Coincidence radius in pixels for direct-detector modelling.",
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
    """Parameters for hard-sphere-packed tomogram specimen generation, loaded
    from a TOML config file.

    Drives `specter.specimen.SpherePackingSpecimenGenerator`: packs several
    PDB-derived protein species -- each at its own real physical size -- via
    hard-sphere Random Sequential Addition (see `specter.crowding.
    pack_hard_spheres_3d`), renders each placed instance's real scattering
    potential, and saves the assembled volume as .mrc (directly usable as
    `TiltSeriesConfig.volume_path`) plus one copick-style .ndjson pick file
    per species.
    """

    # --- Specimen ---
    # One dict per protein species, e.g. {"pdb_source": "6qzp", "ratio": 1.0}.
    # Only "pdb_source" is required ("ratio" defaults to 1.0, i.e. uniform
    # across species left at default). In TOML, provide as [[protein_specs]]
    # tables.
    protein_specs: list[dict[str, Any]] = field(default_factory=list)
    target_shape: list[int] = field(
        default_factory=lambda: [128, 256, 256]
    )  # (Z, Y, X) voxels
    v_size: float = 5.0  # Å/voxel
    occupancy_fraction: float = 0.2  # target bare-sphere volume fraction
    gap_angstrom: float = 5.0  # minimum clearance between placed spheres
    pdb_savefolder: str = "pdb-data"  # resolved against REPO_ROOT if relative
    parameterization: Literal["kirkland", "lobato"] = "kirkland"
    seed: int | None = None

    # --- Ground-truth picks ---
    write_picks: bool = True
    annotation_version: str = "1.0"

    # --- Compute ---
    device: str = "cpu"

    # --- Output ---
    output_dir: str = "./output/"
    filename: str = "tomogram"


TOMOGRAM_HELP: dict[str, str] = {
    "protein_specs": "Protein species to pack (TOML-only, [[protein_specs]] "
    "tables), each {'pdb_source': <code or path>, 'ratio': <relative "
    "abundance weight, default 1.0>}.",
    "target_shape": "Output specimen volume shape in voxels (Z, Y, X).",
    "v_size": "Voxel size in Angstrom.",
    "occupancy_fraction": "Target packing density: candidates are drawn "
    "(species-ratio-weighted) until their combined bare-sphere volume "
    "reaches this fraction of the box volume. May not be fully reachable "
    "at high values -- see SpherePackingSpecimenGenerator's docstring.",
    "gap_angstrom": "Minimum clearance between placed spheres' surfaces, in Angstrom.",
    "pdb_savefolder": "Folder to cache downloaded PDB files.",
    "parameterization": "Atomic scattering-factor parameterization.",
    "seed": "Random seed.",
    "write_picks": "Write one copick-style .ndjson pick file per species "
    "alongside the volume.",
    "annotation_version": "Version string used in pick filenames "
    "('{species}-{version}_orientedpoint.ndjson').",
    "device": "Device for the packing step: cpu | cuda | cuda:0. Rendering "
    "always runs on CPU regardless.",
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
