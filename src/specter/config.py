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
    "pixel_size/voltage/alpha/poses/CTF instead of random sampling.",
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

    Loads a pre-built specimen volume and simulates the tilted acquisition
    (multislice scattering, CTF, dose, detector, noise) -- this is the
    imaging half of the cryo-ET pipeline only. For the specimen-building
    half, see `specter build tomogram` (`TomogramConfig`/
    `specter.specimen.tomogram.MembraneTomogramGenerator`), which writes a
    `.mrc` volume that `volume_path` below then loads directly.
    """

    # --- Specimen source ---
    # Path to a pre-built (Z, Y, X) scattering-potential volume (.mrc/.mrcs/.pt),
    # e.g. `specter build tomogram`'s own output. Required.
    volume_path: str = ""

    # --- Microscope / physics ---
    # Å/voxel of the loaded volume -- must match whatever produced it (e.g.
    # `TomogramConfig.v_size`), not auto-detected from the file itself.
    target_v_size: float = 5.0
    micrograph_size: int | None = (
        None  # pixels, square; defaults to the loaded volume's own XY extent
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
    "(.mrc/.mrcs/.pt), already in scattering-potential units -- e.g. `specter "
    "build tomogram`'s own output. Required.",
    "target_v_size": "Voxel size of the loaded volume, in Angstrom -- must "
    "match whatever produced it.",
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
    """Parameters for tomogram specimen generation, loaded from a TOML
    config file.

    Drives `specter.specimen.tomogram.MembraneTomogramGenerator`, the ONE
    generator behind `specter build tomogram` -- an optional composited
    organic membrane (`membrane`), optional scattered filament species
    (`filaments`/`actin`), and densely packed protein species
    (`targets`/`filler`, region-gated to `location: "cytosol"|"lumen"` when
    a membrane is present, otherwise everywhere is "cytosol" -- see
    `MembraneTomogramGenerator`'s own module docstring). Generation order
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
    v_size: float = 5.0  # Å/voxel
    # Target packing density for `ratio`-mode filler species, as a bare-
    # sphere fraction of EACH REGION's own volume it's placed in (the whole
    # box when `membrane` is empty, since then "cytosol" IS the whole box --
    # see MembraneTomogramGenerator's own occupancy_fraction docstring).
    # Deliberately high by default -- RSA self-limits at its own physical
    # jamming ceiling rather than erroring, so filler simply packs until it
    # jams rather than needing this hand-tuned. Lower it for a deliberately
    # sparser filler layer, or if a small region (e.g. a tight vesicle
    # lumen) makes the candidate pool this implies impractically large.
    filler_occupancy_fraction: float = 0.5
    gap_angstrom: float = 5.0  # minimum clearance between placed spheres
    # (z, y, x), matching target_shape's axis order. True on an axis lets a
    # placed instance's center stay in-bounds while its body pokes past
    # that wall (truncated naturally at render time) instead of being
    # rejected outright -- e.g. for a tomogram whose xy field of view is a
    # crop of a larger cellular region.
    clip_axes: list[bool] = field(default_factory=lambda: [False, False, False])
    pdb_savefolder: str = "pdb-data"  # resolved against REPO_ROOT if relative
    seed: int | None = None

    # --- Organic membrane (optional) ---
    # One or more dicts, [[membrane]] tables -- one membrane TEMPLATE each,
    # composited into the shared tomogram (see specter.specimen.tomogram.
    # MembraneInstance). Empty (default): no membrane at all -- the whole
    # tomogram is then one cytosol region. Keys are passed as **kwargs
    # straight into specter.specimen.membrane.MembraneGenerator -- e.g.
    # {"shape_backend": "spherical_harmonics", "sh_axes_a": [300.0, 300.0,
    # 300.0], "sh_amplitude": 0.15, "bilayer_thickness_a": 30.0} -- PLUS
    # three keys not real MembraneGenerator kwargs, popped before that call:
    #   - "n_instances" (int, default 1): expands this one entry into that
    #     many independent instances sharing the same template, each its
    #     own seed (config.seed + i, restarting at i=0 per entry -- editing/
    #     adding another [[membrane]] entry never perturbs an earlier
    #     entry's own instances). Can't be combined with an explicit
    #     "position_xyz" in the same entry (every copy would want the same
    #     spot) -- raises if both are given.
    #   - "position_xyz" = [x, y, z] (physical Angstrom offset from the
    #     tomogram's own center). Default omitted (None): resolved via
    #     collision-rejecting random placement against every other
    #     omitted-position instance (see MembraneTomogramGenerator's own
    #     docstring) -- an instance that doesn't fit is dropped, not
    #     retried. Give it explicitly for manual placement instead (then
    #     n_instances must be 1).
    #   - "target_shape_zyx" = [Z, Y, X] voxels. Default omitted (None):
    #     MembraneGenerator auto-sizes a small local working grid from the
    #     organelle's own size (see its own docstring) instead of every
    #     instance rendering on a grid the size of the WHOLE tomogram
    #     canvas. Give it explicitly only if this instance genuinely needs
    #     a specific working-grid size.
    # v_size/seed/device/pdb_cache_dir still come from this config's own
    # v_size/seed/device/pdb_savefolder fields for every instance, not from
    # this dict (shape_backend one of "spherical_harmonics" (default) or
    # "swept_spline").
    membrane: list[dict[str, Any]] = field(default_factory=list)
    # Each {"pdb_source": <code or path>, "frequency": 1, "parameterization":
    # "shtyrov"}. In TOML, provide as [[membrane_transmembrane_specs]] tables.
    # Applies across ALL membrane instances (not per-instance in v1). Only
    # meaningful when `membrane` is set (no bilayer to embed into otherwise).
    membrane_transmembrane_specs: list[dict[str, Any]] = field(default_factory=list)
    membrane_region_density_threshold: float | None = None
    membrane_region_max_passes: int = 300
    membrane_min_transmembrane_spacing_a: float = 40.0
    # Atomic scattering-factor parameterization for the targets/filler
    # protein-fill step (MembraneTomogramGenerator's own `parameterization`,
    # distinct from each MembraneGenerator instance's own "parameterization"
    # key inside its [[membrane]] dict, if set there -- that one's for the
    # bilayer/transmembrane step specifically). A top-level field rather
    # than read from membrane[0] since it applies regardless of whether
    # membrane is even set.
    parameterization: str = "shtyrov"

    # --- Filaments (optional, additive on top of membranes if present) ---
    # One dict per filament species, mapping straight onto
    # specter.specimen.filament.FilamentSpec's own kwargs, e.g.
    # {"code": "1TUB", "step": 85.0, "flex_deg": 3.0, "n_filaments": 4}.
    # Placed via specter.specimen.filament.place_filaments -- specter-native
    # random-walk placement, with no region-gating and no collision
    # avoidance against the membrane shell or each other, but DOES get
    # avoided by targets/filler packing (placed right after membranes,
    # before protein fill -- see MembraneTomogramGenerator's own
    # docstring). In TOML, provide as [[filaments]] tables.
    filaments: list[dict[str, Any]] = field(default_factory=list)
    # Convenience toggle: also place the bundled ACTIN_SPEC preset (real
    # F-actin helical repeat -- step/twist from Holmes/Egelman) without
    # hand-writing a [[filaments]] entry. Additive to filaments above (both
    # may be set at once). ACTIN_SPEC's own n_filaments default (1) applies
    # here too -- for more instances or other filament species (e.g.
    # microtubules, MICROTUBULE_SPEC), use [[filaments]] instead.
    actin: bool = False

    # --- Carbon support film (optional, single film) ---
    # Zero or one [[grid]] table, mapping onto
    # specter.specimen.GridSpec's own kwargs (thickness, hole_radius,
    # edge_fraction, edge_side, edge_roughness, edge_grain_size) -- e.g.
    # {"hole_radius": 6000.0, "edge_fraction": [0.02, 0.05]}. Painted
    # directly into the volume before anything else is placed; placement
    # (membranes/targets/filler) is NOT carbon-aware (a documented,
    # CTS-parity limitation -- see MembraneTomogramGenerator's own
    # docstring). More than one entry raises. Empty (default): no carbon
    # film, pure ice.
    grid: list[dict[str, Any]] = field(default_factory=list)

    # --- Gold fiducial beads (optional) ---
    # One dict per bead population, {"radius": <Angstrom>, "count": 1}.
    # "radius" is required. Placed via the same RSA packing used for
    # membranes/targets/filler, avoiding the membrane shell and any
    # already-placed filaments -- NOT region-gated to cytosol/lumen (see
    # specter.specimen.TomogramBeadSpec's own docstring). In TOML, provide
    # as [[beads]] tables.
    beads: list[dict[str, Any]] = field(default_factory=list)

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
    device: str = "cpu"
    # Device for the shared canvas tensors (volume/instance_labels/
    # membrane_labels), decoupled from `device` above (which stays the
    # compute device for rendering/rotation/field-generation regardless).
    # None (default): same as `device` -- one device for everything, the
    # original behaviour. "auto": estimate the canvas' own memory
    # footprint from target_shape/v_size and fall back to "cpu" if it
    # would exceed half of `device`'s currently free memory (see
    # MembraneTomogramGenerator.recommend_accumulator_device's own
    # docstring). Explicit "cpu" always works regardless of that
    # estimate, keeping `device`="cuda" (fast rendering/rotation) while
    # letting the canvas itself be sized by system RAM instead of GPU
    # VRAM -- e.g. a large field of view at fine v_size can need tens of
    # GB, past any single GPU's VRAM but often fine in system RAM on a
    # workstation/cluster node. Rotation/rendering speed is unaffected
    # either way; the only added cost is moving each already-small
    # rotated chunk across devices once, not the canvas itself.
    accumulator_device: str | None = None
    # How many PDB species render/fetch concurrently within a single
    # tomogram: membrane_transmembrane_specs (rendered once, shared across
    # every [[membrane]] entry/n_instances copy, membrane mode only) and
    # targets/filler (cytosol/lumen protein-fill, rendered + PDB-fetched
    # once per tomogram, always) -- see MembraneGenerator/
    # MembraneTomogramGenerator's own render_workers docstrings. Default 1:
    # fully serial, identical to the original behaviour. "auto" resolves
    # per-pool via specter.specimen._parallel_render.
    # recommend_render_workers -- min(n_species, 8), the measured sweet spot
    # from a full production-scale sweep (see that function's own
    # docstring); recommended over hand-picking a number.
    render_workers: int | Literal["auto"] = 1
    # Optional device pool to round-robin those concurrent species across
    # (e.g. ["cuda:0", "cuda:1"] on a multi-GPU machine). None (default):
    # every species renders on `device` above, still concurrently across
    # render_workers threads, just not spread across multiple physical
    # devices. "auto": every visible CUDA GPU (or None/CPU-fallback if
    # there aren't any) -- device choice was measured to barely matter at
    # the recommended worker count, so this doesn't try to be clever about
    # which subset to use. TOML-only (list[str] | "auto").
    render_devices: list[str] | Literal["auto"] | None = None
    # Instances rotated per GPU batch, per species, in the targets/filler
    # protein-fill stage (MembraneTomogramGenerator's own chunk_size --
    # rotate_volume batches ALL of a species' accepted instances into one
    # call when this is None, the original behaviour). Fine at small
    # scale, but a species with hundreds of instances (a real filler
    # species count at production target_shape/occupancy_fraction) can
    # then need many GB for one rotation call alone -- confirmed directly:
    # an 8.7 GB single allocation from one such batch, on top of whatever
    # else was already resident, was what actually tipped a production-
    # scale run into a CUDA OOM. None (default) preserves the original,
    # small-scale-safe behaviour; set e.g. 32-64 once a config's species
    # counts get into the hundreds.
    chunk_size: int | None = None

    # --- Output ---
    output_dir: str = "./output/"
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
    "v_size": "Voxel size in Angstrom.",
    "filler_occupancy_fraction": "Target packing density for filler "
    "species, as a bare-sphere fraction of EACH REGION's own volume it's "
    "placed in (the whole box when [[membrane]] is empty -- 'cytosol' is "
    "then the whole box). Deliberately high by default -- RSA self-limits "
    "at its own physical jamming ceiling rather than erroring, so filler "
    "simply packs until it jams rather than needing this hand-tuned. Lower "
    "it for a sparser filler layer, or if a small region (e.g. a tight "
    "vesicle lumen) makes the implied candidate pool impractically large.",
    "gap_angstrom": "Minimum clearance between placed spheres' surfaces, in Angstrom.",
    "clip_axes": "(z, y, x) -- True on an axis lets a placed instance's "
    "body extend past that wall (truncated at render time) instead of "
    "being rejected outright. TOML-only (list[bool]).",
    "pdb_savefolder": "Folder to cache downloaded PDB files.",
    "seed": "Random seed.",
    "membrane": "One or more MembraneGenerator kwargs dicts (TOML-only, "
    "[[membrane]] tables, one per composited TEMPLATE) -- optional, empty "
    "by default (no membrane at all; the whole tomogram is then one "
    "cytosol region). e.g. {'shape_backend': 'spherical_harmonics', "
    "'n_instances': 3}. See MembraneGenerator's own docstring for the full "
    "per-backend parameter set; plus 'n_instances' (int, default 1, "
    "expands one entry into that many independently-seeded instances), "
    "'position_xyz' (physical Angstrom offset from the tomogram center, "
    "default omitted = collision-rejecting random placement), and "
    "'target_shape_zyx' (default omitted = auto-sized per instance).",
    "membrane_transmembrane_specs": "Transmembrane protein species (TOML-"
    "only, [[membrane_transmembrane_specs]] tables), each {'pdb_source': "
    "<code or path>, 'frequency': 1, 'parameterization': 'shtyrov'}. Only "
    "meaningful when [[membrane]] is set, applies across all instances.",
    "membrane_region_density_threshold": "Passed through to "
    "MembraneTomogramGenerator's own region_density_threshold.",
    "membrane_region_max_passes": "Passed through to "
    "MembraneTomogramGenerator's own region_max_passes.",
    "membrane_min_transmembrane_spacing_a": "Minimum center-to-center "
    "spacing between placed transmembrane proteins, Angstrom. Only "
    "meaningful when [[membrane]] is set.",
    "parameterization": "Atomic scattering-factor parameterization for "
    "the targets/filler protein-fill step.",
    "filaments": "Filament species to scatter through the tomogram (TOML-"
    "only, [[filaments]] tables), each mapping onto "
    "specter.specimen.filament.FilamentSpec kwargs, e.g. {'code': '1TUB', "
    "'step': 85.0, 'flex_deg': 3.0, 'n_filaments': 4}. Placed right after "
    "membranes, before targets/filler -- no region-gating, no collision "
    "avoidance against the membrane shell or each other, but targets/"
    "filler DO avoid already-placed filaments.",
    "actin": "Convenience toggle: also place the bundled ACTIN_SPEC preset "
    "(real F-actin helical repeat) without writing a [[filaments]] entry. "
    "Additive to filaments above. For more instances or other filament "
    "species (e.g. microtubules), use [[filaments]] instead.",
    "grid": "Zero or one [[grid]] table (TOML-only) describing a carbon "
    "support film, mapping onto specter.specimen.GridSpec kwargs "
    "(thickness, hole_radius, edge_fraction, edge_side, edge_roughness, "
    "edge_grain_size). Painted into the volume before anything else is "
    "placed; not carbon-aware for placement (see MembraneTomogramGenerator's "
    "own docstring). Empty (default): no carbon film.",
    "beads": "Gold fiducial bead populations to pack (TOML-only, [[beads]] "
    "tables), each {'radius': <Angstrom>, 'count': 1}. Placed via the same "
    "RSA packing as membranes/targets/filler, avoiding the membrane shell "
    "and already-placed filaments -- not region-gated to cytosol/lumen.",
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
    "device": "cpu | cuda | cuda:0. Drives the whole "
    "MembraneGenerator/MembraneTomogramGenerator pipeline (shape field, "
    "bilayer profile, rasterization, transmembrane/targets/filler "
    "PotentialBuilder rendering) -- packing itself always runs on CPU "
    "regardless (vesin's neighbor list is both slower and OOM-prone on "
    "GPU at realistic particle counts).",
    "accumulator_device": "Device for the shared canvas tensors "
    "(volume/instance_labels/membrane_labels), decoupled from 'device' "
    "above (which stays the compute device regardless). None (default): "
    "same as 'device'. 'auto': estimate the canvas' own memory footprint "
    "and fall back to 'cpu' if it would exceed half of 'device''s "
    "currently free memory. Explicit 'cpu' always keeps 'device'='cuda' "
    "for fast rendering/rotation while letting the canvas itself be sized "
    "by system RAM instead of GPU VRAM -- useful for a large field of "
    "view at fine v_size whose canvas alone would exceed any single "
    "GPU's VRAM. Rotation/rendering speed is unaffected either way; only "
    "each already-small rotated chunk crosses devices, never the canvas "
    "itself.",
    "render_workers": "Number of PDB species rendered concurrently within "
    "one tomogram (membrane_transmembrane_specs and targets/filler each "
    "get their own concurrent build pass). Default 1 (serial, original "
    "behaviour) -- raise for tomograms with several species, especially "
    "with n_instances>1 [[membrane]] entries (all instances of one entry "
    "share one render pass). Set to 'auto' (TOML/Python config only -- the "
    "--render_workers CLI flag stays integer-only) to pick min(n_species, "
    "8) per pool automatically, the measured sweet spot from a full "
    "production-scale sweep -- see "
    "specter.specimen._parallel_render.recommend_render_workers.",
    "render_devices": "TOML-only (list[str] | 'auto'): device pool to "
    "round-robin those concurrent species across, e.g. ['cuda:0', "
    "'cuda:1']. None (default) keeps every species on `device` above, "
    "still concurrent across render_workers threads. 'auto' uses every "
    "visible CUDA GPU (falls back to `device` if none) -- device choice "
    "was measured to barely matter at the recommended worker count.",
    "chunk_size": "Instances rotated per GPU batch, per species, when "
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
    config = config_cls(**flat)
    # Not every config dataclass has pdb_savefolder (e.g. TiltSeriesConfig,
    # the imaging-only half of the cryo-ET pipeline, has no PDB fetching of
    # its own) -- resolve it against REPO_ROOT only when present.
    if hasattr(config, "pdb_savefolder") and not os.path.isabs(config.pdb_savefolder):
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
