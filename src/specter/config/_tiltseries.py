"""TiltSeriesConfig: parameters for cryoET tilt-series generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
    noise_model: Literal["poisson", "none"] = "poisson"
    coincidence_radius: float = 0.0  # pixels; 0 = plain Poisson
    ice_model: Literal["gd", "random", "none"] = "gd"
    ice_cache_dir: str | None = None  # defaults to the bundled ice_data/ice_cache
    ice_relax_steps: int = 0  # local MLBOP seam-relaxation steps for ice_model="gd"
    # Fraction of Nyquist, not 1/A: Scattering masks k <= klim * k_nyquist.
    # Kirkland recommends 0.66 (2/3) to prevent multislice FFT aliasing, but
    # that costs real spatial resolution, so the default keeps the full range
    # and accepts the aliasing. Exposed so a caller can make the other choice.
    klim: float | None = None  # fraction of Nyquist
    bfactor: float | None = None  # A^2
    pad_fft: bool = False
    detector_model: Literal["none", "perfect", "k3_300kv", "k3_200kv"] = "none"

    # --- Post-processing ---
    normalize_tilt_series: bool = False
    save_exitwaves: bool = False

    # --- Compute ---
    device: str = "cuda"  # falls back to CPU when none is available

    # --- Reproducibility ---
    seed: int | None = None

    # --- Output ---
    # One path field, not one per layout: this is the single directory a run
    # writes under, read as the leaf when untracked and as the root of the
    # numbered job tree when tracked. `None` rather than a baked-in default
    # because which default applies is not knowable until tracking is -- see
    # pipelines._common.resolve_output_dir.
    output_dir: str | None = None
    filename: str = "tilt_series"

    # --- Job tracking (opt-in) ---
    # Setting `project` or `job_id` routes output through `specter.jobs`
    # instead of the flat output_dir/filename layout above: the directory
    # becomes output_dir/[project/]tiltseries/J00N/, numbered and with a
    # job.json recording the full parameter set, git commit and status.
    # Neither is required -- leaving both unset keeps today's exact flat
    # behavior.
    project: str | None = None
    job_id: str | None = None


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
    "noise_model": "Noise model. Use 'none' for no noise.",
    "coincidence_radius": "Effective coincidence exclusion radius in pixels "
    "(exclusion area = pi*r^2) for direct-detector modelling.",
    "ice_model": "Ice generation algorithm: 'gd' (IceBank cache), 'random' "
    "(cheap RandomIcemaker), or 'none'.",
    "ice_cache_dir": "Directory of cached ice configs for ice_model='gd'. "
    "Defaults to the bundled ice_data/ice_cache.",
    "ice_relax_steps": "Local MLBOP relaxation steps used to heal ice tile "
    "seams (ice_model='gd' only).",
    "klim": "Bandlimit for Kirkland's FFT anti-aliasing, as a fraction of "
    "Nyquist. Kirkland recommends 0.66 (2/3), which prevents aliasing but "
    "discards real spatial frequency content above it. Unset (the default) "
    "keeps the full Nyquist range and accepts the aliasing.",
    "bfactor": "Isotropic B-factor envelope in Angstrom^2.",
    "pad_fft": "Pad volume for FFT to avoid multislice edge-wraparound "
    "artifacts under tilt.",
    "detector_model": "Detector model.",
    "normalize_tilt_series": "Normalize each tilt image to zero mean and unit std.",
    "save_exitwaves": "Save exit wave magnitude and phase as separate .mrcs files.",
    "device": "Device to use: cpu | cuda | cuda:0.",
    "seed": "RNG seed for ice, crowding, pose and noise sampling. Auto-generated and logged if unset.",
    "output_dir": "Directory to save output files when untracked. Setting --project or --job_id instead makes this the root of the numbered job tree, so tracking organises output within the folder you chose rather than moving it elsewhere. Unset defaults to <artifact>/ untracked, and to the project root found by walking up from cwd for an existing .specter marker when tracked.",
    "filename": "Base name for output files (no extension).",
    "project": "Optional: number and track this run through specter.jobs. "
    "Not required for tracking -- job_id alone also triggers it. The run "
    "lands in "
    "<output_dir>/[<project>/]tiltseries/J00N/ with a job.json "
    "recording every parameter, the git commit and the run's status.",
    "job_id": "Pin the job directory (e.g. J001) rather than auto-assigning "
    "the next one: resumes into it if it exists, creates it otherwise.",
}
