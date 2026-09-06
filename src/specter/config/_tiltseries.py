"""TiltSeriesConfig: parameters for cryoET tilt-series generation."""

from __future__ import annotations

from dataclasses import dataclass

from ._field import help_of, setting
from typing import Literal
from specter.options import IceModel, NoiseModel, ScatteringFactors, TiltAxis


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
    volume_path: str = setting(
        "",
        help=(
            "Path to a pre-built (Z, Y, X) scattering-potential volume "
            "(.mrc/.mrcs/.pt), already in scattering-potential units -- e.g. `specter "
            "build tomogram`'s own output. Required."
        ),
    )

    # --- Microscope / physics ---
    # Å/voxel of the loaded volume -- must match whatever produced it (e.g.
    # `TomogramConfig.voxel_size`), not auto-detected from the file itself.
    voxel_size: float = setting(
        5.0,
        help=(
            "Voxel size of the loaded volume, in Angstrom -- must "
            "match whatever produced it."
        ),
        check="positive",
    )
    micrograph_size: int | None = setting(
        None,
        help=(
            "Output tilt-image size in pixels (square). Defaults "
            "to the XY dimension of the specimen volume."
        ),
        check="positive",
    )
    voltage: float = setting(
        300.0, help="Electron beam accelerating voltage in kV.", check="positive"
    )  # kV
    dose_per_tilt: float = setting(
        3.0,
        help="Total dose for each tilt image in e-/Angstrom^2.",
        check="non_negative",
    )  # e⁻/Å², per tilt angle
    n_frames: int = setting(
        10,
        help=(
            "Number of movie frames per tilt. Only affects the image "
            "when coincidence_radius > 0, which is what splits the dose into "
            "frames; ignored otherwise."
        ),
        check="positive",
    )  # movie frames per tilt
    cs: float = setting(
        2.0, help="Spherical aberration in mm (1-3 mm typical).", check="non_negative"
    )  # mm
    alpha: float = setting(
        0.1, help="Amplitude contrast ratio.", range=(0.0, 1.0)
    )  # unitless, amplitude contrast ratio

    # --- Envelopes ---
    convergence_angle: float | None = setting(
        None,
        help=(
            "Beam convergence semi-angle in mrad, for the Cs "
            "(spatial coherence) envelope."
        ),
        check="non_negative",
    )  # mrad
    cc: float | None = setting(
        None,
        help=(
            "Chromatic aberration coefficient in mm, for the Cc (temporal "
            "coherence) envelope."
        ),
        check="non_negative",
    )  # mm
    energy_spread: float = setting(
        0.7,
        help="FWHM of the beam energy spread in eV, used by the Cc envelope.",
        check="non_negative",
    )  # eV (FWHM)
    deltaV_V: float = setting(
        0.06e-6,
        help="Relative high-voltage instability, used by the Cc envelope.",
        check="non_negative",
    )  # unitless (ΔV/V)
    deltaI_I: float = setting(
        0.01e-6,
        help="Relative objective-lens current instability, used by the Cc envelope.",
        check="non_negative",
    )  # unitless (ΔI/I)
    dose_envelope: bool = setting(
        False, help="Apply the Grant & Grigorieff (2015) cumulative-dose envelope."
    )

    # --- Defocus ---
    defocus: float = setting(
        22000.0,
        help="Defocus in Angstrom (positive = underfocus).",
        check="non_negative_ordered",
    )  # Å (positive = underfocus)

    # --- Tilt geometry ---
    min_tilt_angle: float = setting(
        -45.0, help="Minimum tilt angle in degrees."
    )  # degrees
    max_tilt_angle: float = setting(
        45.0, help="Maximum tilt angle in degrees."
    )  # degrees
    n_tilts: int = setting(
        61,
        help="Number of tilt angles (evenly spaced from min to max).",
        check="positive",
    )
    tilt_axis: TiltAxis = setting("y", help="Tilt axis.")

    # --- Models ---
    scattering_model: Literal["multislice", "firstborn", "projection", "ctf"] = setting(
        "multislice", help="Scattering model."
    )
    noise_model: NoiseModel = setting(
        "poisson", help="Noise model. Use 'none' for no noise."
    )
    coincidence_radius: float = setting(
        0.0,
        help=(
            "Effective coincidence exclusion radius in pixels "
            "(exclusion area = pi*r^2) for direct-detector modelling."
        ),
        check="non_negative_ordered",
    )  # pixels; 0 = plain Poisson
    ice_model: IceModel = setting(
        "gd",
        help=(
            "Ice generation algorithm: 'gd' (IceBank cache), 'random' "
            "(cheap RandomIcemaker), or 'none'."
        ),
    )
    ice_cache_dir: str | None = setting(
        None,
        help=(
            "Directory of cached ice configs for ice_model='gd'. "
            "Defaults to the bundled ice_data/ice_cache."
        ),
    )  # defaults to the bundled ice_data/ice_cache
    ice_relax_steps: int = setting(
        0,
        help=(
            "Local MLBOP relaxation steps used to heal ice tile "
            "seams (ice_model='gd' only)."
        ),
        check="non_negative",
    )  # local MLBOP seam-relaxation steps for ice_model="gd"
    # Everything specter renders that is NOT a biomolecule: the ice.
    # Kept separate from `scattering_factors` on purpose -- Shtyrov fits bonded
    # species of biomolecules over 0.011-0.62 1/A, so bulk materials are out of
    # its domain and a mean inner potential (a k=0 quantity) extrapolates below
    # the fitted range. Kirkland/Lobato/Peng are per-element, valid at k=0, and
    # agree there. See ice._kernels.build_water_kernel for the measurements.
    bulk_scattering_factors: ScatteringFactors = setting(
        "kirkland",
        help=(
            "Atomic scattering-factor parameterization for the ice -- everything rendered that is not a biomolecule. Deliberately separate from scattering_factors: Shtyrov is fitted for biomolecules, and these materials are outside that domain."
        ),
    )
    # Fraction of Nyquist, not 1/A: Scattering masks k <= klim * k_nyquist.
    # Kirkland recommends 0.66 (2/3) to prevent multislice FFT aliasing, but
    # that costs real spatial resolution, so the default keeps the full range
    # and accepts the aliasing. Exposed so a caller can make the other choice.
    klim: float | None = setting(
        None,
        help=(
            "Bandlimit for Kirkland's FFT anti-aliasing, as a fraction of "
            "Nyquist. Kirkland recommends 0.66 (2/3), which prevents aliasing but "
            "discards real spatial frequency content above it. Unset (the default) "
            "keeps the full Nyquist range and accepts the aliasing."
        ),
        check="non_negative",
    )  # fraction of Nyquist
    bfactor: float | None = setting(
        None, help="Isotropic B-factor envelope in Angstrom^2.", check="non_negative"
    )  # A^2
    pad_fft: bool = setting(
        False,
        help=(
            "Pad volume for FFT to avoid multislice edge-wraparound "
            "artifacts under tilt."
        ),
    )
    detector_model: Literal["none", "perfect", "k3_300kv", "k3_200kv", "k2_300kv"] = (
        setting("none", help="Detector model.")
    )

    # --- Post-processing ---
    normalize_tilt_series: bool = setting(
        False, help="Normalize each tilt image to zero mean and unit std."
    )
    save_exitwaves: bool = setting(
        False, help="Save exit wave magnitude and phase as separate .mrcs files."
    )

    # --- Compute ---
    device: str = setting(
        "cuda", help="Device to use: cpu | cuda | cuda:0."
    )  # falls back to CPU when none is available

    # --- Reproducibility ---
    seed: int | None = setting(
        None,
        help=(
            "RNG seed for ice, crowding, pose and noise sampling. Auto-generated and logged if unset."
        ),
    )

    # --- Output ---
    # One path field, not one per layout: this is the single directory a run
    # writes under, read as the leaf when untracked and as the root of the
    # numbered job tree when tracked. `None` rather than a baked-in default
    # because which default applies is not knowable until tracking is -- see
    # pipelines._common.resolve_output_dir.
    output_dir: str | None = setting(
        None,
        help=(
            "Directory to save output files when untracked. Setting --project or --job_id instead makes this the root of the numbered job tree, so tracking organises output within the folder you chose rather than moving it elsewhere. Unset defaults to <artifact>/ untracked, and to the project root found by walking up from cwd for an existing .specter marker when tracked."
        ),
    )
    filename: str = setting(
        "tilt_series", help="Base name for output files (no extension)."
    )

    # --- Job tracking (opt-in) ---
    # Setting `project` or `job_id` routes output through `specter.jobs`
    # instead of the flat output_dir/filename layout above: the directory
    # becomes output_dir/[project/]tiltseries/J00N/, numbered and with a
    # job.json recording the full parameter set, git commit and status.
    # Neither is required -- leaving both unset keeps today's exact flat
    # behavior.
    project: str | None = setting(
        None,
        help=(
            "Optional: number and track this run through specter.jobs. "
            "Not required for tracking -- job_id alone also triggers it. The run "
            "lands in "
            "<output_dir>/[<project>/]tiltseries/J00N/ with a job.json "
            "recording every parameter, the git commit and the run's status."
        ),
    )
    job_id: str | None = setting(
        None,
        help=(
            "Pin the job directory (e.g. J001) rather than auto-assigning "
            "the next one: resumes into it if it exists, creates it otherwise."
        ),
    )


TILT_SERIES_HELP: dict[str, str] = help_of(TiltSeriesConfig)
