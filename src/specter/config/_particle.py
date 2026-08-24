"""ParticleStackConfig: parameters for particle-stack generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ._paths import default_output_dir, default_pdb_cache_dir
from ._scalar_range import ScalarOrRange


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

    # --- Job tracking (opt-in) ---
    # Setting `project` or `job_id` routes output through `specter.jobs`
    # instead of the flat output_dir/filename layout above: the directory
    # becomes job_base_dir/[project/]particles/J00N/, numbered and with a
    # job.json recording the full parameter set, git commit and status.
    # Neither is required -- leaving both unset keeps today's exact flat
    # behavior. Unlike `specter reconstruct particle`, which is always
    # tracked, this command runs far more often and more casually (quick
    # sanity checks, notebooks, CI), so tracking stays opt-in here.
    project: str | None = None
    job_id: str | None = None
    # Defaults to the project root found by walking up from cwd for an
    # existing specter-data/ (find_specter_project_root), the same way git
    # finds the nearest .git.
    job_base_dir: str | None = None

    # --- Advanced ---
    # Relative to the current working directory, like any other CLI path
    # argument -- see default_pdb_cache_dir for the unset case.
    pdb_cache_dir: str = field(default_factory=default_pdb_cache_dir)
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
    # bool | Literal["auto"] so TOML can write the natural `true`/`false` as
    # well as "auto"; the CLI flag flattens to bool (same as batchsize's
    # int | Literal["auto"]), so "auto" is config-only there -- it is the
    # default, so a flag for it would only ever undo an explicit setting.
    readd_hydrogens: bool | Literal["auto"] = "auto"

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
    "project": "Optional: route output through specter.jobs instead of "
    "--output_dir/--filename. Not required for tracking -- job_id alone "
    "also triggers it. The run lands in "
    "<job_base_dir>/[<project>/]particles/J00N/ with a job.json recording "
    "every parameter, the git commit and the run's status.",
    "job_id": "Pin the job directory (e.g. J001) rather than auto-assigning "
    "the next one: resumes into it if it exists, creates it otherwise. "
    "Required (not just recommended) when combining tracking with "
    "multi-GPU device strings -- auto-numbering needs one process to "
    "decide, but multi-GPU dispatch re-runs this pipeline once per rank.",
    "job_base_dir": "Root directory for job folders. Defaults to the "
    "project root found by walking up from cwd looking for an existing "
    "specter-data/, the same way git finds the nearest .git.",
    "pdb_cache_dir": "Folder to cache downloaded PDB files.",
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
    "readd_hydrogens": "Whether to replace a structure's own hydrogens with "
    "the monomer library's ideal geometry: 'auto' (default) keeps hydrogens "
    "the file already carries and adds them only when it has none, 'true' "
    "always re-adds, 'false' never adds hydrogen density (they still inform "
    "atom typing). Only meaningful when a monomer library is available.",
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
