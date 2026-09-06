"""ParticleStackConfig: parameters for particle-stack generation."""

from __future__ import annotations

from dataclasses import dataclass

from ._field import help_of, setting
from typing import Literal

from ._paths import default_pdb_cache_dir
from ._scalar_range import ScalarOrRange
from specter.options import (
    ConvBackend,
    EwaldSphereSign,
    IceModel,
    NoiseModel,
    PotentialMethod,
    RotateMode,
    ScatteringFactors,
)


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
    pdb_source: str = setting(
        help=(
            "Path to a local .cif/.pdb file, or a 4-character PDB "
            "accession code to fetch and cache. A local file is read where it "
            "lies and never copied into the cache; an existing file wins over a "
            "same-named accession code."
        )
    )
    assembly: bool = setting(True, help="Fetch the biological assembly.")
    n_pixels: int = setting(
        256,
        help="Number of pixels per axis for the 3-D potential box.",
        check="positive",
    )
    pixel_size: float = setting(
        1.0, help="Pixel size in Angstrom.", check="positive"
    )  # Å

    # --- Microscope (basic) ---
    voltage: float = setting(
        300.0, help="Electron beam accelerating voltage in kV.", check="positive"
    )  # kV
    dose: ScalarOrRange = setting(
        20.0,
        help=(
            "Total dose per particle in e-/Angstrom^2: a single value (e.g. 20) for a "
            "constant dose per particle, or 'low,high' (e.g. 20,60) to sample uniformly per particle "
            "(in a TOML config, write the range as [20, 60])."
        ),
        check="positive_ordered",
    )  # e⁻/Å²
    cs: float = setting(
        2.0, help="Spherical aberration in mm (1-3 mm typical).", check="non_negative"
    )  # mm
    alpha: float = setting(
        0.1, help="Amplitude contrast ratio.", range=(0.0, 1.0)
    )  # unitless, amplitude contrast ratio

    # --- Sampling (basic) ---
    defocus: ScalarOrRange = setting(
        factory=lambda: [5000.0, 15000.0],
        help=(
            "Defocus in Angstrom: a single value (e.g. 8000) for constant "
            "defocus, or 'low,high' (e.g. 5000,15000) to sample uniformly per particle "
            "(in a TOML config, write the range as [5000, 15000])."
        ),
        check="non_negative_ordered",
    )  # Å
    shift: float = setting(
        2.0,
        help="Max in-plane shift in Angstrom (uniform +/-shift).",
        check="non_negative",
    )  # Å, max in-plane shift (uniform ±shift)
    n_particles: int = setting(
        20, help="Number of particles to simulate.", check="positive"
    )

    # --- Models (basic) ---
    scattering_model: Literal["multislice", "firstborn", "projection", "ctf"] = setting(
        "multislice", help="Scattering model."
    )
    detector_model: Literal[
        "none",
        "perfect",
        "k3_300kv",
        "k3_200kv",
        "k2_300kv",
        "falcon4i_300kv",
        "falcon4i_200kv",
    ] = setting("none", help="Detector model.")

    # --- Post-processing (basic) ---
    normalize_particles: bool = setting(
        True, help="Normalize particles to zero mean and unit std."
    )
    save_exitwaves: bool = setting(
        False, help="Save exit wave magnitude and phase as separate .mrcs files."
    )
    save_clean_exitwaves: bool = setting(
        False,
        help="Save clean (particle-only, no ice) exit wave magnitude and phase.",
    )

    # --- Compute (basic) ---
    device: str = setting(
        "cuda",
        help=(
            "Device to use: cpu | cuda | cuda:0 | 0,1,2,3. "
            "Comma-separated integers trigger multi-GPU Lightning DDP."
        ),
    )  # falls back to CPU when none is available
    # "auto" (the default) sizes the batch to the memory free on `device` at
    # run time, from the box geometry -- see specter.memory. An int pins it,
    # which is what a benchmark or a shared-GPU run wants; nothing about the
    # physics or the output depends on this, only speed and peak memory.
    batchsize: int | Literal["auto"] = setting(
        "auto",
        help=(
            "Number of particles per forward pass. Unset (or 'auto' in "
            "a TOML config, which is the default) takes the smaller of what fits in "
            "the memory free on --device at run time and what is worth batching -- "
            "past the point where one forward pass already saturates the device, a "
            "bigger batch costs memory without going faster. See "
            "specter.memory.recommend_batchsize."
        ),
        check="positive",
    )

    # --- Output (basic) ---
    # One path field, not one per layout: this is the single directory a run
    # writes under, read as the leaf when untracked and as the root of the
    # numbered job tree when tracked. `None` rather than a baked-in default
    # because which default applies is not knowable until tracking is -- see
    # pipelines._common.resolve_output_dir.
    output_dir: str | None = setting(
        None,
        help=(
            "Directory to save .mrcs and .star files when untracked. Setting --project or --job_id instead makes this the root of the numbered job tree, so tracking organises output within the folder you chose rather than moving it elsewhere. Unset defaults to <artifact>/ untracked, and to the project root found by walking up from cwd for an existing .specter marker when tracked."
        ),
    )
    filename: str = setting(
        "particles", help="Base name for output files (no extension)."
    )

    # --- Job tracking (opt-in) ---
    # Setting `project` or `job_id` routes output through `specter.jobs`
    # instead of the flat output_dir/filename layout above: the directory
    # becomes output_dir/[project/]particles/J00N/, numbered and with a
    # job.json recording the full parameter set, git commit and status.
    # Neither is required -- leaving both unset keeps today's exact flat
    # behavior. Unlike `specter reconstruct particle`, which is always
    # tracked, this command runs far more often and more casually (quick
    # sanity checks, notebooks, CI), so tracking stays opt-in here.
    project: str | None = setting(
        None,
        help=(
            "Optional: number and track this run through specter.jobs. "
            "Not required for tracking -- job_id alone also triggers it. The run "
            "lands in "
            "<output_dir>/[<project>/]particles/J00N/ with a job.json recording "
            "every parameter, the git commit and the run's status."
        ),
    )
    job_id: str | None = setting(
        None,
        help=(
            "Pin the job directory (e.g. J001) rather than auto-assigning "
            "the next one: resumes into it if it exists, creates it otherwise. "
            "Mandatory when combining tracking with "
            "multi-GPU device strings -- auto-numbering needs one process to "
            "decide, but multi-GPU dispatch re-runs this pipeline once per rank."
        ),
    )

    # --- Advanced ---
    # Relative to the current working directory, like any other CLI path
    # argument -- see default_pdb_cache_dir for the unset case.
    pdb_cache_dir: str = setting(
        factory=default_pdb_cache_dir,
        help=(
            "Where downloaded PDB/mmCIF structures are cached. An "
            "input cache shared by every run, not an output location -- job tracking "
            "does not redirect it."
        ),
    )
    # if set, poses/CTF/pixel_size/voltage/alpha come from here (pick one)
    cs_path: str | None = setting(
        None,
        help=(
            "Path to a CryoSPARC .cs file to drive generation from real "
            "pixel_size/voltage/alpha/poses/CTF instead of random sampling."
        ),
        check="existing_file",
    )
    star_path: str | None = setting(
        None,
        help=(
            "Path to a RELION .star file to drive generation from real "
            "pixel_size/voltage/alpha/poses/CTF instead of random sampling. Mutually "
            "exclusive with --cs_path."
        ),
        check="existing_file",
    )
    n_frames: int | None = setting(
        None,
        help=(
            "Number of movie frames. Defaults to int(dose) if not set. "
            "Only affects the image when coincidence_radius > 0, which is what "
            "splits the dose into frames; ignored otherwise."
        ),
        check="positive",
    )
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
    bfactor: float | None = setting(
        None, help="Isotropic B-factor envelope in Angstrom^2.", check="non_negative"
    )  # Å²
    noise_model: NoiseModel = setting(
        "poisson", help="Noise model. Use 'none' for no noise."
    )
    coincidence_radius: ScalarOrRange = setting(
        0.0,
        help=(
            "Effective coincidence exclusion radius in pixels "
            "(exclusion area = pi*r^2): a single value for constant radius, or "
            "'low,high' ([low, high] in TOML) to sample uniformly per particle."
        ),
        check="non_negative_ordered",
    )  # pixels
    ice_model: IceModel = setting(
        "gd",
        help=(
            "Ice model: 'gd' (samples the pre-generated IceBank "
            "cache), 'random' (cheap, low-realism), or 'none'."
        ),
    )
    ice_thickness: float = setting(
        0.0,
        help="Ice thickness in Angstrom. 0 = minimum (particle box size).",
        check="non_negative",
    )  # Å, 0 = minimum (particle box size)
    ice_cache_dir: str | None = setting(
        None,
        help=(
            "Directory of cached ice configs for ice_model='gd'. "
            "Defaults to the bundled ice_data/ice_cache."
        ),
    )  # defaults to the bundled ice_data/ice_cache
    crowd_min_distance: float | None = setting(
        None,
        help=(
            "Minimum distance between crowded particles in "
            "Angstrom. Unset disables crowding."
        ),
        check="non_negative",
    )  # Å
    crowd_max_distance_z: float | None = setting(
        None,
        help="Maximum z-distance between crowded particles in Angstrom.",
        check="non_negative",
    )  # Å
    potential_scale: ScalarOrRange = setting(
        1.0,
        help=(
            "Potential scale factor (unitless, values < 1 "
            "approximate thicker ice): a single value for constant scale, or "
            "'low,high' ([low, high] in TOML) to sample uniformly per particle."
        ),
        check="positive_ordered",
    )  # unitless
    pad_fft: bool = setting(
        True, help="Pad the volume for FFT to avoid edge artifacts."
    )

    # --- Advanced: potential building ---
    scattering_factors: ScatteringFactors = setting(
        "shtyrov",
        help=(
            "Atomic scattering-factor parameterization used to "
            "build the structure's scattering potential."
        ),
    )
    potential_method: PotentialMethod = setting(
        "analytic",
        help=(
            "Voxelization method for the structure's own "
            "potential: 'analytic' (per-atom closed-form, no splat/FFT), '2d' "
            "(soft XY, hard Z), or '3d' (trilinear). Ice is built by its own "
            "path and is unaffected by this."
        ),
    )
    rcut: float | None = setting(
        None,
        help=(
            "Cutoff radius in Angstrom for the atomic potential kernel. "
            "Auto-detected per-structure if unset."
        ),
        check="non_negative",
    )  # Å, auto-detected per-structure if unset
    conv_backend: ConvBackend = setting(
        "fftconvolve",
        help=(
            "Convolution backend for potential building. Unused for "
            "potential_method='analytic'."
        ),
    )
    periodic: bool = setting(
        False,
        help=(
            "Wrap atom density across the box faces when voxelizing. "
            "Keep false for a particle -- a protein is a finite object, so wrapping "
            "smears its edge density onto the opposite face. Requires "
            "potential_method='3d'; 'analytic' and '2d' raise."
        ),
    )
    # per-atom bonded species for Shtyrov typing; auto-detected from PDB bonds if
    # unset. Sized to the structure's atom count -- config-only, not a CLI flag.
    atom_species: list[str] | None = setting(None)
    shtyrov_params_path: str | None = setting(
        None,
        help="Override the bundled Shtyrov parameter table.",
        check="existing_file",
    )
    # bool | Literal["auto"] so TOML can write the natural `true`/`false` as
    # well as "auto"; the CLI flag flattens to bool (same as batchsize's
    # int | Literal["auto"]), so "auto" is config-only there -- it is the
    # default, so a flag for it would only ever undo an explicit setting.
    readd_hydrogens: bool | Literal["auto"] = setting(
        "auto",
        help=(
            "Whether to replace a structure's own hydrogens with "
            "the monomer library's ideal geometry: 'auto' (default) keeps hydrogens "
            "the file already carries and adds them only when it has none, 'true' "
            "always re-adds, 'false' never adds hydrogen density (they still inform "
            "atom typing). Only meaningful when a monomer library is available."
        ),
    )
    # Unset falls back to $CLIBD_MON, so the CCP4 variable still works and
    # nothing needs setting per config. This exists because the library
    # changes the RESULT (79% -> 100% typed on 1mbo; 20-30% relative RMS on
    # the rendered potential), and a run should be reproducible from its own
    # recorded config rather than from an environment nobody wrote down --
    # the same reason pdb_cache_dir is a field despite $SPECTER_PDB_CACHE.
    monomer_library_path: str | None = setting(
        None,
        help=(
            "Path to a Monomer Library "
            "(https://github.com/MonomerLibrary/monomers), which completes a "
            "structure's bond topology and hydrogens so Shtyrov species resolve. "
            "Unset falls back to $CLIBD_MON. Without one, around 44% of a "
            "hydrogen-free protein falls back to per-element Peng factors."
        ),
    )
    # Off by default: a deposited B-factor is refinement output, not a
    # measured mean-square displacement, and applying a structure's own
    # column silently makes the rendered specimen depend on who deposited
    # it. Requires scattering_factors="shtyrov" and
    # potential_method="analytic"; anything else raises.
    use_deposited_bfactors: bool = setting(
        False,
        help=(
            "Damp each atom by the B-factor its structure "
            "deposits, instead of rendering the model statically. Only a PER-ATOM "
            "B adds anything an envelope cannot: a uniform one is the same "
            "exp(-B k^2/4) as --bfactor, so setting both double-counts. A deposited "
            "column is refinement output rather than a measured displacement, and "
            "cryo-EM entries often carry a constant or zero one. Requires "
            "scattering_factors='shtyrov' and potential_method='analytic'."
        ),
    )

    # --- Advanced: scattering ---
    ews_curvature_sign: EwaldSphereSign = setting(
        "positive",
        help="Ewald sphere curvature sign, matching CryoSPARC's convention.",
    )
    # Fraction of Nyquist, not 1/A: Scattering masks k <= klim * k_nyquist.
    # Kirkland recommends 0.66 (2/3) to prevent multislice FFT aliasing, but
    # that costs real spatial resolution, so the default keeps the full range
    # and accepts the aliasing. Exposed so a caller can make the other choice.
    klim: float | None = setting(
        None,
        help=(
            "Bandlimit for Kirkland's FFT anti-aliasing, as a fraction of Nyquist. Kirkland recommends 0.66 (2/3), which prevents aliasing but discards real spatial frequency content above it. Unset (the default) keeps the full Nyquist range and accepts the aliasing."
        ),
        check="non_negative",
    )  # fraction of Nyquist
    rotate_mode: RotateMode = setting(
        "real",
        help=(
            "Volume rotation method: 'real' (trilinear interpolation) or "
            "'fourier' (no boundary artifacts)."
        ),
    )

    # --- Advanced: ice ---
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
    ice_relax_steps: int = setting(
        0,
        help=(
            "Local MLBOP seam-relaxation steps, only used when "
            "ice_model='gd' tiles multiple cached blocks."
        ),
        check="non_negative",
    )

    # --- Advanced: crowding ---
    crowd_chunk_size: int = setting(
        1,
        help=(
            "Crowding duplicate volumes rotated per batch. Lowering "
            "it to 1 costs no wall time: wall time is flat in this while peak memory "
            "grows linearly with it, so raising it above the default buys nothing."
        ),
        check="positive",
    )
    crowd_max_distance_xy: float | None = setting(
        None,
        help="Maximum xy-distance between crowded particles in Angstrom.",
        check="non_negative",
    )  # Å
    crowd_method: Literal["2d", "3d"] = setting(
        "3d",
        help="Poisson-disk sampling dimensionality for crowding particle placement.",
    )
    crowd_n_points: int | None = setting(
        None,
        help="Cap on the number of crowding duplicates. Unset means no cap.",
        check="positive",
    )
    crowd_seed: Literal["origin", "random"] = setting(
        "origin",
        help=(
            "Crowding placement seed strategy: 'origin' (first point at "
            "the structure's center) or 'random'."
        ),
    )
    crowd_move_to_cpu: bool = setting(
        False,
        help=(
            "Move crowding intermediates to CPU between steps, to "
            "trade speed for lower GPU memory."
        ),
    )
    water_air_interface: bool = setting(
        False,
        help=(
            "Model a water-air interface when placing ice/"
            "crowding (bimodal density along z instead of uniform)."
        ),
    )

    # --- Advanced: reproducibility ---
    seed: int | None = setting(
        None,
        help=(
            "RNG seed for pose/CTF/dose sampling. Auto-generated and logged if unset."
        ),
    )

    # --- Advanced: aberration richness for synthetic (non-.cs-driven) generation ---
    astigmatism: ScalarOrRange = setting(
        0.0,
        help=(
            "Astigmatism magnitude (dfu - dfv) in Angstrom: a single "
            "value for constant, or 'low,high' ([low, high] in TOML) to sample "
            "uniformly per particle."
        ),
        check="non_negative_ordered",
    )  # Å, magnitude of dfu - dfv
    astigmatism_angle: ScalarOrRange = setting(
        factory=lambda: [0.0, 180.0],
        help=(
            "Astigmatism angle in degrees: a single value or "
            "'low,high' ([low, high] in TOML) range. Irrelevant when astigmatism is 0."
        ),
        check="ordered",
    )  # degrees, dfang
    phaseshift: ScalarOrRange = setting(
        0.0,
        help=(
            "Phase shift in radians (e.g. from a Volta phase plate): a "
            "single value or 'low,high' ([low, high] in TOML) range."
        ),
        check="ordered",
    )  # radians
    tiltx: ScalarOrRange = setting(
        0.0,
        help=(
            "Beam tilt (x) in radians: a single value or 'low,high' "
            "([low, high] in TOML) range."
        ),
    )  # radians
    tilty: ScalarOrRange = setting(
        0.0,
        help=(
            "Beam tilt (y) in radians: a single value or 'low,high' "
            "([low, high] in TOML) range."
        ),
    )  # radians
    trefoil1: ScalarOrRange = setting(
        0.0,
        help=(
            "First trefoil (3-fold astigmatism) component in Angstrom^3: a "
            "single value or 'low,high' ([low, high] in TOML) range."
        ),
    )  # Å^3, coefficient of k^3 sin(3θ)
    trefoil2: ScalarOrRange = setting(
        0.0,
        help=(
            "Second trefoil component in Angstrom^3: a single value or "
            "'low,high' ([low, high] in TOML) range."
        ),
    )  # Å^3, coefficient of k^3 cos(3θ)
    # 4th-order, non-rotationally-symmetric terms -- 1/2 are secondary
    # astigmatism (n=4, m=±2), 3/4 are true 4-fold tetrafoil (n=4, m=±4);
    # see aberrations._functions.tetrafoil.
    tetrafoil1: ScalarOrRange = setting(
        0.0,
        help=(
            "Secondary astigmatism, k^4 cos(2*theta) coefficient in "
            "Angstrom^4: a single value or 'low,high' ([low, high] in TOML) range."
        ),
    )  # Å^4, coefficient of k^4 cos(2θ)
    tetrafoil2: ScalarOrRange = setting(
        0.0,
        help=(
            "Secondary astigmatism, k^4 sin(2*theta) coefficient in "
            "Angstrom^4: a single value or 'low,high' ([low, high] in TOML) range."
        ),
    )  # Å^4, coefficient of k^4 sin(2θ)
    tetrafoil3: ScalarOrRange = setting(
        0.0,
        help=(
            "Tetrafoil (4-fold astigmatism), k^4 cos(4*theta) coefficient "
            "in Angstrom^4: a single value or 'low,high' ([low, high] in TOML) range."
        ),
    )  # Å^4, coefficient of k^4 cos(4θ)
    tetrafoil4: ScalarOrRange = setting(
        0.0,
        help=(
            "Tetrafoil (4-fold astigmatism), k^4 sin(4*theta) coefficient "
            "in Angstrom^4: a single value or 'low,high' ([low, high] in TOML) range."
        ),
    )  # Å^4, coefficient of k^4 sin(4θ)

    # --- Advanced: anisotropic magnification ---
    # [[m00, m01], [m10, m11]], identity (no correction) by default. One fixed
    # matrix applied to every particle in a run (a microscope/session-level
    # calibration constant, not something that varies per particle).
    anisomag_m00: float = setting(
        1.0,
        help=(
            "Anisotropic magnification matrix element [0,0]. Identity "
            "(1.0) means no correction."
        ),
    )
    anisomag_m01: float = setting(
        0.0,
        help=(
            "Anisotropic magnification matrix element [0,1]. Identity "
            "(0.0) means no correction."
        ),
    )
    anisomag_m10: float = setting(
        0.0,
        help=(
            "Anisotropic magnification matrix element [1,0]. Identity "
            "(0.0) means no correction."
        ),
    )
    anisomag_m11: float = setting(
        1.0,
        help=(
            "Anisotropic magnification matrix element [1,1]. Identity "
            "(1.0) means no correction."
        ),
    )


PARTICLE_STACK_HELP: dict[str, str] = help_of(ParticleStackConfig)
