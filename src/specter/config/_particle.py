"""ParticleStackConfig: parameters for particle-stack generation."""

from __future__ import annotations

from dataclasses import dataclass

from ._common_fields import (
    alpha_setting,
    assembly_setting,
    bfactor_setting,
    bulk_scattering_factors_setting,
    cc_setting,
    convergence_angle_setting,
    crowd_max_distance_z_setting,
    cs_setting,
    deltaI_I_setting,
    deltaV_V_setting,
    detector_model_setting,
    dose_envelope_setting,
    energy_spread_setting,
    ice_cache_dir_setting,
    klim_setting,
    monomer_library_path_setting,
    noise_model_setting,
    oversample_setting,
    pdb_cache_dir_setting,
    pdb_source_setting,
    pixel_size_setting,
    save_clean_exitwaves_setting,
    save_exitwaves_setting,
    scattering_factors_setting,
    scattering_model_setting,
    temporal_coherence_setting,
    voltage_setting,
    absorption_model_setting,
    dose_envelope_target_setting,
    ice_motion_variance_setting,
    inelastic_mfp_solvent_setting,
    inelastic_mfp_specimen_setting,
    objective_aperture_setting,
    job_id_setting,
    output_dir_setting,
    project_setting,
    seed_setting,
)
from ._field import help_of, setting
from typing import Literal

from ._scalar_range import ScalarOrRange
from specter.options import (
    AbsorptionModel,
    ConvBackend,
    DetectorModel,
    DoseEnvelopeTarget,
    EwaldSphereSign,
    IceModel,
    NoiseModel,
    PotentialMethod,
    RotateMode,
    ScatteringFactors,
    TemporalCoherence,
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

    `dose`, `defocus`, `coincidence_radius` and the
    aberration-richness fields each take either a single number (e.g. ``20``,
    constant for every particle) or a ``[low, high]`` pair (e.g.
    ``[5000, 15000]``, sampled uniformly per particle). Comma-separated
    strings (``"5000,15000"``) are still accepted, since that's the only
    spelling a CLI flag can carry -- see :func:`parse_scalar_or_range`.
    """

    # --- Structure & potential (basic) ---
    pdb_source: str = pdb_source_setting()
    assembly: bool = assembly_setting()
    n_pixels: int = setting(
        256,
        help="Number of pixels per axis for the 3-D potential box.",
        check="positive",
    )
    pixel_size: float = pixel_size_setting()  # Å

    # --- Microscope (basic) ---
    voltage: float = voltage_setting()  # kV
    dose: ScalarOrRange = setting(
        20.0,
        help=(
            "Total dose per particle in e-/Angstrom^2: a single value (e.g. 20) for a "
            "constant dose per particle, or 'low,high' (e.g. 20,60) to sample uniformly per particle "
            "(in a TOML config, write the range as [20, 60])."
        ),
        check="positive_ordered",
    )  # e⁻/Å²
    cs: float = cs_setting()  # mm
    alpha: float = alpha_setting()  # unitless, amplitude contrast ratio
    absorption_model: AbsorptionModel = absorption_model_setting()
    inelastic_mfp_solvent: float | None = inelastic_mfp_solvent_setting()  # Å
    dose_weights_path: str | None = setting(
        None,
        help=(
            "Path to a (n_frames, n_bins) .npy of the exposure filter's own "
            "per-frame weights, e.g. CryoSPARC's refm_empirical_dw.npy. Frames are "
            "then summed in Fourier space under those weights instead of equally, "
            "which leaves the signal alone and raises the noise floor toward Nyquist "
            "the way a signal-preserving dose weighting does in real data. Requires "
            "n_frames."
        ),
        check="existing_file",
    )
    dose_weights_max_frequency: float | None = setting(
        None,
        help=(
            "Frequency in 1/Angstrom of the last bin of --dose_weights_path. Unset "
            "derives it from the motion-correction job's own files, which is the "
            "reliable route: the pixel size alone does not determine it."
        ),
        check="positive",
    )
    inelastic_mfp_specimen: float | None = inelastic_mfp_specimen_setting()  # Å
    objective_aperture: float | None = objective_aperture_setting()  # mrad

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
    scattering_model: Literal["multislice", "firstborn", "projection", "ctf"] = (
        scattering_model_setting()
    )
    detector_model: DetectorModel = detector_model_setting()

    # --- Post-processing (basic) ---
    normalize_particles: bool = setting(
        True, help="Normalize particles to zero mean and unit std."
    )
    save_exitwaves: bool = save_exitwaves_setting()
    save_clean_exitwaves: bool = save_clean_exitwaves_setting()

    # --- Compute (basic) ---
    device: str = setting(
        "cuda",
        help=(
            "Device to use: cpu | cuda | cuda:0 | 0,1,2,3. A comma-separated "
            "list splits the stack across the devices, one worker each; the "
            "stack is the one a single device makes."
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
    output_dir: str | None = output_dir_setting(
        "save .mrcs and .star files", "particles"
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
    project: str | None = project_setting("particles")
    job_id: str | None = job_id_setting()

    # --- Advanced ---
    # Relative to the current working directory, like any other CLI path
    # argument -- see default_pdb_cache_dir for the unset case.
    pdb_cache_dir: str = pdb_cache_dir_setting()
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
    convergence_angle: float | None = convergence_angle_setting()  # mrad
    cc: float | None = cc_setting()  # mm
    energy_spread: float = energy_spread_setting()  # eV (FWHM)
    deltaV_V: float = deltaV_V_setting()  # unitless (ΔV/V)
    deltaI_I: float = deltaI_I_setting()  # unitless (ΔI/I)
    temporal_coherence: TemporalCoherence = temporal_coherence_setting()
    dose_envelope: bool = dose_envelope_setting()
    dose_envelope_target: DoseEnvelopeTarget = dose_envelope_target_setting("particle")
    bfactor: float | None = bfactor_setting()  # Å²
    noise_model: NoiseModel = noise_model_setting()
    coincidence_radius: ScalarOrRange = setting(
        0.0,
        help=(
            "Effective coincidence exclusion radius in pixels "
            "(exclusion area = pi*r^2): a single value for constant radius, or "
            "'low,high' ([low, high] in TOML) to sample uniformly per particle."
        ),
        check="non_negative_ordered",
    )  # pixels
    detector_pixel_size: float | None = setting(
        None,
        help=(
            "Pixel size in Angstrom at which the movies were recorded, when the "
            "particles were resampled afterwards (Fourier-cropped or binned). The "
            "detector MTF is evaluated on this physical pixel. Unset: the image "
            "pixel is the physical pixel."
        ),
        check="positive",
    )  # Å
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
    ice_cache_dir: str | None = (
        ice_cache_dir_setting()
    )  # defaults to the bundled ice_data/ice_cache
    crowd_min_distance: float | None = setting(
        None,
        help=(
            "Minimum distance between crowded particles in "
            "Angstrom. Unset disables crowding."
        ),
        check="non_negative",
    )  # Å
    crowd_max_distance_z: float | None = crowd_max_distance_z_setting()  # Å
    pad_fft: bool = setting(
        True, help="Pad the volume for FFT to avoid edge artifacts."
    )

    # --- Advanced: potential building ---
    scattering_factors: ScatteringFactors = scattering_factors_setting()
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
    monomer_library_path: str | None = monomer_library_path_setting()
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
        "negative",
        help=(
            "Ewald-sphere curvature sign. Reconstruct the stack with the same sign: "
            "a mismatch costs resolution wherever the curvature matters."
        ),
    )
    # Fraction of Nyquist, not 1/A: Scattering masks k <= klim * k_nyquist.
    # Kirkland recommends 0.66 (2/3) to prevent multislice FFT aliasing, but
    # that costs real spatial resolution, so the default keeps the full range
    # and accepts the aliasing. Exposed so a caller can make the other choice.
    klim: float | None = klim_setting()  # fraction of Nyquist
    oversample: int = oversample_setting()
    rotate_mode: RotateMode = setting(
        "exact",
        help=(
            "How the template is rotated to each particle's pose. 'exact' (default): no "
            "interpolation; the template's transform is evaluated at the rotated "
            "frequencies with a NUFFT. The transform is periodic, so density a rotation "
            "and shift carry past a face re-enters at the opposite one; it warns when "
            "more than 1% of the template could. 'real': trilinear "
            "interpolation, which keeps about half the amplitude near Nyquist. "
            "'fourier': trilinear on the spectrum, about 0.8 of it in every band."
        ),
    )

    # --- Advanced: ice ---
    # Everything specter renders that is NOT a biomolecule: the ice.
    # Kept separate from `scattering_factors` on purpose -- Shtyrov fits bonded
    # species of biomolecules over 0.011-0.62 1/A, so bulk materials are out of
    # its domain and a mean inner potential (a k=0 quantity) extrapolates below
    # the fitted range. Kirkland/Lobato/Peng are per-element, valid at k=0, and
    # agree there. See ice._kernels.build_water_kernel for the measurements.
    bulk_scattering_factors: ScatteringFactors = bulk_scattering_factors_setting()
    ice_relax_steps: int = setting(
        0,
        help=(
            "Local MLBOP seam-relaxation steps, only used when "
            "ice_model='gd' tiles multiple cached blocks."
        ),
        check="non_negative",
    )
    ice_motion_variance: float | None = ice_motion_variance_setting()

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
    seed: int | None = seed_setting(
        "pose, CTF, dose, ice, crowding and noise sampling",
        "Auto-generated and logged if unset. Each particle's crowding, ice and "
        "noise are drawn from a generator seeded by (seed, particle index), so "
        "the stack is the same for any batchsize and any number of devices.",
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
