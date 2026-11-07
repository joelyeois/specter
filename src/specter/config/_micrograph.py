"""MicrographConfig: parameters for micrograph generation."""

from __future__ import annotations

from dataclasses import dataclass

from ._common_fields import (
    klim_setting,
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
    job_id_setting,
    objective_aperture_setting,
    output_dir_setting,
    project_setting,
    seed_setting,
)
from ._field import help_of, setting
from typing import Literal

from ._scalar_range import ScalarOrRange
from specter.options import (
    AbsorptionModel,
    DetectorModel,
    DoseEnvelopeTarget,
    IceModel,
    NoiseModel,
    ScatteringFactors,
    TemporalCoherence,
)


@dataclass
class MicrographConfig:
    """Parameters for micrograph generation, loaded from a TOML config file.

    `dose`, `defocus` and `coincidence_radius` each take
    either a single number (e.g. ``20``, constant for every micrograph) or a
    ``[low, high]`` pair (e.g. ``[5000, 15000]``, sampled uniformly per
    micrograph) -- the same scalar-or-range convention as
    `ParticleStackConfig`/`TiltSeriesConfig`. Comma-separated strings
    (``"5000,15000"``) are still accepted, since that's the only spelling a
    CLI flag can carry -- see :func:`parse_scalar_or_range`.
    """

    # --- PDB / potential ---
    pdb_source: str = pdb_source_setting()
    assembly: bool = assembly_setting()
    # Relative to the current working directory, like any other CLI path
    # argument -- see default_pdb_cache_dir for the unset case.
    pdb_cache_dir: str = pdb_cache_dir_setting()
    n_pixels: int = setting(
        256,
        help="Number of pixels per axis for the 3-D particle potential box.",
        check="positive",
    )
    pixel_size: float = pixel_size_setting()  # Å
    micrograph_size: int = setting(
        4096, help="Micrograph size in pixels (square).", check="positive"
    )
    scattering_factors: ScatteringFactors = scattering_factors_setting()
    # Everything specter renders that is NOT a biomolecule: the ice.
    # Kept separate from `scattering_factors` on purpose -- Shtyrov fits bonded
    # species of biomolecules over 0.011-0.62 1/A, so bulk materials are out of
    # its domain and a mean inner potential (a k=0 quantity) extrapolates below
    # the fitted range. Kirkland/Lobato/Peng are per-element, valid at k=0, and
    # agree there. See ice._kernels.build_water_kernel for the measurements.
    bulk_scattering_factors: ScatteringFactors = bulk_scattering_factors_setting()
    # Only takes effect for scattering_factors="shtyrov" (the only one that
    # types atoms) and only when a Monomer Library is available.
    readd_hydrogens: bool | Literal["auto"] = setting(
        "auto",
        help=(
            "Whether to replace a structure's own hydrogens with "
            "the monomer library's ideal geometry: 'auto' (default) keeps hydrogens "
            "the file already carries and adds them only when it has none, true "
            "always re-adds, false never adds hydrogen density (they still inform "
            "atom typing). Only meaningful when a monomer library is available."
        ),
    )
    # Unset falls back to $CLIBD_MON. A field as well as a variable because
    # the library changes the rendered potential, so a run should be
    # reproducible from its own config -- see ParticleStackConfig's own note.
    monomer_library_path: str | None = monomer_library_path_setting()
    # Off by default: a deposited B-factor is refinement output, not a
    # measured mean-square displacement, and applying a structure's own
    # column silently makes the rendered specimen depend on who deposited
    # it. Requires scattering_factors="shtyrov"; anything else raises.
    use_deposited_bfactors: bool = setting(
        False,
        help=(
            "Damp each atom by the B-factor its structure "
            "deposits, instead of rendering the model statically. Only a PER-ATOM "
            "B adds anything an envelope cannot: a uniform one is the same "
            "exp(-B k^2/4) as --bfactor, so setting both double-counts. A deposited "
            "column is refinement output rather than a measured displacement, and "
            "cryo-EM entries often carry a constant or zero one. Requires "
            "scattering_factors='shtyrov'."
        ),
    )

    # --- Microscope / physics ---
    voltage: float = voltage_setting()  # kV
    dose: ScalarOrRange = setting(
        20.0,
        help=(
            "Total dose per micrograph in e-/Angstrom^2: a single value (e.g. 20) for a "
            "constant dose per micrograph, or 'low,high' (e.g. 20,60) to sample uniformly per micrograph "
            "(in a TOML config, write the range as [20, 60])."
        ),
        check="positive_ordered",
    )  # e⁻/Å²
    n_frames: int | None = setting(
        None,
        help=(
            "Number of movie frames. Defaults to int(dose) if not set. "
            "Only affects the image when coincidence_radius > 0, which is what "
            "splits the dose into frames; ignored otherwise."
        ),
        check="positive",
    )
    cs: float = cs_setting()  # mm
    alpha: float = alpha_setting()  # unitless, amplitude contrast ratio
    absorption_model: AbsorptionModel = absorption_model_setting()
    inelastic_mfp_solvent: float | None = inelastic_mfp_solvent_setting()  # Å
    inelastic_mfp_specimen: float | None = inelastic_mfp_specimen_setting()  # Å
    objective_aperture: float | None = objective_aperture_setting()  # mrad

    # --- Envelopes ---
    convergence_angle: float | None = convergence_angle_setting()  # mrad
    cc: float | None = cc_setting()  # mm
    energy_spread: float = energy_spread_setting()  # eV (FWHM)
    deltaV_V: float = deltaV_V_setting()  # unitless (ΔV/V)
    deltaI_I: float = deltaI_I_setting()  # unitless (ΔI/I)
    temporal_coherence: TemporalCoherence = temporal_coherence_setting()
    dose_envelope: bool = dose_envelope_setting()
    dose_envelope_target: DoseEnvelopeTarget = dose_envelope_target_setting(
        "specimen",
        "Each micrograph's specimen is built for its own dose.",
    )

    # --- Defocus ---
    defocus: ScalarOrRange = setting(
        factory=lambda: [5000.0, 15000.0],
        help=(
            "Defocus in Angstrom: a single value (e.g. 8000) for constant "
            "defocus, or 'low,high' (e.g. 5000,15000) to sample uniformly per micrograph "
            "(in a TOML config, write the range as [5000, 15000])."
        ),
        check="non_negative_ordered",
    )  # Å

    # --- Dataset size ---
    n_micrographs: int = setting(
        1, help="Number of micrographs to simulate.", check="positive"
    )

    # --- Models ---
    scattering_model: Literal["multislice", "firstborn", "projection", "ctf"] = (
        scattering_model_setting()
    )
    noise_model: NoiseModel = noise_model_setting()
    coincidence_radius: ScalarOrRange = setting(
        0.0,
        help=(
            "Effective coincidence exclusion radius in pixels "
            "(exclusion area = pi*r^2): a single value for constant radius, or "
            "'low,high' ([low, high] in TOML) to sample uniformly per micrograph."
        ),
        check="non_negative_ordered",
    )  # pixels; 0 = plain Poisson
    ice_model: IceModel = setting(
        "gd",
        help=(
            "Ice model: 'gd' (samples the pre-generated IceBank cache), "
            "'random' (cheap, low-realism), or 'none'."
        ),
    )
    ice_thickness: float = setting(
        500.0,
        help=(
            "Ice thickness in Angstrom. 0 = minimum (particle box size). "
            "For ice_profile='wedge' without a range this is the thickness at the "
            "centre of the field; for 'meniscus' it is the thickness at the hole centre."
        ),
        check="non_negative",
    )  # Å, 0 = minimum (particle box size)
    # --- Ice profile (specter.ice.IceProfile) ---
    # "flat" reproduces a bare `ice_thickness` slab exactly and is the default,
    # so none of the fields below do anything until `ice_profile` is changed.
    ice_profile: Literal["flat", "wedge", "meniscus"] = setting(
        "flat",
        help=(
            "Lateral ice thickness profile. 'flat' is a uniform slab "
            "(the default, identical to previous behaviour). 'wedge' ramps linearly "
            "across the field; 'meniscus' is the radial film in a foil hole, with the "
            "field placed anywhere in it via ice_hole_offset. Note that the volume is "
            "sized by the THICKEST column and multislice costs one full-plane FFT per "
            "slice regardless of what it holds, so a profile costs what its thickest "
            "column costs over the whole field."
        ),
    )
    ice_thickness_range: ScalarOrRange | None = setting(
        None,
        help=(
            "ice_profile='wedge' only: thickness in Angstrom "
            "across the field's full width, as 'min,max' (e.g. 250,900; in TOML write "
            "[250, 900]). Overrides ice_thickness."
        ),
        check="positive_ordered",
    )  # Å, wedge: [min, max]
    ice_profile_angle: float = setting(
        0.0,
        help=(
            "ice_profile='wedge' only: direction of the thickness "
            "ramp in degrees from the +x axis."
        ),
    )  # degrees, wedge ramp direction
    ice_hole_radius: float = setting(
        6000.0,
        help=(
            "ice_profile='meniscus' only: foil hole radius in "
            "Angstrom (a 1.2 um hole is 6000)."
        ),
        check="non_negative",
    )  # Å, meniscus (1.2 µm hole)
    ice_rim_thickness: float = setting(
        1500.0,
        help=(
            "ice_profile='meniscus' only: ice thickness in Angstrom at the hole rim."
        ),
        check="non_negative",
    )  # Å, meniscus thickness at the rim
    ice_hole_offset: ScalarOrRange = setting(
        0.0,
        help=(
            "ice_profile='meniscus' only: position of the hole's "
            "centre in field coordinates as 'x,y' in Angstrom ([x, y] in TOML). A "
            "micrograph is a small patch of a hole, so this is what decides whether "
            "it looks flat, wedged, or strongly curved."
        ),
    )  # Å, meniscus: hole centre as [x, y]
    ice_tilt: float = setting(
        0.0,
        help=(
            "Slope of the ice slab's mid-plane, in Angstrom of z per "
            "Angstrom laterally. Moves both surfaces together, leaving thickness "
            "unchanged -- a tilted specimen rather than a varying one. Applies to "
            "every ice_profile mode."
        ),
    )  # Å of z per Å laterally, slab mid-plane slope
    ice_cache_dir: str | None = (
        ice_cache_dir_setting()
    )  # defaults to the bundled ice_data/ice_cache
    ice_motion_variance: float | None = ice_motion_variance_setting()
    crowd_min_distance: float | None = setting(
        None,
        help=(
            "Minimum distance between crowded particles in "
            "Angstrom. Defaults to the structure's max diameter; set to 0 to disable "
            "crowding."
        ),
        check="non_negative",
    )  # Å
    crowd_max_distance_z: float | None = crowd_max_distance_z_setting()  # Å
    water_air_interface: bool = setting(
        True,
        help=(
            "Model a water-air interface when placing ice/"
            "crowding (bimodal density along z instead of uniform)."
        ),
    )
    sigma_frac: float = setting(
        0.05,
        help=(
            "Gaussian width as a fraction of the local ice thickness, "
            "for the water_air_interface bias. Smaller pulls adsorbed particles "
            "into a tighter shell against each surface. water_air_interface=True "
            "only."
        ),
    )  # unitless, fraction of local ice thickness
    peak_amplitude: float = setting(
        1.0,
        help=(
            "Amplitude of the two Gaussians centered on each ice "
            "surface, for the water_air_interface bias. water_air_interface=True "
            "only."
        ),
    )  # unitless, keep-probability at each surface
    baseline: float = setting(
        0.1,
        help=(
            "Minimum keep-probability in the bulk, away from either ice "
            "surface, for the water_air_interface bias -- the fraction of particles "
            "left floating free in solution rather than adsorbed. "
            "water_air_interface=True only."
        ),
    )  # unitless, minimum keep-probability in the bulk
    packing_backend: Literal["poisson_disk", "shape"] = setting(
        "poisson_disk",
        help=(
            "Crowding placement algorithm: 'poisson_disk' "
            "(default) is bounding-sphere-exclusion Poisson-disk sampling; 'shape' "
            "collides the real rotated molecular footprint against a running "
            "occupancy grid (the same packer TomogramSpecimenGenerator uses), "
            "reaching substantially higher crowding density. Both backends confine "
            "placement to a non-flat ice_profile's local slab and apply "
            "water_air_interface the same way."
        ),
    )
    packing_gap: float = setting(
        0.0,
        help=(
            "Extra clearance baked into the shape backend's footprint "
            "mask, Angstrom. packing_backend='shape' only."
        ),
    )  # Å, shape backend only
    n_orientations: int = setting(
        256,
        help=(
            "Size of the shape backend's per-instance rotation "
            "cache. packing_backend='shape' only."
        ),
    )  # shape backend only
    packing_max_retries: int = setting(
        1500,
        help=(
            "Shape backend's attempts-per-instance ceiling -- "
            "the knob that sets achieved density. packing_backend='shape' only."
        ),
    )  # shape backend only -- sets achieved density
    packing_stall_patience: int = setting(
        5000,
        help=(
            "Shape backend's early-stop threshold "
            "(consecutive failed attempts before abandoning). packing_backend="
            "'shape' only."
        ),
    )  # shape backend only
    packing_seed: int | None = setting(
        None,
        help=(
            "Shape backend's RNG seed. With --seed, mixed into each "
            "micrograph's own packing seed. packing_backend='shape' only."
        ),
    )  # shape backend only
    n_candidates: int | None = setting(
        None,
        help=(
            "Shape backend's candidate pool size. Unset: estimated "
            "from grid and footprint volume. packing_backend='shape' only."
        ),
    )  # shape backend only; None = auto-estimated
    # Fraction of Nyquist, not 1/A: Scattering masks k <= klim * k_nyquist.
    # Kirkland recommends 0.66 (2/3) to prevent multislice FFT aliasing, but
    # that costs real spatial resolution, so the default keeps the full range
    # and accepts the aliasing. Exposed so a caller can make the other choice.
    klim: float | None = klim_setting()  # fraction of Nyquist
    oversample: int = oversample_setting()
    bfactor: float | None = bfactor_setting()  # A^2
    pad_fft: bool = setting(
        False,
        help=(
            "Pad the volume for FFT to avoid edge artifacts. Not needed for "
            "level, uniform ice, whose opposite edges match; needed for a "
            "'wedge' profile or a nonzero ice_tilt, which warn without it."
        ),
    )
    crowd_chunk_size: int = setting(
        1,
        help=(
            "Crowding duplicate volumes rotated per batch. Lowering "
            "it to 1 costs no wall time: at micrograph scale wall time is flat in this "
            "while peak memory grows linearly with it, so raising it above the default "
            "buys nothing."
        ),
        check="positive",
    )  # duplicates rotated per batch; see crowding.py
    detector_model: DetectorModel = detector_model_setting()

    # --- Post-processing ---
    normalize_micrographs: bool = setting(
        False, help="Normalize micrographs to zero mean and unit std."
    )
    save_exitwaves: bool = save_exitwaves_setting()
    save_clean_exitwaves: bool = save_clean_exitwaves_setting()

    # --- Compute ---
    device: str = setting(
        "cuda", help="Device to use: cpu | cuda | cuda:0."
    )  # falls back to CPU when none is available

    # --- Reproducibility ---
    seed: int | None = seed_setting("CTF, dose, ice, crowding, pose and noise sampling")

    # --- Output ---
    # One path field, not one per layout: this is the single directory a run
    # writes under, read as the leaf when untracked and as the root of the
    # numbered job tree when tracked. `None` rather than a baked-in default
    # because which default applies is not knowable until tracking is -- see
    # pipelines._common.resolve_output_dir.
    output_dir: str | None = output_dir_setting(
        "save .mrcs and .star files", "micrographs"
    )
    filename: str = setting(
        "micrographs", help="Base name for output files (no extension)."
    )

    # --- Job tracking (opt-in) ---
    # Setting `project` or `job_id` routes output through `specter.jobs`
    # instead of the flat output_dir/filename layout above: the directory
    # becomes output_dir/[project/]micrographs/J00N/, numbered and with
    # a job.json recording the full parameter set, git commit and status.
    # Neither is required -- leaving both unset keeps today's exact flat
    # behavior.
    project: str | None = project_setting("micrographs")
    job_id: str | None = job_id_setting()


MICROGRAPH_HELP: dict[str, str] = help_of(MicrographConfig)
