"""MicrographConfig: parameters for micrograph generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ._paths import default_pdb_cache_dir
from ._scalar_range import ScalarOrRange


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
    pdb_source: str
    assembly: bool = True
    # Relative to the current working directory, like any other CLI path
    # argument -- see default_pdb_cache_dir for the unset case.
    pdb_cache_dir: str = field(default_factory=default_pdb_cache_dir)
    n_pixels: int = 256
    pixel_size: float = 1.0  # Å
    micrograph_size: int = 4096
    scattering_factors: Literal["shtyrov", "kirkland", "lobato"] = "shtyrov"
    # Everything specter renders that is NOT a biomolecule: the ice.
    # Kept separate from `scattering_factors` on purpose -- Shtyrov fits bonded
    # species of biomolecules over 0.011-0.62 1/A, so bulk materials are out of
    # its domain and a mean inner potential (a k=0 quantity) extrapolates below
    # the fitted range. Kirkland/Lobato/Peng are per-element, valid at k=0, and
    # agree there. See ice._kernels.build_water_kernel for the measurements.
    bulk_scattering_factors: Literal["kirkland", "lobato", "shtyrov"] = "kirkland"
    # Only takes effect for scattering_factors="shtyrov" (the only one that
    # types atoms) and only when a Monomer Library is available.
    readd_hydrogens: bool | Literal["auto"] = "auto"
    # Unset falls back to $CLIBD_MON. A field as well as a variable because
    # the library changes the rendered potential, so a run should be
    # reproducible from its own config -- see ParticleStackConfig's own note.
    monomer_library_path: str | None = None
    # Off by default: a deposited B-factor is refinement output, not a
    # measured mean-square displacement, and applying a structure's own
    # column silently makes the rendered specimen depend on who deposited
    # it. Requires scattering_factors="shtyrov"; anything else raises.
    use_deposited_bfactors: bool = False

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
    noise_model: Literal["poisson", "none"] = "poisson"
    coincidence_radius: ScalarOrRange = 0.0  # pixels; 0 = plain Poisson
    ice_model: Literal["gd", "random", "none"] = "gd"
    ice_thickness: float = 500.0  # Å, 0 = minimum (particle box size)
    # --- Ice profile (specter.ice.IceProfile) ---
    # "flat" reproduces a bare `ice_thickness` slab exactly and is the default,
    # so none of the fields below do anything until `ice_profile` is changed.
    ice_profile: Literal["flat", "wedge", "meniscus"] = "flat"
    ice_thickness_range: ScalarOrRange | None = None  # Å, wedge: [min, max]
    ice_profile_angle: float = 0.0  # degrees, wedge ramp direction
    ice_hole_radius: float = 6000.0  # Å, meniscus (1.2 µm hole)
    ice_rim_thickness: float = 1500.0  # Å, meniscus thickness at the rim
    ice_hole_offset: ScalarOrRange = 0.0  # Å, meniscus: hole centre as [x, y]
    ice_tilt: float = 0.0  # Å of z per Å laterally, slab mid-plane slope
    ice_cache_dir: str | None = None  # defaults to the bundled ice_data/ice_cache
    crowd_min_distance: float | None = None  # Å
    crowd_max_distance_z: float | None = None  # Å
    water_air_interface: bool = True
    packing_backend: Literal["poisson_disk", "shape"] = "poisson_disk"
    packing_gap: float = 0.0  # Å, shape backend only
    n_orientations: int = 256  # shape backend only
    packing_max_retries: int = 1500  # shape backend only -- sets achieved density
    packing_stall_patience: int = 5000  # shape backend only
    packing_seed: int | None = None  # shape backend only
    n_candidates: int | None = None  # shape backend only; None = auto-estimated
    potential_scale: ScalarOrRange = 1.0  # unitless
    # Fraction of Nyquist, not 1/A: Scattering masks k <= klim * k_nyquist.
    # Kirkland recommends 0.66 (2/3) to prevent multislice FFT aliasing, but
    # that costs real spatial resolution, so the default keeps the full range
    # and accepts the aliasing. Exposed so a caller can make the other choice.
    klim: float | None = None  # fraction of Nyquist
    bfactor: float | None = None  # A^2
    pad_fft: bool = False
    crowd_chunk_size: int = 1  # duplicates rotated per batch; see crowding.py
    detector_model: Literal["none", "perfect", "k3_300kv", "k3_200kv"] = "none"

    # --- Post-processing ---
    normalize_micrographs: bool = False
    save_exitwaves: bool = False
    save_clean_exitwaves: bool = False

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
    filename: str = "micrographs"

    # --- Job tracking (opt-in) ---
    # Setting `project` or `job_id` routes output through `specter.jobs`
    # instead of the flat output_dir/filename layout above: the directory
    # becomes output_dir/[project/]micrographs/J00N/, numbered and with
    # a job.json recording the full parameter set, git commit and status.
    # Neither is required -- leaving both unset keeps today's exact flat
    # behavior.
    project: str | None = None
    job_id: str | None = None


# Human-readable per-field descriptions for MicrographConfig, used to build
# `specter simulate micrograph --help` (see specter/cli/_click_options.py). Kept
# here, next to the dataclass, so adding/renaming a field and its help text happen
# in the same place.
MICROGRAPH_HELP: dict[str, str] = {
    "pdb_source": "Path to a local .cif/.pdb file, or a 4-character PDB "
    "accession code to fetch and cache. A local file is read where it "
    "lies and never copied into the cache; an existing file wins over a "
    "same-named accession code.",
    "assembly": "Fetch the biological assembly.",
    "pdb_cache_dir": "Where downloaded PDB/mmCIF structures are cached. An "
    "input cache shared by every run, not an output location -- job tracking "
    "does not redirect it.",
    "n_pixels": "Number of pixels per axis for the 3-D particle potential box.",
    "pixel_size": "Pixel size in Angstrom.",
    "micrograph_size": "Micrograph size in pixels (square).",
    "scattering_factors": "Atomic scattering-factor parameterization used to "
    "build the structure's scattering potential.",
    "bulk_scattering_factors": "Atomic scattering-factor parameterization for the ice -- everything rendered that is not a biomolecule. Deliberately separate from scattering_factors: Shtyrov is fitted for biomolecules, and these materials are outside that domain.",
    "readd_hydrogens": "Whether to replace a structure's own hydrogens with "
    "the monomer library's ideal geometry: 'auto' (default) keeps hydrogens "
    "the file already carries and adds them only when it has none, true "
    "always re-adds, false never adds hydrogen density (they still inform "
    "atom typing). Only meaningful when a monomer library is available.",
    "monomer_library_path": "Path to a Monomer Library "
    "(https://github.com/MonomerLibrary/monomers), which completes a "
    "structure's bond topology and hydrogens so Shtyrov species resolve. "
    "Unset falls back to $CLIBD_MON. Without one, around 44% of a "
    "hydrogen-free protein falls back to per-element Peng factors.",
    "use_deposited_bfactors": "Damp each atom by the B-factor its structure "
    "deposits, instead of rendering the model statically. Only a PER-ATOM "
    "B adds anything an envelope cannot: a uniform one is the same "
    "exp(-B k^2/4) as --bfactor, so setting both double-counts. A deposited "
    "column is refinement output rather than a measured displacement, and "
    "cryo-EM entries often carry a constant or zero one. Requires "
    "scattering_factors='shtyrov'.",
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
    "noise_model": "Noise model. Use 'none' for no noise.",
    "coincidence_radius": "Effective coincidence exclusion radius in pixels "
    "(exclusion area = pi*r^2): a single value for constant radius, or "
    "'low,high' ([low, high] in TOML) to sample uniformly per micrograph.",
    "ice_model": "Ice model: 'gd' (samples the pre-generated IceBank cache), "
    "'random' (cheap, low-realism), or 'none'.",
    "ice_thickness": "Ice thickness in Angstrom. 0 = minimum (particle box size). "
    "For ice_profile='wedge' without a range this is the thickness at the "
    "centre of the field; for 'meniscus' it is the thickness at the hole centre.",
    "ice_profile": "Lateral ice thickness profile. 'flat' is a uniform slab "
    "(the default, identical to previous behaviour). 'wedge' ramps linearly "
    "across the field; 'meniscus' is the radial film in a foil hole, with the "
    "field placed anywhere in it via ice_hole_offset. Note that the volume is "
    "sized by the THICKEST column and multislice costs one full-plane FFT per "
    "slice regardless of what it holds, so a profile costs what its thickest "
    "column costs over the whole field.",
    "ice_thickness_range": "ice_profile='wedge' only: thickness in Angstrom "
    "across the field's full width, as 'min,max' (e.g. 250,900; in TOML write "
    "[250, 900]). Overrides ice_thickness.",
    "ice_profile_angle": "ice_profile='wedge' only: direction of the thickness "
    "ramp in degrees from the +x axis.",
    "ice_hole_radius": "ice_profile='meniscus' only: foil hole radius in "
    "Angstrom (a 1.2 um hole is 6000).",
    "ice_rim_thickness": "ice_profile='meniscus' only: ice thickness in "
    "Angstrom at the hole rim.",
    "ice_hole_offset": "ice_profile='meniscus' only: position of the hole's "
    "centre in field coordinates as 'x,y' in Angstrom ([x, y] in TOML). A "
    "micrograph is a small patch of a hole, so this is what decides whether "
    "it looks flat, wedged, or strongly curved.",
    "ice_tilt": "Slope of the ice slab's mid-plane, in Angstrom of z per "
    "Angstrom laterally. Moves both surfaces together, leaving thickness "
    "unchanged -- a tilted specimen rather than a varying one. Applies to "
    "every ice_profile mode.",
    "ice_cache_dir": "Directory of cached ice configs for ice_model='gd'. "
    "Defaults to the bundled ice_data/ice_cache.",
    "crowd_min_distance": "Minimum distance between crowded particles in "
    "Angstrom. Defaults to the structure's max diameter; set to 0 to disable "
    "crowding.",
    "crowd_max_distance_z": "Maximum z-distance between crowded particles in Angstrom.",
    "water_air_interface": "Model a water-air interface when placing ice/"
    "crowding (bimodal density along z instead of uniform).",
    "packing_backend": "Crowding placement algorithm: 'poisson_disk' "
    "(default) is bounding-sphere-exclusion Poisson-disk sampling; 'shape' "
    "collides the real rotated molecular footprint against a running "
    "occupancy grid (the same packer TomogramSpecimenGenerator uses), "
    "reaching substantially higher crowding density. Not yet aware of a "
    "non-flat ice_profile -- placement is unconstrained by the local slab "
    "when 'shape' is selected.",
    "packing_gap": "Extra clearance baked into the shape backend's footprint "
    "mask, Angstrom. packing_backend='shape' only.",
    "n_orientations": "Size of the shape backend's per-instance rotation "
    "cache. packing_backend='shape' only.",
    "packing_max_retries": "Shape backend's attempts-per-instance ceiling -- "
    "the knob that sets achieved density. packing_backend='shape' only.",
    "packing_stall_patience": "Shape backend's early-stop threshold "
    "(consecutive failed attempts before abandoning). packing_backend="
    "'shape' only.",
    "packing_seed": "Shape backend's RNG seed. packing_backend='shape' only.",
    "n_candidates": "Shape backend's candidate pool size. Unset: estimated "
    "from grid and footprint volume. packing_backend='shape' only.",
    "potential_scale": "Potential scale factor (unitless, values < 1 "
    "approximate thicker ice): a single value for constant scale, or "
    "'low,high' ([low, high] in TOML) to sample uniformly per micrograph.",
    "klim": "Bandlimit for Kirkland's FFT anti-aliasing, as a fraction of "
    "Nyquist. Kirkland recommends 0.66 (2/3), which prevents aliasing but "
    "discards real spatial frequency content above it. Unset (the default) "
    "keeps the full Nyquist range and accepts the aliasing.",
    "bfactor": "Isotropic B-factor envelope in Angstrom^2.",
    "pad_fft": "Pad the volume for FFT to avoid edge artifacts.",
    "crowd_chunk_size": "Crowding duplicate volumes rotated per batch. Lowering "
    "it to 1 costs no wall time: at micrograph scale wall time is flat in this "
    "while peak memory grows linearly with it, so raising it above the default "
    "buys nothing.",
    "detector_model": "Detector model.",
    "normalize_micrographs": "Normalize micrographs to zero mean and unit std.",
    "save_exitwaves": "Save exit wave magnitude and phase as separate .mrcs files.",
    "save_clean_exitwaves": "Save clean (particle-only, no ice) exit wave "
    "magnitude and phase.",
    "device": "Device to use: cpu | cuda | cuda:0.",
    "seed": "RNG seed for ice, crowding, pose and noise sampling. Auto-generated and logged if unset.",
    "output_dir": "Directory to save .mrcs and .star files when untracked. Setting --project or --job_id instead makes this the root of the numbered job tree, so tracking organises output within the folder you chose rather than moving it elsewhere. Unset defaults to <artifact>/ untracked, and to the project root found by walking up from cwd for an existing .specter marker when tracked.",
    "filename": "Base name for output files (no extension).",
    "project": "Optional: number and track this run through specter.jobs. "
    "Not required for tracking -- job_id alone also triggers it. The run "
    "lands in "
    "<output_dir>/[<project>/]micrographs/J00N/ with a job.json "
    "recording every parameter, the git commit and the run's status.",
    "job_id": "Pin the job directory (e.g. J001) rather than auto-assigning "
    "the next one: resumes into it if it exists, creates it otherwise.",
}
