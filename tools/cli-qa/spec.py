"""Declarative sweep spec: what to run, with which perturbations, expecting what.

Phase 1 asks one question per flag: *does setting this flag actually change the
output?* A flag that parses cleanly, lands in the config, and then changes
nothing is a silent no-op -- the failure mode a user never notices.

Each `Flag` names a perturbed value distinct from the baseline, plus any
`context` needed for the flag to be meaningful at all (an ice flag means
nothing with `--ice_model none`), plus what we expect to happen:

    "changes"   pixel data must differ from baseline
    "unchanged" pixel data must NOT differ (batching/perf knobs -- an
                invariance check, not a no-op check)
    "metadata"  only the .star may differ
    "artifacts" new output files must appear
    "skip"      excluded from Phase 1 (covered elsewhere, or not user-facing)
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from specter.config import (
    MicrographConfig,
    ParticleStackConfig,
    TiltSeriesConfig,
    TomogramConfig,
)

# Structures kept local so the sweep never depends on network fetches. A cache
# may hold an accession as the deposited file or as its first assembly, so take
# whichever is there.
_PDB_CACHE = Path(
    os.environ.get("SPECTER_PDB_CACHE", "") or Path.home() / ".cache/specter/pdb"
)


def _cached(code: str) -> str:
    for name in (f"{code}.cif", f"{code}-assembly1.cif"):
        if (_PDB_CACHE / name).is_file():
            return str(_PDB_CACHE / name)
    return str(_PDB_CACHE / f"{code}.cif")


PDB_A = _cached("1mbo")
PDB_B = _cached("1A6M")

# --------------------------------------------------------------------------
# Not exposed to users yet -- deliberately out of scope for this sweep.
#   torch_ctf backend  : no CLI flag exists at all (Python API only)
#   fresh ice cubes    : GradientSKIcemaker/build_ice_cache have no CLI path
# What remains here are the expert/internal knobs adjacent to those.
# --------------------------------------------------------------------------
NOT_USER_FACING = {
    "ice_cache_dir",
    "ice_relax_steps",
    "bulk_scattering_factors",
    "shtyrov_params_path",
    "conv_backend",
}


@dataclass
class Flag:
    """One flag perturbation to test."""

    name: str
    value: str | None = None  # None => boolean-style flag already encoded in `value`
    context: list[str] = field(default_factory=list)
    expect: str = "changes"
    note: str = ""

    def argv(self) -> list[str]:
        args = list(self.context)
        if self.value is not None:
            args += [f"--{self.name}", self.value]
        return args


@dataclass
class CommandSpec:
    """A CLI command, its fast baseline, and the flags to sweep against it."""

    key: str
    argv: list[str]  # e.g. ["simulate", "particles"]
    baseline: list[str]
    flags: list[Flag]
    timeout: float = 900.0
    # The config the command loads by default, so perturbations can be checked
    # against the values actually in effect rather than the dataclass defaults.
    config_cls: type | None = None
    config_path: str = ""


# Contexts reused by several flags.
CROWD = ["--crowd_min_distance", "100.0"]
# The meniscus params only bite once the mode selects them, and only when the
# hole is small enough that a micrograph-sized field sees real curvature -- at
# the default 6000 A radius a 200 nm field is nearly flat.
MENISCUS = ["--ice_profile", "meniscus", "--ice_hole_radius", "3000.0"]
DETECTOR = ["--detector_model", "k3_300kv"]
CC_ON = ["--cc", "2.0"]
# Expected counts rather than Poisson draws. A change of a few parts per
# million in the rate (the focal spread at a coarse tilt-series sampling) moves
# no seeded Poisson count at all, so it is only visible noiseless.
NOISELESS = ["--noise_model", "none"]
# The tilt baseline's 6 A voxels put Nyquist where the focal spread barely
# acts (~1e-5 relative, under the sweep's 5e-3 tolerance); at 2 A it is plain.
TILT_FINE = ["--voxel_size", "2.0", *NOISELESS]
MFP = ["--absorption_model", "inelastic_mfp", "--alpha", "0.0"]
SHAPE = ["--packing_backend", "shape"]

_PARTICLE_BASELINE = [
    "--pdb_source",
    PDB_A,
    "--n_particles",
    "2",
    "--n_pixels",
    "48",
    "--pixel_size",
    "3.0",
    "--seed",
    "1234",
    "--batchsize",
    "4",  # pinned: a seeded run rejects batchsize="auto"
    "--device",
    "cpu",
]

PARTICLES = CommandSpec(
    key="particles",
    argv=["simulate", "particles"],
    baseline=_PARTICLE_BASELINE,
    config_cls=ParticleStackConfig,
    config_path="configs/particle.toml",
    flags=[
        # --- structure & potential -------------------------------------
        Flag("pdb_source", PDB_B),
        Flag(
            "assembly",
            None,
            expect="skip",
            note="needs a structure whose assembly expands the AU; every cached "
            "structure has AU == assembly, so this needs a network fetch (Phase 3)",
        ),
        Flag("n_pixels", "64"),
        Flag("pixel_size", "4.0"),
        Flag("scattering_factors", "kirkland"),
        Flag("scattering_factors", "lobato"),
        # A per-atom Debye-Waller damping; only shtyrov + analytic accept it,
        # so it must run WITHOUT the kirkland/lobato perturbations above.
        Flag("use_deposited_bfactors", "true"),
        # 2d needs per-element factors: Shtyrov's species-grouped potentials
        # are refused by it.
        Flag("potential_method", "2d", context=["--scattering_factors", "kirkland"]),
        Flag("potential_method", "3d"),
        Flag(
            "rcut", "8.0"
        ),  # 4.0 would quantize to the same ceil(rcut/dx) as the 3.5 default
        # Only the supersampled methods wrap; analytic + periodic is refused.
        # The box is shrunk to 48 A so the molecule's kernel tails reach its
        # faces: in the 144 A baseline box there is nothing to wrap.
        Flag(
            "periodic",
            "true",
            context=["--potential_method", "3d", "--n_pixels", "16"],
        ),
        Flag("pad_fft", "false"),
        # --- microscope -------------------------------------------------
        Flag("voltage", "200.0"),
        Flag("dose", "40.0"),
        Flag("cs", "2.7"),
        Flag("alpha", "0.07"),
        Flag(
            "objective_aperture",
            "12.0",
            context=["--absorption_model", "inelastic_mfp", "--alpha", "0.0"],
        ),
        Flag(
            "n_frames",
            "5",
            context=["--detector_model", "k3_300kv", "--coincidence_radius", "2.0"],
        ),
        Flag("convergence_angle", "1.0"),
        Flag("cc", "2.0"),
        Flag("energy_spread", "1.4", context=CC_ON),
        Flag("deltaV_V", "1e-5", context=CC_ON),
        Flag("deltaI_I", "1e-5", context=CC_ON),
        Flag("dose_envelope", "true"),
        Flag("dose_envelope_target", "specimen", context=["--dose_envelope", "true"]),
        Flag("ice_motion_variance", "0.38", context=["--ice_model", "gd"]),
        Flag("bfactor", "100.0"),
        Flag("absorption_model", "inelastic_mfp", context=["--alpha", "0.0"]),
        Flag("inelastic_mfp_solvent", "2000.0", context=MFP),
        Flag("inelastic_mfp_specimen", "2460.0", context=MFP),
        Flag("oversample", "1"),
        Flag("temporal_coherence", "envelope"),
        Flag(
            "dose_weights_path",
            None,
            expect="skip",
            note="needs a motion-correction weights fixture, Phase 3",
        ),
        Flag(
            "dose_weights_max_frequency",
            None,
            expect="skip",
            note="only read alongside dose_weights_path",
        ),
        Flag("project", None, expect="skip", note="job tracking: path, Phase 2"),
        Flag("job_id", None, expect="skip", note="job tracking: path, Phase 2"),
        Flag(
            "readd_hydrogens",
            None,
            expect="skip",
            note="adds hydrogens only through a Monomer Library, not installed here",
        ),
        Flag(
            "monomer_library_path",
            None,
            expect="skip",
            note="needs a Monomer Library, not installed here",
        ),
        # --- sampling ---------------------------------------------------
        Flag("defocus", "12000.0"),
        Flag("shift", "20.0"),
        Flag("n_particles", "3"),
        Flag("seed", "999"),
        # --- models -----------------------------------------------------
        Flag("scattering_model", "firstborn"),
        Flag("scattering_model", "projection"),
        Flag("scattering_model", "ctf"),
        Flag("noise_model", "none"),
        Flag("detector_model", "perfect"),
        Flag("detector_model", "k3_300kv"),
        Flag("detector_model", "k3_200kv"),
        Flag("detector_model", "falcon4i_300kv"),
        Flag("detector_model", "falcon4i_200kv"),
        Flag("detector_model", "falcon3ec_300kv"),
        Flag("coincidence_radius", "2.0", context=DETECTOR),
        Flag("detector_pixel_size", "0.5", context=DETECTOR),
        Flag("ews_curvature_sign", "negative"),
        Flag("klim", "0.10"),
        Flag("rotate_mode", "fourier"),
        # --- ice (cache-backed paths only) ------------------------------
        Flag("ice_model", "random"),
        Flag("ice_model", "none"),
        Flag("ice_thickness", "300.0"),
        Flag("water_air_interface", "true"),
        # --- crowding ---------------------------------------------------
        Flag("crowd_min_distance", "100.0"),
        Flag("crowd_max_distance_z", "50.0", context=CROWD),
        Flag("crowd_max_distance_xy", "200.0", context=CROWD),
        Flag("crowd_method", "2d", context=CROWD),
        Flag("crowd_n_points", "5", context=CROWD),
        Flag("crowd_seed", "random", context=CROWD),
        Flag(
            "crowd_chunk_size",
            "2",
            context=CROWD,
            expect="unchanged",
            note="performance knob: batching of crowding rotations",
        ),
        Flag(
            "crowd_move_to_cpu",
            "true",
            context=CROWD,
            expect="unchanged",
            note="performance knob: where crowding volumes live",
        ),
        # --- aberrations ------------------------------------------------
        Flag("astigmatism", "1000.0"),
        Flag("astigmatism_angle", "45.0", context=["--astigmatism", "1000.0"]),
        Flag("phaseshift", "1.0"),
        Flag("tiltx", "0.005"),
        Flag("tilty", "0.005"),
        Flag("trefoil1", "1e7"),
        Flag("trefoil2", "1e7"),
        Flag("tetrafoil1", "1e9"),
        Flag("tetrafoil2", "1e9"),
        Flag("tetrafoil3", "1e9"),
        Flag("tetrafoil4", "1e9"),
        Flag("anisomag_m00", "1.05"),
        Flag("anisomag_m01", "0.05"),
        Flag("anisomag_m10", "0.05"),
        Flag("anisomag_m11", "1.05"),
        # --- post-processing & output -----------------------------------
        Flag("normalize_particles", "false"),
        Flag("save_exitwaves", "true", expect="artifacts"),
        Flag("save_clean_exitwaves", "true", expect="artifacts"),
        # --- invariance checks ------------------------------------------
        Flag(
            "batchsize",
            "1",
            expect="unchanged",
            note="batching must not change physics",
        ),
        # --- covered elsewhere ------------------------------------------
        Flag("config", None, expect="skip", note="baseline mechanism"),
        Flag("device", None, expect="skip", note="GPU pass, Phase 3"),
        Flag("output_dir", None, expect="skip", note="path handling, Phase 2"),
        Flag("filename", None, expect="skip", note="path handling, Phase 2"),
        Flag("pdb_cache_dir", None, expect="skip", note="path handling, Phase 2"),
        Flag("cs_path", None, expect="skip", note="needs a .cs fixture, Phase 3"),
        Flag("star_path", None, expect="skip", note="needs a .star fixture, Phase 3"),
    ],
)

_MICROGRAPH_BASELINE = [
    "--seed",
    "1234",
    "--pdb_source",
    PDB_A,
    "--n_micrographs",
    "1",
    "--n_pixels",
    "48",
    "--micrograph_size",
    "128",
    "--pixel_size",
    "3.0",
    "--device",
    "cpu",
]

MICROGRAPH = CommandSpec(
    key="micrograph",
    argv=["simulate", "micrograph"],
    baseline=_MICROGRAPH_BASELINE,
    flags=[
        Flag("pdb_source", PDB_B),
        Flag(
            "assembly",
            None,
            expect="skip",
            note="needs a structure whose assembly expands the AU; every cached "
            "structure has AU == assembly, so this needs a network fetch (Phase 3)",
        ),
        Flag("n_pixels", "64"),
        Flag("pixel_size", "4.0"),
        Flag("micrograph_size", "160"),
        Flag("scattering_factors", "kirkland"),
        Flag("scattering_factors", "lobato"),
        # A per-atom Debye-Waller damping; only shtyrov + analytic accept it,
        # so it must run WITHOUT the kirkland/lobato perturbations above.
        Flag("use_deposited_bfactors", "true"),
        Flag("voltage", "200.0"),
        Flag("dose", "40.0"),
        Flag(
            "n_frames",
            "5",
            context=["--detector_model", "k3_300kv", "--coincidence_radius", "2.0"],
        ),
        Flag("cs", "2.7"),
        Flag("alpha", "0.07"),
        Flag("absorption_model", "inelastic_mfp"),
        Flag(
            "inelastic_mfp_solvent",
            "2000.0",
            context=["--absorption_model", "inelastic_mfp"],
        ),
        Flag(
            "inelastic_mfp_specimen",
            "2460.0",
            context=["--absorption_model", "inelastic_mfp"],
        ),
        Flag(
            "objective_aperture",
            "12.0",
            context=["--absorption_model", "inelastic_mfp"],
        ),
        Flag("dose_envelope_target", "specimen", context=["--dose_envelope", "true"]),
        Flag("ice_motion_variance", "0.38", context=["--ice_model", "gd"]),
        Flag("convergence_angle", "1.0"),
        Flag("cc", "2.0"),
        Flag("energy_spread", "1.4", context=CC_ON),
        Flag("deltaV_V", "1e-5", context=CC_ON),
        Flag("deltaI_I", "1e-5", context=CC_ON),
        Flag("dose_envelope", "true"),
        Flag("defocus", "12000.0"),
        Flag("n_micrographs", "2"),
        Flag("scattering_model", "firstborn"),
        Flag("scattering_model", "projection"),
        Flag("scattering_model", "ctf"),
        Flag("noise_model", "none"),
        Flag("coincidence_radius", "2.0", context=["--detector_model", "k3_300kv"]),
        Flag("ice_model", "random"),
        Flag("ice_model", "none"),
        Flag("ice_thickness", "300.0"),
        Flag("ice_profile", "wedge", context=["--ice_thickness_range", "200,700"]),
        # The context sets only the radius, which a flat profile ignores; a
        # context holding --ice_profile meniscus itself would make the flag's
        # run identical to its reference by construction.
        Flag("ice_profile", "meniscus", context=["--ice_hole_radius", "3000.0"]),
        Flag(
            "ice_thickness_range",
            "200,700",
            context=["--ice_profile", "wedge"],
        ),
        Flag(
            "ice_profile_angle",
            "90.0",
            context=["--ice_profile", "wedge", "--ice_thickness_range", "200,700"],
        ),
        Flag("ice_hole_radius", "1500.0", context=MENISCUS),
        Flag("ice_rim_thickness", "1200.0", context=MENISCUS),
        Flag("ice_hole_offset", "2000,0", context=MENISCUS),
        Flag("ice_tilt", "0.2"),
        Flag("crowd_min_distance", "100.0"),
        Flag("crowd_max_distance_z", "50.0", context=CROWD),
        Flag("water_air_interface", "true"),
        Flag("pad_fft", "false"),
        Flag("detector_model", "perfect"),
        Flag("detector_model", "k3_300kv"),
        Flag("normalize_micrographs", "false"),
        Flag("save_exitwaves", "true", expect="artifacts"),
        Flag("save_clean_exitwaves", "true", expect="artifacts"),
        Flag("seed", "999"),
        Flag(
            "crowd_chunk_size",
            "2",
            expect="unchanged",
            note="performance knob: specimen assembly chunking",
        ),
        Flag("config", None, expect="skip", note="baseline mechanism"),
        Flag("device", None, expect="skip", note="GPU pass, Phase 3"),
        Flag("output_dir", None, expect="skip", note="path handling, Phase 2"),
        Flag("filename", None, expect="skip", note="path handling, Phase 2"),
        Flag("pdb_cache_dir", None, expect="skip", note="path handling, Phase 2"),
        Flag("bfactor", "100.0"),
        Flag("klim", "0.10"),
        Flag("oversample", "1"),
        Flag("temporal_coherence", "envelope"),
        # The shape packer and its knobs; each knob only acts under it.
        Flag("packing_backend", "shape"),
        Flag("packing_gap", "20.0", context=SHAPE),
        # A retry ceiling only binds once placements start failing, which a
        # large candidate pool forces in this small field.
        Flag("packing_max_retries", "1", context=[*SHAPE, "--n_candidates", "400"]),
        Flag("packing_seed", "7", context=SHAPE),
        Flag("packing_stall_patience", "1", context=SHAPE),
        Flag("n_orientations", "4", context=SHAPE),
        Flag("n_candidates", "8", context=SHAPE),
        # The water-air interface bias, on by default.
        Flag("sigma_frac", "0.3"),
        Flag("peak_amplitude", "0.2"),
        Flag("baseline", "0.9"),
        Flag("project", None, expect="skip", note="job tracking: path, Phase 2"),
        Flag("job_id", None, expect="skip", note="job tracking: path, Phase 2"),
        Flag(
            "readd_hydrogens",
            None,
            expect="skip",
            note="adds hydrogens only through a Monomer Library, not installed here",
        ),
        Flag(
            "monomer_library_path",
            None,
            expect="skip",
            note="needs a Monomer Library, not installed here",
        ),
    ],
    timeout=1800.0,
    config_cls=MicrographConfig,
    config_path="configs/micrograph.toml",
)

# Built once by `specter build tomogram` with tools/cli-qa/qa_tomogram.toml.
# `specter simulate tiltseries` needs a specimen volume to image. Built once by
# `tools/cli-qa/sweep.py tomogram` (or by hand from qa_tomogram.toml) and left where the
# tomogram sweep writes it; $SPECTER_QA_VOLUME overrides.
_QA_RUNS = (
    Path(os.environ["SPECTER_QA_WORKDIR"])
    if os.environ.get("SPECTER_QA_WORKDIR")
    else Path(tempfile.gettempdir()) / "specter-qa-runs"
)
QA_VOLUME = os.environ.get("SPECTER_QA_VOLUME", "") or str(
    _QA_RUNS / "tomogram" / "baseline_a" / "out.mrc"
)

_TILTSERIES_BASELINE = [
    "--seed",
    "1234",
    "--volume_path",
    QA_VOLUME,
    "--micrograph_size",
    "96",
    "--n_tilts",
    "3",
    "--voxel_size",
    "6.0",
    "--device",
    "cpu",
]

TILTSERIES = CommandSpec(
    key="tiltseries",
    argv=["simulate", "tiltseries"],
    baseline=_TILTSERIES_BASELINE,
    flags=[
        Flag("voxel_size", "8.0"),
        Flag("micrograph_size", "128"),
        Flag("voltage", "200.0"),
        Flag("dose_per_tilt", "5.0"),
        Flag(
            "n_frames",
            "5",
            context=["--detector_model", "k3_300kv", "--coincidence_radius", "2.0"],
        ),
        Flag("cs", "2.7"),
        Flag("alpha", "0.07"),
        Flag("absorption_model", "inelastic_mfp"),
        Flag(
            "inelastic_mfp_solvent",
            "2000.0",
            context=["--absorption_model", "inelastic_mfp"],
        ),
        Flag(
            "inelastic_mfp_specimen",
            "2460.0",
            context=["--absorption_model", "inelastic_mfp"],
        ),
        Flag(
            "objective_aperture",
            "12.0",
            context=["--absorption_model", "inelastic_mfp"],
        ),
        Flag("dose_envelope_target", "specimen", context=["--dose_envelope", "true"]),
        Flag("ice_motion_variance", "0.38", context=["--ice_model", "random"]),
        Flag("convergence_angle", "1.0"),
        # Noiseless and finer than the baseline: see TILT_FINE.
        # 2.0 against the 2.7 default moves the image 3.8e-3, under tolerance.
        Flag("cc", "5.0", context=TILT_FINE),
        Flag("energy_spread", "1.4", context=CC_ON + TILT_FINE),
        Flag("deltaV_V", "1e-5", context=CC_ON),
        Flag("deltaI_I", "1e-5", context=CC_ON),
        Flag("dose_envelope", "true"),
        Flag("defocus", "12000.0"),
        Flag("min_tilt_angle", "-30.0"),
        Flag("max_tilt_angle", "30.0"),
        Flag("n_tilts", "5", expect="changes"),
        Flag("tilt_axis", "x"),
        Flag("scattering_model", "firstborn"),
        Flag("scattering_model", "projection"),
        Flag("scattering_model", "ctf"),
        Flag("noise_model", "none"),
        Flag("coincidence_radius", "2.0", context=["--detector_model", "k3_300kv"]),
        Flag("ice_model", "random"),
        Flag("ice_model", "none"),
        Flag("pad_fft", "false"),
        Flag("detector_model", "perfect"),
        Flag("detector_model", "k3_300kv"),
        Flag("normalize_tilt_series", "false"),
        Flag("seed", "999"),
        Flag("save_exitwaves", "true", expect="artifacts"),
        Flag("bfactor", "100.0"),
        Flag("klim", "0.10"),
        Flag("oversample", "1"),
        # The envelope and the integrated sum part at 120 kV, where the focal
        # spread is widest.
        Flag(
            "temporal_coherence",
            "envelope",
            context=[*TILT_FINE, "--voltage", "120.0"],
        ),
        Flag("project", None, expect="skip", note="job tracking: path, Phase 2"),
        Flag("job_id", None, expect="skip", note="job tracking: path, Phase 2"),
        Flag("config", None, expect="skip", note="baseline mechanism"),
        Flag("tomogram_config", None, expect="skip", note="chained run, Phase 3"),
        Flag("volume_path", None, expect="skip", note="set by the baseline"),
        Flag("device", None, expect="skip", note="GPU pass, Phase 3"),
        Flag("output_dir", None, expect="skip", note="path handling, Phase 2"),
        Flag("filename", None, expect="skip", note="path handling, Phase 2"),
    ],
    timeout=1800.0,
    config_cls=TiltSeriesConfig,
    config_path="configs/tiltseries.toml",
)

_TOMOGRAM_BASELINE = [
    "--config",
    "tools/cli-qa/qa_tomogram.toml",
    "--n_tomograms",
    "1",
    "--voxel_size",
    "12.0",
    "--seed",
    "7",
    "--device",
    "cpu",
]

TOMOGRAM = CommandSpec(
    key="tomogram",
    argv=["build", "tomogram"],
    baseline=_TOMOGRAM_BASELINE,
    flags=[
        Flag("voxel_size", "16.0"),
        Flag("seed", "8"),
        Flag("filler_from_pei2016", "false"),
        Flag(
            "filler_pei2016_abundance_weighting",
            "true",
            context=["--filler_from_pei2016", "true"],
        ),
        Flag("filler_from_cryoetsim", "true"),
        # The mass cutoffs act on the reference tables, which qa_tomogram.toml
        # switches off; at 20 -> 250 kDa the PEI2016 population drops 27 -> 19.
        Flag(
            "filler_table_max_mw_kda",
            "100.0",
            context=["--filler_from_pei2016", "true"],
        ),
        Flag(
            "filler_table_min_mw_kda",
            "250.0",
            context=["--filler_from_pei2016", "true"],
        ),
        Flag("filler_occupancy_fraction", "0.02"),
        Flag("membrane_region_density_threshold", "0.3"),
        Flag(
            "membrane_region_max_passes",
            None,
            expect="skip",
            note="gold-bead placement budget; binds only once beads collide, "
            "which this volume's few beads never do",
        ),
        Flag(
            "membrane_min_transmembrane_spacing",
            None,
            expect="skip",
            note="binds at ~60 transmembrane copies per membrane (28.6% of "
            "voxels at spacing 40 vs 120, FINDINGS); the copy count is TOML-only",
        ),
        Flag("scattering_factors", "kirkland"),
        # A per-atom Debye-Waller damping; only shtyrov + analytic accept it,
        # so it must run WITHOUT the kirkland/lobato perturbations above.
        Flag("use_deposited_bfactors", "true"),
        Flag("actin", "true"),
        Flag("bead_roughness", "0.3"),
        Flag("write_picks", "false", expect="artifacts"),
        Flag("annotation_version", "2.0", expect="metadata"),
        Flag("write_segmentation", "true", expect="artifacts"),
        Flag("n_tomograms", "2", expect="artifacts"),
        Flag(
            "render_chunk_size",
            "2",
            expect="unchanged",
            note="performance knob: render chunking",
        ),
        Flag(
            "render_workers",
            "2",
            expect="unchanged",
            note="performance knob: parallel species rendering",
        ),
        Flag(
            "accumulator_device",
            "cpu",
            expect="unchanged",
            note="placement knob: where the accumulator lives",
        ),
        # One trial position per instance instead of the 1500 default: at 20
        # every instance of this sparse baseline still found a place.
        Flag("packing_max_retries", "1"),
        # The occupancy grid the shape packer collides on, coarser than its
        # auto-estimated size: a different grid admits different placements.
        Flag("packing_voxel_size", "40.0"),
        Flag(
            "readd_hydrogens",
            None,
            expect="skip",
            note="adds hydrogens only through a Monomer Library, not installed here",
        ),
        Flag(
            "monomer_library_path",
            None,
            expect="skip",
            note="needs a Monomer Library, not installed here",
        ),
        Flag("project", None, expect="skip", note="job tracking: path, Phase 2"),
        Flag("job_id", None, expect="skip", note="job tracking: path, Phase 2"),
        Flag("config", None, expect="skip", note="baseline mechanism"),
        Flag("device", None, expect="skip", note="GPU pass, Phase 3"),
        Flag("output_dir", None, expect="skip", note="path handling, Phase 2"),
        Flag("filename", None, expect="skip", note="path handling, Phase 2"),
        Flag("pdb_cache_dir", None, expect="skip", note="path handling, Phase 2"),
    ],
    timeout=3600.0,
    config_cls=TomogramConfig,
    config_path="tools/cli-qa/qa_tomogram.toml",
)

SPECS = {s.key: s for s in (PARTICLES, MICROGRAPH, TILTSERIES, TOMOGRAM)}
