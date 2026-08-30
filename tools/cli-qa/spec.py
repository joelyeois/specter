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

# Structures kept local so the sweep never depends on network fetches.
PDB_A = "~/.cache/specter/pdb/1mbo.cif"
PDB_B = "~/.cache/specter/pdb/1A6M.cif"

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
        Flag("potential_method", "2d"),
        Flag("potential_method", "3d"),
        Flag(
            "rcut", "8.0"
        ),  # 4.0 would quantize to the same ceil(rcut/dx) as the 3.5 default
        Flag("periodic", "true"),
        Flag("potential_scale", "0.5"),
        Flag("pad_fft", "false"),
        # --- microscope -------------------------------------------------
        Flag("voltage", "200.0"),
        Flag("dose", "40.0"),
        Flag("cs", "2.7"),
        Flag("alpha", "0.07"),
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
        Flag("bfactor", "100.0"),
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
        Flag("coincidence_radius", "2.0", context=DETECTOR),
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
        Flag("voltage", "200.0"),
        Flag("dose", "40.0"),
        Flag(
            "n_frames",
            "5",
            context=["--detector_model", "k3_300kv", "--coincidence_radius", "2.0"],
        ),
        Flag("cs", "2.7"),
        Flag("alpha", "0.07"),
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
        Flag("ice_profile", "meniscus", context=MENISCUS),
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
        Flag("ice_hole_radius", "3000.0", context=MENISCUS),
        Flag("ice_rim_thickness", "1200.0", context=MENISCUS),
        Flag("ice_hole_offset", "2000,0", context=MENISCUS),
        Flag("ice_tilt", "0.2"),
        Flag("crowd_min_distance", "100.0"),
        Flag("crowd_max_distance_z", "50.0", context=CROWD),
        Flag("water_air_interface", "true"),
        Flag("potential_scale", "0.5"),
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
    ],
    timeout=1800.0,
    config_cls=MicrographConfig,
    config_path="configs/micrograph.toml",
)

# Built once by `specter build tomogram` with tools/cli-qa/qa_tomogram.toml.
# `specter simulate tiltseries` needs a specimen volume to image. Built once by
# `tools/cli-qa/sweep.py tomogram` (or by hand from qa_tomogram.toml) and left where the
# tomogram sweep writes it; $SPECTER_QA_VOLUME overrides.
QA_VOLUME = os.environ.get("SPECTER_QA_VOLUME", "") or str(
    Path(os.environ.get("SPECTER_QA_WORKDIR", ""))
    if os.environ.get("SPECTER_QA_WORKDIR")
    else Path(tempfile.gettempdir())
    / "specter-qa-runs"
    / "tomogram"
    / "baseline_a"
    / "out.mrc"
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
        Flag("convergence_angle", "1.0"),
        Flag("cc", "2.0"),
        Flag("energy_spread", "1.4", context=CC_ON),
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
        Flag("filler_from_cryoetsim", "true"),
        Flag("filler_table_max_mw_kda", "300.0"),
        Flag("filler_table_min_mw_kda", "80.0"),
        Flag("filler_occupancy_fraction", "0.02"),
        Flag("membrane_region_density_threshold", "0.3"),
        Flag("membrane_region_max_passes", "2"),
        Flag("membrane_min_transmembrane_spacing", "80.0"),
        Flag("scattering_factors", "kirkland"),
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
