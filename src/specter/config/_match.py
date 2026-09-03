"""MatchConfig: parameters for `specter match particles`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ._paths import default_pdb_cache_dir


@dataclass
class MatchConfig:
    """Parameters for deriving a simulation config that matches a real particle set.

    ``specter match particles`` takes a refined particle set, its images and
    the atomic model, derives every simulation parameter the forward model has
    from the metadata and from probe simulations against the images, and
    writes a ``matched.toml`` that ``specter simulate particles`` runs as is,
    together with a report of how close the match is and what, if anything,
    no parameter can close.

    Most of what a simulation needs comes from the refinement file itself
    (voltage, Cs, amplitude contrast, pixel size, per-particle defocus and
    poses). What no file carries is the acquisition card below: the detector,
    the total dose, the dose rate, and whether an energy filter was used.
    Those are in every methods section and every EMDB record.
    """

    # --- Inputs (required) ---
    metadata_path: str  # CryoSPARC .cs (passthrough) or RELION .star
    pdb_source: str
    dose: float  # e-/A^2 per movie

    # --- Inputs (optional) ---
    # Particle images in the same order as `metadata_path`. Unset reads them
    # from the paths the metadata file itself points at (`blob/path` for a
    # .cs, `rlnImageName` for a .star), resolved against the file's own
    # project directory.
    images_path: str | None = None
    assembly: bool = True

    # --- Acquisition card ---
    # "unknown" applies no MTF, no DQE(0) and no coincidence loss, and says so
    # in the report; it is never silently replaced by a similar model.
    detector_model: Literal[
        "unknown",
        "perfect",
        "k3_300kv",
        "k3_200kv",
        "k2_300kv",
        "falcon4i_300kv",
        "falcon4i_200kv",
    ] = "unknown"
    dose_rate: float | None = None  # e-/physical px/s; unset -> detector's typical rate
    energy_filter: bool | None = None  # None = not stated

    # --- Probing ---
    n_probe: int = 100  # particles per probe simulation
    n_battery: int = 200  # particles per seed in the final comparison
    # Candidate grids, config-only (a flag cannot carry a list). Ice thickness
    # in Angstrom (0 = the box minimum); neighbour spacing as a multiple of the
    # structure's maximum diameter (0 = no neighbours).
    ice_candidates: list[float] = field(
        default_factory=lambda: [0.0, 400.0, 800.0, 1200.0]
    )
    crowd_candidates: list[float] = field(default_factory=lambda: [0.0, 1.0, 1.3])
    # Number of particles to simulate with the matched config after the
    # report, e.g. for a CryoSPARC mixed classification. 0 skips it.
    write_stack: int = 0

    # --- Compute ---
    device: str = "cuda"
    seed: int | None = None

    # --- Output & job tracking ---
    output_dir: str | None = None
    project: str | None = None
    job_id: str | None = None

    # --- Advanced ---
    pdb_cache_dir: str = field(default_factory=default_pdb_cache_dir)
    monomer_library_path: str | None = None
    n_frames: int = 40  # frames the simulation splits the dose into


MATCH_HELP: dict[str, str] = {
    "metadata_path": "Refined particle set: a CryoSPARC passthrough .cs or a "
    "RELION .star. Supplies voltage, Cs, amplitude contrast, pixel size, "
    "per-particle defocus and poses. The refinement must be aligned to the "
    "atomic model (an Align 3D of its map against the model, then re-extract "
    "the particles from it); the pose-alignment check fails otherwise.",
    "pdb_source": "Atomic model matching the particles: a local .cif/.pdb "
    "file or a 4-character PDB accession code.",
    "dose": "Total electron dose per movie in e-/Angstrom^2, from the methods "
    "section or the EMDB record. Drives the radiation-damage envelope.",
    "images_path": "Particle image stack (.mrcs) in the same order as "
    "--metadata_path. Unset reads the images the metadata file points at.",
    "assembly": "Fetch the biological assembly of --pdb_source.",
    "detector_model": "Detector the data were recorded on. Supplies the MTF, "
    "DQE(0), the coincidence exclusion radius and the hardware frame rate. "
    "'unknown' applies none of them and is reported as such.",
    "dose_rate": "Incident dose rate in electrons per physical pixel per "
    "second, from the acquisition notes. Sets the coincidence-loss occupancy. "
    "Unset falls back to the detector's typical operating rate, and the report "
    "says so.",
    "energy_filter": "Whether an energy filter (slit) was used. Recorded in "
    "the report: on today's evidence unfiltered data carry a residual the "
    "forward model cannot express.",
    "n_probe": "Particles per probe simulation (ice thickness and neighbour "
    "spacing candidates).",
    "n_battery": "Particles per seed in the final two-seed comparison that "
    "the report is computed from.",
    "write_stack": "After the report, simulate this many particles with the "
    "matched config (e.g. for a mixed 2D classification). 0 skips it.",
    "device": "Device to use: cpu | cuda | cuda:0.",
    "seed": "RNG seed for the probe and battery simulations.",
    "output_dir": "Directory to write matched.toml and the report under when "
    "untracked; the root of the numbered job tree when --project or --job_id "
    "is set.",
    "project": "Optional: number and track this run through specter.jobs, "
    "under <output_dir>/[<project>/]match/J00N/.",
    "job_id": "Pin the job directory (e.g. J001) rather than auto-assigning "
    "the next one.",
    "pdb_cache_dir": "Where downloaded PDB/mmCIF structures are cached.",
    "monomer_library_path": "Path to a Monomer Library, so Shtyrov species "
    "resolve for a hydrogen-free deposition. Unset falls back to $CLIBD_MON.",
    "n_frames": "Frames the simulation splits the dose into. Only the "
    "coincidence radius depends on it, and the derived radius is converted to "
    "this frame count.",
}
