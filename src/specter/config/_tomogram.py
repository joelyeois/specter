"""TomogramConfig: parameters for tomogram specimen generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ._paths import default_pdb_cache_dir
from ._scalar_range import ScalarOrRange


@dataclass
class TomogramConfig:
    """Parameters for tomogram specimen generation, loaded from a TOML
    config file.

    Drives `specter.specimen.tomogram.TomogramSpecimenGenerator`, the ONE
    generator behind `specter build tomogram` -- an optional composited
    organic membrane (`membrane`), optional scattered filament species
    (`filaments`/`actin`), and densely packed protein species
    (`targets`/`filler`, region-gated to `location: "cytosol"|"lumen"` when
    a membrane is present, otherwise everywhere is "cytosol" -- see
    `TomogramSpecimenGenerator`'s own module docstring). Generation order
    is membranes, then filaments, then protein fill; each stage avoids the
    previous ones' placements. Renders every placed instance's real
    scattering potential (always Shtyrov-parameterized, `PotentialBuilder`'s
    own default) and saves the assembled volume as .mrc (directly usable as
    `TiltSeriesConfig.volume_path`) plus one copick-style .ndjson pick file
    per species.

    Any combination of membrane/filaments/targets/filler is valid as long
    as at least one is non-empty -- there is no longer a separate non-
    membrane "sphere-packing" mode/generator to choose between.
    """

    # --- Specimen ---
    # One dict per target species, e.g. {"pdb_source": "6qzp", "n_copies":
    # 15}. "pdb_source" and "n_copies" required; "location" optional
    # ("cytosol"|"lumen", default "cytosol" -- only meaningful when
    # `membrane` is set, otherwise there's no "lumen" to place into). Placed
    # FIRST within its location, at this exact count, always exported to
    # picks. In TOML, provide as [[targets]] tables.
    targets: list[dict[str, Any]] = field(default_factory=list)
    # One dict per filler species, e.g. {"pdb_source": "1mbo"}. Only
    # "pdb_source" is required; optional "location" (as `targets` above,
    # default "cytosol") and "ratio" (relative abundance weight among OTHER
    # filler species sharing the same location, default 1.0 -- equal
    # attempt-weight across species if left at the default for all of
    # them). Placed SECOND within its location, around any already-placed
    # targets there, budgeted by filler_occupancy_fraction. Excluded from
    # picks by default (see write_picks/TomogramProteinSpec.role). In TOML,
    # provide as [[filler]] tables.
    filler: list[dict[str, Any]] = field(default_factory=list)
    # Additive to filler above: pull extra filler species from the
    # bundled reference tables (specter.specimen.cytosolic_filler.
    # PEI2016_CROWDING_TABLE and/or specter.specimen.cytosolic_filler.
    # CRYOETSIM_PARTICLE_TABLE), rather than hand-listing every species.
    # Both can be enabled at once -- their results are concatenated. Always
    # placed at location="cytosol" (these tables have no lumen/cytosol
    # distinction of their own).
    filler_from_pei2016: bool = False
    filler_from_cryoetsim: bool = False
    # Only affects filler_from_cryoetsim (CRYOETSIM_PARTICLE_TABLE has a
    # "category" column; PEI2016_CROWDING_TABLE doesn't, so this filter
    # has no effect there). None = all 4 usable categories (macromolecules,
    # distractors, transcription_translation, nucleosomes).
    filler_table_categories: list[str] | None = None
    # Mass range applied to whichever table(s) above are enabled.
    filler_table_max_mw_kda: float | None = None
    filler_table_min_mw_kda: float | None = None
    target_shape: list[int] = field(
        default_factory=lambda: [300, 1200, 1200]
    )  # (Z, Y, X) voxels
    voxel_size: float = 5.0  # Å/voxel
    # Target packing density for `ratio`-mode filler species, as a bare-
    # sphere fraction of EACH REGION's own volume it's placed in (the whole
    # box when `membrane` is empty, since then "cytosol" IS the whole box --
    # see TomogramSpecimenGenerator's own occupancy_fraction docstring).
    # Deliberately high by default -- RSA self-limits at its own physical
    # jamming ceiling rather than erroring, so filler simply packs until it
    # jams rather than needing this hand-tuned. Lower it for a deliberately
    # sparser filler layer, or if a small region (e.g. a tight vesicle
    # lumen) makes the candidate pool this implies impractically large.
    filler_occupancy_fraction: float = 0.5
    # Collision geometry for protein packing: "sphere" collides one
    # circumscribing sphere per instance, "shape" collides the real rotated
    # footprint against a running occupancy grid (what CryoTomoSim does).
    # See TomogramSpecimenGenerator's own packing_backend docstring.
    packing_backend: str = "shape"
    packing_max_retries: int = 1500
    # None = auto (coarsen only when the packing grid gets too large).
    packing_voxel_size: float | None = None
    # (z, y, x), matching target_shape's axis order. True on an axis lets a
    # placed instance's center stay in-bounds while its body pokes past
    # that wall (truncated naturally at render time) instead of being
    # rejected outright -- e.g. for a tomogram whose xy field of view is a
    # crop of a larger cellular region.
    clip_axes: list[bool] = field(default_factory=lambda: [False, False, False])
    # Relative to the current working directory, like any other CLI path
    # argument -- see default_pdb_cache_dir for the unset case.
    pdb_cache_dir: str = field(default_factory=default_pdb_cache_dir)
    seed: int | None = None

    # --- Organic membrane (optional) ---
    # One or more dicts, [[membrane]] tables -- one membrane TEMPLATE each,
    # composited into the shared tomogram (see specter.specimen.tomogram.
    # MembraneInstance). Empty (default): no membrane at all -- the whole
    # tomogram is then one cytosol region. Keys are passed as **kwargs
    # straight into specter.specimen.membrane.MembraneGenerator -- e.g.
    # {"shape_backend": "spherical_harmonics", "sh_axes": [300.0, 300.0,
    # 300.0], "sh_amplitude": 0.15, "bilayer_thickness": 30.0} -- PLUS
    # three keys not real MembraneGenerator kwargs, popped before that call:
    #   - "n_copies" (int, default 1): expands this one entry into that
    #     many independent instances sharing the same template, each its
    #     own seed (config.seed + i, restarting at i=0 per entry -- editing/
    #     adding another [[membrane]] entry never perturbs an earlier
    #     entry's own instances). Every instance's position is resolved via
    #     collision-rejecting random placement against every other instance
    #     (see TomogramSpecimenGenerator's own docstring) -- an instance
    #     that doesn't fit is dropped, not retried. There is no manual-
    #     placement override.
    #   - "target_shape" = [Z, Y, X] voxels. Default omitted (None):
    #     MembraneGenerator auto-sizes a small local working grid from the
    #     organelle's own size (see its own docstring) instead of every
    #     instance rendering on a grid the size of the WHOLE tomogram
    #     canvas. Give it explicitly only if this instance genuinely needs
    #     a specific working-grid size.
    # voxel_size/seed/device/pdb_cache_dir still come from this config's own
    # voxel_size/seed/device/pdb_cache_dir fields for every instance, not from
    # this dict (shape_backend one of "spherical_harmonics" (default) or
    # "swept_spline").
    membrane: list[dict[str, Any]] = field(default_factory=list)
    # Each {"pdb_source": <code or path>, "n_copies": 1, "parameterization":
    # "shtyrov"}. In TOML, provide as [[membrane_transmembrane_specs]] tables.
    # "n_copies" is per membrane instance, and is a request rather than a
    # guarantee -- MembraneGenerator.place_transmembrane warns and places
    # fewer if the surface can't fit that many at
    # membrane_min_transmembrane_spacing.
    # Applies across ALL membrane instances (not per-instance in v1). Only
    # meaningful when `membrane` is set (no bilayer to embed into otherwise).
    membrane_transmembrane_specs: list[dict[str, Any]] = field(default_factory=list)
    membrane_region_density_threshold: float | None = None
    membrane_region_max_passes: int = 300
    membrane_min_transmembrane_spacing: float = 40.0
    # The parameterization for EVERYTHING this command renders from atoms:
    # targets, filler, filaments, microtubules, the carbon film, the bilayer
    # and its transmembrane proteins. Everything placed lands in one summed
    # volume, so modelling parts of it with different scattering factors is a
    # choice to make deliberately -- a [[membrane]] table naming its own
    # "parameterization" still overrides this for that population (see
    # run_build_tomogram), but nothing does so by default. The single
    # exception is gold fiducials: Shtyrov fits per bonded species and has no
    # elemental gold, so beads fall back to Peng, the same per-element
    # fallback PotentialBuilder uses (see TomogramSpecimenGenerator._stamp_beads). Spelled the same as
    # ParticleStackConfig.scattering_factors -- one config vocabulary across
    # commands, distinct from the internal `parameterization=` kwarg it feeds.
    scattering_factors: str = "shtyrov"
    # Forwarded to every PDB built for this tomogram. Only takes effect for
    # scattering_factors="shtyrov" (the only one that types atoms) and
    # only when a Monomer Library is available via $CLIBD_MON.
    readd_hydrogens: bool | Literal["auto"] = "auto"
    # Unset falls back to $CLIBD_MON. A field as well as a variable because
    # the library changes the rendered potential, so a run should be
    # reproducible from its own config -- see ParticleStackConfig's own note.
    monomer_library_path: str | None = None

    # --- Filaments (optional, additive on top of membranes if present) ---
    # One dict per filament species, mapping straight onto
    # specter.specimen.filament.FilamentSpec's own kwargs, e.g.
    # {"code": "1TUB", "step": 85.0, "flex_deg": 3.0, "n_copies": 4}.
    # Placed via specter.specimen.filament.place_filaments -- specter-native
    # random-walk placement, with no region-gating and no collision
    # avoidance against the membrane shell or each other, but DOES get
    # avoided by targets/filler packing (placed right after membranes,
    # before protein fill -- see TomogramSpecimenGenerator's own
    # docstring). In TOML, provide as [[filaments]] tables.
    filaments: list[dict[str, Any]] = field(default_factory=list)
    # Convenience toggle: also place the bundled ACTIN_SPEC preset (real
    # F-actin helical repeat -- step/twist from Holmes/Egelman) without
    # hand-writing a [[filaments]] entry. Additive to filaments above (both
    # may be set at once). ACTIN_SPEC's own n_copies default (1) applies
    # here too -- for more instances or other single-strand species, use
    # [[filaments]] instead. Microtubules are NOT a filament species: they
    # are whole tubes, see [[microtubules]] below.
    actin: bool = False

    # --- Microtubules (optional, additive on top of filaments/membranes) ---
    # One dict per microtubule species, mapping straight onto
    # specter.specimen.filament.MicrotubuleSpec's own kwargs, e.g.
    # {"n_copies": 2, "n_protofilaments": 13, "bend_radius": 3e4}. Real
    # 13-protofilament tubes (lumen, A-lattice seam, tubulin dimer wall),
    # not the single protofilament PROTOFILAMENT_SPEC gives. Placed in the
    # same phase as filaments -- no region-gating, no collision avoidance,
    # but avoided by targets/filler afterwards. `code` defaults to a dimer
    # extracted from a deposited microtubule reconstruction (fetched and
    # cached on first use); `length` defaults to the volume diagonal so a
    # microtubule crosses the field. In TOML, provide as [[microtubules]]
    # tables.
    microtubules: list[dict[str, Any]] = field(default_factory=list)

    # --- Carbon support film (optional, single film) ---
    # Zero or one [[carbon_film]] table, mapping onto
    # specter.specimen.CarbonFilmSpec's own kwargs (thickness, hole_radius,
    # edge_fraction, edge_side, edge_roughness) -- e.g.
    # {"hole_radius": 6000.0, "edge_fraction": [0.02, 0.05]}. Painted
    # directly into the volume before anything else is placed; placement
    # (membranes/targets/filler) is NOT carbon-aware (a documented,
    # CTS-parity limitation -- see TomogramSpecimenGenerator's own
    # docstring). More than one entry raises. Empty (default): no carbon
    # film, pure ice.
    carbon_film: list[dict[str, Any]] = field(default_factory=list)

    # --- Gold fiducial beads (optional) ---
    # One dict per bead population, {"radius": <Angstrom or [low, high]>,
    # "n_copies": 1}. "radius" is required; a [low, high] pair draws a
    # fresh radius per instance, giving the population size dispersity
    # real colloidal gold has (see specter.specimen.TomogramBeadSpec).
    # Placed via the same RSA packing used for membranes/targets/filler,
    # avoiding the membrane shell and any already-placed filaments -- NOT
    # region-gated to cytosol/lumen (see TomogramBeadSpec's own
    # docstring). In TOML, provide as [[beads]] tables.
    beads: list[dict[str, Any]] = field(default_factory=list)

    # How irregular each fiducial's boundary is, as an RMS fraction of its
    # radius -- see specter.specimen._grid's BeadGenerator. Either one
    # number or a [low, high] pair drawn per bead. 0.0 gives clean
    # spheres.
    bead_roughness: ScalarOrRange = 0.12

    # --- Ground-truth picks & segmentation ---
    write_picks: bool = True
    annotation_version: str = "1.0"
    # Save per-instance integer label volumes as .mrc alongside the density
    # volume ({filename}_protein_labels.mrc always; membrane mode also gets
    # {filename}_membrane_labels.mrc and {filename}_regions.mrc (0=cytosol/
    # 1=shell/2=lumen)) -- cast to uint16 (MRC has no int32 mode). The
    # segmentation mask, not a coordinate file, is the intended ground truth
    # for membrane geometry (a membrane surface has no single natural
    # "position" the way a protein does).
    write_segmentation: bool = True

    # --- Compute ---
    # "cpu" | "cuda" | "cuda:0" | a bare GPU index ("0") | a comma-separated
    # list of GPU indices ("0,1,2") to pool multiple GPUs for concurrent
    # per-species rendering (see render_workers below) | "auto" to pool
    # every visible CUDA GPU. A pooled value's first entry becomes the
    # primary device for everything else (see
    # specter.specimen._parallel_render.parse_device_pool).
    device: str = "cuda"  # falls back to CPU when none is available
    # Device for the shared canvas tensors (volume/instance_labels/
    # membrane_labels), decoupled from `device` above (which stays the
    # compute device for rendering/rotation/field-generation regardless).
    # None (default): same as `device` -- one device for everything, the
    # original behaviour. "auto": estimate the canvas' own memory
    # footprint from target_shape/voxel_size and fall back to "cpu" if it
    # would exceed half of `device`'s currently free memory (see
    # TomogramSpecimenGenerator.recommend_accumulator_device's own
    # docstring). Explicit "cpu" always works regardless of that
    # estimate, keeping `device`="cuda" (fast rendering/rotation) while
    # letting the canvas itself be sized by system RAM instead of GPU
    # VRAM -- e.g. a large field of view at fine voxel_size can need tens of
    # GB, past any single GPU's VRAM but often fine in system RAM on a
    # workstation/cluster node. Rotation/rendering speed is unaffected
    # either way; the only added cost is moving each already-small
    # rotated chunk across devices once, not the canvas itself.
    accumulator_device: str | None = None
    # How many PDB species render/fetch concurrently within a single
    # tomogram: membrane_transmembrane_specs (rendered once, shared across
    # every [[membrane]] entry/n_copies copy, membrane mode only) and
    # targets/filler (cytosol/lumen protein-fill, rendered + PDB-fetched
    # once per tomogram, always) -- see MembraneGenerator/
    # TomogramSpecimenGenerator's own render_workers docstrings. Default 1:
    # fully serial, identical to the original behaviour. "auto" resolves
    # per-pool via specter.specimen._parallel_render.
    # recommend_render_workers -- min(n_species, 8), the measured sweet spot
    # from a full production-scale sweep (see that function's own
    # docstring); recommended over hand-picking a number.
    render_workers: int | Literal["auto"] = 1
    # Instances rotated per GPU batch, per species, in the targets/filler
    # protein-fill stage (TomogramSpecimenGenerator's own chunk_size
    # constructor kwarg -- rotate_volume batches ALL of a species' accepted
    # instances into one call when this is None, the original behaviour).
    # Fine at small scale, but a species with hundreds of instances (a real
    # filler species count at production target_shape/occupancy_fraction) can
    # then need many GB for one rotation call alone -- confirmed directly:
    # an 8.7 GB single allocation from one such batch, on top of whatever
    # else was already resident, was what actually tipped a production-
    # scale run into a CUDA OOM. None (default) preserves the original,
    # small-scale-safe behaviour; set e.g. 32-64 once a config's species
    # counts get into the hundreds. Named render_chunk_size (not bare
    # chunk_size) to keep it distinct from MicrographConfig's own
    # crowd_chunk_size, a different chunking knob entirely.
    render_chunk_size: int | None = None

    # --- Output ---
    # One path field, not one per layout: this is the single directory a run
    # writes under, read as the leaf when untracked and as the root of the
    # numbered job tree when tracked. `None` rather than a baked-in default
    # because which default applies is not knowable until tracking is -- see
    # pipelines._common.resolve_output_dir.
    output_dir: str | None = None
    filename: str = "tomogram"

    # --- Job tracking (opt-in) ---
    # Setting `project` or `job_id` routes output through `specter.jobs`
    # instead of the flat output_dir/filename layout above: the directory
    # becomes output_dir/[project/]tomograms/J00N/, numbered and with a
    # job.json recording the full parameter set, git commit and status.
    # Neither is required -- leaving both unset keeps today's exact flat
    # behavior. When chained via `run_tilt_series(..., tomogram_config=...)`,
    # leaving both unset here *and* tracking the outer TiltSeriesConfig
    # cascades that project into this config automatically, so one tracked
    # chained call produces two separate, same-project jobs (one
    # "tomograms", one "tiltseries"), linked implicitly by the resulting
    # tiltseries job's volume_path pointing into this job's directory.
    project: str | None = None
    job_id: str | None = None


TOMOGRAM_HELP: dict[str, str] = {
    "targets": "Target protein species to pack (TOML-only, [[targets]] "
    "tables), each {'pdb_source': <code or path>, 'n_copies': <exact "
    "instance count>, 'location': 'cytosol'|'lumen' (optional, default "
    "'cytosol' -- only meaningful when [[membrane]] is set)}. Placed FIRST "
    "within its location, at this exact count, always exported to picks.",
    "filler": "Filler protein species to pack (TOML-only, [[filler]] "
    "tables), each {'pdb_source': <code or path>, 'location': "
    "'cytosol'|'lumen' (optional, default 'cytosol'), 'ratio': 1.0 "
    "(optional, relative attempt-weight among other filler species sharing "
    "the same location)}. Placed SECOND within its location, around any "
    "already-placed targets there. Excluded from picks by default (see "
    "write_picks).",
    "filler_from_pei2016": "Additive to filler: also pull filler species "
    "(location='cytosol') from the bundled PEI2016_CROWDING_TABLE (Pei et "
    "al. 2016 generic cytosolic crowding reference).",
    "filler_from_cryoetsim": "Additive to filler: also pull filler species "
    "(location='cytosol') from the bundled CRYOETSIM_PARTICLE_TABLE "
    "(CryoETSim dataset reference, Stojanovska et al. 2025).",
    "filler_table_categories": "Only used with filler_from_cryoetsim: "
    "restrict to these CRYOETSIM_PARTICLE_TABLE categories (macromolecules, "
    "distractors, transcription_translation, nucleosomes). None = all.",
    "filler_table_max_mw_kda": "Only used with filler_from_pei2016/"
    "filler_from_cryoetsim: exclude species above this mass, kDa.",
    "filler_table_min_mw_kda": "Only used with filler_from_pei2016/"
    "filler_from_cryoetsim: exclude species below this mass, kDa.",
    "target_shape": "Output specimen volume shape in voxels (Z, Y, X).",
    "voxel_size": "Voxel size in Angstrom.",
    "packing_backend": "Protein collision geometry: 'shape' (default, the "
    "real rotated footprint against an occupancy grid) or 'sphere' (one "
    "circumscribing sphere per instance). 'shape' reaches several times the "
    "density; 'sphere' saturates around 0.03-0.09 volume fraction, but is "
    "faster.",
    "packing_voxel_size": "Run packing_backend='shape' collision on a "
    "coarser grid than the render, an integer multiple of voxel_size. "
    "Unset = automatic, which only coarsens once the packing grid would be "
    "too large to hold; ordinary boxes are unaffected.",
    "packing_max_retries": "Trial positions per instance for "
    "packing_backend='shape'. Sets a packing stage's attempt ceiling; "
    "pairs with the packer's own stall_patience, which cuts that budget "
    "short once a species saturates.",
    "filler_occupancy_fraction": "Target packing density for filler "
    "species, as a bare-sphere fraction of EACH REGION's own volume it's "
    "placed in (the whole box when [[membrane]] is empty -- 'cytosol' is "
    "then the whole box). Deliberately high by default -- RSA self-limits "
    "at its own physical jamming ceiling rather than erroring, so filler "
    "packs until it jams rather than needing this hand-tuned. Lower "
    "it for a sparser filler layer, or if a small region (e.g. a tight "
    "vesicle lumen) makes the implied candidate pool impractically large.",
    "clip_axes": "(z, y, x) -- True on an axis lets a placed instance's "
    "body extend past that wall (truncated at render time) instead of "
    "being rejected outright. TOML-only (list[bool]).",
    "pdb_cache_dir": "Where downloaded PDB/mmCIF structures are cached. An "
    "input cache shared by every run, not an output location -- job tracking "
    "does not redirect it.",
    "seed": "Random seed.",
    "membrane": "One or more MembraneGenerator kwargs dicts (TOML-only, "
    "[[membrane]] tables, one per composited TEMPLATE) -- optional, empty "
    "by default (no membrane at all; the whole tomogram is then one "
    "cytosol region). e.g. {'shape_backend': 'spherical_harmonics', "
    "'n_copies': 3}. See MembraneGenerator's own docstring for the full "
    "per-backend parameter set; plus 'n_copies' (int, default 1, expands "
    "one entry into that many independently-seeded instances, each "
    "collision-rejecting-random-placed) and 'target_shape' (default "
    "omitted = auto-sized per instance).",
    "membrane_transmembrane_specs": "Transmembrane protein species (TOML-"
    "only, [[membrane_transmembrane_specs]] tables), each {'pdb_source': "
    "<code or path>, 'n_copies': 1, 'parameterization': 'shtyrov'}. Only "
    "meaningful when [[membrane]] is set, applies across all instances.",
    "membrane_region_density_threshold": "Passed through to "
    "TomogramSpecimenGenerator's own region_density_threshold.",
    "membrane_region_max_passes": "Passed through to "
    "TomogramSpecimenGenerator's own region_max_passes.",
    "membrane_min_transmembrane_spacing": "Minimum center-to-center "
    "spacing between placed transmembrane proteins, Angstrom. Only "
    "meaningful when [[membrane]] is set.",
    "scattering_factors": "Atomic scattering-factor parameterization for "
    "everything rendered from atoms: targets, filler, filaments, "
    "microtubules, carbon film, bilayer and transmembrane proteins. A "
    "[[membrane]] table naming its own 'parameterization' overrides this for "
    "that population. Gold fiducials fall back to Peng under 'shtyrov', "
    "which has no elemental gold.",
    "readd_hydrogens": "Whether to replace a structure's own hydrogens with "
    "the monomer library's ideal geometry: 'auto' (default) keeps hydrogens "
    "the file already carries and adds them only when it has none, true "
    "always re-adds, false never adds hydrogen density (they still inform "
    "atom typing). Needs a Monomer Library on $CLIBD_MON to have any effect.",
    "monomer_library_path": "Path to a Monomer Library "
    "(https://github.com/MonomerLibrary/monomers), which completes a "
    "structure's bond topology and hydrogens so Shtyrov species resolve. "
    "Unset falls back to $CLIBD_MON. Without one, around 44% of a "
    "hydrogen-free protein falls back to per-element Peng factors.",
    "filaments": "Filament species to scatter through the tomogram (TOML-"
    "only, [[filaments]] tables), each mapping onto "
    "specter.specimen.filament.FilamentSpec kwargs, e.g. {'code': '1TUB', "
    "'step': 85.0, 'flex_deg': 3.0, 'n_copies': 4}. Placed right after "
    "membranes, before targets/filler -- no region-gating, no collision "
    "avoidance against the membrane shell or each other, but targets/"
    "filler DO avoid already-placed filaments.",
    "actin": "Convenience toggle: also place the bundled ACTIN_SPEC preset "
    "(real F-actin helical repeat) without writing a [[filaments]] entry. "
    "Additive to filaments above. For more instances or other single-strand "
    "species, use [[filaments]]; for microtubules use [[microtubules]].",
    "microtubules": "Microtubule species to scatter through the tomogram "
    "(TOML-only, [[microtubules]] tables), each mapping onto "
    "specter.specimen.filament.MicrotubuleSpec kwargs, e.g. {'n_copies': 2, "
    "'n_protofilaments': 13, 'bend_radius': 30000.0}. Real 13-protofilament "
    "tubes with a lumen and an A-lattice seam -- not the single protofilament "
    "[[filaments]] with a tubulin dimer would give. Placed alongside "
    "filaments: no region-gating, no collision avoidance, but targets/filler "
    "DO avoid them.",
    "carbon_film": "Zero or one [[carbon_film]] table (TOML-only) describing a carbon "
    "support film, mapping onto specter.specimen.CarbonFilmSpec kwargs "
    "(thickness, hole_radius, edge_fraction, edge_side, edge_roughness). "
    "Painted into the volume before anything else is "
    "placed; not carbon-aware for placement (see TomogramSpecimenGenerator's "
    "own docstring). Empty (default): no carbon film.",
    "beads": "Gold fiducial bead populations to pack (TOML-only, [[beads]] "
    "tables), each {'radius': <Angstrom or [low, high]>, 'n_copies': 1}. "
    "A [low, high] radius draws a fresh size per bead, giving the population "
    "the dispersity real colloidal gold has. Placed via the same "
    "RSA packing as membranes/targets/filler, avoiding the membrane shell "
    "and already-placed filaments -- not region-gated to cytosol/lumen.",
    "bead_roughness": "How irregular each gold fiducial's boundary is, as an "
    "RMS fraction of its radius. One number, or a [low, high] pair drawn per "
    "bead so a population mixes near-round and misshapen particles. 0.0 gives "
    "clean spheres; 0.12-0.20 reads as an irregular particle.",
    "write_picks": "Write one copick-style .ndjson pick file per species "
    "alongside the volume. Filler species (declared via 'ratio', not "
    "'n_copies') are included by default -- see TomogramProteinSpec's own "
    "role/export_picks docstrings for how to exclude them instead.",
    "annotation_version": "Version string used in pick filenames "
    "('{species}-{version}_orientedpoint.ndjson').",
    "write_segmentation": "Save per-instance integer label volumes as "
    ".mrc alongside the density volume -- the intended ground truth for "
    "membrane geometry specifically (not a coordinate file, since a "
    "membrane surface has no single natural 'position' the way a protein "
    "does). {filename}_protein_labels.mrc is always written; "
    "{filename}_membrane_labels.mrc and {filename}_regions.mrc "
    "(0=cytosol/1=shell/2=lumen) are added when [[membrane]] is set.",
    "device": "cpu | cuda | cuda:0 | 0,1,2. Drives the whole "
    "MembraneGenerator/TomogramSpecimenGenerator pipeline (shape field, "
    "bilayer profile, rasterization, transmembrane/targets/filler "
    "PotentialBuilder rendering) -- packing itself always runs on CPU "
    "regardless (vesin's neighbor list is both slower and OOM-prone on "
    "GPU at realistic particle counts). A comma-separated list of GPU "
    "indices (or 'auto', every visible GPU) pools those GPUs for "
    "concurrent per-species rendering (see render_workers); the first "
    "entry becomes the primary device for everything else.",
    "accumulator_device": "Device for the shared canvas tensors "
    "(volume/instance_labels/membrane_labels), decoupled from 'device' "
    "above (which stays the compute device regardless). None (default): "
    "same as 'device'. 'auto': estimate the canvas' own memory footprint "
    "and fall back to 'cpu' if it would exceed half of 'device''s "
    "currently free memory. Explicit 'cpu' always keeps 'device'='cuda' "
    "for fast rendering/rotation while letting the canvas itself be sized "
    "by system RAM instead of GPU VRAM -- useful for a large field of "
    "view at fine voxel_size whose canvas alone would exceed any single "
    "GPU's VRAM. Rotation/rendering speed is unaffected either way; only "
    "each already-small rotated chunk crosses devices, never the canvas "
    "itself.",
    "render_workers": "Number of PDB species rendered concurrently within "
    "one tomogram (membrane_transmembrane_specs and targets/filler each "
    "get their own concurrent build pass). Default 1 (serial, original "
    "behaviour) -- raise for tomograms with several species, especially "
    "with n_copies>1 [[membrane]] entries (all instances of one entry "
    "share one render pass). Set to 'auto' (TOML/Python config only -- the "
    "--render_workers CLI flag stays integer-only) to pick min(n_species, "
    "8) per pool automatically, the measured sweet spot from a full "
    "production-scale sweep -- see "
    "specter.specimen._parallel_render.recommend_render_workers. Round-"
    "robins across device's GPU pool when device is set to a "
    "comma-separated list or 'auto' (see device above); device choice was "
    "measured to barely matter at the recommended worker count.",
    "render_chunk_size": "Instances rotated per GPU batch, per species, when "
    "rendering targets/filler. None (default) rotates all of a species' "
    "accepted instances in one batched call -- fine at small scale, but a "
    "species with hundreds of instances can then need many GB for that "
    "one call. Set e.g. 32-64 once species counts get into the hundreds.",
    "output_dir": "Directory to save output files when untracked. Setting "
    "--project or --job_id instead makes this the root of the numbered job "
    "tree, so tracking organises output within the folder you chose rather "
    "than moving it elsewhere. Unset defaults to <artifact>/ "
    "untracked, and to the project root found by walking up from cwd for an "
    "existing .specter marker when tracked.",
    "filename": "Base name for the output volume (no extension).",
    "project": "Optional: number and track this run through specter.jobs. "
    "Not required for tracking -- job_id alone also triggers it. The run "
    "lands in "
    "<output_dir>/[<project>/]tomograms/J00N/ with a job.json recording "
    "every parameter, the git commit and the run's status. When chained "
    "via --tomogram_config on `specter simulate tiltseries`, leaving this "
    "unset while tracking the tiltseries run cascades that project here "
    "automatically.",
    "job_id": "Pin the job directory (e.g. J001) rather than auto-assigning "
    "the next one: resumes into it if it exists, creates it otherwise.",
}
