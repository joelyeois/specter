"""IceCacheConfig: parameters for building a replacement IceBank library."""

from __future__ import annotations

from dataclasses import dataclass, field

from ._paths import default_output_dir


@dataclass
class IceCacheConfig:
    """Parameters for generating a library of amorphous-ice configurations,
    loaded from a TOML config file.

    Drives `specter.pipelines.run_build_ice_cache` (the `specter build ice`
    command), which runs `specter.ice.GradientSKIcemaker` once per
    configuration and saves the converged coordinates. The result is a
    directory of `config_NNN.pt` files an `IceBank` can draw crops from --
    point any simulation config's `ice_cache_dir` at it to use these instead
    of the bundled `ice_data/ice_cache`.

    Only the sampling geometry, the convergence budget, and the scheduling
    are configurable. The optimisation recipe itself (S(k) target, MLBOP
    weight, and the `mlbop_target = -0.413` eV/atom energy of real LDA-80K
    ice) is fixed: those are measured properties of the phase of ice being
    reproduced, and a cache mixing recipes would have `IceBank` drawing
    from several different phases interchangeably.
    """

    # --- Library ---
    # Number of independent configurations to generate. Each is a separate
    # optimisation run, so cost scales linearly -- and IceBank draws a random
    # rotation and translation from whichever config it picks, so a handful of
    # configs already gives a large space of distinct crops.
    num_configs: int = 8
    # Voxels along each side of the (cubic) periodic cell; the cell measures
    # n * dx Å. Cubic only -- IceBank stores one scalar box_L per config
    # and filters candidates against it, so a non-cubic cell is unrepresentable.
    n: int = 256
    dx: float = 1.0  # A/voxel
    # First seed; the i-th config uses seed_start + i and is saved under that
    # seed's name. Raise it past an existing library's highest seed to extend
    # that library rather than regenerate it.
    seed_start: int = 0

    # --- Optimisation ---
    # L-BFGS step ceiling per config. An upper bound only: the run stops early
    # once the loss plateaus (GradientSKIcemaker.optimize's tol/patience).
    # 250 steps is ~6000 loss evaluations at optimize's default max_iter, the
    # budget its settings were chosen on; see its docstring.
    n_steps: int = 250

    # --- Compute ---
    # "cpu" | "cuda" | "cuda:0" | a bare GPU index | a comma-separated list of
    # GPU indices ("0,1,2,3"). Multiple devices shard whole configs across one
    # worker process per device, so N GPUs build a library roughly N times
    # faster -- but they have to be named. There is no "auto": it used to mean
    # "every visible GPU" here and nowhere else, which made the same word a
    # crash on `specter simulate particles` and a silent cuda:0 on `specter
    # reconstruct particle`. See `specter.devices.parse_device`.
    device: str = "cuda"

    # --- Output ---
    output_dir: str = field(default_factory=lambda: default_output_dir("ice"))
    # Regenerate configs whose file is already present, instead of skipping
    # them. Skipping is what lets an interrupted multi-hour run resume.
    overwrite: bool = False
    # Save IceBank.plot_diagnostics' energy/S(k) figures for the finished
    # library.
    diagnostics: bool = False


ICE_CACHE_HELP: dict[str, str] = {
    "num_configs": "Number of independent ice configurations to generate. "
    "Each costs a full optimisation run -- tens of minutes at the default "
    "n=256, dx=1.0.",
    "n": "Voxels along each side of the cubic periodic cell (the cell "
    "measures n*dx Angstrom). Must be at least as large as the biggest ice "
    "volume a simulation will request from it in any one dimension, or "
    "IceBank has to tile several crops together to serve the request.",
    "dx": "Voxel size in Angstrom used when optimising and when voxelizing "
    "crops. Set this to the pixel size of the simulations the cache is for.",
    "seed_start": "Seed of the first configuration; the i-th uses "
    "seed_start+i, and each is saved under its own seed's filename. Set "
    "this past an existing library's highest seed to extend it rather than "
    "regenerate it.",
    "n_steps": "L-BFGS step ceiling per configuration. An upper bound only "
    "-- a run whose loss plateaus stops early.",
    "device": "cpu | cuda | cuda:0 | a bare GPU index | a comma-separated "
    "list of GPU indices (0,1,2,3). Several devices shard whole "
    "configurations across one worker process per device, so N GPUs "
    "generate a library roughly N times faster -- name them explicitly, "
    "e.g. 0,1,2,3.",
    "output_dir": "Directory to write config_NNN.pt files and manifest.json "
    "to. Point a simulation config's ice_cache_dir at it to use the result. "
    "Never the bundled ice_data/ice_cache, which ships with the repository.",
    "overwrite": "Regenerate configurations already present in output_dir "
    "instead of skipping them. Skipping is what lets an interrupted run "
    "resume where it left off.",
    "diagnostics": "Save energy and S(k) diagnostic figures for the "
    "finished library to output_dir.",
}
