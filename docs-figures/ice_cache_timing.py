"""
Measure the cost of generating one ice configuration, as a function of cell
size (`--sweep cell`) and of voxel size (`--sweep dx`).

Produces both timing and memory tables in `docs/user-guide/ice-cache.md`, which
readers use to budget a `specter build ice` run before starting one that takes
hours. Run it again on new hardware rather than trusting numbers measured
elsewhere; the absolute seconds are specific to the GPU, though the scaling
with `n` is not.

The two sweeps separate the two things that set the cost. Every loss
evaluation transforms an (n, n, n) grid and sums an ML-BOP energy over the
water beads in it, and those scale with different quantities: the grid with
`n`, the beads with the physical cell volume `(n * dx)**3`. `--sweep cell`
grows both together. `--sweep dx` holds the grid fixed and shrinks the cell,
so what it isolates is how much of the cost is the beads.

Usage
-----
    python docs-figures/ice_cache_timing.py [--device cuda:1] [--sweep dx]

Method
------
Each geometry generates `REPEATS` COMPLETE configurations under the production
recipe, one per seed, and reports the mean wall time and the range across
them. Wall time to completion is the quantity, not a cost per step: a run
stops when its loss plateaus, so how many steps it takes is part of what a
configuration costs, and it varies with the random initialisation. Reporting
seconds per step and multiplying out to the 250-step ceiling would quote a
budget almost no run actually spends.

Running to completion is also what makes the memory figure trustworthy. Peak
memory RISES as the structure converges: the ML-BOP three-body term scales
with neighbour triplets, and local coordination tightens as the structure
approaches real amorphous ice. A 40-step sample at n=256 once reported
10.8 GiB where the converged run peaked near 40 GiB, and that underestimate
is what led a 20-config run on a 44 GiB card to run out of memory at config
17, several hours in. (Both of those figures are from the pre-float64
optimiser, whose peak was several times the current one's. The direction of
the error is the part that still applies.)

Reserved is reported rather than allocated, and as the worst repeat rather
than the mean: it is what the process holds from the driver, what a new
allocation fails against once the caching allocator has fragmented, and what
decides whether a run fits on a given card at all.
"""

from __future__ import annotations

import argparse
import time

import torch

from specter.ice import GradientSKIcemaker

#: Cell sizes to sweep, in voxels per side. 256 is `IceCacheConfig`'s default
#: and the size the bundled library was generated at.
CELL_SIZES: tuple[int, ...] = (64, 96, 128, 192, 256)

#: Voxel sizes to sweep, in Angstrom, with `n` held at `DX_SWEEP_N`. The
#: bundled library is dx=1.0; 0.5 and 0.25 are what a simulation wanting ice
#: constrained above 0.5 1/A would have to generate for itself (see the
#: "When to generate your own" section of ice-cache.md).
DX_VALUES: tuple[float, ...] = (0.25, 0.5, 1.0)

#: Grid held fixed across `DX_VALUES`, so the S(k) transform is the same size
#: in every row and only the physical cell -- and with it the bead count --
#: changes.
DX_SWEEP_N = 256

#: Step budget `build_one_ice_config` uses.
PRODUCTION_STEPS = 250

#: Configurations generated per geometry, each from its own seed. Where a run
#: stops is set by when its loss plateaus, which depends on the random
#: initialisation, so a single run reports one draw from a spread rather than
#: what a configuration costs. Three is enough to place the mean and show the
#: spread without turning the sweep into an hour per table.
REPEATS = 3

#: Early-stopping tolerance, matching `build_one_ice_config`'s own recipe.
#: NOT a knob: at 1e-4 (what this script used before) a config runs well past
#: where production stops, so both columns it reports would describe a run
#: nobody generates. See `GradientSKIcemaker.optimize`'s `tol` docstring for
#: why production trades that tail for time.
PRODUCTION_TOL = 1e-3

#: Step ceiling per cell size, matching `build_one_ice_config`'s production
#: budget. Not a speed knob: a short run both overstates cost per step and
#: understates peak memory (see `time_one_cell`), which is exactly the pair of
#: errors that makes a reader mis-size a GPU. Expect ~20 min for a full sweep.
DEFAULT_STEPS = PRODUCTION_STEPS


def time_one_cell(n: int, dx: float, steps: int, device: str, seed: int) -> dict:
    """
    Generate one complete configuration, timing it end to end and recording
    its peak memory.

    What this reports is the wall time of a whole config, which is what
    ``specter build ice`` actually charges a user, not a cost per step
    multiplied out to a nominal budget. The two differ, and not by a constant:
    a run stops when its loss plateaus, so the step count is part of the cost
    and varies with the random initialisation. `main` therefore repeats each
    geometry over several seeds and averages.

    Peak memory RISES as the structure converges: the ML-BOP three-body term
    scales with neighbour triplets, and local coordination tightens as the
    configuration approaches real amorphous ice. Sampling the opening steps
    understates the peak several-fold, which is enough to send a reader
    looking for a GPU that then runs out of memory partway through a real
    run. Running to completion is what makes the memory figure trustworthy.

    Parameters
    ----------
    n : int
        Voxels per side of the cubic cell.
    dx : float
        Voxel size in Angstrom.
    steps : int
        Ceiling on outer L-BFGS steps. Early stopping still applies, so a
        plateaued run finishes sooner.
    device : str
        CUDA device to run on, e.g. ``"cuda:1"``.
    seed : int
        Seed for the random initialisation, so repeats of one geometry differ
        the way separate configs in a real library do.

    Returns
    -------
    dict
        ``n_atoms`` (water beads in the cell), ``n_steps`` (steps taken before
        the loss plateaued, or `steps`), ``wall_s`` (seconds for the whole
        configuration), ``alloc_gb`` and ``reserved_gb`` (peak CUDA memory).
        Size a GPU against ``reserved_gb``: it is what the process holds from
        the driver, and what a new allocation fails against.
    """
    torch.manual_seed(seed)
    gd = GradientSKIcemaker(n=n, dx=dx, device=device, progressbars=False)
    gd.init_random()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    history = gd.optimize(
        n_steps=steps,
        record_every=steps,
        rep_strength=0.0,
        mlbop_strength=0.5,
        mlbop_target=-0.413,
        tol=PRODUCTION_TOL,
        patience=10,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    assert gd.positions is not None
    result = {
        "n_atoms": int(gd.positions.shape[0]),
        "n_steps": history["step"][-1] + 1 if history["stopped_early"] else steps,
        "wall_s": elapsed,
        "alloc_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
    }
    del gd
    torch.cuda.empty_cache()
    return result


def main() -> None:
    """Sweep either `CELL_SIZES` or `DX_VALUES` and print the docs table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="CUDA device to time on.")
    parser.add_argument(
        "--sweep",
        choices=("cell", "dx"),
        default="cell",
        help="'cell' varies n at fixed --dx (the cost-vs-cell-size table); "
        "'dx' varies voxel size at n=%d (the cost-vs-resolution table)." % DX_SWEEP_N,
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=1.0,
        help="Voxel size in Angstrom for --sweep cell. Ignored by --sweep dx, "
        "which sweeps DX_VALUES.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Step ceiling per size. Lowering it truncates the run AND "
        "understates peak memory -- see the module docstring.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS,
        help="Configurations generated per geometry, each from a different "
        "seed. Where a run stops is init-dependent, so one run is a draw "
        "from a spread rather than the cost of a config.",
    )
    args = parser.parse_args()

    if args.sweep == "cell":
        geometries = [(n, args.dx) for n in CELL_SIZES]
    else:
        geometries = [(DX_SWEEP_N, dx) for dx in DX_VALUES]

    name = torch.cuda.get_device_name(args.device)
    print(
        f"Device: {args.device} ({name}), sweep={args.sweep}, "
        f"{args.steps} step ceiling, tol={PRODUCTION_TOL:g}, "
        f"{args.repeats} repeat(s) per geometry\n"
    )
    header = (
        f"{'n':>5}{'dx (A)':>8}{'cell (A)':>10}{'atoms':>10}"
        f"{'steps':>14}{'wall time':>14}{'range':>16}{'reserved GiB':>14}"
    )
    print(header)
    print("-" * len(header))

    for n, dx in geometries:
        runs = [
            time_one_cell(n, dx, args.steps, args.device, seed)
            for seed in range(args.repeats)
        ]
        walls = [r["wall_s"] for r in runs]
        steps_taken = [r["n_steps"] for r in runs]
        mean_wall = sum(walls) / len(walls)
        # Reserved is reported as the WORST repeat, not the mean: it is used
        # to decide whether a run fits on a card, and a mean would recommend
        # a GPU that one config in three overflows.
        reserved = max(r["reserved_gb"] for r in runs)
        print(
            f"{n:>5}{dx:>8.2f}{n * dx:>10.0f}{runs[0]['n_atoms']:>10,}"
            f"{sum(steps_taken) / len(steps_taken):>14.0f}"
            f"{_format_duration(mean_wall):>14}"
            f"{_format_duration(min(walls)) + '-' + _format_duration(max(walls)):>16}"
            f"{reserved:>14.2f}"
        )


def _format_duration(seconds: float) -> str:
    """Wall time as the docs quote it: seconds under a minute, else m:ss."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"


if __name__ == "__main__":
    main()
