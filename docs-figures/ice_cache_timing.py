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
Each cell size is timed over a short run of outer L-BFGS steps, preceded by a
separate warmup run whose cost is discarded (first-call CUDA autotuning and
allocator growth would otherwise land entirely in the first size measured).
Cost per step is then multiplied out to the 250-step budget
`build_one_ice_config` uses.

Each size therefore runs a COMPLETE configuration under the production recipe,
rather than sampling its opening steps. Both reported quantities drift with
optimisation progress, in opposite directions, so a short sample gets both
wrong at once:

- **Per-step cost falls** as L-BFGS accumulates curvature history. Measured at
  n=256: 2.88 s/step over 8 steps, 2.209 over 40, 2.180 over 120. Early steps
  spend more of their budget in the strong-Wolfe line search, which may call
  the closure up to 10 times per outer step. An 8-step sample overstates cost
  by ~30%, and the effect is weakest at small cell sizes (n=128: 0.301 over 8
  steps vs 0.279 over 120), i.e. it misleads most exactly where cost matters.
- **Peak memory rises**, and by much more. The ML-BOP three-body term scales
  with neighbour triplets, and local coordination tightens as the structure
  approaches real amorphous ice, so allocation grows as the run converges. A
  40-step sample at n=256 reported 10.8 GiB where a converged run peaked near
  40 GiB. That underestimate is not academic: it is what led a 20-config run
  on a 44 GiB card to run out of memory at config 17, several hours in.

Both sets of figures in that list were measured under the pre-float64
optimiser, whose peak was several times the current one's; they are kept
because the direction of each error is what matters, and it has not changed.
The absolute numbers to use are the ones this script prints.

Sizing a GPU should use the reserved column, not allocated: reserved is what
the process actually holds from the driver, and what a new allocation fails
against once the caching allocator has fragmented over a long run.
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

#: Step budget `build_one_ice_config` uses, for extrapolating a full run.
PRODUCTION_STEPS = 250

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


def time_one_cell(
    n: int, dx: float, steps: int, device: str
) -> tuple[int, int, float, float, float]:
    """
    Run one full configuration at a given cell size, timing it and recording
    its peak memory.

    Runs to convergence under the production recipe rather than sampling a
    fixed number of steps, because BOTH quantities this reports depend on how
    far the optimisation has progressed:

    - Cost per step falls as L-BFGS accumulates curvature history (see the
      module docstring).
    - Peak memory RISES as the structure converges. The ML-BOP three-body
      term's cost scales with neighbour triplets, and local coordination
      tightens as the configuration approaches real amorphous ice, so a
      near-converged config allocates several times what an early-stage one
      does. Sampling the first few dozen steps understates the peak by
      roughly 4x at n=256, which is enough to send a reader looking for a
      GPU that then runs out of memory partway through a real run.

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

    Returns
    -------
    n_atoms : int
        Water beads in the cell.
    n_steps_actual : int
        Steps actually taken before the loss plateaued (or ``steps``).
    per_step_s : float
        Seconds per outer L-BFGS step.
    peak_alloc_gb : float
        Peak CUDA memory allocated (live tensors), in GiB.
    peak_reserved_gb : float
        Peak CUDA memory reserved from the driver, in GiB. This is the number
        to size a GPU against: it is what the process actually holds, and
        what an allocation fails against.
    """
    torch.manual_seed(0)
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

    n_steps_actual = history["step"][-1] + 1 if history["stopped_early"] else steps
    peak_alloc_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
    assert gd.positions is not None
    n_atoms = int(gd.positions.shape[0])

    del gd
    torch.cuda.empty_cache()
    return (
        n_atoms,
        n_steps_actual,
        elapsed / n_steps_actual,
        peak_alloc_gb,
        peak_reserved_gb,
    )


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
        help="Step ceiling per size. Lowering it overstates cost per step AND "
        "understates peak memory -- see the module docstring.",
    )
    args = parser.parse_args()

    if args.sweep == "cell":
        geometries = [(n, args.dx) for n in CELL_SIZES]
    else:
        geometries = [(DX_SWEEP_N, dx) for dx in DX_VALUES]

    name = torch.cuda.get_device_name(args.device)
    print(
        f"Device: {args.device} ({name}), sweep={args.sweep}, "
        f"{args.steps} step ceiling, tol={PRODUCTION_TOL:g}\n"
    )
    header = (
        f"{'n':>5}{'dx (A)':>8}{'cell (A)':>10}{'atoms':>10}{'steps':>8}"
        f"{'s/step':>9}{'alloc GiB':>11}{'reserved GiB':>14}{'250 steps':>12}"
    )
    print(header)
    print("-" * len(header))

    for n, dx in geometries:
        n_atoms, n_actual, per_step, alloc_gb, reserved_gb = time_one_cell(
            n, dx, args.steps, args.device
        )
        full = per_step * PRODUCTION_STEPS
        full_str = f"{full / 60:.0f} min" if full < 3600 else f"{full / 3600:.1f} h"
        print(
            f"{n:>5}{dx:>8.2f}{n * dx:>10.0f}{n_atoms:>10,}{n_actual:>8}"
            f"{per_step:>9.3f}{alloc_gb:>11.2f}{reserved_gb:>14.2f}{full_str:>12}"
        )


if __name__ == "__main__":
    main()
