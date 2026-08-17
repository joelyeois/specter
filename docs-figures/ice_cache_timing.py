"""
Measure the cost of generating one ice configuration as a function of cell size.

Produces the timing and memory table in `docs/user-guide/ice-cache.md`, which
readers use to budget a `specter build ice` run before starting one that takes
hours. Run it again on new hardware rather than trusting numbers measured
elsewhere; the absolute seconds are specific to the GPU, though the scaling
with `n` is not.

Usage
-----
    python docs-figures/ice_cache_timing.py [--device cuda:1] [--steps 8]

Method
------
Each cell size is timed over a short run of outer L-BFGS steps, preceded by a
separate warmup run whose cost is discarded (first-call CUDA autotuning and
allocator growth would otherwise land entirely in the first size measured).
Cost per step is then multiplied out to the 600-step budget
`build_one_ice_config` uses.

Two properties of the measurement are worth knowing when reading the result:

- **Per-step cost falls as L-BFGS accumulates curvature history**, so the
  sample has to be long enough to get past that. Measured at n=256 on an L40:
  2.88 s/step over 8 steps, 2.209 over 40, 2.180 over 120. The early steps
  spend more of their budget in the strong-Wolfe line search, which may call
  the closure up to 10 times per outer step. `DEFAULT_STEPS = 40` is where the
  estimate stabilises: it reproduces the 2.20 s/step median that the bundled
  `ice-data/ice_cache` recorded over complete 600-step production runs, while
  an 8-step sample overstates it by ~30%. The effect is much weaker at small
  cell sizes (n=128: 0.301 over 8 steps vs 0.279 over 120), so a short sweep
  misleads specifically where the cost matters most.
- `tol=None` disables early stopping, so every timed step is a real step. A
  production run stops as soon as the loss plateaus, which for the bundled
  library was 407-600 steps rather than always 600, making the extrapolated
  column an upper bound.
"""

from __future__ import annotations

import argparse
import time

import torch

from specter.ice import GradientSKIcemaker

#: Cell sizes to sweep, in voxels per side. 256 is `IceCacheConfig`'s default
#: and the size the bundled library was generated at.
CELL_SIZES: tuple[int, ...] = (64, 96, 128, 192, 256)

#: Step budget `build_one_ice_config` uses, for extrapolating a full run.
PRODUCTION_STEPS = 600

#: Timed steps per cell size. Not a speed/accuracy knob to turn down freely --
#: see the module docstring for why fewer than ~40 overstates the cost.
DEFAULT_STEPS = 40


def time_one_cell(
    n: int, dx: float, steps: int, device: str
) -> tuple[int, float, float, float]:
    """
    Time a short optimisation run at one cell size.

    Parameters
    ----------
    n : int
        Voxels per side of the cubic cell.
    dx : float
        Voxel size in Angstrom.
    steps : int
        Outer L-BFGS steps to time, after a discarded warmup.
    device : str
        CUDA device to run on, e.g. ``"cuda:1"``.

    Returns
    -------
    n_atoms : int
        Water beads in the cell.
    setup_s : float
        Seconds to construct the icemaker (S(k) target, k-grid, repulsion
        kernel), paid once per configuration.
    per_step_s : float
        Seconds per outer L-BFGS step.
    peak_gb : float
        Peak CUDA memory allocated during the timed run, in GiB.
    """
    torch.manual_seed(0)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    gd = GradientSKIcemaker(n=n, dx=dx, device=device, progressbars=False)
    torch.cuda.synchronize(device)
    setup_s = time.perf_counter() - t0

    gd.init_random()
    gd.optimize(n_steps=2, record_every=10**9, tol=None)  # warmup, discarded

    gd.init_random()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    gd.optimize(n_steps=steps, record_every=10**9, tol=None)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    assert gd.positions is not None
    n_atoms = int(gd.positions.shape[0])

    del gd
    torch.cuda.empty_cache()
    return n_atoms, setup_s, elapsed / steps, peak_gb


def main() -> None:
    """Sweep `CELL_SIZES` and print the docs table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="CUDA device to time on.")
    parser.add_argument("--dx", type=float, default=1.0, help="Voxel size, Angstrom.")
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Timed steps per size. Lowering this overstates cost per step.",
    )
    args = parser.parse_args()

    name = torch.cuda.get_device_name(args.device)
    print(f"Device: {args.device} ({name}), dx={args.dx} A, {args.steps} timed steps\n")
    header = (
        f"{'n':>5}{'cell (A)':>10}{'atoms':>10}{'setup (s)':>11}"
        f"{'s/step':>9}{'peak (GiB)':>12}{'600 steps':>12}"
    )
    print(header)
    print("-" * len(header))

    for n in CELL_SIZES:
        n_atoms, setup_s, per_step, peak_gb = time_one_cell(
            n, args.dx, args.steps, args.device
        )
        full = setup_s + per_step * PRODUCTION_STEPS
        full_str = f"{full / 60:.0f} min" if full < 3600 else f"{full / 3600:.1f} h"
        print(
            f"{n:>5}{n * args.dx:>10.0f}{n_atoms:>10,}{setup_s:>11.1f}"
            f"{per_step:>9.3f}{peak_gb:>12.2f}{full_str:>12}"
        )


if __name__ == "__main__":
    main()
