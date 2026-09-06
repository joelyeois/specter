"""Ice-cache generation pipeline: `IceCacheConfig` in, a directory of ice
configurations an `IceBank` can draw from out.

Drives `specter.ice.build_one_ice_config` -- one converged
`GradientSKIcemaker` run per configuration -- and adds the three things a
multi-hour batch job needs on top of that loop:

- **Sharding.** Configurations are fully independent, so several devices
  each take a slice of them (`config.device` as a GPU list or "auto").
  Whole configurations go to whole worker processes rather than any
  finer-grained parallelism: one configuration is tens of minutes of work,
  which makes process-pool startup free in relative terms and buys real
  fault isolation (a GPU OOM on one device costs that shard, not the run)
  plus CUDA-context isolation per device.
- **Resume.** A configuration whose file is already present is skipped, so
  an interrupted run continues instead of restarting. This is why
  `ice_config_filename` names files after their seed: re-running the same
  request reproduces the same seeds and therefore skips cleanly.
- **A quality record.** `build_one_ice_config` measures the S(k) loss and
  ML-BOP energy of every configuration and saves them alongside the
  coordinates, but nothing reads them back. This pipeline reports both and
  writes a `manifest.json` describing the whole library. Both are reported
  as measurements, never as a pass/fail verdict -- see `_report_config` for
  why neither quantity has a threshold worth asserting.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from datetime import datetime, timezone

import torch

from specter.config import IceCacheConfig, validate_config
from specter.ice import build_one_ice_config, ice_config_filename

from specter.progress import console, format_elapsed, section
from ._common import _parse_device_pool


def _build_shard(
    config: IceCacheConfig, seeds: list[int], device: str, quiet: bool
) -> None:
    """
    Generate one worker's share of the library, sequentially, on `device`.

    Module-level (not a closure) so it survives pickling into a spawned
    worker process.

    Parameters
    ----------
    config : IceCacheConfig
        The run configuration; only its geometry/budget/output fields are
        read here, `device` is passed separately.
    seeds : list of int
        Seeds this worker is responsible for.
    device : str
        Device this worker generates on, e.g. ``"cuda:1"``.
    quiet : bool
        Suppress per-configuration optimisation progress bars. Set when
        several workers share one terminal, where interleaved bars are
        unreadable; each finished configuration still prints one summary
        line.
    """
    for seed in seeds:
        started = time.time()
        metadata = build_one_ice_config(
            os.path.join(config.output_dir, ice_config_filename(seed)),
            n=config.n,
            dx=config.dx,
            n_steps=config.n_steps,
            seed=seed,
            device=device,
            progressbars=not quiet,
        )
        _report_config(metadata, device, time.time() - started)
        # Return the finished config's cached blocks to the driver before
        # starting the next one. Live tensors do not accumulate across
        # configs (measured: allocation is flat over repeated calls), but a
        # single config's peak is large and grows as it converges -- near
        # 40 GiB at n=256 -- so the allocator ends each config holding a
        # correspondingly large, fragmented pool. A 20-config run on a 44 GiB
        # card ran out of memory at config 17 requesting 4.91 GiB while
        # nominally holding 38 GiB, and the same config succeeded
        # immediately in a fresh process. Releasing between configs costs one
        # cudaFree/cudaMalloc cycle against tens of minutes of compute.
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()


def _report_config(metadata: dict, device: str, elapsed: float) -> None:
    """
    Print one finished configuration's measured quality.

    Both numbers are reported as plain measurements, with no pass/fail
    verdict attached, because neither has a calibrated threshold:

    - `sk_loss` is measured on the coordinates as stored, so its scale
      depends on the box size and the storage encoding as well as on
      convergence. At n=256, dx=1.0 the bundled `ice_data/ice_cache` spans
      4e-4 to 0.02 under the current fixed-point encoding.
    - `E_per_atom` is not a distance from `mlbop_target` either: that
      target is one weighted term in a combined loss, not a value the
      optimisation is expected to reach. Bundled configs sit between
      -0.19 and -0.27 eV/atom against a -0.413 target.

    `stopped_early` is the one reliable signal here, and it answers a
    narrow question: whether the loss plateaued within the step budget or
    the budget ran out first.
    """
    steps = metadata["n_steps_actual"]
    budget = (
        f"plateaued after {steps}"
        if metadata["stopped_early"]
        else f"used the full {steps}"
    )
    console.print(
        f"  {ice_config_filename(metadata['seed'])} on {device}: "
        f"S(k) loss {metadata['sk_loss']:.4g}, "
        f"E/atom {metadata['energy']['E_per_atom']:+.4f} eV, "
        f"{budget} steps in {format_elapsed(elapsed)} "
        f"({elapsed / max(steps, 1):.2f} s/step)"
    )


def _write_manifest(config: IceCacheConfig, path: str) -> list[dict]:
    """
    Record what the finished library contains, and validate it.

    Reads back every configuration in `config.output_dir` -- including any
    a resumed run skipped, so the manifest describes the whole library
    rather than just this invocation -- and writes each one's geometry,
    recipe and convergence metrics to `path`. Every field is read from the
    configuration's own saved metadata rather than assumed from `config`,
    since one directory can legitimately hold configurations from several
    runs at different geometries. Loading each file is also the run's
    post-condition check: a truncated or otherwise unreadable
    configuration surfaces here rather than at simulation time.

    Returns
    -------
    list of dict
        One entry per configuration, as written to the manifest.
    """
    entries = []
    for name in sorted(os.listdir(config.output_dir)):
        if not name.endswith(".pt"):
            continue
        data = torch.load(os.path.join(config.output_dir, name), weights_only=False)
        entries.append(
            {
                "file": name,
                "seed": data.get("seed"),
                "n": data.get("n"),
                "dx": data.get("dx"),
                "box_L": data.get("box_L"),
                "n_atoms": int(data["positions"].shape[0]),
                "n_steps": data.get("n_steps"),
                "n_steps_actual": data.get("n_steps_actual"),
                "wall_time": data.get("wall_time"),
                "stopped_early": data.get("stopped_early"),
                "sk_loss": data.get("sk_loss"),
                "E_per_atom": data.get("energy", {}).get("E_per_atom"),
                "recipe": data.get("recipe"),
                "optimizer": data.get("optimizer"),
            }
        )

    try:
        from importlib.metadata import version

        specter_version = version("specter")
    except Exception:
        specter_version = "unknown"

    with open(path, "w") as f:
        json.dump(
            {
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "specter_version": specter_version,
                "configs": entries,
            },
            f,
            indent=2,
        )
    return entries


def run_build_ice_cache(config: IceCacheConfig) -> None:
    """
    Generate a library of amorphous-ice configurations for :class:`~specter.ice.IceBank`.

    Writes ``config_NNN.pt`` files plus a ``manifest.json`` into
    ``config.output_dir``. Point any simulation config's ``ice_cache_dir``
    at that directory to sample ice from this library instead of the
    bundled ``ice_data/ice_cache``.

    This is expensive by construction -- on the order of tens of minutes
    per configuration at the default ``n=256, dx=1.0`` -- since it runs the
    full ``GradientSKIcemaker`` optimisation that :class:`~specter.ice.IceBank`
    exists to amortize. Configurations already present are skipped unless
    ``config.overwrite`` is set, so an interrupted run resumes rather than
    restarting.

    Parameters
    ----------
    config : IceCacheConfig
        Fully-resolved run configuration, e.g. from
        :func:`specter.config.load_config`, or constructed directly in
        Python.
    """
    validate_config(config)
    section("Ice cache")
    os.makedirs(config.output_dir, exist_ok=True)

    seeds = [config.seed_start + i for i in range(config.n_configs)]
    pending = [
        seed
        for seed in seeds
        if config.overwrite
        or not os.path.exists(
            os.path.join(config.output_dir, ice_config_filename(seed))
        )
    ]
    devices = _parse_device_pool(config.device)

    console.print(
        f"Library: {config.n_configs} configs at n={config.n}, dx={config.dx} A "
        f"(cell {config.n * config.dx:.0f} A), seeds "
        f"{seeds[0]}-{seeds[-1]}" + f"\nOutput:  {config.output_dir}"
    )
    if len(pending) < len(seeds):
        console.print(
            f"Skipping {len(seeds) - len(pending)} config(s) already present "
            "(pass --overwrite to regenerate them)."
        )

    if pending:
        console.print(
            f"Generating {len(pending)} config(s) on {', '.join(devices)} -- "
            "expect tens of minutes each at production scale."
        )
        started = time.time()
        if len(devices) == 1 or len(pending) == 1:
            _build_shard(config, pending, devices[0], quiet=False)
        else:
            # Round-robin rather than contiguous slices: configurations all
            # cost about the same, so this balances the shards, and it keeps
            # each device's workload interleaved with the others' so an
            # early failure doesn't leave one contiguous seed range missing.
            ctx = multiprocessing.get_context("spawn")
            workers = []
            for i, device in enumerate(devices):
                shard = pending[i :: len(devices)]
                if not shard:
                    continue
                worker = ctx.Process(
                    target=_build_shard, args=(config, shard, device, True)
                )
                worker.start()
                workers.append((worker, device))
            for worker, _ in workers:
                worker.join()
            failed = [
                (device, worker.exitcode)
                for worker, device in workers
                if worker.exitcode != 0
            ]
            if failed:
                detail = ", ".join(f"{d} (exit {code})" for d, code in failed)
                raise RuntimeError(
                    f"run_build_ice_cache: worker(s) failed: {detail}. Any "
                    "config that did finish was saved -- re-run the same "
                    "command to generate only the missing ones."
                )
        console.print(f"Generated in {format_elapsed(time.time() - started)}.")
    else:
        console.print("Nothing to generate; every requested config already exists.")

    entries = _write_manifest(config, os.path.join(config.output_dir, "manifest.json"))
    console.print(f"Library now holds {len(entries)} config(s); wrote manifest.json.")

    # Reported as a spread rather than as a verdict: there is no calibrated
    # threshold separating a good sk_loss from a bad one at this scale (see
    # _report_config). An outlier relative to its own library's spread is
    # something a reader can act on; an absolute cutoff would not be.
    losses = [e["sk_loss"] for e in entries if e["sk_loss"] is not None]
    if len(losses) > 1:
        console.print(
            f"S(k) loss across the library: {min(losses):.4g} - {max(losses):.4g} "
            f"(median {sorted(losses)[len(losses) // 2]:.4g}). For reference, the "
            "bundled ice_data/ice_cache spans 0.0002 - 0.022 at n=256, dx=1.0."
        )

    if config.diagnostics:
        from specter.ice import IceBank

        save_path = os.path.join(config.output_dir, "diagnostics")
        bank = IceBank(config.output_dir, device=devices[0], progressbars=False)
        bank.plot_diagnostics(save_path=save_path, show=False)
        console.print(f"Saved diagnostics to {save_path}_energy.png / _sk.png.")
