"""
The ice library on disk: how a config's coordinates are encoded, where the
bundled cache lives, and how `specter build ice` produces new configs with
`GradientSKIcemaker` under the standard MLBOP recipe.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import torch


from ..ice_data import ICE_CACHE_DIRNAME, bundled_ice_data

if TYPE_CHECKING:
    pass


#: Largest magnitude a signed 16-bit fixed-point coordinate index takes. One
#: short of `int16`'s true minimum (-32768), so the mapping stays symmetric
#: about zero and the box centre lands exactly on index 0.
_FIXED_POINT_SCALE = 32767

#: Value of a config's ``coord_encoding`` key when its positions are stored as
#: :func:`encode_positions`' fixed-point indices. Absent from a config file
#: means the older raw-float16 storage, which :meth:`IceBank._load_config`
#: still reads -- the bundled ``ice_data/ice_cache`` predates this key.
FIXED_POINT_ENCODING = "int16_fixed"


def encode_positions(positions: torch.Tensor, box_L: float) -> torch.Tensor:
    """
    Quantize coordinates to signed 16-bit fixed-point indices for storage.

    Coordinates are bounded (wrapped into ``[-box_L/2, box_L/2)``), which is
    the precondition fixed-point needs and floating point cannot exploit.
    Storing them as raw ``float16`` instead spends 5 of 16 bits on an exponent
    covering a dynamic range these values never use, and leaves *relative*
    precision on an absolute quantity: the spacing between representable
    ``float16`` values grows with distance from the origin, reaching 0.0625 A
    across the outer octave of a 256 A cell -- where, since volume grows as
    r^3, half of all coordinates sit. This maps the box onto a uniform grid
    instead, so every coordinate gets the same ``box_L / (2 * 32767)``
    resolution (0.0039 A at ``box_L=256``) for the same two bytes.

    Measured on a converged 256 A config: raw ``float16`` storage costs 0.0137
    A RMS of coordinate accuracy and inflates the config's S(k) loss ~3800x
    (1.2e-4 -> 0.456), while this encoding costs 0.0011 A and ~13x (-> 0.0016).
    The rendered consequence of that difference is small -- 3.7% vs 0.3%
    relative RMS on the potential, both far below shot noise at any realistic
    dose -- so this is about not discarding the S(k) fidelity the optimiser
    exists to produce, rather than about visible image quality.

    ``int16`` rather than ``uint16``: ``torch.save`` cannot serialize
    ``torch.uint16`` as of torch 2.5.1 (``KeyError: 'dtype torch.uint16 is
    not recognized'``). A symmetric signed map has identical resolution.

    Parameters
    ----------
    positions : torch.Tensor
        Coordinates in Å, shape (N, 3), within ``[-box_L/2, box_L/2]``.
    box_L : float
        Cubic cell side length in Å.

    Returns
    -------
    torch.Tensor
        ``int16`` grid indices, same shape. Decode with
        :func:`decode_positions`.
    """
    scaled = positions.double() / (box_L / 2) * _FIXED_POINT_SCALE
    return scaled.round().clamp(-_FIXED_POINT_SCALE, _FIXED_POINT_SCALE).to(torch.int16)


def decode_positions(indices: torch.Tensor, box_L: float) -> torch.Tensor:
    """
    Reconstruct float32 coordinates from :func:`encode_positions`' indices.

    Parameters
    ----------
    indices : torch.Tensor
        ``int16`` grid indices, shape (N, 3).
    box_L : float
        Cubic cell side length in Å, from the config's ``box_L`` key.

    Returns
    -------
    torch.Tensor
        Coordinates in Å, float32.
    """
    return (indices.double() / _FIXED_POINT_SCALE * (box_L / 2)).float()


def default_ice_cache_dir() -> str:
    """
    Path to the bundled ice cache, so a standard user never needs to run
    :class:`GradientSKIcemaker` themselves -- see :func:`build_ice_cache` for
    how it was produced.

    Resolved through :mod:`importlib.resources` (see
    :func:`specter.ice_data.bundled_ice_data`), which is why the library lives
    at ``src/specter/ice_data/ice_cache`` rather than at the repository root:
    the data ships with the package and is found the same way regardless of
    install layout.

    Returns
    -------
    str
        Absolute path to the bundled ``ice_cache`` directory.
    """
    return str(bundled_ice_data(ICE_CACHE_DIRNAME))


def ice_config_filename(seed: int) -> str:
    """
    Filename a cached ice config generated with ``seed`` is stored under.

    Named after the seed rather than a position in the generation batch so
    that extending an existing cache is safe: a second run started at a
    higher ``seed_start`` writes new files instead of overwriting the
    earlier run's, and re-running an identical request is idempotent (it
    reproduces the same seeds, hence the same filenames). For the usual
    ``seed_start=0`` this is the same ``config_000.pt``, ``config_001.pt``,
    ... sequence the bundled ``ice_data/ice_cache`` uses.

    Parameters
    ----------
    seed : int
        Seed the config was generated with.

    Returns
    -------
    str
        Bare filename, e.g. ``"config_007.pt"``.
    """
    return f"config_{seed:03d}.pt"


def build_one_ice_config(
    save_path: str,
    n: int = 256,
    dx: float = 1.0,
    n_steps: int = 250,
    seed: int = 0,
    device: str | torch.device = "cuda",
    progressbars: bool = True,
) -> dict:
    """
    Generate a single ice coordinate config and save it (fixed-point) to
    ``save_path``.

    One unit of work for :func:`build_ice_cache` and for ``specter build
    ice``, which differ only in how they schedule these calls. A fresh
    :class:`GradientSKIcemaker` run under the standard MLBOP recipe
    (``rep_strength=0, mlbop_strength=0.5, mlbop_target=-0.413``,
    ``tol=1e-3``/``patience=10``). That recipe is deliberately not
    parameterised: ``mlbop_target=-0.413`` is a measured property of real
    LDA-80K MD ice rather than a preference, and configs generated under
    different recipes are different phases of ice that
    :class:`IceBank` would then draw from interchangeably.

    Parameters
    ----------
    save_path : str
        Full path of the ``.pt`` file to write. Its parent directory must
        already exist.
    n : int, optional
        Number of voxels along each side (box size = ``n * dx``). Cubic
        only: :class:`IceBank` stores one scalar ``box_L`` per config and
        filters candidate configs against it, so a non-cubic source cell
        has no representation. Default 256.
    dx : float, optional
        Voxel size in Å. Default 1.0.
    n_steps : int, optional
        L-BFGS step ceiling (an upper bound -- ``tol``/``patience`` still
        stop a plateaued run early). Default 250, about 6000 loss
        evaluations at :meth:`GradientSKIcemaker.optimize`'s default
        ``max_iter``; see that docstring for how the budget was chosen.
    seed : int, optional
        Global torch seed set before initialisation, making this config
        reproducible. Default 0.
    device : str or torch.device, optional
        Computation device. Default ``"cuda"``.
    progressbars : bool, optional
        Whether to show a progress bar over optimisation steps. Default True.

    Returns
    -------
    dict
        The saved metadata, without ``positions``: ``box_L``, ``n``, ``dx``,
        ``seed``, ``coord_encoding``, ``n_steps``, ``n_steps_actual``,
        ``wall_time``, ``stopped_early``, ``energy``, ``sk_loss``, ``recipe``.
        ``sk_loss`` is measured on the coordinates as they will be read back
        (encoded then decoded), so it describes the file rather than the
        optimiser's in-memory state -- see the comment at its computation for
        why neither alternative is trustworthy.

    Notes
    -----
    Coordinates are written as fixed-point indices via
    :func:`encode_positions`, not as raw floats. Configs predating that
    encoding (the bundled ``ice_data/ice_cache``) stay readable; see
    :meth:`IceBank._load_config`.
    """
    from ._gradient import GradientSKIcemaker

    # The standard recipe, bound to names so the values actually optimised
    # with and the values recorded in the saved metadata below cannot drift
    # apart.
    rep_strength, mlbop_strength, mlbop_target = 0.0, 0.5, -0.413
    tol, patience = 1e-3, 10

    started = time.perf_counter()
    torch.manual_seed(seed)
    gd = GradientSKIcemaker(n=n, dx=dx, device=device, progressbars=progressbars)
    gd.init_random()
    history_size, max_iter, prerelax_steps = 100, 25, 30
    history = gd.optimize(
        n_steps=n_steps,
        record_every=n_steps,
        rep_strength=rep_strength,
        mlbop_strength=mlbop_strength,
        mlbop_target=mlbop_target,
        tol=tol,
        patience=patience,
        history_size=history_size,
        max_iter=max_iter,
        prerelax_steps=prerelax_steps,
    )
    energy = gd.mlbop_energy()
    assert gd.positions is not None

    # Measure S(k) fidelity on the coordinates as they will be READ BACK --
    # encoded, then decoded -- rather than on the float32 ones held in memory,
    # and never from `history["sk_loss"]`. Both alternatives misreport:
    #
    # - `history` records at `step % record_every == 0` plus one final entry
    #   only when tol/patience triggers, so at `record_every=n_steps` a run
    #   that uses its whole budget has exactly ONE record -- step 0, the
    #   pre-optimisation value of the random initialisation. The bundled
    #   `ice_data/ice_cache` carries that artifact: its ten budget-exhausting
    #   configs all record ~5e4, which is simply what a fresh random init
    #   scores at this size, while their stored coordinates score ~0.4-2.0.
    # - Storage quantization is not negligible in this metric even at the
    #   fixed-point resolution used here (~13x on S(k) loss; the raw float16
    #   this replaced cost ~3800x -- see `encode_positions`). A number
    #   describing coordinates nobody can load back is not a quality record.
    box_L = n * dx
    positions = encode_positions(gd.positions, box_L)
    with torch.no_grad():
        sk_loss, _ = gd._sk_loss(
            decode_positions(positions, box_L).to(gd.device),
            rep_strength=0.0,
            mlbop_strength=0.0,
        )

    metadata = {
        "box_L": box_L,
        "n": n,
        "dx": dx,
        "seed": seed,
        # How `positions` is stored, read by `IceBank._load_config`. Explicit
        # rather than inferred from dtype so a future encoding can be added
        # without either format having to guess about the other.
        "coord_encoding": FIXED_POINT_ENCODING,
        "n_steps": n_steps,
        # The search settings, so a library's convergence level is on
        # record: two libraries built with different optimiser settings on
        # the same recipe are converged to different degrees, and IceBank
        # should not draw from both (regenerate a library whole).
        "optimizer": {
            "history_size": history_size,
            "max_iter": max_iter,
            "prerelax_steps": prerelax_steps,
        },
        # Number of outer L-BFGS steps actually taken, and the seconds they
        # took. Recorded under the same keys the bundled ice_data/ice_cache
        # uses, so cost per step stays derivable per config -- that ratio,
        # not wall time alone, is what transfers to other hardware and cell
        # sizes. Taken from `history["step"]` only when tol/patience fired,
        # since that is the sole case where a final entry is appended; a run
        # that used its whole budget has just the step-0 record (the same
        # recording quirk that makes `history["sk_loss"]` untrustworthy
        # above), and by definition took all `n_steps` of it.
        "n_steps_actual": (
            history["step"][-1] + 1 if history["stopped_early"] else n_steps
        ),
        "wall_time": time.perf_counter() - started,
        "stopped_early": history["stopped_early"],
        "energy": energy,
        "sk_loss": sk_loss.item(),
        # Stored per config, under the same key the bundled ice_data/ice_cache
        # entries use: a cache directory can hold configs from several runs,
        # and which recipe produced one determines which phase of ice it is.
        "recipe": {
            "rep_strength": rep_strength,
            "mlbop_strength": mlbop_strength,
            "mlbop_target": mlbop_target,
            "tol": tol,
            "patience": patience,
        },
    }
    torch.save({"positions": positions, **metadata}, save_path)
    return metadata


def build_ice_cache(
    cache_dir: str,
    n_configs: int,
    n: int = 256,
    dx: float = 1.0,
    n_steps: int = 250,
    device: str | torch.device = "cuda",
    seed_start: int = 0,
    progressbars: bool = True,
) -> None:
    """
    Generate ``n_configs`` independent ice coordinate configs and save
    them (fixed-point) to ``cache_dir`` for later use with :class:`IceBank`.

    One file per config, via :func:`build_one_ice_config`. This is the
    expensive, one-time cost the whole point of :class:`IceBank` is to
    amortize -- expect on the order of tens of minutes per config at
    production scale (``n=256, dx=1.0``), so this is meant to be run once
    (or occasionally, to extend/refresh the library), not per-session.

    This function runs all ``n_configs`` sequentially, on a single
    device, in one process, and overwrites any config already present for
    the same seed. Prefer ``specter build ice`` (equivalently
    :func:`specter.pipelines.run_build_ice_cache`) for a large generation
    run: it shards configs across several GPUs, skips configs the cache
    already has so an interrupted run resumes, and records a manifest of
    the convergence quality actually achieved.

    ``n_steps=250`` is about 6000 loss evaluations at
    :meth:`GradientSKIcemaker.optimize`'s default ``max_iter``, the budget
    its settings were chosen on (loss 0.024 at 3000 evaluations, 0.022 at
    6000, at n=256). Since these configs are permanent, shared assets, the
    tail of that curve is worth the one-time cost. ``tol``/``patience``
    still apply on top, so a config that plateaus early stops early rather
    than burning the full budget.

    Parameters
    ----------
    cache_dir : str
        Directory to write config files to (named by
        :func:`ice_config_filename`). Created if it doesn't exist.
    n_configs : int
        Number of independent configs to generate.
    n : int, optional
        Number of voxels along each side (box size = ``n * dx``). Default 256.
    dx : float, optional
        Voxel size in Å. Default 1.0.
    n_steps : int, optional
        L-BFGS step ceiling per config (upper bound -- see ``tol``).
        Default 250.
    device : str or torch.device, optional
        Computation device. Default ``"cuda"``.
    seed_start : int, optional
        First seed used; config ``i`` uses seed ``seed_start + i``. Default 0.
    progressbars : bool, optional
        Whether to show progress bars during generation. Default True.
    """
    os.makedirs(cache_dir, exist_ok=True)
    for i in range(n_configs):
        seed = seed_start + i
        build_one_ice_config(
            os.path.join(cache_dir, ice_config_filename(seed)),
            n=n,
            dx=dx,
            n_steps=n_steps,
            seed=seed,
            device=device,
            progressbars=progressbars,
        )
