"""Micrograph generation pipeline: `MicrographConfig` in, .mrcs + .star out.

Moved here from `demo-scripts/generate_micrograph.py` so the logic is a real,
importable function instead of being trapped inside a script's ``main()`` --
callable from `specter.cli`, from plain Python, or from a notebook, with
identical behavior.

Single-device only (unlike `run_particle_stack`/`run_tilt_series`, no
multi-GPU DDP dispatch): each micrograph needs its own freshly regenerated
specimen (ice/crowding) between forward passes
(`MicrographGenerator.regenerate_specimen`), and a single micrograph is
already the unit of GPU-memory-bound work at `micrograph_size` resolution,
so there's no batching (and hence no DDP sharding) to speed up the way
particle stacks batch many small boxes together.
"""

from __future__ import annotations

import logging
import os
import time

import torch

import specter
from specter.arrays import compute_nz
from specter.config import MicrographConfig, validate_config
from specter.ice import resolve_icemaker
from specter.imagegenerator import MicrographGenerator
from specter.io import create_micrograph_starfile
from specter.pdb import PDB
from specter.potential import PotentialBuilder
from specter.progress import track

from ._common import (
    _console,
    _format_elapsed,
    _save_exitwave_pair,
    _section,
    _uniform_sample,
)


def run_micrograph(config: MicrographConfig) -> None:
    """
    Build a scattering potential, simulate one or more cryo-EM micrographs
    (fresh ice/crowding per micrograph, varying CTF), and save them as
    ``.mrcs`` + ``.star``.

    Parameters
    ----------
    config : MicrographConfig
        Fully-resolved run configuration, e.g. from
        :func:`specter.config.load_config`, or constructed directly in
        Python. Defocus, dose, coincidence radius, and potential scale are
        randomly sampled per micrograph from the ranges given in ``config``.
    """
    validate_config(config)

    specter.set_verbosity(logging.INFO)
    t_start = time.perf_counter()

    if config.seed is not None:
        specter.seed(config.seed)
    else:
        generated_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        specter.seed(generated_seed)
        _console.print(f"[dim]No seed given -- using seed={generated_seed}[/dim]")

    # --- Building 3D scattering potential ---
    _section("Building 3D scattering potential")
    pdb = PDB(
        config.pdb_code, assembly=config.assembly, savefolder=config.pdb_savefolder
    )

    cs_angstrom = config.cs * 1e7

    pb = PotentialBuilder(config.n_pixels, config.pixel_size, pdb.atomic_numbers).to(
        "cpu"
    )
    with torch.no_grad():
        V = pb(pdb.coordinates).clone()

    n = config.n_micrographs

    noise_model = None if config.noise_model == "none" else config.noise_model
    ice_model = None if config.ice_model == "none" else config.ice_model
    detector_model = None if config.detector_model == "none" else config.detector_model
    crowd_min_distance = (
        None
        if config.crowd_min_distance == 0
        else config.crowd_min_distance
        if config.crowd_min_distance is not None
        else pdb.max_diameter
    )

    # --- Sampling per-micrograph parameters ---
    _section("Sampling defocus, dose, and coincidence radius")
    defocus_A = _uniform_sample(config.defocus, n)
    dose = _uniform_sample(config.dose, n)
    coincidence_radius = _uniform_sample(config.coincidence_radius, n)
    potential_scale = _uniform_sample(config.potential_scale, n)

    n_frames = (
        config.n_frames if config.n_frames is not None else int(dose.mean().item())
    )
    ctf_params = {
        "cs": torch.tensor([cs_angstrom] * n),
        "dfu": defocus_A,
    }

    # --- Ice ---
    # Built once, upfront, on the target device -- regenerate_specimen() draws
    # a fresh, independently rotated/translated ice sample from the same
    # cached configs on every call. resolve_icemaker's nxy must be
    # config.micrograph_size (not config.n_pixels, the separate, usually much
    # smaller, particle-potential resolution): a RandomIcemaker has no tiling
    # support (unlike IceBank), so its own fixed (n, nz) must exactly match
    # the micrograph volume it gets blended into.
    ice_nz = compute_nz(V.shape[0], config.ice_thickness, config.pixel_size)
    icemaker = resolve_icemaker(
        ice_model,
        config.pixel_size,
        config.micrograph_size,
        ice_nz,
        ice_cache_dir=config.ice_cache_dir,
    )
    if icemaker is not None:
        icemaker = icemaker.to(config.device)

    # Build once -- __init__ generates the first specimen.
    _section("Building specimen and image generator")
    cc_angstrom = config.cc * 1e7 if config.cc is not None else None
    model = MicrographGenerator(
        V,
        config.micrograph_size,
        config.pixel_size,
        ctf_params,
        config.voltage,
        dose,
        icemaker=icemaker,
        ice_thickness=config.ice_thickness,
        scattering_model=config.scattering_model,
        aberration_model=config.aberration_model,
        noise_model=noise_model,
        klim=None,
        alpha=config.alpha,
        crowd_min_distance=crowd_min_distance,
        crowd_max_distance_z=config.crowd_max_distance_z,
        water_air_interface=config.water_air_interface,
        pad_fft=config.pad_fft,
        chunk_size=config.specimen_chunk_size,
        move_to_cpu=True,
        detector_model=detector_model,
        verbose=False,
        progressbars=False,
        coincidence_radius=coincidence_radius,
        n_frames=n_frames,
        potential_scale=potential_scale,
        save_clean_exitwaves=config.save_clean_exitwaves,
        convergence_angle=config.convergence_angle,
        cc=cc_angstrom,
        energy_spread=config.energy_spread,
        deltaV_V=config.deltaV_V,
        deltaI_I=config.deltaI_I,
        dose_envelope=config.dose_envelope,
    ).to(config.device)

    # --- Generating images ---
    # Regenerate a fresh specimen (ice/crowding) for every micrograph after
    # the first (already built in __init__ above), then apply the i-th CTF.
    _section(f"Generating {n} micrograph(s) on {config.device}")
    images: list[torch.Tensor] = []
    exitwaves: list[torch.Tensor] | None = [] if config.save_exitwaves else None
    clean_exitwaves: list[torch.Tensor] | None = (
        [] if config.save_clean_exitwaves else None
    )

    for i in track(range(n), description="Generating micrographs"):
        if i > 0:
            model.regenerate_specimen()
        with torch.no_grad():
            img = model(torch.tensor([i]))
        images.append(img.detach().cpu())
        if exitwaves is not None:
            exitwaves.append(model.exitwaves.detach().cpu())
        if clean_exitwaves is not None:
            clean_exitwaves.append(model.clean_exitwaves.detach().cpu())

    images_t = torch.concat(images, dim=0)
    exitwaves_t = torch.concat(exitwaves, dim=0) if exitwaves is not None else None
    clean_exitwaves_t = (
        torch.concat(clean_exitwaves, dim=0) if clean_exitwaves is not None else None
    )

    # --- Post-processing ---
    _section("Post-processing")
    if config.normalize_micrographs:
        mean = images_t.mean(dim=(-2, -1), keepdim=True)
        std = images_t.std(dim=(-2, -1), keepdim=True)
        images_t = (images_t - mean) / std.clamp(min=1e-8)

    # --- Saving ---
    _section("Saving .mrcs + .star")
    import mrcfile

    os.makedirs(config.output_dir, exist_ok=True)
    mrcs_path = os.path.join(config.output_dir, config.filename + ".mrcs")
    with mrcfile.new(mrcs_path, overwrite=True) as mrc:
        mrc.set_data(images_t.numpy().astype("float32"))
    _console.print(f"  [green]✓[/green] {mrcs_path}")

    create_micrograph_starfile(
        n,
        voltage=config.voltage,
        pixel_size=config.pixel_size,
        alpha=config.alpha,
        ctf_params=ctf_params,
        folderpath=config.output_dir,
        filename=config.filename,
        dose_per_angstrom=dose,
        coincidence_radius=coincidence_radius,
        potential_scale=potential_scale,
    )

    if exitwaves_t is not None:
        _section("Saving exit waves")
        _save_exitwave_pair(
            exitwaves_t,
            "exitwave",
            config.output_dir,
            config.filename,
            config.pad_fft,
            config.micrograph_size,
        )

    if clean_exitwaves_t is not None:
        _section("Saving clean exit waves")
        _save_exitwave_pair(
            clean_exitwaves_t,
            "clean_exitwave",
            config.output_dir,
            config.filename,
            config.pad_fft,
            config.micrograph_size,
        )

    elapsed = time.perf_counter() - t_start
    _console.print(f"\n[bold]Total time:[/bold] {_format_elapsed(elapsed)}")
