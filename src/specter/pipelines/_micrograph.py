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
from specter.config import MicrographConfig, parse_scalar_or_range, validate_config
from specter.ice import IceProfile, resolve_icemaker
from specter.imagegenerator import MicrographGenerator
from specter.io import create_micrograph_starfile
from specter.pdb import PDB
from specter.potential import PotentialBuilder
from specter.devices import resolve_available_device
from specter.settings import Camera, Envelopes, Propagation, bundle_from_config
from specter.progress import console, format_elapsed, section, track

from ._common import (
    _save_exitwave_pair,
    _tracked_output_dir,
    _uniform_sample,
)


def build_ice_profile(config: MicrographConfig) -> IceProfile | None:
    """
    Assemble an :class:`~specter.ice.IceProfile` from a config's flat fields.

    Returns ``None`` for the default flat, untilted case so the no-profile
    code path stays exactly as it was: a flat :class:`IceProfile` tapers both
    faces over ``softness`` and is therefore not bit-identical to a slab that
    simply fills its box.

    Parameters
    ----------
    config : MicrographConfig
        Run configuration. Reads ``ice_profile``, ``ice_thickness``,
        ``ice_thickness_range``, ``ice_profile_angle``, ``ice_hole_radius``,
        ``ice_rim_thickness``, ``ice_hole_offset`` and ``ice_tilt``.

    Returns
    -------
    IceProfile or None
    """
    if config.ice_profile == "flat" and config.ice_tilt == 0.0:
        return None

    thickness_range = (
        parse_scalar_or_range(config.ice_thickness_range)
        if config.ice_thickness_range is not None
        else None
    )
    return IceProfile(
        mode=config.ice_profile,
        mean_thickness=config.ice_thickness,
        thickness_range=thickness_range,
        angle=config.ice_profile_angle,
        hole_radius=config.ice_hole_radius,
        rim_thickness=config.ice_rim_thickness,
        hole_offset=parse_scalar_or_range(config.ice_hole_offset),
        tilt=config.ice_tilt,
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
    device = resolve_available_device(config.device)

    specter.set_verbosity(logging.INFO)
    t_start = time.perf_counter()

    if config.seed is not None:
        specter.seed(config.seed)
    else:
        generated_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        specter.seed(generated_seed)
        console.print(f"[dim]No seed given -- using seed={generated_seed}[/dim]")

    # --- Building 3D scattering potential ---
    section("Building 3D scattering potential")
    # Shtyrov fits scattering factors per bonded species, so derive the bond
    # topology from the structure -- without atom_species every atom silently
    # falls back to per-element Peng. The other parameterizations are
    # per-element and would only pay the extra gemmi pass for nothing.
    _derive_atom_species = config.scattering_factors == "shtyrov"
    pdb = PDB(
        config.pdb_source,
        assembly=config.assembly,
        pdb_cache_dir=config.pdb_cache_dir,
        compute_atom_species=_derive_atom_species,
        readd_hydrogens=config.readd_hydrogens,
        monomer_library_path=config.monomer_library_path,
    )

    cs_angstrom = config.cs * 1e7

    # Built on the compute device and moved to the host: the template is
    # one particle box (a few hundred MB at most), the analytic renderer is
    # a loop of small ops per atom, and on the CPU that loop cost 18 s on a
    # 128-core host against under 2 s on the device.
    pb = PotentialBuilder(
        config.n_pixels,
        config.pixel_size,
        pdb.atomic_numbers,
        parameterization=config.scattering_factors,
        atom_species=pdb.atom_species,
        b_factors=pdb.b_factors if config.use_deposited_bfactors else None,
    ).to(device)
    with torch.no_grad():
        V = pb(pdb.coordinates.to(device)).clone().cpu()

    n = config.n_micrographs

    ice_model = None if config.ice_model == "none" else config.ice_model
    crowd_min_distance = (
        None
        if config.crowd_min_distance == 0
        else config.crowd_min_distance
        if config.crowd_min_distance is not None
        else pdb.max_diameter
    )

    # --- Sampling per-micrograph parameters ---
    section("Sampling defocus, dose, and coincidence radius")
    defocus_angstrom = _uniform_sample(config.defocus, n)
    dose = _uniform_sample(config.dose, n)
    coincidence_radius = _uniform_sample(config.coincidence_radius, n)
    potential_scale = _uniform_sample(config.potential_scale, n)

    n_frames = (
        config.n_frames if config.n_frames is not None else int(dose.mean().item())
    )
    ctf_params = {
        "cs": torch.tensor([cs_angstrom] * n),
        "dfu": defocus_angstrom,
    }

    # --- Ice ---
    # Built once, upfront, on the target device -- regenerate_specimen() draws
    # a fresh, independently rotated/translated ice sample from the same
    # cached configs on every call. resolve_icemaker's nxy must be
    # config.micrograph_size (not config.n_pixels, the separate, usually much
    # smaller, particle-potential resolution): a RandomIcemaker has no tiling
    # support (unlike IceBank), so its own fixed (n, nz) must exactly match
    # the micrograph volume it gets blended into.
    ice_profile = build_ice_profile(config)
    ice_nz = (
        ice_profile.required_nz(config.micrograph_size, config.pixel_size, V.shape[0])
        if ice_profile is not None
        else compute_nz(V.shape[0], config.ice_thickness, config.pixel_size)
    )
    icemaker = resolve_icemaker(
        ice_model,
        config.pixel_size,
        config.micrograph_size,
        ice_nz,
        ice_cache_dir=config.ice_cache_dir,
        parameterization=config.bulk_scattering_factors,
    )
    if icemaker is not None:
        icemaker = icemaker.to(device)

    # Build once -- __init__ generates the first specimen.
    section("Building specimen and image generator")
    cc_angstrom = config.cc * 1e7 if config.cc is not None else None
    model = MicrographGenerator(
        V,
        config.micrograph_size,
        config.pixel_size,
        ctf_params,
        config.voltage,
        dose,
        propagation=bundle_from_config(Propagation, config),
        envelopes=bundle_from_config(Envelopes, config, cc=cc_angstrom),
        camera=bundle_from_config(Camera, config, n_frames=n_frames),
        icemaker=icemaker,
        ice_thickness=config.ice_thickness,
        ice_profile=ice_profile,
        bfactor=config.bfactor,
        crowd_min_distance=crowd_min_distance,
        crowd_max_distance_z=config.crowd_max_distance_z,
        water_air_interface=config.water_air_interface,
        sigma_frac=config.sigma_frac,
        peak_amplitude=config.peak_amplitude,
        baseline=config.baseline,
        packing_backend=config.packing_backend,
        atom_coordinates=pdb.coordinates,
        packing_gap=config.packing_gap,
        n_orientations=config.n_orientations,
        packing_max_retries=config.packing_max_retries,
        packing_stall_patience=config.packing_stall_patience,
        packing_seed=config.packing_seed,
        n_candidates=config.n_candidates,
        chunk_size=config.crowd_chunk_size,
        move_to_cpu=True,
        verbose=False,
        progressbars=False,
        coincidence_radius=coincidence_radius,
        potential_scale=potential_scale,
        save_clean_exitwaves=config.save_clean_exitwaves,
    ).to(device)

    # --- Generating images ---
    # Regenerate a fresh specimen (ice/crowding) for every micrograph after
    # the first (already built in __init__ above), then apply the i-th CTF.
    section(f"Generating {n} micrograph(s) on {device}")
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
    section("Post-processing")
    if config.normalize_micrographs:
        mean = images_t.mean(dim=(-2, -1), keepdim=True)
        std = images_t.std(dim=(-2, -1), keepdim=True)
        images_t = (images_t - mean) / std.clamp(min=1e-8)

    # --- Saving ---
    section("Saving .mrcs + .star")
    import mrcfile

    with _tracked_output_dir(config, "micrographs") as output_dir:
        os.makedirs(output_dir, exist_ok=True)
        mrcs_path = os.path.join(output_dir, config.filename + ".mrcs")
        with mrcfile.new(mrcs_path, overwrite=True) as mrc:
            mrc.set_data(images_t.numpy().astype("float32"))
        console.print(f"  [green]✓[/green] {mrcs_path}")

        create_micrograph_starfile(
            n,
            voltage=config.voltage,
            pixel_size=config.pixel_size,
            alpha=config.alpha,
            ctf_params=ctf_params,
            output_dir=output_dir,
            filename=config.filename,
            dose_per_angstrom=dose,
            coincidence_radius=coincidence_radius,
            potential_scale=potential_scale,
        )

        if exitwaves_t is not None:
            section("Saving exit waves")
            _save_exitwave_pair(
                exitwaves_t,
                "exitwave",
                output_dir,
                config.filename,
                config.pad_fft,
                config.micrograph_size,
            )

        if clean_exitwaves_t is not None:
            section("Saving clean exit waves")
            _save_exitwave_pair(
                clean_exitwaves_t,
                "clean_exitwave",
                output_dir,
                config.filename,
                config.pad_fft,
                config.micrograph_size,
            )

    elapsed = time.perf_counter() - t_start
    console.print(f"\n[bold]Total time:[/bold] {format_elapsed(elapsed)}")
