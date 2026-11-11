"""Micrograph generation pipeline: `MicrographConfig` in, .mrcs + .star out.

An importable function behind `specter simulate micrograph`, callable from
`specter.cli`, from plain Python, or from a notebook, with identical
behavior.

Single-device only (unlike `run_particle_stack`): each micrograph is its own
specimen draw (`Field.draw`), and a single micrograph is already the unit of
GPU-memory-bound work at `micrograph_size` resolution, so there is no
batching to shard the way particle stacks batch many small boxes together.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict

import torch

import specter
from specter.arrays import compute_nz
from specter.config import MicrographConfig, parse_scalar_or_range, validate_config
from specter.ice import IceProfile, resolve_icemaker
from specter.imaging import CTF, Exposure, Field, Structure
from specter.imaging._data import sample_uniform
from specter.imaging._stages import read_dose_weights
from specter.io import create_micrograph_starfile, write_mrc
from specter.devices import resolve_available_device
from specter.settings import (
    Crowding,
    Ice,
    Packing,
    Propagation,
    bundle_from_config,
)
from specter.progress import console, section, track

from ._common import (
    _Microscope,
    _absorption_alpha,
    _ice_model,
    _n_frames,
    _crowd_min_distance,
    _mm_to_angstrom,
    _normalize_images,
    _print_total_time,
    _seed_or_draw,
    _save_exitwave_pair,
    _tracked_output_dir,
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
        Python. Defocus, dose, and coincidence radius are randomly sampled
        per micrograph from the ranges given in ``config``.
    """
    validate_config(config)
    device = resolve_available_device(config.device)

    specter.set_verbosity(logging.INFO)
    t_start = time.perf_counter()

    run_seed = _seed_or_draw(config.seed)

    # --- Building 3D scattering potential ---
    section("Building 3D scattering potential")
    structure = Structure.from_pdb(
        config.pdb_source,
        assembly=config.assembly,
        scattering_factors=config.scattering_factors,
        pdb_cache_dir=config.pdb_cache_dir,
        readd_hydrogens=config.readd_hydrogens,
        monomer_library_path=config.monomer_library_path,
    )
    pdb = structure.pdb

    cs_angstrom = _mm_to_angstrom(config.cs)
    # The mean-free-path model replaces the amplitude-contrast ratio as the
    # source of the imaginary potential; keeping both would absorb twice.
    alpha = _absorption_alpha(config, config.alpha)

    # Built on the compute device and moved to the host: the template is
    # one particle box (a few hundred MB at most), the analytic renderer is
    # a loop of small ops per atom, and on the CPU that loop cost 18 s on a
    # 128-core host against under 2 s on the device.
    V = structure.render(
        config.pixel_size,
        config.n_pixels,
        use_deposited_bfactors=config.use_deposited_bfactors,
        device=device,
    ).potential.cpu()

    n = config.n_micrographs

    ice_model = _ice_model(config)
    crowd_min_distance = _crowd_min_distance(
        config.crowd_min_distance, pdb.max_diameter
    )

    # --- Sampling per-micrograph parameters ---
    section("Sampling defocus, dose, and coincidence radius")
    defocus_angstrom = sample_uniform(config.defocus, n)
    dose = sample_uniform(config.dose, n)
    coincidence_radius = sample_uniform(config.coincidence_radius, n)

    n_frames = _n_frames(config, dose)
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
    ice = bundle_from_config(
        Ice,
        config,
        prefix="ice_",
        profile=ice_profile,
        parameterization=config.bulk_scattering_factors,
    )
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

    section("Building the specimen")
    field = Field(
        V,
        config.pixel_size,
        config.micrograph_size,
        molecular_mass=structure.molecular_mass,
        ice=ice,
    ).pack(
        bundle_from_config(
            Crowding,
            config,
            prefix="crowd_",
            min_distance=crowd_min_distance,
            water_air_interface=config.water_air_interface,
            sigma_frac=config.sigma_frac,
            peak_amplitude=config.peak_amplitude,
            baseline=config.baseline,
        ),
        packing=bundle_from_config(
            Packing,
            config,
            prefix="packing_",
            n_orientations=config.n_orientations,
            n_candidates=config.n_candidates,
        ),
        atom_coordinates=pdb.coordinates,
    )
    if icemaker is not None:
        field = field.solvate(ice, icemaker=icemaker)
    microscope = _Microscope.from_config(config, n_frames=n_frames)
    if microscope.damages_specimen:
        field = field.damage()
    weights, weights_max_frequency = read_dose_weights(microscope.camera)
    exposure = Exposure(
        dose,
        n_frames=n_frames,
        weights=weights,
        weights_max_frequency=weights_max_frequency,
    )
    propagation = asdict(bundle_from_config(Propagation, config, alpha=alpha))
    ctf = CTF(ctf_params)

    # --- Generating images ---
    # A fresh specimen (ice/crowding) for every micrograph, each drawn from
    # its own seed, then the i-th CTF.
    section(f"Generating {n} micrograph(s) on {device}")
    images: list[torch.Tensor] = []
    exitwaves: list[torch.Tensor] | None = [] if config.save_exitwaves else None
    clean_exitwaves: list[torch.Tensor] | None = (
        [] if config.save_clean_exitwaves else None
    )

    for i in track(range(n), description="Generating micrographs"):
        one = slice(i, i + 1)
        with torch.no_grad():
            sample = field.draw(
                voltage=config.voltage,
                exposure=exposure[one],
                seed=run_seed,
                index=i,
                keep_dry=config.save_clean_exitwaves,
                objective_aperture=config.objective_aperture,
                device=device,
                **propagation,
            )
            beam = dict(
                voltage=config.voltage,
                ids=torch.tensor([i]),
                exposure=exposure[one],
                device=device,
                **propagation,
            )
            waves = sample.scatter(**beam)
            if exitwaves is not None:
                exitwaves.append(waves.pixels().cpu())
            if clean_exitwaves is not None:
                clean_exitwaves.append(sample.clean().scatter(**beam).pixels().cpu())
            image = microscope.image(
                waves,
                ctf[one],
                exposure[one],
                coincidence_radius=coincidence_radius[one],
                seed=run_seed,
            )
            del sample, waves
        images.append(image.array.cpu())

    images_t = torch.concat(images, dim=0)
    exitwaves_t = torch.concat(exitwaves, dim=0) if exitwaves is not None else None
    clean_exitwaves_t = (
        torch.concat(clean_exitwaves, dim=0) if clean_exitwaves is not None else None
    )

    # --- Post-processing ---
    section("Post-processing")
    if config.normalize_micrographs:
        images_t = _normalize_images(images_t)

    # --- Saving ---
    section("Saving .mrcs + .star")

    with _tracked_output_dir(config, "micrographs") as output_dir:
        os.makedirs(output_dir, exist_ok=True)
        mrcs_path = os.path.join(output_dir, config.filename + ".mrcs")
        write_mrc(mrcs_path, images_t, config.pixel_size)
        console.print(f"  [green]✓[/green] {mrcs_path}")

        create_micrograph_starfile(
            n,
            voltage=config.voltage,
            pixel_size=config.pixel_size,
            alpha=alpha,
            ctf_params=ctf_params,
            output_dir=output_dir,
            filename=config.filename,
            dose_per_angstrom=dose,
            coincidence_radius=coincidence_radius,
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
                config.pixel_size,
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
                config.pixel_size,
            )

    _print_total_time(t_start)
