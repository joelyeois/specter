"""Particle-stack generation pipeline: `ParticleStackConfig` in, .mrcs + .star out.

Moved here from `demo-scripts/generate_particle_stack.py` so the logic is a real,
importable function instead of being trapped inside a script's ``main()`` -- callable
from `specter.cli`, from plain Python, or from a notebook, with identical behavior.
"""

from __future__ import annotations

import logging
import os
import time

import torch

import specter
from specter import rotations
from specter.arrays import compute_nz
from specter.config import ParticleStackConfig
from specter.ice import resolve_icemaker
from specter.image import normalize_particles
from specter.imagegenerator import ImageGenerator
from specter.io import create_particle_starfile
from specter.pdb import PDB
from specter.potential import PotentialBuilder
from specter.progress import track

from ._common import (
    _console,
    _format_elapsed,
    _generate_multi,
    _generate_single,
    _parse_device,
    _save_exitwave_pair,
    _section,
    _uniform_sample,
)


def run_particle_stack(config: ParticleStackConfig) -> None:
    """
    Build a scattering potential, sample poses/CTF/dose, simulate a cryo-EM
    particle stack, and save it as ``.mrcs`` + ``.star``.

    Parameters
    ----------
    config : ParticleStackConfig
        Fully-resolved run configuration, e.g. from
        :func:`specter.config.load_config`, or constructed directly in
        Python. Poses, defocus, dose, coincidence radius, and potential
        scale are all randomly sampled per particle from the ranges given
        in ``config``.
    """
    specter.set_verbosity(logging.INFO)

    mode, device_target = _parse_device(config.device)
    t_start = time.perf_counter()

    # LOCAL_RANK is absent in the original process and set (0, 1, ...) in DDP workers
    is_main = "LOCAL_RANK" not in os.environ

    if config.seed is not None:
        specter.seed(config.seed)
    else:
        generated_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        specter.seed(generated_seed)
        if is_main:
            _console.print(f"[dim]No seed given -- using seed={generated_seed}[/dim]")

    # --- Building 3D scattering potential ---
    if is_main:
        _section("Building 3D scattering potential")
    pdb = PDB(
        config.pdb_code, assembly=config.assembly, savefolder=config.pdb_savefolder
    )

    # Convert cs from mm -> Angstrom (1 mm = 1e7 Angstrom)
    cs_angstrom = config.cs * 1e7

    # Only the main process (always global rank 0 for this single-node launcher)
    # builds V for real. Other DDP ranks hold a zero placeholder of the same
    # shape: Lightning's trainer.predict() calls _sync_module_states() before
    # the predict loop starts, which broadcasts rank 0's real buffer values
    # (V is a registered buffer) to every rank -- so building it per-rank would
    # just be wasted, redundant compute.
    if is_main:
        pb = PotentialBuilder(
            config.num_pixels,
            config.pixel_size,
            pdb.atomic_numbers,
            parameterization=config.potential_parameterization,
            conv_backend=config.conv_backend,
            mmcif_filepath=config.mmcif_filepath,
            atom_species=config.atom_species,
            shtyrov_params_path=config.shtyrov_params_path,
            rcut=config.rcut,
            periodic=config.periodic,
        ).to("cpu")
        with torch.no_grad():
            V = pb(pdb.coordinates, method=config.potential_method).clone()
    else:
        V = torch.zeros(config.num_pixels, config.num_pixels, config.num_pixels)

    # --- Sampling poses, defocus, and translations ---
    if is_main:
        _section("Sampling poses, defocus, and translations")
    n = config.n_particles

    quats = rotations.random_quaternion(n)

    defocus_A = _uniform_sample(config.defocus, n)
    astigmatism_magnitude = _uniform_sample(config.astigmatism, n)
    dfang = _uniform_sample(config.astigmatism_angle, n)
    dfv = defocus_A - astigmatism_magnitude
    phaseshift = _uniform_sample(config.phaseshift, n)
    tiltx = _uniform_sample(config.tiltx, n)
    tilty = _uniform_sample(config.tilty, n)
    trefoil1 = _uniform_sample(config.trefoil1, n)
    trefoil2 = _uniform_sample(config.trefoil2, n)
    ctf_params = {
        "cs": torch.tensor([cs_angstrom] * n),
        "dfu": defocus_A,
        "dfv": dfv,
        "dfang": dfang,
        "phaseshift": phaseshift,
        "tiltx": tiltx,
        "tilty": tilty,
        "trefoil1": trefoil1,
        "trefoil2": trefoil2,
    }

    anisomag_matrix = (
        config.anisomag_m00,
        config.anisomag_m01,
        config.anisomag_m10,
        config.anisomag_m11,
    )
    anisomag = (
        None
        if anisomag_matrix == (1.0, 0.0, 0.0, 1.0)
        else torch.tensor(
            [
                [config.anisomag_m00, config.anisomag_m01],
                [config.anisomag_m10, config.anisomag_m11],
            ]
        )
        .unsqueeze(0)
        .expand(n, 2, 2)
        .contiguous()
    )

    rlnOriginXAngst = 2 * (torch.rand(n) - 0.5) * config.shift
    rlnOriginYAngst = 2 * (torch.rand(n) - 0.5) * config.shift
    translations = torch.stack([rlnOriginXAngst, rlnOriginYAngst], dim=-1)

    dose = _uniform_sample(config.dose, n)
    coincidence_radius = _uniform_sample(config.coincidence_radius, n)
    potential_scale = _uniform_sample(config.potential_scale, n)

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
    num_frames = (
        config.num_frames if config.num_frames is not None else int(dose.mean().item())
    )
    cc_angstrom = config.cc * 1e7 if config.cc is not None else None

    # --- Ice ---
    # resolve_icemaker derives (n, nz) for a fresh RandomIcemaker itself, so it
    # always matches the particle volume V it gets blended into -- IceBank
    # (cache_dir=...) just loads small pre-generated coordinate files from
    # disk, cheap enough that every DDP rank can construct it independently
    # (no rank-0-builds-then-broadcasts dance needed, unlike V above).
    ice_nz = compute_nz(config.num_pixels, config.ice_thickness, config.pixel_size)
    icemaker = resolve_icemaker(
        ice_model,
        config.pixel_size,
        config.num_pixels,
        ice_nz,
        ice_cache_dir=config.ice_cache_dir,
        parameterization=config.ice_parameterization,
    )

    if icemaker is not None:
        icemaker_device = (
            f"cuda:{device_target[0]}" if mode == "multi" else device_target
        )
        if icemaker_device != "cpu":
            icemaker = icemaker.to(icemaker_device)

    model = ImageGenerator(
        V,
        config.pixel_size,
        quats,
        translations,
        ctf_params,
        config.energy,
        dose,
        anisomag=anisomag,
        icemaker=icemaker,
        ice_thickness=config.ice_thickness,
        ice_relax_steps=config.ice_relax_steps,
        ice_parameterization=config.ice_parameterization,
        scattering_model=config.scattering_model,
        aberration_model=config.aberration_model,
        noise_model=noise_model,
        klim=config.klim,
        ews_curvature_sign=config.ews_curvature_sign,
        alpha=config.alpha,
        crowd_min_distance=crowd_min_distance,
        crowd_max_distance_z=config.crowd_max_distance_z,
        crowd_max_distance_xy=config.crowd_max_distance_xy,
        crowd_chunk_size=config.crowd_chunk_size,
        crowd_method=config.crowd_method,
        crowd_n_points=config.crowd_n_points,
        crowd_seed=config.crowd_seed,
        crowd_move_to_cpu=config.crowd_move_to_cpu,
        water_air_interface=config.water_air_interface,
        pad_fft=config.pad_fft,
        rotate_mode=config.rotate_mode,
        detector_model=detector_model,
        verbose=False,
        coincidence_radius=coincidence_radius,
        num_frames=num_frames,
        potential_scale=potential_scale,
        convergence_angle=config.convergence_angle,
        cc=cc_angstrom,
        energy_spread=config.energy_spread,
        deltaV_V=config.deltaV_V,
        deltaI_I=config.deltaI_I,
        dose_envelope=config.dose_envelope,
        bfactor=config.bfactor,
    )

    if config.save_clean_exitwaves:
        model.save_clean_exitwaves = True  # type: ignore[assignment]

    # --- Generating images ---
    if mode == "multi":
        assert isinstance(device_target, list)
        if is_main:
            _section(f"Initializing multi-GPU on devices {device_target}")
        images, exitwaves, clean_exitwaves = _generate_multi(
            model,
            n,
            config.batchsize,
            device_target,
            config.output_dir,
            collect_exitwaves=config.save_exitwaves,
            collect_clean_exitwaves=config.save_clean_exitwaves,
        )
        if images is None:
            return  # worker rank -- rank 0 handles saving
    else:
        if is_main:
            _section(f"Generating images on {device_target}")
        model = model.to(device_target)
        images, exitwaves, clean_exitwaves = _generate_single(
            model,
            n,
            config.batchsize,
            track,
            collect_exitwaves=config.save_exitwaves,
            collect_clean_exitwaves=config.save_clean_exitwaves,
        )

    # --- Post-processing ---
    if is_main:
        _section("Post-processing")
    if config.normalize_particles:
        particles, _means, _stds = normalize_particles(images)
        particles = -particles
    else:
        particles = images

    if is_main:
        _section("Saving .mrcs + .star")
    create_particle_starfile(
        particles,
        rotations=quats,
        translations=translations,
        ctf_params=ctf_params,
        dx=config.pixel_size,
        energy=config.energy,
        alpha=config.alpha,
        filename=config.filename,
        folderpath=config.output_dir,
        dose_per_angstrom=dose,
        coincidence_radius=coincidence_radius,
        potential_scale=potential_scale,
    )

    if is_main:
        if exitwaves is not None:
            _section("Saving exit waves")
            _save_exitwave_pair(
                exitwaves,
                "exitwave",
                config.output_dir,
                config.filename,
                config.pad_fft,
                config.num_pixels,
            )

        if clean_exitwaves is not None:
            _section("Saving clean exit waves")
            _save_exitwave_pair(
                clean_exitwaves,
                "clean_exitwave",
                config.output_dir,
                config.filename,
                config.pad_fft,
                config.num_pixels,
            )

    elapsed = time.perf_counter() - t_start
    _console.print(f"\n[bold]Total time:[/bold] {_format_elapsed(elapsed)}")
