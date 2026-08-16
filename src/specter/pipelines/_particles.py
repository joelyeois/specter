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
from specter.io import (
    create_particle_starfile,
    extract_parameters_from_csfile,
    extract_parameters_from_starfile,
)
from specter.memory import (
    available_memory_bytes,
    estimate_peak_bytes,
    recommend_batchsize,
)
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
        in ``config``, unless ``config.cs_path``/``config.star_path`` points
        at a real dataset to take poses and CTF from instead.
    """
    if config.cs_path is not None and config.star_path is not None:
        raise ValueError(
            "run_particle_stack: cs_path and star_path can't both be set -- "
            "pass exactly one dataset to take poses/CTF from, or neither to "
            "sample them synthetically."
        )

    specter.set_verbosity(logging.INFO)

    mode, device_target = _parse_device(config.device)
    t_start = time.perf_counter()

    # LOCAL_RANK is absent in the original process and set (0, 1, ...) in DDP workers
    is_main = "LOCAL_RANK" not in os.environ

    if config.seed is not None:
        # Stochastic stages (ice crop selection, Poisson noise) draw from the
        # global RNG stream inside each forward pass, so the batch boundaries
        # decide which draw lands on which particle -- change the batch size
        # and the same seed gives a different stack. That is harmless when the
        # batch size is pinned, but `batchsize="auto"` is sized to whatever
        # memory happens to be free on the device at run time, so it can differ
        # between machines, GPUs, or two runs on a busy node. Rather than
        # quietly hand back an unreproducible stack, refuse the combination.
        if config.batchsize == "auto":
            raise ValueError(
                "seed is set but batchsize='auto': 'auto' sizes the batch to "
                "the memory free on the device at run time, and batching "
                "changes which random draw reaches which particle, so the "
                "same seed would not reproduce the same stack on another "
                "machine. Pin an integer batchsize (e.g. batchsize=32, or "
                "--batchsize 32) to make the run reproducible, or drop the "
                "seed to accept a non-reproducible run."
            )
        specter.seed(config.seed)
    else:
        generated_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        specter.seed(generated_seed)
        if is_main:
            _console.print(f"[dim]No seed given -- using seed={generated_seed}[/dim]")

    # --- Building 3D scattering potential ---
    if is_main:
        _section("Building 3D scattering potential")
    # Shtyrov fits scattering factors per bonded species, so derive the
    # bond topology from the structure unless the config supplies its own
    # atom_species list. Other parameterizations are per-element and would
    # only pay the extra gemmi pass for nothing.
    _derive_atom_species = (
        config.potential_parameterization == "shtyrov"
        and config.atom_species is None
        and config.mmcif_filepath is None
    )
    pdb = PDB(
        config.pdb_code,
        assembly=config.assembly,
        savefolder=config.pdb_savefolder,
        compute_atom_species=_derive_atom_species,
    )

    # pixel_size/voltage/alpha come from the dataset when cs_path/star_path is
    # set -- extract once, up front, since PotentialBuilder below needs the
    # resolved pixel_size (matches ImageGeneratorFromCoordinates' notebook
    # counterpart). Both extractors return the same 10-tuple, so everything
    # downstream is agnostic to which file format it came from.
    dataset_particles = None
    if config.cs_path is not None:
        dataset_particles = extract_parameters_from_csfile(
            config.cs_path, n_particles=config.n_particles
        )
    elif config.star_path is not None:
        dataset_particles = extract_parameters_from_starfile(
            config.star_path, n_particles=config.n_particles
        )

    if dataset_particles is not None:
        pixel_size = dataset_particles[1].item()
        voltage = dataset_particles[0].item()
        alpha = dataset_particles[2].item()
    else:
        pixel_size = config.pixel_size
        voltage = config.voltage
        alpha = config.alpha

    # Convert cs from mm -> Angstrom (1 mm = 1e7 Angstrom); unused when
    # cs_path/star_path is set, since Cs then comes per-particle from the file.
    cs_angstrom = config.cs * 1e7

    # Only the main process (always global rank 0 for this single-node launcher)
    # builds V for real. Other DDP ranks hold a zero placeholder of the same
    # shape: Lightning's trainer.predict() calls _sync_module_states() before
    # the predict loop starts, which broadcasts rank 0's real buffer values
    # (V is a registered buffer) to every rank -- so building it per-rank would
    # just be wasted, redundant compute.
    if is_main:
        pb = PotentialBuilder(
            config.n_pixels,
            pixel_size,
            pdb.atomic_numbers,
            parameterization=config.potential_parameterization,
            conv_backend=config.conv_backend,
            mmcif_filepath=config.mmcif_filepath,
            atom_species=config.atom_species or pdb.atom_species,
            shtyrov_params_path=config.shtyrov_params_path,
            rcut=config.rcut,
            periodic=config.periodic,
        ).to("cpu")
        with torch.no_grad():
            V = pb(pdb.coordinates, method=config.potential_method).clone()
    else:
        V = torch.zeros(config.n_pixels, config.n_pixels, config.n_pixels)

    # --- Sampling poses, defocus, and translations ---
    if is_main:
        _section("Sampling poses, defocus, and translations")

    if dataset_particles is not None:
        # Poses/CTF/translations/anisomag come straight from the dataset --
        # dose, coincidence radius, and potential scale aren't recorded in
        # either a CryoSPARC passthrough .cs or a RELION .star, so those three
        # are still sampled below.
        (
            _,
            _,
            _,
            quats,
            translations,
            ctf_params,
            _scale,
            anisomag,
            _indices,
            _split,
        ) = dataset_particles
        n = quats.shape[0]
    else:
        n = config.n_particles

        # random_quaternion squeezes the batch axis at n == 1, returning (4,)
        # rather than (1, 4) -- indexing that per-particle later yields a
        # length-1 vector and roma rejects it. Same guard as CrowdingSimulator
        # and TomogramSpecimenGenerator apply to random_rotation_matrix.
        quats = rotations.random_quaternion(n).reshape(n, 4)

        defocus_A = _uniform_sample(config.defocus, n)
        astigmatism_magnitude = _uniform_sample(config.astigmatism, n)
        dfang = _uniform_sample(config.astigmatism_angle, n)
        dfv = defocus_A - astigmatism_magnitude
        phaseshift = _uniform_sample(config.phaseshift, n)
        tiltx = _uniform_sample(config.tiltx, n)
        tilty = _uniform_sample(config.tilty, n)
        trefoil1 = _uniform_sample(config.trefoil1, n)
        trefoil2 = _uniform_sample(config.trefoil2, n)
        tetrafoil1 = _uniform_sample(config.tetrafoil1, n)
        tetrafoil2 = _uniform_sample(config.tetrafoil2, n)
        tetrafoil3 = _uniform_sample(config.tetrafoil3, n)
        tetrafoil4 = _uniform_sample(config.tetrafoil4, n)
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
            "tetrafoil1": tetrafoil1,
            "tetrafoil2": tetrafoil2,
            "tetrafoil3": tetrafoil3,
            "tetrafoil4": tetrafoil4,
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
    n_frames = (
        config.n_frames if config.n_frames is not None else int(dose.mean().item())
    )
    cc_angstrom = config.cc * 1e7 if config.cc is not None else None

    # --- Ice ---
    # resolve_icemaker derives (n, nz) for a fresh RandomIcemaker itself, so it
    # always matches the particle volume V it gets blended into -- IceBank
    # (cache_dir=...) just loads small pre-generated coordinate files from
    # disk, cheap enough that every DDP rank can construct it independently
    # (no rank-0-builds-then-broadcasts dance needed, unlike V above).
    ice_nz = compute_nz(config.n_pixels, config.ice_thickness, pixel_size)
    # Unset, the ice follows the structure's own parameterization: the two are
    # summed into one volume, so modelling the protein with one set of
    # scattering factors and the ice around it with another is a choice worth
    # making deliberately rather than inheriting from a default.
    ice_parameterization = (
        config.ice_parameterization or config.potential_parameterization
    )
    icemaker = resolve_icemaker(
        ice_model,
        pixel_size,
        config.n_pixels,
        ice_nz,
        ice_cache_dir=config.ice_cache_dir,
        parameterization=ice_parameterization,
    )

    if icemaker is not None:
        icemaker_device = (
            f"cuda:{device_target[0]}" if mode == "multi" else device_target
        )
        if icemaker_device != "cpu":
            icemaker = icemaker.to(icemaker_device)

    model = ImageGenerator(
        V,
        pixel_size,
        quats,
        translations,
        ctf_params,
        voltage,
        dose,
        anisomag=anisomag,
        icemaker=icemaker,
        ice_thickness=config.ice_thickness,
        ice_relax_steps=config.ice_relax_steps,
        ice_parameterization=ice_parameterization,
        scattering_model=config.scattering_model,
        aberration_model=config.aberration_model,
        noise_model=noise_model,
        klim=config.klim,
        ews_curvature_sign=config.ews_curvature_sign,
        alpha=alpha,
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
        n_frames=n_frames,
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

    # --- Batch size ---
    # "auto" sizes the batch to the memory actually free on the target device
    # right now, from the box geometry the model was just built with -- see
    # specter.memory for the measured peak-memory model behind it.
    if config.batchsize == "auto":
        if mode == "multi":
            assert isinstance(device_target, list)
            # Every rank builds an identically-sized batch, so size to
            # whichever GPU in the pool has the least room.
            sizing_device = min(
                (f"cuda:{i}" for i in device_target), key=available_memory_bytes
            )
        else:
            sizing_device = str(device_target)
        batchsize = recommend_batchsize(
            config.n_pixels, model.nz, model.pad_nxy, sizing_device, n_particles=n
        )
        if is_main:
            peak_gib = (
                estimate_peak_bytes(batchsize, config.n_pixels, model.nz, model.pad_nxy)
                / 1024**3
            )
            free_gib = available_memory_bytes(sizing_device) / 1024**3
            _console.print(
                f"  batchsize='auto' -> {batchsize} particle(s) per pass "
                f"(~{peak_gib:.1f} GiB estimated peak, {free_gib:.1f} GiB free "
                f"on {sizing_device})"
            )
    else:
        batchsize = int(config.batchsize)

    # --- Generating images ---
    if mode == "multi":
        assert isinstance(device_target, list)
        if is_main:
            _section(f"Initializing multi-GPU on devices {device_target}")
        images, exitwaves, clean_exitwaves = _generate_multi(
            model,
            n,
            batchsize,
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
            batchsize,
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
        dx=pixel_size,
        voltage=voltage,
        alpha=alpha,
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
                config.n_pixels,
            )

        if clean_exitwaves is not None:
            _section("Saving clean exit waves")
            _save_exitwave_pair(
                clean_exitwaves,
                "clean_exitwave",
                config.output_dir,
                config.filename,
                config.pad_fft,
                config.n_pixels,
            )

    elapsed = time.perf_counter() - t_start
    _console.print(f"\n[bold]Total time:[/bold] {_format_elapsed(elapsed)}")
