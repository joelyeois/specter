"""Particle-stack generation pipeline: `ParticleStackConfig` in, .mrcs + .star out.

Moved here from `demo-scripts/generate_particle_stack.py` so the logic is a real,
importable function instead of being trapped inside a script's ``main()`` -- callable
from `specter.cli`, from plain Python, or from a notebook, with identical behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import logging
import multiprocessing
import os
import time

import torch

import specter
from specter import rotations
from specter.arrays import compute_nz
from specter.config import ParticleStackConfig, validate_config
from specter.ice import IceBank, RandomIcemaker, resolve_icemaker
from specter.image import normalize_particles
from specter.imaging import CTF, Exposure, Poses, Specimen, Structure
from specter.imaging._stages import read_dose_weights
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
from specter.potential import molecular_mass_from_atoms
from specter.devices import parse_device, resolve_available_device
from specter.settings import (
    Camera,
    Crowding,
    Envelopes,
    Ice,
    Propagation,
    bundle_from_config,
)
from specter.progress import console, section, track

from ._common import (
    _crowd_min_distance,
    _mm_to_angstrom,
    _print_total_time,
    _seed_or_draw,
    _HostStack,
    _save_exitwave_pair,
    _tracked_output_dir,
    _uniform_sample,
)


#: (structure, potential) pairs already built in this process, keyed on every
#: config field that changes them. `specter match particles` calls
#: `run_particle_stack` ten times per run on the same structure at the same
#: pixel size, and the parse-plus-build was 40-50 s of a five-minute run; a
#: normal CLI invocation builds once per process, so it never hits this.
_POTENTIAL_CACHE: dict[tuple, tuple[PDB, torch.Tensor]] = {}
_POTENTIAL_CACHE_MAX = 4


def _potential_cache_key(config: ParticleStackConfig, pixel_size: float) -> tuple:
    source = config.pdb_source
    if os.path.exists(source):
        source = os.path.realpath(source)
    species = tuple(config.atom_species) if config.atom_species is not None else None
    return (
        source,
        config.assembly,
        config.pdb_cache_dir,
        str(config.readd_hydrogens),
        config.monomer_library_path,
        config.n_pixels,
        round(float(pixel_size), 6),
        config.scattering_factors,
        config.conv_backend,
        species,
        config.shtyrov_params_path,
        config.rcut,
        config.use_deposited_bfactors,
        config.periodic,
        config.potential_method,
        _potential_device(config),
    )


def _potential_device(config: ParticleStackConfig) -> str:
    """The device the template potential is rendered on: the run's primary."""
    return parse_device(resolve_available_device(config.device)).primary


def _structure_and_potential(
    config: ParticleStackConfig, pixel_size: float, build: bool
) -> tuple[PDB, torch.Tensor]:
    """Parse the structure and build its potential, reusing a same-process result."""
    key = _potential_cache_key(config, pixel_size)
    if build and key in _POTENTIAL_CACHE:
        pdb, V = _POTENTIAL_CACHE[key]
        return pdb, V.clone()  # callers pad and move it; keep the cached copy pristine
    # Shtyrov fits scattering factors per bonded species, so derive the
    # bond topology from the structure unless the config supplies its own
    # atom_species list. Other parameterizations are per-element and would
    # only pay the extra gemmi pass for nothing.
    _derive_atom_species = (
        config.scattering_factors == "shtyrov" and config.atom_species is None
    )
    pdb = PDB(
        config.pdb_source,
        assembly=config.assembly,
        pdb_cache_dir=config.pdb_cache_dir,
        compute_atom_species=_derive_atom_species,
        readd_hydrogens=config.readd_hydrogens,
        monomer_library_path=config.monomer_library_path,
    )
    if not build:
        return pdb, torch.zeros(config.n_pixels, config.n_pixels, config.n_pixels)
    # Rendered on the compute device and moved to the host, as run_micrograph
    # does: the analytic renderer is a loop of small ops per atom, 3.3 s on
    # the CPU against 0.2 s on the device for 6bdf at 256 px. The device
    # build is not bit-reproducible run to run (~5e-7 of the potential's
    # max), so neither is a seeded stack once Poisson counts round.
    V = (
        Structure(pdb, config.scattering_factors)
        .render(
            pixel_size,
            config.n_pixels,
            method=config.potential_method,
            conv_backend=config.conv_backend,
            atom_species=config.atom_species,
            shtyrov_params_path=config.shtyrov_params_path,
            rcut=config.rcut,
            use_deposited_bfactors=config.use_deposited_bfactors,
            periodic=config.periodic,
            device=_potential_device(config),
        )
        .potential.cpu()
    )
    if len(_POTENTIAL_CACHE) >= _POTENTIAL_CACHE_MAX:
        _POTENTIAL_CACHE.pop(next(iter(_POTENTIAL_CACHE)))
    _POTENTIAL_CACHE[key] = (pdb, V)
    return pdb, V


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

    Notes
    -----
    With several devices the particle indices are split into one contiguous
    share per device, each made by its own worker process. Every particle's
    random draws derive from ``(seed, index)``, so the stack is the one a
    single device makes.
    """
    if config.cs_path is not None and config.star_path is not None:
        raise ValueError(
            "run_particle_stack: cs_path and star_path can't both be set -- "
            "pass exactly one dataset to take poses/CTF from, or neither to "
            "sample them synthetically."
        )

    validate_config(config)

    specter.set_verbosity(logging.INFO)

    devices = parse_device(resolve_available_device(config.device)).devices
    t_start = time.perf_counter()

    # Seeds the up-front draws (poses, defocus, dose) on the global stream;
    # the draws inside each forward pass (crowding, ice, noise) derive from
    # the same seed per particle.
    run_seed = _seed_or_draw(config.seed)

    section("Building 3D scattering potential")
    # pixel_size/voltage/alpha come from the dataset when cs_path/star_path is
    # set; both extractors return the same 10-tuple, so everything downstream
    # is agnostic to which file format it came from.
    dataset_particles, pixel_size, voltage, alpha = _resolve_imaging_parameters(config)
    # Unused when cs_path/star_path is set, since Cs then comes per particle.
    cs_angstrom = _mm_to_angstrom(config.cs)
    pdb, V = _structure_and_potential(config, pixel_size, build=True)

    section("Sampling poses, defocus, and translations")
    n, quats, translations, ctf_params, anisomag = _sample_poses_and_ctf(
        config, dataset_particles, cs_angstrom
    )
    dose = _uniform_sample(config.dose, n)
    coincidence_radius = _uniform_sample(config.coincidence_radius, n)
    n_frames = (
        config.n_frames if config.n_frames is not None else int(dose.mean().item())
    )
    weights, weights_max_frequency = read_dose_weights(
        bundle_from_config(Camera, config, n_frames=n_frames)
    )

    run = _ParticleRun(
        config=config,
        potential=V,
        pixel_size=pixel_size,
        voltage=voltage,
        alpha=alpha,
        molecular_mass=molecular_mass_from_atoms(pdb.atomic_numbers),
        max_diameter=pdb.max_diameter,
        poses=Poses(quats, translations, torch.arange(n)),
        ctf=CTF(ctf_params),
        exposure=Exposure(
            dose,
            n_frames=n_frames,
            weights=weights,
            weights_max_frequency=weights_max_frequency,
        ),
        coincidence_radius=coincidence_radius,
        anisomag=anisomag,
        seed=run_seed,
    )
    batchsize = _resolve_batchsize(config, run, devices, n)

    with _tracked_output_dir(config, "particles", is_main=True) as output_dir:
        section(f"Generating images on {', '.join(devices)}")
        if len(devices) > 1:
            images, exitwaves, clean_exitwaves = _generate_sharded(
                run, devices, batchsize, output_dir
            )
        else:
            images, exitwaves, clean_exitwaves = run.generate(
                0, n, devices[0], batchsize, track
            )

        _save_stack(
            config,
            output_dir,
            images,
            exitwaves,
            clean_exitwaves,
            quats,
            translations,
            ctf_params,
            pixel_size,
            voltage,
            alpha,
            dose,
            coincidence_radius,
            True,
        )

    _print_total_time(t_start)


@dataclass(frozen=True)
class _ParticleRun:
    """
    One particle stack's inputs, resolved once and picklable into a worker.

    The template is held on the host; each device builds its own specimen
    (and ice library) from it, so a worker never re-renders the potential.
    """

    config: ParticleStackConfig
    potential: torch.Tensor
    pixel_size: float
    voltage: float
    alpha: float
    molecular_mass: float
    max_diameter: float
    poses: Poses
    ctf: CTF
    exposure: Exposure
    coincidence_radius: torch.Tensor
    anisomag: torch.Tensor | None
    seed: int

    def specimen(self, device: str) -> Specimen:
        """The template with this run's crowding, ice and damage, on `device`."""
        config = self.config
        specimen = Specimen(
            self.potential.to(device),
            self.pixel_size,
            molecular_mass=self.molecular_mass,
        )
        min_distance = _crowd_min_distance(config.crowd_min_distance, self.max_diameter)
        if min_distance is not None:
            crowding = bundle_from_config(
                Crowding,
                config,
                prefix="crowd_",
                min_distance=min_distance,
                water_air_interface=config.water_air_interface,
            )
            specimen = specimen.crowd(crowding, on_cpu=config.crowd_move_to_cpu)
        ice = bundle_from_config(
            Ice, config, prefix="ice_", parameterization=config.bulk_scattering_factors
        )
        icemaker = _build_icemaker(config, self.pixel_size, device)
        if icemaker is None:
            # The thickness still sizes the box when there is no ice in it.
            specimen = replace(specimen, ice=ice)
        else:
            specimen = specimen.solvate(ice, icemaker=icemaker)
        envelopes = bundle_from_config(Envelopes, self.config, cc=None)
        if envelopes.dose_envelope and envelopes.dose_envelope_target == "specimen":
            specimen = specimen.damage()
        return specimen

    def generate(
        self, start: int, stop: int, device: str, batchsize: int, progress: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Particles ``start`` to ``stop``, and their exit waves when asked.

        Returns
        -------
        images, exitwaves, clean_exitwaves : torch.Tensor or None
            On the host. ``exitwaves`` are on the pixel grid; the clean ones
            are the same specimens with no ice and no damage
            (:meth:`~specter.imaging.Specimen.dry`).
        """
        config = self.config
        specimen = self.specimen(device)
        dry = specimen.dry() if config.save_clean_exitwaves else None
        propagation = asdict(bundle_from_config(Propagation, config, alpha=self.alpha))
        # Its cc is in Angstrom and the config's in mm; cc is read from the config.
        envelopes = bundle_from_config(Envelopes, config, cc=None)
        camera = bundle_from_config(Camera, config)
        damage_on_lens = (
            envelopes.dose_envelope
            and envelopes.dose_envelope_target == "transfer_function"
        )
        n = stop - start
        images, exitwaves, clean = _HostStack(n), _HostStack(n), _HostStack(n)
        with torch.no_grad():
            for lo in progress(
                range(start, stop, batchsize), description="Generating images"
            ):
                i = slice(lo, min(lo + batchsize, stop))
                exposure = self.exposure[i]
                beam = dict(
                    voltage=self.voltage,
                    exposure=exposure,
                    seed=self.seed,
                    objective_aperture=config.objective_aperture,
                    **propagation,
                )
                waves = specimen.scatter(self.poses[i], **beam)
                if config.save_exitwaves:
                    exitwaves.append(waves.pixels())
                if dry is not None:
                    clean.append(dry.scatter(self.poses[i], **beam).pixels())
                waves = waves.apply_ctf(
                    self.ctf[i],
                    bfactor=config.bfactor,
                    convergence_angle=envelopes.convergence_angle,
                    dose_envelope=exposure if damage_on_lens else None,
                )
                image = waves.intensity(
                    cc=config.cc or None,
                    energy_spread=envelopes.energy_spread,
                    deltaV_V=envelopes.deltaV_V,
                    deltaI_I=envelopes.deltaI_I,
                    temporal_coherence=envelopes.temporal_coherence,
                )
                counts = image.detect(
                    exposure,
                    detector=camera.detector_model,
                    noise=camera.noise_model,
                    coincidence_radius=self.coincidence_radius[i],
                    anisomag=None if self.anisomag is None else self.anisomag[i],
                    detector_pixel_size=camera.detector_pixel_size,
                    seed=self.seed,
                )
                images.append(counts.array)
        return (
            images.finish(),
            exitwaves.finish() if config.save_exitwaves else None,
            clean.finish() if dry is not None else None,
        )


def _shard_worker(
    run: _ParticleRun,
    start: int,
    stop: int,
    device: str,
    batchsize: int,
    path: str,
    threads: int | None,
) -> None:
    """
    Make particles ``start`` to ``stop`` on `device` and save them to `path`.

    Module-level so it survives pickling into a spawned process. Several
    CPU workers split the host's cores rather than each sizing its thread
    pool from the whole machine.
    """
    if threads is not None:
        torch.set_num_threads(threads)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    images, exitwaves, clean = run.generate(
        start, stop, device, batchsize, lambda it, **_: it
    )
    torch.save(
        {
            "ids": torch.arange(start, stop),
            "images": images,
            "exitwaves": exitwaves,
            "clean_exitwaves": clean,
        },
        path,
    )


def _generate_sharded(
    run: _ParticleRun, devices: tuple[str, ...], batchsize: int, output_dir: str
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Split the stack into one contiguous share per device, one worker each.

    Each worker saves its share to ``particle_shard_<k>.pt`` in
    `output_dir`; the shares are read back and stitched in index order by
    :func:`_gather_shards`, which fails on a missing or short share rather
    than returning a smaller stack.
    """
    n = len(run.poses)
    world = min(len(devices), n)
    bounds = [n * k // world for k in range(world + 1)]
    cpu_workers = sum(d == "cpu" for d in devices[:world])
    threads = max(1, (os.cpu_count() or 1) // cpu_workers) if cpu_workers > 1 else None
    os.makedirs(output_dir, exist_ok=True)
    paths = [os.path.join(output_dir, f"particle_shard_{k}.pt") for k in range(world)]
    ctx = multiprocessing.get_context("spawn")
    workers = []
    for k in range(world):
        worker = ctx.Process(
            target=_shard_worker,
            args=(
                run,
                bounds[k],
                bounds[k + 1],
                devices[k],
                batchsize,
                paths[k],
                threads if devices[k] == "cpu" else None,
            ),
        )
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()
    failed = [
        (devices[k], w.exitcode) for k, w in enumerate(workers) if w.exitcode != 0
    ]
    if failed:
        detail = ", ".join(f"{d} (exit {code})" for d, code in failed)
        raise RuntimeError(f"run_particle_stack: worker(s) failed: {detail}")
    return _gather_shards(paths, n)


def _gather_shards(
    paths: list[str], n: int
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Stitch the workers' shares back into one stack in particle order.

    A missing share, or shares that do not cover ``0..n-1`` exactly once,
    is an error, raised before any file is deleted so the shares that did
    arrive are still there to diagnose.
    """
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"particle share(s) never written: {missing}")
    shards = [torch.load(p, weights_only=True) for p in paths]
    ids = torch.cat([s["ids"] for s in shards])
    for s in shards:
        if len(s["images"]) != len(s["ids"]):
            raise RuntimeError(
                "a particle share holds a different number of images and ids"
            )
    if not torch.equal(ids.sort().values, torch.arange(n)):
        raise RuntimeError(
            f"particle shares cover {len(ids)} images, expected 0..{n - 1}"
        )
    order = ids.argsort()

    def stitch(key: str) -> torch.Tensor | None:
        parts = [s[key] for s in shards]
        return None if parts[0] is None else torch.cat(parts)[order]

    result = stitch("images"), stitch("exitwaves"), stitch("clean_exitwaves")
    for p in paths:
        os.remove(p)
    assert result[0] is not None
    return result[0], result[1], result[2]


def _resolve_imaging_parameters(
    config: ParticleStackConfig,
) -> tuple[Any, float, float, float]:
    """
    The dataset's particles when ``cs_path``/``star_path`` is set (else None),
    and the pixel size, voltage and amplitude contrast the run images at,
    which come from that dataset when there is one and from the config
    otherwise.
    """
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

    if config.absorption_model == "inelastic_mfp":
        # The dataset's amplitude contrast is not a measurement of absorption
        # -- CTF estimation takes it as an input and never fits it -- so under
        # this model it is dropped rather than allowed to override. Leaving it
        # in would double-count against the mean-free-path term, which
        # `Propagation` rejects outright. See
        # `potential.apply_amplitude_contrast` for the evidence.
        alpha = 0.0

    return dataset_particles, pixel_size, voltage, alpha


def _sample_poses_and_ctf(
    config: ParticleStackConfig, dataset_particles: Any, cs_angstrom: float
) -> tuple[
    int, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None
]:
    """
    Poses, translations, CTF terms and anisotropic magnification for every
    particle: taken from the dataset when there is one, sampled from the
    config's ranges otherwise.
    """
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

        defocus_angstrom = _uniform_sample(config.defocus, n)
        astigmatism_magnitude = _uniform_sample(config.astigmatism, n)
        dfang = _uniform_sample(config.astigmatism_angle, n)
        dfv = defocus_angstrom - astigmatism_magnitude
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
            "dfu": defocus_angstrom,
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

    return n, quats, translations, ctf_params, anisomag


def _build_icemaker(
    config: ParticleStackConfig, pixel_size: float, device: str
) -> IceBank | RandomIcemaker | None:
    """The run's icemaker on `device`, where the images are made, or None."""
    ice_model = None if config.ice_model == "none" else config.ice_model
    ice_nz = compute_nz(config.n_pixels, config.ice_thickness, pixel_size)
    icemaker = resolve_icemaker(
        ice_model,
        pixel_size,
        config.n_pixels,
        ice_nz,
        ice_cache_dir=config.ice_cache_dir,
        parameterization=config.bulk_scattering_factors,
    )
    if icemaker is not None and device != "cpu":
        icemaker = icemaker.to(device)
    return icemaker


def _resolve_batchsize(
    config: ParticleStackConfig, run: _ParticleRun, devices: tuple[str, ...], n: int
) -> int:
    """
    The particles per forward pass: ``config.batchsize``, or for ``"auto"``
    what fits the memory free on the device with the least of it right now,
    from the run's box geometry (see `specter.memory`).
    """
    if config.batchsize != "auto":
        return int(config.batchsize)
    nz = compute_nz(config.n_pixels, config.ice_thickness, run.pixel_size)
    pad_nxy = (
        config.n_pixels + (config.n_pixels // 2) * 2
        if config.pad_fft
        else config.n_pixels
    )
    ice = config.ice_model not in (None, "none")
    sizing_device = min(devices, key=available_memory_bytes)
    batchsize = recommend_batchsize(
        config.n_pixels,
        nz,
        pad_nxy,
        sizing_device,
        n_particles=n,
        oversample=config.oversample,
        ice=ice,
    )
    peak_gib = (
        estimate_peak_bytes(
            batchsize, config.n_pixels, nz, pad_nxy, config.oversample, ice
        )
        / 1024**3
    )
    free_gib = available_memory_bytes(sizing_device) / 1024**3
    console.print(
        f"  batchsize='auto' -> {batchsize} particle(s) per pass "
        f"(~{peak_gib:.1f} GiB estimated peak, {free_gib:.1f} GiB free "
        f"on {sizing_device})"
    )
    return batchsize


def _save_stack(
    config: ParticleStackConfig,
    output_dir: str,
    images: torch.Tensor,
    exitwaves: torch.Tensor | None,
    clean_exitwaves: torch.Tensor | None,
    quats: torch.Tensor,
    translations: torch.Tensor,
    ctf_params: dict[str, torch.Tensor],
    pixel_size: float,
    voltage: float,
    alpha: float,
    dose: torch.Tensor,
    coincidence_radius: torch.Tensor,
    is_main: bool,
) -> None:
    """Normalise the images if asked, then write the ``.mrcs``/``.star`` pair and any exit waves."""
    # --- Post-processing ---
    if is_main:
        section("Post-processing")
    if config.normalize_particles:
        # This pipeline owns the stack. Normalize each independent image in
        # bounded chunks, reusing its storage instead of holding several
        # whole-stack subtraction/division/negation temporaries.
        for start in range(0, len(images), 64):
            chunk = images[start : start + 64]
            normalized, _means, _stds = normalize_particles(chunk)
            chunk.copy_(-normalized)
        particles = images
    else:
        particles = images

    if is_main:
        section("Saving .mrcs + .star")
    create_particle_starfile(
        particles,
        rotations=quats,
        translations=translations,
        ctf_params=ctf_params,
        dx=pixel_size,
        voltage=voltage,
        alpha=alpha,
        filename=config.filename,
        output_dir=output_dir,
        dose_per_angstrom=dose,
        coincidence_radius=coincidence_radius,
    )

    if is_main:
        if exitwaves is not None:
            section("Saving exit waves")
            _save_exitwave_pair(
                exitwaves,
                "exitwave",
                output_dir,
                config.filename,
                config.pad_fft,
                config.n_pixels,
                pixel_size,
            )

        if clean_exitwaves is not None:
            section("Saving clean exit waves")
            _save_exitwave_pair(
                clean_exitwaves,
                "clean_exitwave",
                output_dir,
                config.filename,
                config.pad_fft,
                config.n_pixels,
                pixel_size,
            )
