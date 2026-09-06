"""Tilt-series generation pipeline: `TiltSeriesConfig` in, .mrcs + .star out.

This is the imaging half of the cryo-ET pipeline only -- it loads a
pre-built specimen volume from ``config.volume_path`` and simulates the
tilted acquisition. For the specimen-building half, see `specter build
tomogram` (`specter.pipelines.run_build_tomogram`/`TomogramConfig`), which
writes a `.mrc` volume this pipeline then loads directly.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time

import torch

import specter
from specter.config import (
    TiltSeriesConfig,
    TomogramConfig,
    validate_config,
)
from specter.imagegenerator import TiltSeriesGenerator
from specter.io import create_micrograph_starfile
from specter.specimen import load_specimen_volume

from specter.devices import parse_device, resolve_available_device
from specter.progress import console, format_elapsed, section
from ._common import (
    _reserve_next_job_id,
    _save_exitwave_pair,
    _tracked_output_dir,
    resolve_output_dir,
)
from ._tomogram import run_build_tomogram, tomogram_output_path


def run_tilt_series(
    config: TiltSeriesConfig, *, tomogram_config: TomogramConfig | None = None
) -> None:
    """
    Load a specimen volume, simulate a cryo-ET tilt series, and save it as
    ``.mrcs`` + ``.star``.

    Parameters
    ----------
    config : TiltSeriesConfig
        Fully-resolved run configuration, e.g. from
        :func:`specter.config.load_config`, or constructed directly in
        Python. ``config.volume_path`` must be set -- e.g. the output of
        `specter build tomogram` -- unless ``tomogram_config`` is given.
    tomogram_config : TomogramConfig, optional
        If given, build this specimen volume first (`run_build_tomogram`)
        and use its output as this run's specimen -- chaining `specter
        build tomogram` + `specter simulate tiltseries` in one call.
        ``config.volume_path`` must be unset in this case (the two
        specimen sources are mutually exclusive). Always builds exactly
        one tomogram (`run_build_tomogram`'s ``n_tomograms=1`` default,
        not exposed here) -- for several tomograms, call
        `run_build_tomogram(tomogram_config, n_tomograms=N)` yourself and
        loop `run_tilt_series(config=..., ...)` over each resulting
        ``volume_path`` instead.

        If this run is tracked (``config.project``/``job_id`` set) and
        ``tomogram_config`` wasn't independently configured for tracking,
        ``tomogram_config``'s project and output_dir are set to match --
        one tracked chained call then produces two separate jobs, one
        "tomograms" and one "tiltseries", under the same project, linked
        implicitly by this run's ``volume_path`` pointing into the
        tomogram job's directory. Give ``tomogram_config`` its own
        ``project``/``job_id`` explicitly to opt out of this (e.g. to
        reuse one tracked tomogram build across several tiltseries runs in
        a different project).
    """
    if tomogram_config is not None:
        if config.volume_path:
            raise ValueError(
                "run_tilt_series: tomogram_config and config.volume_path "
                "can't both be set -- pass exactly one specimen source: "
                "tomogram_config to build a fresh volume first, or "
                "config.volume_path to reuse an already-built one."
            )

        # Cascade project (+ output_dir) from the outer run, but only if
        # tomogram_config wasn't independently configured for tracking at
        # all -- an explicit tomogram_config.project must never be
        # silently overridden.
        if (
            tomogram_config.project is None
            and tomogram_config.job_id is None
            and (config.project is not None or config.job_id is not None)
        ):
            tomogram_config = dataclasses.replace(
                tomogram_config,
                project=config.project,
                output_dir=tomogram_config.output_dir or config.output_dir,
            )

        # Whichever way tracking ended up on for tomogram_config -- cascaded
        # above, or the caller's own explicit project -- an unpinned job_id
        # can't be known ahead of time by tomogram_output_path below, which
        # runs *after* run_build_tomogram returns using this same config
        # object. Reserve one now so both agree on the same path without
        # either function handing anything back to the other -- the same
        # can't-coordinate-after-the-fact problem multi-GPU DDP has (see
        # _tracked_output_dir), just between two pipeline calls instead of
        # two processes.
        if (
            tomogram_config.project is not None or tomogram_config.job_id is not None
        ) and tomogram_config.job_id is None:
            root = resolve_output_dir(tomogram_config, "tomograms", create=True)
            job_id = _reserve_next_job_id(tomogram_config.project, root)
            tomogram_config = dataclasses.replace(
                tomogram_config, job_id=job_id, output_dir=root
            )

        section("Building specimen volume (tomogram_config)")
        run_build_tomogram(tomogram_config)
        config = dataclasses.replace(
            config, volume_path=tomogram_output_path(tomogram_config)
        )

    if not config.volume_path:
        raise ValueError(
            "run_tilt_series: config.volume_path must be set to a "
            "pre-built specimen volume -- build one first with `specter "
            "build tomogram`, or pass tomogram_config to build one as "
            "part of this call."
        )

    validate_config(config)

    specter.set_verbosity(logging.INFO)
    mode, device_target = parse_device(
        resolve_available_device(config.device)
    ).ddp_dispatch()
    if mode == "multi":
        raise ValueError(
            "run_tilt_series: multi-GPU device strings (e.g. '0,1,2') aren't "
            "supported -- a tilt series is generated as a single volume, not "
            "batched across particles. Pass a single device, e.g. 'cuda:0'."
        )

    t_start = time.perf_counter()

    if config.seed is not None:
        specter.seed(config.seed)
    else:
        generated_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        specter.seed(generated_seed)
        console.print(f"[dim]No seed given -- using seed={generated_seed}[/dim]")

    # --- Loading specimen volume ---
    section(f"Loading specimen volume from {config.volume_path}")
    volume = load_specimen_volume(config.volume_path)
    console.print(f"  Volume shape: {tuple(volume.shape)}  (Z, Y, X)")

    # --- Building TiltSeriesGenerator ---
    section("Building TiltSeriesGenerator")
    dx = config.voxel_size
    micrograph_size = (
        config.micrograph_size
        if config.micrograph_size is not None
        else int(volume.shape[-1])
    )
    angles = torch.linspace(
        config.min_tilt_angle, config.max_tilt_angle, config.n_tilts
    )
    console.print(
        f"  Tilt angles: {config.n_tilts} steps from {config.min_tilt_angle}° "
        f"to {config.max_tilt_angle}°"
    )

    cs_angstrom = config.cs * 1e7
    cc_angstrom = config.cc * 1e7 if config.cc is not None else None

    ctf_params = {
        "cs": torch.tensor([cs_angstrom]),
        "dfu": torch.tensor([config.defocus]),
    }

    ice_model = None if config.ice_model == "none" else config.ice_model
    noise_model = None if config.noise_model == "none" else config.noise_model
    detector_model = None if config.detector_model == "none" else config.detector_model

    model = TiltSeriesGenerator(
        volume.unsqueeze(0),  # add batch dim: (1, Z, Y, X)
        micrograph_size,
        dx,
        ctf_params,
        config.voltage,
        config.dose_per_tilt,
        angles=angles,
        ice_model=ice_model,
        ice_cache_dir=config.ice_cache_dir,
        ice_relax_steps=config.ice_relax_steps,
        ice_parameterization=config.bulk_scattering_factors,
        scattering_model=config.scattering_model,
        noise_model=noise_model,
        alpha=config.alpha,
        pad_fft=config.pad_fft,
        klim=config.klim,
        bfactor=config.bfactor,
        tilt_axis=config.tilt_axis,
        coincidence_radius=config.coincidence_radius,
        n_frames=config.n_frames,
        convergence_angle=config.convergence_angle,
        cc=cc_angstrom,
        energy_spread=config.energy_spread,
        deltaV_V=config.deltaV_V,
        deltaI_I=config.deltaI_I,
        dose_envelope=config.dose_envelope,
        detector_model=detector_model,
    ).to(device_target)

    # --- Generating ---
    section(f"Generating tilt series on {device_target}")
    with torch.no_grad():
        images, exitwaves, _clean = model.generate_tilt_series(torch.tensor([0]))
    images = images[0].cpu()  # (n_tilts, H, W)
    exitwaves = exitwaves[0].cpu()  # (n_tilts, H, W) complex

    # --- Post-processing ---
    if config.normalize_tilt_series:
        section("Normalizing")
        mean = images.mean(dim=(-2, -1), keepdim=True)
        std = images.std(dim=(-2, -1), keepdim=True)
        images = (images - mean) / std.clamp(min=1e-8)

    # --- Saving ---
    section("Saving")
    import mrcfile

    with _tracked_output_dir(config, "tiltseries") as output_dir:
        os.makedirs(output_dir, exist_ok=True)

        mrcs_path = os.path.join(output_dir, config.filename + ".mrcs")
        with mrcfile.new(mrcs_path, overwrite=True) as mrc:
            mrc.set_data(images.numpy().astype("float32"))
        console.print(f"  [green]✓[/green] {mrcs_path}")

        ctf_params_broadcast = {
            "cs": torch.full((config.n_tilts,), cs_angstrom),
            "dfu": torch.full((config.n_tilts,), config.defocus),
        }
        create_micrograph_starfile(
            n=config.n_tilts,
            voltage=config.voltage,
            pixel_size=dx,
            alpha=config.alpha,
            ctf_params=ctf_params_broadcast,
            output_dir=output_dir,
            filename=config.filename,
            dose_per_angstrom=config.dose_per_tilt,
            coincidence_radius=config.coincidence_radius,
            tilt_angles=angles,
        )

        if config.save_exitwaves:
            ew_prefix = "exitwave" if ice_model is not None else "clean_exitwave"
            section(f"Saving {ew_prefix.replace('_', ' ')}")
            _save_exitwave_pair(
                exitwaves,
                ew_prefix,
                output_dir,
                config.filename,
                config.pad_fft,
                micrograph_size,
            )

    elapsed = time.perf_counter() - t_start
    console.print(f"\n[bold]Total time:[/bold] {format_elapsed(elapsed)}")
