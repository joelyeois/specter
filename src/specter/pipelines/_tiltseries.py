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
from specter.config import TiltSeriesConfig, TomogramConfig
from specter.imagegenerator import TiltSeriesGenerator
from specter.io import create_micrograph_starfile
from specter.specimen import load_specimen_volume

from ._common import (
    _console,
    _format_elapsed,
    _parse_device,
    _save_exitwave_pair,
    _section,
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
    """
    if tomogram_config is not None:
        if config.volume_path:
            raise ValueError(
                "run_tilt_series: tomogram_config and config.volume_path "
                "can't both be set -- pass exactly one specimen source: "
                "tomogram_config to build a fresh volume first, or "
                "config.volume_path to reuse an already-built one."
            )
        _section("Building specimen volume (tomogram_config)")
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

    specter.set_verbosity(logging.INFO)
    mode, device_target = _parse_device(config.device)
    if mode == "multi":
        raise ValueError(
            "run_tilt_series: multi-GPU device strings (e.g. '0,1,2') aren't "
            "supported -- a tilt series is generated as a single volume, not "
            "batched across particles. Pass a single device, e.g. 'cuda:0'."
        )

    t_start = time.perf_counter()

    # --- Loading specimen volume ---
    _section(f"Loading specimen volume from {config.volume_path}")
    volume = load_specimen_volume(config.volume_path)
    _console.print(f"  Volume shape: {tuple(volume.shape)}  (Z, Y, X)")

    # --- Building TiltSeriesGenerator ---
    _section("Building TiltSeriesGenerator")
    dx = config.voxel_size
    micrograph_size = (
        config.micrograph_size
        if config.micrograph_size is not None
        else int(volume.shape[-1])
    )
    angles = torch.linspace(
        config.min_tilt_angle, config.max_tilt_angle, config.n_tilts
    )
    _console.print(
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
        scattering_model=config.scattering_model,
        aberration_model=config.aberration_model,
        noise_model=noise_model,
        alpha=config.alpha,
        pad_fft=config.pad_fft,
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
    _section(f"Generating tilt series on {device_target}")
    with torch.no_grad():
        images, exitwaves, _clean = model.generate_tilt_series(torch.tensor([0]))
    images = images[0].cpu()  # (n_tilts, H, W)
    exitwaves = exitwaves[0].cpu()  # (n_tilts, H, W) complex

    # --- Post-processing ---
    if config.normalize_tilt_series:
        _section("Normalizing")
        mean = images.mean(dim=(-2, -1), keepdim=True)
        std = images.std(dim=(-2, -1), keepdim=True)
        images = (images - mean) / std.clamp(min=1e-8)

    # --- Saving ---
    _section("Saving")
    import mrcfile

    os.makedirs(config.output_dir, exist_ok=True)

    mrcs_path = os.path.join(config.output_dir, config.filename + ".mrcs")
    with mrcfile.new(mrcs_path, overwrite=True) as mrc:
        mrc.set_data(images.numpy().astype("float32"))
    _console.print(f"  [green]✓[/green] {mrcs_path}")

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
        folderpath=config.output_dir,
        filename=config.filename,
        dose_per_angstrom=config.dose_per_tilt,
        coincidence_radius=config.coincidence_radius,
        tilt_angles=angles,
    )

    if config.save_exitwaves:
        ew_prefix = "exitwave" if ice_model is not None else "clean_exitwave"
        _section(f"Saving {ew_prefix.replace('_', ' ')}")
        _save_exitwave_pair(
            exitwaves,
            ew_prefix,
            config.output_dir,
            config.filename,
            config.pad_fft,
            micrograph_size,
        )

    elapsed = time.perf_counter() - t_start
    _console.print(f"\n[bold]Total time:[/bold] {_format_elapsed(elapsed)}")
