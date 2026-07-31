"""Tilt-series generation pipeline: `TiltSeriesConfig` in, .mrcs + .star out.

This first pass only supports the ``volume_path`` specimen source (a
pre-built scattering-potential volume loaded from disk) -- the polnet-driven
placement path (`protein_specs`/`membrane_specs`) stays reachable only via
`demo-scripts/generate_tilt_series.py` / the tilt-series notebook until it
gets its own dispatch branch here.

Mirrors `demo-scripts/generate_tilt_series.py`'s imaging/save logic, with the
specimen-building step replaced by loading `config.volume_path` from disk.
"""

from __future__ import annotations

import logging
import os
import time

import torch

import specter
from specter.config import TiltSeriesConfig
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


def run_tilt_series(config: TiltSeriesConfig) -> None:
    """
    Load a specimen volume, simulate a cryo-ET tilt series, and save it as
    ``.mrcs`` + ``.star``.

    Parameters
    ----------
    config : TiltSeriesConfig
        Fully-resolved run configuration, e.g. from
        :func:`specter.config.load_config`, or constructed directly in
        Python. ``config.volume_path`` must be set -- this pass only
        supports loading a pre-built specimen volume from disk.
    """
    if not config.volume_path:
        raise ValueError(
            "run_tilt_series: config.volume_path must be set. This pass only "
            "supports generating a tilt series from a pre-built specimen "
            "volume -- polnet-driven placement isn't wired into this "
            "pipeline yet (see demo-scripts/generate_tilt_series.py)."
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
    dx = config.target_v_size
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
        num_frames=config.num_frames,
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
