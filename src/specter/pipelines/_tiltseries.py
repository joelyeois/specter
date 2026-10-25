"""Tilt-series generation pipeline: `TiltSeriesConfig` in, .mrcs + .star out.

This is the imaging half of the cryo-ET pipeline only -- it loads a
pre-built specimen volume from ``config.volume_path`` and simulates the
tilted acquisition. For the specimen-building half, see `specter build
tomogram` (`specter.pipelines.run_build_tomogram`/`TomogramConfig`), which
writes a `.mrc` volume this pipeline then loads directly.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from pathlib import Path

import torch

import specter
from specter.config import (
    TiltSeriesConfig,
    TomogramConfig,
    validate_config,
)
from specter.config._scalar_range import parse_scalar_or_range
from specter.imaging import CTF, Exposure, Poses, Sample, TiltSeries
from specter.imaging._stages import SEED_STAGE_SPECIMEN, read_dose_weights
from specter.random_seed import derive_seed
from specter.io import create_micrograph_starfile
from specter.specimen import load_specimen_volume

from specter.devices import parse_device, resolve_available_device
from specter.settings import (
    Camera,
    Deformation,
    Envelopes,
    Ice,
    Propagation,
    TiltGeometry,
    bundle_from_config,
)
from specter.progress import console, section, track
from ._common import (
    _mm_to_angstrom,
    _normalize_images,
    _print_total_time,
    _seed_or_draw,
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
        ``volume_path`` instead. Unless ``tomogram_config.seed`` is set, the
        tomogram is seeded from ``config.seed`` (drawn and printed first if
        unset), so the one seed reproduces specimen and series together.

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

        # One seed reproduces the whole chained run: the series' own seed,
        # drawn now and printed if unset, also seeds the tomogram, unless that
        # config names its own. Derived rather than shared, so the specimen's
        # draws and the series' are separate streams.
        seed = config.seed if config.seed is not None else _seed_or_draw(None)
        config = dataclasses.replace(config, seed=seed)
        if tomogram_config.seed is None:
            tomogram_config = dataclasses.replace(
                tomogram_config, seed=derive_seed(seed, SEED_STAGE_SPECIMEN)
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
    deformation = bundle_from_config(
        Deformation,
        config,
        prefix="deformation_",
        hole_offset=parse_scalar_or_range(config.deformation_hole_offset),
    )
    if config.deformation_picks_dir is not None and deformation.model is None:
        raise ValueError(
            "deformation_picks_dir is set but deformation_model is 'none': "
            "there is no bend to move the picks by"
        )

    specter.set_verbosity(logging.INFO)
    devices = parse_device(resolve_available_device(config.device))
    if devices.is_multi:
        raise ValueError(
            "run_tilt_series: multi-device strings (e.g. '0,1,2') aren't "
            "supported -- a tilt series is generated as a single volume, not "
            "batched across particles. Pass a single device, e.g. 'cuda:0'."
        )
    device_target = devices.primary

    t_start = time.perf_counter()

    run_seed = _seed_or_draw(config.seed)

    # --- Loading specimen volume ---
    section(f"Loading specimen volume from {config.volume_path}")
    volume = load_specimen_volume(config.volume_path)
    console.print(f"  Volume shape: {tuple(volume.shape)}  (Z, Y, X)")

    # --- Preparing the series ---
    section("Preparing the tilt series")
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

    cs_angstrom = _mm_to_angstrom(config.cs)
    # The mean-free-path model replaces the amplitude-contrast ratio as the
    # source of the imaginary potential; keeping both would absorb twice.
    alpha = 0.0 if config.absorption_model == "inelastic_mfp" else config.alpha

    ctf_params = {
        "cs": torch.tensor([cs_angstrom]),
        "dfu": torch.tensor([config.defocus]),
    }

    ice = bundle_from_config(
        Ice, config, prefix="ice_", parameterization=config.bulk_scattering_factors
    )
    # Envelopes.cc is in Angstrom and the config's in mm; cc is read from the config.
    envelopes = bundle_from_config(Envelopes, config, cc=None)
    damage_on_lens = (
        envelopes.dose_envelope
        and envelopes.dose_envelope_target == "transfer_function"
    )
    camera = bundle_from_config(Camera, config)
    weights, weights_max_frequency = read_dose_weights(camera)
    exposure = Exposure.tilt_series(
        config.dose_per_tilt,
        config.n_tilts,
        n_frames=camera.n_frames,
        weights=weights,
        weights_max_frequency=weights_max_frequency,
    )
    tilt = bundle_from_config(TiltGeometry, config)
    series = Sample.from_volume(volume, dx).tilt_series(
        Poses.tilt_series(angles, axis=tilt.tilt_axis),
        exposure,
        voltage=config.voltage,
        micrograph_size=micrograph_size,
        ice=ice,
        damaged=envelopes.dose_envelope and not damage_on_lens,
        tilt=tilt,
        deformation=deformation,
        seed=run_seed,
        objective_aperture=config.objective_aperture,
        propagation=bundle_from_config(Propagation, config, alpha=alpha),
    )
    ctf = CTF(ctf_params)

    # --- Generating ---
    section(f"Generating tilt series on {device_target}")
    images_list: list[torch.Tensor] = []
    exitwaves_list: list[torch.Tensor] = []
    with torch.no_grad():
        for i in track(range(config.n_tilts), description="Generating tilt series"):
            one = slice(i, i + 1)
            waves = series.scatter(i, device=device_target)
            if config.save_exitwaves:
                exitwaves_list.append(waves.pixels().cpu())
            image = (
                waves.apply_ctf(
                    ctf,
                    bfactor=config.bfactor,
                    convergence_angle=envelopes.convergence_angle,
                    dose_envelope=exposure[one] if damage_on_lens else None,
                )
                .intensity(
                    cc=config.cc or None,
                    energy_spread=envelopes.energy_spread,
                    deltaV_V=envelopes.deltaV_V,
                    deltaI_I=envelopes.deltaI_I,
                    temporal_coherence=envelopes.temporal_coherence,
                )
                .detect(
                    exposure[one],
                    detector=camera.detector_model,
                    noise=camera.noise_model,
                    coincidence_radius=config.coincidence_radius,
                    detector_pixel_size=camera.detector_pixel_size,
                    seed=run_seed,
                )
            )
            images_list.append(image.array.cpu())
    images = torch.cat(images_list)  # (n_tilts, H, W)
    exitwaves = torch.cat(exitwaves_list) if config.save_exitwaves else None

    # --- Post-processing ---
    if config.normalize_tilt_series:
        section("Normalizing")
        images = _normalize_images(images)

    # --- Saving ---
    section("Saving")
    import mrcfile

    with _tracked_output_dir(config, "tiltseries") as output_dir:
        os.makedirs(output_dir, exist_ok=True)

        mrcs_path = os.path.join(output_dir, config.filename + ".mrcs")
        with mrcfile.new(mrcs_path, overwrite=True) as mrc:
            mrc.set_data(images.numpy().astype("float32"))
            mrc.voxel_size = dx
        console.print(f"  [green]✓[/green] {mrcs_path}")

        ctf_params_broadcast = {
            "cs": torch.full((config.n_tilts,), cs_angstrom),
            "dfu": torch.full((config.n_tilts,), config.defocus),
        }
        create_micrograph_starfile(
            n=config.n_tilts,
            voltage=config.voltage,
            pixel_size=dx,
            alpha=alpha,
            ctf_params=ctf_params_broadcast,
            output_dir=output_dir,
            filename=config.filename,
            dose_per_angstrom=config.dose_per_tilt,
            coincidence_radius=config.coincidence_radius,
            tilt_angles=angles,
        )

        if series.deflection is not None:
            _save_deformation_truth(
                series,
                tuple(volume.shape[-3:]),
                angles,
                run_seed,
                output_dir,
                config,
            )

        if exitwaves is not None:
            ew_prefix = "exitwave" if ice.model is not None else "clean_exitwave"
            section(f"Saving {ew_prefix.replace('_', ' ')}")
            _save_exitwave_pair(
                exitwaves,
                ew_prefix,
                output_dir,
                config.filename,
                config.pad_fft,
                micrograph_size,
                dx,
            )

    _print_total_time(t_start)


def _save_deformation_truth(
    series: TiltSeries,
    shape: tuple[int, ...],
    angles: torch.Tensor,
    seed: int,
    output_dir: str,
    config: TiltSeriesConfig,
) -> None:
    """
    Write the bend each tilt imaged: the full deflection as an ``.mrc``, a
    ``.json`` of how much of it each tilt saw, and, with
    ``deformation_picks_dir``, the specimen's picks moved per tilt.

    Positions are in Angstrom from voxel ``(nz // 2, ny // 2, nx // 2)`` of
    the specimen volume, the rotation centre, whose z slice is the bend's
    midplane; a pick file's corner-relative location ``i * voxel_size`` is
    shifted by that voxel's.
    """
    import mrcfile

    nz, ny, nx = shape
    dx = config.voxel_size
    deflection = series.deflection
    assert deflection is not None

    y = (torch.arange(ny, dtype=torch.float64) - ny // 2) * dx
    x = (torch.arange(nx, dtype=torch.float64) - nx // 2) * dx
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    w = deflection(gx, gy)[0]
    map_path = os.path.join(output_dir, config.filename + "_deflection.mrc")
    with mrcfile.new(map_path, overwrite=True) as mrc:
        mrc.set_data(w.numpy().astype("float32")[None])
        mrc.voxel_size = dx
    console.print(f"  [green]✓[/green] {map_path}")

    d = series.deformation
    record = {
        "model": d.model,
        "height": d.height,
        "hole_radius": d.hole_radius if d.model == "dome" else None,
        "hole_offset": list(d.hole_offset) if d.model == "dome" else None,
        "correlation_length": (d.correlation_length if d.model == "random" else None),
        "burst_dose": d.burst_dose,
        "seed": seed,
        "voxel_size": dx,
        "origin_voxel_zyx": [nz // 2, ny // 2, nx // 2],
        "half_depth": nz * dx / 2.0,
        "deflection_map": os.path.basename(map_path),
        "kinematics": (
            "A specimen point (x, y, z), in Angstrom from origin_voxel_zyx with "
            "z from the midplane, is imaged by tilt i at (x - f_i z' dw/dx, "
            "y - f_i z' dw/dy, z + f_i w), w(x, y) the deflection map and z' = "
            "z clamped to +-half_depth; its material is turned about x by "
            "f_i dw/dy and about y by -f_i dw/dx."
        ),
        "tilts": [
            {
                "index": i,
                "angle": float(angles[i]),
                "dose": float(series.pre_exposure[i]) + float(series.doses[i]) / 2,
                "fraction": series.bend_fraction(i),
            }
            for i in range(len(series))
        ],
    }
    json_path = os.path.join(output_dir, config.filename + "_deformation.json")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)
    console.print(f"  [green]✓[/green] {json_path}")

    if config.deformation_picks_dir is None:
        return
    sources = sorted(Path(config.deformation_picks_dir).glob("*.ndjson"))
    if not sources:
        raise FileNotFoundError(
            f"deformation_picks_dir={config.deformation_picks_dir!r} holds no "
            ".ndjson pick files"
        )
    origin = torch.tensor([nx // 2, ny // 2, nz // 2], dtype=torch.float64) * dx
    root = Path(output_dir) / (config.filename + "_picks")
    for source in sources:
        rows = [json.loads(line) for line in source.read_text().splitlines() if line]
        for i in range(len(series)):
            out = root / f"tilt_{i:03d}" / source.name
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                for row in rows:
                    f.write(json.dumps(_deform_pick(series, row, i, origin)) + "\n")
    console.print(
        f"  [green]✓[/green] {root}/ ({len(sources)} pick files x {len(series)} tilts)"
    )


def _deform_pick(series: TiltSeries, row: dict, i: int, origin: torch.Tensor) -> dict:
    """One copick row as tilt `i` imaged it."""

    def point(p: dict) -> torch.Tensor:
        return torch.tensor([p["x"], p["y"], p["z"]], dtype=torch.float64) - origin

    def moved(p: torch.Tensor) -> dict:
        x, y, z = (series.displaced(p, i) + origin).tolist()
        return {"x": x, "y": y, "z": z}

    row = dict(row)
    at = point(row["location"])
    row["location"] = moved(at)
    if "path" in row:
        row["path"] = [moved(point(p)) for p in row["path"]]
    if "xyz_rotation_matrix" in row:
        rotation = torch.tensor(row["xyz_rotation_matrix"], dtype=torch.float64)
        row["xyz_rotation_matrix"] = (series.rotation(at, i) @ rotation).tolist()
    return row
