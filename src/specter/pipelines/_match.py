"""`specter match particles`: derive a simulation config that matches a real particle set.

Every stage drives the ordinary `run_particle_stack` with a candidate
`ParticleStackConfig`, so the config this pipeline writes is exercised by
exactly the code that will run it later -- there is no second config-to-model
mapping to drift. The cost is a few small stacks written to the job
directory's ``probes/`` folder.

Stages
------
1. Pose alignment: a noiseless, ice-free render at the refinement's poses is
   correlated with the experimental images pair by pair. A refinement that
   was never aligned to the model fails here, and nothing downstream is
   attempted, because with poses off the mark every later comparison is
   meaningless (see `specter.match.matched_index_correlation`).
2. Metadata to physics: detector MTF and DQE(0) by name, the coincidence
   radius from the detector's exclusion radius and frame rate together with
   the dataset's dose rate, and the radiation-damage envelope from the dose.
   Nothing here is fitted.
3. Probes: ice thickness and neighbour spacing are the two quantities the
   images have to supply. Each candidate is rendered at ``n_probe`` particles
   and scored on the background variance outside the particle against the
   experiment. Probes (and the pose check) run at a box Fourier-cropped by
   ``probe_bin``, against the images cropped the same way, since what they
   measure lives below 10 Å; candidates render concurrently in
   ``probe_workers`` processes.
4. Battery: two seeds at the chosen settings, at the native box, compared
   with the experiment at matched poses. A clearly positive residual
   envelope is applied as a B-factor and the battery is rerun once.
5. Output: ``matched.toml`` for `specter simulate particles`, a Markdown and
   PNG report, and optionally a full stack.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import logging
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import mrcfile
import numpy as np
import torch

import specter
from specter.arrays import fourier_crop
from specter.config import MatchConfig, ParticleStackConfig, validate_config
from specter.cpu_threads import SMALL_OP_THREADS
from specter.detectors import (
    EXCLUSION_RADIUS_PX,
    HARDWARE_FRAME_RATE_HZ,
    TYPICAL_DOSE_RATE_E_PX_S,
    coincidence_occupancy,
    coincidence_radius_for_simulation,
    dqe0_for_detector,
)
from specter.io import extract_parameters_from_csfile, extract_parameters_from_starfile
from specter.match import (
    MatchReport,
    annulus_std_profile,
    band_power_ratio,
    dumps_toml,
    edge_band_means,
    load_experimental_images,
    matched_index_correlation,
    matched_pose_snr,
    render_report,
    twin_test,
    water_ring_excess,
)
from specter.match._metadata import recorded_box, rescale_metadata
from specter.match._report import DerivedValue
from specter.pdb import PDB

from ._common import (
    _console,
    _format_elapsed,
    _parse_device_pool,
    _section,
    _tracked_output_dir,
)
from ._particles import run_particle_stack

#: Detectors that count without coincidence loss at cryo-EM rates.
_NO_COINCIDENCE = {"perfect"}

#: Coarsest pixel a probe may be binned to, in Angstrom. The pose check
#: correlates 60-15 Å and the probes score background variance in wide
#: annuli; past a 10 Å Nyquist both lose what they measure.
_MAX_PROBE_PIXEL_A = 5.0
#: Smallest box a probe may be binned to.
_MIN_PROBE_BOX = 32


def _metadata_kwargs(config: MatchConfig) -> dict[str, str]:
    if config.metadata_path.endswith(".cs"):
        return {"cs_path": config.metadata_path}
    if config.metadata_path.endswith(".star"):
        return {"star_path": config.metadata_path}
    raise ValueError(
        f"metadata_path={config.metadata_path!r}: expected a .cs or .star file"
    )


def _read_stack(path: str) -> torch.Tensor:
    with mrcfile.mmap(path, permissive=True) as m:
        return torch.as_tensor(np.asarray(m.data, dtype=np.float32).copy())


def _job(
    base: dict[str, Any],
    out_dir: str,
    name: str,
    n: int,
    seed: int,
    **overrides: Any,
) -> dict[str, Any]:
    """The `ParticleStackConfig` kwargs for one simulation of ``n`` particles."""
    kwargs = dict(base)
    kwargs.update(overrides)
    kwargs.update(n_particles=n, seed=seed, output_dir=out_dir, filename=name)
    return kwargs


def _simulate_job(
    kwargs: dict[str, Any], log_path: str | None = None, threads: int | None = None
) -> str:
    """
    Render one stack through `run_particle_stack` and return its path.

    Runs in the calling process or in a probe worker. A worker sends the
    simulation's own console output to ``log_path``, so concurrent probes do
    not interleave on the terminal, and mirrors the parent's CPU thread cap.
    """
    if threads is not None:
        torch.set_num_threads(threads)
    cfg = ParticleStackConfig(**kwargs)
    if log_path is None:
        run_particle_stack(cfg)
    else:
        with (
            open(log_path, "w") as fh,
            contextlib.redirect_stdout(fh),
            contextlib.redirect_stderr(fh),
        ):
            specter.set_verbosity(logging.INFO)
            run_particle_stack(cfg)
    return os.path.join(kwargs["output_dir"], f"{kwargs['filename']}.mrcs")


def _warm_job(kwargs: dict[str, Any], threads: int) -> None:
    """Parse the structure and build its potential into a worker's cache."""
    from ._particles import _structure_and_potential

    torch.set_num_threads(threads)
    cfg = ParticleStackConfig(**kwargs)
    if cfg.cs_path is not None:
        pixel_size = float(
            extract_parameters_from_csfile(cfg.cs_path, n_particles=1)[1]
        )
    elif cfg.star_path is not None:
        pixel_size = float(
            extract_parameters_from_starfile(cfg.star_path, n_particles=1)[1]
        )
    else:
        pixel_size = cfg.pixel_size
    with contextlib.redirect_stdout(io.StringIO()):
        _structure_and_potential(cfg, pixel_size, build=True)


class _ProbeRunner:
    """
    Run probe simulations, concurrently across worker processes when asked.

    Processes rather than threads, because `specter.seed` and torch's RNG
    are process-global and two probes in one process would share a stream.
    Every worker keeps `run_particle_stack`'s per-process potential cache,
    so a structure is parsed and rendered once per worker rather than once
    per probe. Jobs are dealt round-robin over the devices ``device`` names.
    """

    def __init__(self, device: str, workers: int) -> None:
        self.devices = _parse_device_pool(device)
        self.workers = len(self.devices) if workers <= 0 else int(workers)
        self._pool: ProcessPoolExecutor | None = None
        # Workers share the host: each gets its slice of the parent's thread
        # pool, so four cold potential builds do not run 4 x 128 threads.
        self.threads = max(SMALL_OP_THREADS, torch.get_num_threads() // self.workers)
        if self.workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=mp.get_context("spawn")
            )

    def warm(self, kwargs: dict[str, Any]) -> None:
        """
        Build the structure's potential in every worker, concurrently.

        A worker's first job otherwise pays the parse and build while the
        others wait on it in the same stage; warmed up front, they overlap.
        """
        if self._pool is None:
            return
        futures = [
            self._pool.submit(_warm_job, dict(kwargs), self.threads)
            for _ in range(self.workers)
        ]
        for f in futures:
            f.result()

    def __enter__(self) -> _ProbeRunner:
        return self

    def __exit__(self, *exc: object) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def run(self, jobs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Render every job and return the stacks in the same order."""
        for i, job in enumerate(jobs):
            job["device"] = self.devices[i % len(self.devices)]
        if self._pool is None:
            paths = [_simulate_job(job) for job in jobs]
        else:
            futures = [
                self._pool.submit(
                    _simulate_job,
                    job,
                    os.path.join(job["output_dir"], f"{job['filename']}.log"),
                    self.threads,
                )
                for job in jobs
            ]
            paths = [f.result() for f in futures]
        return [_read_stack(p) for p in paths]


def _probe_bin(requested: int, box: int, pixel_size: float) -> int:
    """The Fourier-crop factor the probes use: the request, capped by the limits above."""
    b = max(1, int(requested))
    while b > 1 and (pixel_size * b > _MAX_PROBE_PIXEL_A or box // b < _MIN_PROBE_BOX):
        b -= 1
    return b


def _bin_images(
    images: torch.Tensor, pixel_size: float, box_p: int
) -> tuple[torch.Tensor, float]:
    """
    Fourier-crop a stack to ``box_p`` px, keeping each image's own standard deviation.

    The experiment's normalisation is whatever its refinement applied, and the
    probes are scored against it as is; cropping removes the noise power above
    the new Nyquist, so each image is rescaled back to the deviation it had.
    """
    box = int(images.shape[-1])
    if box_p == box:
        return images, pixel_size
    std = images.std(dim=(-2, -1), keepdim=True)
    cropped, new_px = fourier_crop(images, pixel_size, pixel_size * box / box_p)
    if int(cropped.shape[-1]) != box_p:
        raise RuntimeError(
            f"Fourier crop to {box_p} px produced {tuple(cropped.shape)}"
        )
    cropped = cropped * (std / cropped.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8))
    return cropped.contiguous(), pixel_size * box / box_p


def _profile_distance(sim: torch.Tensor, exp: torch.Tensor, first_bin: int) -> float:
    """RMS difference of the low-passed annulus std profiles outside the particle."""
    a = np.array(annulus_std_profile(sim))[first_bin:]
    b = np.array(annulus_std_profile(exp))[first_bin:]
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else float("nan")


def run_match(config: MatchConfig) -> MatchReport:
    """
    Derive a matching `ParticleStackConfig` for a real particle set and report
    how close the match is.

    Parameters
    ----------
    config : MatchConfig
        See :class:`specter.config.MatchConfig`.

    Returns
    -------
    MatchReport
        Everything decided and measured. Also written to the run directory as
        ``matched.toml``, ``match_report.md`` and ``match_report.png``.
    """
    validate_config(config)
    specter.set_verbosity(logging.INFO)
    t_start = time.perf_counter()
    seed = config.seed if config.seed is not None else 0

    with (
        _tracked_output_dir(config, "match") as out_dir,
        _ProbeRunner(config.device, config.probe_workers) as runner,
    ):
        os.makedirs(out_dir, exist_ok=True)
        probe_dir = os.path.join(out_dir, "probes")
        os.makedirs(probe_dir, exist_ok=True)

        # ---------------------------------------------------------------- 1
        _section("Reading the refinement and its images")
        n_needed = max(config.n_probe, config.n_battery)
        meta = _metadata_kwargs(config)
        if "cs_path" in meta:
            params = extract_parameters_from_csfile(
                config.metadata_path, n_particles=n_needed
            )
        else:
            params = extract_parameters_from_starfile(
                config.metadata_path, n_particles=n_needed
            )
        voltage = float(params[0])
        pixel_size = float(params[1])
        n_available = int(params[3].shape[0])
        if n_available < n_needed:
            raise ValueError(
                f"{config.metadata_path} has {n_available} particles; n_probe and n_battery "
                f"need {n_needed}. Lower them or use a larger set."
            )
        exp = load_experimental_images(
            config.metadata_path, n=n_needed, images_path=config.images_path
        )
        if exp.shape[-1] != exp.shape[-2]:
            raise ValueError(f"experimental images are not square: {tuple(exp.shape)}")
        box = int(exp.shape[-1])
        # Images binned after extraction leave the metadata describing a box
        # they no longer have; rendering into the recorded pixel size would
        # then put the structure in a box too small by the same factor. Write
        # a rescaled copy and point every probe at it.
        recorded = recorded_box(config.metadata_path)
        pixel_note = ""
        if recorded is not None and recorded != box:
            ext = os.path.splitext(config.metadata_path)[1]
            rescaled = os.path.join(out_dir, f"metadata_rescaled{ext}")
            new_pixel = rescale_metadata(config.metadata_path, box, rescaled)
            pixel_note = f"rescaled from {pixel_size:.4f} Å at {recorded} px"
            pixel_size = new_pixel
            meta = {next(iter(meta)): rescaled}
            _console.print(
                f"[yellow]Images are {box} px but the metadata records {recorded} px: "
                f"using a rescaled copy at {pixel_size:.4f} Å ({rescaled}).[/yellow]"
            )
        _console.print(
            f"{n_available} particles at {pixel_size:.4f} Å, {box} px box, {voltage:.0f} kV"
        )
        if box > 512:
            _console.print(
                "[yellow]Box is larger than 512 px; the final two-seed comparison simulates "
                "at this size. A Fourier-cropped stack passed as --images_path is faster.[/yellow]"
            )

        # Probes measure the pose alignment (60-15 Å) and the background
        # variance (wide annuli): nothing above 10 Å. They render at a
        # Fourier-cropped box against the images cropped the same way, with a
        # metadata copy rescaled to that box; the battery keeps the native box.
        bin_ = _probe_bin(config.probe_bin, box, pixel_size)
        box_p = box // bin_ if bin_ > 1 else box
        box_p -= box_p % 2
        exp_p, pixel_size_p = _bin_images(exp, pixel_size, box_p)
        probe_geometry: dict[str, Any] = {"n_pixels": box_p}
        if box_p != box:
            ext = os.path.splitext(config.metadata_path)[1]
            probe_meta = os.path.join(out_dir, f"metadata_probe{ext}")
            key, path = next(iter(meta.items()))
            rescale_metadata(path, box_p, probe_meta, current_box=box)
            probe_geometry[key] = probe_meta
        _console.print(
            f"probes: {config.n_probe} particles at {box_p} px / {pixel_size_p:.3f} Å "
            f"({bin_}x binned), {runner.workers} worker(s) on {', '.join(runner.devices)}"
        )

        pdb = PDB(
            config.pdb_source,
            assembly=config.assembly,
            pdb_cache_dir=config.pdb_cache_dir,
            compute_atom_species=False,
        )
        diameter = float(pdb.max_diameter)

        base: dict[str, Any] = dict(
            pdb_source=config.pdb_source,
            assembly=config.assembly,
            n_pixels=box,
            dose=config.dose,
            n_frames=config.n_frames,
            scattering_model="multislice",
            noise_model="poisson",
            ice_model="gd",
            ice_thickness=0.0,
            crowd_min_distance=diameter,
            detector_model="none",
            coincidence_radius=0.0,
            dose_envelope=True,
            potential_scale=1.0,
            pad_fft=True,
            device=config.device,
            batchsize=4,
            normalize_particles=True,
            pdb_cache_dir=config.pdb_cache_dir,
            monomer_library_path=config.monomer_library_path,
            **meta,
        )

        runner.warm(_job(base, probe_dir, "warm", 1, seed, **probe_geometry))

        # ---------------------------------------------------------------- 2
        _section("Checking that the poses are aligned to the model")
        exp_probe = exp_p[: config.n_probe]
        (pose_stack,) = runner.run(
            [
                _job(
                    base,
                    probe_dir,
                    "pose_check",
                    config.n_probe,
                    seed,
                    **probe_geometry,
                    noise_model="none",
                    ice_model="none",
                    crowd_min_distance=0,
                    dose_envelope=False,
                )
            ]
        )
        pose = matched_index_correlation(pose_stack, exp_probe, pixel_size_p)
        report = MatchReport(pose=pose, pixel_size=pixel_size)
        _console.print(
            f"matched {pose.matched:.3f} vs shuffled {pose.shuffled:.3f}, "
            f"{pose.fraction_above:.0%} of pairs above, z = {pose.z_score:.1f}: "
            f"{'PASS' if pose.passed else 'FAIL'}"
        )
        if not pose.passed:
            report.warnings.append(
                "The refinement's poses do not reproduce the experimental views: simulated "
                "particle i does not correlate with experimental particle i above a random "
                "pairing. Either the refinement was not aligned to the atomic model (run an "
                "Align 3D of the map against the model and re-extract the particles from it), "
                "or pdb_source is not the structure in the images. No parameter was derived."
            )
            render_report(report, pose_stack, exp_probe, out_dir)
            _write_matched_toml(out_dir, base, report, complete=False)
            return report

        # ---------------------------------------------------------------- 3
        _section("Deriving detector and envelope settings from the acquisition card")
        det = config.detector_model
        detector_model = "none" if det == "unknown" else det
        report.derived.append(DerivedValue("voltage", voltage, "metadata"))
        report.derived.append(
            DerivedValue("pixel_size", round(pixel_size, 4), "metadata", pixel_note)
        )
        report.derived.append(
            DerivedValue("dose", config.dose, "metadata", "e-/Å² per movie")
        )
        occ: float | None = None  # coincidence occupancy, when a detector is calibrated
        if det == "unknown":
            report.warnings.append(
                "detector_model is unknown: no MTF, DQE(0) or coincidence loss applied. "
                "The simulation will carry more high-frequency signal than the data."
            )
            report.derived.append(
                DerivedValue("detector_model", "none", "fallback", "unknown detector")
            )
            cr = 0.0
        else:
            report.derived.append(
                DerivedValue(
                    "detector_model",
                    det,
                    "detector table",
                    f"DQE(0) = {dqe0_for_detector(det):.2f}",
                )
            )
            if det in _NO_COINCIDENCE:
                cr = 0.0
                report.derived.append(
                    DerivedValue(
                        "coincidence_radius",
                        0.0,
                        "detector table",
                        "no coincidence loss",
                    )
                )
            elif det not in EXCLUSION_RADIUS_PX or det not in HARDWARE_FRAME_RATE_HZ:
                cr = 0.0
                report.warnings.append(
                    f"No coincidence calibration for {det}: coincidence_radius set to 0. A "
                    "beam-only dose-rate series would supply it."
                )
                report.derived.append(
                    DerivedValue(
                        "coincidence_radius", 0.0, "fallback", "uncalibrated detector"
                    )
                )
            else:
                rate = config.dose_rate
                source = "metadata"
                if rate is None:
                    rate = TYPICAL_DOSE_RATE_E_PX_S[det]
                    source = "fallback"
                    report.warnings.append(
                        f"dose_rate not given: using {det}'s typical {rate:g} e/px/s. At physical "
                        "rates the effect of this value on the images is small."
                    )
                occ = coincidence_occupancy(
                    EXCLUSION_RADIUS_PX[det], rate, HARDWARE_FRAME_RATE_HZ[det]
                )
                cr = coincidence_radius_for_simulation(
                    occ, config.dose, pixel_size, config.n_frames
                )
                report.derived.append(
                    DerivedValue(
                        "coincidence_radius",
                        round(cr, 3),
                        source,
                        f"r = {EXCLUSION_RADIUS_PX[det]} physical px, {rate:g} e/px/s at "
                        f"{HARDWARE_FRAME_RATE_HZ[det]:.0f} Hz: occupancy {occ:.3f} per cell per frame",
                    )
                )
        report.derived.append(
            DerivedValue(
                "dose_envelope",
                True,
                "fixed",
                "Grant & Grigorieff 2015, exposure-averaged",
            )
        )
        if config.energy_filter is False:
            report.warnings.append(
                "No energy filter: on every unfiltered dataset tried so far the experiment carried "
                "a broad signal-to-noise deficit no forward-model parameter reproduces (an inelastic "
                "background). Expect a residual in the SNR ratio below."
            )
        elif config.energy_filter is None:
            report.warnings.append("energy_filter not stated; recorded as unknown.")
        base.update(detector_model=detector_model, coincidence_radius=cr)
        # The same occupancy expressed in the probes' coarser pixel.
        cr_p = (
            coincidence_radius_for_simulation(
                occ, config.dose, pixel_size_p, config.n_frames
            )
            if occ is not None
            else 0.0
        )
        probe_base: dict[str, Any] = {
            **base,
            **probe_geometry,
            "coincidence_radius": cr_p,
        }

        # ---------------------------------------------------------------- 4
        _section("Probing ice thickness and neighbour spacing against the images")
        radius_px = 0.5 * diameter / pixel_size_p
        first_bin = int(math.ceil(radius_px / 20.0))  # annuli are 20 px wide
        stacks = runner.run(
            [
                _job(
                    probe_base,
                    probe_dir,
                    f"probe_ice{ice:g}",
                    config.n_probe,
                    seed,
                    ice_thickness=ice,
                )
                for ice in config.ice_candidates
            ]
        )
        scores: list[tuple[float, float]] = [
            (float(ice), _profile_distance(stack, exp_probe, first_bin))
            for ice, stack in zip(config.ice_candidates, stacks, strict=True)
        ]
        report.probe_scores["ice_thickness"] = scores
        ice_best = min(scores, key=lambda t: t[1])[0]
        base.update(ice_thickness=ice_best)
        probe_base.update(ice_thickness=ice_best)
        report.derived.append(
            DerivedValue(
                "ice_thickness",
                ice_best,
                "probe",
                "background variance outside the particle",
            )
        )

        stacks = runner.run(
            [
                _job(
                    probe_base,
                    probe_dir,
                    f"probe_crowd{mult:g}",
                    config.n_probe,
                    seed,
                    crowd_min_distance=0 if mult == 0 else mult * diameter,
                )
                for mult in config.crowd_candidates
            ]
        )
        scores = [
            (float(mult), _profile_distance(stack, exp_probe, first_bin))
            for mult, stack in zip(config.crowd_candidates, stacks, strict=True)
        ]
        report.probe_scores["crowd_multiple"] = scores
        crowd_best = min(scores, key=lambda t: t[1])[0]
        crowd_min_distance = 0 if crowd_best == 0 else crowd_best * diameter
        base.update(crowd_min_distance=crowd_min_distance)
        report.derived.append(
            DerivedValue(
                "crowd_min_distance",
                round(crowd_min_distance, 1),
                "probe",
                f"{crowd_best:g} x the structure's {diameter:.0f} Å diameter"
                if crowd_best
                else "no neighbours",
            )
        )

        # ---------------------------------------------------------------- 5
        _section("Comparing two seeds with the experiment at matched poses")
        exp_b = exp[: config.n_battery]

        def battery(suffix: str) -> tuple[torch.Tensor, torch.Tensor]:
            a, b = runner.run(
                [
                    _job(
                        base,
                        probe_dir,
                        f"battery_seed0{suffix}",
                        config.n_battery,
                        seed,
                    ),
                    _job(
                        base,
                        probe_dir,
                        f"battery_seed1{suffix}",
                        config.n_battery,
                        seed + 1,
                    ),
                ]
            )
            return a, b

        sim_a, sim_b = battery("")
        snr = matched_pose_snr(sim_a, sim_b, exp_b, pixel_size)
        bfactor: float | None = None
        if math.isfinite(snr.residual_bfactor) and snr.residual_bfactor > 20.0:
            bfactor = round(snr.residual_bfactor, 0)
            base.update(bfactor=bfactor)
            _console.print(
                f"residual envelope B = {bfactor:.0f} Å²; applying it and re-rendering"
            )
            sim_a, sim_b = battery("_b")
            snr = matched_pose_snr(sim_a, sim_b, exp_b, pixel_size)
        report.derived.append(
            DerivedValue(
                "bfactor",
                bfactor if bfactor is not None else 0.0,
                "measured" if bfactor else "fixed",
                "Guinier slope of the matched-pose signal ratio, 10-4 Å",
            )
        )
        report.snr = snr
        report.twin = twin_test(sim_a, sim_b, exp_b)
        report.band_ratio = band_power_ratio(sim_a, exp_b, pixel_size)
        report.edge_sim, report.edge_exp = (
            edge_band_means(sim_a),
            edge_band_means(exp_b),
        )
        report.annulus_sim, report.annulus_exp = (
            annulus_std_profile(sim_a),
            annulus_std_profile(exp_b),
        )
        report.ring_sim, report.ring_exp = (
            water_ring_excess(sim_a, pixel_size),
            water_ring_excess(exp_b, pixel_size),
        )
        report.n_battery = config.n_battery
        excess = [r for r in snr.ratio[:3] if math.isfinite(r)]
        if excess and max(excess) > 3.0:
            report.warnings.append(
                f"The simulation is {max(excess):.0f}x cleaner than the experiment in a band coarser "
                "than 6.7 Å after every derivable parameter is set. This is not a parameter; "
                "on the datasets tried so far it tracks the absence of an energy filter."
            )
        _console.print(
            "SNR ratio sim/exp per band: "
            + ", ".join(f"{r:.2f}" for r in snr.ratio)
            + f" | twin d = {report.twin.cohen_d:.2f}"
        )

        # ---------------------------------------------------------------- 6
        _section("Writing matched.toml and the report")
        toml_path = _write_matched_toml(out_dir, base, report, complete=True)
        md_path, png_path = render_report(report, sim_a, exp_b, out_dir)
        _console.print(
            f"[green]✓[/green] {toml_path}\n[green]✓[/green] {md_path}\n[green]✓[/green] {png_path}"
        )
        _console.print(f"[bold]{report.verdict}[/bold]")

        if config.write_stack > 0:
            _section(
                f"Simulating {config.write_stack} particles with the matched config"
            )
            # In-process, so its progress shows on the terminal.
            _simulate_job(
                _job(
                    base,
                    out_dir,
                    "matched_particles",
                    config.write_stack,
                    seed + 2,
                    device=runner.devices[0],
                )
            )

        _log_to_job(out_dir, report)
        _console.print(
            f"[dim]Done in {_format_elapsed(time.perf_counter() - t_start)}[/dim]"
        )
        return report


def _write_matched_toml(
    out_dir: str, base: dict[str, Any], report: MatchReport, complete: bool
) -> str:
    """Write the derived `ParticleStackConfig` as TOML, grouped like `configs/particle.toml`."""
    fields = {f.name for f in dataclasses.fields(ParticleStackConfig)}
    values = {
        k: v
        for k, v in base.items()
        if k in fields and k not in {"output_dir", "filename", "batchsize"}
    }
    values.setdefault("n_particles", None)
    tables = {
        "specimen": {
            k: values.pop(k)
            for k in ("pdb_source", "assembly", "n_pixels", "ice_thickness")
            if k in values
        },
        "dataset": {
            k: values.pop(k)
            for k in ("cs_path", "star_path", "n_particles")
            if k in values
        },
        "microscope": {
            k: values.pop(k)
            for k in (
                "dose",
                "n_frames",
                "detector_model",
                "coincidence_radius",
                "dose_envelope",
                "bfactor",
            )
            if k in values
        },
        "models": {
            k: values.pop(k)
            for k in ("scattering_model", "noise_model", "ice_model")
            if k in values
        },
        "crowding": {k: values.pop(k) for k in ("crowd_min_distance",) if k in values},
        "compute": {k: values.pop(k) for k in ("device",) if k in values},
        "advanced": values,
    }
    header = (
        "Written by `specter match particles`"
        + ("" if complete else " -- INCOMPLETE: the pose-alignment check failed")
        + ".\nRun with: specter simulate particles --config matched.toml --n_particles <N>\n"
        + "\n".join(f"{d.name}: {d.value} ({d.source})" for d in report.derived)
    )
    path = os.path.join(out_dir, "matched.toml")
    with open(path, "w") as fh:
        fh.write(dumps_toml(tables, header=header))
    return path


def _log_to_job(out_dir: str, report: MatchReport) -> None:
    """Merge the report summary into job.json when the run is tracked."""
    job_json = os.path.join(out_dir, "job.json")
    if not os.path.exists(job_json):
        return
    import json

    data = json.loads(open(job_json).read())
    data.setdefault("params", {})["results"] = report.summary()
    with open(job_json, "w") as fh:
        fh.write(json.dumps(data, indent=2))
