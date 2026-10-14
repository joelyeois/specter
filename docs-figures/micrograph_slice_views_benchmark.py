"""Measure the shipped CLI and optionally retain/replay its full specimen.

Runs ``specter simulate micrograph`` on the checkout's ``configs/micrograph.toml``
and times each stage of the object API it drives (`specter.imaging`):
assembly (`Field.draw`), placement (`Sample.placed`), propagation
(`IterativeScattering.forward`, inside `Sample.scatter`) and detection
(`ExitWave.intensity` plus `Image.detect`, which together are what the
generator-era `Detector.forward` did). The CUDA peaks, the pixel-grid exit
wave (`exitwave.pt`) and the noiseless intensity ``|aberrated wave|^2`` on the
image grid (`clean.pt`) are the same quantities the generator-era runs
recorded, so ``measurement.json`` keeps its keys; ``intensity_seconds`` and
``detect_seconds`` split ``detector_seconds`` in two.

``--replay-from`` swaps the assembled volume for a captured ``volume.pt``
(without its absorption field, as before) and draws the detector noise from
seed 193, so two replays image one specimen with one noise realisation. The
hooks are the object API's, so ``--checkout`` must have `specter.imaging`.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path


def main() -> None:
    # Guarded so the module imports without arguments (tests/test_docs_figures_import.py).
    capture_seconds = 0
    p = argparse.ArgumentParser()
    p.add_argument("--checkout", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--capture-volume", action="store_true")
    p.add_argument("--replay-from")
    a = p.parse_args()
    sys.path.insert(0, a.checkout + "/src")
    import torch  # noqa: E402
    from specter.cli._cli import cli  # noqa: E402
    from specter.imaging import ExitWave, Field, Image, Sample  # noqa: E402
    from specter.imaging import _field  # noqa: E402
    from specter.scattering import IterativeScattering  # noqa: E402
    from specter.settings import Propagation  # noqa: E402

    torch.set_num_threads(4)
    torch.cuda.set_device(a.device)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    stats = {
        "checkout": a.checkout,
        "device": a.device,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "stages": {},
    }
    draw = Field.draw
    placed = Sample.placed
    propagate = IterativeScattering.forward
    apply_ctf = ExitWave.apply_ctf
    intensity = ExitWave.intensity
    detect = Image.detect

    def timed(name, fn, *args, **kwargs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        stats["stages"][name] = time.perf_counter() - t
        return result

    def make(
        self,
        *,
        voltage,
        exposure=None,
        seed=None,
        index=0,
        keep_dry=False,
        objective_aperture=None,
        device="cpu",
        **propagation,
    ):
        nonlocal capture_seconds
        torch.cuda.synchronize()
        t = time.perf_counter()
        if a.replay_from:
            volume = torch.load(
                Path(a.replay_from) / "volume.pt", map_location="cpu", weights_only=True
            )
            # No absorption field, as the generator-era replay; a uniform
            # absorption is a property of the settings, not of the draw.
            _, uniform = _field._absorption(
                Propagation(**propagation),
                objective_aperture,
                voltage,
                self.icemaker is not None,
                self.ice,
            )
            sample = Sample(
                volume,
                self.voxel_size,
                uniform_absorption=uniform,
                profile=self.ice.profile,
                damaged=self.damaged,
            )
        else:
            sample = draw(
                self,
                voltage=voltage,
                exposure=exposure,
                seed=seed,
                index=index,
                keep_dry=keep_dry,
                objective_aperture=objective_aperture,
                device=device,
                **propagation,
            )
        torch.cuda.synchronize()
        stats["stages"]["assembly_seconds"] = time.perf_counter() - t
        print("ASSEMBLED", stats["stages"], tuple(sample.volume.shape), flush=True)
        if a.capture_volume:
            t = time.perf_counter()
            torch.save(sample.volume, out / "volume.pt")
            capture_seconds += time.perf_counter() - t
        stats["volume_shape"] = list(sample.volume.shape)
        return sample

    def placement(self, device):
        result = timed("placement_seconds", placed, self, device)
        stats["volume_device"] = str(result.volume.device)
        print("PLACED", stats["volume_device"], flush=True)
        return result

    def scatter(self, *args, **kwargs):
        return timed("propagation_seconds", propagate, self, *args, **kwargs)

    def keep(name, fn):
        # On the host, so the diagnostics add nothing to the CUDA peaks.
        t = time.perf_counter()
        torch.save(fn(), out / name)
        stats["diagnostics_seconds"] = (
            stats.get("diagnostics_seconds", 0.0) + time.perf_counter() - t
        )

    def lens(self, *args, **kwargs):
        host = replace(self, array=self.array.detach().cpu())
        keep("exitwave.pt", host.pixels)
        return apply_ctf(self, *args, **kwargs)

    def form(self, *args, **kwargs):
        keep("clean.pt", lambda: self.array.detach().cpu().abs().square())
        return timed("intensity_seconds", intensity, self, *args, **kwargs)

    def record(self, *args, **kwargs):
        if a.replay_from:
            kwargs["seed"] = 193
        result = timed("detect_seconds", detect, self, *args, **kwargs)
        stages = stats["stages"]
        stages["detector_seconds"] = (
            stages["intensity_seconds"] + stages["detect_seconds"]
        )
        stats["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        stats["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        return result

    Field.draw = make
    Sample.placed = placement
    IterativeScattering.forward = scatter
    ExitWave.apply_ctf = lens
    ExitWave.intensity = form
    Image.detect = record
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    cli(
        prog_name="specter",
        standalone_mode=False,
        args=[
            "simulate",
            "micrograph",
            "--config",
            a.checkout + "/configs/micrograph.toml",
            "--device",
            a.device,
            "--seed",
            "93",
            "--output_dir",
            str(out),
        ],
    )
    torch.cuda.synchronize()
    stats["cli_seconds_excluding_volume_capture"] = (
        time.perf_counter() - t - capture_seconds
    )
    stats["capture_seconds"] = capture_seconds
    (out / "measurement.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
