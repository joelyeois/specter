"""Measure the shipped CLI and optionally retain/replay its full specimen."""

import argparse
import json
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--checkout", required=True)
p.add_argument("--output", required=True)
p.add_argument("--device", default="cuda:3")
p.add_argument("--capture-volume", action="store_true")
p.add_argument("--replay-from")
a = p.parse_args()
sys.path.insert(0, a.checkout + "/src")
import torch  # noqa: E402
import specter  # noqa: E402
from specter.cli._cli import cli  # noqa: E402
from specter.imagegenerator import MicrographGenerator  # noqa: E402

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
capture_seconds = 0
models = []
generate = MicrographGenerator._generate_volume
place = MicrographGenerator._ensure_volume_placed
forward = MicrographGenerator.forward


def make(self, *args, **kwargs):
    global capture_seconds
    torch.cuda.synchronize()
    t = time.perf_counter()
    if a.replay_from:
        self.volume = torch.load(
            Path(a.replay_from) / "volume.pt", map_location="cpu", weights_only=True
        )
        self.absorption_potential = None
    else:
        generate(self, *args, **kwargs)
    torch.cuda.synchronize()
    stats["stages"]["assembly_seconds"] = time.perf_counter() - t
    print("ASSEMBLED", stats["stages"], tuple(self.volume.shape), flush=True)
    if a.capture_volume:
        t = time.perf_counter()
        torch.save(self.volume, out / "volume.pt")
        capture_seconds += time.perf_counter() - t
    stats["volume_shape"] = list(self.volume.shape)


def placement(self):
    torch.cuda.synchronize()
    t = time.perf_counter()
    place(self)
    torch.cuda.synchronize()
    stats["stages"]["placement_seconds"] = time.perf_counter() - t
    stats["volume_device"] = str(self.volume.device)
    print("PLACED", stats["volume_device"], flush=True)


def run(self, *args, **kwargs):
    old_scatter = self.iterative_scattering.forward
    old_detect = self.detector.forward

    def scatter(*sa, **sk):
        torch.cuda.synchronize()
        t = time.perf_counter()
        result = old_scatter(*sa, **sk)
        torch.cuda.synchronize()
        stats["stages"]["propagation_seconds"] = time.perf_counter() - t
        return result

    def detect(*da, **dk):
        if a.replay_from:
            specter.seed(193)
        torch.cuda.synchronize()
        t = time.perf_counter()
        result = old_detect(*da, **dk)
        torch.cuda.synchronize()
        stats["stages"]["detector_seconds"] = time.perf_counter() - t
        return result

    self.iterative_scattering.forward = scatter
    self.detector.forward = detect
    result = forward(self, *args, **kwargs)
    stats["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    stats["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    torch.save(self.exitwaves.detach().cpu(), out / "exitwave.pt")
    torch.save(self.detector_waves.detach().abs().square().cpu(), out / "clean.pt")
    self.iterative_scattering.forward = old_scatter
    self.detector.forward = old_detect
    models.append(self)
    return result


MicrographGenerator._generate_volume = make
MicrographGenerator._ensure_volume_placed = placement
MicrographGenerator.forward = run
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
