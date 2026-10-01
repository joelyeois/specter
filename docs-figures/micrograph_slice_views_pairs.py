"""Compare original and view-based propagation on one full shipped specimen."""

import argparse
import gc
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--checkout", required=True)
p.add_argument("--baseline", required=True)
p.add_argument("--volume", required=True)
p.add_argument("--output", required=True)
p.add_argument("--device", default="cuda:3")
p.add_argument("--placements", nargs="+", default=["resident", "streamed"])
a = p.parse_args()
sys.path.insert(0, a.checkout + "/src")
import torch  # noqa: E402
from specter.scattering import IterativeScattering  # noqa: E402

torch.set_num_threads(4)
torch.cuda.set_device(a.device)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
name = "specter.scattering._micrograph_baseline"
spec = importlib.util.spec_from_file_location(
    name, a.baseline + "/src/specter/scattering/_iterative.py"
)
old = importlib.util.module_from_spec(spec)
sys.modules[name] = old
spec.loader.exec_module(old)
host = torch.load(a.volume, map_location="cpu", mmap=True, weights_only=True)
model = IterativeScattering(
    nxy=4096, pixel_size=1.0, voltage=300, alpha=0.1, progressbars=False
).to(a.device)
new_iter = model._iter_slices
old_iter = types.MethodType(old.IterativeScattering._iter_slices, model)
result = {
    "gpu": torch.cuda.get_device_name(),
    "device": a.device,
    "shape": list(host.shape),
    "trials": [],
}
with torch.no_grad():
    for placement in a.placements:
        volume = host.to(a.device) if placement == "resident" else host
        # Warm FFT plans and storage kernels at the actual field size.
        warm = model(volume[:, :2], pose=0)
        del warm
        for pair in range(3):
            saved = {}
            for variant in ["before", "after"] if pair != 1 else ["after", "before"]:
                gc.collect()
                torch.cuda.empty_cache()
                model._iter_slices = old_iter if variant == "before" else new_iter
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_allocated = torch.cuda.memory_allocated()
                t = time.perf_counter()
                wave = model(volume, pose=0)
                torch.cuda.synchronize()
                trial = {
                    "placement": placement,
                    "pair": pair,
                    "variant": variant,
                    "seconds": time.perf_counter() - t,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "start_allocated_bytes": start_allocated,
                }
                saved[variant] = wave.cpu()
                del wave
                result["trials"].append(trial)
                print(json.dumps(trial), flush=True)
            delta = saved["after"] - saved["before"]
            result.setdefault("precision", []).append(
                {
                    "placement": placement,
                    "pair": pair,
                    "bitwise_equal": torch.equal(saved["before"], saved["after"]),
                    "max_abs_error": delta.abs().max().item(),
                    "relative_l2": (
                        torch.linalg.vector_norm(delta)
                        / torch.linalg.vector_norm(saved["before"])
                    ).item(),
                }
            )
            if pair == 0:
                for variant, wave in saved.items():
                    torch.save(wave, out / f"{placement}_{variant}.pt")
            del saved, delta
        del volume
        torch.cuda.empty_cache()
(out / "pairs.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result["precision"]), flush=True)
