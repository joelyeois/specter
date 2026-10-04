"""Profile unchanged 3D FFTs and bounded radial weighting on a production canvas.

The CLI comparison is separate. This diagnostic uses seeded dense input and
holds one float32 and one float16 canvas, like particle material absorption
and the existing free-water field. Timed filtering excludes input generation
and output comparisons. Each candidate output is compared over every voxel.
"""

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time


def load_fft(path, name):
    spec = importlib.util.spec_from_file_location(
        name, Path(path) / "src/specter/fft.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--shape", nargs=3, type=int, default=[1641, 1024, 1024])
    parser.add_argument("--pixel-size", type=float, default=0.731)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--envelope-elements", type=int, default=2**22)
    parser.add_argument("--fft-elements", type=int, default=2**22)
    args = parser.parse_args()
    sys.path.insert(0, args.candidate + "/src")
    import torch
    from specter.io import load_dose_weights
    from specter.potential._damage import dose_damage_envelope

    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    weights, max_frequency = load_dose_weights(args.weights)
    envelope = dose_damage_envelope(
        40.0,
        n_frames=40,
        frame_weights=weights,
        frame_weights_max_frequency=max_frequency,
        voltage=300.0,
    )
    modules = {
        name: load_fft(path, name)
        for name, path in [("before", args.baseline), ("after", args.candidate)]
    }
    modules["after"]._RADIAL_FFT_MAX_ELEMENTS = args.fft_elements
    volume = torch.empty(args.shape, device=args.device)
    # These canvases are counted but not read. The float16 field is existing
    # production physics, not a precision change introduced here.
    residents = [
        torch.empty_like(volume),
        torch.empty_like(volume, dtype=torch.float16),
    ]
    reference = None
    results = []
    chosen_cap = args.envelope_elements
    variants = [
        ("before", None),
        ("after", chosen_cap),
        ("after", chosen_cap),
        ("before", None),
    ]
    if args.sweep:
        variants = [
            ("before", None),
            ("after", 2**26),
            ("after", 2**24),
            ("after", 2**23),
            ("after", 2**22),
            ("after", 2**20),
            ("before", None),
            ("after", chosen_cap),
        ]
    # Warm exact production FFT plans and weighting kernels before timing
    # either revision. Scratch is released before every trial below.
    torch.cuda.manual_seed(57)
    volume.uniform_(0.0, 0.1)
    with torch.no_grad():
        for module in modules.values():
            module._apply_radial_envelope_on_device_(volume, args.pixel_size, envelope)
    torch.cuda.synchronize()
    for variant, cap in variants:
        module = modules[variant]
        if cap is not None:
            module._RADIAL_ENVELOPE_MAX_ELEMENTS = cap
        torch.cuda.manual_seed(57)
        volume.uniform_(0.0, 0.1)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        stages = []
        originals = {}
        for name in [
            "_radial_rfft3",
            "_multiply_radial_envelope",
            "_radial_irfft3_into",
        ]:
            original = getattr(module, name)
            originals[name] = original

            def measured(*a, _function=original, _name=name, **kw):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                result = _function(*a, **kw)
                torch.cuda.synchronize()
                stages.append(
                    dict(
                        name=_name,
                        seconds=time.perf_counter() - start,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    )
                )
                return result

            setattr(module, name, measured)
        start = time.perf_counter()
        with torch.no_grad():
            module._apply_radial_envelope_on_device_(volume, args.pixel_size, envelope)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        for name, original in originals.items():
            setattr(module, name, original)
        if reference is None:
            reference = volume.cpu()
        exact = True
        maximum = 0.0
        error2, ref2 = 0.0, 0.0
        for start_z in range(0, args.shape[0], 8):
            block = volume[start_z : start_z + 8]
            wanted = reference[start_z : start_z + 8].to(args.device)
            block_equal = torch.equal(block, wanted)
            exact &= block_equal
            if block_equal:
                continue
            difference = block.double() - wanted.double()
            maximum = max(maximum, difference.abs().max().item())
            error2 += difference.square().sum().item()
        del block, wanted
        if not exact:
            for start_z in range(0, args.shape[0], 8):
                ref2 += reference[start_z : start_z + 8].double().square().sum().item()
        row = dict(
            variant=variant,
            envelope_elements=cap,
            fft_elements=args.fft_elements if variant == "after" else 2**26,
            shape=args.shape,
            pixel_size_A=args.pixel_size,
            seconds=seconds,
            stages=stages,
            peak_allocated_bytes=max(s["peak_allocated_bytes"] for s in stages),
            peak_reserved_bytes=max(s["peak_reserved_bytes"] for s in stages),
            bitwise_equal=exact,
            max_abs_difference=maximum,
            relative_l2=(error2 / ref2) ** 0.5 if not exact else 0.0,
        )
        print(json.dumps(row), flush=True)
        results.append(row)
        assert exact, row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                resident_bytes=sum(v.numel() * v.element_size() for v in residents),
                warmed_production_plans=True,
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )
    for variant in ["before", "after"]:
        rows = [
            r
            for r in results
            if r["variant"] == variant and r["envelope_elements"] in [None, chosen_cap]
        ]
        print(
            variant,
            "median seconds",
            statistics.median(r["seconds"] for r in rows),
            flush=True,
        )


if __name__ == "__main__":
    main()
