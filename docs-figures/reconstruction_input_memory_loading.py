"""Measure loading alone and hash every preprocessed particle and parameter."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cs-file", required=True)
    parser.add_argument("--mrc-file", required=True)
    parser.add_argument("--dose", required=True, type=float)
    parser.add_argument("--halfset", default="all")
    args = parser.parse_args()
    sys.path.insert(0, args.checkout + "/src")
    import torch
    from specter.ghostbuster import Ghostbuster

    torch.set_num_threads(4)
    start = time.perf_counter()
    g = Ghostbuster(args.cs_file, args.mrc_file, args.dose, halfset=args.halfset)
    loading_seconds = time.perf_counter() - start
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    def digest(tensor):
        return hashlib.sha256(
            memoryview(tensor.detach().cpu().contiguous().numpy())
        ).hexdigest()

    result = dict(
        checkout=args.checkout,
        cs_file=args.cs_file,
        mrc_file=args.mrc_file,
        dose=args.dose,
        halfset=args.halfset,
        shape=list(g._images.shape),
        dtype=str(g._images.dtype),
        loading_seconds=loading_seconds,
        loading_peak_rss_bytes=peak_rss,
        images_sha256=digest(g._images),
        rotations_sha256=digest(g._rotations),
        translations_sha256=digest(g._translations),
        scale_sha256=digest(g._scale),
        ctf_sha256={k: digest(v) for k, v in g._ctf_params.items()},
        anisomag_sha256=None if g._anisomag is None else digest(g._anisomag),
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
