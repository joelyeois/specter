"""Compare full tomogram voxels, labels and all pick files, in bounded slabs."""

import argparse
import json
from pathlib import Path

import mrcfile
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--before", required=True)
p.add_argument("--after", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
before, after = Path(a.before), Path(a.after)
results = {}
for path in sorted(before.glob("*.mrc")):
    with mrcfile.mmap(path) as b, mrcfile.mmap(after / path.name) as c:
        same = True
        max_error = 0.0
        n_different = 0
        sq_error, sq_reference = 0.0, 0.0
        header_equal = all(
            np.array_equal(b.header[f], c.header[f])
            for f in ["dmin", "dmax", "dmean", "rms", "cella"]
        )
        for z in range(0, b.data.shape[0], 8):
            x, y = b.data[z : z + 8], c.data[z : z + 8]
            difference = y.astype("float64") - x
            same &= np.array_equal(x, y)
            max_error = max(max_error, float(np.abs(difference).max()))
            n_different += int(np.count_nonzero(difference))
            sq_error += float(np.square(difference).sum())
            sq_reference += float(np.square(x.astype("float64")).sum())
        results[path.name] = {
            "shape": list(b.data.shape),
            "bitwise_equal": same,
            "max_abs_error": max_error,
            "different_voxels": n_different,
            "relative_l2": float(np.sqrt(sq_error / sq_reference))
            if sq_reference
            else 0,
            "header_stats_equal": header_equal,
        }
picks = sorted(before.glob("*.ndjson"))
results["picks"] = {
    "files": len(picks),
    "same_names": {p.name for p in picks} == {p.name for p in after.glob("*.ndjson")},
    "bitwise_equal": all(
        p.read_bytes() == (after / p.name).read_bytes() for p in picks
    ),
}
Path(a.output).write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2), flush=True)
