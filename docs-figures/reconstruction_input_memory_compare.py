"""Compare every reconstructed voxel in bounded float64 slabs."""

import argparse
import json
from pathlib import Path
import mrcfile
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = []
    for half in ["A", "B"]:
        old = (
            next((Path(a.before) / "reconstructions").glob("J*")) / f"volume_{half}.mrc"
        )
        new = (
            next((Path(a.after) / "reconstructions").glob("J*")) / f"volume_{half}.mrc"
        )
        with mrcfile.mmap(old) as m, mrcfile.mmap(new) as n:
            assert m.data.shape == n.data.shape
            assert m.voxel_size == n.voxel_size
            d2 = v2 = 0.0
            maximum = 0.0
            equal = True
            for z in range(0, m.data.shape[0], 8):
                x = m.data[z : z + 8].astype(np.float64)
                y = n.data[z : z + 8].astype(np.float64)
                delta = y - x
                d2 += np.sum(delta * delta)
                v2 += np.sum(x * x)
                maximum = max(maximum, float(abs(delta).max()))
                equal = equal and np.array_equal(x, y)
            rows.append(
                dict(
                    halfset=half,
                    shape=list(m.data.shape),
                    bitwise_equal=equal,
                    relative_l2=float(np.sqrt(d2 / v2)),
                    max_abs_V=maximum,
                )
            )
    Path(a.output).write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
