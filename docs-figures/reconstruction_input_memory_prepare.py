"""Recreate the frozen real-data pairs used by the reconstruction comparison.

Paths below identify existing local CryoSPARC data, not bundled test fixtures.
The 256-pixel set uses 10,000 distinct particles in CS row order. Its upstream
ab-initio split is all-zero, so a fixed balanced partition is added solely for
this implementation comparison; its FSC does not certify the upstream poses.
The existing 512-pixel ribosome export retains its original gold splits.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import mrcfile
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument(
        "--project-256", default="/scratch/loh/joel/empiar-10254/CS-empiar-10254"
    )
    p.add_argument(
        "--cs-512",
        default="/scratch/loh/joel/empiar-11377/claude_match/j446_2000_passthrough_512.cs",
    )
    p.add_argument(
        "--stack-512",
        default="/scratch/loh/joel/empiar-11377/claude_match/exp_2000_j446_512.mrcs",
    )
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=False)
    project = Path(a.project_256)
    src = project / "J9/J9_particles.cs"
    d = np.load(src, mmap_mode="r")[:10000].copy()
    assert len(d) == 10000 and len(np.unique(d["uid"])) == 10000
    assert np.all(d["blob/shape"] == [256, 256])
    refs = [
        (project / bytes(path).decode(), int(idx))
        for path, idx in zip(d["blob/path"], d["blob/idx"])
    ]
    by_file = defaultdict(list)
    for row, (path, idx) in enumerate(refs):
        by_file[path].append((row, idx))
    out = root / "empiar10254-10000.mrcs"
    with mrcfile.new_mmap(out, shape=(len(d), 256, 256), mrc_mode=2) as m:
        m.voxel_size = float(d["blob/psize_A"][0])
        for path, entries in by_file.items():
            with mrcfile.mmap(path, permissive=True) as source:
                for row, idx in entries:
                    image = np.asarray(source.data[idx], dtype=np.float64)
                    m.data[row] = (
                        (image - image.mean()) / (image.std() + 1e-12)
                    ).astype(np.float32)
        m.flush()
    assert np.all(d["alignments3D/split"] == 0)
    split = np.zeros(len(d), dtype=d["alignments3D/split"].dtype)
    split[np.random.default_rng(10254).permutation(len(d))[:5000]] = 1
    d["alignments3D/split"] = split
    d["blob/path"] = out.name.encode()
    d["blob/idx"] = np.arange(len(d), dtype=np.uint32)
    with (root / "empiar10254-10000.cs").open("wb") as f:
        np.save(f, d)
    metadata = dict(
        source=str(src),
        count=len(d),
        box=256,
        pixel_size_A=float(d["blob/psize_A"][0]),
        dose_per_angstrom=64.0,
        halfsets={"0": 5000, "1": 5000},
        unique_uids=10000,
        uids_sha256=hashlib.sha256(d["uid"].tobytes()).hexdigest(),
        source_files=len(by_file),
        normalization="Full-image mean/std, float64 arithmetic; float32 output. Same frozen prepared data for both variants.",
        split="Frozen balanced random partition (seed 10254); J9 ab-initio metadata originally assigns every row to split 0. Used to compare implementations; FSC is not an independent certification of the upstream poses.",
    )
    (root / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    ribosome = np.load(a.cs_512).copy()
    with mrcfile.open(a.stack_512, header_only=True) as m:
        assert len(ribosome) == int(m.header.nz) == 2000
        assert int(m.header.nx) == int(m.header.ny) == 512
    # The existing export already pairs images with metadata by row. Only
    # addresses change; poses, CTF, pixel size and gold splits stay untouched.
    ribosome["blob/path"] = b"empiar11377-2000.mrcs"
    ribosome["blob/idx"] = np.arange(len(ribosome), dtype=np.uint32)
    with (root / "empiar11377-2000.cs").open("wb") as f:
        np.save(f, ribosome)
    (root / "empiar11377-2000.mrcs").symlink_to(Path(a.stack_512).resolve())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
