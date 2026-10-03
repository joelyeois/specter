"""Freeze a 256-pixel control from a native 512-pixel matching dataset.

Run in a SPECTER checkout. Paths are explicit because experimental data are
not distributed with the repository. This does not alter the native input.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mrcfile
import torch
from specter.arrays import fourier_crop


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(8 * 2**20):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ["metadata", "images", "pdb", "weights", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    with mrcfile.mmap(args.images, permissive=True) as source:
        native = source.data[:200].copy()
    assert native.shape == (200, 512, 512)
    cropped, dx = fourier_crop(torch.from_numpy(native), 0.731, 1.462)
    control = args.output / "experimental-256.mrcs"
    with mrcfile.new(control, overwrite=False) as target:
        target.set_data(cropped.numpy())
        target.voxel_size = dx
    inputs = (
        "[inputs]\n"
        + "\n".join(
            f"{key} = {json.dumps(str(path.resolve()))}"
            for key, path in [
                ("metadata_path", args.metadata),
                ("images_path", args.images),
                ("pdb_source", args.pdb),
            ]
        )
        + "\nassembly = false\n"
    )
    acquisition = (
        '\n[acquisition]\ndetector_model = "falcon4i_300kv"\n'
        'dose = 40.0\nn_frames = 40\nabsorption_model = "inelastic_mfp"\n'
        "inelastic_mfp_solvent = 3950.0\n"
        f"dose_weights_path = {json.dumps(str(args.weights.resolve()))}\n"
    )
    compute = "\n[compute]\nprobe_workers = 0\n"
    (args.output / "material.toml").write_text(
        inputs + acquisition + "inelastic_mfp_specimen = 2460.0\n" + compute
    )
    (args.output / "uniform-256.toml").write_text(
        inputs.replace(
            json.dumps(str(args.images.resolve())), json.dumps(str(control.resolve()))
        )
        + acquisition
        + compute
    )
    manifest = {
        "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
        "pdb": {"path": str(args.pdb), "sha256": sha256(args.pdb)},
        "weights": {"path": str(args.weights), "sha256": sha256(args.weights)},
        "images": {
            "path": str(args.images),
            "shape_used": list(native.shape),
            "first_200_data_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
        },
        "uniform_control": {
            "shape": list(cropped.shape),
            "pixel_size_A": dx,
            "data_sha256": hashlib.sha256(cropped.numpy().tobytes()).hexdigest(),
        },
    }
    (args.output / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
