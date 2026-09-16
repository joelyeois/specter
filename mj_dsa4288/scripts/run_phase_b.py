"""
Phase B: SPECTER end-to-end physics on a fixed view, sweeping dose instead of sigma.

For every experiment in the config a clean stack and a noisy stack are
generated with identical shifts; the clean stack gives the effective SNR and
the CTF-on reference, EM is run on the noisy stack, and the reconstruction SNR
is recorded. Slopes are then compared with the Phase A baseline.

Usage::

    python mj_dsa4288/scripts/run_phase_b.py [--config ...] [--only b1_ctf_fixed] [--quick]
"""

from __future__ import annotations

import argparse
import dataclasses

import torch
from _common import RESULTS, default_config, load_toml

from mj_dsa4288.mra.em import em_mra
from mj_dsa4288.mra.experiment import plot_sigworth, save_rows, summarise_slopes
from mj_dsa4288.mra.metrics import reconstruction_snr, rho
from mj_dsa4288.mra.phase_b import (
    PhysicsConfig,
    build_generator,
    effective_snr,
    generate_stack,
    standardise_stack,
)
from mj_dsa4288.mra.template import PDB_CACHE


def build_volume(t: dict, quick: bool) -> tuple[torch.Tensor, float]:
    from specter.pdb import PDB
    from specter.potential import PotentialBuilder

    PDB_CACHE.mkdir(parents=True, exist_ok=True)
    pdb = PDB(t["pdb_id"], assembly=True, savefolder=str(PDB_CACHE), verbose=False)
    n = int(t["num_pixels"]) if not quick else 16
    builder = PotentialBuilder(
        n,
        float(t["pixel_size"]),
        pdb.atomic_numbers,
        parameterization=t["parameterization"],
    )
    with torch.no_grad():
        return builder(pdb.coordinates, method="analytic").clone(), float(
            t["pixel_size"]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config("phase_b")))
    parser.add_argument("--only", default=None, help="run a single named experiment")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = load_toml(args.config)
    common = cfg["common"]
    volume, pixel_size = build_volume(cfg["template"], args.quick)
    quat = torch.tensor(cfg["template"]["quaternion"], dtype=torch.float32)
    n = int(common["n_particles"]) if not args.quick else 64
    doses = [float(d) for d in common["doses"]] if not args.quick else [0.1, 10.0]
    known = {f.name for f in dataclasses.fields(PhysicsConfig)}

    for exp in cfg["experiment"]:
        if args.only and exp["name"] != args.only:
            continue
        phys_kwargs = {k: v for k, v in exp.items() if k in known}
        phys = PhysicsConfig(
            max_shift_angstrom=float(common["max_shift_angstrom"]),
            device=common["device"],
            **phys_kwargs,
        )
        clean_cfg = dataclasses.replace(
            phys, noise_model=None, detector_model=None, ice_model=None
        )
        rows = []
        for dose in doses:
            seed = int(common["seed"])
            clean = generate_stack(
                build_generator(volume, pixel_size, n, dose, clean_cfg, quat, seed), n
            )
            noisy = generate_stack(
                build_generator(volume, pixel_size, n, dose, phys, quat, seed), n
            )
            snr_paper, snr_pixel = effective_snr(clean, noisy)
            # reference = clean image of the *unshifted* particle (index 0 shifted back is
            # awkward; instead use the clean stack mean after EM alignment as reference)
            ref_stack, scale = standardise_stack(clean, clean[0])
            y, _ = standardise_stack(noisy, clean[0])
            sigma = float((y - ref_stack).std())
            est = em_mra(
                y, sigma, n_iter=int(common["em_iters"]) if not args.quick else 10
            ).theta
            # Align the clean stack the same way to obtain the physics-consistent reference.
            ref = em_mra(ref_stack, max(sigma, 1e-3), n_iter=20, init=est).theta
            rows.append(
                {
                    "experiment": exp["name"],
                    "dose": dose,
                    "snr": snr_pixel,
                    "snr_paper": snr_paper,
                    "sigma": sigma,
                    "n": n,
                    "rho": rho(est, ref),
                    "rec_snr": reconstruction_snr(est, ref),
                }
            )
            print(
                exp["name"],
                f"dose={dose:g}",
                f"pixel SNR={snr_pixel:.3g}",
                f"rec SNR={rows[-1]['rec_snr']:.3g}",
            )
        out = RESULTS / "phase_b"
        save_rows(rows, out / f"{exp['name']}.json")
        plot_sigworth(rows, out / f"{exp['name']}.png", 1.0, exp["name"])
        print(exp["name"], "slopes:", summarise_slopes(rows, 1.0))


if __name__ == "__main__":
    main()
