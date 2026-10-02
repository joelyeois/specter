# Final Year Project Utilities for cryo-EM Image Generation — replicating the sample complexity of multi-reference alignment

Self-contained workspace (DSA4288 FYP) for replicating and then stress-testing
**Perry, Weed, Bandeira, Rigollet & Singer (2019), "The sample complexity of
multi-reference alignment"** with the SPECTER simulator. Nothing outside this
folder is modified; SPECTER is only *imported*.

The paper's claim: for the model `y_i = R_{l_i} θ + σ ξ_i` (cyclic shift plus
white Gaussian noise) the number of samples needed to estimate a generic `θ`
scales as `1/SNR` at high SNR but as **`1/SNR^3`** at low SNR, for *any*
estimator. The bispectrum (third moment) is the reason. Sigworth (1998) saw the
same crossover empirically for 2-D cryo-EM alignment.

## Layout

Only the image-generation side is kept here; the downstream estimators
(invariants / bispectrum, Jennrich, EM), metrics, sweep scripts and their
configs have been removed.

```
mra/                     library (torch only, except template.py / phase_b.py)
  model.py               forward model: cyclic shifts, optional 90° rotations, Gaussian noise, SNR conventions
  template.py            clean θ: synthetic generic signal, or SPECTER single-view projection of a PDB
  phase_b.py             SPECTER end-to-end stacks with physics switched on one term at a time
tests/                   pytest smoke + correctness tests
data/                    gitignored (PDB cache)
```

## Running

From the repo root with `.venv` active:

```bash
python -m pytest mj_dsa4288/tests -v
```

## Generating data

### A. Paper model with a realistic template
SPECTER contributes only `θ`: one fixed view of a PDB structure, projected with
`scattering_model="projection"`, defocus = Cs = α = 0, no envelopes, no `klim`,
no ice, no Poisson noise, no detector, no crowding
(`mra.template.template_from_pdb`). Shifts and Gaussian noise are added by
`mra.model.sample_mra`, so the data obey the paper's assumptions exactly while
`θ` is a real protein projection.

* **SNR convention.** `"paper"` is `‖θ‖²/σ²` with `‖θ‖ = 1`; `"pixel"` is
  Sigworth's per-pixel variance ratio. They differ by ≈ `d`.

### B. Adding cryo-EM physics back
`mra.phase_b.build_generator` + `generate_stack` run SPECTER end-to-end on a
fixed view with real-space in-plane shifts; noise is set by **dose** instead of
`σ`. `PhysicsConfig` switches terms on one at a time: fixed CTF + Poisson,
per-particle defocus jitter, multislice, K3 detector MTF + coincidence loss,
amorphous ice (`IceBank`). Real-space shifts are not cyclic, so keep
`max_shift_angstrom` small enough that the particle stays in the box.
