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

```
mra/                     library (torch only, except template.py / phase_b.py)
  model.py               forward model: cyclic shifts, optional 90° rotations, Gaussian noise, SNR conventions
  invariants.py          unbiased mean / power-spectrum / bispectrum / T3 estimators  (paper eq. 2–3, Lemma 1)
  jennrich.py            Jennrich's tensor decomposition + homoJen estimator          (paper Section 4, Thm 3)
  em.py                  EM / maximum-likelihood over shifts with known σ             (Sigworth 1998, paper Fig. 2)
  metrics.py             shift-invariant distance ρ, reconstruction SNR, log-log slope fits
  experiment.py          SNR sweeps, sample-complexity (doubling) search, JSON I/O, plots
  template.py            clean θ: synthetic generic signal, or SPECTER single-view projection of a PDB
  phase_b.py             SPECTER end-to-end stacks with physics switched on one term at a time
configs/                 reference_1d.toml, phase_a.toml, phase_b.toml
scripts/                 run_reference_1d.py, run_phase_a.py, run_phase_b.py, plot_results.py
tests/                   pytest smoke + correctness tests
data/, results/          gitignored (PDB cache, JSON rows, PNG plots)
```

## Running

From the repo root with `.venv` active (`uv sync` first; see the note below):

```bash
python -m pytest mj_dsa4288/tests -v                       # ~10 s

python mj_dsa4288/scripts/run_reference_1d.py --quick      # smoke; drop --quick for the real sweep
python mj_dsa4288/scripts/run_phase_a.py --quick
python mj_dsa4288/scripts/run_phase_b.py --quick --only b1_ctf_fixed
python mj_dsa4288/scripts/plot_results.py mj_dsa4288/results/phase_a/sweep.json
```

`--quick` shrinks `n`, box size and iteration counts so each script finishes in
well under a minute; the slopes it prints are **not** meaningful. Real sweeps
need `n` up to 10^5–10^6 at the lowest SNRs (the whole point of `1/SNR^3`), so
set `device = "cuda"` in the TOML for Phase A/B.

## The three stages

### 0. 1-D reference (`reference_1d.toml`)
The paper's exact model, `d = 41`, both estimators. Expected: reconstruction
SNR vs data SNR has slope ≈ 1 above SNR ≈ 1 and slope ≈ 3 below; the
doubling-search `n_required(SNR)` shows the mirror image. Jennrich is the
provably optimal estimator but is numerically fragile for `d ≳ 50`; EM is what
practitioners use and what Sigworth plotted.

### A. Faithful replication with a realistic template (`phase_a.toml`)
SPECTER contributes only `θ`: one fixed view of a PDB structure, projected with
`scattering_model="projection"`, defocus = Cs = α = 0, no envelopes, no `klim`,
no ice, no Poisson noise, no detector, no crowding
(`mra.template.template_from_pdb`). Shifts and Gaussian noise are added by
`mra.model.sample_mra`, so the data obey the paper's assumptions exactly while
`θ` is a real protein projection. Two deliberate choices:

* **SNR convention.** `"paper"` is `‖θ‖²/σ²` with `‖θ‖ = 1`; `"pixel"` is
  Sigworth's per-pixel variance ratio. They differ by ≈ `d`, which moves the
  crossover on the x-axis. State which one you plot.
* **No per-image normalisation.** The template is scaled once to `‖θ‖ = 1`;
  noisy images are never z-scored individually (that would bias the moments).

### B. Adding cryo-EM physics back (`phase_b.toml`)
SPECTER runs end-to-end on a fixed view with real-space in-plane shifts, and
noise is set by **dose** instead of `σ`. Each `[[experiment]]` adds one term:

| step | adds | paper assumption broken |
|---|---|---|
| b1 | fixed CTF (300 kV, Cs 2.7 mm, α 0.1, 1 µm) + Poisson | none (known linear operator); Poisson ≈ Gaussian at low dose |
| b2 | per-particle random defocus | invariants now vary per image |
| b3 | multislice instead of projection | non-linear forward model |
| b4 | K3 detector MTF + coincidence loss | noise no longer white |
| b5 | amorphous ice (`IceBank`) | structured, non-Gaussian "noise" |

The effective SNR is measured from clean-vs-noisy stacks with identical shifts
(`mra.phase_b.effective_snr`), and the reference for `ρ` is the CTF-filtered
projection, since that is the estimand once a CTF is on. `run_phase_b.py` is a
sketch: it wires the pieces together and runs, but the reference-alignment step
and the σ estimate are crude and should be refined before drawing conclusions.

## Caveats and TODOs

* Eq. (3) of the paper omits `σ²` in the `3 sym(y ⊗ I)` correction; the proof
  of Lemma 1 shows it must be there. `invariants.py` includes it and is tested
  against a literal brute-force implementation.
* Real-space shifts in Phase B are not cyclic. Keep `max_shift_angstrom` small
  enough that the particle stays in the box; EM's cyclic search is then a
  superset and harmless.
* Section 3.2 (low-passed signals are *harder*) is easy to test with the
  synthetic template by zeroing high frequencies, but no script does it yet.
* Heterogeneous MRA (Section 5, `heteroJen`) is not implemented.
* Ghostbuster's translation refinement is not used on purpose: it is flagged as
  unverified in the repo's CLAUDE.md.

### Environment note
At the time of writing the checked-in `.venv` was missing several declared
dependencies (`torch-ctf`, `vesin-torch`, `roma`, `click`, `rich-click`), and
`uv.lock` is stale relative to `pyproject.toml`. They were installed into the
venv with `uv pip install` to avoid touching `uv.lock`; a proper `uv sync`
(which rewrites the lock file) should be done on the main branch.
