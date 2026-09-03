# Match an experimental dataset

`specter match particles` derives a simulation config that matches a real
particle set, and reports how close the match is. It takes the refined
particle set, its images and the atomic model, and writes a `matched.toml`
that `specter simulate particles` runs as is. The only settings it asks for
are the four an acquisition record carries and no particle file does.

## What you provide

| input | flag | where it comes from |
|---|---|---|
| refined particle set | `--metadata_path` | a CryoSPARC passthrough `.cs` or a RELION `.star` |
| particle images | `--images_path` (optional) | the stack the metadata refers to, in its order; unset reads the paths the file itself points at |
| atomic model | `--pdb_source` | PDB accession or local file, with the assembly that matches the particle |
| detector | `--detector_model` | methods section, EMDB record |
| total dose per movie, e⁻/Å² | `--dose` | methods section, EMDB record, import settings |
| dose rate, e⁻/physical px/s | `--dose_rate` | methods section; unset falls back to the detector's typical rate and the report says so |
| energy filter | `--energy_filter` | methods section; recorded in the report |

The particle set must be aligned to the atomic model: its poses have to
reproduce the experimental views when the model is rendered at them. A
refinement that started from an ab initio volume is in that volume's own
orientation, not the model's. Run an Align 3D of the refined map against a
map rendered from the model, re-extract the particles from the aligned job,
and pass that particle set. The command checks this first and stops if it
fails, because nothing downstream is meaningful otherwise.

```bash
specter match particles \
    --metadata_path J205_passthrough_particles.cs \
    --images_path exp_2000.mrcs \
    --pdb_source 8b0x \
    --detector_model falcon4i_300kv \
    --dose 40 --dose_rate 4.1 --energy_filter true \
    --device cuda:0
```

## What it does

1. **Pose alignment.** A noiseless, ice-free render at the refinement's
   poses is correlated with the experimental images pair by pair and
   compared with a random pairing. Pass or fail, with the fix named.
2. **Acquisition card to physics.** Detector MTF and DQE(0) by name; the
   coincidence radius from the detector's exclusion radius and hardware
   frame rate together with the dose rate, converted to the simulation's
   pixel size and frame count at constant occupancy; the radiation-damage
   envelope from the dose and voltage. Nothing in this step is fitted.
3. **Probes.** Ice thickness and neighbour spacing are the two quantities
   the images have to supply. Each candidate is rendered at 64 particles
   and scored on the background variance outside the particle against the
   experiment. The probes and the pose check run at a box Fourier-cropped
   by two, against the images cropped the same way, since what they measure
   lies below 10 Å; the candidates render concurrently in worker
   processes.
4. **Comparison at matched poses.** Two seeds at the chosen settings, at
   the native box, against the experiment: the signal-to-noise ratio of each stack per frequency
   band, the twin test, and the screens for fixed patterns and background
   mismatch. A clearly positive residual envelope is applied as a B-factor
   and the comparison rerun once.
5. **Output.** `matched.toml`, `match_report.md`, `match_report.png`, and
   with `--write_stack N` a stack of `N` particles simulated from the
   matched config.

Under three minutes on one GPU at a 256 px box, and under two with four
GPUs named in `--device`; the two-seed comparison at the native box is most
of it. The full stack is extra.

## Reading the report

The figure shows five experimental particles above their simulated twins
and the mean image of each stack, the overlaid radial power spectra with
their ratio, the per-band SNR ratio, and the twin-test histograms. The
table lists every derived value with its provenance: metadata, detector
table, probe, measured, or fallback.

The number to read first is the **matched-pose SNR ratio**, simulated over
experimental, per band. Near 1 in every band means the simulation carries
the experiment's signal and noise at every scale. A ratio that is flat and
modest is a contrast or dose question; a ratio that grows with frequency is
an envelope. If a broad excess remains after every derivable parameter is
set, the report says so: that residual is not a parameter of the forward
model, and no knob should be turned to hide it. On the datasets tried so
far it tracked the absence of an energy filter.

The overlaid **power spectra** are there because they are the view most
people know, not because they decide anything. A stack can have half the
experiment's low-frequency power and still be indistinguishable in a mixed
2D classification, or twice it and separate cleanly.

## What it does not do

It does not run a classification. The report is designed to replace the
loop of simulating thousands of particles and classifying them in
CryoSPARC while parameters are being chosen; a mixed 2D classification
remains the right final validation, run once, with two seeds, reading the
count of classes near the input ratio alongside chi-squared.

It does not fit a B-factor, a potential scale or a coincidence radius to
the classification. Each of those was, in earlier hand tuning, standing in
for something the model lacked, and the report names the lack instead.

It cannot reproduce specimen packing that is not random. A close-packed
monolayer of virus particles has neighbours at fixed angles, and the
simulator places neighbours at random; the neighbour-spacing probe will
report the mismatch it cannot close.

## Limitations

- The refinement file supplies the box and pixel size. The probes run at a
  box Fourier-cropped by `probe_bin` (default 2, capped so the probe pixel
  stays at or below 5 Å), but the final two-seed comparison simulates at
  the native box. Above about 512 px, Fourier-crop the stack first and pass
  it as `--images_path`.
- Probe simulations are dealt round-robin over the devices named by
  `device`, one worker process per device by default (`probe_workers`).
  Naming several devices (`--device 0,1,2,3`) is what shortens a run;
  extra processes on one GPU are time-sliced and gain nothing.
- Only detectors with a calibrated exclusion radius get a coincidence
  term; the others get zero and a warning. Only detectors with a bundled
  MTF get one.
- Ice thickness and neighbour spacing are chosen from the candidate lists
  in the config (`ice_candidates`, `crowd_candidates`), not fitted
  continuously.
- One spacing and one thickness serve the whole stack; the simulator has no
  per-micrograph variation.
