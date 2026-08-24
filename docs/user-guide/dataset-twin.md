# Generate a CryoSPARC dataset twin

`specter simulate particles --cs_path <file>.cs` drives generation from a
real dataset instead of randomly sampling poses and CTF parameters. Pixel
size, voltage, amplitude contrast, per-particle pose, and per-particle CTF
are all read straight from the `.cs` file, so the simulated stack shares
every imaging parameter with the real one it was built from, particle for
particle. `--star_path <file>.star` does the same from a RELION `.star`
file (both the single-block layout `specter` itself writes and the RELION
3.1+ two-block `optics`/`particles` layout). The two flags are mutually
exclusive.

This is the mechanism behind [Generate a particle
stack](particle-stack.md#example-matching-empiar-11377)'s small
five-particle example. This page covers the same workflow at full dataset
scale, and how to check the result against the real data it twins.

## What a `.cs`/`.star` file supplies, and what it doesn't

A `.cs`/`.star` file carries imaging parameters, not structural identity:
pose, CTF, pixel size, voltage, and amplitude contrast come from the file,
but which structure was imaged is not recorded in it. `pdb_source` still
has to be set explicitly, to whatever PDB/mmCIF accession or local file
matches the dataset. Get this wrong and every other parameter still loads
correctly. Only the rendered particle looks wrong.

`--n_particles`, combined with either path flag, takes the *first*
`n_particles` rows of the file rather than a random subset, which is what
makes a small worked example (five rows, as in the particle-stack guide)
and a full-dataset twin (every row) the same flag with a different value.
Omit `--n_particles` entirely to use every particle the file contains.

## Running the full workflow

```bash
specter simulate particles \
    --pdb_source 8b0x \
    --n_pixels 512 \
    --cs_path /path/to/your/dataset_passthrough.cs \
    --dose 40 \
    --ice_model gd \
    --coincidence_radius 0.8 \
    --potential_scale 0.5 \
    --detector_model none \
    --device cuda:0
```

This is the same command as the five-particle example in [Generate a
particle stack](particle-stack.md#example-matching-empiar-11377), with
`--n_particles 5` removed so every row in the file is rendered. A dataset
of any real size benefits from `--device` accepting a comma-separated GPU
list (see [Multi-GPU](particle-stack.md#multi-gpu)) and from job tracking
(`--project`, see [Manage jobs](jobs.md)) to keep a multi-thousand-particle
run's parameters and provenance recorded alongside its output.

`coincidence_radius` and `potential_scale` are not read from the `.cs`/
`.star` file. Set them to whatever matches the detector and specimen
thickness of the dataset being twinned; both default to values with no
effect (`0.0` and `1.0`) if left unset.

## Validating the twin

Because every imaging parameter matches, a twin dataset should look like
the real one, not just plausible in isolation. Two checks, in increasing
order of rigor:

**Visual comparison.** Render the same particles the real stack contains
and compare side by side. [Generate a particle
stack](particle-stack.md#example-matching-empiar-11377)'s five-particle
figure is this check at the smallest useful scale.

**Pooled 2D classification.** Mix a twin dataset with the real particles it
matches into a single stack, and run one CryoSPARC 2D Classification job
over the pool. If the physics is right, CryoSPARC has no basis to separate
simulated particles from real ones. It sorts both into the same classes, in
roughly the same proportion per class, and the per-class real/simulated
split reads as noise around 50/50 rather than as two populations. This is
a stronger check than visual comparison because it doesn't rely on a human
judging similarity: a systematic physics error (a wrong envelope, a
miscalibrated noise model) tends to show up as classes that are
disproportionately one source or the other, even when individual particles
still look convincing.

[Generate a particle stack](particle-stack.md#example-matching-empiar-11377)
shows this check run at 2,000 real + 2,000 simulated EMPIAR-11377 particles,
with the pooled class-count figure and the underlying per-particle
class-assignment CSV both committed to the repo. Reproduce or adapt it with
[`docs-figures/particle_stack_empiar_11377.py`](https://github.com/joelyeois/specter/blob/main/docs-figures/particle_stack_empiar_11377.py),
which builds the twin stack, and requires access to a real CryoSPARC
instance for the classification job itself.

## See also

- [Generate a particle stack](particle-stack.md): the general
  `specter simulate particles` guide, including the small worked example
  this page's workflow extends to full scale.
- [Manage jobs](jobs.md): tracking a full-dataset run's parameters and
  provenance.
- `specter.io.extract_parameters_from_csfile` /
  `extract_parameters_from_starfile` (Python API): the functions behind
  `--cs_path`/`--star_path`, including a `halfset` argument for extracting
  only a CryoSPARC/RELION gold-standard half-set. Not currently exposed as
  a CLI flag.
