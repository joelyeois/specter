# Quickstart

## Simulate a particle stack

The `generate_particle_stack.py` demo script is the quickest way to produce
a simulated cryo-EM particle stack from a PDB code:

```bash
python demo-scripts/generate_particle_stack.py \
    --pdb_code 6bdf \
    --n_particles 20 \
    --num_pixels 256 \
    --pixel_size 1.0 \
    --energy 300 \
    --scattering_model multislice \
    --output_dir ./output
```

This downloads the structure, builds the scattering potential, applies CTF and
detector effects, and writes a `.mrcs` / `.star` file pair. For the other
three workflows (a CryoSPARC dataset twin, full micrographs, cryo-ET tilt
series) and the complete CLI reference, see [Data generation](generation.md).

## Job management

Generation runs can be recorded under a project name in a local job
database. Inspect past runs with the `specter-jobs` CLI (installed
automatically with the package) — see [Job management](jobs.md):

```bash
specter-jobs list --project my-project
specter-jobs show <job_id>
specter-jobs diff <job_id_1> <job_id_2>
```

See `demo-notebooks/` for interactive worked examples, including
micrograph and tilt-series generation (`create_micrograph/`,
`tilt-series-generator.ipynb`). For an example of composing the forward
model's individual modules by hand (e.g. to swap in a custom aberration
model) instead of going through `ImageGenerator`, see
`modular_pipeline/`.
