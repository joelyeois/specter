# Generate a CryoSPARC dataset twin

!!! info "Work in progress"
    Will cover the full dataset-driven workflow: `specter simulate particles
    --cs_path ...` (or `ParticleStackConfig.cs_path` from Python), which
    reads pixel size, voltage, alpha, poses, and CTF straight from a real
    [CryoSPARC](https://cryosparc.com/) passthrough `.cs` file instead of
    sampling them synthetically. `--star_path` does the same from a
    [RELION](https://relion.readthedocs.io/) `.star` file (both the
    single-block and the RELION 3.1+ optics/particles two-block layout).
    See [Generate a particle stack](particle-stack.md#example-matching-empiar-11377)
    for a small worked example (EMPIAR-11377) in the meantime.
