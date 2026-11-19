# Render a potential map

`specter build potential` renders the electrostatic potential of one atomic
model and writes it as an `.mrc` volume. The map is the potential that
`specter simulate particles` builds internally before imaging, saved for use
outside a simulation: as a reference map for a Fourier shell correlation, as a
template, or as an initial model. For the method, see
[Atomic potentials](../concepts/atomic-potentials.md); for the same render
from Python, see [A potential from a PDB entry](../tutorials/pdb-potential.md).

## Basic usage

The structure is the only required setting. It is a four-character PDB
accession code, which is fetched and cached, or a path to a local `.cif` or
`.pdb` file:

```bash
specter build potential --pdb_source 6bdf
```

```text
  92176 atoms, 661 kDa, largest extent 192 A
  Box: 200 voxels at 1 A (200 A across), shtyrov scattering factors, on cuda
  Peak 136 V
  Saved potentials/6bdf.mrc
```

A TOML config is loaded first when one is given, and each flag overrides one
of its fields:

```bash
specter build potential \
    --config configs/potential.toml \
    --pdb_source my_model.cif \
    --voxel_size 0.8 \
    --output_dir maps
```

`configs/potential.toml` is the canonical starting point, and the
[command reference](../api/cli/build.md#specter-build-potential) documents
every field.

## The output

The command writes one file, `<output_dir>/<filename>.mrc`. The volume is in
volts, indexed `(Z, Y, X)`, and its header carries the voxel size. `filename`
defaults to the name of the structure, so the accession code `6bdf` and a local
`6bdf.cif` both give `6bdf.mrc`. Setting `--project` or `--job_id` tracks the
run as a numbered job under `<output_dir>/[<project>/]potentials/`; see
[Manage jobs](jobs.md).

## Box size

With `box` unset, the box is the smallest even number of voxels that holds
the model in every orientation about its centre. Its side is twice the
distance of the farthest atom from the centre, plus the radius out to which
each atom is rendered on either side. A map sized this way can be rotated
without any density leaving the box.

An explicit `--box` fixes the size instead, for example to match the box of an
experimental map. The renderer leaves out atoms that fall outside the box, so
a box smaller than the automatic one produces a clipped map; the command warns
when this happens and reports the size the structure needs.

## Scattering factors

`scattering_factors` selects the parameterization of the atomic potentials.
The default, `shtyrov`, is fitted per bonded species (Shtyrov et al. 2026)
and types every atom from the bond topology of the model, which requires a
Monomer Library, given by `monomer_library_path` or `$CLIBD_MON`. Atoms that
cannot be typed fall back to per-element factors, and the command warns when
any do. `kirkland` and `lobato` are per element and need no library.
[Scattering factors](../walkthrough/potentials.md#scattering-factors) compares
the three.

`use_deposited_bfactors` damps each atom by the B-factor deposited with the
model. It is off by default, so that the map is the static model, and it
requires `scattering_factors = "shtyrov"`.

## Limitations

- The map is centred on the model, not placed in the coordinate frame of the
  deposited file, and the header origin is zero. It does not overlay the
  deposited coordinates in a viewer without a translation.
- The map is the model in vacuum. It contains no solvent, no envelope and no
  imaging effects; [Generate a particle stack](particle-stack.md) covers those.
- Of a file that holds several models, such as an NMR ensemble, only the
  first is rendered.

## References

- Shtyrov, A., Wilson, H., Slowik, D., Yamashita, K., Li, J., Wojdyr, M.,
  et al. (2026). Measurement of atomic scattering factors by cryoelectron
  microscopy. *Proceedings of the National Academy of Sciences*, 123(19),
  e2528758123.
  [doi:10.1073/pnas.2528758123](https://doi.org/10.1073/pnas.2528758123)
