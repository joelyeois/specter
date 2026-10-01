# Configure a run

Two layers drive every `specter` simulate/build/reconstruct command,
stacked in a fixed order: a TOML config file, loaded first, and any CLI
flags you pass on the command line, which override individual fields of
it. Nothing else participates: no environment-variable layer, no implicit
merging of multiple config files.

`--config` is optional. Omit it and every field takes the default declared
on its config dataclass, which is where the defaults live; a TOML file only
ever overrides them. The exception is a field with no default, which names
an input the run is about rather than a setting it has an opinion on. Those
must be passed as flags, and the error names them when they are missing.

The example configs are **not installed with the package.** They live in the
repository at
[`configs/`](https://github.com/joelyeois/specter/tree/main/configs), one per
subcommand, and every `configs/...` path below refers to them. Clone the
repository to get them all, or download a single file:

```bash
curl -O https://raw.githubusercontent.com/joelyeois/specter/main/configs/particle.toml
```

Keep a downloaded config and the installed version in step. Field names change
between releases, and a config naming a field that no longer exists is rejected
by name rather than ignored.

## TOML config files

A config file is a flat set of fields, grouped into tables for
readability only. `specter simulate particles`, for instance, reads a
`ParticleStackConfig`, whose fields sit in `configs/particle.toml` under
`[data]`, `[specimen]`, `[microscope]`, `[absorption]`, `[sampling]`,
`[models]`, `[postprocessing]`, `[compute]`, `[output]`, `[job]`, and
`[advanced]`. The table names carry no meaning for the loader: it
flattens every table into one namespace before validating the fields, so `[specimen]`'s `pdb_source`
and `[microscope]`'s `voltage` end up as siblings on the same dataclass.
Writing a field under the wrong table heading, or with no table at all,
works the same as writing it under the "correct" one.

Each canonical config nonetheless orders its tables to match the panels
of the command's `--help`, ending in `[advanced]`: what a first run has
to decide is above that table, and what has a default that is usually
right is below it. Copies you edit are free to depart from that.

Seven commands each have their own config dataclass, and all but
`specter match particles` ship a canonical TOML:

| Command | Config dataclass | Example TOML |
|---|---|---|
| `specter simulate particles` | `ParticleStackConfig` | `configs/particle.toml` |
| `specter simulate micrograph` | `MicrographConfig` | `configs/micrograph.toml` |
| `specter simulate tiltseries` | `TiltSeriesConfig` | `configs/tiltseries.toml` |
| `specter build tomogram` | `TomogramConfig` | `configs/tomogram.toml` |
| `specter build ice` | `IceCacheConfig` | `configs/ice.toml` |
| `specter reconstruct particle` | `ReconstructionConfig` | `configs/reconstruct.toml` |
| `specter match particles` | `MatchConfig` | none |

Every field a command accepts, along with its default and a one-line
description, appears in that command's `--help` output and, where one
ships, in the comments of its canonical TOML. This page covers how the
two layers combine, not a field-by-field listing.

Copy the canonical file and edit it for a real run rather than starting from
an empty TOML; every field left out of your copy still gets its default
from the dataclass itself.

An unrecognised field in a TOML file causes a load-time error naming the
bad key, not a typo that gets ignored. A handful of fields have been
renamed since they were introduced; if a key was renamed rather than
removed, the error names the current spelling directly.

## CLI overrides

Every field on a config dataclass gets a matching `--field_name` flag;
SPECTER generates it automatically rather than hand-writing it per
command, so the two never drift apart. Two things follow from that:

- **A flag's real default is `None`, not the dataclass default shown next
  to it in `--help`.** `specter` needs to distinguish "you explicitly
  passed `--dose 20`" from "you never mentioned dose," and Click can
  only do that by comparing an option's value against its parameter
  source. If the flag's default were the dataclass's own default, a
  config setting `dose = 40` would get clobbered back to 20, without
  warning, by a flag you never typed. `--help` prints the TOML/dataclass
  default under `[default: ...]` for reference; that is not the value the
  flag takes when omitted.
- **`specter` applies only the flags you type.** `apply_overrides`
  receives the subset of parameters Click's `ParameterSource` reports
  as `COMMANDLINE`, and writes them onto the config loaded from TOML in a
  second pass. A field present in your TOML and never mentioned on the
  command line keeps the TOML's value untouched.

The effective value of any field follows a fixed order: the CLI flag if
you typed it, else the TOML file's value if it sets that field, else the
dataclass's own default. There is no fourth source and no field-by-field
opt-out of this order.

`list[...]`-typed fields (for example a tomogram's `targets` and `filler`
lists, written in `configs/tomogram.toml` as inline tables under
`[targets]` and `[filler]`, or its `[[membrane]]` tables) work in
TOML/Python only: no single CLI token can represent a list of tables, so these fields have no matching flag. Set
them in the config file or from Python.

```bash
# TOML alone
specter simulate particles --config configs/particle.toml

# TOML, then two fields overridden from the command line
specter simulate particles \
    --config configs/particle.toml \
    --pdb_source 6bdf \
    --n_particles 200
```

### Per-particle sampling fields on the command line

Some fields accept either a constant or a `[low, high]` range sampled per
particle, micrograph, or bead:

- `specter simulate particles`: `dose`, `defocus`, `coincidence_radius`,
  `potential_scale`, `astigmatism`, `astigmatism_angle`, `phaseshift`,
  `tiltx`, `tilty`, `trefoil1`, `trefoil2` and `tetrafoil1` to
  `tetrafoil4`.
- `specter simulate micrograph`: `dose`, `defocus`, `coincidence_radius`
  and `potential_scale`.
- `specter build tomogram`: `bead_roughness`.

Two micrograph fields share the same two-number spelling without being
sampled: `ice_thickness_range` gives a wedge's minimum and maximum
thickness, and `ice_hole_offset` gives the hole centre as `[x, y]`.

In TOML a range is naturally a list: `defocus = [5000.0, 15000.0]`. A CLI flag can only carry one token, so you write
the same range as a comma-separated string instead: `--defocus
5000,15000`. `specter` accepts that comma-separated spelling inside a
TOML file too, so a config written before the list form existed keeps
working unchanged.

## Validating a config from Python

Loading a TOML file and applying CLI-style overrides are both plain
functions in `specter.config`, independent of Click and usable from a
notebook or script:

```python
from specter.config import ParticleStackConfig, load_config, apply_overrides

config = load_config("configs/particle.toml", ParticleStackConfig)
apply_overrides(config, {"n_particles": 200, "device": "cuda:0"})
```

`apply_overrides` rejects a key that doesn't name a real field outright,
rather than attaching it under an attribute nothing reads, which is what a
bare `setattr` would do.

## See also

- [Manage jobs](jobs.md): how `output_dir`, `project`, and `job_id` route a
  run's output, independent of this page's TOML/CLI mechanics.
- The individual command pages ([particle stack](particle-stack.md),
  [micrograph](micrograph.md), [tilt series](tilt-series.md), [build a
  tomogram](build-tomogram.md), [reconstruction](reconstruction.md), [ice
  cache](ice-cache.md)) for what each config's fields control.
