# specter.cli

`specter` is the command-line entry point to everything in this package. Every
subcommand loads a TOML config, applies the flags you passed on top of it, and
calls into `specter.pipelines`; the flags and the config fields are the same
set of settings under two spellings, so anything below can be written either
way. [Configure a run](../user-guide/configuration.md) covers the config file
itself, and the [User Guide](../user-guide/particle-stack.md) walks through
each command on a real task.

Three rules apply throughout the reference below.

**A flag you do not pass is not applied.** Options default to nothing rather
than to the value in the Default column, which is what lets a flag override one
field of a config without the rest of the command line silently overwriting the
other fields with built-in defaults. The Default column is the value the field
falls back to when neither the config nor a flag sets it. `_none_` means there
is no fallback: the setting is either optional in the run or must be supplied,
through `--config` or through its flag, before the command will start.

**`--config` is optional.** Without it every setting takes its dataclass
default, so only settings that have none must be passed as flags. The worked
examples in [`configs/`](https://github.com/joelyeois/specter/tree/main/configs)
are a starting point rather than a requirement.

**Flags are grouped into panels.** The bolded group above each table is the
panel `--help` prints the flag under, in the same order, which is also the
order the matching file in `configs/` declares its TOML tables. A panel called
`Advanced` is last in every command and holds the settings with a
usually-correct default.

--8<-- "docs-includes/cli-reference.md"

## Python API

The functions below construct the command objects documented above. They are
of interest when embedding the CLI or adding a subcommand, not when running
one.

::: specter.cli._cli
    options:
      show_root_heading: true

::: specter.cli.simulate
    options:
      show_root_heading: true
