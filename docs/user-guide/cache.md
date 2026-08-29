# Manage the PDB cache

Every SPECTER config that names a structure by accession code
(`pdb_source` in a particle, micrograph, or tomogram config) fetches it
from RCSB the first time it's needed and keeps the downloaded file for
reuse. That cache lives outside any project, at `~/.cache/specter/pdb` by
default, because a structure fetched from RCSB is the same file
regardless of which project needs it. Caching it once per user, rather
than once per working directory, saves you from re-downloading the same
assembly for every simulation that references it.

## Locating the cache

```bash
specter cache dir
```

prints the resolved cache directory without inspecting its contents.
`specter` chooses the directory the same way `torch.hub` and HuggingFace
resolve their own caches:

1. `$SPECTER_PDB_CACHE`, if set: an explicit override.
2. `$XDG_CACHE_HOME/specter/pdb`, if `$XDG_CACHE_HOME` is set.
3. `~/.cache/specter/pdb`, otherwise.

Setting `$XDG_CACHE_HOME` relocates every XDG-aware tool's cache at once;
setting `$SPECTER_PDB_CACHE` moves only this one. Either is the way to point
the cache at scratch space on a cluster, where `$HOME` is often small and
slow and `/scratch` is neither:

```bash
export SPECTER_PDB_CACHE=/scratch/$USER/specter-pdb-cache
```

Set it in your shell profile (or the job script's environment) so every
`specter` invocation agrees on the same location.

## Inspecting the cache

```bash
specter cache info
```

reports the resolved directory, how many structures are stored, and their
total size on disk. On this machine, with a cache built up over ordinary use:

```text
Location: /home/user/.cache/specter/pdb
Structures: 32
Size: 130.4 MB
```

An empty or not-yet-created cache reports as such rather than erroring:

```text
Location: /home/user/.cache/specter/pdb
Empty -- nothing downloaded yet.
```

## Clearing the cache

```bash
specter cache clean
```

deletes the entire cache directory, after confirming how much it will free:

```text
Delete 32 cached structure(s) (130.4 MB) from /home/user/.cache/specter/pdb? [y/N]:
```

Pass `-y`/`--yes` to skip the prompt in a script or CI job:

```bash
specter cache clean --yes
```

```text
✓ Removed 32 structure(s), 130.4 MB freed.
```

Running `clean` against an already-empty or missing cache is a no-op rather
than an error:

```text
Nothing to clean -- /home/user/.cache/specter/pdb does not exist.
```

This is always safe to run, because everything the cache holds can be
produced again. It holds two kinds of thing. Downloads: `pdb_source`
accepts either a 4-character PDB/mmCIF accession code or a path to a local
structure file, and `specter` handles the two differently, fetching an
accession code and writing it into the cache while reading a local path
directly from where it sits and never copying it in. And parsed
structures, under `parsed/`: turning a structure file into atom positions,
elements and bonded-species types is the dominant cost of using one that
is already downloaded, so the result is kept and reused. Parsing a
220,000-atom assembly takes 16.7 seconds against 0.09 seconds to read the
arrays back.

A parsed entry is keyed on its source file's path, size and modification
time, along with every setting that changes the result, so editing a
structure in place produces a fresh parse rather than the previous one.
Clearing the cache therefore discards nothing irreplaceable -- downloads
get re-fetched from RCSB, parses get recomputed from files that are still
where they were -- which is the same guarantee `uv cache clean` and
`pip cache purge` make about their own caches.

## Referencing a structure

`pdb_source` shows up wherever a config names a structure: particle and
micrograph configs at the top level, tomogram configs inside
`[[targets]]` and `[[filler]]` tables:

```toml
pdb_source = "6qzp"                        # fetched into the cache
pdb_source = "/data/structures/mystruct.cif"  # read in place, never cached
```

`specter` resolves both forms the same way at every call site, so a
config can mix accession codes and local files: swap one target for the
other, say from a subdirectory of a colleague's structure library, and
nothing else in the config needs to change.
