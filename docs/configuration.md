# Configuration

## TOML files and how they are overridden

Runs are described by a TOML file, and any single field can be changed on
the command line without editing it.

Two configs ship in `configs/`: `particle.toml` and
`micrograph.toml`. The notebooks under `demo-notebooks/create_*/` keep
their own copies, deliberately tuned differently — the micrograph notebook
uses a 0.83 Å pixel and a 2048-pixel field where the shipped config uses
1.0 Å and 4096.

Precedence, lowest to highest priority:

1. Dataclass defaults in `config.py`
2. The TOML file passed via `--config`
3. Command-line flags, e.g. `--n_particles 3000`

```bash
python demo-scripts/generate_particle_stack.py \
    --config configs/particle.toml \
    --n_particles 3000
```

**Why an unspecified flag does not clobber your file.** Every argument
except `--config` declares `default=argparse.SUPPRESS`. A flag you did
not pass never appears in the parsed namespace at all, so it cannot
overwrite your TOML value with an argparse default. Without this, every
unspecified flag would silently reset its field.

## The file

Sections group related settings for readability. They are **cosmetic**:
all tables are flattened into one namespace before the config object is
built, so `[potential] pdb_code` and a bare top-level `pdb_code` are
equivalent.

```toml
--8<-- "configs/particle.toml"
```

## Reading the conventions

| Convention | What it means |
|---|---|
| A commented-out field | An optional feature. Uncommenting `dose_max` switches dose from fixed to randomised per particle; uncommenting `cc` enables the temporal-coherence envelope. |
| `_min` without `_max` | A fixed value for the whole run. |
| `_min` and `_max` | Sampled uniformly per particle, and recorded per particle in the output STAR file. |
| Booleans | Written as values, not bare flags: `--assembly True`. |
| An unrecognised key | Raises an error rather than being ignored, so typos surface immediately. |

!!! note
    **Only two scripts read TOML.** `generate_particle_stack.py` and
    `generate_micrograph.py` accept `--config`. The CryoSPARC and
    tilt-series scripts are plain argparse with no configuration-file
    layer.
