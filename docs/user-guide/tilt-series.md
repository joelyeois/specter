# Generate a tilt series

`specter simulate tiltseries` loads a pre-built specimen volume and
simulates the tilted acquisition: multislice (or first-Born/projection/CTF)
scattering through the volume at each tilt angle, CTF/aberrations, dose,
amorphous ice, and detector effects. It saves the result as a `.mrcs` tilt
stack plus a RELION-style `.star` file. This is the **imaging half** of the
cryo-ET pipeline; the volume itself comes from `specter build tomogram`.
See [Build a tomogram specimen](build-tomogram.md) for that half.

## Basic usage

Every run loads a TOML config first, then applies any flags given on the
command line as overrides:

```bash
specter simulate tiltseries --config configs/tilt_series.toml \
    --volume_path tomograms/tomogram.mrc
```

```bash
specter simulate tiltseries \
    --config configs/tilt_series.toml \
    --volume_path tomograms/tomogram.mrc \
    --device cuda:0 \
    --n_tilts 41 \
    --min_tilt_angle -60 --max_tilt_angle 60
```

`configs/tilt_series.toml` is the canonical starting point. Copy it and
edit for your own runs; the
[command reference](../api/cli/simulate.md#specter-simulate-tiltseries) documents
every field. `volume_path` has no default and is the one
field you always need to supply, either in the TOML, via `--volume_path`
(pointing at whatever `specter build tomogram` already wrote), or by
passing `--tomogram_config` instead to build it as part of this same
command. See [Chaining with `specter build
tomogram`](#chaining-with-specter-build-tomogram) below.

## What you'll usually tune

- **Specimen**: `volume_path` (the `.mrc`/`.mrcs`/`.pt` scattering-potential
  volume to image, already in scattering-potential units, not a raw density
  map) and `voxel_size` (Å/voxel, **must match** whatever produced the
  volume, e.g. `TomogramConfig.voxel_size`; it is not auto-detected from
  the file).
- **Microscope**: `voltage` (kV), `dose_per_tilt` (e⁻/Å² per tilt angle,
  not total dose), `n_frames` (movie frames per tilt, feeds the coincidence-
  loss model), `cs` (mm), `alpha` (amplitude contrast ratio).
- **Defocus**: `defocus` (Å, positive = underfocus), applied the same at
  every tilt.
- **Tilt geometry**: `min_tilt_angle`/`max_tilt_angle` (degrees) and
  `n_tilts` span an evenly-spaced tilt series; `tilt_axis` (`x` or `y`)
  picks which in-plane axis the specimen rotates about. For a tilt axis
  that isn't exactly horizontal/vertical (typical of real acquisition
  geometry), see [Beyond angles: real tilt geometry](#beyond-angles-real-tilt-geometry-quaternions-and-aretomo3)
  below, which needs the Python API, not this CLI.
- **Models**: `scattering_model` (`multislice` default, most accurate;
  `firstborn`/`projection`/`ctf` are cheaper approximations; see
  [Scattering](../concepts/scattering/index.md)), `noise_model` (`poisson`
  or `none`), `detector_model` (`none`, `perfect`, `k3_300kv`, `k3_200kv`).

Everything else (envelopes, ice, output naming) lives under its own panel
in the [command reference](../api/cli/simulate.md#specter-simulate-tiltseries).

## Advanced: envelopes, ice, and edge handling

- **Envelopes**: `convergence_angle` (mrad) and `cc` (mm, chromatic
  aberration coefficient) are both `None`/unset by default, which disables
  the corresponding Cs (spatial-coherence) and Cc (temporal-coherence)
  envelopes; set either to enable it. `energy_spread`, `deltaV_V`,
  `deltaI_I` feed the Cc envelope specifically. `dose_envelope` applies the
  Grant & Grigorieff (2015) cumulative-dose envelope across tilts.
- **Ice**: `ice_model`: `"gd"` (default, `IceBank`'s cached
  `GradientSKIcemaker` configs, realistic and near-free at this
  cache size), `"random"` (cheap, low-fidelity `RandomIcemaker`), or `"none"`
  (no ice). `ice_cache_dir` overrides the bundled `specter/ice_data/ice_cache`;
  `ice_relax_steps` runs local MLBOP seam relaxation for `"gd"` (0 by
  default). See [Ice structure](../concepts/ice.md).
- **`coincidence_radius`**: effective coincidence-loss exclusion radius in
  pixels (exclusion area = πr²) for direct-detector modelling; matters most
  once `detector_model` is one of the `k3_*` presets.
- **`pad_fft`**: pad the volume before FFT to avoid multislice
  edge-wraparound artifacts under tilt. Off by default; turn on if you see
  streaking artifacts near the image edges at large tilt angles.
- **`micrograph_size`**: output tilt-image size in pixels (square).
  Defaults to the specimen volume's own XY extent; set explicitly to crop
  or pad the field of view independent of the specimen box.

## Output

Alongside `{filename}.mrcs` (the `(n_tilts, H, W)` tilt stack) and
`{filename}.star` (RELION-style, one row per tilt: voltage, pixel size, CTF
params, dose, coincidence radius, tilt angle), `save_exitwaves` additionally
writes the complex exit wave as two separate `.mrcs` files:
`{filename}_exitwave_magnitude.mrcs`/`{filename}_exitwave_phase.mrcs` (or
`clean_exitwave_*` when `ice_model="none"`). `normalize_tilt_series`
zero-means and unit-normalizes each tilt image independently before saving.

## Chaining with `specter build tomogram`

Two ways to run the specimen-building and imaging stages back to back:

**One command**, via `--tomogram_config`: pass a `TomogramConfig` TOML
instead of `--volume_path`/`volume_path`, and `specter simulate tiltseries`
builds that specimen first (identical to `specter build tomogram --config
...`, including writing its own `.mrc` + picks/segmentation to
`configs/tomogram.toml`'s own `output_dir`/`filename`), then feeds the
result straight in as this run's volume, with no path to copy by hand:

```bash
specter simulate tiltseries --config configs/tilt_series.toml \
    --tomogram_config configs/tomogram.toml \
    --device cuda:0
```

`--tomogram_config` and `--volume_path`/`config.volume_path` are mutually
exclusive; passing both raises an error rather than silently picking one.
Species/membrane/filament choices still come from the tomogram TOML (edit
it directly, or see
[`specter build tomogram`](../api/cli/build.md#specter-build-tomogram) for its own
per-field flags, which aren't reachable through `simulate tiltseries`); imaging
choices come from `--config`/its flags as usual.

`--tomogram_config` always builds exactly **one** tomogram. `n_tomograms`
is `specter build tomogram`'s own CLI-only flag (not a `TomogramConfig`
field; `load_config` rejects a TOML with `n_tomograms` set outright) and
isn't settable through this chained path. For several tomograms, use
the two-command form below with `specter build tomogram --n_tomograms N`
(writes `output_dir/0001/`, `0002/`, ...; see [Multiple
tomograms](build-tomogram.md#multiple-tomograms)), then call `specter
simulate tiltseries --volume_path ...` once per numbered subdirectory.

**Two commands**, for full control over the intermediate volume's path,
multiple tomograms, or building once and imaging it several ways:

```bash
specter build tomogram --config configs/tomogram.toml \
    --output_dir tomograms --filename tomogram

specter simulate tiltseries --config configs/tilt_series.toml \
    --volume_path tomograms/tomogram.mrc \
    --device cuda:0
```

## Beyond angles: real tilt geometry (quaternions and AreTomo3)

The CLI's `min_tilt_angle`/`max_tilt_angle`/`n_tilts`/`tilt_axis` fields
only describe an idealized, evenly-spaced series about an exactly
horizontal or vertical axis. Real acquisitions have a tilt axis at some
arbitrary angle and (after alignment) per-tilt shifts. `specter.tilt`
(`read_aretomo3_aln`/`read_aretomo3_xf`/`read_aretomo3_global_shifts`,
`tilt_to_quaternions`) parses [AreTomo3](https://github.com/czimaginginstitute/AreTomo3)'s
`.aln`/`.xf`/global-shifts output
into quaternions/translations for exactly this case. This is a Python-API
feature only: `TiltSeriesGenerator` accepts `quaternions=` as an
alternative to `angles=`, but `TiltSeriesConfig`/the CLI does not currently
expose it, so matching a real tilt series' geometry means calling
`TiltSeriesGenerator` directly rather than `specter simulate tiltseries`.

## Using it from Python instead of the CLI

`run_tilt_series(config)` (`specter.pipelines`) is the same function the
CLI calls. Build a `TiltSeriesConfig` directly in Python (or load one with
`specter.config.load_config`) instead of going through the command line.
For anything the config/CLI doesn't expose (quaternion-based tilt geometry,
per-tilt CTF variation, etc.), drive `TiltSeriesGenerator`
(`specter.imagegenerator`) directly instead; `run_tilt_series` is a thin
wrapper around it.

## See also

- [`configs/tilt_series.toml`](https://github.com/joelyeois/specter/tree/main/configs):
  the canonical, commented example config.
- [Build a tomogram specimen](build-tomogram.md): builds the
  `volume_path` this page images.
- [Forward simulation](../concepts/forward-simulation.md),
  [Scattering](../concepts/scattering/index.md),
  [Aberrations](../concepts/aberrations.md),
  [Detector](../concepts/detector.md): the physics behind each stage.
- [Ice structure](../concepts/ice.md): the `ice_model` options.
- Zheng, S., Wolff, G., Greenan, G., Chen, Z., Faas, F. G. A., Bárcena, M.,
  Koster, A. J., Cheng, Y., & Agard, D. A. (2022). AreTomo: An integrated
  software package for automated marker-free, motion-corrected cryo-electron
  tomographic alignment and reconstruction. *Journal of Structural Biology:
  X*, 6, 100068. [doi:10.1016/j.yjsbx.2022.100068](https://doi.org/10.1016/j.yjsbx.2022.100068)
