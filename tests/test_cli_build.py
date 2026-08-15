from __future__ import annotations

import subprocess as proc
import sys
from pathlib import Path

import mrcfile
import numpy as np

_SMALL_FIXTURE = Path(__file__).parent.parent / "specter-data" / "pdb" / "1mbo.cif"
_LARGE_FIXTURE = (
    Path(__file__).parent.parent / "specter-data" / "pdb" / "1bxn-assembly1.cif"
)


def _write_test_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
[targets]
targets = [
    {{ pdb_source = "{_SMALL_FIXTURE}", n_copies = 2 }},
]

[specimen]
target_shape = [24, 32, 32]
voxel_size = 10.0
filler_occupancy_fraction = 0.0
gap = 5.0
seed = 0

[output]
output_dir = "{output_dir}"
filename = "test_tomogram"
"""
    )


def _run_build_cli(config_path: Path, *extra_args: str) -> proc.CompletedProcess:
    args = [
        sys.executable,
        "-m",
        "specter.cli._cli",
        "build",
        "tomogram",
        "--config",
        str(config_path),
        *extra_args,
    ]
    return proc.run(args, capture_output=True, encoding="utf-8")


def test_cli_build_tomogram_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "tomogram.toml"
    _write_test_config(config_path, tmp_path)

    result = _run_build_cli(config_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test_tomogram.mrc").exists()

    pick_files = list(tmp_path.glob("*.ndjson"))
    assert len(pick_files) == 1
    lines = pick_files[0].read_text().strip().splitlines()
    assert len(lines) > 0
    assert '"type": "orientedPoint"' in lines[0]
    assert '"xyz_rotation_matrix"' in lines[0]


def test_cli_build_tomogram_help_smoke() -> None:
    result = proc.run(
        [sys.executable, "-m", "specter.cli._cli", "build", "tomogram", "--help"],
        capture_output=True,
        encoding="utf-8",
    )
    assert result.returncode == 0
    assert "--filler_occupancy_fraction" in result.stdout
    assert "--gap" in result.stdout
    assert "--n_tomograms" in result.stdout


def test_cli_build_tomogram_n_tomograms(tmp_path: Path) -> None:
    """--n_tomograms 2 writes two distinct, seed-varied tomograms into their
    own numbered subdirectories instead of overwriting a single output."""
    config_path = tmp_path / "tomogram.toml"
    _write_test_config(config_path, tmp_path)

    result = _run_build_cli(config_path, "--n_tomograms", "2")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "0001" / "test_tomogram.mrc").exists()
    assert (tmp_path / "0002" / "test_tomogram.mrc").exists()
    assert not (tmp_path / "test_tomogram.mrc").exists()

    assert list((tmp_path / "0001").glob("*.ndjson"))
    assert list((tmp_path / "0002").glob("*.ndjson"))


def _write_membrane_test_config(
    path: Path, output_dir: Path, *, include_lumen: bool, include_transmembrane: bool
) -> None:
    # Same tuned scale as tests/test_tomogram_generator.py's own
    # A spherical_harmonics ellipsoid whose lumen comfortably holds 1mbo.
    # The radius is load-bearing and was measured, not guessed: 1mbo packs
    # at max_diameter/2 = 31.4 A and gap=5 on top, so a placement
    # needs 36.4 A of clearance from the shell. Lumen voxels clearing that
    # bar, by radius (voxel_size=8, seed=0):
    #
    #   70 A ->     3 of   705 lumen voxels   (RSA reliably finds none)
    #   90 A ->   177 of 3,756
    #  110 A ->   924 of 7,286                (used here)
    #  130 A -> 2,641 of 12,605
    #
    # 70 A was the previous value and is why this test failed: a viable
    # site existed, but at 0.4% of the region RSA never sampled one. 110 A
    # leaves ~300x that margin while still fitting the box with room to
    # spare, and (measured) also places transmembrane sites reliably, so
    # both smoke tests can share one geometry.
    sh_radius_a = 110.0
    transmembrane_block = (
        f"""
[[membrane_transmembrane_specs]]
pdb_source = "{_SMALL_FIXTURE}"
n_copies = 1
"""
        if include_transmembrane
        else ""
    )
    lumen_block = (
        f"""
[[filler]]
pdb_source = "{_SMALL_FIXTURE}"
location = "lumen"
"""
        if include_lumen
        else ""
    )
    path.write_text(
        f"""
[[membrane]]
sh_axes = [{sh_radius_a}, {sh_radius_a}, {sh_radius_a}]
sh_amplitude = 0.15
n_lipids_per_leaflet = 6
{transmembrane_block}
{lumen_block}
[[filler]]
pdb_source = "{_LARGE_FIXTURE}"
location = "cytosol"

filler_occupancy_fraction = 0.1

[specimen]
target_shape = [64, 64, 64]
voxel_size = 8.0
gap = 5.0
seed = 0

[output]
output_dir = "{output_dir}"
filename = "test_membrane_tomogram"
"""
    )


def test_cli_build_tomogram_membrane_smoke(tmp_path: Path) -> None:
    """Cytosol + lumen packing end to end through the CLI."""
    config_path = tmp_path / "membrane_tomogram.toml"
    _write_membrane_test_config(
        config_path, tmp_path, include_lumen=True, include_transmembrane=False
    )

    result = _run_build_cli(config_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test_membrane_tomogram.mrc").exists()

    pick_files = {p.name for p in tmp_path.glob("*.ndjson")}
    assert any("-lumen-" in name for name in pick_files), pick_files
    assert any("-cytosol-" in name for name in pick_files), pick_files


def test_cli_build_tomogram_membrane_transmembrane_smoke(tmp_path: Path) -> None:
    """Cytosol + transmembrane placement end to end through the CLI."""
    config_path = tmp_path / "membrane_tomogram.toml"
    _write_membrane_test_config(
        config_path, tmp_path, include_lumen=False, include_transmembrane=True
    )

    result = _run_build_cli(config_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test_membrane_tomogram.mrc").exists()

    pick_files = {p.name for p in tmp_path.glob("*.ndjson")}
    assert any("-cytosol-" in name for name in pick_files), pick_files
    assert any("-transmembrane-" in name for name in pick_files), pick_files


def test_cli_build_tomogram_membrane_and_targets_combined(tmp_path: Path) -> None:
    """config.membrane, config.targets (exact-count), and config.filler
    (ratio) can all be combined in one tomogram -- no more mutual-exclusivity
    error between "membrane mode" and "sphere-packing mode"."""
    config_path = tmp_path / "combined_tomogram.toml"
    _write_membrane_test_config(
        config_path, tmp_path, include_lumen=False, include_transmembrane=False
    )
    with open(config_path, "a") as f:
        f.write(f'\n[[targets]]\npdb_source = "{_SMALL_FIXTURE}"\nn_copies = 1\n')

    result = _run_build_cli(config_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test_membrane_tomogram.mrc").exists()

    pick_files = {p.name for p in tmp_path.glob("*.ndjson")}
    assert any(
        _SMALL_FIXTURE.stem in name and "-cytosol-" in name for name in pick_files
    ), pick_files
    assert any(
        _LARGE_FIXTURE.stem in name and "-cytosol-" in name for name in pick_files
    ), pick_files


def test_cli_build_tomogram_membrane_multi_instance_smoke(tmp_path: Path) -> None:
    """Two [[membrane]] entries at distinct position_xyz -- composited into
    one tomogram, both membrane instances' worth of cytosol picks and a
    _membrane_labels.mrc with 2 distinct instance IDs appear."""
    config_path = tmp_path / "multi_membrane_tomogram.toml"
    config_path.write_text(
        f"""
[[membrane]]
sh_axes = [55.0, 55.0, 55.0]
sh_amplitude = 0.15
n_lipids_per_leaflet = 6
position_xyz = [-150.0, 0.0, 0.0]

[[membrane]]
shape_backend = "swept_spline"
swept_total_length = 150.0
swept_tube_radius = 35.0
n_lipids_per_leaflet = 6
position_xyz = [150.0, 0.0, 0.0]

[[filler]]
pdb_source = "{_LARGE_FIXTURE}"
location = "cytosol"

filler_occupancy_fraction = 0.05

[specimen]
target_shape = [80, 80, 80]
voxel_size = 8.0
gap = 5.0
seed = 0

[output]
output_dir = "{tmp_path}"
filename = "test_multi_membrane_tomogram"
"""
    )

    result = _run_build_cli(config_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test_multi_membrane_tomogram.mrc").exists()

    with mrcfile.open(
        tmp_path / "test_multi_membrane_tomogram_membrane_labels.mrc"
    ) as mrc:
        unique_ids = set(np.unique(mrc.data).tolist()) - {0}
    assert unique_ids == {1, 2}


def test_cli_build_tomogram_write_segmentation_smoke(tmp_path: Path) -> None:
    """Membrane mode writes protein/membrane/region label .mrc files by
    default (write_segmentation defaults True)."""
    config_path = tmp_path / "membrane_tomogram.toml"
    _write_membrane_test_config(
        config_path, tmp_path, include_lumen=False, include_transmembrane=False
    )

    result = _run_build_cli(config_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test_membrane_tomogram_protein_labels.mrc").exists()
    assert (tmp_path / "test_membrane_tomogram_membrane_labels.mrc").exists()
    assert (tmp_path / "test_membrane_tomogram_regions.mrc").exists()


def test_cli_build_tomogram_write_segmentation_override(tmp_path: Path) -> None:
    """--write_segmentation False overrides the loaded TOML config's default
    (True) end to end -- no label .mrc files should be written."""
    config_path = tmp_path / "membrane_tomogram.toml"
    _write_membrane_test_config(
        config_path, tmp_path, include_lumen=False, include_transmembrane=False
    )

    result = _run_build_cli(config_path, "--write_segmentation", "False")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test_membrane_tomogram.mrc").exists()
    assert not (tmp_path / "test_membrane_tomogram_protein_labels.mrc").exists()
    assert not (tmp_path / "test_membrane_tomogram_membrane_labels.mrc").exists()
    assert not (tmp_path / "test_membrane_tomogram_regions.mrc").exists()


def test_cli_build_tomogram_sphere_packing_write_segmentation(tmp_path: Path) -> None:
    """Sphere-packing (non-membrane) mode also gets a _protein_labels.mrc
    under the generalized write_segmentation."""
    config_path = tmp_path / "tomogram.toml"
    _write_test_config(config_path, tmp_path)

    result = _run_build_cli(config_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test_tomogram_protein_labels.mrc").exists()
    assert not (tmp_path / "test_tomogram_membrane_labels.mrc").exists()
    assert not (tmp_path / "test_tomogram_regions.mrc").exists()


def test_cli_build_tomogram_zero_instances_fit_smoke(tmp_path: Path) -> None:
    """A box too small for the requested species (0/n exact-count instances
    fit) must still complete and save a volume, not crash in export_picks --
    an empty self.placements is a legitimate generate() outcome, not
    evidence generate() was never called."""
    config_path = tmp_path / "tomogram.toml"
    config_path.write_text(
        f"""
[targets]
targets = [
    {{ pdb_source = "{_LARGE_FIXTURE}", n_copies = 3 }},
]

[specimen]
target_shape = [8, 8, 8]
voxel_size = 2.0
filler_occupancy_fraction = 0.0
gap = 5.0
seed = 0

[output]
output_dir = "{tmp_path}"
filename = "test_tomogram"
"""
    )

    result = _run_build_cli(config_path)
    assert result.returncode == 0, result.stderr
    assert "only 0/3 exact-count instances fit" in result.stderr
    assert (tmp_path / "test_tomogram.mrc").exists()
    assert list(tmp_path.glob("*.ndjson")) == []


def test_cli_build_tomogram_write_picks_override(tmp_path: Path) -> None:
    """--write_picks False overrides the loaded TOML config's default (True)
    end to end -- no .ndjson files should be written."""
    config_path = tmp_path / "tomogram.toml"
    _write_test_config(config_path, tmp_path)

    result = _run_build_cli(config_path, "--write_picks", "False")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test_tomogram.mrc").exists()
    assert list(tmp_path.glob("*.ndjson")) == []
