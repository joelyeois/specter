"""`specter align` reads maps, writes the aligned one and reports the pose."""

from __future__ import annotations

from pathlib import Path

import torch
from click.testing import CliRunner
from test_alignment import _blobs, _pose

from specter.cli._cli import cli
from specter.io import read_mrc, write_mrc
from specter.symmetries import get_rotation_matrices, symmetrize_exact

N = 48


def test_align_symmetry_writes_a_symmetric_map(tmp_path: Path) -> None:
    source, output = tmp_path / "in.mrc", tmp_path / "out.mrc"
    write_mrc(source, _blobs(N, "D3", *_pose(3, 4.0)), 1.5)

    args = ["align", "symmetry", str(source), str(output), "--sym", "D3"]
    result = CliRunner().invoke(cli, [*args, "--device", "cpu"])

    assert result.exit_code == 0, result.output
    assert "rotation" in result.output and "significance" in result.output
    aligned, voxel_size = read_mrc(output)
    assert voxel_size == 1.5
    ops = get_rotation_matrices("D3", return_affine=False)
    assert (
        float((symmetrize_exact(aligned, ops) - aligned).norm() / aligned.norm()) < 1e-3
    )

    again = CliRunner().invoke(cli, [*args, "--device", "cpu"])
    assert again.exit_code != 0
    assert "--overwrite" in again.output


def test_align_volumes_writes_the_map_on_the_reference(tmp_path: Path) -> None:
    reference = _blobs(N, "C1")
    paths = [tmp_path / name for name in ("in.mrc", "ref.mrc", "out.mrc")]
    write_mrc(paths[0], _blobs(N, "C1", *_pose(4, 4.0)), 1.5)
    write_mrc(paths[1], reference, 1.5)

    result = CliRunner().invoke(
        cli, ["align", "volumes", *map(str, paths), "--device", "cpu"]
    )

    assert result.exit_code == 0, result.output
    aligned, _ = read_mrc(paths[2])
    assert float((aligned - reference).norm() / reference.norm()) < 1e-3


def test_align_volumes_refuses_maps_on_different_grids(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("in.mrc", "ref.mrc", "out.mrc")]
    write_mrc(paths[0], torch.zeros(N, N, N), 1.5)
    for box, voxel_size, said in [(32, 1.5, "box"), (N, 2.0, "voxel size")]:
        write_mrc(paths[1], torch.zeros(box, box, box), voxel_size)
        result = CliRunner().invoke(
            cli, ["align", "volumes", *map(str, paths), "--device", "cpu"]
        )
        assert result.exit_code != 0
        assert said in result.output
        assert not paths[2].exists()


def test_align_symmetry_refuses_a_volume_that_is_not_a_cube(tmp_path: Path) -> None:
    source = tmp_path / "in.mrc"
    write_mrc(source, torch.zeros(N, N, 32), 1.5)
    result = CliRunner().invoke(
        cli,
        ["align", "symmetry", str(source), str(tmp_path / "out.mrc"), "--sym", "C2"],
    )
    assert result.exit_code != 0
    assert "cubic" in result.output
