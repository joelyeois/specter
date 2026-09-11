"""
Tests for the CryoSPARC ``.cs`` -> RELION ``.star`` metadata converter.

Values in the fake dataset below are deterministic (not sampled) so that
every unit/sign conversion can be asserted against a hand-computed number.
"""

import math
import os

import numpy as np
import pytest
import roma
import starfile
import torch

from specter.io import _convert, convert_csfile_to_starfile


def fake_dataset(**overrides) -> type:
    """
    A stand-in for ``cryosparc.dataset.Dataset`` with three particles and
    fully deterministic values. ``overrides`` replace individual columns.
    """
    n = 3
    dtype = np.float32
    columns = {
        "uid": np.arange(n, dtype=np.uint64),
        "blob/path": np.array(["J1/particles.mrcs"] * n),
        "blob/idx": np.array([0, 1, 2], dtype=np.int64),
        "blob/shape": np.array([[128, 128]] * n, dtype=np.int64),
        # 2 px shift at 1.5 A/px -> 3.0 A
        "alignments3D/shift": np.array([[2.0, -2.0]] * n, dtype=dtype),
        "alignments3D/psize_A": np.full(n, 1.5, dtype=dtype),
        "alignments3D/pose": np.array([[0.1, 0.2, 0.3]] * n, dtype=dtype),
        "alignments3D/split": np.array([0, 1, 0], dtype=np.int64),
        "alignments3D/alpha": np.full(n, 0.75, dtype=dtype),
        "ctf/cs_mm": np.full(n, 2.7, dtype=dtype),
        "ctf/accel_kv": np.full(n, 300.0, dtype=dtype),
        "ctf/amp_contrast": np.full(n, 0.1, dtype=dtype),
        "ctf/df1_A": np.full(n, 10000.0, dtype=dtype),
        "ctf/df2_A": np.full(n, 9800.0, dtype=dtype),
        "ctf/df_angle_rad": np.full(n, math.pi / 4, dtype=dtype),
        "ctf/phase_shift_rad": np.full(n, math.pi / 2, dtype=dtype),
        "ctf/tilt_A": np.zeros((n, 2), dtype=dtype),
        "ctf/shift_A": np.zeros((n, 2), dtype=dtype),
        "ctf/trefoil_A": np.zeros((n, 2), dtype=dtype),
        "ctf/tetra_A": np.zeros((n, 4), dtype=dtype),
        "ctf/anisomag": np.zeros((n, 4), dtype=dtype),
    }
    columns.update(overrides)

    class FakeDataset(dict):
        @classmethod
        def load(cls, csfile_path: str) -> "FakeDataset":
            return cls(columns)

    return FakeDataset


@pytest.fixture(autouse=True)
def _patch_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_convert, "Dataset", fake_dataset())


def _convert_and_read(tmp_path, **overrides):
    """Convert a fake dataset and read the two blocks back off disk."""
    if overrides:
        import specter.io._convert as mod

        mod.Dataset = fake_dataset(**overrides)
    out = tmp_path / "out.star"
    convert_csfile_to_starfile("fake.cs", str(out))
    data = starfile.read(str(out))
    return data["optics"], data["particles"]


def test_writes_a_single_file_with_optics_and_particles_blocks(tmp_path) -> None:
    out = tmp_path / "out.star"
    convert_csfile_to_starfile("fake.cs", str(out))

    assert out.exists()
    data = starfile.read(str(out))
    assert set(data) == {"optics", "particles"}
    assert len(data["particles"]) == 3


def test_image_name_is_one_based_index_at_path(tmp_path) -> None:
    _, particles = _convert_and_read(tmp_path)

    # blob/idx is 0-based; RELION's idx@stack is 1-based.
    assert list(particles["rlnImageName"]) == [
        "000001@J1/particles.mrcs",
        "000002@J1/particles.mrcs",
        "000003@J1/particles.mrcs",
    ]


def test_random_subset_is_one_based(tmp_path) -> None:
    _, particles = _convert_and_read(tmp_path)

    # alignments3D/split is 0/1; rlnRandomSubset is 1/2.
    assert list(particles["rlnRandomSubset"]) == [1, 2, 1]


def test_defocus_angle_converted_to_degrees(tmp_path) -> None:
    _, particles = _convert_and_read(tmp_path)

    assert particles["rlnDefocusAngle"].to_numpy() == pytest.approx(45.0, abs=1e-4)


def test_phase_shift_converted_to_degrees(tmp_path) -> None:
    _, particles = _convert_and_read(tmp_path)

    assert particles["rlnPhaseShift"].to_numpy() == pytest.approx(90.0, abs=1e-4)


def test_defocus_passes_through_in_angstrom(tmp_path) -> None:
    _, particles = _convert_and_read(tmp_path)

    assert particles["rlnDefocusU"].to_numpy() == pytest.approx(10000.0)
    assert particles["rlnDefocusV"].to_numpy() == pytest.approx(9800.0)


def test_origins_convert_pixels_to_angstrom_without_sign_flip(tmp_path) -> None:
    _, particles = _convert_and_read(tmp_path)

    # 2 px * 1.5 A/px = 3.0 A, no negation.
    assert particles["rlnOriginXAngst"].to_numpy() == pytest.approx(3.0)
    assert particles["rlnOriginYAngst"].to_numpy() == pytest.approx(-3.0)


def test_origins_fold_in_beam_shift(tmp_path) -> None:
    _, particles = _convert_and_read(
        tmp_path, **{"ctf/shift_A": np.full((3, 2), 0.5, dtype=np.float32)}
    )

    # RELION has no beam-shift column; an image shift and an origin offset
    # are the same operation, so it folds in losslessly.
    assert particles["rlnOriginXAngst"].to_numpy() == pytest.approx(2.5)
    assert particles["rlnOriginYAngst"].to_numpy() == pytest.approx(-3.5)


def test_beam_tilt_converted_to_milliradians(tmp_path) -> None:
    cs_angstrom = 2.7 * 1e7
    tilt_a = np.zeros((3, 2), dtype=np.float32)
    tilt_a[:, 0] = 0.5 * cs_angstrom  # arcsin(0.5) = pi/6 rad
    _, particles = _convert_and_read(tmp_path, **{"ctf/tilt_A": tilt_a})

    expected_mrad = math.asin(0.5) * 1e3
    assert particles["rlnBeamTiltX"].to_numpy() == pytest.approx(
        expected_mrad, rel=1e-5
    )
    assert particles["rlnBeamTiltY"].to_numpy() == pytest.approx(0.0, abs=1e-9)


def test_ctf_scalefactor_from_alignments_alpha(tmp_path) -> None:
    _, particles = _convert_and_read(tmp_path)

    assert particles["rlnCtfScalefactor"].to_numpy() == pytest.approx(0.75)


def test_optics_block_holds_one_row_for_uniform_particles(tmp_path) -> None:
    optics, particles = _convert_and_read(tmp_path)

    assert len(optics) == 1
    assert optics["rlnVoltage"].iloc[0] == pytest.approx(300.0)
    # Cs is millimetres in both formats -- no conversion.
    assert optics["rlnSphericalAberration"].iloc[0] == pytest.approx(2.7)
    assert optics["rlnAmplitudeContrast"].iloc[0] == pytest.approx(0.1)
    assert optics["rlnImagePixelSize"].iloc[0] == pytest.approx(1.5)
    assert optics["rlnImageSize"].iloc[0] == 128
    assert optics["rlnImageDimensionality"].iloc[0] == 2
    assert list(particles["rlnOpticsGroup"]) == [1, 1, 1]


def test_optics_block_splits_on_differing_pixel_size(tmp_path) -> None:
    psize = np.array([1.5, 1.5, 2.0], dtype=np.float32)
    optics, particles = _convert_and_read(tmp_path, **{"alignments3D/psize_A": psize})

    assert len(optics) == 2
    assert list(particles["rlnOpticsGroup"]) == [1, 1, 2]


def test_identity_anisomag_writes_no_magnification_columns(tmp_path) -> None:
    optics, _ = _convert_and_read(tmp_path)

    # ctf/anisomag is stored as a deviation from identity, so all-zero
    # means no anisotropy and the columns are omitted entirely.
    assert "rlnMagMat00" not in optics.columns


def test_anisomag_written_as_absolute_matrix(tmp_path) -> None:
    aniso = np.tile(np.array([0.02, 0.01, -0.01, 0.03], dtype=np.float32), (3, 1))
    optics, _ = _convert_and_read(tmp_path, **{"ctf/anisomag": aniso})

    # CryoSPARC stores M - I in Fourier space; RELION stores absolute M,
    # also in Fourier space. So the conversion is + I and nothing else --
    # no inverse, no transpose.
    assert optics["rlnMagMat00"].iloc[0] == pytest.approx(1.02)
    assert optics["rlnMagMat01"].iloc[0] == pytest.approx(0.01)
    assert optics["rlnMagMat10"].iloc[0] == pytest.approx(-0.01)
    assert optics["rlnMagMat11"].iloc[0] == pytest.approx(1.03)


def test_anisomag_is_not_folded_into_origins(tmp_path) -> None:
    aniso = np.tile(np.array([0.02, 0.01, -0.01, 0.03], dtype=np.float32), (3, 1))
    _, particles = _convert_and_read(tmp_path, **{"ctf/anisomag": aniso})

    # The matrix travels in the optics block, so applying it to the shifts
    # here as well would make RELION count it twice.
    assert particles["rlnOriginXAngst"].to_numpy() == pytest.approx(3.0)
    assert particles["rlnOriginYAngst"].to_numpy() == pytest.approx(-3.0)


def test_warns_when_dropping_higher_order_aberrations(tmp_path) -> None:
    trefoil = np.full((3, 2), 1e-3, dtype=np.float32)
    out = tmp_path / "out.star"
    import specter.io._convert as mod

    mod.Dataset = fake_dataset(**{"ctf/trefoil_A": trefoil})

    with pytest.warns(UserWarning, match="trefoil"):
        convert_csfile_to_starfile("fake.cs", str(out))


def test_rotations_round_trip_through_the_star_reader(tmp_path) -> None:
    from specter.io import extract_parameters_from_starfile

    out = tmp_path / "out.star"
    convert_csfile_to_starfile("fake.cs", str(out))
    (_, _, _, rotations, translations, ctf_params, scale, _, _, split) = (
        extract_parameters_from_starfile(str(out), halfset="all")
    )

    expected_R = roma.rotvec_to_rotmat(torch.tensor([[0.1, 0.2, 0.3]] * 3))
    assert torch.allclose(roma.unitquat_to_rotmat(rotations), expected_R, atol=1e-4)
    assert torch.allclose(translations, torch.tensor([[3.0, -3.0]] * 3), atol=1e-3)
    assert torch.allclose(ctf_params["dfu"], torch.full((3,), 10000.0), atol=1e-2)
    assert torch.allclose(ctf_params["dfang"], torch.full((3,), 45.0), atol=1e-3)
    assert torch.allclose(
        ctf_params["phaseshift"], torch.full((3,), math.pi / 2), atol=1e-5
    )
    assert torch.allclose(scale, torch.full((3,), 0.75), atol=1e-5)
    assert torch.equal(split, torch.tensor([1, 2, 1]))


def test_cli_converts_a_csfile(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from specter.cli._cli import cli

    monkeypatch.setattr(_convert, "Dataset", fake_dataset())
    out = tmp_path / "cli.star"
    src = tmp_path / "fake.cs"
    src.touch()

    result = CliRunner().invoke(cli, ["convert", "cs2star", str(src), str(out)])

    assert result.exit_code == 0, result.output
    assert set(starfile.read(str(out))) == {"optics", "particles"}


def test_cli_refuses_to_clobber_an_existing_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from specter.cli._cli import cli

    monkeypatch.setattr(_convert, "Dataset", fake_dataset())
    out = tmp_path / "cli.star"
    out.write_text("do not lose me")
    src = tmp_path / "fake.cs"
    src.touch()

    result = CliRunner().invoke(cli, ["convert", "cs2star", str(src), str(out)])

    assert result.exit_code != 0
    assert "--overwrite" in result.output
    assert out.read_text() == "do not lose me"


def test_cli_overwrite_flag_replaces_an_existing_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from specter.cli._cli import cli

    monkeypatch.setattr(_convert, "Dataset", fake_dataset())
    out = tmp_path / "cli.star"
    out.write_text("replace me")
    src = tmp_path / "fake.cs"
    src.touch()

    result = CliRunner().invoke(
        cli, ["convert", "cs2star", str(src), str(out), "--overwrite"]
    )

    assert result.exit_code == 0, result.output
    assert set(starfile.read(str(out))) == {"optics", "particles"}


def test_cli_rejects_a_missing_input_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from specter.cli._cli import cli

    monkeypatch.setattr(_convert, "Dataset", fake_dataset())

    result = CliRunner().invoke(
        cli,
        ["convert", "cs2star", str(tmp_path / "nope.cs"), str(tmp_path / "o.star")],
    )

    assert result.exit_code != 0
    assert not (tmp_path / "o.star").exists()


def _split_datasets(shuffle_passthrough: bool = False) -> type:
    """
    A restack job's pair of files: the particles .cs carries only the image
    address, the passthrough .cs carries pose and CTF. Keyed by path so one
    patched ``Dataset`` serves both loads.
    """
    full = fake_dataset()().load("x")
    blob_keys = {"uid", "blob/path", "blob/idx", "blob/shape"}
    main = {k: v for k, v in full.items() if k in blob_keys}
    passthrough = {k: v for k, v in full.items() if k not in blob_keys - {"uid"}}

    if shuffle_passthrough:
        order = np.array([2, 0, 1])
        passthrough = {k: np.asarray(v)[order] for k, v in passthrough.items()}
        # Make a per-particle value that distinguishes the rows, so a join
        # that silently fell back to row order would be caught.
        main["uid"] = np.array([0, 1, 2], dtype=np.uint64)

    class FakeDataset(dict):
        @classmethod
        def load(cls, csfile_path: str) -> "FakeDataset":
            # Matched on the basename: pytest's tmp_path is named after the
            # test, so "passthrough" appears in the directory of both files.
            return cls(
                passthrough if "passthrough" in os.path.basename(csfile_path) else main
            )

    return FakeDataset


def test_passthrough_file_supplies_pose_and_ctf(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_convert, "Dataset", _split_datasets())
    out = tmp_path / "out.star"

    convert_csfile_to_starfile(
        "restacked.cs", str(out), passthrough_path="J1_passthrough.cs"
    )

    data = starfile.read(str(out))
    assert len(data["particles"]) == 3
    assert data["particles"]["rlnDefocusU"].iloc[0] == pytest.approx(10000.0)
    assert data["particles"]["rlnImageName"].iloc[0] == "000001@J1/particles.mrcs"


def test_passthrough_is_joined_on_uid_not_row_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_convert, "Dataset", _split_datasets(shuffle_passthrough=True))
    out = tmp_path / "out.star"

    convert_csfile_to_starfile(
        "restacked.cs", str(out), passthrough_path="J1_passthrough.cs"
    )

    # uid order in the passthrough is [2, 0, 1]; rlnRandomSubset must still
    # follow the particles file's uid order, i.e. split [0, 1, 0] -> [1, 2, 1].
    particles = starfile.read(str(out))["particles"]
    assert list(particles["rlnRandomSubset"]) == [1, 2, 1]


def test_passthrough_with_mismatched_uids_is_rejected(tmp_path, monkeypatch) -> None:
    cls = _split_datasets()

    class Mismatched(dict):
        @classmethod
        def load(cls_, csfile_path: str) -> "Mismatched":
            d = dict(cls.load(csfile_path))
            if "passthrough" in os.path.basename(csfile_path):
                d["uid"] = np.array([7, 8, 9], dtype=np.uint64)
            return cls_(d)

    monkeypatch.setattr(_convert, "Dataset", Mismatched)

    with pytest.raises(KeyError, match="uid"):
        convert_csfile_to_starfile(
            "restacked.cs",
            str(tmp_path / "o.star"),
            passthrough_path="p_passthrough.cs",
        )


def test_image_prefix_is_prepended_to_the_stack_path(tmp_path) -> None:
    out = tmp_path / "out.star"
    convert_csfile_to_starfile("fake.cs", str(out), image_prefix="/data/CS-tutorial")

    particles = starfile.read(str(out))["particles"]
    assert (
        particles["rlnImageName"].iloc[0]
        == "000001@/data/CS-tutorial/J1/particles.mrcs"
    )


def test_cli_accepts_passthrough_and_image_prefix(tmp_path, monkeypatch) -> None:
    from click.testing import CliRunner

    from specter.cli._cli import cli

    monkeypatch.setattr(_convert, "Dataset", _split_datasets())
    src = tmp_path / "restacked.cs"
    src.touch()
    pt = tmp_path / "J1_passthrough.cs"
    pt.touch()
    out = tmp_path / "cli.star"

    result = CliRunner().invoke(
        cli,
        [
            "convert",
            "cs2star",
            str(src),
            str(out),
            "--passthrough",
            str(pt),
            "--image-prefix",
            "/data/CS-tutorial",
        ],
    )

    assert result.exit_code == 0, result.output
    particles = starfile.read(str(out))["particles"]
    assert (
        particles["rlnImageName"]
        .iloc[0]
        .endswith("/data/CS-tutorial/J1/particles.mrcs")
    )
