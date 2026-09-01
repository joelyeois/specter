import math

import pandas as pd
import pytest
import roma
import starfile
import numpy as np
import torch

from specter.imagegenerator import ImageGenerator
from specter.io import (
    create_particle_starfile,
    create_particle_starfile_from_model,
    extract_parameters_from_starfile,
)

# Effective coincidence exclusion radius; see tests/test_generators.py::_CR.
_CR = 1.8 / math.sqrt(2 * math.pi)


def test_create_particle_starfile_writes_bfactor_column(tmp_path) -> None:
    n = 3
    ctf_params = {
        "dfu": torch.full((n,), 5000.0),
        "dfv": torch.full((n,), 5000.0),
        "dfang": torch.zeros(n),
        "cs": torch.full((n,), 2.7),
        "phaseshift": torch.zeros(n),
    }
    create_particle_starfile(
        torch.randn(n, 8, 8),
        rotations=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * n),
        translations=torch.zeros(n, 2),
        alpha=0.1,
        output_dir=str(tmp_path),
        voltage=300.0,
        dx=1.5,
        filename="particles",
        ctf_params=ctf_params,
        dose_per_angstrom=2.0,
        coincidence_radius=_CR,
        potential_scale=0.75,
        bfactor=42.0,
    )

    df = starfile.read(tmp_path / "particles.star")
    assert (df["specterBfactor"] == 42.0).all()
    assert (df["specterPotentialScale"] == 0.75).all()
    assert df["specterCoincidenceRadius"].to_numpy() == pytest.approx(_CR)
    assert (df["specterDosePerAngstrom"] == 2.0).all()


def test_create_particle_starfile_from_model_matches_model_params(tmp_path) -> None:
    volume = torch.zeros(16, 16, 16)
    volume[6:10, 6:10, 6:10] = 50.0
    ctf_params = {
        "dfu": torch.tensor([5000.0]),
        "dfv": torch.tensor([5000.0]),
        "dfang": torch.tensor([0.0]),
        "cs": torch.tensor([2.7]),
        "phaseshift": torch.tensor([0.0]),
    }

    torch.manual_seed(0)
    model = ImageGenerator(
        scattering_potential=volume,
        pixel_size=2.0,
        quaternions=torch.tensor([[0.7071, 0.0, 0.7071, 0.0]]),
        translations=torch.tensor([[1.0, -1.0]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model=None,
        scattering_model="multislice",
        alpha=0.1,
        coincidence_radius=_CR,
        potential_scale=0.75,
        bfactor=42.0,
        verbose=False,
        progressbars=False,
    )
    particles = model(torch.tensor([0]))

    create_particle_starfile_from_model(
        particles, model, output_dir=str(tmp_path), filename="particles"
    )

    df = starfile.read(tmp_path / "particles.star")
    assert (df["specterBfactor"] == 42.0).all()
    assert (df["specterPotentialScale"] == 0.75).all()
    assert df["specterCoincidenceRadius"].to_numpy() == pytest.approx(_CR)
    assert (df["specterDosePerAngstrom"] == 2.0).all()
    # Multislice internally shifts dfu/dfv to the volume's midplane (see
    # _apply_defocus_shift); the saved STAR file must have the original,
    # externally-meaningful defocus, not the internally-shifted one.
    assert (df["rlnDefocusU"] == 5000.0).all()
    assert (df["rlnImagePixelSize"] == 2.0).all()


def _single_block_star_df() -> pd.DataFrame:
    n = 4
    return pd.DataFrame(
        {
            "rlnVoltage": [300.0] * n,
            "rlnSphericalAberration": [2.7] * n,
            "rlnAmplitudeContrast": [0.1] * n,
            "rlnImagePixelSize": [1.5] * n,
            "rlnDefocusU": [10000.0, 10100.0, 10200.0, 10300.0],
            "rlnDefocusV": [9900.0, 9950.0, 10000.0, 10050.0],
            "rlnDefocusAngle": [0.0, 10.0, 20.0, 30.0],
            "rlnPhaseShift": [0.0, 90.0, 180.0, 0.0],
            "rlnAngleRot": [0.0, 10.0, 20.0, 30.0],
            "rlnAngleTilt": [0.0, 15.0, 30.0, 45.0],
            "rlnAnglePsi": [0.0, 5.0, 10.0, 15.0],
            "rlnOriginXAngst": [0.0, 1.0, -2.0, 3.0],
            "rlnOriginYAngst": [0.0, -1.0, 2.0, -3.0],
            "rlnRandomSubset": [1, 2, 1, 2],
        }
    )


def test_extract_parameters_from_starfile_single_block(tmp_path) -> None:
    df = _single_block_star_df()
    star_path = tmp_path / "particles.star"
    starfile.write(df, star_path, overwrite=True)

    (
        voltage_kv,
        pixel_size,
        alpha,
        rotations,
        translations_A,
        ctf_params,
        scale,
        anisomag,
        indices,
        halfset_labels,
    ) = extract_parameters_from_starfile(str(star_path))

    assert voltage_kv.item() == pytest.approx(300.0)
    assert pixel_size.item() == pytest.approx(1.5)
    assert alpha.item() == pytest.approx(0.1)
    assert ctf_params["cs"].shape == (4,)
    assert ctf_params["cs"][0].item() == pytest.approx(2.7 * 1e7)
    assert torch.allclose(
        ctf_params["dfu"],
        torch.tensor(df["rlnDefocusU"].to_numpy(), dtype=torch.float32),
    )
    assert torch.allclose(
        ctf_params["dfv"],
        torch.tensor(df["rlnDefocusV"].to_numpy(), dtype=torch.float32),
    )
    # rlnPhaseShift is in degrees; specter's internal convention is radians
    expected_phaseshift = (
        torch.tensor(df["rlnPhaseShift"].to_numpy(), dtype=torch.float32)
        * torch.pi
        / 180
    )
    assert torch.allclose(ctf_params["phaseshift"], expected_phaseshift, atol=1e-5)

    expected_translations = torch.tensor(
        df[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(), dtype=torch.float32
    )
    assert torch.allclose(translations_A, expected_translations)

    euler_deg = torch.tensor(
        df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(),
        dtype=torch.float32,
    )
    expected_rotations = roma.euler_to_unitquat("ZYZ", euler_deg, degrees=True)
    assert torch.allclose(rotations, expected_rotations, atol=1e-5)

    assert torch.equal(scale, torch.ones(4))
    assert anisomag is None
    assert torch.equal(indices, torch.arange(4))
    assert torch.equal(halfset_labels, torch.tensor([1, 2, 1, 2]))


def test_extract_parameters_from_starfile_halfset_filters_particles(
    tmp_path,
) -> None:
    df = _single_block_star_df()
    star_path = tmp_path / "particles.star"
    starfile.write(df, star_path, overwrite=True)

    (_, _, _, rotations, _, ctf_params, _, _, indices, halfset_labels) = (
        extract_parameters_from_starfile(str(star_path), halfset="B")
    )

    assert rotations.shape == (2, 4)
    assert ctf_params["cs"].shape == (2,)
    assert torch.equal(indices, torch.tensor([1, 3]))
    assert halfset_labels is None


def test_extract_parameters_from_starfile_n_particles_truncates(tmp_path) -> None:
    df = _single_block_star_df()
    star_path = tmp_path / "particles.star"
    starfile.write(df, star_path, overwrite=True)

    (_, _, _, rotations, _, ctf_params, _, _, indices, _) = (
        extract_parameters_from_starfile(str(star_path), n_particles=2)
    )

    assert rotations.shape == (2, 4)
    assert ctf_params["cs"].shape == (2,)
    assert torch.equal(indices, torch.tensor([0, 1]))


def test_extract_parameters_from_starfile_missing_column_raises(tmp_path) -> None:
    df = _single_block_star_df().drop(columns=["rlnDefocusU"])
    star_path = tmp_path / "particles.star"
    starfile.write(df, star_path, overwrite=True)

    with pytest.raises(KeyError):
        extract_parameters_from_starfile(str(star_path))


def test_extract_parameters_from_starfile_missing_dfv_falls_back_to_dfu(
    tmp_path,
) -> None:
    df = _single_block_star_df().drop(columns=["rlnDefocusV"])
    star_path = tmp_path / "particles.star"
    starfile.write(df, star_path, overwrite=True)

    (_, _, _, _, _, ctf_params, _, _, _, _) = extract_parameters_from_starfile(
        str(star_path)
    )

    assert torch.equal(ctf_params["dfu"], ctf_params["dfv"])


def test_extract_parameters_from_starfile_two_block_optics(tmp_path) -> None:
    n = 3
    optics = pd.DataFrame(
        {
            "rlnOpticsGroup": [1],
            "rlnVoltage": [300.0],
            "rlnSphericalAberration": [2.7],
            "rlnAmplitudeContrast": [0.1],
            "rlnImagePixelSize": [1.5],
        }
    )
    particles = pd.DataFrame(
        {
            "rlnOpticsGroup": [1, 1, 1],
            "rlnAngleRot": [0.0, 10.0, 20.0],
            "rlnAngleTilt": [0.0, 15.0, 30.0],
            "rlnAnglePsi": [0.0, 5.0, 10.0],
            "rlnOriginXAngst": [0.0, 1.0, 2.0],
            "rlnOriginYAngst": [0.0, -1.0, -2.0],
            "rlnDefocusU": [10000.0, 10100.0, 10200.0],
            "rlnDefocusV": [9900.0, 9950.0, 10000.0],
            "rlnDefocusAngle": [0.0, 10.0, 20.0],
        }
    )
    star_path = tmp_path / "particles_optics.star"
    starfile.write(
        {"optics": optics, "particles": particles}, star_path, overwrite=True
    )

    (
        voltage_kv,
        pixel_size,
        alpha,
        rotations,
        translations_A,
        ctf_params,
        scale,
        anisomag,
        indices,
        halfset_labels,
    ) = extract_parameters_from_starfile(str(star_path))

    assert voltage_kv.item() == pytest.approx(300.0)
    assert pixel_size.item() == pytest.approx(1.5)
    assert alpha.item() == pytest.approx(0.1)
    assert rotations.shape == (n, 4)
    assert ctf_params["cs"].shape == (n,)
    assert ctf_params["cs"][0].item() == pytest.approx(2.7 * 1e7)
    assert torch.equal(indices, torch.arange(n))
    assert halfset_labels is None


#: The shapes a per-file scalar arrives in. `extract_parameters_from_csfile`
#: returns 0-dim TENSORS, which is what made the repr bug easy to hit, but a
#: caller reading its own metadata just as reasonably has a python int, a
#: numpy scalar, or a length-1 array.
_WRAPS = {
    "float": lambda v: v,
    "int": lambda v: int(v) if float(v).is_integer() else v,
    "np_0d": lambda v: np.array(v),
    "np_scalar": lambda v: np.float64(v),
    "np_len1": lambda v: np.array([v]),
    "torch_0d": lambda v: torch.tensor(v),
    "torch_int": lambda v: torch.tensor(int(v))
    if float(v).is_integer()
    else torch.tensor(v),
}


class TestScalarsAreWrittenAsNumbers:
    """
    A 0-dim tensor must not reach the file as its repr.

    `extract_parameters_from_csfile` returns voltage, pixel size and alpha as
    0-dim tensors, so passing them straight back into a writer is the natural
    thing to do. Written through unconverted they reach `starfile` as objects
    and land as ``tensor(300.)`` instead of ``300.000000`` -- an unreadable
    file, in the three columns that drive the CTF, from code that raised
    nothing.
    """

    @staticmethod
    def _ctf(n):
        return {
            k: torch.full((n,), v)
            for k, v in dict(
                dfu=12000.0,
                dfv=11500.0,
                dfang=30.0,
                cs=2.7e7,
                phaseshift=0.0,
                tiltx=0.0,
                tilty=0.0,
                trefoil1=0.0,
                trefoil2=0.0,
                tetrafoil1=0.0,
                tetrafoil2=0.0,
                tetrafoil3=0.0,
                tetrafoil4=0.0,
            ).items()
        }

    @pytest.mark.parametrize("wrap_name", list(_WRAPS))
    def test_particle_starfile_scalar_columns_are_numeric(self, tmp_path, wrap_name):
        import starfile

        from specter.io import create_particle_starfile

        n = 3
        wrap = _WRAPS[wrap_name]
        create_particle_starfile(
            torch.randn(n, 16, 16),
            rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
            translations=torch.zeros(n, 2),
            ctf_params=self._ctf(n),
            dx=wrap(1.06),
            voltage=wrap(300.0),
            alpha=wrap(0.1),
            filename="p",
            output_dir=str(tmp_path),
        )
        df = starfile.read(str(tmp_path / "p.star"))
        df = df["particles"] if isinstance(df, dict) else df
        for col, want in (
            ("rlnVoltage", 300.0),
            ("rlnImagePixelSize", 1.06),
            ("rlnAmplitudeContrast", 0.1),
            ("rlnSphericalAberration", 2.7),
        ):
            assert len(df[col]) == n
            assert df[col].dtype.kind == "f", f"{col} is {df[col].dtype}, not numeric"
            assert float(df[col].iloc[0]) == pytest.approx(want, rel=1e-5)

    @pytest.mark.parametrize("wrap_name", list(_WRAPS))
    def test_micrograph_starfile_scalar_columns_are_numeric(self, tmp_path, wrap_name):
        import starfile

        from specter.io import create_micrograph_starfile

        n = 2
        wrap = _WRAPS[wrap_name]
        create_micrograph_starfile(
            n,
            voltage=wrap(300.0),
            pixel_size=wrap(1.06),
            alpha=wrap(0.1),
            ctf_params=self._ctf(n),
            filename="m",
            output_dir=str(tmp_path),
        )
        df = starfile.read(str(tmp_path / "m.star"))
        df = df["micrographs"] if isinstance(df, dict) else df
        for col, want in (
            ("rlnVoltage", 300.0),
            ("rlnImagePixelSize", 1.06),
            ("rlnAmplitudeContrast", 0.1),
        ):
            assert df[col].dtype.kind == "f", f"{col} is {df[col].dtype}, not numeric"
            assert float(df[col].iloc[0]) == pytest.approx(want, rel=1e-5)
