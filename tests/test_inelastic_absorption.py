"""
Absorption derived from a measured inelastic mean free path.

The central test is that propagation delivers the mean free path the potential
encodes -- not that the code returns the number it was given, but that a plane
wave through a slab of that potential is attenuated by ``exp(-t/Lambda)``. That
is what makes the model checkable against a measurement rather than against
itself.
"""

from __future__ import annotations

import math

import pytest
import torch

from specter.constants import interaction_parameter
from specter.potential import (
    FULL_OCCUPANCY_POTENTIAL_V,
    INELASTIC_MFP_ICE_A,
    INELASTIC_MFP_PROTEIN_A,
    absorption_potential,
    apply_amplitude_contrast,
    inelastic_absorption_potential,
)
from specter.scattering import Scattering

VOLTAGE = 300.0


def test_absorption_potential_matches_vulovic_equation_3() -> None:
    """``V_ab = 1/(2 sigma Lambda)``, and 0.194 V for ice at 300 kV."""
    sigma = interaction_parameter(VOLTAGE)
    v_ab = absorption_potential(INELASTIC_MFP_ICE_A, VOLTAGE)
    assert v_ab == pytest.approx(1.0 / (2.0 * sigma * INELASTIC_MFP_ICE_A))
    assert v_ab == pytest.approx(0.194, abs=5e-4)


def test_absorption_potential_rejects_nonpositive_mfp() -> None:
    """A non-positive mean free path is a caller error, not an infinity."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be positive"):
            absorption_potential(bad, VOLTAGE)


def test_uniform_field_when_no_specimen_mfp_given() -> None:
    """Without a specimen mean free path the field is the solvent's, flat."""
    v = torch.rand(3, 5, 5) * 10.0
    v_ab = inelastic_absorption_potential(v, 1.0, VOLTAGE)
    expected = absorption_potential(INELASTIC_MFP_ICE_A, VOLTAGE)
    assert torch.allclose(v_ab, torch.full_like(v, expected))


def test_two_materials_interpolate_between_their_mean_free_paths() -> None:
    """Empty space gets the solvent's V_ab, full material the specimen's."""
    v = torch.zeros(8, 16, 16)
    v[3:5, 6:10, 6:10] = 10.0 * FULL_OCCUPANCY_POTENTIAL_V  # solidly occupied
    v_ab = inelastic_absorption_potential(
        v, 1.0, VOLTAGE, mfp_specimen_A=INELASTIC_MFP_PROTEIN_A
    )
    solvent = absorption_potential(INELASTIC_MFP_ICE_A, VOLTAGE)
    specimen = absorption_potential(INELASTIC_MFP_PROTEIN_A, VOLTAGE)

    assert float(v_ab[0, 0, 0]) == pytest.approx(solvent, rel=1e-5)
    assert float(v_ab[4, 8, 8]) == pytest.approx(specimen, rel=1e-5)
    # Protein's shorter mean free path means it absorbs more than the water it
    # displaces; every voxel lies between the two.
    assert specimen > solvent
    assert float(v_ab.min()) >= solvent - 1e-6
    assert float(v_ab.max()) <= specimen + 1e-6


def test_field_is_pointwise_and_so_pose_invariant() -> None:
    """
    The field depends on the potential, not on the beam direction.

    A transposed volume must give the transposed field. This is what allows it
    to be built in the molecule frame and rotated with the specimen, unlike a
    model carrying the plasmon's transverse point spread function.
    """
    v = torch.rand(12, 12, 12) * 8.0
    kwargs = dict(
        voxel_size=1.0, voltage_kv=VOLTAGE, mfp_specimen_A=INELASTIC_MFP_PROTEIN_A
    )
    direct = inelastic_absorption_potential(v, **kwargs)
    swapped = inelastic_absorption_potential(v.permute(2, 1, 0), **kwargs)
    assert torch.allclose(direct.permute(2, 1, 0), swapped, atol=1e-6)


@pytest.mark.parametrize("mfp_A", [2000.0, INELASTIC_MFP_ICE_A])
def test_multislice_recovers_the_mean_free_path_put_in(mfp_A: float) -> None:
    """
    A plane wave through the potential is attenuated by ``exp(-t/Lambda)``.

    Fitted against a per-thickness elastic-only control, so power the
    multislice band limit removes cannot be mistaken for absorption. This is
    the test that ties the model to a measured quantity.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, dx = 32, 2.0
    thicknesses = [100, 200, 400, 800]
    scattering = Scattering(
        nxy=n,
        pixel_size=dx,
        voltage=VOLTAGE,
        scattering_model="multislice",
        progressbars=False,
    ).to(device)

    ratios = []
    for thickness in thicknesses:
        nz = int(thickness / dx)
        # Uniform solvent: the mean free path is a bulk property, so the test
        # should not depend on any particular specimen structure.
        v = torch.full((1, nz, n, n), 4.55, device=device)
        v_ab = inelastic_absorption_potential(v, dx, VOLTAGE, mfp_solvent_A=mfp_A)
        with torch.no_grad():
            i_full = float(
                (scattering.multislice(torch.complex(v, v_ab)).abs() ** 2).mean()
            )
            i_elastic = float((scattering.multislice(v).abs() ** 2).mean())
        ratios.append(i_full / i_elastic)

    # thickness = Lambda * (-ln I); fit the slope.
    x = torch.tensor([-math.log(r) for r in ratios], dtype=torch.float64)
    y = torch.tensor(thicknesses, dtype=torch.float64)
    design = torch.stack([x, torch.ones_like(x)], dim=1)
    slope = float(torch.linalg.lstsq(design, y.unsqueeze(1)).solution.flatten()[0])
    assert slope == pytest.approx(mfp_A, rel=0.01)


def test_transmitted_fraction_is_physically_available() -> None:
    """
    Ice must not remove more of the beam than its cross section allows.

    Through 1200 A of ice, inelastic scattering removes 26%. The fitted
    ``alpha = 0.1`` removes 51% -- more than inelastic plus *all* elastic
    scattering combined (39%) -- which is the error this model exists to
    avoid. A regression that reintroduces it fails here.
    """
    sigma = interaction_parameter(VOLTAGE)
    thickness = 1200.0

    v_ab = absorption_potential(INELASTIC_MFP_ICE_A, VOLTAGE)
    physical = math.exp(-2.0 * sigma * v_ab * thickness)
    assert physical == pytest.approx(math.exp(-thickness / INELASTIC_MFP_ICE_A))
    assert 1.0 - physical == pytest.approx(0.262, abs=0.005)

    # alpha applied to ice's mean inner potential, for contrast.
    fitted = math.exp(-2.0 * sigma * 0.100 * 4.55 * thickness)
    assert 1.0 - fitted > 0.39


def test_amplitude_contrast_leaves_a_complex_potential_alone() -> None:
    """
    A potential that already carries absorption is returned unchanged.

    ``IterativeScattering`` calls ``apply_amplitude_contrast`` unconditionally,
    so without this guard a complex potential would be silently rotated in the
    complex plane by ``sqrt(1 - alpha^2) + i alpha``.
    """
    v = torch.rand(4, 4, 4)
    v_ab = inelastic_absorption_potential(v, 1.0, VOLTAGE)
    complex_v = torch.complex(v, v_ab)
    assert apply_amplitude_contrast(complex_v, alpha=0.1) is complex_v
    # The real path is untouched by the guard.
    assert torch.allclose(
        apply_amplitude_contrast(v, alpha=0.1),
        v * ((1 - 0.1**2) ** 0.5 + 1j * 0.1),
    )
