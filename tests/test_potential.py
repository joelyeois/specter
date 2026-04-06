"""
Tests for potential volume construction.
"""

import pytest
import torch

from specter.potential import build_potential_volume_fftconvolve_3d


# A small cluster of 5 carbon atoms (Z=6) spread over ~6 Å.
# Positions are in Å, centered at the origin.
_ATOMIC_NUMBERS = torch.tensor([6, 6, 6, 6, 6], dtype=torch.long)
_COORDS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [-2.5, 0.0, 0.0],
        [0.0, 2.5, 0.0],
        [0.0, 0.0, 2.5],
    ]
)


def _column_potential(dx: float, n: int) -> float:
    """
    Build a potential volume and return the total column potential.

    The potential builder stores V in units of V·Å (projected potential per
    slice), because the multislice formula uses exp(i*sigma*V[z]) without
    any dz factor.  The physically conserved quantity is therefore:

        V.sum() * dx²   [V·Å · Å² = V·Å³]

    i.e. the total projected-potential integrated over all XY columns.
    """
    V, _, _ = build_potential_volume_fftconvolve_3d(
        _ATOMIC_NUMBERS, _COORDS, n_xyz=n, dx=dx, disable_tqdm=True
    )
    return (V.sum() * dx**2).item()


@pytest.mark.parametrize(
    "dx1,n1,dx2,n2",
    [
        # Same physical box (32 Å³), different voxel sizes
        (1.0, 32, 2.0, 16),
        (1.0, 32, 1.5, 22),  # ~33 Å at 1.5 Å/px, close enough
    ],
)
def test_potential_integral_pixel_size_invariance(dx1, n1, dx2, n2):
    """
    Total column potential (V.sum() * dx²) should be approximately the same
    when building from identical atomic coordinates at different pixel sizes,
    as long as the physical box is large enough to contain the particle.

    This invariant follows from the multislice convention: V stores V·Å
    (projected potential per slice), so V.sum() * dx² gives the total
    projected-potential integrated over all XY columns.
    """
    col1 = _column_potential(dx1, n1)
    col2 = _column_potential(dx2, n2)

    # Allow up to 2% relative deviation — mainly due to grid discretisation
    # and minor differences in kernel truncation between pixel sizes.
    assert abs(col1 - col2) / abs(col1) < 0.02, (
        f"Column potential changed by more than 2% between "
        f"dx={dx1} (n={n1}, col_pot={col1:.3f}) and "
        f"dx={dx2} (n={n2}, col_pot={col2:.3f})"
    )
