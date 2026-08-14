"""
Gold fiducial bead generation -- a from-scratch port of CryoTomoSim (CTS)'s
``gen_beads.m``. No dependency on polnet or VTK.

Generic, self-contained physics (no CTS-specific placement logic). Used by
``specimen.tomogram.MembraneTomogramGenerator`` (the generator behind
`specter build tomogram`) -- see its own module docstring. Originally
shared with ``specimen.cryotomosim``'s now-deleted CTS-replica generator;
kept as its own module rather than folded into ``tomogram/generator.py``
directly, since nothing about this physics is tomogram-generator-specific.

``BeadGenerator`` is a homogeneous-bulk-material model -- there is no
atomic/molecular structure to hand ``specter.potential.PotentialBuilder``,
so intensity has to be derived from real bulk mass density rather than
from explicit atom coordinates: real bulk mass density -> number density
(Avogadro's number, molar mass) -> mean inner potential, by reusing
specter's own atomic-potential parameterizations (via
``ice._kernels.build_atomic_potential_kernel``) to get the material's
per-atom potential integral, then scaling by number density -- i.e. the
same physics used everywhere else in specter to turn atoms into
scattering potential, just applied to a bulk material's *mean* potential
instead of per-atom positions. This lands in real volts and is
independent of voxel size, and comes out in the literature ballpark for
gold's mean inner potential (~25-30 V at this bulk density), unlike a raw
atom-count value (dimensionally inconsistent with ``PotentialBuilder``'s
V*Å-unit output, and unphysically dependent on voxel size).

``_number_density_per_a3``/``_mean_inner_potential`` are also imported by
``._carbon`` (the carbon support film generator), which shares this same
bulk-material approach for its density calibration but is otherwise
unrelated -- kept here since beads needed them first and there's nothing
carbon-specific about the physics itself.

References
----------
Purnell, C., Heebner, J., Swulius, M. T., Hylton, R., Kabonick, S., Grillo, M.,
Grigoryev, S., Heberle, F., Waxham, M. N., & Swulius, M. T. (2023). Rapid
synthesis of cryo-ET data for training deep learning models. bioRxiv
2023.04.28.538636. https://doi.org/10.1101/2023.04.28.538636
CTS source: https://github.com/carsonpurnell/cryotomosim_CTS
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..ice._kernels import build_atomic_potential_kernel

AVOGADRO = 6.02214076e23  # 1/mol

# Bulk mass density (g/cm^3) and molar mass (g/mol) for gold, CTS's bead
# material.
GOLD_DENSITY_G_CM3 = 19.3
GOLD_MOLAR_MASS = 196.97


def _number_density_per_a3(density_g_cm3: float, molar_mass: float) -> float:
    """Atoms per cubic Angstrom, from bulk mass density and molar mass."""
    density_g_a3 = density_g_cm3 * 1e-24  # g/cm^3 -> g/Angstrom^3
    return (density_g_a3 / molar_mass) * AVOGADRO


def _mean_inner_potential(
    v_size: float,
    number_density: float,
    atomic_number: int,
    parameterization: str,
    shtyrov_species: str = "",
) -> float:
    """
    Mean inner potential (volts) of a homogeneous bulk material.

    Computed as (per-atom potential, volume-integrated) x (number density)
    -- the volume integral of a single atom's real-space potential equals
    its k=0 Fourier component, so this is the same physics
    ``potential.PotentialBuilder`` uses per-atom, just summed to a bulk
    mean instead of kept per-position. Independent of `v_size` (verified:
    the integral is stable across voxel size since it's a physical,
    grid-independent quantity), unlike injecting a raw atom count.
    """
    if parameterization == "shtyrov" and not shtyrov_species:
        raise ValueError(
            "shtyrov parameterization requires a bonded species (there is "
            "no unbonded elemental entry in the bundled species table); "
            "use 'kirkland' or 'lobato' instead."
        )
    kernel = build_atomic_potential_kernel(
        v_size,
        parameterization,
        atomic_number=atomic_number,
        shtyrov_species=shtyrov_species or "O(HH)",
    )
    atom_potential_integral = kernel.sum().item() * v_size**3  # V*Angstrom^3
    return number_density * atom_potential_integral


@dataclass
class BeadSpec:
    """Gold fiducial beads to place, one population per radius.

    Attributes
    ----------
    radii : list of float
        One bead radius (Angstrom) per requested bead population.
    count_per_radius : int, optional
        Number of copies to place per radius. Default 1.
    """

    radii: list[float]
    count_per_radius: int = 1


@dataclass
class BeadInstance:
    """A single solid gold fiducial bead.

    Attributes
    ----------
    density : torch.Tensor
        Local density grid, shape (n, n, n), at gold's real mean inner
        potential (volts).
    radius : float
        Bead radius, Angstrom.
    v_size : float
        Voxel size, Angstrom.
    """

    density: torch.Tensor
    radius: float
    v_size: float


class BeadGenerator:
    """
    Generates solid spherical gold fiducial beads, port of CTS's
    ``gen_beads.m``, at gold's real mean inner potential (volts) -- see
    module docstring and ``_mean_inner_potential``.

    Parameters
    ----------
    v_size : float
        Voxel size, Angstrom.
    parameterization : str, optional
        Atomic-potential parameterization used to compute gold's mean
        inner potential: ``'kirkland'`` (default) or ``'lobato'``.
        ``'shtyrov'`` is not supported here (no unbonded elemental gold
        entry in the bundled species table).
    """

    def __init__(self, v_size: float, parameterization: str = "kirkland"):
        self.v_size = v_size
        self.number_density = _number_density_per_a3(
            GOLD_DENSITY_G_CM3, GOLD_MOLAR_MASS
        )
        self.mean_inner_potential = _mean_inner_potential(
            v_size,
            self.number_density,
            atomic_number=79,
            parameterization=parameterization,
        )

    def generate(self, radius: float) -> BeadInstance:
        """
        Parameters
        ----------
        radius : float
            Bead radius, Angstrom.

        Returns
        -------
        BeadInstance
        """
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")
        n = int(np.ceil(2 * radius / self.v_size)) + 2
        n += n % 2
        zz, yy, xx = torch.meshgrid(
            torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij"
        )
        c = n / 2.0
        r2 = ((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2).float() * self.v_size**2
        inside = r2 <= radius**2
        density = inside.float() * self.mean_inner_potential
        return BeadInstance(density=density, radius=radius, v_size=self.v_size)
