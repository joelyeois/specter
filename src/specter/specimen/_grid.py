"""
Gold fiducial bead generation -- a from-scratch port of CryoTomoSim (CTS)'s
``gen_beads.m``. No dependency on polnet or VTK.

Generic, self-contained physics (no CTS-specific placement logic). Used by
``specimen.tomogram.TomogramSpecimenGenerator`` (the generator behind
`specter build tomogram`) -- see its own module docstring. Originally
shared with ``specimen.cryotomosim``'s now-deleted CTS-replica generator;
kept as its own module rather than folded into ``tomogram/generator.py``
directly, since nothing about this physics is tomogram-generator-specific.

``BeadGenerator`` is a bulk-material model -- there is no atomic/molecular
structure to hand ``specter.potential.PotentialBuilder``, so intensity has
to be derived from real bulk mass density rather than from a PDB: real
bulk mass density -> number density (Avogadro's number, molar mass) ->
per-atom potential integral, by reusing specter's own atomic-potential
parameterizations (via ``ice._kernels.build_atomic_potential_kernel``) --
i.e. the same physics used everywhere else in specter to turn atoms into
scattering potential, just fed a bulk material's number density instead of
explicit PDB coordinates. This lands in real volts and is independent of
voxel size, and the resulting mean inner potential comes out in the
literature ballpark for gold (~29 V at this bulk density), unlike CTS's
raw atom-count value (dimensionally inconsistent with
``PotentialBuilder``'s V*Å-unit output, and unphysically dependent on
voxel size).

Unlike CTS, the bead is not a cloud of point atoms binned into voxels: it
is built from real fcc gold atoms (randomly oriented, Debye-Waller
jittered) splatted through the atomic potential kernel, the same way
``potential.PotentialBuilder`` renders a protein. That is correct at any
voxel size, whereas binning a Poisson gas carries shot noise a crystalline
solid does not have (~10% of the projected signal at a 5 Å voxel,
~22% at 2 Å) and collapses each atom's whole potential integral
into a single voxel.

The boundary is an irregular quasi-sphere -- a band-limited spherical-
harmonic modulation of the radius, standing in for the multiply-twinned
particles citrate-reduced colloidal gold actually produces. CTS's exact
sphere is an idealisation (not a shape gold forms), and fcc gold's
equilibrium truncated octahedron describes an annealed single crystal
rather than a colloidal fiducial; neither is offered. Every realisation is
volume-matched, so bead-to-bead shape variation never changes the total
signal.

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
import roma
import torch

from ..arrays import soft_voxelize_coordinates
from ..config import ScalarOrRange, parse_scalar_or_range
from ..fft import spatial_convolve3d_same
from ..potential import build_atomic_potential_kernel
from .membrane._field_spherical_harmonics import (
    _interpolate_angular_grid,
    _sample_sh_coefficients,
    _synthesize_angular_grid,
)

AVOGADRO = 6.02214076e23  # 1/mol

# fcc lattice constant for gold, Å. 4 atoms per conventional cell
# gives 4/a**3 = 0.0591 atoms/Å³, matching to 0.2% the number
# density `_number_density_per_a3` derives independently from bulk mass
# density -- two unrelated routes to the same material.
GOLD_FCC_A = 4.0782

# RMS thermal displacement per Cartesian direction, Å, from
# B ~ 0.6 Å² for Au near room temperature (u = sqrt(B/(8 pi^2))).
# A literature-typical value, not a measurement on any particular
# specimen; it sets how much the lattice fringes are smeared.
# A material constant, so it is a constant here rather than a
# `BeadGenerator` argument -- tuning gold's Debye-Waller factor to make an
# image look right is not something the API should invite. Override it
# here if you have a real number for your specimen's temperature.
GOLD_U_RMS = 0.087

# Directions used to normalise and volume-match the shape's
# spherical-harmonic radius field. Deterministic (Fibonacci sphere), so
# the volume match does not add its own sampling noise.
_SHAPE_QUADRATURE_POINTS = 4096

# Band limit and power spectrum of the shape's radius modulation. Deliberately NOT constructor arguments: the roughness model
# is phenomenological, and nothing is known well enough to tune these
# against -- exposing them would only invite fitting an uncalibrated model
# to whatever an image happened to look like. `BeadGenerator.roughness`
# (the amplitude) stays exposed because it is the one property you could
# measure off EM of a real bead prep.
#
# l_max=6 puts the finest lumps at ~30 degrees of arc, i.e. features
# roughly a quarter of the bead across, which is the scale colloidal gold
# irregularity actually shows. spectrum_power=0.0 gives equal power per
# mode; note the membrane backend's own default of 2.0 encodes the
# Helfrich thermal bending spectrum of a lipid bilayer, which has no
# analogue in a metal nanoparticle -- a bead's irregularity comes from
# growth and twinning, not bending modes.
_ROUGHNESS_L_MAX = 6
_ROUGHNESS_SPECTRUM_POWER = 0.0

_FCC_BASIS = torch.tensor(
    [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
)

# Bulk mass density (g/cm^3) and molar mass (g/mol) for gold, CTS's bead
# material.
GOLD_DENSITY_G_CM3 = 19.3
GOLD_MOLAR_MASS = 196.97


def _number_density_per_a3(density_g_cm3: float, molar_mass: float) -> float:
    """Atoms per cubic Å, from bulk mass density and molar mass."""
    density_g_a3 = density_g_cm3 * 1e-24  # g/cm^3 -> g/Å³
    return (density_g_a3 / molar_mass) * AVOGADRO


def _mean_inner_potential(
    voxel_size: float,
    number_density: float,
    atomic_number: int,
    parameterization: str,
    shtyrov_species: str | None = None,
) -> float:
    """
    Mean inner potential (volts) of a homogeneous bulk material.

    Computed as (per-atom potential, volume-integrated) x (number density)
    -- the volume integral of a single atom's real-space potential equals
    its k=0 Fourier component, so this is the same physics
    ``potential.PotentialBuilder`` uses per-atom, just summed to a bulk
    mean instead of kept per-position. Independent of `voxel_size` (verified:
    the integral is stable across voxel size since it's a physical,
    grid-independent quantity), unlike injecting a raw atom count.
    """
    # No species means no species: build_atomic_potential_kernel falls back
    # to per-element Peng at `atomic_number`, the same way PotentialBuilder
    # does for an atom it cannot type. Substituting a default species here
    # would render one element with another's kernel.
    kernel = build_atomic_potential_kernel(
        voxel_size,
        parameterization,
        atomic_number=atomic_number,
        shtyrov_species=shtyrov_species,
    )
    atom_potential_integral = kernel.sum().item() * voxel_size**3  # V*Å³
    return number_density * atom_potential_integral


def _fibonacci_directions(n: int) -> torch.Tensor:
    """`n` near-uniformly spaced unit vectors, shape (n, 3)."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    rho = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    phi = i * np.pi * (1.0 + np.sqrt(5.0))
    return torch.stack([rho * torch.cos(phi), rho * torch.sin(phi), z], dim=-1).float()


class _BeadShape:
    """
    A bead's boundary: a quasi-sphere whose radius is modulated by a
    band-limited spherical-harmonic field, normalised to `roughness` RMS
    and rescaled to preserve volume.

    A phenomenological model of the irregular, multiply-twinned particles
    citrate-reduced colloidal gold actually produces -- the amplitude and
    band limit are free parameters, not measured ones. `contains` is used
    for both the atom fill and the segmentation mask, so the two can never
    disagree; `half_extent` (the bounding half-width) sizes the grid.

    Every realisation is volume-matched to a sphere of the nominal
    `radius`, so total integrated potential -- and hence projected signal
    -- never depends on how lumpy a particular bead came out.
    """

    def __init__(
        self,
        radius: float,
        roughness: float,
        generator: torch.Generator | None = None,
    ):
        self.roughness = roughness
        l_max = _ROUGHNESS_L_MAX

        # Reuse the membrane backend's harmonic machinery rather than
        # rolling a second copy: `_sample_sh_coefficients` normalises the
        # coefficient set so the perturbation's RMS over the sphere is
        # exactly 1 by Parseval (no empirical rescaling needed), and its
        # real-SH combination carries the sqrt(2)*(-1)^m factors an
        # orthonormal basis needs -- drop those and the m=0 modes end up
        # over-weighted relative to m != 0, which makes the "random"
        # lumpiness quietly anisotropic about z.
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
        self._coeffs = _sample_sh_coefficients(
            l_max, _ROUGHNESS_SPECTRUM_POWER, np.random.default_rng(seed)
        )

        # Synthesize once onto a small angular grid and interpolate
        # thereafter: `contains` runs over every candidate atom *and* every
        # voxel centre (millions of points at fine voxel sizes), and
        # evaluating the harmonics per point makes a bead take seconds.
        # The field is band-limited to `l_max`, so its finest angular
        # structure is ~180/l_max degrees and this grid oversamples it
        # ~12x. Same trick, and the same helpers,
        # `generate_membrane_field_spherical_harmonics` uses.
        n_theta = 12 * l_max + 1
        self._table = _synthesize_angular_grid(
            self._coeffs, l_max, n_theta, 2 * n_theta
        )

        # Mean is exactly 0 and RMS exactly 1 (no l=0 term, Parseval
        # normalisation), so the modulation needs no further centring.
        # V = (1/3) * integral R^3 dOmega, so matching the sphere's volume
        # is a single scale factor set by the mean cube of the modulation.
        modulation = 1.0 + roughness * self._field(
            _fibonacci_directions(_SHAPE_QUADRATURE_POINTS)
        )
        self.scale = radius / float(modulation.pow(3).mean()) ** (1.0 / 3.0)
        self.half_extent = self.scale * float(modulation.max())

    def _field(self, points: torch.Tensor) -> torch.Tensor:
        """The harmonic radius modulation along each point's direction."""
        r = points.norm(dim=-1).clamp_min(1e-9)
        theta = torch.acos((points[..., 2] / r).clamp(-1.0, 1.0))
        phi = torch.atan2(points[..., 1], points[..., 0])
        return torch.as_tensor(
            _interpolate_angular_grid(
                self._table, theta.numpy(), phi.numpy(), device="cpu"
            ),
            dtype=points.dtype,
        )

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        limit = self.scale * (1.0 + self.roughness * self._field(points))
        return points.norm(dim=-1) <= limit


@dataclass
class BeadSpec:
    """Gold fiducial beads to place, one population per radius.

    Attributes
    ----------
    radii : list of float
        One bead radius (Å) per requested bead population.
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
        Local density grid, shape (n, n, n), in volts. Stochastically
        filled (see `BeadGenerator`), so individual voxels scatter about
        gold's real mean inner potential rather than equalling it.
    mask : torch.Tensor
        Bool grid, shape (n, n, n), marking voxels whose *centre* lies
        inside the bead's boundary. This is the bead's geometry,
        independent of the stochastic fill -- use it for segmentation/
        instance labels, since thresholding `density` would punch holes
        wherever the fill happened to place no atom. Follows whichever
        `shape` the generator was built with, so it stays consistent with
        a faceted or roughened bead too.
    radius : float
        Bead radius, Å.
    voxel_size : float
        Voxel size, Å.
    """

    density: torch.Tensor
    mask: torch.Tensor
    radius: float
    voxel_size: float


class BeadGenerator:
    """
    Generates gold fiducial beads, descended from CTS's ``gen_beads.m``,
    built from real fcc gold atoms at gold's real mean inner potential
    (volts) -- see module docstring and ``_mean_inner_potential``.

    Atoms sit on an fcc gold lattice with Debye-Waller jitter and are
    splatted through the real atomic potential kernel, exactly as
    `potential.PotentialBuilder` renders a protein. The boundary is an
    irregular quasi-sphere -- see `_BeadShape`.

    Every call to :meth:`generate` is an independent realisation: its own
    crystal orientation (drawn uniformly over SO(3), so lattice fringes
    point somewhere different on every bead) and its own boundary shape.
    Do not reuse one result as a template for several fiducials, or the
    whole population will be one particle stamped repeatedly.

    Reproducibility follows the usual specter route -- set
    ``specter.seed(n)`` -- or pass an explicit ``generator`` to
    :meth:`generate`, which controls the orientation, the lattice jitter
    and the shape alike.

    Parameters
    ----------
    voxel_size : float
        Voxel size, Å.
    parameterization : str, optional
        Atomic-potential parameterization used to compute gold's per-atom
        potential integral: ``'kirkland'`` (default), ``'lobato'`` or
        ``'peng'``. ``'shtyrov'`` resolves to ``'peng'`` -- there is no
        elemental gold in the bundled species tables and a fiducial has no
        bonds to type, so it takes the same per-element fallback a structure
        atom does, applied by `build_atomic_potential_kernel` itself.
    roughness : float or [low, high], optional
        RMS radius modulation, as a fraction of the radius. Default 0.12.
        A ``[low, high]`` pair draws a fresh value uniformly per bead, so
        a population contains both near-round and badly misshapen
        particles the way a real colloidal prep does -- with a single
        number, every bead is a *different* lumpy shape but the *same*
        degree of lumpy. Follows the same scalar-or-range spelling as
        `config.ParticleStackConfig`'s `dose`/`defocus`.

        Below ~0.06 a bead is visually indistinguishable from a sphere
        while still costing the harmonic machinery; 0.12-0.20 is where it
        reads as a genuinely irregular particle; 0.0 gives a clean sphere.
        The only exposed knob on the roughness model, since its amplitude
        is the one property you could plausibly measure off EM of a real
        bead prep -- the band limit and spectrum are fixed at
        `_ROUGHNESS_L_MAX`/`_ROUGHNESS_SPECTRUM_POWER`, see those
        constants for why.

    Attributes
    ----------
    mean_inner_potential : float
        Gold's bulk mean inner potential (volts) -- the mean over the
        bead's interior, not the value any given voxel takes.

    Notes
    -----
    Two earlier options were removed rather than kept as knobs.

    ``fill='gas'`` reproduced CTS's own model: atoms scattered uniformly
    at the bulk number density and binned into voxels. It is a Poisson
    gas, so per-voxel occupancy carries shot noise a crystalline solid
    does not have -- ~10% of the projected signal at a 5 Å voxel
    against ~3% for the lattice, rising to ~22% at 2 Å -- and it
    collapses each atom's whole potential integral into one voxel. There
    was no voxel size at which it was more accurate.

    ``shape='sphere'``/``'wulff'`` offered an exact sphere (CTS's
    idealisation, not a shape gold forms) and fcc gold's equilibrium
    truncated octahedron. The latter is the right model for an *annealed
    single crystal*, but commercial cryo-ET fiducials are citrate-reduced
    colloidal gold: multiply-twinned and irregular. It also gave a
    population no shape variety, only pose variety -- every bead the same
    solid seen from a different angle.
    """

    def __init__(
        self,
        voxel_size: float,
        parameterization: str = "kirkland",
        roughness: ScalarOrRange = 0.12,
    ):
        self.roughness_range = parse_scalar_or_range(roughness)
        if self.roughness_range[0] < 0:
            raise ValueError(f"roughness must be >= 0, got {roughness!r}")
        if self.roughness_range[1] < self.roughness_range[0]:
            raise ValueError(f"roughness range must be [low, high], got {roughness!r}")
        self.voxel_size = voxel_size
        self.number_density = _number_density_per_a3(
            GOLD_DENSITY_G_CM3, GOLD_MOLAR_MASS
        )
        # Per-atom potential integral (V*Å³), resolution-
        # independent: passing number_density=1.0 makes
        # _mean_inner_potential return exactly that integral (same trick
        # as ._carbon).
        self.atom_potential_integral = _mean_inner_potential(
            voxel_size,
            1.0,
            atomic_number=79,
            parameterization=parameterization,
        )
        self.mean_inner_potential = self.number_density * self.atom_potential_integral
        self._kernel = build_atomic_potential_kernel(
            voxel_size, parameterization, atomic_number=79
        )

    def _random_orientation(self, generator: torch.Generator | None) -> torch.Tensor:
        """A uniformly random 3x3 rotation, drawn from `generator`.

        Not `rotations.random_rotation_matrix`: that wraps
        ``roma.random_rotmat``, which takes no generator and so would
        always draw from the global RNG -- an explicit `generator` would
        then control the lattice jitter and the shape but silently *not*
        the crystal orientation. A normalised Gaussian 4-vector is uniform
        on the unit quaternion sphere, hence uniform over SO(3).
        """
        quat = torch.randn(4, generator=generator)
        return roma.unitquat_to_rotmat(quat / quat.norm())

    def _raw_lattice(
        self, half_extent: float, rot: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        """Jittered fcc gold lattice covering the bead, in the lab frame.

        Built in the crystal frame and rotated into the lab frame by
        `rot`, so every bead's lattice -- and therefore its fringe
        direction -- points somewhere different.
        """
        reach = half_extent + GOLD_FCC_A
        # Rotation preserves norm, so the lab-frame ball of radius `reach`
        # is the crystal-frame ball of radius `reach`: the cell range only
        # has to cover that ball, with no sqrt(3) allowance for the
        # circumscribing cube (which would generate ~10x the atoms and
        # throw all of them away).
        m = int(np.ceil(reach / GOLD_FCC_A)) + 1
        r = torch.arange(-m, m + 1, dtype=torch.float32)
        cells = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(
            -1, 3
        )
        atoms = (cells[:, None, :] + _FCC_BASIS[None, :, :]).reshape(-1, 3) * GOLD_FCC_A
        atoms = atoms + torch.randn(atoms.shape, generator=generator) * GOLD_U_RMS
        atoms = atoms @ rot.T  # crystal frame -> lab frame
        return atoms[atoms.norm(dim=-1) <= reach]

    def generate(
        self, radius: float, generator: torch.Generator | None = None
    ) -> BeadInstance:
        """
        Parameters
        ----------
        radius : float
            Nominal bead radius, Å. The irregular boundary is
            volume-matched to a sphere of this radius.
        generator : torch.Generator, optional
            RNG for the crystal orientation, lattice jitter and shape
            realisation. Default None (global RNG, so ``specter.seed``
            controls it).

        Returns
        -------
        BeadInstance
        """
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")

        # A fresh crystal orientation per bead, so no two fiducials show
        # their lattice fringes running the same way.
        rot = self._random_orientation(generator)
        # Fresh roughness per bead when a range was given, so a population
        # mixes near-round and badly misshapen particles.
        lo, hi = self.roughness_range
        roughness = (
            lo
            if hi <= lo
            else lo + (hi - lo) * float(torch.rand(1, generator=generator))
        )
        shape = _BeadShape(radius, roughness, generator)
        raw_lattice = self._raw_lattice(shape.half_extent, rot, generator)

        # Odd `n` puts the bead's centre exactly on voxel `n // 2`, which
        # is the origin convention both `soft_voxelize_coordinates` and
        # `arrays.clip_insert_bounds` use -- so the rendered bead lands
        # where the placement code thinks it does. (CTS instead offset its
        # centre by a further +1.5 voxels, leaving the bead off-centre and
        # clipped on three faces.)
        pad = 1 + self._kernel.shape[0] // 2
        half = int(np.ceil(shape.half_extent / self.voxel_size)) + pad
        n = 2 * half + 1

        idx = torch.arange(n, dtype=torch.float32) - half
        zz, yy, xx = torch.meshgrid(idx, idx, idx, indexing="ij")
        centres = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3) * self.voxel_size
        mask = shape.contains(centres).reshape(n, n, n)

        atoms = raw_lattice[shape.contains(raw_lattice)]
        deltas = soft_voxelize_coordinates(atoms, (n, n, n), self.voxel_size)
        density = spatial_convolve3d_same(deltas[None], self._kernel)[0]

        return BeadInstance(
            density=density, mask=mask, radius=radius, voxel_size=self.voxel_size
        )
