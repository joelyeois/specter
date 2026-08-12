"""
Carbon support film and gold fiducial bead generation -- a from-scratch
port of CryoTomoSim (CTS)'s ``gen_carbon.m`` and ``gen_beads.m``. No
dependency on polnet or VTK.

Generic, self-contained physics (no CTS-specific placement logic). Used by
``specimen.tomogram.MembraneTomogramGenerator`` (the generator behind
`specter build tomogram`) -- see its own module docstring. Originally
shared with ``specimen.cryotomosim``'s now-deleted CTS-replica generator;
kept as its own module rather than folded into ``tomogram/generator.py``
directly, since nothing about this physics is tomogram-generator-specific.

Both classes below are homogeneous-bulk-material models -- there is no
atomic/molecular structure to hand ``specter.potential.PotentialBuilder``,
so intensity has to be derived from real bulk mass density rather than
from explicit atom coordinates.

Both do this the same way: real bulk mass density -> number density
(Avogadro's number, molar mass) -> mean inner potential, by reusing
specter's own atomic-potential parameterizations (via
``ice._kernels.build_atomic_potential_kernel``) to get the material's
per-atom potential integral, then scaling by number density -- i.e. the
same physics used everywhere else in specter to turn atoms into
scattering potential, just applied to a bulk material's *mean* potential
instead of per-atom positions (appropriate here since, unlike ice's
amorphous structure, sub-atomic texture isn't being modeled for either
material -- see each class's docstring). This lands in real volts and is
independent of voxel size, and comes out in the literature ballpark for
both materials' mean inner potential (~9-13 V for amorphous carbon,
~25-30 V for gold at these bulk densities), unlike a raw atom-count value
(dimensionally inconsistent with ``PotentialBuilder``'s V*Å-unit output,
and unphysically dependent on voxel size).

``CarbonFilmGenerator``'s hole-edge geometry is also purely analytic (a
random band-limited function of angle), not a resampled point cloud --
see its docstring for why that matters for resolution-independence.

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

# Bulk mass densities (g/cm^3) and molar masses (g/mol) for the two
# materials CTS models as homogeneous bulk fill.
CARBON_DENSITY_G_CM3 = 2.0
CARBON_MOLAR_MASS = 12.011
GOLD_DENSITY_G_CM3 = 19.3
GOLD_MOLAR_MASS = 196.97

# Amorphous/graphitic bulk carbon has no H neighbors to match a specific
# organic shtyrov species against; "C(CCC)" (carbon bonded to three other
# carbons, i.e. sp2-like coordination) is the closest bundled proxy.
CARBON_SHTYROV_SPECIES = "C(CCC)"

# CarbonFilmGenerator.generate's cap on the number of independent boundary
# jitter points (see `edge_grain_size`), to bound memory/compute.
_MAX_EDGE_BINS = 200000

# Quantifoil R1.2/1.3 holey carbon grids: 1.2 micron hole diameter, the
# standard for high-resolution single-particle/cryo-ET data collection at
# 50,000x+ magnification (R2/1, R2/2 ~2 micron holes are used instead at
# lower, 30,000-40,000x magnification). Radius = diameter / 2. Source:
# https://www.emsdiasum.com/docs/technical/datasheet/quantifoil ,
# https://www.quantifoil.com/products/quantifoil/quantifoil-circular-holes
QUANTIFOIL_R1_2_HOLE_RADIUS = 6000.0  # Angstrom (0.6 micron)


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


@dataclass
class GridSpec:
    """
    Carbon support film, forwarded to ``CarbonFilmGenerator.generate``.

    Describes a realistic single-field-of-view carbon film: a big,
    real-scale hole (`hole_radius`) of which only a thin strip
    (`edge_fraction` of the frame) intrudes from one edge (`edge_side`) --
    see ``edge_hole_center``'s docstring for why that, not a small hole
    fully contained in frame, is the realistic case.

    Attributes
    ----------
    thickness : float, optional
        Film thickness, Angstrom. Default 150.
    hole_radius : float, optional
        Real physical hole radius, Angstrom. Default
        ``QUANTIFOIL_R1_2_HOLE_RADIUS`` (6000 A / 0.6 micron) -- the fixed
        real-product spec for Quantifoil R1.2/1.3, the standard grid for
        high-resolution collection (see that constant's comment for the
        source), deliberately much larger than a typical `target_shape`
        field of view.
    edge_fraction : float or tuple of float, optional
        Fraction (0-1) of the frame that ends up carbon, entering from
        `edge_side`. Either a fixed value, or a ``(low, high)`` range to
        draw uniformly at random each ``generate()`` call (using the same
        seed as everything else here) -- real images don't all happen to
        catch exactly the same amount of a hole's edge. Default
        ``(0.02, 0.05)``.
    edge_side : str, optional
        Which frame edge the carbon intrudes from: ``'left'``,
        ``'right'``, ``'top'``, ``'bottom'``, or ``'random'`` (default).
    edge_roughness : float, optional
        Approximate standard deviation of the hole boundary's radial
        perturbation, Angstrom. Default 30.
    edge_grain_size : float or None, optional
        Angstrom arc-length of one independent boundary jitter point (see
        ``CarbonFilmGenerator.generate``'s docstring for why the edge is
        built from independent per-point jitter rather than a smooth
        function). Default None picks ``3 * v_size`` at generate-time.
    """

    thickness: float = 150.0
    hole_radius: float = QUANTIFOIL_R1_2_HOLE_RADIUS
    edge_fraction: float | tuple[float, float] = (0.02, 0.05)
    edge_side: str = "random"
    edge_roughness: float = 30.0
    edge_grain_size: float | None = None


@dataclass
class CarbonFilmInstance:
    """A carbon support film slab with a circular hole cut out.

    Attributes
    ----------
    density : torch.Tensor
        Density grid, shape (nz, ny, nx), matching the requested
        `target_shape`.
    """

    density: torch.Tensor


class CarbonFilmGenerator:
    """
    Generates a carbon support film -- a roughly planar slab, with a
    circular hole cut out and an irregular, randomly-roughened edge -- at
    carbon's real mean inner potential. Port of CTS's ``gen_carbon.m``.

    The film is treated as a spatially uniform bulk material rather than
    resolving individual carbon atoms: at the voxel sizes this is used at,
    atomic-scale texture isn't resolvable anyway, and (unlike ice, where
    ``RandomIcemaker``/``GradientSKIcemaker`` deliberately model amorphous
    structure) there's no physical requirement to reproduce a specific
    carbon structure factor here. So every occupied voxel gets the same
    mean inner potential value, computed from carbon's real bulk number
    density and specter's own atomic-potential parameterizations (see
    ``_mean_inner_potential``) -- real volts, independent of voxel size.

    The hole boundary's irregularity is likewise defined analytically, as
    a seeded random function of angle -- independent per-point radial
    jitter at points spaced around the hole's circumference, linearly
    interpolated in between (see `edge_grain_size` below) -- rather than
    by resampling a fixed-size point cloud through an alpha shape (the
    previous approach here; a point-cloud/alpha-shape wrap would be
    appropriate for a genuinely 3D-irregular blob shape, where the whole
    *shape* is the random part, but is overkill for a hole boundary that's
    a circle plus a 1D perturbation). This matters for two separate
    reasons:

    - Reproducibility across resolutions: a point cloud sampled at a fixed
      count spreads over a `target_shape`-sized footprint, so the same
      `n_points` covers a shrinking *fraction* of a larger volume/grid,
      silently degrading from a solid film into a sparse speckle of
      isolated voxels as `target_shape`/`v_size` change. Classifying every
      voxel by direct evaluation of the boundary function has no such
      dependency: the film is uniformly solid (aside from the deliberate
      hole) at any grid resolution for a fixed seed.
    - Genuine jaggedness: a smooth analytic function (e.g. a low-order sum
      of sine waves) is visually smooth no matter how many terms or how
      they're weighted -- reaching real per-pixel graininess that way
      needs mode counts in the thousands, for no benefit over just
      drawing that many independent points directly. Independent jitter
      with linear (not smooth) interpolation gives real, sharp-cornered
      jaggedness at any target roughness scale, at a fraction of the cost.

    Parameters
    ----------
    v_size : float
        Voxel size, Angstrom.
    parameterization : str, optional
        Atomic-potential parameterization used to compute carbon's mean
        inner potential: ``'kirkland'`` (default), ``'lobato'``, or
        ``'shtyrov'``.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        v_size: float,
        parameterization: str = "kirkland",
        seed: int | None = None,
    ):
        self.v_size = v_size
        self.rng = np.random.default_rng(seed)
        self.number_density = _number_density_per_a3(
            CARBON_DENSITY_G_CM3, CARBON_MOLAR_MASS
        )
        self.mean_inner_potential = _mean_inner_potential(
            v_size,
            self.number_density,
            atomic_number=6,
            parameterization=parameterization,
            shtyrov_species=CARBON_SHTYROV_SPECIES,
        )

    def generate(
        self,
        target_shape: tuple[int, int, int],
        thickness: float = 150.0,
        hole_radius: float = QUANTIFOIL_R1_2_HOLE_RADIUS,
        edge_roughness: float = 30.0,
        edge_grain_size: float | None = None,
        hole_center: tuple[float, float] = (0.0, 0.0),
    ) -> CarbonFilmInstance:
        """
        Parameters
        ----------
        target_shape : tuple of int
            Output grid shape (nz, ny, nx).
        thickness : float, optional
            Film thickness, Angstrom. Default 150.
        hole_radius : float, optional
            Radius of the circular hole cut through the film, Angstrom.
            Default `QUANTIFOIL_R1_2_HOLE_RADIUS` (6000 A / 0.6 micron,
            half of Quantifoil R1.2/1.3's 1.2 micron hole diameter -- the
            real, fixed manufacturing spec for the standard grid used for
            high-resolution single-particle/cryo-ET collection, not a
            free-tunable simulation parameter -- see that constant's
            comment for the source). This is much larger than a single
            micrograph/tomogram field of view, so a realistic single
            volume usually sits entirely inside a hole, entirely on the
            carbon, or straddles just one hole's edge (which looks nearly
            straight/gently curved over a `target_shape`-sized volume, not
            like a small closed circle in frame, because `hole_radius` so
            far exceeds it). Pass a smaller value than the real spec only
            to make the curvature more visible for illustration -- not
            for a production-accurate simulation. Combine with a non-zero
            `hole_center` (below) so the boundary actually falls inside
            the volume instead of nowhere near it.
        edge_roughness : float, optional
            Approximate standard deviation of the random radial
            perturbation applied to the hole boundary, Angstrom (CTS's own
            ``carbonshape`` applies similar edge noise). Default 30.
        edge_grain_size : float or None, optional
            Angstrom arc-length of one independent boundary "grain" --
            the boundary radius is drawn as an i.i.d. Gaussian jitter value
            (std `edge_roughness`) at points spaced `edge_grain_size` apart
            around the hole's full circumference, linearly interpolated in
            between. This is deliberately *not* a smooth analytic function
            (e.g. a low-order Fourier sum): a handful of smooth modes is
            always visually smooth no matter how you weight them, and
            reaching genuine per-pixel graininess that way would need mode
            counts in the thousands to tens of thousands (one per
            resolvable wavelength around a realistic, large `hole_radius`)
            with no accuracy benefit over just drawing that many
            independent points directly, closely matching what CTS's own
            ``carbonshape``/``gen_carbon.m`` actually does (independent
            per-point radial jitter). Piecewise-linear interpolation
            between independent draws gives real jaggedness (sharp slope
            changes at every grain boundary) rather than a smooth wave, at
            the same computational cost.

            Default None picks ``3 * v_size`` -- i.e. grains just above
            voxel-scale, using essentially all the detail the grid can
            resolve. Note this makes the auto default resolution-*dependent*
            (a finer grid gets finer default grains) -- unlike
            `hole_radius`/`hole_center`/`edge_roughness`, which are purely
            physical quantities. That's intentional here (a real torn edge
            has detail at every scale down to whatever you can resolve,
            same reasoning as ice's amorphous texture), but pass an
            explicit `edge_grain_size` for reproducibility across
            `target_shape`/`v_size` changes.
        hole_center : tuple of float, optional
            (x, y) offset of the hole's center relative to the volume's
            center, Angstrom. Default (0, 0) -- hole centered on the
            volume, so the whole hole boundary sits inside the frame if
            `hole_radius` is comparable to the FOV. Combine with a large
            `hole_radius` and an offset a bit smaller than `hole_radius`
            (so the boundary still crosses the volume) to instead show
            only one edge of a much bigger hole, near one side of the
            frame -- the realistic case.

        Returns
        -------
        CarbonFilmInstance
        """
        nz, ny, nx = target_shape
        z = (torch.arange(nz, dtype=torch.float32) - nz / 2) * self.v_size
        y = (torch.arange(ny, dtype=torch.float32) - ny / 2) * self.v_size
        x = (torch.arange(nx, dtype=torch.float32) - nx / 2) * self.v_size
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # (ny, nx)
        yy = yy - hole_center[1]
        xx = xx - hole_center[0]

        theta = torch.atan2(yy, xx)
        rho = torch.sqrt(xx**2 + yy**2)

        if edge_grain_size is None:
            edge_grain_size = 3.0 * self.v_size
        n_bins = int(
            np.clip(round(2 * np.pi * hole_radius / edge_grain_size), 8, _MAX_EDGE_BINS)
        )

        # Independent random radial jitter at n_bins points spaced evenly
        # around the hole's full circumference, linearly interpolated
        # in between -- see `edge_grain_size` docstring for why this (not
        # a smooth analytic function) is used for genuine jaggedness. Still
        # fully determined by physical angle, so it's the same boundary at
        # any target_shape/v_size for a fixed seed (see class docstring),
        # modulo whatever `edge_grain_size` resolves to.
        jitter_bins = torch.as_tensor(
            self.rng.normal(scale=edge_roughness, size=n_bins), dtype=torch.float32
        )
        bin_pos = (theta + np.pi) / (2 * np.pi) * n_bins
        bin_floor = torch.floor(bin_pos)
        frac = bin_pos - bin_floor
        idx0 = bin_floor.long() % n_bins
        idx1 = (idx0 + 1) % n_bins
        jitter = jitter_bins[idx0] * (1 - frac) + jitter_bins[idx1] * frac
        hole_boundary = hole_radius + jitter

        occupied_xy = rho > hole_boundary
        occupied_z = z.abs() <= (thickness / 2)
        occupied = occupied_z[:, None, None] & occupied_xy[None]
        if not occupied.any():
            raise ValueError(
                "hole_radius too large relative to target_shape/v_size: "
                "the entire film footprint falls inside the cut-out hole"
            )

        density = occupied.float() * self.mean_inner_potential
        return CarbonFilmInstance(density=density)


def edge_hole_center(
    target_shape: tuple[int, int, int],
    v_size: float,
    hole_radius: float,
    edge_fraction: float = 0.1,
    side: str = "random",
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Compute a `hole_center` for :meth:`CarbonFilmGenerator.generate` that
    places only a thin strip of carbon -- `edge_fraction` of the XY frame's
    width or height -- along one edge of the volume, with the rest of the
    frame falling inside the (real, much larger) hole.

    This is the realistic single-field-of-view case described in
    :meth:`CarbonFilmGenerator.generate`'s `hole_radius` docstring: a real
    support-film hole is usually far bigger than one micrograph/tomogram
    field of view, so you either see no carbon at all, are entirely on the
    carbon, or catch just one edge of a hole near a border of the frame --
    never a small hole fully contained within it. Getting that by hand
    means solving for where a huge circle's boundary has to sit to land a
    specific-width strip at a specific frame edge; this does that solve.

    Parameters
    ----------
    target_shape : tuple of int
        (nz, ny, nx) grid shape the film will be generated at.
    v_size : float
        Voxel size, Angstrom.
    hole_radius : float
        The real, physical hole radius that will be passed to
        ``CarbonFilmGenerator.generate`` alongside the returned center,
        Angstrom -- e.g. ``QUANTIFOIL_R1_2_HOLE_RADIUS``, the real,
        fixed spec for the standard Quantifoil grid (see that constant's
        comment). Large relative to `target_shape` both for realism and
        so the boundary crossing the frame reads as close to a straight
        edge rather than a visibly curved arc.
    edge_fraction : float, optional
        Fraction (0-1) of the frame's width (`side` ``'left'``/``'right'``)
        or height (``'top'``/``'bottom'``) that ends up carbon. Default
        0.1.
    side : str, optional
        Which frame edge the carbon strip sits along: ``'left'``,
        ``'right'``, ``'top'``, ``'bottom'``, or ``'random'`` (default --
        picks uniformly using `rng`).
    rng : numpy.random.Generator, optional
        Used only when `side="random"`; a fresh default generator is used
        if not given.

    Returns
    -------
    tuple of float
        (x, y) `hole_center`, ready to pass straight through to
        ``CarbonFilmGenerator.generate`` together with `hole_radius`.
    """
    if side == "random":
        side = (rng or np.random.default_rng()).choice(
            ["left", "right", "top", "bottom"]
        )
    nz, ny, nx = target_shape
    half_x, half_y = nx * v_size / 2, ny * v_size / 2
    if side == "right":
        edge = half_x * (1 - 2 * edge_fraction)
        return edge - hole_radius, 0.0
    elif side == "left":
        edge = -half_x * (1 - 2 * edge_fraction)
        return edge + hole_radius, 0.0
    elif side == "top":
        edge = half_y * (1 - 2 * edge_fraction)
        return 0.0, edge - hole_radius
    elif side == "bottom":
        edge = -half_y * (1 - 2 * edge_fraction)
        return 0.0, edge + hole_radius
    raise ValueError(
        f"Unknown side '{side}'; choose 'left', 'right', 'top', 'bottom', or 'random'."
    )
