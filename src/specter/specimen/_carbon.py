"""
Carbon support film generation -- a from-scratch replication of CryoTomoSim
(CTS)'s ``gen_carbon.m``/``carbonshape`` alpha-shape geometry, replacing the
earlier (deleted) analytic-boundary approach that used to live here.

That earlier approach (independent per-point angular jitter on a circle,
linearly interpolated -- see git history for its own module docstring)
looked artificially smooth in practice: flat top/bottom faces (no z
roughness at all -- the noise was confined to the rim), a spatially
constant density (a binary occupancy mask times a scalar mean inner
potential, so no shot noise/granularity), and a rim that was a pure
function of angle, so every z-slice shared the exact same boundary (no
islands, overhangs, or bays -- topologically a perfect disk). It was a
deliberate, reasoned departure from an even earlier point-cloud/alpha-shape
implementation, for real concerns (see that removed docstring): a
fixed-point-count cloud degrades to sparse speckle as `target_shape`/
`voxel_size` grow, and a handful of smooth analytic modes can't reach genuine
per-pixel jaggedness. This implementation restores the alpha-shape
approach but fixes both: point count now scales with a real physical seed
*density* (`_SEED_VOLUME_PER_POINT`, atoms/A^3-like, not a fixed count), so
it doesn't thin out at larger volumes/finer grids, and roughness comes from
displacing real 3D points before tetrahedralizing (correlated at the seed
spacing, genuinely 3D, can pinch off islands and form overhangs) rather
than from independent per-angle noise.

Faithfully ported from a literal, unabridged translation of CTS's MATLAB
source (validated cell-by-cell against github.com/carsonpurnell/
cryotomosim_CTS) in ``dev/gen_carbon_replica.py`` -- see that file's own
module docstring for the full derivation, and ``dev/validate_carbon_mip.py``
for how deposition was calibrated (below). Two things are deliberately
*not* carried over from that from-scratch reference, both by design
decisions specific to production use:

- Hole placement here is a caller-supplied `hole_center` (typically from
  `edge_hole_center`, a deterministic solve for a specific edge_fraction/
  edge_side), not CTS's own semi-random offset formula -- which its own
  MATLAB comment flags as "mostly working, needs better central point
  rng". `edge_hole_center` is strictly better-engineered (controllable,
  reproducible coverage) and predates this module; nothing in this
  rewrite's motivation (rim roughness, deposition physics -- see below)
  concerned hole placement, so it's kept as-is rather than replaced with
  CTS's own weaker mechanism.
- Deposition uses a MIP-calibrated flat weight (`_deposit_splat`), not
  real per-atom scattering physics (`dev/gen_carbon_replica.py`'s
  `use_physics=True` path, ~40x more expensive at equal atom count):
  the film is a support artifact that nothing downstream resolves
  atomically, so the extra cost buys nothing here. `_deposit_splat`'s
  weight is calibrated so the *bulk* result is physically correct anyway
  (matches the real-physics path's mean inner potential (MIP) to within
  measurement noise -- see that function's docstring), just without
  per-atom radial structure.

References
----------
Purnell, C., Heebner, J., Swulius, M. T., Hylton, R., Kabonick, S., Grillo, M.,
Grigoryev, S., Heberle, F., Waxham, M. N., & Swulius, M. T. (2023). Rapid
synthesis of cryo-ET data for training deep learning models. bioRxiv
2023.04.28.538636. https://doi.org/10.1101/2023.04.28.538636
CTS source: https://github.com/carsonpurnell/cryotomosim_CTS
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from ._carbon_delaunay import block_job, circumspheres
from ._grid import _mean_inner_potential, _number_density_per_a3

CARBON_DENSITY_G_CM3 = 2.0
CARBON_MOLAR_MASS = 12.011

# Amorphous/graphitic bulk carbon has no H neighbors to match a specific
# organic shtyrov species against; "C(CCC)" (carbon bonded to three other
# carbons, i.e. sp2-like coordination) is the closest bundled proxy.
CARBON_SHTYROV_SPECIES = "C(CCC)"

# Quantifoil R1.2/1.3 holey carbon grids: 1.2 micron hole diameter, the
# standard for high-resolution single-particle/cryo-ET data collection at
# 50,000x+ magnification (R2/1, R2/2 ~2 micron holes are used instead at
# lower, 30,000-40,000x magnification). Radius = diameter / 2. Source:
# https://www.emsdiasum.com/docs/technical/datasheet/quantifoil ,
# https://www.quantifoil.com/products/quantifoil/quantifoil-circular-holes
QUANTIFOIL_R1_2_HOLE_RADIUS = 6000.0  # Å (0.6 micron)

# --- CTS gen_carbon.m/carbonshape-derived geometry constants ---------------
# Validated faithful port in dev/gen_carbon_replica.py against the original
# CTS source; see that file for the derivation these were carried over
# from verbatim.
_SEED_VOLUME_PER_POINT = 18000.0  # A^3/seed point; CTS's own comment: "just looks nice and is fast, not evaluated or hypothesis-driven"
_ALPHA = 40.0  # alphaShape radius, A
_LATERAL_PAD = 50.0  # A, seed-cloud padding beyond the frame (CTS's own `pad`) -- avoids the rough rim clipping visibly at the volume's own lateral edges

# Placed number density, as a fraction of bulk (rho = 2.0 g/cm^3, 12
# g/mol): full bulk density overshoots the real literature mean inner
# potential (MIP) of amorphous carbon (7.8-9.1 V) because the independent-
# atom model has no bonding correction -- calibrated in
# dev/validate_carbon_mip.py by measuring MIP at several density fractions
# against specter's own real per-atom carbon potential (0.7x bulk ->
# 8.56 +/- 1.4 V).
_PLACED_DENSITY_FRACTION = 0.7

# Below this point count, a single global Delaunay triangulation is faster
# than paying multiprocess pool startup cost; above it, splitting into
# blocks (see `_alpha_shape`) wins substantially (measured ~9x with a real
# CUDA context already held in the parent process, at ~1M points -- see
# dev/gen_carbon_replica.py's own benchmarking). Point count scales with
# film volume, so in practice this threshold separates small (e.g.
# test-scale) fields from real production ones.
_BLOCKED_DELAUNAY_THRESHOLD = 300_000
_BLOCKED_DELAUNAY_BLOCKS = 10
_BLOCKED_DELAUNAY_WORKERS = 32

# Atoms sampled+deposited per batch. `_sample_in_tets`'s per-atom vertex
# gather is the memory bottleneck (~48 bytes/atom); a large film at real
# Quantifoil hole scale can reach hundreds of millions of atoms, which
# would not fit as a single batch (measured: 668M atoms tried to allocate
# 33 GiB in one `_sample_in_tets` call). `_deposit_splat` is cheap by
# comparison (no per-atom local-window tensors, unlike a real per-atom
# potential evaluation), so it shares this same chunk size rather than
# needing its own smaller one.
_SAMPLE_CHUNK = 20_000_000


def _canonical(tets: torch.Tensor) -> torch.Tensor:
    """
    Deterministic tetrahedron order, independent of how they were produced.

    Sorts each tet's vertices, then lexsorts the rows via four stable
    argsorts -- makes the single-shot and blocked `_alpha_shape` paths
    return bit-identical tetrahedra (verified in dev/gen_carbon_replica.py),
    which in turn makes atom sampling (`_sample_in_tets`) bit-identical
    between them for a given seed.
    """
    t, _ = torch.sort(tets, dim=1)
    order = torch.arange(t.shape[0], device=t.device)
    for col in (3, 2, 1, 0):
        order = order[torch.argsort(t[order, col], stable=True)]
    return t[order]


@dataclass
class _AlphaShape:
    """The alpha complex: accepted tetrahedra and their geometry."""

    points: torch.Tensor  # (N, 3)
    tets: torch.Tensor  # (T, 4) int64, circumradius < alpha
    volumes: torch.Tensor  # (T,)

    @property
    def total_volume(self) -> float:
        return float(self.volumes.sum())


def _alpha_shape(points: np.ndarray, alpha: float, device: torch.device) -> _AlphaShape:
    """
    Replicate MATLAB's ``alphaShape(points, alpha)`` solid region.

    Keeps the tetrahedra of the 3D Delaunay triangulation whose circumsphere
    radius is below `alpha` -- the alpha complex restricted to tetrahedra,
    which is exactly MATLAB's solid region.

    Above `_BLOCKED_DELAUNAY_THRESHOLD` points, the domain is split into a
    ``_BLOCKED_DELAUNAY_BLOCKS x _BLOCKED_DELAUNAY_BLOCKS`` grid over x and
    y (not z -- the film is a thin slab), each core padded by an
    `alpha`-wide halo, and each block triangulated independently in a
    separate ``"spawn"``-context worker process (matches
    ``potential.py``/``_parallel_render.py``'s own choice of `spawn` over
    `fork`: fork-after-CUDA-init is unsafe, and by the time this runs the
    host process has very likely already touched CUDA). This is **exact**,
    not an approximation: an accepted tetrahedron has circumradius < alpha,
    so every point that could violate its empty-circumsphere condition lies
    within alpha of its circumcenter and is therefore inside the halo.
    Assigning each tetrahedron to the block containing its circumcenter
    partitions the accepted set with no duplicates and no misses, so the
    union is the global alpha complex -- verified bit-identical to the
    single-shot path in dev/gen_carbon_replica.py.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) point cloud in Å.
    alpha : float
        Alpha radius in Å.
    device : torch.device
        Device for the volume computation.

    Returns
    -------
    _AlphaShape
    """
    if points.shape[0] < _BLOCKED_DELAUNAY_THRESHOLD:
        from scipy.spatial import Delaunay

        simplices = Delaunay(points).simplices
        _, r = circumspheres(points[simplices])
        tets_np = simplices[r < alpha]
    else:
        blocks = _BLOCKED_DELAUNAY_BLOCKS
        xs = np.linspace(points[:, 0].min(), points[:, 0].max(), blocks + 1)
        ys = np.linspace(points[:, 1].min(), points[:, 1].max(), blocks + 1)
        xs[0], ys[0] = -np.inf, -np.inf
        xs[-1], ys[-1] = np.inf, np.inf

        jobs = []
        for i in range(blocks):
            in_x = (points[:, 0] >= xs[i] - alpha) & (points[:, 0] < xs[i + 1] + alpha)
            for j in range(blocks):
                m = (
                    in_x
                    & (points[:, 1] >= ys[j] - alpha)
                    & (points[:, 1] < ys[j + 1] + alpha)
                )
                gidx = np.flatnonzero(m)
                jobs.append(
                    (points[m], gidx, xs[i], xs[i + 1], ys[j], ys[j + 1], alpha)
                )

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=_BLOCKED_DELAUNAY_WORKERS, mp_context=ctx
        ) as ex:
            tets_np = np.concatenate(list(ex.map(block_job, jobs)), axis=0)

    pts = torch.as_tensor(points, dtype=torch.float64, device=device)
    tets = _canonical(
        torch.as_tensor(np.ascontiguousarray(tets_np), dtype=torch.long, device=device)
    )

    v = pts[tets]
    e = v[:, 1:] - v[:, 0][:, None, :]
    volumes = torch.linalg.det(e).abs() / 6.0

    return _AlphaShape(points=pts.float(), tets=tets, volumes=volumes.float())


def _sample_in_tets(
    shape: _AlphaShape, n: int, generator: torch.Generator
) -> torch.Tensor:
    """
    Uniform points inside the alpha shape -- MATLAB's ``randtess(..., 'v')``.

    Chooses a tetrahedron with probability proportional to its volume, then
    maps three uniforms into it with the Rocchini-Cignoni cut-and-fold,
    which is exactly uniform (no rejection).

    Parameters
    ----------
    shape : _AlphaShape
        The solid to sample.
    n : int
        Number of points.
    generator : torch.Generator
        Device RNG.

    Returns
    -------
    torch.Tensor
        (n, 3) coordinates in Å.
    """
    device = shape.volumes.device
    cdf = torch.cumsum(shape.volumes.double(), 0)
    cdf = cdf / cdf[-1]
    u = torch.rand(n, device=device, generator=generator, dtype=torch.float64)
    idx = torch.clamp(torch.searchsorted(cdf, u), max=shape.tets.shape[0] - 1)

    v = shape.points[shape.tets[idx]]  # (n, 4, 3)

    s = torch.rand(n, device=device, generator=generator)
    t = torch.rand(n, device=device, generator=generator)
    u3 = torch.rand(n, device=device, generator=generator)

    # cut'n fold the cube into the tetrahedron
    m = s + t > 1.0
    s = torch.where(m, 1.0 - s, s)
    t = torch.where(m, 1.0 - t, t)

    m2 = t + u3 > 1.0
    m3 = (~m2) & (s + t + u3 > 1.0)

    tmp = u3.clone()
    t_new = torch.where(m2, 1.0 - tmp, t)
    u_new = torch.where(m2, 1.0 - s - t, u3)
    s_new = s

    s_new = torch.where(m3, 1.0 - t - tmp, s_new)
    u_new = torch.where(m3, s + t + tmp - 1.0, u_new)

    s, t, u3 = s_new, t_new, u_new
    a = 1.0 - s - t - u3

    return (
        a[:, None] * v[:, 0]
        + s[:, None] * v[:, 1]
        + t[:, None] * v[:, 2]
        + u3[:, None] * v[:, 3]
    )


def _seed_points(
    target_shape: tuple[int, int, int],
    voxel_size: float,
    thickness: float,
    hole_radius: float,
    hole_center: tuple[float, float],
    edge_roughness: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Jittered seed point cloud for the alpha-shape rim.

    Adapted from CTS's ``carbonshape`` (see `dev/gen_carbon_replica.py` for
    the literal MATLAB port this was derived from) -- differs only in that
    `hole_center` is a caller-supplied argument rather than computed here
    from CTS's own semi-random offset formula (see module docstring).

    Points are seeded uniformly across `target_shape`'s lateral footprint
    (padded by `_LATERAL_PAD` to avoid visible clipping of the rough
    boundary at the frame's own edges) and the film's z-thickness, atoms
    inside the hole radius (measured from `hole_center`) are dropped, and
    every remaining point is displaced by an isotropic 3D jitter vector
    with magnitude drawn uniformly in ``[0, edge_roughness]`` -- large
    relative to the ~26 A mean seed spacing at the default seed density, so
    points shuffle past their neighbours; `_alpha_shape` then turns that
    into a correlated, topologically nontrivial boundary (can pinch off
    islands, form overhangs) rather than a smooth circle.

    Parameters
    ----------
    target_shape : tuple of int
        (nz, ny, nx) grid shape, matching `CarbonFilmGenerator.generate`.
    voxel_size : float
        Voxel size, Å.
    thickness : float
        Film thickness, Å.
    hole_radius : float
        Hole radius, Å.
    hole_center : tuple of float
        (x, y) hole center, Å, relative to the volume's center.
    edge_roughness : float
        Jitter magnitude scale, Å -- see
        `CarbonFilmGenerator.generate`'s docstring.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        (N, 3) jittered points, centered-origin convention (physical
        (0, 0, 0) is the volume's center).
    """
    nz, ny, nx = target_shape
    lx, ly = nx * voxel_size, ny * voxel_size
    filmsize = np.array([lx + 2 * _LATERAL_PAD, ly + 2 * _LATERAL_PAD, thickness])

    n_seed = int(round(np.prod(filmsize) / _SEED_VOLUME_PER_POINT))
    half = filmsize / 2.0
    ps = rng.random((n_seed, 3)) * filmsize - half

    h = np.hypot(ps[:, 0] - hole_center[0], ps[:, 1] - hole_center[1])
    ps = ps[h > hole_radius]

    vec = rng.normal(size=ps.shape)
    mag = rng.random((ps.shape[0], 1)) * edge_roughness
    vec = mag * vec / np.linalg.norm(vec, axis=1, keepdims=True)
    return ps + vec


def _deposit_splat(
    coords: torch.Tensor,
    weight: float,
    target_shape: tuple[int, int, int],
    voxel_size: float,
) -> torch.Tensor:
    """
    Trilinear splat of a MIP-calibrated flat weight (see module docstring).

    Every atom deposits `weight` Volts, split among its 8 neighboring
    voxels by trilinear interpolation (conserves the total per atom; the
    split is cosmetic anti-aliasing, not real radial falloff). Centered
    coordinate convention throughout (physical (0, 0, 0) maps to the grid
    center), matching ``specter.potential``'s convention and the rest of
    this generator.

    `weight` should be ``atom_potential_integral / voxel_size**3`` --
    depositing a single atom's real, physical potential integral (Volts *
    Å³, resolution-independent -- see `_mean_inner_potential`)
    uniformly over one voxel's volume. By the mean-field relation for a
    homogeneous random point-source gas (V0 = density * per-atom potential
    integral), this reproduces the correct bulk mean inner potential (MIP)
    at any placed density -- verified numerically pix-independent in
    dev/gen_carbon_replica.py (8.42-8.44 V at pix=1/10/20 A for the same
    density, matching a real per-atom-physics measurement of 8.56 V to
    within noise).

    Parameters
    ----------
    coords : torch.Tensor
        (N, 3) atom coordinates, Å, centered-origin.
    weight : float
        Per-atom deposited value, Volts.
    target_shape : tuple of int
        (nz, ny, nx) output grid size.
    voxel_size : float
        Voxel size, Å.

    Returns
    -------
    torch.Tensor
        (nz, ny, nx) volume, Volts.
    """
    nz, ny, nx = target_shape
    device = coords.device
    out = torch.zeros(nz * ny * nx, dtype=torch.float32, device=device)

    center = torch.tensor(
        [nx / 2.0, ny / 2.0, nz / 2.0], dtype=coords.dtype, device=device
    )
    g = (
        coords / voxel_size + center - 0.5
    )  # voxel centers at (i + 0.5) * voxel_size - center
    g0 = torch.floor(g)
    f = g - g0
    g0 = g0.long()

    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                ix = g0[:, 0] + dx
                iy = g0[:, 1] + dy
                iz = g0[:, 2] + dz
                w = (
                    (f[:, 0] if dx else 1 - f[:, 0])
                    * (f[:, 1] if dy else 1 - f[:, 1])
                    * (f[:, 2] if dz else 1 - f[:, 2])
                )
                valid = (
                    (ix >= 0)
                    & (ix < nx)
                    & (iy >= 0)
                    & (iy < ny)
                    & (iz >= 0)
                    & (iz < nz)
                )
                flat = (iz * ny + iy) * nx + ix
                out.index_add_(0, flat[valid], (w[valid] * weight).to(torch.float32))

    return out.view(nz, ny, nx)


@dataclass
class CarbonFilmInstance:
    """A carbon support film slab with a circular hole cut out.

    Attributes
    ----------
    density : torch.Tensor
        Potential volume, shape (nz, ny, nx), Volts -- matching the
        requested `target_shape`.
    """

    density: torch.Tensor


class CarbonFilmGenerator:
    """
    Generates a carbon support film -- a roughly planar slab, with a
    circular hole cut out and a genuinely 3D, randomly-roughened rim -- via
    alpha-shape geometry at a MIP-calibrated flat deposition. See module
    docstring for the full derivation and what changed from the earlier
    (deleted) implementation.

    Parameters
    ----------
    voxel_size : float
        Voxel size, Å.
    parameterization : str, optional
        Atomic-potential parameterization used to compute carbon's per-atom
        potential integral: ``'kirkland'`` (default), ``'lobato'``, or
        ``'shtyrov'``.
    seed : int, optional
        Random seed.
    device : torch.device or str, optional
        Compute device for the alpha-shape/sampling/splat steps (the
        Delaunay triangulation itself always runs on CPU -- Qhull has no
        GPU path). Defaults to CUDA when available.
    """

    def __init__(
        self,
        voxel_size: float,
        parameterization: str = "kirkland",
        seed: int | None = None,
        device: torch.device | str | None = None,
    ):
        self.voxel_size = voxel_size
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)

        number_density = _number_density_per_a3(CARBON_DENSITY_G_CM3, CARBON_MOLAR_MASS)
        self.placed_density = number_density * _PLACED_DENSITY_FRACTION
        # Per-atom potential integral, V*A^3 -- resolution-independent (see
        # _mean_inner_potential's own docstring); passing number_density=1.0
        # returns this integral directly rather than a density-scaled MIP.
        self.atom_potential_integral = _mean_inner_potential(
            voxel_size,
            1.0,
            atomic_number=6,
            parameterization=parameterization,
            shtyrov_species=CARBON_SHTYROV_SPECIES,
        )
        self.mean_inner_potential = self.placed_density * self.atom_potential_integral

    def generate(
        self,
        target_shape: tuple[int, int, int],
        thickness: float = 150.0,
        hole_radius: float = QUANTIFOIL_R1_2_HOLE_RADIUS,
        edge_roughness: float = 60.0,
        hole_center: tuple[float, float] = (0.0, 0.0),
    ) -> CarbonFilmInstance:
        """
        Parameters
        ----------
        target_shape : tuple of int
            Output grid shape (nz, ny, nx).
        thickness : float, optional
            Film thickness, Å. Default 150.
        hole_radius : float, optional
            Radius of the circular hole cut through the film, Å.
            Default `QUANTIFOIL_R1_2_HOLE_RADIUS` (6000 A / 0.6 micron,
            half of Quantifoil R1.2/1.3's 1.2 micron hole diameter -- the
            real, fixed manufacturing spec for the standard grid used for
            high-resolution single-particle/cryo-ET collection, not a
            free-tunable simulation parameter -- see that constant's
            comment for the source). This is much larger than a single
            micrograph/tomogram field of view, so a realistic single
            volume usually sits entirely inside a hole, entirely on the
            carbon, or straddles just one hole's edge. Combine with a
            non-zero `hole_center` (below) so the boundary actually falls
            inside the volume instead of nowhere near it -- `edge_hole_center`
            solves for exactly that.
        edge_roughness : float, optional
            Rim jitter magnitude, Å -- each seed point near the hole
            boundary is displaced by an isotropic 3D vector with magnitude
            drawn uniformly in ``[0, edge_roughness]`` before the alpha
            shape is built (see `_seed_points`). Default 60 (CTS's own
            ``carbonshape`` default, ``JITTER_MAX``); the resulting
            boundary's radial standard deviation comes out to roughly a
            third of this (~20 A at the default, measured in
            dev/gen_carbon_replica.py).
        hole_center : tuple of float, optional
            (x, y) offset of the hole's center relative to the volume's
            center, Å. Default (0, 0) -- hole centered on the
            volume, so the whole hole boundary sits inside the frame if
            `hole_radius` is comparable to the FOV. See `edge_hole_center`
            for computing a `hole_center` that instead shows only a thin
            edge strip of carbon along one side of the frame -- the
            realistic case for `hole_radius` at real Quantifoil scale.

        Returns
        -------
        CarbonFilmInstance
        """
        points = _seed_points(
            target_shape,
            self.voxel_size,
            thickness,
            hole_radius,
            hole_center,
            edge_roughness,
            self.rng,
        )
        if points.shape[0] < 4:
            raise ValueError(
                "hole_radius too large relative to target_shape/voxel_size: "
                "the entire film footprint falls inside the cut-out hole"
            )

        shape = _alpha_shape(points, _ALPHA, self.device)
        n_atoms = int(round(self.placed_density * shape.total_volume))
        if n_atoms == 0:
            raise ValueError(
                "no carbon atoms placed -- alpha-shape film volume is zero "
                "relative to the placed density; the seed cloud may be too "
                "sparse for target_shape/voxel_size (increase hole_radius/"
                "edge_fraction, or check for a degenerate target_shape)"
            )

        weight = self.atom_potential_integral / self.voxel_size**3
        density = torch.zeros(target_shape, dtype=torch.float32, device=self.device)
        for start in range(0, n_atoms, _SAMPLE_CHUNK):
            m = min(_SAMPLE_CHUNK, n_atoms - start)
            coords = _sample_in_tets(shape, m, self.gen)
            density += _deposit_splat(coords, weight, target_shape, self.voxel_size)
        return CarbonFilmInstance(density=density)


def edge_hole_center(
    target_shape: tuple[int, int, int],
    voxel_size: float,
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

    Deliberately kept as-is from the earlier implementation this module
    replaces (see module docstring) -- a genuinely better-engineered,
    deterministic mechanism than CTS's own semi-random hole-placement
    formula, and nothing about this rewrite's motivation (rim roughness,
    deposition physics) concerned hole placement.

    Parameters
    ----------
    target_shape : tuple of int
        (nz, ny, nx) grid shape the film will be generated at.
    voxel_size : float
        Voxel size, Å.
    hole_radius : float
        The real, physical hole radius that will be passed to
        ``CarbonFilmGenerator.generate`` alongside the returned center,
        Å -- e.g. ``QUANTIFOIL_R1_2_HOLE_RADIUS``, the real,
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
    half_x, half_y = nx * voxel_size / 2, ny * voxel_size / 2
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


@dataclass
class CarbonFilmSpec:
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
        Film thickness, Å. Default 150.
    hole_radius : float, optional
        Real physical hole radius, Å. Default
        ``QUANTIFOIL_R1_2_HOLE_RADIUS`` (6000 A / 0.6 micron) -- the fixed
        real-product spec for Quantifoil R1.2/1.3, the standard grid for
        high-resolution collection (see that constant's comment for the
        source), deliberately much larger than a typical `target_shape`
        field of view.
    edge_fraction : float or sequence of float, optional
        Fraction (0-1) of the frame that ends up carbon, entering from
        `edge_side`. Either a fixed value, or a ``(low, high)`` range to
        draw uniformly at random each ``generate()`` call (using the same
        seed as everything else here) -- real images don't all happen to
        catch exactly the same amount of a hole's edge. Default
        ``(0.02, 0.05)``. A two-element list is normalised to a tuple:
        TOML has no tuple type, so a ``[[carbon_film]]`` table's
        ``edge_fraction = [0.02, 0.05]`` arrives here as a list, and
        consumers branch on ``isinstance(..., tuple)`` to tell a range
        from a fixed value.
    edge_side : str, optional
        Which frame edge the carbon intrudes from: ``'left'``,
        ``'right'``, ``'top'``, ``'bottom'``, or ``'random'`` (default).
    edge_roughness : float, optional
        Rim jitter magnitude, Å -- see
        ``CarbonFilmGenerator.generate``'s own docstring. Default 60.
    """

    thickness: float = 150.0
    hole_radius: float = QUANTIFOIL_R1_2_HOLE_RADIUS
    edge_fraction: float | tuple[float, float] = (0.02, 0.05)
    edge_side: str = "random"
    edge_roughness: float = 60.0

    def __post_init__(self) -> None:
        if isinstance(self.edge_fraction, (list, tuple)):
            if len(self.edge_fraction) != 2:
                raise ValueError(
                    "CarbonFilmSpec: edge_fraction range must be [low, high], got "
                    f"{self.edge_fraction!r}"
                )
            lo, hi = (float(v) for v in self.edge_fraction)
            if hi < lo:
                raise ValueError(
                    f"CarbonFilmSpec: edge_fraction must be [low, high], got "
                    f"{self.edge_fraction!r}"
                )
            self.edge_fraction = (lo, hi)
