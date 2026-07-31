"""
Carbon support film and gold fiducial bead generation -- a from-scratch
port of CryoTomoSim (CTS)'s ``gen_carbon.m`` and ``gen_beads.m``. No
dependency on polnet or VTK.

Both generators are homogeneous-bulk-material models -- there is no
atomic/molecular structure to hand ``specter.potential.PotentialBuilder``,
so intensity is computed directly from real bulk mass density -> number
density -> per-voxel intensity, the same dimensional-analysis approach CTS
itself used (Avogadro's number, molar mass, voxel volume). These are
*number densities* (atoms per cubic Angstrom), not scattering potentials in
the same physically-calibrated units ``PotentialBuilder`` produces --
flagged here explicitly (see each class's docstring) since, unlike the
particle/ice pipeline, there is no independent "real physics" component in
Specter to calibrate these against; the resulting relative magnitude vs.
protein/membrane density is a reasonable-but-uncalibrated approximation,
same caveat CTS's own gold/carbon constants carried.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ._cts_membrane import _alpha_complex, _sample_from_simplices

AVOGADRO = 6.02214076e23  # 1/mol

# Bulk mass densities (g/cm^3) and molar masses (g/mol) for the two
# materials CTS models as homogeneous bulk fill.
CARBON_DENSITY_G_CM3 = 2.0
CARBON_MOLAR_MASS = 12.011
GOLD_DENSITY_G_CM3 = 19.3
GOLD_MOLAR_MASS = 196.97


def _number_density_per_a3(density_g_cm3: float, molar_mass: float) -> float:
    """Atoms per cubic Angstrom, from bulk mass density and molar mass."""
    density_g_a3 = density_g_cm3 * 1e-24  # g/cm^3 -> g/Angstrom^3
    return (density_g_a3 / molar_mass) * AVOGADRO


@dataclass
class BeadInstance:
    """A single solid gold fiducial bead.

    Attributes
    ----------
    density : torch.Tensor
        Local density grid, shape (n, n, n), at real gold number density.
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
    ``gen_beads.m``.

    Parameters
    ----------
    v_size : float
        Voxel size, Angstrom.
    """

    def __init__(self, v_size: float):
        self.v_size = v_size
        self.number_density = _number_density_per_a3(
            GOLD_DENSITY_G_CM3, GOLD_MOLAR_MASS
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
        voxel_vol = self.v_size**3
        density = inside.float() * (self.number_density * voxel_vol)
        return BeadInstance(density=density, radius=radius, v_size=self.v_size)


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
    circular hole cut out and alpha-shape-smoothed edges -- at real carbon
    bulk number density. Port of CTS's ``gen_carbon.m``.

    Reuses the alpha-shape/simplex-sampling machinery from
    ``_cts_membrane.py`` for the hole edge's irregular geometry (factored
    out there rather than duplicated here).

    Parameters
    ----------
    v_size : float
        Voxel size, Angstrom.
    seed : int, optional
        Random seed.
    """

    def __init__(self, v_size: float, seed: int | None = None):
        self.v_size = v_size
        self.rng = np.random.default_rng(seed)
        self.number_density = _number_density_per_a3(
            CARBON_DENSITY_G_CM3, CARBON_MOLAR_MASS
        )

    def generate(
        self,
        target_shape: tuple[int, int, int],
        thickness: float = 150.0,
        hole_radius: float = 400.0,
        edge_roughness: float = 30.0,
        n_points: int = 4000,
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
            Default 400.
        edge_roughness : float, optional
            Random radial jitter applied to the hole edge, Angstrom
            (CTS's own ``carbonshape`` applies similar edge noise).
            Default 30.
        n_points : int, optional
            Number of points used to build the film's point cloud before
            rasterization. Default 4000.

        Returns
        -------
        CarbonFilmInstance
        """
        nz, ny, nx = target_shape
        # Scatter points across the XY footprint at the requested Z
        # thickness, in physical (Angstrom) units centered at the volume
        # midpoint.
        x = (self.rng.random(n_points) - 0.5) * nx * self.v_size
        y = (self.rng.random(n_points) - 0.5) * ny * self.v_size
        z = (self.rng.random(n_points) - 0.5) * thickness

        rho = np.sqrt(x**2 + y**2)
        jitter = self.rng.normal(scale=edge_roughness, size=n_points)
        keep = rho > (hole_radius + jitter)
        x, y, z = x[keep], y[keep], z[keep]
        if x.shape[0] < 8:
            raise ValueError(
                "hole_radius too large relative to target_shape/v_size: "
                f"only {x.shape[0]} points survived the hole cut"
            )

        points = np.stack([x, y, z], axis=1)
        # Smooth the hole edge by resampling from an alpha shape of the
        # raw scatter, same edge-regularization idea as gen_carbon.m's use
        # of alphaShape for the film's boundary.
        alpha = max(hole_radius * 0.15, 5 * self.v_size)
        try:
            tetra, _ = _alpha_complex(points, alpha=alpha)
            points = _sample_from_simplices(
                points, tetra, n=min(n_points, 20000), rng=self.rng
            )
        except ValueError:
            # Fall back to the raw (unsmoothed) scatter if the point cloud
            # is too sparse/thin for a valid 3D alpha complex (thin slabs
            # with few Z-points commonly triangulate poorly) -- a plain
            # scattered film is still a physically reasonable fallback,
            # just without the smoothed hole edge.
            pass

        density = torch.zeros(target_shape, dtype=torch.float32)
        idx_z = np.round(points[:, 2] / self.v_size + nz / 2).astype(np.int64)
        idx_y = np.round(points[:, 1] / self.v_size + ny / 2).astype(np.int64)
        idx_x = np.round(points[:, 0] / self.v_size + nx / 2).astype(np.int64)
        valid = (
            (idx_z >= 0)
            & (idx_z < nz)
            & (idx_y >= 0)
            & (idx_y < ny)
            & (idx_x >= 0)
            & (idx_x < nx)
        )
        idx_z, idx_y, idx_x = idx_z[valid], idx_y[valid], idx_x[valid]
        voxel_vol = self.v_size**3
        weight = self.number_density * voxel_vol
        flat = torch.as_tensor(idx_z * ny * nx + idx_y * nx + idx_x, dtype=torch.int64)
        density.view(-1).scatter_add_(
            0, flat, torch.full((flat.shape[0],), weight, dtype=torch.float32)
        )
        return CarbonFilmInstance(density=density)
