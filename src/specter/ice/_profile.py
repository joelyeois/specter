"""
Laterally varying ice thickness.

Without a profile, ice is a slab of uniform thickness that exactly fills the
simulation box: :func:`~specter.arrays.compute_nz` sizes the box from
``ice_thickness``, and :func:`~specter.ice.blend_ice_into_volume` generates ice
at the box's full ``(nz, nxy, nxy)`` shape. Real vitrified films are not
uniform. Ice in a foil hole is a meniscus pinned to the rim [1]_, and a
micrograph taken off-centre in that hole sees a thickness ramp across its
field.

An :class:`IceProfile` replaces the single scalar with two scalar fields over
the micrograph plane,

.. math::

    z_\\mathrm{top}(x, y) = c(x, y) + t(x, y) / 2 \\\\
    z_\\mathrm{bot}(x, y) = c(x, y) - t(x, y) / 2

where ``t`` is the local thickness and ``c`` the position of the slab's
mid-plane. The two fields are kept separate because they produce different,
separable optical effects: a varying ``t`` changes how widely particle depths
(hence per-particle defocus) are spread at a given point in the field, while a
sloping ``c`` shifts the mean defocus across the field, which is the classic
tilted-specimen defocus gradient. A single thickness field would render a
tilted film as a thickness artifact.

Two consequences are worth knowing before enabling a profile.

**The box is sized by the thickest column.** ``nz`` has to hold the deepest
part of the film, and multislice runs one full-plane FFT per slice regardless
of whether that slice contains anything, so a 250-900 A wedge costs what a
uniform 900 A slab costs. Padding is not free; :meth:`IceProfile.required_nz`
therefore adds no headroom beyond the profile's own extent.

**The defocus refers to the ice's scattering centroid.** A nominal defocus is
the defocus a CTF fit of the field reports: that of the plane at the
thickness-weighted mean height of the ice, :meth:`IceProfile.centroid`. The
vacuum a profile leaves in the box does not move it. A level mid-plane, which
every profile without a ``tilt`` has, puts the centroid at the box's midplane,
where :func:`~specter.aberrations.defocus_midplane_shift` already places the
nominal defocus; a tilted, unevenly thick film moves it off that plane.

References
----------
.. [1] Noble, A. J., Dandey, V. P., Wei, H., Brasch, J., Chase, J., Acharya,
   P., et al. (2018). Routine single particle CryoEM sample and grid
   characterization by tomography. *eLife*, 7, e34257.
   https://doi.org/10.7554/eLife.34257
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from ..arrays import compute_nz

IceProfileMode = Literal["flat", "wedge", "meniscus"]


def _canvas_axis(
    n: int, pixel_size: float, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Position, Å, of each voxel of an ``n``-voxel canvas axis.

    Voxel ``k`` is at ``(k - n // 2) * pixel_size``: the box-centred
    convention every placement into the canvas uses, which puts the
    physical origin exactly on voxel ``n // 2``. ``(k - (n - 1) / 2)``, the
    geometric centre, is half a voxel away on an even axis.
    """
    return (torch.arange(n, dtype=torch.float32, device=device) - n // 2) * pixel_size


@dataclass
class IceProfile:
    """
    Lateral ice thickness profile over a micrograph field of view.

    Parameters
    ----------
    mode : {'flat', 'wedge', 'meniscus'}, optional
        Shape of the thickness field. ``'flat'`` is a slab of uniform
        thickness, matching the behaviour of a bare ``ice_thickness``.
        Default ``'flat'``.
    mean_thickness : float, optional
        Thickness in Å. For ``'wedge'`` this is the thickness at the centre of
        the field when ``thickness_range`` is not given; for ``'meniscus'`` it
        is the thickness at the hole centre. Default 500.0.
    thickness_range : tuple of float, optional
        ``'wedge'`` only. ``(t_min, t_max)`` in Å spanning the field's full
        width along ``angle``. Overrides ``mean_thickness`` when given.
    angle : float, optional
        ``'wedge'`` only. Direction of the thickness ramp in degrees,
        measured from the +x axis. Default 0.0.
    hole_radius : float, optional
        ``'meniscus'`` only. Radius of the foil hole in Å. A 1.2 µm hole is
        6000 Å. Default 6000.0.
    rim_thickness : float, optional
        ``'meniscus'`` only. Thickness in Å at the hole rim. Default 1500.0.
    exponent : float, optional
        ``'meniscus'`` only. Radial exponent of the meniscus. 2.0 is a
        parabolic film; higher values keep the centre flatter for longer and
        turn up more sharply near the rim. Default 2.0.
    hole_offset : tuple of float, optional
        ``'meniscus'`` only. Position of the hole's centre in field
        coordinates, in Å. A micrograph is a small patch of a hole, so this is
        what decides whether it looks flat, wedged, or strongly curved: at the
        hole centre a 1.2 µm meniscus varies by only a few nm across a 200 nm
        field. Default (0.0, 0.0).
    tilt : float, optional
        Slope of the slab's mid-plane, in Å of z per Å of lateral distance.
        Moves both surfaces together, leaving thickness unchanged. Default 0.0.
    tilt_angle : float, optional
        Direction of the mid-plane tilt in degrees. Default 0.0.
    softness : float, optional
        Width in Å of the taper applied at each surface by :meth:`window`.
        Default 2.0, about one water layer.

    Notes
    -----
    A ``'flat'`` profile is not bit-identical to passing no profile at all:
    :meth:`window` tapers both faces over ``softness``, which costs roughly
    ``softness`` Å of projected thickness out of ``mean_thickness``. Pass no
    profile, rather than a flat one, to reproduce earlier behaviour exactly.
    """

    mode: IceProfileMode = "flat"
    mean_thickness: float = 500.0
    thickness_range: tuple[float, float] | None = None
    angle: float = 0.0
    hole_radius: float = 6000.0
    rim_thickness: float = 1500.0
    exponent: float = 2.0
    hole_offset: tuple[float, float] = (0.0, 0.0)
    tilt: float = 0.0
    tilt_angle: float = 0.0
    softness: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in ("flat", "wedge", "meniscus"):
            raise ValueError(
                f"Unknown ice profile mode {self.mode!r}. "
                "Choose 'flat', 'wedge', or 'meniscus'."
            )
        if self.mean_thickness <= 0:
            raise ValueError(
                f"mean_thickness must be positive, got {self.mean_thickness}."
            )
        if self.thickness_range is not None:
            if self.mode != "wedge":
                raise ValueError(
                    f"thickness_range applies to mode='wedge', not {self.mode!r}."
                )
            t_min, t_max = self.thickness_range
            if t_min <= 0 or t_max <= 0:
                raise ValueError(
                    f"thickness_range values must be positive, got {self.thickness_range}."
                )
        if self.mode == "meniscus":
            if self.hole_radius <= 0:
                raise ValueError(
                    f"hole_radius must be positive, got {self.hole_radius}."
                )
            if self.rim_thickness <= 0:
                raise ValueError(
                    f"rim_thickness must be positive, got {self.rim_thickness}."
                )
        if self.softness < 0:
            raise ValueError(f"softness must be non-negative, got {self.softness}.")

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------

    def _grid(self, nxy: int, pixel_size: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Centred (x, y) coordinate grids in Å, each of shape (nxy, nxy).

        Pixel ``k`` is at ``(k - nxy // 2) * pixel_size``, the canvas
        convention :func:`~specter.crowding.insert_particles_into_micrograph`
        places particles in, so a field value belongs to the column a
        particle at that position is rendered into.
        """
        axis = _canvas_axis(nxy, pixel_size)
        return axis[None, :].expand(nxy, nxy), axis[:, None].expand(nxy, nxy)

    def thickness(self, nxy: int, pixel_size: float) -> torch.Tensor:
        """
        Local ice thickness.

        Parameters
        ----------
        nxy : int
            Field size in pixels (square).
        pixel_size : float
            Pixel size in Å.

        Returns
        -------
        torch.Tensor
            Thickness field of shape (nxy, nxy), in Å.
        """
        if self.mode == "flat":
            return torch.full((nxy, nxy), float(self.mean_thickness))

        x, y = self._grid(nxy, pixel_size)

        if self.mode == "wedge":
            if self.thickness_range is None:
                return torch.full((nxy, nxy), float(self.mean_thickness))
            theta = math.radians(self.angle)
            proj = x * math.cos(theta) + y * math.sin(theta)
            t_min, t_max = self.thickness_range
            span = proj.max() - proj.min()
            if span == 0:
                return torch.full((nxy, nxy), float(self.mean_thickness))
            return t_min + (t_max - t_min) * (proj - proj.min()) / span

        r = torch.sqrt((x - self.hole_offset[0]) ** 2 + (y - self.hole_offset[1]) ** 2)
        frac = (r / self.hole_radius).clamp(max=1.0)
        return (
            self.mean_thickness
            + (self.rim_thickness - self.mean_thickness) * frac**self.exponent
        )

    def center(self, nxy: int, pixel_size: float) -> torch.Tensor:
        """
        Position of the slab's mid-plane, relative to the volume's centre.

        Parameters
        ----------
        nxy : int
            Field size in pixels (square).
        pixel_size : float
            Pixel size in Å.

        Returns
        -------
        torch.Tensor
            Mid-plane field of shape (nxy, nxy), in Å.
        """
        if self.tilt == 0.0:
            return torch.zeros(nxy, nxy)
        x, y = self._grid(nxy, pixel_size)
        theta = math.radians(self.tilt_angle)
        return self.tilt * (x * math.cos(theta) + y * math.sin(theta))

    def surfaces(
        self, nxy: int, pixel_size: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bottom and top ice surfaces, relative to the volume's centre.

        Parameters
        ----------
        nxy : int
            Field size in pixels (square).
        pixel_size : float
            Pixel size in Å.

        Returns
        -------
        z_bot, z_top : torch.Tensor
            Two fields of shape (nxy, nxy), in Å.
        """
        t = self.thickness(nxy, pixel_size)
        c = self.center(nxy, pixel_size)
        return c - t / 2, c + t / 2

    def surfaces_at(
        self, xy: torch.Tensor, nxy: int, pixel_size: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Local surfaces sampled at arbitrary lateral coordinates.

        The consumer API for particle placement: given candidate positions, it
        returns the slab each one would sit in.

        Parameters
        ----------
        xy : torch.Tensor
            Coordinates of shape (N, 2) in Å, origin at the field's centre --
            the convention
            :func:`~specter.crowding.insert_particles_into_micrograph` uses.
        nxy : int
            Field size in pixels (square).
        pixel_size : float
            Pixel size in Å.

        Returns
        -------
        z_bot, z_top : torch.Tensor
            Two tensors of shape (N,), in Å. Coordinates outside the field are
            clamped to its edge.
        """
        z_bot, z_top = self.surfaces(nxy, pixel_size)
        idx = (
            ((xy.detach().cpu() / pixel_size).round() + nxy // 2)
            .long()
            .clamp(0, nxy - 1)
        )
        rows, cols = idx[:, 1], idx[:, 0]
        return z_bot[rows, cols], z_top[rows, cols]

    # ------------------------------------------------------------------
    # Consumers
    # ------------------------------------------------------------------

    def required_nz(self, nxy: int, pixel_size: float, base_nz: int = 0) -> int:
        """
        Number of Z slices needed to hold the whole profile.

        No headroom is added beyond the profile's own extent, since
        multislice runs a full-plane FFT per slice whether or not the slice
        contains anything. The consequence is that :meth:`window`'s
        taper is clipped at the thickest column's own surfaces, which costs
        about ``softness`` Å of projected thickness there.

        Parameters
        ----------
        nxy : int
            Field size in pixels (square).
        pixel_size : float
            Pixel size in Å.
        base_nz : int, optional
            Depth of the particle template volume in slices. The box is never
            shallower than this, matching
            :func:`~specter.arrays.compute_nz`. Default 0.

        Returns
        -------
        int
            Number of Z slices.
        """
        z_bot, z_top = self.surfaces(nxy, pixel_size)
        extent = float(z_top.max() - z_bot.min())
        return compute_nz(base_nz, extent, pixel_size)

    def centroid(self, nxy: int, pixel_size: float) -> float:
        """
        Height of the ice's scattering centroid above the volume's centre.

        The thickness-weighted mean of the mid-plane: a column of ice scatters
        in proportion to its thickness, so this is the plane whose defocus a
        CTF fit of the whole field reports, and the one a nominal defocus
        refers to. Zero for any profile whose mid-plane is level, whatever its
        thickness does; under a ``tilt`` it is non-zero only when the film is
        also thicker on one side.

        Parameters
        ----------
        nxy : int
            Field size in pixels (square).
        pixel_size : float
            Pixel size in Å.

        Returns
        -------
        float
            Height in Å, positive towards ``+z``.
        """
        t = self.thickness(nxy, pixel_size)
        c = self.center(nxy, pixel_size)
        return float((t * c).sum() / t.sum())

    def window(
        self,
        nz: int,
        nxy: int,
        pixel_size: float,
        z_slice: slice | None = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """
        Soft z-window: 1 inside the ice, 0 outside.

        Multiply a box-filling ice potential by this to confine it to the
        profile. The taper is a pair of error functions of width ``softness``,
        which keeps the surface from being a hard voxel step. It does still
        clip individual water molecules mid-atom rather than removing them
        whole, which is a fidelity limit of working in voxel space.

        Slice ``k`` sits at ``z = (k - nz // 2) * pixel_size`` and column
        ``(j, i)`` at the lateral position :meth:`surfaces` evaluates it at,
        the convention particles are placed in, so a particle kept inside
        its column's slab by :meth:`surfaces_at` is rendered inside the
        ice this window confines.

        Parameters
        ----------
        nz : int
            Number of Z slices in the volume.
        nxy : int
            Field size in pixels (square).
        pixel_size : float
            Voxel size in Å.
        z_slice : slice, optional
            Build only this z sub-range, so a caller can apply the window in
            chunks instead of materialising the whole (nz, nxy, nxy) mask
            alongside the volume it multiplies. Default None (whole range).
        device : torch.device or str, optional
            Device to build on. Default ``'cpu'``.

        Returns
        -------
        torch.Tensor
            Window of shape (n_selected_z, nxy, nxy), values in [0, 1].
        """
        z_bot, z_top = self.surfaces(nxy, pixel_size)
        z_bot, z_top = z_bot.to(device), z_top.to(device)
        z_axis = _canvas_axis(nz, pixel_size, device=device)
        if z_slice is not None:
            z_axis = z_axis[z_slice]
        z = z_axis[:, None, None]
        if self.softness == 0:
            return ((z >= z_bot[None]) & (z <= z_top[None])).to(z.dtype)
        scale = self.softness * math.sqrt(2.0)
        return 0.5 * (
            torch.erf((z - z_bot[None]) / scale) - torch.erf((z - z_top[None]) / scale)
        )
