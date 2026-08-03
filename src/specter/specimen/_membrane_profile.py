"""
Physically calibrated bilayer scattering-potential profile.

Builds a small, schematic atomic lipid-bilayer patch and renders it once
through :class:`~specter.potential.PotentialBuilder` to obtain a 1D lookup
``psi(d)``: the laterally-averaged scattering potential as a function of
signed distance ``d`` from the bilayer mid-plane. Downstream membrane
rasterization looks up ``psi(phi(x))`` for every voxel of every generated
membrane instance -- so the atomic-resolution step happens once and is
reused, rather than rendering real lipids everywhere -- and the resulting
density sits on the same physical scale as ``PotentialBuilder``-rendered
protein templates (same atomic form factors), instead of an arbitrary
constant.

The lipid coordinates are a schematic idealized model: per-leaflet atom
z-offsets from the mid-plane (phosphate headgroup peak, glycerol backbone,
acyl chain, terminal methyls) taken from known bilayer structural biology
(e.g. a ~40 A phosphate-to-phosphate spacing for a fluid PC bilayer), with
small per-atom jitter standing in for conformational disorder -- not a
relaxed or MD-equilibrated structure. Good enough to get the profile's
shape and physical length scale right; swap in a real coordinate set later
if higher fidelity is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..atom import atom_number
from ..potential import PotentialBuilder

# Same value as specimen/pdb_packing.py's, specimen/polnet_bridge.py's, and
# specimen/cryotomosim.py's own independent copies of this constant -- each
# specimen generator keeps its own rather than importing, so this module's
# dependencies stay limited to atom/potential.
ATOM_KERNEL_HALF_WIDTH_A = 2.5

# Schematic single-leaflet lipid atom template: (element, z_offset_a, count,
# jitter_scale). z_offset_a is distance from the mid-plane for this leaflet
# (mirrored for the opposite leaflet); jitter_scale multiplies the base
# per-atom jitter for that group. Not a relaxed structure -- see module
# docstring.
#
# The phosphate/choline headgroup is a comparatively rigid group (small
# conformational freedom relative to a floppy hydrocarbon tail), so it gets
# a *tighter* jitter_scale -- real bilayer electron-density profiles show
# the phosphate peak as the tallest, sharpest feature, which only comes
# through here if that cluster is kept spatially tight rather than given
# the same disorder as everything else. The first below-glycerol chain
# carbon level gets a *looser* jitter_scale so it doesn't spatially
# reinforce the glycerol/ester cluster into competing with the phosphate
# peak. Acyl chain carbons are spread across several z-levels rather than a
# few dense clusters, and the terminal methyls get extra jitter_scale --
# both leaflets' chain termini converge near the mid-plane, so
# under-spreading them (or giving them ordinary disorder) creates a
# spurious *central* density peak instead of the real bilayer's "methyl
# trough" (the terminal carbons are the most conformationally disordered
# part of the chain, which is exactly why real bilayers show a dip there,
# not a peak).
_LEAFLET_TEMPLATE: list[tuple[str, float, int, float]] = [
    ("N", 21.5, 1, 0.45),  # choline nitrogen
    ("C", 20.5, 3, 0.45),  # choline methyls
    ("P", 19.5, 1, 0.45),  # phosphate
    ("O", 19.0, 4, 0.45),  # phosphate oxygens
    ("C", 15.0, 3, 1.3),  # glycerol backbone
    ("O", 13.5, 2, 1.3),  # ester oxygens
    ("O", 13.0, 2, 1.3),  # carbonyl oxygens
    ("C", 12.0, 3, 1.6),  # acyl chains
    ("C", 10.0, 4, 1.0),  # acyl chains
    ("C", 8.0, 4, 1.0),  # acyl chains
    ("C", 6.0, 4, 1.0),  # acyl chains
    ("C", 4.0, 4, 1.2),  # acyl chains, distal
    ("C", 2.0, 4, 1.5),  # acyl chains, distal
    ("C", 1.0, 2, 2.5),  # terminal methyls -- high disorder, "methyl trough"
]


def build_reference_lipid_patch(
    n_lipids_per_leaflet: int = 200,
    area_per_lipid_a2: float = 65.0,
    jitter_a: float = 2.5,
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a schematic atomic lipid-bilayer patch.

    Parameters
    ----------
    n_lipids_per_leaflet : int, optional
        Number of lipid copies per leaflet. Default 200 -- this is a
        one-time cost (the resulting profile is cached and reused for every
        subsequently generated membrane instance), and low lipid counts
        (tens) are visibly noisier: the phosphate headgroup peak can come
        out shorter than the glycerol/ester shoulder just from per-leaflet
        sampling noise, even though it is consistently the taller feature
        once averaged over enough lipids.
    area_per_lipid_a2 : float, optional
        Lateral area per lipid, Angstrom^2, used to size the patch. Default
        65.0 (a typical fluid-phase PC value).
    jitter_a : float, optional
        Standard deviation of per-atom positional jitter, Angstrom, standing
        in for conformational/thermal disorder. Default 2.5.
    seed : int, optional
        Random seed. Default None.
    device : str or torch.device, optional
        Device for the returned tensors. Default "cpu".

    Returns
    -------
    atomic_numbers : torch.Tensor
        Shape ``(N,)``.
    coordinates : torch.Tensor
        Shape ``(N, 3)``, ``(x, y, z)`` Angstrom, centered at the bilayer
        mid-plane (``z=0``) and patch center (``x=y=0``).
    """
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    side_a = math.sqrt(n_lipids_per_leaflet * area_per_lipid_a2)

    elements: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for leaflet_sign in (1.0, -1.0):
        for _ in range(n_lipids_per_leaflet):
            lipid_xy = (torch.rand(2, generator=generator) - 0.5) * side_a
            for element, z_offset, count, jitter_scale in _LEAFLET_TEMPLATE:
                jitter = torch.randn((count, 3), generator=generator) * (
                    jitter_a * jitter_scale
                )
                elements.extend([element] * count)
                xs.extend((lipid_xy[0] + jitter[:, 0]).tolist())
                ys.extend((lipid_xy[1] + jitter[:, 1]).tolist())
                zs.extend((leaflet_sign * z_offset + jitter[:, 2]).tolist())

    atomic_numbers = atom_number(elements).to(device)
    coordinates = torch.tensor(
        list(zip(xs, ys, zs)), dtype=torch.float32, device=device
    )
    return atomic_numbers, coordinates


def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Vectorized 1D linear interpolation with flat (clamped) extrapolation."""
    xp = xp.to(device=x.device, dtype=x.dtype)
    fp = fp.to(device=x.device, dtype=x.dtype)
    x_clamped = x.clamp(min=xp[0], max=xp[-1])
    idx = torch.searchsorted(xp, x_clamped.contiguous())
    idx = idx.clamp(1, xp.numel() - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    f0 = fp[idx - 1]
    f1 = fp[idx]
    weight = (x_clamped - x0) / (x1 - x0)
    return f0 + weight * (f1 - f0)


@dataclass
class BilayerProfile:
    """
    1D lookup ``psi(d)``: scattering potential vs. signed distance from the
    bilayer mid-plane.

    Parameters
    ----------
    distance_a : torch.Tensor
        Sorted ascending signed distances, shape ``(N,)``, Angstrom.
    psi : torch.Tensor
        Laterally-averaged potential at each distance, shape ``(N,)``.
    """

    distance_a: torch.Tensor
    psi: torch.Tensor

    def __call__(self, d: torch.Tensor) -> torch.Tensor:
        """
        Interpolate the profile at arbitrary signed distances.

        Parameters
        ----------
        d : torch.Tensor
            Signed distance from the bilayer mid-plane, Angstrom, any shape.

        Returns
        -------
        torch.Tensor
            Interpolated potential, same shape as ``d``. Values beyond the
            table's range are clamped to the nearest tabulated value.
        """
        flat = _interp1d(d.reshape(-1), self.distance_a, self.psi)
        return flat.reshape(d.shape)


def compute_bilayer_profile(
    atomic_numbers: torch.Tensor,
    coordinates: torch.Tensor,
    v_size: float = 2.0,
    parameterization: str = "shtyrov",
    lateral_core_fraction: float = 0.6,
    device: str | torch.device = "cpu",
) -> BilayerProfile:
    """
    Render a lipid patch and average it into a 1D bilayer profile.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        From :func:`build_reference_lipid_patch`.
    coordinates : torch.Tensor
        From :func:`build_reference_lipid_patch`.
    v_size : float, optional
        Voxel size for the one-time atomic render, Angstrom. Default 2.0.
    parameterization : str, optional
        ``PotentialBuilder`` parameterization. Default "shtyrov".
    lateral_core_fraction : float, optional
        Fraction of the patch's lateral (x, y) extent to average over,
        centered, excluding the patch's own edges (which are missing
        neighboring lipids and would otherwise bias the profile low).
        Default 0.6.
    device : str or torch.device, optional
        Device for the render. Default "cpu".

    Returns
    -------
    BilayerProfile
    """
    coordinates = coordinates.to(device)
    extent = (coordinates.max(dim=0).values - coordinates.min(dim=0).values).tolist()
    margin_a = 2 * ATOM_KERNEL_HALF_WIDTH_A
    n_xyz = tuple(
        int(math.ceil((e + 2 * margin_a) / v_size)) // 2 * 2 + 2 for e in extent
    )

    builder = PotentialBuilder(
        n_xyz=n_xyz,
        dx=v_size,
        atomic_numbers=atomic_numbers,
        progressbars=False,
        parameterization=parameterization,
    )
    volume = builder.forward(coordinates, method="analytic")

    nx, ny, nz = n_xyz
    x_idx = torch.arange(nx, device=volume.device) - nx // 2
    y_idx = torch.arange(ny, device=volume.device) - ny // 2
    z_idx = torch.arange(nz, device=volume.device) - nz // 2

    core_half_voxels_x = int(lateral_core_fraction * nx / 2)
    core_half_voxels_y = int(lateral_core_fraction * ny / 2)
    x_mask = x_idx.abs() <= core_half_voxels_x
    y_mask = y_idx.abs() <= core_half_voxels_y

    core = volume[:, y_mask][:, :, x_mask]
    psi = core.mean(dim=(1, 2))
    distance_a = z_idx.to(dtype=psi.dtype) * v_size

    return BilayerProfile(distance_a=distance_a, psi=psi)


__all__ = [
    "build_reference_lipid_patch",
    "compute_bilayer_profile",
    "BilayerProfile",
]
