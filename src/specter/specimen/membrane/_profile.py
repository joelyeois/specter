"""
Physically calibrated bilayer scattering-potential profile.

Builds a small, schematic atomic lipid-bilayer patch and renders it once
through :class:`~specter.potential.PotentialBuilder` to obtain a 1D lookup
``psi(d)``: the scattering potential as a function of signed distance ``d``
from the bilayer mid-plane. Downstream membrane rasterization looks up
``psi(phi(x))`` for every voxel of every generated membrane instance -- so
the atomic-resolution step happens once and is reused, rather than
rendering real lipids everywhere -- and the resulting density sits on the
same physical scale as ``PotentialBuilder``-rendered protein templates
(same atomic form factors), instead of an arbitrary constant.

The lipid coordinates are a schematic idealized model: per-leaflet atom
z-offsets from the mid-plane (phosphate headgroup peak, glycerol backbone,
acyl chain, terminal methyls) taken from known bilayer structural biology
(e.g. a ~40 A phosphate-to-phosphate spacing for a fluid PC bilayer), with
small per-atom jitter standing in for conformational disorder -- not a
relaxed or MD-equilibrated structure. Good enough to get the profile's
shape and physical length scale right; swap in a real coordinate set later
if higher fidelity is needed.

Amplitude calibration is deliberately NOT read off a laterally-averaged
multi-lipid render (an earlier version of ``estimate_bilayer_peak_
amplitude`` did exactly that, and diluted the true peak by ~20x with no
physical justification -- see that function's own docstring for the full
story and the measured numbers). Real membranes aren't laterally
pre-averaged before imaging any more than a protein's atoms are; the
smoothing that makes real cryo-ET membranes look continuous comes from the
microscope's own resolution limits (CTF, multislice, detector MTF),
applied identically to membrane and protein potential alike AFTER the
ground truth is built (and already handled downstream by
``_raster.py``'s anti-aliasing step) -- not baked into the ground truth
itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ...atom import atom_number
from ...potential import PotentialBuilder

# Same value as specimen/packing/algorithms.py's own independent copy of
# this constant -- each specimen generator keeps its own rather than
# importing, so this module's dependencies stay limited to atom/potential.
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
# the same disorder as everything else.
#
# Regression note: an earlier version of the acyl-chain rows below used
# jitter_scale 1.0-1.6 (only slightly looser than the headgroup) across 6
# fixed z-levels 2A apart. That is NOT loose enough for adjacent clusters'
# per-atom Gaussian jitter to actually blend together -- verified directly
# (both on the raw atom z-histogram, before any voxel rendering, and on the
# rendered psi(d) profile): it produced a *second*, nearly phosphate-height
# density hump around +-8A (the acyl-chain region as a whole has ~3-4x more
# atoms than the compact headgroup, spread across only 6 distinct z-values,
# each individually under-blended into its neighbors), instead of a single
# broad, smoothly-declining shoulder -- the opposite of a real bilayer
# electron-density profile's two-peaks-with-a-clearly-weaker-middle shape.
# jitter_scale raised to 2.2-3.0 here (each cluster's std now clearly wider
# than the 2A inter-cluster spacing) fixes this: verified the rendered
# profile no longer has any competing peak in the +-4 to +-14A shoulder
# region (see test_compute_bilayer_profile_no_competing_peak_in_chain_
# region in tests/test_membrane_profile.py). The terminal methyls are
# pushed to the mid-plane itself (0.5A, was 1.0A) with even higher
# jitter_scale (4.5, was 2.5) so both leaflets' chain termini spread widely
# rather than piling up right at z=0 -- real bilayers show a genuine dip
# there (the "methyl trough"), not a peak, precisely because the terminal
# segment is the MOST conformationally disordered part of the chain.
_LEAFLET_TEMPLATE: list[tuple[str, float, int, float]] = [
    ("N", 21.5, 1, 0.45),  # choline nitrogen
    ("C", 20.5, 3, 0.45),  # choline methyls
    ("P", 19.5, 1, 0.45),  # phosphate
    ("O", 19.0, 4, 0.45),  # phosphate oxygens
    ("C", 15.0, 3, 1.3),  # glycerol backbone
    ("O", 13.5, 2, 1.3),  # ester oxygens
    ("O", 13.0, 2, 1.3),  # carbonyl oxygens
    ("C", 12.0, 3, 2.2),  # acyl chains
    ("C", 10.0, 4, 2.2),  # acyl chains
    ("C", 8.0, 4, 2.2),  # acyl chains
    ("C", 6.0, 4, 2.2),  # acyl chains
    ("C", 4.0, 4, 2.6),  # acyl chains, distal
    ("C", 2.0, 4, 3.0),  # acyl chains, distal
    ("C", 0.5, 2, 4.5),  # terminal methyls -- high disorder, "methyl trough"
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
    voxel_size: float = 2.0,
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
    voxel_size : float, optional
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
        int(math.ceil((e + 2 * margin_a) / voxel_size)) // 2 * 2 + 2 for e in extent
    )

    builder = PotentialBuilder(
        n_xyz=n_xyz,
        dx=voxel_size,
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
    distance_a = z_idx.to(dtype=psi.dtype) * voxel_size

    return BilayerProfile(distance_a=distance_a, psi=psi)


def _isolated_atom_peak_potential(
    atomic_number: int,
    voxel_size: float,
    parameterization: str,
    device: str | torch.device,
) -> float:
    """Raw, undiluted peak scattering potential of a single isolated atom
    -- the calibration reference `estimate_bilayer_peak_amplitude` uses.
    Same `PotentialBuilder` atomic-form-factor physics as everything else
    in this module, just with nothing else nearby to average against."""
    n = int(math.ceil(2 * ATOM_KERNEL_HALF_WIDTH_A / voxel_size)) // 2 * 2 + 2
    builder = PotentialBuilder(
        n_xyz=(n, n, n),
        dx=voxel_size,
        atomic_numbers=torch.tensor([atomic_number], device=device),
        progressbars=False,
        parameterization=parameterization,
    )
    volume = builder.forward(torch.zeros((1, 3), device=device), method="analytic")
    return float(volume.max())


def estimate_bilayer_peak_amplitude(
    atomic_numbers: torch.Tensor,
    coordinates: torch.Tensor,
    voxel_size: float = 2.0,
    parameterization: str = "shtyrov",
    device: str | torch.device = "cpu",
) -> float:
    """
    Peak scattering potential to calibrate
    :func:`build_analytic_bilayer_profile`'s `amplitude` against real
    atomic scattering physics -- the same units as ``PotentialBuilder``-
    rendered protein templates, so a membrane and an inserted
    transmembrane protein sit on a physically consistent scale in the same
    rendered volume.

    Renders each unique atomic species present in `atomic_numbers` (via
    :func:`_isolated_atom_peak_potential`) as a single ISOLATED atom, not
    the full reference lipid patch, and returns the tallest of those raw
    peaks -- typically the phosphate phosphorus, the heaviest/most
    strongly-scattering species in the default lipid template, matching
    real bilayer electron-density profiles where the phosphate headgroup
    is the dominant feature.

    An earlier version instead rendered the FULL reference lipid patch
    (:func:`compute_bilayer_profile`) and took its LATERALLY-AVERAGED peak
    (``psi.max()``, averaged over ``lateral_core_fraction`` of the whole
    patch -- ~70 lipids' worth of mostly-empty lateral area at this
    module's own default ``n_lipids_per_leaflet=200``). Measured directly
    on a real patch: that diluted the true single-atom peak by ~20x (97.5
    -> 4.8), with no physical justification -- a real bilayer's phosphate
    atoms don't scatter more weakly than the same atom embedded in a
    protein, and the visual "smoothness" real membranes show in cryo-ET
    data comes from the microscope's own resolution limits applied to the
    ground truth AFTER it's built (already handled by
    ``rasterize_membrane_density``'s anti-aliasing step), not from
    pre-averaging the ground truth itself. This was a real, previously
    unnoticed bug (a user-reported visual mismatch -- a transmembrane
    protein's density looked 5-10x brighter than the surrounding membrane
    in a rendered volume -- led directly to finding and fixing it), not a
    deliberate design choice.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        From :func:`build_reference_lipid_patch` (or any atom set) --
        only the set of unique species present is used; a single-atom
        peak has no position dependence, so patch size no longer matters
        here (unlike `compute_bilayer_profile`, which still needs a large
        patch to converge the profile's SHAPE past per-lipid sampling
        noise -- these are independent uses of the same reference patch).
    coordinates : torch.Tensor
        Unused; accepted for call-site symmetry with
        :func:`compute_bilayer_profile` (both are built together by
        :func:`build_reference_lipid_patch`).
    voxel_size, parameterization, device
        Forwarded to the single-atom :class:`~specter.potential.PotentialBuilder`
        render.

    Returns
    -------
    float
    """
    del coordinates  # unused, see docstring
    unique_z = atomic_numbers.unique().tolist()
    return max(
        _isolated_atom_peak_potential(z, voxel_size, parameterization, device)
        for z in unique_z
    )


def build_analytic_bilayer_profile(
    thickness_a: float = 30.0,
    layer_sigma_a: float = 1.25,
    amplitude: float = 1.0,
    distance_half_range_a: float | None = None,
    n_points: int = 241,
    device: str | torch.device = "cpu",
) -> BilayerProfile:
    """
    Build a bilayer profile as two analytic Gaussian peaks, matching
    real cryo-EM bilayer micrographs' "railroad track" appearance far more
    directly than :func:`compute_bilayer_profile`'s simulated-atom-cloud
    approach does.

    ``compute_bilayer_profile`` derives the profile's SHAPE by jittering a
    schematic atomic lipid model and rendering it -- physically motivated,
    but fragile: getting the acyl-chain region to actually stay featureless
    relative to the phosphate peaks (rather than accidentally forming a
    competing secondary hump, or leaving the center only mildly lower than
    the chain shoulder -- both real bugs found and fixed by direct
    inspection, see this module's git history) means fighting per-cluster
    jitter-scale tuning against emergent Gaussian-sum interference, with no
    guarantee a slightly different lipid count/seed doesn't reintroduce a
    bump. `polnet` (`polnet.sample.membranes.mb_sphere.SphGen._build`) takes
    a structurally different, much more robust approach for exactly this
    problem: build two geometrically-separated thin shells (one per
    leaflet, offset by `+-thickness/2`) and Gaussian-blur them with a small
    sigma -- since the blur is far smaller than the leaflet separation, the
    two peaks cannot bleed into each other or spawn a secondary peak
    in between, by construction, not by tuning. This function is the same
    idea reduced to 1D (`psi(d)` is already evaluated against a signed
    distance from the mid-plane everywhere it's used, so the exact "thin
    shell" step is unnecessary -- a Gaussian centered at each leaflet's
    offset IS the 1D cross-section of that shell-then-blur construction).

    Parameters
    ----------
    thickness_a : float, optional
        Phosphate-to-phosphate (outer leaflet peak to inner leaflet peak)
        spacing, Angstrom. Default 30.0 -- midpoint of `polnet`'s own
        `MB_THICK_RG` default range (25.0, 35.0); matched to polnet
        directly rather than to the old atomic model's own headgroup
        z-offsets (38.0), which sat above polnet's entire range and read
        as visibly too widely spaced.
    layer_sigma_a : float, optional
        Gaussian width of each leaflet peak, Angstrom -- matches `polnet`'s
        own `MB_LAYER_S_RG` parameter. Default 1.25, the midpoint of
        polnet's own default range (0.5, 2.0).
    amplitude : float, optional
        Peak height. Pass :func:`estimate_bilayer_peak_amplitude`'s output
        to calibrate against real scattering-potential units rather than an
        arbitrary constant. Default 1.0 (uncalibrated).
    distance_half_range_a : float, optional
        Half-width of the returned lookup table's domain, Angstrom. Default
        ``thickness_a / 2 + 6 * layer_sigma_a`` (comfortably past both
        peaks' Gaussian tails). Values beyond this range are clamped to the
        table's edge (near-zero) by `BilayerProfile.__call__`.
    n_points : int, optional
        Lookup table resolution. Default 241 -- fine enough for accurate
        interpolation at any downstream rendering voxel size, independent
        of `layer_sigma_a`.
    device : str or torch.device, optional
        Device for the returned tensors. Default "cpu".

    Returns
    -------
    BilayerProfile

    References
    ----------
    Martinez-Sanchez, A., Lamm, L., Jasnin, M., & Phelippeau, H. (2024). Simulating
    the cellular context in synthetic datasets for cryo-electron tomography. IEEE
    Transactions on Medical Imaging, 43(11), 3742–3754.
    https://doi.org/10.1109/TMI.2024.3398401
    polnet source: https://github.com/anmartinezs/polnet
    """
    if distance_half_range_a is None:
        distance_half_range_a = thickness_a / 2.0 + 6.0 * layer_sigma_a

    distance_a = torch.linspace(
        -distance_half_range_a, distance_half_range_a, n_points, device=device
    )
    half_t = thickness_a / 2.0
    outer_leaflet = torch.exp(-0.5 * ((distance_a - half_t) / layer_sigma_a) ** 2)
    inner_leaflet = torch.exp(-0.5 * ((distance_a + half_t) / layer_sigma_a) ** 2)
    psi = amplitude * (outer_leaflet + inner_leaflet)

    return BilayerProfile(distance_a=distance_a, psi=psi)


__all__ = [
    "build_reference_lipid_patch",
    "compute_bilayer_profile",
    "estimate_bilayer_peak_amplitude",
    "build_analytic_bilayer_profile",
    "BilayerProfile",
]
