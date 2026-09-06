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

There is no amplitude scalar anywhere here: ``psi(d)`` is used as
measured, in volts, and nothing rescales it to a peak. Two constructions
that look reasonable are wrong, and are worth knowing about.

An amplitude fitted from a single isolated atom's peak potential, on the
reasoning that a plane average dilutes the true peak ~20x, is wrong twice
over: that dilution comes from averaging over the whole patch including
its under-populated jittered edges, which :func:`compute_bilayer_profile`
never does (``lateral_core_fraction`` 0.6 recovers 64.9 A^2 per lipid
against the 65.0 target), and an atom's own centre is a cusp with no
grid-independent value, so it cannot calibrate anything at any voxel
size. The rule governing any scalar taken from this module: a plane
average is commensurate only with another plane average.

An analytic two-Gaussian profile is wrong for a different reason. Two
Gaussians standing on vacuum model a bilayer's *appearance* in a
micrograph rather than its density, and deleting the acyl core that way
costs 4.8x of the integrated potential -- invisible in a slice, dominant
in a projection. The smoothing that makes real cryo-ET membranes look
continuous comes from the microscope's own resolution limits (CTF,
multislice, detector MTF), applied to membrane and protein alike AFTER
the ground truth is built, and downstream by ``_raster.py``'s
anti-aliasing. It is not something to bake into the ground truth.

Calibration is anchored by one parameter-free identity: integral(psi dz)
is fixed by chemistry alone, at 2 * (scattering per lipid) / (area per
lipid). Measured 254.0 V*A against 254.5 predicted. The same identity on
the protein side predicts 1FA2's mean inner potential as 7.03 V against
7.00 V measured by rendering it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch

from ...atom import atom_number
from ...potential import PotentialBuilder

# Same value as specimen/packing/algorithms.py's own independent copy of
# this constant -- each specimen generator keeps its own rather than
# importing, so this module's dependencies stay limited to atom/potential.
ATOM_KERNEL_HALF_WIDTH_ANGSTROM = 2.5

# Grid spacing the reference lipid patch is rendered on to produce psi(z),
# A. Deliberately independent of the voxel size a membrane is rendered at:
# psi is a continuous physical quantity in volts, and the render grid's
# resolution loss belongs downstream, applied once by
# rasterize_membrane_density's anti-aliasing.
#
# This constant is scaffolding for a SYNTHESIZED profile. psi(z) is
# obtained by rendering a schematic atomic model and averaging laterally,
# and any grid-based estimate has a spacing. Source psi(z) from an MD
# snapshot or from published component density profiles instead (see the
# module docstring's open item) and this constant has nothing left to do.
#
# The value is loose, because what the rasterizer consumes is the whole
# curve and its integral barely moves. Measured over 0.25-2.0 A, integral
# (psi dz) spans 240.4-242.1 V*A -- 0.7%, with no trend. The PEAK over the
# same range spans 7.70-8.59 V and is non-monotonic (8.59, 8.35, 8.39,
# 8.25, 7.70, 7.88): that is grid-alignment scatter on a narrow feature,
# not a convergence sequence, so do not read a converged peak off it or
# tune this constant to make the peak look stable.
CALIBRATION_VOXEL_SIZE_ANGSTROM = 1.0

# Lipids per leaflet in the reference patch psi(z) is measured from. Fixed
# independently of a caller's own n_lipids_per_leaflet, which sizes a patch
# for whatever that caller needs: the profile is a property of the lipid
# model and must not drift with it. The plane average converges from below
# as the patch grows, because a small patch's jittered edges eat a larger
# fraction of its own central window: 3.13 V at 30 lipids/leaflet, 4.10 at
# 60, 5.14 at 200, then flat within sampling noise at 5.29/5.48/5.31 for
# 400/800/1600.
CALIBRATION_N_LIPIDS_PER_LEAFLET = 400

# Seed for that reference patch, so the profile is identical run to run
# rather than drifting with the lipid jitter draw.
CALIBRATION_SEED = 0

# Schematic single-leaflet lipid atom template: (element, z_offset_angstrom, count,
# jitter_scale). z_offset_angstrom is distance from the mid-plane for this leaflet
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
# jitter_scale of 2.2-3.0 here (each cluster's std clearly wider than the
# 2A inter-cluster spacing) fixes this: the rendered profile has no
# competing peak in the +-4 to +-14A shoulder
# region (see test_compute_bilayer_profile_no_competing_peak_in_chain_
# region in tests/test_membrane_profile.py). The terminal methyls are
# pushed to the mid-plane itself (0.5A, was 1.0A) with even higher
# jitter_scale (4.5, was 2.5) so both leaflets' chain termini spread widely
# rather than piling up right at z=0 -- real bilayers show a genuine dip
# there (the "methyl trough"), not a peak, precisely because the terminal
# segment is the MOST conformationally disordered part of the chain.
# Census note: the counts below are the exact stoichiometry of POPC,
# C42 H82 N O8 P -- 1-palmitoyl-2-oleoyl-sn-glycero-3-phosphocholine, the
# standard fluid-phase model lipid, and the one this module's own
# area_per_lipid_a2=65.0 and the phosphate-phosphate spacing tests were
# already calibrated against. A template of the 31 carbons alone, with no
# hydrogens, carries 59% of POPC's electron-scattering power and
# understates the whole bilayer 1.74x.
#
# Hydrogen is not negligible for electrons even though it is nearly so for
# X-rays. Mott-Bethe makes a diffuse one-electron atom screen its own
# proton poorly at low k, so per unit mass hydrogen is 2.5x the scatterer
# carbon is (25.2 vs 10.0 V*A^3/Da here). POPC's 82 hydrogens are 25% of
# its total scattering, and they are why the acyl core -- pure CH2, the
# most hydrogen-rich part of the molecule -- sits ABOVE amorphous ice
# (5.78 V against 4.6 V) despite hydrocarbon at 0.9 g/cm^3 being the less
# dense material. The same effect is documented for the ice water kernel.
#
# Hydrogens are placed at their parent carbon's z rather than given their
# own offsets: a C-H bond is 1.09 A, well inside every jitter_scale here,
# and each H draws its jitter independently, so the pair spreads the way
# the group does. This is a z-DISTRIBUTION model, not a conformer.
_LEAFLET_TEMPLATE: list[tuple[str, float, int, float]] = [
    ("N", 21.5, 1, 0.45),  # choline nitrogen
    ("C", 20.5, 5, 0.45),  # choline: 3 methyls + the -CH2-CH2- linker
    ("H", 20.5, 13, 0.45),  # 9 on the methyls, 4 on the linker
    ("P", 19.5, 1, 0.45),  # phosphate
    ("O", 19.0, 4, 0.45),  # phosphate oxygens
    ("C", 15.0, 3, 1.3),  # glycerol backbone
    ("H", 15.0, 5, 1.3),  # glycerol hydrogens
    ("C", 13.5, 2, 1.3),  # the two ester carbonyl carbons
    ("O", 13.5, 2, 1.3),  # ester oxygens
    ("O", 13.0, 2, 1.3),  # carbonyl oxygens
    # 32 chain carbons and 64 chain hydrogens, tapering toward the
    # mid-plane: palmitoyl is C16 and oleoyl C18, so the two chains do not
    # reach equally deep and fewer carbons occupy the innermost levels.
    # The taper is what brings the integrated profile onto its independent
    # prediction (254.0 vs 254.5 V*A, see the module docstring).
    ("C", 12.0, 6, 2.2),  # acyl chains
    ("H", 12.0, 12, 2.2),
    ("C", 10.0, 5, 2.2),  # acyl chains
    ("H", 10.0, 10, 2.2),
    ("C", 8.0, 5, 2.2),  # acyl chains
    ("H", 8.0, 10, 2.2),
    ("C", 6.0, 5, 2.2),  # acyl chains
    ("H", 6.0, 10, 2.2),
    ("C", 4.0, 5, 2.6),  # acyl chains, distal
    ("H", 4.0, 10, 2.6),
    ("C", 2.0, 4, 3.0),  # acyl chains, distal
    ("H", 2.0, 6, 3.0),
    # Exactly two terminal methyl carbons, one per chain -- the count is
    # chemical, not a tuning knob, and it is what sets the depth of the
    # "methyl trough" both leaflets' termini share at the mid-plane.
    ("C", 0.5, 2, 4.5),
    ("H", 0.5, 6, 4.5),
]


def build_reference_lipid_patch(
    n_lipids_per_leaflet: int = 200,
    area_per_lipid_a2: float = 65.0,
    jitter_angstrom: float = 2.5,
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
        Lateral area per lipid, Å², used to size the patch. Default
        65.0 (a typical fluid-phase PC value).
    jitter_angstrom : float, optional
        Standard deviation of per-atom positional jitter, Å, standing
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
        Shape ``(N, 3)``, ``(x, y, z)`` Å, centered at the bilayer
        mid-plane (``z=0``) and patch center (``x=y=0``).
    """
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    side_angstrom = math.sqrt(n_lipids_per_leaflet * area_per_lipid_a2)

    elements: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for leaflet_sign in (1.0, -1.0):
        for _ in range(n_lipids_per_leaflet):
            lipid_xy = (torch.rand(2, generator=generator) - 0.5) * side_angstrom
            for element, z_offset, count, jitter_scale in _LEAFLET_TEMPLATE:
                jitter = torch.randn((count, 3), generator=generator) * (
                    jitter_angstrom * jitter_scale
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
    distance_angstrom : torch.Tensor
        Sorted ascending signed distances, shape ``(N,)``, Å.
    psi : torch.Tensor
        Laterally-averaged potential at each distance, shape ``(N,)``.
    """

    distance_angstrom: torch.Tensor
    psi: torch.Tensor

    def __call__(self, d: torch.Tensor) -> torch.Tensor:
        """
        Interpolate the profile at arbitrary signed distances.

        Parameters
        ----------
        d : torch.Tensor
            Signed distance from the bilayer mid-plane, Å, any shape.

        Returns
        -------
        torch.Tensor
            Interpolated potential, same shape as ``d``. Values beyond the
            table's range are clamped to the nearest tabulated value.
        """
        flat = _interp1d(d.reshape(-1), self.distance_angstrom, self.psi)
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
        Voxel size for the one-time atomic render, Å. Default 2.0.
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
    margin_angstrom = 2 * ATOM_KERNEL_HALF_WIDTH_ANGSTROM
    n_xyz = tuple(
        int(math.ceil((e + 2 * margin_angstrom) / voxel_size)) // 2 * 2 + 2
        for e in extent
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
    distance_angstrom = z_idx.to(dtype=psi.dtype) * voxel_size

    return BilayerProfile(distance_angstrom=distance_angstrom, psi=psi)


@lru_cache(maxsize=None)
def _measured_bilayer_profile(parameterization: str = "shtyrov") -> BilayerProfile:
    """
    The bilayer's psi(z), measured once per process from the reference
    lipid patch. Cached; see :func:`build_measured_bilayer_profile`.
    """
    atomic_numbers, coordinates = build_reference_lipid_patch(
        n_lipids_per_leaflet=CALIBRATION_N_LIPIDS_PER_LEAFLET,
        seed=CALIBRATION_SEED,
    )
    return compute_bilayer_profile(
        atomic_numbers,
        coordinates,
        voxel_size=CALIBRATION_VOXEL_SIZE_ANGSTROM,
        parameterization=parameterization,
    )


def native_bilayer_thickness_angstrom(parameterization: str = "shtyrov") -> float:
    """
    Phosphate-to-phosphate spacing of the measured profile, A.

    Peak-to-peak of :func:`_measured_bilayer_profile`, which is what
    `thickness_angstrom` means everywhere in this module. About 40 A for the
    default POPC template, against a published fluid-PC range of 36-39 A.

    Parameters
    ----------
    parameterization : str, optional
        ``PotentialBuilder`` parameterization. Default "shtyrov".

    Returns
    -------
    float
    """
    profile = _measured_bilayer_profile(parameterization)
    upper = profile.distance_angstrom > 0
    return 2.0 * float(profile.distance_angstrom[upper][profile.psi[upper].argmax()])


def build_measured_bilayer_profile(
    thickness_angstrom: float = 30.0,
    extra_sigma_angstrom: float = 0.0,
    parameterization: str = "shtyrov",
    device: str | torch.device = "cpu",
) -> BilayerProfile:
    """
    The bilayer profile the rasterizer renders: psi(z) measured from the
    reference lipid patch, rescaled in z to the requested thickness.

    The profile every membrane renders with. See the module docstring for
    why an analytic two-Gaussian form (two peaks on vacuum) discards most
    of a bilayer's integrated potential (53 V*A against this one's 254)
    and why that is invisible in a slice but dominant in a projection.

    Rescaling is a pure z-stretch at fixed amplitude, so the integral
    scales with thickness. That is the physical relationship: lipid
    volume is conserved, so a bilayer of thickness `t` built from lipids
    of volume `V` occupies area ``2V/t`` per lipid and therefore deposits
    ``t * (scattering per lipid) / V`` per unit area. Normalising the
    integral instead would make a thinner membrane denser, which is
    wrong.

    Parameters
    ----------
    thickness_angstrom : float, optional
        Phosphate-to-phosphate spacing to rescale to, A. Default 30.0,
        matching `MembraneGenerator`. See
        :func:`native_bilayer_thickness_angstrom` for the template's own value.
    extra_sigma_angstrom : float, optional
        Additional Gaussian broadening applied along z, A. Default 0.0:
        the measured profile already carries the width its atomic model
        implies, and the rasterizer anti-aliases to the render grid
        separately. Raise it only to deliberately blur a bilayer.
    parameterization : str, optional
        ``PotentialBuilder`` parameterization. Default "shtyrov".
    device : str or torch.device, optional
        Device for the returned tensors. Default "cpu".

    Returns
    -------
    BilayerProfile
    """
    profile = _measured_bilayer_profile(parameterization)
    scale = thickness_angstrom / native_bilayer_thickness_angstrom(parameterization)
    distance_angstrom = profile.distance_angstrom * scale
    psi = profile.psi.clone()

    if extra_sigma_angstrom > 0.0:
        spacing = float(distance_angstrom[1] - distance_angstrom[0])
        sigma_samples = extra_sigma_angstrom / spacing
        half = max(1, int(math.ceil(3.0 * sigma_samples)))
        offsets = torch.arange(-half, half + 1, dtype=psi.dtype, device=psi.device)
        kernel = torch.exp(-0.5 * (offsets / sigma_samples) ** 2)
        kernel = kernel / kernel.sum()
        psi = torch.nn.functional.conv1d(
            psi.view(1, 1, -1), kernel.view(1, 1, -1), padding=half
        ).view(-1)

    return BilayerProfile(
        distance_angstrom=distance_angstrom.to(device), psi=psi.to(device)
    )


__all__ = [
    "build_reference_lipid_patch",
    "compute_bilayer_profile",
    "build_measured_bilayer_profile",
    "native_bilayer_thickness_angstrom",
    "BilayerProfile",
]
