"""
Transmembrane insertion geometry: surface site sampling and normal-aligned
orientation.

Because a :class:`~specter.specimen.membrane._field.MembraneField` is a
dense signed field defined everywhere (not just at mesh vertices), the local
surface normal at any candidate insertion point is just its gradient --
exact, and available via Newton projection without a mesh or a KD-tree
(unlike CTS's fragile skeleton/kNN normal estimate or polnet's
mesh-vertex-normal lookup). Orientation composes the same two-step
placement both CTS and polnet use: align a canonical axis to the local
normal, then apply a free random spin around that same axis.

References
----------
Purnell, C., Heebner, J., Swulius, M. T., Hylton, R., Kabonick, S., Grillo, M.,
Grigoryev, S., Heberle, F., Waxham, M. N., & Swulius, M. T. (2023). Rapid
synthesis of cryo-ET data for training deep learning models. bioRxiv
2023.04.28.538636. https://doi.org/10.1101/2023.04.28.538636
CTS source: https://github.com/carsonpurnell/cryotomosim_CTS

Martinez-Sanchez, A., Lamm, L., Jasnin, M., & Phelippeau, H. (2024). Simulating
the cellular context in synthetic datasets for cryo-electron tomography. IEEE
Transactions on Medical Imaging, 43(11), 3742–3754.
https://doi.org/10.1109/TMI.2024.3398401
polnet source: https://github.com/anmartinezs/polnet
"""

from __future__ import annotations

import math

import torch

from ...rotations import rotation_aligning
from ._field import MembraneField


def _project_to_surface(
    field: MembraneField, points_xyz: torch.Tensor, iterations: int
) -> torch.Tensor:
    """Newton-project points onto ``field``'s zero level set."""
    x = points_xyz.clone()
    for _ in range(iterations):
        phi_val = field.sample(x)
        grad = field.gradient(x)
        x = x - phi_val.unsqueeze(-1) * grad
    return x


def sample_surface_sites(
    field: MembraneField,
    n_sites: int,
    min_spacing_a: float,
    max_attempts: int | None = None,
    projection_iterations: int = 8,
    phi_tolerance_a: float = 1.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample well-spaced points on a membrane's mid-surface.

    Draws random candidate points within ``field``'s working grid,
    Newton-projects each onto the zero level set (gradient descent on
    ``phi``, which converges quickly since ``|grad(phi)| ~= 1`` near the
    surface), then greedily accepts candidates that land close enough to the
    surface and are not within ``min_spacing_a`` of any already-accepted
    site (self-avoiding-walk style, not a true blue-noise Poisson-disc
    sampler).

    Parameters
    ----------
    field : MembraneField
    n_sites : int
        Target number of sites.
    min_spacing_a : float
        Minimum center-to-center spacing between accepted sites, Å.
    max_attempts : int, optional
        Maximum candidate draws before giving up. Default ``20 * n_sites``.
    projection_iterations : int, optional
        Newton projection steps per candidate. Default 8.
    phi_tolerance_a : float, optional
        Maximum ``|phi|`` after projection for a candidate to be accepted as
        genuinely on the surface. Default 1.0.
    seed : int, optional
        Random seed. Default None.

    Returns
    -------
    sites_xyz : torch.Tensor
        Shape ``(M, 3)``, ``M <= n_sites`` (fewer if ``max_attempts`` is
        exhausted first, e.g. because the surface is too small/crowded for
        the requested count at this spacing).
    normals_xyz : torch.Tensor
        Unit outward normals at each site, shape ``(M, 3)``.
    """
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    if max_attempts is None:
        max_attempts = 20 * n_sites

    device = field.phi.device
    nz, ny, nx = field.phi.shape
    extent = (
        torch.tensor([nx, ny, nz], dtype=torch.float32, device=device) * field.spacing_a
    )
    origin = field.origin_xyz.to(device)

    accepted_sites: list[torch.Tensor] = []
    accepted_normals: list[torch.Tensor] = []

    attempts = 0
    while len(accepted_sites) < n_sites and attempts < max_attempts:
        batch = min(64, max_attempts - attempts)
        candidates = (
            origin + torch.rand((batch, 3), generator=generator).to(device) * extent
        )
        projected = _project_to_surface(field, candidates, projection_iterations)
        phi_residual = field.sample(projected).abs()
        valid_mask = phi_residual < phi_tolerance_a

        for i in range(batch):
            attempts += 1
            if len(accepted_sites) >= n_sites:
                break
            if not bool(valid_mask[i]):
                continue
            candidate_site = projected[i]
            if accepted_sites:
                existing = torch.stack(accepted_sites)
                dists = torch.linalg.norm(existing - candidate_site, dim=-1)
                if bool((dists < min_spacing_a).any()):
                    continue
            normal = field.gradient(candidate_site.unsqueeze(0))[0]
            accepted_sites.append(candidate_site)
            accepted_normals.append(normal)

    if not accepted_sites:
        return (
            torch.zeros((0, 3), device=device, dtype=field.phi.dtype),
            torch.zeros((0, 3), device=device, dtype=field.phi.dtype),
        )
    return torch.stack(accepted_sites), torch.stack(accepted_normals)


def orientation_for_normal(
    normal_xyz: torch.Tensor, seed: int | None = None
) -> torch.Tensor:
    """
    Rotation matrix aligning local +Z to ``normal_xyz``, plus a random spin.

    Parameters
    ----------
    normal_xyz : torch.Tensor
        Unit (or non-unit -- renormalized internally) target direction,
        shape ``(3,)``.
    seed : int, optional
        Random seed for the spin angle. Default None.

    Returns
    -------
    torch.Tensor
        Rotation matrix, shape ``(3, 3)``, such that ``R @ [0, 0, 1]`` equals
        ``normal_xyz`` (normalized), with an additional free rotation about
        that axis.
    """
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    theta = float(torch.rand(1, generator=generator).item() * 2 * math.pi)
    return _orientation_for_normal_and_angle(normal_xyz, theta)


def _orientation_for_normal_and_angle(
    normal_xyz: torch.Tensor, theta: float
) -> torch.Tensor:
    device = normal_xyz.device
    dtype = normal_xyz.dtype
    z_hat = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    r_align = rotation_aligning(z_hat, normal_xyz)

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    r_spin = torch.tensor(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return r_align @ r_spin


def align_principal_axis_to_z(coordinates: torch.Tensor) -> torch.Tensor:
    """
    Rotate a structure so its longest principal axis of inertia lies along z.

    A real PDB/mmCIF file's native coordinate axes are an arbitrary artifact
    of how the structure was solved/deposited -- nothing guarantees they
    correlate with the membrane normal. For a membrane protein with a large
    asymmetric extramembrane domain (e.g. a bound cytochrome, G-protein, or
    soluble head group), that domain's real spatial extent -- not the file's
    z-axis -- is the physically meaningful "sticks out of the membrane"
    direction, since evolution orients such domains roughly perpendicular to
    the bilayer. Confirmed empirically on a real structure: its principal
    axis (true longest extent) diverged substantially from its native
    z-axis, and depth-centering along the wrong axis left the extramembrane
    domain pointing off at an angle rather than cleanly away from the
    membrane.

    The sign of the principal axis (which end becomes ``+z``) is not
    resolved -- with no topology information (cytoplasmic vs. extracellular,
    etc.) to break the symmetry, this is an arbitrary but deterministic
    (PCA-sign) choice.

    Parameters
    ----------
    coordinates : torch.Tensor
        Atomic coordinates, shape ``(N, 3)``, ``(x, y, z)``.

    Returns
    -------
    torch.Tensor
        Rotated coordinates, shape ``(N, 3)``, centered at the origin.
    """
    centroid = coordinates.mean(dim=0, keepdim=True)
    centered = coordinates - centroid
    cov = centered.T @ centered / centered.shape[0]
    _, eigvecs = torch.linalg.eigh(cov)
    principal_axis = eigvecs[:, -1]

    z_hat = torch.tensor(
        [0.0, 0.0, 1.0], device=coordinates.device, dtype=coordinates.dtype
    )
    rotation = rotation_aligning(principal_axis, z_hat)
    return centered @ rotation.T


def align_transmembrane_depth(
    coordinates: torch.Tensor, tm_span_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Center a structure's transmembrane span at ``z=0``.

    Translates (never rescales -- hydrophobic mismatch between a protein's
    real transmembrane span and the local bilayer thickness is a genuine
    biophysical effect, not something to hide by stretching the structure)
    ``coordinates`` along ``z`` so the transmembrane domain's midpoint sits
    at the local bilayer mid-plane.

    Parameters
    ----------
    coordinates : torch.Tensor
        Atomic coordinates, shape ``(N, 3)``, ``(x, y, z)``.
    tm_span_mask : torch.Tensor, optional
        Boolean mask, shape ``(N,)``, selecting the atoms spanning the
        membrane. Default None, which centers the full structure's z-extent
        instead.

    Returns
    -------
    torch.Tensor
        Translated coordinates, shape ``(N, 3)``.
    """
    selected = coordinates[tm_span_mask] if tm_span_mask is not None else coordinates
    center_z = 0.5 * (selected[:, 2].max() + selected[:, 2].min())
    shifted = coordinates.clone()
    shifted[:, 2] = coordinates[:, 2] - center_z
    return shifted


__all__ = [
    "sample_surface_sites",
    "orientation_for_normal",
    "align_transmembrane_depth",
    "align_principal_axis_to_z",
]
