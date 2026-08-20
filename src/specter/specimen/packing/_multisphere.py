"""
Multi-sphere ("sphere decomposition") packing.

`pack_hard_spheres_3d` collides one circumscribing sphere per instance, of
radius ``PDB.max_diameter / 2``. That sphere is a poor stand-in for a real
molecule: measured over 180 cached CRYOETSIM/PEI2016 species, a molecule's
6.8 A envelope occupies only ~0.178 of its own bounding sphere, so the
achievable macromolecule volume fraction tops out around 0.043 -- roughly a
quarter of physiological cytoplasm, and about a seventh of what CryoTomoSim
reaches on the same species.

This module keeps the RSA machinery of `pack_hard_spheres_3d` (one trial
position per candidate per pass, `vesin_torch` neighbor list, one-shot
independent-set conflict resolution) and changes only the geometry: each
instance carries K spheres, fitted to its own atoms, rigidly rotated with
the instance. Collision is any inter-instance sphere pair closer than the
sum of their radii plus `gap`.

For the exact-shape alternative, which is what CryoTomoSim itself does, see
`._shape.pack_shapes_3d`.
"""

from __future__ import annotations

import numpy as np
import torch

# Radius of the "average organic atom", A. Added to each fitted cluster's
# extent so the union of spheres encloses the atoms' van der Waals volume
# rather than just their centers. Same value CryoTomoSim uses for the same
# purpose (`helper_atoms2vol.m`'s `avol`).
ATOM_RADIUS_A = 1.9


def fit_multisphere(
    coordinates: torch.Tensor,
    n_spheres: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit `n_spheres` spheres covering a molecule's atoms, by k-means.

    Each cluster contributes one sphere centered on the cluster centroid,
    with radius equal to the farthest member atom's distance plus
    `ATOM_RADIUS_A`, so the union of the returned spheres is guaranteed to
    enclose every atom.

    Parameters
    ----------
    coordinates : torch.Tensor
        Atomic coordinates, shape (N, 3), in the molecule's own frame.
        `specter.pdb.PDB.coordinates` is already centered on the atom
        centroid, which is the same origin `PotentialBuilder` builds its
        template cube about -- pass it unmodified so the fitted spheres and
        the rendered template share one origin.
    n_spheres : int
        Number of spheres, K. K=1 reproduces a single centroid-centered
        bounding sphere (slightly tighter than ``max_diameter / 2``, which
        over-estimates the true minimum enclosing sphere).
    seed : int, optional
        Random seed for k-means initialization.

    Returns
    -------
    offsets : torch.Tensor
        Sphere centers relative to the molecule origin, shape (K', 3).
        K' <= K, since k-means may return empty clusters.
    radii : torch.Tensor
        Sphere radii, A, shape (K',).
    """
    from scipy.cluster.vq import kmeans2

    coords = coordinates.detach().cpu().numpy().astype(np.float64)
    if coords.shape[0] == 0:
        raise ValueError("fit_multisphere: no coordinates")

    k = int(min(n_spheres, coords.shape[0]))
    if k <= 1:
        centre = coords.mean(0, keepdims=True)
        radius = np.linalg.norm(coords - centre, axis=1).max() + ATOM_RADIUS_A
        return (
            torch.tensor(centre, dtype=torch.float32),
            torch.tensor([radius], dtype=torch.float32),
        )

    centroids, labels = kmeans2(
        coords, k, minit="++", seed=0 if seed is None else seed, iter=25
    )
    offsets, radii = [], []
    for j in range(k):
        member = labels == j
        if not member.any():
            continue
        c = centroids[j]
        offsets.append(c)
        radii.append(np.linalg.norm(coords[member] - c, axis=1).max() + ATOM_RADIUS_A)

    return (
        torch.tensor(np.asarray(offsets), dtype=torch.float32),
        torch.tensor(np.asarray(radii), dtype=torch.float32),
    )


def _random_rotations(n: int, generator: torch.Generator, device) -> torch.Tensor:
    """Uniform random rotation matrices, shape (n, 3, 3), via unit quaternions."""
    q = torch.randn(n, 4, generator=generator, device=device)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=1,
    ).reshape(n, 3, 3)
    return R


def pack_multisphere_3d(
    species_offsets: list[torch.Tensor],
    species_radii: list[torch.Tensor],
    species_idx: torch.Tensor,
    box: tuple[float, float, float],
    gap: float = 0.0,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    max_passes: int = 200,
    stall_patience: int = 15,
    clip_axes: tuple[bool, bool, bool] = (False, False, False),
    exclusion_distance_field: torch.Tensor | None = None,
    field_voxel_size: float | None = None,
    sampling_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack rigid multi-sphere bodies into a box via Random Sequential Addition.

    Same pass structure, box/`clip_axes` convention, exclusion field and
    sampling mask semantics as `..algorithms.pack_hard_spheres_3d` -- see
    that function for the shared parameters. The difference is that each
    instance is a rigid assembly of spheres (from `fit_multisphere`) carrying
    its own random orientation, rather than one sphere.

    Unlike `pack_hard_spheres_3d`, this returns the orientation each instance
    was accepted at. That rotation is part of the collision result and must
    be reused when the instance is rendered, or the rendered volume will not
    match the geometry that was actually tested.

    Parameters
    ----------
    species_offsets : list of torch.Tensor
        Per species, sphere centers relative to the molecule origin, each
        shape (K_s, 3). From `fit_multisphere`.
    species_radii : list of torch.Tensor
        Per species, sphere radii in A, each shape (K_s,).
    species_idx : torch.Tensor
        Species index per candidate instance, shape (N,).
    box : tuple of float
        (D, H, W) extents in A (z, y, x), centered at the origin.
    gap : float, optional
        Extra clearance between sphere surfaces, A. Default 0.0.
    seed : int, optional
    device : str or torch.device, optional
    max_passes : int, optional
    stall_patience : int, optional
    clip_axes : tuple of bool, optional
    exclusion_distance_field : torch.Tensor, optional
    field_voxel_size : float, optional
    sampling_mask : torch.Tensor, optional

    Returns
    -------
    coords : torch.Tensor
        Accepted instance centers (x, y, z), A, box-centered, shape (M, 3).
    rotations : torch.Tensor
        Rotation matrix each instance was accepted at, shape (M, 3, 3).
    accepted_idx : torch.Tensor
        Indices into `species_idx` for the accepted instances, shape (M,).
    """
    import vesin_torch

    has_field_consumer = (
        exclusion_distance_field is not None or sampling_mask is not None
    )
    if has_field_consumer != (field_voxel_size is not None):
        raise ValueError(
            "field_voxel_size is required together with (and only with) "
            "exclusion_distance_field and/or sampling_mask"
        )

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    D, H, W = box
    half = torch.tensor([W / 2, H / 2, D / 2], device=device)
    box_t = torch.diag(torch.tensor([W, H, D], device=device))
    clip_xyz = torch.tensor(
        [clip_axes[2], clip_axes[1], clip_axes[0]], dtype=torch.bool, device=device
    )

    offsets = [o.to(device) for o in species_offsets]
    radii = [r.to(device) for r in species_radii]
    species_idx = species_idx.to(device)
    N = int(species_idx.numel())

    # Bounding radius per species, for the box-containment test and for
    # sizing the neighbor-list cutoff.
    bound_r = torch.stack([(o.norm(dim=1) + r).max() for o, r in zip(offsets, radii)])
    max_sphere_r = max(float(r.max()) for r in radii)
    cutoff = 2 * max_sphere_r + gap

    mask_positions: torch.Tensor | None = None
    if sampling_mask is not None:
        assert field_voxel_size is not None
        nz, ny, nx = sampling_mask.shape
        idx = sampling_mask.nonzero(as_tuple=False)
        if idx.shape[0] == 0:
            raise ValueError("sampling_mask has no True voxels")
        extent = (
            torch.tensor([nx, ny, nz], dtype=torch.float32, device=device)
            * field_voxel_size
        )
        origin = -0.5 * extent
        mask_positions = (
            origin
            + (idx[:, [2, 1, 0]].to(device=device, dtype=torch.float32) + 0.5)
            * field_voxel_size
        )

    def sample_field(points: torch.Tensor) -> torch.Tensor:
        assert exclusion_distance_field is not None and field_voxel_size is not None
        nz, ny, nx = exclusion_distance_field.shape
        n_xyz = torch.tensor([nx, ny, nz], dtype=points.dtype, device=device)
        origin = -0.5 * n_xyz * field_voxel_size
        norm = 2.0 * ((points - origin) / field_voxel_size + 0.5) / n_xyz - 1.0
        sampled = torch.nn.functional.grid_sample(
            exclusion_distance_field[None, None].to(device=device, dtype=points.dtype),
            norm.view(1, -1, 1, 1, 3),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.reshape(points.shape[0])

    if N == 0:
        e = (
            torch.empty((0, 3)),
            torch.empty((0, 3, 3)),
            torch.empty((0,), dtype=torch.long),
        )
        return e

    acc_pos = torch.empty((0, 3), device=device)
    acc_rot = torch.empty((0, 3, 3), device=device)
    acc_row = torch.empty((0,), dtype=torch.long, device=device)
    # Flattened spheres of everything accepted so far. Only positions and
    # radii are needed -- collisions against accepted instances never care
    # which instance a sphere belongs to, since a candidate is rejected on
    # the first contact regardless.
    acc_sph_xyz = torch.empty((0, 3), device=device)
    acc_sph_r = torch.empty((0,), device=device)

    def expand(pos: torch.Tensor, rot: torch.Tensor, rows: torch.Tensor):
        """Instance centers + rotations -> flattened world-space spheres."""
        xyz, rr, inst = [], [], []
        sp = species_idx[rows]
        for s in sp.unique():
            m = (sp == s).nonzero(as_tuple=True)[0]
            o, r = offsets[int(s)], radii[int(s)]
            # (m, 3, 3) @ (K, 3) -> (m, K, 3)
            world = torch.einsum("mij,kj->mki", rot[m], o) + pos[m][:, None, :]
            xyz.append(world.reshape(-1, 3))
            rr.append(r.repeat(m.numel()))
            inst.append(m.repeat_interleave(o.shape[0]))
        return torch.cat(xyz), torch.cat(rr), torch.cat(inst)

    # Largest bounding radius first, matching pack_hard_spheres_3d's staging.
    for s_val in bound_r.argsort(descending=True).tolist():
        rows = (species_idx == s_val).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        remaining = rows
        stall = 0
        for _ in range(max_passes):
            if remaining.numel() == 0:
                break
            n_cand = int(remaining.numel())
            if mask_positions is not None:
                assert field_voxel_size is not None
                pick = torch.randint(
                    0, mask_positions.shape[0], (n_cand,), generator=gen, device=device
                )
                jitter = (
                    torch.rand(n_cand, 3, generator=gen, device=device) - 0.5
                ) * field_voxel_size
                cand = mask_positions[pick] + jitter
            else:
                cand = (
                    (torch.rand(n_cand, 3, generator=gen, device=device) - 0.5)
                    * 2
                    * half
                )
            rot = _random_rotations(n_cand, gen, device)

            br = bound_r[species_idx[remaining]]
            margin = torch.where(clip_xyz.unsqueeze(0), 0.0, br.unsqueeze(1))
            blocked = ((cand.abs() + margin) > half.unsqueeze(0)).any(dim=1)

            c_xyz, c_r, c_inst = expand(cand, rot, remaining)

            if exclusion_distance_field is not None:
                d = sample_field(c_xyz)
                bad_inst = c_inst[d < c_r + gap].unique()
                if bad_inst.numel():
                    blocked[bad_inst] = True

            free = (~blocked).nonzero(as_tuple=True)[0]
            if free.numel() == 0:
                stall += 1
                if stall >= stall_patience:
                    break
                continue

            keep_sph = torch.isin(c_inst, free)
            f_xyz, f_r, f_inst = c_xyz[keep_sph], c_r[keep_sph], c_inst[keep_sph]

            # candidates vs. already accepted
            if acc_sph_xyz.numel() > 0:
                combined = torch.cat([f_xyz, acc_sph_xyz])
                nl = vesin_torch.NeighborList(cutoff=cutoff, full_list=False)
                i_idx, j_idx, dist = nl.compute(
                    combined, box_t, periodic=False, quantities="ijd"
                )
                n_f = f_xyz.shape[0]
                cross = (i_idx < n_f) & (j_idx >= n_f)
                ci, aj, cd = i_idx[cross], j_idx[cross] - n_f, dist[cross]
                hit = cd < (f_r[ci] + acc_sph_r[aj] + gap)
                if hit.any():
                    blocked[f_inst[ci[hit]].unique()] = True

            active = (~blocked).nonzero(as_tuple=True)[0]
            if active.numel() == 0:
                stall += 1
                if stall >= stall_patience:
                    break
                continue

            # candidates vs. each other -- independent set over INSTANCES
            if active.numel() > 1:
                keep2 = torch.isin(c_inst, active)
                a_xyz, a_r, a_inst = c_xyz[keep2], c_r[keep2], c_inst[keep2]
                nl2 = vesin_torch.NeighborList(cutoff=cutoff, full_list=False)
                i2, j2, d2 = nl2.compute(a_xyz, box_t, periodic=False, quantities="ijd")
                inter = a_inst[i2] != a_inst[j2]
                conflict = inter & (d2 < (a_r[i2] + a_r[j2] + gap))
                ii, jj = a_inst[i2[conflict]], a_inst[j2[conflict]]
                if ii.numel() > 0:
                    prio = torch.rand(n_cand, generator=gen, device=device)
                    best = torch.full((n_cand,), float("inf"), device=device)
                    best.scatter_reduce_(0, ii, prio[jj], reduce="amin")
                    best.scatter_reduce_(0, jj, prio[ii], reduce="amin")
                    survive = prio < best
                    active = active[survive[active]]

            if active.numel() > 0:
                new_rows = remaining[active]
                acc_pos = torch.cat([acc_pos, cand[active]])
                acc_rot = torch.cat([acc_rot, rot[active]])
                acc_row = torch.cat([acc_row, new_rows])
                s_xyz, s_r, _ = expand(cand[active], rot[active], new_rows)
                acc_sph_xyz = torch.cat([acc_sph_xyz, s_xyz])
                acc_sph_r = torch.cat([acc_sph_r, s_r])
                keep = torch.ones(n_cand, dtype=torch.bool, device=device)
                keep[active] = False
                remaining = remaining[keep]
                stall = 0
            else:
                stall += 1
                if stall >= stall_patience:
                    break

    return acc_pos.cpu(), acc_rot.cpu(), acc_row.cpu()
