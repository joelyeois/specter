"""
Organic, irregular vesicle/membrane shape generation -- a from-scratch port
of CryoTomoSim (CTS)'s ``gen_mem.m`` algorithm (alpha-shape-based blob
geometry + bilayer point sampling), reimplemented in NumPy/PyTorch.

Complements ``specimen/cryoet.py``'s polnet-based ``CryoETSpecimenGenerator``,
which only supports regular analytic membrane primitives (sphere/ellipsoid/
toroid): this module is for irregular, biologically-plausible closed vesicle
shapes instead. It has NO dependency on polnet or VTK, and does not import
anything from ``cryoet.py``/``polnet_bridge.py``.

Algorithm
---------
1. Scatter noisy points on a rough sphere, smooth via iterative
   nearest-neighbor averaging (``_smooth_points``, port of ``gen_mem.m``'s
   ``smiter`` subfunction).
2. Wrap the smoothed points in an alpha shape (Delaunay triangulation +
   circumradius filter, ``_alpha_complex`` -- the actual alpha-shape
   algorithm, hand-rolled rather than pulling in an extra dependency).
3. Resample the alpha shape's surface uniformly (``_sample_from_simplices``,
   a direct port of CTS's ``randtess.m``: pick a simplex weighted by its own
   volume/area, then draw a uniform point inside it via the standard
   recursive "mix toward the next vertex with weight U^(1/i)" method),
   push each point outward by a random vector scaled to the target bilayer
   thickness, and re-wrap -- this gives a genuinely shell-shaped (hollow),
   not solid, blob (CTS's ``shape2shell``).

   Simplification vs. CTS: ``gen_mem.m`` repeats the wrap/resample/smooth
   cycle 2-3 times with a shrinking alpha (using MATLAB's ``criticalAlpha``
   search for the minimal single-region alpha at each pass). This port does
   a single smoothing pass before the first wrap and a fixed alpha (a
   constant multiple of the target size) rather than searching for the
   critical value -- documented here rather than silently dropped. It is
   robust across the tested size/roughness range but does not guarantee a
   single connected component the way ``criticalAlpha`` does.
4. Sample two point populations from the shell (CTS's ``shell2pts``): dense
   "head" points near the two offset leaflet surfaces (with a small radial
   jitter), and sparser "tail" points filling the shell's volume.
5. Bin the sampled points into a small local voxel grid via
   ``torch.scatter_add_`` -- the torch equivalent of MATLAB's ``accumarray``
   in ``helper_atoms2vol.m``.

The returned density is UNSCALED (raw geometric point counts) -- callers are
expected to calibrate its magnitude against real, ``PotentialBuilder``-derived
organic-matter density (see ``cryotomosim.py``'s ``_organic_potential_reference``),
not bake in an arbitrary constant here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import Delaunay


def _tetra_circumradii(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """
    Circumradius of each tetrahedron in `simplices`, via a batched linear
    solve for the circumcenter (relative to each tetrahedron's first
    vertex): for vectors a, b, c from vertex 0 to vertices 1, 2, 3, the
    circumcenter x (relative to vertex 0) satisfies 2*[a;b;c] @ x =
    [a.a, b.b, c.c].

    Parameters
    ----------
    points : np.ndarray
        Shape (P, 3).
    simplices : np.ndarray
        Shape (T, 4), indices into `points`.

    Returns
    -------
    np.ndarray
        Shape (T,), circumradius of each tetrahedron. Degenerate
        (near-singular) tetrahedra get `inf` (naturally excluded by any
        finite alpha threshold).
    """
    p0 = points[simplices[:, 0]]
    a = points[simplices[:, 1]] - p0
    b = points[simplices[:, 2]] - p0
    c = points[simplices[:, 3]] - p0
    M = np.stack([a, b, c], axis=1)  # (T, 3, 3)
    rhs = 0.5 * np.stack(
        [(a * a).sum(-1), (b * b).sum(-1), (c * c).sum(-1)], axis=-1
    )  # (T, 3)

    radii = np.full(len(simplices), np.inf)
    dets = np.linalg.det(M)
    ok = np.abs(dets) > 1e-12
    if ok.any():
        # rhs[ok, :, None] forces the unambiguous batched-matrix-rhs solve
        # signature (..., m, m), (..., m, 1) -> (..., m, 1); passing a plain
        # (K, 3) vector rhs is ambiguous between "K systems of 3 unknowns"
        # and "one batch of 3-vectors" across numpy versions.
        centers = np.linalg.solve(M[ok], rhs[ok][..., None])[..., 0]
        radii[ok] = np.linalg.norm(centers, axis=-1)
    return radii


def _alpha_complex(points: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Delaunay-triangulate `points` and filter to an alpha complex.

    Parameters
    ----------
    points : np.ndarray
        Shape (P, 3).
    alpha : float
        Circumradius threshold; tetrahedra with circumradius >= alpha are
        discarded.

    Returns
    -------
    kept_tetra : np.ndarray
        Shape (M, 4), surviving tetrahedra (indices into `points`).
    boundary_faces : np.ndarray
        Shape (K, 3), triangular faces belonging to exactly one surviving
        tetrahedron -- i.e. the alpha shape's outer surface.

    Raises
    ------
    ValueError
        If no tetrahedra survive filtering (alpha too small for this point
        cloud's typical spacing).
    """
    tri = Delaunay(points)
    simplices = tri.simplices
    radii = _tetra_circumradii(points, simplices)
    kept = simplices[radii < alpha]

    if len(kept) == 0:
        raise ValueError(
            f"alpha={alpha:.4g} too small: no tetrahedra survived out of "
            f"{len(simplices)} candidates. Try a larger alpha."
        )

    face_count: dict[tuple[int, int, int], int] = {}
    face_owner: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for tetra in kept:
        i0, i1, i2, i3 = (int(v) for v in tetra)
        for face in ((i0, i1, i2), (i0, i1, i3), (i0, i2, i3), (i1, i2, i3)):
            a, b, c = sorted(face)
            key = (a, b, c)
            face_count[key] = face_count.get(key, 0) + 1
            face_owner[key] = face
    boundary_faces = np.array(
        [face_owner[k] for k, cnt in face_count.items() if cnt == 1], dtype=np.int64
    )
    if boundary_faces.size == 0:
        raise ValueError(
            f"alpha={alpha:.4g}: kept tetrahedra form no boundary "
            "(degenerate point cloud?)."
        )
    return kept, boundary_faces


def _simplex_measures(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """
    Unsigned measure of each simplex: volume for tetrahedra
    (``simplices.shape[1] == 4``), area for triangles
    (``simplices.shape[1] == 3``), embedded in 3D.
    """
    p0 = points[simplices[:, 0]]
    a = points[simplices[:, 1]] - p0
    b = points[simplices[:, 2]] - p0
    if simplices.shape[1] == 4:
        c = points[simplices[:, 3]] - p0
        return np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=-1)


def _sample_from_simplices(
    points: np.ndarray,
    simplices: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Uniformly sample `n` points from the union of `simplices` (triangles or
    tetrahedra), weighted by each simplex's own volume/area.

    Direct port of CTS's ``randtess.m``: pick a simplex with probability
    proportional to its measure (cumulative-volume-weighted selection),
    then generate a uniformly distributed point inside it via the
    recursive "mix toward the next vertex with weight ``U^(1/i)``" method
    (a standard order-statistic trick for uniform simplex sampling -- see
    ``randtess.m``'s main loop, which this mirrors verbatim).

    Parameters
    ----------
    points : np.ndarray
        Shape (P, 3), the simplices' vertex pool.
    simplices : np.ndarray
        Shape (M, 3) for triangles or (M, 4) for tetrahedra.
    n : int
        Number of points to sample.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        Shape (n, 3).
    """
    k = simplices.shape[1]  # 3 (triangle) or 4 (tetrahedron)
    measures = _simplex_measures(points, simplices)
    total = measures.sum()
    if total <= 0:
        raise ValueError("all simplices have zero measure")
    probs = measures / total
    chosen = rng.choice(len(simplices), size=n, p=probs)
    verts = points[simplices[chosen]]  # (n, k, 3)

    p = rng.random((n, 1))
    X = verts[:, 0] * p + verts[:, 1] * (1 - p)
    for i in range(2, k):
        p = rng.random((n, 1)) ** (1.0 / i)
        X = X * p + verts[:, i] * (1 - p)
    return X


def _smooth_points(points: np.ndarray, k: int = 9, iters: int = 3) -> np.ndarray:
    """
    Iterative nearest-neighbor averaging -- port of ``gen_mem.m``'s
    ``smiter`` subfunction. Each point moves to the mean of its `k` nearest
    neighbors (excluding itself), repeated `iters` times.
    """
    pts = torch.as_tensor(points, dtype=torch.float64)
    k = min(k, pts.shape[0] - 1)
    for _ in range(iters):
        d = torch.cdist(pts, pts)
        idx = d.topk(k + 1, largest=False).indices[:, 1:]  # drop self (distance 0)
        pts = pts[idx].mean(dim=1)
    return pts.numpy()


def _point_in_alpha_solid(
    points: np.ndarray, alpha: float, query_xyz: np.ndarray
) -> np.ndarray:
    """
    Test whether each point in `query_xyz` falls inside the alpha-complex
    solid built from `points` at the given `alpha`.

    Uses ``scipy.spatial.Delaunay.find_simplex`` (an efficient directed
    search, not brute-force) to locate which tetrahedron (if any) each
    query point falls in, then checks whether that specific tetrahedron
    survived the alpha-radius filter. This is robust to any rasterization
    gaps -- unlike a flood fill over a discretized point-cloud raster, a
    Delaunay triangulation always completely tiles its convex hull with no
    gaps, so this needs no morphological closing/dilation to work
    correctly.

    Parameters
    ----------
    points : np.ndarray
        Shape (P, 3), the same point cloud used to build the alpha shape.
    alpha : float
        Circumradius threshold, matching ``_alpha_complex``.
    query_xyz : np.ndarray
        Shape (Q, 3), points to test.

    Returns
    -------
    np.ndarray
        Bool, shape (Q,).
    """
    tri = Delaunay(points)
    radii = _tetra_circumradii(points, tri.simplices)
    kept_mask = radii < alpha
    simplex_idx = tri.find_simplex(query_xyz)
    inside = np.zeros(len(query_xyz), dtype=bool)
    valid = simplex_idx >= 0
    inside[valid] = kept_mask[simplex_idx[valid]]
    return inside


def _compute_geometry_fields(
    base_points: np.ndarray,
    alpha: float,
    density: torch.Tensor,
    center: np.ndarray,
    v_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Derive a shell (bilayer footprint) mask, an interior (vesicle lumen)
    mask, and a local outward-normal vector field, on `density`'s own
    voxel grid.

    The interior/exterior test uses the ORIGINAL (pre-shell-thickening)
    smoothed blob point cloud's alpha-complex solid (`base_points`/`alpha`,
    via ``_point_in_alpha_solid``) rather than the final rasterized,
    randomly-thickened bilayer shell -- the shell's own point raster is not
    reliably watertight (discrete point sampling can leave small gaps a
    flood fill would leak through), whereas a Delaunay-based solid test is
    exact regardless of voxel resolution.

    Parameters
    ----------
    base_points : np.ndarray
        The smoothed base blob's point cloud, shape (P, 3), same
        coordinate frame `density`'s rasterization used (i.e. NOT yet
        re-centered to the grid -- absolute coordinates).
    alpha : float
        Alpha-shape circumradius threshold used to build `base_points`'
        solid (same value passed to ``_alpha_complex`` when it was built).
    density : torch.Tensor
        Rasterized bilayer density, shape (nz, ny, nx).
    center : np.ndarray
        Shape (3,), the physical (x, y, z) coordinate `density`'s own grid
        midpoint corresponds to (as returned by
        ``MembraneBlobGenerator._rasterize``).
    v_size : float
        Voxel size, Angstrom.

    Returns
    -------
    shell_mask : torch.Tensor
        Bool, shape (nz, ny, nx). Voxels considered "in/on the bilayer" --
        simply the (lightly dilated) occupied-voxel mask.
    interior_mask : torch.Tensor
        Bool, shape (nz, ny, nx). Voxels inside the base blob's solid
        alpha-shape but not themselves part of the bilayer shell (the
        vesicle lumen).
    normal_field : torch.Tensor
        Float, shape (3, nz, ny, nx). Component order is PHYSICAL (x, y, z)
        -- matching ``specter.rotations.rotate_volume``'s own affine-grid
        convention (physical x = last array axis / nx, z = first array
        axis / nz) -- NOT (z, y, x) spatial-index order. Computed as the
        gradient of the base blob's signed distance field (positive
        outside its solid, negative inside), so it points outward from the
        membrane surface at any voxel, including shell voxels themselves.
    """
    n = density.shape[0]
    # Dilated generously (not just +1) so that after this membrane is later
    # rotated into the specimen volume, the rotated MASK still fully
    # covers the rotated DENSITY's own nonzero footprint: rotate_volume's
    # bilinear interpolation spreads a sparse, spiky point-count density
    # field to more neighboring voxels than a lightly-dilated binary mask
    # rotated the same way would cover, which would otherwise make a
    # membrane-flagged particle's overlap-ignore mask (see
    # _cts_placement.py's `ignore_overlap_mask`) undercount the membrane's
    # own footprint and spuriously block placement.
    shell_mask_np = ndimage.binary_dilation(density.numpy() > 0, iterations=3)

    zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    query = np.stack(
        [
            (xx.ravel() - n / 2) * v_size + center[0],
            (yy.ravel() - n / 2) * v_size + center[1],
            (zz.ravel() - n / 2) * v_size + center[2],
        ],
        axis=1,
    )
    inside_base_blob = _point_in_alpha_solid(base_points, alpha, query).reshape(n, n, n)
    interior_mask_np = inside_base_blob & ~shell_mask_np

    dist_out = ndimage.distance_transform_edt(~inside_base_blob)
    dist_in = ndimage.distance_transform_edt(inside_base_blob)
    signed_dist = dist_out - dist_in  # >0 outside the blob solid, <0 inside it

    gz, gy, gx = np.gradient(signed_dist)
    normal_xyz = np.stack([gx, gy, gz], axis=0)  # physical (x, y, z) component order
    mag = np.linalg.norm(normal_xyz, axis=0, keepdims=True)
    mag[mag < 1e-8] = 1.0
    normal_xyz = normal_xyz / mag

    return (
        torch.as_tensor(shell_mask_np),
        torch.as_tensor(interior_mask_np),
        torch.as_tensor(normal_xyz, dtype=torch.float32),
    )


@dataclass
class BlobMembraneInstance:
    """
    A single organic-blob membrane, rasterized into a small local grid.

    Attributes
    ----------
    density : torch.Tensor
        Unscaled local density grid, shape (nz, ny, nx), the shape
        generated centered at the grid's own midpoint. Callers must
        calibrate its magnitude before inserting into a specimen volume
        (see module docstring).
    thickness : float
        Bilayer thickness (Angstrom) used to generate this instance.
    v_size : float
        Voxel size (Angstrom) of `density`.
    shell_mask : torch.Tensor
        Bool, shape (nz, ny, nx) -- see ``_compute_geometry_fields``.
    interior_mask : torch.Tensor
        Bool, shape (nz, ny, nx) -- see ``_compute_geometry_fields``.
    normal_field : torch.Tensor
        Float, shape (3, nz, ny, nx), physical (x, y, z) component order
        -- see ``_compute_geometry_fields``.
    """

    density: torch.Tensor
    thickness: float
    v_size: float
    shell_mask: torch.Tensor
    interior_mask: torch.Tensor
    normal_field: torch.Tensor


class MembraneBlobGenerator:
    """
    Generates irregular, organic, closed-surface vesicle/membrane shapes
    with an explicit bilayer density profile (dense "head" points near
    each leaflet surface, sparser "tail" points through the shell
    interior) -- see module docstring for the full algorithm. No
    dependency on polnet or VTK.

    Parameters
    ----------
    v_size : float
        Output grid voxel size, Angstrom.
    seed : int, optional
        Random seed.
    """

    def __init__(self, v_size: float, seed: int | None = None):
        self.v_size = v_size
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        size: float = 300.0,
        roughness: float = 0.7,
        thickness: float = 40.0,
        n_head_points: int = 6000,
        n_tail_points: int = 2000,
        surface_jitter: float | None = None,
    ) -> BlobMembraneInstance:
        """
        Parameters
        ----------
        size : float, optional
            Rough target radius of the generated vesicle, Angstrom.
            Default 300.
        roughness : float, optional
            In (0, 1); higher values give a rounder, less irregular shape
            (matches CTS's `sp` parameter). Default 0.7.
        thickness : float, optional
            Bilayer thickness, Angstrom (outer-to-inner leaflet offset).
            Default 40.
        n_head_points, n_tail_points : int, optional
            Number of sampled points for the dense surface ("head") and
            sparse interior ("tail") populations.
        surface_jitter : float, optional
            Radial spread (Angstrom) applied to head points around the
            offset surfaces. Defaults to ``0.2 * thickness``.

        Returns
        -------
        BlobMembraneInstance
        """
        if not 0.0 < roughness < 1.0:
            raise ValueError(f"roughness must be in (0, 1), got {roughness}")
        if surface_jitter is None:
            surface_jitter = 0.2 * thickness

        base_points = self._build_blob(size, roughness)
        alpha = size * 0.9
        _, base_faces = _alpha_complex(base_points, alpha)

        # shape2shell: resample the smoothed blob's surface, push each
        # point outward by a random vector scaled to the target thickness,
        # and re-wrap -- gives a hollow, shell-shaped blob rather than a
        # solid one.
        n_surf = max(500, int(size // 2))
        surf_pts = _sample_from_simplices(base_points, base_faces, n_surf, self.rng)
        vec = self.rng.normal(size=surf_pts.shape)
        vec *= thickness / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        shell_points = surf_pts + vec
        shell_tetra, shell_faces = _alpha_complex(shell_points, alpha=thickness * 3.0)

        # Head points: dense, near the shell surface, with small jitter
        # along the local outward direction (approximated by direction
        # from the shell centroid, since we don't carry per-face normals).
        head = _sample_from_simplices(
            shell_points, shell_faces, n_head_points, self.rng
        )
        centroid = shell_points.mean(axis=0)
        outward = head - centroid
        outward /= np.linalg.norm(outward, axis=1, keepdims=True) + 1e-12
        jitter = (
            self.rng.random(n_head_points) - self.rng.random(n_head_points)
        ) * surface_jitter
        head = head + outward * jitter[:, None]

        # Tail points: sparse, filling the shell's volume.
        tail = _sample_from_simplices(
            shell_points, shell_tetra, n_tail_points, self.rng
        )

        all_points = np.concatenate([head, tail], axis=0)
        density, raster_center = self._rasterize(all_points)
        shell_mask, interior_mask, normal_field = _compute_geometry_fields(
            base_points, alpha, density, raster_center, self.v_size
        )
        return BlobMembraneInstance(
            density=density,
            thickness=thickness,
            v_size=self.v_size,
            shell_mask=shell_mask,
            interior_mask=interior_mask,
            normal_field=normal_field,
        )

    def _build_blob(self, size: float, sp: float) -> np.ndarray:
        """Initial noisy, roughly-spherical point cloud (port of
        ``gen_mem.m``'s ``blob`` subfunction, points only -- the alpha-shape
        wrapping itself is done by the caller)."""
        n = max(20, int(8 + size ** (0.2 + sp)))
        rad = size * sp
        var = size * (1 - sp) * 2
        az = self.rng.random(n) * np.pi
        el = self.rng.random(n) * np.pi
        r = self.rng.random(n) * var + rad
        x = r * np.sin(el) * np.cos(az)
        y = r * np.sin(el) * np.sin(az)
        z = r * np.cos(el)
        pts = np.stack([x, y, z], axis=1)

        reps = max(2, int(10 * sp**0.5))
        qq = np.tile(pts, (reps, 1))
        noise_scale = 10.0 / sp**2 * (1 - sp) + 1e-3
        d = self.rng.normal(size=qq.shape) * noise_scale
        pts = qq + d
        pts = _smooth_points(pts, k=9, iters=3)
        pts = np.unique(pts.round(6), axis=0)
        if pts.shape[0] < 8:
            raise ValueError(
                f"blob point generation collapsed to {pts.shape[0]} unique "
                "points -- size/roughness combination too extreme"
            )
        return pts

    def _rasterize(self, points: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Bin `points` (already centered near the origin) into a local
        voxel grid via scatter-add -- torch equivalent of
        ``helper_atoms2vol.m``'s ``accumarray`` binning.

        Returns
        -------
        grid : torch.Tensor
        center : np.ndarray
            Shape (3,), the physical (x, y, z) coordinate the grid's own
            midpoint corresponds to -- needed by ``_compute_geometry_fields``
            to reconstruct each voxel's absolute coordinate.
        """
        center = points.mean(axis=0)
        centered = points - center
        half_extent = float(np.abs(centered).max()) + 2 * self.v_size
        n = int(np.ceil(2 * half_extent / self.v_size))
        n += n % 2  # even, for a clean center voxel
        grid = torch.zeros((n, n, n), dtype=torch.float32)

        idx = np.round(centered / self.v_size + n / 2).astype(np.int64)
        valid = np.all((idx >= 0) & (idx < n), axis=1)
        idx = idx[valid]
        flat = torch.as_tensor(
            idx[:, 2] * n * n + idx[:, 1] * n + idx[:, 0], dtype=torch.int64
        )
        grid.view(-1).scatter_add_(0, flat, torch.ones(flat.shape[0]))
        return grid, center
