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


def _skeletonize_shell(
    shell_mask: np.ndarray, edge_border: int = 2, min_component_size: int = 20
) -> np.ndarray:
    """
    Thin a full-thickness bilayer shell mask down to a true 1-voxel-thin
    mid-thickness ridge -- port of CTS's ``vesskeletonize`` (``gen_memvol.m``):

    .. code-block:: matlab

        function skel = vesskeletonize(memvol)
        bw = bwdist(~memvol);                    % distance-to-boundary, inside the shell
        mask = rescale(imgradient3(bw))>0.5;      % high-gradient = near either edge
        skel = (bw.*~mask)>max(bw,[],'all')/2-1;  % keep near-max-distance, low-gradient voxels
        skel = ctsutil('edgeblank',skel,2);       % clear a small edge border
        skel = bwareaopen(skel,20);                % drop small disconnected noise components
        end

    Translation notes:

    - ``bwdist(~memvol)`` in MATLAB finds, per voxel, the distance to the
      nearest TRUE voxel of ``~memvol`` (i.e. nearest voxel OUTSIDE the
      shell) -- MATLAB's ``bwdist`` always measures to the nearest
      foreground (true) voxel, so reaching "distance to background" needs
      an explicit complement. ``scipy.ndimage.distance_transform_edt``
      has the OPPOSITE built-in convention: it measures, per voxel,
      distance to the nearest ZERO-valued (background) voxel of its input
      DIRECTLY -- so the correct call here is
      ``distance_transform_edt(shell_mask)`` with NO complement, not
      ``distance_transform_edt(~shell_mask)`` (verified against a
      hand-computed 1D case and a synthetic 3D hollow-shell case before
      relying on this: a spherical shell's ridge came out at mean radius
      7.92 vs. an expected 8.0 mid-thickness radius).
    - MATLAB's ``rescale`` maps an array's own [min, max] to [0, 1];
      reproduced here as a plain min-max normalization of the gradient
      magnitude (not just a divide-by-max, which would only coincide with
      a true rescale if the gradient's minimum happens to be exactly 0).
    - ``imgradient3`` computes the 3D gradient magnitude; reproduced via
      ``np.gradient`` per axis, combined as a Euclidean norm.
    - ``ctsutil('edgeblank', skel, 2)`` clears a small border around the
      whole array so nothing gets placed touching the local grid's own
      edge; reproduced by zeroing the outermost `edge_border` voxels on
      every face.
    - ``bwareaopen(skel, 20)`` removes connected components smaller than
      20 voxels; reproduced via ``scipy.ndimage.label`` + a size filter
      (`min_component_size`, default 20 -- CTS's own value; not
      independently re-derived for this codebase's own typical grid
      resolutions, so worth revisiting if it turns out too
      aggressive/lenient at very coarse or very fine voxel sizes).

    Parameters
    ----------
    shell_mask : np.ndarray
        Bool, shape (nz, ny, nx) -- the FULL-thickness bilayer footprint
        (both leaflets, head to tail).
    edge_border : int, optional
        Border width (voxels) to blank on every face. Default 2.
    min_component_size : int, optional
        Minimum connected-component size (voxels) to keep. Default 20.

    Returns
    -------
    np.ndarray
        Bool, shape (nz, ny, nx). A thin (ideally 1-voxel) ridge at the
        shell's mid-thickness -- used ONLY for candidate position
        sampling of membrane-embedded proteins (see
        ``BlobMembraneInstance.skeleton_mask``'s docstring for why this
        must stay separate from `shell_mask` itself, which is still what
        the overlap-ignore mechanism in ``_cts_placement.py`` uses).
    """
    bw = ndimage.distance_transform_edt(shell_mask)
    if bw.max() == 0:
        return np.zeros_like(shell_mask, dtype=bool)

    gz, gy, gx = np.gradient(bw)
    grad_mag = np.sqrt(gz**2 + gy**2 + gx**2)
    g_min, g_max = grad_mag.min(), grad_mag.max()
    grad_rescaled = (grad_mag - g_min) / (g_max - g_min) if g_max > g_min else grad_mag
    high_gradient = grad_rescaled > 0.5

    skel = (bw * ~high_gradient) > (bw.max() / 2 - 1)

    if edge_border > 0:
        skel[:edge_border] = False
        skel[-edge_border:] = False
        skel[:, :edge_border] = False
        skel[:, -edge_border:] = False
        skel[:, :, :edge_border] = False
        skel[:, :, -edge_border:] = False

    labeled, num = ndimage.label(skel)
    if num > 0:
        sizes = ndimage.sum(np.ones_like(labeled), labeled, index=np.arange(1, num + 1))
        small_labels = np.nonzero(sizes < min_component_size)[0] + 1
        if small_labels.size > 0:
            skel[np.isin(labeled, small_labels)] = False

    return skel


def _compute_geometry_fields(
    base_points: np.ndarray,
    alpha: float,
    density: torch.Tensor,
    center: np.ndarray,
    v_size: float,
    thickness: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Derive a shell (bilayer footprint) mask, a mid-thickness skeleton
    ridge, an interior (vesicle lumen) mask, and a local outward-normal
    vector field, on `density`'s own voxel grid.

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
        simply the (lightly dilated) occupied-voxel mask. This is the
        FULL bilayer thickness footprint (both leaflets, head to tail) --
        used by ``_cts_placement.py``'s overlap-ignore mechanism (a
        membrane-embedded protein is expected to displace the local lipid
        across its whole insertion footprint), NOT for candidate position
        sampling (see `skeleton_mask` for that).
    skeleton_mask : torch.Tensor
        Bool, shape (nz, ny, nx). A thin mid-thickness ridge through
        `shell_mask`, via ``_skeletonize_shell`` (port of CTS's
        ``vesskeletonize``) -- used ONLY for membrane-embedded protein
        CANDIDATE POSITION SAMPLING, so a placed protein's insertion depth
        is consistently centered in the bilayer rather than uniformly
        random anywhere from outer to inner leaflet (or genuinely
        mid-bilayer) as sampling from the full `shell_mask` would give.
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

    dist_out = ndimage.distance_transform_edt(~inside_base_blob)
    dist_in = ndimage.distance_transform_edt(inside_base_blob)
    signed_dist = dist_out - dist_in  # >0 outside the blob solid, <0 inside it

    gz, gy, gx = np.gradient(signed_dist)
    normal_xyz = np.stack([gx, gy, gz], axis=0)  # physical (x, y, z) component order
    mag = np.linalg.norm(normal_xyz, axis=0, keepdims=True)
    mag[mag < 1e-8] = 1.0
    normal_xyz = normal_xyz / mag

    continuous_density = None
    double_shell_bool = None
    if thickness is not None:
        # Continuous (non-point-cloud) bilayer density, directly from the
        # SAME signed-distance field already computed above for the normal
        # field -- replaces discrete head/tail point scattering, which
        # under-samples true atomic density by ~35-45x at typical cryo-ET
        # voxel sizes (verified empirically: ~1 point/voxel vs. ~46 real
        # atoms/voxel for organic matter at 10 A), causing visible gaps in
        # any single 2D slice even though the underlying shell shape itself
        # is a single connected surface (also verified: shell_mask is
        # always 100% one connected component). A real lipid bilayer's
        # density at typical cryo-ET resolution is a smooth, continuous
        # aggregate over many real atoms per voxel -- same reasoning
        # PotentialBuilder relies on for smooth, gap-free protein density
        # despite proteins also being made of discrete atoms. Mirrors
        # specimen/cryoet.py's analytic-shape membrane path (`_shell` +
        # `gaussian_filter`), just driven by this blob's own SDF instead of
        # a closed-form ellipsoid equation.
        # Anchored entirely on the TRUE, un-offset surface (signed_dist==0
        # -- always well-behaved regardless of local curvature/concavity,
        # since it involves no offsetting at all), not on two
        # independently-offset-then-thresholded leaflet surfaces. An
        # earlier version built each leaflet as its own offset-by-half_t
        # binary band, then blurred -- verified (visually, on a real
        # generated shape, not just reasoned about) to produce solid,
        # filled-looking cross-sections instead of a hollow bilayer for
        # sufficiently irregular blobs. Root cause: offsetting a non-
        # convex surface outward/inward by a fixed distance is a classic
        # ill-behaved operation -- once the offset distance approaches the
        # local curvature radius of a concave/convex feature, the offset
        # surface self-intersects (folds onto itself), which is exactly
        # what a solid-looking cross-section through an otherwise-thin
        # shell indicates. A single band centered on the un-offset surface
        # doesn't have this failure mode. The bilayer's two-leaflet
        # profile is then a smooth analytic function of the (already
        # correctly SDF-computed) continuous distance from that one
        # robust reference -- no independent thresholding of each leaflet,
        # no blur-based bridging risk between them either.
        half_t = (thickness / 2.0) / v_size
        leaflet_sigma = max(0.5, 0.25 * half_t)  # each leaflet peak's own width, voxels
        # `signed_dist` (dist_out - dist_in, both from distance_transform_edt
        # with no `sampling` argument) is ALREADY in raw voxel-index units --
        # scipy's EDT has no notion of physical spacing unless told, so it
        # just counts array steps. `half_t` above is likewise already in
        # voxel units (thickness/v_size). Dividing `signed_dist` by `v_size`
        # a second time here was a real bug: it silently rescaled the SDF by
        # an extra factor of `v_size`, so the two leaflet peaks (meant to
        # sit at +-half_t voxels from the true surface) actually only formed
        # at +-half_t*v_size true voxels out -- e.g. at v_size=10,
        # thickness=40 the intended +-2 voxel offset was actually only
        # reachable +-20 true voxels from the surface, nowhere near the
        # actual near-surface shell. Verified via a direct signed_dist-vs-
        # density binned profile on a real generated blob before/after this
        # fix.
        sd_vox = signed_dist

        outer_peak = np.exp(-0.5 * ((sd_vox - half_t) / leaflet_sigma) ** 2)
        inner_peak = np.exp(-0.5 * ((sd_vox + half_t) / leaflet_sigma) ** 2)
        profile = outer_peak + inner_peak

        band_half_width = half_t + 3 * leaflet_sigma  # generous margin past both peaks
        shell_region = np.abs(sd_vox) < band_half_width
        double_shell_bool = shell_region
        profile[~shell_region] = 0.0

        m = profile.max()
        if m > 0:
            profile = profile / m
        profile[profile < 0.02] = 0.0
        continuous_density = torch.as_tensor(profile, dtype=torch.float32)
        double_shell_bool = profile > 0  # tight, matches what's ACTUALLY rendered,
        # not the more generous pre-threshold band -- continuous rendering
        # doesn't need the old points-mode's large sparse-content safety
        # margin (see dilation note below), so keep this tight to leave
        # real interior lumen for vesicle-flagged placement.

    # shell_mask/skeleton_mask/interior_mask must be derived from whichever
    # footprint is ACTUALLY rendered (continuous_density when available),
    # not the old point-scatter density -- these two were previously
    # computed independently and could disconnect: shell_mask/skeleton_mask
    # from the sparse point cloud's footprint, while the real rendered
    # density was already the (much less sparse) continuous field. That
    # mismatch broke the overlap-ignore mechanism (a membrane-embedded
    # candidate sampled from the "skeleton" no longer reliably landed on
    # real rendered material, so overlap tests silently passed even
    # without the ignore mask) -- verified via
    # test_membrane_overlap_ignore_mask_not_conflated_with_skeleton.
    if double_shell_bool is not None:
        # A much smaller dilation than the old points-mode margin below --
        # continuous density is already a complete, non-sparse field, so
        # it only needs enough margin to cover rotate_volume's local
        # interpolation spread (~1 voxel), not the old large safety
        # margin that compensated for sparse point-cloud undercoverage.
        # Using the old iterations=3 margin here left almost no interior
        # lumen for smaller membranes, breaking vesicle-flagged placement
        # (verified: test_vesicle_and_cytosol_flags_respect_inside_outside
        # and friends went from passing to failing when this was still 3).
        shell_mask_np = ndimage.binary_dilation(double_shell_bool, iterations=1)
        # Continuous mode: the true mid-bilayer surface is already known
        # EXACTLY -- it's the un-offset original surface, signed_dist==0 --
        # so extract the skeleton directly from that, rather than
        # rediscovering it via _skeletonize_shell's generic bwdist-ridge
        # search. That generic approach is biased here: double_shell_bool
        # is symmetric in sd_vox VALUE-space (the two Gaussian peaks are
        # mirror images at +-half_t), but not in physical 3D VOLUME --
        # offsetting outward from a curved surface sweeps more voxels than
        # offsetting inward by the same distance, so the binary mask is
        # measurably "fatter" on the outer side. bwdist's max-distance
        # ridge naturally lands in the fattest part of the mask, i.e. the
        # outer leaflet, not the true center -- verified empirically
        # (skeleton points measured at signed_dist ~= +half_t, not ~=0,
        # which is exactly why sampling the profile outward from a
        # "skeleton" point only ever showed one Gaussian peak: the sampled
        # start point already WAS that peak). A direct threshold on
        # signed_dist has no such bias since it doesn't go through any
        # mask-volume/bwdist step at all.
        # signed_dist is already in voxel units (see the matching fix/note
        # a few lines up in the profile block) -- no division by v_size here.
        # Threshold at 1.5, not something smaller like 0.6: distance_
        # transform_edt on a boolean mask never actually produces values in
        # (-1, 1) -- a voxel is either inside (dist_in>=1, dist_out==0) or
        # outside (dist_out>=1, dist_in==0), so signed_dist jumps straight
        # from >=1 to <=-1 with nothing strictly between. A threshold below
        # 1.0 can never match any voxel at all (verified: gave an empty
        # skeleton_mask). 1.5 catches the one layer of voxels immediately
        # on each side of the true boundary -- a genuine ~2-voxel-thin ridge
        # centered on it.
        sd_vox = signed_dist
        mid_surface = np.abs(sd_vox) < 1.5
        if edge_border := 2:
            mid_surface[:edge_border] = False
            mid_surface[-edge_border:] = False
            mid_surface[:, :edge_border] = False
            mid_surface[:, -edge_border:] = False
            mid_surface[:, :, :edge_border] = False
            mid_surface[:, :, -edge_border:] = False
        skeleton_mask_np = mid_surface
    else:
        shell_mask_np = ndimage.binary_dilation(density.numpy() > 0, iterations=3)
        skeleton_mask_np = _skeletonize_shell(shell_mask_np)
    interior_mask_np = inside_base_blob & ~shell_mask_np

    return (
        torch.as_tensor(shell_mask_np),
        torch.as_tensor(skeleton_mask_np),
        torch.as_tensor(interior_mask_np),
        torch.as_tensor(normal_xyz, dtype=torch.float32),
        continuous_density,
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
        Bool, shape (nz, ny, nx) -- see ``_compute_geometry_fields``. Full
        bilayer thickness footprint -- for the overlap-ignore mechanism,
        NOT candidate sampling.
    skeleton_mask : torch.Tensor
        Bool, shape (nz, ny, nx) -- see ``_compute_geometry_fields``. Thin
        mid-thickness ridge -- for membrane-embedded candidate position
        sampling, NOT the overlap-ignore mechanism. Keep these two masks'
        uses separate downstream (see ``_cts_placement.py``'s
        `ignore_masks` plumbing): sampling from `shell_mask` directly
        would let a placed protein's insertion depth land anywhere from
        outer to inner leaflet at random, instead of consistently at
        mid-bilayer.
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
    skeleton_mask: torch.Tensor
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
        density_mode: str = "continuous",
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
            sparse interior ("tail") populations. Only used when
            `density_mode="points"` -- ignored (but still computed, since
            they also drive the shape/mask geometry) when "continuous".
        surface_jitter : float, optional
            Radial spread (Angstrom) applied to head points around the
            offset surfaces. Defaults to ``0.2 * thickness``.
        density_mode : {"continuous", "points"}, optional
            "continuous" (default): smooth SDF-based double-shell density
            (see `_compute_geometry_fields`) -- physically correct at
            typical cryo-ET voxel sizes, no risk of gaps in a 2D slice.
            "points": the original discrete head/tail point-scatter
            density -- kept available specifically for direct point-cloud-
            vs-continuous comparison, not recommended for real specimen
            generation (verified to under-sample true atomic density by
            ~35-45x at 10A voxels, and to leave visible gaps in any single
            2D slice despite the underlying shape being a single closed
            surface).

        Returns
        -------
        BlobMembraneInstance
        """
        if not 0.0 < roughness < 1.0:
            raise ValueError(f"roughness must be in (0, 1), got {roughness}")
        if density_mode not in ("continuous", "points"):
            raise ValueError(
                f"density_mode must be 'continuous' or 'points', got {density_mode!r}"
            )
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
        # Self-verifying grid sizing for continuous mode: an earlier
        # attempt tried to analytically predict the needed margin from
        # base_points' own extent, but that measurement turned out
        # inconsistent with what the SDF/shell construction actually
        # needs -- verified directly (not assumed): even after that
        # margin, the array's own outermost face still had ~85% nonzero
        # voxels reaching full peak brightness, meaning the shell was
        # genuinely hard-clipped at the array boundary (visible as a
        # square/cube artifact once rotated and inserted into a specimen
        # volume), not tapering to zero as it should. Rather than guess at
        # another formula, grow the grid and re-render until the boundary
        # is actually clean, checked directly each time.
        min_half_extent = 0.0
        for _attempt in range(4):
            point_density, raster_center = self._rasterize(
                all_points, min_half_extent=min_half_extent
            )
            (
                shell_mask,
                skeleton_mask,
                interior_mask,
                normal_field,
                continuous_density,
            ) = _compute_geometry_fields(
                base_points,
                alpha,
                point_density,
                raster_center,
                self.v_size,
                thickness=thickness if density_mode == "continuous" else None,
            )
            if density_mode != "continuous":
                break
            d = continuous_density.numpy()
            face_max = max(
                d[0, :, :].max(),
                d[-1, :, :].max(),
                d[:, 0, :].max(),
                d[:, -1, :].max(),
                d[:, :, 0].max(),
                d[:, :, -1].max(),
            )
            if face_max < 0.02:
                break
            # Boundary still has real content -- grow and retry.
            current_half_extent = point_density.shape[0] * self.v_size / 2.0
            min_half_extent = current_half_extent * 1.6
        density = continuous_density if density_mode == "continuous" else point_density
        return BlobMembraneInstance(
            density=density,
            thickness=thickness,
            v_size=self.v_size,
            shell_mask=shell_mask,
            skeleton_mask=skeleton_mask,
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

    def _rasterize(
        self, points: np.ndarray, min_half_extent: float = 0.0
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Bin `points` (already centered near the origin) into a local
        voxel grid via scatter-add -- torch equivalent of
        ``helper_atoms2vol.m``'s ``accumarray`` binning.

        Parameters
        ----------
        points : np.ndarray
        min_half_extent : float, optional
            Grid half-extent (Angstrom) is at least this large, even if
            `points` itself doesn't reach that far -- needed because the
            continuous SDF-based density (see `_compute_geometry_fields`)
            is driven by the alpha-shape's own true geometric extent, not
            by where the finite, randomly-sampled head/tail points happen
            to land. Sizing the grid from `points` alone (the old
            behavior, still fine for `density_mode="points"`) can clip the
            continuous shell at the array's own boundary wherever the true
            shape reaches farther than the sampled points did -- verified
            visually as a hard square/cube edge cutting across the
            membrane in a rendered slice, not a smooth taper to zero.
            Default 0.0 (no additional minimum, matching the old behavior).

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
        half_extent = max(
            float(np.abs(centered).max()) + 2 * self.v_size, min_half_extent
        )
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
