"""
Membrane geometry on a coarse grid.

A membrane is the boundary of a solid, held as a signed distance (negative
inside) on a coarse `Grid`. The solid is built from primitives
(`_primitives`), regularised by a rolling ball so that no curvature radius
and no half self-distance falls below ``r_min`` (`regularise`), and its
reach is measured (`reach`) so the two leaflets of the bilayer always fit.
The rendered potential is read off the distance later (`_render`), so the
output resolution never enters the geometry.

Arrays here are indexed ``(x, y, z)``; `Grid.lo` is the physical position of
the box's low corner and the grid extends `Grid.pad` past the box on every
side, so a solid that leaves the box is clipped by the renderer rather than
by the grid.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Rolling-ball radius, A: the smallest curvature radius and half the smallest
# self-distance a regularised mid-surface may have.
R_MIN = 30.0
# Accepted reach, A. The leaflets of a 37 A bilayer sit 18.5 A either side of
# the mid-surface.
MIN_REACH = 25.0
# Field value standing for "no surface anywhere", A.
_FAR = 1e4
# Nodes refined per closest-point pass.
_REFINE_CHUNK = 1 << 23


class Grid:
    """
    Coarse working grid of a membrane's signed distance.

    Parameters
    ----------
    box : sequence of float
        Physical extent ``(x, y, z)`` of the box the membrane is rendered
        into, Å.
    h : float, optional
        Node spacing, Å. Default 10.0.
    pad : float, optional
        Margin of grid kept past each face of the box, Å. Default
        ``2 * r_min + 4 * h``, which the rolling ball needs.
    r_min : float, optional
        Rolling-ball radius, Å. Default `R_MIN`.
    lo : sequence of float, optional
        Physical ``(x, y, z)`` of the box's low corner, Å. Default the
        origin.
    device : str or torch.device, optional
        Device the fields live on. Default ``"cpu"``.
    """

    def __init__(
        self,
        box: Sequence[float],
        h: float = 10.0,
        pad: float | None = None,
        r_min: float = R_MIN,
        lo: Sequence[float] = (0.0, 0.0, 0.0),
        device: str | torch.device = "cpu",
    ) -> None:
        self.box = (float(box[0]), float(box[1]), float(box[2]))
        self.lo = (float(lo[0]), float(lo[1]), float(lo[2]))
        self.h = float(h)
        self.r_min = float(r_min)
        self.pad = 2 * self.r_min + 4 * self.h if pad is None else float(pad)
        self.device = torch.device(device)
        self.shape = (
            int(math.ceil((self.box[0] + 2 * self.pad) / self.h)) + 1,
            int(math.ceil((self.box[1] + 2 * self.pad) / self.h)) + 1,
            int(math.ceil((self.box[2] + 2 * self.pad) / self.h)) + 1,
        )
        self.origin = torch.tensor(
            [c - self.pad for c in self.lo], dtype=torch.float32, device=self.device
        )

    @property
    def n_nodes(self) -> int:
        """Number of grid nodes."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def centre(self) -> np.ndarray:
        """Physical centre of the box, Å."""
        return np.array(self.lo) + np.array(self.box) / 2

    def axes(self) -> list[torch.Tensor]:
        """
        Node coordinates per axis, as three mutually broadcastable views.

        Returns
        -------
        list of torch.Tensor
            Shapes ``(X, 1, 1)``, ``(1, Y, 1)`` and ``(1, 1, Z)``, Å.
        """
        out = []
        for k, n in enumerate(self.shape):
            a = self.origin[k] + self.h * torch.arange(
                n, device=self.device, dtype=torch.float32
            )
            out.append(a.view(*[-1 if i == k else 1 for i in range(3)]))
        return out

    def nodes(self, lin: torch.Tensor) -> torch.Tensor:
        """Physical positions ``(N, 3)`` of the nodes at linear indices `lin`."""
        sy, sz = self.shape[1], self.shape[2]
        ijk = torch.stack([lin // (sy * sz), (lin // sz) % sy, lin % sz], 1)
        return self.origin + self.h * ijk.float()

    def linear_index(self, points: torch.Tensor) -> torch.Tensor:
        """Linear index of the node nearest each of `points` ``(N, 3)``."""
        ijk = ((points - self.origin) / self.h).round().long()
        for k in range(3):
            ijk[:, k].clamp_(0, self.shape[k] - 1)
        return (ijk[:, 0] * self.shape[1] + ijk[:, 1]) * self.shape[2] + ijk[:, 2]


def _nearest_feature_gpu(mask: torch.Tensor) -> torch.Tensor | None:
    """Nearest-True-voxel index on the GPU through cupy, or None if it cannot."""
    try:
        import cupy
        import cupyx.scipy.ndimage as cndi
    except ImportError:
        return None
    try:
        # cupy keeps its own pool and cannot use what torch has cached.
        torch.cuda.empty_cache()
        with cupy.cuda.Device(mask.device.index or 0):
            inverted = cupy.from_dlpack((~mask).to(torch.uint8))
            idx = cndi.distance_transform_edt(
                inverted,
                return_distances=False,
                return_indices=True,
                float64_distances=False,
            )
            out = torch.from_dlpack(idx).clone()
            del inverted, idx
            cupy.get_default_memory_pool().free_all_blocks()
        return out
    except Exception as exc:  # noqa: BLE001 - any GPU failure falls back to scipy
        warnings.warn(
            f"membrane geometry: the GPU distance transform failed ({exc!r}); "
            "running it on the CPU instead.",
            stacklevel=3,
        )
        return None


def _nearest_feature(mask: torch.Tensor) -> torch.Tensor:
    """Index ``(3, X, Y, Z)`` of the nearest True voxel of `mask`."""
    if mask.device.type == "cuda":
        out = _nearest_feature_gpu(mask)
        if out is not None:
            return out
    from scipy.ndimage import distance_transform_edt

    idx = distance_transform_edt(
        ~mask.cpu().numpy(), return_distances=False, return_indices=True
    )
    return torch.from_numpy(np.ascontiguousarray(idx)).to(mask.device)


def _distance_to_points(
    grid: Grid, lin: torch.Tensor, pts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Distance from every node to the point stored at its nearest feature node.

    Parameters
    ----------
    grid : Grid
        Working grid.
    lin : torch.Tensor
        Linear indices of the feature nodes, shape ``(N,)``.
    pts : torch.Tensor
        The sub-voxel point each feature node stands for, shape ``(N, 3)``.

    Returns
    -------
    distance : torch.Tensor
        Shape ``grid.shape``, Å.
    nearest : torch.Tensor
        Row of `pts` nearest each node, shape ``grid.shape``.
    """
    dev = grid.device
    mask = torch.zeros(grid.n_nodes, dtype=torch.bool, device=dev)
    mask[lin] = True
    idmap = torch.zeros(grid.n_nodes, dtype=torch.int32, device=dev)
    idmap[lin] = torch.arange(lin.numel(), dtype=torch.int32, device=dev)
    nn = _nearest_feature(mask.view(grid.shape))
    sy, sz = grid.shape[1], grid.shape[2]
    near = nn[0].long().mul_(sy).add_(nn[1]).mul_(sz).add_(nn[2])
    near = idmap[near].long()
    del nn, mask, idmap
    d2 = torch.zeros(grid.shape, device=dev)
    for k, ax in enumerate(grid.axes()):
        d2 += (ax - pts[:, k][near]) ** 2
    return d2.sqrt_(), near


_POLE = math.sqrt(3.0) - 2.0


def prefilter(vol: torch.Tensor) -> torch.Tensor:
    """
    Cubic B-spline coefficients whose spline interpolates `vol`.

    Parameters
    ----------
    vol : torch.Tensor
        Samples, shape ``(..., X, Y, Z)``.

    Returns
    -------
    torch.Tensor
        Coefficients, same shape.
    """
    k = torch.arange(-12, 13, device=vol.device, dtype=torch.float32)
    sign = torch.where(k.long() % 2 == 0, 1.0, -1.0)
    w = (-6 * _POLE / (1 - _POLE * _POLE)) * abs(_POLE) ** k.abs() * sign
    v = vol.reshape(-1, 1, *vol.shape[-3:])
    for ax in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = -1
        pad = [0] * 6
        pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = 12
        v = F.conv3d(F.pad(v, pad, mode="replicate"), w.view(shape))
    return v.reshape(vol.shape)


def bspline_taps(
    u: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Two linear taps per axis that sum to the cubic B-spline.

    Parameters
    ----------
    u : torch.Tensor
        Positions in index coordinates.

    Returns
    -------
    tuple
        Two ``(weight, position)`` pairs; a linear interpolation at each
        position, weighted and summed, is the cubic B-spline at `u`.
    """
    i = torch.floor(u)
    t = u - i
    w0 = (1 - t) ** 3 / 6
    w1 = (3 * t**3 - 6 * t**2 + 4) / 6
    w3 = t**3 / 6
    w2 = 1 - w0 - w1 - w3
    g0, g1 = w0 + w1, w2 + w3
    return (g0, i - 1 + w1 / g0), (g1, i + 1 + w3 / g1)


# Points per CPU thread below which `_sample_linear` stays on one thread.
_CPU_SAMPLE_CHUNK = 4096


def _sample_linear(coef: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    Trilinear samples of ``(C, X, Y, Z)`` volumes at normalised points.

    On the CPU `grid_sample` threads over the batch axis alone, so the
    points are laid out as a batch over one shared (expanded) volume.
    """
    n = g.shape[0]
    rows = 1
    if coef.device.type == "cpu":
        rows = max(1, min(torch.get_num_threads(), n // _CPU_SAMPLE_CHUNK))
    cols = -(-n // rows)
    if rows * cols != n:
        g = torch.cat([g, g[:1].expand(rows * cols - n, 3)])
    out = F.grid_sample(
        coef[None].expand(rows, *coef.shape),
        g.view(rows, cols, 1, 1, 3),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[:, :, :, 0, 0]
    return out.permute(1, 0, 2).reshape(coef.shape[0], rows * cols)[:, :n]


def sample(coef: torch.Tensor, grid: Grid, pts: torch.Tensor) -> torch.Tensor:
    """
    Cubic B-spline samples of coefficient volumes at arbitrary points.

    Parameters
    ----------
    coef : torch.Tensor
        Coefficients from `prefilter`, shape ``(C, X, Y, Z)``.
    grid : Grid
        Grid the coefficients live on.
    pts : torch.Tensor
        Physical points, shape ``(N, 3)``, Å.

    Returns
    -------
    torch.Tensor
        Shape ``(C, N)``.
    """
    size = torch.tensor(grid.shape, device=coef.device, dtype=torch.float32)
    taps = bspline_taps((pts - grid.origin) / grid.h)
    out: torch.Tensor | None = None
    for a in range(8):
        sel = [(a >> k) & 1 for k in range(3)]
        w = torch.stack([taps[sel[k]][0][:, k] for k in range(3)]).prod(0)
        pos = torch.stack([taps[sel[k]][1][:, k] for k in range(3)], 1)
        g = (pos / (size - 1) * 2 - 1).flip(-1)
        term = w * _sample_linear(coef, g)
        out = term if out is None else out + term
    assert out is not None
    return out


def _field_of(phi: torch.Tensor, grid: Grid) -> torch.Tensor:
    """``(4, X, Y, Z)``: spline coefficients of `phi` and of its gradient."""
    grad = torch.stack(torch.gradient(phi, spacing=grid.h))
    return prefilter(torch.cat([phi[None], grad]))


def surface_points(
    phi: torch.Tensor, grid: Grid, newton: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Sub-voxel points on ``{phi = 0}``, one per node next to a sign change.

    Parameters
    ----------
    phi : torch.Tensor
        Signed field, shape ``grid.shape``.
    grid : Grid
        Working grid.
    newton : int, optional
        Newton steps past the first. Default 2.

    Returns
    -------
    tuple or None
        ``(lin, points, normals, field)``: the nodes' linear indices, the
        surface points ``(N, 3)``, unit outward normals ``(N, 3)`` and the
        spline coefficients of `phi` and its gradient. None when `phi` has
        no zero crossing.
    """
    s = phi > 0
    band = torch.zeros_like(s)
    for ax in range(3):
        a: list[slice] = [slice(None)] * 3
        b: list[slice] = [slice(None)] * 3
        a[ax], b[ax] = slice(1, None), slice(None, -1)
        diff = s[tuple(a)] != s[tuple(b)]
        band[tuple(a)] |= diff
        band[tuple(b)] |= diff
    lin = band.view(-1).nonzero()[:, 0]
    if lin.numel() == 0:
        return None
    x0 = grid.nodes(lin)
    field = _field_of(phi, grid)
    p = x0.clone()
    for _ in range(newton + 1):
        v = sample(field, grid, p)
        g = v[1:].T
        p = p - (v[0] / g.pow(2).sum(1).clamp_min(1e-6))[:, None] * g
        # Never leave the node's neighbourhood: a crease has no reliable
        # gradient to follow.
        off = p - x0
        norm = off.norm(dim=1, keepdim=True).clamp_min(1e-6)
        p = x0 + off * (norm.clamp(max=1.5 * grid.h) / norm)
    normal = F.normalize(sample(field[1:], grid, p).T, dim=1)
    return lin, p, normal, field


def redistance(
    phi: torch.Tensor, grid: Grid, refine_within: float | None = None, iters: int = 3
) -> torch.Tensor:
    """
    Signed distance to ``{phi = 0}``, accurate below the node spacing.

    The nearest surface sample (one per boundary node, found by a distance
    transform) seeds a closest-point iteration on the spline's zero set:
    project the node onto the sample's tangent plane, pull that foot back
    onto the surface, repeat.

    Parameters
    ----------
    phi : torch.Tensor
        Any field whose sign marks the solid, shape ``grid.shape``.
    grid : Grid
        Working grid.
    refine_within : float, optional
        Nodes further than this from the surface keep the seed distance, Å.
        Default ``2 * grid.r_min + 4 * grid.h``.
    iters : int, optional
        Closest-point iterations. Default 3.

    Returns
    -------
    torch.Tensor
        Signed distance, shape ``grid.shape``, Å.
    """
    found = surface_points(phi, grid)
    if found is None:
        return torch.where(phi > 0, _FAR, -_FAR).to(torch.float32)
    lin, p, _, field = found
    d, near = _distance_to_points(grid, lin, p)
    if refine_within is None:
        refine_within = 2 * grid.r_min + 4 * grid.h
    dflat, nflat = d.view(-1), near.view(-1)
    todo = (dflat < refine_within).nonzero()[:, 0]
    for i0 in range(0, todo.numel(), _REFINE_CHUNK):
        idx = todo[i0 : i0 + _REFINE_CHUNK]
        x = grid.nodes(idx)
        q = p[nflat[idx]]
        best = dflat[idx]
        for _ in range(iters):
            n = F.normalize(sample(field[1:], grid, q).T, dim=1)
            move = (x - q) - ((x - q) * n).sum(1, keepdim=True) * n
            norm = move.norm(dim=1, keepdim=True).clamp_min(1e-6)
            f = q + move * (norm.clamp(max=2 * grid.h) / norm)
            for _ in range(2):
                v = sample(field, grid, f)
                g = v[1:].T
                f = f - (v[0] / g.pow(2).sum(1).clamp_min(1e-6))[:, None] * g
            ok = sample(field[:1], grid, f)[0].abs() < 0.05 * grid.h
            dn = (x - f).norm(dim=1)
            better = ok & (dn < best)
            q = torch.where(better[:, None], f, q)
            best = torch.where(better, dn, best)
        dflat[idx] = best
    return torch.where(phi > 0, d, -d)


def skeleton_solid(
    grid: Grid, pts: np.ndarray | torch.Tensor, radii: np.ndarray | float
) -> torch.Tensor:
    """
    Signed distance of the union of balls centred on skeleton samples.

    Parameters
    ----------
    grid : Grid
        Working grid.
    pts : array-like
        Skeleton samples, shape ``(N, 3)``, Å. Their spacing should be
        below the node spacing.
    radii : array-like or float
        Ball radius per sample, Å.

    Returns
    -------
    torch.Tensor
        Signed distance, shape ``grid.shape``. Samples outside the grid are
        dropped.
    """
    dev = grid.device
    points = torch.as_tensor(np.asarray(pts), dtype=torch.float32, device=dev)
    points = points.reshape(-1, 3)
    r = torch.as_tensor(
        np.broadcast_to(np.asarray(radii, dtype=np.float32), (points.shape[0],)).copy(),
        device=dev,
    )
    ijk = torch.round((points - grid.origin) / grid.h).long()
    size = torch.tensor(grid.shape, device=dev)
    keep = ((ijk >= 0) & (ijk < size)).all(1)
    ijk, points, r = ijk[keep], points[keep], r[keep]
    if ijk.shape[0] == 0:
        return torch.full(grid.shape, _FAR, device=dev)
    lin = (ijk[:, 0] * grid.shape[1] + ijk[:, 1]) * grid.shape[2] + ijk[:, 2]
    uniq, inv = torch.unique(lin, return_inverse=True)
    cnt = torch.zeros(uniq.numel(), device=dev).index_add_(0, inv, torch.ones_like(r))
    mean_p = torch.zeros(uniq.numel(), 3, device=dev).index_add_(0, inv, points)
    mean_p = mean_p / cnt[:, None]
    mean_r = torch.zeros(uniq.numel(), device=dev).index_add_(0, inv, r) / cnt
    d, near = _distance_to_points(grid, uniq, mean_p)
    return d - mean_r[near]


def smooth_noise(
    grid: Grid, wavelength: float, rng: np.random.Generator
) -> torch.Tensor:
    """
    Unit-variance Gaussian random field on the grid.

    Parameters
    ----------
    grid : Grid
        Working grid.
    wavelength : float
        Typical wavelength of the field, Å. The Gaussian filter's width is
        a quarter of it.
    rng : numpy.random.Generator
        Source of the field's seed.

    Returns
    -------
    torch.Tensor
        Shape ``grid.shape``.
    """
    # Drawn on the host so a seed gives the same field on every device.
    g = torch.Generator().manual_seed(int(rng.integers(2**31)))
    w = torch.randn(grid.shape, generator=g).to(grid.device)
    ks = [torch.fft.fftfreq(n, d=grid.h, device=grid.device) for n in grid.shape[:2]]
    ks.append(torch.fft.rfftfreq(grid.shape[2], d=grid.h, device=grid.device))
    k2 = ks[0].view(-1, 1, 1) ** 2 + ks[1].view(1, -1, 1) ** 2 + ks[2] ** 2
    sigma = wavelength / 4
    out = torch.fft.irfftn(
        torch.fft.rfftn(w) * torch.exp(-2 * (math.pi * sigma) ** 2 * k2), s=grid.shape
    )
    return out / out.std()


def blur(phi: torch.Tensor, sigma_nodes: float) -> torch.Tensor:
    """Gaussian blur of a field, `sigma_nodes` in node spacings."""
    half = max(1, int(math.ceil(3 * sigma_nodes)))
    x = torch.arange(-half, half + 1, device=phi.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma_nodes) ** 2)
    k = k / k.sum()
    v = phi[None, None]
    for ax in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = -1
        pad = [0] * 6
        pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = half
        v = F.conv3d(F.pad(v, pad, mode="replicate"), k.view(shape))
    return v[0, 0]


def _roll(phi: torch.Tensor, grid: Grid, r: float, smooth: float) -> torch.Tensor:
    """Dilate by r, erode by 2r, dilate by r, then smooth."""
    b = redistance(redistance(phi, grid) - r, grid)
    d = redistance(b + 2 * r, grid)
    e = redistance(d - r, grid)
    if smooth > 0:
        e = redistance(blur(e, smooth), grid)
    return e


class ReachError(ValueError):
    """A membrane's reach stayed below the accepted minimum after repair."""


def regularise(
    phi: torch.Tensor,
    grid: Grid,
    r: float | None = None,
    smooth: float = 1.0,
    min_reach: float = MIN_REACH,
    rounds: int = 5,
) -> torch.Tensor:
    """
    Rolling-ball closing then opening at radius `r`, then a one-node smoothing.

    A closing followed by an opening is not idempotent: the opening can
    bridge two pockets of the complement through a neck thinner than the
    bilayer. The reach is therefore measured, and wherever it falls below
    `min_reach` the thin side is filled (a neck of solvent becomes lumen, a
    web of lumen becomes solvent) and the field is rolled again.

    Parameters
    ----------
    phi : torch.Tensor
        Field whose sign marks the solid, shape ``grid.shape``.
    grid : Grid
        Working grid.
    r : float, optional
        Rolling-ball radius, Å. Default ``grid.r_min``.
    smooth : float, optional
        Width of the final Gaussian smoothing, node spacings. Default 1.0.
    min_reach : float, optional
        Accepted reach, Å. Default `MIN_REACH`.
    rounds : int, optional
        Repairs attempted. Default 5.

    Returns
    -------
    torch.Tensor
        Signed distance of the regularised solid, shape ``grid.shape``.

    Raises
    ------
    ReachError
        If `rounds` repairs leave a reach below `min_reach`, so a returned
        field always has room for its leaflets.
    """
    r = grid.r_min if r is None else r
    e = _roll(phi, grid, r, smooth)
    for _ in range(rounds):
        p, _, out, inn = reach(e, grid)
        bad_o, bad_i = out < min_reach, inn < min_reach
        if not bool((bad_o | bad_i).any()):
            return e
        shift = torch.zeros(grid.n_nodes, device=grid.device)
        lin = grid.linear_index(p)
        shift[lin[bad_i]] = 1.0
        shift[lin[bad_o]] = -1.0
        shift = shift.view(1, 1, *grid.shape)
        k = 2 * int(math.ceil(r / grid.h)) + 1
        shift = F.max_pool3d(shift, k, 1, k // 2) - F.max_pool3d(-shift, k, 1, k // 2)
        e = _roll(e + 1.5 * r * blur(shift[0, 0], 1.5), grid, r, smooth)
    _, _, out, inn = reach(e, grid)
    if out.numel():
        worst = float(torch.minimum(out, inn).min())
        if worst < min_reach:
            raise ReachError(
                f"membrane reach {worst:.1f} A is below {min_reach} A "
                f"after {rounds} repairs"
            )
    return e


def reach(
    phi: torch.Tensor, grid: Grid, s_max: float = 80.0, n: int = 33, tol: float = 2.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    How far the field stays an exact distance along each surface normal.

    A reach of ``s`` at a point means no curvature radius below ``s`` and no
    other part of the membrane within ``2 s`` on that side; leaflets at
    half the bilayer thickness need a reach above it.

    Parameters
    ----------
    phi : torch.Tensor
        Signed distance, shape ``grid.shape``.
    grid : Grid
        Working grid.
    s_max : float, optional
        Largest reach measured, Å. Default 80.0.
    n : int, optional
        Samples along the normal, including zero. Default 33.
    tol : float, optional
        Departure from an exact distance that ends the reach, Å. Default 2.0.

    Returns
    -------
    points, normals, outer, inner : torch.Tensor
        Surface points inside the box ``(N, 3)``, their unit normals, and
        the reach on the outer and inner side, Å.
    """
    dev = grid.device
    found = surface_points(phi, grid)
    if found is None:
        z = torch.zeros(0, device=dev)
        return torch.zeros(0, 3, device=dev), torch.zeros(0, 3, device=dev), z, z
    _, p, nrm, _ = found
    lo = torch.tensor(grid.lo, device=dev)
    hi = lo + torch.tensor(grid.box, device=dev)
    inside = ((p > lo) & (p < hi)).all(1)
    p, nrm = p[inside], nrm[inside]
    coef = prefilter(phi)[None]
    ss = torch.linspace(0, s_max, n, device=dev)[1:]
    step = float(ss[1] - ss[0])
    out = torch.full((p.shape[0],), s_max, device=dev)
    inn = out.clone()
    for sgn, res in ((1.0, out), (-1.0, inn)):
        alive = torch.ones(p.shape[0], dtype=torch.bool, device=dev)
        for s in ss:
            v = sample(coef, grid, p + sgn * s * nrm)[0] * sgn
            bad = alive & (v < s - tol)
            res[bad] = float(s) - step
            alive &= ~bad
    return p, nrm, out, inn


def surface_mesh(phi: torch.Tensor, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangle mesh of ``{phi = 0}``, for inspection and figures.

    Returns
    -------
    vertices : numpy.ndarray
        Physical ``(x, y, z)``, shape ``(V, 3)``, Å.
    faces : numpy.ndarray
        Vertex indices, shape ``(F, 3)``.
    """
    from skimage.measure import marching_cubes

    v, f, _, _ = marching_cubes(phi.cpu().numpy(), 0.0, spacing=(grid.h,) * 3)
    return v + grid.origin.cpu().numpy(), f


__all__ = [
    "MIN_REACH",
    "R_MIN",
    "Grid",
    "ReachError",
    "blur",
    "bspline_taps",
    "prefilter",
    "reach",
    "redistance",
    "regularise",
    "sample",
    "skeleton_solid",
    "smooth_noise",
    "surface_mesh",
    "surface_points",
]
