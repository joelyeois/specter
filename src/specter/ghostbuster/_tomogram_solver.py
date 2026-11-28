"""
Full-batch tomogram reconstruction: the volume that minimises the tilt
series' chi-squared plus `TomogramReconstructor`'s priors, found by
preconditioned nonlinear conjugate gradients.

A tilt series holds far fewer pixels than its tomogram holds voxels, so the
data term alone has no unique minimiser, and a fit without a prior ends in
whatever its optimiser's path through the null space happens to reach. The
solver here minimises

    1/2 sum_i sum_p (y_ip - m_ip(V))^2 / n_i  +  R(V)

over every tilt ``i`` and pixel ``p`` at once, where ``m`` is
`TomogramReconstructor`'s prediction, ``n_i`` the tilt's expected counts per
pixel (its Poisson variance) and ``R`` the reconstructor's
:meth:`~TomogramReconstructor.regularisation`. With the single-scatter
models the whole series is one non-uniform FFT of the volume
(`specter.scattering.surface_slice_sum`), so a gradient over every tilt
costs about what one tilt of the slice loop does.

The step along each search direction is the minimum of a parabola through
the objective at the current volume and at the previous step length. With a
quadratic prior alone the search directions are preconditioned by
``1 / (W(k) + lambda(k))``, ``W`` the data term's curvature per Fourier
component and ``lambda`` the prior's. Without a prior no preconditioner is
applied, since it would select a different member of the data term's null
space rather than reach the same one faster.

A volume too large for one device is solved in strips along the tilt axis
(`solve_tilt_series_in_strips`). A ray keeps its coordinate along the tilt
axis at every tilt, so a strip of the volume projects onto the same band of
pixels in every image, and neighbouring strips are coupled only by the
distance the transfer function displaces the image and by the prior. Each
strip is therefore an independent problem of the same form, solved by
`solve_tilt_series` on the pixels whose rays stay inside it, and the strips
are cross-faded where they overlap.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Sequence

import roma
import torch

from specter import logger

from ..constants import interaction_parameter

if TYPE_CHECKING:
    from ._tomogram_reconstructor import TomogramReconstructor

__all__ = [
    "StripPlan",
    "predict_tilt_series",
    "solve_tilt_series",
    "solve_tilt_series_in_strips",
]

#: Preconditioner floor, as a fraction of the largest curvature.
_CURVATURE_FLOOR = 1e-4

#: Phase shift, in radians, of the voxel that probes the data term's curvature.
_PROBE_PHASE = 1e-2

#: Relative decrease of the objective below which an iteration is not
#: resolved: the forward model is evaluated in single precision, whose
#: rounding moves the objective by about this fraction of its value.
_OBJECTIVE_RESOLUTION = 1e-7

#: Largest drift of a ray along the strip axis, in voxels over half the
#: volume's depth, for the series to count as tilting about that axis.
_AXIS_TOLERANCE = 0.25

#: Smallest half-width, in voxels, of the cross-fade between two strips.
_MIN_FADE = 8

#: Voxels over which a strip is tapered to zero at its inner faces.
_TAPER = 8


def _gradient_spectrum(
    shape: tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    """
    Eigenvalues of the squared forward-difference gradient, in ``rfftn`` layout.

    ``sum |grad V|^2 = sum_k L(k) |V(k)|^2 / N`` on a periodic grid, with
    ``L(k) = 4 sum_axis sin^2(pi f_axis)``.
    """
    fz = torch.fft.fftfreq(shape[0], device=device)[:, None, None]
    fy = torch.fft.fftfreq(shape[1], device=device)[None, :, None]
    fx = torch.fft.rfftfreq(shape[2], device=device)[None, None, :]
    return 4 * (
        torch.sin(math.pi * fz) ** 2
        + torch.sin(math.pi * fy) ** 2
        + torch.sin(math.pi * fx) ** 2
    )


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    """The inner product of two volumes, summed in double precision."""
    return float((a * b).sum(dtype=torch.float64))


class _Objective:
    """The solver's objective over the fitted tilts, with full-batch gradients."""

    def __init__(
        self,
        model: "TomogramReconstructor",
        images: torch.Tensor,
        tilts: torch.Tensor,
        tilt_batch: int,
    ) -> None:
        self.model = model
        self.images = images
        self.tilts = tilts
        self.tilt_batch = tilt_batch
        self.passes = 0

    def _data(self, V: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        sim = self.model._forward_tilts(self.model._taper(V), idx)
        return self.model._half_chi2(sim, self.images[idx].to(V.device), idx)

    def value(self, V: torch.Tensor) -> float:
        """The objective at ``V``."""
        total = 0.0
        with torch.no_grad():
            for idx in self.tilts.split(self.tilt_batch):
                total += float(self._data(V, idx))
                self.passes += 1
            total += float(self.model.regularisation(V))
        return total

    def value_and_gradient(self, V: torch.Tensor) -> tuple[float, float, torch.Tensor]:
        """The data term, the prior and the objective's gradient at ``V``."""
        gradient = torch.zeros_like(V)
        data = 0.0
        for idx in self.tilts.split(self.tilt_batch):
            v = V.detach().requires_grad_(True)
            loss = self._data(v, idx)
            (g,) = torch.autograd.grad(loss, v)
            gradient += g
            data += float(loss)
            self.passes += 1
        v = V.detach().requires_grad_(True)
        prior = self.model.regularisation(v)
        if prior.requires_grad:
            (g,) = torch.autograd.grad(prior, v)
            gradient += g
        return data, float(prior), gradient

    def curvature(
        self, shape: tuple[int, int, int], device: torch.device
    ) -> torch.Tensor:
        """
        The data term's curvature per Fourier component, in ``rfftn`` layout.

        The normal operator's response to a voxel at the volume's centre,
        taken as shift invariant: its transform is the curvature. The
        response is probed against the images the empty volume predicts
        rather than the observed ones. The observed images contribute the
        same large term to the gradient at either sign of the probe, and
        the difference of two such gradients keeps their rounding error,
        which is as large as the response itself; against the empty
        volume's prediction that term is absent and the gradient is the
        response alone. The odd part about the empty volume removes what
        the model's curvature adds. The probe is the potential that shifts
        the phase of a voxel by ``_PROBE_PHASE``.
        """
        centre = tuple(s // 2 for s in shape)
        sigma = interaction_parameter(self.model.voltage)
        step = _PROBE_PHASE / (sigma * self.model.voxel_size)
        empty = torch.zeros(shape, device=device)
        probe = torch.zeros(shape, device=device)
        response = torch.zeros(shape, device=device)
        for idx in self.tilts.split(self.tilt_batch):
            with torch.no_grad():
                base = self.model._forward_tilts(self.model._taper(empty), idx)
            for sign in (1.0, -1.0):
                probe[centre] = sign * step
                v = probe.clone().requires_grad_(True)
                sim = self.model._forward_tilts(self.model._taper(v), idx)
                (g,) = torch.autograd.grad(self.model._half_chi2(sim, base, idx), v)
                response.add_(g, alpha=sign / (2 * step))
        response = torch.roll(response, tuple(-c for c in centre), dims=(0, 1, 2))
        return torch.fft.rfftn(response).real.clamp_min(0)


def solve_tilt_series(
    model: "TomogramReconstructor",
    images: torch.Tensor,
    n_iter: int,
    *,
    tilts: torch.Tensor | None = None,
    tilt_batch: int = 8,
    callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """
    Fit ``model.V`` to a whole tilt series by conjugate gradients, in place.

    Parameters
    ----------
    model : TomogramReconstructor
        Supplies the forward model, the loss mask and the priors, and holds
        the volume, which is updated in place. Must be on the device to
        solve on.
    images : torch.Tensor
        The observed tilt series in electron counts, ``(N_tilts, H, W)``.
    n_iter : int
        Most conjugate-gradient iterations to take. An edge-preserving
        prior (``tv_weight``) needs on the order of 100 to 200; a quadratic
        prior alone (``smoothness``) is preconditioned and needs about 20.
        The preconditioned solve ends sooner, once an iteration lowers the
        objective by less than its rounding error.
    tilts : torch.Tensor, optional
        Indices of the tilts to fit. Default all of them. Leaving tilts out
        and scoring the prediction against them is a test of the
        reconstruction that needs no ground truth.
    tilt_batch : int, optional
        Tilts predicted per forward pass. The result does not depend on it;
        device memory grows with it. Default 8.
    callback : callable, optional
        Called as ``callback(iteration, entry)`` after every iteration with
        that iteration's log entry.

    Returns
    -------
    list of dict
        One entry per iteration taken: ``objective``, ``data`` (the chi-squared
        term), ``prior``, ``chi2_per_pixel``, ``step`` and ``seconds``.

    Notes
    -----
    The directions are Polak-Ribiere+ conjugate gradients. Each iteration
    costs one gradient over every tilt and one or more evaluations of the
    objective for the line search.
    """
    # The volume is a parameter when a learning rate was given and a buffer
    # otherwise; the solver updates either in place.
    V = model.V.data if isinstance(model.V, torch.nn.Parameter) else model.V
    device = V.device
    shape = (int(V.shape[0]), int(V.shape[1]), int(V.shape[2]))
    tilts = torch.arange(len(model.quaternions)) if tilts is None else tilts.cpu()
    objective = _Objective(model, images, tilts, tilt_batch)
    n_pixels = sum(model._n_valid_pixels(int(i)) for i in tilts)

    preconditioner: torch.Tensor | None = None
    if model.smoothness > 0 and model.tv_weight == 0:
        W = objective.curvature(shape, device)
        prior = model.smoothness * _gradient_spectrum(shape, device)
        preconditioner = 1.0 / (W + prior + _CURVATURE_FLOOR * float(W.max()))
        del W, prior

    def precondition(g: torch.Tensor) -> torch.Tensor:
        if preconditioner is None:
            return g
        return torch.fft.irfftn(torch.fft.rfftn(g) * preconditioner, s=g.shape)

    start = time.time()
    log: list[dict[str, Any]] = []
    data, prior_value, g = objective.value_and_gradient(V)
    f = data + prior_value
    z = precondition(g)
    d = -z
    gz = _dot(g, z)
    step = 1.0
    for iteration in range(1, n_iter + 1):
        slope = _dot(g, d)
        if slope >= 0:  # not a descent direction: restart from steepest descent
            d, slope = -z, -gz
        # The parabola through f(0), f'(0) and f(step).
        taken = step
        for _ in range(4):
            trial = objective.value(V + step * d)
            curvature = trial - f - slope * step
            if curvature > 0:
                taken = min(-slope * step * step / (2 * curvature), 20 * step)
                break
            step *= 4  # flat or concave along d: look further
        else:
            taken = step
        V.add_(d, alpha=taken)
        data_new, prior_new, g_new = objective.value_and_gradient(V)
        if data_new + prior_new > f and taken > step:
            # The parabola overshot; fall back to the trial point.
            V.add_(d, alpha=step - taken)
            taken = step
            data_new, prior_new, g_new = objective.value_and_gradient(V)
        z_new = precondition(g_new)
        beta = max(0.0, _dot(z_new, g_new - g) / gz) if gz > 0 else 0.0
        # A preconditioned step is close to a Newton step, so a decrease
        # below the objective's own rounding means the remaining gradient is
        # rounding too, and further steps would follow it. Without the
        # preconditioner a small decrease may only mean a short step.
        converged = preconditioner is not None and f - (
            data_new + prior_new
        ) < _OBJECTIVE_RESOLUTION * abs(f)
        g, z, f = g_new, z_new, data_new + prior_new
        gz = _dot(g, z)
        d = -z + beta * d
        step = taken
        entry = {
            "objective": f,
            "data": data_new,
            "prior": prior_new,
            "chi2_per_pixel": 2 * data_new / n_pixels,
            "step": taken,
            "seconds": time.time() - start,
        }
        log.append(entry)
        logger.info(
            f"  iteration {iteration:3d}  chi2/pixel {entry['chi2_per_pixel']:.4f}  "
            f"prior {prior_new:.3e}  step {taken:.2e}"
        )
        if callback is not None:
            callback(iteration, entry)
        if gz == 0 or converged:
            break
    if model.kmask is not None:
        from ._helpers import _apply_kmask_inplace

        _apply_kmask_inplace(V, model._kmask_half())
    return log


# --------------------------------------------------------------------------- #
# Strip-wise solve
# --------------------------------------------------------------------------- #


def _smoothstep(u: torch.Tensor, centre: float, half: float) -> torch.Tensor:
    """A raised-cosine step from 0 to 1 across ``centre -/+ half``."""
    t = ((u - centre) / (2 * half) + 0.5).clamp(0.0, 1.0)
    return 0.5 - 0.5 * torch.cos(math.pi * t)


@dataclasses.dataclass(frozen=True)
class StripPlan:
    """
    How a volume is divided into overlapping strips along its tilt axis.

    Every strip has the same shape, so the strips share their non-uniform
    FFT plans and need the same memory. Neighbours overlap. From an inner
    face inwards, a strip is first tapered to zero over ``taper`` voxels:
    the forward model is band limited at the Nyquist frequency, so a volume
    that ended abruptly would ring across the whole image. The strip's loss
    then starts a further ``reach`` voxels in, beyond what the transfer
    function brings in from the tapered part and from the specimen the strip
    does not hold. At the middle of the overlap one strip is cross-faded
    into the next over ``2 * fade`` voxels; a voxel there is at least
    ``taper + 2 * reach`` from the face of either strip, so every pixel its
    image is displaced onto is compared by both.

    Attributes
    ----------
    dim : int
        Dimension of the ``(Z, Y, X)`` volume the strips divide: 2 for a
        tilt axis along X, 1 for one along Y.
    size : int
        The volume's size along ``dim``.
    width : int
        Each strip's size along ``dim``.
    starts : tuple of int
        First voxel of each strip along ``dim``.
    reach : int
        Voxels the transfer function displaces the image by
        (`TomogramReconstructor._delocalisation`).
    fade : float
        Half-width of the cross-fade, in voxels.
    taper : int
        Voxels over which a strip falls to zero at its inner faces.
    """

    dim: int
    size: int
    width: int
    starts: tuple[int, ...]
    reach: int
    fade: float
    taper: int = _TAPER

    @property
    def seams(self) -> tuple[float, ...]:
        """Where neighbouring strips are cross-faded: the middle of each overlap."""
        return tuple(
            (self.starts[i + 1] + self.starts[i] + self.width - 1) / 2
            for i in range(len(self.starts) - 1)
        )

    def weight(self, i: int, u: torch.Tensor) -> torch.Tensor:
        """
        Strip ``i``'s share at coordinates ``u`` along ``dim``, in voxels.

        The shares of all strips sum to one everywhere.
        """
        seams = self.seams
        w = torch.ones_like(u)
        if i > 0:
            w = w * _smoothstep(u, seams[i - 1], self.fade)
        if i < len(self.starts) - 1:
            w = w * (1 - _smoothstep(u, seams[i], self.fade))
        return w

    def compared(self, i: int, u: torch.Tensor) -> torch.Tensor:
        """
        Whether a ray at coordinate ``u`` along ``dim`` enters strip ``i``'s loss.

        True where the ray is inside the strip's untapered part and at
        least ``reach`` from its ends; the volume's own outer faces set no
        limit.
        """
        keep = torch.ones_like(u, dtype=torch.bool)
        margin = self.taper + self.reach
        if i > 0:
            keep &= u >= self.starts[i] + margin
        if i < len(self.starts) - 1:
            keep &= u <= self.starts[i] + self.width - 1 - margin
        return keep

    def window(self, i: int) -> torch.Tensor:
        """
        The taper of strip ``i`` along ``dim``, ``(width,)``: one inside,
        falling to zero over ``taper`` voxels at each inner face.
        """
        u = torch.arange(self.width, dtype=torch.float64)
        w = torch.ones_like(u)
        half = self.taper / 2
        if i > 0:
            w = w * _smoothstep(u, half - 0.5, half)
        if i < len(self.starts) - 1:
            w = w * (1 - _smoothstep(u, self.width - 0.5 - half, half))
        return w.float()

    @classmethod
    def for_model(
        cls,
        model: "TomogramReconstructor",
        strips: int | None = None,
        max_voxels: int | None = None,
        overlap: int | None = None,
    ) -> "StripPlan":
        """
        Divide ``model``'s volume along its tilt axis.

        Parameters
        ----------
        model : TomogramReconstructor
            The whole problem. Its forward model must be sampled on the
            surface, and its rays must keep their coordinate along volume X
            or Y at every tilt: a series tilting about that axis
            (`specter.tilt.turn_to_tilt_axis` turns any single-axis series'
            frame so that it does).
        strips : int, optional
            Number of strips. Default the fewest that respect ``max_voxels``,
            or one.
        max_voxels : int, optional
            Largest strip, in voxels.
        overlap : int, optional
            Voxels neighbouring strips share. Default twice the taper,
            four times the transfer function's reach, and a cross-fade as
            wide as twice that reach and at least 16 voxels.

        Returns
        -------
        StripPlan
        """
        if not model._surface:
            raise ValueError(
                "a strip-wise solve needs a forward model sampled on the "
                "surface (a single-scatter model with rotate_mode unset)"
            )
        if model.kmask is not None or model.taper_width > 0:
            raise ValueError(
                "a strip-wise solve supports neither a kmask nor a lateral "
                "taper: both act on the whole volume"
            )
        nz, ny, nx = (int(n) for n in model.V.shape)
        rot = roma.unitquat_to_rotmat(model.quaternions.double().cpu())
        # rot takes beam coordinates to the volume's, so rot[:, a, 2] is how
        # far a ray moves along volume axis a per voxel of depth.
        drift = rot[:, :2, 2].abs().amax(0) * nz / 2
        if float(drift[0]) <= _AXIS_TOLERANCE:
            dim, size, other = 2, nx, ny
        elif float(drift[1]) <= _AXIS_TOLERANCE:
            dim, size, other = 1, ny, nx
        else:
            raise ValueError(
                "a strip-wise solve needs the tilt axis along volume X or Y; "
                "rays drift by "
                f"{float(drift[0]):.1f} and {float(drift[1]):.1f} voxels along "
                "them. Turn the volume frame with specter.tilt.turn_to_tilt_axis"
            )
        reach = max(model._delocalisation())
        inner = 2 * (_TAPER + 2 * reach)  # what neither strip of a pair trusts
        if overlap is None:
            overlap = inner + 2 * max(reach, _MIN_FADE) + 2
        if overlap < inner + 4:
            raise ValueError(
                f"overlap={overlap} leaves no cross-fade: the strips' tapers "
                f"and the {reach} voxels the transfer function displaces the "
                f"image by take {inner}"
            )

        def layout(count: int) -> tuple[int, tuple[int, ...]]:
            if count == 1:
                return size, (0,)
            width = math.ceil((size + (count - 1) * overlap) / count)
            width += (width - model.nxy) % 2  # the images' parity, as V has
            if width >= size:
                return size, (0,)
            step = (size - width) / (count - 1)
            return width, tuple(round(i * step) for i in range(count))

        def fits(count: int) -> bool:
            # Successive cross-fades must not meet inside a strip.
            width, starts = layout(count)
            steps = [b - a for a, b in zip(starts, starts[1:])]
            return not steps or min(steps) >= overlap - inner

        if strips is None:
            strips = 1
            if max_voxels is not None:
                while nz * other * layout(strips)[0] > max_voxels:
                    strips += 1
                    if not fits(strips):
                        raise ValueError(
                            f"no division into strips of at most {max_voxels} "
                            f"voxels leaves room for their {overlap}-voxel overlaps"
                        )
        if not fits(strips):
            raise ValueError(
                f"{strips} strips of a {size}-voxel volume, {layout(strips)[0]} "
                f"voxels wide, leave no room for their {overlap}-voxel overlaps"
            )
        width, starts = layout(strips)
        shared = min(
            (starts[i] + width - starts[i + 1] for i in range(len(starts) - 1)),
            default=overlap,
        )
        return cls(dim, size, width, starts, reach, (shared - 1 - inner) / 2)


def _ray_coordinate(
    model: "TomogramReconstructor", plan: StripPlan, tilt_idx: int
) -> torch.Tensor:
    """
    Where each pixel's ray of tilt ``tilt_idx`` lies along the strip axis.

    A volume point ``r``, in voxels from the volume's centre, is imaged at
    ``(R^T r)_xy - t`` from the image's, so the ray through pixel ``p``
    holds the points ``R (p + t, s)`` for every depth ``s``, and with the
    tilt axis along volume axis ``a`` their ``a`` coordinate is ``R[a, 0]
    (p + t)_x + R[a, 1] (p + t)_y`` whatever ``s``. Returned as a voxel
    index along the axis, ``(H, W)``, on the CPU.
    """
    rot = roma.unitquat_to_rotmat(model.quaternions[tilt_idx].double().cpu())
    shift = model.translations[tilt_idx].double().cpu() / model.voxel_size
    pixel = torch.arange(model.nxy, dtype=torch.float64) - model.nxy // 2
    a = 0 if plan.dim == 2 else 1
    return (
        rot[a, 0] * (pixel[None, :] + shift[0])
        + rot[a, 1] * (pixel[:, None] + shift[1])
        + plan.size // 2
    )


def _strip_model(
    model: "TomogramReconstructor",
    plan: StripPlan,
    i: int,
    device: torch.device,
) -> "TomogramReconstructor":
    """
    Strip ``i`` of ``model`` as a reconstruction problem of its own.

    The strip's voxels, tapered at the inner faces, with the whole
    problem's poses, transfer function and priors. Its centre lies ``c`` voxels along the tilt axis from the
    volume's, which displaces its image by the same ``(R^T c)_xy`` at every
    tilt and leaves its defocus alone, so each tilt's shift absorbs it. The
    loss is restricted to the pixels the whole problem compares whose rays
    the strip holds (`StripPlan.compared`).
    """
    from ._tomogram_reconstructor import TomogramReconstructor

    start = plan.starts[i]
    index = [slice(None)] * 3
    index[plan.dim] = slice(start, start + plan.width)
    volume = model.V.detach()[tuple(index)].clone()
    n_tilts = len(model.quaternions)
    rot = roma.unitquat_to_rotmat(model.quaternions.double().cpu())
    offset = start + plan.width // 2 - plan.size // 2
    a = 0 if plan.dim == 2 else 1
    translations = (
        model.translations.double().cpu() - model.voxel_size * offset * rot[:, a, :2]
    )
    mask = torch.empty(n_tilts, model.nxy, model.nxy, dtype=torch.bool)
    for t in range(n_tilts):
        mask[t] = plan.compared(i, _ray_coordinate(model, plan, t))
        whole = model._compared(t)
        if whole is not None:
            mask[t] &= whole.cpu() > 0
    strip = TomogramReconstructor(
        volume,
        model.voxel_size,
        model.quaternions.cpu(),
        translations.float(),
        {name: getattr(model, name).cpu() for name in model.ctf_params},
        model.voltage,
        model.dose_per_pixel.cpu() / model.voxel_size**2,
        tilt=model.tilt,
        use_fov_mask=False,
        propagation=model.propagation,
        optics=model.optics,
        image_size=model.nxy,
        smoothness=model.smoothness,
        tv_weight=model.tv_weight,
        tv_delta=model.tv_delta,
        pixel_mask=mask,
    )
    shape = [1, 1, 1]
    shape[plan.dim] = plan.width
    strip.volume_window = plan.window(i).reshape(shape)
    return strip.to(device)


def _run_on_devices(
    jobs: int,
    devices: Sequence[torch.device],
    work: Callable[[int, torch.device], None],
) -> None:
    """Run ``work(job, device)`` for every job, one at a time per device."""
    pending: queue.SimpleQueue[int] = queue.SimpleQueue()
    for job in range(jobs):
        pending.put(job)
    errors: list[BaseException] = []

    def drain(device: torch.device) -> None:
        while not errors:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                work(job, device)
            except BaseException as err:  # noqa: BLE001 -- re-raised by the caller
                errors.append(err)

    workers = list(devices[: max(1, min(len(devices), jobs))])
    if len(workers) == 1:
        drain(workers[0])
    else:
        threads = [threading.Thread(target=drain, args=(d,)) for d in workers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    if errors:
        raise errors[0]


def _volume_weight(plan: StripPlan, i: int) -> torch.Tensor:
    """Strip ``i``'s share of its own voxels, broadcastable over the strip."""
    u = plan.starts[i] + torch.arange(plan.width, dtype=torch.float64)
    shape = [1, 1, 1]
    shape[plan.dim] = plan.width
    return plan.weight(i, u).float().reshape(shape)


def solve_tilt_series_in_strips(
    model: "TomogramReconstructor",
    images: torch.Tensor,
    n_iter: int,
    *,
    strips: int | None = None,
    max_voxels: int | None = None,
    overlap: int | None = None,
    devices: Sequence[torch.device | str] | None = None,
    tilts: torch.Tensor | None = None,
    tilt_batch: int = 8,
) -> list[list[dict[str, Any]]]:
    """
    Fit ``model.V`` to a tilt series in strips along the tilt axis, in place.

    Each strip is solved by `solve_tilt_series` on a device of its own, so
    ``model`` itself can stay on the CPU and its volume may exceed what one
    device holds. The strips are independent and run one per device at a
    time.

    Parameters
    ----------
    model : TomogramReconstructor
        The whole problem; see `StripPlan.for_model` for what it must
        satisfy. Its volume is the starting point and receives the result.
    images : torch.Tensor
        The observed tilt series in electron counts, ``(N_tilts, H, W)``.
    n_iter : int
        Conjugate-gradient iterations per strip.
    strips, max_voxels, overlap : int, optional
        The division, see `StripPlan.for_model`.
    devices : sequence of torch.device or str, optional
        Devices to solve on. Default the model's own.
    tilts : torch.Tensor, optional
        Indices of the tilts to fit. Default all of them.
    tilt_batch : int, optional
        Tilts predicted per forward pass. Default 8.

    Returns
    -------
    list of list of dict
        `solve_tilt_series`' log for each strip, in order along the axis.

    Notes
    -----
    The stitched volume is not the minimiser of the whole problem's
    objective: each strip's prior ends at its faces, and the lowest spatial
    frequencies, which the data constrain weakly, are settled by each strip
    on its own. The strips agree with a single solve to the solver's own
    convergence in their interiors and differ most near the cross-fades.
    """
    plan = StripPlan.for_model(model, strips, max_voxels, overlap)
    volume = model.V.data if isinstance(model.V, torch.nn.Parameter) else model.V
    targets = [torch.device(d) for d in (devices or [volume.device])]
    count = len(plan.starts)
    parts: list[torch.Tensor | None] = [None] * count
    logs: list[list[dict[str, Any]]] = [[] for _ in range(count)]

    def solve(i: int, device: torch.device) -> None:
        strip = _strip_model(model, plan, i, device)
        logger.info(
            f"strip {i + 1}/{count}: voxels "
            f"{plan.starts[i]}..{plan.starts[i] + plan.width - 1} on {device}"
        )
        logs[i] = solve_tilt_series(
            strip, images, n_iter, tilts=tilts, tilt_batch=tilt_batch
        )
        parts[i] = strip.V.detach().to(volume.device, copy=True)
        del strip
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _run_on_devices(count, targets, solve)
    _stitch(plan, parts, volume)
    return logs


def _stitch(
    plan: StripPlan, parts: Sequence[torch.Tensor | None], out: torch.Tensor
) -> None:
    """
    Cross-fade the solved strips into ``out``, releasing each as it is used.

    Without absorption the data do not determine a strip's mean potential,
    so each strip settles on its own. Before it is faded in, a strip is
    shifted by the constant that makes its mean over the cross-fade equal
    its neighbour's; the shifts are then removed on average, which leaves
    the volume's mean where the strips put it.
    """
    parts = list(parts)
    out.zero_()
    total, previous = 0.0, None
    for i, start in enumerate(plan.starts):
        part = parts[i]
        assert part is not None
        parts[i] = None
        shift = 0.0
        if previous is not None:
            seam = plan.seams[i - 1]
            lo, hi = math.ceil(seam - plan.fade), math.floor(seam + plan.fade)
            ahead, behind = [slice(None)] * 3, [slice(None)] * 3
            ahead[plan.dim] = slice(lo - start, hi + 1 - start)
            behind[plan.dim] = slice(
                lo - plan.starts[i - 1], hi + 1 - plan.starts[i - 1]
            )
            shift = float(previous[tuple(behind)].mean() - part[tuple(ahead)].mean())
            part += shift
        weight = _volume_weight(plan, i).to(out.device)
        total += shift * float(weight.sum())
        index = [slice(None)] * 3
        index[plan.dim] = slice(start, start + plan.width)
        out[tuple(index)] += part * weight
        previous = part
    out -= total / plan.size


def predict_tilt_series(
    model: "TomogramReconstructor",
    tilts: torch.Tensor | None = None,
    *,
    strips: int | None = None,
    max_voxels: int | None = None,
    overlap: int | None = None,
    devices: Sequence[torch.device | str] | None = None,
    tilt_batch: int = 8,
) -> torch.Tensor:
    """
    The noiseless images ``model.V`` predicts, in strips if asked.

    Scoring a prediction against images the fit did not see (the other half
    of the dose, or withheld tilts) validates a reconstruction without a
    ground truth. A volume that was solved in strips is predicted in strips
    too: each strip predicts the pixels whose rays it holds, and the
    predictions are combined with the strips' cross-fade weights.

    Parameters
    ----------
    model : TomogramReconstructor
        The reconstruction.
    tilts : torch.Tensor, optional
        Indices of the tilts to predict. Default all of them.
    strips, max_voxels, overlap : int, optional
        The division, see `StripPlan.for_model`. Default one strip, the
        whole volume on one device.
    devices : sequence of torch.device or str, optional
        Devices to predict on. Default the model's own.
    tilt_batch : int, optional
        Tilts predicted per forward pass. Default 8.

    Returns
    -------
    torch.Tensor
        Expected electron counts, ``(len(tilts), H, W)``, on the CPU.
    """
    tilts = torch.arange(len(model.quaternions)) if tilts is None else tilts.cpu()
    volume = model.V.detach()
    targets = [torch.device(d) for d in (devices or [volume.device])]
    plan = StripPlan.for_model(model, strips, max_voxels, overlap)
    out = torch.zeros(len(tilts), model.nxy, model.nxy)
    if len(plan.starts) == 1:
        whole = model
        if volume.device != targets[0]:
            whole = copy.deepcopy(model).to(targets[0])
        with torch.no_grad():
            prepared = whole._taper(whole.V)
            for n, idx in enumerate(tilts.split(tilt_batch)):
                out[n * tilt_batch : n * tilt_batch + len(idx)] = whole._forward_tilts(
                    prepared, idx
                ).cpu()
        return out
    lock = threading.Lock()

    def predict(i: int, device: torch.device) -> None:
        strip = _strip_model(model, plan, i, device)
        with torch.no_grad():
            prepared = strip._taper(strip.V)
            for n, idx in enumerate(tilts.split(tilt_batch)):
                images = strip._forward_tilts(prepared, idx).cpu()
                share = torch.stack(
                    [
                        plan.weight(i, _ray_coordinate(model, plan, int(t))).float()
                        for t in idx
                    ]
                )
                with lock:
                    out[n * tilt_batch : n * tilt_batch + len(idx)] += images * share
        del strip
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _run_on_devices(len(plan.starts), targets, predict)
    return out
