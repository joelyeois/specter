"""
`TomogramReconstructor`: reconstructs a volume from a tilt series with the
same forward model as `imaging.TiltSeries`.

The single-scatter models predict every tilt from one non-uniform FFT of the
unrotated volume (`specter.scattering.surface_slice_sum`); multislice walks
the tilted volume slice by slice. The volume may be wider than the images,
so that it holds everything the tilted beam passes through.
"""

from __future__ import annotations

from specter import logger

import json
import math
from pathlib import Path
from typing import Any

import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import next_fast_len
from torch.optim.lr_scheduler import LRScheduler

from .._defaults import FFT_PAD_MARGIN, LR_DECAY, SCHEDULER
from .. import tilt as tilt_geometry
from ..aberrations import defocus_midplane_shift
from ..arrays import center_crop
from ..constants import energy_to_wavelength
from ..imaging import CTF, ExitWave
from ..imaging._engines import scattering_engine
from ..imaging._tilt import tilt_defocus_shift, tilt_theta
from ..scattering import IterativeScattering, surface_slice_sum
from ._base_reconstructor import _BaseReconstructor
from ..settings import Optics, Propagation, TiltGeometry
from specter.options import Scheduler


class TomogramReconstructor(_BaseReconstructor):
    """
    Differentiable tomogram reconstruction from a cryo-ET tilt series.

    Reconstructs a 3D electrostatic potential volume from tilt-series images
    by minimising the discrepancy between simulated and observed images, using
    the same forward model as :class:`~specter.imaging.TiltSeries`.

    Under a single-scatter model (projection, CTF, first Born, Rytov) with
    ``propagation.rotate_mode`` unset, each tilt is the volume's transform
    sampled on that tilt's Ewald surface, so a batch of tilts costs one
    non-uniform FFT and nothing is interpolated. Multislice, or an explicit
    ``rotate_mode`` other than ``"surface"``, samples the tilted volume slice
    by slice instead.

    The volume's lateral shape may exceed the images'. A tilted beam passes
    through specimen beside the field of view, and a volume as wide as the
    images cannot hold it: either the loss is masked to the strip of each
    tilt the volume fully explains (``use_fov_mask``), discarding up to half
    of the highest tilts, or the volume is made wide enough to hold every
    ray (:meth:`covering_shape`), which uses all of the data. Only the
    surface path accepts a volume wider than its images.

    The reconstructed potential is relative to the specimen's mean: a
    uniform slab shifts the phase of the whole wave and leaves no contrast.
    With ``dose_per_angstrom`` set to each tilt's mean counts per unit area,
    the volume is the potential's departure from that mean, and the empty
    space beside it stands for more of the same specimen rather than vacuum.

    A tilt series holds fewer pixels than its tomogram holds voxels, so the
    data alone do not determine the volume. ``smoothness`` and ``tv_weight``
    add the priors that do (:meth:`regularisation`).

    Typically constructed and driven by :class:`TomogramGhostbuster`, which
    fits it with `specter.ghostbuster._tomogram_solver.solve_tilt_series`.

    Parameters
    ----------
    V : torch.Tensor
        Initial volume estimate, shape ``(Z, Y, X)``. ``Y`` and ``X`` are at
        least ``image_size``, unless a ``pixel_mask`` is given, and of its
        parity.
    voxel_size : float
        Voxel size in Å.
    quaternions : torch.Tensor
        Per-tilt rotation quaternions, shape ``(N_tilts, 4)``, xyzw convention.
    translations : torch.Tensor
        Per-tilt in-plane translations in Å, shape ``(N_tilts, 2)``.
    ctf_params : dict[str, torch.Tensor]
        Per-tilt CTF parameters; each value has leading dimension ``N_tilts``.
    voltage : float
        Electron beam accelerating voltage in kV.
    dose_per_angstrom : float or torch.Tensor
        Electron dose (fluence) per tilt image in e⁻/Å². Scalar, or a 1-D
        tensor of length ``N_tilts`` giving a separate dose for each tilt, as
        `Exposure.tilt_series` holds it. The forward model predicts electron
        counts per pixel, ``dose_per_angstrom * voxel_size**2 * |CTF(psi)|²``,
        and the loss divides each tilt's squared residual by that tilt's
        expected counts per pixel, its Poisson variance.
    tilt : TiltGeometry, optional
        Tilt axis and edge tapers -- the same object the `TiltSeries` that
        produced the data was built with. The axis decides which image
        dimension shrinks at high tilt for FOV masking; the tapers are applied
        to V before each forward pass. Default ``TiltGeometry()``.
    lr : float, optional
        Learning rate for volume V.  ``None`` freezes V.
    use_fov_mask : bool
        If ``True``, the loss at each tilt is restricted to the pixels the
        forward model can predict: the central strip corresponding to real
        (non-padded) volume at high tilt, less a band at every edge as wide
        as the transfer function delocalises the image (:meth:`_edge_pads`).
        Within that band the data carry whatever surrounds the field, which
        the volume does not contain. Default ``True``.
    propagation : Propagation, optional
        How the forward model computes the exit wave (model, amplitude
        contrast, bandlimit). The same object the `TiltSeries` that produced
        the data was scattered with, when it was simulated. Default
        ``Propagation()``.
    optics : Optics, optional
        The aberration engine and phase plate. Default ``Optics()``. No
        `Envelopes` or `Camera`: the inverse does not model what the data
        cannot determine.
    scheduler : str
        LR scheduler for the volume optimiser.  Default ``"LambdaLR"``.
    lr_decay : float
        Decay coefficient for the reciprocal-sqrt LR schedule.  Default ``0.1``.
    kmask : torch.Tensor, optional
        3-D Fourier-space mask applied to V after every gradient update.
    slice_batchsize : int
        Z-slice chunk size passed to ``IterativeScattering``.  Reduce for large
        volumes to stay within GPU memory.  Default ``1``.
    checkpoint_chunks : int or None
        If set, the multislice slice loop is split into chunks of this size
        and run under gradient checkpointing.  Only the wave function at chunk
        boundaries is retained; intermediates are recomputed on the backward
        pass.  This reduces activation memory from ``O(nz_tilt)`` (tens of GB
        at high tilt) to ``O(checkpoint_chunks × nxy²)`` at the cost of one
        extra forward pass per chunk.  Suggested starting value: ``50``.
        ``None`` disables checkpointing.
    run_dir : str or Path, optional
        Directory for per-epoch MRC volumes and training metrics.  ``None``
        disables all file output.
    image_size : int, optional
        Size of the tilt images in pixels. Default the volume's ``X``.
    smoothness : float
        Weight of the quadratic gradient prior, ``smoothness / 2 * sum
        |grad V|^2`` in the units of the chi-squared sum. Default 0.
    tv_weight : float
        Weight of the edge-preserving prior, a pseudo-Huber total variation
        (:meth:`regularisation`). Default 0.
    tv_delta : float
        The gradient magnitude, in volts per voxel, below which the
        edge-preserving prior is quadratic and above which it grows
        linearly. Default 0.02.
    pixel_mask : torch.Tensor, optional
        Boolean mask of shape ``(N_tilts, H, W)``: the pixels of each tilt
        the loss compares, in addition to what ``use_fov_mask`` leaves out.
        A volume that holds only part of the specimen under the field, such
        as one strip of a strip-wise solve
        (`specter.ghostbuster._tomogram_solver.solve_tilt_series_in_strips`),
        is fitted to the pixels whose rays stay inside it; with a mask the
        volume may be narrower than the images. Default ``None``, every
        pixel.

    Notes
    -----
    Measured on a simulated 100 nm specimen at 5 A per voxel and 3.9
    electrons per square Angstrom per tilt, against the volume the series
    was simulated from: the edge-preserving prior raised the Fourier shell
    correlation with the truth from 0.79 to 0.89 at 400 to 100 A over the
    quadratic prior and recovered part of the missing wedge (0.69 against
    0.04 there), and a volume covering every ray raised it from 0.53 to 0.79
    over the masked loss.
    """

    def __init__(
        self,
        V: torch.Tensor,
        voxel_size: float,
        quaternions: torch.Tensor,
        translations: torch.Tensor,
        ctf_params: dict[str, torch.Tensor],
        voltage: float,
        dose_per_angstrom: float | torch.Tensor,
        tilt: TiltGeometry = TiltGeometry(),
        lr: float | None = None,
        use_fov_mask: bool = True,
        propagation: Propagation = Propagation(),
        optics: Optics = Optics(),
        scheduler: Scheduler = SCHEDULER,
        lr_decay: float = LR_DECAY,
        kmask: torch.Tensor | None = None,
        slice_batchsize: int = 1,
        checkpoint_chunks: int | None = None,
        run_dir: str | Path | None = None,
        image_size: int | None = None,
        smoothness: float = 0.0,
        tv_weight: float = 0.0,
        tv_delta: float = 0.02,
        pixel_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if propagation.absorption_model == "inelastic_mfp":
            raise ValueError(
                "TomogramReconstructor does not support absorption_model='inelastic_mfp': "
                "its iterative volume path requires explicit material/solvent "
                "information to construct the absorption field. Use 'alpha'."
            )
        scattering_model = propagation.scattering_model
        self._save_hyperparameters(
            TomogramReconstructor.__init__,
            locals(),
            ignore=[
                "V",
                "quaternions",
                "translations",
                "ctf_params",
                "kmask",
                "pixel_mask",
            ],
        )
        self._run_dir = Path(run_dir) if run_dir is not None else None

        # Geometry: the images are `nxy` across; the volume may be wider.
        self.nxy = V.shape[-1] if image_size is None else image_size
        self.nz = V.shape[0]
        self._surface = propagation.rotation() == "surface"
        ny, nx = V.shape[-2], V.shape[-1]
        narrow = min(ny, nx) < self.nxy and pixel_mask is None
        if narrow or (ny - self.nxy) % 2 or (nx - self.nxy) % 2:
            raise ValueError(
                f"the volume's lateral shape ({ny}, {nx}) must be at least the "
                f"image size {self.nxy} (unless a pixel_mask says which pixels "
                "it explains) and of its parity"
            )
        if pixel_mask is not None and tuple(pixel_mask.shape) != (
            len(quaternions),
            self.nxy,
            self.nxy,
        ):
            raise ValueError(
                f"pixel_mask has shape {tuple(pixel_mask.shape)}; expected "
                f"{(len(quaternions), self.nxy, self.nxy)}"
            )
        if not self._surface and (ny, nx) != (self.nxy, self.nxy):
            raise ValueError(
                f"a volume wider than its {self.nxy} px images ({ny} x {nx}) "
                "needs a single-scatter model sampled on the surface; "
                f"scattering_model={scattering_model!r} with rotate_mode="
                f"{propagation.rotate_mode!r} walks the tilted volume instead"
            )
        if min(smoothness, tv_weight) < 0 or tv_delta <= 0:
            raise ValueError(
                "smoothness and tv_weight must be non-negative and tv_delta positive"
            )
        self.smoothness = smoothness
        self.tv_weight = tv_weight
        self.tv_delta = tv_delta
        self.voxel_size = voxel_size
        self.voltage = voltage
        self.tilt = tilt
        self.tilt_axis = tilt.tilt_axis

        # Optimisation settings
        self.lr = lr
        self.lr_decay = lr_decay
        self.scheduler = scheduler
        self.taper_width = tilt.taper_width
        self.z_taper_width = tilt.z_taper_width
        self.use_fov_mask = use_fov_mask
        self.slice_batchsize = slice_batchsize
        self.checkpoint_chunks = checkpoint_chunks

        self._init_loss_logs()

        # Volume parameter
        if lr is None:
            self.register_buffer("V", V.float())
        else:
            self.V = nn.Parameter(V.float())

        # Expected electrons per pixel for each tilt, the scale of the
        # predicted counts and the Poisson variance the loss divides by.
        n_tilts = len(quaternions)
        dose = torch.as_tensor(dose_per_angstrom, dtype=torch.float32).flatten()
        if dose.numel() not in (1, n_tilts):
            raise ValueError(
                f"dose_per_angstrom has {dose.numel()} entries; expected 1 or "
                f"one per tilt ({n_tilts})"
            )
        if not bool((dose > 0).all()):
            raise ValueError("dose_per_angstrom must be positive")
        self.register_buffer(
            "dose_per_pixel", dose.expand(n_tilts).clone() * voxel_size**2
        )

        # Tilt geometry buffers (fixed — orientations are known in cryo-ET)
        self.register_buffer(
            "quaternions", torch.as_tensor(quaternions, dtype=torch.float32)
        )
        self.register_buffer(
            "translations", torch.as_tensor(translations, dtype=torch.float32)
        )

        # Per-tilt CTF params as named buffers
        self.ctf_params: dict[str, torch.Tensor] = {}
        for k, v in ctf_params.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
            self.ctf_params[k] = getattr(self, k)

        # Multislice returns the wave at the volume's exit face, while dfu/dfv
        # are the specimen midplane's defocus, the plane a CTF fit reports.
        # The simulators' chain applies the offset through
        # `ExitWave.defocus_shift`, which `_forward_tilt` builds the same way
        # `TiltSeries.scatter` does (`imaging._tilt.tilt_defocus_shift`), so
        # the two cannot drift apart. Recorded for the parity test.
        self._defocus_shift_angstrom = (
            0.0
            if scattering_model in ("projection", "ctf")
            else defocus_midplane_shift(self.nz, voxel_size)
        )

        # Fourier-space mask applied after each gradient step
        self.register_buffer("kmask", kmask)
        self.register_buffer(
            "pixel_mask",
            None if pixel_mask is None else torch.as_tensor(pixel_mask).bool(),
            persistent=False,
        )
        # A window the forward model multiplies the volume by, broadcastable
        # over it; a strip of a strip-wise solve is tapered at its inner faces.
        self.register_buffer("volume_window", None, persistent=False)

        # Physics modules
        self.propagation = propagation
        self.optics = optics
        self._pad_nxy = self._surface_grid() if self._surface else self.nxy
        self.iterative_scattering = IterativeScattering(
            self.nxy,
            voxel_size,
            voltage,
            scattering_model=scattering_model,
            klim=propagation.klim,
            oversample=propagation.oversample,
            crop=False,
            alpha=propagation.alpha,
            ews_curvature_sign=propagation.ews_curvature_sign,
            # The same canvas `TiltSeries` propagates on, kept through the
            # transfer function as it keeps it, so the two agree.
            pad_fft=propagation.pads(True),
            keep_pad=True,
        )
        # The image grid: the transfer function is applied and the intensity
        # formed on the propagation grid, then sampled to the pixel grid, as
        # in every simulator.
        self._image_oversample = (
            propagation.oversample if scattering_model != "ctf" else 1
        )

    # ------------------------------------------------------------------ #
    # Forward helpers                                                      #
    # ------------------------------------------------------------------ #

    def _prepare_volume(self) -> torch.Tensor:
        """Apply cosine taper to V for the current tilt pass.

        No XY padding is applied. At the maximum tilt angle the projected
        volume extent (nxy·cos θ + nz·sin θ) is smaller than nxy, so all
        volume content fits within the existing grid. Pixels sampled outside
        the volume boundary by grid_sample return zero (vacuum), which is
        physically correct. The real-FOV loss mask (use_fov_mask) already
        discards gradient contributions from the zero-padded periphery.
        """
        return self._taper(self.V)

    def _taper(self, V: torch.Tensor) -> torch.Tensor:
        """`V` under the cosine tapers of `tilt` and the strip window, if any."""
        if self.taper_width > 0 or self.z_taper_width > 0:
            V = tilt_geometry.apply_volume_cosine_taper(
                V, taper_xy=self.taper_width, taper_z=self.z_taper_width
            )
        if self.volume_window is not None:
            V = V * self.volume_window
        return V

    def _surface_grid(self) -> int:
        """
        Size of the grid the surface path propagates on.

        The sampled spectrum is that of an image periodic over the grid. A
        volume wider than the images gets a grid holding its whole
        projection at every tilt, found from the projected corners of its
        box, plus the margin every tilt series keeps against wrap
        (`FFT_PAD_MARGIN`); it has the images' parity, which keeps the volume
        and the field on one centre. The projection is displaced by each
        tilt's shift, and the grid is also wide enough that none of it
        wraps into the field from the far side, which matters for a strip
        of a larger volume, posed far from the field's centre by its shift.
        A volume as wide as the images is
        propagated on the image grid itself, as `TiltSeries` propagates the
        single-scatter models, so that the two predict the same image; what
        then wraps at a tilt lies inside the band the loss mask leaves out.
        """
        nz, ny, nx = self.V.shape
        if (ny, nx) == (self.nxy, self.nxy):
            return self.nxy
        rot = roma.unitquat_to_rotmat(self.quaternions.double().cpu())
        corners = torch.tensor(
            [
                [sx * nx / 2, sy * ny / 2, sz * nz / 2]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ],
            dtype=torch.float64,
        )
        # `rot` takes beam-frame coordinates to the volume's, so its
        # transpose projects the volume's corners into the image plane.
        projected = torch.einsum("bji,cj->bci", rot, corners)[..., :2]
        shift = self.translations.double().cpu() / self.voxel_size
        projected = projected - shift[:, None, :]
        # The projection must not overlap itself on the periodic grid, and
        # its farthest point must not wrap round into the field.
        span = float((projected.amax(1) - projected.amin(1)).max())
        reach = float(projected.abs().max())
        need = max(math.ceil(span), math.ceil(self.nxy / 2 + reach), self.nxy)
        need += 2 * FFT_PAD_MARGIN
        need += (need - self.nxy) % 2
        while next_fast_len(need) != need:
            need += 2
        return need

    @staticmethod
    def covering_shape(
        image_size: int,
        nz: int,
        quaternions: torch.Tensor,
        margin: int = 16,
    ) -> tuple[int, int, int]:
        """
        The smallest volume holding every ray that reaches the detector.

        Each tilt's field of view is traced through the slab ``nz`` thick;
        the volume's lateral shape is the bounding box of where its corners
        enter and leave, plus ``margin`` voxels on each side for what the
        transfer function delocalises into the field. With the tilt axis
        along a volume axis the box grows across it only, by
        ``image_size / cos(tilt) + nz * tan(tilt)``; an oblique axis grows
        both, so a series whose axis is neither is better posed in a volume
        frame turned to it first.

        Parameters
        ----------
        image_size : int
            Size of the tilt images in pixels.
        nz : int
            Depth of the volume in voxels.
        quaternions : torch.Tensor
            Per-tilt rotation quaternions, ``(N_tilts, 4)``.
        margin : int, optional
            Voxels added on each side. Default 16.

        Returns
        -------
        tuple of int
            ``(nz, Y, X)``, with ``Y`` and ``X`` of ``image_size``'s parity.
        """
        rot = roma.unitquat_to_rotmat(torch.as_tensor(quaternions).double().cpu())
        half = image_size / 2
        reach = torch.zeros(2, dtype=torch.float64)
        for cx in (-half, half):
            for cy in (-half, half):
                for face in (-nz / 2, nz / 2):
                    # The point of the ray through detector corner (cx, cy)
                    # at volume depth `face`: rot @ (cx, cy, s), s chosen so
                    # that its z component is `face`.
                    along = (face - rot[:, 2, 0] * cx - rot[:, 2, 1] * cy) / rot[
                        :, 2, 2
                    ]
                    point = (
                        rot[:, :2, 0] * cx
                        + rot[:, :2, 1] * cy
                        + rot[:, :2, 2] * along[:, None]
                    )
                    reach = torch.maximum(reach, point.abs().amax(0))
        sizes = []
        for r in (float(reach[1]), float(reach[0])):  # (Y, X)
            size = max(image_size, 2 * math.ceil(r)) + 2 * margin
            sizes.append(size + (size - image_size) % 2)
        return nz, sizes[0], sizes[1]

    def field_volume(self) -> torch.Tensor:
        """The part of ``V`` under the images: its central ``(nz, nxy, nxy)``."""
        ny, nx = self.V.shape[-2], self.V.shape[-1]
        y0, x0 = (ny - self.nxy) // 2, (nx - self.nxy) // 2
        return self.V.detach()[:, y0 : y0 + self.nxy, x0 : x0 + self.nxy]

    def _loss_mask(self, tilt_idx: int) -> torch.Tensor | None:
        """
        The pixels of tilt `tilt_idx` the loss compares, or ``None`` for all.

        The real-FOV strip (:meth:`_fov_mask`) less the delocalisation band
        at every edge (:meth:`_edge_pads`).
        """
        mask = self._fov_mask(tilt_idx)
        edge = self._edge_pads()[tilt_idx]
        # A volume reaching past the field by the band's width models what
        # the transfer function carries in, so that edge keeps its pixels.
        ny, nx = self.V.shape[-2], self.V.shape[-1]
        edge_y = 0 if (ny - self.nxy) // 2 >= edge else edge
        edge_x = 0 if (nx - self.nxy) // 2 >= edge else edge
        if edge_y == 0 and edge_x == 0:
            return mask
        if mask is None:
            mask = torch.ones(
                self.nxy, self.nxy, device=self.device, dtype=torch.float32
            )
        if edge_y:
            mask[:edge_y, :] = 0.0
            mask[self.nxy - edge_y :, :] = 0.0
        if edge_x:
            mask[:, :edge_x] = 0.0
            mask[:, self.nxy - edge_x :] = 0.0
        return mask

    def _edge_pads(self) -> list[int]:
        """
        Per-tilt width, in pixels, of the band at the field's edges that the
        transfer function fills from outside the field.

        The lens displaces the image of each spatial frequency ``k`` by
        ``|dchi/dk| / (2 pi) = lambda k |Df - Cs lambda^2 k^2|``, the
        delocalisation of an aberrated image. Out to the pixel Nyquist, and
        over every plane of the tilted column (the nominal defocus, the
        midplane's, plus or minus half the column's depth, and both
        astigmatic axes), the largest displacement bounds how far the
        specimen's surroundings reach into the field. The forward model
        cannot predict those pixels, since its volume ends at the field, so
        the loss leaves them out: fitting them anyway makes the edge voxels
        explain what lies outside the volume, and the interior degrades the
        longer the fit runs.

        A band wider than a quarter of the field per side leaves no pixel the
        surroundings cannot reach, so excluding it only discards data; such a
        tilt gets no band (0), and the log says so.

        Computed once per device and cached, as :meth:`_fov_pads` is.
        """
        key = self.quaternions.device
        cached = getattr(self, "_edge_pads_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        cap = self.nxy // 4
        reach = self._delocalisation()
        pads = [0 if r > cap else r for r in reach]
        too_wide = [i for i, r in enumerate(reach) if r > cap]
        if too_wide:
            logger.warning(
                "the transfer function delocalises the image of tilts %s "
                "further than a quarter of the %d px field, so no part of it "
                "is free of the specimen's surroundings; those tilts are "
                "compared without an edge band",
                too_wide,
                self.nxy,
            )
        self._edge_pads_cache = (key, pads)
        return pads

    def _delocalisation(self) -> list[int]:
        """
        Per-tilt distance, in pixels, over which the transfer function
        displaces the image: the bound :meth:`_edge_pads` describes, uncapped.
        """
        wavelength = energy_to_wavelength(self.voltage)
        k = torch.linspace(0.0, 0.5 / self.voxel_size, 1025, dtype=torch.float64)
        params = {name: getattr(self, name).detach().cpu() for name in self.ctf_params}
        n_tilts = len(self.quaternions)
        zero = torch.zeros(n_tilts)
        dfu = params.get("dfu", zero)
        dfv = params.get("dfv", dfu)
        cs = params.get("cs", zero)
        model = self.propagation.scattering_model
        pads: list[int] = []
        for i in range(n_tilts):
            theta = tilt_theta(
                self.quaternions[i : i + 1].detach().cpu(),
                self.translations[i : i + 1].detach().cpu(),
                self.nxy,
                self.voxel_size,
            )
            half_depth = abs(
                tilt_defocus_shift(
                    (1, self.nz, self.nxy, self.nxy), theta, self.voxel_size, model
                )
            )
            low = float(min(dfu[i], dfv[i])) - half_depth
            high = float(max(dfu[i], dfv[i])) + half_depth
            spherical = float(cs[i]) * wavelength**2 * k**2
            reach = max(
                float((wavelength * k * (defocus - spherical).abs()).max())
                for defocus in (low, high)
            )
            pads.append(math.ceil(reach / self.voxel_size))
        return pads

    def _fov_mask(self, tilt_idx: int) -> torch.Tensor | None:
        """
        Binary mask for the real-FOV region at the given tilt.

        Returns ``None`` for near-zero tilts where the full image is real FOV.

        The shrinking dimension is X for ``tilt_axis="y"`` and Y for
        ``tilt_axis="x"``.
        """
        pad = self._fov_pads()[tilt_idx]
        if pad is None:
            return None

        mask = torch.ones(self.nxy, self.nxy, device=self.device, dtype=torch.float32)
        if self.tilt_axis == "x":
            # rotation around X → Y shrinks
            mask[:pad, :] = 0.0
            mask[self.nxy - pad :, :] = 0.0
        else:
            # rotation around Y → X shrinks
            mask[:, :pad] = 0.0
            mask[:, self.nxy - pad :] = 0.0
        return mask

    def _fov_pads(self) -> list[int | None]:
        """
        Per-tilt border width of the real-FOV mask, ``None`` where there is none.

        The tilt poses are fixed buffers, so this is computed once, on first
        use, rather than read back from the device with an ``.item()`` sync on
        every training step. Keyed on the poses' device, since the widths are
        evaluated there and the trigonometry may round differently on another.
        The mask itself is rebuilt per call from the cached width: it is an
        asynchronous fill, and caching one ``nxy**2`` mask per tilt would grow
        device memory with the tilt count.
        """
        key = self.quaternions.device
        cached = getattr(self, "_fov_pads_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        pads: list[int | None] = []
        # The volume's width across the tilt axis, which the tilt foreshortens.
        width = self.V.shape[-2] if self.tilt_axis == "x" else self.V.shape[-1]
        for Q in self.quaternions:
            rotvec = roma.unitquat_to_rotvec(Q.unsqueeze(0))[0]
            theta = rotvec.norm()  # radians (scalar tensor)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            real_fov = int((width * cos_t - self.nz * sin_t).clamp(min=1).item())
            pads.append(None if real_fov >= self.nxy else (self.nxy - real_fov) // 2)
        self._fov_pads_cache = (key, pads)
        return pads

    def _forward_tilt(self, V_prepared: torch.Tensor, tilt_idx: int) -> torch.Tensor:
        """
        Simulate one noiseless tilt image.

        Parameters
        ----------
        V_prepared : torch.Tensor
            Tapered volume from ``_prepare_volume``, shape ``(Z, Y, X)``.
        tilt_idx : int
            Index into ``self.quaternions`` / ``self.translations``.

        Returns
        -------
        torch.Tensor
            Noiseless expected electron counts per pixel, shape ``(H, W)``:
            the intensity a `TiltSeries` forms for the tilt without noise
            (``ExitWave.intensity(cc=None).pixels()``), times the tilt's dose
            per pixel.
        """
        if self._surface:
            idx = torch.tensor([tilt_idx], device=self.quaternions.device)
            return self._forward_tilts(V_prepared, idx)[0]
        theta_matrix = tilt_theta(
            self.quaternions[tilt_idx : tilt_idx + 1],
            self.translations[tilt_idx : tilt_idx + 1],
            V_prepared.shape[-1],
            self.voxel_size,
        )
        V_batched = V_prepared.unsqueeze(0)  # (1, Z, Y', X')
        exitwave = self.iterative_scattering(
            V_batched,
            theta_matrix,
            slice_batchsize=self.slice_batchsize,
            checkpoint_chunks=self.checkpoint_chunks,
        )
        model = self.propagation.scattering_model
        waves = ExitWave(
            array=exitwave,
            pixel_size=self.voxel_size,
            voltage=self.voltage,
            nxy=self.nxy,
            pad_nxy=self.iterative_scattering.output_nxy,
            oversample=self._image_oversample,
            defocus_shift=tilt_defocus_shift(
                tuple(V_batched.shape), theta_matrix, self.voxel_size, model
            ),
            ids=torch.zeros(1, dtype=torch.long),
            tilt=tilt_idx,
            linear=model == "ctf",
            alpha=self.propagation.alpha,
        )
        # getattr, not the dict built at __init__: `.to(device)` replaces
        # buffer tensors, and the dict would keep the old ones.
        ctf = CTF(
            {k: getattr(self, k)[tilt_idx : tilt_idx + 1] for k in self.ctf_params}
        )
        image = waves.apply_ctf(
            ctf,
            backend=self.optics.aberration_backend,
            lpp_params=self.optics.lpp_params,
            phase_plate=self.optics.phase_plate,
            phase_plate_cuton=self.optics.phase_plate_cuton,
        ).intensity(cc=None)
        field = center_crop(image.pixels()[0], self.nxy, dim=(-2, -1))
        return self.dose_per_pixel[tilt_idx] * field

    def _forward_tilts(
        self, V_prepared: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Simulate the noiseless images of tilts ``idx``, ``(len(idx), H, W)``.

        On the surface path the tilts share one non-uniform FFT of the
        volume; otherwise each is propagated in turn (:meth:`_forward_tilt`).
        """
        if not self._surface:
            return torch.stack([self._forward_tilt(V_prepared, int(i)) for i in idx])
        idx = idx.to(self.quaternions.device)
        p = self.propagation
        model = p.scattering_model
        flat = model in ("projection", "ctf")
        scattering = scattering_engine(
            self._pad_nxy,
            self.voxel_size,
            self.voltage,
            model,
            p.klim,
            p.oversample,
            p.ews_curvature_sign,
            self.nz,
            p.alpha,
            0.0,
            str(V_prepared.device),
        )
        spectrum = surface_slice_sum(
            V_prepared,
            self.quaternions[idx],
            self.translations[idx],
            pad_nxy=self._pad_nxy,
            pixel_size=self.voxel_size,
            wavelength=scattering.wavelength,
            ews_curvature_sign=p.ews_curvature_sign,
            flat=flat,
        )
        waves = ExitWave(
            array=scattering.from_slice_sum(spectrum, self.nz),
            pixel_size=self.voxel_size,
            voltage=self.voltage,
            nxy=self.nxy,
            pad_nxy=self._pad_nxy,
            oversample=self._image_oversample,
            # The surface's wave is taken at the unrotated depth's exit
            # face whatever the tilt, so the shift is the volume's own.
            defocus_shift=self._defocus_shift_angstrom,
            ids=idx.cpu(),
            linear=model == "ctf",
            alpha=p.alpha,
        )
        ctf = CTF({k: getattr(self, k)[idx] for k in self.ctf_params})
        image = waves.apply_ctf(
            ctf,
            backend=self.optics.aberration_backend,
            lpp_params=self.optics.lpp_params,
            phase_plate=self.optics.phase_plate,
            phase_plate_cuton=self.optics.phase_plate_cuton,
        ).intensity(cc=None)
        field = center_crop(image.pixels(), self.nxy, dim=(-2, -1))
        return self.dose_per_pixel[idx][:, None, None] * field

    def forward(self, tilt_idx: int) -> torch.Tensor:
        """Simulate tilt image ``tilt_idx`` using the current volume."""
        return self._forward_tilt(self._prepare_volume(), tilt_idx)

    # ------------------------------------------------------------------ #
    # Loss                                                                 #
    # ------------------------------------------------------------------ #

    def _compute_loss(
        self,
        sim: torch.Tensor,
        obs: torch.Tensor,
        tilt_idx: int,
    ) -> torch.Tensor:
        """
        Squared residual per pixel over the tilt's Poisson variance, over the
        pixels the forward model can predict (:meth:`_loss_mask`) when
        ``use_fov_mask``.

        Dividing by the expected counts per pixel makes the loss a per-pixel
        chi-squared under Poisson statistics. Shot noise then contributes
        about one per pixel at any dose, while a real misfit weighs in
        proportion to the dose, as the evidence for it does. Without the
        division both would scale with the square of the dose.
        """
        mse = F.mse_loss(sim, obs, reduction="none") / self.dose_per_pixel[tilt_idx]
        mask = self._compared(tilt_idx)
        if mask is not None:
            # Normalise by the number of valid pixels so that all tilt
            # angles contribute equal gradient signal regardless of FOV
            # size (which shrinks by cos(θ) at high tilt).
            return (mse * mask).sum() / mask.sum().clamp_min(1.0)
        return mse.mean()

    def _compared(self, tilt_idx: int) -> torch.Tensor | None:
        """
        Weights of tilt ``tilt_idx``'s pixels in the loss, or ``None`` for all.

        The product of :meth:`_loss_mask`, when ``use_fov_mask``, and the
        tilt's ``pixel_mask``.
        """
        mask = self._loss_mask(tilt_idx) if self.use_fov_mask else None
        if self.pixel_mask is None:
            return mask
        chosen = self.pixel_mask[tilt_idx].float()
        return chosen if mask is None else mask * chosen

    def _half_chi2(
        self, sim: torch.Tensor, obs: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Half the chi-squared of tilts ``idx``, summed over their compared pixels.

        The full-batch solver's data term: every compared pixel weighs the
        same, where :meth:`_compute_loss` weighs every tilt the same.
        """
        idx = idx.to(self.dose_per_pixel.device)
        r2 = (sim - obs) ** 2 / self.dose_per_pixel[idx][:, None, None]
        masks = [self._compared(int(i)) for i in idx]
        if any(m is not None for m in masks):
            ones = torch.ones(self.nxy, self.nxy, device=r2.device)
            r2 = r2 * torch.stack([ones if m is None else m for m in masks])
        # In double precision, as `regularisation` sums.
        return 0.5 * r2.sum(dtype=torch.float64)

    def _n_valid_pixels(self, tilt_idx: int) -> int:
        """Pixels of tilt ``tilt_idx`` the loss compares."""
        mask = self._compared(tilt_idx)
        return self.nxy * self.nxy if mask is None else int(mask.sum().item())

    def regularisation(self, V: torch.Tensor) -> torch.Tensor:
        """
        The priors on the volume, in the units of the chi-squared sum.

        ``smoothness / 2 * sum |grad V|^2`` plus ``tv_weight * sum
        h(|grad V|)``, with forward differences and the pseudo-Huber function
        ``h(g) = tv_delta * (sqrt(1 + g^2 / tv_delta^2) - 1)``, which is
        ``g^2 / (2 tv_delta)`` for gradients well below ``tv_delta`` and
        ``|g|`` well above it. The quadratic prior damps what the data do
        not determine without preferring any structure. The edge-preserving
        one prefers compact objects in a flat solvent, which is what lets it
        restore part of the missing wedge; it is the stronger assumption.
        A voxel on the volume's last face along an axis has no forward
        neighbour there, and that difference is taken as zero.
        """
        out = V.new_zeros((), dtype=torch.float64)
        if self.smoothness == 0 and self.tv_weight == 0:
            return out
        # Forward differences at every voxel, zero where the neighbour would
        # lie outside the volume, so that the voxels of the last faces are
        # penalised like the rest.
        g2 = torch.zeros_like(V)
        g2[:-1] += (V[1:] - V[:-1]) ** 2
        g2[:, :-1] += (V[:, 1:] - V[:, :-1]) ** 2
        g2[:, :, :-1] += (V[:, :, 1:] - V[:, :, :-1]) ** 2
        # Summed in double precision: the solver's line search takes
        # differences of the objective that single precision does not resolve.
        if self.smoothness > 0:
            out = out + 0.5 * self.smoothness * g2.sum(dtype=torch.float64)
        if self.tv_weight > 0:
            d = self.tv_delta
            huber = torch.sqrt(1 + g2 / (d * d)) - 1
            out = out + self.tv_weight * d * huber.sum(dtype=torch.float64)
        return out

    # ------------------------------------------------------------------ #
    # Optimisation                                                         #
    # ------------------------------------------------------------------ #

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[LRScheduler]]:
        """Build AdamW optimiser and LR scheduler for V."""
        return self._volume_optimizer()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Run one manual optimisation step over a batch of tilt images.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            ``(tilt_images, tilt_indices)`` from the dataloader, shapes
            ``(B, H, W)`` and ``(B,)``.
        batch_idx : int
            Batch index (unused).

        Returns
        -------
        torch.Tensor
            Scalar loss for the current batch, over every rank, detached.

        Notes
        -----
        Under several devices each rank holds a share of the global batch
        and weights its mean loss by that share, so the gradients `_step`
        sums across ranks are the global batch's.
        """
        obs_images, tilt_indices = batch
        n = len(tilt_indices)
        if n == 0:
            return self._idle_step()

        # Prepare V once per step (the taper is a deterministic transform)
        V_prepared = self._prepare_volume()

        total_loss = torch.tensor(0.0, device=self.device)
        # One host transfer for the batch's tilt indices, not a sync per tilt.
        sims = self._forward_tilts(V_prepared, tilt_indices)
        for i, idx in enumerate(tilt_indices.tolist()):
            total_loss = total_loss + self._compute_loss(sims[i], obs_images[i], idx)

        loss = total_loss / n
        if self.smoothness > 0 or self.tv_weight > 0:
            # The priors are in the units of the chi-squared sum over the
            # series; the step's loss is a mean per pixel over its tilts.
            series = 0.5 * self.nxy * self.nxy * len(self.quaternions)
            loss = loss + (self.regularisation(self.V) / series).to(loss.dtype)
        total = self._batch_total(n)
        if total != n:
            loss = loss * (n / total)

        # Summed across ranks under several devices for logging only --
        # `loss` itself (fed to `_optimise` below) stays this rank's
        # share-weighted value; `_step` sums the gradients itself. Kept on
        # the device, as in `Reconstructor`: a `.cpu()` here is a host sync
        # per step. The lists are read once, when the metrics are built.
        logged = self._gather_for_logging(loss).detach()
        self.log_total_loss.append(logged)

        self._optimise(loss)
        return logged

    # ------------------------------------------------------------------ #
    # Epoch / fit callbacks                                                #
    # ------------------------------------------------------------------ #

    def on_fit_start(self) -> None:
        """Create run directory and write hyperparameter metadata.

        Rank-0-only under multi-GPU: every replica would otherwise
        race to create/write the same files.
        """
        if not self._make_run_dir():
            return
        assert self._run_dir is not None

        if self.kmask is not None:
            torch.save(self.kmask.detach().cpu(), self._run_dir / "kmask.pt")

        meta: dict[str, Any] = dict(self.hparams)
        (self._run_dir / "params.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
        logger.info(f"Run directory: {self._run_dir}")

    def on_train_epoch_end(self) -> None:
        """Save per-epoch volume as MRC.

        Rank-0-only under multi-GPU: ``self.V`` is already identical
        across replicas at this point (kept in sync by the gradient
        all-reduce in ``_optimise`` every step), so every rank would
        otherwise race to write the same file.
        """
        if self._run_dir is None or not self.is_global_zero:
            return
        epoch = self.current_epoch + 1
        self._save_volume(
            self._volume_cpu(), self._run_dir / "epochs" / f"{epoch:03d}.mrc"
        )

    def on_fit_end(self) -> None:
        """Save final reconstructed volume and training metrics.

        Rank-0-only under multi-GPU, same reasoning as
        ``on_train_epoch_end``.
        """
        self._save_metrics()
        if self._run_dir is None or not self.is_global_zero:
            return
        volume_path = self._run_dir / "volume.mrc"
        self._save_volume(self._volume_cpu(), volume_path)
        logger.info(f"Saved final volume → {volume_path}")
