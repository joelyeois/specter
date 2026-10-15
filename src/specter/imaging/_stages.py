"""
The stages of an image's forward pass, as plain functions.

Each takes what it needs as arguments rather than reading it off an object,
so `Specimen`, `Sample`, `TiltSeries`, and anything else that composes the
forward model, call the same code: crowding, the free fraction a specimen leaves to
water, the ice draw and its blend, the solvent's exposure filter, the
mean-free-path absorption field, the transfer function on the image grid,
and the sampling back to the pixel grid. Damage, the aperture low-pass,
scattering and detection are already free functions or modules of their own
(:func:`~specter.potential.apply_dose_damage`,
:func:`~specter.potential.aperture_lowpass`,
:class:`~specter.scattering.Scattering`,
:class:`~specter.microscope.Detector`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch

from ..cpu_threads import limited_cpu_threads
from ..crowding import CrowdWithDuplicates
from ..ice import IceBank, RandomIcemaker
from ..ice._blend import IceSlabBlender
from ..potential import (
    FULL_OCCUPANCY_POTENTIAL_V,
    aperture_mfp_ice,
    aperture_mfp_protein,
    ice_inelastic_mfp,
    inelastic_absorption_potential,
    potential_occupancy_slabs,
)
from ..random_seed import derive_seed, seeded_rng
from ..scattering._kernels import fourier_crop_xy, fourier_upsample_xy
from ..settings import Camera, Ice, Optics, Propagation

__all__ = [
    "RemovalMFPs",
    "aberrate",
    "crowd_into",
    "draw_ice",
    "free_fraction",
    "mfp_absorption_field",
    "read_dose_weights",
    "reflect_index",
    "resolve_removal_mfps",
    "solvate",
    "solvate_chunk_slices",
    "solvent_exposure_filter",
    "to_pixel_grid",
]


def read_dose_weights(camera: Camera) -> tuple[torch.Tensor | None, float | None]:
    """
    The exposure filter's per-frame weights and their frequency axis, if any.

    Both come from :func:`~specter.io.load_dose_weights`, which derives the
    axis from the motion-correction job's own files rather than letting a
    caller assume one.

    Parameters
    ----------
    camera : Camera
        Supplies ``dose_weights_path``, ``dose_weights_max_frequency`` and
        ``n_frames``.

    Returns
    -------
    tuple of (torch.Tensor or None, float or None)
        Weights of shape ``(n_frames, n_bins)`` and the frequency of their
        last bin in 1/Angstrom, or ``(None, None)`` without a weights file.

    Raises
    ------
    ValueError
        If weights are given without ``n_frames``, or the frame count
        disagrees with the file's.
    """
    path = camera.dose_weights_path
    if path is None:
        return None, None
    if camera.n_frames is None:
        raise ValueError("dose_weights_path requires n_frames to be set.")
    from ..io import load_dose_weights

    w, max_freq = load_dose_weights(
        path, max_frequency=camera.dose_weights_max_frequency
    )
    if w.shape[0] != camera.n_frames:
        raise ValueError(
            f"{path} has {w.shape[0]} frames but n_frames={camera.n_frames}"
        )
    return w, max_freq


def solvent_exposure_filter(
    ice: Ice,
    camera: Camera,
    pixel_size: float,
    dose: float,
    weights: torch.Tensor | None,
    weights_max_frequency: float | None,
) -> Callable[[torch.Tensor], None] | None:
    """
    The in-place solvent-exposure filter for one exposure, or None.

    :func:`~specter.ice.apply_solvent_exposure` with the exposure's frame
    structure: ``camera.n_frames`` frames (one when unset) summed under
    `weights`. Shared by `Sample` and `TiltSeries`, which filter a whole ice
    canvas before it is blended, and by `Specimen`, which hands it to
    :func:`solvate` to filter each particle's ice draw.

    Parameters
    ----------
    ice : Ice
        Supplies ``motion_variance``; None returns None.
    camera : Camera
        Supplies ``n_frames``.
    pixel_size : float
        Voxel size in Angstrom.
    dose : float
        The exposure's dose in e-/A^2.
    weights, weights_max_frequency
        As returned by :func:`read_dose_weights`.

    Returns
    -------
    callable or None
        Filters a ``(B, Z, Y, X)`` ice canvas in place.
    """
    if ice.motion_variance is None:
        return None
    from ..ice._exposure import apply_solvent_exposure

    variance = ice.motion_variance
    n_frames = camera.n_frames or 1

    def apply(canvas: torch.Tensor) -> None:
        apply_solvent_exposure(
            canvas,
            pixel_size,
            dose,
            variance,
            n_frames,
            weights,
            weights_max_frequency,
        )

    return apply


@dataclass(frozen=True)
class RemovalMFPs:
    """
    Resolved mean free paths for an electron leaving the image, in Angstrom.

    Built by :func:`resolve_removal_mfps` from the settings every simulation
    shares, so particles, micrographs and tilt series charge the same loss
    for the same configuration.

    Attributes
    ----------
    inelastic_solvent : float
        Inelastic mean free path of the solvent: the configured
        ``inelastic_mfp_solvent``, else the measured or estimated value for
        amorphous ice at this voltage.
    inelastic_specimen : float or None
        Inelastic mean free path of the specimen, or None when the specimen
        absorbs at the solvent's rate.
    objective_aperture : float or None
        Objective aperture semi-angle in milliradians, or None.
    voltage : float
        Accelerating voltage in kV.
    """

    inelastic_solvent: float
    inelastic_specimen: float | None
    objective_aperture: float | None
    voltage: float

    def removal(self, material: str) -> float:
        """
        Mean free path for leaving the image, inelastic plus aperture, in A.

        Two independent loss channels add as rates. Without an objective
        aperture this is the configured inelastic mean free path unchanged.

        Parameters
        ----------
        material : {"solvent", "specimen"}
            Which material. ``"specimen"`` requires ``inelastic_specimen``.

        Returns
        -------
        float
            Mean free path in Angstrom.
        """
        if material == "solvent":
            inelastic = self.inelastic_solvent
        else:
            inelastic = cast(float, self.inelastic_specimen)
        if self.objective_aperture is None:
            return inelastic
        aperture = (
            aperture_mfp_ice if material == "solvent" else aperture_mfp_protein
        )(self.objective_aperture, self.voltage)
        return 1.0 / (1.0 / inelastic + 1.0 / aperture)


def resolve_removal_mfps(
    propagation: Propagation, optics: Optics | None, voltage: float
) -> RemovalMFPs | None:
    """
    Resolve the mean-free-path absorption settings, or None under ``"alpha"``.

    Resolved once, where the voltage is known, so a voltage with no measured
    ice value fails or warns at construction rather than mid-run.

    Parameters
    ----------
    propagation : Propagation
        Supplies ``absorption_model`` and the two mean-free-path overrides.
    optics : Optics or None
        Supplies the objective aperture, if any.
    voltage : float
        Accelerating voltage in kV.

    Returns
    -------
    RemovalMFPs or None
        The resolved mean free paths, or None when
        ``absorption_model != "inelastic_mfp"``.

    References
    ----------
    .. [1] Rice, W. J., Cheng, A., Noble, A. J., Eng, E. T., Kim, L. Y.,
       Carragher, B., & Potter, C. S. (2018). Routine determination of ice
       thickness for cryo-EM grids. *Journal of Structural Biology*, 204(1),
       38–44. https://doi.org/10.1016/j.jsb.2018.06.007
    .. [2] Grimm, R., Typke, D., Bärmann, M., & Baumeister, W. (1996).
       Determination of the inelastic mean free path in ice by examination of
       tilted vesicles and automated most probable loss imaging.
       *Ultramicroscopy*, 63(3–4), 169–179.
       https://doi.org/10.1016/0304-3991(96)00035-6
    """
    if propagation.absorption_model != "inelastic_mfp":
        return None
    return RemovalMFPs(
        inelastic_solvent=(
            propagation.inelastic_mfp_solvent
            if propagation.inelastic_mfp_solvent is not None
            else ice_inelastic_mfp(voltage)
        ),
        inelastic_specimen=propagation.inelastic_mfp_specimen,
        objective_aperture=None if optics is None else optics.objective_aperture,
        voltage=voltage,
    )


def mfp_absorption_field(
    specimen: torch.Tensor,
    pixel_size: float,
    removal_mfps: RemovalMFPs,
    has_solvent: bool,
    full_potential: float | torch.Tensor = FULL_OCCUPANCY_POTENTIAL_V,
) -> torch.Tensor:
    """
    The imaginary potential of a dry specimen from its removal mean free paths.

    :func:`~specter.potential.inelastic_absorption_potential` with the
    solvent and specimen rates of `removal_mfps`: inelastic plus, with an
    objective aperture, elastic scattering beyond it. Shared by `Specimen`,
    `Sample` and `TiltSeries`.

    Parameters
    ----------
    specimen : torch.Tensor
        The potential of the specimen alone, before any solvent was blended
        in, shape ``(..., Z, Y, X)``. Occupancy read off a solvated volume is
        full everywhere.
    pixel_size : float
        Voxel size in Angstrom.
    removal_mfps : RemovalMFPs
        The resolved mean free paths.
    has_solvent : bool
        Whether solvent surrounds the specimen. Without it the solvent term
        is dropped: vacuum absorbs nothing.
    full_potential : float or torch.Tensor, optional
        Occupancy reference. Default
        :data:`~specter.potential.FULL_OCCUPANCY_POTENTIAL_V`.

    Returns
    -------
    torch.Tensor
        Absorption potential in volts, same shape as `specimen`. Uniform when
        `removal_mfps` has no specimen mean free path.

    References
    ----------
    .. [1] Vulović, M., Ravelli, R. B. G., van Vliet, L. J., Koster, A. J.,
       Lazić, I., Lücken, U., Rullgård, H., Öktem, O., & Rieger, B. (2013). Image
       formation modeling in cryo-electron microscopy. *Journal of Structural
       Biology*, 183(1), 19–32. https://doi.org/10.1016/j.jsb.2013.05.008
    """
    return inelastic_absorption_potential(
        specimen,
        pixel_size,
        removal_mfps.voltage,
        mfp_solvent_A=(
            removal_mfps.removal("solvent") if has_solvent else float("inf")
        ),
        mfp_specimen_A=(
            None
            if removal_mfps.inelastic_specimen is None
            else removal_mfps.removal("specimen")
        ),
        full_potential=full_potential,
    )


# The stochastic stages of one image's forward pass, each seeded from
# (run seed, particle index, stage) when the generator has a seed, so no stage
# can read another's stream. Values are arbitrary but fixed: changing one
# changes every seeded stack.
SEED_STAGE_CROWD = 0
SEED_STAGE_ICE = 1
SEED_STAGE_DETECTOR = 2
SEED_STAGE_SPECIMEN = 3


def packer_seed(item_seed: int, packing_seed: int | None) -> int:
    """
    The shape packer's seed for one item of a seeded run.

    The item's own seed, or, when the caller also fixed the packer's seed
    (`Packing.seed`), one derived from both, so that setting still changes
    the packing without the item ceasing to be a function of its index.
    """
    return item_seed if packing_seed is None else derive_seed(item_seed, packing_seed)


#: Voxel budget for one solvate slab, and the slice count it is capped at.
#: The occupancy blur reads `halo` slices past each slab on both sides, so
#: the redundant work is (chunk + 2 halo) / chunk: 2x at the 16 slices a
#: 2**24 budget gave a 1024^2 canvas, 1.25x at 64. Past 64 the slab only
#: grows the blur's transposed intermediates (a whole 256-slice volume in
#: one slab took a 256-pixel box from 2.4 to 3.3 GB) for a halo saving that
#: has already flattened out.
_SOLVATE_SLAB_VOXELS = 2**26
_SOLVATE_MAX_SLICES = 64


def solvate_chunk_slices(nxy: int) -> int:
    """Slices per solvate slab for an ``nxy``-wide canvas."""
    return max(1, min(_SOLVATE_SLAB_VOXELS // (nxy * nxy), _SOLVATE_MAX_SLICES))


def reflect_index(n: int, pad: int, device: torch.device) -> torch.Tensor:
    """
    Indices into an axis of length `n` that reproduce
    ``F.pad(..., (pad, pad), mode="reflect")`` along it by gathering.

    Parameters
    ----------
    n : int
        Axis length before padding.
    pad : int
        Padding on each side; must be smaller than `n`, as for ``F.pad``.
    device : torch.device
        Device for the index tensor.

    Returns
    -------
    torch.Tensor
        Shape ``(n + 2 * pad,)``, int64.
    """
    if pad >= n:
        raise ValueError(f"reflect padding {pad} must be smaller than the axis {n}")
    i = torch.arange(-pad, n + pad, device=device)
    i = torch.where(i < 0, -i, i)
    return torch.where(i >= n, 2 * (n - 1) - i, i)


def crowd_into(
    V: torch.Tensor, crowd: CrowdWithDuplicates, seeds: list[int] | None
) -> None:
    """
    Stamp neighbouring copies of the template into each item of `V`, in place.

    Stamped straight onto ``V[i]``: a separate accumulator is a second padded
    canvas held only to be added once, and a host copy of it is a 2 GB
    pageable device-to-host transfer per particle, 40% of the forward pass at
    a 512-pixel box. One thread cap for the whole batch: each duplicate
    insertion caps the pool itself, but resizing a 128-thread pool costs
    ~12 ms each way, and per chunk that was 21 s of a 28 s 64-particle run at
    128 px. Under this outer cap the inner ones find the pool already small.

    Parameters
    ----------
    V : torch.Tensor
        ``(B, Z, Y, X)`` canvases, modified in place.
    crowd : CrowdWithDuplicates
        The crowding stage.
    seeds : list of int or None
        One seed per item for its placement draws, or None for the global RNG.
    """
    with torch.no_grad(), limited_cpu_threads():
        if seeds is None:
            for i in range(len(V)):
                crowd(into=V[i])
            return
        # Every crowd draw is on the CPU (candidate positions, orientations,
        # the thinning), so a CPU-only fork is enough; the shape packer's own
        # NumPy stream is seeded through `packing_seed`, restored afterwards.
        packing_seed = crowd.packing_seed
        try:
            for i, item_seed in enumerate(seeds):
                crowd.packing_seed = packer_seed(item_seed, packing_seed)
                with seeded_rng(item_seed):
                    crowd(into=V[i])
        finally:
            crowd.packing_seed = packing_seed


def free_fraction(
    V: torch.Tensor, pixel_size: float, full: torch.Tensor
) -> torch.Tensor:
    """
    The fraction of each voxel still available to water, ``1 - occupancy``.

    Computed from `V` as given, so a caller can take it from the UNDAMAGED
    specimen and hand it to :func:`solvate` after the specimen has been
    damaged. A molecule occupies the same volume whatever its radiation
    history, and the dose envelope conserves the potential's integral while
    spreading it, so reading occupancy after damage admits ice into a solid
    interior (peak core occupancy 1.00 -> 0.81 on 1A6M at 50 e-/A^2).

    Stored as float16: a weight in [0, 1] multiplying the ice, where half
    precision costs ~5e-4 relative against a blur that is itself an
    approximation, at half the memory of a full-precision canvas.

    Parameters
    ----------
    V : torch.Tensor
        Specimen potential, ``(B, Z, Y, X)``.
    pixel_size : float
        Voxel size in Angstrom.
    full : torch.Tensor
        Occupancy reference per item, ``(B or 1, 1, 1, 1)``: the potential, in
        volts, of a voxel the specimen fills.

    Returns
    -------
    torch.Tensor
        ``(B, Z, Y, X)`` float16 in [0, 1], on `V`'s device.
    """
    chunk = solvate_chunk_slices(V.shape[-1])
    free = torch.empty(V.shape, dtype=torch.float16, device=V.device)
    for b in range(V.shape[0]):
        full_b = full[b : b + 1] if full.shape[0] > 1 else full
        for start, end, occ in potential_occupancy_slabs(
            V[b : b + 1], pixel_size, chunk, full_potential=full_b
        ):
            free[b : b + 1, start:end] = (
                occ.neg_().add_(1.0).clamp_(0.0, 1.0).to(torch.float16)
            )
            del occ
    return free


def draw_ice(
    icemaker: IceBank | RandomIcemaker,
    nxy: int,
    nz: int,
    pixel_size: float,
    relax_steps: int,
    seed: int | None,
) -> torch.Tensor:
    """One ``(1, nz, nxy, nxy)`` ice canvas, from `seed` or the global RNG."""
    if isinstance(icemaker, IceBank):
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        return icemaker.generate_big_ice(
            n=nxy,
            dx=pixel_size,
            nz=nz,
            batchsize=1,
            relax_steps=relax_steps,
            generator=generator,
        )
    # RandomIcemaker draws from the global RNG, so a seeded item runs it on
    # a forked, seeded copy of that stream.
    icemaker = cast(RandomIcemaker, icemaker)
    if seed is None:
        return icemaker.generate_ice(batchsize=1)
    with seeded_rng(seed):
        return icemaker.generate_ice(batchsize=1)


def solvate(
    V: torch.Tensor,
    icemaker: IceBank | RandomIcemaker,
    *,
    pixel_size: float,
    nxy: int,
    pad: int,
    full: torch.Tensor,
    relax_steps: int = 0,
    free: torch.Tensor | None = None,
    seeds: list[int] | None = None,
    exposure_filter: Callable[[torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """
    Embed each item of `V` in amorphous ice, in place.

    Blended a z-slab at a time, for the same reason
    :func:`~specter.ice.blend_ice_into_volume` is: the whole-volume spelling
    holds V, the ice, the occupancy, the weight and two clamp/subtract
    temporaries at once, 12 GB at ``nxy=512`` with FFT padding. The ice is
    drawn at the UNPADDED box and, with `pad`, reflect-padded into the FFT
    margin by index per slab, never materialised (0.5 GB against 2 GB at
    ``nxy=512``). One batch item at a time: the slab temporaries would
    otherwise all scale with the batch (7 GB over the live volume at three
    512-pixel boxes).

    Parameters
    ----------
    V : torch.Tensor
        ``(B, Z, Y, X)`` volume; modified in place.
    icemaker : IceBank or RandomIcemaker
        Draws each item's ``(1, Z, nxy, nxy)`` ice canvas (:func:`draw_ice`).
    pixel_size : float
        Voxel size in Angstrom.
    nxy : int
        Unpadded box size the ice is drawn at.
    pad : int
        FFT margin on each side of the box, 0 for none.
    full : torch.Tensor
        Occupancy reference per item, ``(B or 1, 1, 1, 1)``. Divided in HERE:
        the occupancy clamps to [0, 1] against whatever reference it is
        given.
    relax_steps : int, optional
        Seam relaxation steps for a tiled IceBank draw. Default 0.
    free : torch.Tensor, optional
        Precomputed free fraction (:func:`free_fraction`) to weight the ice
        by, instead of reading occupancy off `V`.
    seeds : list of int, optional
        One seed per item for its ice draw.
    exposure_filter : callable, optional
        Filters an ice canvas in place to what survives the exposure
        (solvent decorrelation); applied to the pure ice, before the
        particle's hole is cut, since the hole moves with the particle.

    Returns
    -------
    torch.Tensor
        `V`.
    """
    if seeds is not None and len(seeds) != V.shape[0]:
        raise ValueError(f"{len(seeds)} ice seeds for {V.shape[0]} volumes")
    ridx = reflect_index(nxy, pad, V.device) if pad else None
    nz = V.shape[1]
    chunk = solvate_chunk_slices(V.shape[-1])
    for b in range(V.shape[0]):
        Vb = V[b : b + 1]
        seed = None if seeds is None else seeds[b]
        ice_b = draw_ice(icemaker, nxy, nz, pixel_size, relax_steps, seed)
        ice_b = ice_b.to(V.device)
        if exposure_filter is not None:
            exposure_filter(ice_b)
        full_b = full[b : b + 1] if full.shape[0] > 1 else full
        blender = IceSlabBlender(pixel_size, full_potential=full_b)
        for start in range(0, nz, chunk):
            end = min(start + chunk, nz)
            slab = ice_b[:, start:end]
            if ridx is not None:
                slab = slab.index_select(-2, ridx).index_select(-1, ridx)
            else:
                slab = slab.clone()
            blender.add(
                Vb, slab, start, end, free=None if free is None else free[b : b + 1]
            )
            del slab
        del ice_b
    return V


def aberrate(
    exitwave: torch.Tensor,
    aberration: Callable[[torch.Tensor, dict[str, torch.Tensor]], torch.Tensor] | None,
    ctf: dict[str, torch.Tensor],
    image_oversample: int,
    pad_nxy: int,
) -> torch.Tensor:
    """
    The aberration stage, or the exit wave itself when there is none.

    On the image grid, `image_oversample` times finer than the pixels: an
    exit wave still on the pixel grid (first Born, kinematic, or an
    oversample of 1) is interpolated there first, which adds nothing but
    gives ``|psi|^2`` room for its own bandwidth. That holds with no
    aberration stage too, since the detector samples whatever it receives
    down by `image_oversample`.

    Parameters
    ----------
    exitwave : torch.Tensor
        Complex ``(B, Y, X)`` exit wave, on the pixel or the image grid.
    aberration : callable or None
        ``Aberration`` (or its adapter) built on the image grid, or None.
    ctf : dict of str to torch.Tensor
        Per-image transfer-function parameters.
    image_oversample : int
        Image grid points per pixel, per axis.
    pad_nxy : int
        Size of the (FFT-padded) pixel grid.

    Returns
    -------
    torch.Tensor
        The wave at the detector, on the image grid.
    """
    if image_oversample > 1 and exitwave.shape[-1] == pad_nxy:
        exitwave = fourier_upsample_xy(exitwave, image_oversample)
    if aberration is None:
        return exitwave
    return aberration(exitwave, ctf)


def to_pixel_grid(x: torch.Tensor, pad_nxy: int, image_oversample: int) -> torch.Tensor:
    """
    A wave or an intensity on the pixel grid, whichever grid it was formed on.

    Fourier-cropped from the image grid when it is there; returned as given
    when it is already on the pixel grid.
    """
    if image_oversample > 1 and x.shape[-1] == pad_nxy * image_oversample:
        return fourier_crop_xy(x, pad_nxy)
    return x
