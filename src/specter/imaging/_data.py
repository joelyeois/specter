"""
Per-image data: where the specimen sits for each image (`Poses`), how the lens
images it (`CTF`), and the electrons each image received (`Exposure`).

Each is matched to images by position. `Poses.ids` numbers every image within
its stack and is what the per-image seeds derive from, so a slice of a stack
draws exactly what the whole stack drew for those images; a `CTF` read from a
dataset carries the same ids, and the stage that pairs it with a wave checks
them.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import Any, TypeVar

import roma
import torch

from .. import rotations
from ..config._scalar_range import ScalarOrRange, parse_scalar_or_range

__all__ = ["CTF", "Exposure", "Poses"]

_Group = TypeVar("_Group")


def settings_group(
    cls: type[_Group], given: _Group | None, settings: dict[str, Any]
) -> _Group:
    """
    A stage's settings group: the object given, or one built from its fields
    passed as keywords. Every verb takes its group either way, never both.
    """
    if given is None:
        return cls(**settings)
    if settings:
        raise TypeError(
            f"pass a {cls.__name__} or its fields as keywords, not both "
            f"(got {sorted(settings)})"
        )
    return given


def per_image(
    value: float | torch.Tensor,
    n: int,
    name: str,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    `value` as a float32 tensor of length `n`, a scalar broadcast; on
    `device` when given, else where `value` already is (the CPU for a
    number).
    """
    t = torch.as_tensor(value, dtype=torch.float32, device=device).flatten()
    if len(t) == 1:
        return t.expand(n).clone()
    if len(t) != n:
        raise ValueError(f"{name} has {len(t)} entries for {n} images")
    return t


def sample_uniform(value: ScalarOrRange, n: int) -> torch.Tensor:
    """`n` draws from a scalar or a ``[low, high]`` range, uniform per image."""
    low, high = parse_scalar_or_range(value)
    return torch.rand(n) * (high - low) + low


#: Keys for the seeds of the per-image draws made here, distinct from the
#: imaging stages' (``imaging._stages.SEED_STAGE_*``, which take an image id
#: as well) so that a run's poses and its specimens never share a stream.
_SEED_POSES = 100
_SEED_CTF = 101


def _drawn(seed: int | None, key: int) -> contextlib.AbstractContextManager[object]:
    """The global CPU generator seeded from ``(seed, key)``, or left alone."""
    if seed is None:
        return contextlib.nullcontext()
    from ..random_seed import derive_seed, seeded_rng

    return seeded_rng(derive_seed(seed, key))


@dataclass(frozen=True)
class Poses:
    """
    One orientation and in-plane shift per image.

    Attributes
    ----------
    quaternions : torch.Tensor
        Unit quaternions, ``(n, 4)``, in roma's convention.
    shifts : torch.Tensor
        In-plane shifts ``(x, y)`` in Angstrom, ``(n, 2)``.
    ids : torch.Tensor
        Each image's index in its stack, ``(n,)``. Seeds derive from it.
    """

    quaternions: torch.Tensor
    shifts: torch.Tensor
    ids: torch.Tensor

    @classmethod
    def random(cls, n: int, shift: float = 0.0, *, seed: int | None = None) -> Poses:
        """
        `n` uniformly random orientations, numbered from 0.

        Parameters
        ----------
        n : int
            Image count.
        shift : float, optional
            Largest in-plane shift along each axis, in Angstrom; each image
            is shifted uniformly within ``[-shift, shift]``. Default 0.
        seed : int, optional
            Draw from this seed rather than the global generator, which is
            left untouched. Default None.

        References
        ----------
        .. [1] Shoemake, K. (1992). Uniform random rotations. In *Graphics
           Gems III* (pp. 124–132). Elsevier.
           https://doi.org/10.1016/B978-0-08-050755-2.50036-1
        """
        with _drawn(seed, _SEED_POSES):
            quaternions = rotations.random_quaternion(n).reshape(n, 4)
            shifts = 2 * (torch.rand(n, 2) - 0.5) * shift
        return cls(quaternions, shifts, torch.arange(n))

    @classmethod
    def tilt_series(
        cls,
        angles: torch.Tensor | list[float],
        *,
        axis: str = "x",
        shifts: torch.Tensor | None = None,
    ) -> Poses:
        """
        One pose per tilt angle, in degrees, about the ``"x"`` or ``"y"`` axis.

        The axis defaults to `TiltGeometry`'s. The ids number the tilts in
        acquisition order.
        """
        theta = torch.deg2rad(torch.as_tensor(angles, dtype=torch.float32))
        zeros = torch.zeros_like(theta)
        if axis == "x":
            rotvecs = torch.stack([theta, zeros, zeros], dim=-1)
        elif axis == "y":
            rotvecs = torch.stack([zeros, theta, zeros], dim=-1)
        else:
            raise ValueError(f"tilt axis must be 'x' or 'y', not {axis!r}")
        n = len(theta)
        return cls(
            roma.rotvec_to_unitquat(rotvecs),
            torch.zeros(n, 2) if shifts is None else torch.as_tensor(shifts),
            torch.arange(n),
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: slice | torch.Tensor) -> Poses:
        return Poses(self.quaternions[i], self.shifts[i], self.ids[i])


@dataclass(frozen=True)
class CTF:
    """
    Per-image transfer-function terms, the defocus that of the specimen's midplane.

    The defocus refers to the specimen's scattering centroid, the plane a CTF
    fit reports and a CryoSPARC/RELION file records: the box's midplane, or a
    tilt's or an ice profile's own centroid. The exit wave carries the offset
    from where it was computed (:attr:`~specter.imaging.ExitWave.defocus_shift`).

    Held as the ``ctf_params`` dict the aberration engines take, in
    CryoSPARC's units: ``dfu``, ``dfv`` and ``cs`` in Angstrom, ``dfang`` in
    degrees, ``phaseshift`` and the beam tilts in radians.

    Attributes
    ----------
    params : dict of str to torch.Tensor
        One value per image for each term.
    ids : torch.Tensor, optional
        The images the terms belong to, when they were read from a dataset;
        :meth:`~specter.imaging.ExitWave.apply_ctf` refuses a wave whose ids
        differ.
    """

    params: dict[str, torch.Tensor]
    ids: torch.Tensor | None = None

    @classmethod
    def from_values(
        cls,
        defocus: float | torch.Tensor,
        *,
        n: int | None = None,
        astigmatism: float | torch.Tensor = 0.0,
        astigmatism_angle: float | torch.Tensor = 0.0,
        cs: float = 2.7,
        phase_shift: float | torch.Tensor = 0.0,
    ) -> CTF:
        """
        Build from readable units, one value or one per image for each term.

        Parameters
        ----------
        defocus : float or torch.Tensor
            Underfocus along the major axis, in Angstrom.
        n : int, optional
            Image count, when every argument is a scalar.
        astigmatism : float or torch.Tensor, optional
            ``dfu - dfv`` in Angstrom. Default 0.
        astigmatism_angle : float or torch.Tensor, optional
            Major-axis angle in degrees. Default 0.
        cs : float, optional
            Spherical aberration in millimetres. Default 2.7.
        phase_shift : float or torch.Tensor, optional
            Phase-plate shift in degrees. Default 0.
        """
        n = n if n is not None else torch.as_tensor(defocus).numel()
        # Every term on the defocus's device, so a CUDA defocus gives a CUDA
        # CTF rather than a mix the first subtraction refuses.
        device = defocus.device if isinstance(defocus, torch.Tensor) else None
        dfu = per_image(defocus, n, "defocus")
        return cls(
            {
                "cs": torch.full((n,), cs * 1e7, device=dfu.device),
                "dfu": dfu,
                "dfv": dfu - per_image(astigmatism, n, "astigmatism", device),
                "dfang": per_image(astigmatism_angle, n, "astigmatism_angle", device),
                "phaseshift": torch.deg2rad(
                    per_image(phase_shift, n, "phase_shift", device)
                ),
            }
        )

    @classmethod
    def sample(
        cls,
        n: int,
        *,
        defocus: ScalarOrRange,
        astigmatism: ScalarOrRange = 0.0,
        astigmatism_angle: ScalarOrRange = 0.0,
        cs: float = 2.7,
        phase_shift: ScalarOrRange = 0.0,
        seed: int | None = None,
    ) -> CTF:
        """
        Draw `n` sets of terms, each a scalar or a ``[low, high]`` range
        sampled uniformly per image; units as :meth:`from_values`. Draws from
        `seed` when given, else from the global generator.
        """
        with _drawn(seed, _SEED_CTF):
            return cls.from_values(
                sample_uniform(defocus, n),
                n=n,
                astigmatism=sample_uniform(astigmatism, n),
                astigmatism_angle=sample_uniform(astigmatism_angle, n),
                cs=cs,
                phase_shift=sample_uniform(phase_shift, n),
            )

    def __len__(self) -> int:
        return len(next(iter(self.params.values())))

    def __getitem__(self, i: slice | torch.Tensor) -> CTF:
        return CTF(
            {k: v[i] for k, v in self.params.items()},
            None if self.ids is None else self.ids[i],
        )


@dataclass(frozen=True)
class Exposure:
    """
    The electrons each image received, and how they were fractionated.

    Given to :meth:`~specter.imaging.Specimen.scatter`, where the specimen is
    damaged and the solvent decorrelates during the exposure, and to
    :meth:`~specter.imaging.Image.detect`, where the dose becomes counts and
    the frames set the coincidence loss.

    Attributes
    ----------
    dose : torch.Tensor or float
        Dose per image in e-/A^2, ``(n,)``, or one value for every image.
    pre_exposure : torch.Tensor, optional
        Dose already received when each exposure started, ``(n,)``.
    n_frames : int, optional
        Movie frames the dose was split into.
    weights : torch.Tensor, optional
        Per-frame exposure-filter weights, ``(n_frames, n_bins)``.
    weights_max_frequency : float, optional
        Frequency of the weights' last bin, in 1/Angstrom.
    weighted : bool
        Whether the image is an exposure-filtered (dose-weighted) movie sum,
        as single-particle images and micrographs are, rather than a plain
        sum of its frames, as a tilt is. Sets the dose envelope's form.
        Default True.

    References
    ----------
    .. [1] Grant, T., & Grigorieff, N. (2015). Measuring the optimal exposure
       for single particle cryo-EM using a 2.6 Å reconstruction of rotavirus
       VP6. *eLife*, 4, e06980. https://doi.org/10.7554/eLife.06980
    """

    dose: torch.Tensor | float
    pre_exposure: torch.Tensor | None = None
    n_frames: int | None = None
    weights: torch.Tensor | None = None
    weights_max_frequency: float | None = None
    weighted: bool = True

    @classmethod
    def tilt_series(
        cls,
        dose: torch.Tensor | float,
        n_tilts: int,
        *,
        n_frames: int | None = None,
        weights: torch.Tensor | None = None,
        weights_max_frequency: float | None = None,
    ) -> Exposure:
        """
        The exposures of a tilt series, in acquisition order.

        Each tilt is a plain sum of its frames, taken after the summed dose
        of the tilts before it.
        """
        doses = per_image(dose, n_tilts, "dose")
        return cls(
            doses,
            pre_exposure=torch.cumsum(doses, 0) - doses,
            n_frames=n_frames,
            weights=weights,
            weights_max_frequency=weights_max_frequency,
            weighted=False,
        )

    @classmethod
    def from_weights_file(
        cls,
        dose: torch.Tensor | float,
        path: str,
        *,
        n_frames: int | None = None,
        max_frequency: float | None = None,
    ) -> Exposure:
        """
        An exposure summed under a motion-correction job's own frame weights.

        The frame count and the weights' frequency axis come from the files
        (:func:`~specter.io.load_dose_weights`); a disagreeing `n_frames` is
        an error.
        """
        from ..io import load_dose_weights

        weights, axis = load_dose_weights(path, max_frequency=max_frequency)
        if n_frames is not None and weights.shape[0] != n_frames:
            raise ValueError(f"{path} has {weights.shape[0]} frames, not {n_frames}")
        return cls(
            dose,
            n_frames=int(weights.shape[0]),
            weights=weights,
            weights_max_frequency=axis,
        )

    def doses(self, n: int) -> torch.Tensor:
        """The dose of each of `n` images, float32."""
        return per_image(self.dose, n, "dose")

    def __getitem__(self, i: slice | torch.Tensor) -> Exposure:
        dose = torch.as_tensor(self.dose)
        pre = self.pre_exposure
        return replace(
            self,
            dose=dose if dose.numel() == 1 else dose[i],
            pre_exposure=None if pre is None else pre[i],
        )
