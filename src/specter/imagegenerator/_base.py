"""
`BaseImager`: the Lightning base every simulator shares -- settings
bundles, per-image parameters, the optics stage and the detector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from collections.abc import Callable, Sequence

if TYPE_CHECKING:
    from ..inelastic import (
        ExposureStep,
        PotentialState,
        PlasmonFilter,
        FrozenPlasmonResult,
    )

import lightning as L
import torch

from specter.detectors import (
    dqe0_for_detector,
    falcon4i_200kv,
    falcon4i_300kv,
    k2_300kv,
    k3_200kv,
    k3_300kv,
    perfect_detector,
)

from ..aberrations import (
    Aberration,
    aberration_model_for_scattering,
    defocus_midplane_shift,
)
from ..arrays import compute_nz, pad_volume
from ..ctf import LegacyAberrationAdapter
from ..microscope import Detector
from ..potential import (
    FULL_OCCUPANCY_POTENTIAL_V,
    absorption_potential,
    aperture_mfp_ice,
    aperture_mfp_protein,
    ice_inelastic_mfp,
    inelastic_absorption_potential,
)
from ..settings import Camera, Envelopes, Optics, Propagation

__all__ = [
    "BaseImager",
    "compute_nz",
    "pad_volume",
]


class BaseImager(L.LightningModule):
    """
    Shared base for all image-generation classes.

    Registers CTF parameters, per-image scalars (dose, coincidence radius,
    potential scale) as Lightning buffers, initialises the detector MTF, and
    exposes helpers for optics initialisation and defocus shifting.

    Every concrete generator should call ``super().__init__()`` with these
    arguments, then call ``_init_optics()`` once ``nz`` and ``pad_nxy`` are
    known.

    Parameters
    ----------
    pixel_size : float
        Pixel size in Å.
    voltage : float
        Electron beam accelerating voltage in kV.
    dose_per_angstrom : float or torch.Tensor
        Total electron dose (fluence) per image in e⁻/Å². Scalar, or a 1-D
        tensor of length n giving a separate dose for each image.
    nxy : int
        Unpadded image size in pixels.
    nz : int
        Number of Z slices in the simulation volume.
    pad_nxy : int, optional
        XY size after FFT padding. Defaults to ``nxy``.
    propagation : Propagation, optional
        How the exit wave is computed. Default ``Propagation()``.
    optics : Optics, optional
        The aberration stage. ``None`` skips it: the detector sees the bare
        exit wave, and ``ctf_params`` may be omitted. Default ``Optics()``.
    envelopes : Envelopes, optional
        Coherence and radiation-damage envelopes. Default ``Envelopes()``,
        every envelope off.
    camera : Camera, optional
        The detector chain. Default ``Camera()``: no MTF, Poisson noise.
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices, shape (n, 2, 2).
    ctf_params : dict[str, torch.Tensor], optional
        Per-image CTF parameters; each value is a 1-D tensor of length n.
        Required unless ``optics`` is ``None``.
    progressbars : bool, optional
        Whether to show progress bars. Default True.
    verbose : bool, optional
        Whether to emit debug-level log messages. Default True.
    coincidence_radius : float or torch.Tensor, optional
        Coincidence radius in pixels. Scalar or 1-D tensor of length n.
        Default 0.0.
    potential_scale : float or torch.Tensor, optional
        Multiplier applied to the scattering potential before propagation.
        Scalar or 1-D tensor of length n. Default 1.0.
    bfactor : float or torch.Tensor or None, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function. None or 0.0 means no envelope. Default None.
    n_images : int, optional
        Number of images, which scalar per-image inputs are expanded to.
        Taken from ``ctf_params`` when those are given; otherwise from this,
        and failing that from the longest per-image input. Without any of
        the three, a scalar stays length 1 and only image 0 can be
        simulated. Default None.
    """

    def __init__(
        self,
        pixel_size: float,
        voltage: float,
        dose_per_angstrom: float | torch.Tensor,
        nxy: int,
        nz: int,
        pad_nxy: int | None = None,
        propagation: Propagation = Propagation(),
        optics: Optics | None = Optics(),
        envelopes: Envelopes = Envelopes(),
        camera: Camera = Camera(),
        anisomag: torch.Tensor | None = None,
        ctf_params: dict[str, torch.Tensor] | None = None,
        progressbars: bool = True,
        verbose: bool = True,
        coincidence_radius: float | torch.Tensor = 0.0,
        potential_scale: float | torch.Tensor = 1.0,
        bfactor: float | torch.Tensor | None = None,
        n_images: int | None = None,
    ):
        super().__init__()
        # The settings are frozen dataclasses, so a shared default instance
        # is safe. `optics=None` is meaningful (no aberration stage), which is
        # why its default is an instance rather than None.
        self.propagation = propagation
        self.optics = optics
        self.envelopes = envelopes
        self.camera = camera
        if self.optics is not None and ctf_params is None:
            raise ValueError(
                "ctf_params is required when optics is given; pass optics=None "
                "to skip the aberration stage."
            )
        self.pixel_size = pixel_size
        self.voltage = voltage
        self.progressbars = progressbars
        self.verbose = verbose
        self.nxy = nxy
        self.nz = nz
        self.pad_nxy = pad_nxy if pad_nxy is not None else nxy
        # Read often enough downstream to be worth mirroring as attributes.
        self.scattering_model = self.propagation.scattering_model
        self.alpha = self.propagation.alpha
        self.absorption_model = self.propagation.absorption_model
        self.objective_aperture = (
            None if self.optics is None else self.optics.objective_aperture
        )
        if (
            self.objective_aperture is not None
            and self.absorption_model != "inelastic_mfp"
        ):
            raise ValueError(
                "Optics(objective_aperture=...) requires "
                "Propagation(absorption_model='inelastic_mfp'): under 'alpha' the "
                "fitted amplitude contrast already stands in for aperture loss, "
                "and applying both would count it twice."
            )
        if (
            self.envelopes.dose_envelope_target == "specimen"
            and not self._supports_specimen_damage
        ):
            raise ValueError(
                f"{type(self).__name__} cannot apply the dose envelope to the "
                "specimen (Envelopes(dose_envelope_target='specimen')): it is "
                "given a volume with the solvent already in it. Use "
                "'transfer_function'."
            )
        # Set by the generators before this constructor runs.
        ice = getattr(self, "ice", None)
        if getattr(ice, "motion_variance", None) is not None:
            if not self._supports_specimen_damage:
                raise ValueError(
                    f"{type(self).__name__} does not support Ice(motion_variance=...): "
                    "the solvent exposure filter needs the ice as its own field, "
                    "which only the particle generators build."
                )
            if self.envelopes.dose_envelope and not self._damages_potential:
                raise ValueError(
                    "Ice(motion_variance=...) with the dose envelope on the "
                    "transfer function would take the solvent's structure away "
                    "twice: the envelope fades the water ring, and the exposure "
                    "filter decorrelates it. Set "
                    "Envelopes(dose_envelope_target='specimen')."
                )
        # Resolved here, where the voltage is known, so a voltage with no
        # measured ice value fails at construction rather than mid-run.
        self._inelastic_mfp_solvent: float | None = None
        if self.absorption_model == "inelastic_mfp":
            self._inelastic_mfp_solvent = (
                self.propagation.inelastic_mfp_solvent
                if self.propagation.inelastic_mfp_solvent is not None
                else ice_inelastic_mfp(self.voltage)
            )
        self.klim = self.propagation.klim
        self.ews_curvature_sign = self.propagation.ews_curvature_sign
        self.noise_model = self.camera.noise_model
        self.detector_model = self.camera.detector_model
        self.n_frames = self.camera.n_frames
        self._init_detector_mtf()

        if anisomag is None:
            self.anisomag = None
        else:
            self.register_buffer("anisomag", torch.as_tensor(anisomag))

        n: int | None
        if ctf_params is not None:
            for k, v in ctf_params.items():
                v_tensor = torch.as_tensor(v)
                if v_tensor.ndim == 0:
                    v_tensor = v_tensor.unsqueeze(0)
                self.register_buffer(k, v_tensor)
            self._ctf_param_names = list(ctf_params.keys())
            n = len(next(iter(ctf_params.values())))
        else:
            self._ctf_param_names = []
            # No CTF to count images by (optics=None): the caller's count,
            # else the longest per-image input. Without this a scalar dose
            # stayed length 1 and indexing image 1 asserted on the device.
            n = n_images
            if n is None:
                lengths = [
                    torch.as_tensor(v).numel()
                    for v in (dose_per_angstrom, coincidence_radius, potential_scale)
                ]
                if bfactor is not None:
                    lengths.append(torch.as_tensor(bfactor).numel())
                n = max(lengths) if max(lengths) > 1 else None

        # Scalar inputs are expanded to length-n so forward() can index with [idx].
        def _to_buffer(val: float | torch.Tensor, name: str) -> None:
            t = torch.as_tensor(val, dtype=torch.float32).flatten()
            if n is not None and len(t) == 1:
                t = t.expand(n).clone()
            self.register_buffer(name, t)

        _to_buffer(dose_per_angstrom, "dose_per_angstrom")
        _to_buffer(coincidence_radius, "coincidence_radius")
        _to_buffer(potential_scale, "potential_scale")
        # Decided once, on the host: testing the buffer in `forward` every
        # step (`bool(torch.all(scale == 1))`) is a device sync per step,
        # which in a training loop exposes the CPU's kernel-launch time
        # instead of overlapping it with the GPU.
        self._potential_scale_is_unity = bool(torch.all(self.potential_scale == 1.0))
        if bfactor is not None:
            if "bfactor" in self._ctf_param_names:
                self._buffers.pop("bfactor")
                self._ctf_param_names.remove("bfactor")
            _to_buffer(bfactor, "bfactor")

    # Whether each image stands for an exposure-filtered (dose-weighted) movie
    # sum. True for single-particle images and micrographs; TiltSeriesGenerator
    # overrides it, since a tilt is a plain short exposure after a pre-exposure.
    # See specter.aberrations.dose_envelope.
    _dose_weighted: bool = True

    # Whether this imager can apply the dose envelope to the specimen's own
    # potential, before the solvent is added (`potential.apply_dose_damage`).
    # The particle generators can; micrograph and tilt-series imagers receive
    # a volume with the ice already in it, so for them the envelope can only
    # act on the transfer function, where it filters the ice as hard as the
    # protein.
    _supports_specimen_damage: bool = False

    @property
    def _damages_potential(self) -> bool:
        """The dose envelope acts on the specimen potential, not the transfer function."""
        env = self.envelopes
        return env.dose_envelope and env.dose_envelope_target == "specimen"

    def _ctf_batch(self, idx: torch.Tensor | int) -> dict[str, torch.Tensor]:
        """Collect per-image transfer-function parameters for a batch."""
        ctf_batch = {k: getattr(self, k)[idx] for k in self._ctf_param_names}
        if getattr(self, "bfactor", None) is not None:
            ctf_batch["bfactor"] = self.bfactor[idx]
        ctf_batch["dose"] = self.dose_per_angstrom[idx]
        if getattr(self, "pre_exposure", None) is not None:
            ctf_batch["pre_exposure"] = self.pre_exposure[idx]
        return ctf_batch

    def ctf_params_dict(self) -> dict[str, torch.Tensor]:
        """
        Full (unindexed) CTF parameter buffers, keyed by name.

        Useful for re-exporting exactly the CTF parameters a model was
        constructed with, e.g. when writing a particle stack to a STAR file.

        ``dfu``/``dfv`` are un-shifted back to the convention they were
        constructed with (e.g. CryoSPARC/RELION's) if :meth:`_apply_defocus_shift`
        moved them to the volume's midplane for internal multislice use — this
        method always returns the original, externally-meaningful values.
        """
        params = {k: getattr(self, k) for k in self._ctf_param_names}
        shift = getattr(self, "_defocus_shift_angstrom", 0.0)
        if shift:
            if "dfu" in params:
                params["dfu"] = params["dfu"] + shift
            if "dfv" in params:
                params["dfv"] = params["dfv"] + shift
        return params

    def _init_detector_mtf(self) -> None:
        """Register the detector MTF buffer based on the model name."""
        # return1d defaults to False, so these always return a single Tensor here.
        if self.detector_model == "k3_300kv":
            mtf = cast(torch.Tensor, k3_300kv(self.nxy, self.pixel_size))
            self.register_buffer("detector_mtf", mtf)
        elif self.detector_model == "k3_200kv":
            mtf = cast(torch.Tensor, k3_200kv(self.nxy, self.pixel_size))
            self.register_buffer("detector_mtf", mtf)
        elif self.detector_model == "perfect":
            mtf = cast(torch.Tensor, perfect_detector(self.nxy, self.pixel_size))
            self.register_buffer("detector_mtf", mtf)
        elif self.detector_model == "falcon4i_300kv":
            mtf = cast(torch.Tensor, falcon4i_300kv(self.nxy, self.pixel_size))
            self.register_buffer("detector_mtf", mtf)
        elif self.detector_model == "falcon4i_200kv":
            mtf = cast(torch.Tensor, falcon4i_200kv(self.nxy, self.pixel_size))
            self.register_buffer("detector_mtf", mtf)
        elif self.detector_model == "k2_300kv":
            mtf = cast(torch.Tensor, k2_300kv(self.nxy, self.pixel_size))
            self.register_buffer("detector_mtf", mtf)
        else:
            self.detector_mtf = None

    def _apply_defocus_shift(
        self, shift_required: bool = True, shift: float | None = None
    ) -> None:
        """
        Shift ``dfu`` and ``dfv`` to account for the volume's Z extent.

        The CTF is evaluated at the centre of the volume; without this shift the
        defocus would be measured from the top face rather than the midplane.
        Only applied when ``shift_required`` is True (i.e. for multislice — not
        needed for projection or CTF-only models). See
        :func:`~specter.aberrations.defocus_midplane_shift` for the public,
        standalone version of this correction.

        Parameters
        ----------
        shift_required : bool, optional
            Whether the scattering model has a Z extent to offset from.
            Default True.
        shift : float, optional
            Distance in Å from the volume's midplane to the *specimen's* entry
            face. Defaults to :func:`~specter.aberrations.defocus_midplane_shift`,
            i.e. the *box's* entry face -- correct whenever the specimen fills
            the box, which is every case except an
            :class:`~specter.ice.IceProfile`. A caller that pads the box beyond
            the specimen must pass
            :meth:`~specter.ice.IceProfile.entry_face_shift` instead, or the
            defocus silently picks up the padding (see that method).
        """
        if not shift_required:
            shift = 0.0
        elif shift is None:
            shift = defocus_midplane_shift(self.nz, self.pixel_size)
        self._defocus_shift_angstrom = shift
        if shift:
            if hasattr(self, "dfu"):
                setattr(self, "dfu", getattr(self, "dfu") - shift)
            if hasattr(self, "dfv"):
                setattr(self, "dfv", getattr(self, "dfv") - shift)

    @property
    def _uniform_absorption(self) -> float:
        """
        The constant part of the absorption potential, in volts.

        Nonzero only when the imaginary potential is spatially uniform: the
        mean-free-path model with no separate specimen value, where every
        material absorbs at the medium's rate. A constant factorises out of
        the transmission exponential, so `Scattering` applies it as a scalar
        and nothing is allocated -- which is what makes this model usable on
        the volumes a 512-pixel box in thick ice produces, where the field
        alone would be 25.6 GiB.

        Returns
        -------
        float
            Absorption potential in volts, or 0.0.
        """
        if self.absorption_model != "inelastic_mfp":
            return 0.0
        if self.propagation.inelastic_mfp_specimen is not None:
            return 0.0  # not uniform; built as a field instead
        if getattr(self, "icemaker", None) is None:
            return 0.0  # vacuum around the specimen, nothing to absorb in
        return absorption_potential(self._removal_mfp("solvent"), self.voltage)

    def _removal_mfp(self, material: str) -> float:
        """
        Mean free path for leaving the image, inelastic plus aperture, in A.

        Two independent loss channels add as rates. Without an objective
        aperture this is the configured inelastic mean free path unchanged.

        Parameters
        ----------
        material : {"solvent", "specimen"}
            Which material. ``"specimen"`` requires
            ``inelastic_mfp_specimen`` to be set.

        Returns
        -------
        float
            Mean free path in Angstrom.
        """
        if material == "solvent":
            inelastic = cast(float, self._inelastic_mfp_solvent)
        else:
            inelastic = cast(float, self.propagation.inelastic_mfp_specimen)
        if self.objective_aperture is None:
            return inelastic
        aperture = (
            aperture_mfp_ice if material == "solvent" else aperture_mfp_protein
        )(self.objective_aperture, self.voltage)
        return 1.0 / (1.0 / inelastic + 1.0 / aperture)

    def _absorption_field(
        self,
        specimen: torch.Tensor,
        full_potential: float | torch.Tensor = FULL_OCCUPANCY_POTENTIAL_V,
    ) -> torch.Tensor | None:
        """
        The imaginary potential from material mean free paths, or None.

        None under ``absorption_model="alpha"``, where the imaginary part is
        applied downstream by ``Scattering`` a slice chunk at a time from a
        real volume. Otherwise a real-valued field to be combined with the
        propagating potential as ``torch.complex(V, v_ab)``, after which
        ``Scattering`` uses the potential as given and ignores its ``alpha``.

        Parameters
        ----------
        specimen : torch.Tensor
            The potential of the specimen **alone**, before any solvent was
            blended in, shape ``(B, Z, Y, X)``. This must not be a solvated
            volume: occupancy read off one is full everywhere, which would
            hand the whole box the specimen's mean free path. It is also why
            this is computed before ``solvate``, which writes into its input.
        full_potential : float or torch.Tensor, optional
            Occupancy reference, with any per-image potential scale folded
            in; the particle generators pass their template's own. Default
            :data:`~specter.potential.FULL_OCCUPANCY_POTENTIAL_V`.

        Returns
        -------
        torch.Tensor or None
            The absorption potential in volts, or None when absorption is
            left to ``alpha``.

        Notes
        -----
        The solvent term is dropped when there is no icemaker: what surrounds
        the specimen is then vacuum, not ice, and absorbing in it would remove
        electrons nothing scattered.

        The complex volume this feeds costs ~3.5x the resident memory of the
        real one at a 512-pixel box, because it defeats
        ``Scattering.multislice``'s per-chunk complexification. Building it
        inside that loop instead keeps it near 2x, and is worth doing when
        this moves past single-particle boxes.
        """
        if self.absorption_model != "inelastic_mfp":
            return None
        if self.propagation.inelastic_mfp_specimen is None:
            # Uniform, so it goes to `Scattering` as a scalar instead.
            return None
        has_solvent = getattr(self, "icemaker", None) is not None
        return inelastic_absorption_potential(
            specimen,
            self.pixel_size,
            self.voltage,
            mfp_solvent_A=(
                self._removal_mfp("solvent") if has_solvent else float("inf")
            ),
            mfp_specimen_A=self._removal_mfp("specimen"),
            full_potential=full_potential,
        )

    def _init_optics(self) -> None:
        """Instantiate the aberration engine and ``Detector`` from the
        stored settings. Which class ``self.aberration`` is depends on
        ``optics.aberration_backend`` -- both have the exact same call
        signature (``self.aberration(exitwave, ctf_batch_dict)``), so no
        other code needs to know or care which one is in use. With
        ``optics=None`` there is no aberration stage and
        ``self.aberration`` is ``None``.

        ``self.aberration_model`` is derived from ``self.scattering_model``
        rather than user-configurable: ``"linear"`` for
        ``scattering_model="ctf"`` (whose exit wave is a real-valued
        projected potential), ``"nonlinear"`` for every other
        ``scattering_model`` (a complex exit wave from full wave-optics
        propagation). The two must agree or the aberration/detector stage
        misinterprets the exit wave it's given -- see
        :func:`~specter.aberrations.aberration_model_for_scattering`.
        """
        self.aberration_model = aberration_model_for_scattering(self.scattering_model)
        env = self.envelopes
        self.aberration: Aberration | LegacyAberrationAdapter | None
        if self.optics is None:
            self.aberration = None
        elif self.optics.aberration_backend == "torch_ctf":
            self.aberration = LegacyAberrationAdapter(
                self.pad_nxy,
                self.pixel_size,
                self.voltage,
                aberration_model=self.aberration_model,
                # See the "legacy" branch below for why this depends only
                # on scattering_model, not a fixed value.
                specimen_absorption=self.scattering_model != "ctf",
                bfactor=getattr(self, "bfactor", None),
                convergence_angle=env.convergence_angle,
                cc=env.cc,
                energy_spread=env.energy_spread,
                deltaV_V=env.deltaV_V,
                deltaI_I=env.deltaI_I,
                # Once, in one place: on the specimen's potential when the
                # generator damages it there, otherwise here.
                dose_envelope=env.dose_envelope and not self._damages_potential,
                dose_weighted=self._dose_weighted,
                lpp_params=self.optics.lpp_params,
            )  # _dose_weighted: class attribute, False on TiltSeriesGenerator
        else:
            self.aberration = Aberration(
                self.pad_nxy,
                self.pixel_size,
                self.voltage,
                aberration_model=self.aberration_model,
                alpha=self.alpha,
                # scattering_model="ctf" is the only mode whose exit wave has
                # no complex/absorptive component of its own (Scattering.ctf()
                # returns a real projection; every other scattering_model
                # applies alpha upstream via potential.apply_amplitude_contrast) --
                # so it's the only case where amplitude contrast needs to be
                # folded into the transfer function itself.
                specimen_absorption=self.scattering_model != "ctf",
                convergence_angle=env.convergence_angle,
                cc=env.cc,
                energy_spread=env.energy_spread,
                deltaV_V=env.deltaV_V,
                deltaI_I=env.deltaI_I,
                dose_envelope=env.dose_envelope and not self._damages_potential,
                dose_weighted=self._dose_weighted,
                progressbars=self.progressbars,
            )
        dose_weights = self._load_dose_weights()
        self.detector = Detector(
            self.pixel_size,
            aberration_model=self.aberration_model,
            noise_model=self.noise_model,
            mtf=self.detector_mtf,
            dqe0=dqe0_for_detector(self.detector_model),
            n_frames=self.n_frames,
            dose_weights=dose_weights,
            dose_weights_max_frequency=self._dose_weights_max_frequency,
            progressbars=self.progressbars,
        )

    def _load_dose_weights(self) -> torch.Tensor | None:
        """
        The exposure filter's per-frame weights, with their frequency axis.

        Both come from :func:`~specter.io.load_dose_weights`, which derives
        the axis from the motion-correction job's own files rather than
        letting a caller assume one. The frequency is stashed for the
        detector, since a wrong axis fails silently.

        Returns
        -------
        torch.Tensor or None
            Shape ``(n_frames, n_bins)``.

        Raises
        ------
        ValueError
            If weights are given without ``n_frames``, or the frame count
            disagrees with the file's.
        """
        self._dose_weights_max_frequency: float | None = None
        path = self.camera.dose_weights_path
        if path is None:
            return None
        if self.n_frames is None:
            raise ValueError("dose_weights_path requires n_frames to be set.")
        from ..io import load_dose_weights

        w, max_freq = load_dose_weights(
            path, max_frequency=self.camera.dose_weights_max_frequency
        )
        if w.shape[0] != self.n_frames:
            raise ValueError(
                f"{path} has {w.shape[0]} frames but n_frames={self.n_frames}"
            )
        self._dose_weights_max_frequency = max_freq
        return w

    def _aberrate(
        self, exitwave: torch.Tensor, ctf_batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """The aberration stage, or the exit wave itself when there is none."""
        if self.aberration is None:
            return exitwave
        return self.aberration(exitwave, ctf_batch)

    def simulate_frozen(
        self,
        specimen: Callable[[ExposureStep], PotentialState],
        plasmon_filter: PlasmonFilter | None,
        frame_doses: Sequence[float],
        *,
        idx: int = 0,
        pose: float | torch.Tensor = 0.0,
        poses: Sequence[float | torch.Tensor] | None = None,
        substeps: int = 1,
        pre_exposure: float = 0.0,
        detector_seed: int | None = None,
        checkpoint_chunks: int | None = None,
    ) -> FrozenPlasmonResult:
        """Render explicit frozen states with this imager's optics and camera.

        ``specimen`` returns PotentialState fields in specimen coordinates;
        it owns all particle/solvent assembly. Supply poses explicitly (one
        per raw frame for a tilt sequence). Existing cached/solvated volumes
        are not used. Frame doses are physical readouts, not substep doses.
        Use FrozenPlasmonForward directly for per-frame CTF/tilt-defocus
        control. This convenience method uses CTF entry ``idx`` for all frames.
        """
        from ..inelastic import FrozenPlasmonForward
        from ..scattering import IterativeScattering

        if self.alpha != 0 or self.scattering_model != "multislice":
            raise ValueError("simulate_frozen requires multislice with alpha=0")
        if self.camera.dose_weights_path is not None:
            raise ValueError(
                "apply external dose weights after simulating physical frames"
            )
        if self.envelopes.dose_envelope:
            raise ValueError(
                "simulate_frozen requires dose_envelope=False; use a specimen "
                "damage model such as PotentialDoseDamage for exposure evolution"
            )
        scattering = IterativeScattering(
            self.pad_nxy,
            self.pixel_size,
            self.voltage,
            alpha=0,
            klim=self.klim,
            ews_curvature_sign=self.ews_curvature_sign,
            progressbars=self.progressbars,
        ).to(self.device)
        detector = Detector(
            self.pixel_size,
            aberration_model="nonlinear",
            noise_model=self.noise_model,
            mtf=self.detector_mtf,
            dqe0=self.detector.dqe0,
            n_frames=1,
            progressbars=self.progressbars,
        ).to(self.device)

        def optics(wave: torch.Tensor, step: ExposureStep) -> torch.Tensor:
            params = self._ctf_batch(torch.tensor([idx], device=self.device))
            params["dose"] = wave.real.new_full((len(wave),), step.dose)
            params["pre_exposure"] = wave.real.new_full((len(wave),), step.start)
            return self._aberrate(wave, params)

        model = FrozenPlasmonForward(
            scattering, detector, plasmon_filter, optics=optics
        )
        return model(
            specimen,
            frame_doses,
            pose=pose,
            poses=poses,
            substeps=substeps,
            pre_exposure=pre_exposure,
            detector_seed=detector_seed,
            checkpoint_chunks=checkpoint_chunks,
            coincidence_radius=float(self.coincidence_radius[idx]),
            anisomag=None if self.anisomag is None else self.anisomag[idx : idx + 1],
            nxy=self.nxy,
        )

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Standard Lightning predict step."""
        return self(batch)
