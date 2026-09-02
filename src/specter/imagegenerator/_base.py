from __future__ import annotations

from typing import Any, Literal, cast

import lightning as L
import torch

from specter.detectors import (
    dqe0_for_detector,
    falcon4i_200kv,
    falcon4i_300kv,
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
    noise_model : str, optional
        Noise model ('poisson', 'gaussian', None). Default 'poisson'.
    alpha : float, optional
        Amplitude contrast ratio. Default 0.0.
    detector_model : str, optional
        Detector MTF model ('k3_300kv', 'k3_200kv', 'perfect', None).
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices, shape (n, 2, 2).
    ctf_params : dict[str, torch.Tensor], optional
        CTF parameters; each value is a 1-D tensor of length n.
    progressbars : bool, optional
        Whether to show progress bars. Default True.
    verbose : bool, optional
        Whether to emit debug-level log messages. Default True.
    coincidence_radius : float or torch.Tensor, optional
        Coincidence radius in pixels. Scalar or 1-D tensor of length n.
        Default 0.0.
    n_frames : int, optional
        Number of detector frames to simulate. Default None (single frame).
    potential_scale : float or torch.Tensor, optional
        Multiplier applied to the scattering potential before propagation.
        Scalar or 1-D tensor of length n. Default 1.0.
    bfactor : float or torch.Tensor or None, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function. None or 0.0 means no envelope. Default None.
    convergence_angle : float, optional
        Beam convergence semi-angle in milliradians, used for the Cs
        (spatial coherence) envelope. Default None (envelope disabled).
    cc : float, optional
        Chromatic aberration coefficient in Å, used for the Cc
        (temporal coherence) envelope. Default None (envelope disabled).
    energy_spread : float, optional
        FWHM of the beam energy spread in eV, used by the Cc envelope.
        Default 0.7.
    deltaV_V : float, optional
        Relative high-voltage instability, used by the Cc envelope.
        Default 0.06e-6.
    deltaI_I : float, optional
        Relative objective-lens current instability, used by the Cc
        envelope. Default 0.01e-6.
    dose_envelope : bool, optional
        Whether to apply the Grant & Grigorieff (2015) cumulative-dose
        envelope, using ``dose_per_angstrom``. Default False.
    aberration_backend : {"legacy", "torch_ctf"}, optional
        Which engine computes the CTF/aberration transfer function.
        ``"legacy"`` (default) uses ``aberrations.Aberration`` unchanged --
        no behaviour change for any existing caller. ``"torch_ctf"`` uses
        ``ctf.LegacyAberrationAdapter`` (torch-ctf-backed, verified
        bit-identical to ``"legacy"`` term-by-term and against real
        multi-particle CryoSPARC .cs data -- see
        tests/test_ctf_legacy_adapter.py). Opt-in only; not yet the
        default.
    lpp_params : dict[str, float], optional
        Laser-phase-plate config, in ``ctf.CTFParameters``-native units
        (see ``ctf.LegacyAberrationAdapter``). Requires
        ``aberration_backend="torch_ctf"`` -- ``aberrations.Aberration``
        has no LPP model, so this raises at construction time if set with
        the default ``"legacy"`` backend rather than silently having no
        effect. A single shared laser-instrument config, never
        per-particle, so it's a constructor argument here rather than a
        ``ctf_params`` dict key.
    """

    def __init__(
        self,
        pixel_size: float,
        voltage: float,
        dose_per_angstrom: float | torch.Tensor,
        nxy: int,
        nz: int,
        pad_nxy: int | None = None,
        noise_model: str | None = "poisson",
        alpha: float = 0.0,
        detector_model: str | None = None,
        anisomag: torch.Tensor | None = None,
        ctf_params: dict[str, torch.Tensor] | None = None,
        progressbars: bool = True,
        verbose: bool = True,
        coincidence_radius: float | torch.Tensor = 0.0,
        n_frames: int | None = None,
        potential_scale: float | torch.Tensor = 1.0,
        bfactor: float | torch.Tensor | None = None,
        convergence_angle: float | None = None,
        cc: float | None = None,
        energy_spread: float = 0.7,
        deltaV_V: float = 0.06e-6,
        deltaI_I: float = 0.01e-6,
        dose_envelope: bool = False,
        aberration_backend: Literal["legacy", "torch_ctf"] = "legacy",
        lpp_params: dict[str, float] | None = None,
    ):
        super().__init__()
        if lpp_params is not None and aberration_backend != "torch_ctf":
            raise ValueError(
                "lpp_params requires aberration_backend='torch_ctf' -- "
                "aberrations.Aberration has no laser-phase-plate model."
            )
        self.pixel_size = pixel_size
        self.voltage = voltage
        self.noise_model = noise_model
        self.alpha = alpha
        self.aberration_backend = aberration_backend
        self.lpp_params = lpp_params
        self.progressbars = progressbars
        self.verbose = verbose
        self.nxy = nxy
        self.nz = nz
        self.pad_nxy = pad_nxy if pad_nxy is not None else nxy
        self.detector_model = detector_model
        self.n_frames = n_frames
        self.convergence_angle = convergence_angle
        self.cc = cc
        self.energy_spread = energy_spread
        self.deltaV_V = deltaV_V
        self.deltaI_I = deltaI_I
        self.dose_envelope = dose_envelope
        self._init_detector_mtf()

        if anisomag is None:
            self.anisomag = None
        else:
            self.register_buffer("anisomag", torch.as_tensor(anisomag))

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
            n = None

        # Scalar inputs are expanded to length-n so forward() can index with [idx].
        def _to_buffer(val: float | torch.Tensor, name: str) -> None:
            t = torch.as_tensor(val, dtype=torch.float32).flatten()
            if n is not None and len(t) == 1:
                t = t.expand(n).clone()
            self.register_buffer(name, t)

        _to_buffer(dose_per_angstrom, "dose_per_angstrom")
        _to_buffer(coincidence_radius, "coincidence_radius")
        _to_buffer(potential_scale, "potential_scale")
        # Decided once, on the host: `forward` used to test the buffer's
        # batch every step (`bool(torch.all(scale == 1))`), a device sync
        # per step that, in a training loop, exposed the CPU's kernel-launch
        # time instead of overlapping it with the GPU.
        self._potential_scale_is_unity = bool(torch.all(self.potential_scale == 1.0))
        if bfactor is not None:
            if "bfactor" in self._ctf_param_names:
                self._buffers.pop("bfactor")
                self._ctf_param_names.remove("bfactor")
            _to_buffer(bfactor, "bfactor")

    def _ctf_batch(self, idx: torch.Tensor | int) -> dict[str, torch.Tensor]:
        """Collect per-image transfer-function parameters for a batch."""
        ctf_batch = {k: getattr(self, k)[idx] for k in self._ctf_param_names}
        if getattr(self, "bfactor", None) is not None:
            ctf_batch["bfactor"] = self.bfactor[idx]
        ctf_batch["dose"] = self.dose_per_angstrom[idx]
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
        shift = getattr(self, "_defocus_shift_A", 0.0)
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
        self._defocus_shift_A = shift
        if shift:
            if hasattr(self, "dfu"):
                setattr(self, "dfu", getattr(self, "dfu") - shift)
            if hasattr(self, "dfv"):
                setattr(self, "dfv", getattr(self, "dfv") - shift)

    def _init_optics(self) -> None:
        """Instantiate the aberration engine and ``Detector`` from the
        stored parameters. Which class ``self.aberration`` is depends on
        ``self.aberration_backend`` -- both have the exact same call
        signature (``self.aberration(exitwave, ctf_batch_dict)``), so no
        other code needs to know or care which one is in use.

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
        self.aberration: Aberration | LegacyAberrationAdapter
        if self.aberration_backend == "torch_ctf":
            self.aberration = LegacyAberrationAdapter(
                self.pad_nxy,
                self.pixel_size,
                self.voltage,
                aberration_model=self.aberration_model,
                # See the "legacy" branch below for why this depends only
                # on scattering_model, not a fixed value.
                specimen_absorption=self.scattering_model != "ctf",
                bfactor=getattr(self, "bfactor", None),
                convergence_angle=self.convergence_angle,
                cc=self.cc,
                energy_spread=self.energy_spread,
                deltaV_V=self.deltaV_V,
                deltaI_I=self.deltaI_I,
                dose_envelope=self.dose_envelope,
                lpp_params=self.lpp_params,
            )
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
                convergence_angle=self.convergence_angle,
                cc=self.cc,
                energy_spread=self.energy_spread,
                deltaV_V=self.deltaV_V,
                deltaI_I=self.deltaI_I,
                dose_envelope=self.dose_envelope,
                progressbars=self.progressbars,
            )
        self.detector = Detector(
            self.pixel_size,
            aberration_model=self.aberration_model,
            noise_model=self.noise_model,
            mtf=self.detector_mtf,
            dqe0=dqe0_for_detector(self.detector_model),
            n_frames=self.n_frames,
            progressbars=self.progressbars,
        )

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Standard Lightning predict step."""
        return self(batch)
