from __future__ import annotations

from typing import Any, cast

import lightning as L
import torch

from specter.detectors import (
    falcon4i_200kv,
    falcon4i_300kv,
    k3_200kv,
    k3_300kv,
    perfect_detector,
)

from ..aberrations import Aberration, defocus_midplane_shift
from ..arrays import compute_nz, pad_volume
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
    energy : float
        Electron beam energy in kV.
    dose_per_angstrom : float or torch.Tensor
        Electron dose per Å². Scalar or 1-D tensor of length n.
    nxy : int
        Unpadded image size in pixels.
    nz : int
        Number of Z slices in the simulation volume.
    pad_nxy : int, optional
        XY size after FFT padding. Defaults to ``nxy``.
    aberration_model : str, optional
        Aberration model ('holography', 'phase_plate'). Default 'holography'.
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
    num_frames : int, optional
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
        Chromatic aberration coefficient in Angstrom, used for the Cc
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
    """

    def __init__(
        self,
        pixel_size: float,
        energy: float,
        dose_per_angstrom: float | torch.Tensor,
        nxy: int,
        nz: int,
        pad_nxy: int | None = None,
        aberration_model: str = "holography",
        noise_model: str | None = "poisson",
        alpha: float = 0.0,
        detector_model: str | None = None,
        anisomag: torch.Tensor | None = None,
        ctf_params: dict[str, torch.Tensor] | None = None,
        progressbars: bool = True,
        verbose: bool = True,
        coincidence_radius: float | torch.Tensor = 0.0,
        num_frames: int | None = None,
        potential_scale: float | torch.Tensor = 1.0,
        bfactor: float | torch.Tensor | None = None,
        convergence_angle: float | None = None,
        cc: float | None = None,
        energy_spread: float = 0.7,
        deltaV_V: float = 0.06e-6,
        deltaI_I: float = 0.01e-6,
        dose_envelope: bool = False,
    ):
        super().__init__()
        self.pixel_size = pixel_size
        self.energy = energy
        self.aberration_model = aberration_model
        self.noise_model = noise_model
        self.alpha = alpha
        self.progressbars = progressbars
        self.verbose = verbose
        self.nxy = nxy
        self.nz = nz
        self.pad_nxy = pad_nxy if pad_nxy is not None else nxy
        self.detector_model = detector_model
        self.num_frames = num_frames
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

    def _apply_defocus_shift(self, shift_required: bool = True) -> None:
        """
        Shift ``dfu`` and ``dfv`` to account for the volume's Z extent.

        The CTF is evaluated at the centre of the volume; without this shift the
        defocus would be measured from the top face rather than the midplane.
        Only applied when ``shift_required`` is True (i.e. for multislice — not
        needed for projection or CTF-only models). See
        :func:`~specter.aberrations.defocus_midplane_shift` for the public,
        standalone version of this correction.
        """
        shift = (
            defocus_midplane_shift(self.nz, self.pixel_size) if shift_required else 0.0
        )
        self._defocus_shift_A = shift
        if shift:
            if hasattr(self, "dfu"):
                setattr(self, "dfu", getattr(self, "dfu") - shift)
            if hasattr(self, "dfv"):
                setattr(self, "dfv", getattr(self, "dfv") - shift)

    def _init_optics(self) -> None:
        """Instantiate ``Aberration`` and ``Detector`` from the stored parameters."""
        self.aberration = Aberration(
            self.pad_nxy,
            self.pixel_size,
            self.energy,
            aberration_model=self.aberration_model,
            alpha=self.alpha,
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
            num_frames=self.num_frames,
            progressbars=self.progressbars,
        )

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Standard Lightning predict step."""
        return self(batch)

    def predict_epoch_end(self, outputs: list[torch.Tensor]) -> torch.Tensor | None:
        """
        Gather predictions from all GPUs at epoch end.

        Parameters
        ----------
        outputs : list[torch.Tensor]
            Per-batch predictions from this GPU.

        Returns
        -------
        preds : torch.Tensor or None
            Concatenated predictions from all ranks; None on non-zero ranks.
        """
        preds = torch.cat(outputs, dim=0)
        preds_all = self.trainer.strategy.all_gather(preds)
        if self.trainer.is_global_zero:
            return preds_all.cpu()
        return None
