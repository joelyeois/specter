"""
Exposure-filter weights, read together with the frequency axis they belong to.

A motion-correction job writes its per-frame weights as a bare
``(n_frames, n_bins)`` array with no statement of what frequency the bins
run to. Pairing that array with a particle stack therefore needs a convention,
and getting it wrong is silent: the weights still apply, still rise toward the
edge of the array, and simply rise at the wrong rate. Measured on
EMPIAR-11377, reading the axis as the particles' own Nyquist rather than the
array's overstated the noise gain threefold.

It is also not enough to carry the pixel size, which is the obvious thing to
reach for and the reason this went wrong twice. CryoSPARC records
``refmotion_doseweights/psize_A`` in the job's ``hyperparams.cs``, and for
EMPIAR-11377 it reads 0.731 -- exactly the particles' pixel size. The factor
of two is not a sampling difference at all: the weights array simply carries
twice the radial sampling of the ``refm_fcc.npy`` written beside it, so its
axis runs to twice Nyquist while the FCC's runs to Nyquist.

So the quantity that travels with the weights here is the **frequency of the
last bin, in 1/Angstrom**, and the mapping onto an image is done in absolute
frequency. A stack that was downsampled, binned or Fourier-cropped after
motion correction then gets the weights that apply at the frequencies it
actually kept, with no convention left for a caller to get wrong.
"""

from __future__ import annotations

import os

import numpy as np
import torch

__all__ = ["load_dose_weights"]

#: Name of the Fourier-cross-correlation array a CryoSPARC reference-motion
#: job writes beside its weights. Its radial axis runs to Nyquist, so the
#: ratio of bin counts is what says how far the weights' own axis runs.
_FCC_FILENAME = "refm_fcc.npy"
#: Where the same job records the pixel size it worked at.
_HYPERPARAMS_FILENAME = "hyperparams.cs"


def _pixel_size_from_hyperparams(directory: str) -> float | None:
    """
    The pixel size a CryoSPARC reference-motion job recorded, if present.

    Parameters
    ----------
    directory : str
        Directory holding the weights, i.e. the job directory.

    Returns
    -------
    float or None
        Pixel size in Angstrom, or None when the file or field is absent.
    """
    path = os.path.join(directory, _HYPERPARAMS_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        from cryosparc.dataset import Dataset

        data = Dataset.load(path)
        field = "refmotion_doseweights/psize_A"
        if field not in data.fields():
            return None
        return float(np.asarray(data[field]).reshape(-1)[0])
    except Exception:
        # A metadata file that cannot be read is not a reason to fail: the
        # caller can still supply the frequency explicitly.
        return None


def load_dose_weights(
    path: str, max_frequency: float | None = None
) -> tuple[torch.Tensor, float]:
    """
    Read exposure-filter weights and the frequency their last bin sits at.

    Parameters
    ----------
    path : str
        ``.npy`` of shape ``(n_frames, n_bins)`` -- CryoSPARC's
        ``refm_empirical_dw.npy``, say.
    max_frequency : float or None, optional
        Frequency of the last bin, in 1/Angstrom. Default None derives it
        from the job's own files: the pixel size from ``hyperparams.cs`` gives
        Nyquist, and the bin count of ``refm_fcc.npy`` -- whose axis runs to
        Nyquist -- says how many Nyquists the weights' axis spans.

    Returns
    -------
    tuple
        The weights as a ``(n_frames, n_bins)`` float tensor, and the
        frequency of the last bin in 1/Angstrom.

    Raises
    ------
    ValueError
        If the array is not 2-D, or the frequency axis can be neither derived
        nor read. Failing here is deliberate: a wrong axis does not raise
        anywhere downstream, it just applies the weights at the wrong
        frequencies.
    """
    weights = torch.as_tensor(np.load(path)).float()
    if weights.ndim != 2:
        raise ValueError(
            f"{path}: expected (n_frames, n_bins), got {tuple(weights.shape)}"
        )
    if max_frequency is not None:
        if max_frequency <= 0.0:
            raise ValueError(f"max_frequency={max_frequency} must be positive")
        return weights, float(max_frequency)

    directory = os.path.dirname(os.path.abspath(path))
    pixel_size = _pixel_size_from_hyperparams(directory)
    fcc_path = os.path.join(directory, _FCC_FILENAME)
    if pixel_size is None or not os.path.exists(fcc_path):
        raise ValueError(
            f"{path}: cannot determine which frequencies these weights cover. "
            f"Deriving it needs both {_HYPERPARAMS_FILENAME} (for the pixel "
            f"size) and {_FCC_FILENAME} (whose {weights.shape[1]}-bin axis "
            "runs to Nyquist) beside the weights. Pass the frequency of the "
            "last bin explicitly instead, in 1/Angstrom. This is raised "
            "rather than guessed because the wrong axis fails silently -- the "
            "weights still apply, at the wrong frequencies."
        )

    fcc_bins = int(np.load(fcc_path).shape[-1])
    nyquist = 1.0 / (2.0 * pixel_size)
    max_frequency = nyquist * (weights.shape[1] / fcc_bins)
    return weights, float(max_frequency)
