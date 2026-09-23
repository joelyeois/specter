"""Load an independently calibrated effective counting/detector response."""

import numpy as np
import torch


def load_detector_calibration(path: str, n: int, pixel_size: float):
    """Return effective pre-counting MTF, post-counting transfer and DQE(0).

    The total signal MTF is effective_mtf * noise_transfer. Thus
    DQE(k) = dqe0 * effective_mtf(k)^2 for this equivalent counting model.
    This representation specifies second moments, not microscopic event shapes.
    """
    with np.load(path, allow_pickle=False) as d:
        k = d["k"].astype(float)
        mtf = d["effective_mtf"].astype(float)
        noise = d["noise_transfer"].astype(float)
        dqe0 = float(d["dqe0"])
    if (
        k.ndim != 1
        or len(k) < 2
        or mtf.shape != k.shape
        or noise.shape != k.shape
        or not np.all(np.diff(k) > 0)
        or k[0] != 0
        or not np.isfinite(np.r_[k, mtf, noise, dqe0]).all()
        or np.any(mtf < 0)
        or np.any(noise < 0)
        or not 0 < dqe0 <= 1
        or not np.isclose(mtf[0], 1)
        or not np.isclose(noise[0], 1)
        or np.any(dqe0 * mtf**2 > 1.000001)
    ):
        raise ValueError("invalid physical detector calibration")
    axis = np.fft.fftfreq(n, pixel_size)
    radius = np.hypot(axis[:, None], axis[None, :])
    if radius.max() > k[-1] + 1e-6:
        raise ValueError(
            "detector calibration does not cover the simulation frequencies"
        )
    return (
        torch.tensor(np.interp(radius, k, mtf), dtype=torch.float32),
        torch.tensor(np.interp(radius, k, noise), dtype=torch.float32),
        dqe0,
    )
