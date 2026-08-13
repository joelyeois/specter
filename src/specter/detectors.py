from __future__ import annotations

import torch
from scipy.interpolate import interp1d as scipy_interp1d
from torchinterp1d import interp1d

# Shared K3 frequency axis (RELION inverse-pixel units, 0 to ~0.508 Nyquist).
# Identical for both 200 kV and 300 kV K3 datasheets.
_K3_FREQ = [
    0.000000000,
    0.015873016,
    0.031746032,
    0.047619047,
    0.063492064,
    0.07936508,
    0.095238095,
    0.111111111,
    0.126984127,
    0.142857143,
    0.158730159,
    0.174603175,
    0.190476191,
    0.206349207,
    0.222222222,
    0.238095238,
    0.253968254,
    0.26984127,
    0.285714286,
    0.301587302,
    0.317460318,
    0.333333334,
    0.349206349,
    0.365079365,
    0.380952381,
    0.396825397,
    0.412698413,
    0.428571429,
    0.444444445,
    0.460317461,
    0.476190476,
    0.492063492,
    0.507936508,
]


def _k3_mtf(
    n: int,
    dx: float,
    device: str | torch.device,
    return1d: bool,
    mtf_values: list[float],
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Shared implementation for K3 MTF functions."""
    k_data = torch.tensor(_K3_FREQ, dtype=torch.float32, device=device) / dx
    mtf = torch.tensor(mtf_values, dtype=torch.float32, device=device)

    if return1d:
        return k_data, mtf

    k = torch.fft.fftfreq(n, dx, device=device)
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    k_rad = torch.sqrt(kx**2 + ky**2)
    return interp1d(k_data, mtf, k_rad.ravel()).reshape(n, n)


def _falcon4i_mtf(
    n: int,
    dx: float,
    device: str | torch.device,
    return1d: bool,
    dqe_values: list[float],
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Shared implementation for Falcon 4i MTF functions.

    The published data are DQE, not MTF. Under a white-noise approximation
    ``DQE(k) = MTF(k)**2``, so the *shape* is recovered as
    ``MTF(k) = sqrt(DQE(k) / DQE(0))`` -- normalised by the zero-frequency
    value so that ``MTF(0) == 1``, as an MTF must be.

    Normalising matters: an MTF is a pure blur and conserves total counts.
    The zero-frequency efficiency ``DQE(0)`` is a *separate* physical effect
    (electrons that never register at all) and is applied as a counting
    efficiency via ``Detector(dqe0=...)``, not folded in here. Folding it in
    would both scale counts by ``sqrt(DQE(0))`` instead of ``DQE(0)`` and give
    the wrong shot-noise statistics -- see :func:`dqe0_for_detector`.
    """
    k_nyquist = 1 / 2 / dx
    k_points = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32) * k_nyquist
    dqe = torch.tensor(dqe_values, dtype=torch.float32)
    mtf_data = torch.sqrt(dqe / dqe[0])

    # Quadratic interpolation via scipy (torchinterp1d does not support quadratic)
    interp = scipy_interp1d(
        k_points.numpy(), mtf_data.numpy(), kind="quadratic", fill_value="extrapolate"
    )

    k = torch.fft.fftfreq(n, dx, device=device)
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    k_rad = torch.sqrt(kx**2 + ky**2)
    mtf = torch.from_numpy(interp(k_rad.cpu().numpy())).to(device).reshape(n, n)

    if return1d:
        return k_rad[n // 2 :, n // 2], mtf[n // 2 :, n // 2]
    return mtf


def k3_200kv(
    n: int, dx: float, device: str | torch.device = "cpu", return1d: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Return the MTF of a Gatan K3 detector at 200 kV.

    Uses measured MTF values from the datasheet.

    Parameters
    ----------
    n : int
        Number of pixels along each axis of the output MTF.
    dx : float
        Pixel size of the simulated image (same units as spatial frequency).
    device : str or torch.device, optional
        Device to create tensors on ('cpu' or 'cuda'). Default is 'cpu'.
    return1d : bool, optional
        If True, return 1D MTF sampled at radial frequencies. Default is False.

    Returns
    -------
    mtf : torch.Tensor
        - If return1d=False: 2D NxN MTF array (radially symmetric).
        - If return1d=True: Tuple (k_data, mtf_values) for 1D MTF.

    References
    ----------
    https://www.gatan.com/sites/default/files/images/mtf_k3_standard_200kV_FL2.star
    """
    return _k3_mtf(
        n,
        dx,
        device,
        return1d,
        [
            1.000000000,
            0.992006436,
            0.984482532,
            0.977220107,
            0.970036615,
            0.96277324,
            0.955293437,
            0.947481509,
            0.939241219,
            0.930494442,
            0.921179852,
            0.911251641,
            0.900678283,
            0.889441327,
            0.877534229,
            0.864961225,
            0.851736231,
            0.837881788,
            0.823428039,
            0.808411746,
            0.792875339,
            0.776866005,
            0.760434809,
            0.743635861,
            0.726525508,
            0.70916157,
            0.69160261,
            0.67390724,
            0.656133466,
            0.638338062,
            0.620575997,
            0.602899875,
            0.585359435,
        ],
    )


def k3_300kv(
    n: int, dx: float, device: str | torch.device = "cpu", return1d: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Return the MTF of a Gatan K3 detector at 300 kV.

    Uses measured MTF values from the datasheet.

    Parameters
    ----------
    n : int
        Number of pixels along each axis of the output MTF.
    dx : float
        Pixel size of the simulated image.
    device : str or torch.device, optional
        Device to create tensors on ('cpu' or 'cuda'). Default is 'cpu'.
    return1d : bool, optional
        If True, return 1D MTF sampled at radial frequencies. Default is False.

    Returns
    -------
    mtf : torch.Tensor
        - If return1d=False: 2D NxN MTF array.
        - If return1d=True: Tuple (k_data, mtf_values) for 1D MTF.

    References
    ----------
    https://www.gatan.com/sites/default/files/images/mtf_k3_standard_300kV_FL2.star
    """
    return _k3_mtf(
        n,
        dx,
        device,
        return1d,
        [
            0.999999586,
            0.994780311,
            0.990164462,
            0.985857811,
            0.981606672,
            0.977194975,
            0.972441438,
            0.96719683,
            0.961341332,
            0.954781993,
            0.947450278,
            0.939299717,
            0.930303644,
            0.920453031,
            0.909754424,
            0.898227965,
            0.885905514,
            0.872828868,
            0.85904807,
            0.844619817,
            0.829605963,
            0.814072114,
            0.798086321,
            0.78171787,
            0.765036162,
            0.748109691,
            0.731005117,
            0.713786436,
            0.69651424,
            0.67924508,
            0.662030915,
            0.644918663,
            0.627949844,
        ],
    )


def perfect_detector(
    n: int, dx: float, device: str | torch.device = "cpu", return1d: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Return the MTF of a perfect pixel detector limited only by pixel integration.

    The ideal MTF is given by sinc(pi * f / f_Nyquist), where f_Nyquist = 1/(2*dx).

    Parameters
    ----------
    n : int
        Number of pixels along each axis.
    dx : float
        Pixel size of the simulated image.
    device : str or torch.device, optional
        Device to create tensors on ('cpu' or 'cuda'). Default is 'cpu'.
    return1d : bool, optional
        If True, return 1D radial MTF instead of 2D. Default is False.

    Returns
    -------
    mtf : torch.Tensor
        - If return1d=False: 2D NxN MTF array (radially symmetric).
        - If return1d=True: Tuple (k_data, mtf_1d) for 1D MTF along a radial line.
    """
    k = torch.fft.fftfreq(n, d=dx, device=device)

    if return1d:
        omega = k / (1 / (2 * dx))
        return k[n // 2 :], torch.sinc(omega / 2)[n // 2 :]

    kx, ky = torch.meshgrid(k, k, indexing="ij")
    k_rad = torch.sqrt(kx**2 + ky**2)
    omega = k_rad / (1 / (2 * dx))
    return torch.sinc(omega / 2)


def falcon4i_300kv(
    n: int, dx: float, device: str | torch.device = "cpu", return1d: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Return the MTF of a Thermo Fisher Falcon 4i detector at 300 kV.

    Approximates the MTF using measured DQE at 0, 0.5, and 1 Nyquist and assumes
    MTF ≈ sqrt(DQE). Quadratic interpolation is used.

    Parameters
    ----------
    n : int
        Number of pixels along each axis.
    dx : float
        Pixel size of the simulated image.
    device : str or torch.device, optional
        Device to create tensors on ('cpu' or 'cuda'). Default is 'cpu'.
    return1d : bool, optional
        If True, return 1D MTF along a radial line. Default is False.

    Returns
    -------
    mtf : torch.Tensor
        - If return1d=False: 2D NxN MTF array.
        - If return1d=True: Tuple (k_data, mtf_1d) for 1D radial MTF.

    References
    ----------
    https://www.thermofisher.com/sg/en/home/electron-microscopy/products/accessories-em/falcon-detector.html
    """
    return _falcon4i_mtf(n, dx, device, return1d, [0.92, 0.72, 0.50])


def falcon4i_200kv(
    n: int, dx: float, device: str | torch.device = "cpu", return1d: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Return the MTF of a Thermo Fisher Falcon 4i detector at 200 kV.

    Approximates the MTF using measured DQE at 0, 0.5, and 1 Nyquist and assumes
    MTF ≈ sqrt(DQE). Quadratic interpolation is used.

    Parameters
    ----------
    n : int
        Number of pixels along each axis.
    dx : float
        Pixel size of the simulated image.
    device : str or torch.device, optional
        Device to create tensors on ('cpu' or 'cuda'). Default is 'cpu'.
    return1d : bool, optional
        If True, return 1D MTF along a radial line. Default is False.

    Returns
    -------
    mtf : torch.Tensor
        - If return1d=False: 2D NxN MTF array.
        - If return1d=True: Tuple (k_data, mtf_1d) for 1D radial MTF.

    References
    ----------
    https://www.thermofisher.com/sg/en/home/electron-microscopy/products/accessories-em/falcon-detector.html
    """
    return _falcon4i_mtf(n, dx, device, return1d, [0.91, 0.62, 0.33])


# Zero-frequency detective quantum efficiency, DQE(0), per detector model.
#
# This is the *counting efficiency*: the fraction of incident electrons that
# register at all. For a detector that records each electron independently with
# probability p, thinning a Poisson process leaves a Poisson process of mean
# p*N, so DQE(0) = p exactly. It is applied by scaling the expected electron
# count (see ``Detector(dqe0=...)``), which reproduces both the reduced mean and
# the correct shot noise -- unlike a multiplicative filter, which gets the
# variance wrong.
#
# Kept separate from the MTF because they are two halves of one measured curve:
# the MTF carries the frequency dependence (normalised to 1 at DC, conserving
# counts) and DQE(0) carries the overall efficiency.
#
# Only entries with a traceable published source are listed. Models absent here
# default to 1.0 (an ideal counter), which is optimistic -- real direct
# detectors sit around 0.5-0.95 at 300 kV.
#
# IMPORTANT: these must be *low-dose-rate* values. Published DQE(0) falls with
# dose rate precisely because of coincidence loss, which specter models
# separately; using a high-flux figure here would count that loss twice. The
# Falcon 4i value is consistent with the low-flux limit measured from beam-only
# micrographs (fitted zero-dose intercept 0.926 vs 0.92 published).
DQE0: dict[str, float] = {
    "falcon4i_300kv": 0.92,
    "falcon4i_200kv": 0.91,
    "perfect": 1.0,
}


def dqe0_for_detector(detector_model: str | None) -> float:
    """
    Return the zero-frequency DQE (counting efficiency) for a detector model.

    Parameters
    ----------
    detector_model : str or None
        Detector model name, e.g. ``"falcon4i_300kv"``. ``None`` or an unknown
        name returns 1.0.

    Returns
    -------
    float
        DQE(0) in (0, 1]. 1.0 means every incident electron is counted.

    Notes
    -----
    The K3 entries deliberately have no value: their bundled response is a
    measured MTF (already normalised to 1 at DC) with no accompanying DQE(0)
    figure in the same datasheet, so assuming one would be inventing physics.
    They therefore behave as ideal counters until a sourced value is added.
    """
    if detector_model is None:
        return 1.0
    return DQE0.get(detector_model, 1.0)
