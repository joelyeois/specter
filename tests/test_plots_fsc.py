"""
Tests for the FSC resolution readout in specter.plots.
"""

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

from specter.plots import (  # noqa: E402
    MAP_TO_MODEL_FSC_THRESHOLD,
    fsc_resolution,
    plot_map_to_model_fsc,
    resolution_between,
)


def _band_limited_pair(
    n: int, voxel_size: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two noisy observations of one band-limited signal, giving an FSC that
    falls off smoothly like a real half-map pair rather than dropping to zero
    after one shell."""
    torch.manual_seed(seed)
    kf = torch.fft.fftfreq(n, d=voxel_size)
    k_mag = torch.sqrt(sum(g**2 for g in torch.meshgrid(kf, kf, kf, indexing="ij")))
    signal = torch.fft.ifftn(
        torch.fft.fftn(torch.randn(n, n, n)) * torch.exp(-((k_mag / 0.12) ** 2))
    ).real
    signal = signal / signal.std()
    return (
        signal + 0.6 * torch.randn(n, n, n),
        signal + 0.6 * torch.randn(n, n, n),
    )


def test_fsc_resolution_k_max_ignores_crossing_in_tail():
    """`fourier_shell_correlation` returns shells past Nyquist (the corners of
    the Fourier cube). A threshold crossing there is aliasing, not signal, and
    `k_max` exists to keep it out of the reported resolution."""
    # Crosses 0.5 at k = 0.18 (below Nyquist=0.25), recovers, crosses again at
    # k = 0.38 -- deep in the corner tail.
    k = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    fsc = torch.tensor([1.0, 0.9, 0.4, 0.9, 0.4, 0.0])

    # Without k_max, fsc_resolution takes the *last* crossing -- the aliased one.
    assert fsc_resolution(k, fsc, 0.5) == "2.632 Å"
    # With it, the tail is ignored.
    assert fsc_resolution(k, fsc, 0.5, k_max=0.25) == "5.556 Å"


def test_map_to_model_plot_resolution_matches_resolution_between():
    """`plot_map_to_model_fsc` labels each curve with a resolution, and
    `resolution_between` returns the same number without drawing. They must
    agree: both are the map-to-model resolution of the same pair, and only one
    of them capping the curve at Nyquist would report the aliased corner tail."""
    # This particular pair is chosen because the two paths *disagreed* before
    # the k_max fix: the plot labelled a genuinely 5.065 Å map as 1.749 Å,
    # read off a spurious crossing in the corner shells past Nyquist (3.0 Å).
    n, voxel_size = 48, 1.5
    a, b = _band_limited_pair(n, voxel_size, seed=5)

    expected = resolution_between(a, b, voxel_size, MAP_TO_MODEL_FSC_THRESHOLD)
    assert expected == "5.065 Å"

    fig = plot_map_to_model_fsc(
        [a], b, voxel_size=voxel_size, labels=["vol"], show=False
    )
    assert fig is not None
    (label,) = fig.axes[0].get_legend_handles_labels()[1]
    try:
        assert label == f"vol ({expected})"
    finally:
        matplotlib.pyplot.close(fig)


def test_map_to_model_resolution_never_finer_than_nyquist():
    """A reported resolution finer than Nyquist is unphysical: it can only come
    from a crossing in the aliased corner shells."""
    torch.manual_seed(1)
    n, voxel_size = 32, 2.0
    nyquist_res = 2.0 * voxel_size
    a = torch.randn(n, n, n)
    b = torch.randn(n, n, n)

    fig = plot_map_to_model_fsc(
        [a], b, voxel_size=voxel_size, labels=["vol"], show=False
    )
    assert fig is not None
    (label,) = fig.axes[0].get_legend_handles_labels()[1]
    matplotlib.pyplot.close(fig)

    if ">Nyquist" not in label:
        reported = float(label.split("(")[1].split()[0])
        assert reported >= nyquist_res, (
            f"reported {reported} Å, finer than Nyquist {nyquist_res} Å"
        )


@pytest.mark.parametrize("threshold", [0.143, 0.5])
def test_fsc_resolution_reports_no_crossing(threshold):
    k = torch.linspace(0.0, 0.5, 8)
    fsc = torch.ones(8)
    assert fsc_resolution(k, fsc, threshold) == ">Nyquist"
