"""Unrotated propagation must read views without changing order or gradients."""

import pytest
import torch

from specter.scattering import IterativeScattering


@pytest.mark.parametrize("sign", ["positive", "negative"])
@pytest.mark.parametrize("roi", [6, 8, 12])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_identity_slice_views_match_indexed_fetch_and_gradients(sign, roi, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(91)
    volume = torch.randn(2, 5, 8, 8, device=device, requires_grad=True)
    scattering = IterativeScattering(
        nxy=8,
        pixel_size=1.0,
        voltage=300,
        ews_curvature_sign=sign,
        progressbars=False,
    ).to(device)
    theta = torch.eye(3, 4, device=device).unsqueeze(0).expand(2, -1, -1)
    ys, xs = scattering._roi_start(volume, roi)
    samples = []
    references = []
    for i, nz, sample in scattering._iter_slices(volume, theta, 2, "test", roi):
        z = nz - 1 - i if sign == "negative" else i
        reference = scattering._fetch_volume_slices(
            volume,
            torch.tensor([z], device=device),
            True,
            None,
            theta,
            nz,
            ys,
            xs,
            device,
            roi,
        )[:, 0]
        assert torch.equal(sample, reference)
        if roi <= 8:
            assert (
                sample.untyped_storage().data_ptr()
                == volume.untyped_storage().data_ptr()
            )
        samples.append(sample)
        references.append(reference)
    actual_grad = torch.autograd.grad(torch.stack(samples).square().sum(), volume)[0]
    reference_grad = torch.autograd.grad(
        torch.stack(references).square().sum(), volume
    )[0]
    assert torch.equal(actual_grad, reference_grad)
