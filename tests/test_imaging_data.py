"""
The per-image data objects of `specter.imaging` and the device they live on.
"""

from __future__ import annotations

import pytest
import torch

from specter.imaging import CTF


def test_a_ctf_from_numbers_is_built_on_the_cpu():
    ctf = CTF.from_values(10000.0, n=3, astigmatism=200.0)
    assert all(v.device.type == "cpu" for v in ctf.params.values())
    assert torch.equal(ctf.params["dfv"], torch.full((3,), 9800.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_ctf_follows_its_defocus_onto_the_device():
    """
    Every term lands on the defocus tensor's device. The others used to be
    built on the CPU whatever the defocus, so a CUDA defocus failed at the
    first subtraction.
    """
    defocus = torch.tensor([10000.0, 12000.0], device="cuda")
    ctf = CTF.from_values(
        defocus, astigmatism=torch.tensor([100.0, 200.0]), phase_shift=90.0
    )
    assert all(v.device.type == "cuda" for v in ctf.params.values())
    on_host = CTF.from_values(
        defocus.cpu(), astigmatism=torch.tensor([100.0, 200.0]), phase_shift=90.0
    )
    for name, value in on_host.params.items():
        assert torch.equal(ctf.params[name].cpu(), value), name
