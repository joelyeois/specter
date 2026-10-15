"""
Tiling an IceBank canvas larger than one cached config, on the bank's device.
"""

from __future__ import annotations

import pytest
import torch

from specter.ice import GradientSKIcemaker, IceBank

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _bank(tmp_path) -> IceBank:
    """A one-config library with a 32 A cell, so a 64 A request tiles 2x2x2."""
    torch.manual_seed(0)
    gd = GradientSKIcemaker(n=32, dx=1.0, progressbars=False)
    gd.init_random()
    gd.optimize(n_steps=5, record_every=5, tol=None)
    torch.save(
        {"positions": gd.positions.half(), "box_L": 32.0, "n": 32, "dx": 1.0},
        tmp_path / "config_000.pt",
    )
    return IceBank(str(tmp_path), progressbars=False)


@needs_cuda
def test_seam_relaxation_runs_on_the_bank_device(tmp_path):
    """
    A box larger than one cell relaxes its seams where the bank lives. The
    relaxed molecules used to come back on the host while the untouched bulk
    stayed on the device, and joining the two raised. The result is the one
    the host path gives: the same draws, the same molecules in the same
    order, relaxed on the same device. Unrelaxed they agree bit for bit;
    relaxed, to the ~5e-6 A that two identical relaxations on a GPU differ
    by.
    """
    bank = _bank(tmp_path).to("cuda")

    def positions(relax_steps: int, to_host: bool) -> torch.Tensor:
        return bank.big_ice_positions(
            n=64,
            dx=1.0,
            relax_steps=relax_steps,
            generator=torch.Generator().manual_seed(3),
            to_host=to_host,
        )

    assert torch.equal(positions(0, False).cpu(), positions(0, True))
    on_device, on_host = positions(5, False), positions(5, True)
    assert on_device.device.type == "cuda"
    assert on_host.device.type == "cpu"
    torch.testing.assert_close(on_device.cpu(), on_host, rtol=0, atol=1e-4)

    ice = bank.generate_big_ice(
        n=64, dx=1.0, relax_steps=5, generator=torch.Generator().manual_seed(3)
    )
    assert ice.device.type == "cuda"
    assert ice.shape == (1, 64, 64, 64)
    assert torch.isfinite(ice).all()


@needs_cuda
def test_a_generator_on_the_gpu_draws_reproducibly(tmp_path):
    """
    The crops' few random numbers are drawn where the generator lives, so a
    CUDA generator works and repeats itself under the same seed, and a
    different seed draws different ice.
    """
    bank = _bank(tmp_path)

    def ice(seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        return bank.generate_ice(n=24, dx=1.0, generator=generator)

    first = ice(0)
    assert torch.isfinite(first).all()
    assert torch.equal(first, ice(0))
    assert not torch.equal(first, ice(1))
