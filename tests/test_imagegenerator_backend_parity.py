"""End-to-end parity between aberration_backend="legacy" (aberrations.
Aberration) and aberration_backend="torch_ctf" (ctf.LegacyAberrationAdapter),
through the actual ImageGenerator forward pass -- not just the isolated
transfer-function classes tested elsewhere.
"""

from __future__ import annotations

import pytest
import torch

from specter.imagegenerator import ImageGenerator


@pytest.fixture
def small_volume():
    vol = torch.zeros(32, 32, 32)
    vol[12:20, 12:20, 12:20] = 50.0
    return vol


def _build(small_volume, ctf_params, aberration_backend, seed=0):
    torch.manual_seed(seed)
    gen = ImageGenerator(
        scattering_potential=small_volume,
        pixel_size=2.0,
        quaternions=torch.tensor([[0.7071, 0.0, 0.7071, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[1.0, -1.0], [0.0, 0.5]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model=None,
        scattering_model="multislice",
        ice_model=None,
        alpha=0.1,
        verbose=False,
        progressbars=False,
        aberration_backend=aberration_backend,
    )
    torch.manual_seed(seed)
    return gen(torch.tensor([0, 1]))


def test_realistic_nonzero_ctf_params_match_across_backends(small_volume):
    """Every CTF term nonzero at once (defocus, astigmatism, Cs, trefoil,
    beam tilt, phase shift), two particles -- exercises the full
    conversion path, not just defocus/Cs like the plain regression
    fixture's all-zero ctf_params does."""
    ctf_params = {
        "dfu": torch.tensor([5200.0, 4800.0]),
        "dfv": torch.tensor([4900.0, 4600.0]),
        "dfang": torch.tensor([12.0, -30.0]),
        "cs": torch.tensor([2.7e7, 2.7e7]),
        "phaseshift": torch.tensor([0.1, 0.0]),
        "tiltx": torch.tensor([5e-4, -3e-4]),
        "tilty": torch.tensor([-2e-4, 4e-4]),
        "trefoil1": torch.tensor([0.3, -0.2]),
        "trefoil2": torch.tensor([-0.15, 0.25]),
    }

    legacy_images = _build(small_volume, ctf_params, "legacy")
    torch_ctf_images = _build(small_volume, ctf_params, "torch_ctf")

    assert legacy_images.shape == torch_ctf_images.shape
    assert torch.allclose(legacy_images, torch_ctf_images, atol=1e-3)


def test_ctf_aberration_model_matches_across_backends(small_volume):
    """aberration_model="ctf" (real-valued path), not just the default
    "holography" complex path."""
    torch.manual_seed(0)
    ctf_params = {
        "dfu": torch.tensor([5000.0, 5000.0]),
        "cs": torch.tensor([2.7e7, 2.7e7]),
    }

    def build(backend):
        torch.manual_seed(0)
        gen = ImageGenerator(
            scattering_potential=small_volume,
            pixel_size=2.0,
            quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            translations=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            ctf_params=ctf_params,
            voltage=300.0,
            dose_per_angstrom=2.0,
            noise_model=None,
            scattering_model="multislice",
            ice_model=None,
            aberration_model="ctf",
            alpha=0.0,
            verbose=False,
            progressbars=False,
            aberration_backend=backend,
        )
        torch.manual_seed(0)
        return gen(torch.tensor([0, 1]))

    legacy_images = build("legacy")
    torch_ctf_images = build("torch_ctf")
    assert torch.allclose(legacy_images, torch_ctf_images, atol=1e-3)


@pytest.mark.skipif(
    not __import__("os").path.exists(
        "/scratch/loh/joel/empiar-10202/CS-aav2/J247/J247_passthrough_particles.cs"
    ),
    reason="real .cs file not available",
)
def test_real_csfile_particles_match_across_backends_end_to_end(small_volume):
    """First 5 real particles from the same .cs file used throughout this
    migration, through the actual ImageGenerator forward pass end to end
    -- the strongest available validation that aberration_backend
    switches nothing but which engine computes the transfer function."""
    from specter.io import extract_parameters_from_csfile

    CS_PATH = (
        "/scratch/loh/joel/empiar-10202/CS-aav2/J247/J247_passthrough_particles.cs"
    )
    (
        voltage_kv,
        pixel_size,
        alpha,
        rotations,
        translations_A,
        ctf_params,
        scale,
        anisomag,
        indices,
        split,
    ) = extract_parameters_from_csfile(CS_PATH, return_class="all", n_particles=5)

    def build(backend):
        torch.manual_seed(0)
        gen = ImageGenerator(
            scattering_potential=small_volume,
            pixel_size=2.0,
            quaternions=rotations,
            translations=torch.zeros(5, 2),
            ctf_params=ctf_params,
            voltage=float(voltage_kv),
            dose_per_angstrom=2.0,
            noise_model=None,
            scattering_model="multislice",
            ice_model=None,
            alpha=float(alpha),
            verbose=False,
            progressbars=False,
            aberration_backend=backend,
        )
        torch.manual_seed(0)
        return gen(torch.arange(5))

    legacy_images = build("legacy")
    torch_ctf_images = build("torch_ctf")

    assert legacy_images.shape == torch_ctf_images.shape == (5, 32, 32)
    assert torch.allclose(legacy_images, torch_ctf_images, atol=1e-3)
