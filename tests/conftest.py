import pytest
import torch


@pytest.fixture
def dummy_volume():
    """Returns a simple 4D dummy volume (B=1, Z=32, Y=64, X=64)."""
    vol = torch.zeros(1, 32, 64, 64)
    # Add a central 'particle'
    vol[0, 12:20, 24:40, 24:40] = 1.0
    return vol


@pytest.fixture
def ctf_params_standard():
    """Returns standard CTF parameters."""
    return {
        "dfu": torch.tensor([10000.0]),
        "dfv": torch.tensor([10000.0]),
        "dfang": torch.tensor([0.0]),
        "cs": torch.tensor([2.7]),
        "phaseshift": torch.tensor([0.0]),
        "tiltx": torch.tensor([0.0]),
        "tilty": torch.tensor([0.0]),
        "trefoil1": torch.tensor([0.0]),
        "trefoil2": torch.tensor([0.0]),
    }


@pytest.fixture
def basic_setup_params():
    """Returns basic microscope parameters shared across generators."""
    return {
        "pixel_size": 1.0,
        "energy": 300.0,
        "dose_per_angstrom": 1.0,
        "micrograph_size": 64,
    }
