"""
Every generator pads its potential with "reflect", not `pad_volume`'s default.

`pad_volume`'s ``xy_pad_mode`` defaults to "constant" because it is a generic
array helper, but no imaging path wants vacuum in the margin. Padding does not
move the specimen's edge, it decides what lies beyond it, and the crop at the
end returns the original field, so whatever fills the margin leaks back into
roughly the outer ``wavelength * defocus * k_max`` pixels. "constant" puts an
ice/vacuum cliff there; "reflect" continues the specimen with matching
statistics.

`ImageGenerator` was the one call site that omitted the argument while the
other three passed it, which is exactly the kind of drift a per-call-site
convention invites -- hence the AST check rather than a comment.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import specter.imagegenerator._generator as generator_module
from specter.imagegenerator import ImageGenerator

PACKAGE = Path(generator_module.__file__).parent
MODULES = sorted(PACKAGE.glob("*.py"))


def _pad_volume_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "pad_volume"
    ]


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_every_pad_volume_call_asks_for_reflect(path: Path) -> None:
    calls = _pad_volume_calls(path)
    if not calls:
        pytest.skip(f"{path.name} does not call pad_volume")
    for call in calls:
        modes = [k.value for k in call.keywords if k.arg == "xy_pad_mode"]
        assert modes, (
            f"{path.name}:{call.lineno} calls pad_volume without xy_pad_mode, "
            "so it silently gets the generic 'constant' (vacuum) default."
        )
        assert isinstance(modes[0], ast.Constant) and modes[0].value == "reflect", (
            f"{path.name}:{call.lineno} pads with {modes[0]!r}, expected 'reflect'."
        )


def test_image_generator_actually_pads_with_reflect() -> None:
    """The AST check alone would pass if the call became unreachable."""
    seen: list[str] = []
    original = generator_module.pad_volume

    def recording(V, nxy, nz, ice_thickness, pad_fft, xy_pad_mode="constant"):
        seen.append(xy_pad_mode)
        return original(V, nxy, nz, ice_thickness, pad_fft, xy_pad_mode=xy_pad_mode)

    generator_module.pad_volume = recording
    try:
        volume = torch.zeros(16, 16, 16)
        volume[6:10, 6:10, 6:10] = 50.0
        gen = ImageGenerator(
            scattering_potential=volume,
            pixel_size=2.0,
            quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            translations=torch.zeros(1, 2),
            ctf_params={
                "cs": torch.full((1,), 2.0e7),
                "dfu": torch.full((1,), 8000.0),
            },
            voltage=300.0,
            dose_per_angstrom=20.0,
            ice_model=None,
            noise_model=None,
            scattering_model="multislice",
            pad_fft=True,
            verbose=False,
            progressbars=False,
        )
        with torch.no_grad():
            gen(torch.tensor([0]))
    finally:
        generator_module.pad_volume = original

    assert seen == ["reflect"]
