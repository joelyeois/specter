"""
Every generator states its XY pad mode explicitly, and states the right one.

`pad_volume`'s ``xy_pad_mode`` defaults to "constant" because it is a generic
array helper. Relying on that default is what let `ImageGenerator` and
`ImageGeneratorFromCoordinates` -- the same job, one class apart -- disagree
silently, so every call site here names its mode even when it matches.

The two modes are not a style choice, and the split is not by module:

- **Single-particle boxes zero-pad the protein channel.** "reflect" would
  mirror the target particle into the margin, placing deterministic copies of
  it just outside the box, correlated with the particle being imaged. Real
  neighbours sit at random positions and orientations, which is what `crowd`
  models. Note this pads the protein only: `solvate` reflect-pads the ice
  separately and crowding is generated at the padded size, so the margin is
  never vacuum overall.
- **`MicrographGenerator` reflects.** Its volume is a whole specimen field, so
  mirroring continues a statistically similar one rather than inventing
  correlated copies of a single target.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import specter.imagegenerator._generator as generator_module
from specter.imagegenerator import ImageGenerator

PACKAGE = Path(generator_module.__file__).parent

#: module file name -> the xy_pad_mode every pad_volume call in it must name.
EXPECTED_MODE = {
    "_generator.py": "constant",  # ImageGenerator, ImageGeneratorFromCoordinates
    "_micrograph.py": "reflect",  # MicrographGenerator
}


def _pad_volume_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "pad_volume"
    ]


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_MODE.items()))
def test_pad_volume_calls_name_the_expected_mode(name: str, expected: str) -> None:
    calls = _pad_volume_calls(PACKAGE / name)
    assert calls, f"{name} no longer calls pad_volume -- update EXPECTED_MODE"
    for call in calls:
        modes = [k.value for k in call.keywords if k.arg == "xy_pad_mode"]
        assert modes, (
            f"{name}:{call.lineno} calls pad_volume without xy_pad_mode, so it "
            "inherits the generic default instead of stating a choice."
        )
        assert isinstance(modes[0], ast.Constant) and modes[0].value == expected, (
            f"{name}:{call.lineno} pads with {modes[0].value!r}, expected {expected!r}."
        )


def test_no_unlisted_module_pads() -> None:
    """A new generator must make the same decision deliberately."""
    unlisted = sorted(
        p.name
        for p in PACKAGE.glob("*.py")
        if p.name not in EXPECTED_MODE and _pad_volume_calls(p)
    )
    assert not unlisted, f"pad_volume called in unlisted module(s): {unlisted}"


def test_image_generator_actually_pads_with_constant() -> None:
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

    assert seen == ["constant"]
