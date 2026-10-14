"""
Image formation as data objects and the verbs between them.

A `Specimen` is scattered into an `ExitWave`, which the lens turns into an
`Image` that the camera records::

    specimen = Structure.from_pdb("6bdf").render(voxel_size=1.0, box=256)
    specimen = specimen.crowd(min_distance=360.0)
    specimen = specimen.solvate(model="gd", thickness=500.0)

    poses = Poses.random(16, shift=2.0)
    exposure = Exposure(dose=40.0)
    waves = specimen.scatter(poses, voltage=300.0, exposure=exposure, seed=0)
    waves = waves.apply_ctf(CTF.sample(16, defocus=[5000.0, 15000.0]))
    images = waves.intensity(cc=2.7).detect(exposure, seed=0)

A micrograph's specimen is a `Field` of copies through a slab of ice, drawn
once into a `Sample` and imaged as it lies::

    field = Field(template, 1.0, 4096).pack(min_distance=360.0)
    field = field.solvate(model="gd", thickness=500.0)
    sample = field.draw(voltage=300.0, seed=0, index=0)
    waves = sample.scatter(voltage=300.0, device="cuda:0")

Each verb takes the parameters of its own stage, named as in the settings
dataclasses and the configs. An `ExitWave` or `Image` holds its tensor as
``.array``, a `Specimen` its potential as ``.potential`` and a `Sample` its
volume as ``.volume``. The stages themselves are the same functions every
simulator and the reconstructor call (:mod:`specter.imaging._stages`).
"""

from ._data import CTF, Exposure, Poses
from ._field import Field, Sample
from ._frozen import simulate_frozen
from ._recipes import simulate_particles
from ._specimen import Specimen, Structure
from ._tilt import TiltSeries
from ._waves import ExitWave, Image

__all__ = [
    "CTF",
    "ExitWave",
    "Exposure",
    "Field",
    "Image",
    "Poses",
    "Sample",
    "Specimen",
    "Structure",
    "TiltSeries",
    "simulate_frozen",
    "simulate_particles",
]
