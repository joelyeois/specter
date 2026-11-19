"""Potential-map pipeline: `PotentialConfig` in, one structure's potential out as .mrc."""

from __future__ import annotations

import logging
import math
import os
import warnings

import specter
from specter.config import PotentialConfig, validate_config
from specter.devices import parse_device, resolve_available_device
from specter.imaging import Structure
from specter.io import write_mrc
from specter.potential import recommended_rcut
from specter.progress import console, section

from ._common import _tracked_output_dir


def _required_extent(structure: Structure, use_deposited_bfactors: bool) -> float:
    """
    Side of the smallest cube, in Angstrom, that holds the rendered structure
    in any orientation about its centre.

    Twice the farthest atom's distance from the centre the model is rendered
    about, plus the radius each atom is rendered out to on either side. The
    bounding sphere rather than the axis-aligned extent, so that the map can
    be rotated (as a template, or by a simulation) without leaving its box.
    """
    pdb = structure.pdb
    b_max = float(pdb.b_factors.max()) if use_deposited_bfactors else 0.0
    rcut = recommended_rcut(
        pdb.atomic_numbers,
        structure.scattering_factors,
        pdb.atom_species,
        b_factor_max=b_max,
    )
    radius = float(pdb.coordinates.norm(dim=-1).max())
    return 2.0 * (radius + rcut)


def _auto_box(extent: float, voxel_size: float) -> int:
    """The smallest even box, in voxels, at least `extent` Angstrom across."""
    return 2 * math.ceil(extent / voxel_size / 2.0)


def _default_filename(pdb_source: str) -> str:
    """`pdb_source` without its directory and extension(s): ``6bdf`` for ``x/6bdf.cif.gz``."""
    return os.path.basename(pdb_source).split(".")[0]


def run_build_potential(config: PotentialConfig) -> None:
    """
    Render one atomic model's electrostatic potential and save it as ``.mrc``.

    The volume is the model alone, in vacuum, in volts and indexed
    ``(Z, Y, X)``, centred in a cubic box, with the voxel size in the header.
    It is what `specter simulate particles` builds internally before imaging,
    written out for use elsewhere: a reference for an FSC, a template, or an
    initial model.

    Parameters
    ----------
    config : PotentialConfig
        Fully-resolved run configuration, e.g. from
        :func:`specter.config.load_config`, or constructed directly in
        Python. ``config.box`` unset sizes the box from the structure.
    """
    validate_config(config)
    specter.set_verbosity(logging.INFO)
    device = parse_device(resolve_available_device(config.device)).primary

    section("Building 3D scattering potential")
    structure = Structure.from_pdb(
        config.pdb_source,
        assembly=config.assembly,
        scattering_factors=config.scattering_factors,
        pdb_cache_dir=config.pdb_cache_dir,
        readd_hydrogens=config.readd_hydrogens,
        monomer_library_path=config.monomer_library_path,
    )
    extent = _required_extent(structure, config.use_deposited_bfactors)
    box = config.box
    if box is None:
        box = _auto_box(extent, config.voxel_size)
    elif box * config.voxel_size < extent:
        warnings.warn(
            f"run_build_potential: box={box} at voxel_size={config.voxel_size} A "
            f"is {box * config.voxel_size:.0f} A across, and the structure needs "
            f"{extent:.0f} A to fit in every orientation "
            f"(box={_auto_box(extent, config.voxel_size)}). Atoms outside the "
            "box are left out of the map. Leave box unset to size it from the "
            "structure.",
            stacklevel=2,
        )
    console.print(
        f"  {len(structure.pdb.coordinates)} atoms, "
        f"{structure.molecular_mass / 1e3:.0f} kDa, "
        f"largest extent {structure.pdb.max_diameter:.0f} A"
    )
    console.print(
        f"  Box: {box} voxels at {config.voxel_size:g} A "
        f"({box * config.voxel_size:.0f} A across), "
        f"{config.scattering_factors} scattering factors, on {device}"
    )

    potential = structure.render(
        config.voxel_size,
        box,
        use_deposited_bfactors=config.use_deposited_bfactors,
        device=device,
    ).potential

    with _tracked_output_dir(config, "potentials") as output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = config.filename or _default_filename(config.pdb_source)
        path = os.path.join(output_dir, f"{filename}.mrc")
        write_mrc(path, potential, config.voxel_size)
    console.print(f"  Peak {float(potential.max()):.0f} V")
    console.print(f"  Saved {path}")
