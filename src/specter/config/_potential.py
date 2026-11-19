"""PotentialConfig: parameters for rendering one structure's potential map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from specter.options import ScatteringFactors

from ._common_fields import (
    assembly_setting,
    job_id_setting,
    monomer_library_path_setting,
    output_dir_setting,
    pdb_cache_dir_setting,
    pdb_source_setting,
    project_setting,
    scattering_factors_setting,
)
from ._field import help_of, setting


@dataclass
class PotentialConfig:
    """Parameters for rendering an atomic model's electrostatic potential,
    loaded from a TOML config file.

    Drives `specter.pipelines.run_build_potential` (the `specter build
    potential` command), which parses one structure, renders it with
    `specter.imaging.Structure.render` and saves the volume as .mrc, in
    volts, with the voxel size in the header. The map is the model alone, in
    vacuum and centred in the box: no solvent, no envelope and no imaging.
    """

    # --- Structure ---
    pdb_source: str = pdb_source_setting()
    assembly: bool = assembly_setting()

    # --- Grid ---
    voxel_size: float = setting(1.0, help="Voxel size in Angstrom.", check="positive")
    # Unset sizes the box from the structure rather than defaulting to a
    # number: a box smaller than the model clips it without complaint, and
    # the clipped map is still a plausible-looking volume.
    box: int | None = setting(
        None,
        help=(
            "Voxels along each side of the cubic box. Unset takes the "
            "smallest even box that holds the structure in any orientation "
            "about its centre, atomic tails included. A box smaller than "
            "the structure clips it, with a warning."
        ),
        check="positive",
    )

    # --- Potential ---
    scattering_factors: ScatteringFactors = scattering_factors_setting()
    # Off by default, as on every other config: a deposited B-factor is
    # refinement output, not a measured displacement.
    use_deposited_bfactors: bool = setting(
        False,
        help=(
            "Damp each atom by the B-factor its structure deposits, instead "
            "of rendering the model statically. A deposited column is "
            "refinement output rather than a measured displacement, and "
            "cryo-EM entries often carry a constant or zero one. Requires "
            "scattering_factors='shtyrov'."
        ),
    )

    # --- Compute ---
    # One structure is one render, so there is nothing to shard: of a
    # comma-separated list only the first device is used.
    device: str = setting(
        "cuda",
        help=(
            "cpu | cuda | cuda:0 | a bare GPU index. The render is a loop of "
            "small operations per atom, about an order of magnitude faster "
            "on a GPU. Falls back to cpu when no GPU is available."
        ),
    )

    # --- Output ---
    output_dir: str | None = output_dir_setting("save the .mrc file", "potentials")
    filename: str | None = setting(
        None,
        help=(
            "Output filename, without the .mrc extension. Unset names the "
            "map after the structure: 6bdf.mrc for the accession code 6bdf "
            "or for a local 6bdf.cif."
        ),
    )
    project: str | None = project_setting("potentials")
    job_id: str | None = job_id_setting()

    # --- Advanced ---
    pdb_cache_dir: str = pdb_cache_dir_setting()
    readd_hydrogens: bool | Literal["auto"] = setting(
        "auto",
        help=(
            "Whether to replace a structure's own hydrogens with "
            "the monomer library's ideal geometry: 'auto' (default) keeps hydrogens "
            "the file already carries and adds them only when it has none, true "
            "always re-adds, false never adds hydrogen density (they still inform "
            "atom typing). Only meaningful when a monomer library is available."
        ),
    )
    monomer_library_path: str | None = monomer_library_path_setting()


POTENTIAL_HELP: dict[str, str] = help_of(PotentialConfig)
