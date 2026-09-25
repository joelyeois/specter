"""
Pick export for `TomogramSpecimenGenerator`: copick/CryoET-Data-Portal-style
.ndjson files for every placed protein, transmembrane protein, filament,
microtubule and gold bead (see `export_picks`).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..filament import FilamentInstance, MicrotubuleInstance
from ..membrane import TransmembranePlacement
from ._filaments import _filament_runs
from ._specs import BeadPlacement, TomogramPlacement


class _PickExportMixin:
    """`TomogramSpecimenGenerator`'s pick export (see module docstring)."""

    # Set by `TomogramSpecimenGenerator.__init__`/`generate`; declared for
    # type checking.
    target_shape: tuple[int, int, int]
    voxel_size: float
    instance_labels: torch.Tensor | None
    placements: list[TomogramPlacement]
    transmembrane_placements: list[TransmembranePlacement]
    filament_instances: list[FilamentInstance]
    microtubule_instances: list[MicrotubuleInstance]
    bead_instances: list[BeadPlacement]

    def export_picks(
        self,
        output_dir: str | Path,
        annotation_version: str = "1.0",
        oriented: bool = True,
        include_transmembrane: bool = True,
        include_filaments: bool = True,
        include_microtubules: bool = True,
        include_filler: bool = True,
        include_beads: bool = True,
    ) -> dict[str, Path]:
        """
        Write one copick/CryoET-Data-Portal-style .ndjson pick file per
        placed cytosol/lumen species (grouped by `(location, species_id)`
        so the same `pdb_source` declared at both locations never collides
        in one file) plus, by default, one per transmembrane species --
        one JSON object per line: ``{"type": "point"|"orientedPoint",
        "location": {"x", "y", "z"}[, "xyz_rotation_matrix"]}``.

        `TomogramPlacement.role == "filler"` placements (species declared
        via `ratio`, not `n_copies`) are INCLUDED by default, so a
        `ratio`-only config exports every species it declares. Pass
        `include_filler=False` to export only `n_copies`-declared species. A
        `(species_id, location)` pair placed as BOTH a target and filler
        (declared twice, once with `n_copies` and once with just `ratio`)
        keeps its filler instances in a separate ``-filler``-suffixed file,
        never merged with the target file.

        Transmembrane picks are oriented (a real `rotation_matrix`, unlike
        other membrane picks here, which are plain points) since
        `TransmembranePlacement` actually carries one.

        Coordinates are converted from this generator's box-centered
        convention (`position_xyz`/`center_xyz`, origin at the volume's
        center, matching `MembraneGenerator`'s own convention) to the
        corner-relative (``0..extent``) convention copick/the portal
        actually use -- the same conversion the other two generators'
        `export_picks` perform.

        Must be called after `generate()`.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Directory to write the .ndjson files into.
        annotation_version : str, optional
            Used only in the output filename
            (``"{name}-{version}_{type}.ndjson"``). Default "1.0".
        oriented : bool, optional
            If True (default), picks are written as ``"orientedPoint"``
            with each instance's rotation matrix included; if False, as
            plain ``"point"`` (location only).
        include_transmembrane : bool, optional
            If True (default), also write pick file(s) for transmembrane
            species, suffixed ``-transmembrane``.
        include_filaments : bool, optional
            If True (default), also write one pick file per filament
            species, suffixed ``-filament``. Each `FilamentInstance`'s own
            `position_xyz` is already in the corner-relative convention
            used here (see `_stamp_filaments`), so -- unlike
            placements/transmembrane above -- it's written directly, with
            no `+ extent_xyz / 2` conversion.
        include_microtubules : bool, optional
            If True (default), also write one pick file per microtubule
            species, suffixed ``-microtubule``: one entry per TUBE, whose
            ``path`` is the axis polyline, not one entry per dimer. A tube
            is a ~950-dimer object, and a pick file listing every dimer is
            rarely what a consumer wants; the per-dimer copies remain in
            `microtubule_dimer_instances` for anyone who does.
        include_filler : bool, optional
            If True, also write pick files for `role == "filler"`
            cytosol/lumen placements (suffixed ``-filler`` on a
            target/filler `(species_id, location)` collision, to avoid
            overwriting the target's own file). Default False.
        include_beads : bool, optional
            If True (default), also write every gold fiducial to a single
            ``gold-bead`` pick file, regardless of radius or which
            `bead_specs` population it came from -- nothing downstream
            distinguishes bead sizes. Always written as plain ``"point"``
            regardless of `oriented`: a bead has no meaningful
            per-instance orientation for picking purposes.

        Returns
        -------
        dict[str, pathlib.Path]
            Mapping of a grouping key (``"{species}-{location}"`` for
            cytosol/lumen instances, ``"{species}-transmembrane"`` for
            transmembrane instances, ``"{species}-filament"`` for filament
            instances) to written file path.
        """
        if self.instance_labels is None:
            raise RuntimeError("call generate() before export_picks()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        target_shape = self.target_shape
        voxel_size = self.voxel_size
        extent_xyz = (
            torch.tensor(
                [target_shape[2], target_shape[1], target_shape[0]],
                dtype=torch.float32,
            )
            * voxel_size
        )
        point_type = "orientedPoint" if oriented else "point"

        self._export_protein_picks(
            written,
            output_dir,
            annotation_version,
            point_type,
            oriented,
            extent_xyz,
            include_filler,
        )
        if include_transmembrane and self.transmembrane_placements:
            self._export_transmembrane_picks(
                written,
                output_dir,
                annotation_version,
                point_type,
                oriented,
                extent_xyz,
            )
        if include_filaments and self.filament_instances:
            self._export_filament_picks(
                written, output_dir, annotation_version, point_type, oriented
            )
        if include_microtubules and self.microtubule_instances:
            self._export_microtubule_paths(written, output_dir, annotation_version)
        if include_beads and self.bead_instances:
            self._export_bead_picks(written, output_dir, annotation_version, extent_xyz)

        return written

    def _export_protein_picks(
        self,
        written: dict[str, Path],
        output_dir: Path,
        annotation_version: str,
        point_type: str,
        oriented: bool,
        extent_xyz: torch.Tensor,
        include_filler: bool,
    ) -> None:
        """One file per ``(species, location)`` of the cytosol/lumen placements."""
        target_keys = {
            (placed.species_id, placed.location)
            for placed in self.placements
            if placed.role == "target"
        }
        by_key: dict[str, list[TomogramPlacement]] = {}
        for placed in self.placements:
            if placed.role == "filler" and not include_filler:
                continue
            name = Path(placed.species_id).stem
            key = f"{name}-{placed.location}"
            if (
                placed.role == "filler"
                and (placed.species_id, placed.location) in target_keys
            ):
                key = f"{key}-filler"
            by_key.setdefault(key, []).append(placed)
        for key, placed_list in by_key.items():
            path = (
                output_dir / f"{key}-{annotation_version}_{point_type.lower()}.ndjson"
            )
            with open(path, "w") as f:
                for placed in placed_list:
                    corner_xyz = placed.position_xyz + extent_xyz / 2
                    x, y, z = (float(v) for v in corner_xyz)
                    row: dict = {
                        "type": point_type,
                        "location": {"x": x, "y": y, "z": z},
                    }
                    if oriented:
                        row["xyz_rotation_matrix"] = (
                            placed.rotation_matrix.numpy().tolist()
                        )
                    f.write(json.dumps(row) + "\n")
            written[key] = path

    def _export_transmembrane_picks(
        self,
        written: dict[str, Path],
        output_dir: Path,
        annotation_version: str,
        point_type: str,
        oriented: bool,
        extent_xyz: torch.Tensor,
    ) -> None:
        """One file per transmembrane species, oriented."""
        by_species: dict[str, list[TransmembranePlacement]] = {}
        for tp in self.transmembrane_placements:
            by_species.setdefault(Path(tp.species_id).stem, []).append(tp)
        for species, tps in by_species.items():
            key = f"{species}-transmembrane"
            path = (
                output_dir / f"{key}-{annotation_version}_{point_type.lower()}.ndjson"
            )
            with open(path, "w") as f:
                for tp in tps:
                    corner_xyz = tp.center_xyz + extent_xyz / 2
                    x, y, z = (float(v) for v in corner_xyz)
                    row = {"type": point_type, "location": {"x": x, "y": y, "z": z}}
                    if oriented:
                        row["xyz_rotation_matrix"] = tp.rotation_matrix.numpy().tolist()
                    f.write(json.dumps(row) + "\n")
            written[key] = path

    def _export_filament_picks(
        self,
        written: dict[str, Path],
        output_dir: Path,
        annotation_version: str,
        point_type: str,
        oriented: bool,
    ) -> None:
        """Per filament species: the monomer picks and, beside them, one path per filament."""
        by_filament_code: dict[str, list[FilamentInstance]] = {}
        for inst in self.filament_instances:
            by_filament_code.setdefault(inst.code, []).append(inst)
        for code, insts in by_filament_code.items():
            key = f"{Path(code).stem}-filament"
            path = (
                output_dir / f"{key}-{annotation_version}_{point_type.lower()}.ndjson"
            )
            with open(path, "w") as f:
                for inst in insts:
                    x, y, z = (float(v) for v in inst.position_xyz)
                    row = {"type": point_type, "location": {"x": x, "y": y, "z": z}}
                    if oriented:
                        row["xyz_rotation_matrix"] = (
                            inst.rotation_matrix.numpy().tolist()
                        )
                    f.write(json.dumps(row) + "\n")
            written[key] = path

            # One `path` per filament as well, so the picks agree with
            # the label volume about what an object is: both now say a
            # filament, where the labels said one object and these
            # points said several dozen.
            #
            # Written in ADDITION to the oriented points rather than
            # instead of them. A path carries no orientations, and the
            # per-monomer rotation matrices above are what subtomogram
            # averaging of F-actin needs; nothing in the volume can
            # recover them. Microtubules ship only a path and so have
            # no per-dimer poses at all.
            path_file = output_dir / f"{key}-{annotation_version}_path.ndjson"
            with open(path_file, "w") as f:
                for run in _filament_runs(insts):
                    points = torch.stack([i.position_xyz for i in run])
                    centre = points.mean(dim=0)
                    f.write(
                        json.dumps(
                            {
                                "type": "path",
                                "location": {
                                    "x": float(centre[0]),
                                    "y": float(centre[1]),
                                    "z": float(centre[2]),
                                },
                                "path": [
                                    {
                                        "x": float(p[0]),
                                        "y": float(p[1]),
                                        "z": float(p[2]),
                                    }
                                    for p in points
                                ],
                                "n_monomers": len(run),
                            }
                        )
                        + "\n"
                    )
            written[f"{key}-path"] = path_file

    def _export_microtubule_paths(
        self, written: dict[str, Path], output_dir: Path, annotation_version: str
    ) -> None:
        """One path per microtubule, per species."""
        by_tube_code: dict[str, list[MicrotubuleInstance]] = {}
        for tube in self.microtubule_instances:
            by_tube_code.setdefault(tube.code, []).append(tube)
        for code, tubes in by_tube_code.items():
            key = f"{Path(code).stem}-microtubule"
            path = output_dir / f"{key}-{annotation_version}_path.ndjson"
            with open(path, "w") as f:
                for tube in tubes:
                    axis = tube.axis_xyz
                    centre = axis.mean(dim=0)
                    f.write(
                        json.dumps(
                            {
                                "type": "path",
                                "location": {
                                    "x": float(centre[0]),
                                    "y": float(centre[1]),
                                    "z": float(centre[2]),
                                },
                                "path": [
                                    {
                                        "x": float(p[0]),
                                        "y": float(p[1]),
                                        "z": float(p[2]),
                                    }
                                    for p in axis
                                ],
                                "radius": tube.lattice.radius,
                                "n_protofilaments": (tube.lattice.n_protofilaments),
                            }
                        )
                        + "\n"
                    )
            written[key] = path

    def _export_bead_picks(
        self,
        written: dict[str, Path],
        output_dir: Path,
        annotation_version: str,
        extent_xyz: torch.Tensor,
    ) -> None:
        """Every gold fiducial, as plain points, in one file."""
        # Every fiducial goes in one file regardless of radius or
        # population: nothing downstream distinguishes bead sizes, and
        # under a [low, high] radius each instance has a unique size,
        # so grouping by radius would write one file per bead.
        key = "gold-bead"
        path = output_dir / f"{key}-{annotation_version}_point.ndjson"
        with open(path, "w") as f:
            for bead in self.bead_instances:
                corner_xyz = bead.position_xyz + extent_xyz / 2
                x, y, z = (float(v) for v in corner_xyz)
                f.write(
                    json.dumps({"type": "point", "location": {"x": x, "y": y, "z": z}})
                    + "\n"
                )
        written[key] = path
