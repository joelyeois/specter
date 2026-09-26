"""
Filament and microtubule stage of `TomogramSpecimenGenerator`: placing
`FilamentSpec`/`MicrotubuleSpec` species, dropping copies that land in the
carbon film, assigning one instance id per filament or tube, and rendering
the copies into the shared canvas. See `._generator`'s module docstring for
why filaments are neither region-gated nor collision-avoiding.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from ...progress import phase_done, phase_start, status
from ...rotations import build_affine_matrix
from ..filament import (
    FilamentInstance,
    FilamentSpec,
    MicrotubuleInstance,
    MicrotubuleSpec,
    place_filaments,
    place_microtubules,
)
from ..membrane._placement import align_principal_axis_to_z
from ._helpers import _insert_rotated_copies

if TYPE_CHECKING:
    from ...pdb import PDB


def _filament_runs(
    instances: list[FilamentInstance],
) -> Iterator[list[FilamentInstance]]:
    """Split placed monomers into one list per filament, in order.

    The single definition of where one filament ends and the next begins,
    shared by instance labelling and pick export so the two ground truths
    cannot disagree about what an object is.

    A boundary is a change of ``(code, filament_id)`` between CONSECUTIVE
    instances, not a change of the key alone. Both placers number
    filaments with ``range(spec.n_copies)``, restarting per spec, so every
    spec contributes a filament 0; each emits one filament's monomers
    consecutively, which is what makes runs the right unit.

    One case this cannot separate: a spec contributing exactly one
    filament, followed immediately by a same-`code` filament 0 from the
    next spec. Separating those needs the placers to number filaments
    globally, which is the real fix if it ever matters -- `filament_id` is
    internal and never written to picks.
    """
    run: list[FilamentInstance] = []
    previous: tuple[str, int] | None = None
    for inst in instances:
        key = (inst.code, inst.filament_id)
        if previous is not None and key != previous:
            yield run
            run = []
        previous = key
        run.append(inst)
    if run:
        yield run


class _FilamentStageMixin:
    """`TomogramSpecimenGenerator`'s filament/microtubule stage (see module docstring)."""

    # Set by `TomogramSpecimenGenerator.__init__`; declared for type checking.
    target_shape: tuple[int, int, int]
    filament_specs: list[FilamentSpec]
    microtubule_specs: list[MicrotubuleSpec]
    pdb_cache_dir: str
    seed: int | None
    device: str | torch.device
    chunk_size: int | None
    progressbars: bool
    filament_instances: list[FilamentInstance]
    microtubule_instances: list[MicrotubuleInstance]
    microtubule_dimer_instances: list[FilamentInstance]

    if TYPE_CHECKING:
        # Defined on `TomogramSpecimenGenerator`, shared with its protein stage.
        def _load_pdb(self, source: str) -> PDB: ...

        def _build_species_template(
            self,
            pdb: PDB,
            voxel_size: float,
            device: str | torch.device,
            coordinates: torch.Tensor | None = None,
        ) -> torch.Tensor: ...

    def _stage_filaments(
        self,
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        voxel_size: float,
        carbon_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor | None]:
        """Stamp the filaments and microtubules and return the obstacle mask the
        later stages avoid (None when there are none)."""
        # Filaments (then gold fiducial beads, see below) render right
        # after membranes, BEFORE cytosol/lumen protein packing (see module
        # docstring) -- obstacle_mask (voxels they actually occupy, from
        # instance_labels before any protein instance touches it) is then
        # folded into the beads' sampling mask and the protein packer's
        # occupancy grid, alongside the membrane shell.
        if self.filament_specs or self.microtubule_specs:
            _filament_phase_start = phase_start(
                "Filaments", disable=not self.progressbars
            )
            if self.filament_specs:
                with status(
                    f"Placing {len(self.filament_specs)} filament species",
                    disable=not self.progressbars,
                ):
                    volume, instance_labels, next_instance_id = self._stamp_filaments(
                        volume,
                        instance_labels,
                        next_instance_id,
                        voxel_size,
                        carbon_mask,
                    )
            else:
                self.filament_instances = []
            if self.microtubule_specs:
                with status(
                    f"Placing {len(self.microtubule_specs)} microtubule species",
                    disable=not self.progressbars,
                ):
                    volume, instance_labels, next_instance_id = (
                        self._stamp_microtubules(
                            volume,
                            instance_labels,
                            next_instance_id,
                            voxel_size,
                            carbon_mask,
                        )
                    )
            phase_done(
                f"Filaments ({len(self.filament_instances)} monomer instance(s), "
                f"{len(self.microtubule_instances)} microtubule(s))",
                _filament_phase_start,
                disable=not self.progressbars,
            )
            obstacle_mask = instance_labels > 0
            if self.microtubule_instances:
                # A microtubule's lumen is EMPTY but not accessible: it is
                # sealed by the tube wall. Occupied-voxel exclusion alone
                # would happily pack cytosolic protein inside it, which is
                # exactly what lumenal particles are not (microtubule inner
                # proteins are explicitly out of scope -- see `_lattice`).
                obstacle_mask = obstacle_mask | self._microtubule_lumen_mask(
                    voxel_size, obstacle_mask.device
                )
        else:
            self.filament_instances = []
            obstacle_mask = None
        return volume, instance_labels, next_instance_id, obstacle_mask

    def _one_id_per_filament(
        self, instances: list[FilamentInstance], next_instance_id: int
    ) -> tuple[torch.Tensor, int]:
        """One instance id per filament, rather than one per monomer.

        Segmentation ground truth should mark a filament as an object, the
        way it already marked a microtubule as one rather than as ~950
        loose dimers. Labelled per monomer, 20 actin filaments appear as
        765 separate objects, and a picker evaluated against them is being
        asked to find monomers.

        Grouped on runs of equal ``(code, filament_id)`` rather than on
        the key alone. Both placers number filaments with
        ``range(spec.n_copies)``, restarting per spec, so two specs each
        contribute a filament 0; keying on the pair alone would merge
        them. Every placer emits one filament's monomers consecutively,
        so a change of key is a filament boundary.

        That leaves one case this cannot separate: a spec contributing
        exactly one filament, immediately followed by a filament of the
        same `code` and id from the next spec. Distinguishing those needs
        the placers to number filaments globally, which is the real fix if
        it ever matters -- `filament_id` is internal, used only here and
        never written to picks.
        """
        ids: list[int] = []
        current = next_instance_id
        for run in _filament_runs(instances):
            ids.extend([current] * len(run))
            current += 1
        return torch.tensor(ids, dtype=torch.int32), current

    def _stamp_filaments(
        self,
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        voxel_size: float,
        carbon_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Place and render every `filament_specs` species, continuing
        `instance_labels`'s own instance-id counter from the transmembrane
        proteins. Filaments render before beads and the cytosol/lumen
        protein fill, which then avoid them.

        `place_filaments` draws positions in `[0, extent)` -- a corner-
        relative box -- while `volume`/`instance_labels` (and
        `insert_particles_into_micrograph`/`_insert_instance_labels`, the
        same helpers the cytosol/lumen proteins use) are centered at
        physical (0,0,0). Only the LOCAL `positions_centered` used for
        rendering is shifted by `-extent/2` to bridge that; `self.
        filament_instances` keeps each `FilamentInstance`'s original
        corner-relative `position_xyz` untouched, since that's the
        convention `export_picks` itself writes out directly.

        `place_filaments` itself has no obstacle awareness (a genuine
        collision-avoiding random walk is a bigger algorithmic change than
        this needs -- see module docstring): individual monomer instances
        that land inside `carbon_mask`, if given, are dropped here after
        the fact instead, the same "truncated at render/insert time"
        treatment already applied to monomers that wander outside the
        volume entirely (see `place_filaments`'s own docstring). A dropped
        monomer mid-path just leaves a gap in that filament, not a
        redirected walk around the film.
        """
        target_shape = self.target_shape

        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)
        instances = place_filaments(self.filament_specs, target_shape, voxel_size, rng)
        instances = self._drop_instances_in_carbon(
            instances, carbon_mask, voxel_size, "filament monomer"
        )

        self.filament_instances = instances
        if not instances:
            return volume, instance_labels, next_instance_id
        instance_ids, after = self._one_id_per_filament(instances, next_instance_id)
        return self._render_filament_instances(
            volume,
            instance_labels,
            after,
            voxel_size,
            instances,
            instance_ids=instance_ids,
        )

    def _drop_instances_in_carbon(
        self,
        instances: list[FilamentInstance],
        carbon_mask: torch.Tensor | None,
        voxel_size: float,
        what: str,
    ) -> list[FilamentInstance]:
        """Drop copies whose centre lands inside the carbon film.

        Shared by filament and microtubule stamping -- neither placer is
        obstacle-aware, so this is the same "reject after the fact" pass
        described in `_stamp_filaments`.
        """
        if carbon_mask is None or not instances:
            return instances

        nz, ny, nx = self.target_shape
        pos = torch.stack([inst.position_xyz for inst in instances])  # (N,3) x,y,z
        ix = (pos[:, 0] / voxel_size).long().clamp(0, nx - 1)
        iy = (pos[:, 1] / voxel_size).long().clamp(0, ny - 1)
        iz = (pos[:, 2] / voxel_size).long().clamp(0, nz - 1)
        cm_dev = carbon_mask.device
        in_carbon = carbon_mask[iz.to(cm_dev), iy.to(cm_dev), ix.to(cm_dev)].cpu()
        n_dropped = int(in_carbon.sum())
        if n_dropped:
            warnings.warn(
                f"TomogramSpecimenGenerator: dropped {n_dropped} {what} "
                "instance(s) that landed inside the carbon film.",
                stacklevel=2,
            )
            instances = [
                inst for inst, drop in zip(instances, in_carbon.tolist()) if not drop
            ]
        return instances

    def _stamp_microtubules(
        self,
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        voxel_size: float,
        carbon_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Place and render every `microtubule_specs` species.

        A microtubule reaches the renderer as many rigid copies of one
        alpha-beta tubulin dimer -- the same `FilamentInstance` form
        filaments use -- so this shares `_render_filament_instances`
        wholesale. The one difference is instance labelling: every dimer of
        one tube gets the SAME instance id, so segmentation ground truth
        marks microtubules as objects rather than as ~950 loose dimers.
        """
        rng = torch.Generator()
        if self.seed is not None:
            # Offset from the filament seed so two species with the same
            # spec don't land on identical paths.
            rng.manual_seed(self.seed + 1)
        instances, tubes = place_microtubules(
            self.microtubule_specs,
            self.target_shape,
            voxel_size,
            generator=rng,
            pdb_cache_dir=self.pdb_cache_dir,
        )
        instances = self._drop_instances_in_carbon(
            instances, carbon_mask, voxel_size, "microtubule dimer"
        )

        self.microtubule_instances = tubes
        self.microtubule_dimer_instances = instances
        if not instances:
            return volume, instance_labels, next_instance_id

        # One instance id per tube, via the same helper actin uses. The
        # previous spelling grouped on `filament_id` alone, which merged
        # tube 0 of one [[microtubules]] spec with tube 0 of the next --
        # every spec resolves to the same cached dimer `code`, so nothing
        # else separated them.
        instance_ids, after = self._one_id_per_filament(instances, next_instance_id)
        return self._render_filament_instances(
            volume,
            instance_labels,
            after,
            voxel_size,
            instances,
            instance_ids=instance_ids,
            align_to_z=False,
        )

    def _microtubule_lumen_mask(
        self, voxel_size: float, device: torch.device
    ) -> torch.Tensor:
        """Voxels enclosed by a placed microtubule's wall, lumen included.

        Built by stamping a sphere of the tube's own radius at every ring of
        every axis polyline. Consecutive rings are one dimer repeat apart
        (82 A) while the radius is ~111 A, so the stamped spheres overlap
        and seal the tube along its whole length without needing a real
        distance transform over the full canvas.
        """
        nz, ny, nx = self.target_shape
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)

        for tube in self.microtubule_instances:
            radius_vox = tube.lattice.radius / voxel_size
            reach = int(math.ceil(radius_vox))
            for point in tube.axis_xyz:
                cx, cy, cz = (float(v) / voxel_size for v in point)
                ix0, ix1 = max(0, int(cx) - reach), min(nx, int(cx) + reach + 1)
                iy0, iy1 = max(0, int(cy) - reach), min(ny, int(cy) + reach + 1)
                iz0, iz1 = max(0, int(cz) - reach), min(nz, int(cz) + reach + 1)
                if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
                    continue
                zz, yy, xx = torch.meshgrid(
                    torch.arange(iz0, iz1, device=device, dtype=torch.float32),
                    torch.arange(iy0, iy1, device=device, dtype=torch.float32),
                    torch.arange(ix0, ix1, device=device, dtype=torch.float32),
                    indexing="ij",
                )
                inside = (
                    (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
                ) <= radius_vox**2
                mask[iz0:iz1, iy0:iy1, ix0:ix1] |= inside
        return mask

    def _render_filament_instances(
        self,
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        voxel_size: float,
        instances: list[FilamentInstance],
        instance_ids: torch.Tensor | None = None,
        align_to_z: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Render placed monomer/dimer copies: one template per species,
        rotated and inserted once per instance.

        Parameters
        ----------
        instance_ids : torch.Tensor, optional
            Per-instance segmentation ids, shape ``(len(instances),)``.
            Default None: number them sequentially from
            ``next_instance_id``, which is then advanced. Microtubule
            stamping passes explicit ids so a whole tube shares one.
        align_to_z : bool, optional
            Pre-rotate the template's longest principal axis onto ``+Z``.
            Default True, as filament monomers need. Microtubules pass
            False: their dimer template is already in the microtubule frame
            (`_tubulin.extract_mt_dimer`), where the roll about ``+Z``
            carries the radial orientation that principal-axis alignment
            has no way to know about and would be free to destroy.
        """
        extent_xyz = (
            torch.tensor(self.target_shape[::-1], dtype=torch.float32) * voxel_size
        )

        if instance_ids is None:
            ids = torch.arange(
                next_instance_id,
                next_instance_id + len(instances),
                dtype=torch.int32,
            )
            next_instance_id += len(instances)
        else:
            ids = instance_ids.to(torch.int32)

        by_code: dict[str, list[tuple[FilamentInstance, int]]] = {}
        for inst, inst_id in zip(instances, ids.tolist()):
            by_code.setdefault(inst.code, []).append((inst, inst_id))

        pdb_cache: dict[str, PDB] = {}
        templates: dict[str, torch.Tensor] = {}
        for code in by_code:
            if code not in pdb_cache:
                pdb_cache[code] = self._load_pdb(code)
            pdb = pdb_cache[code]
            coordinates = (
                align_principal_axis_to_z(pdb.coordinates)
                if align_to_z
                else pdb.coordinates
            )
            templates[code] = self._build_species_template(
                pdb, voxel_size, self.device, coordinates=coordinates
            )

        offset = (extent_xyz / 2).to(self.device)
        for code, entries in by_code.items():
            template = templates[code]

            insts = [inst for inst, _ in entries]
            positions_centered = (
                torch.stack([inst.position_xyz for inst in insts]).to(self.device)
                - offset
            )
            R = torch.stack([inst.rotation_matrix for inst in insts]).to(self.device)
            theta = build_affine_matrix(R)

            instance_ids = torch.tensor(
                [inst_id for _, inst_id in entries],
                dtype=torch.int32,
                device=self.device,
            )

            volume, instance_labels = _insert_rotated_copies(
                template,
                theta,
                positions_centered,
                instance_ids,
                volume,
                instance_labels,
                voxel_size,
                self.chunk_size,
            )

        return volume, instance_labels, next_instance_id
