"""
Membrane stage of `TomogramSpecimenGenerator`: placing each
:class:`MembraneInstance` by bounding sphere, generating it with its
transmembrane proteins, max-merging it into the shared canvas, then
classifying shell/lumen/cytosol regions on the composite and labelling each
instance's shell. See `._generator`'s module docstring for where this runs
in the generation order.
"""

from __future__ import annotations

import gc
import warnings

import torch

from ...arrays import clip_insert_bounds
from ...progress import TqdmProgress, phase_done, phase_start, status
from ..membrane import TransmembranePlacement
from ..packing import pack_hard_spheres_3d
from ._helpers import (
    _allowed_region_exclusion_field,
    _canvas_box_center,
    _canvas_indices,
    _insert_local_labels,
    _insert_shell_label,
    _insert_volume_max,
    _instance_bounding_radius,
    _mask_at_positions,
    _position_to_center_index,
    _snap_to_canvas_voxel,
)
from ._regions import classify_membrane_regions
from ._specs import MembraneInstance


class _MembraneStageMixin:
    """`TomogramSpecimenGenerator`'s membrane stage (see module docstring)."""

    # Set by `TomogramSpecimenGenerator.__init__`; declared for type checking.
    membrane_instances: list[MembraneInstance]
    target_shape: tuple[int, int, int]
    region_density_threshold: float | None
    min_transmembrane_spacing: float
    gap: float
    clip_axes: tuple[bool, bool, bool]
    seed: int | None
    device: str | torch.device
    accumulator_device: torch.device
    progressbars: bool
    retain_membrane_fields: bool
    regions: dict[str, torch.Tensor] | None
    membrane_labels: torch.Tensor | None
    placed_membrane_instances: list[MembraneInstance]
    transmembrane_placements: list[TransmembranePlacement]

    def _stage_membranes(
        self,
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        carbon_mask: torch.Tensor | None,
        box: tuple[float, float, float],
        voxel_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Place, generate and composite the membrane instances with their
        transmembrane proteins, then classify regions and label the shells."""
        # Resolve any omitted position_xyz via collision-rejecting random
        # placement, treating each instance as a bounding sphere (see
        # _instance_bounding_radius) -- an instance that doesn't fit
        # without colliding is dropped (never .generate()-called at all,
        # cheaper than generating first and rejecting after), matching
        # this module's own "reject and move on" packing philosophy rather
        # than retrying at new positions. Every instance is placed this
        # way: `MembraneInstance.position_xyz` is not settable at
        # construction, and is written here for the instances accepted.
        # The phases below stay on phase_start/phase_done rather than
        # `with phase(...)`: their completion line reports a count (instances
        # placed, structures loaded) that only exists once the block has run.
        _membrane_phase_start = phase_start(
            "Membranes", disable=not self.progressbars or not self.membrane_instances
        )
        # Sub-phases, so this reports like the species phases below rather
        # than as one opaque total: at the 2 A production grid this phase is
        # over half the run.
        _membrane_place_start = phase_start(
            "  Placement", disable=not self.progressbars or not self.membrane_instances
        )
        to_composite = self._place_membrane_instances(
            box, carbon_mask, self.target_shape, voxel_size
        )
        phase_done(
            "  Placement",
            _membrane_place_start,
            disable=not self.progressbars or not self.membrane_instances,
        )
        _membrane_build_start = phase_start(
            "  Generation & compositing",
            disable=not self.progressbars or not to_composite,
        )
        volume, instance_labels, next_instance_id, instance_shell_masks = (
            self._composite_membranes(
                to_composite,
                volume,
                instance_labels,
                next_instance_id,
                carbon_mask,
                box,
                voxel_size,
            )
        )
        phase_done(
            "  Generation & compositing",
            _membrane_build_start,
            disable=not self.progressbars or not to_composite,
        )
        # classify_membrane_regions' own threshold: needs the FULL
        # composite's peak (unlike `_instance_shell_mask`'s per-instance
        # thresholds), so can only be resolved after every instance
        # is merged into volume.
        _membrane_regions_start = phase_start(
            "  Region classification",
            disable=not self.progressbars or not self.membrane_instances,
        )
        self._classify_regions(volume)
        phase_done(
            "  Region classification",
            _membrane_regions_start,
            disable=not self.progressbars or not self.membrane_instances,
        )

        _membrane_label_start = phase_start(
            "  Shell labelling",
            disable=not self.progressbars or not self.membrane_instances,
        )
        self._label_membrane_shells(
            instance_shell_masks, tuple(volume.shape), voxel_size
        )
        phase_done(
            "  Shell labelling",
            _membrane_label_start,
            disable=not self.progressbars or not self.membrane_instances,
        )
        phase_done(
            f"Membranes ({len(instance_shell_masks)}/"
            f"{len(self.membrane_instances)} instance(s) placed)",
            _membrane_phase_start,
            disable=not self.progressbars or not self.membrane_instances,
        )
        return volume, instance_labels, next_instance_id

    def _place_membrane_instances(
        self,
        box: tuple[float, float, float],
        carbon_mask: torch.Tensor | None,
        target_shape: tuple[int, int, int],
        voxel_size: float,
    ) -> list[MembraneInstance]:
        """Resolve every membrane instance's position by collision-rejecting
        random placement and return the ones that fit."""
        to_composite: list[MembraneInstance] = []
        if self.membrane_instances:
            radii = torch.tensor(
                [
                    _instance_bounding_radius(mi.generator)
                    for mi in self.membrane_instances
                ]
            )
            if carbon_mask is not None:
                allowed_field, exclusion_field, field_voxel_size, field_origin = (
                    _allowed_region_exclusion_field(
                        (~carbon_mask).cpu(), target_shape, voxel_size
                    )
                )
            with status(
                f"Placing {len(self.membrane_instances)} membrane instance(s)",
                disable=not self.progressbars,
            ):
                coords, accepted_idx = pack_hard_spheres_3d(
                    radii,
                    box,
                    gap=self.gap,
                    seed=self.seed,
                    device="cpu",  # see self.device's own docstring
                    clip_axes=self.clip_axes,
                    # A handful of large spheres, each alone in its radius
                    # stage: a pass is one trial, and an organelle spanning
                    # most of the slab has a narrow window to land in.
                    max_passes=2000,
                    stall_patience=2000,
                    exclusion_distance_field=(
                        exclusion_field if carbon_mask is not None else None
                    ),
                    field_voxel_size=field_voxel_size
                    if carbon_mask is not None
                    else None,
                    sampling_mask=(allowed_field if carbon_mask is not None else None),
                    field_origin=field_origin if carbon_mask is not None else None,
                    box_center=_canvas_box_center(target_shape, voxel_size),
                )
            n_dropped = len(self.membrane_instances) - accepted_idx.numel()
            if n_dropped:
                warnings.warn(
                    f"TomogramSpecimenGenerator: {n_dropped}/"
                    f"{len(self.membrane_instances)} membrane instances did not "
                    "fit without colliding (with each other, the box walls, or "
                    "the carbon film, if any) and were dropped (never "
                    "generated).",
                    stacklevel=2,
                )
            # Snapped to the canvas voxel the instance will be merged at,
            # before anything reads the position (see
            # `_snap_to_canvas_voxel`). The snap moves a centre at most half
            # a voxel per axis, against a bounding sphere packed with `gap`.
            coords = _snap_to_canvas_voxel(coords, voxel_size)
            for k, orig_idx in enumerate(accepted_idx.tolist()):
                mi = self.membrane_instances[orig_idx]
                mi.position_xyz = tuple(coords[k].tolist())
                to_composite.append(mi)
        return to_composite

    def _composite_membranes(
        self,
        to_composite: list[MembraneInstance],
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        carbon_mask: torch.Tensor | None,
        box: tuple[float, float, float],
        voxel_size: float,
    ) -> tuple[
        torch.Tensor, torch.Tensor, int, list[tuple[MembraneInstance, torch.Tensor]]
    ]:
        """Generate each placed instance, embed its transmembrane proteins,
        max-merge it into the canvas and keep its shell mask for labelling."""
        # Generate + place transmembrane proteins per instance, each in its
        # own centered local frame, then composite densities into the
        # shared canvas (max-merge) before any region classification --
        # classify_membrane_regions needs the full composite, not
        # per-instance pieces.
        self.transmembrane_placements = []
        self.placed_membrane_instances = []
        instance_shell_masks: list[tuple[MembraneInstance, torch.Tensor]] = []
        membrane_progress = TqdmProgress(
            transient=True, disable=not self.progressbars or not to_composite
        )
        with membrane_progress as progress:
            membrane_task = progress.add_task(
                "Generating membrane instances", total=len(to_composite)
            )
            for mi in to_composite:
                # Every mi here has a concrete position_xyz by construction:
                # to_composite only ever collects accepted instances, whose
                # position_xyz was just set above.
                assert mi.position_xyz is not None
                generated = self._generate_instance(mi)
                progress.update(membrane_task, advance=1)
                if generated is None:
                    continue
                bare_peak, tm_placements = generated

                tm_placements = self._drop_placements_outside(
                    tm_placements, tuple(volume.shape), voxel_size
                )
                local_volume = mi.generator.volume
                assert local_volume is not None
                if carbon_mask is not None:
                    self._clip_instance_to_carbon(
                        mi, local_volume, carbon_mask, voxel_size
                    )
                    tm_placements = self._drop_placements_in_carbon(
                        tm_placements, carbon_mask, voxel_size
                    )
                self.transmembrane_placements.extend(tm_placements)
                volume = _insert_volume_max(
                    volume, local_volume, mi.position_xyz, voxel_size
                )

                shell_mask = self._instance_shell_mask(local_volume, bare_peak)
                # The membrane label is the BILAYER, not the bilayer plus
                # whatever is embedded in it. `_insert_blend` has already
                # decided which voxels the protein displaced lipid from --
                # reuse that decision rather than making a second, looser
                # one here. The proteins themselves become ordinary protein
                # instances just below, so nothing goes unlabelled: the two
                # volumes partition the membrane between them.
                tm_labels = mi.generator.transmembrane_labels
                if tm_labels is not None:
                    shell_mask = shell_mask & (tm_labels == 0)
                    instance_labels = _insert_local_labels(
                        instance_labels,
                        tm_labels,
                        id_offset=next_instance_id - 1,
                        position_xyz=mi.position_xyz,
                        voxel_size=voxel_size,
                    )
                    # Reserve from the count `place_transmembrane` actually
                    # LABELLED (its own `placements`, ids 1..n), not from
                    # `tm_placements` -- the carbon drop may have pruned that
                    # list, and reserving the short count would let the next
                    # membrane's offset collide with this one's higher ids.
                    next_instance_id += len(mi.generator.placements)
                    mi.generator.transmembrane_labels = None
                del tm_labels
                instance_shell_masks.append((mi, shell_mask.cpu()))
                self.placed_membrane_instances.append(mi)
                mi.generator.volume = None
                del local_volume
                self._release_device_memory()
        return volume, instance_labels, next_instance_id, instance_shell_masks

    def _generate_instance(
        self, mi: MembraneInstance
    ) -> tuple[float, list[TransmembranePlacement]] | None:
        """Generate `mi` and embed its transmembrane proteins.

        Returns the bare bilayer's peak and the placements, offset into the
        canvas frame, or None when the instance's working grid clipped the
        organelle it drew.
        """
        assert mi.position_xyz is not None
        mi.generator.generate()
        if mi.generator.clipped_at_boundary:
            if not self.retain_membrane_fields:
                mi.generator.field = None
                mi.generator.volume = None
            warnings.warn(
                "TomogramSpecimenGenerator: a membrane instance's own "
                "working grid was too small for the organelle size it "
                "actually drew (clipped_at_boundary=True on its "
                "MembraneGenerator) -- skipped rather than compositing a "
                "visibly truncated shape. Increase that instance's own "
                "target_shape, or omit it for auto-sizing.",
                stacklevel=3,
            )
            return None
        # The bilayer's own peak, read BEFORE place_transmembrane inserts
        # protein density. A protein template's peak is typically several
        # times a smoothed bilayer's, so taking this afterwards would
        # inflate the shell threshold and erode the very thing it is meant
        # to outline.
        bare_bilayer = mi.generator.volume
        assert bare_bilayer is not None  # generate() just ran
        bare_peak = float(bare_bilayer.max())
        del bare_bilayer
        tm_placements = mi.generator.place_transmembrane(
            min_spacing_angstrom=self.min_transmembrane_spacing
        )
        # The field is needed for surface-site placement, but not for
        # compositing, labels or subsequent tomogram stages. CLI runs need
        # only those outputs; retaining every dense field would keep
        # completed organelles in GPU memory.
        if not self.retain_membrane_fields:
            mi.generator.field = None
        offset = torch.tensor(mi.position_xyz, dtype=torch.float32)
        for tp in tm_placements:
            tp.center_xyz = tp.center_xyz + offset
        return bare_peak, tm_placements

    @staticmethod
    def _drop_placements_outside(
        tm_placements: list[TransmembranePlacement],
        shape_zyx: tuple[int, ...],
        voxel_size: float,
    ) -> list[TransmembranePlacement]:
        """Drop placements whose centre's canvas voxel is outside the box.

        A membrane instance may extend past the tomogram walls (clip_axes),
        and `_insert_volume_max` clips the part that does -- so a protein
        embedded out there contributes no density and gets no instance
        label. Its ground-truth entry goes too, rather than shipping a pick
        at coordinates the volume does not cover.
        """
        if not tm_placements:
            return tm_placements
        centers = torch.stack([tp.center_xyz for tp in tm_placements])
        index = _canvas_indices(centers, shape_zyx, voxel_size)
        shape = torch.tensor(shape_zyx, device=index.device)
        outside = ((index < 0) | (index >= shape)).any(dim=1)
        n_outside = int(outside.sum())
        if n_outside:
            warnings.warn(
                f"TomogramSpecimenGenerator: dropped {n_outside} "
                "transmembrane protein placement(s) that fell "
                "outside the tomogram box, on the part of a "
                "membrane instance clipped by the volume walls "
                "(no density was rendered for them, so a pick "
                "there would claim a particle the volume does "
                "not contain).",
                stacklevel=3,
            )
        return [tp for tp, drop in zip(tm_placements, outside.tolist()) if not drop]

    @staticmethod
    def _clip_instance_to_carbon(
        mi: MembraneInstance,
        local_volume: torch.Tensor,
        carbon_mask: torch.Tensor,
        voxel_size: float,
    ) -> None:
        """Zero `local_volume` and its transmembrane labels, in place,
        wherever they would land on the carbon film.

        Placed instances shouldn't reach here at all in the common case
        (their bounding sphere was already kept clear of carbon by the RSA
        exclusion field), so this is normally a no-op -- a safety net for an
        irregular organelle whose true rendered shape extends past its own
        bounding-sphere approximation. It runs BEFORE both compositing and
        the shell mask/ground-truth labelling, so both reflect the clip,
        and reuses the index math `_insert_volume_max` itself uses.
        """
        assert mi.position_xyz is not None
        center_zyx = _position_to_center_index(
            mi.position_xyz, tuple(carbon_mask.shape), voxel_size
        )
        bounds = clip_insert_bounds(center_zyx, local_volume.shape, carbon_mask.shape)
        if bounds is None:
            return
        dst, src = bounds
        forbidden = carbon_mask[dst].to(local_volume.device)
        if not forbidden.any():
            return
        warnings.warn(
            "TomogramSpecimenGenerator: clipped part of a "
            "membrane instance (an irregular shape "
            "exceeding its own bounding-sphere estimate) "
            "that overlapped the carbon film.",
            stacklevel=3,
        )
        local_volume[src] = local_volume[src] * (~forbidden).to(local_volume.dtype)
        # Same clip on the protein labels, so a transmembrane instance whose
        # density was just removed doesn't keep a ground-truth label sitting
        # on carbon.
        tm_labels = mi.generator.transmembrane_labels
        if tm_labels is not None:
            tm_labels[src] = tm_labels[src] * (~forbidden).to(tm_labels.dtype)

    @staticmethod
    def _drop_placements_in_carbon(
        tm_placements: list[TransmembranePlacement],
        carbon_mask: torch.Tensor,
        voxel_size: float,
    ) -> list[TransmembranePlacement]:
        """Drop placements whose centre's canvas voxel is carbon.

        `place_transmembrane` baked these placements' density into the
        instance before `_clip_instance_to_carbon` zeroed it, so this only
        fixes the ground-truth list (what export_picks writes), which would
        otherwise claim a particle where no density is left. Checked by
        centre point, not full footprint (as for filament copies): a
        placement whose centre is just outside carbon keeps its partially
        clipped entry, matching how bead and protein exclusion are also
        voxel-level rather than footprint-exact.
        """
        if not tm_placements:
            return tm_placements
        centers = torch.stack([tp.center_xyz for tp in tm_placements])
        in_carbon = _mask_at_positions(carbon_mask, centers, voxel_size)
        n_dropped = int(in_carbon.sum())
        if n_dropped:
            warnings.warn(
                f"TomogramSpecimenGenerator: dropped "
                f"{n_dropped} transmembrane protein "
                "placement(s) clipped by the carbon film "
                "(density already removed above; this drops "
                "their now-stale ground-truth pick entries "
                "too).",
                stacklevel=3,
            )
        return [tp for tp, drop in zip(tm_placements, in_carbon.tolist()) if not drop]

    def _instance_shell_mask(
        self, local_volume: torch.Tensor, bare_peak: float
    ) -> torch.Tensor:
        """Where `local_volume` is membrane shell, thresholded per instance.

        Computed now, while `local_volume` is at hand, and kept as a bool:
        a GLOBAL threshold would need the full composite's peak, forcing
        every instance's full-resolution array to stay resident until the
        loop finishes, which OOMs once one instance reaches tens of GB. This
        instance's own peak (when `region_density_threshold` is auto) is
        also the right reference for labelling it: an instance with a lower
        peak (a smaller organelle on a coarser working grid, say) should
        not have its shell mislabelled as background because a brighter
        sibling set a higher bar. An explicit `region_density_threshold` is
        an absolute density, identical to a shared global threshold.
        """
        if self.region_density_threshold is not None:
            threshold = self.region_density_threshold
        else:
            threshold = 0.05 * bare_peak if bare_peak > 0 else 0.0
        return local_volume > threshold

    def _release_device_memory(self) -> None:
        """Hand freed CUDA blocks back to the driver between instances.

        Dropping the last reference is not enough: the caching allocator
        keeps freed blocks in its own pool, and each instance's grid can be
        a different size (random per-instance organelle size), so the next
        allocation can fail on fragmentation with several GiB "reserved but
        unallocated" -- observed directly. gc.collect() first in case a
        tensor is only reachable through a reference cycle.
        """
        if torch.device(self.device).type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    def _classify_regions(self, volume: torch.Tensor) -> None:
        """Classify the composite into shell, lumen and cytosol (``self.regions``)."""
        # classify_membrane_regions' own threshold: needs the FULL
        # composite's peak (unlike `_instance_shell_mask`'s per-instance
        # thresholds), so can only be resolved after every instance
        # is merged into volume.
        threshold = self.region_density_threshold
        if threshold is None:
            peak = float(volume.max())
            threshold = 0.05 * peak if peak > 0 else 0.0
        self.regions = classify_membrane_regions(volume, threshold)

    def _label_membrane_shells(
        self,
        instance_shell_masks: list[tuple[MembraneInstance, torch.Tensor]],
        target_shape: tuple[int, ...],
        voxel_size: float,
    ) -> None:
        """Write each instance's shell mask into ``self.membrane_labels``."""
        membrane_labels = torch.zeros(
            target_shape, dtype=torch.int32, device=self.accumulator_device
        )
        for instance_id, (mi, shell_mask) in enumerate(instance_shell_masks, start=1):
            assert mi.position_xyz is not None  # see identical assert above
            membrane_labels, overlap = _insert_shell_label(
                membrane_labels, shell_mask, instance_id, mi.position_xyz, voxel_size
            )
            if overlap:
                warnings.warn(
                    f"TomogramSpecimenGenerator: membrane instance {instance_id} "
                    "(1-indexed, in membrane_instances order) overlaps a voxel "
                    "already claimed by an earlier instance in membrane_labels "
                    "-- the earlier instance's label wins there (first-write-"
                    "wins). This can happen even with collision-checked "
                    "placement, since the RSA solve treats each instance as a "
                    "bounding sphere while an irregular organelle's true "
                    "rendered shape can extend past that estimate.",
                    stacklevel=2,
                )
        self.membrane_labels = membrane_labels
