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
    _insert_local_labels,
    _insert_shell_label,
    _insert_volume_max,
    _instance_bounding_radius,
    _position_to_center_index,
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
        # than retrying at new positions. Instances with an explicit
        # position_xyz are placed as given and NOT included in this
        # collision check.
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
        # composite's peak (unlike instance_shell_masks' per-instance
        # thresholds above), so can only be resolved after every instance
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
                allowed_field, exclusion_field, field_voxel_size = (
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
                    exclusion_distance_field=(
                        exclusion_field if carbon_mask is not None else None
                    ),
                    field_voxel_size=field_voxel_size
                    if carbon_mask is not None
                    else None,
                    sampling_mask=(allowed_field if carbon_mask is not None else None),
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
                mi.generator.generate()
                progress.update(membrane_task, advance=1)
                if mi.generator.clipped_at_boundary:
                    warnings.warn(
                        "TomogramSpecimenGenerator: a membrane instance's own "
                        "working grid was too small for the organelle size it "
                        "actually drew (clipped_at_boundary=True on its "
                        "MembraneGenerator) -- skipped rather than compositing a "
                        "visibly truncated shape. Increase that instance's own "
                        "target_shape/voxel_size, or omit target_shape "
                        "entirely for auto-sizing.",
                        stacklevel=2,
                    )
                    continue
                # The bilayer's own peak, read BEFORE place_transmembrane
                # inserts protein density. A protein template's peak is
                # typically several times a smoothed bilayer's, so taking
                # this afterwards would inflate the shell threshold below
                # and erode the very thing it is meant to outline.
                bare_bilayer = mi.generator.volume
                assert bare_bilayer is not None  # generate() just ran
                bare_peak = float(bare_bilayer.max())
                tm_placements = mi.generator.place_transmembrane(
                    min_spacing_angstrom=self.min_transmembrane_spacing
                )
                offset = torch.tensor(mi.position_xyz, dtype=torch.float32)
                for tp in tm_placements:
                    tp.center_xyz = tp.center_xyz + offset

                # A membrane instance may extend past the tomogram walls
                # (clip_axes), and `_insert_volume_max` clips the part that
                # does -- so a protein embedded out there contributes no
                # density and gets no instance label. Drop its ground-truth
                # entry too, rather than shipping a pick at coordinates the
                # volume does not cover. Same reasoning, and the same
                # center-point granularity, as the carbon-film drop below.
                if tm_placements:
                    half = torch.tensor(
                        [box[2] / 2, box[1] / 2, box[0] / 2], dtype=torch.float32
                    )
                    centers = torch.stack([tp.center_xyz for tp in tm_placements])
                    outside = (centers.abs() >= half).any(dim=1)
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
                            stacklevel=2,
                        )
                        tm_placements = [
                            tp
                            for tp, drop in zip(tm_placements, outside.tolist())
                            if not drop
                        ]

                local_volume = mi.generator.volume
                assert local_volume is not None
                if carbon_mask is not None:
                    # Placed instances shouldn't reach here at all in the
                    # common case (their bounding sphere was already kept
                    # clear of carbon by the RSA exclusion field above), so
                    # this is normally a no-op -- it's a safety net for an
                    # irregular organelle whose true rendered shape extends
                    # past its own bounding-sphere approximation. Zeroes
                    # local_volume wherever it would
                    # land on carbon, BEFORE both compositing into `volume`
                    # and the shell_mask/ground-truth labeling below, so
                    # both consistently reflect the clip -- reuses the same
                    # index math `_insert_volume_max` itself uses, rather
                    # than duplicating it.
                    center_zyx = _position_to_center_index(
                        mi.position_xyz, tuple(volume.shape), voxel_size
                    )
                    bounds = clip_insert_bounds(
                        center_zyx, local_volume.shape, volume.shape
                    )
                    if bounds is not None:
                        dst, src = bounds
                        forbidden = carbon_mask[dst].to(local_volume.device)
                        if forbidden.any():
                            warnings.warn(
                                "TomogramSpecimenGenerator: clipped part of a "
                                "membrane instance (an irregular shape "
                                "exceeding its own bounding-sphere estimate) "
                                "that overlapped the carbon film.",
                                stacklevel=2,
                            )
                            local_volume[src] = local_volume[src] * (~forbidden).to(
                                local_volume.dtype
                            )
                            # Same clip on the protein labels, so a
                            # transmembrane instance whose density was just
                            # removed doesn't keep a ground-truth label
                            # sitting on carbon.
                            tm_labels_clip = mi.generator.transmembrane_labels
                            if tm_labels_clip is not None:
                                tm_labels_clip[src] = tm_labels_clip[src] * (
                                    ~forbidden
                                ).to(tm_labels_clip.dtype)

                    # `place_transmembrane` (above) already baked these
                    # placements' own density into local_volume before this
                    # point, so a placement whose center lands on carbon
                    # just had its density zeroed by the clip above too --
                    # this only fixes the separate ground-truth bookkeeping
                    # list (self.transmembrane_placements, what export_picks
                    # writes out), which would otherwise still claim a
                    # particle sits somewhere with no actual density left.
                    # Checked by center point, not full rendered footprint
                    # (same granularity already used for filament monomers
                    # in _stamp_filaments) -- a placement whose center is
                    # just outside carbon but whose template partially
                    # overlapped it keeps its (partially clipped) entry,
                    # matching how e.g. bead/protein exclusion is also
                    # voxel-level, not footprint-exact.
                    if tm_placements:
                        shape_zyx = tuple(volume.shape)
                        z_c, y_c, x_c = (s // 2 for s in shape_zyx)
                        centers = torch.stack([tp.center_xyz for tp in tm_placements])
                        iz = (
                            (z_c + torch.round(centers[:, 2] / voxel_size))
                            .long()
                            .clamp(0, shape_zyx[0] - 1)
                        )
                        iy = (
                            (y_c + torch.round(centers[:, 1] / voxel_size))
                            .long()
                            .clamp(0, shape_zyx[1] - 1)
                        )
                        ix = (
                            (x_c + torch.round(centers[:, 0] / voxel_size))
                            .long()
                            .clamp(0, shape_zyx[2] - 1)
                        )
                        cm_dev = carbon_mask.device
                        in_carbon = carbon_mask[
                            iz.to(cm_dev), iy.to(cm_dev), ix.to(cm_dev)
                        ].cpu()
                        n_dropped_tm = int(in_carbon.sum())
                        if n_dropped_tm:
                            warnings.warn(
                                f"TomogramSpecimenGenerator: dropped "
                                f"{n_dropped_tm} transmembrane protein "
                                "placement(s) clipped by the carbon film "
                                "(density already removed above; this drops "
                                "their now-stale ground-truth pick entries "
                                "too).",
                                stacklevel=2,
                            )
                            tm_placements = [
                                tp
                                for tp, drop in zip(tm_placements, in_carbon.tolist())
                                if not drop
                            ]
                self.transmembrane_placements.extend(tm_placements)
                volume = _insert_volume_max(
                    volume, local_volume, mi.position_xyz, voxel_size
                )
                # Per-instance shell mask, computed and stashed as a bool
                # (~4x smaller than float32, and ~4-8x smaller again than
                # keeping the full density array around) NOW, while
                # local_volume is still cheaply available, rather than in a
                # second pass after every instance has run. A GLOBAL
                # threshold (shared across every instance) would need the
                # full composite's peak, which isn't known until the loop
                # finishes -- forcing every instance's full-resolution
                # array to stay resident simultaneously until then.
                # Confirmed directly: that OOMs well before this loop even
                # finishes, now that generation-resolution decoupling
                # (MembraneGenerator's max_field_voxels) lets a single
                # instance's own volume reach tens of GB. Using THIS
                # instance's own peak instead when region_density_threshold
                # is auto (None) -- a per-instance peak is also the more
                # correct reference for per-instance shell LABELING, since
                # an instance whose own peak is lower (a smaller organelle
                # resolved on a coarser working grid, say) should not have
                # its true shell mislabeled as background just
                # because a brighter sibling set a higher global bar).
                # When region_density_threshold is explicitly set, it's
                # already an absolute density value (not a fraction, see
                # this same fallback below for self.regions), so using it
                # directly here is identical to a shared global threshold
                # -- no behaviour change in that case.
                if self.region_density_threshold is not None:
                    instance_threshold = self.region_density_threshold
                else:
                    instance_threshold = 0.05 * bare_peak if bare_peak > 0 else 0.0
                shell_mask = local_volume > instance_threshold
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
                    # `tm_placements` -- the carbon block above may have
                    # pruned that list, and reserving the short count would
                    # let the next membrane's offset collide with this
                    # one's higher ids.
                    next_instance_id += len(mi.generator.placements)
                    mi.generator.transmembrane_labels = None
                shell_mask = shell_mask.cpu()
                instance_shell_masks.append((mi, shell_mask))
                self.placed_membrane_instances.append(mi)
                mi.generator.volume = None
                # Dropping the last reference above is not enough by
                # itself: PyTorch's CUDA caching allocator keeps freed
                # blocks in its own pool rather than returning them to the
                # driver, and each instance's own working/output grid can
                # be a DIFFERENT size (random per-instance organelle size),
                # so the next instance's allocation can fail on
                # fragmentation even though the previous instance's memory
                # was already dereferenced -- confirmed directly: a second
                # instance's OOM here, with the traceback showing several
                # GiB "reserved but unallocated" at the same time as the
                # failing allocation. gc.collect() first in case any
                # tensor is only reachable via a reference cycle (autograd
                # graphs can create these) that plain refcounting wouldn't
                # free promptly.
                del local_volume
                if torch.device(self.device).type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()
        return volume, instance_labels, next_instance_id, instance_shell_masks

    def _classify_regions(self, volume: torch.Tensor) -> None:
        """Classify the composite into shell, lumen and cytosol (``self.regions``)."""
        # classify_membrane_regions' own threshold: needs the FULL
        # composite's peak (unlike instance_shell_masks' per-instance
        # thresholds above), so can only be resolved after every instance
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
