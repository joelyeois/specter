"""Microtubule surface-lattice geometry, frames and tube placement.

Pure geometry -- no PDB fetch, no rendering -- except for the two tests
marked ``network``, which check the hardcoded lattice constants against the
deposited structures they were measured from.
"""

from __future__ import annotations

import math

import pytest
import torch

from specter.specimen.filament import (
    DIMER_REPEAT,
    LATERAL_SPACING,
    MONOMER_RISE,
    MicrotubuleSpec,
    build_tube_instances,
    microtubule_axis_path,
    parallel_transport_frames,
    place_microtubules,
    solve_tube_lattice,
    thermal_flex_deg,
)
from conftest import seeded


# --------------------------------------------------------------------------
# Lattice
# --------------------------------------------------------------------------
def test_lattice_13_protofilaments_matches_3jal():
    """Measured off 3JAL: R = 111.1 A, stagger = 9.43 A."""
    lattice = solve_tube_lattice(13)
    assert lattice.radius == pytest.approx(111.1, abs=1.5)
    assert lattice.stagger == pytest.approx(9.43, abs=0.2)
    assert lattice.dimer_repeat == pytest.approx(82.0)


def test_lattice_14_protofilaments_matches_6dpu():
    """Measured off 6DPU: R = 118.5 A, stagger = 9.00 A.

    A 14-protofilament tube is wider and its stagger is *smaller* -- the
    lattice accommodates the extra protofilament by sliding the lateral
    bond, not by skewing the protofilaments. Nothing here asserts a
    supertwist: deposited helical parameters are not precise enough to pin
    one down (see `_lattice`'s module docstring).
    """
    lattice = solve_tube_lattice(14)
    assert lattice.radius == pytest.approx(118.5, abs=1.5)
    assert lattice.stagger == pytest.approx(9.00, abs=0.3)
    assert lattice.stagger < solve_tube_lattice(13).stagger


def test_seam_is_exactly_one_monomer_at_every_protofilament_number():
    """The A-lattice seam is a consequence of the arithmetic, not a case."""
    for n_pf in range(10, 17):
        lattice = solve_tube_lattice(n_pf)
        assert lattice.seam_offset == pytest.approx(MONOMER_RISE, abs=1e-6), n_pf


def test_seam_shows_up_as_a_single_odd_junction():
    """Every inter-protofilament junction steps by `stagger` except one."""
    lattice = solve_tube_lattice(13)
    registers = [p * lattice.stagger for p in range(lattice.n_protofilaments)]
    # Junction p -> p+1, closing back onto protofilament 0 through the seam.
    steps = [
        (registers[(p + 1) % 13] - registers[p]) % lattice.dimer_repeat
        for p in range(13)
    ]
    odd = [s for s in steps if abs(s - lattice.stagger) > 1e-6]
    assert len(odd) == 1
    # The odd junction is offset from a normal one by exactly one monomer:
    # an alpha meets a beta there (A-lattice) where every other junction is
    # alpha-alpha/beta-beta (B-lattice).
    assert (odd[0] - lattice.stagger) % lattice.dimer_repeat == pytest.approx(
        MONOMER_RISE, abs=1e-6
    )


def test_radius_scales_with_protofilament_number():
    radii = [solve_tube_lattice(n).radius for n in range(10, 17)]
    assert all(b > a for a, b in zip(radii, radii[1:]))
    # Circumference is set by the (constant) lateral spacing.
    lattice = solve_tube_lattice(13)
    assert 2 * math.pi * lattice.radius == pytest.approx(13 * LATERAL_SPACING)


def test_lattice_rejects_degenerate_tubes():
    with pytest.raises(ValueError):
        solve_tube_lattice(2)
    with pytest.raises(ValueError):
        solve_tube_lattice(13, n_start=0)


def test_thermal_flex_matches_worm_like_chain():
    """flex = sqrt(6*step/Lp); ~0.40 deg for tubulin at Lp = 1 mm."""
    assert thermal_flex_deg() == pytest.approx(0.40, abs=0.02)
    # Stiffer -> straighter, and the scaling is 1/sqrt(Lp).
    stiff = thermal_flex_deg(persistence_length=5.0e7)
    assert stiff == pytest.approx(thermal_flex_deg() / math.sqrt(5), rel=1e-6)


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------
def test_parallel_transport_frames_are_orthonormal_and_follow_the_tangent():
    positions = torch.cumsum(torch.randn(40, 3, generator=seeded(0)) * 5.0, dim=0)
    frames = parallel_transport_frames(positions)
    assert frames.shape == (40, 3, 3)

    identity = torch.eye(3).expand(40, 3, 3)
    assert torch.allclose(frames @ frames.transpose(1, 2), identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(frames), torch.ones(40), atol=1e-5)

    tangents = positions[1:] - positions[:-1]
    tangents = tangents / tangents.norm(dim=-1, keepdim=True)
    assert torch.allclose(frames[:-1, :, 2], tangents, atol=1e-5)


def test_parallel_transport_does_not_twist():
    """The whole point of parallel transport: no roll about the tangent.

    A frame built independently per point (align +Z to the tangent, e.g.
    `filament_orientations`) accumulates roll along a curved path, which
    would shear a tube's protofilaments apart.
    """
    t = torch.linspace(0, 2.0, 60)
    positions = torch.stack([torch.sin(t) * 300, torch.cos(t) * 300, t * 100], dim=1)
    frames = parallel_transport_frames(positions)

    for i in range(len(positions) - 2):
        # Rotation carrying frame i to frame i+1, expressed in frame i.
        relative = frames[i].T @ frames[i + 1]
        # Its component about the shared tangent (local z) must vanish.
        roll = math.atan2(float(relative[1, 0]), float(relative[0, 0]))
        assert abs(roll) < 1e-3, f"frame twisted by {math.degrees(roll):.4f} deg"


# --------------------------------------------------------------------------
# Tube construction
# --------------------------------------------------------------------------
def test_tube_instances_sit_on_the_wall():
    lattice = solve_tube_lattice(13)
    axis = torch.stack(
        [torch.zeros(30), torch.zeros(30), torch.arange(30.0) * lattice.dimer_repeat],
        dim=1,
    )
    instances = build_tube_instances("x.cif", 0, axis, lattice)
    assert len(instances) == 30 * 13

    positions = torch.stack([i.position_xyz for i in instances])
    radii = positions[:, :2].norm(dim=-1)
    assert torch.allclose(radii, torch.full_like(radii, lattice.radius), atol=1e-3)


def test_tube_instances_stay_on_the_wall_around_a_bend():
    """Rigid ring on a curved axis: distance to the axis must not drift."""
    spec = MicrotubuleSpec(code="x.cif", bend_radius=3.0e4)
    lattice = spec.lattice()
    axis = microtubule_axis_path(
        spec,
        60,
        torch.zeros(3),
        torch.tensor([1.0, 0.0, 0.0]),
        generator=seeded(1),
    )
    instances = build_tube_instances("x.cif", 0, axis, lattice)
    frames = parallel_transport_frames(axis)

    # Perpendicular distance from each dimer to its own ring's axis point.
    # (Distance to the nearest *sampled* axis point would instead measure
    # the polyline's 82 A sampling, since a dimer's register carries it up
    # to half a ring away along the tangent.)
    radii = []
    for instance in instances:
        ring = instance.monomer_index
        offset = instance.position_xyz - axis[ring]
        tangent = frames[ring][:, 2]
        perpendicular = offset - (offset @ tangent) * tangent
        radii.append(float(perpendicular.norm()))

    radii = torch.tensor(radii)
    assert torch.allclose(radii, torch.full_like(radii, lattice.radius), atol=1e-3)


def test_tube_rotations_point_x_outwards_and_z_along_the_axis():
    """The template's radial face must face out, or the wall is wrong."""
    lattice = solve_tube_lattice(13)
    axis = torch.stack(
        [torch.zeros(5), torch.zeros(5), torch.arange(5.0) * lattice.dimer_repeat],
        dim=1,
    )
    instances = build_tube_instances("x.cif", 0, axis, lattice)

    x_hat = torch.tensor([1.0, 0.0, 0.0])
    z_hat = torch.tensor([0.0, 0.0, 1.0])
    for instance in instances:
        radial = instance.position_xyz.clone()
        radial[2] = 0.0
        radial = radial / radial.norm()
        assert torch.allclose(instance.rotation_matrix @ x_hat, radial, atol=1e-4)
        assert torch.allclose(instance.rotation_matrix @ z_hat, z_hat, atol=1e-4)


def test_protofilament_registers_reproduce_the_stagger():
    lattice = solve_tube_lattice(13)
    axis = torch.stack(
        [torch.zeros(4), torch.zeros(4), torch.arange(4.0) * lattice.dimer_repeat],
        dim=1,
    )
    instances = build_tube_instances("x.cif", 0, axis, lattice)
    ring0 = [i for i in instances if i.monomer_index == 0]
    ring0.sort(key=lambda i: i.protofilament_index)
    z = torch.tensor([float(i.position_xyz[2]) for i in ring0])
    steps = z[1:] - z[:-1]
    assert torch.allclose(steps, torch.full_like(steps, lattice.stagger), atol=1e-4)


def test_place_microtubules_counts_and_bookkeeping():
    spec = MicrotubuleSpec(code="x.cif", n_copies=2, length=2000.0)
    instances, tubes = place_microtubules(
        [spec], (128, 256, 256), 8.0, generator=seeded(2)
    )
    n_rings = round(2000.0 / DIMER_REPEAT)
    assert len(tubes) == 2
    assert len(instances) == 2 * n_rings * 13
    assert {t.tube_id for t in tubes} == {0, 1}
    assert all(t.axis_xyz.shape == (n_rings, 3) for t in tubes)
    assert {i.protofilament_index for i in instances} == set(range(13))


def test_slab_confinement_keeps_long_tubes_in_plane():
    """A 4000 A tube cannot be steeply tilted in 800 A of ice."""
    spec = MicrotubuleSpec(code="x.cif", n_copies=12, length=4000.0)
    shape = (100, 512, 512)  # 800 A thick at 8 A/voxel
    _, tubes = place_microtubules([spec], shape, 8.0, generator=seeded(3))

    for tube in tubes:
        span_z = float(tube.axis_xyz[:, 2].max() - tube.axis_xyz[:, 2].min())
        assert span_z <= 800.0 * 1.35, span_z

    unconfined = MicrotubuleSpec(
        code="x.cif", n_copies=12, length=4000.0, confine_to_slab=False
    )
    _, wild = place_microtubules([unconfined], shape, 8.0, generator=seeded(3))
    assert max(
        float(t.axis_xyz[:, 2].max() - t.axis_xyz[:, 2].min()) for t in wild
    ) > max(float(t.axis_xyz[:, 2].max() - t.axis_xyz[:, 2].min()) for t in tubes)


def test_thermal_tube_is_nearly_straight():
    """At tubulin's persistence length a microtubule barely bends over a
    600 nm field -- transverse wander of order half a diameter."""
    spec = MicrotubuleSpec(code="x.cif", length=6000.0, confine_to_slab=False)
    axis = microtubule_axis_path(
        spec, 74, torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]), generator=seeded(4)
    )
    chord = axis[-1] - axis[0]
    contour = float((axis[1:] - axis[:-1]).norm(dim=-1).sum())
    assert float(chord.norm()) / contour > 0.999

    # Deviation from the straight line through the ends.
    unit = chord / chord.norm()
    offsets = axis - axis[0]
    perpendicular = offsets - (offsets @ unit).unsqueeze(1) * unit
    assert float(perpendicular.norm(dim=-1).max()) < 400.0


def test_arc_bend_is_smooth_unlike_a_wiggly_walk():
    """`bend_radius` curves; a bigger flex angle would only kink."""
    spec = MicrotubuleSpec(code="x.cif", bend_radius=1.0e4)
    axis = microtubule_axis_path(
        spec, 74, torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]), generator=seeded(5)
    )
    tangents = axis[1:] - axis[:-1]
    tangents = tangents / tangents.norm(dim=-1, keepdim=True)
    turns = torch.arccos((tangents[:-1] * tangents[1:]).sum(dim=-1).clamp(-1, 1))
    # Constant curvature: every step turns by step/radius, identically.
    expected = DIMER_REPEAT / 1.0e4
    assert torch.allclose(turns, torch.full_like(turns, expected), atol=1e-4)


def test_arc_bend_rejects_nonpositive_radius():
    spec = MicrotubuleSpec(code="x.cif", bend_radius=-5.0)
    with pytest.raises(ValueError):
        microtubule_axis_path(spec, 10, torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]))


# --------------------------------------------------------------------------
# Against the deposited structures the constants came from
# --------------------------------------------------------------------------
@pytest.mark.network
@pytest.mark.parametrize(
    "source, n_pf",
    [("3JAL", 13), ("6DPU", 14)],
)
def test_constants_match_deposited_structures(source, n_pf, tmp_path):
    """Guards the hardcoded constants against the data they came from."""
    from specter.specimen.filament import measure_source_lattice

    measured = measure_source_lattice(source, pdb_cache_dir=str(tmp_path))
    model = solve_tube_lattice(n_pf)

    assert round(measured["n_protofilaments"]) == n_pf
    assert measured["monomer_rise"] == pytest.approx(MONOMER_RISE, abs=0.6)
    assert measured["lateral_spacing"] == pytest.approx(LATERAL_SPACING, abs=1.0)
    assert measured["radius"] == pytest.approx(model.radius, rel=0.02)


@pytest.mark.network
def test_rendered_microtubule_is_a_hollow_tube(tmp_path):
    """End-to-end: the thing that actually has to be true of the density.

    Checks the wall against the real microtubule it is meant to be -- a
    ~250 A tube around an empty ~170 A lumen -- which is what would break
    if the dimer template's radial orientation were wrong (nothing in the
    pure-geometry tests above can catch that).
    """
    from specter.specimen import TomogramSpecimenGenerator

    shape, voxel_size = (64, 224, 224), 8.0
    generator = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=shape,
        voxel_size=voxel_size,
        protein_specs=[],
        microtubule_specs=[MicrotubuleSpec(n_copies=1, length=1200.0)],
        pdb_cache_dir=str(tmp_path),
        seed=0,
        progressbars=False,
    )
    volume = generator.generate().cpu()
    assert len(generator.microtubule_instances) == 1

    tube = generator.microtubule_instances[0]
    nz, ny, nx = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij"
    )
    points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], 1).float()
    radius = torch.cdist(points * voxel_size, tube.axis_xyz).min(dim=1).values

    density = volume.flatten()
    lattice = tube.lattice

    in_wall = (radius > lattice.radius - 25.0) & (radius < lattice.radius + 25.0)
    wall_density = float(density[in_wall].mean())

    # Hollow: the lumen is empty relative to the wall. Stated as a ratio
    # rather than "exactly zero at the lumen boundary" because the wall's
    # rendered density has a soft inward tail -- the tubulin dimer is a
    # real structure, not a shell of zero thickness.
    assert float(density[radius < lattice.radius / 3].max()) == 0.0
    assert float(density[radius < lattice.radius / 2].mean()) < 0.01 * wall_density

    # Wall: density concentrates at the protofilament radius, not outside.
    beyond = radius > lattice.radius + 60.0
    assert wall_density > 10 * float(density[beyond].mean() + 1e-9)

    # One instance-label id for the whole tube, not one per dimer.
    labels = torch.unique(generator.instance_labels.cpu())
    assert labels.numel() == 2  # background + the tube
    assert len(generator.microtubule_dimer_instances) > 100


@pytest.mark.network
def test_protein_fill_does_not_pack_into_the_lumen(tmp_path):
    """The lumen is empty but sealed -- nothing cytosolic gets inside.

    Occupied-voxel exclusion alone does not cover this: the lumen contains
    no density, so without `_microtubule_lumen_mask` the packer treats it
    as free space and puts protein inside a closed tube.
    """
    from specter.specimen import TomogramProteinSpec, TomogramSpecimenGenerator

    shape, voxel_size = (48, 224, 224), 8.0
    generator = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=shape,
        voxel_size=voxel_size,
        protein_specs=[TomogramProteinSpec(pdb_source="1BXN", location="cytosol")],
        microtubule_specs=[MicrotubuleSpec(n_copies=1)],
        occupancy_fraction=0.25,
        pdb_cache_dir=str(tmp_path),
        seed=7,
        progressbars=False,
    )
    generator.generate()
    assert generator.placements, "no protein placed -- test would be vacuous"

    extent = (
        torch.tensor([shape[2], shape[1], shape[0]], dtype=torch.float32) * voxel_size
    )
    centres = torch.stack([p.position_xyz for p in generator.placements]) + extent / 2
    axis = torch.cat([t.axis_xyz for t in generator.microtubule_instances])
    closest = torch.cdist(centres, axis).min(dim=1).values
    radius = generator.microtubule_instances[0].lattice.radius
    assert float(closest.min()) > radius


@pytest.mark.network
def test_extracted_dimer_is_in_the_microtubule_frame(tmp_path):
    """+Z must be the protofilament axis: the dimer is ~82 A long there and
    much narrower across, which is what makes the wall come out right."""
    from specter.pdb import PDB
    from specter.specimen.filament import extract_mt_dimer

    path = extract_mt_dimer(pdb_cache_dir=str(tmp_path))
    pdb = PDB(path, assembly=False, verbose=False)
    coordinates = pdb.coordinates

    extent = coordinates.max(dim=0).values - coordinates.min(dim=0).values
    assert float(extent[2]) == pytest.approx(2 * MONOMER_RISE, abs=25.0)
    assert float(extent[2]) > float(extent[0])
    assert float(extent[2]) > float(extent[1])


def test_one_instance_id_per_filament_not_per_monomer():
    """A filament is one object in the segmentation, like a microtubule.

    Microtubules already shared an id across their dimers, so a tube reads
    as one object rather than ~950 loose pieces. Actin did not: 20
    filaments appeared as 765 separate instances, and a picker evaluated
    against that ground truth was being asked to find monomers."""
    import torch

    from specter.specimen.filament import FilamentSpec, place_filaments
    from specter.specimen.tomogram import TomogramSpecimenGenerator

    rng = torch.Generator()
    rng.manual_seed(11)
    spec = FilamentSpec(
        code="1J6Z",
        step=27.3,
        flex_deg=12.0,
        twist_deg=166.15,
        n_copies=20,
        n_monomers=(20, 60),
    )
    instances = place_filaments([spec], (300, 1200, 1200), 5.0, rng)
    ids, after = TomogramSpecimenGenerator._one_id_per_filament(None, instances, 100)

    assert len(instances) > 100, "expected many monomers, else the test proves nothing"
    assert len(set(ids.tolist())) == spec.n_copies
    assert int(ids.min()) == 100 and after == 100 + spec.n_copies
    # Ids are contiguous, and every monomer of one filament shares one.
    assert sorted(set(ids.tolist())) == list(range(100, 100 + spec.n_copies))


def test_filament_ids_do_not_collide_across_species():
    """Both placers number filaments with `range(spec.n_copies)`, so every
    spec contributes a filament 0. Grouping on that key alone would merge
    them -- which is what `_stamp_microtubules` used to do, and every
    microtubule spec resolves to the same cached dimer `code`, so nothing
    else separated them."""
    import torch

    from specter.specimen.filament import FilamentSpec, place_filaments
    from specter.specimen.tomogram import TomogramSpecimenGenerator

    rng = torch.Generator()
    rng.manual_seed(0)
    specs = [
        FilamentSpec(
            code="1J6Z",
            step=27.3,
            flex_deg=12.0,
            twist_deg=166.15,
            n_copies=3,
            n_monomers=(5, 5),
        ),
        FilamentSpec(
            code="1TUB",
            step=85.0,
            flex_deg=3.0,
            twist_deg=0.0,
            n_copies=3,
            n_monomers=(5, 5),
        ),
    ]
    instances = place_filaments(specs, (300, 1200, 1200), 5.0, rng)
    ids, _ = TomogramSpecimenGenerator._one_id_per_filament(None, instances, 1)
    assert len(set(ids.tolist())) == 6, "two specs x 3 filaments must give 6 ids"


def test_filaments_export_one_path_per_filament_alongside_oriented_points():
    """Picks and labels must agree on what an object is.

    The label volume gives a filament one id. Its picks gave one entry per
    MONOMER, so the two ground truths disagreed: 20 objects against 765
    points. A `path` per filament is written as well, matching both the
    label volume and how microtubules were already exported.

    Written IN ADDITION to the oriented points, not instead: a path has no
    orientations, and the per-monomer rotation matrices are what
    subtomogram averaging of F-actin needs. Nothing in the density volume
    can recover them."""

    import torch

    from specter.specimen.filament import FilamentSpec, place_filaments
    from specter.specimen.tomogram._generator import _filament_runs

    rng = torch.Generator()
    rng.manual_seed(11)
    spec = FilamentSpec(
        code="1J6Z",
        step=27.3,
        flex_deg=12.0,
        twist_deg=166.15,
        n_copies=7,
        n_monomers=(20, 40),
    )
    instances = place_filaments([spec], (300, 1200, 1200), 5.0, rng)
    runs = list(_filament_runs(instances))

    assert len(runs) == spec.n_copies
    assert sum(len(r) for r in runs) == len(instances), "every monomer in exactly one"
    # Each run is one filament: same code and filament_id throughout.
    for run in runs:
        assert len({(i.code, i.filament_id) for i in run}) == 1
    # And the runs are distinct filaments, not one repeated.
    assert len({(r[0].code, r[0].filament_id) for r in runs}) == spec.n_copies


def test_filament_runs_handles_a_single_monomer_filament():
    """A filament of one monomer is still one run, not a boundary case that
    swallows its neighbour."""
    import torch

    from specter.specimen.filament import FilamentSpec, place_filaments
    from specter.specimen.tomogram._generator import _filament_runs

    rng = torch.Generator()
    rng.manual_seed(2)
    spec = FilamentSpec(
        code="1J6Z",
        step=27.3,
        flex_deg=12.0,
        twist_deg=166.15,
        n_copies=4,
        n_monomers=(1, 1),
    )
    instances = place_filaments([spec], (300, 1200, 1200), 5.0, rng)
    runs = list(_filament_runs(instances))
    assert len(runs) == 4 and all(len(r) == 1 for r in runs)
