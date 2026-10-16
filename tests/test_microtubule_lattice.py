"""Microtubule surface-lattice geometry, frames and tube placement.

Pure geometry -- no PDB fetch, no rendering -- except for the tests marked
``network``, which check the hardcoded lattice constants and the extracted
dimer against the deposited structures they come from. The dimer's polarity
is also pinned offline, on a synthetic protofilament patch.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from specter.specimen.filament import (
    ACTIN_SPEC,
    DIMER_REPEAT,
    LATERAL_SPACING,
    MONOMER_RISE,
    PERSISTENCE_LENGTH,
    PROTOFILAMENT_SPEC,
    MicrotubuleSpec,
    build_tube_instances,
    microtubule_axis_path,
    parallel_transport_frames,
    place_microtubules,
    solve_tube_lattice,
    thermal_flex_deg,
)
from conftest import seeded

# Deposited lateral helix of 3JAL (13 protofilaments), measured by
# superposing laterally adjacent alpha or beta monomers of the deposited
# coordinates: -27.7 degrees about the tube axis for +9.46 A of rise. A
# negative twist for a positive rise is a left-handed 3-start helix
# (Mandelkow et al. 1986).
_3JAL_LATERAL_TWIST_DEG = -27.7
_3JAL_LATERAL_RISE = 9.46


def _screw(rotation: torch.Tensor, translation: torch.Tensor) -> tuple[float, float]:
    """(twist in degrees, rise) of the rigid motion ``x -> R x + t``, about
    the screw axis oriented so that the rise is positive."""
    import roma

    rotvec = roma.rotmat_to_rotvec(rotation.double())
    angle = float(rotvec.norm())
    axis = rotvec / angle
    rise = float(translation.double() @ axis)
    twist = math.degrees(angle)
    return (twist, rise) if rise >= 0 else (-twist, -rise)


# --------------------------------------------------------------------------
# Lattice
# --------------------------------------------------------------------------
def test_lattice_13_protofilaments_matches_3jal():
    """Measured off 3JAL: R = 110.9 A, stagger (its deposited helical rise)
    = 9.46 A."""
    lattice = solve_tube_lattice(13)
    assert lattice.radius == pytest.approx(110.9, abs=1.5)
    assert lattice.stagger == pytest.approx(9.46, abs=0.05)
    assert lattice.dimer_repeat == pytest.approx(82.0)


def test_lattice_14_protofilaments_matches_6dpv():
    """Measured off 6DPV (14 protofilaments, GDP): R = 118.5 A, stagger =
    8.75 A. (6DPU, the same lattice with GMPCPP, is expanded: 8.99 A.)

    A 14-protofilament tube is wider and its stagger is *smaller* -- the
    lattice accommodates the extra protofilament by sliding the lateral
    bond, not by skewing the protofilaments. Nothing here asserts a
    supertwist: deposited helical parameters are not precise enough to pin
    one down (see `_lattice`'s module docstring).
    """
    lattice = solve_tube_lattice(14)
    assert lattice.radius == pytest.approx(118.5, abs=1.5)
    assert lattice.stagger == pytest.approx(8.75, abs=0.1)
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
    """flex = sqrt(6*step/Lp); ~2.3 deg for a microtubule at the in-cell
    apparent persistence length, 30 um."""
    assert PERSISTENCE_LENGTH == pytest.approx(3.0e5)
    assert thermal_flex_deg() == pytest.approx(2.32, abs=0.01)
    # Stiffer -> straighter, and the scaling is 1/sqrt(Lp).
    stiff = thermal_flex_deg(persistence_length=5.0e7)
    assert stiff == pytest.approx(
        thermal_flex_deg() * math.sqrt(PERSISTENCE_LENGTH / 5.0e7), rel=1e-6
    )


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


def test_lattice_is_a_left_handed_three_start_helix():
    """Laterally adjacent dimers are related by the deposited screw.

    Stepping from one protofilament to the next must turn -360/13 degrees
    about the tube axis for +stagger of rise, as 3JAL's coordinates do
    (`_3JAL_LATERAL_TWIST_DEG`, `_3JAL_LATERAL_RISE`). The mirror lattice
    has the same radius, stagger and seam, so nothing else catches it.
    """
    lattice = solve_tube_lattice(13)
    axis = torch.stack(
        [torch.zeros(3), torch.zeros(3), torch.arange(3.0) * lattice.dimer_repeat],
        dim=1,
    )
    instances = build_tube_instances("x.cif", 0, axis, lattice)
    ring = sorted(
        (i for i in instances if i.monomer_index == 1),
        key=lambda i: i.protofilament_index,
    )
    for a, b in zip(ring, ring[1:]):
        rotation = b.rotation_matrix @ a.rotation_matrix.T
        translation = b.position_xyz - rotation @ a.position_xyz
        twist, rise = _screw(rotation, translation)
        assert twist == pytest.approx(_3JAL_LATERAL_TWIST_DEG, abs=0.1)
        assert rise == pytest.approx(_3JAL_LATERAL_RISE, abs=0.05)


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


def test_axis_walk_reproduces_the_persistence_length():
    """The default walk is a worm-like chain of persistence length
    `PERSISTENCE_LENGTH`: tangents decorrelate as exp(-s / L_p)."""
    spec = MicrotubuleSpec(code="x.cif", confine_to_slab=False)
    n_rings, lag = 64, 60
    generator = seeded(4)
    cosines = []
    for _ in range(300):
        axis = microtubule_axis_path(
            spec, n_rings, torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]), generator
        )
        steps = axis[1:] - axis[:-1]
        tangents = steps / steps.norm(dim=-1, keepdim=True)
        cosines.append(float(tangents[0] @ tangents[lag]))
    expected = math.exp(-lag * DIMER_REPEAT / PERSISTENCE_LENGTH)
    assert 1 - sum(cosines) / len(cosines) == pytest.approx(1 - expected, rel=0.2)


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
    "source, n_pf, rise",
    # Deposited helical rise per lateral step x N / 3: 9.46 x 13/3 (3JAL,
    # GMPCPP compacted by EB3), 8.99 x 14/3 (6DPU, GMPCPP, expanded) and
    # 8.754 x 14/3 (6DPV, GDP).
    [("3JAL", 13, 40.99), ("6DPU", 14, 41.95), ("6DPV", 14, 40.85)],
)
def test_constants_match_deposited_structures(source, n_pf, rise, tmp_path):
    """Guards the hardcoded constants against the data they came from.

    The rise is set by the nucleotide, not by N: `MONOMER_RISE` is the
    compacted GDP lattice (3JAL, 6DPV), and the expanded GMPCPP lattice of
    6DPU sits about 1 A above it.
    """
    from specter.specimen.filament import measure_source_lattice

    measured = measure_source_lattice(source, pdb_cache_dir=str(tmp_path))
    model = solve_tube_lattice(n_pf)

    assert round(measured["n_protofilaments"]) == n_pf
    assert measured["monomer_rise"] == pytest.approx(rise, abs=0.15)
    if source != "6DPU":
        assert measured["monomer_rise"] == pytest.approx(MONOMER_RISE, abs=0.2)
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


@pytest.mark.network
def test_extracted_3jal_dimer_is_one_heterodimer_pointing_plus_end_up(tmp_path):
    """The two chains are one dimer, and +Z is its plus end.

    In 3JAL alpha binds GTP at the N-site and beta GMPCPP (G2P) at the
    E-site. The N-site lies inside the dimer, covered by its own beta (whose
    Lys254 comes within 3.6 A of the GTP); the E-site faces the next dimer
    and is exposed at the plus end. An inter-dimer pair has the two
    nucleotides the other way round, with the GTP at the template's end and
    the G2P between the monomers.
    """
    import gemmi

    from specter.specimen.filament import extract_mt_dimer

    model = gemmi.read_structure(extract_mt_dimer(pdb_cache_dir=str(tmp_path)))[0]

    def xyz(residues) -> np.ndarray:
        return np.array([[a.pos.x, a.pos.y, a.pos.z] for r in residues for a in r])

    alpha, beta = (
        next(c for c in model if any(r.name == name for r in c))
        for name in ("GTP", "G2P")
    )
    z_alpha = xyz(r for r in alpha if r.name != "GTP")[:, 2].mean()
    z_beta = xyz(r for r in beta if r.name != "G2P")[:, 2].mean()
    n_site = xyz(r for r in alpha if r.name == "GTP")
    e_site = xyz(r for r in beta if r.name == "G2P")

    assert z_alpha < 0 < z_beta
    assert z_alpha < n_site[:, 2].mean() < z_beta  # inside the dimer
    assert e_site[:, 2].mean() > z_beta  # at the plus end
    lys254 = xyz(r for r in beta if r.seqid.num == 254)
    assert next(r.name for r in beta if r.seqid.num == 254) == "LYS"
    assert np.linalg.norm(lys254[:, None] - n_site[None], axis=-1).min() < 5.0


def _write_tubulin_patch(path, plus_sign: int) -> dict[str, str]:
    """A synthetic 3-protofilament x 4-monomer microtubule patch, as mmCIF.

    Each monomer is a 420-atom cloud on the 13-protofilament lattice, with
    its nucleotide on the face pointing to the plus end, at
    ``plus_sign * z``. Chains run A..L protofilament by protofilament from
    the minus end, alpha first, so the intra-dimer pairs are (A, B), (C, D),
    ... and an alpha's beta neighbour on its minus side belongs to the
    previous dimer. Returns chain name -> "alpha" / "beta".
    """
    import gemmi

    rng = np.random.default_rng(0)
    structure = gemmi.Structure()
    structure.name = "SYNT"
    model = gemmi.Model("1")
    kinds: dict[str, str] = {}

    def atom(name: str, element: str, xyz: np.ndarray) -> gemmi.Atom:
        a = gemmi.Atom()
        a.name, a.element, a.occ = name, gemmi.Element(element), 1.0
        a.pos = gemmi.Position(*map(float, xyz))
        return a

    names = iter("ABCDEFGHIJKL")
    for p in range(3):
        phi = math.radians(60.0 + p * _3JAL_LATERAL_TWIST_DEG)
        for m in range(4):
            kind = "alpha" if m % 2 == 0 else "beta"
            z = plus_sign * (m * MONOMER_RISE + p * _3JAL_LATERAL_RISE)
            centre = np.array([110.9 * math.cos(phi), 110.9 * math.sin(phi), z])
            chain = gemmi.Chain(next(names))
            kinds[chain.name] = kind
            directions = rng.normal(size=(420, 3))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            cloud = centre + 24.0 * directions * rng.uniform(size=(420, 1)) ** (1 / 3)
            for i, point in enumerate(cloud):
                residue = gemmi.Residue()
                residue.name, residue.seqid = "ALA", gemmi.SeqId(i + 1, " ")
                residue.entity_type = gemmi.EntityType.Polymer
                residue.add_atom(atom("CA", "C", point))
                chain.add_residue(residue)
            nucleotide = gemmi.Residue()
            nucleotide.name = "GTP" if kind == "alpha" else "GDP"
            nucleotide.seqid, nucleotide.het_flag = gemmi.SeqId(500, " "), "H"
            nucleotide.entity_type = gemmi.EntityType.NonPolymer
            site = centre + np.array([0.0, 0.0, plus_sign * 17.0])
            for j in range(5):
                nucleotide.add_atom(atom(f"P{j}", "P", site + rng.normal(0, 0.5, 3)))
            chain.add_residue(nucleotide)
            model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()

    # gemmi does not write entity descriptions; the extraction reads them to
    # tell alpha from beta, as in a deposited file.
    document = structure.make_mmcif_document()
    rows: dict[str, list[str]] = {"id": [], "type": [], "pdbx_description": []}
    for entity in structure.entities:
        chain_name = entity.subchains[0][0]
        polymer = entity.entity_type == gemmi.EntityType.Polymer
        rows["id"].append(entity.name)
        rows["type"].append("polymer" if polymer else "non-polymer")
        rows["pdbx_description"].append(
            f"Tubulin {kinds[chain_name]} chain" if polymer else "guanine nucleotide"
        )
    document.sole_block().set_mmcif_category("_entity.", rows)
    document.write_file(str(path))
    return kinds


@pytest.mark.parametrize("plus_sign", [-1, 1])
def test_extracted_dimer_pairs_alpha_with_its_own_beta_and_points_plus_end_up(
    plus_sign, tmp_path
):
    """Offline guard on the dimer's polarity, whichever way the source runs.

    Both betas next to an alpha are one monomer rise away, so geometry along
    the axis cannot pick the partner; the beta capping alpha's nucleotide
    can. ``plus_sign=-1`` is 3JAL's case, plus end along the deposited -z,
    in which picking "the beta one rise further along +z" takes the
    previous dimer's beta and points the template at the minus end.
    """
    import gemmi

    from specter.specimen.filament import extract_mt_dimer

    kinds = _write_tubulin_patch(tmp_path / "SYNT.cif", plus_sign)
    model = gemmi.read_structure(extract_mt_dimer("SYNT", pdb_cache_dir=str(tmp_path)))[
        0
    ]

    def z_of(chain, residue_names) -> float:
        return float(
            np.mean([a.pos.z for r in chain if r.name in residue_names for a in r])
        )

    alpha, beta = model[0], model[1]
    assert (kinds[alpha.name], kinds[beta.name]) == ("alpha", "beta")
    # One dimer: in the patch, chains of a dimer are named consecutively.
    assert ord(beta.name) - ord(alpha.name) == 1

    z_alpha, z_beta = z_of(alpha, {"ALA"}), z_of(beta, {"ALA"})
    assert z_alpha == pytest.approx(-MONOMER_RISE / 2, abs=1.5)
    assert z_beta == pytest.approx(MONOMER_RISE / 2, abs=1.5)
    # alpha's nucleotide between the monomers, beta's at the +Z (plus) end.
    assert z_alpha < z_of(alpha, {"GTP"}) < z_beta
    assert z_of(beta, {"GDP"}) > z_beta


def test_a_cached_dimer_from_an_older_extraction_is_re_extracted(tmp_path):
    """A stale template must not survive a change to the extraction: the
    cached file carries its version, and an older one is rewritten."""
    from specter.specimen.filament import extract_mt_dimer
    from specter.specimen.filament._tubulin import _dimer_block_name

    _write_tubulin_patch(tmp_path / "SYNT.cif", plus_sign=-1)
    stale = tmp_path / "SYNT-mtdimer.cif"
    stale.write_text("data_SYNT_mtdimer\n")

    path = extract_mt_dimer("SYNT", pdb_cache_dir=str(tmp_path))
    assert path == str(stale)
    first_line = stale.read_text().splitlines()[0]
    assert first_line == f"data_{_dimer_block_name('SYNT')}"

    # And a current one is reused as-is.
    mtime = stale.stat().st_mtime_ns
    extract_mt_dimer("SYNT", pdb_cache_dir=str(tmp_path))
    assert stale.stat().st_mtime_ns == mtime


# --------------------------------------------------------------------------
# A free protofilament (PROTOFILAMENT_SPEC) stacks the microtubule's dimer
# --------------------------------------------------------------------------
def test_a_straight_free_protofilament_is_one_protofilament_of_a_straight_tube():
    """Every protofilament of a straight tube is the free protofilament,
    moved rigidly.

    `PROTOFILAMENT_SPEC` takes the tube's own dimer in the microtubule
    frame, steps it by `DIMER_REPEAT` along +Z with no twist, and puts the
    path through the dimer's origin. One rigid motion that carries the
    first free copy onto the first copy of a tube protofilament must then
    carry every free copy, position and rotation, onto that protofilament.
    """
    from specter.specimen.filament import MT_DIMER_SOURCE, place_filaments

    assert PROTOFILAMENT_SPEC.frame == "microtubule_dimer"
    assert PROTOFILAMENT_SPEC.code == MT_DIMER_SOURCE
    assert PROTOFILAMENT_SPEC.twist_deg == 0.0

    lattice = solve_tube_lattice(13)
    assert lattice.dimer_repeat == pytest.approx(PROTOFILAMENT_SPEC.step)
    n = 6
    axis = torch.stack(
        [torch.zeros(n), torch.zeros(n), torch.arange(float(n)) * lattice.dimer_repeat],
        dim=1,
    )
    tube = build_tube_instances("x.cif", 0, axis, lattice)
    spec = replace(PROTOFILAMENT_SPEC, n_monomers=n, flex_deg=0.0)
    free = place_filaments([spec], (64, 64, 64), 5.0, generator=seeded(0))

    for p in range(lattice.n_protofilaments):
        column = [inst for inst in tube if inst.protofilament_index == p]
        rotation = (
            column[0].rotation_matrix.double() @ free[0].rotation_matrix.double().T
        )
        shift = (
            column[0].position_xyz.double() - rotation @ free[0].position_xyz.double()
        )
        for f, t in zip(free, column, strict=True):
            moved = rotation @ f.position_xyz.double() + shift
            assert torch.allclose(moved, t.position_xyz.double(), atol=1e-3)
            assert torch.allclose(
                rotation @ f.rotation_matrix.double(),
                t.rotation_matrix.double(),
                atol=1e-5,
            )


def test_a_microtubule_dimer_species_renders_the_extracted_dimer_as_written(
    tmp_path,
):
    """``frame="microtubule_dimer"`` renders the dimer `extract_mt_dimer`
    writes, without re-orienting it, and places it by the file's origin.

    The template must be the extracted coordinates as written: a
    principal-axis alignment would choose its own roll about +Z and lose
    the radial face. The centre is where `PDB`'s re-centring put the
    file's origin, which a copy's path point is (two chains here, so every
    chain has to count).
    """
    import gemmi

    from specter.specimen.filament import FilamentSpec, extract_mt_dimer
    from specter.specimen.tomogram import TomogramSpecimenGenerator

    _write_tubulin_patch(tmp_path / "SYNT.cif", plus_sign=-1)
    voxel_size = 4.0
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=(32, 32, 32),
        voxel_size=voxel_size,
        protein_specs=[],
        filament_specs=[
            FilamentSpec(
                code="SYNT",
                step=DIMER_REPEAT,
                flex_deg=0.0,
                frame="microtubule_dimer",
            )
        ],
        pdb_cache_dir=str(tmp_path),
        parameterization="kirkland",
        progressbars=False,
    )
    template, centre = gen._filament_template("SYNT", voxel_size, as_given=False)

    path = extract_mt_dimer("SYNT", pdb_cache_dir=str(tmp_path))
    as_written = gen._build_species_template(
        gen._load_pdb(path), voxel_size, gen.device
    )
    assert torch.allclose(template, as_written)

    model = gemmi.read_structure(path)[0]
    assert len(model) == 2
    xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for c in model for r in c for a in r])
    np.testing.assert_allclose(centre.numpy(), xyz.mean(axis=0), atol=1e-3)


@pytest.mark.network
def test_straight_protofilament_superposes_on_the_3jal_protofilaments(tmp_path):
    """Two dimers placed by `PROTOFILAMENT_SPEC` on a straight path
    superpose on two consecutive dimers of each 3JAL protofilament.

    Measured: CA RMSD 0.14 A on each of the three protofilaments of the
    deposited patch (1722 CA). 3JAL's own dimer repeat is 81.76 A, 0.24 A
    short of `DIMER_REPEAT`; stepped at 81.76 A the RMSD is 0.085 A. For
    comparison, the previous preset (1TUB, aligned by its longest principal
    axis) gave 4.7 A, its stacking axis 7.5 deg off the protofilament.
    """
    import gemmi

    from specter.pdb import PDB
    from specter.specimen.filament import extract_mt_dimer, place_filaments
    from specter.specimen.filament._actin import _kabsch, _residue_ca
    from specter.specimen.filament._tubulin import (
        _N_SITE_CONTACT,
        _ca_centroid,
        _chain_entities,
        _fit_tube_axis,
        _nucleotide_xyz,
        _polymer_xyz,
    )

    cache = str(tmp_path)
    template = [
        _residue_ca(chain)
        for chain in gemmi.read_structure(
            extract_mt_dimer(PROTOFILAMENT_SPEC.code, pdb_cache_dir=cache)
        )[0]
    ]  # alpha, beta

    cif = PDB.fetch_pdb_file(
        PROTOFILAMENT_SPEC.code,
        ext="cif",
        pdb_cache_dir=cache,
        assembly=False,
        verbose=False,
    )
    st = gemmi.read_structure(cif)
    st.setup_entities()
    entities = _chain_entities(st, cif)
    centroids = {}
    for chain in st[0]:
        if "tubulin" in entities.get(chain.name, ""):
            centroid = _ca_centroid(chain)
            if centroid is not None:
                centroids[chain.name] = centroid
    direction, _, columns = _fit_tube_axis(centroids)
    assert len(columns) == 3

    for column in columns:
        # Each alpha with the beta capping its nucleotide, as the
        # extraction pairs them, ordered toward the plus end.
        betas = [c for c in column if "beta" in entities[c]]
        dimers = []
        for alpha in (c for c in column if "alpha" in entities[c]):
            site = _nucleotide_xyz(st[0][alpha])
            contact = {
                b: np.linalg.norm(
                    _polymer_xyz(st[0][b])[:, None] - site[None], axis=-1
                ).min()
                for b in betas
            }
            beta = min(contact, key=contact.__getitem__)
            if contact[beta] <= _N_SITE_CONTACT:
                dimers.append((alpha, beta))
        plus = direction * np.sign(
            (centroids[dimers[0][1]] - centroids[dimers[0][0]]) @ direction
        )
        dimers.sort(key=lambda d: float(centroids[d[0]] @ plus))
        assert len(dimers) == 2

        spec = replace(PROTOFILAMENT_SPEC, n_monomers=len(dimers), flex_deg=0.0)
        instances = place_filaments([spec], (200, 200, 200), 5.0, generator=seeded(3))
        placed, deposited = [], []
        for inst, names in zip(instances, dimers, strict=True):
            position = inst.position_xyz.double().numpy()
            rotation = inst.rotation_matrix.double().numpy()
            for monomer, name in zip(template, names, strict=True):
                ca = _residue_ca(st[0][name])
                keys = sorted(set(monomer) & set(ca))
                placed.append(
                    np.stack([position + rotation @ monomer[k] for k in keys])
                )
                deposited.append(np.stack([ca[k] for k in keys]))
        placed_xyz, deposited_xyz = np.concatenate(placed), np.concatenate(deposited)
        rotation, translation = _kabsch(placed_xyz, deposited_xyz)
        rmsd = np.sqrt(
            ((placed_xyz @ rotation.T + translation - deposited_xyz) ** 2).sum(1).mean()
        )
        assert len(placed_xyz) > 1600
        assert rmsd < 0.25


def test_one_instance_id_per_filament_not_per_monomer():
    """A filament is one object in the segmentation, like a microtubule.

    Microtubules already shared an id across their dimers, so a tube reads
    as one object rather than ~950 loose pieces. Actin did not: 20
    filaments appeared as 765 separate instances, and a picker evaluated
    against that ground truth was being asked to find monomers."""
    import torch

    from specter.specimen.filament import place_filaments
    from specter.specimen.tomogram import TomogramSpecimenGenerator

    rng = torch.Generator()
    rng.manual_seed(11)
    spec = replace(ACTIN_SPEC, n_copies=20, n_monomers=(20, 60))
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

    from specter.specimen.filament import place_filaments
    from specter.specimen.tomogram import TomogramSpecimenGenerator

    rng = torch.Generator()
    rng.manual_seed(0)
    specs = [
        replace(ACTIN_SPEC, n_copies=3, n_monomers=(5, 5)),
        replace(PROTOFILAMENT_SPEC, n_copies=3, n_monomers=(5, 5)),
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

    from specter.specimen.filament import place_filaments
    from specter.specimen.tomogram._filaments import _filament_runs

    rng = torch.Generator()
    rng.manual_seed(11)
    spec = replace(ACTIN_SPEC, n_copies=7, n_monomers=(20, 40))
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

    from specter.specimen.filament import place_filaments
    from specter.specimen.tomogram._filaments import _filament_runs

    rng = torch.Generator()
    rng.manual_seed(2)
    spec = replace(ACTIN_SPEC, n_copies=4, n_monomers=(1, 1))
    instances = place_filaments([spec], (300, 1200, 1200), 5.0, rng)
    runs = list(_filament_runs(instances))
    assert len(runs) == 4 and all(len(r) == 1 for r in runs)
