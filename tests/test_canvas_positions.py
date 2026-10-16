"""
Every placed thing renders at the position recorded for it.

The canvas convention is voxel ``k`` of an axis of ``n`` at
``(k - n // 2) * voxel_size``, box-centred and in Angstrom; it is what
`insert_particles_into_micrograph` and `clip_insert_bounds` stamp with. These
tests pin the paths that used to disagree with it, each by half a voxel on
some axis: gold beads and transmembrane proteins inserted at the nearest voxel,
membrane grids and the ice profile centred on ``n / 2`` or ``(n - 1) / 2``, the
sphere packer's walls at ``+-box / 2``, and coarse-grid footprints reaching
past the canvas.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from specter.crowding import CrowdWithDuplicates
from specter.ice import IceProfile
from specter.rotations._volume import build_affine_matrix, rotate_volume
from specter.specimen._grid import BeadGenerator
from specter.specimen.membrane import MembraneGenerator, TransmembraneSpec
from specter.specimen.membrane._field_spherical_harmonics import (
    generate_membrane_field_spherical_harmonics,
)
from specter.specimen.packing import (
    build_species_mask,
    pack_hard_spheres_3d,
    pack_shapes_3d,
)
from specter.specimen.tomogram import (
    MembraneInstance,
    TomogramBeadSpec,
    TomogramSpecimenGenerator,
)
from specter.specimen.tomogram._helpers import _canvas_box_center
from conftest import blob


def _centroid_zyx(volume: torch.Tensor, centre_zyx: torch.Tensor, half: int):
    """Density-weighted centroid, in voxel indices, of a cube about a voxel."""
    lo = (centre_zyx - half).clamp_min(0)
    hi = centre_zyx + half + 1
    window = volume[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].double()
    grids = torch.meshgrid(*[torch.arange(s) for s in window.shape], indexing="ij")
    mass = window.sum()
    return torch.stack([(g * window).sum() / mass for g in grids]) + lo


def _canvas_index_zyx(position_xyz: torch.Tensor, shape_zyx, voxel: float):
    """Continuous canvas index (z, y, x) of a position (x, y, z)."""
    centre = torch.tensor([s // 2 for s in shape_zyx], dtype=torch.float64)
    return position_xyz.double().flip(-1) / voxel + centre


def _gaussian_template(n: int = 9, sigma: float = 1.0, peak: float = 1e5):
    """A blob centred exactly on voxel n // 2, bright enough that the bilayer
    around it shifts a centroid by under 0.01 voxel."""
    idx = torch.arange(n, dtype=torch.float32) - n // 2
    zz, yy, xx = torch.meshgrid(idx, idx, idx, indexing="ij")
    return peak * torch.exp(-(zz**2 + yy**2 + xx**2) / (2 * sigma**2))


# ----------------------------------------------------------------------
# Gold beads
# ----------------------------------------------------------------------


def test_a_bead_offset_moves_its_density_by_exactly_that_offset():
    """`offset_xyz` displaces the atoms before splatting, which conserves the
    first moment, so the density centroid moves by the offset to float
    accuracy; nothing random is redrawn."""
    voxel = 4.0
    offset = torch.tensor([0.3, -0.45, 0.2]) * voxel  # (x, y, z)
    a = BeadGenerator(voxel_size=voxel).generate(
        radius=20.0, generator=torch.Generator().manual_seed(3)
    )
    b = BeadGenerator(voxel_size=voxel).generate(
        radius=20.0, generator=torch.Generator().manual_seed(3), offset_xyz=offset
    )
    n = a.density.shape[0]
    c = torch.tensor([n // 2] * 3)
    shift = _centroid_zyx(b.density, c, n) - _centroid_zyx(a.density, c, n)
    assert torch.allclose(shift.flip(0), offset.double() / voxel, atol=2e-3), shift


def test_tomogram_beads_render_at_their_picks():
    """Each bead's density centroid is its recorded position. Inserted at the
    nearest voxel, it was up to half a voxel off per axis (0.49 voxel worst
    axis over this seed's beads)."""
    shape = (48, 48, 48)
    voxel = 8.0
    half = 5  # window half-width, voxels: the 20 A bead plus its kernel tail
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=shape,
        voxel_size=voxel,
        protein_specs=[],
        bead_specs=[TomogramBeadSpec(radius=20.0, count=4)],
        bead_roughness=0.0,
        seed=2,
        progressbars=False,
    )
    volume = gen.generate()
    pos = torch.stack([b.position_xyz for b in gen.bead_instances])
    worst, checked = 0.0, 0
    for i, bead in enumerate(gen.bead_instances):
        others = torch.cat([pos[:i], pos[i + 1 :]])
        # Skip a bead whose window would hold a neighbour's density.
        if (
            len(others)
            and float((others - bead.position_xyz).abs().max(dim=1).values.min())
            < (2 * half + 1) * voxel
        ):
            continue
        target = _canvas_index_zyx(bead.position_xyz, shape, voxel)
        got = _centroid_zyx(volume, target.round().long(), half)
        worst = max(worst, float((got - target).abs().max()))
        checked += 1
    assert checked >= 2
    assert worst < 0.05, f"bead density is {worst:.3f} voxel from its pick"


# ----------------------------------------------------------------------
# Membranes and transmembrane proteins
# ----------------------------------------------------------------------


def test_a_spherical_membrane_field_is_centred_on_voxel_n_over_2():
    """On an odd grid the field's origin must be a voxel. Built about
    ``-n / 2`` it was half a voxel off, and the field of a sphere centred on
    the origin was not mirror-symmetric about any voxel."""
    shape = (31, 33, 35)
    field = generate_membrane_field_spherical_harmonics(
        shape_zyx=shape,
        spacing_angstrom=4.0,
        sh_axes=(40.0, 40.0, 40.0),
        sh_amplitude=0.0,
        seed=0,
        device="cpu",
    )
    for axis in range(3):
        assert torch.allclose(field.phi, field.phi.flip(axis), atol=1e-4), axis


def test_a_membrane_volume_is_centred_on_voxel_n_over_2():
    """The rendered membrane, not only its field: a spherical vesicle on an
    odd grid is mirror-symmetric about voxel ``n // 2``, the voxel
    `clip_insert_bounds` puts on the instance's position."""
    gen = MembraneGenerator(
        target_shape=(33, 33, 33),
        voxel_size=6.0,
        sh_axes=(50.0, 50.0, 50.0),
        sh_amplitude=0.0,
        seed=0,
    )
    volume = gen.generate()
    scale = float(volume.abs().max())
    for axis in range(3):
        assert torch.allclose(volume, volume.flip(axis), atol=1e-4 * scale), axis


def test_transmembrane_density_lands_on_its_site():
    """A transmembrane protein's density centroid is its recorded site, in the
    grid's own canvas convention. Inserted at the nearest voxel of a grid
    centred on ``n / 2`` it was up to a voxel off per axis on an odd grid."""
    voxel = 6.0
    gen = MembraneGenerator(
        target_shape=(33, 33, 33),
        voxel_size=voxel,
        sh_axes=(50.0, 50.0, 50.0),
        sh_amplitude=0.15,
        transmembrane_specs=[
            TransmembraneSpec("blob", template=_gaussian_template(), frequency=6)
        ],
        seed=0,
    )
    gen.generate()
    placements = gen.place_transmembrane(min_spacing_angstrom=60.0)
    assert len(placements) >= 3
    assert gen.volume is not None
    worst = 0.0
    for p in placements:
        target = _canvas_index_zyx(p.center_xyz, gen.volume.shape, voxel)
        got = _centroid_zyx(gen.volume, target.round().long(), 4)
        worst = max(worst, float((got - target).abs().max()))
    assert worst < 0.03, f"protein density is {worst:.3f} voxel from its site"


def test_tomogram_membranes_snap_and_their_proteins_land_on_their_picks():
    """A membrane instance is merged a whole voxel at a time, so its recorded
    position is the voxel it lands on, and a transmembrane pick (offset by
    that position) is where the protein's density is in the canvas."""
    voxel = 8.0
    shape = (40, 40, 40)
    mgen = MembraneGenerator(
        target_shape=(25, 25, 25),
        voxel_size=voxel,
        sh_axes=(60.0, 60.0, 60.0),
        sh_amplitude=0.15,
        transmembrane_specs=[
            TransmembraneSpec("blob", template=_gaussian_template(), frequency=6)
        ],
        seed=1,
    )
    instance = MembraneInstance(generator=mgen)
    gen = TomogramSpecimenGenerator(
        membrane_instances=[instance],
        target_shape=shape,
        voxel_size=voxel,
        protein_specs=[],
        min_transmembrane_spacing=90.0,  # windows clear of neighbours
        seed=0,
        progressbars=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volume = gen.generate()
    assert instance.position_xyz is not None
    pos = torch.tensor(instance.position_xyz)
    assert torch.allclose(pos, torch.round(pos / voxel) * voxel, atol=1e-4)
    assert len(gen.transmembrane_placements) >= 3
    worst = 0.0
    for p in gen.transmembrane_placements:
        target = _canvas_index_zyx(p.center_xyz, shape, voxel)
        got = _centroid_zyx(volume, target.round().long(), 4)
        worst = max(worst, float((got - target).abs().max()))
    assert worst < 0.03, f"protein density is {worst:.3f} voxel from its pick"


# ----------------------------------------------------------------------
# Ice profile
# ----------------------------------------------------------------------


@pytest.mark.parametrize("nxy", [64, 65])
def test_ice_profile_grid_puts_the_origin_on_pixel_n_over_2(nxy):
    """A meniscus centred on the origin is thinnest exactly at pixel
    ``nxy // 2``; on the ``(n - 1) / 2`` grid an even field had no pixel
    there."""
    prof = IceProfile(mode="meniscus", mean_thickness=300.0, hole_radius=200.0)
    t = prof.thickness(nxy, 2.0)
    assert float(t[nxy // 2, nxy // 2]) == pytest.approx(300.0)
    assert float(t.min()) == pytest.approx(300.0)


@pytest.mark.parametrize("nxy", [32, 33])
def test_surfaces_at_reads_the_pixel_a_particle_renders_into(nxy):
    """A particle at pixel ``k``'s position, ``(k - n // 2) * px``, reads that
    pixel's surfaces."""
    prof = IceProfile(
        mode="meniscus",
        mean_thickness=200.0,
        hole_radius=60.0,
        rim_thickness=900.0,
        hole_offset=(7.0, -5.0),
        tilt=0.2,
    )
    px = 2.0
    z_bot_f, z_top_f = prof.surfaces(nxy, px)
    k = torch.arange(nxy)
    jj, ii = torch.meshgrid(k, k, indexing="ij")
    xy = torch.stack([ii.flatten(), jj.flatten()], 1).float()
    xy = (xy - nxy // 2) * px
    z_bot, z_top = prof.surfaces_at(xy, nxy, px)
    assert torch.equal(z_bot, z_bot_f.flatten())
    assert torch.equal(z_top, z_top_f.flatten())


def test_ice_window_is_centred_on_slice_nz_over_2():
    """A level slab is centred on ``z = 0``, which is slice ``nz // 2``."""
    prof = IceProfile(mode="flat", mean_thickness=60.0, softness=2.0)
    nz = 40
    w = prof.window(nz, 4, 2.0)[:, 0, 0]
    c = nz // 2
    assert torch.allclose(w[c - 19 : c], w[c + 1 : c + 20].flip(0), atol=1e-6)


def test_shape_crowding_keeps_copies_inside_the_ice():
    """The shape backend seeds everything outside the ice as occupied, on the
    canvas grid; every rendered footprint voxel of every copy is then inside
    its column's slab. Seeded on the ``(n - 1) / 2`` grid, the slab was half a
    voxel low, and copies stuck out of the bottom surface."""
    dx = 4.0
    nz, nxy = 48, 48
    prof = IceProfile(mode="flat", mean_thickness=100.0, softness=0.0)
    atoms = blob(300, radius=10.0)
    crowd = CrowdWithDuplicates(
        torch.zeros(8, 8, 8),
        dx=dx,
        min_distance=20.0,
        nxy_out=nxy,
        nz_out=nz,
        ice_profile=prof,
        particle_radius=10.0,
        packing_backend="shape",
        atom_coordinates=atoms,
        packing_max_retries=300,
        packing_stall_patience=3000,
        packing_seed=0,
        progressbars=False,
    )
    torch.manual_seed(0)
    crowd.generate_coordinates()
    assert len(crowd.coords) > 5
    mask = build_species_mask(atoms, dx, gap=0.0).float()
    z_axis = (torch.arange(nz) - nz // 2) * dx
    lowest, highest = float("inf"), -float("inf")
    for i in range(len(crowd.coords)):
        theta = build_affine_matrix(crowd._shape_rotations[i : i + 1])
        copy = rotate_volume(mask, theta, padding_mode="zeros")[0] > 0.5
        zs = torch.nonzero(copy.any(dim=(1, 2)))[:, 0] - copy.shape[0] // 2
        z_pos = zs * dx + float(crowd.coords[i, 2])
        lowest = min(lowest, float(z_pos.min()))
        highest = max(highest, float(z_pos.max()))
    assert z_axis.min() < -50.0 and z_axis.max() > 50.0
    assert lowest >= -50.0 and highest <= 50.0, (lowest, highest)


# ----------------------------------------------------------------------
# Sphere packer walls
# ----------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(20, 24, 24), (21, 25, 23)])
def test_sphere_packer_walls_are_the_canvas_faces(shape):
    """With the canvas's own centre, every sphere fits between the canvas's
    faces, ``-(n // 2 + 1/2)`` and ``n - n // 2 - 1/2`` voxels. Walls at
    ``+-box / 2`` let spheres cross the upper face of an even axis by half a
    voxel."""
    voxel = 4.0
    box = tuple(float(n * voxel) for n in shape)
    radii = torch.full((400,), 6.0)
    coords, idx = pack_hard_spheres_3d(
        radii,
        box,
        seed=0,
        box_center=_canvas_box_center(shape, voxel),
        max_passes=400,
        stall_patience=50,
    )
    n_xyz = torch.tensor(shape[::-1], dtype=torch.float32)
    lo = -(n_xyz // 2 + 0.5) * voxel
    hi = (n_xyz - n_xyz // 2 - 0.5) * voxel
    r = radii[idx][:, None]
    assert bool(((coords - r) >= lo - 1e-4).all())
    assert bool(((coords + r) <= hi + 1e-4).all())
    # dense enough that the faces are actually reached
    assert float((hi - (coords + r)).min()) < 0.5 * voxel
    assert float(((coords - r) - lo).min()) < 0.5 * voxel


def test_tomogram_beads_stay_inside_an_even_canvas():
    """Beads, placed by the sphere packer with walls on the canvas faces, fit
    inside the canvas on an even grid."""
    shape = (24, 24, 24)
    voxel = 8.0
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=shape,
        voxel_size=voxel,
        protein_specs=[],
        bead_specs=[TomogramBeadSpec(radius=12.0, count=150)],
        bead_roughness=0.0,
        seed=0,
        progressbars=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen.generate()
    assert len(gen.bead_instances) > 20
    pos = torch.stack([b.position_xyz for b in gen.bead_instances])
    r = torch.tensor([b.radius for b in gen.bead_instances])[:, None]
    n_xyz = torch.tensor(shape[::-1], dtype=torch.float32)
    lo = -(n_xyz // 2 + 0.5) * voxel
    hi = (n_xyz - n_xyz // 2 - 0.5) * voxel
    assert bool(((pos - r) >= lo - 1e-4).all())
    assert bool(((pos + r) <= hi + 1e-4).all())


# ----------------------------------------------------------------------
# Coarse packing grids that overhang the canvas
# ----------------------------------------------------------------------


@pytest.mark.parametrize("canvas_shape,factor", [((33, 39, 41), 2), ((31, 34, 35), 3)])
def test_coarse_footprints_stay_inside_an_overhung_canvas(canvas_shape, factor):
    """With no axis clippable, no rendered copy loses a voxel to the canvas
    edge, although the coarse grid's last packing voxel covers render voxels
    past it."""
    from specter.crowding import fold_subvoxel_offsets, insert_particles_into_micrograph

    voxel = 2.0
    grid = tuple(-(-n // factor) for n in canvas_shape)
    mask = build_species_mask(blob(200, radius=6.0), voxel, gap=0.0)
    torch.manual_seed(0)
    coords, rots, accepted, _ = pack_shapes_3d(
        [mask],
        torch.zeros(400, dtype=torch.long),
        grid,
        voxel * factor,
        pool_factor=factor,
        canvas_shape=canvas_shape,
        seed=1,
        n_orientations=16,
        max_retries=200,
        stall_patience=4000,
    )
    assert accepted.numel() > 20
    lost = 0
    for i in range(len(coords)):
        theta = build_affine_matrix(rots[i].unsqueeze(0))
        theta, whole = fold_subvoxel_offsets(
            theta, coords[i : i + 1], voxel, mask.shape
        )
        rotated = rotate_volume(mask.float(), theta, padding_mode="zeros")
        copy = insert_particles_into_micrograph(
            rotated, whole, pixel_size=voxel, micro_shape=canvas_shape
        )
        lost += int((rotated > 0.5).sum()) - int((copy > 0.5).sum())
    assert lost == 0, f"{lost} footprint voxels fell past the canvas"
