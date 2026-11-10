import warnings
from pathlib import Path

import pytest
import torch

from specter.specimen.membrane._generator import (
    MembraneGenerator,
    TransmembraneSpec,
)

# A small vesicle on a coarse grid: 80 A radius, mildly elongated and
# undulated, inside a 320 A box.
_SMALL_KWARGS = dict(
    target_shape=(40, 40, 40),
    voxel_size=8.0,
    radius=80.0,
    elongation=1.1,
    undulation=8.0,
    n_lipids_per_leaflet=6,
)

# A plain sphere, for the tests that compare against its radius.
_SPHERE_KWARGS = dict(
    voxel_size=10.0,
    radius=100.0,
    elongation=1.0,
    undulation=0.0,
    n_lipids_per_leaflet=6,
)


def test_generate_produces_correct_shape_with_membrane_density() -> None:
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    volume = gen.generate()

    assert volume.shape == _SMALL_KWARGS["target_shape"]
    assert torch.isfinite(volume).all()
    assert volume.max() > 0
    assert gen.field is not None
    assert gen.profile is not None
    assert gen.kind == "vesicle"


def test_an_unknown_kind_or_setting_is_refused() -> None:
    with pytest.raises(ValueError, match="kind"):
        MembraneGenerator(kind="nucleus")
    with pytest.raises(ValueError, match="no setting"):
        MembraneGenerator(kind="vesicle", length=900.0)


def test_same_seed_gives_the_same_draw_and_the_same_volume() -> None:
    """Ranges are drawn at construction, and a seed fixes both the draw and
    the geometry built from it."""
    kwargs = dict(
        voxel_size=10.0,
        radius=[80.0, 110.0],
        elongation=[1.0, 1.3],
        undulation=[0.0, 10.0],
        n_lipids_per_leaflet=6,
    )
    a = MembraneGenerator(seed=5, **kwargs)
    b = MembraneGenerator(seed=5, **kwargs)
    other = MembraneGenerator(seed=6, **kwargs)

    assert a.params == b.params
    assert a.params != other.params
    assert 80.0 <= a.params["radius"] <= 110.0
    assert a.target_shape == b.target_shape
    assert a.bounding_radius == b.bounding_radius
    assert torch.equal(a.generate(), b.generate())


def test_an_auto_sized_box_contains_the_membrane() -> None:
    gen = MembraneGenerator(seed=0, **_SPHERE_KWARGS)
    volume = gen.generate()

    assert gen.clipped_at_boundary is False
    # Nothing reaches the faces of the box.
    peak = float(volume.max())
    for face in (volume[0], volume[-1], volume[:, 0], volume[:, -1]):
        assert float(face.abs().max()) < 1e-3 * peak
    assert float(volume[:, :, 0].abs().max()) < 1e-3 * peak
    assert float(volume[:, :, -1].abs().max()) < 1e-3 * peak
    # The box is not larger than the bounding sphere needs.
    extent = min(gen.target_shape) * gen.voxel_size
    assert extent <= 2 * gen.bounding_radius + 4 * gen.voxel_size


def test_a_box_smaller_than_the_membrane_is_reported_as_clipped() -> None:
    gen = MembraneGenerator(seed=0, target_shape=(12, 12, 12), **_SPHERE_KWARGS)
    gen.generate()

    assert gen.clipped_at_boundary is True
    assert gen.field is not None and gen.field.clipped_at_boundary is True


def test_the_field_zero_level_set_lies_on_the_rendered_shell() -> None:
    """`field` is the signed distance to the bilayer's mid-surface: the
    rendered density sits where it is small, and nowhere else."""
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    volume = gen.generate()
    assert gen.field is not None and gen.profile is not None

    nz, ny, nx = volume.shape
    idx = torch.stack(
        torch.meshgrid(
            torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij"
        ),
        dim=-1,
    ).reshape(-1, 3)
    centre = torch.tensor([nz // 2, ny // 2, nx // 2])
    points_xyz = ((idx - centre).flip(-1) * gen.voxel_size).to(torch.float32)
    distance = gen.field.sample(points_xyz).reshape(volume.shape)

    bright = volume > 0.5 * volume.max()
    assert bool(bright.any())
    # Every bright voxel is within the bilayer's half-thickness (plus a
    # voxel) of the zero level set...
    half_thickness = gen.bilayer_thickness / 2
    assert float(distance[bright].abs().max()) < half_thickness + gen.voxel_size
    # ...and well away from it there is no density at all.
    far = distance.abs() > 60.0
    assert bool(far.any())
    assert float(volume[far].abs().max()) < 1e-3 * float(volume.max())
    # The mid-surface itself is inside the slab: the acyl core carries
    # potential, so no voxel on the zero level set is empty.
    on_surface = distance.abs() < 0.5 * gen.voxel_size
    assert bool(on_surface.any())
    assert float(volume[on_surface].min()) > 0.3 * float(volume.max())


def test_a_sphere_s_shell_sits_at_its_radius() -> None:
    gen = MembraneGenerator(seed=0, **_SPHERE_KWARGS)
    volume = gen.generate()

    nz, ny, nx = volume.shape
    z, y, x = torch.meshgrid(
        torch.arange(nz) - nz // 2,
        torch.arange(ny) - ny // 2,
        torch.arange(nx) - nx // 2,
        indexing="ij",
    )
    r = torch.sqrt((x**2 + y**2 + z**2).float()) * gen.voxel_size
    mean_radius = float((volume * r).sum() / volume.sum())
    # Regularising the surface rounds a sphere inwards by about an angstrom.
    assert mean_radius == pytest.approx(100.0, abs=4.0)
    assert float(volume[r < 50.0].abs().max()) < 1e-3 * float(volume.max())
    assert float(volume[r > 150.0].abs().max()) < 1e-3 * float(volume.max())


def test_place_transmembrane_before_generate_raises():
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    with pytest.raises(RuntimeError):
        gen.place_transmembrane()


def test_insert_blend_replaces_occupied_region_and_preserves_untouched_background():
    # Regression test for the additive-insertion fix: a transmembrane
    # protein displaces lipid where it sits rather than coexisting with it,
    # so plain addition (the old behaviour) double-counts density in the
    # overlap. _insert_blend should instead fully REPLACE membrane density
    # where the protein template is occupied, and leave it EXACTLY
    # untouched elsewhere -- the latter also catches a real bug an earlier
    # sigmoid-based (rather than smoothstep) blend had: its tail never
    # truly reaches 0, so a template's zero background picked up a small
    # but nonzero weight too, silently attenuating membrane density across
    # the whole inserted bounding box, not just near the protein's own mass.
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    gen.generate()

    # Pick an actual on-membrane voxel (nonzero density), not an arbitrary
    # point that might land in empty space, so replace-vs-add is a
    # meaningfully different outcome.
    peak_idx = (gen.volume == gen.volume.max()).nonzero()[0]
    center_zyx = peak_idx.clone()
    baseline_center = gen.volume[tuple(peak_idx.tolist())].clone()
    assert baseline_center.item() > 0  # sanity: real density here, not empty space
    offset_idx = peak_idx + torch.tensor([2, 0, 0])
    baseline_offset = gen.volume[tuple(offset_idx.tolist())].clone()

    template = torch.zeros(5, 5, 5)
    template[2, 2, 2] = 1000.0  # far above any plausible bilayer peak amplitude
    gen._insert_blend(template, center_zyx)

    # occupied voxel: replaced by the protein value, not baseline + protein
    # (an absolute, not relative, tolerance -- the baseline density here is
    # tiny relative to 1000.0, so a relative check can't tell replace and
    # add apart).
    assert gen.volume[tuple(peak_idx.tolist())].item() == pytest.approx(
        1000.0, abs=1e-3
    )
    assert gen.volume[tuple(peak_idx.tolist())].item() != pytest.approx(
        (baseline_center + 1000.0).item(), abs=1e-3
    )
    # background voxel within the inserted bounding box but zero protein
    # density there: left exactly as it was.
    assert gen.volume[tuple(offset_idx.tolist())].item() == pytest.approx(
        baseline_offset.item()
    )


def test_place_transmembrane_with_prebuilt_template_inserts_and_records_placement():
    template = torch.zeros(9, 9, 9)
    template[4, 4, 4] = 50.0

    gen = MembraneGenerator(
        transmembrane_specs=[TransmembraneSpec("synthetic", template=template)],
        seed=0,
        **_SMALL_KWARGS,
    )
    baseline = gen.generate().clone()
    baseline_sum = baseline.sum()

    placements = gen.place_transmembrane(min_spacing_angstrom=20.0)

    assert len(placements) >= 1
    for placement in placements:
        assert placement.species_id == "synthetic"
        R = placement.rotation_matrix
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-3)
        assert torch.isclose(torch.det(R), torch.tensor(1.0), atol=1e-3)

    assert gen.volume.sum() > baseline_sum


def test_place_transmembrane_species_selection_seed_reproducible():
    template_a = torch.zeros(7, 7, 7)
    template_a[3, 3, 3] = 10.0
    template_b = torch.zeros(7, 7, 7)
    template_b[3, 3, 3] = 20.0

    specs = [
        TransmembraneSpec("species_a", frequency=1, template=template_a),
        TransmembraneSpec("species_b", frequency=3, template=template_b),
    ]

    gen_a = MembraneGenerator(transmembrane_specs=specs, seed=7, **_SMALL_KWARGS)
    gen_a.generate()
    placements_a = gen_a.place_transmembrane(min_spacing_angstrom=15.0)

    gen_b = MembraneGenerator(transmembrane_specs=specs, seed=7, **_SMALL_KWARGS)
    gen_b.generate()
    placements_b = gen_b.place_transmembrane(min_spacing_angstrom=15.0)

    assert [p.species_id for p in placements_a] == [p.species_id for p in placements_b]
    for pa, pb in zip(placements_a, placements_b):
        assert torch.allclose(pa.center_xyz, pb.center_xyz)


def test_build_template_uses_analytic_method_matching_membrane_profile():
    # Regression test: PDB-backed templates were rendered with method="3d"
    # while the bilayer profile is rendered with method="analytic". Both
    # integrate to the same total potential (confirmed empirically, ratio
    # ~1.001 across a 10x span of voxel sizes on a real structure), but
    # "3d" spreads that total over a 25-45% lower peak for identical atoms
    # -- an unintended cross-method contrast mismatch between membrane and
    # protein, not a deliberate physical difference. Verify the template
    # builder now actually calls PotentialBuilder with method="analytic".
    #
    # The reference below also mirrors the bonded-species path: under
    # parameterization="shtyrov" the shipped builder types atoms via
    # compute_atom_species, and comparing against an untyped reference would
    # measure that (~4% relative RMS) rather than the method it is testing.
    from specter.pdb import PDB
    from specter.potential import PotentialBuilder
    from specter.specimen.membrane._placement import (
        align_principal_axis_to_z,
        align_transmembrane_depth,
    )
    from specter.specimen.packing import estimate_protein_box_size

    gen = MembraneGenerator(
        pdb_cache_dir=str(Path(__file__).parent / "test_data"),
        **_SMALL_KWARGS,
    )
    spec = TransmembraneSpec("1C3W", parameterization="shtyrov")
    template = gen._build_template(spec)

    pdb = PDB(
        "1C3W",
        pdb_cache_dir=str(Path(__file__).parent / "test_data"),
        verbose=False,
        compute_atom_species=True,
    )
    coordinates = align_principal_axis_to_z(pdb.coordinates)
    coordinates = align_transmembrane_depth(coordinates, None)
    n = estimate_protein_box_size(pdb.max_diameter, gen.voxel_size)
    builder = PotentialBuilder(
        n_xyz=n,
        dx=gen.voxel_size,
        atomic_numbers=pdb.atomic_numbers,
        progressbars=False,
        parameterization="shtyrov",
        atom_species=pdb.atom_species,
    )
    expected = builder.forward(coordinates, method="analytic")

    assert torch.allclose(template.cpu(), expected, atol=1e-3)


def test_transmembrane_labels_mark_exactly_the_displaced_lipid():
    """`transmembrane_labels` is the same decision the density blend makes.

    `_insert_blend` replaces lipid with protein wherever the template's own
    density exceeds `transmembrane_occupancy_fraction * psi.max()`, which is
    where its smoothstep weight reaches 0.5. The label has to use that same
    boundary, not a second cutoff of its own, or the ground truth and the
    density disagree about where the protein is.
    """
    fixture = Path(__file__).parent / "test_data" / "1mbo.cif"
    if not fixture.exists():
        pytest.skip("bundled PDB fixture missing")

    gen = MembraneGenerator(
        transmembrane_specs=[TransmembraneSpec(pdb_source=str(fixture), frequency=3)],
        seed=0,
        **_SMALL_KWARGS,
    )
    gen.generate()
    placements = gen.place_transmembrane(min_spacing_angstrom=20.0)

    assert len(placements) > 0
    labels = gen.transmembrane_labels
    assert labels is not None
    assert labels.shape == gen.volume.shape
    assert labels.dtype == torch.int16

    ids = set(torch.unique(labels).tolist()) - {0}
    assert ids, "no voxel was labelled for any placement"
    # Local, 1-based, and never more than the number of placements.
    assert ids <= set(range(1, len(placements) + 1))

    # Labelled voxels carry real density -- the label tracks the protein,
    # not the template's zero-padded bounding box.
    assert float(gen.volume[labels > 0].min()) > 0.0


def test_transmembrane_labels_stay_none_without_specs():
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    gen.generate()
    assert gen.place_transmembrane() == []
    assert gen.transmembrane_labels is None


def test_place_transmembrane_warns_on_partial_or_zero_placement():
    """Pre-existing gap: place_transmembrane silently returned fewer (or
    zero) placements than requested with no warning at all. Fixed."""
    pdb_path = Path(__file__).parent / "test_data" / "1mbo.cif"
    if not pdb_path.exists():
        pytest.skip("bundled PDB fixture missing")
    gen = MembraneGenerator(
        target_shape=(32, 32, 32),
        voxel_size=8.0,
        radius=60.0,
        elongation=1.0,
        undulation=0.0,
        n_lipids_per_leaflet=6,
        # frequency=40: a 60 A vesicle cannot hold that many sites 60 A
        # apart (about a dozen at best), so the shortfall is geometric.
        transmembrane_specs=[TransmembraneSpec(pdb_source=str(pdb_path), frequency=40)],
        seed=0,
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")  # let generate()'s own warning through quietly
        gen.generate()
    with pytest.warns(UserWarning, match="requested transmembrane sites were found"):
        placements = gen.place_transmembrane(min_spacing_angstrom=60.0)
    assert len(placements) < 40


def test_collision_radius_and_generator_sizing_agree() -> None:
    """
    The tomogram placer and the generator's own auto-sizing must measure the
    same organelle the same way.
    """
    from specter.specimen.tomogram._helpers import _instance_bounding_radius

    for kind, settings in (
        ("vesicle", {"radius": 150.0, "elongation": 1.3}),
        ("tubules", {"radius": 120.0, "length": 900.0}),
    ):
        generator = MembraneGenerator(voxel_size=10.0, kind=kind, seed=0, **settings)
        assert _instance_bounding_radius(generator) == generator.bounding_radius
        # The auto-sized box holds the bounding sphere's reach on its
        # shortest axis no more than loosely: both come from one plan.
        assert generator.bounding_radius > settings["radius"]


def test_render_transmembrane_template_honours_use_deposited_bfactors():
    """
    The membrane path builds its own `PotentialBuilder`, so it needs its own
    check that the flag arrives -- see the tomogram generator's counterpart.
    """
    from specter.specimen.membrane._generator import render_transmembrane_template

    pdb_path = Path(__file__).parent / "test_data" / "1mbo.cif"
    if not pdb_path.exists():
        pytest.skip("bundled PDB fixture missing")

    spec = TransmembraneSpec(str(pdb_path))
    kwargs = dict(
        spec=spec,
        voxel_size=2.0,
        pdb_cache_dir=str(pdb_path.parent),
        device="cpu",
    )
    static = render_transmembrane_template(**kwargs)
    damped = render_transmembrane_template(**kwargs, use_deposited_bfactors=True)

    # A B-factor redistributes potential without removing any (exp(-B k^2/4)
    # is 1 at k=0) while strictly reducing its power at every other frequency.
    # Power rather than the peak: a peak sits in one voxel, so it moves with
    # sub-voxel alignment (0.92 here against 0.90 for the same structure
    # unrotated in the tomogram path), while the power ratio is 0.777 in both.
    assert damped.sum().item() == pytest.approx(static.sum().item(), rel=5e-3)
    assert damped.pow(2).sum() < 0.9 * static.pow(2).sum()


def test_a_transmembrane_template_turns_by_its_recorded_rotation():
    """The template's +z lands along the membrane normal, ``R @ z``.

    `orientation_for_normal` builds R so that R maps the template's +z (its
    membrane axis) onto the outward normal, and `rotate_volume` turns
    density by the transpose of the matrix it is given. Until 2026-10-16 R
    itself was handed over, so a protein's axis landed on ``R^T @ z``:
    pointing inward at some sites and tipped tens of degrees off the normal
    at most. A bright voxel three voxels up the template's +z must sit along
    ``R @ z`` from the placed centre.

    One site per membrane, over several seeds: with several sites 15 A
    apart, a neighbour's tip falls inside the search window and is often
    the voxel the argmax finds (7 of 12 sites on seed 0 read another
    site's tip).
    """
    template = torch.zeros(9, 9, 9)
    template[7, 4, 4] = 1000.0  # (z, y, x): +3 voxels along the template's +z
    z_axis = torch.tensor([0.0, 0.0, 1.0])
    checked = 0
    for seed in range(6):
        gen = MembraneGenerator(
            transmembrane_specs=[
                TransmembraneSpec("tip", template=template, frequency=1)
            ],
            seed=seed,
            **_SMALL_KWARGS,
        )
        gen.generate()
        (placement,) = gen.place_transmembrane()
        R = placement.rotation_matrix.float()
        if abs(float((R @ z_axis) @ (R.T @ z_axis))) > 0.8:
            continue  # this R and its transpose agree on z: no information
        centre = gen._physical_to_voxel_index(placement.center_xyz)
        lo = (centre - 6).clamp_min(0)
        window = gen.volume[
            lo[0] : centre[0] + 7, lo[1] : centre[1] + 7, lo[2] : centre[2] + 7
        ]
        peak = torch.stack(torch.unravel_index(window.argmax(), window.shape)) + lo
        offset_zyx = (peak - centre).float()
        offset_xyz = offset_zyx.flip(0)
        direction = offset_xyz / offset_xyz.norm()
        assert float(direction @ (R @ z_axis)) > 0.8
        checked += 1
    assert checked
