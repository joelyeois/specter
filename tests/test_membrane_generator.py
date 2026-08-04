import warnings

import pytest
import torch

from specter.specimen.membrane._generator import (
    MembraneGenerator,
    TransmembraneSpec,
)

_SMALL_KWARGS = dict(
    target_shape_zyx=(32, 32, 32),
    v_size=6.0,
    n_sources=3,
    radius_range_a=(20.0, 30.0),
    spread_a=10.0,
    noise_amplitude_a=0.0,
    curvature_iterations=5,
    n_lipids_per_leaflet=6,
)


def test_generate_produces_correct_shape_with_membrane_density():
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    volume = gen.generate()

    assert volume.shape == _SMALL_KWARGS["target_shape_zyx"]
    assert torch.isfinite(volume).all()
    assert volume.max() > 0
    assert gen.field is not None
    assert gen.profile is not None


def test_generate_is_seed_reproducible():
    gen_a = MembraneGenerator(seed=3, **_SMALL_KWARGS)
    gen_b = MembraneGenerator(seed=3, **_SMALL_KWARGS)
    volume_a = gen_a.generate()
    volume_b = gen_b.generate()
    assert torch.equal(volume_a, volume_b)


def test_place_transmembrane_before_generate_raises():
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    with pytest.raises(RuntimeError):
        gen.place_transmembrane()


def test_place_transmembrane_with_no_specs_returns_empty_list():
    gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
    gen.generate()
    assert gen.place_transmembrane() == []


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

    placements = gen.place_transmembrane(min_spacing_a=20.0)

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
    placements_a = gen_a.place_transmembrane(min_spacing_a=15.0)

    gen_b = MembraneGenerator(transmembrane_specs=specs, seed=7, **_SMALL_KWARGS)
    gen_b.generate()
    placements_b = gen_b.place_transmembrane(min_spacing_a=15.0)

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
    from specter.pdb import PDB
    from specter.potential import PotentialBuilder
    from specter.specimen.membrane._placement import (
        align_principal_axis_to_z,
        align_transmembrane_depth,
    )
    from specter.specimen.packing import estimate_protein_box_size

    gen = MembraneGenerator(
        pdb_cache_dir="pdb-data/",
        **_SMALL_KWARGS,
    )
    spec = TransmembraneSpec("1C3W", parameterization="shtyrov")
    template = gen._build_template(spec)

    pdb = PDB("1C3W", savefolder="pdb-data/", verbose=False)
    coordinates = align_principal_axis_to_z(pdb.coordinates)
    coordinates = align_transmembrane_depth(coordinates, None)
    n = estimate_protein_box_size(pdb.max_diameter, gen.v_size)
    builder = PotentialBuilder(
        n_xyz=n,
        dx=gen.v_size,
        atomic_numbers=pdb.atomic_numbers,
        progressbars=False,
        parameterization="shtyrov",
    )
    expected = builder.forward(coordinates, method="analytic")

    assert torch.allclose(template.cpu(), expected, atol=1e-3)


def test_max_field_voxels_coarsens_and_warns_instead_of_exploding_memory():
    """Regression test for a real production-scale finding: the naive
    field spacing heuristic, applied uniformly to a large target_shape_zyx,
    produced a ~1.1 billion voxel working grid (50+ GB resident across the
    several such arrays field generation allocates) for a real
    (200, 600, 600)-voxel/10A tomogram. A tiny max_field_voxels here forces
    the same coarsening path at a scale cheap enough to test quickly."""
    gen = MembraneGenerator(
        target_shape_zyx=(32, 32, 32),
        v_size=6.0,
        n_sources=3,
        radius_range_a=(20.0, 30.0),
        spread_a=10.0,
        noise_amplitude_a=0.0,
        curvature_iterations=5,
        n_lipids_per_leaflet=6,
        max_field_voxels=1000,
        seed=0,
    )
    with pytest.warns(UserWarning, match="coarsened field_spacing_a"):
        volume = gen.generate()

    assert volume.shape == (32, 32, 32)
    assert torch.isfinite(volume).all()
    assert gen.field is not None
    n_field_voxels = (
        gen.field.phi.shape[0] * gen.field.phi.shape[1] * gen.field.phi.shape[2]
    )
    assert n_field_voxels <= 1000 * 1.5  # ceil() rounding can push slightly over


def test_max_field_voxels_default_does_not_warn_at_small_scale():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        gen = MembraneGenerator(seed=0, **_SMALL_KWARGS)
        gen.generate()  # would raise if the coarsening warning fired
