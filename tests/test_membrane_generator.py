import warnings
from pathlib import Path

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


_ALPHA_SHAPE_KWARGS = dict(
    target_shape_zyx=(80, 80, 80),
    v_size=4.0,
    shape_backend="alpha_shape",
    blob_size_a=40.0,
    blob_roughness=0.4,
    n_lipids_per_leaflet=6,
)


def test_alpha_shape_backend_produces_correct_shape_with_membrane_density():
    """shape_backend="alpha_shape" (CTS-derived) must be a drop-in for
    "metaball" through the rest of the pipeline -- same output contract,
    same calibrated BilayerProfile underneath, only the shape geometry
    differs."""
    gen = MembraneGenerator(seed=0, **_ALPHA_SHAPE_KWARGS)
    volume = gen.generate()

    assert volume.shape == _ALPHA_SHAPE_KWARGS["target_shape_zyx"]
    assert torch.isfinite(volume).all()
    assert volume.max() > 0
    assert gen.field is not None
    assert gen.profile is not None


def test_alpha_shape_backend_is_seed_reproducible():
    gen_a = MembraneGenerator(seed=3, **_ALPHA_SHAPE_KWARGS)
    gen_b = MembraneGenerator(seed=3, **_ALPHA_SHAPE_KWARGS)
    volume_a = gen_a.generate()
    volume_b = gen_b.generate()
    assert torch.equal(volume_a, volume_b)


def test_alpha_shape_backend_supports_transmembrane_placement():
    """The whole point of building this backend on top of MembraneGenerator
    rather than reusing CTS's own MembraneBlobGenerator directly: correct
    normal-aligned transmembrane placement, which CTS's own "location"
    mechanism does not provide for arbitrary (non-pre-oriented) PDB
    structures."""
    pdb_path = Path(__file__).parent.parent / "pdb-data" / "1mbo.cif"
    if not pdb_path.exists():
        pytest.skip("bundled PDB fixture missing")
    gen = MembraneGenerator(
        transmembrane_specs=[TransmembraneSpec(pdb_source=str(pdb_path), frequency=3)],
        seed=0,
        **_ALPHA_SHAPE_KWARGS,
    )
    gen.generate()
    placements = gen.place_transmembrane(min_spacing_a=15.0)
    assert len(placements) > 0


def test_shape_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="shape_backend"):
        MembraneGenerator(
            target_shape_zyx=(32, 32, 32), v_size=6.0, shape_backend="not_a_backend"
        )


def test_alpha_shape_surface_smoothing_reduces_facet_kinks():
    """Regression test for a real user-reported issue: the alpha-shape's
    Delaunay-triangulated boundary is faceted/kinked rather than smooth at
    any realistic blob_size_a (verified visually, including against CTS's
    own native generator reusing the same underlying point-cloud/alpha-shape
    algorithm directly -- the faceting is inherent to it, not something this
    backend introduced; raising point count alone doesn't fix it either,
    since surface area grows as blob_size_a**2 while _build_blob's point
    count grows far more slowly). surface_smoothing_voxels (blur-then-
    rethreshold the solid mask, default 2.0) fixes this. Verify
    quantitatively via the isoperimetric ratio (surface area / volume^(2/3),
    from face-adjacency transitions on the phi<0 mask) -- a faceted/kinked
    surface has excess area for its enclosed volume relative to a smooth
    one; smoothing should measurably reduce it."""
    from specter.specimen.membrane._field_alpha import (
        generate_membrane_field_alpha_shape,
    )

    kwargs = dict(
        shape_zyx=(80, 80, 80),
        spacing_a=4.0,
        blob_size_a=60.0,
        blob_roughness=0.3,
        seed=0,
    )
    field_kinked = generate_membrane_field_alpha_shape(
        surface_smoothing_voxels=0.0, **kwargs
    )
    field_smooth = generate_membrane_field_alpha_shape(
        surface_smoothing_voxels=2.0, **kwargs
    )

    def isoperimetric_ratio(phi: torch.Tensor, spacing_a: float) -> float:
        import numpy as np

        mask = phi.numpy() < 0
        volume = mask.sum() * spacing_a**3
        area_voxels = sum(np.sum(mask != np.roll(mask, 1, axis=ax)) for ax in range(3))
        area = area_voxels * spacing_a**2
        return float(area / volume ** (2.0 / 3.0))

    ratio_kinked = isoperimetric_ratio(field_kinked.phi, 4.0)
    ratio_smooth = isoperimetric_ratio(field_smooth.phi, 4.0)
    assert ratio_smooth < 0.75 * ratio_kinked


def test_alpha_shape_backend_warns_when_blob_too_small_for_reliable_resolution():
    """Regression test for a real finding: a blob_size_a too small relative
    to the working grid's spacing makes sample_surface_sites' Newton
    projection unreliable, previously silently finding zero transmembrane
    sites with no warning at all. Both this proactive check (fires during
    generate()) and the reactive one in place_transmembrane below must
    fire, not just one -- verified directly across a v_size sweep that a
    too-small blob keeps failing regardless of v_size once field spacing
    saturates at its own resolution floor (driven by the bilayer profile,
    not v_size), so this is a real, not incidental, gap to guard."""
    gen = MembraneGenerator(
        target_shape_zyx=(80, 80, 80),
        v_size=4.0,
        shape_backend="alpha_shape",
        blob_size_a=18.0,  # ~6.4 voxels/radius at this spacing -- below the
        # verified-reliable floor (see _field_alpha.py's
        # _MIN_RELIABLE_VOXELS_PER_RADIUS)
        blob_roughness=0.5,
        n_lipids_per_leaflet=6,
        seed=0,
    )
    with pytest.warns(UserWarning, match="voxels/radius"):
        gen.generate()


def test_place_transmembrane_warns_on_partial_or_zero_placement():
    """Pre-existing gap, not specific to the new backend: place_transmembrane
    silently returned fewer (or zero) placements than requested with no
    warning at all, for either shape_backend. Fixed for both."""
    pdb_path = Path(__file__).parent.parent / "pdb-data" / "1mbo.cif"
    if not pdb_path.exists():
        pytest.skip("bundled PDB fixture missing")
    gen = MembraneGenerator(
        target_shape_zyx=(80, 80, 80),
        v_size=4.0,
        shape_backend="alpha_shape",
        blob_size_a=18.0,
        blob_roughness=0.5,
        n_lipids_per_leaflet=6,
        # frequency=8 (not e.g. 3): verified directly this many requested
        # sites reliably falls short at this resolution/spacing (a lower
        # frequency, e.g. 3-5, sometimes fully succeeds by chance despite
        # the low-resolution risk generate()'s own proactive warning
        # already flags -- that warning is a risk indicator, not a
        # guarantee of shortfall, so this reactive-warning test needs its
        # own separately-verified failure case).
        transmembrane_specs=[TransmembraneSpec(pdb_source=str(pdb_path), frequency=8)],
        seed=0,
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")  # let generate()'s own warning through quietly
        gen.generate()
    with pytest.warns(UserWarning, match="requested transmembrane sites were found"):
        placements = gen.place_transmembrane(min_spacing_a=15.0)
    assert len(placements) < 8
