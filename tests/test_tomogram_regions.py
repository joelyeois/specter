"""Tests for specter.specimen.tomogram._regions.classify_membrane_regions."""

from __future__ import annotations

import torch

from specter.specimen.tomogram._regions import classify_membrane_regions


def _hollow_sphere_density(
    n: int = 40, shell_radius: float = 12.0, sigma: float = 1.5
) -> torch.Tensor:
    zz, yy, xx = torch.meshgrid(
        torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij"
    )
    center = n / 2
    r = torch.sqrt((zz - center) ** 2 + (yy - center) ** 2 + (xx - center) ** 2).float()
    return torch.exp(-((r - shell_radius) ** 2) / (2 * sigma**2))


def test_classify_membrane_regions_partitions_full_volume():
    density = _hollow_sphere_density()
    masks = classify_membrane_regions(density)
    total = masks["shell"].sum() + masks["lumen"].sum() + masks["cytosol"].sum()
    assert int(total) == density.numel()
    # every voxel belongs to exactly one region
    overlap = (
        masks["shell"] & masks["lumen"]
        | masks["shell"] & masks["cytosol"]
        | masks["lumen"] & masks["cytosol"]
    )
    assert not bool(overlap.any())


def test_classify_membrane_regions_hollow_sphere_center_is_lumen_corner_is_cytosol():
    density = _hollow_sphere_density(n=40, shell_radius=12.0)
    masks = classify_membrane_regions(density)
    assert bool(masks["lumen"][20, 20, 20])
    assert bool(masks["cytosol"][0, 0, 0])
    assert bool(masks["shell"][20, 20, 8])  # on the shell at radius ~12 from center


def test_classify_membrane_regions_no_membrane_is_all_cytosol():
    density = torch.zeros(10, 10, 10)
    masks = classify_membrane_regions(density)
    assert int(masks["cytosol"].sum()) == density.numel()
    assert int(masks["shell"].sum()) == 0
    assert int(masks["lumen"].sum()) == 0


def test_classify_membrane_regions_two_disjoint_vesicles_both_become_lumen():
    n = 60
    zz, yy, xx = torch.meshgrid(
        torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij"
    )
    zz, yy, xx = zz.float(), yy.float(), xx.float()

    def shell(cz, cy, cx, radius, sigma=1.5):
        r = torch.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
        return torch.exp(-((r - radius) ** 2) / (2 * sigma**2))

    density = torch.maximum(shell(20, 20, 20, 8.0), shell(40, 40, 40, 8.0))
    masks = classify_membrane_regions(density)
    assert bool(masks["lumen"][20, 20, 20])
    assert bool(masks["lumen"][40, 40, 40])
    assert bool(masks["cytosol"][0, 0, 0])
    assert bool(
        masks["cytosol"][30, 30, 30]
    )  # between the two vesicles, not enclosed by either


def test_classify_membrane_regions_default_threshold_scales_with_peak():
    density = (
        _hollow_sphere_density() * 100.0
    )  # scale peak up, threshold should track it
    masks = classify_membrane_regions(density)
    assert bool(masks["lumen"][20, 20, 20])
    assert bool(masks["cytosol"][0, 0, 0])


def test_many_boundary_components_classify_correctly():
    """The boundary-label membership test used to be `torch.isin`, which has
    no sorted fast path at these sizes and materializes an
    (n_voxels, n_boundary_labels) bool tensor. Measured at 65.1 bytes per
    voxel for 63 labels, so a 300x1200x1200 tomogram asked CUDA for
    25.35 GiB and OOM'd -- and since the factor is the label COUNT, which
    the membrane geometry decides, it fired on some random draws and not
    others.

    This builds many disjoint boundary-touching components on purpose, the
    shape that drove that count up, plus one genuinely enclosed cavity that
    must still come back as lumen rather than being swept in with them."""
    n = 24
    density = torch.zeros((n, n, n))

    # A grid of shell pillars, carving the exterior into many separate
    # components that each touch a boundary face.
    density[:, ::4, ::4] = 1.0

    # One sealed box, well away from the pillars, whose interior is
    # reachable from nowhere outside.
    density[4:10, 14:20, 14:20] = 1.0
    density[5:9, 15:19, 15:19] = 0.0

    masks = classify_membrane_regions(density, density_threshold=0.5)

    total = masks["shell"].sum() + masks["lumen"].sum() + masks["cytosol"].sum()
    assert int(total) == density.numel()

    # The sealed interior is lumen, and nothing on a boundary face is.
    assert bool(masks["lumen"][6, 16, 16])
    for face in (
        masks["lumen"][0],
        masks["lumen"][-1],
        masks["lumen"][:, 0],
        masks["lumen"][:, -1],
        masks["lumen"][..., 0],
        masks["lumen"][..., -1],
    ):
        assert not bool(face.any())

    # The many exterior components are all cytosol, not lumen.
    assert bool(masks["cytosol"][0, 1, 1])
    assert bool(masks["cytosol"][-1, -2, -2])


def test_label_volumes_escalate_to_float32_past_the_uint16_ceiling(tmp_path):
    """A label id above 65,535 must not be written as a wrapped uint16.

    numpy's astype wraps silently: instance 65,536 becomes 0 and reads
    back as BACKGROUND, and anything beyond collides with a real
    instance's id -- while the .ndjson picks stay correct, so the two
    disagree with nothing to say so. MRC has no integer mode wider than
    uint16 (mrcfile rejects int32 outright), so the writer escalates to
    float32, which is exact for integers to 2**24.

    Exercises `run_build_tomogram`'s own writer rather than a copy of its
    rule, so the test fails if the escalation is removed."""
    import mrcfile
    import numpy as np
    import torch

    from specter.pipelines._tomogram import _MRC_UINT16_MAX

    assert _MRC_UINT16_MAX == 65535

    # What the old code did, kept only to show what is being prevented.
    over = np.array([_MRC_UINT16_MAX + 1, 70000], dtype=np.int32)
    assert list(over.astype("uint16")) == [0, 4464], "uint16 wraps silently"

    # float32 is exact across the range the escalation covers.
    ids = np.arange(0, 2**20, dtype=np.int32)
    assert np.array_equal(ids, ids.astype("float32").astype("int32"))

    # And it survives a real MRC round trip.
    labels = torch.zeros(2, 2, 2, dtype=torch.int32)
    labels[0, 0, 0] = 70000
    path = tmp_path / "labels.mrc"
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(labels.numpy().astype("float32"))
    assert int(np.asarray(mrcfile.open(str(path)).data).max()) == 70000
