"""Tests for specter.specimen.tomogram._regions.classify_membrane_regions."""

from __future__ import annotations

import pytest
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


def test_cc3d_labelling_matches_scipy_partition():
    """The classifier labels the non-shell space with cc3d; scipy.ndimage.label
    at full 26-connectivity must give the same partition (one scipy label per
    cc3d label and vice versa), and the same region masks through the classifier."""
    import numpy as np
    from scipy import ndimage

    torch.manual_seed(0)
    n = 48
    zz, yy, xx = torch.meshgrid(
        torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij"
    )
    density = torch.zeros(n, n, n)
    # three hollow spheres: two closed, one cut open by the volume wall
    for c, r in (((16, 16, 16), 9.0), ((32, 30, 28), 10.0), ((44, 24, 24), 8.0)):
        d = torch.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2).float()
        density += torch.exp(-((d - r) ** 2) / (2 * 1.2**2))
    # a little noise-like speckle so components have ragged edges
    density += 0.02 * torch.rand(n, n, n)

    masks = classify_membrane_regions(density)
    non_shell = (~masks["shell"]).numpy()
    scipy_labels, n_scipy = ndimage.label(non_shell, structure=np.ones((3, 3, 3)))
    import cc3d

    cc_labels, n_cc = cc3d.connected_components(
        non_shell, connectivity=26, return_N=True
    )
    assert n_scipy == n_cc
    pairs = np.unique(np.stack([scipy_labels.ravel(), cc_labels.ravel()], 1), axis=0)
    assert len(pairs) == n_scipy + 1  # a bijection between the two label sets

    # and the classifier's own answer, rebuilt with scipy's labels
    boundary = set()
    for face in (
        scipy_labels[0],
        scipy_labels[-1],
        scipy_labels[:, 0],
        scipy_labels[:, -1],
        scipy_labels[:, :, 0],
        scipy_labels[:, :, -1],
    ):
        boundary.update(int(v) for v in np.unique(face) if v > 0)
    is_boundary = np.zeros(n_scipy + 1, dtype=bool)
    is_boundary[list(boundary)] = True
    cytosol_ref = non_shell & is_boundary[scipy_labels]
    assert np.array_equal(masks["cytosol"].numpy(), cytosol_ref)
    assert np.array_equal(masks["lumen"].numpy(), non_shell & ~cytosol_ref)
    assert bool(masks["lumen"][16, 16, 16]) and bool(masks["lumen"][32, 30, 28])
    assert bool(masks["cytosol"][0, 0, 0])


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


def _full_volume_reference(
    density: torch.Tensor, density_threshold: float | None = None
) -> dict[str, torch.Tensor]:
    """
    Label the WHOLE volume, the way this module did before the shell's
    bounding box was used to bound the work.

    Kept verbatim as the oracle for
    `test_shell_bbox_labelling_matches_full_volume_labelling`: the point of
    that test is that restricting `ndimage.label` to the shell's bounding
    box is an exact optimization, so it needs the unrestricted answer to
    compare against, not a second expression of the restricted one.
    """
    import numpy as np
    from scipy import ndimage

    if density_threshold is None:
        peak = float(density.max())
        density_threshold = 0.05 * peak if peak > 0 else 0.0
    shell_mask = density > density_threshold
    non_shell = ~shell_mask
    structure = np.ones((3, 3, 3), dtype=np.int8)
    labeled, num_features = ndimage.label(non_shell.cpu().numpy(), structure=structure)
    boundary_labels: set[int] = set()
    if num_features > 0:
        for face in (
            labeled[0, :, :],
            labeled[-1, :, :],
            labeled[:, 0, :],
            labeled[:, -1, :],
            labeled[:, :, 0],
            labeled[:, :, -1],
        ):
            boundary_labels.update(int(v) for v in face.ravel() if v > 0)
    if boundary_labels:
        is_boundary = np.zeros(num_features + 1, dtype=bool)
        is_boundary[np.fromiter(boundary_labels, dtype=np.int64)] = True
        cytosol = non_shell & torch.from_numpy(is_boundary[labeled])
    else:
        cytosol = torch.zeros_like(non_shell)
    return {"shell": shell_mask, "lumen": non_shell & ~cytosol, "cytosol": cytosol}


def _shell(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    r_inner: float,
    r_outer: float,
    value: float = 7.7,
) -> torch.Tensor:
    zz, yy, xx = torch.meshgrid(
        *(torch.arange(s, dtype=torch.float32) for s in shape), indexing="ij"
    )
    r = ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2).sqrt()
    volume = torch.zeros(shape)
    volume[(r > r_inner) & (r < r_outer)] = value
    return volume


def test_shell_bbox_labelling_matches_full_volume_labelling() -> None:
    """Bounding the connected-component labelling to the shell's own bbox
    is exact, including for the geometries where it could plausibly not be.

    `generator.py` stamps the carbon film into the volume BEFORE the
    membrane phase, so the density handed to `classify_membrane_regions` is
    carbon plus membranes and its shell is not membrane-only -- which is
    why the box is taken from the shell mask rather than from any
    membrane's geometry, and why the carbon-slab cases below are here."""
    shape = (48, 64, 64)
    cases = {
        "single vesicle": _shell(shape, (24, 32, 32), 14, 18),
        "no membrane at all": torch.zeros(shape),
        "two disjoint vesicles": (
            _shell(shape, (24, 18, 18), 8, 11) + _shell(shape, (24, 46, 46), 8, 11)
        ),
        "nested vesicles": (
            _shell(shape, (24, 32, 32), 19, 22) + _shell(shape, (24, 32, 32), 7, 10)
        ),
        "clipped at a face": _shell(shape, (0, 32, 32), 14, 18),
        "shell everywhere": torch.full(shape, 9.0),
    }
    # A hole punched through the shell, so the lumen escapes to the cytosol
    # and there is no enclosed region left at all.
    leaky = _shell(shape, (24, 32, 32), 14, 18)
    leaky[:, 30:35, 30:35] = 0.0
    cases["leaky shell"] = leaky
    # Carbon-like slab spanning the full cross-section: shell material that
    # no membrane put there, which must still bound the box correctly.
    slab = torch.zeros(shape)
    slab[4:8, :, :] = 12.0
    cases["carbon slab only"] = slab
    cases["carbon slab and vesicle"] = slab + _shell(shape, (28, 32, 32), 11, 14)

    for name, density in cases.items():
        got = classify_membrane_regions(density)
        want = _full_volume_reference(density)
        for key in ("shell", "lumen", "cytosol"):
            assert torch.equal(got[key], want[key]), f"{name}: {key} differs"
        partition = got["shell"].int() + got["lumen"].int() + got["cytosol"].int()
        assert torch.equal(partition, torch.ones(shape, dtype=torch.int32)), name


def test_streaming_mrc_writer_matches_mrcfile_set_data(tmp_path) -> None:
    """`_write_volume_mrc` writes the same file mrcfile's `set_data` would.

    It exists to avoid `set_data`'s `update_header_stats`, which fills the
    header's rms field with `ndarray.std()` over the whole volume and
    allocates a full-size temporary to do it -- 27 GB, and half the wall
    clock, on a 2 A production tomogram. So the bytes and all four header
    statistics have to come back unchanged.

    The large-mean case is the one with teeth: deriving rms from a running
    `E[x^2] - mean^2` in a single pass cancels catastrophically when the
    values sit far from zero, which is why the implementation takes a
    second pass over the slabs instead.
    """
    import mrcfile
    import numpy as np

    from specter.pipelines._tomogram import _write_volume_mrc

    rng = np.random.default_rng(0)
    cases = {
        "potential-like": rng.gamma(2.0, 1.5, size=(24, 32, 32)).astype("float32"),
        "all zeros": np.zeros((12, 20, 20), dtype="float32"),
        "large mean": rng.normal(500.0, 0.5, size=(16, 24, 24)).astype("float32"),
        "uint16 labels": rng.integers(0, 5000, size=(14, 24, 24)).astype("uint16"),
    }
    for name, array in cases.items():
        new_path = str(tmp_path / f"{name}-new.mrc")
        ref_path = str(tmp_path / f"{name}-ref.mrc")
        _write_volume_mrc(new_path, array, 3.25)
        with mrcfile.new(ref_path, overwrite=True) as mrc:
            mrc.set_data(array.copy())
            mrc.voxel_size = 3.25

        with mrcfile.open(new_path) as new, mrcfile.open(ref_path) as ref:
            assert np.array_equal(np.asarray(new.data), np.asarray(ref.data)), name
            assert float(new.voxel_size.x) == float(ref.voxel_size.x), name
            for field in ("dmin", "dmax", "dmean", "rms"):
                got, want = float(new.header[field]), float(ref.header[field])
                assert got == pytest.approx(want, rel=1e-6, abs=1e-6), (
                    f"{name}: header {field} {got!r} != {want!r}"
                )
