from __future__ import annotations

import pytest
import roma
import torch

import specter
from specter import rotations
from specter.crowding import CrowdWithDuplicates, rotation_safe_crop

# Small enough to stay quick, large enough that the Poisson-disk sampling
# places several duplicates -- with fewer than two there is nothing for
# chunking to reorder and the invariance below would hold vacuously.
_GEOM = dict(dx=2.0, min_distance=30.0, nxy_out=192, nz_out=64)


def _crowd(chunk_size: int, device: str) -> torch.Tensor:
    specter.seed(0)
    V = torch.rand(32, 32, 32, device=device)
    crowd = CrowdWithDuplicates(V, chunk_size=chunk_size, progressbars=False, **_GEOM)
    specter.seed(0)
    return crowd()


def test_chunk_size_does_not_change_the_result_on_cpu() -> None:
    """
    `chunk_size` is a memory knob, and on the CPU it is exactly that.

    The chunk loop inserts duplicates in the same global order whatever the
    chunk size, so the only thing that can differ is the reduction order
    *inside* one batched insert. On the CPU that reduction is deterministic,
    so the result is bit-for-bit identical -- which is what lets the CLI QA
    sweep assert `expect="unchanged"` for this flag.
    """
    assert torch.equal(_crowd(1, "cpu"), _crowd(4, "cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunk_size_perturbs_only_at_rounding_level_on_gpu() -> None:
    """
    On the GPU the same knob is not bit-exact, but is bounded well below noise.

    A batched insert accumulates through atomics there, whose ordering is not
    fixed, so `chunk_size` perturbs the sum -- the constructor docstring quotes
    ~4e-6 relative. That is float-rounding, not a physics change, and this
    pins it as such: a regression that made chunking actually *move* density
    would blow through this bound rather than hide behind "GPU is nondeterministic".
    """
    a, b = _crowd(1, "cuda:0"), _crowd(4, "cuda:0")
    rel = (a - b).abs().max().item() / max(a.abs().max().item(), 1e-30)
    assert rel < 1e-4, f"chunking perturbed the result by {rel:.2e} relative"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_seeded_run_is_reproducible_on_gpu() -> None:
    """
    The atomics above must not leak into run-to-run reproducibility.

    Every simulate/build config defaults to `device="cuda"`, so `specter.seed`
    has to mean the same thing there as on the CPU. Holding `chunk_size` fixed,
    two seeded runs must agree exactly.
    """
    assert torch.equal(_crowd(1, "cuda:0"), _crowd(1, "cuda:0"))


def _blob(n_atoms: int = 300, radius: float = 20.0, seed: int = 0) -> torch.Tensor:
    """A compact random point cloud standing in for a small globular protein.

    Same convention as `test_packing_shape.py`'s own helper, kept local
    rather than imported since it's a one-line synthetic fixture, not
    shared test infrastructure.
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n_atoms, 3, generator=g)
    v = v / v.norm(dim=1, keepdim=True)
    r = radius * torch.rand(n_atoms, 1, generator=g) ** (1 / 3)
    coords = v * r
    return coords - coords.mean(0)


def test_shape_backend_requires_atom_coordinates() -> None:
    """
    `packing_backend='shape'` cannot collide real footprints without atoms.

    A silent fallback (e.g. to poisson_disk, or to an empty placement) would
    be far worse than an explicit error here -- a caller who asked for
    shape-aware crowding and got bounding-sphere crowding instead with no
    warning would draw wrong conclusions from the resulting density.
    """
    V = torch.rand(16, 16, 16)
    with pytest.raises(ValueError, match="atom_coordinates"):
        CrowdWithDuplicates(V, dx=2.0, min_distance=30.0, packing_backend="shape")


def test_shape_backend_places_denser_than_poisson_disk() -> None:
    """
    The entire point of `packing_backend='shape'`: it should out-crowd
    `poisson_disk` at the same `min_distance` and box, because it collides
    the real (non-spherical) footprint instead of a bounding-sphere
    distance -- see `CrowdWithDuplicates`'s own docstring for the measured
    ~8.6x figure this is a smaller-scale version of. A rod-shaped template
    (`_blob` is closer to spherical, where the two backends should be
    closer) makes the difference maximal and the test fast at a small box.
    """
    g = torch.Generator().manual_seed(1)
    z = (torch.rand(300, 1, generator=g) * 2 - 1) * 45.0
    xy = torch.randn(300, 2, generator=g) * 4.0
    atoms = torch.cat([xy, z], dim=1)
    atoms = atoms - atoms.mean(0)

    dx = 4.0
    V = torch.rand(32, 32, 32)
    max_diameter = float((atoms.max(0).values - atoms.min(0).values).norm())
    geom = dict(dx=dx, min_distance=max_diameter, nxy_out=128, nz_out=64)

    specter.seed(0)
    n_poisson = CrowdWithDuplicates(V, progressbars=False, **geom)
    n_poisson.generate_coordinates()

    specter.seed(0)
    n_shape = CrowdWithDuplicates(
        V,
        progressbars=False,
        packing_backend="shape",
        atom_coordinates=atoms,
        packing_max_retries=300,
        packing_stall_patience=2000,
        **geom,
    )
    n_shape.generate_coordinates()

    assert len(n_shape.coords) > len(n_poisson.coords), (
        f"shape backend placed {len(n_shape.coords)} instances, "
        f"poisson_disk placed {len(n_poisson.coords)} -- expected shape to "
        "place strictly more for an elongated template"
    )


def test_shape_backend_reuses_committed_rotations_for_render() -> None:
    """
    Shape-backend coordinates and rotations must round-trip together.

    `pack_shapes_3d` commits to an orientation per instance as part of the
    collision test; `generate_affine_matrices` must reuse it rather than
    draw a fresh one, or the rendered volume would depict geometry that was
    never actually collision-tested (see that method's own docstring).
    """
    atoms = _blob()
    dx = 4.0
    V = torch.rand(16, 16, 16)
    crowd = CrowdWithDuplicates(
        V,
        dx=dx,
        min_distance=30.0,
        nxy_out=96,
        nz_out=48,
        packing_backend="shape",
        atom_coordinates=atoms,
        packing_max_retries=300,
        packing_stall_patience=2000,
        progressbars=False,
    )
    crowd.generate_coordinates()
    assert len(crowd.coords) > 0, "expected at least one accepted instance"
    committed_rotations = crowd._shape_rotations.clone()

    crowd.generate_affine_matrices()

    assert torch.equal(crowd.theta[:, :3, :3], committed_rotations), (
        "generate_affine_matrices must reuse the packer's committed rotations"
    )


def test_water_air_interface_knobs_are_threaded_for_poisson_disk() -> None:
    """
    `sigma_frac`/`peak_amplitude`/`baseline` are constructor knobs, not just
    `filter_by_local_z_density`'s own hardcoded defaults -- a caller raising
    `baseline` should keep a visibly larger bulk fraction than one that
    doesn't, at the same seed and raw candidate pool.
    """
    geom = dict(
        dx=2.0,
        min_distance=6.0,
        nxy_out=128,
        nz_out=128,
        water_air_interface=True,
        progressbars=False,
    )
    V = torch.rand(16, 16, 16)

    specter.seed(0)
    tight = CrowdWithDuplicates(V, sigma_frac=0.02, baseline=0.0, **geom)
    tight.generate_coordinates()

    specter.seed(0)
    loose = CrowdWithDuplicates(V, sigma_frac=0.02, baseline=1.0, **geom)
    loose.generate_coordinates()

    assert len(loose.coords) > len(tight.coords), (
        f"baseline=1.0 kept {len(loose.coords)}, baseline=0.0 kept "
        f"{len(tight.coords)} -- expected the higher bulk baseline to keep "
        "strictly more, at the same seed and raw candidate pool"
    )


def test_water_air_interface_knobs_are_threaded_for_shape_backend() -> None:
    """
    Same wiring check as the poisson_disk test above, for
    `_generate_coordinates_shape`'s own thinning step.
    """
    atoms = _blob()
    geom = dict(
        dx=4.0,
        min_distance=30.0,
        nxy_out=96,
        nz_out=64,
        packing_backend="shape",
        atom_coordinates=atoms,
        packing_max_retries=300,
        packing_stall_patience=2000,
        water_air_interface=True,
        progressbars=False,
    )
    V = torch.rand(16, 16, 16)

    specter.seed(0)
    tight = CrowdWithDuplicates(V, sigma_frac=0.02, baseline=0.0, **geom)
    tight.generate_coordinates()

    specter.seed(0)
    loose = CrowdWithDuplicates(V, sigma_frac=0.02, baseline=1.0, **geom)
    loose.generate_coordinates()

    assert len(loose.coords) > len(tight.coords), (
        f"baseline=1.0 kept {len(loose.coords)}, baseline=0.0 kept "
        f"{len(tight.coords)} -- expected the higher bulk baseline to keep "
        "strictly more, at the same seed and raw jammed pool"
    )


# ---------------------------------------------------------------------------
# rotation_safe_crop
# ---------------------------------------------------------------------------


def _centred_blob(n: int, radius: float) -> torch.Tensor:
    """A centred spherical blob of `radius` voxels in an ``(n, n, n)`` box."""
    g = torch.arange(n, dtype=torch.float32) - (n - 1) / 2
    r2 = g.view(-1, 1, 1) ** 2 + g.view(1, -1, 1) ** 2 + g.view(1, 1, -1) ** 2
    return torch.where(r2 <= radius**2, 1.0 + r2, torch.zeros(()))


def test_rotation_safe_crop_keeps_every_voxel_a_rotation_can_reach() -> None:
    """
    The crop must contain the whole sphere the density sweeps out.

    Rotation preserves distance from the centre, so the guarantee is
    geometric: a crop whose half-width covers the furthest occupied voxel
    cannot lose density to any rotation. Checked directly rather than by
    trusting the radius arithmetic -- every occupied voxel of the original
    has to survive the crop.
    """
    V = _centred_blob(64, 12.0)
    crop = rotation_safe_crop(V)
    assert crop.shape[0] < V.shape[0], "a blob in a big box should crop"
    assert float(crop.abs().sum()) == float(V.abs().sum())
    lo, side = (V.shape[0] - crop.shape[0]) // 2, crop.shape[0]
    assert torch.equal(crop, V[lo : lo + side, lo : lo + side, lo : lo + side])


def test_rotation_safe_crop_loses_no_density_under_rotation() -> None:
    """
    The sharp invariant: rotating the crop must reproduce rotating the whole
    box, over its own extent, and must not have thrown density away.

    The half-width has to clear the furthest occupied voxel *plus the
    interpolation's reach* -- ``rotate_volume`` is trilinear, so a voxel at
    radius R lands density out to R + sqrt(3). Cropping at R alone passes
    every geometric check and still clips that outer shell; it cost 20% of
    this blob before the margin was added, which is why the test rotates
    rather than measuring the crop.
    """
    V = _centred_blob(64, 12.0)
    crop = rotation_safe_crop(V)
    lo, side = (V.shape[0] - crop.shape[0]) // 2, crop.shape[0]

    specter.seed(3)
    quats = rotations.random_quaternion(8)
    theta = rotations.build_affine_matrix(
        roma.unitquat_to_rotmat(quats), torch.zeros(8, 3)
    )
    whole = rotations.rotate_volume(V, theta, padding_mode="zeros")
    part = rotations.rotate_volume(crop, theta, padding_mode="zeros")

    beyond = whole.clone()
    beyond[..., lo : lo + side, lo : lo + side, lo : lo + side] = 0.0
    assert float(beyond.abs().max()) == 0.0, (
        "rotating the whole box put density outside the crop, so the crop "
        "would have thrown it away"
    )

    inside = whole[..., lo : lo + side, lo : lo + side, lo : lo + side]
    rel = (part - inside).abs().max().item() / inside.abs().max().item()
    assert rel < 1e-4, f"cropping changed the rotated template by {rel:.2e}"


@pytest.mark.parametrize("n", [64, 65])
def test_rotation_safe_crop_keeps_each_axis_parity(n: int) -> None:
    """
    A crop that flipped an axis's parity would move the duplicate half a voxel.

    ``grid_sample`` rotates about ``(n - 1) / 2`` while
    `clip_insert_bounds` positions index ``n // 2``; the two coincide for an
    odd axis and sit half a voxel apart for an even one. Cropping keeps that
    offset only if the parity is kept, so this is a correctness rule, not a
    tidiness one.
    """
    crop = rotation_safe_crop(_centred_blob(n, 9.0))
    assert crop.shape[0] % 2 == n % 2
    assert crop.shape[0] < n


def test_rotation_safe_crop_declines_when_there_is_nothing_to_crop() -> None:
    """Density reaching the faces, or none at all, returns the input itself."""
    full = _centred_blob(32, 40.0)  # radius past the box: occupied to every face
    assert rotation_safe_crop(full) is full
    empty = torch.zeros(16, 16, 16)
    assert rotation_safe_crop(empty) is empty


def test_cropping_the_template_does_not_move_the_duplicates() -> None:
    """
    Cropping is a speed change, not a placement change.

    `CrowdWithDuplicates` rotates one template once per duplicate, and the
    box a template is rendered in is sized for the IMAGE, not the molecule:
    6bdf at 384 px fills a 103-voxel radius, so the full box resamples 6.3x
    the voxels the content needs. Cropping first is exact in principle --
    what lies outside is zero and rotates to zero -- and the residual here is
    the interpolation grid's own rounding, since normalised sample
    coordinates divide by a different half-width. Bounded well under the
    ~2e-5 the forward model already varies by between runs.
    """
    specter.seed(0)
    V = _centred_blob(48, 11.0)
    crowd = CrowdWithDuplicates(V, chunk_size=1, progressbars=False, **_GEOM)
    specter.seed(0)
    cropped = crowd()

    crowd_full = CrowdWithDuplicates(V, chunk_size=1, progressbars=False, **_GEOM)
    crowd_full._rotation_template = lambda: crowd_full.V  # type: ignore[method-assign]
    specter.seed(0)
    whole = crowd_full()

    assert crowd._rotation_template().shape[0] < V.shape[0], "expected a crop"
    rel = (cropped - whole).abs().max().item() / whole.abs().max().item()
    assert rel < 1e-4, f"cropping moved density by {rel:.2e} relative"


def test_rotation_template_is_cached_and_follows_a_swapped_buffer() -> None:
    """
    The crop is computed once per template, and re-computed when `V` changes.

    `ImageGenerator._process_volume` aliases its own volume buffer onto the
    crowd's after a device move, so a cache keyed on anything but the
    buffer's identity would keep serving a crop of the old tensor.
    """
    crowd = CrowdWithDuplicates(_centred_blob(48, 11.0), progressbars=False, **_GEOM)
    first = crowd._rotation_template()
    assert crowd._rotation_template() is first

    crowd.V = _centred_blob(48, 20.0)
    second = crowd._rotation_template()
    assert second is not first
    assert second.shape[0] > first.shape[0]
