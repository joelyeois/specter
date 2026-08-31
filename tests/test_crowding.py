from __future__ import annotations

import pytest
import torch

import specter
from specter.crowding import CrowdWithDuplicates

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
