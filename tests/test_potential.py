"""
Tests for potential volume construction.
"""

import warnings
from importlib import resources

import gemmi
import pytest
import torch

from specter.arrays import radial_grid_3d
from specter.atom import (
    load_shtyrov_species_parameters,
    shtyrov_atomic_potential_3d_by_species,
)
from specter.potential import (
    GemmiPotentialBuilder,
    PotentialBuilder,
    build_atomic_potential_kernel,
    build_potential_volume_analytic_scatter_kirkland,
    build_potential_volume_analytic_scatter_lobato,
    build_potential_volume_fftconvolve_3d,
    potential_from_deltas,
    recommended_rcut,
)


# A small cluster of 5 carbon atoms (Z=6) spread over ~6 Å.
# Positions are in Å, centered at the origin.
_ATOMIC_NUMBERS = torch.tensor([6, 6, 6, 6, 6], dtype=torch.long)
_COORDS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [-2.5, 0.0, 0.0],
        [0.0, 2.5, 0.0],
        [0.0, 0.0, 2.5],
    ]
)


def _column_potential(dx: float, n: int) -> float:
    """
    Build a potential volume and return the total volume integral of V.

    The potential builder stores V in units of V (3D electrostatic potential).
    The dz integration factor is applied in scattering.py. The physically
    conserved quantity is therefore:

        V.sum() * dx³   [V · Å³]

    i.e. the total potential integrated over the full 3D volume.
    """
    V, _ = build_potential_volume_fftconvolve_3d(
        _ATOMIC_NUMBERS, _COORDS, n_xyz=n, dx=dx, disable_tqdm=True
    )
    return (V.sum() * dx**3).item()


@pytest.mark.parametrize(
    "dx1,n1,dx2,n2",
    [
        # Same physical box (32 Å³), different voxel sizes
        (1.0, 32, 2.0, 16),
        (1.0, 32, 1.5, 22),  # ~33 Å at 1.5 Å/px, close enough
    ],
)
def test_potential_integral_pixel_size_invariance(dx1, n1, dx2, n2):
    """
    Total volume integral (V.sum() * dx³) should be approximately the same
    when building from identical atomic coordinates at different pixel sizes,
    as long as the physical box is large enough to contain the particle.

    This invariant follows from V storing the 3D electrostatic potential in
    volts, so V.sum() * dx³ gives the total potential integrated over the
    full volume.
    """
    col1 = _column_potential(dx1, n1)
    col2 = _column_potential(dx2, n2)

    # Allow up to 2% relative deviation — mainly due to grid discretisation
    # and minor differences in kernel truncation between pixel sizes.
    assert abs(col1 - col2) / abs(col1) < 0.02, (
        f"Column potential changed by more than 2% between "
        f"dx={dx1} (n={n1}, col_pot={col1:.3f}) and "
        f"dx={dx2} (n={n2}, col_pot={col2:.3f})"
    )


def test_potential_builder_shtyrov_species_smoke():
    """
    PotentialBuilder(parameterization='shtyrov', atom_species=...) should
    build a volume end-to-end: matched species use Shtyrov kernels, atoms
    with no matching species (None, or missing from the table) fall back
    to gemmi's built-in Peng et al. scattering factors with a warning, and
    the result should differ from a pure-Kirkland volume built from the
    same coordinates.
    """
    atomic_numbers = torch.tensor([8, 6, 26], dtype=torch.long)
    coords = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    # 'O(HH)' is in the bundled species table (matched); the other two are
    # deliberately unmatched to exercise the fallback path.
    atom_species = ["O(HH)", "C(unmatched)", None]

    with pytest.warns(UserWarning, match="falling back"):
        pb = PotentialBuilder(
            32,
            1.0,
            atomic_numbers,
            parameterization="shtyrov",
            atom_species=atom_species,
            progressbars=False,
        )

    assert pb.shtyrov_groups == [
        ("species", "O(HH)"),
        ("element", 6),
        ("element", 26),
    ]

    volume = pb.forward(coords)
    assert torch.isfinite(volume).all()
    assert volume.abs().sum() > 0

    pb_kirkland = PotentialBuilder(
        32, 1.0, atomic_numbers, parameterization="kirkland", progressbars=False
    )
    volume_kirkland = pb_kirkland.forward(coords)
    assert not torch.allclose(volume, volume_kirkland)


def test_shtyrov_species_potential_matches_analytic_closed_form():
    """
    shtyrov_atomic_potential_3d_by_species must match the analytic 3D
    inverse Fourier transform of its Gaussian-sum form factor.

    Regression test for two historical bugs, both silent until this check
    (and the atom_species smoke test above) started actually exercising
    this code path:
      - real_to_kgrid_3d inferring pixel spacing from a radial-*distance*
        grid (R[1,0,0]-R[0,0,0]) instead of the true spacing, silently
        corrupting the whole reciprocal-space grid.
      - computing the potential via `abs(fftn(...))`, discarding the sign
        of what should be a signed, positive-peaked real-space function.

    The reference here is derived independently (a plain closed-form
    Gaussian-sum evaluation, not a call into the library code under test):
    the 3D inverse Fourier transform of a single Gaussian term
    `a*exp(-b*k^2/4)` is `a * 8 * pi^1.5 * b^-1.5 * exp(-4*pi^2*r^2/b)`
    (standard 3D isotropic Gaussian FT pair, unitary e^{2*pi*i*k.r}
    convention) -- see also the equivalent (4*pi/b)^1.5 form the library
    code itself now uses.
    """
    species_path = resources.files("specter.atom_data").joinpath("params_cat.json")
    with resources.as_file(species_path) as fpath:
        table = load_shtyrov_species_parameters(str(fpath))
    P = table["O(HH)"].double()
    a, b = P[:, 0], P[:, 1]

    a0 = 0.529  # Bohr radius, [Å]
    e = 14.4  # electron charge, [V·Å]
    c1 = 2 * torch.pi * e * a0

    def independent_reference(r: float) -> float:
        r_t = torch.tensor(r, dtype=torch.float64)
        terms = (
            a
            * 8
            * torch.pi**1.5
            * b ** (-1.5)
            * torch.exp(-4 * torch.pi**2 * r_t**2 / b)
        )
        return (c1 * terms.sum()).item()

    # Small grid matching what PotentialBuilder actually supersamples to
    # (~5 Å box) -- this exact regime silently produced up to ~200%
    # relative error with the pre-fix FFT-based implementation.
    n, dx = 50, 0.1
    r_xyz = radial_grid_3d(n, dx, convention="torch")
    potential = shtyrov_atomic_potential_3d_by_species("O(HH)", r_xyz, table)

    for target_r in [0.0, 0.5, 1.0, 1.5, 2.0]:
        idx = int(torch.argmin((r_xyz - target_r).abs()))
        actual_r = r_xyz.flatten()[idx].item()
        numeric = potential.flatten()[idx].item()
        reference = independent_reference(actual_r)
        assert numeric == pytest.approx(reference, rel=1e-3), (
            f"mismatch at r={actual_r:.4f}: numeric={numeric}, "
            f"analytic reference={reference}"
        )


def _write_single_atom_shtyrov_mmcif(path: str, a_coefs, b_coefs) -> None:
    """Build a one-atom mmCIF with a custom `_lmb_scat_coef` table via gemmi's
    own API (not hand-written CIF text, which gemmi is picky about)."""
    st = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    res = gemmi.Residue()
    res.name = "HOH"
    res.seqid = gemmi.SeqId(1, " ")
    atom = gemmi.Atom()
    atom.name = "O"
    atom.element = gemmi.Element("O")
    atom.pos = gemmi.Position(0.0, 0.0, 0.0)
    atom.occ = 1.0
    atom.b_iso = 0.0
    atom.serial = 1
    res.add_atom(atom)
    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()

    doc = st.make_mmcif_document()
    block = doc.sole_block()
    block.find_loop("_atom_site.id").get_loop().add_columns(
        ["_atom_site.scat_id"], value="0"
    )
    sfloop = block.init_loop(
        "_lmb_scat_coef.",
        [
            "scat_id",
            "coef_a1",
            "coef_a2",
            "coef_a3",
            "coef_a4",
            "coef_a5",
            "coef_b1",
            "coef_b2",
            "coef_b3",
            "coef_b4",
            "coef_b5",
        ],
    )
    sfloop.add_row(
        ["0"] + [f"{v:.6f}" for v in a_coefs] + [f"{v:.6f}" for v in b_coefs]
    )
    doc.write_file(path)


def test_shtyrov_species_potential_matches_gemmi_density_calculator(tmp_path):
    """
    Cross-check shtyrov_atomic_potential_3d_by_species against gemmi's own
    DensityCalculatorC — an independent numerical implementation (not
    something derived by this codebase), fed the exact same 5-Gaussian
    coefficients via a custom `_lmb_scat_coef` mmCIF table
    (GemmiPotentialBuilder.build_potential_from_custom_mmcif).

    Matched voxel-for-voxel by each voxel's own radius (not raw array
    index, to stay robust to any axis-order differences between the two
    grid constructions). gemmi truncates the density to exactly zero past
    its own evaluation cutoff radius, so only radii within that cutoff
    (~1 Å here) are compared.
    """
    species_path = resources.files("specter.atom_data").joinpath("params_cat.json")
    with resources.as_file(species_path) as fpath:
        table = load_shtyrov_species_parameters(str(fpath))
    P = table["O(HH)"].numpy()
    a_coefs, b_coefs = P[:, 0], P[:, 1]

    n, dx = 50, 0.1
    mmcif_path = str(tmp_path / "single_O_HH.cif")
    _write_single_atom_shtyrov_mmcif(mmcif_path, a_coefs, b_coefs)

    gpb = GemmiPotentialBuilder(n_xyz=n, dx=dx, b_factor=0.0)
    gemmi_pot = gpb.build_potential_from_custom_mmcif(mmcif_path)

    idx = torch.arange(n) * dx
    X, Y, Z = torch.meshgrid(idx, idx, idx, indexing="ij")
    center = (n // 2) * dx
    r_gemmi = torch.sqrt((X - center) ** 2 + (Y - center) ** 2 + (Z - center) ** 2)
    r_gemmi_sorted, order = torch.sort(r_gemmi.flatten())
    v_gemmi_sorted = gemmi_pot.flatten()[order]

    r_ours = radial_grid_3d(n, dx, convention="relion")
    pot_ours = shtyrov_atomic_potential_3d_by_species("O(HH)", r_ours, table)
    r_ours_sorted, order_ours = torch.sort(r_ours.flatten())
    v_ours_sorted = pot_ours.flatten()[order_ours]

    for target_r in [0.0, 0.2, 0.5, 1.0]:
        gi = min(
            torch.searchsorted(r_gemmi_sorted, torch.tensor(target_r)).item(),
            len(r_gemmi_sorted) - 1,
        )
        oi = min(
            torch.searchsorted(r_ours_sorted, torch.tensor(target_r)).item(),
            len(r_ours_sorted) - 1,
        )
        assert v_gemmi_sorted[gi].item() == pytest.approx(
            v_ours_sorted[oi].item(), rel=1e-3
        ), f"mismatch at r~{target_r}"


@pytest.mark.parametrize(
    "species",
    [
        "H(N)",
        "C(HHCN)",
        "C(HCCC)",
        "C(HCCN)",
        "C(HCCO)",
        "N(CCFe)",
        "O(C, amide)",
    ],
)
def test_shtyrov_species_with_zero_b_coefficient_is_finite(species):
    """
    Regression test: 7 of the 42 bundled Shtyrov species have a b_i=0 term
    (a frequency-independent, i.e. Dirac-delta-like, contribution). The
    closed-form real-space formula divides by b, so before _MIN_GAUSSIAN_B
    was introduced this silently produced NaN for exactly these species —
    never caught earlier because the O(HH) species used throughout prior
    testing happens not to have this pattern.
    """
    species_path = resources.files("specter.atom_data").joinpath("params_cat.json")
    with resources.as_file(species_path) as fpath:
        table = load_shtyrov_species_parameters(str(fpath))
    assert any(b == 0.0 for b in table[species][:, 1].tolist()), (
        f"expected {species} to have a b_i=0 term; bundled data may have changed"
    )

    r_xyz = radial_grid_3d(20, 0.5, convention="torch")
    potential = shtyrov_atomic_potential_3d_by_species(species, r_xyz, table)
    assert torch.isfinite(potential).all()


def _write_multi_atom_shtyrov_mmcif(path, elements, coords, scat_coeffs):
    """
    Build a multi-atom mmCIF with a custom `_lmb_scat_coef` table via gemmi's
    own API, one row per unique coefficient set in `scat_coeffs` (a list of
    (a, b) arrays, one per atom, matching `elements`/`coords` in order).
    """
    st = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    res = gemmi.Residue()
    res.name = "LIG"
    res.seqid = gemmi.SeqId(1, " ")

    scat_ids = []
    unique_coeffs: list[tuple] = []
    for i, (elem, pos, (a, b)) in enumerate(zip(elements, coords, scat_coeffs)):
        key = (tuple(round(x, 6) for x in a), tuple(round(x, 6) for x in b))
        match = next((j for j, u in enumerate(unique_coeffs) if u == key), None)
        if match is None:
            unique_coeffs.append(key)
            match = len(unique_coeffs) - 1
        scat_ids.append(match)

        atom = gemmi.Atom()
        atom.name = f"{elem}{i}"
        atom.element = gemmi.Element(elem)
        atom.pos = gemmi.Position(*pos)
        atom.occ = 1.0
        atom.b_iso = 0.0
        atom.serial = i + 1
        res.add_atom(atom)

    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()

    doc = st.make_mmcif_document()
    block = doc.sole_block()
    block.find_loop("_atom_site.id").get_loop().add_columns(
        ["_atom_site.scat_id"], value="0"
    )
    table_block = block.find("_atom_site.", ["id", "scat_id"])
    for row, sid in zip(table_block, scat_ids):
        row[1] = str(sid)

    sfloop = block.init_loop(
        "_lmb_scat_coef.",
        [
            "scat_id",
            "coef_a1",
            "coef_a2",
            "coef_a3",
            "coef_a4",
            "coef_a5",
            "coef_b1",
            "coef_b2",
            "coef_b3",
            "coef_b4",
            "coef_b5",
        ],
    )
    for sid, (a, b) in enumerate(unique_coeffs):
        sfloop.add_row([str(sid)] + [f"{v:.6f}" for v in a] + [f"{v:.6f}" for v in b])
    doc.write_file(path)


def test_potential_builder_analytic_matches_gemmi_multi_atom(tmp_path):
    """
    End-to-end cross-check of PotentialBuilder(...).forward(method="analytic")
    — the production code path, not a standalone prototype — against gemmi's
    own DensityCalculatorC on a small multi-atom cluster mixing matched
    Shtyrov species and Peng-fallback atoms, on the exact same coordinate
    frame (both place physical (0,0,0) at voxel [n//2, n//2, n//2]).

    Note: our implementation evaluates the *exact voxel average* of the
    potential (see build_potential_volume_analytic_scatter's docstring for
    why — point-sampling is wildly sensitive to sub-voxel atom position),
    while gemmi point-samples. So we don't expect a near-exact voxel-by-voxel
    match right at atom peaks — check overall correlation (still very high)
    and total integrated potential (a physically meaningful quantity that
    should survive the different discretization convention) instead of a
    tight per-voxel tolerance.
    """
    species_path = resources.files("specter.atom_data").joinpath("params_cat.json")
    with resources.as_file(species_path) as fpath:
        table = load_shtyrov_species_parameters(str(fpath))

    n, dx = 40, 0.25
    elements = ["O", "O", "N", "C", "S"]
    atomic_numbers = torch.tensor([8, 8, 7, 6, 16], dtype=torch.long)
    atom_species = ["O(HH)", "O(HH)", "N(CC)", None, "S(unmatched)"]
    coords = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.5, 0.0],
            [0.0, 1.2, 0.3],
            [0.5, -1.0, -0.5],
            [-0.8, -0.6, 0.9],
        ]
    )
    coords = coords - coords.mean(dim=0, keepdim=True)

    with pytest.warns(UserWarning, match="falling back"):
        pb = PotentialBuilder(
            n,
            dx,
            atomic_numbers,
            parameterization="shtyrov",
            atom_species=atom_species,
            progressbars=False,
        )
    volume_ours = pb.forward(coords, method="analytic")

    scat_coeffs = []
    for sp, z in zip(atom_species, atomic_numbers.tolist()):
        if sp is not None and sp in table:
            P = table[sp].numpy()
        else:
            c4322 = gemmi.Element(z).c4322
            P = None
        if P is None:
            a, b = list(c4322.a), list(c4322.b)
        else:
            a, b = list(P[:, 0]), list(P[:, 1])
        scat_coeffs.append((a, b))

    mmcif_path = str(tmp_path / "multi_atom.cif")
    _write_multi_atom_shtyrov_mmcif(mmcif_path, elements, coords.tolist(), scat_coeffs)

    gpb = GemmiPotentialBuilder(n_xyz=n, dx=dx, b_factor=0.0)
    volume_gemmi = gpb.build_potential_from_custom_mmcif(mmcif_path)

    corr = torch.corrcoef(torch.stack([volume_ours.flatten(), volume_gemmi.flatten()]))[
        0, 1
    ]
    assert corr.item() > 0.97, f"correlation too low: {corr.item()}"

    total_ours = (volume_ours.sum() * dx**3).item()
    total_gemmi = (volume_gemmi.sum() * dx**3).item()
    assert abs(total_ours - total_gemmi) / abs(total_gemmi) < 0.1, (
        f"total integrated potential differs too much: ours={total_ours}, "
        f"gemmi={total_gemmi}"
    )


def test_potential_builder_analytic_gradient_matches_finite_difference():
    """
    forward(method="analytic") must be differentiable w.r.t. coordinates,
    and the gradient must match a central finite-difference estimate.

    Uses a random-weighted sum as the loss, not a bare .sum(): the plain
    volume sum is (by design, since build_potential_volume_analytic_scatter
    evaluates the exact voxel average) the conserved total integrated
    potential, which is intentionally near-invariant to atom position — its
    gradient is correctly near-zero, which isn't a useful differentiability
    check. A weighted sum breaks that symmetry and actually depends on
    position.
    """
    atomic_numbers = torch.tensor([8, 6], dtype=torch.long)
    atom_species = ["O(HH)", None]
    coords = torch.tensor([[0.3, -0.2, 0.1], [3.1, 0.0, 0.0]])

    pb = PotentialBuilder(
        32,
        1.0,
        atomic_numbers,
        parameterization="shtyrov",
        atom_species=atom_species,
        progressbars=False,
    )

    torch.manual_seed(0)
    weight = torch.randn(32, 32, 32)

    def total_potential(c):
        return (pb.forward(c, method="analytic") * weight).sum()

    c = coords.clone().requires_grad_(True)
    total_potential(c).backward()
    analytic_grad = c.grad[0, 0].item()

    eps = 1e-3
    plus = coords.clone()
    plus[0, 0] += eps
    minus = coords.clone()
    minus[0, 0] -= eps
    fd_grad = (total_potential(plus).item() - total_potential(minus).item()) / (2 * eps)

    assert analytic_grad == pytest.approx(fd_grad, rel=1e-2)


def test_potential_builder_analytic_handles_out_of_bounds_atom():
    """
    An atom placed (partially or fully) outside the grid should be clipped,
    not crash — matching soft_voxelize_coordinates' existing behavior for
    the '2d'/'3d' methods.
    """
    atomic_numbers = torch.tensor([8, 8], dtype=torch.long)
    atom_species = ["O(HH)", "O(HH)"]
    coords = torch.tensor([[0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]])

    pb = PotentialBuilder(
        16,
        1.0,
        atomic_numbers,
        parameterization="shtyrov",
        atom_species=atom_species,
        progressbars=False,
    )
    volume = pb.forward(coords, method="analytic")
    assert torch.isfinite(volume).all()
    assert volume.sum() > 0


def test_potential_builder_analytic_robust_to_subvoxel_position():
    """
    Regression test for the point-sampling aliasing bug: before
    build_potential_volume_analytic_scatter switched to exact erf-based
    voxel-averaging, moving one atom by a fraction of a voxel swung its
    peak value by ~26x (1012 -> 38.6) purely from where it landed relative
    to the grid, with no physical meaning. With proper voxel-averaging:
      - the total integrated potential (sum * dx^3) must stay constant
        regardless of sub-voxel position (matches the analytic total
        c1 * sum(a_i), since voxel-averaging exactly conserves the
        integral under translation);
      - the peak value should vary only mildly with position, not swing
        by an order of magnitude.
    """
    atomic_numbers = torch.tensor([8], dtype=torch.long)
    atom_species = ["O(HH)"]
    n, dx = 16, 1.0

    pb = PotentialBuilder(
        n,
        dx,
        atomic_numbers,
        parameterization="shtyrov",
        atom_species=atom_species,
        progressbars=False,
    )

    a0 = 0.529
    e = 14.4
    c1 = 2 * torch.pi * e * a0
    a_coefs, _ = pb._get_analytic_atom_coefficients()
    expected_total = (c1 * a_coefs[0].sum()).item()

    peaks = []
    for offset in [0.0, 0.1, 0.25, 0.5]:
        coords = torch.tensor([[offset, 0.0, 0.0]])
        volume = pb.forward(coords, method="analytic")
        total = (volume.sum() * dx**3).item()
        assert total == pytest.approx(expected_total, rel=1e-3), (
            f"total at offset={offset} is {total}, expected {expected_total}"
        )
        peaks.append(volume.max().item())

    assert max(peaks) / min(peaks) < 3.0, (
        f"peak varies by {max(peaks) / min(peaks):.1f}x across sub-voxel "
        f"positions ({peaks}) -- should be a mild, smooth variation, not "
        "an aliasing-driven blow-up"
    )


@pytest.mark.parametrize("parameterization", ["kirkland", "lobato"])
def test_potential_builder_analytic_matches_3d_splat(parameterization):
    """
    Cross-check PotentialBuilder(...).forward(method="analytic") for
    Kirkland/Lobato against the existing, production-validated method="3d"
    (splat + convolve) path on the same coordinates -- gemmi has no native
    support for Kirkland/Lobato's Yukawa-type terms, so unlike Shtyrov there
    is no independent external reference tool; method="3d" is the closest
    available same-repo cross-check.

    The two methods use different approximations (splat+convolve smooths
    the true peak; analytic's shell-average slightly overestimates it -- see
    build_potential_volume_analytic_scatter_kirkland/_lobato's docstrings),
    so we check the total integrated potential (a physically robust,
    peak-insensitive quantity) and overall correlation rather than a tight
    per-voxel tolerance, exactly as
    test_potential_builder_analytic_matches_gemmi_multi_atom does for Shtyrov.
    """
    torch.manual_seed(2)
    atomic_numbers = torch.tensor([6, 7, 8, 16, 6, 7], dtype=torch.long)
    n, dx = 24, 1.0
    coords = torch.rand(6, 3) * 8 - 4

    pb = PotentialBuilder(
        n, dx, atomic_numbers, parameterization=parameterization, progressbars=False
    )
    volume_analytic = pb.forward(coords, method="analytic")
    volume_3d = pb.forward(coords, method="3d")

    total_analytic = (volume_analytic.sum() * dx**3).item()
    total_3d = (volume_3d.sum() * dx**3).item()
    assert abs(total_analytic - total_3d) / abs(total_3d) < 0.1, (
        f"total integrated potential differs too much: analytic={total_analytic}, "
        f"3d={total_3d}"
    )

    corr = torch.corrcoef(
        torch.stack([volume_analytic.flatten(), volume_3d.flatten()])
    )[0, 1]
    assert corr.item() > 0.9, f"correlation too low: {corr.item()}"


@pytest.mark.parametrize("parameterization", ["kirkland", "lobato"])
def test_potential_builder_analytic_gradient_matches_finite_difference_kl(
    parameterization,
):
    """
    forward(method="analytic") for Kirkland/Lobato must be differentiable
    w.r.t. coordinates, matching a central finite-difference estimate.

    Regression test for the NaN-gradient bug found and fixed this session:
    yukawa_shell_average/plain_exp_shell_average's far/near torch.where
    branches were each evaluated everywhere (not just where selected), and
    the unselected branch could overflow to inf for out-of-domain inputs,
    corrupting the gradient via 0*inf=nan even though the forward value was
    finite and correct. Uses a random-weighted sum as the loss, not a bare
    .sum(), following the same reasoning as the Shtyrov analytic gradient
    test (Kirkland/Lobato's shell-average is only an approximation, not
    exact, so a bare sum isn't perfectly position-invariant here either --
    but a weighted sum gives a robust, clearly nonzero gradient regardless).
    """
    atomic_numbers = torch.tensor([16, 6], dtype=torch.long)
    coords = torch.tensor([[0.3, -0.2, 0.1], [3.1, 0.0, 0.0]])

    pb = PotentialBuilder(
        32, 1.0, atomic_numbers, parameterization=parameterization, progressbars=False
    )

    torch.manual_seed(0)
    weight = torch.randn(32, 32, 32)

    def total_potential(c):
        return (pb.forward(c, method="analytic") * weight).sum()

    c = coords.clone().requires_grad_(True)
    total_potential(c).backward()
    assert torch.isfinite(c.grad).all(), f"non-finite gradient: {c.grad}"
    analytic_grad = c.grad[0, 0].item()

    eps = 1e-2
    plus = coords.clone()
    plus[0, 0] += eps
    minus = coords.clone()
    minus[0, 0] -= eps
    fd_grad = (total_potential(plus).item() - total_potential(minus).item()) / (2 * eps)

    assert analytic_grad == pytest.approx(fd_grad, rel=1e-2)


@pytest.mark.parametrize("parameterization", ["kirkland", "lobato"])
def test_potential_builder_analytic_handles_out_of_bounds_atom_kl(parameterization):
    """
    An atom placed (partially or fully) outside the grid should be clipped,
    not crash, for Kirkland/Lobato's analytic path too.
    """
    atomic_numbers = torch.tensor([8, 8], dtype=torch.long)
    coords = torch.tensor([[0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]])

    pb = PotentialBuilder(
        16, 1.0, atomic_numbers, parameterization=parameterization, progressbars=False
    )
    volume = pb.forward(coords, method="analytic")
    assert torch.isfinite(volume).all()
    assert volume.sum() > 0


@pytest.mark.parametrize("parameterization", ["kirkland", "lobato"])
def test_potential_builder_analytic_robust_to_subvoxel_position_kl(parameterization):
    """
    Sub-voxel robustness for Kirkland/Lobato's analytic path. Unlike
    Shtyrov/Peng's pure-Gaussian erf treatment (exact voxel average, total
    integral exactly invariant under translation), Kirkland/Lobato's Yukawa
    terms use a sphere-of-equal-volume shell-average -- a bounded
    *approximation*, not an exact integral, so the total is only
    approximately invariant (~2-20% per the derivation's Monte Carlo
    validation, confirmed empirically here at ~11%). The key regression
    guard is that neither the total nor the peak blows up or swings by
    orders of magnitude the way naive point-sampling does.
    """
    atomic_numbers = torch.tensor([8], dtype=torch.long)
    n, dx = 16, 1.0

    pb = PotentialBuilder(
        n, dx, atomic_numbers, parameterization=parameterization, progressbars=False
    )

    totals, peaks = [], []
    for offset in [0.0, 0.1, 0.25, 0.5]:
        coords = torch.tensor([[offset, 0.0, 0.0]])
        volume = pb.forward(coords, method="analytic")
        assert torch.isfinite(volume).all()
        totals.append((volume.sum() * dx**3).item())
        peaks.append(volume.max().item())

    total_spread = (max(totals) - min(totals)) / (sum(totals) / len(totals))
    assert total_spread < 0.2, (
        f"total integral varies by {total_spread * 100:.1f}% across sub-voxel "
        f"positions ({totals}) -- should stay within the shell-average "
        "approximation's known ~20% worst-case bound"
    )
    assert max(peaks) / min(peaks) < 3.0, (
        f"peak varies by {max(peaks) / min(peaks):.1f}x across sub-voxel "
        f"positions ({peaks}) -- should be a mild, smooth variation, not "
        "an aliasing-driven blow-up"
    )


def test_kirkland_lobato_analytic_no_overflow_for_heavy_elements():
    """
    Regression test for a float32 overflow bug found and fixed this session:
    the textbook far/near shell-average formulas each contain a bare
    exp(2*R*lam) factor that, evaluated as its own intermediate value (rather
    than algebraically pre-combined with the accompanying decaying factor),
    overflows to inf for fast-decaying (large lam) terms -- confirmed for
    Kirkland sulfur's (Z=16) b=167 Yukawa term (lam~81/A) and, more severely,
    Kirkland selenium's (Z=34) b=196.8 term (the largest in the whole table).
    Even though this term is masked out by torch.where in the *value* it
    contributes to, the inf still corrupted gradients (0*inf=nan). Sweeps
    every element Z=1..103 in both tables to guard against regressions for
    any future table changes, not just the two elements found empirically.
    """
    n, dx = 12, 1.0
    for z in range(1, 104):
        atomic_numbers = torch.tensor([z])
        for fn in (
            build_potential_volume_analytic_scatter_kirkland,
            build_potential_volume_analytic_scatter_lobato,
        ):
            coords = torch.tensor([[0.3, 0.1, -0.2]], requires_grad=True)
            volume = fn(atomic_numbers, coords, (n, n, n), dx)
            assert torch.isfinite(volume).all(), (
                f"{fn.__name__} Z={z}: non-finite volume"
            )
            volume.sum().backward()
            assert coords.grad is not None and torch.isfinite(coords.grad).all(), (
                f"{fn.__name__} Z={z}: non-finite gradient"
            )


def test_potential_builder_shtyrov_peng_only_analytic():
    """
    forward(method="analytic") for parameterization="shtyrov" with
    atom_species=None entirely (no bonded-species typing at all) should use
    plain per-element Peng c4322 factors for every atom, matching the
    equivalent atom_species=[None]*N list (the pre-existing way to reach a
    pure-Peng fallback for every atom via _get_analytic_atom_coefficients).
    """
    torch.manual_seed(0)
    atomic_numbers = torch.tensor([6, 7, 8, 16])
    n, dx = 20, 1.0
    coords = torch.rand(4, 3) * 5 - 2.5

    pb_none = PotentialBuilder(
        n, dx, atomic_numbers, parameterization="shtyrov", progressbars=False
    )
    volume_none = pb_none.forward(coords, method="analytic")

    with pytest.warns(UserWarning, match="falling back"):
        pb_list = PotentialBuilder(
            n,
            dx,
            atomic_numbers,
            parameterization="shtyrov",
            atom_species=[None] * len(atomic_numbers),
            progressbars=False,
        )
    volume_list = pb_list.forward(coords, method="analytic")

    assert torch.isfinite(volume_none).all()
    assert torch.allclose(volume_none, volume_list)


@pytest.mark.parametrize("parameterization", ["kirkland", "lobato"])
def test_recommended_rcut_scales_with_slowest_element_present(parameterization):
    """
    recommended_rcut should return a small radius for a light-element-only
    (H/C/N/O) structure, and a larger one once a slower-decaying element
    (K) is added -- and should equal the max over whatever's present, not
    e.g. the mean.
    """
    light_only = torch.tensor([1, 6, 6, 7, 8, 8], dtype=torch.long)
    with_potassium = torch.tensor([1, 6, 6, 7, 8, 8, 19], dtype=torch.long)

    r_light = recommended_rcut(light_only, parameterization)
    r_k = recommended_rcut(with_potassium, parameterization)

    assert r_light < 3.0
    assert r_k > r_light
    assert r_k == max(
        recommended_rcut(torch.tensor([z]), parameterization)
        for z in with_potassium.unique().tolist()
    )


def test_recommended_rcut_shtyrov_species_and_peng_fallback():
    """
    recommended_rcut for parameterization="shtyrov" should use the species
    table for matched atoms and the per-element Peng table as a fallback
    for unmatched/None species, taking the max across all atoms present.
    """
    atomic_numbers = torch.tensor([8, 6], dtype=torch.long)

    # O(HH) needs ~3.6A (see _SHTYROV_MIN_RCUT); an unmatched carbon falls
    # back to the Peng table's value for Z=6.
    r = recommended_rcut(atomic_numbers, "shtyrov", atom_species=["O(HH)", None])
    assert r == pytest.approx(3.6)

    # atom_species=None entirely -> every atom uses the Peng table.
    r_no_species = recommended_rcut(atomic_numbers, "shtyrov", atom_species=None)
    assert r_no_species < 3.0  # O and C are both fast-decaying in Peng's table


def test_recommended_rcut_handles_shtyrov_mixed_sign_species():
    """
    Regression test for the mixed-sign-Gaussian-term bug found while
    building this table: Shtyrov's O(HH) has two negative-amplitude terms
    (a=-0.60, b=64.2 and a=-0.15, b=121.4) that decay slower than its
    positive terms, so a naive bisection assuming the captured-integral
    fraction is monotonic in radius converges on a radius far too small
    (found: 0.6A, giving ~17.5% too much total potential in practice).
    The corrected, monotonicity-robust table entry must be large enough
    to actually be safe.
    """
    r = recommended_rcut(torch.tensor([8]), "shtyrov", atom_species=["O(HH)"])
    assert r >= 3.5


def test_potential_builder_default_rcut_is_auto_detected():
    """
    PotentialBuilder's rcut defaults to None, which auto-selects via
    recommended_rcut based on the elements actually present -- not a fixed
    constant. An explicitly-passed rcut must override the auto-detection.
    """
    light_only = torch.tensor([1, 6, 6, 7, 8, 8], dtype=torch.long)
    with_potassium = torch.tensor([1, 6, 6, 7, 8, 8, 19], dtype=torch.long)

    pb_light = PotentialBuilder(
        16, 1.0, light_only, parameterization="kirkland", progressbars=False
    )
    pb_k = PotentialBuilder(
        16, 1.0, with_potassium, parameterization="kirkland", progressbars=False
    )
    assert pb_light.rcut == recommended_rcut(light_only, "kirkland")
    assert pb_k.rcut == recommended_rcut(with_potassium, "kirkland")
    assert pb_light.rcut < pb_k.rcut

    pb_explicit = PotentialBuilder(
        16,
        1.0,
        with_potassium,
        parameterization="kirkland",
        rcut=2.0,
        progressbars=False,
    )
    assert pb_explicit.rcut == 2.0


def test_potential_builder_defaults_to_shtyrov_analytic():
    """
    PotentialBuilder's parameterization defaults to 'shtyrov' and
    forward()'s method defaults to 'analytic' -- constructing and calling
    with no explicit overrides should just work (falling back to plain
    per-element Peng factors, with no warning, since no atom_species was
    given at all -- see test_potential_builder_shtyrov_peng_only_analytic
    for the atom_species=None-vs-explicit-None-list distinction).
    """
    atomic_numbers = torch.tensor([6, 7, 8, 16], dtype=torch.long)
    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pb = PotentialBuilder(16, 1.0, atomic_numbers, progressbars=False)
        assert pb.parameterization == "shtyrov"
        volume = pb.forward(coords)  # no explicit method
    assert torch.isfinite(volume).all()
    assert volume.sum() > 0


def test_potential_builder_analytic_rejects_periodic():
    """
    method="analytic" (now the default) does not implement periodic
    boundary wrapping, unlike '2d'/'3d' -- must raise a clear error rather
    than silently ignoring periodic=True and producing wrong results.
    """
    atomic_numbers = torch.tensor([6], dtype=torch.long)
    coords = torch.tensor([[0.0, 0.0, 0.0]])
    pb = PotentialBuilder(16, 1.0, atomic_numbers, periodic=True, progressbars=False)
    with pytest.raises(ValueError, match="periodic"):
        pb.forward(coords)
    with pytest.raises(ValueError, match="periodic"):
        pb.forward(coords, method="analytic")
    # '3d' should still work fine with periodic=True.
    volume = pb.forward(coords, method="3d")
    assert torch.isfinite(volume).all()


@pytest.mark.parametrize("parameterization", ["kirkland", "lobato"])
def test_potential_builder_kernels_match_standalone_builder(parameterization):
    """
    PotentialBuilder's per-element kernels are the standalone helper's output.

    The ice and gold-bead generators call `build_atomic_potential_kernel`
    directly for their single fixed species, while PotentialBuilder assembles
    a stack of them for a structure. Both used to construct the kernel with
    their own copy of the supersample -> sample -> bin recipe; this pins them
    to one implementation so they cannot silently diverge again.
    """
    atomic_numbers = torch.tensor([6, 7, 8, 16])
    pb = PotentialBuilder(
        32, 1.0, atomic_numbers, parameterization=parameterization, progressbars=False
    )
    for i, z in enumerate(pb.unique_elements.tolist()):
        direct = build_atomic_potential_kernel(
            1.0, parameterization=parameterization, atomic_number=int(z)
        )
        assert torch.equal(pb.atomic_potentials_3d[i], direct), (
            f"{parameterization} Z={int(z)} kernel differs between "
            "PotentialBuilder and build_atomic_potential_kernel"
        )


def test_build_atomic_potential_kernel_rejects_unknown_parameterization():
    with pytest.raises(ValueError, match="Unknown parameterization"):
        build_atomic_potential_kernel(1.0, parameterization="nonesuch")


@pytest.mark.parametrize("dx", [1.0, 1.5])
def test_conv_backends_agree_including_even_kernels(dx):
    """
    'fftconvolve' and 'conv3d' must produce the same potential.

    conv3d used to be F.conv3d(padding="same"), which pads asymmetrically
    relative to fftconvolve's centered crop whenever the kernel has an even
    number of voxels -- shifting the potential half a voxel off the
    coordinates it was built from. dx=1.5 gives a 4-voxel kernel and used to
    disagree by ~3.5 (against a signal max of ~4.3); dx=1.0 gives a 5-voxel
    kernel and always agreed. Even kernels occur for 20 of the 36 pixel sizes
    between 0.5 and 4.0 A, so this is the common case, not the exotic one.
    """
    torch.manual_seed(0)
    atomic_numbers = torch.tensor([6, 7, 8, 8, 7, 6])
    coords = (torch.rand(6, 3) - 0.5) * 20

    out = {}
    for backend in ("fftconvolve", "conv3d"):
        pb = PotentialBuilder(
            24,
            dx,
            atomic_numbers,
            parameterization="kirkland",
            conv_backend=backend,
            progressbars=False,
        )
        out[backend] = pb(coords, method="3d")

    torch.testing.assert_close(out["fftconvolve"], out["conv3d"], atol=1e-4, rtol=1e-4)


def test_potential_from_deltas_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        potential_from_deltas(
            torch.zeros(1, 4, 4, 4), torch.ones(3, 3, 3), backend="nope"
        )


@pytest.mark.parametrize("dx", [1.0, 1.5])
def test_2d_and_3d_methods_agree_on_density_position(dx):
    """
    method='2d' must place density at the same in-plane position as '3d'.

    '2d' is a projection approximation, so the two differ in value -- but not
    in *where* the density sits. Its convolution used to be
    F.conv2d(padding="same"), which pads asymmetrically for an even-sized
    kernel, offsetting the whole projected potential half a voxel from the
    coordinates it was built from. dx=1.5 gives a 4-voxel kernel and used to
    land a full pixel off by this measure; dx=1.0 gives 5 voxels and always
    agreed.
    """
    torch.manual_seed(0)
    atomic_numbers = torch.tensor([6, 7, 8, 8, 7, 6])
    coords = (torch.rand(6, 3) - 0.5) * 20
    pb = PotentialBuilder(
        24, dx, atomic_numbers, parameterization="kirkland", progressbars=False
    )

    proj_2d = pb(coords, method="2d").sum(0)
    proj_3d = pb(coords, method="3d").sum(0)

    def centroid(m):
        return torch.nonzero(m > 0.5 * m.max()).float().mean(0)

    assert torch.allclose(centroid(proj_2d), centroid(proj_3d), atol=1e-6)


# ---------------------------------------------------------------------------
# The lazily-built analytic coefficient cache must follow the module's device
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_analytic_coefs_cache_follows_device():
    """
    A builder used on CPU and then moved to GPU must keep working.

    `_analytic_coefs` is a plain attribute, not a registered buffer, so
    `nn.Module.to()` cannot migrate it. Without an explicit device check the
    second call returns CPU coefficients and dies inside `forward`. Anything
    that rebuilds the potential every forward pass -- notably
    `ImageGeneratorFromCoordinates` -- hits this on every GPU run.
    """
    torch.manual_seed(0)
    coords = torch.randn(50, 3) * 5.0
    atomic_numbers = torch.full((50,), 6, dtype=torch.long)

    builder = PotentialBuilder(32, 2.0, atomic_numbers, progressbars=False)
    on_cpu = builder(coords)
    assert builder._analytic_coefs is not None  # populated by the first call

    builder = builder.to("cuda")
    on_gpu = builder(coords.to("cuda"))

    assert builder._analytic_coefs[0].device.type == "cuda"
    assert torch.allclose(on_cpu, on_gpu.cpu(), atol=1e-4)


def test_potential_kernel_cache_is_transparent_and_isolated():
    """
    Reusing a cached potential kernel is invisible to callers.

    Specimen assembly builds one `PotentialBuilder` per species, so the same few
    kernels get rebuilt dozens of times -- measured 321 builds of 29 distinct
    kernels on the default tomogram config. The cache keys on everything a
    kernel depends on (`dx`, parameterization, element/species), so two builders
    that agree on those must produce identical stacks.

    Also pins that the cache hands back a COPY: a caller that mutates its own
    `atomic_potentials_3d` (or moves it to a device) must not corrupt the entry
    every later builder reads.
    """
    import torch

    from specter.potential import PotentialBuilder
    from specter.potential._potential_builder import _KERNEL_CACHE

    znums = torch.tensor([6, 7, 8, 16] * 10)
    kwargs = dict(
        n_xyz=24,
        dx=2.0,
        atomic_numbers=znums,
        progressbars=False,
        parameterization="kirkland",
    )

    _KERNEL_CACHE.clear()
    first = PotentialBuilder(**kwargs)
    assert len(_KERNEL_CACHE) == 4, "one entry per unique element"
    cold = first.atomic_potentials_3d.clone()

    second = PotentialBuilder(**kwargs)
    assert torch.equal(second.atomic_potentials_3d, cold)

    # Mutating a builder's own stack must not reach through to the cache.
    second.atomic_potentials_3d.zero_()
    third = PotentialBuilder(**kwargs)
    assert torch.equal(third.atomic_potentials_3d, cold)

    # A different voxel size is a different kernel, not a cache hit.
    other = PotentialBuilder(**{**kwargs, "dx": 1.0})
    assert len(_KERNEL_CACHE) == 8
    assert not torch.equal(other.atomic_potentials_3d, cold)


def test_potential_kernel_cache_separates_parameterizations():
    """
    Kernels for the same element under different parameterizations stay distinct.

    The cheapest way to get this cache wrong is to key it on the element alone,
    which would silently serve Kirkland kernels to a Lobato builder -- a physics
    error that no shape or dtype check would catch.
    """
    import torch

    from specter.potential import PotentialBuilder
    from specter.potential._potential_builder import _KERNEL_CACHE

    znums = torch.tensor([6, 8])
    _KERNEL_CACHE.clear()
    kirkland = PotentialBuilder(
        n_xyz=24,
        dx=2.0,
        atomic_numbers=znums,
        progressbars=False,
        parameterization="kirkland",
    ).atomic_potentials_3d.clone()
    lobato = PotentialBuilder(
        n_xyz=24,
        dx=2.0,
        atomic_numbers=znums,
        progressbars=False,
        parameterization="lobato",
    ).atomic_potentials_3d.clone()

    assert not torch.allclose(kirkland, lobato)
