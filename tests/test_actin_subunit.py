"""F-actin subunits lifted out of a deposited filament, in the filament frame.

The offline tests run the extraction on a synthetic filament whose screw,
axis and polarity are known exactly: copies of one rigid, asymmetric
"subunit" with actin residue numbering, related by F-actin's screw about an
oblique axis. The two tests marked ``network`` run it on the deposited
filament `ACTIN_SPEC` is built from.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import gemmi
import numpy as np
import pytest
import roma
import torch

from specter.specimen.filament import (
    ACTIN_SPEC,
    FilamentInstance,
    FilamentSpec,
    extract_actin_subunit,
    measure_actin_filament,
    place_filaments,
)
from specter.specimen.filament._actin import (
    _actin_chains,
    _fit_actin_filament,
    _kabsch,
    _source_cif,
)
from specter.specimen.filament._placement import subunit_centre_in_filament_frame
from specter.specimen.tomogram import TomogramSpecimenGenerator
from conftest import seeded

_TWIST_DEG = -166.6
_RISE = 27.5
#: The synthetic subunit's centroid sits this far from its filament axis.
_RADIUS = 15.7
#: Residues of actin's subdomains 2 and 4, which face the pointed end.
_POINTED = list(range(33, 70)) + list(range(181, 270))


def _screw_matrix(k: int) -> tuple[np.ndarray, np.ndarray]:
    """``k`` steps of the genetic-helix screw about +Z: rotation, translation."""
    angle = math.radians(k * _TWIST_DEG)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array(
        [0.0, 0.0, k * _RISE]
    )


def _synthetic_filament(path: Path, n_subunits: int = 5) -> dict:
    """Write an mmCIF of a synthetic F-actin filament; return its ground truth.

    Each subunit is 375 CA atoms scattered through an elongated blob, with
    subdomains 2 and 4 displaced towards the pointed end (-Z of the true
    filament frame). The filament is then rotated and shifted to an
    arbitrary place in the deposited frame, and its chains are named out
    of axial order, so neither the axis nor the chain names give the answer
    away.
    """
    rng = np.random.default_rng(0)
    residues = np.arange(1, 376)
    local = rng.normal(size=(375, 3)) * np.array([11.0, 8.0, 6.0])
    local[np.isin(residues, _POINTED), 2] -= 8.0
    local -= local.mean(axis=0)
    # Subunit 0 in the true filament frame: centre at (_RADIUS, 0, 0).
    subunit = local + np.array([_RADIUS, 0.0, 0.0])

    lab_rotation = roma.rotvec_to_rotmat(
        torch.tensor([0.4, -1.1, 0.7], dtype=torch.float64)
    ).numpy()
    lab_shift = np.array([120.0, -40.0, 75.0])

    st = gemmi.Structure()
    model = gemmi.Model(1)
    names = ["C", "E", "A", "D", "B"][:n_subunits]
    chains_true = {}
    for k, name in enumerate(names):
        rotation, translation = _screw_matrix(k)
        xyz = (subunit @ rotation.T + translation) @ lab_rotation.T + lab_shift
        chains_true[name] = xyz
        chain = gemmi.Chain(name)
        for num, (x, y, z) in zip(residues, xyz):
            res = gemmi.Residue()
            res.name = "ALA"
            res.seqid = gemmi.SeqId(int(num), " ")
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(x), float(y), float(z))
            atom.occ = 1.0
            res.add_atom(atom)
            chain.add_residue(res)
        model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    doc = st.make_mmcif_document()
    block = doc.sole_block()
    ids = [row.str(0) for row in block.find("_entity.", ["id"])]
    block.set_mmcif_category(
        "_entity.",
        {
            "id": ids,
            "type": ["polymer"] * len(ids),
            "pdbx_description": ["Actin, alpha skeletal muscle"] * len(ids),
        },
    )
    doc.write_file(str(path))
    return {
        "chains": chains_true,  # chain name -> xyz, in subunit order along +Z
        "order": names,
        "subunit": subunit,
        "lab_rotation": lab_rotation,
        "lab_shift": lab_shift,
    }


@pytest.fixture
def synthetic_filament(tmp_path: Path) -> tuple[Path, dict]:
    path = tmp_path / "synthetic_factin.cif"
    return path, _synthetic_filament(path)


def _subunit_xyz(path: str) -> np.ndarray:
    st = gemmi.read_structure(path)
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for res in st[0][0] for a in res])


def test_fit_recovers_the_screw_axis_and_polarity(synthetic_filament):
    # mmCIF stores coordinates to 1e-3 A, which bounds every tolerance here.
    path, truth = synthetic_filament
    measured = measure_actin_filament(str(path))
    assert measured["twist_deg"] == pytest.approx(_TWIST_DEG, abs=1e-3)
    assert measured["rise"] == pytest.approx(_RISE, abs=1e-3)
    assert measured["radius"] == pytest.approx(_RADIUS, abs=1e-3)
    assert measured["n_subunits"] == 5

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    fit = _fit_actin_filament(_actin_chains(st, str(path), "synthetic"), "synthetic")
    # Pointed end first, whatever the chains are called.
    assert fit.chains == truth["order"]
    barbed = truth["lab_rotation"] @ np.array([0.0, 0.0, 1.0])
    assert fit.barbed @ barbed == pytest.approx(1.0, abs=1e-6)


def test_extracted_subunit_is_in_the_filament_frame(synthetic_filament, tmp_path):
    """Centre at (r, 0, 0), +Z to the barbed end, and the shape untouched."""
    path, truth = synthetic_filament
    out = extract_actin_subunit(str(path), pdb_cache_dir=tmp_path / "cache")
    xyz = _subunit_xyz(out)

    np.testing.assert_allclose(xyz.mean(axis=0), [_RADIUS, 0.0, 0.0], atol=1e-3)
    pointed = np.isin(np.arange(1, 376), _POINTED)
    assert xyz[pointed, 2].mean() < xyz[~pointed, 2].mean() - 5.0

    # The central subunit sits two screw steps up the filament; re-expressed
    # about its own height and azimuth it is the subunit as built.
    np.testing.assert_allclose(xyz, truth["subunit"], atol=1e-3)


def test_cached_subunit_is_reused_and_refreshed_with_its_source(
    synthetic_filament, tmp_path
):
    path, _ = synthetic_filament
    cache = tmp_path / "cache"
    out = extract_actin_subunit(str(path), pdb_cache_dir=cache)
    stamp = Path(out).stat().st_mtime_ns
    assert extract_actin_subunit(str(path), pdb_cache_dir=cache) == out
    assert Path(out).stat().st_mtime_ns == stamp, "a current cache was rewritten"

    # A stale version stamp is re-extracted rather than reused.
    text = Path(out).read_text().splitlines(keepends=True)
    Path(out).write_text("data_old_block\n" + "".join(text[1:]))
    extract_actin_subunit(str(path), pdb_cache_dir=cache)
    assert Path(out).read_text().splitlines()[0].endswith("_v1")


def test_straight_filament_rebuilds_the_synthetic_one(synthetic_filament, tmp_path):
    """Placing the extracted subunit by the real placement code rebuilds
    every subunit of the source filament, up to one rigid motion."""
    path, truth = synthetic_filament
    out = extract_actin_subunit(str(path), pdb_cache_dir=tmp_path / "cache")
    template = _subunit_xyz(out)

    spec = FilamentSpec(
        code=str(path),
        step=_RISE,
        flex_deg=0.0,
        twist_deg=_TWIST_DEG,
        n_monomers=5,
        frame="actin_filament",
    )
    instances = place_filaments([spec], (100, 100, 100), 5.0, generator=seeded(4))
    rendered = np.concatenate(
        [
            inst.position_xyz.double().numpy()
            + template @ inst.rotation_matrix.double().numpy().T
            for inst in instances
        ]
    )
    deposited = np.concatenate([truth["chains"][n] for n in truth["order"]])
    rotation, translation = _kabsch(rendered, deposited)
    rmsd = np.sqrt(
        ((rendered @ rotation.T + translation - deposited) ** 2).sum(1).mean()
    )
    # float32 positions and rotations along a 110 A filament.
    assert rmsd < 2e-3


def test_filament_spec_rejects_an_unknown_frame():
    with pytest.raises(ValueError, match="frame"):
        FilamentSpec(code="1J6Z", step=27.5, flex_deg=12.0, frame="actin")


def test_actin_filament_copies_render_off_the_axis(synthetic_filament, tmp_path):
    """A copy's density sits where the subunit sits, ``r`` off the axis.

    The instance's position is a point on the filament axis. Rendering the
    subunit centred on that point, as an isolated monomer is, would put its
    density 15.7 A from where the source filament has it. Compared by
    centre of mass, which the template's trilinear rotation preserves; the
    insertion rounds a copy's centre to the nearest voxel, so the two can
    differ by up to half a voxel diagonal.
    """
    from specter.potential import PotentialBuilder

    path, _ = synthetic_filament
    n, voxel_size = 64, 2.0
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=(n, n, n),
        voxel_size=voxel_size,
        protein_specs=[],
        filament_specs=[
            FilamentSpec(
                code=str(path),
                step=_RISE,
                flex_deg=0.0,
                twist_deg=_TWIST_DEG,
                frame="actin_filament",
            )
        ],
        pdb_cache_dir=str(tmp_path / "cache"),
        parameterization="kirkland",
        progressbars=False,
    )
    rotation = roma.rotvec_to_rotmat(torch.tensor([0.9, -0.4, 0.3]))
    instance = FilamentInstance(
        code=str(path),
        filament_id=0,
        monomer_index=0,
        position_xyz=torch.full((3,), n * voxel_size / 2),
        rotation_matrix=rotation,
    )
    volume, _, _ = gen._render_filament_instances(
        torch.zeros((n, n, n)),
        torch.zeros((n, n, n), dtype=torch.int32),
        1,
        voxel_size,
        [instance],
    )

    subunit = torch.tensor(
        _subunit_xyz(
            extract_actin_subunit(str(path), pdb_cache_dir=tmp_path / "cache")
        ),
        dtype=torch.float32,
    )
    builder = PotentialBuilder(
        n_xyz=n,
        dx=voxel_size,
        atomic_numbers=torch.full((len(subunit),), 6),
        progressbars=False,
        parameterization="kirkland",
    )
    zyx = torch.meshgrid(*(torch.arange(n, dtype=torch.float64),) * 3, indexing="ij")
    xyz = torch.stack(zyx[::-1], dim=-1) * voxel_size

    def centre_of_mass(density: torch.Tensor) -> torch.Tensor:
        density = density.double()
        return (density[..., None] * xyz).sum(dim=(0, 1, 2)) / density.sum()

    rendered = centre_of_mass(volume)
    off_axis = centre_of_mass(builder.forward(subunit @ rotation.T, method="analytic"))
    on_axis = centre_of_mass(
        builder.forward((subunit - subunit.mean(dim=0)) @ rotation.T, method="analytic")
    )
    assert float((rendered - off_axis).norm()) < math.sqrt(3) / 2 * voxel_size + 0.5
    assert float((rendered - on_axis).norm()) > 12.0


def test_subunit_centre_undoes_pdb_centring(synthetic_filament, tmp_path):
    """`PDB` re-centres the subunit; the centre it removed is recovered."""
    from specter.pdb import PDB

    path, _ = synthetic_filament
    out = extract_actin_subunit(str(path), pdb_cache_dir=tmp_path / "cache")
    pdb = PDB(out, pdb_cache_dir=str(tmp_path / "cache"), verbose=False)
    centre = subunit_centre_in_filament_frame(out, pdb.coordinates, pdb.atomic_numbers)
    np.testing.assert_allclose(centre.numpy(), [_RADIUS, 0.0, 0.0], atol=1e-3)


# --------------------------------------------------------------------------
# Against the deposited filament ACTIN_SPEC's subunit comes from
# --------------------------------------------------------------------------
@pytest.mark.network
def test_actin_spec_matches_its_source_filament(tmp_path):
    """ACTIN_SPEC's screw is the source's own, and its subunits sit off the
    axis the way F-actin's do."""
    measured = measure_actin_filament(ACTIN_SPEC.code, pdb_cache_dir=tmp_path)
    assert measured["twist_deg"] == pytest.approx(ACTIN_SPEC.twist_deg, abs=0.2)
    assert measured["rise"] == pytest.approx(ACTIN_SPEC.step, abs=0.1)
    assert 15.0 < measured["radius"] < 16.5
    assert 75.0 < measured["principal_axis_angle_deg"] < 85.0


@pytest.mark.network
def test_straight_actin_filament_superposes_on_the_deposited_one(tmp_path):
    """Five subunits placed by `ACTIN_SPEC` on a straight path superpose on
    the five of the deposited filament.

    Measured on 7BT7: CA RMSD 0.31 A over all five, every centre within
    0.07 A and every orientation within 0.21 deg. The residual is the
    deposited model's own: its chains were refined independently and differ
    from one another by as much (0.6-0.75 A all-atom).
    """
    out = extract_actin_subunit(ACTIN_SPEC.code, pdb_cache_dir=tmp_path)
    sub = gemmi.read_structure(out)
    template = {
        res.seqid.num: np.array(res.find_atom("CA", "*").pos.tolist())
        for res in sub[0][0].get_polymer()
    }

    cif = _source_cif(ACTIN_SPEC.code, str(tmp_path), False)
    st = gemmi.read_structure(cif)
    st.setup_entities()
    chains = _actin_chains(st, cif, ACTIN_SPEC.code)
    fit = _fit_actin_filament(chains, ACTIN_SPEC.code)

    spec = replace(ACTIN_SPEC, n_copies=1, n_monomers=len(fit.chains), flex_deg=0.0)
    instances = place_filaments([spec], (200, 200, 200), 5.0, generator=seeded(3))
    rendered, deposited = [], []
    for inst, name in zip(instances, fit.chains):
        keys = sorted(set(template) & set(chains[name]))
        position = inst.position_xyz.double().numpy()
        rotation = inst.rotation_matrix.double().numpy()
        rendered.append(np.stack([position + rotation @ template[k] for k in keys]))
        deposited.append(np.stack([chains[name][k] for k in keys]))

    rotation, translation = _kabsch(np.concatenate(rendered), np.concatenate(deposited))
    fitted = [x @ rotation.T + translation for x in rendered]
    rmsd = np.sqrt(
        ((np.concatenate(fitted) - np.concatenate(deposited)) ** 2).sum(1).mean()
    )
    assert rmsd < 0.5
    for x, y in zip(fitted, deposited):
        assert np.linalg.norm(x.mean(axis=0) - y.mean(axis=0)) < 0.2
        r_i, _ = _kabsch(x, y)
        angle = math.degrees(math.acos(min(1.0, (np.trace(r_i) - 1) / 2)))
        assert angle < 0.5
