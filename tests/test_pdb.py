"""
Tests for PDB parsing, in particular bonded-species atom typing.
"""

import gzip
from pathlib import Path

import pytest

import specter.pdb as pdb_module
from specter.pdb import PDB

# Myoglobin (oxy form): standard amino acids, a heme group coordinated by a
# His side chain and a bound O2 — exercises intra-residue bonds
# (_chem_comp_bond), the carboxyl/amide flag override, and inter-residue
# metal-coordination bonds (_struct_conn), all without needing an external
# CCP4 Monomer Library.
_FIXTURE = Path(__file__).parent.parent / "specter-data" / "pdb" / "1mbo.cif"


@pytest.fixture(scope="module")
def species():
    return PDB.get_atom_species(str(_FIXTURE), verbose=False)


def _atom_species_by_name(structure_species, filepath, resname, resseq, atom_name):
    """
    Look up the computed species string for a specific (residue, atom).

    Mirrors get_atom_species's altloc de-duplication (first-seen wins) so
    this stays index-aligned with the list it's checking against.
    """
    import gemmi

    st = gemmi.read_structure(str(filepath))
    st.setup_entities()

    names = []
    seen = set()
    for cra in st[0].all():
        if cra.atom.occ == 0.0:
            continue
        key = (cra.chain.name, cra.residue.seqid.num or 0, cra.atom.name)
        if key in seen:
            continue
        seen.add(key)
        names.append((cra.residue.name, cra.residue.seqid.num, cra.atom.name))

    for (resn, seq, atn), sp in zip(names, structure_species):
        if resn == resname and seq == resseq and atn == atom_name:
            return sp
    raise AssertionError(f"Atom {resname}{resseq}:{atom_name} not found")


class _FakeResponse:
    def __init__(self, content: bytes = b"", json_data: dict | None = None):
        self.content = content
        self._json_data = json_data or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def test_fetch_pdb_file_creates_missing_savefolder(tmp_path, monkeypatch):
    # Regression test: savefolder is a plain filesystem path (resolved
    # against the caller's own cwd, not the specter repo root), and
    # fetch_pdb_file used to crash with a raw FileNotFoundError from
    # open(file_path, "w") if that directory didn't already exist, rather
    # than creating it on demand.
    fake_cif = "data_1ABC\n#\n"
    compressed = gzip.compress(fake_cif.encode())

    def fake_get(url, *args, **kwargs):
        if "rest/v1/core/entry" in url:
            return _FakeResponse(
                json_data={"rcsb_entry_container_identifiers": {"assembly_ids": []}}
            )
        return _FakeResponse(content=compressed)

    monkeypatch.setattr(pdb_module.requests, "get", fake_get)

    missing_dir = tmp_path / "not" / "yet" / "created"
    assert not missing_dir.exists()

    filepath = PDB.fetch_pdb_file("1abc", savefolder=str(missing_dir), verbose=False)

    assert missing_dir.exists()
    assert Path(filepath).read_text() == fake_cif


def test_get_atom_species_requires_mmcif(tmp_path):
    """A legacy .pdb source has no chem-comp bond dictionary; species is None."""
    fake_pdb = tmp_path / "fake.pdb"
    fake_pdb.write_text(
        "ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "END\n"
    )
    with pytest.warns(UserWarning):
        result = PDB.get_atom_species(str(fake_pdb), verbose=False)
    assert result == [None]


def test_carboxyl_oxygen_flag(species):
    """Asp OD1/OD2 side-chain oxygens get the ', carboxyl' species suffix."""
    sp = _atom_species_by_name(species, _FIXTURE, "ASP", 20, "OD1")
    assert sp is not None
    assert sp.startswith("O(")
    assert sp.endswith(", carboxyl)")


def test_heme_iron_coordination(species):
    """
    Heme Fe shows up bonded to porphyrin N atoms (+ His/O2 axial ligands),
    confirming _struct_conn metal-coordination bonds are picked up.

    This is the no-monomer-library path, where those links are kept: without a
    library the porphyrin Fe-N bonds are not fully defined by the file's own
    _chem_comp_bond, so dropping the links would leave Fe(NNN) here and Fe(N)
    for 1A6M. See test_metal_links_dropped_with_library for the other path.
    """
    sp = _atom_species_by_name(species, _FIXTURE, "HEM", 155, "FE")
    assert sp is not None
    assert sp.startswith("Fe(")
    assert "N" in sp

    # matches the bundled Shtyrov species table's Fe environment format
    from specter.atom import load_shtyrov_species_parameters
    from importlib import resources

    path = resources.files("specter.atom_data").joinpath("params_cat.json")
    with resources.as_file(path) as fpath:
        table = load_shtyrov_species_parameters(str(fpath))
    assert "Fe(NNNN)" in table  # the 4-coordinate deoxy-heme entry exists


def test_unresolved_atoms_are_none(species):
    """Atoms whose component/bonds can't be resolved fall back to None, not an error."""
    assert any(s is None for s in species)


def test_species_aligns_with_atomic_numbers():
    """atom_species (opt-in) is the same length as atomic_numbers/coordinates."""
    pdb = PDB(str(_FIXTURE), verbose=False, compute_atom_species=True)
    assert pdb.atom_species is not None
    assert len(pdb.atom_species) == pdb.atomic_numbers.shape[0]
    assert len(pdb.atom_species) == pdb.coordinates.shape[0]


# Beta-amylase: 208 of its deposited atoms carry occupancy 0.0, which is also
# the marker get_atom_species stamps on the dummy hydrogens it adds while
# building the bond graph. Typing once dropped the deposited ones too, so
# atom_species came back 208 entries short of atomic_numbers and the Shtyrov
# potential path died on the length mismatch.
_ZERO_OCC_FIXTURE = (
    Path(__file__).parent.parent / "specter-data" / "pdb" / "1fa2-assembly1.cif"
)


@pytest.mark.skipif(
    not _ZERO_OCC_FIXTURE.exists(), reason="1fa2 not in the local PDB cache"
)
def test_zero_occupancy_atoms_are_still_typed():
    """Deposited zero-occupancy atoms are not mistaken for added dummy atoms."""
    import gemmi

    st = gemmi.read_structure(str(_ZERO_OCC_FIXTURE))
    st.setup_entities()
    assert sum(1 for cra in st[0].all() if cra.atom.occ == 0.0) > 0, (
        "fixture no longer has zero-occupancy atoms; pick another structure"
    )

    pdb = PDB(str(_ZERO_OCC_FIXTURE), verbose=False, compute_atom_species=True)
    assert pdb.atom_species is not None
    assert len(pdb.atom_species) == pdb.atomic_numbers.shape[0]


# Each of these once returned an atom_species list of the wrong length, which
# surfaced far downstream as an opaque IndexError inside the Shtyrov potential
# builder (`shape of the mask [N] does not match the indexed tensor [M]`) --
# and shtyrov is the default parameterization, so these structures could not be
# rendered at all. One entry per distinct root cause:
#   3DY4  residues numbered 10, 10A, 10B... -- the atom key ignored the
#         insertion code, so distinct atoms collided and were dropped as
#         though they were alternate conformers.
#   7a4m  carries explicit hydrogens -- HydrogenChange.ReAdd strips every
#         existing H before re-adding it from monomer-library geometry, so
#         without a library the strip happened and the re-add did not.
_ALIGNMENT_FIXTURES = ["3DY4-assembly1.cif", "7a4m.cif"]


@pytest.mark.parametrize("name", _ALIGNMENT_FIXTURES)
def test_atom_species_aligns_for_awkward_structures(name):
    """atom_species stays index-aligned with atomic_numbers, element for element."""
    from specter.atom.atom import atom_symbol

    path = Path(__file__).parent.parent / "specter-data" / "pdb" / name
    if not path.exists():
        pytest.skip(f"{name} not in the local PDB cache")

    pdb = PDB(str(path), verbose=False, compute_atom_species=True)
    assert pdb.atom_species is not None
    assert len(pdb.atom_species) == pdb.atomic_numbers.shape[0]

    # Equal lengths alone would still pass if the two lists were shifted
    # relative to each other, so check the elements agree entry by entry.
    mismatched = [
        i
        for i, (z, s) in enumerate(zip(pdb.atomic_numbers.tolist(), pdb.atom_species))
        if s is not None and str(atom_symbol(int(z))).upper() != s.split("(")[0].upper()
    ]
    assert not mismatched, f"{len(mismatched)} atoms typed as the wrong element"


# A monomer library is what supplies the hydrogens a deposited structure omits,
# and without it every H-containing Shtyrov species ("C(HHHC)", "O(HH)") misses
# the table and falls back to per-element Peng. Typing then covers ~56% of a
# protein's atoms instead of ~99%. These tests need a real library, so they are
# skipped wherever one is not configured.
def _monomer_library() -> str | None:
    from os import environ

    env = environ.get("CLIBD_MON")
    if env and Path(env).is_dir():
        return env
    bundled = Path.home() / "sffit" / "monomers"
    return str(bundled) if bundled.is_dir() else None


@pytest.mark.skipif(_monomer_library() is None, reason="no monomer library available")
def test_monomer_library_adds_hydrogens_and_stays_aligned():
    """With a library, all three arrays come from the H-completed model."""
    monlib = _monomer_library()

    plain = PDB(str(_FIXTURE), verbose=False, compute_atom_species=True)
    assert (plain.atomic_numbers == 1).sum().item() == 0, (
        "fixture already carries hydrogens; pick one without them"
    )

    completed = PDB(
        str(_FIXTURE),
        verbose=False,
        compute_atom_species=True,
        monomer_library_path=monlib,
    )
    n_h = (completed.atomic_numbers == 1).sum().item()
    assert n_h > 0, "library did not add any hydrogens"

    # The three arrays are taken from one iteration, so they cannot disagree.
    assert len(completed.atom_species) == completed.atomic_numbers.shape[0]
    assert completed.coordinates.shape[0] == completed.atomic_numbers.shape[0]
    assert completed.atomic_numbers.shape[0] > plain.atomic_numbers.shape[0]


@pytest.mark.skipif(_monomer_library() is None, reason="no monomer library available")
def test_ambiguous_hydrogens_are_not_rendered():
    """Zero-occupancy H (rotatable -OH, His tautomers) must not reach the model.

    PotentialBuilder applies no occupancy weighting, so keeping them would
    render both tautomer hydrogens of every histidine. sffit selects ";q>0"
    for the same reason.
    """
    import gemmi

    monlib = _monomer_library()
    st = gemmi.read_structure(str(_FIXTURE))
    st.setup_entities()
    lib = gemmi.read_monomer_lib(monlib, st[0].get_all_residue_names())
    gemmi.prepare_topology(st, lib, h_change=gemmi.HydrogenChange.ReAdd)
    n_zero = sum(1 for cra in st[0].all() if cra.atom.occ == 0.0)
    assert n_zero > 0, "fixture no longer exercises the ambiguous-hydrogen path"

    completed = PDB(
        str(_FIXTURE),
        verbose=False,
        compute_atom_species=True,
        monomer_library_path=monlib,
    )
    n_kept = completed.atomic_numbers.shape[0]
    n_all = sum(1 for _ in st[0].all())
    assert n_kept <= n_all - n_zero, (
        f"{n_all - n_zero - n_kept} ambiguous hydrogens were not excluded"
    )


@pytest.mark.skipif(_monomer_library() is None, reason="no monomer library available")
def test_metal_links_dropped_with_library():
    """With a library, the heme iron types as Fe(NNNN) -- an entry in the table.

    sffit deletes ConnectionType.MetalC before typing, so the iron keeps only
    the four porphyrin nitrogens its HEM component defines rather than also
    picking up the axial His/O2 ligands. Keeping them gives Fe(NNNNNO...),
    which no fitted table contains, so the iron would fall back to Peng.
    """
    from specter.pdb import PDB as _PDB

    _, _, species, used = _PDB._build_typed_model(
        str(_FIXTURE), _monomer_library(), False
    )
    assert used, "library did not load"
    iron = [s for s in species if s is not None and s.startswith("Fe(")]
    assert iron == ["Fe(NNNN)"], iron
