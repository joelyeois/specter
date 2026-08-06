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
_FIXTURE = Path(__file__).parent.parent / "pdb-data" / "1mbo.cif"


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
    Heme Fe should show up bonded to porphyrin N atoms (+ His/O2 axial
    ligands), confirming _struct_conn metal-coordination bonds are picked up.
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
