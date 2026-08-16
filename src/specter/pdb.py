from __future__ import annotations

import gzip
import io
import os
import sys
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import gemmi
import numpy as np
import requests
import torch
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning  # correct import
from .progress import track
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure

from .atom import atom_number
from .config import default_pdb_cache_dir

# `specter-data/pdb`, relative to the caller's cwd -- one rule shared
# with every other path in a specter config, rather than this one field being
# special. Previously anchored to REPO_ROOT, which only resolves to the repo
# for an editable install and would land inside the virtualenv for a wheel
# install. Set $SPECTER_PDB_CACHE to an absolute path to share one cache
# across working directories. See `default_pdb_cache_dir` in config.py.
DEFAULT_PDB_SAVEFOLDER = default_pdb_cache_dir()

# Suppress only PDBConstructionWarnings
warnings.simplefilter("ignore", PDBConstructionWarning)


# (chain, residue number, insertion code, atom name)
_AtomKey = tuple[str, int, str, str]


def _atom_key(
    chain: "gemmi.Chain", residue: "gemmi.Residue", atom: "gemmi.Atom"
) -> _AtomKey:
    """
    Identify one atom uniquely within a gemmi model.

    The insertion code is part of the identity, not decoration: an entry like
    3DY4 numbers whole stretches of residues `10`, `10A`, `10B`, ... , so a key
    without it collapses thousands of genuinely distinct atoms onto each other.
    `get_atom_species` treats a repeated key as an alternate conformer to be
    dropped, so those atoms went missing and the returned species list no
    longer lined up with `atomic_numbers`.
    """
    return (chain.name, residue.seqid.num or 0, residue.seqid.icode, atom.name)


class PDB:
    def __init__(
        self,
        pdb_source: str,
        assembly: bool = True,
        savefolder: str = DEFAULT_PDB_SAVEFOLDER,
        origin: tuple[float, float, float] | None = None,
        verbose: bool = True,
        compute_atom_species: bool = False,
    ) -> None:
        """
        Create a PDB object from either a PDB ID or a local file path.

        Parameters
        ----------
        pdb_source : str
            Either a 4-character PDB ID (e.g., '1abc') or a local file path
            to a PDB/mmCIF structure file.
        assembly : bool, optional
            Whether to fetch the biological assembly when using a PDB ID.
            Default is True.
        savefolder : str, optional
            Folder to store downloaded PDB/mmCIF files. Default is
            `specter-data/pdb`, relative to the caller's cwd -- see
            `config.default_pdb_cache_dir`.
        origin : tuple[float, float, float] or None, optional
            Custom origin to subtract from coordinates. If None, coordinates
            are centered on their geometric center. If a tuple, that point is
            subtracted directly without auto-centering.
        verbose : bool, optional
            Print fetch/assembly status and show the atom-extraction progress
            bar. Default True; set False to silence per-fetch chatter (e.g.
            when constructing many PDB objects in a loop with your own
            higher-level progress reporting).
        compute_atom_species : bool, optional
            Also compute `atom_species` (bonded-neighbor species descriptors,
            e.g. for the Shtyrov potential parameterization). Requires an
            mmCIF source and re-parses it via gemmi, so it is opt-in.
            Default is False.

        Attributes
        ----------
        pdb_id : str
            The PDB ID if pdb_source is a PDB ID.
        filepath : str
            Path to the PDB/mmCIF file.
        atomic_numbers : torch.Tensor
            Atomic numbers of all atoms in the structure, shape (N,).
        coordinates : torch.Tensor
            Coordinates shifted by origin, shape (N, 3).
        atom_species : list of str or None, optional
            Bonded-neighbor species descriptor per atom (e.g. `"O(HH)"`,
            `"C(HHHC)"`), only set when `compute_atom_species=True`.
        """

        # Determine whether pdb_source is a PDB ID or file path
        if (
            isinstance(pdb_source, str)
            and len(pdb_source) == 4
            and pdb_source.isalnum()
        ):
            # Treat as PDB ID
            self.pdb_id = pdb_source
            self.filepath = PDB.fetch_pdb_file(
                pdb_source, savefolder=savefolder, assembly=assembly, verbose=verbose
            )
            self.assembly = assembly
            self.savefolder = savefolder
        elif os.path.isfile(pdb_source):
            self.filepath = pdb_source
        else:
            raise ValueError(
                f"Invalid pdb_source: '{pdb_source}'. Must be a 4-character PDB ID or a valid file path."
            )

        # get pdb structure
        self.structure = PDB.get_pdb_structure(self.filepath)
        """Bio.PDB.Structure.Structure: Parsed PDB structure object."""

        # get atomic elements and coordinates
        self.atomic_numbers, self.coordinates = PDB.get_atoms_and_coordinates(
            self.structure, verbose=verbose
        )

        # bonded-neighbor species descriptors (e.g. for Shtyrov potentials)
        self.atom_species: list[str | None] | None = None
        if compute_atom_species:
            self.atom_species = PDB.get_atom_species(self.filepath, verbose=verbose)
            # Coordinates come from Biopython and species from gemmi, so the
            # two lists are only meaningful zipped together. Catch any residual
            # disagreement here, where the structure can be named, rather than
            # letting it surface as an opaque mask/tensor shape error deep in
            # PotentialBuilder. A multi-model file trips this: Biopython walks
            # every model, gemmi types only the first.
            if len(self.atom_species) != self.atomic_numbers.shape[0]:
                n_models = len(list(self.structure))
                raise ValueError(
                    f"{self.filepath}: bonded-species typing produced "
                    f"{len(self.atom_species)} entries for "
                    f"{self.atomic_numbers.shape[0]} atoms"
                    + (
                        f" -- the file holds {n_models} models, and only the "
                        "first can be typed. Extract a single model, or use "
                        "potential_parameterization='kirkland'/'lobato'."
                        if n_models > 1
                        else ". Use potential_parameterization="
                        "'kirkland'/'lobato' for this structure."
                    )
                )

        # center coordinates
        if origin is None:
            self.coordinates = PDB.center_coordinates(self.coordinates)
        else:
            self.coordinates = self.coordinates - torch.tensor(
                origin, dtype=self.coordinates.dtype
            )

        # estimate max diameter
        self.max_diameter = PDB.estimate_max_diameter(self.coordinates)
        """float: Maximum diameter of the structure based on convex hull."""

    @staticmethod
    def fetch_pdb_file(
        pdb_id: str,
        ext: str = "cif",
        savefolder: str = DEFAULT_PDB_SAVEFOLDER,
        assembly: bool | int = True,
        verbose: bool = True,
    ) -> str:
        """
        Download a PDB file and save it in a given location.

        Parameters
        ----------
        pdb_id : str
            A valid PDB ID.
        ext : str
            File ext ('cif' or 'pdb').
        savefolder : str
            Destination folder.
        assembly : bool or int
            - True  → fetch default biological assembly (assembly 1 if available).
            - False → fetch asymmetric unit.
            - int   → fetch that specific assembly if available, fallback to default
               PDBx/mmCIF file.
        verbose : bool, optional
            Print fetch/assembly status. Default True.

        Returns
        -------
        str
            Path to the saved PDB file
        """
        PDB.get_available_assemblies(pdb_id, verbose=verbose)

        # Decide what to fetch
        if assembly is True:
            if verbose:
                print(f"{pdb_id}: Fetching default Biological Assembly 1")
            filename = f"{pdb_id}-assembly{1}.{ext}"
        elif assembly is False:
            if verbose:
                print(f"{pdb_id}: Fetching default PDBx/mmCIF file.")
            filename = f"{pdb_id}.{ext}"
        elif isinstance(assembly, int):
            if verbose:
                print(f"{pdb_id}: Fetching Biological Assembly {assembly}")
            filename = f"{pdb_id}-assembly{assembly}.{ext}"
        else:
            raise ValueError("assembly must be True, False, or int")

        # Build filepath
        file_path = os.path.join(savefolder, filename)

        # Return existing file if available
        if os.path.exists(file_path):
            if verbose:
                print(f"File already exists: {file_path}, skip fetching.")
            return file_path

        # Fetch
        if verbose:
            print("File does not exist, fetching.")
        url = "https://files.rcsb.org/download/" + filename + ".gz"
        r = requests.get(url)
        r.raise_for_status()

        # Decompress in memory
        with gzip.open(io.BytesIO(r.content), "rt") as f:
            cif_content = f.read()

        # Save to file -- savefolder is just a plain relative-or-absolute
        # path (resolved against the current process's cwd, not the
        # specter repo root), so create it on demand rather than crashing
        # with a raw FileNotFoundError if it doesn't exist yet.
        os.makedirs(savefolder, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(cif_content)

        if verbose:
            print(f"Downloaded to: {file_path}")
        return file_path

    @staticmethod
    def get_available_assemblies(pdb_id: str, verbose: bool = True) -> None:
        """
        Print the available biological assembly IDs for a PDB entry.

        Parameters
        ----------
        pdb_id : str
            4-character PDB ID.
        verbose : bool, optional
            Print the result (or fetch error). Default True.

        Notes
        -----
        Prints the available assemblies to console. If the PDB entry cannot be
        accessed or does not have assembly information, prints an error message.
        """
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            assemblies = data.get("rcsb_entry_container_identifiers", {}).get(
                "assembly_ids", []
            )
            if verbose:
                print("Assemblies available: " + ", ".join(assemblies))
        except Exception as e:
            if verbose:
                print(f"Error fetching assemblies for {pdb_id}: {e}")
            return

    @staticmethod
    def get_pdb_structure(filepath: str) -> Structure:
        """
        Parse a PDB or mmCIF file and return the structure object.

        Parameters
        ----------
        filepath : str
            Path to PDB (.pdb) or mmCIF (.cif) file.

        Returns
        -------
        structure : Bio.PDB.Structure.Structure
            Parsed structure object from Biopython.

        Raises
        ------
        ValueError
            If the file format is not 'pdb' or 'cif'.
        """
        ext = filepath[-3:]
        parser: PDBParser | MMCIFParser
        if ext == "pdb":
            parser = PDBParser()
        elif ext == "cif":
            parser = MMCIFParser()
        else:
            raise ValueError(f"Invalid file format '{ext}'. Must be 'cif' or 'pdb'.")
        structure = parser.get_structure("structure", filepath)
        return structure

    @staticmethod
    def get_atoms_and_coordinates(
        structure: Structure | str,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract atomic elements and coordinates from PDB structure.

        Parameters
        ----------
        structure : Bio.PDB.Structure.Structure or str
            Either a parsed Biopython structure object or a filepath to a
            PDB/mmCIF file. If a filepath is provided, the structure will be
            loaded automatically.
        verbose : bool, optional
            Show the extraction progress bar. Default True.

        Returns
        -------
        elements : torch.Tensor
            Atomic numbers for each atom, shape (N,).
        coords : torch.Tensor
            Atomic coordinates (x, y, z) for each atom, shape (N, 3).

        Notes
        -----
        Uses Rich progress bar to display extraction progress for large structures.
        """

        # if filepath, load the structure.
        if isinstance(structure, str):
            structure = PDB.get_pdb_structure(structure)

        # Extract atomic data
        coords_list = []
        element_symbols = []
        for atom in track(
            structure.get_atoms(),
            description="Extracting atom coordinates and element",
            disable=not verbose,
        ):
            element_symbol = atom.element.strip().upper()
            element_symbols.append(element_symbol)

            coord = atom.get_coord()
            coords_list.append(coord)
        coords = torch.as_tensor(np.array(coords_list))
        elements = atom_number(element_symbols)

        return elements, coords

    @staticmethod
    def get_atom_species(
        filepath: str,
        monomer_library_path: str | None = None,
        verbose: bool = True,
    ) -> list[str | None]:
        """
        Determine each atom's bonded-neighbor species descriptor.

        Mirrors the atom-typing approach used by `sffit
        <https://github.com/as2875/sffit>`_ to fit bonded-species electron
        scattering factors: builds a bond topology with
        ``gemmi.prepare_topology`` and describes each atom as its element
        plus the elements of its directly bonded neighbors, e.g. `"O(HH)"`
        for a water oxygen or `"C(HHHC)"` for a methyl carbon, with the same
        carboxyl/amide override sffit uses for Asp/Glu/Asn/Gln side-chain
        oxygens.

        Parameters
        ----------
        filepath : str
            Path to an mmCIF file. Bond templates come from the file's own
            embedded chemical-component definitions (`_chem_comp_bond`) plus
            explicit links (`_struct_conn`, e.g. disulfides or metal
            coordination) and standard polymer backbone connectivity, all
            resolved by gemmi. The legacy PDB format carries none of this,
            so this function requires mmCIF.
        monomer_library_path : str, optional
            Path to a CCP4 Monomer Library (as used by REFMAC/Coot/
            Servalcat). If None, falls back to the `CLIBD_MON` environment
            variable, and if that is unset too, to gemmi's built-in
            defaults. Bond topology for standard residues is available
            either way, but hydrogens can only be added with a real monomer
            library — without one, H-containing species (e.g. `"O(HH)"`,
            `"C(HHHC)"`) will not be resolved and those atoms fall back to
            `None`.
        verbose : bool, optional
            Print how many atoms were successfully typed. Default True.

        Returns
        -------
        species : list of str or None
            One entry per atom, in the same file/chain/residue/atom order
            as `get_atoms_and_coordinates`. `None` where no neighbors could
            be determined (e.g. isolated ions, or unresolved components).
        """
        if not filepath.lower().endswith("cif"):
            warnings.warn(
                "get_atom_species requires an mmCIF source (bond templates "
                "are not available in legacy PDB format); returning None "
                "for all atoms.",
                stacklevel=2,
            )
            structure = PDB.get_pdb_structure(filepath)
            return [None for _ in structure.get_atoms()]

        monlib_path = monomer_library_path or os.environ.get("CLIBD_MON")

        st = gemmi.read_structure(filepath)
        st.setup_entities()

        if monlib_path:
            try:
                monlib = gemmi.read_monomer_lib(
                    monlib_path, st[0].get_all_residue_names()
                )
            except RuntimeError as err:
                warnings.warn(str(err), RuntimeWarning, stacklevel=2)
                monlib = gemmi.MonLib()
        else:
            warnings.warn(
                "CLIBD_MON is not set; falling back to gemmi's built-in "
                "chemical-component definitions. Bond topology for standard "
                "residues is still derived from the file itself, but "
                "hydrogens cannot be added, so H-containing Shtyrov species "
                "(e.g. 'O(HH)', 'C(HHHC)') will not be resolved.",
                RuntimeWarning,
                stacklevel=2,
            )
            monlib = gemmi.MonLib()

        warnings_sink = sys.stderr if verbose else io.StringIO()

        # First pass: complete the model (adds hydrogens as dummy,
        # zero-occupancy atoms if the monomer library has geometry for
        # them), mirroring sffit's from_gemmi().
        #
        # This only runs with a real monomer library. `HydrogenChange.ReAdd`
        # *removes* every existing hydrogen before re-adding it from library
        # geometry, so without a library to re-add from it is not a no-op: it
        # strips the file's own hydrogens outright (22FX loses 4968 atoms),
        # leaving fewer species than there are atoms in `atomic_numbers`.
        #
        # The added atoms are tracked by identity rather than recognised by
        # their zero occupancy later: deposited structures may legitimately
        # contain zero-occupancy atoms (1FA2 has 208 of them), and dropping
        # those as "dummies" would misalign the two lists the same way.
        added_keys: set[_AtomKey] = set()
        if len(monlib.monomers):
            topo = gemmi.prepare_topology(
                st, monlib, h_change=gemmi.HydrogenChange.ReAdd, warnings=warnings_sink
            )
            for m in topo.find_missing_atoms(including_hydrogen=True):
                # A residue the library has no entry for (including the
                # blank-named components some entries carry) can't be
                # completed; its atoms keep whatever bonds the file provides.
                if m.res_id.name not in monlib.monomers:
                    continue
                mon = monlib.monomers[m.res_id.name]
                monat = mon.find_atom(m.atom_name)
                if monat is None:
                    continue
                atom = gemmi.Atom()
                atom.occ = 0.0
                atom.element = monat.el
                atom.name = m.atom_name
                cra = st[0].find_cra(m)
                added_keys.add(_atom_key(cra.chain, cra.residue, atom))
                cra.residue.add_atom(atom)

        # Second pass: final bond graph, now including any added atoms.
        topo = gemmi.prepare_topology(
            st, monlib, h_change=gemmi.HydrogenChange.NoChange, warnings=warnings_sink
        )

        # Atom objects returned by gemmi are stable dict keys for the same
        # underlying atom (mirrors sffit's `lookup` pattern in from_gemmi).
        # Neighbors are tracked by atom identity, not element, so that e.g.
        # a methyl carbon's three separate hydrogen neighbors are each
        # counted (not collapsed into one "H").
        identity = {
            cra.atom: _atom_key(cra.chain, cra.residue, cra.atom) for cra in st[0].all()
        }
        element_of = {key: atom.element.name for atom, key in identity.items()}
        neighbor_keys: dict[_AtomKey, set[_AtomKey]] = defaultdict(set)
        for bond in topo.bonds:
            a, b = bond.atoms
            ka, kb = identity[a], identity[b]
            neighbor_keys[ka].add(kb)
            neighbor_keys[kb].add(ka)

        species: list[str | None] = []
        n_matched = 0
        seen: set[_AtomKey] = set()
        for cra in st[0].all():
            key = _atom_key(cra.chain, cra.residue, cra.atom)
            if key in added_keys:
                # dummy atom added only to inform its neighbors' typing
                continue
            if key in seen:
                # alternate conformer of an atom already typed (gemmi's
                # `all()` visits every altloc; Biopython/get_atoms_and_
                # coordinates collapses them to one atom per position, so
                # we keep only the first here to stay index-aligned with
                # atomic_numbers/coordinates — bonding topology doesn't
                # differ between altlocs of the same atom in practice).
                continue
            seen.add(key)
            neighbors = neighbor_keys.get(key)
            if not neighbors:
                species.append(None)
                continue

            elem = cra.atom.element.name
            neighbor_elems = [element_of[nk] for nk in neighbors]
            neighbor_str = "".join(
                sorted(neighbor_elems, key=lambda e: gemmi.Element(e).atomic_number)
            )
            descriptor = f"{elem}({neighbor_str})"
            if cra.atom.name in ("OD1", "OD2", "OE1", "OE2"):
                if cra.residue.name in ("ASP", "GLU"):
                    descriptor = f"{elem}({neighbor_str}, carboxyl)"
                elif cra.residue.name in ("ASN", "GLN"):
                    descriptor = f"{elem}({neighbor_str}, amide)"
            species.append(descriptor)
            n_matched += 1

        if verbose:
            print(f"[get_atom_species] {n_matched}/{len(species)} atoms typed")

        return species

    @staticmethod
    def center_of_particle(coords: torch.Tensor) -> torch.Tensor:
        """
        Return a particle's geometric center.

        Parameters
        ----------
        coords : torch.Tensor
            Atom coordinates of molecule with N atoms, shape (N, 3).

        Returns
        -------
        center : torch.Tensor
            Geometric center of the molecule, shape (3,).
        """
        center = coords.mean(dim=0)
        return center

    @staticmethod
    def center_coordinates(coords: torch.Tensor) -> torch.Tensor:
        """
        Centers coordinates on its geometric center.

        Parameters
        ----------
        coords : tensor
            Atom coordinates of molecule with N atoms, shape (N,3)

        Returns
        -------
        centered_coordinates : tensor
            Centered coordinates, shape (N,3)
        """
        center = PDB.center_of_particle(coords)
        return coords - center

    @staticmethod
    def estimate_max_diameter(coordinates: torch.Tensor) -> float:
        """
        Estimate the maximum diameter of a structure using convex hull.

        Parameters
        ----------
        coordinates : torch.Tensor
            Atomic coordinates with shape (N, 3).

        Returns
        -------
        max_diameter : float
            Maximum pairwise distance between convex hull vertices.

        Notes
        -----
        Computes the convex hull of the coordinates and returns the maximum
        distance between any two hull vertices.
        """
        hull = ConvexHull(coordinates)
        hull_points = coordinates[hull.vertices]
        max_diameter = pdist(hull_points).max()
        return max_diameter
