"""GemmiPotentialBuilder: scattering-potential volumes via the Gemmi library."""

from __future__ import annotations

from specter import logger

import multiprocessing as mp
from typing import Any, Sequence

import gemmi
import numpy as np
import torch


class GemmiPotentialBuilder:
    """
    Build electrostatic potential volumes using Gemmi library.

    Uses Gemmi's density calculator with Gaussian atomic form factors.
    Supports custom scattering factors from mmCIF files.

    Parameters
    ----------
    n_xyz : int or tuple of int
        Grid size (nx, ny, nz). If int, assumes cubic grid.
    dx : float
        Pixel/voxel size in Å.
    atomic_numbers : torch.Tensor, optional
        Atomic numbers of all atoms. Default is None.
    b_factor : float, optional
        Isotropic B-factor. Default is 20.0.
    """

    def __init__(
        self,
        n_xyz: int | Sequence[int],
        dx: float,
        atomic_numbers: torch.Tensor | None = None,
        b_factor: float | None = None,
    ):
        if isinstance(n_xyz, (int, float)):
            self.nx = self.ny = self.nz = n_xyz
        else:
            self.nx, self.ny, self.nz = n_xyz
        self.dx = dx
        self.b_factor = b_factor
        self.translate_to_center = torch.tensor(
            [[self.nx // 2 * dx, self.ny // 2 * dx, self.nz // 2 * dx]]
        )
        """torch.Tensor: Translation vector to center atoms in grid."""
        self.atomic_numbers = atomic_numbers

        # scaling prefactor
        a0 = 0.529  # Bohr radius, [Å]
        e = 14.4  # electron charge, [V·Å]
        self.c1 = 2 * torch.pi * e * a0
        """float: Scaling factor for electrostatic potential (2π*e*a₀)."""

    def build_model(
        self,
        atom_coordinates: torch.Tensor,
        atom_elements: torch.Tensor,
    ) -> gemmi.Model:
        """
        Build Gemmi structure model from atomic coordinates and elements.

        Parameters
        ----------
        atom_coordinates : torch.Tensor
            Atomic coordinates in Å, shape (N, 3).
        atom_elements : torch.Tensor
            Atomic numbers, shape (N,).

        Returns
        -------
        model : gemmi.Model
            Gemmi model containing atoms within grid bounds.

        Notes
        -----
        Filters out atoms outside the grid boundaries.
        All atoms are assigned to chain A, residue 1.
        """
        st = gemmi.Structure()
        model = st.add_model(gemmi.Model(1))
        chain = model.add_chain("A")
        res = chain.add_residue(gemmi.Residue())

        mask = (
            (atom_coordinates[:, 0] >= 0.0)
            & (atom_coordinates[:, 0] <= self.nx * self.dx)
            & (atom_coordinates[:, 1] >= 0.0)
            & (atom_coordinates[:, 1] <= self.ny * self.dx)
            & (atom_coordinates[:, 2] >= 0.0)
            & (atom_coordinates[:, 2] <= self.nz * self.dx)
        )

        filtered_coords = atom_coordinates[mask]
        filtered_elements = atom_elements[mask]
        b_iso = self.b_factor if self.b_factor is not None else 20.0

        for i, (pos, z) in enumerate(zip(filtered_coords, filtered_elements), start=1):
            # if not (0. <= pos[0] <= self.nx * self.dx and 0. <= pos[1] <= self.ny * self.dx and 0. <= pos[2] <= self.nz * self.dx):
            #     continue
            atom = gemmi.Atom()
            atom.pos = gemmi.Position(float(pos[0]), float(pos[1]), float(pos[2]))
            atom.element = gemmi.Element(int(z))
            atom.occ = 1.0
            atom.b_iso = b_iso
            atom.serial = i
            res.add_atom(atom)
        return model

    def build_dencalc(self) -> gemmi.DensityCalculatorE:
        """
        Build a fresh Gemmi density calculator with current grid settings.

        Returns
        -------
        dencalc : gemmi.DensityCalculatorE
            Configured density calculator.

        Notes
        -----
        Creates a new calculator to avoid state contamination between
        multiple potential calculations.
        """
        dencalc = gemmi.DensityCalculatorE()
        unit_cell = gemmi.UnitCell(
            self.nx * self.dx,
            self.ny * self.dx,
            self.nz * self.dx,
            90.0,
            90.0,
            90.0,
        )
        dencalc.grid.unit_cell = unit_cell
        dencalc.grid.spacegroup = gemmi.SpaceGroup("P1")
        dencalc.grid.set_size(self.nx, self.ny, self.nz)
        return dencalc

    def build_potential_from_custom_mmcif(self, mmcif_filepath: str) -> torch.Tensor:
        """
        Build potential using custom scattering factors from mmCIF file.

        Reads scattering factor coefficients from '_lmb_scat_coef' table
        in mmCIF file for high-accuracy potential calculations.

        Parameters
        ----------
        mmcif_filepath : str
            Path to mmCIF file containing structure and scattering factors.

        Returns
        -------
        potential : torch.Tensor
            Electrostatic potential volume, shape (nz, ny, nx).

        Notes
        -----
        Uses only the first atom from the structure and applies custom
        form factors from the mmCIF file. Atoms are recentered to grid center.
        """
        st = gemmi.read_structure(mmcif_filepath)

        # --- keep only the first atom ---
        # first_atom = st[0][0][0][0]
        # new_st = gemmi.Structure()
        # model = new_st.add_model(gemmi.Model(1))
        # chain = model.add_chain("A")
        # residue = chain.add_residue(gemmi.Residue())
        # residue.add_atom(first_atom.clone())
        # st = new_st
        # --------------------------------

        block = gemmi.cif.read_file(mmcif_filepath).sole_block()
        ctable = block.find(
            "_lmb_scat_coef.",
            [
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

        coefs = np.empty((len(ctable), 10))
        for ind, row in enumerate(ctable):
            coefs[ind] = [float(field) for field in row]
        max_serial = max(cra.atom.serial for cra in st[0].all())
        custom_form_factors = np.zeros((max_serial + 1, 10))
        itable = block.find("_atom_site.", ["id", "scat_id"])
        for row in itable:
            serial, scat_id = row
            custom_form_factors[int(serial)] = coefs[int(scat_id)]
            # print(scat_id)
            # break
        gemmi.set_custom_form_factors(custom_form_factors.tolist())
        dencalc = gemmi.DensityCalculatorC()

        coords = np.array([cra.atom.pos for cra in st[0].all()])  # (N, 3)
        center_geom = coords.mean(axis=0)
        for cra in st[0].all():
            translate = -np.asarray(
                center_geom.tolist()
            ) + self.translate_to_center.numpy().squeeze(0)
            cra.atom.pos += gemmi.Position(
                float(translate[0]), float(translate[1]), float(translate[2])
            )
            if self.b_factor is not None:
                cra.atom.b_iso = self.b_factor

        if self.b_factor is None:
            logger.warning(
                "Using default B-factor in mmcif file. Set b_factor to 0 if not intended."
            )

        unit_cell = gemmi.UnitCell(
            self.nx * self.dx,
            self.ny * self.dx,
            self.nz * self.dx,
            90.0,
            90.0,
            90.0,
        )
        dencalc.grid.unit_cell = unit_cell
        dencalc.grid.spacegroup = gemmi.SpaceGroup("P1")
        dencalc.grid.set_size(self.nx, self.ny, self.nz)
        dencalc.put_model_density_on_grid(st[0])
        return self.c1 * torch.as_tensor(dencalc.grid.array).transpose(0, 2)

    def _build_single_potential(
        self, coords_elements_tuple: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Build potential for a single set of coordinates (non-parallel).

        Parameters
        ----------
        coords_elements_tuple : tuple
            (coordinates, elements) where coordinates is (N, 3) and
            elements is (N,).

        Returns
        -------
        potential : torch.Tensor
            Potential volume with shape (nz, ny, nx).
        """
        coords, elements = coords_elements_tuple
        model = self.build_model(coords, elements)
        dencalc = self.build_dencalc()
        dencalc.put_model_density_on_grid(model)
        return torch.as_tensor(dencalc.grid.array).transpose(0, 2)

    @staticmethod
    def _build_parallelizable_single_potential(args: tuple[Any, ...]) -> torch.Tensor:
        """
        Build potential for parallel processing (static method).

        Parameters
        ----------
        args : tuple
            (coords, elements, nx, ny, nz, dx, b_factor) containing all
            necessary parameters for building potential.

        Returns
        -------
        potential : torch.Tensor
            Potential volume with shape (nz, ny, nx).

        Notes
        -----
        Static method to enable multiprocessing with spawn context.
        Creates fresh Gemmi objects to avoid pickling issues.
        """
        coords, elements, nx, ny, nz, dx, b_factor = args
        st = gemmi.Structure()
        model = st.add_model(gemmi.Model(1))
        chain = model.add_chain("A")
        res = chain.add_residue(gemmi.Residue())

        mask = (
            (coords[:, 0] >= 0.0)
            & (coords[:, 0] <= nx * dx)
            & (coords[:, 1] >= 0.0)
            & (coords[:, 1] <= ny * dx)
            & (coords[:, 2] >= 0.0)
            & (coords[:, 2] <= nz * dx)
        )

        filtered_coords = coords[mask]
        filtered_elements = elements[mask]

        for i, (pos, z) in enumerate(zip(filtered_coords, filtered_elements), start=1):
            atom = gemmi.Atom()
            atom.pos = gemmi.Position(float(pos[0]), float(pos[1]), float(pos[2]))
            atom.element = gemmi.Element(int(z))
            atom.occ = 1.0
            atom.b_iso = b_factor
            atom.serial = i
            res.add_atom(atom)

        dencalc = gemmi.DensityCalculatorE()
        unit_cell = gemmi.UnitCell(
            nx * dx,
            ny * dx,
            nz * dx,
            90.0,
            90.0,
            90.0,
        )
        dencalc.grid.unit_cell = unit_cell
        dencalc.grid.spacegroup = gemmi.SpaceGroup("P1")
        dencalc.grid.set_size(nx, ny, nz)
        dencalc.put_model_density_on_grid(model)
        return torch.as_tensor(dencalc.grid.array).transpose(0, 2)

    def build_potential(
        self,
        atom_coordinates: torch.Tensor,
        atomic_numbers: torch.Tensor | None = None,
        n_processes: int | None = None,
    ) -> torch.Tensor:
        """
        Build electrostatic potential volume from atomic coordinates.

        Parameters
        ----------
        atom_coordinates : torch.Tensor
            Atomic coordinates in Ų, shape (N, 3). Centered at origin.
        atomic_numbers : torch.Tensor, optional
            Atomic numbers, shape (N,). If None, uses self.atomic_numbers.
            Default is None.
        n_processes : int, optional
            Number of parallel processes. If None, runs serially.
            Default is None.

        Returns
        -------
        potential : torch.Tensor
            Electrostatic potential volume in Volts, shape (nz, ny, nx).

        Notes
        -----
        Coordinates are automatically translated to place origin at grid center.
        Parallel processing splits atoms across processes and sums results.
        Scaling factor c1 = 2π*e*a₀ is applied to match physical units.
        """
        if atomic_numbers is None:
            atomic_numbers = self.atomic_numbers
        else:
            self.atomic_numbers = atomic_numbers
        if atomic_numbers is None:
            raise ValueError(
                "atomic_numbers must be provided, either as an argument or set on "
                "the builder at construction time."
            )
        translated_coordinates = atom_coordinates + self.translate_to_center

        if n_processes is None:
            volume = self._build_single_potential(
                (translated_coordinates, atomic_numbers)
            )
            return self.c1 * volume

        chunks_coords = torch.split(translated_coordinates, n_processes)
        chunks_elements = torch.split(atomic_numbers, n_processes)
        args_list = [
            (
                chunks_coords[i],
                chunks_elements[i],
                self.nx,
                self.ny,
                self.nz,
                self.dx,
                self.b_factor,
            )
            for i in range(n_processes)
        ]

        with mp.get_context("spawn").Pool(processes=n_processes) as pool:
            results = pool.map(self._build_parallelizable_single_potential, args_list)

        total_volume = torch.stack(results).sum(0)
        return self.c1 * total_volume
