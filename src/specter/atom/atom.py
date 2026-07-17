from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

ELEMENT_SYMBOLS = np.array(
    [
        None,
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
    ]
)

ATOMIC_MASSES = torch.tensor(
    [
        0.0,  # placeholder for Z=0
        1.008,
        4.0026,
        6.94,
        9.0122,
        10.81,
        12.011,
        14.007,
        15.999,
        18.998,
        20.180,  # 1-10
        22.990,
        24.305,
        26.982,
        28.085,
        30.974,
        32.06,
        35.45,
        39.948,
        39.098,
        40.078,  # 11-20
        44.956,
        47.867,
        50.942,
        51.996,
        54.938,
        55.845,
        58.933,
        58.693,
        63.546,
        65.38,  # 21-30
        69.723,
        72.630,
        74.922,
        78.971,
        79.904,
        83.798,
        85.468,
        87.62,
        88.906,
        91.224,  # 31-40
        92.906,
        95.95,
        98.0,
        101.07,
        102.91,
        106.42,
        107.87,
        112.41,
        114.82,
        118.71,  # 41-50
        121.76,
        127.60,
        126.90,
        131.29,
        132.91,
        137.33,
        138.91,
        140.12,
        140.91,
        144.24,  # 51-60
        145.0,
        150.36,
        151.96,
        157.25,
        158.93,
        162.50,
        164.93,
        167.26,
        168.93,
        173.05,  # 61-70
        174.97,
        178.49,
        180.95,
        183.84,
        186.21,
        190.23,
        192.22,
        195.08,
        196.97,
        200.59,  # 71-80
        204.38,
        207.2,
        208.98,
        209.0,
        210.0,
        222.0,
        223.0,
        226.0,
        227.0,
        232.04,  # 81-90
        231.04,
        238.03,
        237.0,
        244.0,
        243.0,
        247.0,
        247.0,
        251.0,
        252.0,
        257.0,  # 91-100
        258.0,
        259.0,
        262.0,  # 101-103
    ]
)


def atom_symbol(z: int | Sequence[int] | torch.Tensor) -> np.ndarray:
    """
    Convert atomic numbers to element symbols.

    Supports scalar, list, or PyTorch tensor input of any shape.
    Returns a NumPy array of strings with the same shape as the input.

    Parameters
    ----------
    z : int, list, or torch.Tensor
        Atomic number(s).

    Returns
    -------
    symbols : np.ndarray
        Element symbol(s) as strings.
    """
    # Convert torch tensor to numpy if needed
    z_np = z.cpu().numpy() if isinstance(z, torch.Tensor) else z
    # Convert to numpy array for indexing
    z_np = np.asarray(z_np)
    return ELEMENT_SYMBOLS[z_np]


def atom_number(symbols: str | Sequence[str] | np.ndarray) -> torch.Tensor:
    """
    Convert element symbols to atomic numbers using ELEMENT_SYMBOLS array.

    Case-insensitive.

    Parameters
    ----------
    symbols : str, list of str, or np.ndarray of str
        Element symbol(s).

    Returns
    -------
    numbers : torch.Tensor
        Atomic number(s) as a tensor of dtype long.
    """
    arr = np.atleast_1d(symbols).astype(str)

    # Convert ELEMENT_SYMBOLS to string, skip None at index 0
    element_str = np.array([str(s) if s is not None else "" for s in ELEMENT_SYMBOLS])
    element_upper = np.char.upper(element_str)

    numbers = np.array([np.where(element_upper == s.upper())[0][0] for s in arr])

    return torch.tensor(numbers, dtype=torch.long)


def atom_mass(
    x: str | int | Sequence[str] | Sequence[int] | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """
    Return atomic mass of atoms.

    Parameters
    ----------
    x : str, int, list, np.ndarray, or torch.Tensor
        - Atom symbol(s) as string, list of strings, or np.ndarray of strings
        - Atomic number(s) as int, list, np.ndarray of integers, or torch.Tensor

    Returns
    -------
    masses : torch.Tensor
        Atomic masses corresponding to input.
    """
    # If input is string or list of strings, convert to atomic numbers
    if isinstance(x, str) or (
        isinstance(x, list) and all(isinstance(s, str) for s in x)
    ):
        Z = atom_number(x)
    elif isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.integer):
            Z = torch.as_tensor(x, dtype=torch.long)
        else:  # assume array of strings
            Z = atom_number(x)
    elif torch.is_tensor(x):
        Z = x
    else:  # assume scalar integer
        Z = torch.tensor(x, dtype=torch.long)

    return ATOMIC_MASSES[Z]
