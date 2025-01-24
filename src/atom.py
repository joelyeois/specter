import numpy as np
import os

def get_atom_symbols():
    """
    Return list of atom symbols.

    Returns
    -------
    names : ndarray
        Atom symbols in the periodic table.
    """
    elem = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "../atom-data", "atom_mass.txt"),
        delimiter=" ",
        dtype="str",
        usecols=0,
    )
    return np.array([e.upper() for e in elem])


def get_atom_masses():
    """
    Return list of atom masses.

    Returns
    -------
    masses : ndarray
        Atom masses
    """
    masses = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "../atom-data", "atom_mass.txt"),
        delimiter=" ",
        dtype="float",
        usecols=1,
    )
    return masses


def atom_symbol(atomic_number):
    """
    Return atomic symbol.

    Parameters
    ----------
    atomic_number : ndarray
        List of atomic numbers. Hydrogen is 1.

    Returns
    -------
    atom_symbols : str
        Atomic symbol
    """
    atom_symbols = get_atom_symbols()
    return atom_symbols[atomic_number - 1]


def atom_number(symbol):
    """
    Return atomic number.

    Parameters
    ----------
    symbol : str
        Atom symbol, e.g. H,C,N,...

    Returns
    -------
    atom_numbers : int
        Atomic numbers, Hydrogen has number 1.
    """

    atom_symbols = get_atom_symbols()
    return np.where(atom_symbols == symbol.upper())[0][0] + 1

def get_atom_params_dict():
    """
    Returns a dictionary of atoms and their interaction parameters. 
    (taken from Appendix C in Kirkland, Advanced Computing in Electron Microscopy)

    The (4x3) parameter array has the form
    a1 a2 a3
    b1 b2 b3
    c1 c2 c3
    d1 d2 d3
    
    This parameterization  can be used to calculate the atomic potential for both X-ray and electron interaction.

    Returns
    -------
    params : dict
        Dictionary with atomic number as keys. Hydrogen has key 1.
        Each entry contains 'params' (4x3 ndarray) and 'chisquared' (float).
    """
    params = {}
    with open(os.path.join(os.path.dirname(__file__),'../atom-data', "atom_params.txt"), 'r') as f:
        lines = f.read().split('Z= ')
        for line in lines:
            if(len(line)):
                llist = line.split('\n')
                l0 = llist[0].split(',')
                Z = int(l0[0])
                CH = float(l0[1].split('chisq= ')[1])
                P1 = [float(e) for e in llist[1].split(' ')]
                P2 = [float(e) for e in llist[2].split(' ')]
                P3 = [float(e) for e in llist[3].split(' ')]
                A = [P1[0], P1[2], P2[0]]
                B = [P1[1], P1[3], P2[1]]
                C = [P2[2], P3[0], P3[2]]
                D = [P2[3], P3[1], P3[3]]
                P = np.vstack([A,B,C,D])
                params[Z] = {'chisquared':CH, 'params':P}
    return params