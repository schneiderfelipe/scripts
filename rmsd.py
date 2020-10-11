#!/usr/bin/python3

"""Calculate the RMSD between structures."""

import argparse

import numpy as np

# Inspired by
# https://github.com/cclib/cclib/blob/master/cclib/parser/utils.py#L159
element = [
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
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

# Inspired by
# https://github.com/cclib/cclib/blob/master/cclib/parser/utils.py#L159
atomic_number = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}


def read_xyz(path_file_or_str):
    """Read a xyz file and return structures.

    Parameters
    ----------
    path_file_or_str : str

    Returns
    -------
    atomnos : array-like
    comments : str
    atomcoords : array-like
    """

    def _process(lines):
        natom = int(lines[0])
        m = 2 + natom
        nstruct = len(lines) // m

        atomnos = []
        comments = []
        atomcoords = []
        for i in range(nstruct):
            nos = []
            coords = []
            initial = i * m + 2
            for j in range(initial, initial + natom):
                fields = lines[j].split()
                nos.append(atomic_number[fields[0]])
                coords.append([float(x) for x in fields[1:]])
            atomnos.append(np.array(nos))
            comments.append(lines[i * m + 1].strip("\n"))
            atomcoords.append(np.array(coords))
        return atomnos, comments, atomcoords

    try:
        with open(path_file_or_str, "r") as xyz_file:
            return _process(xyz_file.readlines())
    except TypeError:
        return _process(path_file_or_str.readlines())
    except FileNotFoundError:
        return _process(path_file_or_str.split("\n"))


def write_xyz(atomnos, atomcoords, comment=""):
    """Format a string as xyz coordinates.

    Parameters
    ----------
    atomnos, atomcoords : array-like

    Returns
    -------
    str
    """
    lines = []
    for no, coord in zip(atomnos, atomcoords):
        lines.append(
            f"{element[no]:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}"
        )

    lines = "\n".join(lines)
    return f"{len(atomnos)}\n{comment}\n{lines}"


def calc_rmsd(P, Q, translate=True):
    """Calculate the RMSD between P and Q using Kabsch algorithm.

    Parameters
    ----------
    P, Q : array-like
    translate : bool

    Returns
    -------
    float

    Notes
    -----
    If structures are not comparable, np.inf is returned.

    Examples
    --------
    >>> P = [[0, 0, 0],
    ...      [1, 1, 1]]
    >>> calc_rmsd(P, P)
    0.0
    >>> Q = [[1, 1, 1],
    ...      [2, 2, 2]]
    >>> calc_rmsd(P, Q)
    0.0
    >>> P = [[-3.652796902, 0.000000000, -4.445975658],
    ...      [-3.527558151, 0.000000000, -3.430150234],
    ...      [-4.501637325, 0.000000000, -3.833697320]]
    >>> Q = [[ 1.885538972, 0.000000000, -0.577489796],
    ...      [ 0.911459798, 0.000000000,  0.201773543],
    ...      [ 1.165525626, 1.078706033, -0.490031274]]
    >>> calc_rmsd(P, Q)
    0.140663071340813
    """
    P, Q = np.asanyarray(P), np.asanyarray(Q)
    if translate:
        P = P - P.mean(axis=0)
        Q = Q - Q.mean(axis=0)

    try:
        V, s, W = np.linalg.svd(P.T @ Q)
    except ValueError:
        return np.inf
    if np.linalg.det(V) * np.linalg.det(W) < 0:
        s[-1] = -s[-1]
        V[:, -1] = -V[:, -1]

    U = V @ W
    P = P @ U
    return np.sqrt(np.sum((P - Q) ** 2) / len(P))


def main():
    """Run main procedure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xyz_files", type=argparse.FileType("r"), default="-", nargs="+"
    )
    args = parser.parse_args()

    coords = []
    names = []
    for xyz_file in args.xyz_files:
        coords.append(read_xyz(xyz_file)[2])
        names.append(xyz_file.name)

    print("RMSD:")
    for i in range(len(coords)):
        for j in range(i):
            rmsd = calc_rmsd(coords[j][-1], coords[i][-1])
            print(f"{names[j]:14s} ~ {names[i]:14s} = {rmsd:6.4f} Å")


if __name__ == "__main__":
    main()
