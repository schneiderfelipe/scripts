#!/usr/bin/python3

"""Calculate the RMSD between structures."""

import argparse

import numpy as np


def read_xyz(path_or_file):
    """Read a xyz file and return structures.

    Parameters
    ----------
    path_or_file : str

    Returns
    -------
    array-like
    """

    def _process(xyz_file):
        lines = xyz_file.readlines()
        natom = int(lines[0])
        m = 2 + natom
        nstruct = len(lines) // m

        coords = []
        for i in range(nstruct):
            struct = []
            initial = i * m + 2
            for j in range(initial, initial + natom):
                struct.append([float(x) for x in lines[j].split()[1:]])
            coords.append(np.array(struct))
        return coords

    try:
        with open(path_or_file, "r") as xyz_file:
            return _process(xyz_file)
    except TypeError:
        return _process(path_or_file)


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
        coords.append(read_xyz(xyz_file))
        names.append(xyz_file.name)

    print("RMSD:")
    for i in range(len(coords)):
        for j in range(i):
            rmsd = calc_rmsd(coords[j][-1], coords[i][-1])
            print(f"{names[j]:14s} ~ {names[i]:14s} = {rmsd:6.4f} Å")


if __name__ == "__main__":
    main()
