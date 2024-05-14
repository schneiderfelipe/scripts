#!/usr/bin/python3

"""Reorder atoms such that molecules resemble each other the most."""

import argparse
from functools import lru_cache

import numpy as np
from mendeleev import element
from scipy.spatial import distance_matrix

from rmsd import read_xyz, write_xyz


@lru_cache(maxsize=None)
def radius(i, typ="vdw"):
    if typ == "vdw":
        return element(i).vdw_radius / 100.0
    elif typ == "cov":
        return element(i).covalent_radius / 100.0
    else:
        raise ValueError(f"unknown radius type: {typ}")


def bond_matrix(atomno, coord, typ="cov", kind="dist"):
    radii = np.array([radius(int(i), typ) for i in atomno])
    S = radii[np.newaxis].T + radii[np.newaxis]
    D = distance_matrix(coord, coord)

    if kind == "dist":
        return D / S
    elif kind == "bo":
        # A small correction based on the deviation from carbon
        # 0.05 is a value I got by testing
        c = 0.353 + 0.05 * (S - 2.0 * radius(6, "cov"))

        bo = np.exp(-(D - S) / c)
        np.fill_diagonal(bo, 0)
        return bo
    else:
        raise ValueError(f"unknown kind: {kind}")


# https://github.com/holoviz/holoviews/blob/6d20a0662b566079be7fa39cc6d47e79a99769e9/holoviews/core/util.py#L1914
def arglexsort(arrays):
    """
    Returns the indices of the lexicographical sorting
    order of the supplied arrays.

    Examples
    --------
    >>> arglexsort([[1, 0], [0, 1], [2, 1], [0, 1]])
    array([1, 3, 0, 2])
    """
    arrays = np.asanyarray(arrays).T
    dtypes = ",".join(array.dtype.str for array in arrays)
    recarray = np.empty(len(arrays[0]), dtype=dtypes)
    for i, array in enumerate(arrays):
        recarray["f%s" % i] = array
    return recarray.argsort()


def regularize(atomno, coord, typ="cov"):
    orders = [
        np.round(np.sum(a), 0)
        for a in bond_matrix(atomno, coord, typ, kind="bo").tolist()
    ]

    # either reverse=True is required or reverse=False plus round to zero
    # decimal places
    dists = np.round(
        [
            sorted(a, reverse=False)
            for a in bond_matrix(atomno, coord, typ, kind="dist").tolist()
        ],
        0,
    )

    traits = [(-atomno[i], orders[i], *dists[i]) for i in range(len(orders))]

    indices = arglexsort(traits)

    return atomno[indices], coord[indices]


def main():
    """Run main procedure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xyz_files", type=argparse.FileType("r"), default="-", nargs="+"
    )
    args = parser.parse_args()

    for xyz_file in args.xyz_files:
        atomno, comment, coord = read_xyz(xyz_file)
        atomno, comment, coord = atomno[-1], comment[-1], coord[-1]
        atomno, coord = regularize(atomno, coord)

        with open(xyz_file.name, "w") as stream:
            stream.write(write_xyz(atomno, coord, comment))


if __name__ == "__main__":
    main()
