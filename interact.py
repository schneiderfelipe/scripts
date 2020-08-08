#!/usr/bin/python3

"""Non-covalently interact a set of coordinates or named structures."""

import argparse

import cirpy
import numpy as np

from rmsd import read_xyz, write_xyz


def main():
    """Run main procedure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identifiers", nargs="+")
    args = parser.parse_args()

    atomnos = []
    atomcoords = []
    for identifier in args.identifiers:
        try:
            nos, _, coords = read_xyz(cirpy.resolve(identifier, "xyz"))
        except AttributeError:
            nos, _, coords = read_xyz(identifier)
        atomnos.append(nos[-1])
        atomcoords.append(coords[-1] - np.mean(coords[-1], axis=0))

    curnos = atomnos[0]
    curcoords = atomcoords[0]
    for nos, coords in zip(atomnos[1:], atomcoords[1:]):
        curdim = curcoords.max(axis=0) - curcoords.min(axis=0)
        extradim = coords.max(axis=0) - coords.min(axis=0)
        axis = curdim.argmin()

        v = np.zeros(3)
        v[axis] = (curdim[axis] + extradim[axis]) / 2 + 2.83
        coords = coords + v
        print(write_xyz(nos, coords))


if __name__ == "__main__":
    main()
