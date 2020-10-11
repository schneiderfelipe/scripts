#!/usr/bin/python3

"""Reorder atoms such that molecules resemble each other the most."""

import argparse
from itertools import permutations

import numpy as np

from rmsd import read_xyz, write_xyz, calc_rmsd


def reorder(uniquenos, refatomno, refcoord, atomno, coord):
    coord_no = {}
    for no in uniquenos:
        refids = refatomno == no
        refcoord_no = refcoord[refids]

        ids = atomno == no
        coord_no[no] = coord[ids]

        rmsd_no = np.inf
        # TODO(schneiderfelipe): there might be better ways to do it than loop
        # over permutations (dynamic programming maybe?).
        for perm in permutations(coord_no[no]):
            # TODO(schneiderfelipe): if RMSD is *very* small (almost zero),
            # could we simply stop?
            perm = np.asanyarray(perm)
            r = calc_rmsd(refcoord_no, coord_no[no])
            if rmsd_no > r:
                rmsd_no = r
                coord_no[no] = perm

        coord_no[no] = list(coord_no[no])

    atomno = []
    coord = []
    for no in refatomno:
        atomno.append(no)
        # TODO(schneiderfelipe): pop(0) is probably very inefficient.
        coord.append(coord_no[no].pop(0))

    return atomno, coord


def main():
    """Run main procedure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xyz_files", type=argparse.FileType("r"), default="-", nargs="+"
    )
    args = parser.parse_args()

    refname = args.xyz_files[0].name
    refatomno, _, refcoord = read_xyz(args.xyz_files[0])
    refatomno, refcoord = refatomno[-1], refcoord[-1]
    uniquenos = sorted(set(refatomno))

    for xyz_file in args.xyz_files[1:]:
        atomno, comment, coord = read_xyz(xyz_file)
        atomno, coord = reorder(
            uniquenos, refatomno, refcoord, atomno[-1], coord[-1]
        )

        with open(xyz_file.name, "w") as stream:
            stream.write(write_xyz(atomno, coord, comment[-1]))


if __name__ == "__main__":
    main()
