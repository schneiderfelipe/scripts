#!/usr/bin/python3

"""Optimize the geometry of molecules or fragments."""

import argparse

from ase import Atoms
from ase.optimize import BFGS
from xtb.ase.calculator import XTB


from rmsd import read_xyz, write_xyz
from swing import smd2gbsa


def minimize(
    atomno,
    coord,
    method="GFN2-xTB",
    accuracy=1.0,
    electronic_temperature=300.0,
    max_iterations=250,
    solvent="water",
    cache_api=True,
):
    atoms = Atoms(numbers=atomno, positions=coord)
    calc = XTB(
        method=method,
        accuracy=accuracy,
        electronic_temperature=electronic_temperature,
        max_iterations=max_iterations,
        solvent=solvent,
        cache_api=cache_api,
    )
    atoms.set_calculator(calc)
    opt = BFGS(atoms)
    opt.run(fmax=0.05)

    return atoms.numbers, atoms.get_positions()


def main():
    """Run main procedure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xyz_files", type=argparse.FileType("r"), default="-", nargs="+"
    )
    # TODO(schneiderfelipe): set charge and multiplicity
    parser.add_argument(
        "-a",
        "--acc",
        help="accuracy for SCC calculation, lower is better",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--iterations",
        help="number of iterations in SCC",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--gfn",
        help="specify parametrisation of GFN-xTB",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--etemp", help="electronic temperature", type=float, default=300.0
    )
    parser.add_argument(
        "-s",
        "--solvent",
        help=("solvent (SMD/GBSA implicit solvation models)"),
        default="none",
    )
    parser.add_argument(
        "--do-not-cache-api",
        dest="cache_api",
        help="Do not reuse generate API objects (not recommended)",
        action="store_false",
    )
    args = parser.parse_args()
    print(args)

    method = f"GFN{args.gfn}-xTB"
    solvent = smd2gbsa[args.solvent.lower()]

    for xyz_file in args.xyz_files:
        atomno, comment, coord = read_xyz(xyz_file)
        atomno, comment, coord = atomno[-1], comment[-1], coord[-1]

        atomno, coord = minimize(
            atomno,
            coord,
            method=method,
            accuracy=args.acc,
            electronic_temperature=args.etemp,
            max_iterations=args.iterations,
            solvent=solvent,
            cache_api=args.cache_api,
        )

        with open(xyz_file.name, "w") as stream:
            stream.write(write_xyz(atomno, coord, comment))


if __name__ == "__main__":
    main()
