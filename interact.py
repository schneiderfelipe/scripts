#!/usr/bin/python3

"""Non-covalently interact a set of coordinates or named structures."""

import argparse

import cirpy

from rmsd import read_xyz


def main():
    """Run main procedure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identifiers", nargs="+")
    args = parser.parse_args()

    for identifier in args.identifiers:
        try:
            nos, coords = read_xyz(cirpy.resolve(identifier, "xyz"))
        except AttributeError:
            nos, coords = read_xyz(identifier)
        print(nos, coords)


if __name__ == "__main__":
    main()
