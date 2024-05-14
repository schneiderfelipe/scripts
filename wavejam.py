#!/usr/bin/python3

"""Convert wavefunction files using MultiWFN."""

import argparse
from functools import partial
import os
import subprocess


def convert(infile, outfile):
    """Convert wavefunction files.

    Parameters
    ----------
    infile, outfile : str
    """
    inbase, inext = os.path.splitext(infile)
    inext = inext[1:]
    outbase, outext = os.path.splitext(outfile)
    outext = outext[1:]

    if inext == "gbw":
        if outext == "mkl":
            p = subprocess.run(
                ["orca_2mkl", inbase, "-mkl"], stdout=subprocess.PIPE
            )
        elif outext == "molden":
            p = subprocess.run(
                ["orca_2mkl", inbase, "-molden"], stdout=subprocess.PIPE
            )
            os.rename(f"{inbase}.molden.input", outfile)
    elif inext == "mkl":
        if outext == "gbw":
            p = subprocess.run(
                ["orca_2mkl", inbase, "-gbw"], stdout=subprocess.PIPE
            )
    else:
        # TODO(schneiderfelipe): convert to pdb using OpenBabel? If possible,
        # the result is improved, but use the code below if something fails.
        selector = """100        // Other functions (Part 1)
                      2          // Export various files...
                      {code}     // Output current wavefunction as...
                      {outfile}
                      0          // Return
                      q          // Exit program gracefully"""
        input = {
            "pdb": partial(selector.format, code="1"),
            "xyz": partial(selector.format, code="2"),
            "chg": partial(selector.format, code="3"),
            "wfx": partial(selector.format, code="4"),
            "wfn": partial(selector.format, code="5"),
            "molden": partial(selector.format, code="6"),
            "fch": partial(selector.format, code="7"),
            "47": partial(selector.format, code="8"),
            "mkl": partial(selector.format, code="9"),
        }

        p = subprocess.run(
            ["Multiwfn", infile],
            stdout=subprocess.PIPE,
            input=input[outext](outfile=outfile),
            encoding="ascii",
        )
        # os.rename(f"{inbase}.{outext}", outfile)

    if p.returncode:
        raise ValueError(f"conversion {p.args} failed: {p.stdout}")


def main():
    """Run main procedure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("-O", "--outfile", help="specify the output file")
    args = parser.parse_args()

    convert(args.infile, args.outfile)


if __name__ == "__main__":
    main()
