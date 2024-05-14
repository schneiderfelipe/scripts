#!/usr/bin/python3

"""Interpolate between structures.

It reads a single command, path to coordinates file or single integer per line.
The program attempts to interpolate that integer number of points between given
structures.

Commands indicate how this is done. Understood commands are "linear", "idpp",
"pm3", "am1" and "xtb2".
"""

import os
import argparse
from contextlib import redirect_stdout

# Test https://wiki.fysik.dtu.dk/ase/ase/optimize.html#scipy-optimizers
# Test https://wiki.fysik.dtu.dk/ase/ase/optimize.html#preconditioned-optimizers
# Test https://wiki.fysik.dtu.dk/ase/ase/optimize.html#global-optimization

from ase import io
from ase.neb import NEB
from ase.optimize import LBFGS
from ase.optimize import FIRE
from ase.calculators.orca import ORCA

from rmsd import calc_rmsd

# TODO(schneiderfelipe): transform in command-line parameters
prefix = "interpolator"
neb_kwargs = {
    "dynamic_relaxation": True,
    "scale_fmax": 3.0,
    "method": "aseneb",
}
calc_kwargs = {"charge": 0, "mult": 1, "label": prefix}
opt_kwargs = {"trajectory": f"{prefix}.traj"}
run_kwargs = {"fmax": 3.0, "steps": 10}


# TODO(schneiderfelipe): support reading pieces of the path from .xyz files
# TODO(schneiderfelipe): support minimizing ends
# TODO(schneiderfelipe): support constraints (https://wiki.fysik.dtu.dk/ase/tutorials/neb/diffusion.html#diffusion-tutorial)
# TODO(schneiderfelipe): support writing final complete traj file (https://wiki.fysik.dtu.dk/ase/ase/neb.html#trajectories)
# TODO(schneiderfelipe): support NEBTools analysis? (https://wiki.fysik.dtu.dk/ase/ase/neb.html#analysis-of-output)
def main():
    """Run main procedure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "commands",
        metavar="input_file",
        nargs="?",
        type=argparse.FileType("r"),
        default="-",
        help="input text file with commands",
    )
    parser.add_argument(
        "-O",
        "--output-file",
        type=argparse.FileType("w"),
        default="-",
        help="output coordinates file",
    )
    parser.add_argument(
        "--no-final-neb",
        dest="final_neb",
        action="store_false",
        help="piecewise nudged-elastic band only",
    )
    parser.add_argument(
        "--no-opt",
        dest="opt",
        action="store_false",
        help="use final states as given",
    )
    # TODO(schneiderfelipe): check extension .allxyz instead of using this flag!
    parser.add_argument(
        "--use-allxyz-format",
        action="store_true",
        help="use ORCA's .allxyz format",
    )
    args = parser.parse_args()

    n = 0
    auto_n = True
    images = []
    method, theory = "linear", None
    for line in args.commands:
        for command in line.split():
            command = command.strip()
            if command == "linear":
                method, theory = "linear", None
                print("@ switching to linear interpolation")
            elif command == "idpp":
                method, theory = "idpp", None
                print("@ switching to IDPP interpolation")
            elif command in {"pm3", "am1", "xtb2"}:
                # method == "idpp"
                theory = command
                print(f"@ switching to {command.upper()}")
            elif command == "auto":
                auto_n = True
                print("@ using automatic number of points")
            else:
                try:
                    structure = io.read(command)
                    print(f"@ reading structure from '{command}'")
                except FileNotFoundError:
                    try:
                        n = int(command)
                        auto_n = False
                        print(f"@ using {n} points")
                    except ValueError:
                        parser.error(
                            f"could not understand command or find file: '{command}'"
                        )
                    continue

                if len(images) > 0 and (auto_n or n > 0):
                    if auto_n:
                        rmsd = calc_rmsd(
                            images[-1].get_positions(),
                            structure.get_positions(),
                        )
                        n = max(1, int(rmsd / 0.3))

                    pieces = [images[-1]]
                    pieces += [images[-1].copy() for _ in range(n)]
                    pieces += [structure]
                    if method == "idpp":
                        neb = NEB(
                            pieces,
                            remove_rotation_and_translation=False,
                            **neb_kwargs,
                        )
                    else:
                        neb = NEB(
                            pieces,
                            remove_rotation_and_translation=True,
                            **neb_kwargs,
                        )
                    neb.interpolate(method)
                    if theory is not None:
                        # TODO(schneiderfelipe): avoid repeat the code for NEB
                        orcablocks = ""
                        if theory not in {"pm3", "am1"}:
                            orcablocks = f"%pal nprocs {os.cpu_count()} end"
                        for piece in pieces[1:-1]:
                            piece.calc = ORCA(
                                orcasimpleinput=f"{theory} loosescf nososcf",
                                orcablocks=orcablocks,
                                **calc_kwargs,
                            )
                        if n < 5:
                            # best for 2 and 4 structures
                            opt = FIRE(neb, **opt_kwargs)
                        else:
                            # best for 3 and 10 structures
                            opt = LBFGS(neb, **opt_kwargs)
                        opt.run(**run_kwargs)
                    images = images[:-1] + pieces
                else:
                    images.append(structure)

    run_kwargs.update({"fmax": run_kwargs["fmax"] / 5})
    if args.opt and theory is not None:
        # TODO(schneiderfelipe): avoid repeat the code for NEB
        orcablocks = ""
        if theory not in {"pm3", "am1"}:
            orcablocks = f"%pal nprocs {os.cpu_count()} end"
        images[0].calc = ORCA(
            orcasimpleinput=f"{theory} loosescf nososcf",
            orcablocks=orcablocks,
            **calc_kwargs,
        )
        images[-1].calc = ORCA(
            orcasimpleinput=f"{theory} loosescf nososcf",
            orcablocks=orcablocks,
            **calc_kwargs,
        )

        opt = LBFGS(images[0], **opt_kwargs)
        opt.run(**run_kwargs)

        opt = LBFGS(images[-1], **opt_kwargs)
        opt.run(**run_kwargs)

    if args.final_neb and theory is not None and len(images) > 2:
        neb = NEB(images, remove_rotation_and_translation=True, **neb_kwargs)

        # TODO(schneiderfelipe): avoid repeat the code for NEB
        orcablocks = ""
        if theory not in {"pm3", "am1"}:
            orcablocks = f"%pal nprocs {os.cpu_count()} end"
        for image in images[1:-1]:
            image.calc = ORCA(
                orcasimpleinput=f"{theory} loosescf nososcf",
                orcablocks=orcablocks,
                **calc_kwargs,
            )
        if len(images) < 5:
            # best for 2 and 4 structures
            opt = FIRE(neb, **opt_kwargs)
        else:
            # best for 3 and 10 structures
            opt = LBFGS(neb, **opt_kwargs)

        opt.run(**run_kwargs)

    with redirect_stdout(args.output_file):
        io.write("-", images[0], format="xyz", plain=True)
        for image in images:
            if args.use_allxyz_format:
                print(">")
            io.write("-", image, format="xyz", plain=True)

    # some programs (e.g., Chemcraft) won't read without a newline at the end
    # but even with this, Chemcraft won't read if there are only 2 strucures
    args.output_file.write("\n")


if __name__ == "__main__":
    main()
