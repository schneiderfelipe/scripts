#!/usr/bin/python3

"""Read .interp files from ORCA NEB calculations and create graphs."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import calorie
from scipy.constants import kilo
from scipy.constants import N_A
from scipy.constants import physical_constants

hartree, _, _ = physical_constants["Hartree energy"]


def main():
    """Run main procedure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "interp_file", type=argparse.FileType("r"), default="-"
    )
    parser.add_argument("-a", "--all", action="store_true")
    args = parser.parse_args()

    images = []
    interps = []
    for line in args.interp_file:
        line = line.strip()
        if line:
            fields = line.split()
            if fields[0] == "Iteration:":
                try:
                    images.append(np.array(image))
                    interps.append(np.array(interp))
                except UnboundLocalError:
                    pass
                image = []
                interp = []
            elif fields[0] == "Images:":
                mode = "images"
            elif fields[0] == "Interp.:":
                mode = "interps"
            else:
                if mode == "images":
                    image.append(np.array([float(entry) for entry in fields]))
                if mode == "interps":
                    interp.append(np.array([float(entry) for entry in fields]))

    images = np.array(images)
    interps = np.array(interps)

    i_max = images[-1, :, 2].argmax()
    forward_barrier = images[-1, i_max, 2] - images[-1, 0, 2]
    backward_barrier = images[-1, i_max, 2] - images[-1, -1, 2]
    print(
        f"barrier is at the {i_max + 1}th (out of {len(images[-1, :, 2])}) position"
    )
    print(
        f"forward barrier  = {forward_barrier:6.4f} Eh "
        f"= {forward_barrier * hartree * N_A / kilo:5.1f} kJ/mol "
        f"= {forward_barrier * hartree * N_A / (kilo * calorie):5.1f} kcal/mol"
    )
    print(
        f"backward barrier = {backward_barrier:6.4f} Eh "
        f"= {backward_barrier * hartree * N_A / kilo:5.1f} kJ/mol "
        f"= {backward_barrier * hartree * N_A / (kilo * calorie):5.1f} kcal/mol"
    )

    if args.all:
        for i in range(len(images) - 1):
            plt.plot(images[i, :, 1], images[i, :, 2], "ok")
            plt.plot(interps[i, :, 1], interps[i, :, 2], "--", label=i)

    plt.plot(images[-1, :, 1], images[-1, :, 2], "ok")
    plt.plot(interps[-1, :, 1], interps[-1, :, 2], "--", label=-1)

    plt.xlabel("Distance  (Bohr)")
    plt.ylabel("Energy (Eh)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
