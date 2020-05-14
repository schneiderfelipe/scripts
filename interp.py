#!/usr/bin/env python3

"""Read .interp files from ORCA NEB calculations and create graphs."""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main():
    """Run main procedure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("interp_file", type=argparse.FileType("r"), default="-")
    parser.add_argument("-a", "--all", action="store_true")
    args = parser.parse_args()

    image = []
    interp = []
    images = []
    interps = []
    for line in args.interp_file:
        line = line.strip()
        if line:
            fields = line.split()
            if fields[0] == "Iteration:":
                try:
                    images.append(image)
                    interps.append(interp)
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
                    image.append([float(entry) for entry in fields])
                if mode == "interps":
                    interp.append([float(entry) for entry in fields])

    images = np.array(images)
    interps = np.array(interps)

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
