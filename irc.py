#!/usr/bin/env python3

"""Read .out files from ORCA IRC calculations and create graphs."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def smooth(values):
    """Smooth a sequence of values."""
    return savgol_filter(values, 1 + 2 * len(values) // 8, 4, mode="nearest")


def main():
    """Run main procedure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("out_file", type=argparse.FileType("r"), default="-")
    args = parser.parse_args()

    irc = False
    data = []
    for line in args.out_file:
        if "Step        E(Eh)      dE(kcal/mol)  max(|G|)   RMS(G)" in line:
            irc = True
        elif irc:
            line = line.strip()
            if line:
                fields = line.split()
                data.append([float(x) for x in fields[:5]])
                if len(fields) > 5 and fields[5] == "<=":
                    ts_step = int(data[-1][0]) - 1  # first step is one
            else:
                break

    data = np.array(data)
    data[:, 0] = (data[:, 0] - data[:, 0].min()) / data[:, 0].max()
    data[:, 1] = data[:, 1] - data[:, 1].min()

    # TODO(schneiderfelipe): interpolate before and use small step for
    # differentiation.
    # reaction_force = -np.gradient(data[:, 1], data[:, 0])
    # reaction_force_constant = -np.gradient(reaction_force, data[:, 0])

    plt.subplot(211)
    plt.plot(data[:, 0], data[:, 1], "o--")
    plt.vlines(data[ts_step, 0], 0, data[:, 1].max())
    plt.xlabel(r"IRC ($\xi$)")
    plt.ylabel(r"Potential Energy, V($\xi$) [Eh]")

    # plt.subplot(412)
    # plt.plot(data[:, 0], reaction_force, "o--")
    # plt.vlines(data[ts_step, 0], reaction_force.min(), reaction_force.max())
    # plt.xlabel(r"IRC ($\xi$)")
    # plt.ylabel(r"Reaction Force, F($\xi$) [Eh/$\Delta\xi$]")

    # plt.subplot(413)
    # plt.plot(data[:, 0], reaction_force_constant, "o--")
    # plt.vlines(data[ts_step, 0], reaction_force_constant.min(),
    #            reaction_force_constant.max())
    # plt.xlabel(r"IRC ($\xi$)")
    # plt.ylabel(
    #     r"Reaction Force Constant, $\kappa$($\xi$) [Eh/$\Delta\xi^2$]"
    # )

    plt.subplot(212)
    plt.plot(data[:, 0], data[:, 3], "o--", label="max(|G|)")
    plt.vlines(data[ts_step, 0], 0, data[:, 3].max())
    plt.plot(data[:, 0], data[:, 4], "x", label="RMS(G)")
    plt.xlabel(r"IRC ($\xi$)")
    plt.ylabel(r"Gradient, G($\xi$) [Eh/Bohr]")
    plt.legend()

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
