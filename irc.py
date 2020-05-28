#!/usr/bin/python3

"""Read .out files from ORCA IRC calculations and create graphs."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import calorie
from scipy.constants import kilo
from scipy.constants import N_A
from scipy.constants import physical_constants
from scipy import interpolate

from rmsd import calc_rmsd
from rmsd import read_xyz

hartree, _, _ = physical_constants["Hartree energy"]


def main():
    """Run main procedure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("out_file", type=argparse.FileType("r"), default="-")
    parser.add_argument("--classic", action="store_true")
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

    coords = read_xyz(args.out_file.name.replace(".out", "_TSOpt_IRC_Full_trj.xyz"))[1]
    rmsd = [0.0]
    for i in range(1, len(coords)):
        rmsd.append(rmsd[-1] + calc_rmsd(coords[i - 1], coords[i]))
    rmsd = np.array(rmsd)

    data = np.array(data)
    # xi = (data[:, 0] - data[:, 0].min()) / data[:, 0].max()
    xi = rmsd
    y = data[:, 1] - data[:, 1].min()

    forward_barrier = y.max() - y[0]
    backward_barrier = y.max() - y[-1]
    print(
        f"forward barrier  = {forward_barrier:6.4f} Eh = {forward_barrier * hartree * N_A / kilo:5.1f} kJ/mol = {forward_barrier * hartree * N_A / (kilo * calorie):5.1f} kcal/mol"
    )
    print(
        f"backward barrier = {backward_barrier:6.4f} Eh = {backward_barrier * hartree * N_A / kilo:5.1f} kJ/mol = {backward_barrier * hartree * N_A / (kilo * calorie):5.1f} kcal/mol"
    )

    if not args.classic:
        xi_new = np.linspace(xi.min(), xi.max(), 10000)

        # points = ~np.isclose(xi, xi[ts_step])
        # f = interpolate.InterpolatedUnivariateSpline(xi[points], y[points])
        f = interpolate.InterpolatedUnivariateSpline(xi, y, k=4)
        fp = f.derivative()
        fpp = f.derivative(n=2)

        plt.subplot(311)
    else:
        plt.subplot(211)

    plt.plot(xi, y, "o", label="calculated")
    plt.vlines(xi[ts_step], y.min(), y.max())
    plt.xlabel(r"IRC ($\xi$)")
    plt.ylabel(r"Potential Energy, V($\xi$) [Eh]")

    if not args.classic:
        # add interpolation
        pass
        plt.plot(xi_new, f(xi_new), "--", label="interpolated")
        plt.legend()

        plt.subplot(312)
        rf = -fp(xi_new)
        plt.plot(xi_new, rf, "--")
        plt.vlines(xi[ts_step], rf.min(), rf.max())
        plt.xlabel(r"IRC ($\xi$)")
        plt.ylabel(r"Reaction Force, F($\xi$) [Eh/$\Delta\xi$]")

        # TODO(schneiderfelipe): get max and min force along the coordinate
        # and do the usual analysis
        plt.subplot(313)
        rfc = fpp(xi_new)
        plt.plot(xi_new, rfc, "--")
        plt.vlines(xi[ts_step], rfc.min(), rfc.max())
        plt.xlabel(r"IRC ($\xi$)")
        plt.ylabel(r"Reaction Force Constant, $\kappa$($\xi$) [Eh/$\Delta\xi^2$]")
    else:
        plt.subplot(212)
        plt.plot(xi, data[:, 3], "o", label="max(|G|)")
        plt.vlines(xi[ts_step], 0, data[:, 3].max())
        plt.plot(xi, data[:, 4], "o", label="RMS(G)")
        plt.xlabel(r"IRC ($\xi$)")
        plt.ylabel(r"Gradient, G($\xi$) [Eh/Bohr]")
        plt.legend()

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
