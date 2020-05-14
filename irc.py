#!/usr/bin/env python3

"""Read .out files from ORCA IRC calculations and create graphs."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def calc_rmsd(P, Q, translate=True):
    """Calculate the RMSD between P and Q using Kabsch algorithm.

    Parameters
    ----------
    P, Q : array-like
    translate : bool

    Returns
    -------
    float

    Examples
    --------
    >>> P = [[0, 0, 0],
    ...      [1, 1, 1]]
    >>> calc_rmsd(P, P)
    0.0
    >>> Q = [[1, 1, 1],
    ...      [2, 2, 2]]
    >>> calc_rmsd(P, Q)
    0.0
    >>> P = [[-3.652796902, 0.000000000, -4.445975658],
    ...      [-3.527558151, 0.000000000, -3.430150234],
    ...      [-4.501637325, 0.000000000, -3.833697320]]
    >>> Q = [[ 1.885538972, 0.000000000, -0.577489796],
    ...      [ 0.911459798, 0.000000000,  0.201773543],
    ...      [ 1.165525626, 1.078706033, -0.490031274]]
    >>> calc_rmsd(P, Q)
    0.140663071340813
    """
    P, Q = np.asanyarray(P), np.asanyarray(Q)
    if translate:
        P = P - P.mean(axis=0)
        Q = Q - Q.mean(axis=0)

    V, s, W = np.linalg.svd(P.T @ Q)
    if np.linalg.det(V) * np.linalg.det(W) < 0:
        s[-1] = -s[-1]
        V[:, -1] = -V[:, -1]

    U = V @ W
    P = P @ U
    return np.sqrt(np.sum((P - Q) ** 2) / len(P))


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

    with open(
        args.out_file.name.replace(".out", "_TSOpt_IRC_Full_trj.xyz"), "r"
    ) as xyz_file:
        lines = xyz_file.readlines()
        natom = int(lines[0])
        m = 2 + natom
        nstruct = len(lines) // m

        coords = []
        for i in range(nstruct):
            struct = []
            initial = i * m + 2
            for j in range(initial, initial + natom):
                struct.append([float(x) for x in lines[j].split()[1:]])
            coords.append(np.array(struct))

        rmsd = [0.0]
        for i in range(1, len(coords)):
            rmsd.append(rmsd[-1] + calc_rmsd(coords[i - 1], coords[i]))
        rmsd = np.array(rmsd)

    data = np.array(data)
    # xi = (data[:, 0] - data[:, 0].min()) / data[:, 0].max()
    xi = rmsd
    y = data[:, 1] - data[:, 1].min()

    if not args.classic:
        xi_new = np.linspace(xi.min(), xi.max(), 10000)

        # points = ~np.isclose(xi, xi[ts_step])
        # f = interpolate.InterpolatedUnivariateSpline(xi[points], y[points])
        f = interpolate.UnivariateSpline(xi, y, k=5, s=1e-8)
        print(f.get_residual())
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
