#!/usr/bin/python3

"""Analyse metadynamics trajectories from xtb."""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.pca as pca
from MDAnalysis.lib import distances
from scipy.constants import calorie
from scipy.constants import kilo
from scipy.constants import N_A
from scipy.constants import physical_constants
from scipy import stats

from rmsd import element
from rmsd import read_xyz

hartree, _, _ = physical_constants["Hartree energy"]


commands = {
    "q": "exit",
    "h": "help",
    "e": "energies",
    "s": "current selection",
    "p": "graph",
    "*anything else*": "any MDAnalysis selection, see <https://userguide.mdanalysis.org/selections.html>.",
}


def main():
    """Run main procedure."""
    # TODO(schneiderfelipe): accept multiple files
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traj_files", nargs="+")
    args = parser.parse_args()

    gnorms = []
    energies = []
    all_atomnos = []
    all_atomcoords = []
    for traj_file in args.traj_files:
        atomnos, comments, atomcoords = read_xyz(traj_file)
        all_atomnos.extend(atomnos)
        all_atomcoords.extend(atomcoords)
        for comment in comments:
            fields = comment.split()
            gnorms.append(float(fields[3]))
            energies.append(float(fields[1]))
    energies = np.array(energies)
    energies -= energies.min()
    energies *= hartree * N_A / (kilo * calorie)

    u = mda.Universe.empty(n_atoms=len(all_atomnos[0]), trajectory=True)
    u.add_TopologyAttr("type", [element[i] for i in all_atomnos[0]])
    u.load_new(all_atomcoords, order="fac")
    print(u)

    selection = None
    print("(enter 'q' for exit, 'h' for help)")
    while True:
        code = input("select> ").strip().split()
        if code[0] == "q":
            break
        elif code[0] == "h":
            for key in commands:
                print(f"{key:15s}: {commands[key]}")
        elif code[0] == "e":
            fig, ax = plt.subplots(2)
            ax[0].plot(energies)
            ax[0].set_xlabel("frame")
            ax[0].set_ylabel("energy (kcal/mol)")

            ax[1].plot(gnorms)
            ax[1].set_xlabel("frame")
            ax[1].set_ylabel("grad. norm (Eh/a0)")
            plt.show()
        elif code[0] == "s":
            print(selection)
            if selection is not None:
                print(selection_text)
        elif code[0] == "pca":
            if selection is None:
                print("empty selection, doing nothing")
                continue

            p = pca.PCA(u, select=selection_text)
            p.run()

            n_pcs = np.where(p.cumulated_variance > 0.95)[0][0]
            print(n_pcs)
            print(p.cumulated_variance[0:n_pcs])
            pca_space = p.transform(selection, n_components=n_pcs)
            print(pca_space)
            print(pca.cosine_content(pca_space, 0))
        elif code[0] == "p":
            if selection is None:
                print("empty selection, doing nothing")
                continue

            n = len(selection)
            if n == 2:
                data_label = "bond length (Å)"
            elif n == 3:
                data_label = "bond angle (°)"
            elif n == 4:
                data_label = "dihedral angle (°)"
            else:
                print("too few or too many indices")
                continue

            data = []
            for i, (e, ts) in enumerate(zip(energies, u.trajectory)):
                if n == 2:
                    d = distances.calc_bonds(
                        selection[0].position, selection[1].position
                    )
                elif n == 3:
                    d = np.degrees(
                        distances.calc_angles(
                            selection[0].position,
                            selection[1].position,
                            selection[2].position,
                        )
                    )
                elif n == 4:
                    d = np.degrees(
                        distances.calc_dihedrals(
                            selection[0].position,
                            selection[1].position,
                            selection[2].position,
                            selection[3].position,
                        )
                    )

                data.append(d)
                if i % 100 == 0 or i == len(u.trajectory) - 1:
                    print(
                        f"frame = {ts.frame:4d}: e = {e:5.1f} kcal/mol, {data_label.split('(')[0][:-1]} = {d:7.3f} {data_label[-2]}"
                    )
            data = np.array(data)

            fig, ax = plt.subplots(1, 2)
            ax[0].plot(data)
            ax[0].set_xlabel("frame")
            ax[0].set_ylabel(data_label)

            ax[1].plot(energies, data, "o", label="data points")
            ax[1].set_xlabel("energy (kcal/mol)")
            ax[1].set_ylabel(data_label)

            if n == 2:
                dx = 0.1
            elif n == 3:
                dx = 10.0
            elif n == 4:
                dx = 10.0
            res = stats.binned_statistic(
                data, energies, "min", min(25, (data.max() - data.min()) / dx)
            )

            # print(res.statistic)
            mask = np.isnan(res.statistic)
            res.statistic[mask] = np.interp(
                np.flatnonzero(mask),
                np.flatnonzero(~mask),
                res.statistic[~mask],
            )
            # print(res.statistic)

            # ax[1].hlines(res.statistic, res.bin_edges[:-1], res.bin_edges[1:], colors='g', lw=2, label='binned min. energies')
            ax[1].barh(
                (res.bin_edges[:-1] + res.bin_edges[1:]) / 2,
                res.statistic,
                align="center",
                height=res.bin_edges[1:] - res.bin_edges[:-1],
                alpha=0.25,
                label="binned min. energies",
            )
            ax[1].legend()

            plt.show()

        else:
            try:
                selection_text = " ".join(code)
                selection = u.select_atoms(selection_text)
            except mda.exceptions.SelectionError as e:
                print(e)

    print("bye")


if __name__ == "__main__":
    main()
