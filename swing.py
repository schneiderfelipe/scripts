#!/usr/bin/python3

"""Swing molecules based on normal coordinates."""

import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.calculators.orca import ORCA
from cclib import ccopen
from scipy.constants import physical_constants
from scipy.optimize import root
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from xtb.ase.calculator import XTB

hartree, _, _ = physical_constants["Hartree energy"]

smd2gbsa = defaultdict(
    lambda: "none",
    {
        "acetone": "acetone",
        "mecn": "acetonitrile",
        "acetonitrile": "acetonitrile",
        "benzene": "benzene",
        "dichloromethane": "ch2cl2",
        "chloroform": "chcl3",
        "carbon disulfide": "cs2",
        "dmf": "dmf",
        "n,n-dimethylformamide": "dmf",
        "dmso": "dmso",
        "dimethylsulfoxide": "dmso",
        "diethyl ether": "ether",
        "water": "water",
        "methanol": "methanol",
        "n-hexane": "n-hexan",
        "thf": "thf",
        "tetrahydrofuran": "thf",
        "toluene": "toluene",
    },
)


def main():
    """Run main procedure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logfile")
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
        "--gfn", help="specify parametrisation of GFN-xTB", type=int
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

    parser.add_argument(
        "--minimize", action="store_true",
    )
    parser.add_argument(
        "--transition-state", action="store_true",
    )
    parser.add_argument("--max-omega", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()
    print(args)

    data = ccopen(args.logfile).parse()
    initial_positions = data.atomcoords[-1]
    atoms = Atoms(numbers=data.atomnos, positions=initial_positions)

    if args.gfn:
        method = f"GFN{args.gfn}-xTB"
        solvent = smd2gbsa[args.solvent.lower()]

        calc = XTB(
            method=method,
            accuracy=args.acc,
            electronic_temperature=args.etemp,
            max_iterations=args.iterations,
            solvent=solvent,
            cache_api=args.cache_api,
        )
    else:

        def allow_keyword(keyword):
            for forbidden in {"freq", "opt", "print"}:
                if forbidden in keyword.lower():
                    return False
            return True

        keywords = [
            keyword
            for keyword in data.metadata["keywords"]
            if allow_keyword(keyword)
        ]
        method = " ".join(keywords)
        solvent = args.solvent
        blocks = f"%pal\n nprocs 6\nend\n%scf\n maxiter {args.iterations}\nend"
        if solvent != "none":
            blocks += f'\n%cpcm\n smd true\n smdsolvent "{solvent}"\nend'

        calc = ORCA(
            label="012345_swing", orcasimpleinput=method, orcablocks=blocks
        )

    print(f"*** {method} ***")
    print(f"    : solvent:              {solvent}")

    atoms.set_calculator(calc)
    potential_min = atoms.get_potential_energy()
    print(f"@ potential energy:        {potential_min} eV")

    indices = np.where(data.vibfreqs < 0)[0]
    n_indices = len(indices)
    print(f"@ imaginary frequencies:   {data.vibfreqs[indices]}")
    if not n_indices:
        print("    : nothing to be done, bye")
        return

    ignoring = None
    if args.transition_state:
        ignoring = 0
        print("    : transition state:    ignoring first imaginary frequency")

    omegas = []
    potentials = []

    def f(omega):
        atoms.set_positions(
            initial_positions
            + np.einsum("i,ijk->jk", omega, data.vibdisps[indices])
        )

        potential = 1e3 * (atoms.get_potential_energy() - potential_min)

        omegas.append(omega)
        potentials.append(potential)
        print(f"    : omega:               {omega}")
        print(f"    : potential:           {potential} meV")

        return potential

    if args.minimize:
        guesses = [np.zeros_like(indices, dtype=float)]

    for i in indices:
        if ignoring is not None and i == ignoring:
            continue

        print(f"@ searching in direction   #{i}")

        def g(w):
            z = np.zeros_like(indices, dtype=float)
            z[i] = w
            return f(z)

        if args.minimize:
            res = minimize_scalar(
                g,
                method="bounded",
                bounds=(-args.max_omega, args.max_omega),
                tol=args.tol,
            )
            print(res)

            guess = np.zeros_like(indices, dtype=float)
            guess[i] = res.x
            guesses.append(guess)
        else:
            dx = args.max_omega / 100
            x = [-dx, 0.0, dx]
            y = [g(-dx), 0.0, g(dx)]

            # p[0] * x**2 + p[1] * x + p[2] == k * (x - x0)**2 == k * x**2 - 2 * x0 * k * x + k * x0**2
            p = np.polyfit(x, y, 2)
            print(p)
            print(np.roots(p))

            dp = np.polyder(p)
            print(dp)
            r = np.roots(dp)
            print(r)

            # k = p[0]
            # x0 = np.sqrt(p[2] / k)
            # print(k, x0)
            # print(root(lambda z: [p[0] - z[0], p[1] + 2 * z[0] * z[1], p[2] - z[0] * z[1] ** 2], [k, x0]))

            best_positions = initial_positions + np.einsum(
                "i,ijk->jk", r, data.vibdisps[indices]
            )

    if args.minimize:
        print("@ choosing initial guess for global search")
        if n_indices > 1:
            guesses.append(np.sum(guesses, axis=0))
        x0 = guesses[np.argmin([f(guess) for guess in guesses])]

        print("@ searching in all directions")
        constraints = ()
        if args.transition_state and ignoring is not None:
            constraints = (
                {"type": "eq", "fun": lambda omega: omega[ignoring]},
            )
        res = minimize(
            f,
            x0=x0,
            bounds=n_indices * [(-args.max_omega, args.max_omega)],
            constraints=constraints,
            tol=args.tol,
        )
        print(res)
        best_positions = initial_positions + np.einsum(
            "i,ijk->jk", res.x, data.vibdisps[indices]
        )

        # TODO(schneiderfelipe): correct for when using --transition-state
        omegas = np.array(omegas)
        fig, ax = plt.subplots(n_indices, 1)
        if n_indices == 1:
            ax = [ax]
        xlim = (-args.max_omega - 0.05, args.max_omega + 0.05)
        ylim = (np.min(potentials) - 2.0, 40.0)
        for i in indices:
            if ignoring is not None and i == ignoring:
                continue

            ax[i].plot(omegas[:, i], potentials, "o")
            ax[i].set_title(f"view of normal mode #{i}")
            ax[i].set_ylabel(r"potential energy (meV)")
            ax[i].set_xlabel(rf"$\omega_{i}$")
            ax[i].set_ylim(ylim)
            ax[i].set_xlim(xlim)
        plt.tight_layout()
        plt.show()

    print("@ writing best geometry to swinged.xyz")
    # TODO(schneiderfelipe): print a RMSD between initial and final structures
    atoms.set_positions(best_positions)
    atoms.write("swinged.xyz", format="xyz", plain=True)


if __name__ == "__main__":
    main()
