#!/usr/bin/python3

import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from cclib import ccopen
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.stats import cauchy, norm

sns.set(style="ticks", palette="colorblind", font_scale=1.25)


def process_name(name: str) -> str:
    """A hack to make compound names bold in some common cases."""
    if "+" in name:
        pieces = name.split("+", 1)
        name = f"$\\bf{{{pieces[0]}}}$+{pieces[1]}"
    return name


def broaden_spectrum(
    x,
    x0,
    y0,
    distribution="gaussian",
    scale=1.0,
    fit_points=True,
    *args,
    **kwargs,
):
    if distribution in {"gaussian", "norm"}:
        distribution = norm
    elif distribution in {"lorentzian", "cauchy"}:
        distribution = cauchy

    s = np.sum(
        [
            yp * distribution.pdf(x, xp, scale=scale, *args, **kwargs)
            for xp, yp in zip(x0, y0)
        ],
        axis=0,
    )

    if fit_points:
        s_max = np.max(s)
        if s_max == 0.0:
            s_max = 1.0
        return s * np.max(y0) / s_max
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfiles", metavar="logfile", nargs="+")
    parser.add_argument("--xmin", default=400.0, type=float)
    parser.add_argument("--xmax", default=700.0, type=float)
    parser.add_argument("--xshift", default=80.0, type=float)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--broad-only", action="store_true")
    parser.add_argument("--save-path")
    args = parser.parse_args()

    for logfile_path in args.logfiles:
        # We ask for label=path pairs.
        parts = logfile_path.split("=")
        name = parts[0]
        logfile_path = parts[-1]
        print(name.center(80, "-"))
        print(logfile_path)

        spectrum_path = logfile_path.replace(".out", ".spectrum")
        spectrum_path_found = os.path.isfile(spectrum_path)
        if not args.broad_only and spectrum_path_found:
            print(".spectrum file found")

            spectrum = pd.read_csv(spectrum_path, sep="\s+", index_col=0)

            x = spectrum.index
            if args.normalize:
                y = spectrum["TotalSpectrum"] / spectrum["TotalSpectrum"].max()
        else:
            if spectrum_path_found:
                print("Ignoring found .spectrum file, using broadened data")
            else:
                print("No .spectrum file found, using broadened data")
            data = ccopen(logfile_path).parse()
            wavelengths = 1e7 / data.etenergies  # nm conversion

            xmin = min(args.xmin, wavelengths.min())
            xmax = max(args.xmax, wavelengths.max())
            x = np.linspace(xmin - 100.0, xmax + 100.0, num=1000)
            y = broaden_spectrum(x, wavelengths, data.etoscs, scale=40.0)
            if args.normalize:
                y = y / y.max()

        if args.xshift:
            print(f"Shifting all wavelengths by {args.xshift} nm")
            x += args.xshift
        plt.plot(x, y, label=process_name(name))
        xmin = max(args.xmin, x.min())
        xmax = min(args.xmax, x.max())

        f = interp1d(x, y, kind="cubic")
        res = minimize_scalar(
            lambda t: -f(t),
            bracket=(xmin, xmax),
            bounds=(xmin, xmax),
            method="bounded",
        )
        print(res)

    plt.xlim(args.xmin, args.xmax)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Arbitrary units")
    plt.legend()

    if args.save_path:
        plt.savefig(args.save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
