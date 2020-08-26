#!/usr/bin/python3

import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cclib import ccopen
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.stats import cauchy, norm

sns.set(style="white", palette="colorblind")


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
    parser.add_argument("--xshift", default=90.0, type=float)
    parser.add_argument("--broad-only", action="store_true")
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
            y = spectrum["TotalSpectrum"] / spectrum["TotalSpectrum"].max()
        else:
            if spectrum_path_found:
                print("Ignoring found .spectrum file, using broadened data")
            else:
                print("No .spectrum file found, using broadened data")
            data = ccopen(logfile_path).parse()
            wavelengths = 1e7 / data.etenergies  # nm conversion

            x = np.linspace(
                wavelengths.min() - 100.0, wavelengths.max() + 100.0, num=1000
            )
            y = broaden_spectrum(x, wavelengths, data.etoscs, scale=40.0)
            y = y / y.max()

        if args.xshift:
            print(f"Shifting all wavelengths by {args.xshift} nm")
            x += args.xshift
        plt.plot(x, y, label=name)

        f = interp1d(
            x, y, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )
        res = minimize_scalar(
            lambda t: -f(t),
            bracket=(args.xmin, args.xmax),
            bounds=(args.xmin, args.xmax),
            method="bounded",
        )
        print(res)

    plt.xlim(args.xmin, args.xmax)
    plt.xlabel("wavelength (nm)")
    plt.ylabel("arbitrary units")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
