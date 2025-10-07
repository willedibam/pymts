from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from pymts import Generator
from pymts.plotting import plot_heatmap, plot_phase, plot_timeseries


def test_plotting_helpers_smoke() -> None:
    gen = Generator()
    ds = gen.generate(
        [
            {
                "model": "noise_gaussian",
                "M": 3,
                "T": 32,
                "sigma": 0.4,
                "n_realizations": 2,
                "seed": 14,
            }
        ]
    )[0]

    ax1 = plot_heatmap(ds, realization=0)
    ax1.figure.canvas.draw()

    ax2 = plot_timeseries(ds, realization=0, stems=True)
    ax2.figure.canvas.draw()

    ax3 = plot_phase(ds, realization=0, max_channels=2)
    ax3.figure.canvas.draw()

    plt.close(ax1.figure)
    plt.close(ax2.figure)
    plt.close(ax3.figure)
