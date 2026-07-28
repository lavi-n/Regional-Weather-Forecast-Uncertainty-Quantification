"""
precip_uncertainty.py
----------------------
Temperature residuals are roughly symmetric and unimodal, so an AR(1) /
Gaussian-ish approach was reasonable. Daily precipitation is a different
animal: it's zero-inflated (most days are dry) and heavy-tailed on wet
days, so a symmetric +/- interval around a mean forecast is close to
meaningless. Instead, this treats each calendar day as having its own
empirical predictive DISTRIBUTION, estimated from a +/-10-day window
across all historical years (borrowing strength across years while
respecting seasonality), and reports quantiles rather than a mean+interval.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def conditional_quantiles(df, window=10, quantiles=(0.1, 0.5, 0.9)):
    doy = df["date"].dt.dayofyear.values
    precip = df["precip"].values
    n = len(df)
    out = {q: np.empty(366) for q in quantiles}
    prob_wet = np.empty(366)

    for target_day in range(1, 367):
        # circular distance across the day-of-year wrap (Dec 31 -> Jan 1)
        diff = np.abs(doy - target_day)
        diff = np.minimum(diff, 365 - diff)
        mask = diff <= window
        vals = precip[mask]
        prob_wet[target_day - 1] = np.mean(vals > 0.1)
        for q in quantiles:
            out[q][target_day - 1] = np.quantile(vals, q)
    return out, prob_wet


def plot_precip_climatology(out, prob_wet, out_path):
    days = np.arange(1, 367)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].fill_between(days, 0, out[0.9], color="#2980b9", alpha=0.2, label="90th percentile")
    axes[0].plot(days, out[0.5], color="#2c3e50", linewidth=1.5, label="Median")
    axes[0].set_ylabel("Precip (mm)")
    axes[0].set_title("Day-of-Year Conditional Precipitation Quantiles (+/-10 day window)",
                       fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].plot(days, prob_wet, color="#16a085", linewidth=1.5)
    axes[1].set_ylabel("P(wet day)")
    axes[1].set_xlabel("Day of year")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv("weather_data.csv", parse_dates=["date"])
    out, prob_wet = conditional_quantiles(df)
    plot_precip_climatology(out, prob_wet, "precip_climatology.png")

    # Quick calibration check: for a held-out final year, what fraction of days
    # fall at/below the day-specific 90th percentile learned from all other years?
    test = df[df["date"] >= df["date"].max() - pd.Timedelta(days=365)]
    train = df[df["date"] < df["date"].max() - pd.Timedelta(days=365)]
    out_train, _ = conditional_quantiles(train)
    doy_test = test["date"].dt.dayofyear.values
    q90_for_test_days = np.array([out_train[0.9][d - 1] for d in doy_test])
    coverage = np.mean(test["precip"].values <= q90_for_test_days)
    print(f"Held-out check: {coverage*100:.1f}% of days fell at/below the day-specific 90th percentile "
          f"(target: 90%)")
