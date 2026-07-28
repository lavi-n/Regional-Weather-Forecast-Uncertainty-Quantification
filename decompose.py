"""
decompose.py
------------
Time-series decomposition of the daily mean temperature series using STL
(Seasonal-Trend decomposition using LOESS). STL is a good fit for daily
weather data because, unlike classical additive/multiplicative decomposition,
it allows the seasonal component itself to change shape slowly over years
(e.g. a shifting monsoon onset) and is robust to outliers (cold snaps /
heat waves) via its inner/outer iteration scheme.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL

from generate_data import generate_weather


def run_stl(series: pd.Series, period: int = 365, robust: bool = True) -> STL:
    """Fit STL to a daily series. period=365 captures the annual cycle."""
    stl = STL(series, period=period, robust=robust, seasonal=25)
    return stl.fit()


def plot_decomposition(dates, result: "STLResult", out_path: str):
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    components = [
        (result.observed, "Observed", "#2c3e50"),
        (result.trend, "Trend", "#c0392b"),
        (result.seasonal, "Seasonal (annual cycle)", "#2980b9"),
        (result.resid, "Residual", "#7f8c8d"),
    ]
    for ax, (data, title, color) in zip(axes, components):
        ax.plot(dates, data, color=color, linewidth=0.8)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_title("STL Decomposition of Daily Mean Temperature", fontsize=13, fontweight="bold")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    df = generate_weather()
    ts = pd.Series(df["tmean"].values, index=pd.DatetimeIndex(df["date"]))

    result = run_stl(ts)
    plot_decomposition(df["date"], result, "stl_decomposition.png")

    # Quantify how much variance each component explains
    var_obs = ts.var()
    var_trend = result.trend.var()
    var_seasonal = result.seasonal.var()
    var_resid = result.resid.var()

    print("Variance decomposition of daily mean temperature:")
    print(f"  Total variance:     {var_obs:8.2f}")
    print(f"  Trend variance:     {var_trend:8.2f}  ({100*var_trend/var_obs:5.1f}%)")
    print(f"  Seasonal variance:  {var_seasonal:8.2f}  ({100*var_seasonal/var_obs:5.1f}%)")
    print(f"  Residual variance:  {var_resid:8.2f}  ({100*var_resid/var_obs:5.1f}%)")
    print(f"\nResidual std dev (irreducible day-to-day uncertainty): {result.resid.std():.2f} C")

    result.resid.to_frame("resid").assign(date=df["date"].values).to_csv("residuals.csv", index=False)
    df.to_csv("weather_data.csv", index=False)
