"""
uncertainty.py
--------------
Quantifies forecast uncertainty for daily mean temperature, comparing two
approaches, then BACKTESTS both against real held-out data to check whether
the stated confidence level actually matches observed coverage. This
calibration check is the piece that's easy to skip and is exactly the kind
of thing that separates "made a forecast" from "quantified how much to
trust the forecast."

Method A - Analytic (parametric) intervals:
    Model the STL residual as an AR(1) process: e_t = phi*e_{t-1} + w_t.
    Under that model, the h-step-ahead forecast variance has a closed form:
        Var(e_{t+h} | e_t) = sigma_w^2 * (1 - phi^(2h)) / (1 - phi^2)
    which grows with horizon and asymptotes to the unconditional residual
    variance. Intervals are trend + seasonal +/- z * sqrt(that variance).
    Fast, interpretable, but assumes Gaussian, stationary AR(1) residuals.

Method B - Residual block bootstrap:
    Resample contiguous blocks (not single points, to preserve the
    day-to-day autocorrelation weather actually has) from the historical
    STL residuals and add them to the trend+seasonal forecast. Repeat
    thousands of times to build an empirical predictive distribution.
    Fewer distributional assumptions, but relies on residual behavior
    staying similar in the future.

Backtest:
    Hold out the final year of data. Fit STL on everything before it,
    forecast into the held-out year with both methods, and check what
    fraction of actual observations fall inside the nominal 90% interval.
    A well-calibrated method should cover ~90% of held-out days.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

from generate_data import generate_weather


def fit_ar1(resid: np.ndarray):
    """Fit AR(1) by OLS on lagged residuals; return phi and innovation std."""
    x, y = resid[:-1], resid[1:]
    phi = np.sum(x * y) / np.sum(x * x)
    innov = y - phi * x
    sigma_w = innov.std(ddof=1)
    return phi, sigma_w


def analytic_interval(last_resid, phi, sigma_w, horizon, z=1.645):
    """Method A: AR(1)-implied prediction interval half-width per lead time."""
    h = np.arange(1, horizon + 1)
    var_h = sigma_w**2 * (1 - phi ** (2 * h)) / (1 - phi**2)
    center = last_resid * phi**h
    half_width = z * np.sqrt(var_h)
    return center, half_width


def block_bootstrap_paths(resid: np.ndarray, horizon: int, n_boot: int, block_len: int, rng):
    """Method B: sample forecast residual paths via moving-block bootstrap."""
    n = len(resid)
    n_blocks_needed = int(np.ceil(horizon / block_len))
    paths = np.empty((n_boot, horizon))
    max_start = n - block_len
    for b in range(n_boot):
        chunks = []
        for _ in range(n_blocks_needed):
            start = rng.integers(0, max_start)
            chunks.append(resid[start:start + block_len])
        path = np.concatenate(chunks)[:horizon]
        paths[b] = path
    return paths


def backtest(df, horizon=90, n_boot=2000, block_len=10, seed=11):
    rng = np.random.default_rng(seed)
    cutoff = len(df) - horizon
    train, test = df.iloc[:cutoff], df.iloc[cutoff:]

    ts_train = pd.Series(train["tmean"].values, index=pd.DatetimeIndex(train["date"]))
    stl = STL(ts_train, period=365, robust=True, seasonal=25).fit()
    resid = stl.resid.values

    phi, sigma_w = fit_ar1(resid)

    # Extend trend with its recent slope; extend seasonal by repeating the fitted annual cycle
    recent_slope = np.mean(np.diff(stl.trend.values[-365:]))
    trend_fc = stl.trend.values[-1] + recent_slope * np.arange(1, horizon + 1)
    seasonal_fc = stl.seasonal.values[-365:][:horizon]  # same calendar days, one year later

    # --- Method A ---
    center_a, half_width_a = analytic_interval(resid[-1], phi, sigma_w, horizon)
    point_fc = trend_fc + seasonal_fc + center_a
    lo_a, hi_a = point_fc - half_width_a, point_fc + half_width_a

    # --- Method B ---
    paths = block_bootstrap_paths(resid, horizon, n_boot, block_len, rng)
    boot_fc = trend_fc + seasonal_fc + paths  # broadcast over n_boot
    lo_b = np.percentile(boot_fc, 5, axis=0)
    hi_b = np.percentile(boot_fc, 95, axis=0)

    actual = test["tmean"].values
    cover_a = np.mean((actual >= lo_a) & (actual <= hi_a))
    cover_b = np.mean((actual >= lo_b) & (actual <= hi_b))
    width_a = np.mean(hi_a - lo_a)
    width_b = np.mean(hi_b - lo_b)

    return {
        "dates": test["date"].values,
        "actual": actual,
        "point_fc": point_fc,
        "lo_a": lo_a, "hi_a": hi_a, "cover_a": cover_a, "width_a": width_a,
        "lo_b": lo_b, "hi_b": hi_b, "cover_b": cover_b, "width_b": width_b,
        "phi": phi, "sigma_w": sigma_w,
    }


def plot_backtest(res, out_path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.fill_between(res["dates"], res["lo_b"], res["hi_b"], color="#3498db", alpha=0.25,
                     label=f"Bootstrap 90% interval (coverage={res['cover_b']*100:.0f}%)")
    ax.plot(res["dates"], res["lo_a"], color="#e67e22", linestyle="--", linewidth=1,
            label=f"AR(1) analytic 90% interval (coverage={res['cover_a']*100:.0f}%)")
    ax.plot(res["dates"], res["hi_a"], color="#e67e22", linestyle="--", linewidth=1)
    ax.plot(res["dates"], res["point_fc"], color="#2c3e50", linewidth=1.3, label="Point forecast")
    ax.scatter(res["dates"], res["actual"], color="black", s=6, alpha=0.6, label="Actual (held out)")
    ax.set_title("90-Day Held-Out Forecast: Two Uncertainty Quantification Methods", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Temp (C)")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv("weather_data.csv", parse_dates=["date"])
    res = backtest(df, horizon=90)

    print(f"AR(1) fit: phi={res['phi']:.3f}, innovation std={res['sigma_w']:.2f} C")
    print()
    print("Backtest over 90 held-out days (nominal 90% interval):")
    print(f"  Method A (AR(1) analytic):    empirical coverage = {res['cover_a']*100:5.1f}%   avg width = {res['width_a']:.2f} C")
    print(f"  Method B (block bootstrap):   empirical coverage = {res['cover_b']*100:5.1f}%   avg width = {res['width_b']:.2f} C")

    plot_backtest(res, "backtest_uncertainty.png")
