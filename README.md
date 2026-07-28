# Regional Weather Uncertainty Quantification

A time series analysis project that decomposes historical daily temperature and precipitation into interpretable components, then compares methods for quantifying forecast uncertainty and evaluates those methods through backtesting on held-out observations.

## Data

The dataset consists of **daily weather observations from 2014–2023** for **Memphis International Airport (NOAA station USW00013893)** obtained from the **NOAA Global Historical Climatology Network (GHCN-Daily)**.

The data are downloaded directly from NOAA and include:

- Daily maximum temperature (`TMAX`)
- Daily minimum temperature (`TMIN`)
- Daily precipitation (`PRCP`)

Daily mean temperature is computed as:

```text
Tmean = (TMAX + TMIN) / 2
```

Quality-controlled observations are retained by removing records with NOAA quality flags before analysis.

The download and preprocessing are handled by `fetch.py`, which converts NOAA's raw observations into a clean daily dataset (`weather_data.csv`) used by all subsequent analyses.

---

## 1. Decomposition (`decompose.py`)

STL (Seasonal-Trend decomposition using LOESS) separates daily mean temperature into trend, seasonal, and residual components. STL was chosen over classical decomposition because it allows the seasonal pattern to vary gradually over time while remaining robust to extreme weather events such as cold snaps and heat waves.

The decomposition provides:

- Long-term temperature trend
- Annual seasonal cycle
- Residual day-to-day variability

The residual component represents the portion of temperature that cannot be explained by seasonality or long-term trends and forms the basis for the uncertainty analysis.

![STL decomposition](stl_decomposition.png)

---

## 2. Forecast Uncertainty (`uncertainty.py`)

Two methods are compared for estimating uncertainty in daily mean temperature forecasts.

### Method A — AR(1) Analytic Intervals

The STL residual is modeled as

```text
eₜ = φeₜ₋₁ + wₜ
```

allowing prediction intervals to be computed analytically from the AR(1) forecast variance. This method is computationally efficient but assumes approximately Gaussian, stationary residuals.

### Method B — Moving Block Bootstrap

Instead of assuming a particular distribution, contiguous blocks of historical residuals are resampled to generate empirical forecast distributions while preserving short-term temporal dependence.

### Backtesting

The final 90 days of observations are held out as a test set. Both uncertainty methods are fit using only the earlier observations, then evaluated by comparing their nominal 90% prediction intervals against the actual held-out temperatures.

![Backtest](backtest_uncertainty.png)

| Method | Nominal | Empirical Coverage | Average Interval Width |
|---------|---------|-------------------:|-----------------------:|
| AR(1) Analytic | 90% | 68.9% | 12.9°C |
| Moving Block Bootstrap | 90% | 70.0% | 12.6°C |

Both methods underestimate uncertainty: although they produce nominal 90% prediction intervals, the true temperatures fall inside those intervals only about 70% of the time. This indicates that while residual uncertainty is modeled reasonably well, additional uncertainty arising from trend extrapolation and future seasonal variation is not fully captured.

---

## 3. Precipitation Uncertainty (`precip_uncertainty.py`)

Daily precipitation is highly skewed and contains many zero-rainfall days, making symmetric prediction intervals inappropriate.

Instead, the project estimates an empirical conditional distribution for each day of the year using observations within a ±10-day window across all historical years. From this distribution, it computes:

- Median precipitation
- 90th percentile precipitation
- Probability of a wet day

![Precipitation Climatology](precip_climatology.png)

A held-out validation compares observed precipitation against the estimated conditional distributions to assess calibration.

---

## Project Workflow

1. Download and clean historical NOAA weather observations.
2. Decompose daily temperature into trend, seasonal, and residual components using STL.
3. Quantify forecast uncertainty using two different statistical approaches.
4. Backtest both methods using held-out observations.
5. Model precipitation uncertainty using empirical conditional distributions.

---

## Files

| File | Description |
|------|-------------|
| `fetch.py` | Downloads and preprocesses NOAA GHCN-Daily weather observations |
| `weather_data.csv` | Cleaned historical weather dataset |
| `decompose.py` | STL decomposition and variance analysis |
| `uncertainty.py` | AR(1) and moving block bootstrap uncertainty estimation with backtesting |
| `precip_uncertainty.py` | Empirical precipitation uncertainty model |
| `residuals.csv` | STL residual series |
| `*.png` | Generated figures |

---

## Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib
- statsmodels
- requests
