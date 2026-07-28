import gzip
import io
import requests
import pandas as pd
 
STATION_ID = "USW00013893"  #Memphis Intl Airport, TN
START_YEAR = 2014
END_YEAR = 2023
URL = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{STATION_ID}.csv.gz"
 
 
def fetch_station_data(station_id: str = STATION_ID) -> pd.DataFrame:
    print(f"Downloading {URL} ...")
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
 
    with gzip.open(io.BytesIO(resp.content)) as f:
        raw = pd.read_csv(
            f, header=None,
            names=["id", "date", "element", "value", "mflag", "qflag", "sflag", "obstime"],
            dtype={"date": str},
        )
 
    raw["date"] = pd.to_datetime(raw["date"], format="%Y%m%d")
    raw = raw[(raw["date"].dt.year >= START_YEAR) & (raw["date"].dt.year <= END_YEAR)]
 
    raw = raw[raw["qflag"].isna()]
 
    pivot = raw.pivot_table(index="date", columns="element", values="value", aggfunc="first")
 
    df = pd.DataFrame({
        "date": pivot.index,
        "tmax": pivot.get("TMAX") / 10.0,
        "tmin": pivot.get("TMIN") / 10.0,
        "precip": pivot.get("PRCP") / 10.0, 
    }).reset_index(drop=True)
 
    df["tmean"] = (df["tmax"] + df["tmin"]) / 2
    df = df.dropna(subset=["tmax", "tmin"]).sort_values("date").reset_index(drop=True)
 
    print(f"Got {len(df)} days, {df['date'].min().date()} to {df['date'].max().date()}")
    return df
 
 
if __name__ == "__main__":
    df = fetch_station_data()
    import os
    df.to_csv("weather_data.csv", index=False)
    print("Saved to weather_data.csv")
