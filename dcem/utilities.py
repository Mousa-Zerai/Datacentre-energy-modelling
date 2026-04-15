import argparse
from typing import Optional
import requests
import time
import pandas as pd
import numpy as np
from datetime import date, timedelta

from globals import _CURRENT_DATE, _GEOCODING_URL, _HISTORICAL_URL, _CURRENT_YEAR, CURRENT_DATE, HISTORICAL_URL, WMO_CODES

def construct_argument_parser():
    parser = argparse.ArgumentParser(description="DCEM: Data Centre Energy Modeling")
    parser.add_argument("--city", type=str, required=True, help="City for which to run the simulation.")
    parser.add_argument("--year", type=int, required=True, help="Year for which to run the simulation.")
    return parser

def fetch_chunk(lat, lon, start, end, tz, retries=3) -> pd.Series:
    """Fetch temperature_2m for a date range; return a Series indexed by datetime."""
    params = {
        "latitude":          lat,
        "longitude":         lon,
        "start_date":        start,
        "end_date":          end,
        "hourly":            "temperature_2m",
        "timezone":          tz,
        "temperature_unit":  "celsius",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(_HISTORICAL_URL, params=params, timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"         Retry {attempt}/{retries} after {wait}s ({exc})")
            time.sleep(wait)

    raw = resp.json()
    if "error" in raw:
        raise RuntimeError(f"API error: {raw.get('reason', raw)}")

    hourly = raw.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.Series(dtype=float)

    series = pd.Series(
        hourly["temperature_2m"],
        index=pd.to_datetime(hourly["time"]),
        name="temperature_2m",
    )
    return series

def fetch_year(lat, lon, year, tz) -> pd.Series:
    """Fetch a full calendar year of hourly temperature in two 6-month chunks."""
    chunks = [
        (f"{year}-01-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-12-31"),
    ]
    parts = []
    for start, end in chunks:
        s = fetch_chunk(lat, lon, start, end, tz)
        if not s.empty:
            parts.append(s)
        time.sleep(0.3)
    if not parts:
        raise RuntimeError(f"No temperature data returned for {year}.")
    return pd.concat(parts).sort_index()

def geocode_city(city_name: str) -> dict:
    """Resolve a city name to lat/lon/timezone via Open-Meteo Geocoding API."""
    print(f"\n[Weather] Geocoding '{city_name}' ...")
    resp = requests.get(
        _GEOCODING_URL,
        params={"name": city_name, "count": 5, "language": "en", "format": "json"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("results"):
        raise ValueError(
            f"City '{city_name}' not found. "
            "Try a more specific name, e.g. 'Paris, France'."
        )
    r = data["results"][0]
    info = {
        "name":      r["name"],
        "country":   r.get("country", ""),
        "latitude":  r["latitude"],
        "longitude": r["longitude"],
        "timezone":  r.get("timezone", "UTC"),
        "elevation": r.get("elevation", 0),
    }
    print(
        f"         Found: {info['name']}, {info['country']} "
        f"(lat={info['latitude']:.4f}, lon={info['longitude']:.4f})"
    )
    return info

def build_normals(lat, lon, tz, n_years=10, ref_end=None) -> pd.DataFrame:
    """Compute hourly climatological normals (mean ± std) over n_years."""
    if ref_end is None:
        ref_end = _CURRENT_YEAR - 1
    ref_start = ref_end - n_years + 1
    print(f"[Weather] Building climatological normals ({ref_start}-{ref_end}) ...")

    frames = []
    for yr in range(ref_start, ref_end + 1):
        print(f"          Fetching reference year {yr} ...", end="", flush=True)
        try:
            s = fetch_year(lat, lon, yr, tz)
            df = s.to_frame()
            df["month"] = df.index.month
            df["day"]   = df.index.day
            df["hour"]  = df.index.hour
            frames.append(df)
            print(" ok")
        except Exception as exc:
            print(f" skipped ({exc})")

    if not frames:
        raise RuntimeError("Could not fetch any reference years for normals.")

    combined = pd.concat(frames)
    normals = (
        combined.groupby(["month", "day", "hour"])["temperature_2m"]
        .agg(["mean", "std"])
        .reset_index()
        .fillna(0)
    )
    return normals

def sample_synthetic_year(normals: pd.DataFrame, year: int, seed=42) -> pd.Series:
    """Sample a synthetic temperature year from climatological normals."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00:00", freq="h")
    df = pd.DataFrame({"month": idx.month, "day": idx.day, "hour": idx.hour}, index=idx)
    df = df.merge(normals, on=["month", "day", "hour"], how="left")
    df.index = idx
    noise  = rng.normal(0, df["std"].fillna(0).values)
    temps  = df["mean"].values + noise
    return pd.Series(temps, index=idx, name="temperature_2m")

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable and derived columns to the DataFrame."""
    df = df.copy()

    # Weather code description
    if "weather_code" in df.columns:
        df["weather_description"] = (
            df["weather_code"]
            .round()
            .astype("Int64")
            .map(WMO_CODES)
            .fillna("Unknown")
        )

    # Heat index / feels-like already in apparent_temperature
    # Add season
    if df.index.dtype == "datetime64[ns]" or hasattr(df.index, "month"):
        month = df.index.month
        df["season"] = pd.cut(
            month,
            bins=[0, 2, 5, 8, 11, 12],
            labels=["Winter", "Spring", "Summer", "Autumn", "Winter"],
            ordered=False,
        )

    return df


def _fetch_historical_chunk(
    lat: float,
    lon: float,
    start: str,
    end: str,
    variables: list[str],
    timezone: str,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch a single date-range chunk from the Open-Meteo Historical Weather API.
    Returns a DataFrame indexed by datetime.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": timezone,
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(_HISTORICAL_URL, params=params, timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"    Retry {attempt}/{retries} after {wait}s ({exc})")
            time.sleep(wait)

    raw = resp.json()
    if "error" in raw:
        raise RuntimeError(f"API error: {raw.get('reason', raw)}")

    hourly = raw.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df

def fetch_historical_year(
    lat: float,
    lon: float,
    year: int,
    variables: list[str],
    timezone: str,
) -> pd.DataFrame:
    """
    Fetch a full calendar year of hourly historical data.
    Splits into two 6-month chunks to stay within API limits.
    """
    chunks = [
        (f"{year}-01-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-12-31"),
    ]
    frames = []
    for start, end in chunks:
        df = _fetch_historical_chunk(lat, lon, start, end, variables, timezone)
        if not df.empty:
            frames.append(df)
        time.sleep(0.3)  # polite rate limiting

    if not frames:
        raise RuntimeError(f"No data returned for year {year}.")
    return pd.concat(frames).sort_index()

def build_climatological_normals(
    lat: float,
    lon: float,
    variables: list[str],
    timezone: str,
    n_years: int = 10,
    reference_end_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute hourly climatological normals (mean and std) over the past
    `n_years` years.  Returns a DataFrame indexed by (month, day, hour)
    with columns like `temperature_2m_mean`, `temperature_2m_std`, etc.
    """
    if reference_end_year is None:
        reference_end_year = _CURRENT_YEAR - 1

    reference_start_year = reference_end_year - n_years + 1
    print(
        f"\n[2/4] Building climatological normals "
        f"({reference_start_year}-{reference_end_year}, {n_years} years) ..."
    )

    all_frames = []
    for yr in range(reference_start_year, reference_end_year + 1):
        print(f"    Fetching reference year {yr} ...", end="", flush=True)
        try:
            df = fetch_historical_year(lat, lon, yr, variables, timezone)
            df["year"] = df.index.year
            df["month"] = df.index.month
            df["day"] = df.index.day
            df["hour"] = df.index.hour
            all_frames.append(df)
            print(" done")
        except Exception as exc:
            print(f" skipped ({exc})")

    if not all_frames:
        raise RuntimeError("Could not fetch any reference years for normals.")

    combined = pd.concat(all_frames)

    # Group by (month, day, hour) and compute mean + std
    group_cols = ["month", "day", "hour"]
    agg = {}
    for v in variables:
        if v in combined.columns:
            agg[f"{v}_mean"] = (v, "mean")
            agg[f"{v}_std"] = (v, "std")

    normals = combined.groupby(group_cols).agg(**agg).reset_index()
    # Fill NaN std with 0 (e.g. for Feb 29 with only 1 sample)
    normals = normals.fillna(0)
    return normals

def generate_predicted_year(
    normals: pd.DataFrame,
    year: int,
    variables: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic hourly year by sampling from Gaussian distributions
    centred on the climatological normals for each hour-of-year.

    For non-negative variables (precipitation, radiation, etc.) the result
    is clipped to zero.
    """
    rng = np.random.default_rng(seed)

    # Build a datetime index for the target year (handles leap years)
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31 23:00:00")
    idx = pd.date_range(start, end, freq="h")

    df = pd.DataFrame(index=idx)
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["hour"] = df.index.hour

    # Merge normals
    df = df.merge(normals, on=["month", "day", "hour"], how="left")
    df.index = idx

    non_negative = {
        "precipitation", "rain", "snowfall", "snow_depth",
        "shortwave_radiation", "direct_radiation", "diffuse_radiation",
        "et0_fao_evapotranspiration", "vapour_pressure_deficit",
        "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm",
    }
    # Variables with hard physical bounds [0, 100]
    percent_bounded = {
        "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
        "relative_humidity_2m",
    }

    result_cols = {}
    for v in variables:
        mean_col = f"{v}_mean"
        std_col = f"{v}_std"
        if mean_col not in df.columns:
            continue

        means = df[mean_col].values
        stds = df[std_col].fillna(0).values

        if v == "weather_code":
            # Use the modal weather code (round mean to nearest valid code)
            sampled = np.round(means).astype(int)
        elif v == "wind_direction_10m" or v == "wind_direction_100m":
            # Circular variable: sample uniformly within ±std degrees
            noise = rng.uniform(-stds, stds)
            sampled = (means + noise) % 360
        else:
            noise = rng.normal(0, stds)
            sampled = means + noise
            if v in percent_bounded:
                sampled = np.clip(sampled, 0, 100)
            elif v in non_negative:
                sampled = np.clip(sampled, 0, None)

        result_cols[v] = sampled

    out = pd.DataFrame(result_cols, index=idx)
    out.index.name = "time"
    return out

def fetch_current_year_hybrid(
    lat: float,
    lon: float,
    year: int,
    variables: list[str],
    timezone: str,
) -> tuple[pd.DataFrame, str]:
    """
    For the current calendar year:
    - Fetch actual data from Jan 1 up to yesterday.
    - Fill the rest of the year with climatological normals (mean values).
    Returns (DataFrame, mode_description).
    """
    yesterday = _CURRENT_DATE - timedelta(days=1)
    actual_end = yesterday.strftime("%Y-%m-%d")
    actual_start = f"{year}-01-01"

    print(f"\n[2/4] Fetching actual data {actual_start} → {actual_end} ...")
    actual_df = _fetch_historical_chunk(
        lat, lon, actual_start, actual_end, variables, timezone
    )

    # Build normals for the remainder of the year
    normals = build_climatological_normals(
        lat, lon, variables, timezone, n_years=10,
        reference_end_year=year - 1
    )

    # Generate synthetic data for the remaining days
    remaining_start = CURRENT_DATE
    remaining_end = date(year, 12, 31)

    if remaining_start <= remaining_end:
        print(
            f"\n[3/4] Generating predicted data "
            f"{remaining_start} → {remaining_end} ..."
        )
        full_predicted = generate_predicted_year(normals, year, variables)
        # Slice only the remaining portion
        predicted_slice = full_predicted[
            full_predicted.index.date >= remaining_start
        ]
        combined = pd.concat([actual_df, predicted_slice]).sort_index()
        mode = (
            f"Hybrid: actual data Jan 1-{yesterday.strftime('%b %d')}, "
            f"predicted {remaining_start.strftime('%b %d')}-Dec 31"
        )
    else:
        combined = actual_df
        mode = "Actual (full year already elapsed)"

    return combined, mode
