#!/usr/bin/env python3
"""
weather_fetcher.py
==================
Retrieve open-access hourly weather data for any city and year using the
Open-Meteo API (backed by ERA5 / ECMWF reanalysis data).

Usage
-----
    python3 weather_fetcher.py --city "Tokyo" --year 2022
    python3 weather_fetcher.py --city "New York" --year 2025
    python3 weather_fetcher.py --city "London" --year 2030   # future → statistical prediction
    python3 weather_fetcher.py --city "Paris" --year 2023 --output csv
    python3 weather_fetcher.py --city "Berlin" --year 2021 --output parquet
    python3 weather_fetcher.py --city "Sydney" --year 2020 --variables temperature_2m,precipitation,wind_speed_10m

Modes
-----
- **Historical** (year ≤ current year − 1):
    Fetches real ERA5/IFS reanalysis data from the Open-Meteo Historical Weather API.

- **Current year** (year == current year):
    Fetches available actual data up to yesterday, then fills the remaining
    days of the year with climatological averages derived from the previous
    10 years of the same calendar dates.

- **Future year** (year > current year):
    Generates a statistical prediction by computing hourly climatological
    normals (mean ± std) from the previous 10 years of ERA5 data for each
    hour-of-year, then samples from a Gaussian distribution around those
    normals to produce a plausible synthetic year.

Output
------
Saves a CSV / Parquet / Excel file and prints a summary table to stdout.
"""

import argparse
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CURRENT_YEAR = date.today().year
CURRENT_DATE = date.today()

# Default set of hourly variables to retrieve
DEFAULT_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "surface_pressure",
    "cloud_cover",
    "weather_code",
    "shortwave_radiation",
]

# WMO weather code descriptions
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ slight hail",
    99: "Thunderstorm w/ heavy hail",
}

# ─────────────────────────────────────────────────────────────────────────────
# Geocoding
# ─────────────────────────────────────────────────────────────────────────────

def geocode_city(city_name: str) -> dict:
    """
    Resolve a city name to geographic coordinates using the Open-Meteo
    Geocoding API (powered by GeoNames).

    Returns a dict with keys: name, country, latitude, longitude, timezone.
    Raises ValueError if the city cannot be found.
    """
    print(f"\n[1/4] Geocoding '{city_name}' ...")
    params = {"name": city_name, "count": 5, "language": "en", "format": "json"}
    resp = requests.get(GEOCODING_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("results"):
        raise ValueError(
            f"City '{city_name}' not found. "
            "Try a more specific name, e.g. 'Paris, France'."
        )

    result = data["results"][0]
    info = {
        "name": result["name"],
        "country": result.get("country", ""),
        "admin1": result.get("admin1", ""),
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "timezone": result.get("timezone", "UTC"),
        "elevation": result.get("elevation", 0),
        "population": result.get("population", 0),
    }

    print(
        f"    Found: {info['name']}, {info['admin1']}, {info['country']} "
        f"(lat={info['latitude']:.4f}, lon={info['longitude']:.4f}, "
        f"tz={info['timezone']})"
    )
    return info


# ─────────────────────────────────────────────────────────────────────────────
# API fetching helpers
# ─────────────────────────────────────────────────────────────────────────────

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
            resp = requests.get(HISTORICAL_URL, params=params, timeout=60)
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


# ─────────────────────────────────────────────────────────────────────────────
# Climatological normals (for prediction)
# ─────────────────────────────────────────────────────────────────────────────

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
        reference_end_year = CURRENT_YEAR - 1

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


# ─────────────────────────────────────────────────────────────────────────────
# Prediction / synthetic year generation
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Current-year hybrid fetch
# ─────────────────────────────────────────────────────────────────────────────

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
    yesterday = CURRENT_DATE - timedelta(days=1)
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


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def print_summary(df: pd.DataFrame, city_info: dict, year: int, mode: str) -> None:
    """Print a concise summary of the retrieved dataset."""
    print("\n" + "=" * 65)
    print(f"  WEATHER DATA SUMMARY")
    print("=" * 65)
    print(f"  City       : {city_info['name']}, {city_info['country']}")
    print(f"  Year       : {year}")
    print(f"  Mode       : {mode}")
    print(f"  Records    : {len(df):,} hourly observations")
    print(f"  Date range : {df.index.min()} → {df.index.max()}")
    print(f"  Timezone   : {city_info['timezone']}")
    print("-" * 65)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats_vars = [
        v for v in [
            "temperature_2m", "apparent_temperature",
            "relative_humidity_2m", "precipitation",
            "wind_speed_10m", "cloud_cover", "surface_pressure",
        ]
        if v in numeric_cols
    ]

    if stats_vars:
        stats = df[stats_vars].describe().loc[["mean", "min", "max", "std"]]
        print(stats.round(2).to_string())

    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def save_output(
    df: pd.DataFrame,
    city_info: dict,
    year: int,
    mode: str,
    output_format: str,
    output_dir: str = ".",
) -> str:
    """Save the DataFrame to the requested format and return the file path."""
    city_slug = (
        city_info["name"]
        .lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("'", "")
    )
    base = f"{output_dir}/weather_{city_slug}_{year}"

    # Add metadata row at top for CSV/Excel
    meta = {
        "city": city_info["name"],
        "country": city_info["country"],
        "latitude": city_info["latitude"],
        "longitude": city_info["longitude"],
        "timezone": city_info["timezone"],
        "year": year,
        "mode": mode,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "Open-Meteo (ERA5 / ECMWF IFS reanalysis)",
    }

    df_out = df.reset_index()
    df_out.rename(columns={"index": "time"}, inplace=True)

    if output_format == "csv":
        path = base + ".csv"
        # Write metadata as header comments
        with open(path, "w") as f:
            for k, v in meta.items():
                f.write(f"# {k}: {v}\n")
        df_out.to_csv(path, mode="a", index=False)

    elif output_format == "parquet":
        path = base + ".parquet"
        df_out.to_parquet(path, index=False)

    elif output_format == "excel":
        path = base + ".xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            # Metadata sheet
            meta_df = pd.DataFrame(
                list(meta.items()), columns=["Key", "Value"]
            )
            meta_df.to_excel(writer, sheet_name="Metadata", index=False)
            df_out.to_excel(writer, sheet_name="Hourly Data", index=False)

    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"\n[4/4] Saved to: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────────────────

def get_weather_data(
    city: str,
    year: int,
    variables: Optional[list[str]] = None,
    output_format: str = "csv",
    output_dir: str = ".",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Main entry point.  Fetches or predicts hourly weather data for a given
    city and year.

    Parameters
    ----------
    city : str
        City name (e.g. "Tokyo", "New York", "Paris, France").
    year : int
        Calendar year (1940-future).
    variables : list[str], optional
        List of Open-Meteo hourly variable names to retrieve.
        Defaults to DEFAULT_VARIABLES.
    output_format : str
        One of "csv", "parquet", "excel".
    output_dir : str
        Directory where the output file will be saved.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        Hourly weather data indexed by datetime.
    """
    if variables is None:
        variables = DEFAULT_VARIABLES

    # ── Step 1: Geocode ──────────────────────────────────────────────────────
    city_info = geocode_city(city)
    lat = city_info["latitude"]
    lon = city_info["longitude"]
    tz = city_info["timezone"]

    # ── Step 2: Determine mode and fetch data ────────────────────────────────
    if year < 1940:
        raise ValueError("ERA5 data is only available from 1940 onwards.")

    if year < CURRENT_YEAR:
        # ── Historical mode ──────────────────────────────────────────────────
        print(f"\n[2/4] Fetching historical data for {year} ...")
        df = fetch_historical_year(lat, lon, year, variables, tz)
        mode = "Historical (ERA5 / ECMWF IFS reanalysis)"

    elif year == CURRENT_YEAR:
        # ── Hybrid mode ──────────────────────────────────────────────────────
        df, mode = fetch_current_year_hybrid(lat, lon, year, variables, tz)

    else:
        # ── Prediction mode ──────────────────────────────────────────────────
        print(
            f"\n[2/4] Year {year} is in the future. "
            "Generating statistical prediction from climatological normals ..."
        )
        normals = build_climatological_normals(
            lat, lon, variables, tz, n_years=10,
            reference_end_year=CURRENT_YEAR - 1
        )
        print(f"\n[3/4] Sampling synthetic year {year} from normals ...")
        df = generate_predicted_year(normals, year, variables)
        mode = (
            f"Predicted (climatological normals from "
            f"{CURRENT_YEAR - 10}-{CURRENT_YEAR - 1}, Gaussian sampling)"
        )

    # ── Step 3: Post-process ─────────────────────────────────────────────────
    df = add_derived_columns(df)

    if verbose:
        print_summary(df, city_info, year, mode)

    # ── Step 4: Save ─────────────────────────────────────────────────────────
    save_output(df, city_info, year, mode, output_format, output_dir)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch open-access hourly weather data for a city and year.\n"
            "Uses Open-Meteo (ERA5/ECMWF) for historical data and\n"
            "climatological Gaussian sampling for future predictions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--city", "-c",
        required=True,
        help='City name, e.g. "Tokyo" or "Paris, France"',
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        required=True,
        help="Calendar year (1940-future). Past → actual data; future → prediction.",
    )
    parser.add_argument(
        "--variables", "-v",
        default=None,
        help=(
            "Comma-separated list of Open-Meteo hourly variable names. "
            "Defaults to a standard set of 13 variables."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        choices=["csv", "parquet", "excel"],
        default="csv",
        help="Output file format (default: csv).",
    )
    parser.add_argument(
        "--output-dir", "-d",
        default=".",
        help="Directory to save the output file (default: current directory).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    variables = None
    if args.variables:
        variables = [v.strip() for v in args.variables.split(",") if v.strip()]

    try:
        get_weather_data(
            city=args.city,
            year=args.year,
            variables=variables,
            output_format=args.output,
            output_dir=args.output_dir,
        )
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
