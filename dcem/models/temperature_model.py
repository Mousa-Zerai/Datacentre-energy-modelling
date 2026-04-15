import pandas as pd

from datetime import date, timedelta

from dcem.globals import _CURRENT_YEAR, _CURRENT_DATE
from dcem.utilities import geocode_city, fetch_chunk, fetch_year, build_normals, sample_synthetic_year

class TemperatureModel:
    def __init__(self, city, year):
        """
        Main entry point: fetch or predict hourly outdoor temperature for a city/year.

        Constructs
        -------
        series   : pd.Series  - hourly temperature indexed by datetime
        mode     : str        - human-readable description of data source
        city_info: dict       - geocoding metadata
        """
        self.city_info = geocode_city(city)
        lat, lon, tz = self.city_info["latitude"], self.city_info["longitude"], self.city_info["timezone"]

        if year < 1940:
            raise ValueError("ERA5 data is only available from 1940 onwards.")

        if year < _CURRENT_YEAR:
            print(f"[Weather] Fetching historical ERA5 temperature for {year} ...")
            self.series = fetch_year(lat, lon, year, tz)
            self.mode   = f"Historical ERA5 (Open-Meteo) — {year}"

        elif year == _CURRENT_YEAR:
            yesterday    = _CURRENT_DATE - timedelta(days=1)
            actual_start = f"{year}-01-01"
            actual_end   = yesterday.strftime("%Y-%m-%d")
            print(f"[Weather] Fetching actual data {actual_start} → {actual_end} ...")
            actual = fetch_chunk(lat, lon, actual_start, actual_end, tz)

            normals = build_normals(lat, lon, tz, n_years=10, ref_end=year - 1)
            full_pred = sample_synthetic_year(normals, year)
            predicted = full_pred[full_pred.index.date >= _CURRENT_DATE]

            self.series = pd.concat([actual, predicted]).sort_index()
            self.mode   = (
                f"Hybrid: ERA5 Jan 1-{yesterday.strftime('%b %d')}, "
                f"predicted {_CURRENT_DATE.strftime('%b %d')}-Dec 31"
            )

        else:
            print(f"[Weather] Year {year} is in the future — generating prediction ...")
            normals     = build_normals(lat, lon, tz, n_years=10, ref_end=_CURRENT_YEAR - 1)
            self.series = sample_synthetic_year(normals, year)
            self.mode   = (
                f"Predicted (climatological normals "
                f"{_CURRENT_YEAR - 10}-{_CURRENT_YEAR - 1}, Gaussian sampling)"
            )

        print(f"[Weather] Temperature data ready: {len(self.series):,} hourly records. Mode: {self.mode}")
