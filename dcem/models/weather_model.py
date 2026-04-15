import pandas as pd
from datetime import datetime

from dcem.utilities import (
    geocode_city,
    fetch_historical_year,
    fetch_current_year_hybrid,
    build_climatological_normals,
    generate_predicted_year,
    add_derived_columns)

from dcem.globals import CURRENT_YEAR, DEFAULT_VARIABLES


class WeatherModel:
    def __init__(self, city, year):
        """
        Main entry point.  Fetches or predicts hourly weather data for a given
        city and year.

        Constructs
        ----------
        city : str
            City name (e.g. "Tokyo", "New York", "Paris, France").
        year : int
            Calendar year (1940-future).
        """
        if variables is None:
            variables = DEFAULT_VARIABLES

        # ── Step 1: Geocode ──────────────────────────────────────────────────────
        self.city_info = geocode_city(city)
        lat = self.city_info["latitude"]
        lon = self.city_info["longitude"]
        tz  = self.city_info["timezone"]

        self.year = year
        self.mode = "None"
        self.df   = None

        # ── Step 2: Determine mode and fetch data ────────────────────────────────
        if year < 1940:
            raise ValueError("ERA5 data is only available from 1940 onwards.")

        if year < CURRENT_YEAR:
            # ── Historical mode ──────────────────────────────────────────────────
            print(f"\n[2/4] Fetching historical data for {year} ...")
            df = fetch_historical_year(lat, lon, year, variables, tz)

        elif year == CURRENT_YEAR:
            # ── Hybrid mode ──────────────────────────────────────────────────────
            df, self.mode = fetch_current_year_hybrid(lat, lon, year, variables, tz)

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


        # ── Step 3: Post-process ─────────────────────────────────────────────────
        self.df = add_derived_columns(df)
    
    def save(self, outformat, output_dir):
        #save_output(self.df, self.city_info, self.year, self.mode, outformat, output_dir)
        base = f"{output_dir}/weather_{self.city_info['name'].replace(' ', '_')}_{self.year}"

        # Add metadata row at top for CSV/Excel
        meta = {
            "city": self.city_info["name"],
            "country": self.city_info["country"],
            "latitude": self.city_info["latitude"],
            "longitude": self.city_info["longitude"],
            "timezone": self.city_info["timezone"],
            "year": self.year,
            "mode": self.mode,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source": "Open-Meteo (ERA5 / ECMWF IFS reanalysis)",
        }

        df_out = self.df.reset_index()
        df_out.rename(columns={"index": "time"}, inplace=True)

        if outformat == "csv":
            path = base + ".csv"
            # Write metadata as header comments
            with open(path, "w") as f:
                for k, v in meta.items():
                    f.write(f"# {k}: {v}\n")
            df_out.to_csv(path, mode="a", index=False)

        elif outformat == "parquet":
            path = base + ".parquet"
            df_out.to_parquet(path, index=False)

        elif outformat == "excel":
            path = base + ".xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                # Metadata sheet
                meta_df = pd.DataFrame(
                    list(meta.items()), columns=["Key", "Value"]
                )
                meta_df.to_excel(writer, sheet_name="Metadata", index=False)
                df_out.to_excel(writer, sheet_name="Hourly Data", index=False)

        else:
            raise ValueError(f"Unsupported output format: {outformat}")

        print(f"\n[4/4] Saved to: {path}")
        return path