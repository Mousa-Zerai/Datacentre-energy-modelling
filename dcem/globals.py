
from datetime import date

# ── Open-Meteo endpoints ──────────────────────────────────────────────────────
_GEOCODING_URL   = "https://geocoding-api.open-meteo.com/v1/search"
_HISTORICAL_URL  = "https://archive-api.open-meteo.com/v1/archive"
_WEATHER_VARS    = ["temperature_2m"]      # only temperature needed
_CURRENT_YEAR    = date.today().year
_CURRENT_DATE    = date.today()


GEOCODING_URL  = "https://geocoding-api.open-meteo.com/v1/search"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL   = "https://api.open-meteo.com/v1/forecast"

CURRENT_YEAR = date.today().year
CURRENT_DATE = date.today()

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
