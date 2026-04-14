# Datacentre Energy Modelling

Physics-based data centre energy consumption and waste heat simulation for district heating integration studies.

## What It Does

Simulates hourly or half-hourly power consumption and recoverable waste heat for data centres. Built for Virtual Heat Plant and demand-side management research.

## Features

- Physics-based IT load modelling (daily/weekly patterns)
- Real weather data via Open-Meteo / ERA5 reanalysis (no API key needed)
- Temperature-dependent cooling calculations
- Dynamic PUE (Power Usage Effectiveness) analysis
- Waste heat quantification for district heating
- Outputs to CSV, Excel, and PNG plots

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Edit the CONFIG block at the top of `DC_energy_modelling_v2.py`:

```python
CONFIG = {
    "city":             "Edinburgh",
    "year":             2024,
    "timestep_hours":   1,        # 1 = hourly, 0.5 = half-hourly
    "it_capacity_kw":   500,
    "pue":              1.4,
    "base_utilization": 0.65,
    "output_csv":       True,
    "output_excel":     True,
    "output_plots":     True,
    "output_dir":       ".",
}
```

Then run:

```bash
python DC_energy_modelling_v2.py
```

## Requirements

- Python 3.8+
- numpy, pandas, matplotlib, requests, openpyxl

## License

MIT License — see LICENSE file

## Author

Mousa Zerai  
University of Edinburgh
