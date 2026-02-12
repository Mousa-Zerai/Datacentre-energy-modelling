# Datacenter-energy-modelling

Physics-based data centre energy consumption and waste heat 
simulation for district heating integration studies.

## What It Does

Simulates hourly power consumption and recoverable waste heat 
for data centres across 24 global locations. Built for 
Virtual Heat Plant research.

## Features

- Physics-based IT load modelling (daily/weekly patterns)
- Temperature-dependent cooling calculations
- Dynamic PUE (Power Usage Effectiveness) analysis
- Waste heat quantification for district heating
- 24 global climate locations

## Installation

pip install -r requirements.txt

## Quick Start

from datacenter_model.model import DataCenterModel

dc = DataCenterModel(it_capacity_kw=500, pue=1.4, location='London')
df = dc.simulate(hours=8760)
dc.print_summary(df)

## Available Locations

Edinburgh, London, Philadelphia, Pennsylvania, Cambridge MA, 
New York, Phoenix, Singapore, Tokyo... and 15 more.

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib

## License

MIT License - see LICENSE file

## Author

Mousa Zerai 
University of Edinburgh
email@ed.ac.uk

## Citation

If you use this software in your research, please cite:
[JOSS paper link - to be added after publication]
```

---

### **Step 4: requirements.txt**

**Create this file with:**
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
