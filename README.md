# Wildfire Analysis Tools

This repository contains a collection of Python scripts for analyzing wildfire risk factors, including weather data, vegetation indices, and burned areas in California.

## Scripts Overview

### 1. NOAA Weather Data Fetcher (`noaa.py`)
Fetches and processes weather data from NOAA stations, including:
- Maximum and minimum temperature
- Precipitation
- Wind speed and direction
- Estimated relative humidity
- Vapor Pressure Deficit (VPD)

### 2. Vegetation Analysis (`vegetation.py`)
Analyzes vegetation health using Sentinel-2 satellite imagery:
- Calculates NDVI (Normalized Difference Vegetation Index)
- Masks water bodies using NDWI
- Processes and aggregates data by coordinates
- Exports results as CSV and GeoTIFF

### 3. Enhanced Vegetation Analysis (`veg2.py`)
Extended version of vegetation analysis with additional features:
- Multi-temporal vegetation analysis
- Advanced water body masking
- Improved data aggregation
- Enhanced error handling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/peteteaa/SizzlerRizzler.git
cd SizzlerRizzler
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dependencies

Create a `requirements.txt` file with the following dependencies:
```
earthengine-api>=0.1.355
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
tqdm>=4.65.0
google-cloud-storage>=2.10.0
```

## Authentication Setup

### NOAA API
1. Get a NOAA API token from: https://www.ncdc.noaa.gov/cdo-web/token
2. Set your token as an environment variable:
```bash
export NOAA_TOKEN='your-token-here'
```

### Google Earth Engine
1. Sign up for Google Earth Engine: https://earthengine.google.com/signup/
2. Install the Earth Engine CLI:
```bash
pip install earthengine-api --upgrade
```
3. Authenticate:
```bash
earthengine authenticate
```

## Usage

### NOAA Weather Data
```python
from noaa import NOAADataFetcher

# Initialize fetcher with your NOAA API token
fetcher = NOAADataFetcher(token='your-token-here')

# Fetch and process data
fetcher.fetch_and_process()
```

### Vegetation Analysis
```python
from vegetation import process_ndvi_data

# Process vegetation data
process_ndvi_data('2023-01-01', '2023-12-31')
```

### Enhanced Vegetation Analysis
```python
from veg2 import VegetationAnalyzer

# Initialize analyzer
analyzer = VegetationAnalyzer()

# Run analysis
analyzer.process_vegetation_data('2023-01-01', '2023-12-31')
```

## Output Data

All scripts save their output in organized directories:
- Weather data: `data/weather/`
- Vegetation data: `data/vegetation/`
- Processed results include both CSV files and metadata JSON files

## Data Structure

### NOAA Weather Data
```csv
date,latitude,longitude,TMAX,TMIN,PRCP,AWND,WDF2,RH,VPD
2023-01-01,37.7749,-122.4194,15.6,8.3,0.0,3.1,270,65.2,0.82
```

### Vegetation Data
```csv
latitude,longitude,ndvi,pixel_count
37.7749,-122.4194,0.65,3
```

## Error Handling

All scripts include robust error handling and logging:
- Failed API requests are retried automatically
- Data validation checks are performed
- Detailed error messages are logged
- Missing data is handled gracefully

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
