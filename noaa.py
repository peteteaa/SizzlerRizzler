import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import time
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NOAADataFetcher:
    """
    Fetches weather data from NOAA API for San Francisco Bay Area,
    including temperature, precipitation, wind speed, wind direction, and relative humidity.
    """
    
    def __init__(self):
        """Initialize the NOAA data fetcher"""
        self.api_key = os.getenv('NOAA_API_KEY')
        if not self.api_key:
            raise ValueError("NOAA_API_KEY environment variable not set")
            
        self.base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
        self.headers = {'token': self.api_key}
        
        # Expanded Bay Area bounding box
        self.sf_bbox = {
            "minlat": 37.2,  # Expanded south to include more of the Bay Area
            "minlon": -122.8,  # Expanded west
            "maxlat": 38.2,  # Expanded north
            "maxlon": -122.0  # Expanded east
        }
        
        # Create data directory
        os.makedirs("data/weather", exist_ok=True)
        
    def _make_request(self, endpoint: str, params: Dict, max_retries: int = 3, retry_delay: int = 5) -> requests.Response:
        """Make a request to the NOAA API with retry logic"""
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params)
                if response.status_code == 200:
                    return response
                elif response.status_code == 503:
                    logger.warning(f"Service unavailable (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    response.raise_for_status()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        raise Exception(f"Failed after {max_retries} attempts")

    def _get_stations(self) -> List[Dict]:
        """Get list of NOAA stations in San Francisco Bay Area"""
        # Search for stations with temperature and dew point data
        params = {
            'extent': f'{self.sf_bbox["minlat"]},{self.sf_bbox["minlon"]},{self.sf_bbox["maxlat"]},{self.sf_bbox["maxlon"]}',
            'datasetid': 'GHCND',
            'datacategoryid': ['TEMP', 'WIND'],  # Temperature includes dew point
            'startdate': '2023-01-01',
            'enddate': '2023-12-31',
            'limit': 1000
        }
        
        response = self._make_request('stations', params)
        stations = response.json().get('results', [])
        logger.info(f"Found {len(stations)} stations in Bay Area with temperature and wind data")
        
        # Filter for stations that have recent data
        valid_stations = []
        for station in stations:
            params = {
                'datasetid': 'GHCND',
                'stationid': station['id'],
                'startdate': '2023-01-01',  # Check if station has recent data
                'enddate': '2023-12-31',
                'limit': 1
            }
            
            response = self._make_request('data', params)
            if response.status_code == 200 and response.json().get('results'):
                valid_stations.append(station)
            time.sleep(0.5)  # Rate limiting
            
        logger.info(f"Found {len(valid_stations)} stations with recent data")
        return valid_stations
    
    def _fetch_data(self, station_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data for a specific station and date range"""
        # Split date range into yearly chunks
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        all_data = []
        current_start = start
        
        while current_start < end:
            current_end = min(
                current_start + pd.DateOffset(years=1) - pd.DateOffset(days=1),
                end
            )
            
            # First try with all data types
            datatypes = ['TMAX', 'TMIN', 'PRCP', 'AWND', 'WDF2']  
            params = {
                'datasetid': 'GHCND',
                'stationid': station_id,
                'startdate': current_start.strftime('%Y-%m-%d'),
                'enddate': current_end.strftime('%Y-%m-%d'),
                'datatypeid': datatypes,
                'limit': 1000,
                'units': 'metric'
            }
            
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    response = self._make_request('data', params)
                    if response.status_code == 200:
                        data = response.json()
                        chunk_data = data.get('results', [])
                        if chunk_data:
                            all_data.extend(chunk_data)
                            
                            # Check if we need to fetch more pages
                            while 'metadata' in data and len(chunk_data) >= 1000:
                                params['offset'] = len(chunk_data)
                                time.sleep(0.5)  # Rate limiting
                                response = self._make_request('data', params)
                                if response.status_code == 200:
                                    data = response.json()
                                    chunk_data = data.get('results', [])
                                    if chunk_data:
                                        all_data.extend(chunk_data)
                                    else:
                                        break
                                else:
                                    break
                        break
                    elif response.status_code == 503:
                        logger.warning(f"Service temporarily unavailable, retrying in 5 seconds...")
                        time.sleep(5)
                        retry_count += 1
                    else:
                        logger.error(f"Failed to fetch data: {response.text}")
                        break
                except Exception as e:
                    logger.error(f"Error fetching data: {str(e)}")
                    retry_count += 1
                    time.sleep(5)
            
            current_start = current_end + pd.DateOffset(days=1)
            
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()
    
    def fetch_and_process(self) -> None:
        """Main method to fetch and process all weather data"""
        logger.info("Starting NOAA data fetch process...")
        
        # Get available stations
        stations = self._get_stations()
        logger.info(f"Found {len(stations)} NOAA stations in Bay Area with required data")
        
        # Fetch data for each station
        all_data = []
        for station in tqdm(stations, desc="Fetching station data"):
            df = self._fetch_data(
                station['id'],
                '2023-01-01',
                '2023-12-31'
            )
            
            if not df.empty:
                df['latitude'] = station['latitude']
                df['longitude'] = station['longitude']
                all_data.append(df)
        
        if not all_data:
            raise Exception("No data fetched from any station")
        
        # Combine all station data
        df = pd.concat(all_data, ignore_index=True)
        
        # Convert to wide format
        df['date'] = pd.to_datetime(df['date'])
        df_wide = df.pivot_table(
            index=['date', 'latitude', 'longitude'],
            columns='datatype',
            values='value'
        ).reset_index()
        
        # Calculate monthly averages
        monthly = df_wide.groupby([
            df_wide['date'].dt.to_period('M'),
            'latitude',
            'longitude'
        ]).agg({
            'TMAX': 'mean',
            'TMIN': 'mean',
            'PRCP': 'sum',  # Monthly total precipitation
            'AWND': 'mean',  # Average wind speed
            'WDF2': lambda x: np.rad2deg(np.arctan2(
                np.mean(np.sin(np.deg2rad(x))),
                np.mean(np.cos(np.deg2rad(x)))
            )) % 360  # Circular mean for wind direction
        }).reset_index()
        
        # Clean up date format
        monthly['date'] = monthly['date'].astype(str)
        
        # Calculate average temperature
        monthly['TAVG'] = (monthly['TMAX'] + monthly['TMIN']) / 2
        
        # Estimate relative humidity based on temperature and precipitation
        # This is a simplified model that assumes:
        # 1. Higher precipitation = higher humidity
        # 2. Higher temperature = lower humidity
        # 3. Base RH of 60% adjusted by temp and precip
        
        # Normalize temperature and precipitation to 0-1 scale
        temp_norm = (monthly['TAVG'] - monthly['TAVG'].min()) / (monthly['TAVG'].max() - monthly['TAVG'].min())
        precip_norm = (monthly['PRCP'] - monthly['PRCP'].min()) / (monthly['PRCP'].max() - monthly['PRCP'].min())
        
        # Calculate estimated RH
        # Base RH of 60%, increased by precipitation (up to +20%) and decreased by temperature (up to -20%)
        monthly['RH'] = 60 + (20 * precip_norm) - (20 * temp_norm)
        monthly['RH'] = monthly['RH'].clip(30, 90)  # Clip to reasonable range for Bay Area
        
        # Calculate VPD using estimated RH
        t = monthly['TAVG']
        rh = monthly['RH']
        
        # Calculate saturation vapor pressure
        svp = 0.61078 * np.exp((17.27 * t) / (t + 237.3))  # in kPa
        
        # Calculate actual vapor pressure
        vp = svp * (rh / 100)
        
        # Calculate VPD
        monthly['VPD'] = svp - vp  # in kPa
        
        # Drop intermediate calculation column
        monthly = monthly.drop('TAVG', axis=1)
        
        # Save to CSV
        output_file = "data/weather/noaa_ca.csv"
        monthly.to_csv(output_file, index=False)
        logger.info(f"Saved monthly weather data to {output_file}")
        
        # Save metadata
        metadata = {
            "date_range": ["2023-01-01", "2023-12-31"],
            "variables": {
                "TMAX": "Maximum temperature (Celsius)",
                "TMIN": "Minimum temperature (Celsius)",
                "PRCP": "Total precipitation (mm)",
                "AWND": "Average wind speed (meters/second)",
                "WDF2": "Wind direction (degrees from north)",
                "RH": "Estimated relative humidity (%) - derived from temperature and precipitation patterns",
                "VPD": "Vapor Pressure Deficit (kPa) - calculated from temperature and estimated humidity"
            },
            "stations_used": len(stations),
            "temporal_resolution": "Monthly averages",
            "notes": [
                "Wind direction is calculated using circular mean",
                "Precipitation is monthly total, all other variables are monthly averages",
                "Wind direction is measured in degrees from north (0° = N, 90° = E, 180° = S, 270° = W)",
                "Relative humidity is estimated using a simple model based on temperature and precipitation patterns",
                "VPD indicates the atmospheric moisture demand; higher values = drier conditions",
                "Note: The humidity estimation is approximate and should be used with caution"
            ]
        }
        
        with open("data/weather/noaa_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    try:
        fetcher = NOAADataFetcher()
        fetcher.fetch_and_process()
    except Exception as e:
        logger.error(f"Error in NOAA data fetching: {str(e)}")
        raise