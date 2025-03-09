import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import time
import numpy as np
from tqdm import tqdm

class NOAADataFetcher:
    """
    A class to fetch, process, and save NOAA climate data for San Francisco for 2023.
    """
    
    def __init__(self, api_key):
        """
        Initialize the fetcher with API key and San Francisco bounding box.
        
        Args:
            api_key (str): Your NOAA API key
        """
        self.api_key = api_key
        self.base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"
        self.headers = {"token": api_key}
        
        # San Francisco bounding box (minlat, minlon, maxlat, maxlon)
        # This covers the city and immediate surrounding areas
        self.sf_bbox = {"minlat": 37.70, "minlon": -122.51, "maxlat": 37.83, "maxlon": -122.35}
        
        # Create data directory if it doesn't exist
        os.makedirs("data/weather", exist_ok=True)
        
    def fetch_stations(self):
        """
        Fetch all weather stations in San Francisco.
        
        Returns:
            list: List of station metadata dictionaries
        """
        print("Fetching San Francisco weather stations...")
        
        stations = []
        offset = 1
        limit = 1000
        
        while True:
            params = {
                "extent": f"{self.sf_bbox['minlat']},{self.sf_bbox['minlon']},{self.sf_bbox['maxlat']},{self.sf_bbox['maxlon']}",
                "datasetid": "GHCND",  # Global Historical Climatology Network Daily
                "limit": limit,
                "offset": offset
            }
            
            response = requests.get(f"{self.base_url}stations", headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"Error fetching stations: {response.text}")
                break
            
            data = response.json()
            if "results" not in data or len(data["results"]) == 0:
                break
                
            stations.extend(data["results"])
            offset += limit
            
            # Sleep to avoid hitting rate limits
            time.sleep(0.5)
        
        print(f"Found {len(stations)} stations in San Francisco")
        return stations
    
    def fetch_data_for_station(self, station_id, start_date, end_date):
        """
        Fetch temperature and precipitation data for a specific station and date range.
        
        Args:
            station_id (str): The station ID
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            pandas.DataFrame: Data for the station
        """
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "datatypeid": "TMAX,TMIN,PRCP",  # Max temp, Min temp, Precipitation
            "units": "standard",  # Use Fahrenheit for temp, inches for precip
            "limit": 1000
        }
        
        all_data = []
        offset = 1
        
        while True:
            params["offset"] = offset
            response = requests.get(f"{self.base_url}data", headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"Error fetching data for station {station_id}: {response.text}")
                break
            
            data = response.json()
            if "results" not in data or len(data["results"]) == 0:
                break
                
            all_data.extend(data["results"])
            if len(data["results"]) < params["limit"]:
                break
                
            offset += params["limit"]
            
            # Sleep to avoid hitting rate limits
            time.sleep(0.5)
        
        if not all_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        return df
    
    def fetch_all_data(self, year=2023):
        """
        Fetch data for all stations in San Francisco for 2023.
        
        Args:
            year (int): Year to fetch data for (default: 2023)
            
        Returns:
            pandas.DataFrame: Raw data from all stations
            dict: Station coordinate information
        """
        stations = self.fetch_stations()
        
        # Extract station coordinates
        station_coords = {
            station["id"]: {
                "name": station["name"],
                "latitude": station["latitude"],
                "longitude": station["longitude"]
            } 
            for station in stations
        }
        
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        print(f"Fetching San Francisco weather data for {year}...")
        
        all_data = []
        for station in tqdm(stations):
            station_id = station["id"]
            df = self.fetch_data_for_station(station_id, start_date, end_date)
            
            if not df.empty:
                # Add station coordinates
                df["latitude"] = station["latitude"]
                df["longitude"] = station["longitude"]
                all_data.append(df)
        
        # Combine all station data
        if not all_data:
            raise ValueError("No data was retrieved. Check API key or try different stations.")
            
        full_df = pd.concat(all_data, ignore_index=True)
        return full_df, station_coords
    
    def process_data(self, df, station_coords):
        """
        Process the raw NOAA data into a clean format with daily values.
        
        Args:
            df (pandas.DataFrame): Raw NOAA data
            station_coords (dict): Dictionary of station coordinates
            
        Returns:
            pandas.DataFrame: Processed daily data
        """
        print("Processing data...")
        
        # Extract date from 'date' column
        df['date'] = pd.to_datetime(df['date'])
        
        # Create separate dataframes for each data type
        tmax_df = df[df['datatype'] == 'TMAX'].copy()
        tmin_df = df[df['datatype'] == 'TMIN'].copy()
        prcp_df = df[df['datatype'] == 'PRCP'].copy()
        
        # Rename 'value' column to the respective measurement
        tmax_df = tmax_df.rename(columns={'value': 'tmax'})
        tmin_df = tmin_df.rename(columns={'value': 'tmin'})
        prcp_df = prcp_df.rename(columns={'value': 'prcp'})
        
        # Merge into a new dataframe with date, station, and measurements
        daily_data = []
        
        # Process TMAX data
        if not tmax_df.empty:
            tmax_data = tmax_df[['date', 'station', 'tmax', 'latitude', 'longitude']]
            daily_data.append(tmax_data)
            
        # Process TMIN data
        if not tmin_df.empty:
            tmin_data = tmin_df[['date', 'station', 'tmin', 'latitude', 'longitude']]
            if daily_data:
                daily_data[0] = pd.merge(daily_data[0], tmin_data, on=['date', 'station', 'latitude', 'longitude'], how='outer')
            else:
                daily_data.append(tmin_data)
                
        # Process PRCP data
        if not prcp_df.empty:
            prcp_data = prcp_df[['date', 'station', 'prcp', 'latitude', 'longitude']]
            if daily_data:
                daily_data[0] = pd.merge(daily_data[0], prcp_data, on=['date', 'station', 'latitude', 'longitude'], how='outer')
            else:
                daily_data.append(prcp_data)
        
        if not daily_data:
            raise ValueError("No valid temperature or precipitation data found")
            
        result_df = daily_data[0]
        
        # Add station name
        result_df['station_name'] = result_df['station'].apply(
            lambda x: station_coords.get(x, {}).get('name', 'Unknown')
        )
        
        # Extract year, month from date
        result_df['year'] = result_df['date'].dt.year
        result_df['month'] = result_df['date'].dt.month
        
        # Calculate average temperature
        result_df['tavg'] = result_df[['tmax', 'tmin']].mean(axis=1, skipna=True)
        
        # Convert temperatures from tenths of degrees to degrees
        if 'tmax' in result_df.columns:
            result_df['tmax'] = result_df['tmax'] / 10
        if 'tmin' in result_df.columns:
            result_df['tmin'] = result_df['tmin'] / 10
        if 'tavg' in result_df.columns:
            result_df['tavg'] = result_df['tavg'] / 10
            
        # Convert precipitation from tenths of mm to inches
        if 'prcp' in result_df.columns:
            result_df['prcp'] = result_df['prcp'] / 10
        
        return result_df
    
    def aggregate_to_monthly(self, df):
        """
        Aggregate data into monthly averages for San Francisco.
        Since we're focusing on a small area, we'll aggregate by month across all stations.
        
        Args:
            df (pandas.DataFrame): Processed daily data
            
        Returns:
            pandas.DataFrame: Aggregated monthly data
        """
        print("Aggregating data into monthly averages...")
        
        # Group by month
        grouped = df.groupby(['year', 'month'])
        
        # Calculate aggregates
        aggregated = grouped.agg({
            'tmax': lambda x: np.nanmean(x) if 'tmax' in df.columns else np.nan,
            'tmin': lambda x: np.nanmean(x) if 'tmin' in df.columns else np.nan,
            'tavg': lambda x: np.nanmean(x) if 'tavg' in df.columns else np.nan,
            'prcp': lambda x: np.nansum(x) if 'prcp' in df.columns else np.nan,
            'station': 'nunique',
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        # Rename station count column
        aggregated = aggregated.rename(columns={'station': 'station_count'})
        
        return aggregated
    
    def run(self, year=2023):
        """
        Run the complete pipeline: fetch, process, aggregate, and save data for San Francisco.
        
        Args:
            year (int): Year to fetch data for (default: 2023)
            
        Returns:
            str: Path to saved CSV file
        """
        print(f"Fetching NOAA data for San Francisco ({year})...")
        
        # Fetch raw data
        raw_data, station_coords = self.fetch_all_data(year)
        
        # Process data
        processed_data = self.process_data(raw_data, station_coords)
        
        # Aggregate into monthly averages
        monthly_data = self.aggregate_to_monthly(processed_data)
        
        # Save to CSV
        output_path = "data/weather/noaa_sf.csv"
        monthly_data.to_csv(output_path, index=False)
        
        print(f"Data successfully saved to {output_path}")
        
        # Also save daily data for more detailed analysis
        daily_output_path = "data/weather/noaa_sf_daily.csv"
        processed_data.to_csv(daily_output_path, index=False)
        print(f"Daily data saved to {daily_output_path}")
        
        return output_path

# Example usage
if __name__ == "__main__":
    # Replace with your actual NOAA API key
    api_key = "pEnzQayogxrBBCFXAGoGwbwbdCbpQYZy"
    
    fetcher = NOAADataFetcher(api_key)
    output_file = fetcher.run(year=2023)
    
    print(f"Completed! Monthly data saved to {output_file}")