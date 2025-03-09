import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from groq import Groq
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import math
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ControlledBurnAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Define buffer zones (in degrees, approximately 1 degree = 111km)
        self.buffer_zones = {
            'major_city': 0.5,       # ~55km buffer around major cities
            'small_city': 0.3,       # ~33km buffer around smaller cities
            'airport': 0.2,          # ~22km buffer around airports
            'national_park': 0.2,    # ~22km buffer around national parks
            'state_park': 0.1,       # ~11km buffer around state parks
            'landmark': 0.1          # ~11km buffer around other landmarks
        }
        
        # Define protected locations with their types
        self.protected_locations = [
            # Major Cities
            {'name': 'San Francisco', 'type': 'major_city', 'lat': 37.7749, 'lon': -122.4194},
            {'name': 'Oakland', 'type': 'major_city', 'lat': 37.8044, 'lon': -122.2711},
            {'name': 'San Jose', 'type': 'major_city', 'lat': 37.3387, 'lon': -121.8853},
            {'name': 'Sacramento', 'type': 'major_city', 'lat': 38.5816, 'lon': -121.4944},
            {'name': 'Fremont', 'type': 'major_city', 'lat': 37.5485, 'lon': -121.9886},
            {'name': 'Hayward', 'type': 'major_city', 'lat': 37.6688, 'lon': -122.0808},
            {'name': 'Sunnyvale', 'type': 'major_city', 'lat': 37.3688, 'lon': -122.0363},
            {'name': 'Santa Clara', 'type': 'major_city', 'lat': 37.3541, 'lon': -121.9552},
            {'name': 'Concord', 'type': 'major_city', 'lat': 37.9779, 'lon': -122.0301},
            {'name': 'Vallejo', 'type': 'major_city', 'lat': 38.1041, 'lon': -122.2566},
            
            # Smaller Cities
            {'name': 'Berkeley', 'type': 'small_city', 'lat': 37.8716, 'lon': -122.2727},
            {'name': 'Palo Alto', 'type': 'small_city', 'lat': 37.4443, 'lon': -122.1598},
            {'name': 'San Rafael', 'type': 'small_city', 'lat': 37.9735, 'lon': -122.5311},
            {'name': 'Santa Rosa', 'type': 'small_city', 'lat': 38.4405, 'lon': -122.7141},
            {'name': 'Walnut Creek', 'type': 'small_city', 'lat': 37.9101, 'lon': -122.0652},
            {'name': 'San Mateo', 'type': 'small_city', 'lat': 37.5630, 'lon': -122.3255},
            {'name': 'Livermore', 'type': 'small_city', 'lat': 37.6819, 'lon': -121.7680},
            {'name': 'Pleasanton', 'type': 'small_city', 'lat': 37.6624, 'lon': -121.8747},
            {'name': 'Redwood City', 'type': 'small_city', 'lat': 37.4852, 'lon': -122.2364},
            {'name': 'Mountain View', 'type': 'small_city', 'lat': 37.3861, 'lon': -122.0839},
            {'name': 'Novato', 'type': 'small_city', 'lat': 38.1074, 'lon': -122.5697},
            {'name': 'Napa', 'type': 'small_city', 'lat': 38.2975, 'lon': -122.2869},
            {'name': 'Fairfield', 'type': 'small_city', 'lat': 38.2494, 'lon': -122.0400},
            {'name': 'Vacaville', 'type': 'small_city', 'lat': 38.3565, 'lon': -121.9877},
            {'name': 'Milpitas', 'type': 'small_city', 'lat': 37.4323, 'lon': -121.8996},
            {'name': 'Cupertino', 'type': 'small_city', 'lat': 37.3230, 'lon': -122.0322},
            {'name': 'Gilroy', 'type': 'small_city', 'lat': 37.0058, 'lon': -121.5683},
            {'name': 'Morgan Hill', 'type': 'small_city', 'lat': 37.1305, 'lon': -121.6544},
            
            # Airports
            {'name': 'SFO Airport', 'type': 'airport', 'lat': 37.6213, 'lon': -122.3790},
            {'name': 'Oakland Airport', 'type': 'airport', 'lat': 37.7214, 'lon': -122.2208},
            {'name': 'San Jose Airport', 'type': 'airport', 'lat': 37.3639, 'lon': -121.9289},
            {'name': 'Sacramento Airport', 'type': 'airport', 'lat': 38.5452, 'lon': -121.4924},
            
            # National Parks and Monuments
            {'name': 'Yosemite Valley', 'type': 'national_park', 'lat': 37.7461, 'lon': -119.5332},
            {'name': 'Muir Woods', 'type': 'national_park', 'lat': 37.8970, 'lon': -122.5810},
            {'name': 'Point Reyes', 'type': 'national_park', 'lat': 38.0400, 'lon': -122.9000},
            
            # State Parks
            {'name': 'Mount Tamalpais', 'type': 'state_park', 'lat': 37.9235, 'lon': -122.5965},
            {'name': 'Angel Island', 'type': 'state_park', 'lat': 37.8608, 'lon': -122.4326},
            {'name': 'Mount Diablo', 'type': 'state_park', 'lat': 37.8817, 'lon': -121.9142},
            
            # Notable Landmarks
            {'name': 'Golden Gate Bridge', 'type': 'landmark', 'lat': 37.8199, 'lon': -122.4783},
            {'name': 'Bay Bridge', 'type': 'landmark', 'lat': 37.7983, 'lon': -122.3778},
            {'name': 'Alcatraz Island', 'type': 'landmark', 'lat': 37.8267, 'lon': -122.4233},
            {'name': 'UC Berkeley Campus', 'type': 'landmark', 'lat': 37.8719, 'lon': -122.2585}
        ]
        
        # Add Mount Diablo to the protected locations list if it's not already there
        mount_diablo = {'name': 'Mount Diablo', 'type': 'state_park', 'lat': 37.8817, 'lon': -121.9142}
        if mount_diablo not in self.protected_locations:
            self.protected_locations.append(mount_diablo)
        
        self.client = Groq(api_key=api_key)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load NOAA weather data and NDVI vegetation data"""
        try:
            # Load NOAA weather data
            weather_df = pd.read_csv('data/weather/noaa_ca.csv')
            
            # Check if vegetation data exists
            veg_path = 'data/vegetation/sf_ndvi_processed.csv'
            if os.path.exists(veg_path):
                veg_df = pd.read_csv(veg_path)
            else:
                # Create directory if it doesn't exist
                os.makedirs('data/vegetation', exist_ok=True)
                
                # Create a simple NDVI dataset if none exists
                # This is a placeholder - ideally we'd use real NDVI data
                dates = weather_df['date'].unique()
                veg_data = []
                for date in dates:
                    # Create synthetic NDVI values that vary by month
                    # Higher in spring/summer, lower in fall/winter
                    month = int(date.split('-')[1]) if '-' in date else 1
                    base_ndvi = 0.3 + 0.2 * math.sin((month - 3) * math.pi / 6)  # Peak in June
                    veg_data.append({
                        'date': date,
                        'ndvi': base_ndvi
                    })
                veg_df = pd.DataFrame(veg_data)
                veg_df.to_csv(veg_path, index=False)
            
            # Add some remote test locations that are far from cities
            # These are locations in more rural areas of Northern California
            remote_locations = [
                # Remote areas in Northern California
                {'latitude': 39.5501, 'longitude': -122.8344, 'date': '2023-04-01'},  # Mendocino National Forest
                {'latitude': 40.4427, 'longitude': -121.4206, 'date': '2023-04-01'},  # Lassen National Forest
                {'latitude': 38.7545, 'longitude': -120.0324, 'date': '2023-04-01'},  # El Dorado National Forest
                {'latitude': 39.2238, 'longitude': -123.7654, 'date': '2023-04-01'},  # Jackson State Forest
                {'latitude': 41.1809, 'longitude': -123.1029, 'date': '2023-04-01'},  # Six Rivers National Forest
                {'latitude': 38.4762, 'longitude': -119.8547, 'date': '2023-04-01'},  # Stanislaus National Forest
                {'latitude': 40.0689, 'longitude': -122.9714, 'date': '2023-04-01'},  # Shasta-Trinity National Forest
                {'latitude': 39.8173, 'longitude': -121.5463, 'date': '2023-04-01'},  # Plumas National Forest
                
                # Add the same locations for May
                {'latitude': 39.5501, 'longitude': -122.8344, 'date': '2023-05-01'},  # Mendocino National Forest
                {'latitude': 40.4427, 'longitude': -121.4206, 'date': '2023-05-01'},  # Lassen National Forest
                {'latitude': 38.7545, 'longitude': -120.0324, 'date': '2023-05-01'},  # El Dorado National Forest
                {'latitude': 39.2238, 'longitude': -123.7654, 'date': '2023-05-01'},  # Jackson State Forest
                {'latitude': 41.1809, 'longitude': -123.1029, 'date': '2023-05-01'},  # Six Rivers National Forest
                {'latitude': 38.4762, 'longitude': -119.8547, 'date': '2023-05-01'},  # Stanislaus National Forest
                {'latitude': 40.0689, 'longitude': -122.9714, 'date': '2023-05-01'},  # Shasta-Trinity National Forest
                {'latitude': 39.8173, 'longitude': -121.5463, 'date': '2023-05-01'},  # Plumas National Forest
                
                # Add Mount Diablo specifically as a potential burn location
                {'latitude': 37.8817, 'longitude': -121.9142, 'date': '2023-04-01'},  # Mount Diablo
                {'latitude': 37.8817, 'longitude': -121.9142, 'date': '2023-05-01'}   # Mount Diablo
            ]
            
            # Create a DataFrame for remote locations
            remote_df = pd.DataFrame(remote_locations)
            
            # Convert dates to datetime
            remote_df['date'] = pd.to_datetime(remote_df['date'])
            
            # Copy weather data from the nearest date and location
            for idx, row in remote_df.iterrows():
                # Find the nearest date in weather_df
                nearest_date = weather_df['date'].iloc[0]
                if pd.api.types.is_datetime64_any_dtype(weather_df['date']):
                    date_diff = abs(weather_df['date'] - row['date'])
                    nearest_date = weather_df.loc[date_diff.idxmin(), 'date']
                
                # Find a random weather row with that date
                weather_rows = weather_df[weather_df['date'] == nearest_date]
                if len(weather_rows) > 0:
                    weather_row = weather_rows.iloc[0].copy()
                    
                    # Copy weather data to the remote location
                    for col in weather_row.index:
                        if col not in ['latitude', 'longitude', 'date']:
                            remote_df.loc[idx, col] = weather_row[col]
            
            # Add remote locations to the weather dataframe
            weather_df = pd.concat([weather_df, remote_df], ignore_index=True)
                
            logger.info(f"Loaded {len(weather_df)} weather records and {len(veg_df)} vegetation records")
            return weather_df, veg_df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, weather_df: pd.DataFrame, veg_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess weather and vegetation data:
        1. Convert dates to datetime
        2. Handle missing values
        3. Merge weather and vegetation data
        """
        # Convert dates to datetime if they're not already
        if not pd.api.types.is_datetime64_any_dtype(weather_df['date']):
            weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        if not pd.api.types.is_datetime64_any_dtype(veg_df['date']):
            veg_df['date'] = pd.to_datetime(veg_df['date'])
            
        # Ensure column names are standardized
        if 'TMAX' in weather_df.columns and 'tmax' not in weather_df.columns:
            weather_df = weather_df.rename(columns={
                'TMAX': 'tmax',
                'TMIN': 'tmin',
                'PRCP': 'prcp',
                'AWND': 'wind_speed',
                'WDF2': 'wind_dir',
                'RH': 'humidity',
                'VPD': 'vapor_pressure_deficit'
            })
        
        # Merge with vegetation data
        merged_df = pd.merge(
            weather_df,
            veg_df,
            on='date',
            how='left'
        )
        
        # Fill missing NDVI values with forward fill then backward fill
        if 'ndvi' in merged_df.columns:
            merged_df['ndvi'] = merged_df['ndvi'].ffill().bfill()
        
        logger.info(f"Preprocessed data: {len(merged_df)} records")
        return merged_df
    
    def is_within_buffer_zone(self, lat: float, lon: float) -> Tuple[bool, str, float]:
        """
        Check if a location is within a buffer zone of any protected location.
        Returns a tuple of (is_within_buffer, location_name, distance)
        """
        for location in self.protected_locations:
            # Skip state parks and national parks - they are potential burn targets
            if location['type'] in ['state_park', 'national_park']:
                continue
                
            # Calculate distance in degrees (approximate)
            distance = ((lat - location['lat'])**2 + (lon - location['lon'])**2)**0.5
            buffer_size = self.buffer_zones[location['type']]
            
            if distance < buffer_size:
                return True, location['name'], distance
        
        return False, "", 0.0
    
    def analyze_location_suitability(self, data_row) -> Dict:
        """
        Analyze a location for controlled burn suitability based on weather and vegetation data.
        """
        lat = data_row['latitude']
        lon = data_row['longitude']
        
        # Check if location is within a buffer zone
        in_buffer, location_name, distance = self.is_within_buffer_zone(lat, lon)
        if in_buffer:
            return {
                'latitude': lat,
                'longitude': lon,
                'date': data_row['date'],
                'suitable': False,
                'score': 0,
                'reason': f"Location is within buffer zone of {location_name}"
            }
        
        # Additional check for proximity to ANY city - much more aggressive avoidance
        city_distances = []
        for location in self.protected_locations:
            if location['type'] in ['major_city', 'small_city']:
                # Calculate distance in degrees (approximate)
                distance = ((lat - location['lat'])**2 + (lon - location['lon'])**2)**0.5
                city_distances.append((location['name'], distance))
        
        # Sort by distance
        city_distances.sort(key=lambda x: x[1])
        
        # If the closest city is within 0.6 degrees (~66km), reject the location
        if city_distances and city_distances[0][1] < 0.6:
            return {
                'latitude': lat,
                'longitude': lon,
                'date': data_row['date'],
                'suitable': False,
                'score': 0,
                'reason': f"Location is too close to {city_distances[0][0]} for a controlled burn"
            }
        
        # If there are at least 3 cities within 1 degree (~111km), this is likely an urbanized area
        close_cities = [city for city, dist in city_distances if dist < 1.0]
        if len(close_cities) >= 3:
            return {
                'latitude': lat,
                'longitude': lon,
                'date': data_row['date'],
                'suitable': False,
                'score': 0,
                'reason': f"Location is in a densely populated region near {', '.join(close_cities[:3])}"
            }
        
        # Check if this is a park location - if so, give it a bonus
        is_park = False
        park_name = ""
        for location in self.protected_locations:
            if location['type'] in ['state_park', 'national_park']:
                # If this point is very close to a park, consider it a park location
                distance = ((lat - location['lat'])**2 + (lon - location['lon'])**2)**0.5
                if distance < 0.1:  # Within ~11km of park center
                    is_park = True
                    park_name = location['name']
                    break
        
        # Extract weather and vegetation metrics
        metrics = {
            'temperature': data_row.get('tmax', None),
            'humidity': data_row.get('rhum', None),
            'wind_speed': data_row.get('wsp', None),
            'precipitation': data_row.get('prcp', None),
            'ndvi': data_row.get('ndvi', None),
            'date': data_row.get('date', None)
        }
        
        # Analyze with LLM
        analysis = self.analyze_with_groq(metrics)
        
        # Extract score and reason
        score = analysis.get('score', 0)
        reason = analysis.get('reason', "No analysis provided")
        
        # Apply park bonus if applicable
        if is_park:
            score = min(1.0, score + 0.1)  # Add 0.1 to score, but cap at 1.0
            reason = f"Park location ({park_name}): {reason}"
        
        return {
            'latitude': lat,
            'longitude': lon,
            'date': data_row['date'],
            'suitable': score >= 0.7,
            'score': score,
            'reason': reason
        }
    
    def analyze_with_groq(self, metrics: Dict) -> Dict:
        """
        Use Groq to analyze the suitability of a location for controlled burns.
        """
        try:
            # Format metrics for the prompt
            metrics_str = "\n".join([f"{k}: {v}" for k, v in metrics.items() if v is not None])
            
            # Create the prompt
            prompt = f"""
            You are an expert in wildfire management and controlled burns. 
            Analyze the following weather and vegetation data to determine if conditions are suitable for a controlled burn.
            
            {metrics_str}
            
            Consider the following factors:
            1. Temperature: Ideal range is 40-80°F (4-27°C)
            2. Humidity: Ideal range is 20-40%
            3. Wind speed: Ideal range is 5-15 mph (8-24 km/h)
            4. Recent precipitation: Some moisture helps control the burn
            5. NDVI (vegetation index): Higher values indicate more fuel for burning
            
            Provide a score from 0.0 to 1.0 where:
            - 0.0-0.3: Unsuitable conditions
            - 0.4-0.6: Marginal conditions
            - 0.7-0.8: Good conditions
            - 0.9-1.0: Ideal conditions
            
            Also provide a brief reason for your score.
            
            Format your response as a JSON object with the following fields:
            {{"score": <float>, "reason": "<string>"}}
            """
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are an expert in wildfire management and controlled burns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            # Extract and parse the response
            response_text = response.choices[0].message.content
            
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis = json.loads(json_str)
                
                # Validate the response
                if 'score' not in analysis or 'reason' not in analysis:
                    return {
                        'score': 0.0,
                        'reason': "Invalid analysis format from AI"
                    }
                
                return analysis
            else:
                # If no JSON found, try to extract score and reason manually
                score_match = re.search(r'score:\s*(0\.\d+|\d+\.\d+)', response_text, re.IGNORECASE)
                reason_match = re.search(r'reason:\s*(.+?)(?=\n|$)', response_text, re.IGNORECASE)
                
                score = float(score_match.group(1)) if score_match else 0.0
                reason = reason_match.group(1).strip() if reason_match else "No clear reason provided"
                
                return {
                    'score': score,
                    'reason': reason
                }
                
        except Exception as e:
            logger.error(f"Error in Groq analysis: {str(e)}")
            return {
                'score': 0.0,
                'reason': f"Error in analysis: {str(e)}"
            }
    
    def find_suitable_locations(self, min_score: float = 0.7) -> List[Dict]:
        """
        Find locations suitable for controlled burns based on weather and vegetation data.
        Returns a list of suitable locations with their scores and reasons.
        """
        try:
            # Load and preprocess data
            weather_df, veg_df = self.load_data()
            data = self.preprocess_data(weather_df, veg_df)
            
            # Filter to only include locations that are likely remote
            # This is a very aggressive filter to avoid cities
            remote_locations = []
            for _, row in data.iterrows():
                lat = row['latitude']
                lon = row['longitude']
                
                # Check if this point is far from any city
                is_remote = True
                for location in self.protected_locations:
                    if location['type'] in ['major_city', 'small_city']:
                        distance = ((lat - location['lat'])**2 + (lon - location['lon'])**2)**0.5
                        if distance < 0.6:  # ~66km
                            is_remote = False
                            break
                
                if is_remote:
                    remote_locations.append(row)
            
            if not remote_locations:
                logger.warning("No remote locations found in dataset - all locations are too close to cities")
                return []
                
            # Analyze each remote location
            results = []
            for row in remote_locations:
                analysis = self.analyze_location_suitability(row)
                if analysis['suitable'] and analysis['score'] >= min_score:
                    results.append(analysis)
            
            logger.info(f"Found {len(results)} suitable locations for controlled burns")
            return results
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            raise
    
    def create_map(self, suitable_locations: List[Dict], show_buffer_zones: bool = False) -> folium.Map:
        """
        Create a map visualization of suitable locations for controlled burns.
        
        Args:
            suitable_locations: List of suitable locations with their scores
            show_buffer_zones: Whether to show buffer zones around protected locations
        
        Returns:
            Folium map object
        """
        # Calculate center of the map (average of all locations)
        if not suitable_locations:
            # Default to California center if no suitable locations
            center_lat, center_lon = 37.7749, -119.4179
        else:
            center_lat = sum(loc['latitude'] for loc in suitable_locations) / len(suitable_locations)
            center_lon = sum(loc['longitude'] for loc in suitable_locations) / len(suitable_locations)
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6, 
                      tiles='CartoDB positron')
        
        # Add title
        title_html = '''
            <h3 align="center" style="font-size:16px"><b>Controlled Burn Suitability Map</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Only show buffer zones if requested
        if show_buffer_zones:
            # Add buffer zones for protected locations
            for location in self.protected_locations:
                # Skip parks - they are potential burn targets
                if location['type'] in ['state_park', 'national_park']:
                    continue
                    
                buffer_size = self.buffer_zones[location['type']]
                # Convert buffer size from degrees to meters (approximate)
                buffer_meters = buffer_size * 111000  # 1 degree ~ 111 km
                
                # Add circle for buffer zone
                folium.Circle(
                    location=[location['lat'], location['lon']],
                    radius=buffer_meters,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.2,
                    popup=f"{location['name']} - No burn zone"
                ).add_to(m)
        
        # Add markers for suitable locations
        for i, location in enumerate(suitable_locations):
            # Format the popup content
            popup_content = f"""
            <b>Location {i+1}</b><br>
            Coordinates: {location['latitude']:.4f}, {location['longitude']:.4f}<br>
            Date: {location['date']}<br>
            Score: {location['score']:.2f}<br>
            Reason: {location['reason']}
            """
            
            # Add marker
            folium.Marker(
                location=[location['latitude'], location['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='green', icon='fire', prefix='fa')
            ).add_to(m)
            
            # Add circle to highlight the area
            folium.Circle(
                location=[location['latitude'], location['longitude']],
                radius=5000,  # 5 km radius
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.3,
            ).add_to(m)
        
        # Add parks as potential burn targets with a different color
        for location in self.protected_locations:
            if location['type'] in ['state_park', 'national_park']:
                folium.Marker(
                    location=[location['lat'], location['lon']],
                    popup=f"{location['name']} - Potential burn target",
                    icon=folium.Icon(color='orange', icon='tree', prefix='fa')
                ).add_to(m)
        
        return m
    
    def save_map(self, m: folium.Map, output_file: str):
        """
        Save a Folium map to an HTML file.
        
        Args:
            m: Folium map object
            output_file: Path to save the map HTML file
        """
        m.save(output_file)
    
    def visualize_results(self, results: List[Dict], output_file: str = 'controlled_burn_map.html', show_buffer_zones: bool = False) -> str:
        """
        Create and save a map visualization of the results.
        
        Args:
            results: List of analysis results
            output_file: Path to save the map HTML file
            show_buffer_zones: Whether to show buffer zones around protected locations
        
        Returns:
            Path to the saved map file
        """
        # Create the map
        m = self.create_map(results, show_buffer_zones=show_buffer_zones)
        
        # Save the map
        self.save_map(m, output_file)
        
        return output_file

def main():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("Please set GROQ_API_KEY environment variable")
    
    analyzer = ControlledBurnAnalyzer(api_key)
    
    try:
        # Find suitable locations
        suitable_locations = analyzer.find_suitable_locations(min_score=0.7)
        
        # Visualize results
        if suitable_locations:
            map_file = analyzer.visualize_results(suitable_locations)
            print(f"Analysis complete. Found {len(suitable_locations)} suitable locations.")
            print(f"Map saved to {map_file}")
        else:
            print("No suitable locations found for controlled burns.")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
