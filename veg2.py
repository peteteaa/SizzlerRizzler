import os
import gdown
import rasterio
import pandas as pd
import numpy as np
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime

def get_latest_ndvi_file():
    """Get the latest NDVI file from GEE_Exports folder in Google Drive."""
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    try:
        # Load credentials from the token file if it exists
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            # Save credentials
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        # Build the Drive API service
        service = build('drive', 'v3', credentials=creds)
        
        # Search for the GEE_Exports folder
        folder_results = service.files().list(
            q="name='GEE_Exports' and mimeType='application/vnd.google-apps.folder'",
            spaces='drive',
            fields='files(id)'
        ).execute()
        
        if not folder_results['files']:
            raise Exception("GEE_Exports folder not found")
            
        folder_id = folder_results['files'][0]['id']
        
        # Search for NDVI files in the folder
        results = service.files().list(
            q=f"'{folder_id}' in parents and name contains 'sanfrancisco_ndvi' and mimeType='image/tiff'",
            spaces='drive',
            fields='files(id, name, createdTime)',
            orderBy='createdTime desc'
        ).execute()
        
        if not results['files']:
            raise Exception("No NDVI files found in GEE_Exports folder")
            
        latest_file = results['files'][0]
        return latest_file['id'], latest_file['name']
        
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None, None

def download_geotiff(file_id, output_geotiff):
    """Download a file from Google Drive using its file ID."""
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_geotiff, quiet=False)

def extract_ndvi_to_csv(input_geotiff, output_csv):
    with rasterio.open(input_geotiff) as src:
        # Read the NDVI band (assuming it's the first band in the TIFF)
        ndvi = src.read(1)
        no_data_value = src.nodata

        # Get the affine transformation
        transform = src.transform

        # Create a meshgrid of coordinates
        rows, cols = np.indices(ndvi.shape)
        xs, ys = rasterio.transform.xy(transform, rows, cols)

        # Flatten the arrays and filter out no-data values
        lat = np.array(ys).flatten()
        lon = np.array(xs).flatten()
        ndvi_flat = ndvi.flatten()
        
        # Handle no-data values
        mask = ndvi_flat != no_data_value
        lat = lat[mask]
        lon = lon[mask]
        ndvi_values = ndvi_flat[mask]

        # Create a DataFrame and save to CSV
        df = pd.DataFrame({'lat': lat, 'lon': lon, 'ndvi': ndvi_values})
        df.to_csv(output_csv, index=False)
        print(f"NDVI values extracted to {output_csv}")

def main():
    print("Searching for latest NDVI file...")
    file_id, file_name = get_latest_ndvi_file()
    
    if not file_id:
        print("Failed to find NDVI file in GEE_Exports folder")
        return
        
    output_geotiff = 'ndvi.tif'  # Desired output GeoTIFF file name
    output_csv = 'ndvi_values.csv'  # Desired output CSV file name

    # Download the GeoTIFF from Google Drive
    print(f"Downloading {file_name}...")
    download_geotiff(file_id, output_geotiff)

    # Check if the download was successful
    if not os.path.exists(output_geotiff):
        print("Error: GeoTIFF file was not downloaded successfully.")
        return

    # Attempt to open and process the GeoTIFF file
    try:
        print("Extracting NDVI values...")
        extract_ndvi_to_csv(output_geotiff, output_csv)
    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening the GeoTIFF file: {e}")

if __name__ == "__main__":
    main()
