#!/usr/bin/env python3
"""
Run the Controlled Burn Analyzer to find suitable locations for controlled burns
in the San Francisco Bay Area based on weather and vegetation data.
"""

import os
import sys
import logging
from controlled_burn_analyzer import ControlledBurnAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Check for Groq API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Please set it using: export GROQ_API_KEY=your_api_key")
        sys.exit(1)
    
    print("Starting Controlled Burn Analysis...")
    
    # Initialize the analyzer
    analyzer = ControlledBurnAnalyzer(api_key)
    
    try:
        # Find suitable locations with a minimum score of 0.7
        suitable_locations = analyzer.find_suitable_locations(min_score=0.7)
        
        # Visualize results
        if suitable_locations:
            map_file = analyzer.visualize_results(suitable_locations, show_buffer_zones=False)
            print(f"\nAnalysis complete! Found {len(suitable_locations)} suitable locations.")
            print(f"Map saved to {map_file}")
            
            # Print top 5 locations
            print("\nTop locations for controlled burns:")
            sorted_locations = sorted(suitable_locations, key=lambda x: x['score'], reverse=True)
            for i, location in enumerate(sorted_locations[:5], 1):
                print(f"{i}. Score: {location['score']:.2f} - {location['reason']}")
                print(f"   Location: {location['latitude']:.4f}, {location['longitude']:.4f}")
                print(f"   Date: {location['date']}")
                print()
        else:
            print("\nNo suitable locations found for controlled burns.")
            print("Consider adjusting the minimum score threshold or collecting more data.")
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
