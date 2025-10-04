import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List


class DataProcessor:
    """Process raw energy system data into structured DataFrames."""

    def __init__(self):
        """Initialize with empty storage."""
        self.data = {}

    def process_all(self, raw_data, question_name):
        """
        Process all raw data for a question.
        
        Args:
            raw_data: Dictionary from DataLoader.get_data()
            question_name: Name of the question
            
        Returns:
            Dictionary with processed DataFrames
        """
        result = {}
        
        # Process bus parameters (electricity grid info)
        if 'bus_params' in raw_data:
            bus = raw_data['bus_params'][0]  # Get first entry from list
            result['bus_info'] = pd.DataFrame(raw_data['bus_params'])
            result['hourly_prices'] = pd.DataFrame({
                'hour': range(24),
                'hourly_energy_price': bus['energy_price_DKK_per_kWh']
            })


        
        # Process consumer parameters
        if 'consumer_params' in raw_data:
            result['consumer_info'] = pd.DataFrame(raw_data['consumer_params'])
        
        # Process appliance parameters (solar, loads, batteries)
        if 'appliance_params' in raw_data:
            appl = raw_data['appliance_params']
            if appl.get('DER'):
                result['DER'] = pd.DataFrame(appl['DER'])
            if appl.get('load'):
                result['load'] = pd.DataFrame(appl['load'])
            if appl.get('storage'):
                result['storage'] = pd.DataFrame(appl['storage'])
        
        # Process solar production profile
        if 'DER_production' in raw_data:
            der = raw_data['DER_production'][0]  # Get first entry from list
            result['solar_profile'] = pd.DataFrame({
                'hour': range(24),
                'production_ratio': der['hourly_profile_ratio']
            })
        
        # Process usage preferences (load profile)
        if 'usage_preferences' in raw_data:
            usage = raw_data['usage_preferences'][0]  # Get first entry from list
            if usage.get('load_preferences'):
                load_pref = usage['load_preferences'][0]  # Get first load preference
                if load_pref.get('hourly_profile_ratio'):
                    result['load_profile'] = pd.DataFrame({
                        'hour': range(24),
                        'load_ratio': load_pref['hourly_profile_ratio']
                    })

        # Store and return
        self.data[question_name] = result
        return result