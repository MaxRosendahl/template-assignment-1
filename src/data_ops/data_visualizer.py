import json
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly

class DataVisualizer:
    """Visualize processed energy system data using plots."""


    def __init__(self):
        """Initialize with default plot settings."""
        plt.style.use('default')

    
    def plot_prices(self, hourly_data, question_name):
        """Plot energy prices over 24 hours
        
        Args: 
        hourly_data (pd.DataFrame): DataFrame with 'hour' and 'price' columns
        question_name (str): Name of the question/scenario for title and saving the plot
        """
        plt.figure(figsize=(10, 6))

        plt.plot(hourly_data['hour'], hourly_data['hourly_energy_price'], marker='o')

        plt.xlabel('Hour of Day')
        plt.ylabel('Price (DKK/kWh)')
        plt.title(f'Energy Prices Over 24 Hours - {question_name}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_solar(self, solar_data, question_name):
        """Plot solar production profile over 24 hours
        
        Args: 
        solar_data (pd.DataFrame): DataFrame with 'hour' and 'production ratio' columns
        question_name (str): Name of the question/scenario for title
        """
        plt.figure(figsize=(10, 6))

        plt.plot(solar_data['hour'], solar_data['production_ratio'], marker='o', color='orange')

        plt.xlabel('Hour of Day')
        plt.ylabel('Production Ratio')
        plt.title(f'Solar Production Profile Over 24 Hours - {question_name}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_load(self, load_data, question_name):
        """Plot load profile over 24 hours
        
        Args: 
        load_data (pd.DataFrame): DataFrame with 'hour' and 'load_ratio' columns
        question_name (str): Name of the question/scenario for title and saving the plot
        """
        plt.figure(figsize=(10, 6))

        plt.plot(load_data['hour'], load_data['load_ratio'], marker='o', color='green')

        plt.xlabel('Hour of Day')
        plt.ylabel('Load Ratio')
        plt.title(f'Load Profile Over 24 Hours - {question_name}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def create_all_plots(self, processed_data, question_name):
        """
        Create all available plots for a question.
        
        Args:
            processed_data: Dictionary from DataProcessor
            question_name: Name of the question
        """
        print(f"\nCreating plots for: {question_name}")
        print("="*50)
        
        # Plot prices (always available)
        if 'hourly_prices' in processed_data:
            print("Plotting energy prices...")
            self.plot_prices(processed_data['hourly_prices'], question_name)
        
        # Plot solar (if available)
        if 'solar_profile' in processed_data:
            print("Plotting solar production...")
            self.plot_solar(processed_data['solar_profile'], question_name)
        
        # Plot load (if available)
        if 'load_profile' in processed_data:
            print("Plotting load profile...")
            self.plot_load(processed_data['load_profile'], question_name)
        
        print(f"All plots created for {question_name}!\n")
