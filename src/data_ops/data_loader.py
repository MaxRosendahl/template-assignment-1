# -----------------------------
# Load Data
# -----------------------------
import os
import json
import csv
import pandas as pd
from pathlib import Path

from pathlib import Path
from dataclasses import dataclass
from logging import Logger
import pandas as pd
import xarray as xr
import numpy as np
import yaml


class DataLoader:
    """
    Loads energy system input data for a given configuration/question from structured CSV and json files
    and an auxiliary configuration metadata file.
    """
    question: str
    input_path: Path

    def __init__(self, question: str, input_path: str = 'data/'):
        """
        Initialize DataLoader with question name and input path.
        Args:
            question (str): Name of the question/scenario to load data for
            input_path (str): Path to the directory containing input data files
        """
        self.question = question
        self.input_path = Path(input_path) / self.question
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path {self.input_path} does not exist.")
        
        self._load_dataset()
    

    def _load_dataset(self):
        """
        Load all relevant CSV and json files from the input_path directory.
        """
        for file_name in os.listdir(self.input_path):
            if file_name.endswith('.csv') or file_name.endswith('.json'):
                self._load_data_file(file_name)
        


    def _load_data_file(self, file_name: str):
        """
        Load a single data file (CSV or JSON) and store it as a class attribute.
        Args:
            file_name (str): Name of the file to load
        """
        file_path = self.input_path / file_name
        data_name = file_name.split('.')[0]  # Use the file name (without extension) as the attribute name

        try:
            if file_name.endswith('.csv'):
                setattr(self, data_name, pd.read_csv(file_path))
            elif file_name.endswith('.json'):
                with open(file_path, 'r') as f:
                    setattr(self, data_name, json.load(f))
            else: 
                print(f"Unsupported file format for {file_name}. Only .csv and .json are supported.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Required file {file_name} not found in {self.input_path}")
        
    def get_data(self):
        """
        Get all loaded data as a dictionary for DataProcessor.
        
        Returns:
            Dictionary with all loaded data files
        """
        data_dict = {}
        
        # Get all attributes that are data (not methods or private attributes)
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name not in ['question', 'input_path', 'get_data', 'load_aux_data']:
                data_dict[attr_name] = getattr(self, attr_name)
        
        return data_dict

    def load_aux_data(self, filename: str):
        """
        Load auxiliary configuration data from a YAML file.
        Args:
            filename (str): Name of the YAML file to load   
        """

        file_path = self.input_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Auxiliary file {filename} not found in {self.input_path}")
        
        with open(file_path, 'r') as f:
            self.aux_data = yaml.safe_load(f)

        for key, value in self.aux_data.items():
            setattr(self, key, value)
        