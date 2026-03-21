"""
base_csv_reader.py

CSV Reader model for parsing and serving time-series data within the simulation framework.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""
import os
import pandas as pd
import datetime
import copy
from .base_model import BaseModel


class BaseCSVReader(BaseModel):
    """
    Base class for CSV readers. This class provides common functionality for reading CSV files and extracting data.
    Subclasses should implement the specific logic for processing the CSV data and defining the model's inputs, outputs, and parameters.
    """

    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)
        



    
    def initialize(self):
        self._data = None
        self.starting_row = 0
        self._data = self._load_csv_data()
        self._data = self._resample_if_required(self._data)
        self.starting_row = self._start_point() 
        self._row_idx = self.starting_row
        self._set_init_state()
    
    def _set_init_state(self):
        self.logger.debug(f"Setting initial state: for cols {self._data.columns} from row {self.starting_row}")
        for col in self._data.columns:
            self.init_state.outputs[col] = self._data.iloc[self.starting_row][col]
            self.state.outputs[col] = self._data.iloc[self.starting_row][col] 
            
    def _load_csv_data(self):
         
        csv_path = self.state.parameters.get("csv_path", None)
        if csv_path is None:
            self.logger.error("CSV path not specified in parameters")
            raise ValueError("CSV path not specified in parameters")

        if not os.path.isabs(csv_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, csv_path)

        if not os.path.exists(csv_path):
            self.logger.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        skip = int(self.state.parameters.get("skip_rows", 0))
        df = pd.read_csv(csv_path, skiprows=skip)
        df = self._check_existing_columns(df)
        self.logger.debug(f"CSV data loaded from {csv_path} with columns: {df.columns.tolist()}")
        self.logger.debug(f"CSV data preview:\n{df.head()}")
        
        return df

    def _check_existing_columns(self, df):
        """Check if all required columns exist in the CSV data and return a DataFrame with only those columns."""
        needed = self.config.outputs
        missing = [col for col in needed if col not in df.columns]
        if missing:
            self.logger.error(f"Missing columns in CSV: {missing}")
            raise ValueError(f"Missing columns in CSV: {missing}")
        return df[needed]
    
    def _resample_if_required(self, df):
        # TODO: implement resampling logic
        return df
    
    def _start_point(self):
        # TODO: implement logic of starting point maybe from datetime
        # need to return an index
        return 0
    
    def step(self):
        """Advance the CSV reader by one step. Populates self.state.outputs with values from the current row of self._data."""
        if self._row_idx >= len(self._data):
            self.logger.warning("End of CSV data reached. Resetting to the beginning.")
            self._row_idx = self.starting_row
            
        row = self._data.iloc[self._row_idx]
        for col in self._data.columns:
            self.state.outputs[col] = row[col]
        self._row_idx += 1
    
    def finalize(self):
        """Finalize the CSV reader. Subclasses can implement this method to perform any cleanup if necessary."""
        pass

    def reset(self, mode='full', ts=None, time=None):
        
        if mode == 'full':
            # for now only implement full reset with redundant output setting because the initial condition should be already published from federate
            self._row_idx = self.starting_row
            first_row = self._data.iloc[self._row_idx]
            for col in self._data.columns:
                self.state.outputs[col] = first_row[col]
        elif mode == 'soft':
            self.logger.debug("Soft reset: maintaining current state, no changes made.")
        
        elif mode == 'rolling':
            # `ts` is the absolute rolling start point provided by the federate.
            # Assign directly to avoid cumulative triangular jumps.
            if ts is None:
                self.starting_row += 1
            else:
                self.starting_row = int(ts)
            if self.starting_row >= len(self._data):
                self.logger.warning("Rolling reset: new starting point exceeds data length, resetting to the beginning.")
                self.starting_row = 0
            self._row_idx = self.starting_row
            self.logger.debug(f"Rolling reset: resetting to new starting point: {self.starting_row}.")

    
