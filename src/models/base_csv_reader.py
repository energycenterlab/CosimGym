"""
base_csv_reader.py

CSV Reader model for parsing and serving time-series data within the simulation framework.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""
import inspect
import os
import pandas as pd
from .base_model import BaseModel

#TODO add functionalities to the csv reader resampling and starting dates
class BaseCSVReader(BaseModel):
    """
    Base class for CSV readers. This class provides common functionality for reading CSV files and extracting data.
    Subclasses should implement the specific logic for processing the CSV data and defining the model's inputs, outputs, and parameters.

    The row served at each step is derived from the model's clock, not from a cursor:
    ``starting_row + horizon_pos_s() / real_period`` (modulo the file length), so a reset
    (RL episode, rolling or full) or the end of the file needs no logic of its own. A file is
    the model's horizon: unless the catalog or the scenario declares ``max_sim_time``, it is
    ``len(data) * real_period``.
    """

    def __init__(self, name, metadata, config, logger):
        """Delegate to BaseModel.__init__; CSV-specific state is set up in `initialize()`."""
        super().__init__(name, metadata, config, logger)

    def initialize(self):
        """Load the CSV, resample it (currently a no-op — see `_resample_if_required`), and seed the initial output state."""
        self._data = self._load_csv_data()
        self._data = self._resample_if_required(self._data)
        self.starting_row = self._start_point()
        if not self.max_sim_time:
            self.max_sim_time = len(self._data) * self.real_period
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
            # relative to the script of the (sub)class that reads the file
            base_dir = os.path.dirname(os.path.abspath(inspect.getfile(type(self))))
            csv_path = os.path.join(base_dir, csv_path)

        if not os.path.exists(csv_path):
            self.logger.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        skip = int(self.state.parameters.get("skip_rows", 0))
        df = pd.read_csv(csv_path)
        df = self._check_existing_columns(df).iloc[skip:].reset_index(drop=True)
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
        # TODO: implement resampling logic — currently always a no-op identity pass-through.
        return df

    def _start_point(self):
        # TODO: implement logic of starting point maybe from datetime — currently always returns 0.
        # need to return an index
        return 0
    
    def step(self):
        """Populate self.state.outputs with the row the model's clock points at (wrapping around the file)."""
        row = self._data.iloc[(self.starting_row + int(self.horizon_pos_s() // self.real_period)) % len(self._data)]
        for col in self._data.columns:
            self.state.outputs[col] = row[col]
    
    def finalize(self):
        """Finalize the CSV reader. Subclasses can implement this method to perform any cleanup if necessary."""
        pass
