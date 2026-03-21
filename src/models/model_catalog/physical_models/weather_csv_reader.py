import os
import csv
from ...base_model import BaseModel


class WeatherCSVReader(BaseModel):
    """
    Weather data reader that streams temperature from a CSV file.

    At each simulation step the model advances one row in the CSV file and
    publishes the outdoor temperature (T_ext).  When the end of the file is
    reached the reader wraps around (cyclic replay), so the simulation
    length is not constrained by the file length.

    The expected CSV format (header required):
        DateTime, ..., T_ext, ...
    Any extra columns are silently ignored.

    Inputs:  none

    Outputs:
        - T_ext : Outdoor air temperature read from the CSV [°C]

    Parameters:
        - csv_path   : Absolute or workspace-relative path to the CSV file
        - column     : Name of the temperature column to read (default "T_ext")
        - skip_rows  : Number of header rows already consumed (default 0 = start
                       at the first data row)
    """

    MODEL_NAME = "weather_csv_reader"

    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)

    def initialize(self):
        """Open the CSV file and load all rows into memory."""
        csv_path = self.state.parameters.get("csv_path", "")
        if not os.path.isabs(csv_path):
            # Resolve relative path from the directory of this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, csv_path)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"WeatherCSVReader: CSV file not found: {csv_path}")

        self._column = self.state.parameters.get("column", "T_ext")
        skip = int(self.state.parameters.get("skip_rows", 0))

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if self._column not in rows[0]:
            raise KeyError(
                f"WeatherCSVReader: column '{self._column}' not found in CSV. "
                f"Available columns: {list(rows[0].keys())}"
            )

        # Store only the temperature values; skip the requested leading rows
        self._data = [float(r[self._column]) for r in rows[skip:]]
        self._n_rows = len(self._data)
        self._row_idx = 0

        self.logger.info(
            f"WeatherCSVReader '{self.name}': loaded {self._n_rows} rows "
            f"from '{csv_path}', column='{self._column}'"
        )

        # Seed initial output
        self.state.outputs["T_ext"] = self._data[0]

    def step(self) -> None:
        """Advance one CSV row and publish T_ext (wraps around at end of file)."""
        self.state.outputs["T_ext"] = self._data[self._row_idx % self._n_rows]
        self._row_idx += 1

    def finalize(self):
        self.logger.info(
            f"WeatherCSVReader '{self.name}' finalized after {self._row_idx} steps."
        )

    def reset(self, mode='full', ts=None, time=None) :
        if mode == 'full':
            # for now only implement full reset with redunndant output setting beacuse the initial condition should be already piublished from federate
            self._row_idx = 0
            self.state.outputs["T_ext"] = self._data[0]
