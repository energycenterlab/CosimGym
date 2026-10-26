from ...base_csv_reader import BaseCSVReader


class WeatherCSVReader(BaseCSVReader):
    """
    Weather data reader that streams temperature from a CSV file.

    At each simulation step the model serves the next row of the CSV file and
    publishes the outdoor temperature (T_ext). The row comes from the model's clock
    (see ``BaseCSVReader``), and when the end of the file is reached the reader
    wraps around (cyclic replay), so the simulation length is not constrained by
    the file length.

    The expected CSV format (header required):
        DateTime, ..., T_ext, ...
    Any extra columns are silently ignored.

    Inputs:  none

    Outputs:
        - T_ext : Outdoor air temperature read from the CSV [°C]

    Parameters:
        - csv_path   : Absolute or workspace-relative path to the CSV file
        - column     : Name of the temperature column to read (default "T_ext")
        - skip_rows  : Number of data rows to skip at the start of the file
                       (default 0 = start at the first data row)
    """

    MODEL_NAME = "weather_csv_reader"

    def _check_existing_columns(self, df):
        """Keep only the configured temperature column, served as ``T_ext``."""
        column = self.state.parameters.get("column", "T_ext")
        if column not in df.columns:
            raise KeyError(
                f"WeatherCSVReader: column '{column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        return df[[column]].rename(columns={column: "T_ext"})
