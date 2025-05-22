from glob import glob
from typing import Tuple
import numpy as np
import pandas as pd
import random
import os

# website for the dataset
# https://datadryad.org/


class RamanDataLoader:
    """
    A class to load Raman spectral data from .txt or .csv files.

    Attributes:
    -----------
    directory_path : str
        Path to the directory containing Raman data files
    csv_filepath : str
        Path to the CSV file containing Raman spectral data
    pk_filepath : str
        Path to the pickle file containing Raman spectral data
    """

    def __init__(self, path=None):
        self.data = None

        if not os.path.exists(path):
            print(f"Path {path} does not exist")
            raise FileNotFoundError(f"Path {path} does not exist")

        if ".pkl" in path:
            self.data = self.load_pkl(path)

        elif ".csv" in path:
            self.data = self.load_csv(path)

        elif os.path.isdir(path):
            self.data = self.load_txt(path)

        else:
            print("Invalid file type or directory path")
            raise ValueError("Invalid file type or directory path")

    def load_txt(self, directory_path) -> pd.DataFrame:
        """
        Load all .txt Raman data files from a directory into a pandas DataFrame.

        Parameters:
        -----------
        directory_path : str
            Path to the directory containing Raman data files

        Returns:
        --------
        DataFrame with columns:
            - wavenumber: The Raman shift in cm^-1
            - Multiple columns for each sample's intensity values
        """
        # Check if directory path is provided
        if directory_path is None:
            print("No directory path provided")
            raise ValueError("No directory path provided")
            return None

        # Check if directory exists
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist")
            raise FileNotFoundError(
                f"Directory {directory_path} does not exist")

        # Find all txt files in the directory
        file_paths = glob.glob(os.path.join(directory_path, "*.txt"))

        if not file_paths:
            print(f"No .txt files found in {directory_path}")
            raise ValueError(f"No .txt files found in {directory_path}")

        # Create a dictionary to store all dataframes
        data_dict = {}
        wavenumbers = None

        # Process each file
        for file_path in file_paths:
            # Get the filename without extension for column naming
            file_name = os.path.basename(file_path).split('.')[0]

            # Load data from the file
            df = pd.read_csv(file_path, sep=',', header=None,
                             names=['wavenumber', file_name])

            # If this is our first file, save the wavenumbers
            if wavenumbers is None:
                wavenumbers = df['wavenumber'].values
                data_dict['wavenumber'] = wavenumbers

            # Store the intensity values
            data_dict[file_name] = df[file_name].values

        # Create a combined dataframe
        combined_df = pd.DataFrame(data_dict)

        print(f"Loaded {len(file_paths)} Raman spectra files")
        return combined_df

    def load_csv(self, csv_filepath, wavenumberColName: str = 'wavenumber', wavenumberColIndex: int = 0) -> pd.DataFrame:
        """
        Load Raman spectral data from a CSV file into a pandas DataFrame.

        Parameters:
        -----------
        csv_filepath : str
            Path to the CSV file containing Raman spectral data

        Returns:
        --------
        DataFrame with columns:
            - wavenumber: The Raman shift in cm^-1
            - Multiple columns for each sample's intensity values
        """
        # Check if CSV file path is provided
        if csv_filepath is None:
            print("No CSV file path provided")
            raise ValueError("No CSV file path provided")
            return None

        # Check if file exists
        if not os.path.exists(csv_filepath):
            print(f"File {csv_filepath} does not exist")
            raise FileNotFoundError(f"File {csv_filepath} does not exist")

        # Load data
        try:
            # Read CSV file - header is in row 0
            df = pd.read_csv(csv_filepath, header=0)

            # Set the first column as the wavenumber column
            wavenumber_col = df.columns[wavenumberColIndex]
            df = df.rename(columns={wavenumber_col: wavenumberColName})
            df = df.set_index(wavenumberColName)

            # Check if data has expected structure
            if len(df.columns) < 2:
                raise ValueError(
                    f"CSV file {csv_filepath} does not contain enough columns")

            print(
                f"Loaded Raman spectra with {len(df.columns)-1} samples and {len(df)} data points")
            return df

        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise

    def load_pkl(self, pk_filepath) -> pd.DataFrame:
        """
        Load Raman spectral data from a pickle file into a pandas DataFrame.

        Parameters:
        -----------
        pk_filepath : str
            Path to the pickle file containing Raman spectral data

        Returns:
        --------
        DataFrame with columns:
            - wavenumber: The Raman shift in cm^-1
            - Multiple columns for each sample's intensity values
        """

        # Check if pickle file path is provided
        if pk_filepath is None:
            print("No pickle file path provided")
            raise ValueError("No pickle file path provided")
            return None

        # Check if file exists
        if not os.path.exists(pk_filepath):
            print(f"File {pk_filepath} does not exist")
            raise FileNotFoundError(f"File {pk_filepath} does not exist")

        # Load data
        try:
            df = pd.read_pickle(pk_filepath)

            # Check if data has expected structure
            if len(df.columns) < 2:
                raise ValueError(
                    f"Pickle file {pk_filepath} does not contain enough columns")

            print(
                f"Loaded Raman spectra with {len(df.columns)-1} samples and {len(df)} data points")
            return df

        except Exception as e:
            print(f"Error loading pickle file: {e}")
            raise


def processDFA1(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Process the DataFrame to create a new DataFrame with interpolated spectra for dataset A1.
    Args:
        df (pd.DataFrame): Input DataFrame with columns 'xAxis', 'RawSpectra', 'CoreID', and 'Label'.

    Returns:
        pd.DataFrame: New DataFrame with interpolated spectra.
        list: List of labels corresponding to the spectra.
    """
    all_wavenumbers = np.unique(np.concatenate(df["xAxis"].values))
    all_wavenumbers.sort()

    data = {}
    for idx, (wn, spec, cid) in enumerate(zip(
            df["xAxis"],
            df["RawSpectra"],
            df["CoreID"])):
        # Make unique column name if needed
        col_name = f"{cid}_{idx}"
        interp_spec = np.interp(all_wavenumbers, wn, spec)
        data[col_name] = interp_spec

    # 3. Create DataFrame: index=wavenumber, columns=ids
    merged_df = pd.DataFrame(data, index=all_wavenumbers)
    merged_df.index.name = "wavenumber"

    return merged_df
