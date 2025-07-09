import os
import glob
import json
from typing import Tuple, Any, Union, List
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from configs.configs import create_logs

def load_data_from_path(path: str) -> Union[pd.DataFrame, str]:
    """
    Loads spectral data from a given path, which can be a single file
    (.csv, .pkl) or a directory of .txt files.
    """
    if not os.path.exists(path):
        error_msg = f"Path does not exist: {path}"
        create_logs("DataLoader", "data_loading", error_msg, status='error')
        return error_msg

    if os.path.isdir(path):
        return _load_from_txt_directory(path)
    
    _, extension = os.path.splitext(path)
    if extension == '.csv':
        return _load_from_csv(path)
    elif extension == '.pkl':
        return _load_from_pkl(path)
    elif extension == '.txt':
        return _load_from_txt_directory(os.path.dirname(path))
    else:
        error_msg = f"Unsupported file type: {extension}"
        create_logs("DataLoader", "data_loading", error_msg, status='error')
        return error_msg

def _load_from_txt_directory(directory_path: str) -> Union[pd.DataFrame, str]:
    """Loads all .txt files from a directory into a DataFrame."""
    try:
        file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
        if not file_paths:
            raise ValueError(f"No .txt files found in {directory_path}")

        data_dict = {}
        wavenumbers = None

        for file_path in file_paths:
            file_name = os.path.basename(file_path).split('.')[0]
            df = pd.read_csv(file_path, sep=',', header=None, names=['wavenumber', 'intensity'])

            if df.shape[1] != 2 or df.isnull().any().any():
                create_logs("DataLoader", "data_loading", f"Skipping invalid file: {file_name}", status='warning')
                continue

            if wavenumbers is None:
                wavenumbers = df['wavenumber'].values
            elif not np.allclose(wavenumbers, df['wavenumber'].values):
                create_logs("DataLoader", "data_loading", f"Wavenumber mismatch in {file_name}, skipping.", status='warning')
                continue
            
            data_dict[file_name] = df['intensity'].values

        if not data_dict:
            raise ValueError("No valid .txt files could be loaded.")

        combined_df = pd.DataFrame(data_dict)
        combined_df.index = wavenumbers
        combined_df.index.name = 'wavenumber'
        
        create_logs("DataLoader", "data_loading", f"Successfully loaded {len(data_dict)} spectra from {directory_path}", status='info')
        return combined_df

    except Exception as e:
        create_logs("DataLoader", "data_loading", f"Error loading from {directory_path}: {e}", status='error')
        return str(e)

def _load_from_csv(csv_filepath: str) -> Union[pd.DataFrame, str]:
    """Loads spectral data from a CSV file."""
    try:
        df = pd.read_csv(csv_filepath, header=0, index_col=0)
        df.index.name = 'wavenumber'
        create_logs("DataLoader", "data_loading", f"Successfully loaded CSV: {csv_filepath}", status='info')
        return df
    except Exception as e:
        create_logs("DataLoader", "data_loading", f"Error loading CSV {csv_filepath}: {e}", status='error')
        return str(e)

def _load_from_pkl(pk_filepath: str) -> Union[pd.DataFrame, str]:
    """Loads spectral data from a pickle file."""
    try:
        df = pd.read_pickle(pk_filepath)
        df.index.name = 'wavenumber'
        create_logs("DataLoader", "data_loading", f"Successfully loaded PKL: {pk_filepath}", status='info')
        return df
    except Exception as e:
        create_logs("DataLoader", "data_loading", f"Error loading PKL {pk_filepath}: {e}", status='error')
        return str(e)

def load_metadata_from_json(json_path: str) -> Union[dict, str]:
    """Loads metadata from a specified JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        create_logs("DataLoader", "metadata_loading", f"Successfully loaded metadata: {json_path}", status='info')
        return metadata
    except Exception as e:
        create_logs("DataLoader", "metadata_loading", f"Error loading metadata {json_path}: {e}", status='error')
        return str(e)

def plot_spectra(df: pd.DataFrame) -> Figure:
    """
    Generates a matplotlib Figure object containing a plot of the spectra.
    Plots a maximum of 10 spectra for clarity.
    """
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # --- Robustness Check ---
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data to display.", ha='center', va='center', fontsize=14, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    # --- Plotting Logic ---
    num_spectra = df.shape[1]
    plot_title = "Loaded Raman Spectra"
    
    # Limit the number of plotted spectra for clarity
    if num_spectra > 10:
        df_to_plot = df.iloc[:, :10]
        plot_title += f" (showing first 10 of {num_spectra})"
    else:
        df_to_plot = df

    # Plot each spectrum
    for column in df_to_plot.columns:
        ax.plot(df_to_plot.index, df_to_plot[column], label=column)
    
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add a legend
    ax.legend()

    fig.tight_layout()
    return fig
