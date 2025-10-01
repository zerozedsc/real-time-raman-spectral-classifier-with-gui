import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ramanspy as rp

import pickle
import traceback
import random

from functions.configs import *
from functions.utils import translate_confusion_matrix
from functions.data_loader import RamanDataLoader
from functions.noise_func import RamanNoiseProcessor
from functions.preprocess import RamanPipeline, SNV, BaselineCorrection, Transformer1DBaseline, MultiScaleConv1D
from functions.visualization import RamanVisualizer
from functions.ML import RamanML, MLModel

try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    device = 'cpu'
    print("PyTorch not installed, defaulting to CPU.")

def processDFA1(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Process the DataFrame to create a new DataFrame with interpolated spectra.
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


def load_pickle(path):
    if not os.path.exists(path):
        console_log(f"File not found: {path}")
        return None

    data = None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
