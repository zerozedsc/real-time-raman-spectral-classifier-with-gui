from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import ramanspy as rp
import matplotlib.pyplot as plt

class RamanPipeline:
    """
    A class to handle the preprocessing of Raman spectral data.
    
    Attributes:
    -----------
    region : tuple
        The range of wavenumbers to consider for analysis.
    """
    
    def __init__(self, region: tuple[int, int]=(1050, 1700)):
        """
        Initializes the RamanPipeline class.
        """
        self.region = region


    def pipeline_hirschsprung_multi(self,
        hirsch_dfs: List[pd.DataFrame], 
        normal_dfs: List[pd.DataFrame],
        wavenumber_rowname: str = 'wavenumber',
        region: Tuple[int, int] = (1050, 1700)
    ) -> Tuple[np.ndarray, list, np.ndarray, pd.DataFrame]:
        """
        Preprocessing pipeline for multiple Hirshsprung disease and normal Raman DataFrames.
        
        Parameters:
        ----------
        hirsch_dfs : List[pd.DataFrame]
            List of DataFrames containing Hirshsprung disease Raman spectra.
        normal_dfs : List[pd.DataFrame]
            List of DataFrames containing normal Raman spectra.
        region : Tuple[int, int], optional
            The range of wavenumbers to consider for analysis (default is (1050, 1700)).
            
        Returns:
        -------
        processed_data : np.ndarray
            The processed spectral data.
        labels : list
            List of labels indicating the source of each spectrum (Hirshsprung or normal).
        wavenumbers : np.ndarray
            The wavenumbers corresponding to the spectral data.
        merged_df : pd.DataFrame
            The merged DataFrame containing all spectra, with wavenumbers as the first column.
        
        """
        
        # Handle empty input cases
        if not hirsch_dfs and not normal_dfs:
            raise ValueError("Both hirsch_dfs and normal_dfs are empty. At least one must be provided.")
        elif hirsch_dfs:
            wavenumbers = hirsch_dfs[0]['wavenumber'].values
        else:
            wavenumbers = normal_dfs[0]['wavenumber'].values

        # Concatenate all hirsch and normal DataFrames (drop wavenumber, keep only intensity columns)
        all_hirsch = [df.drop(wavenumber_rowname, axis=1) for df in hirsch_dfs] if hirsch_dfs else []
        all_normal = [df.drop(wavenumber_rowname, axis=1) for df in normal_dfs] if normal_dfs else []
        merged_df = pd.concat(all_hirsch + all_normal, axis=1)

        intensities = merged_df.values.T  # shape: (n_samples, n_wavenumbers)

        # Labels
        labels = []
        if hirsch_dfs:
            labels += ['hirsch'] * sum(df.shape[1] - 1 for df in hirsch_dfs)
        if normal_dfs:
            labels += ['normal'] * sum(df.shape[1] - 1 for df in normal_dfs)

        # Preprocessing pipeline
        pipeline = rp.preprocessing.Pipeline([
            rp.preprocessing.misc.Cropper(region=region),
            rp.preprocessing.despike.WhitakerHayes(),
            rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
            rp.preprocessing.baseline.ASPLS(lam=1e5, tol=0.01),
            rp.preprocessing.normalise.Vector()
        ])
        spectra = rp.SpectralContainer(intensities, wavenumbers)
        data = pipeline.apply(spectra)

        # Return processed data, labels, and wavenumbers for further analysis
        return data.spectral_data, labels, data.spectral_axis, merged_df


    def preprocess(
        self,
        dfs: List[pd.DataFrame],
        label: str,
        wavenumber_col: str = 'wavenumber',
        intensity_cols: Optional[List[str]] = None,
        region: Tuple[int, int] = (1050, 1700),
        preprocessing_steps: Optional[List[Callable]] = None,
        visualize_steps: bool = False,
        max_plot_visualize_steps: int = 10
    ) -> dict[str, Any]:
        """
        Dynamic preprocessing pipeline for generic Raman spectral DataFrames.

        Parameters
        ----------
        dfs : List[pd.DataFrame]
            List of DataFrames containing Raman spectra.
        label : str
            Label for the spectra (e.g., 'hirsch' or 'normal').
        wavenumber_col : str
            Column name for wavenumber axis.
        intensity_cols : Optional[List[str]]
            List of columns for intensity data. If None, use all except wavenumber_col and label_col.
        region : Tuple[int, int]
            Wavenumber region to crop.
        preprocessing_steps : Optional[List[Callable]]
            List of ramanspy preprocessing steps. If None, use default.
        visualize_steps : bool
            If True, visualize each preprocessing step.
        max_plot_visualize_steps : int
            Maximum number of spectra to visualize at each step.

        Returns
        -------
        dict:
            Dictionary containing processed spectra, labels, and raw DataFrame.
            {
                'processed': SpectralContainer,
                'labels': List[str],
                'raw': pd.DataFrame
            }
        """
        
        if type(dfs) is not list:
            dfs = [dfs]

        # Merge all DataFrames
        merged_df = pd.concat(dfs, axis=1 if dfs[0].index.name == wavenumber_col else 0)

        # Check if index is wavenumber
        if merged_df.index.name == wavenumber_col:
            wavenumbers = merged_df.index.values
            if intensity_cols is None:
                intensity_cols = merged_df.columns.tolist()
            intensities = merged_df[intensity_cols].values.T
        elif wavenumber_col in merged_df.columns:
            wavenumbers = merged_df[wavenumber_col].values
            if intensity_cols is None:
                exclude = {wavenumber_col}
                intensity_cols = [col for col in merged_df.columns if col not in exclude]
            intensities = merged_df[intensity_cols].values
        else:
            raise ValueError(f"Wavenumber column '{wavenumber_col}' not found in DataFrame index or columns.")

        # Labels: assign the provided label to all spectra
        n_spectra = intensities.shape[0] if intensities.ndim == 2 else 1
        labels = [label] * n_spectra if label is not None else []

        # Default preprocessing pipeline
        if preprocessing_steps is None:
            preprocessing_steps = [
                rp.preprocessing.misc.Cropper(region=region),
                rp.preprocessing.despike.WhitakerHayes(),
                rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
                rp.preprocessing.baseline.ASPLS(lam=1e5, tol=0.01),
                rp.preprocessing.normalise.Vector()
            ]

        # Apply steps one-by-one (if visualization enabled)
        spectra = rp.SpectralContainer(intensities, wavenumbers)

        if visualize_steps:
            fig, axes = plt.subplots(len(preprocessing_steps) + 1, 1, figsize=(12, 3 * (len(preprocessing_steps) + 1)), sharex=True)
            axes[0].set_title("Raw Spectra")
            for spectrum in spectra.spectral_data[:max_plot_visualize_steps]:
                axes[0].plot(spectra.spectral_axis, spectrum, alpha=0.6)

        for i, step in enumerate(preprocessing_steps):
            spectra = step.apply(spectra)
            if visualize_steps:
                axes[i + 1].set_title(f"After {step.__class__.__name__}")
                for spectrum in spectra.spectral_data[:max_plot_visualize_steps]:
                    axes[i + 1].plot(spectra.spectral_axis, spectrum, alpha=0.6)

        if visualize_steps:
            for ax in axes:
                ax.set_ylabel("Intensity")
            axes[-1].set_xlabel("Wavenumber (cm⁻¹)")
            plt.tight_layout()
            plt.show()

        return {"processed": spectra, 
                "labels": labels, 
                "raw" : merged_df}
    
     
class SNV:
    def __call__(self, spectra):
        # spectra: 2D numpy array (n_samples, n_features)
        return np.apply_along_axis(self.snv_normalisation, 1, spectra) 
    
    def apply(self, spectra):
        data = spectra.spectral_data
        snv_data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
        return rp.SpectralContainer(snv_data, spectra.spectral_axis)
    
    def snv_normalisation(spectrum):
        # spectrum: 1D numpy array
        return (spectrum - np.mean(spectrum)) / np.std(spectrum) 
    

class MovingAverage:
    def __init__(self, window_length=15):
        self.window_length = window_length

    def __call__(self, spectra):
        return self.apply(spectra)

    def apply(self, spectra):
        # spectra: rp.SpectralContainer
        smoothed_data = []
        for spectrum in spectra.spectral_data:
            smoothed = np.convolve(
                spectrum,
                np.ones(self.window_length) / self.window_length,
                mode='same'
            )
            smoothed_data.append(smoothed)
        return spectra.__class__(np.array(smoothed_data), spectra.spectral_axis)