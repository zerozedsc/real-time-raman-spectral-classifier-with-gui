
from functions.ML import RamanML
from functions.configs import *
from numpy import trapz

from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import numpy as np
import pandas as pd
import seaborn as sns
import ramanspy as rp
import matplotlib.pyplot as plt
import shap
import time
import ramanspy as rp


class RamanVisualizer:
    """
    A class to visualize Raman spectra data.

    Attributes:
    -----------
    df : pandas DataFrame
        DataFrame containing Raman data with wavenumber column and intensity columns
    """

    def __init__(self, df: pd.DataFrame = None,
                 spectral_container: List[rp.SpectralContainer] = None,
                 labels: List[str] = None,
                 common_axis: np.ndarray = None,
                 n_features: int = None,
                 ramanML: RamanML = None):
        self.ramanML = ramanML
        self.df = df
        self.spectral_container = spectral_container
        self.labels = labels
        if self.ramanML is not None:
            self.common_axis = self.ramanML.common_axis
            self.n_features = self.ramanML.n_features_in
        else:
            self.common_axis = common_axis
            self.n_features = n_features

    def get_peak_assignment(
        self,
        peak: Union[int, float, str],
        pick_near: bool = False,
        tolerance: int = 5,
        json_file_path: str = "data/raman_peaks.json"
    ) -> dict:
        """
        Get the assignment meaning of a Raman peak based on the raman_peaks.json database.

        Parameters:
        -----------
        peak : Union[int, float, str]
            The wavenumber peak to look up. If float, will be rounded to int.
        pick_near : bool, optional
            If True, will search for the nearest peak within tolerance if exact match not found.
            Default is False.
        tolerance : int, optional
            Maximum distance (in cm⁻¹) to search for nearby peaks when pick_near=True.
            Default is 5.
        json_file_path : str, optional
            Path to the raman_peaks.json file. Default is "data/raman_peaks.json".

        Returns:
        --------
        dict
            Dictionary containing peak assignment information:
            - If found: {"peak": peak, "assignment": assignment, "reference_number": ref_num}
            - If not found and pick_near=False: {"assignment": "Not Found"}
            - If not found and pick_near=True: nearest match or "Not Found"
        """
        try:
            # Convert peak to integer (round if float)
            if isinstance(peak, (float, str)):
                try:
                    peak_int = int(round(float(peak)))
                except (ValueError, TypeError):
                    return {"assignment": "Invalid peak value"}
            else:
                peak_int = int(peak)

            # Load the JSON data
            try:
                if hasattr(self, '_raman_peaks_cache'):
                    # Use cached data if available
                    raman_data = self._raman_peaks_cache
                else:
                    # Load and cache the data
                    import json
                    import os

                    # Handle relative path from current working directory
                    if not os.path.isabs(json_file_path):
                        # Try relative to current working directory first
                        if os.path.exists(json_file_path):
                            full_path = json_file_path
                        else:
                            # Try relative to the script directory
                            script_dir = os.path.dirname(
                                os.path.abspath(__file__))
                            full_path = os.path.join(
                                script_dir, "..", json_file_path)
                    else:
                        full_path = json_file_path

                    with open(full_path, 'r', encoding='utf-8') as file:
                        raman_data = json.load(file)

                    # Cache the data for future use
                    self._raman_peaks_cache = raman_data

            except FileNotFoundError:
                create_logs("get_peak_assignment", "ML",
                            f"Raman peaks file not found: {json_file_path}", status='error')
                return {"assignment": "Database file not found"}
            except json.JSONDecodeError as e:
                create_logs("get_peak_assignment", "ML",
                            f"Error parsing JSON file: {e}", status='error')
                return {"assignment": "Database file corrupted"}

            # Convert peak to string for lookup (JSON keys are strings)
            peak_str = str(peak_int)

            # Direct lookup first
            if peak_str in raman_data:
                result = raman_data[peak_str].copy()
                result["peak"] = peak_int
                return result

            # If not found and pick_near is False, return "Not Found"
            if not pick_near:
                return {"assignment": "Not Found"}

            # Find nearest peak within tolerance
            nearest_peak = None
            min_distance = float('inf')

            for db_peak_str in raman_data.keys():
                try:
                    db_peak_int = int(db_peak_str)
                    distance = abs(peak_int - db_peak_int)

                    if distance <= tolerance and distance < min_distance:
                        min_distance = distance
                        nearest_peak = db_peak_str
                except ValueError:
                    # Skip invalid peak keys
                    continue

            # Return nearest peak if found within tolerance
            if nearest_peak is not None:
                result = raman_data[nearest_peak].copy()
                result["peak"] = int(nearest_peak)
                result["distance"] = min_distance
                result["original_peak"] = peak_int
                return result
            else:
                return {"assignment": "Not Found"}

        except Exception as e:
            create_logs("get_peak_assignment", "ML",
                        f"Error in get_peak_assignment: {e}", status='error')
            return {"assignment": "Error occurred during lookup"}

    def get_multiple_peak_assignments(
        self,
        peaks: List[Union[int, float, str]],
        pick_near: bool = False,
        tolerance: int = 5,
        json_file_path: str = "data/raman_peaks.json"
    ) -> List[dict]:
        """
        Get assignments for multiple peaks at once.

        Parameters:
        -----------
        peaks : List[Union[int, float, str]]
            List of wavenumber peaks to look up.
        pick_near : bool, optional
            If True, will search for the nearest peak within tolerance if exact match not found.
        tolerance : int, optional
            Maximum distance (in cm⁻¹) to search for nearby peaks when pick_near=True.
        json_file_path : str, optional
            Path to the raman_peaks.json file.

        Returns:
        --------
        List[dict]
            List of dictionaries containing peak assignment information for each input peak.
        """
        results = []
        for peak in peaks:
            result = self.get_peak_assignment(
                peak, pick_near, tolerance, json_file_path)
            results.append(result)
        return results

    def find_peaks_in_range(
        self,
        min_wavenumber: Union[int, float],
        max_wavenumber: Union[int, float],
        json_file_path: str = "data/raman_peaks.json"
    ) -> List[dict]:
        """
        Find all peaks within a specified wavenumber range.

        Parameters:
        -----------
        min_wavenumber : Union[int, float]
            Minimum wavenumber of the range.
        max_wavenumber : Union[int, float]
            Maximum wavenumber of the range.
        json_file_path : str, optional
            Path to the raman_peaks.json file.

        Returns:
        --------
        List[dict]
            List of all peaks within the specified range.
        """
        try:
            # Load data using the existing method
            dummy_result = self.get_peak_assignment(
                1000, json_file_path=json_file_path)
            if "Database file" in str(dummy_result.get("assignment", "")):
                return []

            # Get cached data
            raman_data = self._raman_peaks_cache

            results = []
            for peak_str, data in raman_data.items():
                try:
                    peak_int = int(peak_str)
                    if min_wavenumber <= peak_int <= max_wavenumber:
                        result = data.copy()
                        result["peak"] = peak_int
                        results.append(result)
                except ValueError:
                    continue

            # Sort by wavenumber
            results.sort(key=lambda x: x["peak"])
            return results

        except Exception as e:
            create_logs("find_peaks_in_range", "ML",
                        f"Error in find_peaks_in_range: {e}", status='error')
            return []

    def visualize_raman_spectra(self, wavenumber_colname: str = "wavenumber", title="Raman Spectra", figsize=(12, 6),
                                xlim=None, ylim=None, legend=True,
                                legend_loc='best', sample_limit=10) -> plt:
        """
        Visualize the Raman spectra data.
        Clears any existing plots before creating a new one.

        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame containing Raman data with wavenumber column and intensity columns
        title : str
            Plot title
        figsize : tuple
            Figure size (width, height) in inches
        xlim : tuple or None
            Optional x-axis limits (min, max)
        ylim : tuple or None
            Optional y-axis limits (min, max)
        legend : bool
            Whether to display the legend
        legend_loc : str
            Legend location
        sample_limit : int
            Maximum number of samples to plot (to avoid overcrowding)
        """
        df = self.df

        # Check if DataFrame is empty
        if df.empty:
            print("DataFrame is empty. No data to plot.")
            return None

        # Determine wavenumber axis
        if df.index.name == wavenumber_colname:
            wavenumbers = df.index.values
            intensity_columns = df.columns
        elif wavenumber_colname in df.columns:
            wavenumbers = df[wavenumber_colname].values
            intensity_columns = [
                col for col in df.columns if col != wavenumber_colname]
        else:
            print(
                f"Wavenumber column '{wavenumber_colname}' not found in DataFrame.")
            return None

        # Limit number of samples to plot if needed
        if len(intensity_columns) > sample_limit:
            print(
                f"Limiting plot to {sample_limit} samples out of {len(intensity_columns)}")
            intensity_columns = intensity_columns[:sample_limit]

        # Clear any existing plots
        plt.clf()
        plt.close('all')
        fig = plt.figure(num=1, figsize=figsize, clear=True)

        # Plot each spectrum
        for col in intensity_columns:
            if df.index.name == wavenumber_colname:
                plt.plot(wavenumbers, df[col], label=col)
            else:
                plt.plot(wavenumbers, df[col].values, label=col)

        # Set plot attributes
        plt.title(title)
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity')
        plt.grid(True, alpha=0.3)

        if legend:
            plt.legend(loc=legend_loc, fontsize='small')

        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)

        plt.tight_layout()
        return plt

    def visualize_processed_spectra(self, spectral_data, spectral_axis,
                                    title="Processed Raman Spectra", figsize=(12, 6),
                                    xlim=None, ylim=None, legend=True,
                                    sample_names=None, legend_loc='best',
                                    sample_limit=10, cmap='viridis',
                                    add_mean=False) -> plt:
        """
        Visualize processed spectral data from ramanspy.

        Parameters:
        -----------
        spectral_data : np.ndarray
            Array of shape (n_samples, n_wavenumbers) containing processed spectra
        spectral_axis : np.ndarray
            Array containing wavenumber axis
        title : str
            Plot title
        figsize : tuple
            Figure size (width, height) in inches
        xlim : tuple or None
            Optional x-axis limits (min, max)
        ylim : tuple or None
            Optional y-axis limits (min, max)
        legend : bool
            Whether to display the legend
        sample_names : list or None
            List of names for each spectrum (if None, will use Sample 1, Sample 2, etc.)
        legend_loc : str
            Legend location
        sample_limit : int
            Maximum number of samples to plot (to avoid overcrowding)
        cmap : str
            Matplotlib colormap for spectra if no sample names provided
        add_mean : bool
            Whether to add the mean spectrum in bold

        Returns:
        --------
        plt : matplotlib.pyplot
            The plot object for further customization
        """
        # Check inputs
        if spectral_data.shape[1] != len(spectral_axis):
            # Transpose if needed (sometimes ramanspy returns (n_wavenumbers, n_samples))
            if spectral_data.shape[0] == len(spectral_axis):
                spectral_data = spectral_data.T
            else:
                raise ValueError(f"Spectral data shape {spectral_data.shape} doesn't match " +
                                 f"spectral axis length {len(spectral_axis)}")

        # Limit samples if needed
        n_samples = spectral_data.shape[0]
        if n_samples > sample_limit:
            print(
                f"Limiting plot to {sample_limit} samples out of {n_samples}")
            plot_samples = min(n_samples, sample_limit)
        else:
            plot_samples = n_samples

        # Prepare sample names
        if sample_names is None:
            sample_names = [f"Sample {i+1}" for i in range(plot_samples)]
        elif len(sample_names) < plot_samples:
            # Extend sample names if too few
            sample_names = list(
                sample_names) + [f"Sample {i+1}" for i in range(len(sample_names), plot_samples)]

        # Clear any existing plots
        plt.clf()
        plt.close('all')
        fig = plt.figure(num=1, figsize=figsize, clear=True)

        # Plot each spectrum
        cmap_func = plt.get_cmap(cmap)
        for i in range(plot_samples):
            color = cmap_func(i / max(1, plot_samples-1)
                              ) if n_samples > 1 else cmap_func(0.5)
            plt.plot(spectral_axis, spectral_data[i], label=sample_names[i],
                     color=color, alpha=0.7)

        # Add mean spectrum if requested
        if add_mean and n_samples > 1:
            mean_spectrum = np.mean(spectral_data[:plot_samples], axis=0)
            plt.plot(spectral_axis, mean_spectrum, 'k-',
                     linewidth=2, label='Mean Spectrum')

        # Set plot attributes
        plt.title(title)
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity (normalized)')
        plt.grid(True, alpha=0.3)

        if legend:
            if n_samples <= 15:  # Only show legend if not too many samples
                plt.legend(loc=legend_loc, fontsize='small')

        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)

        plt.tight_layout()
        return plt

    def extract_raman_characteristics(x: np.ndarray, y: np.ndarray, sample_name: str = "Sample", show_plot: bool = False) -> tuple[list[tuple], float]:
        """
        Extract Raman characteristics from the spectrum.
        Parameters:
        ----------
        x : np.ndarray
            Wavenumber array
        y : np.ndarray
            Intensity array
        sample_name : str
            Name of the sample for labeling
        show_plot : bool
            Whether to show the plot of the spectrum
        Returns:
        -------
        top_peaks : list[tuple]
            List of tuples containing peak positions and intensities
        auc : float
            Area under the curve

        """

        # Find peaks
        peaks, props = find_peaks(
            y, height=np.percentile(y, 80))  # simple threshold
        peak_positions = x[peaks]
        peak_intensities = y[peaks]

        # Sort peaks by intensity
        sorted_idx = np.argsort(peak_intensities)[::-1]
        top_peaks = [(peak_positions[i], peak_intensities[i])
                     for i in sorted_idx[:5]]

        # Area under curve
        auc = trapz(y, x)

        # Plot
        if show_plot:
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, label="Raman Spectrum")
            plt.scatter(peak_positions, peak_intensities,
                        color='red', label="Detected Peaks")
            for pos, inten in top_peaks:
                plt.text(pos, inten, f"{int(pos)}", fontsize=8, ha='center')
            plt.title(f"Raman Spectrum - {sample_name}")
            plt.xlabel("Wavenumber (cm⁻¹)")
            plt.ylabel("Intensity (a.u.)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Print characteristic summary
        print(f"----- {sample_name} Characteristics -----")
        print(f"Top Peaks:")
        for i, (pos, inten) in enumerate(top_peaks):
            print(f"  {i+1}. {pos:.1f} cm⁻¹  (Intensity: {inten:.1f})")
        print(f"Total Area Under Curve (AUC): {auc:.2f}")
        print(f"Approximate Noise Level (std): {np.std(y):.2f}")

        return top_peaks, auc

    def pca2d(
        self,
        df: pd.DataFrame = None,
        containers: List[rp.SpectralContainer] = None,
        labels: List[str] = None,
        common_axis: np.ndarray = None,
        current_n_features: int = None,
        title: str = "PCA",
        figsize: tuple = (12, 6),
        sample_limit: int = 100,
        cmap: str = 'tab10',
        add_centroids: bool = True,
        show_centroid_line: bool = True,
        legend: bool = True,
        legend_loc: str = 'best'
    ) -> plt:
        """
        Perform PCA on a list of ramanspy SpectralContainer objects and plot the first two PCs.

        Args:
            containers (List[rp.SpectralContainer]):
                List of SpectralContainer objects containing spectral data
            labels (List[str]):
                List of labels for each spectrum
            title (str):
                Title of the plot
            figsize (tuple):
                Size of the figure (width, height)
            sample_limit (int):
                Maximum number of samples to plot (to avoid overcrowding)
            cmap (str):
                Colormap for the plot
            add_centroids (bool):
                Whether to add centroids for each class
            show_centroid_line (bool):
                Whether to show a line between centroids
            legend (bool):
                Whether to show the legend
            legend_loc (str):
                Location of the legend

        Returns:
            plt (matplotlib.pyplot):
                The plot object for further customization
        """

        labels = labels if labels is not None else self.labels
        if labels is None:
            raise ValueError(
                "You must provide labels for DataFrame input.")

        # --- Data extraction ---
        df = df if df is not None else self.df
        containers = containers if containers is not None else self.spectral_container
        if df is not None:
            # DataFrame mode

            # Assume spectra are all columns except label column
            if isinstance(labels, str):
                # If labels is a column name
                y = df[labels].values
                X = df.drop(columns=[labels]).values
            else:
                y = np.array(labels)
                X = df.values
            if X.shape[0] != len(y):
                print("Label length mismatch, using integer indices for legend.")
                y = np.arange(X.shape[0])
        elif containers is not None:
            # SpectralContainer mode
            common_axis = common_axis if common_axis is not None else self.common_axis
            if common_axis is None:
                raise ValueError(
                    "You must provide a common axis for interpolation.")
            current_n_features = current_n_features if current_n_features is not None else self.n_features
            if current_n_features is None:
                raise ValueError(
                    "You must provide the number of features for interpolation.")

            X_list = []
            for s_container in containers:
                if s_container.spectral_data is None or s_container.spectral_data.size == 0:
                    continue
                for single_spectrum_original_axis in s_container.spectral_data:
                    if single_spectrum_original_axis.ndim != 1:
                        continue
                    if len(s_container.spectral_axis) != current_n_features:
                        interp_spectrum = np.interp(
                            common_axis, s_container.spectral_axis, single_spectrum_original_axis)
                        X_list.append(interp_spectrum)
                    else:
                        X_list.append(single_spectrum_original_axis)
            if not X_list:
                raise ValueError(
                    "No valid spectra found in the provided containers.")
            X = np.array(X_list)
            y = np.array(labels[:X.shape[0]])
            if X.shape[0] != len(y):
                print("Label length mismatch, using integer indices for legend.")
                y = np.arange(X.shape[0])
        else:
            raise ValueError(
                "You must provide either a DataFrame or a list of SpectralContainer objects.")

        print(f"Generating PCA plot for {X.shape[0]} spectra.")

        # Limit number of samples if needed
        if X.shape[0] > sample_limit:
            print(
                f"Limiting plot to {sample_limit} samples out of {X.shape[0]}")
            X = X[:sample_limit]
            y = y[:sample_limit]

        # --- Fit PCA ---
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Plot PCA
        plt.figure(figsize=figsize)
        unique_labels = np.unique(y)
        colors = plt.get_cmap(cmap, len(unique_labels))
        label_to_color = {label: colors(i)
                          for i, label in enumerate(unique_labels)}
        for label in unique_labels:
            idxs = np.where(y == label)[0]
            plt.scatter(X_pca[idxs, 0], X_pca[idxs, 1], label=label,
                        color=label_to_color[label], alpha=0.7)

        # Centroids and line between them
        centroids = []
        for label in unique_labels:
            idxs = np.where(y == label)[0]
            centroid = X_pca[idxs].mean(axis=0)
            centroids.append(centroid)
            if add_centroids:
                plt.scatter(
                    *centroid, color=label_to_color[label], edgecolor='black', s=200, marker='X', zorder=5)
                plt.text(centroid[0], centroid[1],
                         f"{label} centroid", fontsize=12, weight='bold')
        if show_centroid_line and len(centroids) == 2:
            plt.plot([centroids[0][0], centroids[1][0]], [centroids[0][1], centroids[1][1]],
                     'k--', lw=2, label='Centroid Line')

        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        if legend:
            plt.legend(loc=legend_loc)
        plt.tight_layout()
        plt.show()
        return plt

    def confusion_matrix_heatmap(
        self,
        y_true: list,
        y_pred: list,
        class_labels: list,
        title: str = "Confusion Matrix",
        figsize: tuple = (10, 8),
        cmap: str = 'Blues',
        normalize: bool = True,
        show_counts: bool = True,
        fmt: str = None,
        show_heatmap: bool = True,
    ) -> tuple[dict, sns.heatmap]:
        """
        Plot a confusion matrix as a heatmap.

        Parameters:
            y_true (list):
                True labels
            y_pred (list):
                Predicted labels
            class_labels (list):
                List of class labels
            title (str):
                Plot title
            figsize (tuple):
                Figure size (width, height) in inches
            cmap (str):
                Matplotlib colormap for the heatmap
            normalize (bool):
                Whether to normalize the confusion matrix (default: True)
            show_counts (bool):
                Whether to show raw counts in each cell (default: True)
            fmt (str):
                Format for annotations (default: '.2f' for normalized, 'd' for counts)
            show_heatmap (bool):
                Whether to show the heatmap (default: True)

        Returns:
            plt (matplotlib.pyplot):
                The plot object for further customization
        """
        # Check input lengths
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have the same length. Got {len(y_true)} and {len(y_pred)}.")

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)

        # Calculate per-class prediction accuracy (recall)
        per_class_accuracy = {}
        for idx, label in enumerate(class_labels):
            total = cm[idx, :].sum()
            correct = cm[idx, idx]
            acc = (correct / total) * 100 if total > 0 else 0
            per_class_accuracy[label] = acc

        # Normalize if requested
        if normalize and fmt is None:
            with np.errstate(all='ignore'):
                cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            if fmt is None:
                fmt = '.2f'
        elif fmt is None:
            cm_display = cm
            fmt = 'd'
        else:
            cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = fmt

        # Prepare annotation labels
        if show_counts and normalize:
            annot = np.empty_like(cm).astype(str)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    annot[i, j] = f"{cm_display[i, j]:.2f}\n({cm[i, j]})"
        elif show_counts:
            annot = cm
        else:
            annot = cm_display

        ax = None

        if show_heatmap:
            plt.figure(figsize=figsize)
            ax = sns.heatmap(
                cm_display, annot=annot, fmt=fmt, cmap=cmap,
                xticklabels=class_labels, yticklabels=class_labels,
                cbar=False, square=True, linewidths=.5
            )
            plt.title(title)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.show()

        return per_class_accuracy, ax

    def shap_explain(
        self,
        ramanML: RamanML = None,
        nsamples: int = 50,  # Reduced default for better performance
        show_plots: bool = True,
        max_display: int = 15,  # Reduced default
        wavenumber_axis: np.ndarray = None,
        # Performance optimization parameters
        max_background_samples: int = 50,  # Reduced from 100
        max_test_samples: int = 20,  # Limit test samples for explanation
        reduce_features: bool = True,  # Enable feature reduction for SVC
        max_features: int = 300,  # Maximum features after reduction
        use_kmeans_sampling: bool = False,  # Use K-means for background sampling
        fast_mode: bool = False  # Ultra-fast mode with minimal samples
    ) -> dict:
        """
        Generate SHAP explanations using stored training/test data from previous model training.

        Args:
            nsamples (int):
                Number of samples for SHAP KernelExplainer. Lower = faster.
            show_plots (bool):
                Whether to generate and show SHAP plots.
            max_display (int):
                Maximum number of features to display in plots.
            wavenumber_axis (np.ndarray, optional):
                Wavenumber axis for feature names. If None, uses stored common_axis.
            max_background_samples (int):
                Maximum background samples for SHAP explainer. Lower = faster.
            max_test_samples (int):
                Maximum test samples to explain. Lower = faster.
            reduce_features (bool):
                Whether to use feature reduction for SVC models.
            max_features (int):
                Maximum features after reduction (only for SVC).
            use_kmeans_sampling (bool):
                Use K-means clustering for background sample selection.
            fast_mode (bool):
                Enable ultra-fast mode with minimal samples.

        Returns:
            dict: Comprehensive SHAP analysis results including performance metrics.
        """
        try:
            start_time = time.time()
            ramanML = ramanML if ramanML is not None else self.ramanML
            if ramanML is None:
                raise ValueError(
                    "No RamanML instance provided. Please provide a RamanML instance with trained data.")

            # Check if training data exists
            if ramanML.X_train is None or ramanML.y_train is None or ramanML.X_test is None or ramanML.y_test is None:
                raise ValueError(
                    "No training/test data found. Please run train_svc() or train_rf() first to generate shap data.")
            else:
                X_train = ramanML.X_train
                y_train = ramanML.y_train
                X_test = ramanML.X_test
                y_test = ramanML.y_test

            # Determine which model to use
            if hasattr(ramanML, "_model") and ramanML._model is not None:
                model = ramanML._model
            else:
                raise ValueError(
                    "No trained model found. Train a model first or provide one.")

            # Get wavenumber axis
            if wavenumber_axis is None:
                if hasattr(ramanML, "common_axis") and ramanML.common_axis is not None:
                    wavenumber_axis = ramanML.common_axis
                else:
                    wavenumber_axis = None

            # Use stored training/test data
            background_data = X_train.copy()
            test_data = X_test.copy()
            labels = list(np.unique(y_train))

            print(
                f"Using stored data: {background_data.shape[0]} background samples, {test_data.shape[0]} test samples")

            # Apply fast mode settings
            if fast_mode:
                max_background_samples = min(max_background_samples, 20)
                max_test_samples = min(max_test_samples, 10)
                nsamples = min(nsamples, 20)
                max_features = min(max_features, 200)
                print("Fast mode enabled: Using minimal samples for quick computation")

            # Feature reduction for SVC models (performance optimization)
            feature_selector = None
            if isinstance(model, SVC) and reduce_features and background_data.shape[1] > max_features:

                print(
                    f"Reducing features from {background_data.shape[1]} to {max_features} for faster SHAP computation...")

                # Use training labels for feature selection
                selector = SelectKBest(f_classif, k=max_features)
                background_data = selector.fit_transform(
                    background_data, y_train)
                test_data = selector.transform(test_data)

                # Update wavenumber axis if provided
                if wavenumber_axis is not None:
                    wavenumber_axis = wavenumber_axis[selector.get_support()]

                feature_selector = selector
                print(
                    f"Feature reduction completed: {background_data.shape[1]} features selected")

            # Optimize background data sampling
            if background_data.shape[0] > max_background_samples:
                if use_kmeans_sampling:

                    print(
                        f"Using K-means to select {max_background_samples} representative background samples...")
                    kmeans = KMeans(
                        n_clusters=max_background_samples, random_state=42, n_init=10)
                    kmeans.fit(background_data)
                    background_data = kmeans.cluster_centers_
                else:
                    indices = np.random.choice(
                        background_data.shape[0], max_background_samples, replace=False)
                    background_data = background_data[indices]
                print(
                    f"Background data reduced to {background_data.shape[0]} samples")

            # Limit test data for explanation
            if test_data.shape[0] > max_test_samples:
                print(
                    f"Limiting SHAP explanation to {max_test_samples} samples out of {test_data.shape[0]}")
                test_data = test_data[:max_test_samples]

            print(
                f"Computing SHAP values for {test_data.shape[0]} samples with {background_data.shape[0]} background samples...")

            # Choose SHAP explainer based on model type
            if isinstance(model, RandomForestClassifier):
                print("Using TreeExplainer for RandomForest model...")
                explainer = shap.TreeExplainer(model)

                # TreeExplainer handles feature selection differently
                if feature_selector is not None:
                    # For RF with feature selection, we need to transform the data first
                    shap_values = explainer.shap_values(test_data)
                    expected_value = explainer.expected_value
                else:
                    shap_values = explainer.shap_values(test_data)
                    expected_value = explainer.expected_value
            else:
                # FIXED: Add SVC and other model handling
                print(
                    f"Using KernelExplainer for {type(model).__name__} model...")
                explainer = shap.KernelExplainer(
                    model.predict_proba, background_data, link="logit")
                shap_values = explainer.shap_values(
                    test_data, nsamples=nsamples)
                expected_value = explainer.expected_value

            # Handle RandomForest specific output format
            if isinstance(model, RandomForestClassifier):
                # TreeExplainer returns different formats for binary vs multiclass
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    # Binary classification: TreeExplainer returns [class_0_shap, class_1_shap]
                    n_classes = 2
                    # For binary classification, we typically use the positive class (class 1)
                    # But keep both for completeness
                    mean_shap_values = [
                        np.mean(np.abs(sv), axis=0) for sv in shap_values]
                elif isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 3:
                        # 3D array: (n_samples, n_features, n_classes)
                        n_classes = shap_values.shape[2]
                        shap_values = [shap_values[:, :, i]
                                       for i in range(n_classes)]
                        mean_shap_values = [
                            np.mean(np.abs(sv), axis=0) for sv in shap_values]
                    else:
                        # 2D array: single class or regression
                        n_classes = 1
                        shap_values = [shap_values]
                        mean_shap_values = [
                            np.mean(np.abs(shap_values), axis=0)]
                else:
                    raise ValueError(
                        f"Unexpected shap_values format: {type(shap_values)}")
            else:
                # Handle SVC and other models - IMPROVED
                if isinstance(shap_values, list):
                    n_classes = len(shap_values)
                    # FIXED: Ensure each SHAP value array is properly shaped
                    processed_shap_values = []
                    for sv in shap_values:
                        if hasattr(sv, 'ndim'):
                            if sv.ndim > 2:
                                # If 3D, take the first dimension (samples)
                                sv = sv.reshape(sv.shape[0], -1)
                            processed_shap_values.append(sv)
                        else:
                            processed_shap_values.append(np.asarray(sv))
                    shap_values = processed_shap_values
                    mean_shap_values = [
                        np.mean(np.abs(sv), axis=0) for sv in shap_values]
                else:
                    if hasattr(shap_values, 'values'):
                        shap_values_array = shap_values.values
                        if shap_values_array.ndim == 3:
                            n_classes = shap_values_array.shape[-1]
                            shap_values = [shap_values_array[:, :, i]
                                           for i in range(n_classes)]
                            mean_shap_values = [
                                np.mean(np.abs(sv), axis=0) for sv in shap_values]
                        else:
                            n_classes = 1
                            shap_values = [shap_values_array]
                            mean_shap_values = [
                                np.mean(np.abs(shap_values_array), axis=0)]
                    else:
                        # FIXED: Handle direct numpy array or other formats
                        shap_values_array = np.asarray(shap_values)

                        if shap_values_array.ndim == 3:
                            n_classes = shap_values_array.shape[-1]
                            shap_values = [shap_values_array[:, :, i]
                                           for i in range(n_classes)]
                            mean_shap_values = [
                                np.mean(np.abs(sv), axis=0) for sv in shap_values]
                        elif shap_values_array.ndim == 2:
                            # Could be (n_samples, n_features) for single class
                            n_classes = 1
                            shap_values = [shap_values_array]
                            mean_shap_values = [
                                np.mean(np.abs(shap_values_array), axis=0)]
                        else:
                            # 1D array - single sample, single class
                            n_classes = 1
                            shap_values = [shap_values_array.reshape(1, -1)]
                            mean_shap_values = [np.abs(shap_values_array)]

                        expected_value = [expected_value] if not isinstance(
                            expected_value, list) else expected_value

                # FIXED: Debug the mean_shap_values
                for i, msv in enumerate(mean_shap_values):
                    if hasattr(msv, 'ndim') and msv.ndim > 1:
                        print(
                            f"Warning: mean_shap_values[{i}] has {msv.ndim} dimensions: {msv.shape}")
                        mean_shap_values[i] = msv.flatten()

            # Calculate feature importance
            overall_importance = np.mean(
                [msv for msv in mean_shap_values], axis=0)
            feature_importance_indices = np.argsort(overall_importance)[::-1]

            # Get top features for each class with robust error handling
            top_features = {}
            for class_idx in range(n_classes):
                if n_classes == 1:
                    class_shap = mean_shap_values[0]
                else:
                    class_shap = mean_shap_values[class_idx]

                # Robust index handling
                top_indices = np.argsort(class_shap)[-max_display:][::-1]

                class_features = []
                for rank, idx in enumerate(top_indices):
                    try:
                        # FIXED: Handle numpy scalar and array indexing properly
                        idx_val = int(idx.item() if hasattr(
                            idx, 'item') else idx)

                        # Same robust conversion for importance values
                        importance_val = float(class_shap[idx_val].item() if hasattr(
                            class_shap[idx_val], 'item') else class_shap[idx_val])

                        # Robust wavenumber conversion
                        if wavenumber_axis is not None and idx_val < len(wavenumber_axis):
                            wavenumber_val = float(wavenumber_axis[idx_val].item() if hasattr(
                                wavenumber_axis[idx_val], 'item') else wavenumber_axis[idx_val])
                        else:
                            wavenumber_val = float(idx_val)

                        class_features.append({
                            "rank": int(rank + 1),
                            "feature_index": idx_val,
                            "wavenumber": wavenumber_val,
                            "importance": importance_val
                        })
                    except (ValueError, TypeError, IndexError) as e:
                        print(f"Warning: Error processing feature {idx}: {e}")
                        class_features.append({
                            "rank": int(rank + 1),
                            "feature_index": 0,
                            "wavenumber": 0.0,
                            "importance": 0.0
                        })
                        continue

                top_features[f"class_{class_idx}"] = class_features

            # Generate plots if requested
            plots_dict = {}
            try:
                # Summary plot with proper handling for RF
                try:
                    if isinstance(model, RandomForestClassifier) and n_classes == 2:
                        # For RF binary classification, use the positive class (class 1)
                        shap.summary_plot(shap_values[1], test_data,
                                          feature_names=[
                                              f"Feature_{i}" for i in range(test_data.shape[1])],
                                          max_display=max_display, show=show_plots)
                    elif n_classes == 1:
                        shap.summary_plot(shap_values[0], test_data,
                                          feature_names=[
                                              f"Feature_{i}" for i in range(test_data.shape[1])],
                                          max_display=max_display, show=show_plots)
                    else:
                        # Multi-class: pass the list of arrays
                        shap.summary_plot(shap_values, test_data,
                                          feature_names=[
                                              f"Feature_{i}" for i in range(test_data.shape[1])],
                                          max_display=max_display, show=show_plots)

                    summary_plot = plt.gcf()
                    plots_dict["summary_plot"] = summary_plot
                    if not show_plots:
                        plt.close()
                except Exception as e:
                    print(f"Warning: Could not generate summary plot: {e}")

                # Feature importance plot with proper array handling
                try:
                    if n_classes == 1:
                        importance_data = np.abs(mean_shap_values[0])
                    else:
                        # For multi-class, average the importance across classes
                        importance_data = np.mean(
                            [np.abs(sv) for sv in mean_shap_values], axis=0)

                    # Get top features
                    top_indices = np.argsort(importance_data)[-max_display:]
                    top_importance = importance_data[top_indices]

                    # Create feature names with robust wavenumber handling
                    if wavenumber_axis is not None:
                        try:
                            feature_names = [
                                f"{float(wavenumber_axis[i]):.1f} cm⁻¹" for i in top_indices]
                        except (IndexError, TypeError, ValueError):
                            feature_names = [
                                f"Feature_{i}" for i in top_indices]
                    else:
                        feature_names = [f"Feature_{i}" for i in top_indices]

                    plt.figure(figsize=(10, 6))
                    plt.barh(range(len(top_importance)), top_importance)
                    plt.yticks(range(len(feature_names)), feature_names)
                    plt.xlabel('Mean |SHAP value|')
                    plt.title('Feature Importance (SHAP)')
                    plt.tight_layout()

                    importance_plot = plt.gcf()
                    plots_dict["importance_plot"] = importance_plot
                    if show_plots:
                        plt.show()
                    else:
                        plt.close()
                except Exception as e:
                    print(f"Warning: Could not generate importance plot: {e}")

                # Waterfall plots for first few test samples
                if not fast_mode:
                    waterfall_plots = []
                    n_waterfall = min(3, test_data.shape[0])
                    for i in range(n_waterfall):
                        try:
                            if isinstance(model, RandomForestClassifier) and n_classes == 2:
                                # For RF binary classification, use positive class
                                sample_shap = shap_values[1][i]
                                expected_val = expected_value[1] if isinstance(
                                    expected_value, list) else expected_value

                                # FIXED: More comprehensive numpy array to scalar conversion
                                if isinstance(expected_val, np.ndarray):
                                    if expected_val.size == 1:
                                        expected_val = float(
                                            expected_val.item())
                                    elif expected_val.size > 1:
                                        # Use positive class for binary RF
                                        expected_val = float(expected_val[1])
                                    else:
                                        expected_val = 0.0
                                elif hasattr(expected_val, 'item'):
                                    expected_val = float(expected_val.item())
                                elif isinstance(expected_val, (list, tuple)):
                                    expected_val = float(expected_val[1]) if len(expected_val) > 1 else float(
                                        expected_val[0]) if len(expected_val) > 0 else 0.0
                                else:
                                    expected_val = float(expected_val)

                            elif isinstance(model, SVC) and n_classes == 2:
                                # FIXED: Add SVC binary classification handling
                                # Use positive class
                                sample_shap = shap_values[1][i]
                                expected_val = expected_value[1] if isinstance(
                                    expected_value, list) else expected_value

                                # Convert expected_val to scalar
                                if isinstance(expected_val, np.ndarray):
                                    if expected_val.size == 1:
                                        expected_val = float(
                                            expected_val.item())
                                    elif expected_val.size > 1:
                                        expected_val = float(expected_val[1])
                                    else:
                                        expected_val = 0.0
                                elif hasattr(expected_val, 'item'):
                                    expected_val = float(expected_val.item())
                                elif isinstance(expected_val, (list, tuple)):
                                    expected_val = float(expected_val[1]) if len(expected_val) > 1 else float(
                                        expected_val[0]) if len(expected_val) > 0 else 0.0
                                else:
                                    expected_val = float(expected_val)

                            elif n_classes == 1:
                                sample_shap = shap_values[0][i] if shap_values[0].ndim > 1 else shap_values[0]
                                expected_val = expected_value[0] if isinstance(
                                    expected_value, list) else expected_value

                                # FIXED: More comprehensive numpy array to scalar conversion
                                if isinstance(expected_val, np.ndarray):
                                    if expected_val.size == 1:
                                        expected_val = float(
                                            expected_val.item())
                                    elif expected_val.size > 1:
                                        expected_val = float(expected_val[0])
                                    else:
                                        expected_val = 0.0
                                elif hasattr(expected_val, 'item'):
                                    expected_val = float(expected_val.item())
                                elif isinstance(expected_val, (list, tuple)):
                                    expected_val = float(expected_val[0]) if len(
                                        expected_val) > 0 else 0.0
                                else:
                                    expected_val = float(expected_val)

                            else:
                                # Multi-class: show waterfall for predicted class
                                pred_class = model.predict(test_data[i:i+1])[0]
                                class_idx = list(labels).index(
                                    pred_class) if pred_class in labels else 0
                                sample_shap = shap_values[class_idx][i]
                                expected_val = expected_value[class_idx] if isinstance(
                                    expected_value, list) else expected_value

                                # FIXED: More comprehensive numpy array to scalar conversion
                                if isinstance(expected_val, np.ndarray):
                                    if expected_val.size == 1:
                                        expected_val = float(
                                            expected_val.item())
                                    elif expected_val.size > 1:
                                        expected_val = float(
                                            expected_val[class_idx])
                                    else:
                                        expected_val = 0.0
                                elif hasattr(expected_val, 'item'):
                                    expected_val = float(expected_val.item())
                                elif isinstance(expected_val, (list, tuple)):
                                    expected_val = float(expected_val[class_idx]) if len(
                                        expected_val) > class_idx else 0.0
                                else:
                                    expected_val = float(expected_val)

                            # FIXED: Ensure sample_shap is also properly handled and converted to Python floats
                            if hasattr(sample_shap, 'ndim') and sample_shap.ndim > 1:
                                sample_shap = sample_shap.flatten()

                            # FIXED: Convert to native Python float array - this is crucial!
                            if isinstance(sample_shap, np.ndarray):
                                sample_shap = sample_shap.astype(
                                    float).tolist()
                                sample_shap = np.array(
                                    sample_shap, dtype=float)

                            # FIXED: Also ensure test_data[i] is properly converted
                            test_sample = test_data[i]
                            if isinstance(test_sample, np.ndarray):
                                test_sample = test_sample.astype(float)

                            # FINAL CHECK: Verify all inputs are proper Python/numpy types
                            print(
                                f"Debug - Final types: expected_val={type(expected_val)} (value: {expected_val})")
                            print(
                                f"Debug - sample_shap type: {type(sample_shap)}, shape: {sample_shap.shape if hasattr(sample_shap, 'shape') else 'no shape'}")
                            print(
                                f"Debug - test_sample type: {type(test_sample)}, shape: {test_sample.shape if hasattr(test_sample, 'shape') else 'no shape'}")

                            # Create explanation object for new SHAP API
                            if hasattr(shap, 'Explanation'):
                                try:
                                    exp = shap.Explanation(
                                        values=sample_shap,
                                        base_values=expected_val,
                                        data=test_sample
                                    )
                                    shap.waterfall_plot(exp, show=show_plots)
                                except Exception as exp_error:
                                    print(
                                        f"Warning: SHAP Explanation failed: {exp_error}")
                                    print(
                                        f"Trying alternative force_plot approach...")
                                    # Try alternative approach with force_plot
                                    try:
                                        shap.force_plot(
                                            expected_val, sample_shap, test_sample, show=show_plots, matplotlib=True)
                                    except Exception as force_error:
                                        print(
                                            f"Warning: force_plot also failed: {force_error}")
                                        print("Skipping this waterfall plot...")
                                        continue
                            else:
                                # Fallback for older SHAP versions
                                try:
                                    shap.force_plot(
                                        expected_val, sample_shap, test_sample, show=show_plots, matplotlib=True)
                                except Exception as force_error:
                                    print(
                                        f"Warning: force_plot failed: {force_error}")
                                    print("Skipping this waterfall plot...")
                                    continue

                            waterfall_plots.append(plt.gcf())
                            if not show_plots:
                                plt.close()

                        except Exception as e:
                            print(
                                f"Warning: Could not generate waterfall plot {i+1}: {e}")
                            print(
                                f"Debug info - expected_val type: {type(expected_val)}")
                            if 'sample_shap' in locals():
                                print(
                                    f"Debug info - sample_shap type: {type(sample_shap)}")
                                if hasattr(sample_shap, 'shape'):
                                    print(
                                        f"Debug info - sample_shap shape: {sample_shap.shape}")
                            continue
                    plots_dict["waterfall_plots"] = waterfall_plots

            except Exception as e:
                create_logs("shap_plots", "ML",
                            f"Error generating plots: {e}", status='warning')
                plots_dict = {"plot_error": str(e)}

            # Prepare class-specific explanations with robust conversion
            class_explanations = {}
            for i, label in enumerate(labels[:len(shap_values)]):
                # Fix the problematic list comprehension
                top_features_list = []

                # FIXED: Ensure we're working with the correct class SHAP values
                if i < len(mean_shap_values):
                    class_shap_values = mean_shap_values[i]
                else:
                    print(
                        f"Warning: Class index {i} out of range for mean_shap_values")
                    continue

                # FIXED: Handle potential dimension issues with class_shap_values
                if hasattr(class_shap_values, 'ndim'):
                    if class_shap_values.ndim > 1:
                        # If it's 2D, flatten it or take first row
                        if class_shap_values.shape[0] == 1:
                            class_shap_values = class_shap_values.flatten()
                        else:
                            print(
                                f"Warning: Unexpected class_shap_values shape: {class_shap_values.shape}")
                            class_shap_values = class_shap_values.flatten()

                # Convert to numpy array and ensure it's 1D
                class_shap_values = np.asarray(class_shap_values)
                if class_shap_values.ndim > 1:
                    class_shap_values = class_shap_values.flatten()

                print(
                    f"Debug: class_shap_values shape for {label}: {class_shap_values.shape}")
                print(
                    f"Debug: class_shap_values type: {type(class_shap_values)}")

                # FIXED: Get indices properly - ensure we get scalar indices
                try:
                    # Ensure class_shap_values is a proper 1D array
                    if class_shap_values.size == 0:
                        print(
                            f"Warning: Empty class_shap_values for class {label}")
                        continue

                    # Force conversion to 1D array of floats
                    class_shap_values_1d = np.asarray(
                        class_shap_values, dtype=float).flatten()

                    # Get top feature indices
                    top_feature_indices = np.argsort(class_shap_values_1d)[
                        ::-1][:max_display]

                    # Ensure indices are proper integers
                    top_feature_indices = np.asarray(
                        top_feature_indices, dtype=int).flatten()

                except Exception as e:
                    print(
                        f"Error in argsort for class {label}: {e} \n {traceback.format_exc()}")
                    continue

                for rank, idx in enumerate(top_feature_indices):
                    try:
                        # FIXED: Convert to standard Python int
                        idx_val = int(idx.item()) if hasattr(
                            idx, 'item') else int(idx)

                        # Validate index
                        if idx_val < 0 or idx_val >= len(class_shap_values_1d):
                            print(
                                f"Warning: Invalid index {idx_val} for class_shap_values length {len(class_shap_values_1d)}")
                            continue

                        # FIXED: Safe access to importance value
                        importance_val = float(class_shap_values_1d[idx_val])

                        # FIXED: Safe access to wavenumber
                        if wavenumber_axis is not None and idx_val < len(wavenumber_axis):
                            try:
                                wavenumber_val = float(
                                    wavenumber_axis[idx_val])
                                if hasattr(wavenumber_val, 'item'):
                                    wavenumber_val = wavenumber_val.item()
                            except (IndexError, TypeError):
                                wavenumber_val = float(idx_val)
                        else:
                            wavenumber_val = float(idx_val)

                        top_features_list.append({
                            "feature_index": idx_val,
                            "wavenumber": wavenumber_val,
                            "importance": importance_val
                        })
                    except (ValueError, TypeError, IndexError) as e:
                        print(
                            f"Warning: Error processing top feature {idx} for class {label}: {e}")
                        continue

                # FIXED: Safe access to SHAP values and expected values
                try:
                    if i < len(shap_values):
                        class_shap_vals = shap_values[i]
                    else:
                        class_shap_vals = shap_values[0] if len(
                            shap_values) > 0 else []

                    if isinstance(expected_value, list) and i < len(expected_value):
                        exp_val = expected_value[i]
                    elif isinstance(expected_value, np.ndarray) and i < len(expected_value):
                        exp_val = expected_value[i]
                    else:
                        exp_val = expected_value if not isinstance(expected_value, (list, np.ndarray)) else (
                            expected_value[0] if len(expected_value) > 0 else 0)

                    class_explanations[label] = {
                        "shap_values": class_shap_vals,
                        "expected_value": exp_val,
                        "mean_abs_shap": class_shap_values_1d,  # Use the cleaned 1D array
                        "top_features": top_features_list
                    }
                except Exception as e:
                    print(
                        f"Warning: Error creating class explanation for {label}: {e}")
                    class_explanations[label] = {
                        "shap_values": [],
                        "expected_value": 0,
                        "mean_abs_shap": [],
                        "top_features": top_features_list
                    }

            processing_time = time.time() - start_time
            return {
                "success": True,
                "msg": "shap_explain_success",
                "shap_values": shap_values,
                "expected_value": expected_value,
                "explainer": explainer,
                "feature_importance": overall_importance,
                "feature_importance_ranking": feature_importance_indices,
                "mean_shap_values": mean_shap_values,
                "top_features": top_features,
                "class_explanations": class_explanations,
                "n_classes": n_classes,
                "n_features": int(test_data.shape[1]),
                "background_size": int(background_data.shape[0]),
                "test_size": int(test_data.shape[0]),
                "processing_time": float(processing_time),
                "wavenumber_axis": wavenumber_axis.tolist() if wavenumber_axis is not None else None,
                "model_type": type(model).__name__,
                "class_labels": labels,
                # Performance metrics
                "optimization_settings": {
                    "max_background_samples": max_background_samples,
                    "max_test_samples": max_test_samples,
                    "nsamples": nsamples,
                    "feature_reduction_used": feature_selector is not None,
                    # Fixed: was self.X_train
                    "features_before_reduction": X_train.shape[1] if feature_selector is not None else test_data.shape[1],
                    "features_after_reduction": test_data.shape[1],
                    "kmeans_sampling": use_kmeans_sampling,
                    "fast_mode": fast_mode
                },
                **plots_dict
            }

        except Exception as e:
            create_logs("shap_explain", "ML",
                        f"Error in SHAP explanation: {e} \n {traceback.format_exc()}", status='error')
            return {
                "success": False,
                "msg": "shap_explain_error",
                "detail": f"{e} \n {traceback.format_exc()}",
            }

    def inspect_random_spectra(
        self,
        test_spectra: list[rp.SpectralContainer] = None,
        true_labels: list[str] = None,
        n_samples: int = 5,
        seed: int = None,
        class_to_inspect: str = None,
        show_plots: bool = True,
        max_display: int = 10,
        nsamples: int = 100,
        fast_mode: bool = False,
        show_shap_plots: bool = False,
        positive_label: str = "cancer",
        negative_label: str = "benign"
    ) -> dict:
        """
        Randomly select spectra and show detailed SHAP explanations for why they were
        classified as positive_label or negative_label.

        Parameters:
        -----------
        test_spectra : list, optional
            List of SpectralContainer objects. If None, uses stored test data.
        true_labels : list, optional
            True labels for the spectra. If None, uses stored test labels.
        n_samples : int
            Number of random spectra to inspect
        seed : int, optional
            Random seed for reproducibility
        class_to_inspect : str, optional
            If provided, only inspect spectra from this class ("benign" or "cancer")
        show_plots : bool
            Whether to show plots
        max_display : int
            Maximum number of features to display in SHAP plots
        nsamples : int
            Number of samples for SHAP computation
        fast_mode : bool
            Enable fast mode for SHAP computation
        show_shap_plots : bool
            Whether to show SHAP plots for each inspected spectrum
        positive_label : str
            Label for positive class (default: "cancer")
        negative_label : str
            Label for negative class (default: "benign")

        Returns:
        --------
        dict
            Dictionary of inspection results
        """
        if self.ramanML is None:
            raise ValueError(
                "RamanML instance is required for spectrum inspection.")

        # Use provided data or fall back to stored data
        if test_spectra is None and true_labels is None:
            # Use stored test data from ramanML
            if (self.ramanML.X_test is None or self.ramanML.y_test is None or
                    self.ramanML.common_axis is None):
                raise ValueError(
                    "No test data provided and no stored test data found. "
                    "Please provide test_spectra and true_labels or train a model first."
                )

            # Convert stored test data back to spectral format for analysis
            X_test = self.ramanML.X_test
            y_test = self.ramanML.y_test
            common_axis = self.ramanML.common_axis

            # Create artificial spectral containers from stored data
            all_spectra = []
            for idx, (spectrum, label) in enumerate(zip(X_test, y_test)):
                all_spectra.append({
                    "container_idx": 0,
                    "spectrum_idx": idx,
                    "true_label": label,
                    "spectrum": spectrum,
                    "spectral_axis": common_axis
                })
        else:
            if test_spectra is None or true_labels is None:
                raise ValueError(
                    "Both test_spectra and true_labels must be provided together.")

            # Flatten test spectra into a list of individual spectra with indices
            all_spectra = []
            label_idx = 0

            for container_idx, container in enumerate(test_spectra):
                for spectrum_idx, spectrum in enumerate(container.spectral_data):
                    if label_idx < len(true_labels):
                        all_spectra.append({
                            "container_idx": container_idx,
                            "spectrum_idx": spectrum_idx,
                            "true_label": true_labels[label_idx],
                            "spectrum": spectrum,
                            "spectral_axis": container.spectral_axis
                        })
                        label_idx += 1

        # Filter by class if requested
        if class_to_inspect:
            filtered_spectra = [
                s for s in all_spectra if s["true_label"] == class_to_inspect]
            if not filtered_spectra:
                print(f"No spectra found with class {class_to_inspect}")
                return {"results": [], "summary": f"No spectra found for class: {class_to_inspect}"}
            all_spectra = filtered_spectra

        # Select random spectra
        rng = np.random.RandomState(seed) if seed is not None else np.random
        selected_indices = rng.choice(len(all_spectra), min(
            n_samples, len(all_spectra)), replace=False)
        selected_spectra = [all_spectra[idx] for idx in selected_indices]

        print(f"Inspecting {len(selected_spectra)} random spectra...")

        results = []

        # FIXED: Changed variable name
        for spectrum_idx, spec_info in enumerate(selected_spectra):
            print(
                f"\nAnalyzing spectrum {spectrum_idx+1}/{len(selected_spectra)}...")

            spectrum = spec_info["spectrum"]

            # Interpolate to common axis if needed
            if len(spec_info["spectral_axis"]) != len(self.ramanML.common_axis):
                spectrum = np.interp(self.ramanML.common_axis,
                                     spec_info["spectral_axis"], spectrum)

            # Get prediction using MLModel predict method
            try:
                pred_result = self.ramanML.predict(
                    [rp.SpectralContainer(np.array([spectrum]), self.ramanML.common_axis)])
                pred_label = pred_result["y_pred"][0]
                confidence = pred_result.get("confidences", [None])[0]
            except Exception as e:
                print(
                    f"Error in prediction for spectrum {spectrum_idx+1}: {e}")
                continue

            # Get SHAP explanation for this individual spectrum
            try:
                # FIXED: Instead of modifying ramanML data, call shap_explain with custom parameters
                # that disable feature reduction to avoid dimension mismatch

                # Create a temporary single-spectrum dataset
                single_spectrum_data = np.array([spectrum]).reshape(1, -1)

                # Store original test data
                original_X_test = self.ramanML.X_test.copy(
                ) if self.ramanML.X_test is not None else None
                original_y_test = self.ramanML.y_test.copy(
                ) if self.ramanML.y_test is not None else None

                # Temporarily replace with single spectrum (ensuring same dimensions)
                self.ramanML.X_test = single_spectrum_data
                self.ramanML.y_test = np.array([spec_info["true_label"]])

                # Get SHAP explanation with reduced feature reduction to avoid dimension issues
                shap_result = self.shap_explain(
                    show_plots=show_shap_plots,  # We'll create custom plots
                    max_display=max_display,
                    nsamples=nsamples,
                    fast_mode=fast_mode,
                    max_test_samples=1,
                    max_background_samples=30,
                    reduce_features=False,  # FIXED: Disable feature reduction to avoid dimension mismatch
                    # Use full feature set
                    max_features=single_spectrum_data.shape[1]
                )

                # Restore original test data
                if original_X_test is not None:
                    self.ramanML.X_test = original_X_test
                if original_y_test is not None:
                    self.ramanML.y_test = original_y_test

            except Exception as e:
                print(
                    f"Error in SHAP explanation for spectrum {spectrum_idx+1}: {e}")
                # Restore original test data even on error
                if 'original_X_test' in locals() and original_X_test is not None:
                    self.ramanML.X_test = original_X_test
                if 'original_y_test' in locals() and original_y_test is not None:
                    self.ramanML.y_test = original_y_test
                continue

            # Process SHAP results and create visualizations
            if shap_result["success"]:
                # FIXED: More robust SHAP value extraction for SVC
                shap_values_raw = shap_result["shap_values"]

                # Extract SHAP values based on model type and format - Handle SVC specifically
                if hasattr(self.ramanML, '_model') and isinstance(self.ramanML._model, SVC):
                    print("Debug: Using SVC-specific SHAP value extraction")
                    # For SVC, we need to be extra careful with the SHAP values format
                    if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                        # Binary classification with SVC - use positive class (index 1)
                        sv_positive = shap_values_raw[1]
                        if isinstance(sv_positive, np.ndarray):
                            if sv_positive.ndim == 1:
                                # Already 1D array
                                shap_values = sv_positive
                            elif sv_positive.ndim == 2 and sv_positive.shape[0] == 1:
                                # 2D array with single sample - flatten
                                shap_values = sv_positive[0]
                            else:
                                # More complex structure - take first sample to be safe
                                shap_values = sv_positive[0] if sv_positive.shape[0] > 0 else sv_positive
                        else:
                            # Convert to numpy array
                            shap_values = np.asarray(sv_positive)
                    else:
                        # Fallback for unexpected format
                        print(
                            "Warning: Unexpected SHAP values format for SVC model")
                        if isinstance(shap_values_raw, np.ndarray):
                            shap_values = shap_values_raw.flatten()
                        elif hasattr(shap_values_raw, 'values'):
                            shap_values = np.asarray(
                                shap_values_raw.values).flatten()
                        else:
                            shap_values = np.asarray(shap_values_raw[0] if isinstance(
                                shap_values_raw, list) and len(shap_values_raw) > 0 else shap_values_raw)
                else:
                    # Standard extraction for other model types
                    if isinstance(shap_values_raw, list) and len(shap_values_raw) > 0:
                        # For binary SVC, we typically want the positive class (index 1)
                        if len(shap_values_raw) == 2:
                            # Binary classification - use positive class
                            if isinstance(shap_values_raw[1], np.ndarray):
                                if shap_values_raw[1].ndim > 1:
                                    # Take first sample
                                    shap_values = shap_values_raw[1][0]
                                else:
                                    shap_values = shap_values_raw[1]
                            else:
                                shap_values = shap_values_raw[1]
                        else:
                            # Multi-class or single class
                            if isinstance(shap_values_raw[0], np.ndarray):
                                if shap_values_raw[0].ndim > 1:
                                    # Take first sample
                                    shap_values = shap_values_raw[0][0]
                                else:
                                    shap_values = shap_values_raw[0]
                            else:
                                shap_values = shap_values_raw[0]
                    else:
                        # Handle newer SHAP format or single array
                        if hasattr(shap_values_raw, 'values'):
                            # New SHAP Explanation object
                            shap_values_array = shap_values_raw.values
                            if shap_values_array.ndim == 3:
                                # (n_samples, n_features, n_classes) - take positive class for binary
                                if shap_values_array.shape[2] == 2:
                                    # First sample, positive class
                                    shap_values = shap_values_array[0, :, 1]
                                else:
                                    # First sample, first class
                                    shap_values = shap_values_array[0, :, 0]
                            elif shap_values_array.ndim == 2:
                                # First sample
                                shap_values = shap_values_array[0]
                            else:
                                shap_values = shap_values_array
                        elif hasattr(shap_values_raw, '__getitem__'):
                            shap_values = shap_values_raw[0]
                        else:
                            shap_values = shap_values_raw

                # Ensure shap_values is 1D array and proper numpy array
                if hasattr(shap_values, 'ndim') and shap_values.ndim > 1:
                    shap_values = shap_values.flatten()

                # Convert to numpy array if it's not already
                if not isinstance(shap_values, np.ndarray):
                    shap_values = np.asarray(shap_values)

                # Ensure we have float dtype to avoid conversion issues
                shap_values = shap_values.astype(float)

                # Ensure we have the right length
                # FIXED: Add proper type checking for common_axis
                if hasattr(self.ramanML, 'common_axis') and self.ramanML.common_axis is not None:
                    try:
                        if isinstance(self.ramanML.common_axis, (np.ndarray, list, tuple)):
                            common_axis_length = len(self.ramanML.common_axis)
                            common_axis_subset = np.asarray(
                                self.ramanML.common_axis, dtype=float)
                        elif np.isscalar(self.ramanML.common_axis):
                            # If it's a scalar, we can't use it for length comparison
                            print(
                                f"Warning: common_axis is a scalar value: {self.ramanML.common_axis}")
                            # Use the length of shap_values as fallback
                            common_axis_length = len(shap_values)
                            # Create a dummy axis
                            common_axis_subset = np.arange(
                                len(shap_values), dtype=float)
                        else:
                            print(
                                f"Warning: Unexpected common_axis type: {type(self.ramanML.common_axis)}")
                            common_axis_length = len(shap_values)
                            common_axis_subset = np.arange(
                                len(shap_values), dtype=float)
                    except (TypeError, ValueError) as e:
                        print(f"Warning: Error accessing common_axis: {e}")
                        common_axis_length = len(shap_values)
                        common_axis_subset = np.arange(
                            len(shap_values), dtype=float)
                else:
                    print("Warning: common_axis is None or doesn't exist")
                    common_axis_length = len(shap_values)
                    common_axis_subset = np.arange(
                        len(shap_values), dtype=float)

                # Ensure common_axis_subset is always a numpy array with float dtype
                if not isinstance(common_axis_subset, np.ndarray):
                    common_axis_subset = np.asarray(
                        common_axis_subset, dtype=float)
                else:
                    common_axis_subset = common_axis_subset.astype(float)

                if len(shap_values) != common_axis_length:
                    print(
                        f"Warning: SHAP values length ({len(shap_values)}) doesn't match common axis length ({common_axis_length})")
                    # Take only the length we need
                    min_length = min(len(shap_values), common_axis_length)
                    shap_values = shap_values[:min_length]

                    # Only slice if we have a proper array
                    if len(common_axis_subset) >= min_length:
                        common_axis_subset = common_axis_subset[:min_length]
                    else:
                        common_axis_subset = np.arange(min_length, dtype=float)

                # Get top contributing wavenumbers
                sorted_idx = np.argsort(shap_values)
                # Most negative (toward negative_label)
                top_negative_idx = sorted_idx[:5]
                # Most positive (toward positive_label)
                top_positive_idx = sorted_idx[-5:][::-1]

                # FIXED: Convert to Python lists first to avoid scalar conversion issues
                def safe_extract(arr, idx):
                    try:
                        val = arr[idx]
                        if hasattr(val, 'item'):
                            return val.item()
                        elif isinstance(val, (np.ndarray, list)):
                            return float(val[0]) if len(val) > 0 else 0.0
                        else:
                            return float(val)
                    except:
                        return 0.0

                try:
                    top_negative = [(safe_extract(common_axis_subset, idx),
                                     safe_extract(shap_values, idx))
                                    for idx in top_negative_idx]

                    top_positive = [(safe_extract(common_axis_subset, idx),
                                     safe_extract(shap_values, idx))
                                    for idx in top_positive_idx]
                except Exception as e:
                    print(
                        f"Error with safe extraction method: {e} \n {traceback.format_exc()}")
                    raise Exception(
                        f"Failed to extract top contributors for spectrum {spectrum_idx+1}: {e}")

                # Create visualizations if requested
                if show_plots:
                    # Figure 1: Original spectrum with highlighted regions
                    fig1 = plt.figure(figsize=(12, 6))
                    plt.plot(common_axis_subset, spectrum[:len(common_axis_subset)],
                             'k-', linewidth=2, label='Spectrum')

                    # Highlight top positive and negative regions
                    for wn, sv in top_positive:
                        plt.axvline(x=wn, color='red', alpha=0.6,
                                    linestyle='--', linewidth=1.5, label=f'{positive_label} Contributors' if wn == top_positive[0][0] else "")

                    for wn, sv in top_negative:
                        plt.axvline(x=wn, color='blue', alpha=0.6,
                                    linestyle='--', linewidth=1.5, label=f'{negative_label} Contributors' if wn == top_negative[0][0] else "")

                    # Enhanced title with spectrum index information
                    title_text = (f"Spectrum {spectrum_idx+1} [Container: {spec_info['container_idx']}, "
                                  f"Index: {spec_info['spectrum_idx']}] - "
                                  f"True: {spec_info['true_label']}, Predicted: {pred_label} "
                                  f"(Confidence: {confidence:.3f})")

                    plt.title(title_text, fontsize=12, fontweight='bold')
                    plt.xlabel('Raman Shift (cm⁻¹)')
                    plt.ylabel("Intensity")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()

                    # Figure 2: SHAP values plot with wavenumbers on x-axis
                    fig2 = plt.figure(figsize=(12, 6))
                    colors = ['red' if sv >
                              0 else 'blue' for sv in shap_values]

                    # Plot against wavenumbers instead of feature indices
                    plt.bar(common_axis_subset, shap_values,
                            color=colors, alpha=0.6, width=1.0)

                    # Add color legend
                    red_patch = plt.Rectangle(
                        (0, 0), 1, 1, facecolor='red', alpha=0.6, label=f'{positive_label} (Positive)')
                    blue_patch = plt.Rectangle(
                        (0, 0), 1, 1, facecolor='blue', alpha=0.6, label=f'{negative_label} (Negative)')
                    plt.legend(handles=[red_patch, blue_patch])

                    plt.title(f"SHAP Values - Spectrum {spectrum_idx+1} [Container: {spec_info['container_idx']}, Index: {spec_info['spectrum_idx']}]",
                              fontsize=12, fontweight='bold')
                    plt.ylabel("SHAP Value")
                    plt.xlabel("Raman Shift (cm⁻¹)")

                    # Set y-axis limits based on SHAP values range
                    shap_min = np.min(shap_values)
                    shap_max = np.max(shap_values)
                    shap_range = shap_max - shap_min
                    padding = shap_range * 0.05  # 5% padding
                    plt.ylim(shap_min - padding, shap_max + padding)

                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()

                    # Figure 3: Enhanced Top contributors table with peak assignments
                    fig3 = plt.figure(figsize=(14, 12))
                    plt.axis('off')

                    # Calculate prediction mathematics
                    total_shap_sum = np.sum(shap_values)
                    expected_val = shap_result.get("expected_value", 0)
                    if isinstance(expected_val, (list, np.ndarray)):
                        expected_val = float(expected_val[0]) if len(
                            expected_val) > 0 else 0.0
                    else:
                        expected_val = float(expected_val)

                    prediction_score = expected_val + total_shap_sum

                    # Calculate contribution percentages
                    positive_sum = sum(sv for _, sv in top_positive)
                    negative_sum = sum(sv for _, sv in top_negative)
                    total_abs_contribution = abs(
                        positive_sum) + abs(negative_sum)

                    # Get peak assignments for top contributors
                    def get_peak_info(wavenumber):
                        """Helper function to get peak assignment with fallback"""
                        try:
                            assignment_info = self.get_peak_assignment(
                                wavenumber, pick_near=True, tolerance=10)
                            if assignment_info.get("assignment") == "Not Found":
                                # Try with larger tolerance
                                assignment_info = self.get_peak_assignment(
                                    wavenumber, pick_near=True, tolerance=20)
                            return assignment_info
                        except:
                            return {"assignment": "Error in lookup", "peak": wavenumber}

                    # Create enhanced table data with 4 columns
                    table_data = []

                    # Header section with mathematical explanation
                    table_data.append(
                        ["MATHEMATICAL PREDICTION BREAKDOWN", "", "", ""])
                    table_data.append(
                        ["Base Model Expectation", f"{expected_val:.4f}", "", ""])
                    table_data.append(
                        ["Total SHAP Contribution", f"{total_shap_sum:.4f}", "", ""])
                    table_data.append(
                        ["Final Prediction Score", f"{prediction_score:.4f}", "", ""])
                    table_data.append(
                        ["Predicted Class", pred_label, f"({confidence:.3f})", ""])
                    table_data.append(["", "", "", ""])  # Empty row

                    # positive_label contributors section with peak assignments
                    table_data.append(
                        [f"Top {positive_label} Contributors", "Wavenumber", "SHAP Value", "Peak Assignment"])

                    for idx, (wn, sv) in enumerate(top_positive):
                        percentage = (abs(sv) / total_abs_contribution *
                                      100) if total_abs_contribution > 0 else 0

                        # Get peak assignment
                        peak_info = get_peak_info(wn)
                        assignment = peak_info.get("assignment", "Unknown")

                        # Truncate assignment if too long
                        if len(assignment) > 50:
                            assignment = assignment[:47] + "..."

                        # Add distance info if found nearby peak
                        if "distance" in peak_info and peak_info["distance"] > 0:
                            wn_display = f"{wn:.1f} cm⁻¹ (~{peak_info['peak']} cm⁻¹)"
                        else:
                            wn_display = f"{wn:.1f} cm⁻¹"

                        table_data.append([
                            f"#{idx+1}",
                            wn_display,
                            f"{sv:.4f} ({percentage:.1f}%)",
                            assignment
                        ])

                    # Summary row for positive contributors
                    table_data.append([
                        f"{positive_label} Total:",
                        f"Net Push: +{positive_sum:.4f}",
                        f"({abs(positive_sum)/total_abs_contribution*100:.1f}%)",
                        ""
                    ])
                    table_data.append(["", "", "", ""])  # Empty row

                    # negative_label contributors section with peak assignments
                    table_data.append(
                        [f"Top {negative_label} Contributors", "Wavenumber", "SHAP Value", "Peak Assignment"])

                    for idx, (wn, sv) in enumerate(top_negative):
                        percentage = (abs(sv) / total_abs_contribution *
                                      100) if total_abs_contribution > 0 else 0

                        # Get peak assignment
                        peak_info = get_peak_info(wn)
                        assignment = peak_info.get("assignment", "Unknown")

                        # Truncate assignment if too long
                        if len(assignment) > 50:
                            assignment = assignment[:47] + "..."

                        # Add distance info if found nearby peak
                        if "distance" in peak_info and peak_info["distance"] > 0:
                            wn_display = f"{wn:.1f} cm⁻¹ (~{peak_info['peak']} cm⁻¹)"
                        else:
                            wn_display = f"{wn:.1f} cm⁻¹"

                        table_data.append([
                            f"#{idx+1}",
                            wn_display,
                            f"{sv:.4f} ({percentage:.1f}%)",
                            assignment
                        ])

                    # Summary row for negative contributors
                    table_data.append([
                        f"{negative_label} Total:",
                        f"Net Push: {negative_sum:.4f}",
                        f"({abs(negative_sum)/total_abs_contribution*100:.1f}%)",
                        ""
                    ])
                    table_data.append(["", "", "", ""])  # Empty row

                    # Enhanced decision logic with peak context
                    net_direction = positive_label if total_shap_sum > 0 else negative_label
                    confidence_explanation = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"

                    # Get the most influential peak
                    most_influential_wn = top_positive[0][0] if abs(
                        top_positive[0][1]) > abs(top_negative[0][1]) else top_negative[0][0]
                    most_influential_sv = top_positive[0][1] if abs(
                        top_positive[0][1]) > abs(top_negative[0][1]) else top_negative[0][1]
                    most_influential_info = get_peak_info(most_influential_wn)
                    most_influential_assignment = most_influential_info.get(
                        "assignment", "Unknown")

                    # Truncate if too long
                    if len(most_influential_assignment) > 60:
                        most_influential_assignment = most_influential_assignment[:57] + "..."

                    table_data.append(
                        ["DECISION LOGIC & PEAK ANALYSIS", "", "", ""])
                    table_data.append(
                        ["Net SHAP Direction", f"→ {net_direction}", f"(Δ={total_shap_sum:.4f})", ""])
                    table_data.append(
                        ["Confidence Level", confidence_explanation, f"({confidence:.3f})", ""])
                    table_data.append([
                        "Key Decision Factor",
                        f"{most_influential_wn:.1f} cm⁻¹",
                        f"SHAP: {most_influential_sv:.4f}",
                        most_influential_assignment
                    ])

                    # Add biological interpretation if available
                    biological_context = []
                    for wn, sv in (top_positive + top_negative):
                        peak_info = get_peak_info(wn)
                        assignment = peak_info.get("assignment", "")
                        if assignment and assignment not in ["Not Found", "Unknown", "Error in lookup"]:
                            # Look for biological keywords
                            bio_keywords = ["protein", "lipid", "nucleic", "collagen", "DNA", "RNA",
                                            "amino", "carbohydrate", "phosphate", "membrane"]
                            if any(keyword.lower() in assignment.lower() for keyword in bio_keywords):
                                biological_context.append(
                                    (wn, assignment[:40] + "..." if len(assignment) > 40 else assignment))

                    if biological_context:
                        table_data.append(["", "", "", ""])  # Empty row
                        table_data.append(["BIOLOGICAL CONTEXT", "", "", ""])
                        # Show top 3
                        for wn, bio_info in biological_context[:3]:
                            table_data.append([
                                "Biomolecular Signal",
                                f"{wn:.1f} cm⁻¹",
                                "",
                                bio_info
                            ])

                    # Create table with adjusted column widths for 4 columns
                    table = plt.table(cellText=table_data,
                                      cellLoc='left',  # Changed to left align for better readability
                                      loc='center',
                                      colWidths=[0.25, 0.20, 0.20, 0.35])  # Adjusted widths for 4 columns
                    table.auto_set_font_size(False)
                    # Slightly smaller font to fit more content
                    table.set_fontsize(9)
                    table.scale(1, 1.8)

                    # Enhanced styling for different sections
                    # FIXED: Changed variable name from i to table_row_idx
                    for table_row_idx, row in enumerate(table_data):
                        for j in range(4):  # Now 4 columns
                            cell = table[(table_row_idx, j)]

                            # Header sections
                            if "BREAKDOWN" in str(row[0]) or "Contributors" in str(row[0]) or "DECISION LOGIC" in str(row[0]) or "BIOLOGICAL CONTEXT" in str(row[0]):
                                cell.set_facecolor('#E8E8E8')
                                cell.set_text_props(
                                    weight='bold', color='black')
                                cell.set_height(0.08)

                            # Positive contributors
                            elif positive_label in str(row[0]) and "Total" not in str(row[0]):
                                cell.set_facecolor('#FFE6E6')
                                cell.set_text_props(color='#8B0000')

                            # Negative contributors
                            elif negative_label in str(row[0]) and "Total" not in str(row[0]):
                                cell.set_facecolor('#E6F3FF')
                                cell.set_text_props(color='#000080')

                            # Individual contributor rows
                            elif str(row[0]).startswith('#'):
                                if any(positive_label in prev_row[0] for prev_row in table_data[max(0, table_row_idx-5):table_row_idx] if prev_row):
                                    cell.set_facecolor('#FFF0F0')
                                else:
                                    cell.set_facecolor('#F0F8FF')

                            # Summary rows
                            elif "Total:" in str(row[0]) or "Net Push:" in str(row[1]):
                                cell.set_facecolor('#F5F5F5')
                                cell.set_text_props(weight='bold')

                            # Peak assignment column styling
                            if j == 3 and str(row[j]) and str(row[j]) not in ["", "Peak Assignment"]:
                                cell.set_text_props(style='italic', size=8)

                            # Mathematical values
                            if j in [1, 2] and any(char in str(row[j]) for char in ['±', '(', ':', 'Δ']):
                                cell.set_text_props(family='monospace', size=8)

                    # Use spectrum_idx instead of any loop variable
                    plt.suptitle(f"ENHANCED PREDICTION ANALYSIS WITH PEAK ASSIGNMENTS\nSpectrum {spectrum_idx+1} [Container: {spec_info['container_idx']}, Index: {spec_info['spectrum_idx']}]",
                                 fontsize=14, fontweight='bold', y=0.98)
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.show()

                # Store results with proper type conversion
                spec_result = {
                    "spectrum_info": spec_info,
                    "prediction": {
                        "label": pred_label,
                        "confidence": float(confidence) if confidence is not None else None,
                        "is_correct": pred_label == spec_info["true_label"]
                    },
                    "explanation": {
                        "top_positive_contributors": top_positive,
                        "top_negative_contributors": top_negative,
                        "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else list(shap_values)
                    }
                }
                results.append(spec_result)

                # Print summary for this spectrum - FIXED: Use spectrum_idx instead of i
                print(
                    f"Spectrum {spectrum_idx+1} Summary [Container: {spec_info['container_idx']}, Index: {spec_info['spectrum_idx']}]:")
                print(f"  True Label: {spec_info['true_label']}")
                print(
                    f"  Predicted: {pred_label} (Confidence: {confidence:.3f})")
                print(
                    f"  Correct: {'Yes' if pred_label == spec_info['true_label'] else 'No'}")
                print(
                    f"  Top {positive_label} contributor: {top_positive[0][0]:.1f} cm⁻¹ (SHAP: {top_positive[0][1]:.4f})")
                print(
                    f"  Top {negative_label} contributor: {top_negative[0][0]:.1f} cm⁻¹ (SHAP: {top_negative[0][1]:.4f})")

        # Create summary statistics
        if results:
            correct_predictions = sum(
                1 for r in results if r["prediction"]["is_correct"])
            accuracy = correct_predictions / len(results)

            summary = {
                "total_spectra_analyzed": len(results),
                "correct_predictions": correct_predictions,
                "accuracy": accuracy,
                "class_filter": class_to_inspect,
                "random_seed": seed
            }
        else:
            summary = {
                "total_spectra_analyzed": 0,
                "error": "No spectra could be analyzed successfully"
            }

        return {
            "results": results,
            "summary": summary
        }
