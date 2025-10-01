import os
from typing import Dict, Any, List, Tuple, Optional
from functions.ML import RamanML, MLModel
from functions.configs import *
from numpy import trapz

# Import extracted modules (Phase 1, 2 & 3 refactoring)
from . import peak_assignment
from . import basic_plots
from . import model_evaluation
from . import ml_visualization
from . import explainability

from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import lime
import lime.lime_tabular
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
                 ML_PROPERTY: Union[RamanML, MLModel] = None,
                 ):
        """



        """

        self.ML_PROPERTY = ML_PROPERTY
        self.df = df
        self.spectral_container = spectral_container
        self.labels = labels
        if self.ML_PROPERTY is not None:
            self.common_axis = self.ML_PROPERTY.common_axis
            self.n_features = self.ML_PROPERTY.n_features_in
        else:
            self.common_axis = common_axis
            self.n_features = n_features

    def get_peak_assignment(
        self,
        peak: Union[int, float, str],
        pick_near: bool = False,
        tolerance: int = 5,
        json_file_path: str = "assets/data/raman_peaks.json"
    ) -> dict:
        """
        Get the assignment meaning of a Raman peak based on the raman_peaks.json database.
        
        **NOTE**: This method now delegates to the peak_assignment module.

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
            Path to the raman_peaks.json file. Default is "assets/data/raman_peaks.json".

        Returns:
        --------
        dict
            Dictionary containing peak assignment information:
            - If found: {"peak": peak, "assignment": assignment, "reference_number": ref_num}
            - If not found and pick_near=False: {"assignment": "Not Found"}
            - If not found and pick_near=True: nearest match or "Not Found"
        """
        # Delegate to extracted module
        return peak_assignment.get_peak_assignment(peak, pick_near, tolerance, json_file_path)

    def get_multiple_peak_assignments(
        self,
        peaks: List[Union[int, float, str]],
        pick_near: bool = False,
        tolerance: int = 5,
        json_file_path: str = "assets/data/raman_peaks.json"
    ) -> List[dict]:
        """
        Get assignments for multiple peaks at once.
        
        **NOTE**: This method now delegates to the peak_assignment module.

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
        # Delegate to extracted module
        return peak_assignment.get_multiple_peak_assignments(peaks, pick_near, tolerance, json_file_path)

    def find_peaks_in_range(
        self,
        min_wavenumber: Union[int, float],
        max_wavenumber: Union[int, float],
        json_file_path: str = "assets/data/raman_peaks.json"
    ) -> List[dict]:
        """
        Find all peaks within a specified wavenumber range.
        
        **NOTE**: This method now delegates to the peak_assignment module.

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
        # Delegate to extracted module
        return peak_assignment.find_peaks_in_range(min_wavenumber, max_wavenumber, json_file_path)

    def visualize_raman_spectra(self, wavenumber_colname: str = "wavenumber", title="Raman Spectra", figsize=(12, 6),
                                xlim=None, ylim=None, legend=True,
                                legend_loc='best', sample_limit=10) -> plt:
        """
        Visualize the Raman spectra data.
        
        **NOTE**: This method now delegates to the basic_plots module.

        Parameters:
        -----------
        wavenumber_colname : str
            Name of the wavenumber column
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

        Returns:
        --------
        plt : matplotlib.pyplot
            The plot object for further customization
        """
        # Delegate to extracted module
        return basic_plots.visualize_raman_spectra(
            df=self.df,
            wavenumber_colname=wavenumber_colname,
            title=title,
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
            legend=legend,
            legend_loc=legend_loc,
            sample_limit=sample_limit
        )

    def visualize_processed_spectra(self, spectral_data, spectral_axis,
                                    title="Processed Raman Spectra", figsize=(12, 6),
                                    xlim=None, ylim=None, legend=True,
                                    sample_names=None, legend_loc='best',
                                    sample_limit=10, cmap='viridis',
                                    add_mean=False) -> plt:
        """
        Visualize processed spectral data from ramanspy.
        
        **NOTE**: This method now delegates to the basic_plots module.

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
        # Delegate to extracted module
        return basic_plots.visualize_processed_spectra(
            spectral_data=spectral_data,
            spectral_axis=spectral_axis,
            title=title,
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
            legend=legend,
            sample_names=sample_names,
            legend_loc=legend_loc,
            sample_limit=sample_limit,
            cmap=cmap,
            add_mean=add_mean
        )

    def extract_raman_characteristics(x: np.ndarray, y: np.ndarray, sample_name: str = "Sample", show_plot: bool = False) -> tuple[list[tuple], float]:
        """
        Extract Raman characteristics from the spectrum.
        
        **NOTE**: This method now delegates to the basic_plots module.
        
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
        # Delegate to extracted module
        return basic_plots.extract_raman_characteristics(x, y, sample_name, show_plot)

    def pca2d(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
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
        legend_loc: str = 'best',
        show_decision_boundary: bool = False,
        decision_boundary_alpha: float = 0.3,
        # Parameters for using pre-calculated boundary data
        use_precalculated_boundary: bool = True,
    ) -> plt:
        """
        Perform PCA on DataFrame/numpy array or SpectralContainer objects and plot the first two PCs.
        Now automatically uses data from ML_PROPERTY when parameters are not provided.

        Args:
            df (pd.DataFrame or np.ndarray, optional):
                DataFrame or numpy array containing spectral data. If None, uses data from ML_PROPERTY.
            containers (List[rp.SpectralContainer], optional):
                List of SpectralContainer objects containing spectral data. If None, uses data from ML_PROPERTY.
            labels (List[str], optional):
                List of labels corresponding to the data points. If None, uses data from ML_PROPERTY.
            common_axis (np.ndarray, optional):
                Common spectral axis for interpolation. If None, uses data from ML_PROPERTY.
            current_n_features (int, optional):
                Number of features in the current dataset. If None, uses data from ML_PROPERTY.
            title (str):
                Title for the PCA plot.
            figsize (tuple):
                Size of the figure (width, height).
            sample_limit (int):
                Maximum number of samples to plot (to avoid overcrowding).
            cmap (str):
                Colormap for the plot.
            add_centroids (bool):
                Whether to add centroids for each class in the plot.
            show_centroid_line (bool):
                Whether to draw a line between centroids of the two classes.
            legend (bool):
                Whether to show the legend in the plot.
            legend_loc (str):
                Location of the legend in the plot.
            show_decision_boundary (bool):
                Whether to show the decision boundary using pre-calculated data from ML_PROPERTY.
            decision_boundary_alpha (float):
                Transparency of the decision boundary contour.
            use_precalculated_boundary (bool):
                Whether to use pre-calculated boundary data from ML_PROPERTY if available.

        Returns:
            plt: The matplotlib plot object.

        Notes:
            - If no parameters are provided, the function will automatically use training data from ML_PROPERTY
            - Decision boundary data must be pre-calculated using predict() with calculate_pca_boundary=True
            - No on-the-fly boundary calculation is performed to avoid redundant predictions
        """
        # Delegate to extracted module, passing ML_PROPERTY for auto-detection
        return ml_visualization.pca2d(
            df=df,
            containers=containers,
            labels=labels,
            common_axis=common_axis,
            current_n_features=current_n_features,
            ml_property=self.ML_PROPERTY if hasattr(self, 'ML_PROPERTY') else None,
            title=title,
            figsize=figsize,
            sample_limit=sample_limit,
            cmap=cmap,
            add_centroids=add_centroids,
            show_centroid_line=show_centroid_line,
            legend=legend,
            legend_loc=legend_loc,
            show_decision_boundary=show_decision_boundary,
            decision_boundary_alpha=decision_boundary_alpha,
            use_precalculated_boundary=use_precalculated_boundary
        )

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
        
        **NOTE**: This method now delegates to the model_evaluation module.

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
        # Delegate to extracted module
        return model_evaluation.confusion_matrix_heatmap(
            y_true=y_true,
            y_pred=y_pred,
            class_labels=class_labels,
            title=title,
            figsize=figsize,
            cmap=cmap,
            normalize=normalize,
            show_counts=show_counts,
            fmt=fmt,
            show_heatmap=show_heatmap
        )

    def shap_explain(
        self,
        ML_PROPERTY: Union[RamanML, MLModel] = None,
        nsamples: int = 50,
        show_plots: bool = True,
        max_display: int = 15,
        wavenumber_axis: np.ndarray = None,
        # Performance optimization parameters
        max_background_samples: int = 50,
        max_test_samples: int = 20,
        reduce_features: bool = True,
        max_features: int = 300,
        use_kmeans_sampling: bool = False,
        fast_mode: bool = False,
        # NEW: Use base estimator for CalibratedClassifierCV
        use_base_estimator: bool = True,
        # NEW: Force KernelExplainer for all models
        force_kernel_explainer: bool = False,
        shap_output_mode: str = "auto",  # NEW: "auto", "full", "sparse"
        # NEW: Increase samples for KernelExplainer
        kernel_nsamples_multiplier: float = 2.0,
    ) -> dict:
        """
        Generate SHAP explanations using stored training/test data from previous model training.
        REFACTORED: Now uses nested functions for better organization and tracking.

        Args:
            ML_PROPERTY (ML_PROPERTY): ML_PROPERTY instance with trained model
            nsamples (int): Number of samples for SHAP KernelExplainer. Lower = faster.
            show_plots (bool): Whether to generate and show SHAP plots.
            max_display (int): Maximum number of features to display in plots.
            wavenumber_axis (np.ndarray, optional): Wavenumber axis for feature names.
            max_background_samples (int): Maximum background samples for SHAP explainer.
            max_test_samples (int): Maximum test samples to explain.
            reduce_features (bool): Whether to use feature reduction for SVC models.
            max_features (int): Maximum features after reduction (only for SVC).
            use_kmeans_sampling (bool): Use K-means clustering for background sample selection.
            fast_mode (bool): Enable ultra-fast mode with minimal samples.
            use_base_estimator (bool): For CalibratedClassifierCV, use base estimator for SHAP
            force_kernel_explainer (bool): Force KernelExplainer for all models, even SVC.
            shap_output_mode (str): "auto", "full", or "sparse" for SHAP output mode.
            kernel_nsamples_multiplier (float): Multiplier for KernelExplainer samples.

        Returns:
            dict: Comprehensive SHAP analysis results including performance metrics.
        
        **NOTE**: This method now delegates to the explainability module.
        """
        
        # Delegate to extracted module, passing ML_PROPERTY for data extraction
        # Use instance ML_PROPERTY if none provided
        if ML_PROPERTY is None:
            ML_PROPERTY = self.ML_PROPERTY
        
        return explainability.shap_explain(
            ml_property=ML_PROPERTY,
            nsamples=nsamples,
            show_plots=show_plots,
            max_display=max_display,
            wavenumber_axis=wavenumber_axis,
            max_background_samples=max_background_samples,
            max_test_samples=max_test_samples,
            reduce_features=reduce_features,
            max_features=max_features,
            use_kmeans_sampling=use_kmeans_sampling,
            fast_mode=fast_mode,
            use_base_estimator=use_base_estimator,
            force_kernel_explainer=force_kernel_explainer,
            shap_output_mode=shap_output_mode,
            kernel_nsamples_multiplier=kernel_nsamples_multiplier
        )

    def lime_explain(
        self,
        ML_PROPERTY: Union[RamanML, MLModel] = None,
        test_spectra: list[rp.SpectralContainer] = None,
        true_labels: list[str] = None,
        num_features: int = 20,  # Number of features to include in the explanation
        show_plots: bool = True,
        num_samples: int = 1000,  # Number of samples to generate for LIME explanations
        # Sample selection parameters
        max_test_samples: int = 5,  # Maximum number of test samples to explain
        # Specific indices to explain (overrides max_test_samples)
        sample_indices: List[int] = None,
        seed: int = 42,  # Random seed for reproducibility
        # Visualization parameters
        figsize: tuple = (10, 6),
        positive_label: str = "disease",
        negative_label: str = "normal",
        # Performance parameters
        # Use feature selection for faster computation
        use_feature_selection: bool = True,
        max_features: int = 300,  # Maximum features to use if feature selection enabled
        use_kmeans_sampling: bool = False  # Use K-means for background sampling
    ) -> dict:
        """
        Generate LIME explanations for the model predictions with unified prediction function.

        LIME (Local Interpretable Model-agnostic Explanations) creates a simple surrogate model
        that explains individual predictions by perturbing the input and seeing how predictions change.

        Args:
            ML_PROPERTY (ML_PROPERTY): ML_PROPERTY instance with trained model
            test_spectra (list[rp.SpectralContainer]): Test spectra to explain. If None, uses stored test data.
            true_labels (list[str]): True labels for test spectra. If None, uses stored test labels.
            num_features (int): Number of features to include in the explanation
            show_plots (bool): Whether to show plots
            num_samples (int): Number of samples to generate for LIME explanations
            max_test_samples (int): Maximum number of test samples to explain
            sample_indices (List[int]): Specific indices to explain (overrides max_test_samples)
            seed (int): Random seed for reproducibility
            figsize (tuple): Figure size for plots
            positive_label (str): Label for positive class (default: "disease")
            negative_label (str): Label for negative class (default: "normal")
            use_feature_selection (bool): Use feature selection for faster computation
            max_features (int): Maximum features to use if feature selection enabled
            use_kmeans_sampling (bool): Use K-means for background sampling

        Returns:
            dict: Dictionary containing LIME explanations and visualizations
        """
        try:
            start_time = time.time()
            ML_PROPERTY = ML_PROPERTY if ML_PROPERTY is not None else self.ML_PROPERTY

            if ML_PROPERTY is None:
                raise ValueError(
                    "No ML_PROPERTY instance provided. Please provide a ML_PROPERTY instance with trained data.")

            # Handle test data input - either from parameters or stored data
            if test_spectra is not None and true_labels is not None:
                # Convert spectral containers to numpy arrays
                all_spectra = []
                for container_idx, container in enumerate(test_spectra):
                    for spectrum_idx, spectrum in enumerate(container.spectral_data):
                        # Interpolate to common axis if needed
                        if len(container.spectral_axis) != len(ML_PROPERTY.common_axis):
                            interp_spectrum = np.interp(
                                ML_PROPERTY.common_axis, container.spectral_axis, spectrum)
                            all_spectra.append(interp_spectrum)
                        else:
                            all_spectra.append(spectrum)

                X_test = np.array(all_spectra)
                y_test = np.array(true_labels[:len(all_spectra)])

                console_log(
                    f"Using provided test data: {X_test.shape[0]} test samples")
            else:
                # Use stored test data
                if ML_PROPERTY.X_train is None or ML_PROPERTY.y_train is None or ML_PROPERTY.X_test is None or ML_PROPERTY.y_test is None:
                    raise ValueError(
                        "No training/test data found. Please run train_svc() or train_rf() first or provide test_spectra and true_labels.")

                X_test = ML_PROPERTY.X_test
                y_test = ML_PROPERTY.y_test
                console_log(
                    f"Using stored test data: {X_test.shape[0]} test samples")

            # Get training data (always from stored data)
            X_train = ML_PROPERTY.X_train
            y_train = ML_PROPERTY.y_train

            # Get model and data
            if hasattr(ML_PROPERTY, "_model") and ML_PROPERTY._model is not None:
                model = ML_PROPERTY._model
            else:
                raise ValueError(
                    "No trained model found. Train a model first or provide one.")

            # === UNIFIED MODEL TYPE DETECTION (same as SHAP) ===
            is_svc = isinstance(model, SVC)
            is_rf = isinstance(model, RandomForestClassifier)
            is_linear_svc = is_svc and hasattr(
                model, 'kernel') and model.kernel == 'linear'

            console_log(
                f"Creating unified prediction function for {type(model).__name__}")
            console_log(
                f"Model type detection: SVC={is_svc}, RF={is_rf}, Linear SVC={is_linear_svc}")

            # Get class names and wavenumber axis
            class_names = list(np.unique(y_train))
            if len(class_names) != 2:
                console_log(f"Warning: LIME visualization works best for binary classification. "
                            f"Found {len(class_names)} classes: {class_names}")

            wavenumber_axis = ML_PROPERTY.common_axis if hasattr(
                ML_PROPERTY, "common_axis") else None

            console_log(
                f"Original data shapes: X_train: {X_train.shape}, X_test: {X_test.shape}")

            # Handle sample indices BEFORE feature selection
            if sample_indices is not None:
                # Use specified indices - check against test set size
                indices = sample_indices
                if any(idx >= len(X_test) for idx in indices):
                    console_log(
                        f"Warning: Some indices exceed the test set size ({len(X_test)}). Filtering valid indices.")
                    indices = [idx for idx in indices if idx < len(X_test)]

                if not indices:
                    raise ValueError(
                        "No valid sample indices provided for the test set.")

                console_log(
                    f"Selected sample indices from test set: {indices}")
            else:
                # Randomly select samples from test set
                np.random.seed(seed)
                n_samples = min(max_test_samples, len(X_test))
                indices = np.random.choice(
                    len(X_test), n_samples, replace=False)
                console_log(
                    f"Randomly selected {n_samples} samples from test set")

            # === FEATURE SELECTION WITH UNIFIED PREDICTION FUNCTION ===
            feature_selector = None
            if use_feature_selection and X_train.shape[1] > max_features:
                console_log(
                    f"Applying feature selection to reduce from {X_train.shape[1]} to {max_features} features")
                selector = SelectKBest(f_classif, k=max_features)
                X_train_reduced = selector.fit_transform(X_train, y_train)
                X_test_reduced = selector.transform(X_test)
                feature_selector = selector

                # Update wavenumber axis if provided
                if wavenumber_axis is not None:
                    wavenumber_axis_reduced = wavenumber_axis[selector.get_support(
                    )]
                else:
                    wavenumber_axis_reduced = np.arange(
                        X_train_reduced.shape[1])

                # === UNIFIED PREDICTION FUNCTION WITH FEATURE SELECTION ===
                def unified_predict_fn(x_reduced):
                    """
                    Unified prediction function that handles CalibratedClassifierCV and feature selection.
                    """
                    # Ensure input is 2D
                    if x_reduced.ndim == 1:
                        x_reduced = x_reduced.reshape(1, -1)

                    # Handle feature selection if applied
                    if feature_selector is not None:
                        if hasattr(model, 'n_features_in_') and model.n_features_in_ != x_reduced.shape[1]:
                            # Model was trained on original features, need to expand
                            original_shape = (
                                x_reduced.shape[0], X_train.shape[1])
                            x_expanded = np.zeros(original_shape)
                            selected_indices_feat = selector.get_support(
                                indices=True)
                            x_expanded[:, selected_indices_feat] = x_reduced
                            x_input = x_expanded
                        else:
                            x_input = x_reduced
                    else:
                        x_input = x_reduced

                    # FIXED: Handle CalibratedClassifierCV first
                    from sklearn.calibration import CalibratedClassifierCV
                    if isinstance(model, CalibratedClassifierCV):
                        # CalibratedClassifierCV always has predict_proba and it's calibrated
                        return model.predict_proba(x_input)
                    elif is_rf:
                        # RandomForest always has predict_proba
                        return model.predict_proba(x_input)
                    elif is_svc:
                        # For SVC, check if probability is enabled
                        if hasattr(model, 'predict_proba') and model.probability:
                            return model.predict_proba(x_input)
                        else:
                            # For SVC without probability, use decision_function (same as SHAP)
                            decision = model.decision_function(x_input)
                            from scipy.special import expit
                            if decision.ndim == 1:
                                # Binary classification
                                prob_pos = expit(decision)
                                return np.column_stack([1 - prob_pos, prob_pos])
                            else:
                                # Multiclass - use softmax
                                exp_scores = np.exp(
                                    decision - np.max(decision, axis=1, keepdims=True))
                                return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                    else:
                        # For other model types, try predict_proba first
                        if hasattr(model, 'predict_proba'):
                            return model.predict_proba(x_input)
                        else:
                            # Fallback for models without predict_proba
                            predictions = model.predict(x_input)
                            # Convert to dummy probabilities (not ideal but works)
                            n_classes = len(np.unique(predictions))
                            dummy_probs = np.eye(n_classes)[predictions]
                            return dummy_probs

                feature_names = [f"{wavenumber_axis_reduced[i]:.1f} cm⁻¹"
                                 for i in range(len(wavenumber_axis_reduced))]

            else:
                # No feature selection
                X_train_reduced = X_train
                X_test_reduced = X_test

                # === UNIFIED PREDICTION FUNCTION WITHOUT FEATURE SELECTION ===
                def unified_predict_fn(x):
                    """Unified prediction function without feature selection."""
                    if x.ndim == 1:
                        x = x.reshape(1, -1)

                    if hasattr(model, 'predict_proba') and model.probability:
                        return model.predict_proba(x)
                    else:
                        # Same logic as SHAP for SVC without probability
                        decision = model.decision_function(x)
                        from scipy.special import expit
                        if decision.ndim == 1:
                            prob_pos = expit(decision)
                            return np.column_stack([1 - prob_pos, prob_pos])
                        else:
                            exp_scores = np.exp(
                                decision - np.max(decision, axis=1, keepdims=True))
                            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

                if wavenumber_axis is not None:
                    feature_names = [f"{wavenumber_axis[i]:.1f} cm⁻¹"
                                     for i in range(len(wavenumber_axis))]
                else:
                    feature_names = [f"Feature_{i}" for i in range(
                        X_train_reduced.shape[1])]

            console_log(
                f"After feature selection: X_train_reduced: {X_train_reduced.shape}, X_test_reduced: {X_test_reduced.shape}")

            # Now use the unified_predict_fn for both prediction verification AND LIME
            console_log(
                f"Setting up LIME explainer with unified prediction function...")

            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train_reduced,
                feature_names=feature_names,
                class_names=class_names,
                mode='classification',
                random_state=seed,
                discretize_continuous=False  # Keep continuous for spectral data
            )

            # Select the samples from the reduced test set using the same indices
            selected_samples_reduced = X_test_reduced[indices]
            selected_labels = y_test[indices]

            console_log(
                f"Selected samples shape after feature selection: {selected_samples_reduced.shape}")

            # Generate explanations
            console_log(
                f"Generating LIME explanations for {len(indices)} samples...")
            explanations = []

            for i, (original_idx, x_sample, y_true) in enumerate(zip(indices, selected_samples_reduced, selected_labels)):
                console_log(
                    f"\nExplaining sample {i+1}/{len(indices)} (original index {original_idx})...")

                # === USE UNIFIED PREDICTION FUNCTION FOR CONSISTENCY ===
                y_pred_proba = unified_predict_fn(x_sample.reshape(1, -1))[0]
                y_pred_idx = np.argmax(y_pred_proba)

                # Get class names - handle CalibratedClassifierCV
                from sklearn.calibration import CalibratedClassifierCV
                if isinstance(model, CalibratedClassifierCV):
                    if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                        model_class_names = model.estimators_[0].classes_
                    elif hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'classes_'):
                        model_class_names = model.base_estimator.classes_
                    else:
                        model_class_names = class_names  # Use the ones we defined earlier
                else:
                    model_class_names = model.classes_ if hasattr(
                        model, 'classes_') else class_names

                y_pred = model_class_names[y_pred_idx]
                confidence = y_pred_proba[y_pred_idx]

                console_log(
                    f"True class: {y_true}, Predicted: {y_pred} (confidence: {confidence:.3f})")

                # Generate explanation using the SAME unified function
                try:
                    exp = explainer.explain_instance(
                        x_sample,
                        unified_predict_fn,  # <- Use the same function!
                        num_features=min(num_features, len(feature_names)),
                        num_samples=num_samples,
                        top_labels=len(class_names)
                    )

                    # Store explanation info
                    exp_dict = {
                        'sample_idx': original_idx,  # Use original index
                        'true_label': y_true,
                        'predicted_label': y_pred,
                        'confidence': float(confidence),
                        'explanation': exp
                    }

                    # Visualize if requested
                    if show_plots:
                        self._visualize_lime_explanation(
                            exp, class_names, feature_names,
                            y_true, y_pred, confidence,
                            positive_label, negative_label,
                            figsize=figsize, sample_index=original_idx,
                            wavenumber_axis=wavenumber_axis_reduced if feature_selector else wavenumber_axis
                        )

                    explanations.append(exp_dict)

                except Exception as e:
                    console_log(
                        f"Error explaining instance {original_idx}: {str(e)}")
                    continue

            # Create data arrays for wavenumbers with most impact
            top_features = {}
            for class_name in class_names:
                # Collect feature weights across all explanations for this class
                class_feature_weights = {}

                for exp_dict in explanations:
                    exp = exp_dict['explanation']

                    # Get the feature list for this class
                    try:
                        class_idx = class_names.index(class_name)
                        for feature_id, weight in exp.local_exp[class_idx]:
                            feature_name = feature_names[feature_id]

                            if feature_name not in class_feature_weights:
                                class_feature_weights[feature_name] = []

                            class_feature_weights[feature_name].append(weight)
                    except:
                        # This explanation might not have this class
                        continue

                # Calculate average weights
                avg_weights = {
                    feature: np.mean(weights)
                    for feature, weights in class_feature_weights.items()
                }

                # Sort features by absolute weight and take top ones
                sorted_features = sorted(
                    avg_weights.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:num_features]

                top_features[class_name] = sorted_features

            # Create comparison visualization of top features across classes if binary
            if len(class_names) == 2 and show_plots:
                self._visualize_lime_comparison(
                    top_features,
                    class_names,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    figsize=figsize
                )

            return {
                "success": True,
                "explanation_objects": explanations,
                "num_samples": len(indices),
                "top_features_by_class": top_features,
                "class_names": class_names,
                "feature_names": feature_names,
                "feature_selection_used": feature_selector is not None,
                "processing_time": time.time() - start_time,
                "wavenumber_axis": (wavenumber_axis_reduced if feature_selector else wavenumber_axis).tolist() if wavenumber_axis is not None else None,
                "unified_prediction_function_used": True  # New flag to indicate fix
            }

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            console_log(f"LIME explanation error: {str(e)}")
            console_log(error_details)

            return {
                "success": False,
                "error": str(e),
                "error_details": error_details
            }

    def _visualize_lime_explanation(
        self,
        explanation,
        class_names,
        feature_names,
        true_label,
        predicted_label,
        confidence,
        positive_label="cancer",
        negative_label="benign",
        figsize=(12, 8),
        sample_index=0,
        wavenumber_axis=None
    ):
        """
        Visualize a LIME explanation with customized styling for Raman spectroscopy.

        Args:
            explanation: The LIME explanation object
            class_names: List of class names
            feature_names: List of feature names
            true_label: The true label of the sample
            predicted_label: The predicted label
            confidence: The prediction confidence
            positive_label: Label for positive class (default: "cancer")
            negative_label: Label for negative class (default: "benign")
            figsize: Figure size for the plot
            sample_index: Index of the sample in the test set
            wavenumber_axis: The wavenumber axis (if available)
        """
        # Plot info
        plt.figure(figsize=figsize)

        # Pick the class index to show (predicted class)
        if len(class_names) == 2:
            # For binary classification
            show_class_idx = class_names.index(predicted_label)
        else:
            # For multiclass, show the predicted class
            show_class_idx = class_names.index(
                predicted_label) if predicted_label in class_names else 0

        # Get explanation data
        try:
            # Extract explanation for the selected class
            exp_list = explanation.as_list(label=show_class_idx)

            # Sort by absolute impact
            exp_list = sorted(exp_list, key=lambda x: abs(
                float(x[1])), reverse=True)

            # Extract features and weights
            features = [x[0] for x in exp_list]
            weights = [float(x[1]) for x in exp_list]

            # Split into positive and negative impacts
            pos_features = []
            pos_weights = []
            neg_features = []
            neg_weights = []

            for f, w in zip(features, weights):
                if w > 0:
                    pos_features.append(f)
                    pos_weights.append(w)
                else:
                    neg_features.append(f)
                    neg_weights.append(abs(w))  # Use absolute for plotting

            # Formatting and colors
            pos_color = '#D62728'  # Red for positive impacts (cancer)
            neg_color = '#1F77B4'  # Blue for negative impacts (benign)

            # Create horizontal bar chart with separate bars
            plt.figure(figsize=figsize)

            # Calculate positions for bars
            num_features = max(len(pos_features), len(neg_features))
            bar_height = 0.4
            y_pos = np.arange(num_features) * 1.0

            # Ensure we have labels for all positions
            all_features = []
            for i in range(num_features):
                if i < len(pos_features) and i < len(neg_features):
                    all_features.append(
                        f"{pos_features[i]} / {neg_features[i]}")
                elif i < len(pos_features):
                    all_features.append(pos_features[i])
                elif i < len(neg_features):
                    all_features.append(neg_features[i])

            # Plot bars for positive impact (right side)
            if pos_weights:
                plt.barh(y_pos[:len(pos_weights)], pos_weights, height=bar_height,
                         color=pos_color, alpha=0.7, label=f'Toward {positive_label.upper()}')

            # Plot bars for negative impact (left side)
            if neg_weights:
                plt.barh(y_pos[:len(neg_weights)], [-w for w in neg_weights], height=bar_height,
                         color=neg_color, alpha=0.7, label=f'Toward {negative_label.upper()}')

            # Set up the plot
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            plt.grid(alpha=0.3)

            # Custom y-ticks for wavenumbers
            try:
                # Try to extract numerical wavenumbers from feature names
                wavenumbers = []
                for feature in features[:num_features]:
                    match = re.search(r'(\d+\.\d+)', feature)
                    if match:
                        wavenumbers.append(f"{float(match.group(1)):.1f} cm⁻¹")
                    else:
                        wavenumbers.append(feature)

                plt.yticks(y_pos[:num_features], wavenumbers)
            except:
                plt.yticks(y_pos[:num_features], [f.split(' ≤')[0].split(
                    ' >')[0] for f in all_features[:num_features]])

            # Add labels
            plt.xlabel('Impact on Prediction')
            plt.ylabel('Raman Shift')

            # Title with accuracy information
            correct_str = "✓ Correct" if predicted_label == true_label else "✗ Incorrect"
            title = (f"LIME Explanation (Sample #{sample_index})\n"
                     f"True: {true_label}, Predicted: {predicted_label} ({confidence:.3f}) - {correct_str}")

            plt.title(title, fontweight='bold')
            plt.legend(loc='best')
            plt.tight_layout()

            # Add a table with key information
            table_data = [
                ['Prediction', predicted_label,
                    f'Confidence: {confidence:.3f}'],
                ['True Label', true_label, f'Result: {correct_str}'],
                ['Class Probabilities', '', '']
            ]

            # Add probabilities for each class
            class_probas = explanation.predict_proba.tolist()
            for i, (cls, proba) in enumerate(zip(class_names, class_probas)):
                formatted_proba = f"{proba:.3f}"
                highlight = '→ ' if i == show_class_idx else ''
                table_data.append([f'{highlight}{cls}', formatted_proba, ''])

            # Compute table position
            table = plt.table(
                cellText=table_data,
                cellLoc='left',
                loc='bottom',
                bbox=[0.0, -0.32, 0.6, 0.20]  # Adjust as needed
            )

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)

            # Color the table cells
            for i, row in enumerate(table_data):
                for j in range(3):
                    cell = table[(i, j)]
                    if i == 0:  # Prediction row
                        if j == 1:  # Class name
                            text_color = '#D62728' if predicted_label == positive_label else '#1F77B4'
                            cell.set_text_props(
                                weight='bold', color=text_color)
                    elif i == 1:  # True label row
                        if j == 1:  # Class name
                            text_color = '#D62728' if true_label == positive_label else '#1F77B4'
                            cell.set_text_props(
                                weight='bold', color=text_color)
                    elif i == 2:  # Header for probabilities
                        cell.set_facecolor('#F0F0F0')
                        cell.set_text_props(weight='bold')
                    elif i > 2:  # Class probability rows
                        if f"{class_names[i-3]}" == positive_label:
                            cell.set_text_props(color='#D62728')
                        elif f"{class_names[i-3]}" == negative_label:
                            cell.set_text_props(color='#1F77B4')

            plt.subplots_adjust(bottom=0.25)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            console_log(f"Error in LIME visualization: {str(e)}")
            import traceback
            traceback.print_exc()

    def _visualize_lime_comparison(
        self,
        top_features,
        class_names,
        positive_label="cancer",
        negative_label="benign",
        figsize=(14, 10)
    ):
        """
        Visualize comparison of feature importance between classes.

        Args:
            top_features: Dictionary with class names as keys and lists of (feature, weight) tuples as values
            class_names: List of class names
            positive_label: Name for positive label
            negative_label: Name for negative label
            figsize: Figure size for plot
        """
        if len(class_names) != 2:
            console_log(
                "Feature comparison visualization requires exactly 2 classes.")
            return

        plt.figure(figsize=figsize)

        # Get the class names in correct order
        pos_class = positive_label
        neg_class = negative_label

        if pos_class not in top_features or neg_class not in top_features:
            # Try case-insensitive matching
            for cls in top_features.keys():
                if cls.lower() == pos_class.lower():
                    pos_class = cls
                if cls.lower() == neg_class.lower():
                    neg_class = cls

        # Organize the feature data
        pos_data = dict(top_features.get(pos_class, []))
        neg_data = dict(top_features.get(neg_class, []))

        # Get all unique features
        all_features = set(list(pos_data.keys()) + list(neg_data.keys()))

        # Extract wavenumbers for sorting
        feature_wavenumbers = {}
        for feature in all_features:
            # Extract wavenumber from feature name (format: "123.4 cm⁻¹")
            match = re.search(r'(\d+\.\d+)', feature)
            if match:
                feature_wavenumbers[feature] = float(match.group(1))
            else:
                # If we can't extract, use a default
                feature_wavenumbers[feature] = 0

        # Sort features by wavenumber
        sorted_features = sorted(
            all_features, key=lambda f: feature_wavenumbers[f])

        # Prepare data for plotting
        x = np.arange(len(sorted_features))
        bar_width = 0.35

        # Get feature importance values in the sorted order
        pos_values = [pos_data.get(f, 0) for f in sorted_features]
        neg_values = [neg_data.get(f, 0) for f in sorted_features]

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        bars1 = ax.bar(x - bar_width/2, pos_values, bar_width,
                       label=f'{pos_class}', color='#D62728', alpha=0.7)
        bars2 = ax.bar(x + bar_width/2, neg_values, bar_width,
                       label=f'{neg_class}', color='#1F77B4', alpha=0.7)

        # Add some text for labels, title and axes ticks
        ax.set_xlabel('Raman Shift (cm⁻¹)')
        ax.set_ylabel('Average Feature Importance')
        ax.set_title('Comparison of Important Features by Class')
        ax.set_xticks(x)

        # Format x-tick labels to show just the wavenumber
        x_labels = []
        for feature in sorted_features:
            match = re.search(r'(\d+\.\d+)', feature)
            if match:
                x_labels.append(f"{float(match.group(1)):.1f}")
            else:
                x_labels.append(feature.split(' ')[0])

        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.legend()

        # Add data labels above bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                if abs(height) > 0.01:  # Only label significant values
                    ax.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                # 3 points vertical offset for positive, -10 for negative
                                xytext=(0, 3 if height > 0 else -10),
                                textcoords="offset points",
                                ha='center', va='bottom' if height > 0 else 'top',
                                fontsize=8)

        autolabel(bars1)
        autolabel(bars2)

        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.tight_layout()

        # Add "cm⁻¹" to the x-label
        ax.set_xlabel('Raman Shift (cm⁻¹)')

        # Adjust bottom margin to fit labels
        plt.subplots_adjust(bottom=0.15)

        plt.show()

        # Also create a table view of the data
        plt.figure(figsize=(10, len(sorted_features)*0.4 + 3))
        plt.axis('off')

        table_data = []
        table_data.append(
            ["Raman Shift (cm⁻¹)", f"{pos_class} Importance", f"{neg_class} Importance", "Difference"])

        for i, feature in enumerate(sorted_features):
            pos_val = pos_data.get(feature, 0)
            neg_val = neg_data.get(feature, 0)
            diff = abs(pos_val) - abs(neg_val)

            row_data = [
                x_labels[i],
                f"{pos_val:.4f}",
                f"{neg_val:.4f}",
                f"{diff:.4f}"
            ]
            table_data.append(row_data)

        table = plt.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            colWidths=[0.2, 0.26, 0.26, 0.26]
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Color the table cells
        for i, row in enumerate(table_data):
            for j in range(4):
                cell = table[(i, j)]

                # Header row
                if i == 0:
                    cell.set_facecolor('#F0F0F0')
                    cell.set_text_props(weight='bold')
                    if j == 1:
                        # Positive class (red)
                        cell.set_text_props(color='#D62728')
                    elif j == 2:
                        # Negative class (blue)
                        cell.set_text_props(color='#1F77B4')
                # Data rows
                else:
                    # Positive class column
                    if j == 1 and float(row[j]) > 0:
                        cell.set_facecolor('#FFEDED')
                        cell.set_text_props(color='#D62728')
                    # Negative class column
                    elif j == 2 and float(row[j]) > 0:
                        cell.set_facecolor('#EDF6FF')
                        cell.set_text_props(color='#1F77B4')
                    # Difference column
                    elif j == 3:
                        try:
                            diff_val = float(row[j])
                            if diff_val > 0:  # Positive class more important
                                cell.set_facecolor('#FFEDED')
                                cell.set_text_props(color='#D62728')
                            elif diff_val < 0:  # Negative class more important
                                cell.set_facecolor('#EDF6FF')
                                cell.set_text_props(color='#1F77B4')
                        except:
                            pass

        plt.title('Feature Importance Comparison Table', fontweight='bold')
        plt.tight_layout()
        plt.show()

    def inspect_spectra(
        self,
        ML_PROPERTY: Union[RamanML, MLModel] = None,
        test_spectra: list[rp.SpectralContainer] = None,
        true_labels: list[str] = None,
        n_samples: int = 5,
        sample_indices: List[int] = None,
        seed: int = None,
        class_to_inspect: str = None,
        show_plots: bool = True,
        max_display: int = 10,
        nsamples: int = 100,
        fast_mode: bool = False,
        show_shap_plots: bool = False,
        show_lime_plots: bool = False,
        positive_label: str = "cancer",
        negative_label: str = "benign",
        lime_num_features: int = 20,
        lime_num_samples: int = 1000,
        lime_figsize: tuple = (10, 6),
        lime_seed: int = 42,
        lime_use_feature_selection: bool = True,
        lime_max_features: int = 300,
        lime_use_kmeans_sampling: bool = False,
        **kwargs
    ) -> dict:
        """
        Randomly select spectra and show detailed SHAP and/or LIME explanations for why they were
        classified as positive_label or negative_label. Compatible with both RamanML and MLModel classes.

        Parameters:
        -----------
        ML_PROPERTY : Union[RamanML, MLModel], optional
            ML_PROPERTY instance with trained model. If None, uses stored ML_PROPERTY.
        test_spectra : list, optional
            List of SpectralContainer objects. If None, uses stored test data.
        true_labels : list, optional
            True labels for the spectra. If None, uses stored test labels.
        n_samples : int
            Number of random spectra to inspect
        sample_indices : list, optional
            Specific indices of spectra to inspect. If provided, overrides n_samples.
        seed : int, optional
            Random seed for reproducibility
        class_to_inspect : str, optional
            If provided, only inspect spectra from this class ("benign" or "cancer")
        show_plots : bool
            Whether to show general plots (spectrum with highlights, SHAP plots, tables)
        max_display : int
            Maximum number of features to display in SHAP plots
        nsamples : int
            Number of samples for SHAP computation
        fast_mode : bool
            Enable fast mode for SHAP computation
        show_shap_plots : bool
            Whether to show SHAP plots for each inspected spectrum
        show_lime_plots : bool
            Whether to show LIME plots for each inspected spectrum (independent of show_plots)
        positive_label : str
            Label for positive class (default: "cancer")
        negative_label : str
            Label for negative class (default: "benign")
        lime_num_features : int
            Number of features to include in LIME explanation
        lime_num_samples : int
            Number of samples to generate for LIME explanations
        lime_figsize : tuple
            Figure size for LIME plots
        lime_seed : int
            Random seed for LIME explanations
        lime_use_feature_selection : bool
            Whether to use feature selection for LIME explanations
        lime_max_features : int
            Maximum number of features to use in LIME explanations
        lime_use_kmeans_sampling : bool
            Whether to use KMeans sampling for LIME explanations

        Returns:
        --------
        dict
            Dictionary of inspection results including both SHAP and LIME explanations
        """

        def _validate_and_prepare_ml_property():
            """Validate and prepare ML_PROPERTY instance."""
            nonlocal ML_PROPERTY

            ML_PROPERTY = ML_PROPERTY if ML_PROPERTY is not None else self.ML_PROPERTY

            if ML_PROPERTY is None:
                raise ValueError(
                    "ML_PROPERTY instance is required for spectrum inspection.")

            # Detect ML_PROPERTY type and get the actual sklearn model
            ml_type = type(ML_PROPERTY).__name__

            if ml_type == "RamanML":
                actual_model = getattr(ML_PROPERTY, '_model', None)
                if actual_model is None:
                    raise ValueError(
                        "No trained model found in RamanML instance.")
            elif ml_type == "MLModel":
                actual_model = getattr(ML_PROPERTY, 'sklearn_model', None)
                if actual_model is None:
                    raise ValueError(
                        "No sklearn model found in MLModel instance.")
            else:
                raise ValueError(f"Unsupported ML_PROPERTY type: {ml_type}")

            console_log(
                f"Using {ml_type} with model type: {type(actual_model).__name__}")

            return actual_model, ml_type

        def _prepare_test_data():
            """Prepare test data from various sources with priority logic."""

            # UPDATED: Priority logic for data source selection
            # 1. Use provided test_spectra and true_labels if both given
            # 2. Use stored X_test and y_test if available
            # 3. Fall back to X_train and y_train only for RamanML

            if test_spectra is not None and true_labels is not None:
                console_log("📊 Using provided test_spectra and true_labels")

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

                return all_spectra

            # Try to use stored test data first (works for both RamanML and MLModel)
            elif (hasattr(ML_PROPERTY, 'X_test') and ML_PROPERTY.X_test is not None and
                  hasattr(ML_PROPERTY, 'y_test') and ML_PROPERTY.y_test is not None and
                  hasattr(ML_PROPERTY, 'common_axis') and ML_PROPERTY.common_axis is not None):

                console_log("📊 Using stored test data (X_test, y_test)")

                # Convert stored test data back to spectral format for analysis
                X_test = ML_PROPERTY.X_test
                y_test = ML_PROPERTY.y_test
                common_axis = ML_PROPERTY.common_axis

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

                return all_spectra

            # Fall back to training data only for RamanML
            elif (ml_type == "RamanML" and
                  hasattr(ML_PROPERTY, 'X_train') and ML_PROPERTY.X_train is not None and
                  hasattr(ML_PROPERTY, 'y_train') and ML_PROPERTY.y_train is not None and
                  hasattr(ML_PROPERTY, 'common_axis') and ML_PROPERTY.common_axis is not None):

                console_log(
                    "📊 Using stored training data (X_train, y_train) - RamanML fallback")

                # Convert stored training data back to spectral format for analysis
                X_train = ML_PROPERTY.X_train
                y_train = ML_PROPERTY.y_train
                common_axis = ML_PROPERTY.common_axis

                # Create artificial spectral containers from stored data
                all_spectra = []
                for idx, (spectrum, label) in enumerate(zip(X_train, y_train)):
                    all_spectra.append({
                        "container_idx": 0,
                        "spectrum_idx": idx,
                        "true_label": label,
                        "spectrum": spectrum,
                        "spectral_axis": common_axis
                    })

                return all_spectra

            else:
                # No data available
                ml_type_str = ml_type if 'ml_type' in locals() else type(ML_PROPERTY).__name__

                if ml_type_str == "MLModel":
                    error_msg = (
                        "No data available for inspection. MLModel requires either:\n"
                        "1. Provide test_spectra and true_labels explicitly, OR\n"
                        "2. Call predict() first to populate X_test and y_test"
                    )
                else:
                    error_msg = (
                        "No data available for inspection. Please either:\n"
                        "1. Provide test_spectra and true_labels explicitly, OR\n"
                        "2. Train a model first to populate training/test data"
                    )

                raise ValueError(error_msg)

        def _filter_and_select_spectra(all_spectra):
            """Filter by class and select random spectra."""
            # Filter by class if requested
            if class_to_inspect:
                filtered_spectra = [
                    s for s in all_spectra if s["true_label"] == class_to_inspect]
                if not filtered_spectra:
                    console_log(
                        f"No spectra found with class {class_to_inspect}")
                    return {"results": [], "summary": f"No spectra found for class: {class_to_inspect}"}
                all_spectra = filtered_spectra

            # Select random spectra
            selected_spectra = []
            if sample_indices is None:
                if n_samples <= 0 or n_samples > len(all_spectra):
                    raise ValueError(
                        f"n_samples must be between 1 and {len(all_spectra)}. Provided: {n_samples}")
                rng = np.random.RandomState(
                    seed) if seed is not None else np.random
                selected_indices = rng.choice(len(all_spectra), min(
                    n_samples, len(all_spectra)), replace=False)
                selected_spectra = [all_spectra[idx]
                                    for idx in selected_indices]
            else:
                for idx in sample_indices:
                    if 0 <= idx < len(all_spectra):
                        selected_spectra.append(all_spectra[idx])
                    else:
                        console_log(
                            f"Warning: Index {idx} out of range, skipping.")

            return selected_spectra

        def _get_unified_prediction_function(model):
            """Create a unified prediction function that handles all model types including CalibratedClassifierCV."""
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC

            is_calibrated = isinstance(model, CalibratedClassifierCV)

            if is_calibrated:
                # For CalibratedClassifierCV, always use predict_proba (it's calibrated)
                def predict_fn(X):
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    return model.predict_proba(X)
                return predict_fn
            else:
                # For non-calibrated models, use the original logic
                is_svc = isinstance(model, SVC)
                is_rf = isinstance(model, RandomForestClassifier)

                def predict_fn(X):
                    if X.ndim == 1:
                        X = X.reshape(1, -1)

                    if is_rf or (hasattr(model, 'predict_proba') and
                                 hasattr(model, 'probability') and model.probability):
                        return model.predict_proba(X)
                    elif is_svc:
                        # For SVC without probability, use decision_function
                        decision = model.decision_function(X)
                        from scipy.special import expit
                        if decision.ndim == 1:
                            prob_pos = expit(decision)
                            return np.column_stack([1 - prob_pos, prob_pos])
                        else:
                            exp_scores = np.exp(
                                decision - np.max(decision, axis=1, keepdims=True))
                            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                    else:
                        # Fallback for other model types
                        if hasattr(model, 'predict_proba'):
                            return model.predict_proba(X)
                        else:
                            predictions = model.predict(X)
                            n_classes = len(np.unique(predictions))
                            dummy_probs = np.eye(n_classes)[predictions]
                            return dummy_probs
                return predict_fn

        def _get_model_class_names(model):
            """Get class names from the model."""
            from sklearn.calibration import CalibratedClassifierCV

            if hasattr(model, 'classes_'):
                return model.classes_
            elif isinstance(model, CalibratedClassifierCV):
                # For CalibratedClassifierCV, get classes from base estimator
                if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                    return model.estimators_[0].classes_
                elif hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'classes_'):
                    return model.base_estimator.classes_
                else:
                    return np.array(['class_0', 'class_1'])  # Fallback
            else:
                return np.array(['class_0', 'class_1'])  # Fallback

        def _predict_single_spectrum(spectrum, spec_info, model, unified_predict):
            """Make prediction for a single spectrum."""
            # Interpolate to common axis if needed
            if len(spec_info["spectral_axis"]) != len(ML_PROPERTY.common_axis):
                spectrum = np.interp(ML_PROPERTY.common_axis,
                                     spec_info["spectral_axis"], spectrum)

            try:
                # Get prediction probabilities
                pred_proba = unified_predict(spectrum.reshape(1, -1))[0]

                # Get class names from model
                class_names = _get_model_class_names(model)

                # Get predicted class and confidence
                pred_idx = np.argmax(pred_proba)
                pred_label = class_names[pred_idx]
                confidence = pred_proba[pred_idx]

                return pred_label, confidence, spectrum

            except Exception as e:
                console_log(f"Error in prediction: {e}")
                return None, None, spectrum

        def _generate_lime_explanations(selected_spectra):
            """Generate LIME explanations if requested."""
            lime_results = None
            if show_lime_plots and len(selected_spectra) > 0:
                console_log("\nGenerating LIME explanations...")
                try:
                    # Extract sample indices for LIME
                    lime_sample_indices = [spec_info["spectrum_idx"]
                                           for spec_info in selected_spectra]

                    lime_results = self.lime_explain(
                        test_spectra=test_spectra,
                        true_labels=true_labels,
                        sample_indices=lime_sample_indices,
                        num_features=lime_num_features,
                        show_plots=show_lime_plots,  # LIME has its own show_plots parameter
                        num_samples=lime_num_samples,
                        figsize=lime_figsize,
                        positive_label=positive_label,
                        negative_label=negative_label,
                        use_feature_selection=lime_use_feature_selection,
                        max_features=lime_max_features,
                        seed=lime_seed,
                        use_kmeans_sampling=lime_use_kmeans_sampling,
                    )

                    if lime_results["success"]:
                        console_log(
                            f"LIME explanations generated successfully for {lime_results['num_samples']} samples")
                    else:
                        console_log(
                            f"LIME explanation failed: {lime_results.get('error', 'Unknown error')}")

                except Exception as e:
                    console_log(f"Error generating LIME explanations: {e}")
                    lime_results = {"success": False, "error": str(e)}

            return lime_results

        def _get_shap_explanation(spectrum, spec_info, spectrum_idx):
            """Generate SHAP explanation for a single spectrum with updated background data logic."""
            try:
                # Create a temporary single-spectrum dataset
                single_spectrum_data = np.array([spectrum]).reshape(1, -1)

                # Store original data
                original_X_test = getattr(ML_PROPERTY, 'X_test', None)
                original_y_test = getattr(ML_PROPERTY, 'y_test', None)
                original_X_train = getattr(ML_PROPERTY, 'X_train', None)
                original_y_train = getattr(ML_PROPERTY, 'y_train', None)

                # Set up data for SHAP
                ML_PROPERTY.X_test = single_spectrum_data
                ML_PROPERTY.y_test = np.array([spec_info["true_label"]])

                # UPDATED: Smart background data selection
                background_data_source = None

                # Priority 1: Use existing training data if available (RamanML)
                if original_X_train is not None and len(original_X_train) > 0:
                    ML_PROPERTY.X_train = original_X_train
                    ML_PROPERTY.y_train = original_y_train
                    background_data_source = "existing_training_data"
                    console_log(
                        f"Using existing training data with {len(original_X_train)} samples for SHAP background")

                # Priority 2: Use existing test data if available (both RamanML and MLModel)
                elif original_X_test is not None and len(original_X_test) > 1:
                    # Use other test samples as background (exclude current sample)
                    background_indices = [i for i in range(
                        len(original_X_test)) if i != spectrum_idx]
                    if len(background_indices) >= 10:
                        # Use subset for efficiency
                        selected_bg_indices = np.random.choice(
                            background_indices, min(50, len(background_indices)), replace=False)
                        ML_PROPERTY.X_train = original_X_test[selected_bg_indices]
                        if original_y_test is not None:
                            ML_PROPERTY.y_train = original_y_test[selected_bg_indices]
                        else:
                            # Create dummy labels
                            available_labels = ['benign', 'cancer']
                            if hasattr(ML_PROPERTY, 'metadata') and 'labels' in ML_PROPERTY.metadata:
                                available_labels = ML_PROPERTY.metadata['labels']
                            ML_PROPERTY.y_train = np.random.choice(
                                available_labels, len(selected_bg_indices))

                        background_data_source = "test_data_subset"
                        console_log(
                            f"Using test data subset with {len(ML_PROPERTY.X_train)} samples for SHAP background")
                    else:
                        raise ValueError(
                            "Insufficient test data for background (need at least 10 samples)")

                # Priority 3: Create synthetic background from provided test_spectra
                elif test_spectra is not None:
                    console_log(
                        "Creating background data from provided test_spectra...")

                    all_background_data = []
                    for container in test_spectra:
                        for spec in container.spectral_data:
                            if len(spec) == len(ML_PROPERTY.common_axis):
                                all_background_data.append(spec)
                            else:
                                # Interpolate if needed
                                interp_spec = np.interp(ML_PROPERTY.common_axis,
                                                        container.spectral_axis, spec)
                                all_background_data.append(interp_spec)

                            # Limit background size for efficiency
                            if len(all_background_data) >= 100:
                                break

                        if len(all_background_data) >= 100:
                            break

                    if len(all_background_data) < 10:
                        # Create synthetic background if still not enough
                        console_log(
                            "Warning: Very few background samples. Creating synthetic data.")
                        n_synthetic = max(20, 50 - len(all_background_data))
                        for _ in range(n_synthetic):
                            synthetic_spec = spectrum + \
                                np.random.normal(
                                    0, spectrum.std() * 0.1, len(spectrum))
                            all_background_data.append(synthetic_spec)

                    ML_PROPERTY.X_train = np.array(all_background_data)

                    # Create corresponding labels
                    available_labels = ['benign', 'cancer']
                    if hasattr(ML_PROPERTY, 'metadata') and 'labels' in ML_PROPERTY.metadata:
                        available_labels = ML_PROPERTY.metadata['labels']
                    ML_PROPERTY.y_train = np.random.choice(
                        available_labels, len(all_background_data))

                    background_data_source = "synthetic_from_test_spectra"
                    console_log(
                        f"Created background dataset with {len(ML_PROPERTY.X_train)} samples")

                else:
                    raise ValueError(
                        "No background data available for SHAP explanation")

                # Get SHAP explanation
                shap_result = self.shap_explain(
                    show_plots=show_shap_plots,
                    max_display=max_display,
                    nsamples=nsamples,
                    fast_mode=fast_mode,
                    max_test_samples=1,
                    max_background_samples=min(50, len(ML_PROPERTY.X_train)),
                    reduce_features=False,
                    max_features=single_spectrum_data.shape[1],
                    use_base_estimator=kwargs.get('use_base_estimator', True),
                    force_kernel_explainer=kwargs.get(
                        'force_kernel_explainer', False),
                    shap_output_mode=kwargs.get('shap_output_mode', 'auto'),
                    kernel_nsamples_multiplier=kwargs.get(
                        'kernel_nsamples_multiplier', 1),
                )

                # Add background data source info to result
                if shap_result and shap_result.get("success"):
                    shap_result["background_data_source"] = background_data_source

                # Restore original data
                if original_X_test is not None:
                    ML_PROPERTY.X_test = original_X_test
                else:
                    ML_PROPERTY.X_test = None

                if original_y_test is not None:
                    ML_PROPERTY.y_test = original_y_test
                else:
                    ML_PROPERTY.y_test = None

                if original_X_train is not None:
                    ML_PROPERTY.X_train = original_X_train
                else:
                    ML_PROPERTY.X_train = None

                if original_y_train is not None:
                    ML_PROPERTY.y_train = original_y_train
                else:
                    ML_PROPERTY.y_train = None

                return shap_result

            except Exception as e:
                console_log(
                    f"Error in SHAP explanation for spectrum {spectrum_idx+1}: {e}")

                # Restore original data even on error
                if 'original_X_test' in locals() and original_X_test is not None:
                    ML_PROPERTY.X_test = original_X_test
                if 'original_y_test' in locals() and original_y_test is not None:
                    ML_PROPERTY.y_test = original_y_test
                if 'original_X_train' in locals() and original_X_train is not None:
                    ML_PROPERTY.X_train = original_X_train
                if 'original_y_train' in locals() and original_y_train is not None:
                    ML_PROPERTY.y_train = original_y_train

                return None

        def _process_shap_values(shap_result, model):
            """Process SHAP values and extract top contributors."""
            if not shap_result or not shap_result.get("success"):
                return None, None, None

            shap_values_raw = shap_result["shap_values"]

            # Extract SHAP values based on model type
            if hasattr(model, '_model') and isinstance(model._model, SVC):
                if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                    sv_positive = shap_values_raw[1]
                    if isinstance(sv_positive, np.ndarray):
                        if sv_positive.ndim == 1:
                            shap_values = sv_positive
                        elif sv_positive.ndim == 2 and sv_positive.shape[0] == 1:
                            shap_values = sv_positive[0]
                        else:
                            shap_values = sv_positive[0] if sv_positive.shape[0] > 0 else sv_positive
                    else:
                        shap_values = np.asarray(sv_positive)
                else:
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
                    if len(shap_values_raw) == 2:
                        if isinstance(shap_values_raw[1], np.ndarray):
                            if shap_values_raw[1].ndim > 1:
                                shap_values = shap_values_raw[1][0]
                            else:
                                shap_values = shap_values_raw[1]
                        else:
                            shap_values = shap_values_raw[1]
                    else:
                        if isinstance(shap_values_raw[0], np.ndarray):
                            if shap_values_raw[0].ndim > 1:
                                shap_values = shap_values_raw[0][0]
                            else:
                                shap_values = shap_values_raw[0]
                        else:
                            shap_values = shap_values_raw[0]
                else:
                    if hasattr(shap_values_raw, 'values'):
                        shap_values_array = shap_values_raw.values
                        if shap_values_array.ndim == 3:
                            if shap_values_array.shape[2] == 2:
                                shap_values = shap_values_array[0, :, 1]
                            else:
                                shap_values = shap_values_array[0, :, 0]
                        elif shap_values_array.ndim == 2:
                            shap_values = shap_values_array[0]
                        else:
                            shap_values = shap_values_array
                    elif hasattr(shap_values_raw, '__getitem__'):
                        shap_values = shap_values_raw[0]
                    else:
                        shap_values = shap_values_raw

            # Ensure shap_values is proper format
            if hasattr(shap_values, 'ndim') and shap_values.ndim > 1:
                shap_values = shap_values.flatten()

            if not isinstance(shap_values, np.ndarray):
                shap_values = np.asarray(shap_values)

            shap_values = shap_values.astype(float)

            # Process common axis
            if hasattr(ML_PROPERTY, 'common_axis') and ML_PROPERTY.common_axis is not None:
                try:
                    if isinstance(ML_PROPERTY.common_axis, (np.ndarray, list, tuple)):
                        common_axis_length = len(ML_PROPERTY.common_axis)
                        common_axis_subset = np.asarray(
                            ML_PROPERTY.common_axis, dtype=float)
                    else:
                        common_axis_length = len(shap_values)
                        common_axis_subset = np.arange(
                            len(shap_values), dtype=float)
                except (TypeError, ValueError) as e:
                    common_axis_length = len(shap_values)
                    common_axis_subset = np.arange(
                        len(shap_values), dtype=float)
            else:
                common_axis_length = len(shap_values)
                common_axis_subset = np.arange(len(shap_values), dtype=float)

            if not isinstance(common_axis_subset, np.ndarray):
                common_axis_subset = np.asarray(
                    common_axis_subset, dtype=float)
            else:
                common_axis_subset = common_axis_subset.astype(float)

            if len(shap_values) != common_axis_length:
                min_length = min(len(shap_values), common_axis_length)
                shap_values = shap_values[:min_length]
                if len(common_axis_subset) >= min_length:
                    common_axis_subset = common_axis_subset[:min_length]
                else:
                    common_axis_subset = np.arange(min_length, dtype=float)

            return shap_values, common_axis_subset, shap_result

        def _extract_top_contributors(shap_values, common_axis_subset):
            """Extract top positive and negative contributors."""
            sorted_idx = np.argsort(shap_values)
            top_negative_idx = sorted_idx[:5]
            top_positive_idx = sorted_idx[-5:][::-1]

            # Safe extraction function
            def safe_extract_to_float(arr_input, idx_input):
                try:
                    current_idx = int(idx_input.item() if hasattr(
                        idx_input, 'item') else idx_input)
                    val = arr_input[current_idx]
                    if hasattr(val, 'item'):
                        return float(val.item())
                    if isinstance(val, (int, float)):
                        return float(val)
                    if isinstance(val, np.ndarray):
                        if val.size == 1:
                            return float(val.flat[0])
                        else:
                            return float(val.flat[0])
                    if isinstance(val, list):
                        if len(val) == 1 and isinstance(val[0], (int, float)):
                            return float(val[0])
                        elif len(val) > 0 and isinstance(val[0], (int, float)):
                            return float(val[0])
                        else:
                            return 0.0
                    return float(val)
                except Exception:
                    return 0.0

            try:
                top_negative = [(safe_extract_to_float(common_axis_subset, idx),
                                 safe_extract_to_float(shap_values, idx))
                                for idx in top_negative_idx]

                top_positive = [(safe_extract_to_float(common_axis_subset, idx),
                                 safe_extract_to_float(shap_values, idx))
                                for idx in top_positive_idx]
            except Exception as e:
                console_log(f"Error with safe extraction method: {e}")
                top_positive = [(0.0, 0.0)] * 5
                top_negative = [(0.0, 0.0)] * 5

            return top_positive, top_negative

        def _add_lime_explanation_data(spec_result, lime_results, spec_info):
            """Add LIME explanation data if available."""
            if lime_results and lime_results["success"]:
                # Find corresponding LIME explanation for this spectrum
                for lime_exp in lime_results["explanation_objects"]:
                    if lime_exp["sample_idx"] == spec_info["spectrum_idx"]:
                        spec_result["lime_explanation"] = {
                            "sample_idx": lime_exp["sample_idx"],
                            "true_label": lime_exp["true_label"],
                            "predicted_label": lime_exp["predicted_label"],
                            "confidence": lime_exp["confidence"],
                            "explanation_available": True
                        }
                        break
            return spec_result

        def _create_summary_statistics(results):
            """Create summary statistics for the inspection results."""
            if results:
                correct_predictions = sum(
                    1 for r in results if r["prediction"]["is_correct"])
                accuracy = correct_predictions / len(results)

                summary = {
                    "total_spectra_analyzed": len(results),
                    "correct_predictions": correct_predictions,
                    "accuracy": accuracy,
                    "class_filter": class_to_inspect,
                    "random_seed": seed,
                    "shap_explanations_generated": sum(1 for r in results if r["shap_explanation"] is not None),
                    "lime_explanations_generated": sum(1 for r in results if r["lime_explanation"] is not None),
                    "lime_results_summary": None
                }
            else:
                summary = {
                    "total_spectra_analyzed": 0,
                    "error": "No spectra could be analyzed successfully"
                }

            return summary

        # === MAIN EXECUTION ===
        try:
            # 1. Validate and prepare ML_PROPERTY
            actual_model, ml_type = _validate_and_prepare_ml_property()

            # 2. Prepare test data
            all_spectra = _prepare_test_data()

            # 3. Filter and select spectra
            selected_spectra = _filter_and_select_spectra(all_spectra)
            if isinstance(selected_spectra, dict) and "results" in selected_spectra:
                return selected_spectra  # Early return for error case

            console_log(f"Inspecting {len(selected_spectra)} spectra...")

            # 4. Generate LIME explanations if requested
            lime_results = _generate_lime_explanations(selected_spectra)

            # 5. Get unified prediction function
            unified_predict = _get_unified_prediction_function(actual_model)

            results = []

            # 6. Process each spectrum
            for spectrum_idx, spec_info in enumerate(selected_spectra):
                console_log(
                    f"\nAnalyzing spectrum {spectrum_idx+1}/{len(selected_spectra)}...")

                spectrum = spec_info["spectrum"]

                # 7. Make prediction
                pred_label, confidence, processed_spectrum = _predict_single_spectrum(
                    spectrum, spec_info, actual_model, unified_predict)

                if pred_label is None:
                    continue

                # 8. Initialize spectrum result
                spec_result = {
                    "spectrum_info": spec_info,
                    "prediction": {
                        "label": pred_label,
                        "confidence": float(confidence) if confidence is not None else None,
                        "is_correct": pred_label == spec_info["true_label"]
                    },
                    "shap_explanation": None,
                    "lime_explanation": None
                }

                # 9. Get SHAP explanation if requested
                if show_plots:
                    shap_result = _get_shap_explanation(
                        processed_spectrum, spec_info, spectrum_idx)

                    if shap_result:
                        # Process SHAP values
                        shap_values, common_axis_subset, shap_result_processed = _process_shap_values(
                            shap_result, ML_PROPERTY)

                        if shap_values is not None:
                            # Extract top contributors
                            top_positive, top_negative = _extract_top_contributors(
                                shap_values, common_axis_subset)

                            # Generate visualizations
                            spectrum_with_highlights_spectrum(
                                processed_spectrum, common_axis_subset, top_positive, top_negative,
                                positive_label, negative_label, spec_info, pred_label,
                                confidence, spectrum_idx)

                            create_shap_plots(
                                spectrum_idx, spec_info, shap_values, common_axis_subset,
                                top_positive, top_negative, positive_label, negative_label)

                            create_enhanced_table(self,
                                                  spectrum_idx, spec_info, pred_label, shap_result_processed, shap_values,
                                                  top_positive, top_negative, positive_label, negative_label,
                                                  confidence)

                            # Store SHAP results
                            spec_result["shap_explanation"] = {
                                "top_positive_contributors": top_positive,
                                "top_negative_contributors": top_negative,
                                "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else list(shap_values)
                            }

                # 10. Add LIME explanation data if available
                spec_result = _add_lime_explanation_data(
                    spec_result, lime_results, spec_info)

                results.append(spec_result)

                # 11. console_log summary for this spectrum
                console_log(
                    f"Spectrum {spectrum_idx+1} Summary [Container: {spec_info['container_idx']}, Index: {spec_info['spectrum_idx']}]:")
                console_log(f"  True Label: {spec_info['true_label']}")
                console_log(
                    f"  Predicted: {pred_label} (Confidence: {confidence:.3f})")
                console_log(
                    f"  Correct: {'Yes' if pred_label == spec_info['true_label'] else 'No'}")

                if spec_result["shap_explanation"]:
                    top_pos = spec_result["shap_explanation"]["top_positive_contributors"][0]
                    top_neg = spec_result["shap_explanation"]["top_negative_contributors"][0]
                    console_log(
                        f"  Top {positive_label} contributor: {top_pos[0]:.1f} cm⁻¹ (SHAP: {top_pos[1]:.4f})")
                    console_log(
                        f"  Top {negative_label} contributor: {top_neg[0]:.1f} cm⁻¹ (SHAP: {top_neg[1]:.4f})")

                if spec_result["lime_explanation"]:
                    console_log(f"  LIME explanation: Available")

            # 12. Create summary statistics
            summary = _create_summary_statistics(results)
            if lime_results:
                summary["lime_results_summary"] = lime_results

            return {
                "results": results,
                "summary": summary,
                "lime_full_results": lime_results,  # Include full LIME results for reference
                "ml_type_used": ml_type  # Include the ML type that was used
            }

        except Exception as e:
            console_log(f"Error in inspect_spectra: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def plot_container_distribution(self,
                                    spectral_containers: List[rp.SpectralContainer],
                                    container_labels: List[str],
                                    **kwargs) -> dict:
        """
        Method version for RamanVisualizer class.

        Usage:
        ------
        visualizer = RamanVisualizer()
        results = visualizer.plot_container_distribution(
            spectral_containers=[chumDF_benign["processed"], chumDF_cancer["processed"], ...],
            container_labels=['CHUM_benign', 'CHUM_cancer', ...],
            sample_limit=1000
        )
        """
        return plot_institution_distribution(spectral_containers, container_labels, **kwargs)


def spectrum_with_highlights_spectrum(
        spectrum, common_axis_subset, top_positive, top_negative,
        positive_label, negative_label, spec_info, pred_label,
        confidence: Union[int, float] = 0, spectrum_idx: int = -1) -> plt:
    """
    Original spectrum with highlighted regions

    Args:
        spectrum (np.ndarray): The Raman spectrum data.
        common_axis_subset (np.ndarray): The common wavenumber axis subset.
        top_positive (list): List of tuples with top positive contributors (wavenumber, SHAP value).
        top_negative (list): List of tuples with top negative contributors (wavenumber, SHAP value).
        positive_label (str): Label for positive contributors.
        negative_label (str): Label for negative contributors.
        spec_info (dict): Dictionary containing spectrum information like container index, spectrum index, and true label.
        pred_label (str): Predicted label for the spectrum.
        confidence (Union[int, float]): Confidence score for the prediction.
        spectrum_idx (int): Index of the spectrum in the dataset.

    Returns:
        plt: The matplotlib figure object containing the plot.
    """
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(common_axis_subset, spectrum[:len(common_axis_subset)],
             'k-', linewidth=2, label='Spectrum')

    # Highlight top positive and negative regions
    # Ensure top_positive and top_negative are not empty before accessing [0][0]
    if top_positive:
        for wn, sv in top_positive:
            plt.axvline(x=wn, color='red', alpha=0.6,
                        linestyle='--', linewidth=1.5, label=f'{positive_label} Contributors' if wn == top_positive[0][0] else "")
    if top_negative:
        for wn, sv in top_negative:
            plt.axvline(x=wn, color='blue', alpha=0.6,
                        linestyle='--', linewidth=1.5, label=f'{negative_label} Contributors' if wn == top_negative[0][0] else "")

    # Enhanced title with spectrum index information
    # Ensure confidence is scalar before formatting
    confidence_scalar = confidence
    if hasattr(confidence_scalar, 'item'):
        confidence_scalar = confidence_scalar.item()
    elif isinstance(confidence_scalar, (np.ndarray, list)) and len(confidence_scalar) == 1:
        confidence_scalar = confidence_scalar[0]
        if hasattr(confidence_scalar, 'item'):
            confidence_scalar = confidence_scalar.item()

    try:
        confidence_float = float(confidence_scalar)
    except (TypeError, ValueError):
        console_log(
            f"Warning: Could not convert confidence '{confidence_scalar}' to float. Using 0.0.")
        confidence_float = 0.0

    title_text = (f"Spectrum {spectrum_idx+1} [Container: {spec_info['container_idx']}, "
                  f"Index: {spec_info['spectrum_idx']}] - "
                  f"True: {spec_info['true_label']}, Predicted: {pred_label} "
                  f"(Confidence: {confidence_float:.3f})")  # Use confidence_float

    plt.title(title_text, fontsize=12, fontweight='bold')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig1


def create_shap_plots(spectrum_idx, spec_info, shap_values, common_axis_subset, top_positive, top_negative, positive_label, negative_label) -> plt:
    """
    Helper function to create SHAP plots.

    Args:
        spectrum_idx (int): Index of the spectrum in the dataset.
        spec_info (dict): Dictionary containing spectrum information like container index, spectrum index, and true label.
        shap_values (np.ndarray): SHAP values for the spectrum.
        common_axis_subset (np.ndarray): The common wavenumber axis subset.
        top_positive (list): List of tuples with top positive contributors (wavenumber, SHAP value).
        top_negative (list): List of tuples with top negative contributors (wavenumber, SHAP value).
        positive_label (str): Label for positive contributors.
        negative_label (str): Label for negative contributors.

    Returns:
        plt: The matplotlib figure object containing the SHAP plot.

    """
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
    return fig2


def create_enhanced_table(self: RamanVisualizer,
                          spectrum_idx, spec_info, pred_label, shap_result, shap_values,
                          top_positive, top_negative, positive_label, negative_label,
                          confidence: Union[int, float] = 0
                          ) -> plt:
    """
    Enhanced Top contributors table with peak assignments

    Args:
        self (RamanVisualizer): Instance of the RamanVisualizer class.
        spectrum_idx (int): Index of the spectrum in the dataset.
        spec_info (dict): Dictionary containing spectrum information like container index, spectrum index, and true label.
        pred_label (str): Predicted label for the spectrum.
        shap_result (dict): SHAP result dictionary containing expected value and other metadata.
        shap_values (np.ndarray): SHAP values for the spectrum.
        top_positive (list): List of tuples with top positive contributors (wavenumber, SHAP value).
        top_negative (list): List of tuples with top negative contributors (wavenumber, SHAP value).
        positive_label (str): Label for positive contributors.
        negative_label (str): Label for negative contributors.
        confidence (Union[int, float]): Confidence score for the prediction.

    Returns:
        plt: The matplotlib figure object containing the enhanced table.

    """
    positive_label = positive_label.upper()
    negative_label = negative_label.upper()

    fig3 = plt.figure(figsize=(14, 12))
    plt.axis('off')

    # Calculate prediction mathematics
    total_shap_sum = np.sum(shap_values)
    expected_val = shap_result.get("expected_value", 0)
    if isinstance(expected_val, (list, np.ndarray)):
        expected_val = float(expected_val[0]) if len(expected_val) > 0 else 0.0
    else:
        expected_val = float(expected_val)

    prediction_score = expected_val + total_shap_sum

    # Calculate contribution percentages
    positive_sum = sum(sv for _, sv in top_positive)
    negative_sum = sum(sv for _, sv in top_negative)
    total_abs_contribution = abs(positive_sum) + abs(negative_sum)

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
        except Exception as e:
            console_log(
                f"Error in get_peak_info for {wavenumber}: {e} \n {traceback.format_exc()}")
            return {"assignment": e, "peak": wavenumber}

    # Create enhanced table data with 4 columns
    table_data = []

    # Header section with mathematical explanation
    table_data.append(["MATHEMATICAL PREDICTION BREAKDOWN", "", "", ""])
    table_data.append(
        ["Base Model Expectation", f"{expected_val:.4f}", "", ""])
    table_data.append(["Total SHAP Contribution",
                      f"{total_shap_sum:.4f}", "", ""])
    table_data.append(
        ["Final Prediction Score", f"{prediction_score:.4f}", "", ""])
    table_data.append(["Predicted Class", pred_label,
                      f"({confidence:.3f})", ""])
    table_data.append(["", "", "", ""])  # Empty row

    # positive_label contributors section with peak assignments
    table_data.append([f"Top {positive_label} Contributors",
                      "Wavenumber", "SHAP Value", "Peak Assignment"])

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
    table_data.append([f"Top {negative_label} Contributors",
                      "Wavenumber", "SHAP Value", "Peak Assignment"])

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

    table_data.append(["DECISION LOGIC & PEAK ANALYSIS", "", "", ""])
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
    table.set_fontsize(9)  # Slightly smaller font to fit more content
    table.scale(1, 1.8)

    # UPDATED: Enhanced styling with fixed coloring for positive (red) and negative (blue)
    for table_row_idx, row in enumerate(table_data):
        for j in range(4):  # Now 4 columns
            cell = table[(table_row_idx, j)]

            # Header sections
            if "BREAKDOWN" in str(row[0]) or "DECISION LOGIC" in str(row[0]) or "BIOLOGICAL CONTEXT" in str(row[0]):
                cell.set_facecolor('#E8E8E8')  # Light gray for headers
                cell.set_text_props(weight='bold', color='black')
                cell.set_height(0.08)

            # FIXED: Always use consistent coloring for positive_label (cancer) and negative_label (benign)
            # Positive label header (cancer)
            elif f"Top {positive_label}" in str(row[0]):
                cell.set_facecolor('#FFE6E6')  # Light red for positive header
                cell.set_text_props(
                    weight='bold', color='#8B0000')  # Dark red text

            # Negative label header (benign)
            elif f"Top {negative_label}" in str(row[0]):
                cell.set_facecolor('#E6F3FF')  # Light blue for negative header
                cell.set_text_props(
                    weight='bold', color='#000080')  # Dark blue text

            # Individual positive contributor rows
            elif str(row[0]).startswith("#") and table_row_idx > 7 and table_row_idx < len(table_data) and "Top " + positive_label in table_data[table_row_idx-int(row[0][1])][0]:
                cell.set_facecolor('#FFF0F0')  # Very light red background

            # Individual negative contributor rows
            elif str(row[0]).startswith("#") and table_row_idx > 15:
                cell.set_facecolor('#F0F8FF')  # Very light blue background

            # Total rows with special emphasis
            elif f"{positive_label} Total:" in str(row[0]):
                cell.set_facecolor('#FFE6E6')  # Light red for total
                cell.set_text_props(
                    weight='bold', color='#8B0000')  # Dark red text

            elif f"{negative_label} Total:" in str(row[0]):
                cell.set_facecolor('#E6F3FF')  # Light blue for total
                cell.set_text_props(
                    weight='bold', color='#000080')  # Dark blue text

            # Net direction row
            elif "Net SHAP Direction" in str(row[0]):
                color = '#FFE6E6' if net_direction == positive_label else '#E6F3FF'
                text_color = '#8B0000' if net_direction == positive_label else '#000080'
                cell.set_facecolor(color)
                if j == 1 or j == 2:  # For the arrow and value columns
                    cell.set_text_props(weight='bold', color=text_color)

            # Key decision factor
            elif "Key Decision Factor" in str(row[0]):
                # Color based on whether most_influential_sv is positive or negative
                color = '#FFF0F0' if most_influential_sv > 0 else '#F0F8FF'
                text_color = '#8B0000' if most_influential_sv > 0 else '#000080'
                cell.set_facecolor(color)
                if j == 2:  # SHAP value column
                    cell.set_text_props(color=text_color)

            # Peak assignment column styling
            if j == 3 and str(row[j]) and str(row[j]) not in ["", "Peak Assignment"]:
                cell.set_text_props(style='italic', size=8)

            # Mathematical values
            if j in [1, 2] and any(char in str(row[j]) for char in ['±', '(', ':', 'Δ']):
                cell.set_text_props(family='monospace', size=8)

    plt.suptitle(f"ENHANCED PREDICTION ANALYSIS WITH PEAK ASSIGNMENTS\nSpectrum {spectrum_idx+1} [Container: {spec_info['container_idx']}, Index: {spec_info['spectrum_idx']}]",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig3


def plot_institution_distribution(
    spectral_containers: List[rp.SpectralContainer],
    container_labels: List[str],
    container_names: List[str] = None,
    sample_limit: int = 500,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    figsize: tuple = (12, 8),
    alpha: float = 0.6,
    point_size: int = 50,
    title: str = "Institution Distribution in Feature Space",
    show_legend: bool = True,
    save_plot: bool = False,
    save_path: str = None,
    color_palette: str = 'tab10',
    interpolate_to_common_axis: bool = True,
    common_axis: np.ndarray = None,
    show_class_info: bool = True,
    class_labels: List[str] = None
) -> dict:
    """
    Plot t-SNE visualization of spectral data distribution across different containers/institutions.

    Parameters:
    -----------
    spectral_containers : List[rp.SpectralContainer]
        List of SpectralContainer objects to compare
    container_labels : List[str]
        Labels for each container (e.g., ['CHUM_benign', 'CHUM_cancer', 'UHN_benign', ...])
    container_names : List[str], optional
        Institution/group names for each container. If None, extracted from container_labels
    sample_limit : int
        Maximum number of samples to use for t-SNE (for performance)
    perplexity : int
        t-SNE perplexity parameter
    n_iter : int
        Number of iterations for t-SNE
    random_state : int
        Random seed for reproducibility
    figsize : tuple
        Figure size (width, height)
    alpha : float
        Point transparency
    point_size : int
        Size of scatter plot points
    title : str
        Plot title
    show_legend : bool
        Whether to show legend
    save_plot : bool
        Whether to save the plot
    save_path : str
        Path to save the plot (if save_plot=True)
    color_palette : str
        Matplotlib colormap name
    interpolate_to_common_axis : bool
        Whether to interpolate all spectra to a common axis
    common_axis : np.ndarray, optional
        Common axis for interpolation. If None, uses the first container's axis
    show_class_info : bool
        Whether to show class information in the legend
    class_labels : List[str], optional
        Class labels (e.g., ['benign', 'cancer']) for extracting class info

    Returns:
    --------
    dict : Results dictionary containing embedded data, labels, and metadata
    """
    try:
        console_log("🔬 Starting spectral distribution analysis...")

        # Validate inputs
        if len(spectral_containers) != len(container_labels):
            raise ValueError(
                f"Number of containers ({len(spectral_containers)}) must match number of labels ({len(container_labels)})")

        # Extract institution names if not provided
        if container_names is None:
            container_names = []
            for label in container_labels:
                # Extract institution name (everything before first underscore)
                inst_name = label.split('_')[0] if '_' in label else label
                container_names.append(inst_name)

        # Get unique institutions
        unique_institutions = list(set(container_names))
        console_log(
            f"📊 Found {len(unique_institutions)} unique institutions: {unique_institutions}")

        # Determine common axis for interpolation
        if interpolate_to_common_axis:
            if common_axis is None:
                # Use the axis from the container with the most features
                max_features = 0
                best_axis = None
                for container in spectral_containers:
                    if len(container.spectral_axis) > max_features:
                        max_features = len(container.spectral_axis)
                        best_axis = container.spectral_axis
                common_axis = best_axis
                console_log(
                    f"🔧 Using common axis with {len(common_axis)} features")
            else:
                console_log(
                    f"🔧 Using provided common axis with {len(common_axis)} features")

        # Collect all spectral data
        all_data = []
        all_labels = []
        all_institutions = []
        all_classes = []
        container_info = []

        for i, (container, label, inst_name) in enumerate(zip(spectral_containers, container_labels, container_names)):
            if container.spectral_data is None or len(container.spectral_data) == 0:
                console_log(
                    f"⚠️  Warning: Container {i} ({label}) has no spectral data, skipping...")
                continue

            # Extract class information if available
            class_info = "unknown"
            if class_labels:
                for class_label in class_labels:
                    if class_label.lower() in label.lower():
                        class_info = class_label
                        break

            console_log(
                f"📈 Processing container {i+1}/{len(spectral_containers)}: {label} ({len(container.spectral_data)} spectra)")

            # Process each spectrum in the container
            for spectrum_idx, spectrum in enumerate(container.spectral_data):
                try:
                    # Interpolate to common axis if needed
                    if interpolate_to_common_axis and len(container.spectral_axis) != len(common_axis):
                        interpolated_spectrum = np.interp(
                            common_axis, container.spectral_axis, spectrum)
                        all_data.append(interpolated_spectrum)
                    else:
                        all_data.append(spectrum)

                    all_labels.append(label)
                    all_institutions.append(inst_name)
                    all_classes.append(class_info)
                    container_info.append({
                        'container_idx': i,
                        'spectrum_idx': spectrum_idx,
                        'label': label,
                        'institution': inst_name,
                        'class': class_info
                    })

                except Exception as e:
                    console_log(
                        f"⚠️  Warning: Error processing spectrum {spectrum_idx} in container {i}: {e}")
                    continue

        if not all_data:
            raise ValueError("No valid spectral data found in any container")

        # Convert to numpy array
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        all_institutions = np.array(all_institutions)
        all_classes = np.array(all_classes)

        console_log(
            f"📊 Total collected data: {all_data.shape[0]} spectra with {all_data.shape[1]} features")

        # Sample data if too large
        if len(all_data) > sample_limit:
            console_log(
                f"🎲 Sampling {sample_limit} spectra from {len(all_data)} total for t-SNE performance")
            indices = np.random.choice(
                len(all_data), sample_limit, replace=False)
            all_data = all_data[indices]
            all_labels = all_labels[indices]
            all_institutions = all_institutions[indices]
            all_classes = all_classes[indices]
            container_info = [container_info[i] for i in indices]

        # Perform t-SNE
        console_log(
            f"🧮 Running t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
        tsne = TSNE(n_components=2, perplexity=perplexity,
                    n_iter=n_iter, random_state=random_state, verbose=1)
        embedded = tsne.fit_transform(all_data)

        # Create the plot
        plt.figure(figsize=figsize)

        # Get colors
        cmap = plt.get_cmap(color_palette)
        colors = [cmap(i/len(unique_institutions))
                  for i in range(len(unique_institutions))]
        institution_colors = {inst: colors[i]
                              for i, inst in enumerate(unique_institutions)}

        # Plot by institution
        for inst in unique_institutions:
            mask = all_institutions == inst

            # Count classes for this institution if class info is available
            class_counts = {}
            if show_class_info and class_labels:
                for class_label in class_labels:
                    class_count = np.sum((all_institutions == inst) & (
                        all_classes == class_label))
                    if class_count > 0:
                        class_counts[class_label] = class_count

            # Create legend label
            if class_counts:
                class_info_str = ", ".join(
                    [f"{cls}:{cnt}" for cls, cnt in class_counts.items()])
                legend_label = f"{inst} ({class_info_str})"
            else:
                legend_label = f"{inst} (n={np.sum(mask)})"

            plt.scatter(embedded[mask, 0], embedded[mask, 1],
                        label=legend_label,
                        alpha=alpha,
                        s=point_size,
                        color=institution_colors[inst])

        # Customize plot
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)

        if show_legend:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot if requested
        if save_plot:
            if save_path is None:
                save_path = f"institution_distribution_tsne_{random_state}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            console_log(f"💾 Plot saved to: {save_path}")

        plt.show()

        # Calculate and display statistics
        console_log("\n📈 Distribution Statistics:")
        console_log("=" * 50)
        for inst in unique_institutions:
            mask = all_institutions == inst
            count = np.sum(mask)
            percentage = (count / len(all_institutions)) * 100
            console_log(f"{inst}: {count} spectra ({percentage:.1f}%)")

            if show_class_info and class_labels:
                for class_label in class_labels:
                    class_count = np.sum((all_institutions == inst) & (
                        all_classes == class_label))
                    if class_count > 0:
                        class_percentage = (class_count / count) * 100
                        console_log(
                            f"  └─ {class_label}: {class_count} ({class_percentage:.1f}%)")

        # Return results
        results = {
            'success': True,
            'embedded_data': embedded,
            'institutions': all_institutions,
            'labels': all_labels,
            'classes': all_classes,
            'container_info': container_info,
            'unique_institutions': unique_institutions,
            'tsne_params': {
                'perplexity': perplexity,
                'n_iter': n_iter,
                'random_state': random_state
            },
            'data_info': {
                'total_spectra': len(all_data),
                'n_features': all_data.shape[1],
                'sampled': len(all_data) < len(container_info)
            }
        }

        console_log(f"\n✅ Analysis completed successfully!")
        return results

    except Exception as e:
        console_log(f"❌ Error in plot_institution_distribution: {e}")
        import traceback
        traceback.print_exc()
