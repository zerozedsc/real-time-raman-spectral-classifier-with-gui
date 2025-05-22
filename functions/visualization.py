import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from numpy import trapz

import numpy as np
import pandas as pd
import seaborn as sns


class RamanVisualizer:
    """
    A class to visualize Raman spectra data.

    Attributes:
    -----------
    df : pandas DataFrame
        DataFrame containing Raman data with wavenumber column and intensity columns
    """

    def __init__(self, df):
        self.df = df

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
        spectral_data: np.ndarray,
        spectral_axis: np.ndarray,
        title: str = "PCA of Raman Spectra",
        figsize: tuple = (12, 6),
        xlim: tuple = None,
        ylim: tuple = None,
        legend: bool = True,
        sample_names: list = None,
        legend_loc: str = 'best',
        sample_limit: int = 10,
        cmap: str = 'viridis',
    ) -> plt:
        """
        Perform PCA on the spectral data and visualize the first two principal components.

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
        """

        # Check inputs
        if spectral_data.shape[1] != len(spectral_axis):
            if spectral_data.shape[0] == len(spectral_axis):
                spectral_data = spectral_data.T
            else:
                raise ValueError(
                    "Spectral data shape does not match spectral axis.")

        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(spectral_data)

        # Limit samples if needed
        n_samples = pca_result.shape[0]
        plot_samples = min(n_samples, sample_limit)

        if sample_names is None or len(sample_names) != n_samples:
            sample_names = [f"Sample {i}" for i in range(plot_samples)]

        # Auto-assign colors to classes
        unique_labels = list(sorted(set(sample_names)))
        if isinstance(cmap, str):
            base_colors = plt.get_cmap(cmap, len(unique_labels))
            color_map = {label: base_colors(i)
                         for i, label in enumerate(unique_labels)}
        elif isinstance(cmap, dict):
            color_map = cmap
        else:
            # fallback to Set1 if cmap is None or invalid
            base_colors = plt.get_cmap("Set1", len(unique_labels))
            color_map = {label: base_colors(i)
                         for i, label in enumerate(unique_labels)}

        # Plot setup
        plt.clf()
        plt.close('all')
        fig = plt.figure(num=1, figsize=figsize, clear=True)

        shown_labels = set()
        for i in range(plot_samples):
            label = sample_names[i]
            color = color_map.get(label, 'gray')
            if label not in shown_labels:
                plt.scatter(pca_result[i, 0], pca_result[i, 1],
                            color=color, label=label, alpha=0.7)
                shown_labels.add(label)
            else:
                plt.scatter(pca_result[i, 0],
                            pca_result[i, 1], color=color, alpha=0.7)

        # Axes and legend
        plt.title(title)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, alpha=0.3)

        if legend:
            plt.legend(loc=legend_loc)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)

        plt.tight_layout()
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
            plt.tight_layout()

        return per_class_accuracy, ax
