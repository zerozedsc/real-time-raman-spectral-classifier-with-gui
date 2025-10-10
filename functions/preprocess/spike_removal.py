"""
Cosmic Ray and Spike Removal Methods

This module contains methods for removing cosmic ray spikes and other 
artifacts from Raman spectra.
"""

import numpy as np
try:
    import ramanspy as rp
    from scipy.ndimage import gaussian_filter1d, median_filter
    from scipy.stats import median_abs_deviation as median_absolute_deviation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from ..utils import create_logs
except ImportError:
    # Fallback logging function
    def create_logs(log_id, source, message, status='info'):
        print(f"[{status.upper()}] {source}: {message}")


class Gaussian:
    """
    A callable class to remove cosmic ray spikes from Raman spectra using a
    Gaussian filter-based approach.

    This method works by smoothing the spectrum with a Gaussian filter and
    identifying points that deviate significantly from the smoothed version.
    The deviation is measured against the noise level, estimated using the
    Median Absolute Deviation (MAD).

    Attributes:
        kernel (int): The standard deviation (sigma) for the Gaussian kernel.
                      This controls the degree of smoothing. Must be an odd integer.
        threshold (float): The number of standard deviations a point must be
                           from the smoothed spectrum to be considered a spike.
    """

    def __init__(self, kernel: int = 5, threshold: float = 3.0):
        """
        Initializes the Gaussian despike processor.

        Args:
            kernel (int): The size of the Gaussian kernel. It's used as the
                          standard deviation (sigma) for the filter.
                          A larger kernel results in more smoothing.
                          Defaults to 5.
            threshold (float): The modified Z-score threshold for spike detection.
                               Defaults to 3.0.
        """
        if not isinstance(kernel, int) or kernel <= 0:
            raise ValueError("Kernel size must be a positive integer")
        if not isinstance(threshold, (float, int)) or threshold <= 0:
            raise ValueError("Threshold must be a positive number")
            
        self.kernel = kernel
        self.threshold = threshold

    def __call__(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian despiking to a 2D numpy array of spectra.
        This makes the class compatible with scikit-learn pipelines.

        Args:
            spectra (np.ndarray): A 2D numpy array where each row is a spectrum.

        Returns:
            np.ndarray: The despiked spectra.
        """
        if spectra.ndim != 2:
            raise ValueError("Input spectra must be a 2D array")
        
        # Apply the despiking function along each row (each spectrum)
        return np.apply_along_axis(self._despike_spectrum, 1, spectra)

    def apply(self, spectra: 'rp.SpectralContainer') -> 'rp.SpectralContainer':
        """
        Apply Gaussian despiking to a ramanspy SpectralContainer.

        Args:
            spectra (SpectralContainer): The ramanspy container holding the spectral data.

        Returns:
            SpectralContainer: A new container with the despiked data.
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy and ramanspy are required for SpectralContainer operations")
            
        data = spectra.spectral_data
        
        # Handle both single (1D) and multiple (2D) spectra
        if data.ndim == 1:
            despiked_data = self._despike_spectrum(data)
        else:
            despiked_data = np.apply_along_axis(self._despike_spectrum, 1, data)

        return rp.SpectralContainer(despiked_data, spectra.spectral_axis)

    def _despike_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Core logic to remove spikes from a single 1D spectrum.

        Args:
            spectrum (np.ndarray): A 1D numpy array representing a single spectrum.

        Returns:
            np.ndarray: The despiked spectrum.
        """
        if spectrum is None or spectrum.size == 0:
            return spectrum

        # Create a copy to avoid modifying the original data
        despiked = np.copy(spectrum)
        
        if not SCIPY_AVAILABLE:
            # Simple fallback without scipy
            return despiked
        
        # 1. Smooth the spectrum using a Gaussian filter
        # The `sigma` of the filter is controlled by our `kernel` parameter.
        smoothed_spectrum = gaussian_filter1d(despiked, sigma=self.kernel)

        # 2. Calculate the difference (residual)
        residual = despiked - smoothed_spectrum

        # 3. Estimate noise using Median Absolute Deviation (MAD) for robustness
        # The factor 0.6745 makes MAD an unbiased estimator for the standard deviation
        # for normally distributed data.
        mad = np.median(np.abs(residual - np.median(residual)))
        if mad < 1e-9:
            return despiked

        # 4. Calculate the modified Z-score for each point
        # This score tells us how many standard deviations away each point is.
        modified_z_score = 0.6745 * residual / mad

        # 5. Identify spike locations
        # Spikes are points where the absolute Z-score exceeds the threshold.
        spike_indices = np.where(np.abs(modified_z_score) > self.threshold)[0]

        # 6. Replace spikes with the value from the smoothed spectrum
        # This is a simple and effective way to correct the spike.
        for i in spike_indices:
            despiked[i] = smoothed_spectrum[i]

        return despiked


class MedianDespike:
    """
    Median filter-based cosmic ray removal for Raman spectra.
    
    This method uses median filtering to identify and remove cosmic ray spikes.
    It's particularly effective for narrow, high-intensity spikes that are
    characteristic of cosmic ray events.
    
    Attributes:
        kernel_size (int): Size of median filter kernel
        threshold (float): Threshold for spike detection (in MAD units)
    """
    
    def __init__(self, kernel_size: int = 5, threshold: float = 3.0):
        """
        Initialize MedianDespike processor.
        
        Args:
            kernel_size (int): Size of median filter kernel (default: 5)
            threshold (float): Threshold for spike detection in MAD units (default: 3.0)
        """
        if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd integer")
        if not isinstance(threshold, (float, int)) or threshold <= 0:
            raise ValueError("Threshold must be a positive number")
            
        self.kernel_size = kernel_size
        self.threshold = threshold
    
    def __call__(self, spectra: np.ndarray) -> np.ndarray:
        """Apply median despiking to numpy array format (for sklearn pipelines)."""
        if spectra.ndim != 2:
            raise ValueError("Input spectra must be a 2D array")
        
        return np.apply_along_axis(self._despike_spectrum, 1, spectra)
    
    def _despike_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Remove spikes from a single spectrum using median filtering.
        
        Args:
            spectrum (np.ndarray): 1D spectrum to despike
            
        Returns:
            np.ndarray: Despiked spectrum
        """
        if spectrum is None or spectrum.size == 0:
            return spectrum
            
        if not SCIPY_AVAILABLE:
            # Simple fallback without scipy
            return spectrum.copy()
        
        # Create a copy to avoid modifying the original
        despiked = spectrum.copy()
        
        # Apply median filter
        filtered = median_filter(despiked, size=self.kernel_size)
        
        # Calculate residual
        residual = despiked - filtered
        
        # Calculate MAD-based threshold
        mad = np.median(np.abs(residual - np.median(residual)))
        if mad < 1e-9:
            return despiked
            
        # Identify spikes
        modified_z_score = 0.6745 * residual / mad
        spike_indices = np.where(np.abs(modified_z_score) > self.threshold)[0]
        
        # Replace spikes with filtered values
        despiked[spike_indices] = filtered[spike_indices]
        
        return despiked
    
    def apply(self, spectra: 'rp.SpectralContainer') -> 'rp.SpectralContainer':
        """Apply median despiking to ramanspy SpectralContainer."""
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy and ramanspy are required for SpectralContainer operations")
            
        data = spectra.spectral_data
        
        # Handle both 1D and 2D data
        if data.ndim == 1:
            despiked_data = self._despike_spectrum(data)
        else:
            despiked_data = np.apply_along_axis(self._despike_spectrum, 1, data)
            
        return rp.SpectralContainer(despiked_data, spectra.spectral_axis)
