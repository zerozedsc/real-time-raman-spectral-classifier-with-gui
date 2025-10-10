"""
Preprocessing Step Registry for Dynamic UI Generation

This module contains the comprehensive registry of all available preprocessing
methods organized by category, with parameter specifications for UI generation.
"""

from typing import Any, Dict, List
try:
    import ramanspy as rp
    RAMANSPY_AVAILABLE = True
except ImportError:
    RAMANSPY_AVAILABLE = False

try:
    from ..utils import create_logs
except ImportError:
    # Fallback logging function
    def create_logs(log_id, source, message, status='info'):
        print(f"[{status.upper()}] {source}: {message}")

from .spike_removal import Gaussian, MedianDespike
from .calibration import WavenumberCalibration, IntensityCalibration
from .normalization import SNV, MSC, MovingAverage
from .baseline import MultiScaleConv1D
from .derivatives import Derivative
from .advanced_normalization import QuantileNormalization, RankTransform, ProbabilisticQuotientNormalization
from .feature_engineering import PeakRatioFeatures
from .advanced_baseline import ButterworthHighPass
from .kernel_denoise import Kernel
from .background_subtraction import BackgroundSubtractor

# Try to import deep learning module (requires PyTorch)
try:
    from .deep_learning import ConvolutionalAutoencoder
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False


class PreprocessingStepRegistry:
    """Registry for managing all available preprocessing steps with their parameters."""
    
    def __init__(self):
        self._steps = self._build_step_registry()
        
    def _build_step_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive registry of preprocessing steps organized by category."""
        registry = {
            "miscellaneous": {},
            "calibration": {},
            "denoising": {},
            "cosmic_ray_removal": {},
            "baseline_correction": {},
            "derivatives": {},
            "normalisation": {},
        }

        # Add ramanspy methods if available
        if RAMANSPY_AVAILABLE:
            ramanspy_methods = self._build_ramanspy_methods()
            for category, methods in ramanspy_methods.items():
                registry[category].update(methods)
        
        # Add custom methods (always available)
        custom_methods = self._build_custom_methods()
        for category, methods in custom_methods.items():
            registry[category].update(methods)
        
        return registry

    def _build_ramanspy_methods(self) -> Dict[str, Dict[str, Any]]:
        """Build registry entries for ramanspy methods."""
        return {
            "miscellaneous": {
                "Cropper": {
                    "class": rp.preprocessing.misc.Cropper,
                    "default_params": {"region": (800, 1800)},
                    "param_info": {
                        "region": {"type": "tuple", "range": [(400, 4000)], "description": "Wavenumber range to extract (start, end)"}
                    },
                    "description": "Crop the intensity values and wavenumber axis to the specified range"
                },
                "BackgroundSubtractor": {
                    "class": BackgroundSubtractor,
                    "default_params": {"background": None},
                    "param_info": {
                        "background": {"type": "optional", "description": "Fixed reference background to subtract"}
                    },
                    "description": "Subtract a fixed reference background"
                }
            },
            
            "denoising": {
                "SavGol": {
                    "class": rp.preprocessing.denoise.SavGol,
                    "default_params": {"window_length": 7, "polyorder": 3},
                    "param_info": {
                        "window_length": {"type": "int", "range": [3, 99], "step": 2, "description": "Window length (must be odd)"},
                        "polyorder": {"type": "int", "range": [1, 10], "description": "Polynomial order"}
                    },
                    "description": "Denoising based on Savitzky-Golay filtering"
                },
                "Whittaker": {
                    "class": rp.preprocessing.denoise.Whittaker,
                    "default_params": {"lam": 1e5, "d": 2},
                    "param_info": {
                        "lam": {"type": "scientific", "range": [1e2, 1e12], "description": "Smoothing parameter"},
                        "d": {"type": "int", "range": [1, 4], "description": "Order of differences"}
                    },
                    "description": "Denoising based on Discrete Penalised Least Squares (Whittaker−Henderson smoothing)"
                },
                "Kernel": {
                    "class": Kernel,
                    "default_params": {"kernel_type": "uniform", "kernel_size": 7},
                    "param_info": {
                        "kernel_type": {"type": "choice", "choices": ["uniform", "gaussian", "triangular"], "description": "Type of kernel"},
                        "kernel_size": {"type": "int", "range": [3, 21], "step": 2, "description": "Size of kernel (must be odd)"}
                    },
                    "description": "Denoising based on kernel/window smoothers"
                },
                "Gaussian": {
                    "class": rp.preprocessing.denoise.Gaussian,
                    "default_params": {"sigma": 1.0, "order": 0},
                    "param_info": {
                        "sigma": {"type": "float", "range": [0.1, 5.0], "step": 0.1, "description": "Standard deviation for Gaussian kernel"},
                        "order": {"type": "int", "range": [0, 3], "description": "Order of the filter (0 = Gaussian, 1 = first derivative, etc.)"}
                    },
                    "description": "Denoising based on a Gaussian filter"
                }
            },

            "cosmic_ray_removal": {
                "WhitakerHayes": {
                    "class": rp.preprocessing.despike.WhitakerHayes,
                    "default_params": {"kernel_size": 7, "threshold": 8},
                    "param_info": {
                        "kernel_size": {"type": "int", "range": [3, 15], "step": 2, "description": "Kernel size for filtering"},
                        "threshold": {"type": "float", "range": [1, 20], "step": 0.1, "description": "Threshold for spike detection"}
                    },
                    "description": "Cosmic rays removal based on modified z-scores filtering"
                }
            },

            "baseline_correction": self._build_baseline_methods(),
            
            "normalisation": {
                "Vector": {
                    "class": rp.preprocessing.normalise.Vector,
                    "default_params": {"pixelwise": False},
                    "param_info": {
                        "pixelwise": {"type": "bool", "description": "Apply normalisation pixelwise (True) or spectrumwise (False)"}
                    },
                    "description": "Vector normalisation"
                },
                "MinMax": {
                    "class": rp.preprocessing.normalise.MinMax,
                    "default_params": {"pixelwise": False, "a": 0, "b": 1},
                    "param_info": {
                        "pixelwise": {"type": "bool", "description": "Apply normalisation pixelwise (True) or spectrumwise (False)"},
                        "a": {"type": "float", "range": [-10.0, 10.0], "description": "Minimum value for scaling"},
                        "b": {"type": "float", "range": [-10.0, 10.0], "description": "Maximum value for scaling"}
                    },
                    "description": "Min-max normalisation"
                },
                "MaxIntensity": {
                    "class": rp.preprocessing.normalise.MaxIntensity,
                    "default_params": {"pixelwise": False},
                    "param_info": {
                        "pixelwise": {"type": "bool", "description": "Apply normalisation pixelwise (True) or spectrumwise (False)"}
                    },
                    "description": "Max intensity normalisation"
                },
                "AUC": {
                    "class": rp.preprocessing.normalise.AUC,
                    "default_params": {"pixelwise": False},
                    "param_info": {
                        "pixelwise": {"type": "bool", "description": "Apply normalisation pixelwise (True) or spectrumwise (False)"}
                    },
                    "description": "Area under the curve normalisation"
                }
            }
        }

    def _build_baseline_methods(self) -> Dict[str, Dict[str, Any]]:
        """Build baseline correction methods registry."""
        return {
            # Least squares methods
            "ASLS": {
                "class": rp.preprocessing.baseline.ASLS,
                "default_params": {"lam": 1e6, "p": 0.01, "diff_order": 2, "max_iter": 50, "tol": 1e-6},
                "param_info": {
                    "lam": {"type": "scientific", "range": [1e2, 1e12], "description": "Smoothing parameter"},
                    "p": {"type": "float", "range": [0.001, 0.999], "step": 0.001, "description": "Asymmetry parameter"},
                    "diff_order": {"type": "int", "range": [1, 3], "description": "Difference order"},
                    "max_iter": {"type": "int", "range": [1, 1000], "description": "Maximum iterations"},
                    "tol": {"type": "scientific", "range": [1e-9, 1e-3], "description": "Convergence tolerance"}
                },
                "description": "Baseline correction based on Asymmetric Least Squares (AsLS)"
            },
            "IASLS": {
                "class": rp.preprocessing.baseline.IASLS,
                "default_params": {"lam": 1e6, "p": 0.01, "lam_1": 1e-4, "max_iter": 50, "tol": 1e-6},
                "param_info": {
                    "lam": {"type": "scientific", "range": [1e2, 1e12], "description": "Smoothing parameter"},
                    "p": {"type": "float", "range": [0.001, 0.999], "step": 0.001, "description": "Asymmetry parameter"},
                    "lam_1": {"type": "scientific", "range": [1e-6, 1e-2], "description": "Secondary smoothing parameter"},
                    "max_iter": {"type": "int", "range": [1, 1000], "description": "Maximum iterations"},
                    "tol": {"type": "scientific", "range": [1e-9, 1e-3], "description": "Convergence tolerance"}
                },
                "description": "Baseline correction based on Improved Asymmetric Least Squares (IAsLS)"
            },
            "AIRPLS": {
                "class": rp.preprocessing.baseline.AIRPLS,
                "default_params": {"lam": 1e6, "diff_order": 2, "max_iter": 50, "tol": 1e-6},
                "param_info": {
                    "lam": {"type": "scientific", "range": [1e2, 1e12], "description": "Smoothing parameter"},
                    "diff_order": {"type": "int", "range": [1, 3], "description": "Difference order"},
                    "max_iter": {"type": "int", "range": [1, 1000], "description": "Maximum iterations"},
                    "tol": {"type": "scientific", "range": [1e-9, 1e-3], "description": "Convergence tolerance"}
                },
                "description": "Baseline correction based on Adaptive Iteratively Reweighted Penalized Least Squares (airPLS)"
            },
            "ARPLS": {
                "class": rp.preprocessing.baseline.ARPLS,
                "default_params": {"lam": 1e5, "diff_order": 2, "max_iter": 50, "tol": 1e-6},
                "param_info": {
                    "lam": {"type": "scientific", "range": [1e2, 1e12], "description": "Smoothing parameter"},
                    "diff_order": {"type": "int", "range": [1, 3], "description": "Difference order"},
                    "max_iter": {"type": "int", "range": [1, 1000], "description": "Maximum iterations"},
                    "tol": {"type": "scientific", "range": [1e-9, 1e-3], "description": "Convergence tolerance"}
                },
                "description": "Baseline correction based on Asymmetrically Reweighted Penalized Least Squares (arPLS)"
            },
            "DRPLS": {
                "class": rp.preprocessing.baseline.DRPLS,
                "default_params": {"lam": 1e5, "eta": 0.5, "max_iter": 50, "tol": 1e-6},
                "param_info": {
                    "lam": {"type": "scientific", "range": [1e2, 1e12], "description": "Smoothing parameter"},
                    "eta": {"type": "float", "range": [0.1, 0.9], "step": 0.1, "description": "Reweighting parameter"},
                    "max_iter": {"type": "int", "range": [1, 1000], "description": "Maximum iterations"},
                    "tol": {"type": "scientific", "range": [1e-9, 1e-3], "description": "Convergence tolerance"}
                },
                "description": "Baseline correction based on Doubly Reweighted Penalized Least Squares (drPLS)"
            },
            "IARPLS": {
                "class": rp.preprocessing.baseline.IARPLS,
                "default_params": {"lam": 1e5, "diff_order": 2, "max_iter": 50, "tol": 1e-6},
                "param_info": {
                    "lam": {"type": "scientific", "range": [1e2, 1e12], "description": "Smoothing parameter"},
                    "diff_order": {"type": "int", "range": [1, 3], "description": "Difference order"},
                    "max_iter": {"type": "int", "range": [1, 1000], "description": "Maximum iterations"},
                    "tol": {"type": "scientific", "range": [1e-9, 1e-3], "description": "Convergence tolerance"}
                },
                "description": "Baseline correction based on Improved Asymmetrically Reweighted Penalized Least Squares (IarPLS)"
            },
            "ASPLS": {
                "class": rp.preprocessing.baseline.ASPLS,
                "default_params": {"lam": 1e6, "diff_order": 2, "max_iter": 50, "alpha": 0.95},
                "param_info": {
                    "lam": {"type": "scientific", "range": [1e2, 1e12], "description": "Smoothing parameter"},
                    "diff_order": {"type": "int", "range": [1, 3], "description": "Difference order"},
                    "max_iter": {"type": "int", "range": [1, 1000], "description": "Maximum iterations"},
                    "alpha": {"type": "float", "range": [0.5, 0.99], "step": 0.01, "description": "Adaptive parameter"}
                },
                "description": "Baseline correction based on Adaptive Smoothness Penalized Least Squares (asPLS)"
            },
            
            # Polynomial fitting methods
            "Poly": {
                "class": rp.preprocessing.baseline.Poly,
                "default_params": {"poly_order": 3, "regions": None},
                "param_info": {
                    "poly_order": {"type": "int", "range": [1, 10], "description": "Polynomial order"},
                    "regions": {"type": "optional", "description": "Regions for polynomial fitting (optional)"}
                },
                "description": "Baseline correction based on polynomial fitting"
            },
            "ModPoly": {
                "class": rp.preprocessing.baseline.ModPoly,
                "default_params": {"poly_order": 3, "tol": 0.001, "max_iter": 250},
                "param_info": {
                    "poly_order": {"type": "int", "range": [1, 10], "description": "Polynomial order"},
                    "tol": {"type": "float", "range": [0.0001, 0.1], "step": 0.0001, "description": "Tolerance for convergence"},
                    "max_iter": {"type": "int", "range": [1, 500], "description": "Maximum iterations"}
                },
                "description": "Baseline correction based on modified polynomial fitting"
            },
            "PenalisedPoly": {
                "class": rp.preprocessing.baseline.PenalisedPoly,
                "default_params": {"poly_order": 6, "tol": 0.001, "max_iter": 100},
                "param_info": {
                    "poly_order": {"type": "int", "range": [1, 15], "description": "Polynomial order"},
                    "tol": {"type": "float", "range": [0.0001, 0.1], "step": 0.0001, "description": "Tolerance for convergence"},
                    "max_iter": {"type": "int", "range": [1, 500], "description": "Maximum iterations"}
                },
                "description": "Baseline correction based on penalised polynomial fitting"
            },
            "IModPoly": {
                "class": rp.preprocessing.baseline.IModPoly,
                "default_params": {"poly_order": 3, "tol": 0.001, "max_iter": 200},
                "param_info": {
                    "poly_order": {"type": "int", "range": [1, 10], "description": "Polynomial order"},
                    "tol": {"type": "float", "range": [0.0001, 0.1], "step": 0.0001, "description": "Tolerance for convergence"},
                    "max_iter": {"type": "int", "range": [1, 500], "description": "Maximum iterations"}
                },
                "description": "Baseline correction based on improved modified polynomial fitting"
            },
            
            # Other methods
            "Goldindec": {
                "class": rp.preprocessing.baseline.Goldindec,
                "default_params": {"poly_order": 4, "tol": 0.001, "max_iter": 100},
                "param_info": {
                    "poly_order": {"type": "int", "range": [1, 10], "description": "Polynomial order"},
                    "tol": {"type": "float", "range": [0.0001, 0.1], "step": 0.0001, "description": "Tolerance for convergence"},
                    "max_iter": {"type": "int", "range": [1, 500], "description": "Maximum iterations"}
                },
                "description": "Baseline correction based on Goldindec"
            },
            "IRSQR": {
                "class": rp.preprocessing.baseline.IRSQR,
                "default_params": {"lam": 1e5, "quantile": 0.05, "max_iter": 50, "tol": 1e-6},
                "param_info": {
                    "lam": {"type": "scientific", "range": [1e2, 1e12], "description": "Smoothing parameter"},
                    "quantile": {"type": "float", "range": [0.01, 0.2], "step": 0.01, "description": "Quantile for regression"},
                    "max_iter": {"type": "int", "range": [1, 500], "description": "Maximum iterations"},
                    "tol": {"type": "scientific", "range": [1e-9, 1e-3], "description": "Convergence tolerance"}
                },
                "description": "Baseline correction based on Iterative Reweighted Spline Quantile Regression (IRSQR)"
            },
            "CornerCutting": {
                "class": rp.preprocessing.baseline.CornerCutting,
                "default_params": {"max_iter": 100},
                "param_info": {
                    "max_iter": {"type": "int", "range": [1, 500], "description": "Maximum iterations"}
                },
                "description": "Baseline correction based on Corner Cutting"
            },
            "FABC": {
                "class": rp.preprocessing.baseline.FABC,
                "default_params": {"lam": 1e6, "scale": 1.0, "num_std": 3.0, "max_iter": 50},
                "param_info": {
                    "lam": {"type": "scientific", "range": [1e2, 1e12], "description": "Smoothing parameter"},
                    "scale": {"type": "float", "range": [0.1, 5.0], "step": 0.1, "description": "Scale parameter"},
                    "num_std": {"type": "float", "range": [1.0, 10.0], "step": 0.5, "description": "Number of standard deviations"},
                    "max_iter": {"type": "int", "range": [1, 500], "description": "Maximum iterations"}
                },
                "description": "Baseline correction based on Fully automatic baseline correction (FABC)"
            }
        }

    def _build_custom_methods(self) -> Dict[str, Dict[str, Any]]:
        """Build registry entries for custom methods."""
        return {
            "cosmic_ray_removal": {
                "Gaussian": {
                    "class": Gaussian,
                    "default_params": {"kernel": 5, "threshold": 3.0},
                    "param_info": {
                        "kernel": {"type": "int", "range": [3, 15], "step": 2, "description": "Gaussian kernel size (standard deviation)"},
                        "threshold": {"type": "float", "range": [1.0, 10.0], "step": 0.1, "description": "MAD-based threshold for spike detection"}
                    },
                    "description": "Cosmic ray removal using Gaussian filter and MAD-based detection"
                },
                "MedianDespike": {
                    "class": MedianDespike,
                    "default_params": {"kernel_size": 5, "threshold": 3.0},
                    "param_info": {
                        "kernel_size": {"type": "int", "range": [3, 15], "step": 2, "description": "Median filter kernel size"},
                        "threshold": {"type": "float", "range": [1.0, 10.0], "step": 0.1, "description": "MAD-based threshold for spike detection"}
                    },
                    "description": "Cosmic ray removal using median filtering"
                }
            },
            
            "baseline_correction": {
                "MultiScaleConv1D": {
                    "class": MultiScaleConv1D,
                    "default_params": {"kernel_sizes": [5, 11, 21, 41], "weights": None, "mode": "reflect", "iterations": 1},
                    "param_info": {
                        "kernel_sizes": {"type": "list", "description": "List of kernel sizes for multi-scale convolution"},
                        "weights": {"type": "optional", "description": "Weights for combining different scales (optional)"},
                        "mode": {"type": "choice", "choices": ["reflect", "constant", "nearest", "mirror", "wrap"], "description": "Boundary condition for convolution"},
                        "iterations": {"type": "int", "range": [1, 10], "description": "Number of correction iterations"}
                    },
                    "description": "Multi-scale convolutional baseline correction"
                },
                "ButterworthHighPass": {
                    "class": ButterworthHighPass,
                    "default_params": {"cutoff_freq": 0.01, "filter_order": 3, "validate_peaks": True},
                    "param_info": {
                        "cutoff_freq": {"type": "float", "range": [0.001, 0.4], "step": 0.001, "description": "Normalized cutoff frequency (0 < fc < 0.5)"},
                        "filter_order": {"type": "int", "range": [1, 6], "description": "Filter order (higher = steeper rolloff)"},
                        "validate_peaks": {"type": "bool", "description": "Warn if peak areas change significantly"}
                    },
                    "description": "Digital Butterworth high-pass filter for baseline removal with smooth phase response"
                }
            },
            
            "calibration": {
                "WavenumberCalibration": {
                    "class": WavenumberCalibration,
                    "default_params": {"reference_peaks": {"Si": 520.5}, "poly_order": 3},
                    "param_info": {
                        "reference_peaks": {"type": "dict", "description": "Reference peak positions {name: wavenumber}"},
                        "poly_order": {"type": "int", "range": [1, 5], "description": "Polynomial order for wavenumber correction"}
                    },
                    "description": "Wavenumber axis calibration using reference peaks"
                },
                "IntensityCalibration": {
                    "class": IntensityCalibration,
                    "default_params": {"reference": None},
                    "param_info": {
                        "reference": {"type": "optional", "description": "Reference standard spectrum for calibration"}
                    },
                    "description": "Intensity calibration using reference standards"
                }
            },
            
            "derivatives": {
                "Derivative": {
                    "class": Derivative,
                    "default_params": {"order": 1, "window_length": 5, "polyorder": 2},
                    "param_info": {
                        "order": {"type": "choice", "choices": [1, 2], "default": 1, "description": "Derivative order (1st or 2nd)"},
                        "window_length": {"type": "int", "range": [3, 21], "step": 2, "description": "Savitzky-Golay window length"},
                        "polyorder": {"type": "int", "range": [1, 6], "description": "Polynomial order for fitting"}
                    },
                    "description": "Spectral derivatives using Savitzky-Golay method"
                }
            },
            
            "denoising": {
                "MovingAverage": {
                    "class": MovingAverage,
                    "default_params": {"window_length": 15},
                    "param_info": {
                        "window_length": {"type": "int", "range": [3, 51], "step": 2, "description": "Window length for moving average"}
                    },
                    "description": "Simple moving average smoothing"
                }
            },
            
            "normalisation": {
                "SNV": {
                    "class": SNV,
                    "default_params": {},
                    "param_info": {},
                    "description": "Standard Normal Variate normalisation"
                },
                "MSC": {
                    "class": MSC,
                    "default_params": {},
                    "param_info": {},
                    "description": "Multiplicative Scatter Correction for scattering effects"
                },
                "QuantileNormalization": {
                    "class": QuantileNormalization,
                    "default_params": {"method": "median"},
                    "param_info": {
                        "method": {"type": "choice", "choices": ["median", "mean"], "default": "median", "description": "Aggregation method for reference quantiles"}
                    },
                    "description": "Quantile normalization for cross-platform distribution alignment (robust to domain shift)"
                },
                "RankTransform": {
                    "class": RankTransform,
                    "default_params": {"scale_range": (0, 1), "standardize": False},
                    "param_info": {
                        "scale_range": {"type": "tuple", "range": [(-1, 1)], "description": "Target range for scaled ranks (min, max)"},
                        "standardize": {"type": "bool", "description": "Standardize features after ranking"}
                    },
                    "description": "Rank transform for intensity-independent relative ordering (domain-shift robust)"
                },
                "ProbabilisticQuotientNormalization": {
                    "class": ProbabilisticQuotientNormalization,
                    "default_params": {},
                    "param_info": {},
                    "description": "Probabilistic Quotient Normalization for dilution correction in biological samples"
                }
            },
            
            "miscellaneous": {
                "Cropper": {
                    "class": rp.preprocessing.misc.Cropper,
                    "default_params": {"region": (800, 1800)},
                    "param_info": {
                        "region": {"type": "tuple", "range": [(400, 4000)], "description": "Wavenumber range to extract (start, end)"}
                    },
                    "description": "Crop the intensity values and wavenumber axis to the specified range"
                },
                "PeakRatioFeatures": {
                    "class": PeakRatioFeatures,
                    "default_params": {
                        "peak_positions": None,  # Uses default MGUS/MM peaks
                        "window_size": 10.0,
                        "extraction_method": "local_max",
                        "ratio_mode": "log",
                        "epsilon": 1e-10
                    },
                    "param_info": {
                        "window_size": {"type": "float", "range": [5.0, 50.0], "step": 1.0, "description": "Half-width of window around peak (cm⁻¹)"},
                        "extraction_method": {"type": "choice", "choices": ["local_max", "local_integral", "gaussian_fit"], "default": "local_max", "description": "Method for extracting peak intensity"},
                        "ratio_mode": {"type": "choice", "choices": ["simple", "log", "both"], "default": "log", "description": "Type of ratios to compute"},
                        "epsilon": {"type": "scientific", "range": [1e-12, 1e-6], "description": "Small constant to avoid division by zero"}
                    },
                    "description": "Peak-ratio feature engineering: dimensionless batch-invariant descriptors for MGUS/MM classification"
                }
            },
        }
        
        # Add deep learning methods if PyTorch is available
        if DL_AVAILABLE:
            custom_methods["denoising"] = custom_methods.get("denoising", {})
            custom_methods["denoising"]["ConvolutionalAutoencoder"] = {
                "class": ConvolutionalAutoencoder,
                "default_params": {
                    "input_size": None,  # Auto-detected
                    "latent_dim": 32,
                    "kernel_sizes": (7, 11, 15),
                    "tv_weight": 0.01,
                    "device": None
                },
                "param_info": {
                    "latent_dim": {"type": "int", "range": [8, 128], "description": "Latent space dimensionality (16-32 typical)"},
                    "tv_weight": {"type": "float", "range": [0.0, 0.1], "step": 0.001, "description": "Total variation regularization weight"}
                },
                "description": "Convolutional autoencoder for unified denoising/baseline correction (requires training on clean/noisy pairs)"
            }
        
        return custom_methods
    
    def get_categories(self) -> List[str]:
        """Get all available preprocessing categories."""
        return list(self._steps.keys())
    
    def get_methods_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all methods in a specific category."""
        return self._steps.get(category, {})
    
    def get_method_info(self, category: str, method: str) -> Dict[str, Any]:
        """Get information about a specific method."""
        return self._steps.get(category, {}).get(method, {})
    
    def create_method_instance(self, category: str, method: str, params: Dict[str, Any] = None) -> Any:
        """Create an instance of a preprocessing method with given parameters."""
        method_info = self.get_method_info(category, method)
        if not method_info:
            raise ValueError(f"Method {method} not found in category {category}")
        
        method_class = method_info["class"]
        final_params = method_info["default_params"].copy()
        if params:
            # Convert parameter types based on param_info
            param_info = method_info.get("param_info", {})
            converted_params = {}
            for key, value in params.items():
                if key in param_info:
                    param_type = param_info[key].get("type")
                    if param_type == "int":
                        converted_params[key] = int(value) if not isinstance(value, int) else value
                    elif param_type in ("float", "scientific"):
                        # scientific notation parameters are also floats
                        converted_params[key] = float(value) if not isinstance(value, (int, float)) else value
                    elif param_type == "list":
                        # Convert string representation to list of integers
                        if isinstance(value, str):
                            import ast
                            try:
                                converted_params[key] = ast.literal_eval(value)
                            except (ValueError, SyntaxError):
                                converted_params[key] = value
                        else:
                            converted_params[key] = value
                    elif param_type == "choice":
                        # For choice parameters, try to convert to the appropriate type
                        choices = param_info[key].get("choices", [])
                        if choices and isinstance(choices[0], int):
                            converted_params[key] = int(value) if not isinstance(value, int) else value
                        elif choices and isinstance(choices[0], float):
                            converted_params[key] = float(value) if not isinstance(value, float) else value
                        else:
                            converted_params[key] = value
                    else:
                        converted_params[key] = value
                else:
                    converted_params[key] = value
            final_params.update(converted_params)
        
        try:
            return method_class(**final_params)
        except Exception as e:
            create_logs("method_creation_error", "PreprocessingStepRegistry",
                       f"Error creating {method}: {e}", status='error')
            raise

    def get_all_methods(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get all methods organized by category."""
        return self._steps
