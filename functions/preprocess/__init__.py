"""
Enhanced Raman Preprocessing Package

A comprehensive preprocessing package for Raman spectroscopy data,
specifically designed for medical applications and disease detection.

This package provides:
- Cosmic ray and spike removal
- Wavenumber and intensity calibration  
- Baseline correction methods
- Derivative processing
- Normalization techniques
- Dynamic pipeline building
- Registry system for UI integration
"""

# Import all preprocessing classes and functions
from .spike_removal import Gaussian, MedianDespike
from .calibration import WavenumberCalibration, IntensityCalibration
from .normalization import SNV, MSC, MovingAverage
from .baseline import (
    BaselineCorrection, MultiScaleConv1D, Transformer1DBaseline,
    LightweightTransformer1D
)
from .derivatives import Derivative
from .pipeline import RamanPipeline, EnhancedRamanPipeline
from .registry import PreprocessingStepRegistry

# Create global registry instance
PREPROCESSING_REGISTRY = PreprocessingStepRegistry()

# Export all main classes and the registry
__all__ = [
    # Spike removal
    'Gaussian', 'MedianDespike',
    
    # Calibration
    'WavenumberCalibration', 'IntensityCalibration',
    
    # Normalization
    'SNV', 'MSC', 'MovingAverage',
    
    # Baseline correction
    'BaselineCorrection', 'MultiScaleConv1D', 'Transformer1DBaseline',
    'LightweightTransformer1D',
    
    # Derivatives
    'Derivative',
    
    # Pipeline
    'RamanPipeline', 'EnhancedRamanPipeline',
    
    # Registry
    'PreprocessingStepRegistry', 'PREPROCESSING_REGISTRY'
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Raman Spectroscopy Team"
__description__ = "Medical-grade Raman spectroscopy preprocessing package"
