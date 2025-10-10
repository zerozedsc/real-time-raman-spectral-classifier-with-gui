# Enhanced Raman Preprocessing Implementation Documentation

## Overview

This document provides comprehensive documentation for the enhanced `preprocess.py` implementation in the Real-time Raman Spectral Classifier application. The module provides a complete, medical-grade preprocessing pipeline specifically designed for Raman spectroscopy data used in disease detection and diagnostic applications.

## Architecture

### Core Components

1. **Preprocessing Classes** - Individual method implementations
2. **RamanPipeline** - Sequential processing pipeline
3. **EnhancedRamanPipeline** - Pipeline with progress tracking
4. **PreprocessingStepRegistry** - Dynamic method registry for UI integration

## Preprocessing Categories & Methods

### 1. Cosmic Ray & Spike Removal

Critical for removing measurement artifacts that can interfere with diagnostic accuracy.

#### `Gaussian` Class
- **Purpose**: MAD-based cosmic ray spike detection and removal
- **Method**: Gaussian filter smoothing with statistical outlier detection
- **Parameters**:
  - `kernel` (int): Gaussian kernel standard deviation (default: 5)
  - `threshold` (float): MAD-based detection threshold (default: 3.0)
- **Medical Application**: Essential for tissue measurements where cosmic rays can mimic disease biomarkers

#### `MedianDespike` Class ⭐ **NEW**
- **Purpose**: Alternative cosmic ray removal using median filtering
- **Method**: Median filter-based spike detection with MAD threshold
- **Parameters**:
  - `kernel_size` (int): Median filter kernel size (default: 5)
  - `threshold` (float): MAD-based threshold (default: 3.0)
- **Advantages**: More robust for narrow, high-intensity spikes

### 2. Wavenumber & Intensity Calibration ⭐ **NEW CATEGORY**

Essential for ensuring measurement accuracy and reproducibility across instruments.

#### `WavenumberCalibration` Class ⭐ **NEW**
- **Purpose**: Correct systematic wavenumber axis errors using reference peaks
- **Method**: Polynomial correction based on known vs measured peak positions
- **Parameters**:
  - `reference_peaks` (dict): Reference peak positions {name: wavenumber} (default: {"Si": 520.5})
  - `poly_order` (int): Polynomial correction order (default: 3)
- **Medical Application**: Critical for multi-instrument studies and longitudinal patient monitoring

#### `IntensityCalibration` Class
- **Purpose**: Correct instrument wavelength-dependent response
- **Method**: Reference standard-based intensity correction
- **Parameters**:
  - `reference` (array): Known reference spectrum
- **Medical Application**: Ensures consistent intensity measurements for quantitative biomarker analysis

### 3. Denoising & Smoothing

Improves signal-to-noise ratio while preserving diagnostic spectral features.

#### `SavGol` (Savitzky-Golay)
- **Purpose**: Polynomial-based smoothing with feature preservation
- **Parameters**: `window_length`, `polyorder`
- **Medical Application**: Standard for biological samples where peak shape is critical

#### `Whittaker`
- **Purpose**: Penalized least squares smoothing
- **Parameters**: `lam` (smoothing), `d` (difference order)
- **Medical Application**: Excellent for noisy tissue spectra

#### `MovingAverage` ⭐ **ENHANCED**
- **Purpose**: Simple moving average smoothing
- **Parameters**: `window_length` (default: 15)
- **Medical Application**: Quick noise reduction for real-time applications

#### `Gaussian`, `Kernel`
- **Purpose**: Various kernel-based smoothing approaches
- **Medical Application**: Customizable smoothing for different tissue types

### 4. Baseline Correction

Critical for removing fluorescence background that can mask disease biomarkers.

#### Least Squares Methods
- **ASLS, IASLS, AIRPLS, ARPLS, DRPLS, IARPLS, ASPLS**
- **Purpose**: Asymmetric least squares baseline removal
- **Medical Application**: Removes tissue autofluorescence while preserving Raman peaks

#### Polynomial Methods
- **Poly, ModPoly, PenalisedPoly, IModPoly**
- **Purpose**: Polynomial baseline fitting and subtraction
- **Medical Application**: Simple baseline removal for well-behaved samples

#### Advanced Methods
- **MultiScaleConv1D** ⭐ **ENHANCED**
  - **Purpose**: Multi-scale convolutional baseline correction
  - **Parameters**: `kernel_sizes`, `weights`, `mode`, `iterations`
  - **Medical Application**: Effective for complex tissue backgrounds

- **Transformer1DBaseline** (PyTorch-based)
  - **Purpose**: Deep learning-based baseline correction
  - **Medical Application**: Adaptive baseline removal for challenging samples

### 5. Derivative Processing ⭐ **NEW CATEGORY**

Enhances peak resolution and removes baseline variations.

#### `Derivative` Class ⭐ **NEW**
- **Purpose**: Calculate 1st and 2nd order spectral derivatives
- **Method**: Savitzky-Golay derivative calculation
- **Parameters**:
  - `order` (int): Derivative order (1 or 2)
  - `window_length` (int): Filter window length (default: 5)
  - `polyorder` (int): Polynomial order (default: 2)
- **Medical Applications**:
  - **1st Derivative**: Peak detection, baseline removal
  - **2nd Derivative**: Overlapping peak separation, enhanced resolution

### 6. Normalization

Ensures consistent intensity scales for machine learning and multi-sample comparison.

#### `Vector`
- **Purpose**: L1, L2, or max normalization
- **Medical Application**: Standard preprocessing for classification algorithms

#### `MinMax`
- **Purpose**: Scale intensities to specified range
- **Medical Application**: Ensures consistent dynamic range across samples

#### `SNV` (Standard Normal Variate)
- **Purpose**: Mean-centering and variance normalization
- **Medical Application**: Compensates for scattering differences between tissue samples

#### `MSC` (Multiplicative Scatter Correction) ⭐ **NEW**
- **Purpose**: Correct multiplicative scattering effects
- **Method**: Linear regression-based scatter correction
- **Medical Application**: Essential for biological samples with varying scattering properties
- **Features**:
  - Fit-transform pattern for training/application
  - Automatic mean spectrum calculation
  - Handles both single spectra and batch processing

#### `AUC` (Area Under Curve)
- **Purpose**: Normalize by total spectral area
- **Medical Application**: Concentration-independent measurements

## Pipeline Integration

### Dynamic Registry System

The `PreprocessingStepRegistry` provides dynamic method discovery and parameter management:

```python
# Registry categories with new additions
{
    "cosmic_ray_removal": {
        "WhitakerHayes": {...},
        "Gaussian": {...},          # Enhanced
        "MedianDespike": {...}      # NEW
    },
    "calibration": {                # NEW CATEGORY
        "WavenumberCalibration": {...},  # NEW
        "IntensityCalibration": {...}
    },
    "derivatives": {                # NEW CATEGORY
        "Derivative": {...}         # NEW
    },
    "denoising": {
        "SavGol": {...},
        "Whittaker": {...},
        "Kernel": {...},
        "Gaussian": {...},
        "MovingAverage": {...}      # Enhanced
    },
    "baseline_correction": {
        # ... all existing methods ...
        "MultiScaleConv1D": {...}   # Enhanced
    },
    "normalisation": {
        # ... all existing methods ...
        "MSC": {...}                # NEW
    },
    "miscellaneous": {
        "Cropper": {...},
        "BackgroundSubtractor": {...}
    }
}
```

### Parameter Specifications

Each method includes comprehensive parameter information for UI generation:

```python
"param_info": {
    "parameter_name": {
        "type": "int|float|choice|bool|scientific|tuple|list|dict|optional",
        "range": [min, max],
        "step": increment,
        "choices": [...],
        "description": "User-friendly description"
    }
}
```

## Medical Applications & Benefits

### Disease Detection Workflow

1. **Calibration**: Ensure accurate wavenumber and intensity measurements
2. **Spike Removal**: Remove cosmic ray artifacts that could mimic biomarkers
3. **Baseline Correction**: Remove tissue autofluorescence background
4. **Derivative Processing**: Enhance overlapping disease biomarker peaks
5. **Normalization**: Prepare data for machine learning classification

### Clinical Advantages

#### **Reproducibility**
- Wavenumber calibration ensures consistent measurements across instruments
- Standardized processing pipelines enable multi-center studies

#### **Sensitivity**
- Advanced baseline correction reveals weak disease biomarkers
- Derivative processing separates overlapping peaks

#### **Specificity**
- MSC normalization reduces false positives from scattering variations
- Systematic artifact removal improves diagnostic accuracy

#### **Real-time Capability**
- Optimized algorithms enable point-of-care applications
- Progress tracking provides user feedback during processing

## Implementation Features

### Error Handling & Logging
- Comprehensive error handling with medical-grade logging
- Graceful fallbacks for missing dependencies
- Parameter validation with user-friendly error messages

### Performance Optimization
- Vectorized operations for large datasets
- Memory-efficient processing for real-time applications
- Optional GPU acceleration (PyTorch methods)

### Extensibility
- Modular architecture for easy method addition
- Registry-based system for dynamic UI generation
- Standardized parameter interface

## Code Quality & Standards

### Documentation
- Comprehensive docstrings for all methods
- Parameter descriptions with medical context
- Usage examples for clinical applications

### Testing & Validation
- Parameter validation for all methods
- Error handling for edge cases
- Logging for debugging and audit trails

### Dependencies
- Core functionality uses standard libraries (NumPy, SciPy)
- Optional advanced features (PyTorch for deep learning methods)
- Graceful degradation when dependencies are missing

## Usage Examples

### Basic Pipeline
```python
from functions.preprocess import PREPROCESSING_REGISTRY

# Create pipeline steps
pipeline_steps = [
    PREPROCESSING_REGISTRY.create_method_instance(
        "calibration", "WavenumberCalibration", 
        {"reference_peaks": {"Si": 520.5}}
    ),
    PREPROCESSING_REGISTRY.create_method_instance(
        "cosmic_ray_removal", "MedianDespike",
        {"kernel_size": 5, "threshold": 3.0}
    ),
    PREPROCESSING_REGISTRY.create_method_instance(
        "baseline_correction", "ASPLS",
        {"lam": 1e6, "p": 0.01}
    ),
    PREPROCESSING_REGISTRY.create_method_instance(
        "normalisation", "MSC", {}
    ),
    PREPROCESSING_REGISTRY.create_method_instance(
        "derivatives", "Derivative",
        {"order": 1, "window_length": 5}
    )
]
```

### Medical Diagnostic Pipeline
```python
# Optimized for tissue classification
medical_pipeline = [
    WavenumberCalibration(reference_peaks={"Si": 520.5}),  # Calibrate axis
    MedianDespike(kernel_size=5, threshold=3.0),           # Remove spikes  
    ASPLS(lam=1e6, p=0.01),                               # Remove fluorescence
    MSC(),                                                 # Correct scattering
    Derivative(order=1, window_length=7, polyorder=3),    # Enhance peaks
    Vector(pixelwise=False)                               # Normalize for ML
]
```

## Future Enhancements

### Planned Additions
1. **Peak Alignment**: Spectral registration for multi-sample studies
2. **EMSC**: Extended Multiplicative Scatter Correction
3. **OSC**: Orthogonal Signal Correction
4. **Advanced Derivatives**: Higher-order and custom derivative operators

### Research Integration
- Integration with latest Raman spectroscopy research
- Collaboration with medical institutions for method validation
- Continuous improvement based on clinical feedback

## Conclusion

The enhanced preprocess.py implementation provides a comprehensive, medical-grade preprocessing pipeline for Raman spectroscopy applications. With the addition of calibration methods, derivative processing, and advanced normalization techniques, the system now covers all essential preprocessing categories required for robust disease detection and diagnostic applications.

The modular architecture, comprehensive error handling, and dynamic registry system ensure that the implementation is both powerful and user-friendly, suitable for both research applications and clinical deployment.
