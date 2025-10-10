# Deep Analysis of RamanVisualizer Class

**Date**: 2025-10-01  
**File**: `functions/visualization/core.py`  
**Total Lines**: 4,405 lines  
**Class**: `RamanVisualizer`

## Executive Summary

The `RamanVisualizer` class is a **monolithic visualization and explainability engine** that combines:
1. **Raman peak database querying** (peak assignment lookups)
2. **Basic visualization** (raw spectra, processed spectra)
3. **Dimensionality reduction** (PCA with decision boundaries)
4. **Model evaluation** (confusion matrices)
5. **ML explainability** (SHAP with 10+ nested helper functions, LIME with helpers)
6. **Spectrum inspection** (deep dive into individual predictions)
7. **Statistical analysis** (t-SNE distribution plotting)

**Key Finding**: This is not just a "visualization" class - it's a **full-featured ML explainability framework** embedded in what appears to be a visualization module.

---

## Class Structure Analysis

### 1. Initialization & State Management

```python
def __init__(self, df, spectral_container, labels, common_axis, n_features, ML_PROPERTY)
```

**Purpose**: Flexible initialization supporting multiple data sources  
**Lines**: 40-62 (23 lines)

**State Variables**:
- `self.ML_PROPERTY`: Reference to ML model wrapper (RamanML or MLModel)
- `self.df`: Pandas DataFrame for raw data
- `self.spectral_container`: List of ramanspy SpectralContainer objects
- `self.labels`: Class labels
- `self.common_axis`: Shared wavenumber axis for interpolation
- `self.n_features`: Number of spectral features

**Design Issue**: Accepts both ML_PROPERTY (which contains axis/features) AND separate parameters. This dual initialization creates complexity.

---

### 2. Peak Database Methods (Domain-Specific Utility)

#### 2.1 `get_peak_assignment()`
- **Lines**: 64-187 (124 lines)
- **Purpose**: Query `raman_peaks.json` for peak assignments
- **Features**:
  - Exact peak lookup
  - Fuzzy matching with tolerance
  - Caching mechanism (`self._raman_peaks_cache`)
  - Error handling for missing files

#### 2.2 `get_multiple_peak_assignments()`
- **Lines**: 188-220 (33 lines)
- **Purpose**: Batch wrapper for `get_peak_assignment()`

#### 2.3 `find_peaks_in_range()`
- **Lines**: 221-273 (53 lines)
- **Purpose**: Find all peaks within wavenumber range

**Analysis**: These methods are **domain-specific utilities** that should be **separated** from visualization logic. They interact with external data (`raman_peaks.json`) and are reused across multiple visualization methods.

**Suggested Module**: `peak_assignment.py` or `raman_database.py`

---

### 3. Basic Visualization Methods

#### 3.1 `visualize_raman_spectra()`
- **Lines**: 274-354 (81 lines)
- **Purpose**: Plot raw spectra from DataFrame
- **Dependencies**: `self.df`

#### 3.2 `visualize_processed_spectra()`
- **Lines**: 355-458 (104 lines)
- **Purpose**: Plot preprocessed spectral data
- **Features**: Colormap support, mean spectrum overlay

#### 3.3 `extract_raman_characteristics()` (STATIC METHOD - BUG!)
- **Lines**: 459-520 (62 lines)
- **Purpose**: Extract peaks and AUC from single spectrum
- **Design Issue**: Declared as instance method but uses `@staticmethod` behavior
- **Dependencies**: scipy.signal.find_peaks, numpy.trapz

**Analysis**: These are **pure visualization functions** with minimal state dependency. They could be standalone functions or grouped in a simple `BasicPlots` class.

**Suggested Module**: `basic_plots.py`

---

### 4. PCA Visualization (Complex, Feature-Rich)

#### 4.1 `pca2d()`
- **Lines**: 521-933 (413 lines - **9.4% of entire file!**)
- **Purpose**: 2D PCA plot with decision boundaries
- **Features**:
  - Auto-detects data source from ML_PROPERTY
  - Handles DataFrame, numpy array, and SpectralContainer inputs
  - Pre-calculated decision boundary support
  - Centroid calculation and visualization
  - Class-specific coloring

**Complexity Breakdown**:
```python
# Data source detection: ~100 lines
# Data preprocessing: ~80 lines
# PCA computation: ~40 lines
# Decision boundary handling: ~120 lines
# Plotting: ~70 lines
```

**Key Dependencies**:
- `ML_PROPERTY.X_train`, `ML_PROPERTY.y_train`, `ML_PROPERTY.X_test`
- `ML_PROPERTY.pca_boundary_data` (pre-calculated)
- `ML_PROPERTY.common_axis`

**Analysis**: This method is **too complex** and tries to do too much:
1. Data source negotiation (DataFrame vs containers vs ML_PROPERTY)
2. PCA computation
3. Decision boundary rendering
4. Statistical reporting

**Suggested Refactoring**:
- Split into `_prepare_pca_data()`, `_compute_pca()`, `_plot_pca()`, `_add_decision_boundary()`
- Or create dedicated `PCAVisualizer` class

---

### 5. Model Evaluation

#### 5.1 `confusion_matrix_heatmap()`
- **Lines**: 934-1033 (100 lines)
- **Purpose**: Plot confusion matrix with per-class accuracy
- **Dependencies**: sklearn.metrics, seaborn
- **State**: Completely stateless (could be standalone function)

**Analysis**: **Low coupling**, easy to extract.

**Suggested Module**: `model_evaluation.py`

---

### 6. SHAP Explainability (MASSIVE, Nested, Complex)

#### 6.1 `shap_explain()` - MAIN METHOD
- **Lines**: 1034-1995 (**962 lines - 21.8% of entire file!!!**)
- **Purpose**: Generate SHAP explanations for model predictions
- **Architecture**: **10 nested helper functions** + main execution flow

**Nested Functions**:
1. `_validate_and_prepare_data()` (92 lines) - Data validation
2. `_detect_model_type_enhanced()` (138 lines) - Model type detection (CalibratedClassifierCV, SVC, RF)
3. `_optimize_data_for_performance()` (77 lines) - Feature selection, sampling
4. `_create_enhanced_shap_explainer()` (91 lines) - Explainer factory
5. `_create_kernel_fallback()` (58 lines) - KernelExplainer wrapper
6. `_create_kernel_explainer_for_svc()` (79 lines) - SVC-specific explainer
7. `_process_shap_values()` (142 lines) - SHAP value normalization
8. `_extract_top_features()` (68 lines) - Feature importance ranking
9. `_generate_plots()` (121 lines) - SHAP summary/importance plots
10. `_create_final_results()` (96 lines) - Result dictionary construction

**Total Nested Code**: ~962 lines (85% of method is nested functions!)

**Key Features**:
- Supports multiple model types (SVC, RandomForest, CalibratedClassifierCV)
- Multiple explainer strategies (TreeExplainer, LinearExplainer, KernelExplainer)
- Feature selection for performance
- K-means sampling for background data
- Fast mode with reduced samples

**Critical Dependencies**:
- `ML_PROPERTY._model` or `ML_PROPERTY.sklearn_model`
- `ML_PROPERTY.X_train`, `ML_PROPERTY.y_train`, `ML_PROPERTY.X_test`, `ML_PROPERTY.y_test`
- `ML_PROPERTY.common_axis`, `ML_PROPERTY.n_features_in`
- SHAP library

**Analysis**: This is **an entire SHAP framework** hiding inside a visualization class! The nested structure makes it hard to:
- Test individual components
- Reuse helper functions
- Understand control flow
- Debug failures

**Suggested Refactoring**:
- Extract to `shap_explainer.py` with proper class structure
- Convert nested functions to class methods
- Create `SHAPExplainerFactory` for model type detection
- Separate plotting logic from computation

---

### 7. LIME Explainability (Large, Feature-Rich)

#### 7.1 `lime_explain()`
- **Lines**: 1996-2419 (424 lines)
- **Purpose**: Generate LIME explanations for predictions
- **Features**:
  - Unified prediction function for all model types
  - Feature selection support
  - K-means sampling
  - Batch explanation generation

**Key Implementation Details**:
```python
# Lines 2120-2180: Unified prediction function (60 lines)
def unified_predict_fn(x_reduced):
    # Handles CalibratedClassifierCV, RF, SVC
    # Manages feature selection
    # Converts decision functions to probabilities
```

**Dependencies**:
- lime.lime_tabular
- ML_PROPERTY (same as SHAP)
- `_visualize_lime_explanation()` helper

#### 7.2 `_visualize_lime_explanation()`
- **Lines**: 2420-2613 (194 lines)
- **Purpose**: Create detailed LIME visualization with Raman-specific styling

#### 7.3 `_visualize_lime_comparison()`
- **Lines**: 2614-2808 (195 lines)
- **Purpose**: Compare feature importance across classes

**Analysis**: LIME methods are **more cohesive** than SHAP (no deep nesting), but still complex due to:
- Model type handling
- Feature selection integration
- Custom visualization

**Suggested Refactoring**:
- Extract to `lime_explainer.py`
- Share model prediction logic with SHAP (create `model_wrapper.py`)

---

### 8. Spectrum Inspection (Integration Method)

#### 8.1 `inspect_spectra()`
- **Lines**: 2809-3683 (**875 lines - 19.9% of entire file!**)
- **Purpose**: Comprehensive analysis of individual spectra
- **Architecture**: **9 nested helper functions** + main flow

**Nested Functions**:
1. `_validate_and_prepare_ml_property()` (25 lines)
2. `_prepare_test_data()` (120 lines)
3. `_filter_and_select_spectra()` (25 lines)
4. `_get_unified_prediction_function()` (42 lines)
5. `_get_model_class_names()` (15 lines)
6. `_predict_single_spectrum()` (20 lines)
7. `_generate_lime_explanations()` (38 lines)
8. `_get_shap_explanation()` (140 lines - **complex background data logic!**)
9. `_process_shap_values()` (98 lines)
10. `_extract_top_contributors()` (40 lines)
11. `_add_lime_explanation_data()` (10 lines)
12. `_create_summary_statistics()` (20 lines)

**What it does**:
1. Selects random or specific spectra
2. Makes predictions
3. Generates SHAP explanations (calls `shap_explain()`)
4. Generates LIME explanations (calls `lime_explain()`)
5. Creates visualizations (calls module-level functions below)
6. Compiles comprehensive reports

**Critical Integration Point**: This method **orchestrates** SHAP, LIME, peak assignment, and visualization. It's the **main entry point for end users**.

**Analysis**: This is a **high-level workflow method** that should stay with the main class, but its nested helper functions could be extracted to make it more readable.

---

### 9. Module-Level Helper Functions (Outside Class)

#### 9.1 `spectrum_with_highlights_spectrum()`
- **Lines**: 3704-3781 (78 lines)
- **Purpose**: Plot spectrum with SHAP-highlighted regions

#### 9.2 `create_shap_plots()`
- **Lines**: 3784-3823 (40 lines)
- **Purpose**: Bar chart of SHAP values

#### 9.3 `create_enhanced_table()`
- **Lines**: 3826-4146 (321 lines)
- **Purpose**: Create detailed prediction breakdown table with peak assignments
- **Calls**: `self.get_peak_assignment()` (uses RamanVisualizer instance)

#### 9.4 `plot_institution_distribution()`
- **Lines**: 4149-4405 (257 lines)
- **Purpose**: t-SNE visualization of spectral data across institutions
- **Note**: Also has a **class method wrapper** at line 3684

**Analysis**: These are **module-level functions** but some (`create_enhanced_table()`) still **need access to RamanVisualizer instance** for peak assignments. This creates a **circular dependency**.

**Suggested Refactoring**:
- Pass peak assignment function as parameter
- Or extract peak assignment to separate module

---

## Dependency Analysis

### Internal Dependencies

```
RamanVisualizer.__init__
    ├── self.ML_PROPERTY (RamanML or MLModel)
    ├── self.df (pandas DataFrame)
    ├── self.spectral_container (List[SpectralContainer])
    ├── self.common_axis (np.ndarray)
    └── self.labels (List[str])

get_peak_assignment
    └── self._raman_peaks_cache (instance variable)
    └── raman_peaks.json (external file)

pca2d
    ├── self.ML_PROPERTY.X_train, y_train, X_test, y_test
    ├── self.ML_PROPERTY.pca_boundary_data
    ├── self.ML_PROPERTY.common_axis
    └── self.ML_PROPERTY.n_features_in

shap_explain
    ├── self.ML_PROPERTY._model or .sklearn_model
    ├── self.ML_PROPERTY.X_train, y_train, X_test, y_test
    ├── self.ML_PROPERTY.common_axis
    └── self.ML_PROPERTY.n_features_in

lime_explain
    ├── (Same as shap_explain)
    └── Calls self._visualize_lime_explanation(), self._visualize_lime_comparison()

inspect_spectra
    ├── Calls self.shap_explain()
    ├── Calls self.lime_explain()
    ├── Calls self.get_peak_assignment()
    └── Calls module-level functions (spectrum_with_highlights_spectrum, etc.)

create_enhanced_table (MODULE-LEVEL!)
    └── Calls visualizer_instance.get_peak_assignment()
```

### External Dependencies

```python
# Standard Library
import os, time, traceback, re, json

# Scientific Computing
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.special import expit
from numpy import trapz

# Machine Learning
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# Explainability
import shap
import lime
import lime.lime_tabular

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Domain-Specific
import ramanspy as rp

# Project Modules
from functions.ML import RamanML, MLModel
from functions.configs import *  # Includes console_log, create_logs
```

---

## Complexity Metrics

| Method | Lines | % of File | Complexity Level | Coupling Level |
|--------|-------|-----------|-----------------|----------------|
| `shap_explain()` | 962 | 21.8% | **VERY HIGH** | **VERY HIGH** |
| `inspect_spectra()` | 875 | 19.9% | **VERY HIGH** | **VERY HIGH** |
| `lime_explain()` | 424 | 9.6% | HIGH | HIGH |
| `pca2d()` | 413 | 9.4% | HIGH | HIGH |
| `create_enhanced_table()` | 321 | 7.3% | MEDIUM | MEDIUM |
| `plot_institution_distribution()` | 257 | 5.8% | MEDIUM | LOW |
| `_visualize_lime_comparison()` | 195 | 4.4% | MEDIUM | LOW |
| `_visualize_lime_explanation()` | 194 | 4.4% | MEDIUM | LOW |
| `get_peak_assignment()` | 124 | 2.8% | MEDIUM | LOW |
| `visualize_processed_spectra()` | 104 | 2.4% | LOW | LOW |
| `confusion_matrix_heatmap()` | 100 | 2.3% | LOW | LOW |

**Total "Explainability Code"** (SHAP + LIME + inspect_spectra): **2,261 lines (51.3%)**

**Total "Visualization Code"**: ~600 lines (13.6%)

**Total "Peak Database Code"**: ~210 lines (4.8%)

---

## Design Issues

### 1. **Violation of Single Responsibility Principle**
The class handles:
- Peak database querying
- Basic plotting
- PCA visualization
- Model evaluation
- SHAP explainability
- LIME explainability
- Spectrum inspection
- Statistical analysis

**Impact**: Hard to maintain, test, and extend.

### 2. **Massive Methods with Nested Functions**
- `shap_explain()`: 962 lines with 10 nested functions
- `inspect_spectra()`: 875 lines with 12 nested functions

**Impact**:
- Cannot unit test nested functions
- Hard to debug
- Cannot reuse helper logic
- Poor code navigation

### 3. **Tight Coupling to ML_PROPERTY**
Almost every method depends on `ML_PROPERTY` structure:
```python
self.ML_PROPERTY.X_train
self.ML_PROPERTY.y_train
self.ML_PROPERTY.common_axis
self.ML_PROPERTY._model or .sklearn_model
```

**Impact**: Cannot use visualization without ML model, hard to test.

### 4. **Mixed Instance and Module-Level Functions**
Module-level functions (`create_enhanced_table()`) still need class instance for peak assignments.

**Impact**: Circular dependency, unclear API.

### 5. **Inconsistent Error Handling**
Some methods use:
- `raise ValueError()`
- `return {"success": False, "error": ...}`
- `console_log()` + return None

**Impact**: Unpredictable error behavior for callers.

---

## Suggested Refactoring Strategy

### Phase 1: Extract Domain Utilities (LOW RISK, HIGH VALUE)
**Files to Create**:
1. `peak_assignment.py` - Peak database query functions
   - `get_peak_assignment()`, `get_multiple_peak_assignments()`, `find_peaks_in_range()`
   - Lines: ~210

2. `basic_plots.py` - Simple visualization functions
   - `visualize_raman_spectra()`, `visualize_processed_spectra()`, `extract_raman_characteristics()`
   - Lines: ~250

3. `model_evaluation.py` - Evaluation metrics
   - `confusion_matrix_heatmap()`, `plot_institution_distribution()`
   - Lines: ~360

**Estimated Time**: 2-3 hours  
**Risk**: LOW (minimal dependencies)  
**Benefit**: Immediate code organization improvement

---

### Phase 2: Extract PCA Visualization (MEDIUM RISK, MEDIUM VALUE)
**Files to Create**:
1. `pca_visualization.py`
   - Extract `pca2d()` method
   - Split into smaller functions:
     - `prepare_pca_data()`
     - `compute_pca()`
     - `plot_pca_scatter()`
     - `add_decision_boundary()`
   - Lines: ~413

**Estimated Time**: 3-4 hours  
**Risk**: MEDIUM (complex data source logic)  
**Benefit**: Cleaner main class, easier PCA customization

---

### Phase 3: Refactor SHAP Explainability (HIGH RISK, HIGH VALUE)
**Files to Create**:
1. `shap_explainer.py`
   - Create `SHAPExplainer` class
   - Convert nested functions to methods:
     ```python
     class SHAPExplainer:
         def __init__(self, ml_property):
             self.ml_property = ml_property
         
         def validate_and_prepare_data(self)
         def detect_model_type(self)
         def optimize_data(self)
         def create_explainer(self)
         def process_shap_values(self)
         def extract_top_features(self)
         def generate_plots(self)
         def explain(self, **kwargs)  # Main entry point
     ```
   - Lines: ~962

2. `model_wrapper.py` (SHARED with LIME)
   - Unified prediction function
   - Model type detection
   - Feature selection wrapper
   - Lines: ~150

**Estimated Time**: 5-6 hours  
**Risk**: HIGH (complex logic, many edge cases)  
**Benefit**: Testable SHAP logic, shared model handling

---

### Phase 4: Refactor LIME Explainability (MEDIUM RISK, MEDIUM VALUE)
**Files to Create**:
1. `lime_explainer.py`
   - Create `LIMEExplainer` class
   - Extract visualization helpers:
     - `visualize_lime_explanation()`
     - `visualize_lime_comparison()`
   - Reuse `model_wrapper.py` from Phase 3
   - Lines: ~424 + 389 (visualizations)

**Estimated Time**: 3-4 hours  
**Risk**: MEDIUM (depends on Phase 3 model wrapper)  
**Benefit**: Cleaner LIME logic, shared prediction functions

---

### Phase 5: Refactor Spectrum Inspection (HIGH RISK, MEDIUM VALUE)
**Decision**: **Keep in main class** but refactor nested functions

**Rationale**:
- `inspect_spectra()` is an **integration method** that orchestrates multiple components
- It's a **primary user-facing API**
- Moving it would break the main class's purpose

**Refactoring Approach**:
1. Keep `inspect_spectra()` as main class method
2. Extract nested helpers to **private instance methods** (not nested functions):
   ```python
   class RamanVisualizer:
       def inspect_spectra(self, **kwargs):
           ml_property = self._validate_ml_property()
           data = self._prepare_inspection_data()
           shap_results = self.shap_explain(...)
           lime_results = self.lime_explain(...)
           return self._compile_results(...)
       
       def _validate_ml_property(self):
           # Extracted from nested function
       
       def _prepare_inspection_data(self):
           # Extracted from nested function
   ```

**Estimated Time**: 2-3 hours  
**Risk**: MEDIUM (main API, but internal refactoring only)  
**Benefit**: More testable, better code navigation

---

## Final Refactored Structure

```
functions/visualization/
├── __init__.py                     # Public API exports
├── core.py                         # Main RamanVisualizer class (REDUCED to ~1,200 lines)
│   ├── __init__()
│   ├── inspect_spectra()           # Main integration method
│   ├── plot_container_distribution() # Wrapper
│   └── Private helpers for inspect_spectra
│
├── peak_assignment.py              # Domain utilities (~210 lines)
│   ├── get_peak_assignment()
│   ├── get_multiple_peak_assignments()
│   └── find_peaks_in_range()
│
├── basic_plots.py                  # Simple visualizations (~250 lines)
│   ├── visualize_raman_spectra()
│   ├── visualize_processed_spectra()
│   └── extract_raman_characteristics()
│
├── pca_visualization.py            # PCA plotting (~413 lines)
│   ├── prepare_pca_data()
│   ├── compute_pca()
│   ├── plot_pca_scatter()
│   └── add_decision_boundary()
│
├── model_evaluation.py             # Metrics (~360 lines)
│   ├── confusion_matrix_heatmap()
│   └── plot_institution_distribution()
│
├── model_wrapper.py                # Shared ML utilities (~150 lines)
│   ├── ModelPredictor class
│   ├── detect_model_type()
│   └── create_unified_predict_fn()
│
├── shap_explainer.py               # SHAP framework (~962 lines)
│   └── SHAPExplainer class
│       ├── validate_and_prepare_data()
│       ├── detect_model_type()
│       ├── optimize_data()
│       ├── create_explainer()
│       ├── process_shap_values()
│       ├── extract_top_features()
│       ├── generate_plots()
│       └── explain()
│
├── lime_explainer.py               # LIME framework (~813 lines)
│   └── LIMEExplainer class
│       ├── explain()
│       ├── visualize_explanation()
│       └── visualize_comparison()
│
└── inspection_helpers.py           # Helpers for inspect_spectra (~400 lines)
    ├── spectrum_with_highlights()
    ├── create_shap_plots()
    └── create_enhanced_table()
```

**Total Lines After Refactoring**:
- Reduced `core.py`: ~1,200 lines (**72% reduction**)
- New modules: ~3,558 lines (well-organized, testable)
- Total package: ~4,758 lines (+8% for additional structure, but much cleaner)

---

## Risks and Mitigation

### Risk 1: Breaking Existing Code
**Mitigation**:
- Maintain backward compatibility in `__init__.py`:
  ```python
  from .core import RamanVisualizer
  from .peak_assignment import get_peak_assignment  # Also export individually
  ```
- Keep old API: `visualizer.shap_explain()` calls new `SHAPExplainer.explain()`
- Add deprecation warnings for direct function imports

### Risk 2: SHAP/LIME Refactoring Complexity
**Mitigation**:
- **DO NOT REFACTOR LOGIC** - just extract structure
- Copy nested functions to class methods first
- Test extensively with existing test cases
- Keep original method as fallback during migration

### Risk 3: Testing Burden
**Mitigation**:
- Write integration tests first (test current behavior)
- Extract modules one at a time
- Run full test suite after each extraction
- Use `git bisect` if issues arise

---

## Recommendation

### ✅ **Do Now** (Phase 1 - Safe):
1. Extract `peak_assignment.py` (LOW RISK, HIGH VALUE)
2. Extract `basic_plots.py` (LOW RISK, MEDIUM VALUE)
3. Extract `model_evaluation.py` (LOW RISK, MEDIUM VALUE)

**Time**: 2-3 hours  
**Benefit**: 820 lines (~18%) removed from core.py with minimal risk

### 🤔 **Consider Later** (Phases 2-4 - Medium Risk):
4. Extract `pca_visualization.py` after Phase 1 stabilizes
5. Extract `shap_explainer.py` + `model_wrapper.py` after Phase 2
6. Extract `lime_explainer.py` reusing model_wrapper from Phase 3

**Time**: 11-14 hours total  
**Benefit**: 2,362 lines (~53%) removed, better testing, shared model handling

### ⏸️ **Skip** (Phase 5 - Marginal Benefit):
7. Refactor `inspect_spectra()` nested functions
   - **Only if** you need to debug/extend inspection logic
   - **Keep** as integration method in main class

---

## Next Steps

**If you approve Phase 1**, I will:
1. Create `peak_assignment.py` with 3 functions + cache logic
2. Create `basic_plots.py` with 3 visualization functions
3. Create `model_evaluation.py` with 2 evaluation functions
4. Update `core.py` to import and delegate to new modules
5. Update `__init__.py` for backward compatibility
6. Test all existing functionality

**Estimated Time**: 2-3 hours  
**Lines Moved**: ~820 lines  
**Risk**: **LOW** (these functions have minimal dependencies)

Would you like me to proceed with **Phase 1**?
