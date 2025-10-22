# Raman Spectroscopy Analysis Application - AI Agent Knowledge Base

## Project Overview
This is a comprehensive PySide6-based desktop application for Raman spectroscopy data analysis and preprocessing. The application provides a modern GUI interface for loading, visualizing, and processing Raman spectral data with a sophisticated preprocessing pipeline system.

## Recent Updates & Improvements (October 2025)

### Build System Implementation (October 21, 2025) 🔧
- **PyInstaller Build Automation**: Complete Windows build system for portable and installer distributions
  - **Spec Files**: `raman_app.spec` (portable) and `raman_app_installer.spec` (installer)
  - **Build Scripts**: PowerShell automation with validation (`build_portable.ps1`, `build_installer.ps1`)
  - **Test Suite**: Comprehensive `test_build_executable.py` validating all components
  - **NSIS Support**: Professional installer template (`raman_app_installer.nsi`)
  - **Documentation**: Complete guide in `.docs/building/PYINSTALLER_GUIDE.md`
- **Two Distribution Methods**:
  - **Portable**: Single .exe (50-80 MB) with all dependencies, no installation required
  - **Installer**: NSIS-based setup (30-50 MB) for professional deployment
- **Quality Assurance**: Automated testing validates executable structure, assets, binaries, launch
- **Production Status**: ✅ Ready for testing and deployment

### Critical System Enhancement (October 15, 2025) 🔒⭐
- **Robust Parameter Type Validation System**: Implemented two-layer type validation architecture
  - **Problem Solved**: FABC baseline correction failing with "expected int, got '1.0'" errors
  - **Root Cause**: UI sliders emit floats (2.0) when pybaselines expects strict integers
  - **Solution**: Two-stage integer conversion `int(float(value))` handles all input types
  - **Coverage**: All 40 preprocessing methods validated (100% pass rate)
  - **Testing**: 45 tests executed, 100% success rate
  - **Documentation**: 4 files updated/created (BASE_MEMORY, RECENT_CHANGES, IMPLEMENTATION_PATTERNS, comprehensive guide)
  - **Production Status**: ✅ Ready - zero breaking changes, full backwards compatibility
  - **Edge Cases**: Handles floats, strings, decimals, None, booleans
  - **Performance**: 99.3% baseline reduction verified in FABC execution tests
- **Two-Layer Architecture**:
  - **Layer 1 (Registry)**: Universal type conversion for all methods
  - **Layer 2 (Class)**: Defensive programming in critical methods (FABCFixed)
- **Quality Assurance**: Comprehensive test suites for parameter types, FABC-specific edge cases, and functional execution

### Major Feature Additions (October 7, 2025 Afternoon)
- **6 Advanced Preprocessing Methods**: Implemented research-based methods for MGUS/MM classification
  - Quantile Normalization (robust distribution alignment)
  - Rank Transform (intensity-independent ordering)
  - Probabilistic Quotient Normalization (dilution correction)
  - Peak-Ratio Feature Engineering (classification features)
  - Butterworth High-Pass Filter (IIR baseline removal)
  - Convolutional Autoencoder (unified denoising/baseline via PyTorch)
- **New Feature Engineering Category**: Dedicated preprocessing category for feature extraction
- **Version Bump**: functions/preprocess module upgraded to v1.1.0

### Critical Bug Fixes (October 7, 2025)
- **Derivative Order Parameter Empty Field**: Fixed choice parameter default handling in registry and parameter widgets
- **Feature Engineering Enumerate Bug**: Corrected runtime error in peak extraction loop
- **Deep Learning Module Syntax**: Fixed class indentation inside conditional block
- **Pipeline Eye Button Crash**: Fixed "list index out of range" error by implementing robust step index validation

### UI/UX Enhancements (October 7, 2025 Morning)
- **Input Datasets Layout Optimization**: Moved refresh and export buttons to title bar, reduced padding to show 3-4 items minimum
- **Pipeline Step Selection Visual Feedback**: Added darker background highlighting when pipeline steps are selected
- **Pipeline Add Button Color**: Changed from blue to green (#28a745) for better visual consistency
- **Section Title Standardization**: All sections now use consistent custom title widgets with hint button pattern

### Code Quality Improvements
- Enhanced parameter widget choice handling with proper type conversion
- Improved error handling and logging for pipeline operations
- Added comprehensive documentation for PipelineStepWidget class
- Standardized layout margins and spacing across preprocessing page
- All new preprocessing methods fully documented with research citations
- 100% docstring coverage on new preprocessing classes
- Comprehensive syntax validation (all files pass py_compile)

## Architecture Overview

### Core Technologies
- **Frontend**: PySide6 (Qt6 for Python) - Modern cross-platform GUI framework
- **Visualization**: matplotlib with Qt backend integration
- **Data Processing**: pandas, numpy, scipy for numerical analysis
- **Spectroscopy**: Custom Raman analysis algorithms and preprocessing methods
- **Hardware Integration**: Andor SDK for camera/spectrometer control
- **Configuration**: JSON-based configuration system with live reloading

### Application Structure
```
raman-app/
├── main.py                 # Application entry point
├── configs/                # Configuration management
├── components/             # Reusable UI components
├── pages/                  # Application pages/views
├── functions/              # Core processing algorithms
├── assets/                 # Static resources (icons, fonts, locales)
├── projects/               # User project data storage
└── logs/                   # Application logging
```

### Key Design Patterns
1. **Page-Based Navigation**: Modular page system with dedicated controllers
2. **Component Architecture**: Reusable widgets with standardized interfaces
3. **Pipeline Pattern**: Configurable preprocessing pipeline with step chaining
4. **Observer Pattern**: Real-time parameter updates and preview generation
5. **Factory Pattern**: Dynamic method instantiation for preprocessing steps
6. **MVC Separation**: Clear separation of data, view, and control logic

## Core Components

### 1. Main Application (`main.py`)
- **Purpose**: Application bootstrap and global state management
- **Key Features**: Window management, global configuration, application lifecycle
- **Integration Points**: Configuration loading, logging setup, page routing

### 2. Configuration System (`configs/`)
- **Purpose**: Centralized configuration management
- **Files**:
  - `configs.py`: Configuration loading and validation
  - `app_configs.json`: Application settings
  - `style/stylesheets.py`: UI theming and styling
- **Features**: Live configuration reloading, validation, default fallbacks

### 3. UI Components (`components/`)
- **Widget Library**: Custom parameter input widgets with validation
- **matplotlib Integration**: Specialized plotting widgets for spectral data
- **Features**:
  - Real-time parameter validation
  - Color-coded status indicators
  - Automatic unit conversion and display
  - Touch/mouse-friendly interface design

### 4. Application Pages (`pages/`)

#### Home Page (`home_page.py`)
- **Purpose**: Project management and data loading interface
- **Features**: Recent projects, new project creation, data import

#### Preprocessing Page (`preprocess_page.py`)
- **Purpose**: Core spectral preprocessing interface
- **Key Features**:
  - Interactive preprocessing pipeline builder
  - Real-time preview with comparison views
  - Parameter widgets with live validation
  - **Intelligent Auto-Focus**: Conditional spectral region focusing
  - Export and batch processing capabilities

#### Data Package Page (`data_package_page.py`)
- **Purpose**: Advanced data management and export
- **Features**: Batch operations, format conversion, metadata management

### 5. Processing Functions (`functions/`)
- **Data Loading**: Multi-format support (CSV, Excel, custom formats)
- **Preprocessing**: Comprehensive spectral preprocessing methods
- **Visualization**: Advanced plotting with customization options
- **ML Integration**: Machine learning preprocessing and analysis
- **Hardware Control**: Andor camera/spectrometer integration

## Preprocessing Pipeline System

### Architecture
The preprocessing system uses a flexible pipeline architecture where each step is:
- **Configurable**: Parameters adjustable via dynamic UI
- **Chainable**: Output of one step feeds into the next
- **Reversible**: Steps can be disabled/enabled individually
- **Previewable**: Real-time visualization of processing effects

### Preprocessing Method Categories
1. **Range Operations**: Cropper, spectral region selection
2. **Baseline Correction**: 
   - Polynomial baseline correction
   - Asymmetric Least Squares (ASLS)
   - Rolling ball baseline
   - **NEW**: Butterworth High-Pass Filter (IIR digital filtering)
3. **Normalization**: 
   - Standard normalization (min-max, z-score)
   - Vector normalization
   - **NEW**: Quantile Normalization (robust distribution alignment)
   - **NEW**: Rank Transform (intensity-independent ordering)
   - **NEW**: Probabilistic Quotient Normalization (PQN) (dilution correction)
4. **Smoothing & Derivatives**:
   - Savitzky-Golay smoothing
   - Savitzky-Golay derivatives (1st, 2nd order)
   - Moving average
5. **Feature Engineering** (NEW Category):
   - **NEW**: Peak-Ratio Features (MGUS/MM classification)
6. **Deep Learning** (NEW Category):
   - **NEW**: Convolutional Autoencoder (CDAE) (unified denoising/baseline)
7. **Advanced**: Cosmic ray removal, noise reduction

### NEW: Advanced Preprocessing Methods (October 2025)
**Purpose**: Research-based methods for MGUS/MM Raman spectral classification

#### 1. Quantile Normalization
- **File**: `functions/preprocess/advanced_normalization.py`
- **Method**: Maps intensity distributions to reference quantiles (median-based)
- **Use Cases**: Cross-platform normalization, batch effect removal
- **Parameters**: n_quantiles (100), reference strategy (median/mean/custom)
- **Performance**: O(n log n), ~1-5ms per spectrum
- **Citation**: Bolstad et al. 2003

#### 2. Rank Transform
- **File**: `functions/preprocess/advanced_normalization.py`
- **Method**: Replaces intensities with ranks (dense/average/min/max)
- **Use Cases**: Outlier suppression, non-parametric analysis
- **Parameters**: method (average/min/max/dense/ordinal)
- **Performance**: O(n log n), ~1-3ms per spectrum

#### 3. Probabilistic Quotient Normalization (PQN)
- **File**: `functions/preprocess/advanced_normalization.py`
- **Method**: Corrects dilution using median quotient of intensity ratios
- **Use Cases**: Sample dilution correction, concentration normalization
- **Parameters**: reference (median/mean/custom), auto_select
- **Performance**: O(n), ~2-5ms per spectrum
- **Citation**: Dieterle et al. 2006

#### 4. Peak-Ratio Feature Engineering
- **File**: `functions/preprocess/feature_engineering.py`
- **Method**: Extracts discriminative peak ratios for classification
- **Use Cases**: MGUS vs MM discrimination, dimensionality reduction
- **Parameters**: 
  - peak_indices (custom or auto wavenumber ranges)
  - extraction_method (local_max/local_integral/gaussian_fit)
  - ratio_type (all_pairs/sequential/relative_to_first)
- **Performance**: O(p²) where p=peaks, ~10-50ms for 10 peaks
- **Citation**: Deeley et al. 2010

#### 5. Butterworth High-Pass Filter
- **File**: `functions/preprocess/advanced_baseline.py`
- **Method**: IIR digital filtering with zero-phase (filtfilt)
- **Use Cases**: Baseline removal with sharp cutoff, narrow peak preservation
- **Parameters**: cutoff_freq (0.001-0.5 Hz), order (1-10), auto_cutoff
- **Performance**: O(n), ~2-10ms per spectrum
- **Citation**: Butterworth 1930

#### 6. Convolutional Autoencoder (CDAE)
- **File**: `functions/preprocess/deep_learning.py`
- **Method**: 1D CNN encoder-decoder for end-to-end signal cleanup
- **Use Cases**: Unified denoising and baseline removal
- **Parameters**: 
  - Architecture: latent_dim, n_layers, kernel_size, stride
  - Training: learning_rate, batch_size, num_epochs
- **Performance**: ~1-5s training, ~10ms inference per spectrum
- **Dependencies**: PyTorch (optional, graceful fallback)
- **Citation**: Vincent et al. 2010

### Key Implementation Details
- **Method Registry**: Centralized registration of preprocessing methods (`functions/preprocess/registry.py`)
- **Dynamic Parameter Generation**: Automatic UI generation based on method signatures
- **State Management**: Pipeline state persistence and restoration
- **Real-time Preview**: Conditional auto-focus based on pipeline contents
- **Fit/Transform Pattern**: Scikit-learn compatible preprocessing pipeline
- **Cross-Platform**: Full Windows/Linux/Mac compatibility
- **Error Handling**: Comprehensive validation with user-friendly messages

## Intelligent Auto-Focus System

### Purpose
Automatically optimize the display range for spectral data to highlight regions of interest, particularly when range-limiting preprocessing steps are applied.

### Implementation
- **Trigger Conditions**: Only activates when range-limiting steps (e.g., Cropper) are present
- **Detection Algorithm**: Variance-based signal analysis for optimal range selection
- **Integration**: Seamlessly integrated into all plotting methods
- **User Control**: Conditional activation preserves user workflow expectations

### Technical Details
```python
def _should_auto_focus(self) -> bool:
    """Check if auto-focus should be enabled based on pipeline contents."""
    range_limiting_steps = ['Cropper', 'Range Selector', 'Spectral Window']
    enabled_steps = [step for step in self.pipeline if step.enabled]
    return any(step.method in range_limiting_steps for step in enabled_steps)
```

## Component Architecture & Recent Improvements

### Pipeline Step Widget (`pages/preprocess_page_utils/pipeline.py`)
- **Purpose**: Interactive widgets for individual pipeline steps with visual state management
- **Key Features**:
  - Enable/disable toggle with eye icon buttons
  - **NEW**: Selection visual feedback with darker blue background highlighting
  - Color-coded states: new steps (green ✚), existing steps (gray ⚙), disabled (muted)
  - Hover effects and visual state transitions
  - **FIXED**: Robust step index validation to prevent "list index out of range" errors
- **Technical Implementation**: 
  - `set_selected(bool)`: Method for managing selection appearance
  - `_update_appearance()`: Comprehensive state-based styling system
  - Error-safe signal handling with dynamic index resolution

### Parameter Widget System (`components/widgets/parameter_widgets.py`)
- **Purpose**: Dynamic UI generation for preprocessing method parameters
- **Key Features**:
  - Automatic widget creation based on parameter type definitions (int, float, choice, bool, tuple, scientific)
  - **FIXED**: Enhanced choice parameter handling with proper integer/float type conversion
  - Real-time parameter validation and live preview updates
  - Range-aware tuple parameters for spectral data bounds
- **Technical Improvement**:
  ```python
  # NEW: Proper choice parameter type mapping
  choice_mapping = {str(choice): choice for choice in choices}
  widget.choice_mapping = choice_mapping  # Preserve original types
  ```

### Preprocessing Page Layout (`pages/preprocess_page.py`)
- **Purpose**: Main preprocessing interface with optimized space utilization
- **Recent Enhancements**:
  - **Input Datasets**: Moved refresh/export buttons to title bar, increased list height to show 3-4 items minimum
  - **Pipeline Building**: Changed add button from blue to green (#28a745) for visual consistency
  - **Section Titles**: Standardized all sections to use custom title widgets with hint button pattern
  - **Layout Optimization**: Reduced padding (20px→12px top, 16px spacing) for better space utilization
- **Visual Improvements**:
  - Consistent 24px title bar action buttons with hover effects
  - Transparent title bar button styling for clean integration
  - Selection highlighting for pipeline steps with 2px blue border

## Data Flow Architecture

### Input Processing
1. **Data Loading**: Multi-format file support with automatic format detection
2. **Validation**: Data integrity checks and error handling
3. **Normalization**: Standardized internal representation
4. **Caching**: Intelligent caching for performance optimization

### Processing Pipeline
1. **Step Validation**: Parameter validation before processing
2. **Sequential Processing**: Ordered execution of enabled steps
3. **Error Handling**: Graceful failure with user feedback
4. **Result Caching**: Intermediate result storage for preview generation

### Output Generation
1. **Visualization**: Real-time plot updates with conditional auto-focus
2. **Export**: Multiple format support with metadata preservation
3. **Project Storage**: Persistent project state management

## UI/UX Design Principles

### Visual Design
- **Modern Flat Design**: Clean, professional interface suitable for scientific use
- **Color Coding**: Intuitive status indicators (green/red for validation states)
- **Icon System**: Consistent iconography using SVG assets
- **Typography**: Professional fonts optimized for data display

### Interaction Design
- **Immediate Feedback**: Real-time parameter validation and preview updates
- **Progressive Disclosure**: Complex features revealed as needed
- **Keyboard Navigation**: Full keyboard accessibility
- **Touch-Friendly**: Mouse and touch interaction support

### Internationalization & Localization
- **Multi-Language Support**: Full English and Japanese localization
- **Dynamic UI Sizing**: UI elements automatically adapt to text length across languages
- **Font Configuration**: Locale-specific font families (Inter for English, Noto Sans JP for Japanese)
- **Cultural Considerations**: UI layouts and interactions optimized for target locales

**Technical Implementation:**
```python
# Dynamic button sizing for localized content
def _adjust_button_width_to_text(self):
    """Automatically adjust button width based on localized text content."""
    font_metrics = QFontMetrics(self.button.font())
    text_width = font_metrics.horizontalAdvance(self.button.text())
    dynamic_width = max(min_width, text_width + padding + margins)
    self.button.setFixedWidth(dynamic_width)
```

**Localization Assets:**
- `assets/locales/en.json`: English translations
- `assets/locales/ja.json`: Japanese translations
- Comprehensive coverage of all UI elements, tooltips, and status messages

### Accessibility
- **Color Contrast**: High contrast for readability
- **Size Flexibility**: Scalable UI elements
- **Screen Reader Support**: Proper labeling and structure
- **Keyboard Shortcuts**: Power user efficiency features

## Performance Considerations

### Optimization Strategies
1. **Lazy Loading**: Components loaded on demand
2. **Efficient Rendering**: matplotlib optimization for large datasets
3. **Memory Management**: Careful handling of large spectral datasets
4. **Caching**: Intelligent caching of processed results
5. **Background Processing**: Non-blocking operations where possible

### Scalability Features
- **Batch Processing**: Handle multiple files efficiently
- **Memory Monitoring**: Prevent memory overflow with large datasets
- **Progress Indication**: User feedback for long-running operations

## Integration Points

### Hardware Integration
- **Andor SDK**: Direct camera/spectrometer control
- **Driver Management**: Automatic driver loading and configuration
- **Real-time Acquisition**: Live spectral data capture

### External Libraries
- **Scientific Computing**: numpy, scipy, pandas for numerical operations
- **Visualization**: matplotlib with Qt backend for plotting
- **UI Framework**: PySide6 for cross-platform GUI development
- **File I/O**: Support for multiple spectroscopic file formats

## Development Guidelines

### Code Organization
- **Feature-Sliced Architecture**: Group related functionality together
- **Single Responsibility**: Each module has a clear, focused purpose
- **Dependency Injection**: Loose coupling between components
- **Interface Segregation**: Clean interfaces between layers

### Naming Conventions
- **Files**: snake_case for Python files
- **Classes**: PascalCase for class names
- **Methods**: snake_case for method names
- **Constants**: UPPER_CASE for constants
- **Private Members**: Leading underscore for internal methods

### Error Handling
- **Graceful Degradation**: Application continues functioning when possible
- **User-Friendly Messages**: Clear error communication to users
- **Logging**: Comprehensive logging for debugging and monitoring
- **Recovery Mechanisms**: Automatic recovery from common error states

## Future Extensibility

### Plugin Architecture
The application is designed to support future plugin development:
- **Method Registration**: Easy addition of new preprocessing methods
- **UI Generation**: Automatic parameter widget generation
- **Integration Points**: Clear interfaces for third-party extensions

### API Design
- **Consistent Interfaces**: Standardized method signatures
- **Documentation**: Comprehensive inline documentation
- **Version Compatibility**: Backward compatibility considerations
- **Testing Framework**: Comprehensive test coverage for reliability

## Security Considerations

### Data Protection
- **Input Validation**: All user inputs validated and sanitized
- **File Access**: Controlled file system access
- **Memory Safety**: Careful memory management to prevent leaks
- **Error Information**: Secure error handling without information disclosure

### Configuration Security
- **Default Settings**: Secure defaults for all configurations
- **Validation**: Configuration validation and sanitization
- **Access Control**: Appropriate file permissions for sensitive data

This knowledge base provides a comprehensive foundation for understanding and extending the Raman spectroscopy application. The architecture supports both current functionality and future enhancements while maintaining code quality and user experience standards.