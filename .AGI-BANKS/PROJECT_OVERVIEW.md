# Raman Spectroscopy Analysis Application - AI Agent Knowledge Base

## Project Overview
This is a comprehensive PySide6-based desktop application for Raman spectroscopy data analysis and preprocessing. The application provides a modern GUI interface for loading, visualizing, and processing Raman spectral data with a sophisticated preprocessing pipeline system.

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

### Step Types
1. **Range Operations**: Cropper, spectral region selection
2. **Baseline Correction**: Polynomial, ASLS, rolling ball
3. **Normalization**: Various normalization methods
4. **Smoothing**: Savitzky-Golay, moving average
5. **Calibration**: Wavenumber and intensity calibration
6. **Advanced**: Cosmic ray removal, noise reduction

### Key Implementation Details
- **Method Registry**: Centralized registration of preprocessing methods
- **Dynamic Parameter Generation**: Automatic UI generation based on method signatures
- **State Management**: Pipeline state persistence and restoration
- **Real-time Preview**: Conditional auto-focus based on pipeline contents

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