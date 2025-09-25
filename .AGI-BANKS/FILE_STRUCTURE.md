# File Structure and Component Details

## Directory Structure
```
raman-app/
├── main.py                     # Application entry point
├── pyproject.toml             # Project configuration and dependencies
├── uv.lock                    # Dependency lock file
├── README.md                  # Project documentation
├── LICENSE                    # License information
├── test.py                    # Test runner
├── fix_backslash.py          # Utility script
├── utils.py                  # General utilities
├── main.md                   # Main documentation
├── utils.md                  # Utilities documentation
├── .AI-AGENT/                # AI agent knowledge base
│   ├── PROJECT_OVERVIEW.md
│   ├── IMPLEMENTATION_PATTERNS.md
│   └── RECENT_CHANGES.md
├── assets/                   # Static resources
│   ├── data/
│   │   └── raman_peaks.json  # Reference Raman peak data
│   ├── fonts/                # Application fonts
│   │   ├── Inter-Italic.ttf
│   │   ├── Inter.ttf
│   │   └── Noto Sans JP.ttf
│   ├── icons/                # SVG icons
│   │   ├── chevron-down.svg
│   │   ├── eye-close.svg
│   │   ├── eye-open.svg
│   │   ├── load-project.svg
│   │   ├── minus.svg         # Parameter widget icons
│   │   ├── new-project.svg
│   │   ├── plus.svg          # Parameter widget icons
│   │   ├── recent-project.svg
│   │   ├── reload.svg
│   │   ├── trash-bin.svg
│   │   └── trash-xmark.svg
│   └── locales/              # Internationalization
│       ├── en.json           # English translations
│       └── ja.json           # Japanese translations
├── components/               # Reusable UI components
│   ├── app_tabs.py          # Tab management component
│   ├── toast.py             # Notification system
│   └── widgets/             # Custom widget library
│       ├── __init__.py
│       ├── constrained_parameter_widgets.py
│       ├── enhanced_parameter_widgets.py
│       ├── icons.py         # Icon loading utilities
│       ├── matplotlib_widget.py      # Matplotlib integration
│       ├── parameter_widgets.py      # Parameter input widgets
│       ├── utils.py         # Widget utilities
│       └── docs/            # Widget documentation
├── configs/                  # Configuration management
│   ├── __init__.py
│   ├── app_configs.json     # Main application configuration
│   ├── configs.py           # Configuration loading logic
│   └── style/               # UI styling
│       └── stylesheets.py   # Qt stylesheets
├── docs/                     # Documentation
├── drivers/                  # Hardware drivers
│   ├── atmcd32d.dll         # Andor camera driver (32-bit)
│   └── atmcd64d.dll         # Andor camera driver (64-bit)
├── functions/                # Core processing functions
│   ├── _utils_.py           # Internal utilities
│   ├── andorsdk.py          # Andor SDK interface
│   ├── configs.py           # Function-specific configs
│   ├── data_loader.py       # Data loading and parsing
│   ├── ML.py                # Machine learning integration
│   ├── noise_func.py        # Noise reduction algorithms
│   ├── utils.py             # General function utilities
│   ├── visualization.py     # Advanced plotting functions
│   ├── andorsdk/            # Andor SDK components
│   └── preprocess/          # Preprocessing algorithms
├── logs/                     # Application logs
│   ├── config.log
│   ├── data_loading.log
│   ├── localization.log
│   ├── open_recent.log
│   ├── PreprocessPage.log
│   ├── projects.log
│   ├── RamanPipeline.log
│   └── reset_state.log
├── pages/                    # Application pages/views
│   ├── data_package_page.md  # Data package documentation
│   ├── data_package_page.py  # Data management interface
│   ├── home_page.md         # Home page documentation
│   ├── home_page.py         # Project management interface
│   ├── preprocess_page.md   # Preprocessing documentation
│   ├── preprocess_page.py   # Main preprocessing interface (refactored)
│   ├── workspace_page.py    # Workspace management
│   └── preprocess_page_utils/ # Preprocessing handler classes
│       ├── __init__.py      # Package initialization
│       ├── data_manager.py  # Data loading and management handler
│       ├── pipeline_manager.py # Pipeline creation and step management
│       ├── parameter_manager.py # Parameter widget management
│       ├── preview_manager.py   # Real-time preview system
│       ├── processing_manager.py # Processing execution and export
│       └── widgets.py       # Preprocessing-specific widgets
└── projects/                 # User project storage
    ├── taketani-sensei-data/
    └── test/
```

## Core Component Analysis

### 1. Application Entry Point
**File:** `main.py`
- **Purpose**: Bootstrap application, initialize global state
- **Key Features**: 
  - Window management and lifecycle
  - Global configuration loading
  - Page routing and navigation setup
  - Hardware initialization (Andor SDK)

### 2. Configuration System

#### `configs/configs.py`
- **Purpose**: Centralized configuration management
- **Key Features**:
  - JSON-based configuration loading
  - Environment-specific settings
  - Live configuration reloading
  - Validation and default values

#### `configs/app_configs.json`
- **Purpose**: Main application settings
- **Contains**:
  - UI preferences and themes
  - Default processing parameters
  - Hardware configuration
  - User interface settings

#### `configs/style/stylesheets.py`
- **Purpose**: Qt stylesheet management
- **Features**:
  - Consistent theming across components
  - Dynamic style application
  - Color scheme management
  - Responsive design elements

### 3. UI Components

#### `components/widgets/matplotlib_widget.py`
**Critical Component for Visualization**
- **Purpose**: Scientific plotting with Qt integration
- **Key Features**:
  - `detect_signal_range()`: Variance-based auto-focus algorithm
  - `plot_spectra()`: Enhanced plotting with conditional auto-focus
  - `plot_comparison_spectra_with_wavenumbers()`: Comparison visualization
- **Recent Enhancements**:
  - Added auto_focus parameter to plotting methods
  - Implemented intelligent signal detection for Raman spectra
  - Optimized for large dataset handling

#### `components/widgets/parameter_widgets.py`
**Core Component for User Input**
- **Purpose**: Dynamic parameter input widgets
- **Key Classes**:
  - `FloatParameterWidget`: Floating-point input with validation
  - `IntParameterWidget`: Integer input with range constraints
  - `RangeParameterWidget`: Two-ended range selection
  - `ChoiceParameterWidget`: Dropdown selection
  - `DynamicParameterWidget`: Auto-generating parameter interface
- **Recent Fixes**:
  - Fixed icon paths for plus/minus buttons
  - Enhanced real-time value updates
  - Improved unit display and validation

#### `components/widgets/icons.py`
- **Purpose**: Icon loading and management utilities
- **Features**:
  - SVG icon loading from assets
  - Icon caching for performance
  - Consistent icon sizing and styling

### 4. Application Pages

#### `pages/preprocess_page.py` (Refactored Architecture)
**Primary Interface for Spectral Preprocessing**
- **Purpose**: Main preprocessing workflow coordination and UI management  
- **Architecture**: Composition-based design with specialized handler classes
- **Current State**: Dramatically simplified from 1839 → 346 lines (81% reduction)
- **Key Components**:
  - Handler initialization and delegation system
  - Signal forwarding and backward compatibility  
  - UI setup and layout management
  - Graceful fallback for method availability
- **Delegation Pattern**: All business logic delegated to specialized handlers

#### `pages/preprocess_page_utils/` (Handler Classes)
**Specialized handler classes implementing single responsibility principle**

#### `pages/preprocess_page_utils/data_manager.py`
- **Purpose**: Data loading, previewing, and management operations
- **Implementation**: 200+ lines of focused data handling logic
- **Key Methods**:
  - `load_project_data()`: Project data loading with validation
  - `preview_raw_data()`: Data preview with automatic visualization
  - `get_data_wavenumber_range()`: Spectral range extraction
- **Features**: Preprocessing history management, multi-dataset handling

#### `pages/preprocess_page_utils/pipeline_manager.py`  
- **Purpose**: Pipeline creation, modification, and step management
- **Implementation**: 280+ lines of pipeline orchestration logic
- **Key Methods**:
  - `add_pipeline_step()`: Step addition with parameter initialization
  - `remove_pipeline_step()`: Step removal with UI updates
  - `toggle_step()`: Enable/disable individual steps
- **Features**: Drag-and-drop reordering, step configuration, method validation

#### `pages/preprocess_page_utils/parameter_manager.py`
- **Purpose**: Parameter widget display and real-time parameter management  
- **Implementation**: 125+ lines of parameter coordination logic
- **Key Methods**:
  - `show_parameter_widget()`: Dynamic widget creation and display
  - `update_step_parameters()`: Real-time parameter synchronization
  - `clear_parameter_widget()`: Widget cleanup and state management
- **Features**: Dynamic parameter widgets, real-time validation, method-specific UIs

#### `pages/preprocess_page_utils/preview_manager.py`
- **Purpose**: Real-time preview system with automatic updates
- **Implementation**: 170+ lines of preview coordination logic  
- **Key Methods**:
  - `toggle_preview()`: Preview system activation/deactivation
  - `schedule_preview_update()`: Debounced update scheduling
  - `force_preview_update()`: Immediate preview refresh
- **Features**: Debounced updates, conditional auto-focus, error handling, status indicators

#### `pages/preprocess_page_utils/processing_manager.py`
- **Purpose**: Pipeline execution and result export
- **Implementation**: 320+ lines of processing orchestration logic
- **Key Methods**:
  - `start_processing()`: Background pipeline execution
  - `export_results()`: CSV/Excel export with metadata
  - Worker thread management with progress tracking
- **Features**: Background processing, progress indicators, multiple export formats, error recovery

#### `pages/preprocess_page_utils/widgets.py`
- **Purpose**: Preprocessing-specific widget extensions  
- **Features**:
  - Specialized parameter widgets for spectroscopy
  - Method-specific parameter generation
  - Pipeline step configuration interfaces
- **Recent Fixes**:
  - Corrected icon path references
  - Enhanced parameter extraction logic
  - Removed debug output for cleaner interface

#### `pages/home_page.py`
- **Purpose**: Project management and data loading
- **Features**:
  - Recent project access
  - New project creation workflow
  - Data import and format handling
  - Project state management

#### `pages/data_package_page.py`
- **Purpose**: Advanced data management and export
- **Features**:
  - Batch processing capabilities
  - Multiple export format support
  - Metadata management
  - Data package creation

### 5. Processing Functions

#### `functions/data_loader.py`
- **Purpose**: Multi-format data loading and parsing
- **Supported Formats**:
  - CSV files with automatic delimiter detection
  - Excel files (.xlsx, .xls)
  - Custom spectroscopy formats
  - Andor camera data files
- **Features**:
  - Automatic format detection
  - Data validation and cleaning
  - Metadata extraction
  - Error handling and recovery

#### `functions/visualization.py`
- **Purpose**: Advanced plotting and visualization
- **Features**:
  - Scientific plotting templates
  - Customizable plot styles
  - Export capabilities
  - Interactive plot elements

#### `functions/preprocess/`
- **Purpose**: Preprocessing algorithm implementations
- **Contains**:
  - Baseline correction methods
  - Smoothing algorithms
  - Normalization techniques
  - Calibration functions
  - Noise reduction methods

### 6. Hardware Integration

#### `functions/andorsdk.py`
- **Purpose**: Andor camera/spectrometer control
- **Features**:
  - Camera initialization and configuration
  - Real-time data acquisition
  - Hardware status monitoring
  - Temperature control

#### `drivers/`
- **Purpose**: Hardware driver libraries
- **Contains**:
  - Andor camera drivers (32-bit and 64-bit)
  - Hardware interface libraries

### 7. Asset Management

#### `assets/icons/`
- **Purpose**: SVG icon collection
- **Key Icons**:
  - `plus.svg`, `minus.svg`: Parameter widget controls
  - `eye-open.svg`, `eye-close.svg`: Visibility toggles
  - `load-project.svg`, `new-project.svg`: Project actions
  - `reload.svg`: Refresh operations
  - `trash-bin.svg`, `trash-xmark.svg`: Deletion actions

#### `assets/fonts/`
- **Purpose**: Typography assets
- **Fonts**:
  - Inter: Modern sans-serif for UI elements
  - Noto Sans JP: Japanese text support

#### `assets/locales/`
- **Purpose**: Internationalization support
- **Languages**:
  - English (en.json): Default language
  - Japanese (ja.json): Japanese translations

#### `assets/data/raman_peaks.json`
- **Purpose**: Reference Raman peak database
- **Contents**:
  - Known Raman peak positions
  - Compound identification data
  - Peak assignment references

### 8. Project Management

#### `projects/`
- **Purpose**: User project storage
- **Structure**:
  - Individual project directories
  - Project configuration files
  - Processed data storage
  - Pipeline state persistence

### 9. Logging System

#### `logs/`
- **Purpose**: Application event logging
- **Log Files**:
  - `PreprocessPage.log`: Preprocessing operation logs
  - `data_loading.log`: Data loading operation tracking
  - `config.log`: Configuration change tracking
  - `RamanPipeline.log`: Pipeline execution logs

## Key Integration Points

### 1. Widget-Page Integration
- Parameter widgets automatically connect to page update methods
- Real-time value changes trigger preview updates
- Validation states propagate to user interface

### 2. Plotting-Processing Integration
- matplotlib widgets receive processed data directly
- Auto-focus decisions based on pipeline analysis
- Coordinated updates between original and preview plots

### 3. Configuration-Component Integration
- All components respect global configuration settings
- Dynamic theming and styling application
- User preferences persist across sessions

### 4. Hardware-Software Integration
- Direct camera control through Andor SDK
- Real-time data acquisition and display
- Hardware status integration with UI

This comprehensive file structure analysis provides the foundation for understanding how components interact and where to make modifications for future enhancements.