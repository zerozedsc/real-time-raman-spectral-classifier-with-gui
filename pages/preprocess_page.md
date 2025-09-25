# Preprocessing Page Documentation

## 🎯 Recent Updates (2025)

### **Latest Critical Bug Fixes and Enhancements (September 2025)**

#### **Global Pipeline Memory System (NEW)**
- **✅ Persistent Pipeline Steps**: Implemented global memory system to prevent pipeline steps from vanishing when switching between datasets
- **✅ Cross-Dataset State Management**: Pipeline steps now persist across dataset switches using `_global_pipeline_memory`
- **✅ Automatic Save/Restore**: Pipeline state automatically saved on modifications and restored on dataset selection
- **✅ UI Reconstruction**: Added `_rebuild_pipeline_ui()` for seamless interface rebuilding with saved steps

#### **Enhanced X-axis Padding for Cropped Regions (NEW)**
- **✅ Crop Boundary Visualization**: When cropping regions (e.g., 600-1800 cm⁻¹), boundaries are now properly visible with padding
- **✅ Smart Crop Detection**: Added `_extract_crop_bounds()` method to automatically detect cropping steps in pipeline
- **✅ Matplotlib Integration**: Enhanced `matplotlib_widget.py` with `crop_bounds` parameter for proper boundary handling
- **✅ Fixed Padding Implementation**: Default padding changed from percentage-based to fixed ±50 wavenumber units
- **✅ Parameter Persistence**: Pipeline parameters now preserved when switching datasets (not just steps)
- **✅ Enhanced Global Memory**: `_update_current_step_parameters()` captures widget values before saving to memory

#### **Preview OFF Functionality Fix (NEW)**
- **✅ Original Data Display**: Preview OFF now properly shows original dataset data instead of empty graphs
- **✅ Data Loading Fix**: Enhanced `_update_preview()` method to correctly load data from `RAMAN_DATA` when preview disabled
- **✅ State Management**: Proper handling of preview toggle between processed and original data views

#### **Professional UI Color Scheme (NEW)**
- **✅ Removed Orange Elements**: Replaced all orange UI elements with professional gray/blue color scheme
- **✅ Processing Status**: Changed processing indicators from orange `#f39c12` to dark gray `#666666`
- **✅ Pipeline Widget Colors**: Updated imported step colors to blue scheme (Dark blue `#1976d2` enabled, Light blue `#64b5f6` disabled)
- **✅ Better Accessibility**: Improved color contrast and visual distinction for enabled/disabled states

#### **Previous Enhancements**

#### **Real-time UI State Management**
- **✅ Fixed Enable/Disable Button States**: Pipeline step eye buttons now update in real-time when toggled
- **✅ Enhanced Toggle All Operations**: Toggle all existing steps now properly updates both step state and visual indicators
- **✅ Improved State Synchronization**: UI elements maintain consistency with underlying data model

#### **Enhanced Dataset Switching Logic**
- **✅ Intelligent Step Persistence**: When switching between raw datasets, only enabled steps are maintained
- **✅ Prevented Unwanted Step Propagation**: Preprocessed dataset steps no longer automatically follow to raw datasets
- **✅ Smart State Management**: Raw dataset switches preserve user-enabled pipeline configuration

#### **Advanced Graph Visualization**
- **✅ Improved Auto-focus Padding**: Both auto-focus and manual focus now properly show 50-100 units of X-axis padding
- **✅ Enhanced Preview OFF Behavior**: Preview OFF correctly shows original unprocessed data without pipeline effects
- **✅ Optimized Signal Range Detection**: Better automatic range detection for meaningful Raman signals

#### **Robust Warning System**
- **✅ Enhanced Preview Warnings**: Comprehensive dialog system warns users about:
  - Double preprocessing risks when enabling preview on preprocessed data
  - Hidden processing effects when disabling preview on raw data with active pipeline
- **✅ Intelligent Context Detection**: System automatically detects dataset types and potential user errors

#### **Complete Localization Support**
- **✅ Verified UI Text**: All interface elements properly use localization keys from assets/locales/en.json
- **✅ Internationalization Ready**: Full support for English/Japanese text switching

---

## 1. Overview

The **Preprocessing** page is an advanced, interactive module for cleaning and preparing Raman spectral data for medical diagnosis applications. It provides a comprehensive, pipeline-based workflow with detailed parameter controls, **automatic real-time preview**, and robust error handling designed specifically for disease detection scenarios.

This page has been **completely refactored** using a **composition-based architecture** that separates concerns into specialized handler classes, dramatically improving maintainability while preserving all functionality.

---

## 2. Architecture Overview

### **New Composition-Based Architecture (2025)**

The preprocessing page now uses a modern **handler composition pattern** that separates the monolithic 1800+ line class into focused, specialized components:

#### **Core Handler Classes**

1. **`DataManagerHandler`** (`pages/preprocess_page_utils/data_manager.py`)
   - **Responsibility**: Data loading, previewing, and management operations
   - **Key Methods**: `load_project_data()`, `preview_raw_data()`, `get_data_wavenumber_range()`
   - **Features**: Preprocessing history management, multi-dataset handling

2. **`PipelineManagerHandler`** (`pages/preprocess_page_utils/pipeline_manager.py`)  
   - **Responsibility**: Pipeline creation, modification, and step management
   - **Key Methods**: `add_pipeline_step()`, `remove_pipeline_step()`, `toggle_operations()`
   - **Features**: Drag-and-drop reordering, step enabling/disabling, pipeline persistence

3. **`ParameterManagerHandler`** (`pages/preprocess_page_utils/parameter_manager.py`)
   - **Responsibility**: Parameter widget display and real-time parameter management
   - **Key Methods**: `show_parameter_widget()`, `update_step_parameters()`, `clear_parameter_widget()`
   - **Features**: Dynamic parameter widgets, real-time validation, parameter persistence

4. **`PreviewManagerHandler`** (`pages/preprocess_page_utils/preview_manager.py`)
   - **Responsibility**: Real-time preview system with automatic updates
   - **Key Methods**: `toggle_preview()`, `schedule_preview_update()`, `force_preview_update()`
   - **Features**: Debounced updates, error handling, preview status indicators

5. **`ProcessingManagerHandler`** (`pages/preprocess_page_utils/processing_manager.py`)
   - **Responsibility**: Pipeline execution and result export
   - **Key Methods**: `start_processing()`, `export_results()`, worker thread management
   - **Features**: Background processing, progress tracking, CSV/Excel export

#### **Main Page Class** (`pages/preprocess_page.py`)
- **Size**: Reduced from 1839 → 346 lines (**81% reduction**)
- **Role**: UI coordination, handler initialization, signal delegation
- **Pattern**: Delegation methods maintain backward compatibility
- **Fallback**: Graceful degradation if handlers fail to load

### **Technical Benefits Achieved**

- **🎯 Single Responsibility**: Each handler focuses on one specific domain
- **🔧 Maintainability**: Clear separation makes debugging and modification easier
- **🧪 Testability**: Handler classes can be unit tested independently
- **🔄 Reusability**: Handlers can be reused in other preprocessing interfaces
- **📈 Scalability**: Easy to add new features to specific responsibility areas
- **⚡ Performance**: Background processing with worker threads prevents UI blocking

---

## 3. Enhanced Page Layout & Components

The page features a modern, two-panel layout optimized for efficient preprocessing workflows with **automatic visual feedback**:

### **Left Panel: Enhanced Control Interface**

#### **入力データセット (Input Datasets)**
- **Multi-selection Support**: Select multiple datasets simultaneously for batch processing
- **Visual Feedback**: Clear selection indicators and hover states
- **Data Validation**: Automatic validation of selected datasets
- **Status Display**: Shows when no datasets are available
- **Auto-Preview**: Automatically loads data for real-time preview when selected

#### **パイプライン設定 (Pipeline Configuration)**
- **Advanced Step Selection**: Comprehensive dropdown with Japanese preprocessing method names:
  - スペクトル切り出し (Spectral Cropping)
  - Savitzky-Golay フィルタ (Savitzky-Golay Filter)
  - ASPLS ベースライン補正 (ASPLS Baseline Correction)
  - マルチスケール畳み込み (Multi-scale Convolution)
  - ベクトル正規化 (Vector Normalization)
  - SNV 正規化 (SNV Normalization)
  - コズミックレイ除去 (Cosmic Ray Removal)
  - 微分処理 (Derivative Processing)

- **Real-time Pipeline Management**:
  - **Automatic Preview**: Visual updates immediately when steps are added/removed
  - Drag-and-drop reordering of processing steps with instant preview update
  - **Dynamic Parameter Preview**: Changes to parameters trigger automatic graph updates
  - Visual step indicators with medical-themed styling
  - One-click step removal and pipeline clearing
  - Real-time parameter validation with visual feedback

#### **出力設定 (Output Configuration)**
- **Smart Naming**: Intelligent output dataset naming with validation
- **Overwrite Protection**: Confirmation dialogs for existing datasets
- **Execution Control**: Prominent run button with progress indication

---

### **Right Panel: Advanced Display & Control**

#### **パラメータ (Parameters)**
- **Detailed Parameter Controls**: Each processing method includes comprehensive parameter options:

  **スペクトル切り出し (Spectral Cropping)**:
  - 開始波数/終了波数 with cm⁻¹ units
  - Contextual help explaining fingerprint region usage
  - Real-time range validation

  **Savitzky-Golay フィルタ**:
  - ウィンドウ長 (Window Length) with odd-number validation
  - 多項式次数 (Polynomial Order) with medical-appropriate defaults
  - 微分次数 (Derivative Order) for spectral derivatives
  - 境界処理モード (Boundary Mode) selection
  - Medical context explanations

  **ASPLS ベースライン補正**:
  - 平滑化パラメータ (λ) with scientific notation support
  - 非対称パラメータ (p) with precision controls
  - 差分次数 (Difference Order)
  - 最大反復回数 (Maximum Iterations)
  - 許容誤差 (Tolerance) with scientific precision
  - Medical application guidance

  **マルチスケール畳み込み**:
  - カーネルサイズ (Kernel Sizes) with comma-separated input
  - 重み (Weights) configuration
  - 畳み込みモード (Convolution Mode)
  - 反復回数 (Iterations)

  **正規化手法**:
  - SNV: Contextual medical explanations
  - Vector: Norm type selection (L1, L2, Max)

- **Parameter Validation**: Real-time validation with medical context
- **Contextual Help**: Each parameter includes tooltips and explanations
- **Scrollable Interface**: Accommodates detailed parameter sets

#### **可視化 (Visualization)**
- **Real-time Automatic Preview**: Instant visual feedback during pipeline building
  - **Automatic Updates**: Graph updates immediately when steps are added/removed/reordered
  - **Parameter Live Preview**: Real-time visualization as parameters are adjusted
  - **Debounced Updates**: Smooth performance with 300ms delay for parameter changes
  - **Preview Status Indicator**: Visual status showing preview state (ready/processing/error)

- **Smart Comparison View**: 
  - **Original vs Processed**: Side-by-side comparison with color-coded visualization
  - **Sample Data Preview**: Uses subset of data for fast preview performance
  - **Toggle Control**: Enable/disable automatic preview with manual fallback

- **Interactive Matplotlib Integration**: 
  - High-quality spectral plots optimized for medical applications
  - **Performance Optimized**: Uses sampled data (every 10th spectrum) for preview
  - Medical-grade plotting with professional styling
  - Responsive design with adaptive plot sizing

---

## 3. Enhanced Real-time Preview Workflow

### **Automatic Preview System**
1. **Data Loading**: Original data automatically stored when datasets are selected
2. **Pipeline Building**: 
   - Each step addition triggers automatic preview update
   - Parameter changes update preview with 300ms debouncing
   - Step toggling (enable/disable) updates preview instantly
   - Pipeline reordering via drag-drop updates preview automatically

3. **Visual Feedback**:
   - **Green Status**: Preview ready and current
   - **Orange Status**: Processing preview update
   - **Red Status**: Preview error (falls back to original data)
   - **Gray Status**: Preview disabled (manual mode)

### **Data Preparation Phase**
1. **Dataset Selection**: Choose single or multiple datasets for processing
2. **Data Validation**: Automatic validation of spectral data format and quality  
3. **Preview Generation**: Immediate visualization with automatic preview system

### **Pipeline Configuration Phase**
1. **Method Selection**: Choose from medical-optimized preprocessing methods
2. **Real-time Building**: 
   - Add steps and see immediate visual impact
   - Adjust parameters with live preview updates
   - Reorder steps with instant visual feedback
3. **Parameter Optimization**: Detailed parameter adjustment with automatic visual validation
4. **Live Validation**: Real-time parameter validation with preview confirmation

### **Execution Phase**
1. **Preview Verification**: Final review of preprocessing effects before execution
2. **Output Naming**: Intelligent naming with conflict resolution
3. **Processing**: Robust execution with comprehensive error handling  
4. **Result Visualization**: Immediate visualization of processed spectra

---

## 4. Medical Application Features

### **Disease Detection Optimization**
- **Fingerprint Region Focus**: Default spectral cropping for medical analysis
- **Fluorescence Removal**: ASPLS baseline correction for tissue fluorescence
- **Scatter Compensation**: SNV normalization for tissue variation
- **Signal Enhancement**: Savitzky-Golay filtering preserving diagnostic features

### **Quality Assurance**
- **Parameter Validation**: Medical-appropriate parameter ranges
- **Processing Verification**: Comprehensive error handling and validation
- **Metadata Tracking**: Complete processing history for regulatory compliance
- **Reproducibility**: Consistent processing parameters across sessions

### **User Experience**
- **Japanese Interface**: Native Japanese labels and explanations
- **Medical Context**: Parameter explanations specific to medical applications
- **Workflow Guidance**: Clear step-by-step processing guidance
- **Error Prevention**: Proactive validation and user feedback

---

## 5. Technical Implementation

### **Handler Composition Architecture**

#### **Initialization Pattern**
```python
def _initialize_handlers(self):
    """Initialize all handler classes for composition architecture."""
    try:
        # Import and instantiate handlers
        self.data_manager = DataManagerHandler(self)
        self.pipeline_manager = PipelineManagerHandler(self)
        self.parameter_manager = ParameterManagerHandler(self)
        self.preview_manager = PreviewManagerHandler(self)
        self.processing_manager = ProcessingManagerHandler(self)
        
        # Setup handler-specific controls
        self.preview_manager.setup_preview_controls()
        self.processing_manager.setup_processing_controls()
        
    except ImportError as e:
        # Graceful fallback to legacy methods
        self._setup_legacy_fallbacks()
```

#### **Delegation Pattern**
```python
def load_project_data(self):
    """Public API method with handler delegation."""
    if hasattr(self, 'data_manager'):
        return self.data_manager.load_project_data()
    else:
        return self._legacy_load_project_data()  # Fallback
```

#### **Inter-Handler Communication**
```python
class ParameterManagerHandler:
    def _on_parameter_changed(self):
        """Handle parameter changes with cross-handler communication."""
        # Update step parameters
        self.update_step_parameters()
        
        # Trigger preview update through parent reference
        if hasattr(self.parent, 'preview_manager'):
            self.parent.preview_manager.schedule_preview_update()
```

### **Background Processing**
- **Worker Threads**: Processing operations run in separate QThread instances
- **Progress Tracking**: Real-time progress updates without UI blocking
- **Error Handling**: Comprehensive error capture and user feedback
- **Export Integration**: Seamless CSV/Excel export functionality

### **Real-Time Preview System**
- **Debounced Updates**: 500ms delay prevents excessive updates during parameter changes
- **Pipeline Application**: Live preview applies current pipeline to sample data
- **Status Indicators**: Visual feedback for preview state (ready/processing/error)
- **Performance Optimization**: Uses data sampling for large datasets

### **Legacy Compatibility**
- **Backward Compatibility**: All existing APIs preserved through delegation
- **Graceful Degradation**: Fallback methods ensure functionality if handlers fail
- **Property Compatibility**: Pipeline steps accessible through compatibility properties
- **Signal Preservation**: All Qt signal connections maintained across handlers

---

## 6. Integration

### **Architecture**
- **Modular Design**: Uses reusable widgets from `components/widgets/` package
- **Professional SVG Icons**: Custom spinboxes with decrease-circle.svg and increase-circle.svg
- **Extensible Framework**: Easy addition of new preprocessing methods
- **Robust Error Handling**: Comprehensive exception handling and user feedback
- **Performance Optimization**: Efficient processing for large spectral datasets

### **Widget Integration**
- **Custom Parameter Widgets**: Professional CustomSpinBox and CustomDoubleSpinBox with SVG icons
- **Range Parameter Widget**: Sophisticated dual-input with synchronized sliders
- **Dynamic Parameter System**: Automatic widget creation based on method metadata
- **Real-time Validation**: Immediate parameter constraint enforcement
- **Medical-grade Styling**: Professional appearance optimized for scientific applications

### **Integration**
- **Project Management**: Seamless integration with project workflow through handler composition
- **Data Persistence**: Automatic saving of processed datasets via DataManagerHandler
- **Metadata Management**: Complete processing history and parameter tracking via ProcessingManagerHandler  
- **Visualization Pipeline**: Real-time plotting and display updates via PreviewManagerHandler

This enhanced preprocessing page provides a professional, medical-grade interface for preparing Raman spectral data for disease detection applications, now built on a robust, maintainable architecture that combines advanced technical capabilities with clean code principles.

---

## 6. Architecture Documentation

### **Handler Classes Architecture**
- **[DataManagerHandler](../pages/preprocess_page_utils/data_manager.py)**: Data loading, previewing, and management operations
- **[PipelineManagerHandler](../pages/preprocess_page_utils/pipeline_manager.py)**: Pipeline creation, modification, and step management  
- **[ParameterManagerHandler](../pages/preprocess_page_utils/parameter_manager.py)**: Parameter widget display and real-time management
- **[PreviewManagerHandler](../pages/preprocess_page_utils/preview_manager.py)**: Real-time preview system with automatic updates
- **[ProcessingManagerHandler](../pages/preprocess_page_utils/processing_manager.py)**: Pipeline execution and result export

### **Implementation Patterns**
- **Composition Pattern**: Main class delegates responsibilities to specialized handlers
- **Single Responsibility**: Each handler focuses on one business domain
- **Backward Compatibility**: Delegation pattern preserves existing APIs
- **Graceful Fallbacks**: Robust error handling with method availability checks

---

## 7. Related Documentation

- **[Widgets Component Package](../docs/widgets-component-package.md)**: Comprehensive documentation for the reusable parameter widgets used throughout the preprocessing interface
- **[Enhanced Parameter Widgets](../docs/enhanced-parameter-widgets.md)**: Technical details on widget implementation and customization
- **[Parameter Widgets Fixes](../docs/parameter-widgets-fixes.md)**: Historical documentation of widget improvements and bug fixes

