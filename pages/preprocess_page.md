# Preprocessing Page Documentation

## 1. Overview

The **Preprocessing** page is an advanced, interactive module for cleaning and preparing Raman spectral data for medical diagnosis applications. It provides a comprehensive, pipeline-based workflow with detailed parameter controls, **automatic real-time preview**, and robust error handling designed specifically for disease detection scenarios.

This page fulfills the core requirement of **"前処理" (Preprocessing)** as outlined in the project's development plan for real-time Raman spectral classification.

---

## 2. Enhanced Page Layout & Components

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
- **Project Management**: Seamless integration with project workflow
- **Data Persistence**: Automatic saving of processed datasets
- **Metadata Management**: Complete processing history and parameter tracking
- **Visualization Pipeline**: Real-time plotting and display updates

This enhanced preprocessing page provides a professional, medical-grade interface for preparing Raman spectral data for disease detection applications, combining advanced technical capabilities with user-friendly design principles.

---

## 6. Related Documentation

- **[Widgets Component Package](../docs/widgets-component-package.md)**: Comprehensive documentation for the reusable parameter widgets used throughout the preprocessing interface
- **[Enhanced Parameter Widgets](../docs/enhanced-parameter-widgets.md)**: Technical details on widget implementation and customization
- **[Parameter Widgets Fixes](../docs/parameter-widgets-fixes.md)**: Historical documentation of widget improvements and bug fixes

