# Preprocessing Page Documentation

## 1. Overview

The **Preprocessing** page is an advanced, interactive module for cleaning and preparing Raman spectral data for medical diagnosis applications. It provides a comprehensive, pipeline-based workflow with detailed parameter controls, real-time visualization, and robust error handling designed specifically for disease detection scenarios.

This page fulfills the core requirement of **"前処理" (Preprocessing)** as outlined in the project's development plan for real-time Raman spectral classification.

---

## 2. Enhanced Page Layout & Components

The page features a modern, two-panel layout optimized for efficient preprocessing workflows:

### **Left Panel: Enhanced Control Interface**

#### **入力データセット (Input Datasets)**
- **Multi-selection Support**: Select multiple datasets simultaneously for batch processing
- **Visual Feedback**: Clear selection indicators and hover states
- **Data Validation**: Automatic validation of selected datasets
- **Status Display**: Shows when no datasets are available

#### **パイプライン設定 (Pipeline Configuration)**
- **Advanced Step Selection**: Comprehensive dropdown with Japanese preprocessing method names:
  - スペクトル切り出し (Spectral Cropping)
  - Savitzky-Golay フィルタ (Savitzky-Golay Filter)
  - ASPLS ベースライン補正 (ASPLS Baseline Correction)
  - マルチスケール畳み込み (Multi-scale Convolution)
  - ベクトル正規化 (Vector Normalization)
  - SNV 正規化 (SNV Normalization)

- **Pipeline Management**:
  - Drag-and-drop reordering of processing steps
  - Visual step indicators with medical-themed styling
  - One-click step removal and pipeline clearing
  - Real-time parameter validation

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
- **Interactive Matplotlib Integration**: High-quality spectral plots
- **Real-time Updates**: Immediate visualization of raw data selection
- **Before/After Comparison**: Clear visualization of preprocessing effects
- **Medical-grade Plotting**: Optimized for diagnostic applications
- **Responsive Design**: Adaptive plot sizing and formatting

---

## 3. Enhanced Workflow

### **Data Preparation Phase**
1. **Dataset Selection**: Choose single or multiple datasets for processing
2. **Data Validation**: Automatic validation of spectral data format and quality
3. **Preview Generation**: Immediate visualization of selected raw data

### **Pipeline Configuration Phase**
1. **Method Selection**: Choose from medical-optimized preprocessing methods
2. **Step Organization**: Drag-and-drop reordering for optimal processing sequence
3. **Parameter Optimization**: Detailed parameter adjustment with medical context
4. **Validation**: Real-time parameter validation and feedback

### **Execution Phase**
1. **Output Naming**: Intelligent naming with conflict resolution
2. **Processing**: Robust execution with comprehensive error handling
3. **Result Visualization**: Immediate visualization of processed spectra
4. **Project Integration**: Automatic integration into project workflow

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
- **Modular Design**: Separate parameter widgets for each processing method
- **Extensible Framework**: Easy addition of new preprocessing methods
- **Robust Error Handling**: Comprehensive exception handling and user feedback
- **Performance Optimization**: Efficient processing for large spectral datasets

### **Integration**
- **Project Management**: Seamless integration with project workflow
- **Data Persistence**: Automatic saving of processed datasets
- **Metadata Management**: Complete processing history and parameter tracking
- **Visualization Pipeline**: Real-time plotting and display updates

This enhanced preprocessing page provides a professional, medical-grade interface for preparing Raman spectral data for disease detection applications, combining advanced technical capabilities with user-friendly design principles.

