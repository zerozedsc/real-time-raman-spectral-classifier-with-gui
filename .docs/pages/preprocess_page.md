# Preprocessing Page Documentation

## 🎯 Recent Updates (2025)

### **Preview Toggle State Management Fix (October 21, 2025) ✅**

#### **Critical Bug Fix: Preview Toggle State Synchronization**
Fixed critical bugs where preview toggle button showed incorrect state, causing confusion about preview mode status.

**Issues Resolved**:
1. ❌ **Non-existent method calls** - Code called `_update_preview_toggle_button_style()` which didn't exist (6 occurrences)
2. ❌ **Duplicate method definitions** - Two `_update_preview_button_state()` methods (one hardcoded Japanese, one localized)
3. ❌ **Desynchronized state** - Button visual didn't match `checked` state or `preview_enabled` flag

**Root Cause**:
- Methods like `_on_dataset_tab_changed()` and `_on_dataset_selection_changed()` called non-existent `_update_preview_toggle_button_style()`
- Python used last method definition (localized version), but duplicate caused maintenance issues
- Missing `self.preview_enabled` flag updates after toggle state changes

**Solution - Three-Way State Synchronization**:
```python
# Correct pattern used throughout codebase
if condition:
    # 1. Update checked state
    self.preview_toggle_btn.blockSignals(True)
    self.preview_toggle_btn.setChecked(True)
    self.preview_toggle_btn.blockSignals(False)
    
    # 2. Update visual appearance
    self._update_preview_button_state(True)
    
    # 3. Update internal flag
    self.preview_enabled = True
```

**Preview Toggle Behavior** (User-Facing):
- **Preview ON (プレビュー ON)**: 
  - Shows real-time processed data with pipeline preview
  - Graph updates as you add/modify preprocessing steps
  - Best for raw datasets to see processing effects
  
- **Preview OFF (プレビュー OFF)**:
  - Shows original/current dataset state **without** processing
  - Graph still displayed (NOT cleared/disabled)
  - Best for preprocessed datasets to avoid double-preprocessing confusion

**Smart Default State**:
- **Raw datasets** → Preview defaults to **ON** (see processing effects)
- **Preprocessed datasets** → Preview defaults to **OFF** (avoid double preprocessing)
- **Tab switching** → Preview auto-adjusts based on active tab (All/Raw = ON, Preprocessed = OFF)

**Files Modified**:
- Replaced 6 calls to non-existent method with correct `_update_preview_button_state()`
- Removed duplicate hardcoded method definition (lines 3261-3303)
- Added `preview_enabled` flag synchronization in tab and selection handlers

**Impact**:
- ✅ Preview button text now always matches actual state
- ✅ No more silent errors from missing methods
- ✅ Preview mode correctly indicated at all times
- ✅ Users can trust the displayed preview state
- ✅ Prevents preprocessing confusion with preprocessed datasets

**Related Documentation**:
- `.AGI-BANKS/IMPLEMENTATION_PATTERNS.md` section 0.4: "Preview Toggle State Synchronization Pattern"
- `.AGI-BANKS/RECENT_CHANGES.md`: "October 21, 2025 - Preview Toggle & Dataset Info Enhancements"

---

### **Height Optimization for Non-Maximized Windows (October 6 Evening #2, 2025) ⚙️**

#### **Critical Design Constraint**
**Non-Maximized Window Support**: Application must work in non-maximized mode (e.g., 800x600 resolution). All sections optimized for smaller heights.

#### **Dataset List Height Reduction (NEW)**
- **Previous**: 280-350px (showed 3-4 items, too tall for small windows)
- **New**: **140-165px** (shows exactly **4 items** before scrolling)
- **Calculation**: 4 items × 40px/item + padding = 165px
- **Space saved**: 185px (-53% reduction)
- **User Experience**: Perfect for non-maximized, y-scroll for 5+ items

#### **Pipeline List Height Optimization (NEW)**
- **Previous**: 300-400px (showed 8-10 steps, text cutoff issues)
- **New**: **180-215px** (shows exactly **5 steps** before scrolling)
- **Item height increased**: 32px → **38px** min-height
- **Padding increased**: 8px 6px → **10px 8px** (more vertical space)
- **Calculation**: 5 steps × 40px/step + padding = 215px
- **Space saved**: 185px (-46% reduction)
- **Text visibility**: **FIXED** - "その他前処理 - Cropper" fully visible

#### **Visualization Header Compactification (NEW)**
- **Button size reduction**:
  - Preview toggle: 32px → **28px** height, 120px → **110px** width
  - Manual refresh/focus: 32x32px → **28x28px**
  - Icon sizes: 16x16px → **14x14px**
- **Layout optimization**:
  - Explicit margins: **12px** all sides
  - Spacing: 15px → **8px** between controls
  - Removed redundant "Preview:" label container
- **Font size reduction**:
  - Status dot: 14px → **12px**
  - Status text: 11px → **10px**
- **Space saved**: 14px (-28% reduction)

#### **Height Calculation Formula**
```python
# List widget height calculation
list_height = (items_to_show × item_height) + padding + borders

# Dataset list example
dataset_height = (4 items × 40px/item) + 25px padding = 165px

# Pipeline list example  
pipeline_height = (5 steps × 40px/step) + 15px padding = 215px
```

#### **Design Principles for Non-Maximized Windows**
1. **List Heights**: Calculate based on items × item_height + padding
2. **Maximum Items**: Show 4-5 items before scrolling
3. **Button Sizes**: Use 28x28px for compact controls
4. **Icon Sizes**: Use 14x14px for compact buttons
5. **Spacing**: Use 8px between controls in compact headers
6. **Font Sizes**: Reduce by 1-2px in compact layouts
7. **Margins**: Use explicit 12px margins for consistency

#### **Implementation Code**

**Dataset List Configuration**:
```python
# Configure all dataset list widgets (all tabs)
for list_widget in [self.dataset_list_all, self.dataset_list_raw, self.dataset_list_preprocessed]:
    list_widget.setObjectName("datasetList")
    list_widget.setSelectionMode(QListWidget.ExtendedSelection)
    list_widget.setMinimumHeight(140)
    list_widget.setMaximumHeight(165)  # Show max 4 items before scrolling
```

**Pipeline List Configuration**:
```python
# Pipeline steps list (optimized for non-maximized windows)
self.pipeline_list = QListWidget()
self.pipeline_list.setMinimumHeight(180)
self.pipeline_list.setMaximumHeight(215)  # Show max 5 steps before scrolling
self.pipeline_list.setStyleSheet("""
    QListWidget#modernPipelineList::item {
        padding: 10px 8px;    /* Increased from 8px 6px */
        min-height: 38px;     /* Increased from 32px */
        font-size: 12px;
    }
""")
```

**Visualization Header Configuration**:
```python
# Visualization section with compact layout
plot_layout = QVBoxLayout(plot_group)
plot_layout.setContentsMargins(12, 12, 12, 12)  # Explicit margins
plot_layout.setSpacing(8)  # Compact spacing

# Preview toggle - compact
self.preview_toggle_btn.setFixedHeight(28)  # Reduced from 32
self.preview_toggle_btn.setMinimumWidth(110)  # Reduced from 120

# Compact control buttons
self.manual_refresh_btn.setFixedSize(28, 28)  # Reduced from 32x32
reload_icon = load_svg_icon(get_icon_path("reload"), "#7f8c8d", QSize(14, 14))  # Reduced from 16x16

# Compact status indicators
self.preview_status.setStyleSheet("font-size: 12px;")  # Reduced from 14px
self.preview_status_text.setStyleSheet("font-size: 10px;")  # Reduced from 11px
```

#### **Space Savings Summary**
| Section | Before | After | Savings | Items Shown |
|---------|--------|-------|---------|-------------|
| Dataset List | 280-350px | 140-165px | -185px | 4 items max |
| Pipeline List | 300-400px | 180-215px | -185px | 5 steps max |
| Viz Header | ~50px | ~36px | -14px | Same controls |
| **TOTAL** | - | - | **-384px** | - |

---

### **Input Dataset & Pipeline Section Redesign (October 6 Evening, 2025) ✨✨**

#### **Input Dataset Section - Maximum Content Visibility (NEW)**
- **✅ Hint Button in Title Bar**: 
  - Moved from bottom info row to title bar with "?" icon
  - Fixed size: 20x20px circular blue button
  - Combined multi-select and multi-dataset hints in single tooltip
  - Hover effect: blue background with white text
  - Saves ~40px vertical space
- **✅ Dataset List Height Increase**:
  - Increased from 200px → 350px maximum height
  - Added minimum height: 280px for consistency
  - **Shows 3-4 dataset items before scrolling** (target achieved!)
  - Net space gain: ~110px more content visibility
- **✅ Removed Info Icons Row**:
  - Deleted ℹ️ and 💡 emoji icons row
  - All hint information consolidated in title bar button
  - Cleaner, more professional appearance
- **✅ Custom Title Widget**:
  - QGroupBox with empty title
  - Separate widget for title + hint button layout
  - Better visual hierarchy

#### **Pipeline Section - Professional SVG Icons (NEW)**
- **✅ Plus Icon (Add Step)**:
  - Replaced ➕ emoji with `plus.svg` icon
  - White 24x24px SVG on blue background
  - Professional appearance in 60x50px button
- **✅ Trash Icon (Remove Step)**:
  - Replaced 🗑️ emoji with `trash-bin.svg` icon
  - Red-colored (#dc3545) 14x14px SVG icon
  - Proper danger color scheme for delete action
- **✅ Text Overflow Fix**:
  - Increased pipeline item padding: 6px → 8px vertical
  - Added min-height: 32px for items
  - Prevents text cutoff for long names like "その他前処理 - Cropper"

#### **Implementation Code Examples**

**Custom Title with Hint Button**:
```python
# Create custom title widget with hint button
title_widget = QWidget()
title_layout = QHBoxLayout(title_widget)
title_layout.setContentsMargins(0, 0, 0, 0)
title_layout.setSpacing(8)

title_label = QLabel(LOCALIZE("PREPROCESS.input_datasets_title"))
title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
title_layout.addWidget(title_label)

# Hint button with ? icon
hint_btn = QPushButton("?")
hint_btn.setObjectName("hintButton")
hint_btn.setFixedSize(20, 20)
hint_btn.setToolTip(
    LOCALIZE("PREPROCESS.multi_select_hint") + "\\n\\n" +
    LOCALIZE("PREPROCESS.multi_dataset_hint")
)
hint_btn.setStyleSheet("""
    QPushButton#hintButton {
        background-color: #e7f3ff;
        color: #0078d4;
        border: 1px solid #90caf9;
        border-radius: 10px;
        font-weight: bold;
        font-size: 11px;
    }
    QPushButton#hintButton:hover {
        background-color: #0078d4;
        color: white;
    }
""")
```

**SVG Icons in Pipeline**:
```python
# Add step button with plus.svg icon
add_step_btn = QPushButton()
plus_icon = load_svg_icon(get_icon_path("plus"), "white", QSize(24, 24))
add_step_btn.setIcon(plus_icon)
add_step_btn.setIconSize(QSize(24, 24))
add_step_btn.setFixedSize(60, 50)

# Remove button with trash-bin.svg icon
remove_btn = QPushButton()
trash_icon = load_svg_icon(get_icon_path("trash_bin"), "#dc3545", QSize(14, 14))
remove_btn.setIcon(trash_icon)
remove_btn.setIconSize(QSize(14, 14))
```

**Dataset List Height**:
```python
# Configure all dataset list widgets
for list_widget in [self.dataset_list_all, self.dataset_list_raw, self.dataset_list_preprocessed]:
    list_widget.setSelectionMode(QListWidget.ExtendedSelection)
    list_widget.setMinimumHeight(280)
    list_widget.setMaximumHeight(350)  # Shows 3-4 items before scrolling
```

---

### **UI Optimization & Refactoring Plan (October 6, 2025) ✨**

#### **Icon-Only Buttons (NEW)**
- **✅ Space-Efficient Design**: Converted refresh and export buttons to icon-only format
- **✅ Saved Space**: ~200px horizontal space reclaimed in button row
- **✅ SVG Icons**: Using centralized icon system with color customization
  - Refresh: Blue reload icon (#0078d4, 18x18px)
  - Export: Green export icon (#2e7d32, 18x18px)
- **✅ Hover Tooltips**: Full text shown on hover for accessibility
- **✅ Visual Feedback**: Hover and pressed states with color transitions
- **✅ Fixed Size**: 36x36px buttons with 6px rounded corners
- **✅ Pointer Cursor**: Indicates clickability on hover

#### **Optimized Pipeline Construction Section (NEW)**
- **✅ Compact Layout**: Single-row layout for category and method selection
  - Replaced vertical card layout with horizontal row
  - Reduced spacing from 12px to 8px
  - Smaller labels (11px font) and dropdowns
- **✅ Enlarged Pipeline List**: 
  - Increased minimum height from 250px to 300px
  - Added maximum height of 400px for better scrolling
  - More compact item styling (6px padding, 12px font)
- **✅ Icon-Only Control Buttons**:
  - Remove (🗑️), Clear (🧹), Toggle (🔄)
  - Compact 28px height
  - Tooltips provide full descriptions
- **✅ Prominent Add Button**:
  - Large plus icon (➕, 20px font)
  - 60x50px size to match two-row height
  - Blue background (#0078d4)

#### **Icon-Only Button Implementation**
```python
# Refresh button - icon only
refresh_btn = QPushButton()
refresh_btn.setObjectName("iconOnlyButton")
reload_icon = load_svg_icon(get_icon_path("reload"), "#0078d4", QSize(18, 18))
refresh_btn.setIcon(reload_icon)
refresh_btn.setIconSize(QSize(18, 18))
refresh_btn.setFixedSize(36, 36)
refresh_btn.setToolTip(LOCALIZE("PREPROCESS.refresh_datasets_tooltip"))
refresh_btn.setCursor(Qt.PointingHandCursor)

# Export button - icon only (green)
export_btn = QPushButton()
export_btn.setObjectName("iconOnlyButtonGreen")
export_icon = load_svg_icon(get_icon_path("export"), "#2e7d32", QSize(18, 18))
export_btn.setIcon(export_icon)
export_btn.setIconSize(QSize(18, 18))
export_btn.setFixedSize(36, 36)
export_btn.setToolTip(LOCALIZE("PREPROCESS.export_button_tooltip"))
export_btn.setCursor(Qt.PointingHandCursor)
```

#### **Compact Pipeline Layout**
```python
# Single-row layout for category and method
selection_row = QHBoxLayout()
selection_row.setSpacing(8)

# Category dropdown (compact)
cat_container = QVBoxLayout()
cat_container.setSpacing(4)
cat_label = QLabel("📂 " + LOCALIZE("PREPROCESS.category"))
cat_label.setStyleSheet("font-weight: 500; color: #495057; font-size: 11px;")
self.category_combo.setStyleSheet("""
    QComboBox {
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 5px 8px;
        background: white;
        font-size: 12px;
    }
""")

# Method dropdown (compact)
# ... similar structure

# Add button (tall, icon-only)
add_step_btn = QPushButton("➕")
add_step_btn.setFixedSize(60, 50)  # Tall enough for two rows
```

#### **Comprehensive Refactoring Plan (NEW)**
Created detailed 800+ line refactoring plan document:

**Current State**:
- 3060 lines in single file
- 75 methods in one class
- 40+ inline style definitions
- Mixed UI, business logic, and data handling

**Proposed Structure**:
- Main coordinator: 600-800 lines
- 7 specialized modules: 1900+ lines
- Centralized styles: 800 lines

**New Modules**:
1. `ui_components.py` (400 lines) - UI creation methods
2. `dataset_manager.py` (300 lines) - Dataset operations
3. `pipeline_manager.py` (250 lines) - Pipeline operations
4. `preview_manager.py` (300 lines) - Preview functionality
5. `parameter_manager.py` (200 lines) - Parameter widgets
6. `history_manager.py` (250 lines) - History display
7. `styles.py` (800 lines) - All style definitions

**Migration Strategy**:
- **Phase 1**: Style extraction (2-3 hours)
- **Phase 2**: UI component extraction (3-4 hours)
- **Phase 3**: Manager classes (4-5 hours)
- **Phase 4**: Integration & testing (2-3 hours)
- **Phase 5**: Optimization & polish (1-2 hours)
- **Total**: 12-17 hours

**Success Metrics**:
- ✅ 70-75% reduction in main file size
- ✅ Zero inline styles
- ✅ Clear separation of concerns
- ✅ Improved testability
- ✅ No performance regression

**Documentation**:
- See `.docs/pages/PREPROCESS_PAGE_REFACTORING_PLAN.md` for full details

---

### **UI/UX Modernization (October 2025) ✨**

#### **Hover Tooltip System (NEW)**
- **✅ Space-Saving Design**: Replaced always-visible hint label with hover tooltips
- **✅ Interactive Icons**: Info icon (ℹ️) and lightbulb icon (💡) with hover states
- **✅ Rich Tooltips**: HTML-formatted tooltips with bullet points and bold text
- **✅ Visual Feedback**: Icons highlight on hover with subtle background color
- **✅ Medical Theme**: Blue accent color (#0078d4) consistent with application theme

#### **Multi-Selection & Multi-Dataset Hints**
- **ℹ️ Multi-Selection Tooltip**:
  - Instructions for Ctrl/Cmd + Click to select multiple datasets
  - Instructions for Shift + Click to select range
  - Appears on hover over info icon below dataset tabs
  
- **💡 Multi-Dataset Processing Tooltip**:
  - Explains that multiple datasets are combined via `pd.concat()`
  - Clarifies that preprocessing pipeline applies to all selected data simultaneously
  - One output dataset is created from combined processing
  - Appears on hover over lightbulb icon below dataset tabs

#### **Dataset Selection Synchronization (FIXED)**
- **✅ Cross-Tab Selection Events**: All three dataset tabs (All, Raw, Preprocessed) now trigger graph updates
- **✅ Unified Handler**: `_on_dataset_selection_changed()` handles selection from all tabs consistently
- **✅ Tab Switch Sync**: Changing tabs automatically triggers selection event and updates visualization
- **✅ Signal Architecture**: All QListWidget instances connected to same selection handler
- **✅ Backward Compatible**: Maintains reference to active list for legacy code

#### **Modern Confirmation Dialog**
Redesigned preprocessing confirmation dialog with clean, professional layout:

- **Header Section**:
  - Clean white gradient background (#ffffff → #f8fbff)
  - Subtle border (#d0dae6)
  - Large icon (🔬) separated from title text
  - Elegant divider line (#e1e4e8)
  
- **Metric Display**:
  - Three card-based metrics in grid layout
  - Vertical layout: Icon (20px) → Value (24px bold #0078d4) → Label (11px uppercase)
  - Hover effect: Border changes to blue, gradient shifts to blue tint
  - Professional spacing and sizing
  - Values: Input dataset count, Pipeline step count, Output name (truncated at 25 chars)

- **Modern Styling**:
  - Replaced old badge-style with card-based metrics
  - Removed colorful gradients in favor of subtle white gradients
  - Consistent medical theme throughout
  - Better visual hierarchy with improved typography

#### **Modernized Pipeline Creation Section**
Complete redesign of pipeline building interface with enhanced medical theme:

- **Selection Card**:
  - White card container (#ffffff) with subtle border (#e1e4e8)
  - Rounded corners (8px radius)
  - Category and Method dropdowns with icons (📂, ⚙️)
  - Enhanced dropdown styling:
    - White background with border (#ced4da)
    - Blue border on hover/focus (#0078d4)
    - Rounded corners (6px)
    - Proper padding (6px 10px)
  
- **Primary Add Button**:
  - Prominent blue background (#0078d4)
  - White text with plus icon (➕)
  - Rounded corners (6px)
  - Hover/pressed states (#106ebe, #005a9e)
  - Full-width within card
  
- **Pipeline Steps List**:
  - Gray background (#f8f9fa) with subtle border
  - White item cards with rounded corners (6px)
  - Selection highlight with blue tint (#e7f3ff)
  - Hover effect with light blue (#f0f8ff)
  - 250px max height with scrolling
  
- **Control Buttons**:
  - Secondary button style (gray background #f8f9fa)
  - Icons: 🗑️ Remove, 🧹 Clear, 🔄 Toggle
  - Hover states with darker gray (#e9ecef)
  - Consistent spacing (8px between buttons)
  - Font size 12px for compact look

- **Enhanced Typography**:
  - Pipeline Steps label with emoji (📋) and bold font
  - Category/Method labels with medium weight (500)
  - Professional color scheme (#495057 for labels)

#### **Stylesheet Additions**
- **`selection_card`**: White card style for category/method selection
- **`modern_pipeline_group`**: QGroupBox styling for pipeline section
- **`metricItem`**: Card-based metric display for confirmation dialog
- **Enhanced button styles**: Primary and secondary button variants

#### **Technical Implementation**
- **Signal/Slot Architecture**: 
  - All three QListWidget instances connect to `_on_dataset_selection_changed()`
  - Tab change event triggers selection handler
  - Single unified handler manages all selection events
  
- **Helper Methods**:
  - `_create_metric_item()`: Creates modern metric cards for confirmation dialog
  - Replaced old `_create_info_badge()` method
  
- **Localization**:
  - `multi_select_hint`: HTML-formatted multi-selection instructions
  - `multi_dataset_hint`: HTML-formatted multi-dataset processing explanation
  - `pipeline_steps_label`: Label for pipeline steps list
  - Full English and Japanese support

---

### **Export Feature Enhancements (October 2025) ✨**

#### **Metadata JSON Export (NEW)**
- **✅ Automatic Metadata Export**: When exporting datasets, companion metadata JSON files are automatically generated
- **✅ Comprehensive Metadata**: Includes export date, preprocessing pipeline, source datasets, spectral info
- **✅ Optional Export**: User can toggle metadata export via checkbox
- **✅ File Format**: `{filename}_metadata.json` alongside dataset file

#### **Location Validation (NEW)**
- **✅ Smart Validation**: Warning dialog prevents export without selected location
- **✅ Path Validation**: Checks for both empty and non-existent paths
- **✅ User-Friendly**: Clear instructions to select location before proceeding
- **✅ Prevents Errors**: Stops file system errors from invalid paths

#### **Default Location Persistence (NEW)**
- **✅ Remember Last Path**: Automatically remembers and pre-fills last used export location
- **✅ Session Persistence**: Location persists during application session
- **✅ Browse Optimization**: File browser starts from last location
- **✅ Improved Workflow**: Saves time on repeated exports

#### **Multiple Dataset Export (NEW)**
- **✅ Batch Processing**: Export multiple selected datasets simultaneously
- **✅ Dynamic UI**: Shows dataset count, adapts interface for batch mode
- **✅ Individual Naming**: Each dataset exported with original name
- **✅ Comprehensive Feedback**: Success count, failure tracking, partial success warnings
- **✅ Error Recovery**: Continues processing remaining datasets if one fails

#### **Export Dialog Features**
- Format selection: CSV, TXT, ASC, Pickle
- Browse button with smart location memory
- Filename input (single export) or automatic naming (multiple export)
- Metadata checkbox with tooltip
- Full English/Japanese localization

#### **Metadata JSON Structure**
```json
{
  "export_info": {
    "export_date": "ISO timestamp",
    "dataset_name": "string",
    "data_shape": {"rows": int, "columns": int}
  },
  "preprocessing": {
    "is_preprocessed": boolean,
    "processing_date": "ISO timestamp",
    "source_datasets": ["list"],
    "pipeline": [{"category": "str", "method": "str", "params": {}, "enabled": bool}],
    "pipeline_summary": {"total_steps": int, "successful_steps": int, ...}
  },
  "spectral_info": {
    "num_spectra": int,
    "spectral_axis_start": float,
    "spectral_axis_end": float,
    "spectral_points": int
  }
}
```

---

### **Critical Bug Fixes and Enhancements (September 2025)**

#### **Global Pipeline Memory System**
- **✅ Persistent Pipeline Steps**: Implemented global memory system to prevent pipeline steps from vanishing when switching between datasets
- **✅ Cross-Dataset State Management**: Pipeline steps now persist across dataset switches using `_global_pipeline_memory`
- **✅ Automatic Save/Restore**: Pipeline state automatically saved on modifications and restored on dataset selection
- **✅ UI Reconstruction**: Added `_rebuild_pipeline_ui()` for seamless interface rebuilding with saved steps

#### **Enhanced X-axis Padding for Cropped Regions**
- **✅ Crop Boundary Visualization**: When cropping regions (e.g., 600-1800 cm⁻¹), boundaries are now properly visible with padding
- **✅ Smart Crop Detection**: Added `_extract_crop_bounds()` method to automatically detect cropping steps in pipeline
- **✅ Matplotlib Integration**: Enhanced `matplotlib_widget.py` with `crop_bounds` parameter for proper boundary handling
- **✅ Fixed Padding Implementation**: Default padding changed from percentage-based to fixed ±50 wavenumber units
- **✅ Parameter Persistence**: Pipeline parameters now preserved when switching datasets (not just steps)
- **✅ Enhanced Global Memory**: `_update_current_step_parameters()` captures widget values before saving to memory

#### **Preview OFF Functionality Fix**
- **✅ Original Data Display**: Preview OFF now properly shows original dataset data instead of empty graphs
- **✅ Data Loading Fix**: Enhanced `_update_preview()` method to correctly load data from `RAMAN_DATA` when preview disabled
- **✅ State Management**: Proper handling of preview toggle between processed and original data views

#### **Professional UI Color Scheme**
- **✅ Removed Orange Elements**: Replaced all orange UI elements with professional gray/blue color scheme
- **✅ Processing Status**: Changed processing indicators from orange `#f39c12` to dark gray `#666666`
- **✅ Pipeline Widget Colors**: Updated imported step colors to blue scheme (Dark blue `#1976d2` enabled, Light blue `#64b5f6` disabled)
- **✅ Better Accessibility**: Improved color contrast and visual distinction for enabled/disabled states

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

