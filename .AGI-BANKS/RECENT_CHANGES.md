# Recent Changes and UI Improvements

## Summary of Recent Changes
This document tracks the most recent modifications made to the Raman spectroscopy application, focusing on preprocessing interface fixes, pipeline management improvements, enable/disable toggles, and localization enhancements.

## Latest Updates (Comprehensive Preprocessing Fixes - September 25, 2025)

### Global Pipeline Memory System & UI Improvements

#### 1. Implemented Global Pipeline Memory System
**Issue:** Pipeline steps vanishing when switching between datasets
**Solution:** Added persistent global memory system in `pages/preprocess_page.py`
- Added `_global_pipeline_memory: List[PipelineStep]` for cross-dataset persistence
- Implemented `_save_to_global_memory()`, `_restore_global_pipeline_memory()`, `_clear_global_memory()`
- Added `_rebuild_pipeline_ui()` for reconstructing interface with saved steps
- Pipeline steps now automatically saved on add/remove/toggle operations

#### 2. Enhanced X-axis Padding for Cropped Regions
**Issue:** Cropped spectrum (e.g., 600-1800 cm⁻¹) not showing boundaries with proper padding
**Solution:** Enhanced matplotlib_widget.py and plot method calls
- Added `crop_bounds` parameter to `detect_signal_range()` function
- Modified `plot_spectra()` and related methods to accept `crop_bounds`
- Added `_extract_crop_bounds()` method to detect cropping steps in pipeline
- **Final Fix:** Changed default padding from percentage-based (10% of range) to fixed ±50 wavenumber units
- Ensures consistent cropped region boundaries are always visible with appropriate context

#### 6. Parameter Persistence Enhancement (September 25, 2025)
**Issue:** Pipeline steps persisted but parameter values reset to defaults when switching datasets
**Solution:** Enhanced global memory system to preserve parameter values
- Added `_update_current_step_parameters()` method to capture widget values before saving
- Enhanced `_save_to_global_memory()` to preserve both steps and their parameter values
- Parameters now correctly restored when switching back to datasets with existing pipelines

#### 3. Fixed Preview OFF Functionality
**Issue:** Preview OFF showing empty graph instead of original dataset
**Solution:** Updated `_update_preview()` method in `pages/preprocess_page.py`
- Preview OFF now properly loads original dataset data from `RAMAN_DATA`
- Calls `_show_original_data()` to display selected dataset without preprocessing
- Ensures `self.original_data` is populated correctly when preview disabled

#### 4. Updated Color Scheme - Removed Orange UI Elements
**Issue:** Orange fonts should be black/gray with proper enabled/disabled coding
**Solution:** Comprehensive color scheme updates
- Changed processing color from orange `#f39c12` to dark gray `#666666`
- Updated pipeline widget imported step colors: Orange → Dark blue `#1976d2` (enabled), Light blue `#64b5f6` (disabled)
- Updated all `_update_preview_status()` calls to use "processing" color instead of "orange"
- Maintains visual distinction while improving accessibility and professionalism

#### 5. Manual Preview Button Verification
**Issue:** Manual preview button behavior unclear
**Solution:** Verified `_manual_refresh_preview()` method functionality
- Confirmed proper operation for both preview ON/OFF modes
- Appropriate error handling with user notifications
- Uses debounced updates for responsive UI

## Previous Updates (September 16, 2025)

### Major Bug Fixes and Feature Additions

#### 1. Fixed Preprocessed Dataset Auto-Visualization
**Issue:** Preprocessed datasets showed yellow "processing" alert but never completed visualization
**Root Cause:** `_update_preview()` method used hardcoded emoji replacements that didn't include 🔬 emoji for preprocessed datasets
**Files Modified:** `pages/preprocess_page.py`

**Solution:**
- Changed `dataset_name = first_item.text().replace("📊 ", "").replace("🔄 ", "")` 
- To `dataset_name = self._clean_dataset_name(first_item.text())`
- Uses centralized emoji cleaning method that handles all dataset types

**Result:** Preprocessed datasets now properly load and display visualization automatically

#### 2. Fixed Pipeline Step Processing Errors
**Issue:** `'PipelineStep' object has no attribute 'get_method_instance'` and `'get_parameters'` errors
**Root Cause:** Incorrect method names being called on PipelineStep objects
**Files Modified:** `pages/preprocess_page.py`

**Solution:**
- Changed `step.get_method_instance()` → `step.create_instance()`
- Changed `step.get_parameters()` → `step.params` (direct attribute access)
- Fixed attribute access pattern to match PipelineStep class design

**Result:** Pipeline processing now works without attribute errors

#### 3. Fixed Raw Data Preview Logic
**Issue:** Raw datasets showed preprocessing results from previously selected preprocessed datasets
**Root Cause:** `_clear_preprocessing_history()` only cleared UI but not pipeline steps used in preview
**Files Modified:** `pages/preprocess_page.py`

**Solution:**
Enhanced `_clear_preprocessing_history()` to also clear:
- `self.pipeline_steps.clear()`
- `self.pipeline_list.clear()`
- `self.toggle_all_btn.setVisible(False)`

**Result:** Raw datasets now correctly show original data without previous preprocessing

#### 4. Improved Pipeline Step Transfer Logic
**Issue:** Dialog appeared when moving from preprocessed to raw datasets (unwanted)
**Requirement:** Dialog should only appear when moving between preprocessed datasets
**Files Modified:** `pages/preprocess_page.py`

**Solution:**
- Added `self._last_selected_was_preprocessed` tracking variable
- Modified logic to only show dialog when: previous selection was preprocessed AND current selection is also preprocessed
- Raw dataset selection always clears pipeline without prompting

**Result:** Intuitive pipeline transfer behavior matching user expectations

#### 5. Added Individual Step Enable/Disable Toggles
**New Feature:** Eye icon toggle buttons for each preprocessing step
**Files Modified:** `pages/preprocess_page_utils/pipeline.py`, localization files
**Icons Used:** `eye_open` (to enable), `eye_close` (to disable)

**Implementation:**
```python
def _toggle_enabled(self):
    """Toggle the enabled state of the preprocessing step."""
    self.step.enabled = not self.step.enabled
    self._update_enable_button()
    self._update_appearance()
    self.toggled.emit(self.step_index, self.step.enabled)

def _update_enable_button(self):
    """Update the enable/disable button icon and tooltip."""
    if self.step.enabled:
        icon = load_icon("eye_close", "button")  # Show eye_close to disable
        tooltip = LOCALIZE("PREPROCESS.disable_step_tooltip")
    else:
        icon = load_icon("eye_open", "button")   # Show eye_open to enable
        tooltip = LOCALIZE("PREPROCESS.enable_step_tooltip")
```

**Features:**
- Real-time preview updates when steps are toggled
- Visual feedback with grayed-out text for disabled steps
- Proper localization support
- Integration with existing pipeline processing

#### 6. Enhanced Localization Support
**Files Modified:** `assets/locales/en.json`, `assets/locales/ja.json`

**New Keys Added:**
```json
"enable_step_tooltip": "Click to enable this preprocessing step",
"disable_step_tooltip": "Click to disable this preprocessing step",
"step_enabled": "Enabled",
"step_disabled": "Disabled"
```

**Japanese Translations:**
```json
"enable_step_tooltip": "この前処理ステップを有効にするにはクリック",
"disable_step_tooltip": "この前処理ステップを無効にするにはクリック",
"step_enabled": "有効",
"step_disabled": "無効"
```

### Technical Improvements

#### Icon Management System
**File Updated:** `components/widgets/icons.py`
- Centralized icon path management with `ICON_PATHS` registry
- Support for `eye_open` and `eye_close` icons used in toggle buttons
- Backward compatibility with existing icon usage

#### Pipeline Step Widget Enhancement
**File Updated:** `pages/preprocess_page_utils/pipeline.py`
- Added enable/disable toggle buttons to `PipelineStepWidget`
- Enhanced visual feedback for step states
- Improved appearance management with enable/disable styling

### Debug and Testing Results

**Comprehensive Testing Results:**
- ✅ Preprocessed dataset auto-visualization: Working
- ✅ Manual preview button functionality: Working  
- ✅ Pipeline step processing: No attribute errors
- ✅ Raw data preview logic: Correct behavior
- ✅ Pipeline transfer dialog: Only between preprocessed datasets
- ✅ Individual step toggles: Working with real-time preview updates



## Previous Updates (Dynamic UI Sizing - Previous Session)

### Dynamic Preview Toggle Button Sizing
**Issue:** Preview ON/OFF button had fixed width (120px) that couldn't accommodate longer localized text
**Files Modified:** `pages/preprocess_page.py`

**Background:**
The preview toggle button displayed different text lengths:
- English: " Preview ON" / " Preview OFF" 
- Japanese: " プレビュー ON" / " プレビュー OFF" (longer text)

The fixed width (120px) caused text truncation in Japanese locale.

**Solution Implemented:**
1. **Removed Fixed Width**: Changed from `setFixedSize(120, 32)` to `setFixedHeight(32)` only
2. **Added Dynamic Width Calculation**: New method `_adjust_button_width_to_text()`
3. **Font Metrics Integration**: Uses `QFontMetrics` to measure actual text width
4. **Comprehensive Width Calculation**: Accounts for icon (16px), spacing (8px), padding (16px), border (4px)
5. **Minimum Width Guarantee**: Ensures button never gets smaller than 80px

**Technical Implementation:**
```python
def _adjust_button_width_to_text(self):
    """Adjust button width dynamically based on text content."""
    text = self.preview_toggle_btn.text()
    font = self.preview_toggle_btn.font()
    
    font_metrics = QFontMetrics(font)
    text_width = font_metrics.horizontalAdvance(text)
    
    # Calculate total width: text + icon + spacing + padding + border
    total_width = text_width + 16 + 8 + 16 + 4
    dynamic_width = max(80, total_width)  # Minimum 80px
    
    self.preview_toggle_btn.setFixedWidth(dynamic_width)
```

**Benefits:**
- Proper text display in all locales
- Responsive UI that adapts to text length
- Maintains visual consistency across languages
- Prevents UI layout issues with longer translations

## Previous Updates (Localization and Data Format Fixes)

### Data Format Conversion Issue Resolution
**Issue:** "x and y must have same first dimension, but have shapes (2000,) and (28,)" error in preview pipeline after auto-focus implementation
**Files Modified:** `pages/preprocess_page.py`

**Root Cause Analysis:**
The conditional auto-focus implementation introduced a complex data format conversion chain that conflicted with the original DataFrame-based approach:

1. **Original Working Pattern**: DataFrame → `plot_spectra()` directly
2. **Broken Pattern**: DataFrame → SpectralContainer → DataFrame → Array extraction → `plot_comparison_spectra_with_wavenumbers()`
3. **Core Issue**: Dimensional mismatch between expected plotting format `(n_spectra, n_wavenumbers)` vs actual `(n_wavenumbers, n_spectra)`

**Solution Implemented:**
Restored **DataFrame-first approach** in `_apply_preview_pipeline()`:
- Simplified conversion: DataFrame in, DataFrame out
- Minimal SpectralContainer usage: Only temporary conversion for processing step
- Direct DataFrame plotting: Use `plot_spectra(dataframe)` instead of complex array extraction
- Consistent with original working pattern before auto-focus was added

**Key Changes:**
```python
# Old broken approach
processed_data = SpectralContainer(...)  # Complex conversion chain
# ... multiple steps
original_array = original_data.values.T  # Manual array extraction 
plot_comparison_spectra_with_wavenumbers(original_array, processed_array, ...)

# New working approach  
processed_data = data.copy()  # Stay in DataFrame
# ... convert to SpectralContainer only for processing step
# ... immediately convert back to DataFrame
plot_spectra(processed_data, ...)  # Direct DataFrame plotting
```

**Technical Insight:**
The ramanspy library requires SpectralContainer format for processing, but the matplotlib widgets work best with DataFrame format. The solution is to minimize conversion scope - only convert temporarily for each processing step, then immediately back to DataFrame.

## Previous Updates (DataFrame Comparison and Parameter Widget Fixes)

### 1. DataFrame Comparison Bug Fix
**Issue:** "The truth value of a DataFrame is ambiguous" error when toggling preview mode
**Files Modified:** `pages/preprocess_page.py`

**Root Cause:** Direct DataFrame comparison using `not processed_data.equals(original_data)` was causing ambiguity errors in pandas.

**Solution Implemented:**
```python
# Safe DataFrame comparison
data_modified = False
try:
    if not processed_data.shape == original_data.shape:
        data_modified = True
    elif not processed_data.equals(original_data):
        data_modified = True
except Exception:
    # If comparison fails, assume data is modified
    data_modified = True
```

**Locations Fixed:**
- `_update_preview()` method - Lines for comparison plot logic
- `_manual_focus()` method - Lines for forced auto-focus logic

### 2. Parameter Widget Initialization Bug Fix
**Issue:** `TypeError: cannot unpack non-iterable PreprocessPage object` when adding cropper to pipeline
**Files Modified:** `pages/preprocess_page.py`

**Root Cause:** `DynamicParameterWidget` constructor was receiving parameters in wrong order, causing `self` (PreprocessPage instance) to be interpreted as `data_range`.

**Solution Implemented:**
```python
# Added helper method for data range
def _get_data_wavenumber_range(self):
    """Get the wavenumber range from the currently loaded data."""
    if self.original_data is not None and not self.original_data.empty:
        wavenumbers = self.original_data.index.values
        min_wn = float(wavenumbers.min())
        max_wn = float(wavenumbers.max())
        return (min_wn, max_wn)
    else:
        return (400.0, 4000.0)  # Fallback range

# Fixed parameter widget creation
data_range = self._get_data_wavenumber_range()
param_widget = DynamicParameterWidget(method_info, step.params, data_range, self)
```

### 3. Preview Toggle State Management Fix
**Issue:** Graph disappears when toggling preview off and on, requiring manual dataset reload
**Files Modified:** `pages/preprocess_page.py`

**Solution Implemented:**
```python
def _toggle_preview_mode(self, enabled):
    """Toggle preview mode on/off."""
    self.preview_enabled = enabled
    self._update_preview_button_state(enabled)
    
    if enabled:
        # Ensure we have data loaded when turning preview back on
        if not self.original_data:
            self.preview_raw_data()  # This will load data and trigger preview
        else:
            self._update_preview()
    else:
        self.plot_widget.clear_plot()
```

**Enhanced `_update_preview()` method:**
- Better handling when `original_data` is None
- Automatic setting of `self.original_data` from selected dataset
- Improved error handling and fallbacks

## Previous Updates (Auto-Focus Implementation)

### 1. Auto-Focus Implementation Fix
**Issue Identified:** The conditional auto-focus was not working because the core preview and auto-focus methods were missing from the preprocess page.

**Files Modified:**
- `pages/preprocess_page.py`
- `components/widgets/icons.py`

**Key Additions:**

#### Missing Preview Methods Implementation
Added complete preview system with these critical methods:
- `_should_auto_focus()`: Analyzes pipeline for range-limiting steps
- `_schedule_preview_update()`: Debounced preview updates
- `_update_preview()`: Main preview generation with conditional auto-focus
- `_apply_preview_pipeline()`: Pipeline execution for preview
- `preview_raw_data()`: Initial data display without auto-focus

#### Manual Focus Button
**New Feature:** Added manual focus button using `focus-horizontal-round.svg` icon
- **Location**: Next to refresh button in preview controls
- **Functionality**: Forces auto-focus even when conditions aren't met
- **Implementation**: `_manual_focus()` method with forced `auto_focus=True`

#### Icon Registry Update
Added focus icon to the icon registry:
```python
"focus_horizontal": "focus-horizontal-round.svg",  # Manual focus button
```

### 2. Conditional Auto-Focus Logic
**Implementation Details:**
```python
def _should_auto_focus(self) -> bool:
    """Check if auto-focus should be enabled based on pipeline contents."""
    range_limiting_steps = ['Cropper', 'Range Selector', 'Spectral Window', 'Baseline Range']
    enabled_steps = [step for step in self.pipeline_steps if step.enabled]
    return any(step.method in range_limiting_steps for step in enabled_steps)
```

**Trigger Conditions:**
- Cropper: Range selection preprocessing
- Range Selector: Spectral window selection
- Spectral Window: Frequency range limiting
- Baseline Range: Baseline correction with range limits

### 3. Preview System Integration
**Auto-Focus Behavior:**
- **Original Data View**: Always `auto_focus=False` to show full spectrum
- **Comparison View**: Uses `_should_auto_focus()` result for conditional focusing
- **Manual Focus**: Forces `auto_focus=True` regardless of pipeline contents

**Preview Generation:**
```python
def _update_preview(self):
    # Determine if auto-focus should be used
    auto_focus = self._should_auto_focus()
    
    # Safe comparison and plotting logic
    if processed_data is not None:
        data_modified = False
        try:
            if not processed_data.shape == original_data.shape:
                data_modified = True
            elif not processed_data.equals(original_data):
                data_modified = True
        except Exception:
            data_modified = True
        
        if data_modified:
            self.plot_widget.plot_comparison_spectra_with_wavenumbers(
                original_data.values, processed_data.values,
                original_data.index.values, processed_data.index.values,
                titles=["Original", "Processed"],
                auto_focus=auto_focus  # Conditional auto-focus
            )
```

## User Experience Improvements

### 1. Error Prevention
- **Safe DataFrame Operations**: Prevents pandas ambiguity errors
- **Robust Parameter Widget Creation**: Handles data range calculation safely
- **State Management**: Preview mode toggling works reliably

### 2. Technical Improvements
- **Graceful Error Handling**: Try-catch blocks for DataFrame operations
- **Data Range Calculation**: Centralized method for wavenumber range determination
- **Preview State Consistency**: Automatic data reloading when needed

### 3. Stability Enhancements
- **No More Graph Disappearing**: Preview toggle works without requiring manual reload
- **Parameter Widget Stability**: Cropper and other range widgets initialize correctly
- **Error Recovery**: Fallback mechanisms for various edge cases

## Testing Status

### ✅ Fixed Issues
- ✅ DataFrame comparison errors resolved
- ✅ Parameter widget initialization fixed
- ✅ Preview toggle state management working
- ✅ Auto-focus conditional logic implemented
- ✅ Manual focus button functional

### 🔄 Testing Recommendations
- Test with various dataset sizes and types
- Verify parameter widgets work across all preprocessing methods
- Check preview behavior with complex multi-step pipelines
- Test error recovery scenarios

## Future Enhancements

### 1. User Customization
- Allow users to configure auto-focus sensitivity
- Provide manual override options for auto-focus behavior
- Add user preferences for focus detection parameters

### 2. Advanced Features
- Multi-region focus detection for complex spectra
- Machine learning-based region of interest detection
- Integration with peak detection algorithms

### 3. Performance Optimization
- Caching of focus detection results
- Background processing for large datasets
- GPU acceleration for intensive calculations

This implementation successfully resolves the critical bugs while maintaining all the previously implemented conditional auto-focus functionality. The application should now work reliably without the DataFrame comparison errors or parameter widget initialization issues.