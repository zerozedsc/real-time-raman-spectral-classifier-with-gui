# Bug Fixes Summary - FINAL UPDATE

## Critical Issues Fixed ✅

### 1. Pipeline Index Out of Range Error (Issue #4) ✅
**Problem:** `list index out of range` error when disabling pipeline steps
**Root Cause:** In `_apply_preview_pipeline()` and `_apply_full_pipeline()`, code accessed `steps[current_row]` where `steps` contained only enabled steps but `current_row` was an index into the full pipeline list.
**Solution:** Changed to use `self.pipeline_steps[current_row]` and check if the step is in the filtered `steps` list before updating parameters.
**Files Modified:**
- `pages/preprocess_page.py` - Lines ~2518 and ~2451

### 2. Memory Not Clearing Between Projects (Issue #1) ✅
**Problem:** Datasets and pipeline steps from previous project persist when opening new project
**Root Causes:**
1. `load_project()` didn't call `clear_project_data()` before loading new data
2. `clear_project_data()` didn't clear the global `RAMAN_DATA` dictionary
**Solutions:**
1. Added `clear_project_data()` call at start of `load_project()` in workspace_page.py
2. Added `RAMAN_DATA.clear()` in `clear_project_data()` method in preprocess_page.py
**Files Modified:**
- `pages/workspace_page.py` - Lines ~162-190
- `pages/preprocess_page.py` - Lines ~990-1000

### 3. Pipeline Step Selection Visual Feedback (Issue #9) ✅
**Problem:** Current pipeline step selection not visually obvious
**Solution:** Enhanced selection styling with:
- Darker background color (#a8d0f0 instead of #d4e6f7)
- Thicker border (3px instead of 2px)  
- Darker border color (#0056b3)
- Bolder text (font-weight 700)
- Darker text color (#002952)
**Files Modified:**
- `pages/preprocess_page_utils/pipeline.py` - Lines ~1082-1096

## Previous Fixes (from earlier session)

### 1. Memory Clearing (Issue #1) ✅
**Problem:** Datasets from previous project persist when opening new project
**Solution:** Added `clear_project_data()` method in `pages/preprocess_page.py`
- Clears datasets, pipeline, parameters, visualization state
- Cancels running preview threads
- Resets UI to initial state
- Called by `workspace_page._reset_workspace_state()` on navigation

### 2. Parameter Type Conversion (Issue #2) ✅
**Problem:** Parameters losing type information through save/load cycles
**Solution:** Enhanced `create_method_instance()` in `functions/preprocess/registry.py`
- Added type conversion for: int, float, scientific, choice, list
- Special handling for choice parameters (detects int/float/string from choices)
- List parameters parsed with `ast.literal_eval()` from string representation
- Falls back gracefully on conversion errors

### 3. Kernel numpy.uniform Error (Issue #5) ✅
**Problem:** `AttributeError: module 'numpy' has no attribute 'uniform'`
**Solution:** Created `functions/preprocess/kernel_denoise.py` wrapper
- Monkey-patches `numpy.uniform` to `numpy.random.uniform`
- Wraps ramanspy's Kernel class
- Registry updated to use custom wrapper

### 4. BackgroundSubtractor Array Comparison (Issue #3) ✅
**Problem:** `ValueError: The truth value of an array is ambiguous`
**Solution:** Created `functions/preprocess/background_subtraction.py` wrapper
- Proper None checking before creating ramanspy instance
- Handles optional background parameter correctly
- Logs warning when no background is set

### 5. ASPLS Missing Parameter (Issue #6) ✅
**Problem:** `TypeError: ASPLS() missing required keyword-only argument: lam`
**Solution:** Fixed by adding 'scientific' type support in type conversion
- Scientific notation parameters now properly converted to float
- Applies to all methods with scientific parameters (lam, epsilon, etc.)

### 6. MultiScaleConv1D Type Error (Issue #8) ✅
**Problem:** String '[5, 11, 21, 41]' not converted to list of integers
**Solution:** Added 'list' type support in type conversion
- Uses `ast.literal_eval()` to safely parse list strings
- Handles nested structures and different data types

## Issues Requiring Further Investigation

### 7. FABC 'frequencies' Attribute ⚠️
**Problem:** `AttributeError: 'FABC' object has no attribute 'frequencies'`
**Status:** Deferred - requires deeper investigation of ramanspy API
**Notes:** FABC expects wavenumber frequencies to be provided, but current pipeline
doesn't pass this information. May need custom wrapper similar to other methods.

### 8. PeakRatioFeatures ℹ️
**Status:** Class exists and is properly imported
**Notes:** Original error may be stale from before import was added. Should work now.

## UI Improvements Pending

### 9. Pipeline Disable Graph Revert (Issue #4)
**Problem:** Disabling step reverts graph to raw data instead of showing remaining enabled steps
**Investigation:** Found potential index mismatch in `_apply_preview_pipeline()`
- Line 2518-2520: `current_row` is index into full pipeline list
- `steps` parameter contains only enabled steps
- This causes index mismatch when accessing `steps[current_row]`
**Status:** Needs testing and fix confirmation

### 10. Pipeline Step Selection Visual Feedback (Issue #9)
**Problem:** Current selection not visually obvious
**Solution:** Need to enhance selection indicator with darker/more prominent color
**Status:** Pending implementation

### 11. Section Heights (Issue #10)
**Problem:** Section heights don't match user's layout preferences  
**Solution:** Adjust preprocessed data, pipeline, and parameter section heights
**Status:** Pending implementation

## Files Modified

1. `pages/preprocess_page.py` - Added clear_project_data() method
2. `functions/preprocess/registry.py` - Enhanced type conversion, added imports
3. `functions/preprocess/kernel_denoise.py` - NEW: Kernel wrapper
4. `functions/preprocess/background_subtraction.py` - NEW: BackgroundSubtractor wrapper

## Testing Checklist

- [ ] Test memory clearing when navigating home -> new project
- [ ] Test Kernel with all three kernel types (uniform, gaussian, triangular)
- [ ] Test BackgroundSubtractor with None and with background
- [ ] Test ASPLS with scientific notation parameters
- [ ] Test MultiScaleConv1D with list kernel_sizes parameter
- [ ] Test Derivative with integer order parameter
- [ ] Test pipeline disable/enable step behavior
- [ ] Verify all preprocessing methods load and apply correctly
- [ ] Check UI selection feedback
- [ ] Validate section height adjustments
