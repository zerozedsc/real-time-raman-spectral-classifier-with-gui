# Export Feature Testing Plan

> **Test Plan for Enhanced Export Functionality**  
> Created: October 3, 2025  
> Features: Metadata export, location validation, default location, multiple dataset export

## Overview

This document provides a comprehensive testing plan for the newly implemented export features in the preprocessing page.

## Features to Test

### 1. Metadata JSON Export
- **Feature**: Automatic export of preprocessing metadata alongside dataset
- **File format**: `{filename}_metadata.json`
- **Content**: Preprocessing pipeline, source datasets, dates, data shape, spectral info

### 2. Location Validation
- **Feature**: Warning dialog when user clicks OK without selecting location
- **Behavior**: Prevents export if no location selected
- **UI**: Modal warning dialog with clear message

### 3. Default Location Persistence
- **Feature**: Remember last used export location
- **Persistence**: Session-level (stored in class instance)
- **Behavior**: Pre-fills location field on next export

### 4. Multiple Dataset Export
- **Feature**: Export multiple selected datasets simultaneously
- **Behavior**: Each dataset exported with original name
- **Feedback**: Success count and failure summary

## Test Scenarios

### Test 1: Single Dataset Export with Metadata

**Objective**: Verify single dataset export with metadata JSON

**Prerequisites**:
- Load project with at least one preprocessed dataset
- Ensure dataset has preprocessing metadata

**Steps**:
1. Start application: `uv run main.py`
2. Navigate to Preprocessing page
3. Select a single preprocessed dataset
4. Click "Export" button
5. Select export format (CSV recommended)
6. Click "Browse..." and select export location
7. Ensure "Export metadata (JSON)" checkbox is checked
8. Enter filename or use default
9. Click OK

**Expected Results**:
- ✅ Two files created: `{filename}.csv` and `{filename}_metadata.json`
- ✅ CSV file contains spectral data
- ✅ Metadata JSON contains:
  - `export_info`: export date, dataset name, data shape
  - `preprocessing`: pipeline steps, source datasets, dates
  - `spectral_info`: num_spectra, axis range, spectral points
- ✅ Success notification displayed
- ✅ No errors in logs

### Test 2: Location Validation

**Objective**: Verify warning when no location selected

**Prerequisites**:
- Load project with at least one dataset

**Steps**:
1. Start application
2. Navigate to Preprocessing page
3. Select a dataset
4. Click "Export" button
5. Select export format
6. **Do NOT click Browse or select location**
7. Enter filename
8. Click OK

**Expected Results**:
- ✅ Warning dialog appears
- ✅ Dialog title: "Export Warning"
- ✅ Dialog message: "Please select a folder location before exporting..."
- ✅ Export dialog remains open
- ✅ No files created

### Test 3: Default Location Persistence

**Objective**: Verify last location is remembered

**Prerequisites**:
- Load project with datasets

**Steps**:
1. Start application
2. Navigate to Preprocessing page
3. Select dataset
4. Click "Export" button
5. Browse and select location (e.g., Desktop)
6. Complete export
7. Click "Export" button again
8. Check location field

**Expected Results**:
- ✅ Location field pre-filled with previous location
- ✅ Browse dialog starts from previous location
- ✅ Location persists across multiple exports in same session

### Test 4: Multiple Dataset Export

**Objective**: Verify batch export of multiple datasets

**Prerequisites**:
- Load project with 3+ datasets
- Enable multi-selection in dataset list

**Steps**:
1. Start application
2. Navigate to Preprocessing page
3. Select 3 datasets (Ctrl+Click or Shift+Click)
4. Click "Export" button
5. Verify dialog shows: "Exporting {count} datasets"
6. Note: filename field should be replaced with info text
7. Select format and location
8. Ensure metadata checkbox checked
9. Click OK

**Expected Results**:
- ✅ Dialog shows count of datasets to export
- ✅ Info text: "Multiple datasets will be exported using their original names"
- ✅ No filename field shown (uses original names)
- ✅ 3 data files created
- ✅ 3 metadata files created (if metadata enabled)
- ✅ Success notification: "Successfully exported {count} dataset(s) to {path}"
- ✅ No errors in logs

### Test 5: Metadata Export Toggle

**Objective**: Verify metadata export can be disabled

**Prerequisites**:
- Load project with preprocessed dataset

**Steps**:
1. Start application
2. Navigate to Preprocessing page
3. Select dataset
4. Click "Export" button
5. **Uncheck "Export metadata (JSON)" checkbox**
6. Select format and location
7. Complete export

**Expected Results**:
- ✅ Only data file created (e.g., `dataset.csv`)
- ✅ No metadata JSON file created
- ✅ Success notification still shown

### Test 6: Japanese Localization

**Objective**: Verify all new strings work in Japanese

**Prerequisites**:
- Set language to Japanese in config

**Steps**:
1. Edit `configs/app_configs.json`: Set `"language": "ja"`
2. Start application
3. Navigate to Preprocessing page
4. Test all export scenarios (1-5) in Japanese
5. Verify:
   - Button text: "エクスポート"
   - Dialog title: "データセットのエクスポート"
   - All labels in Japanese
   - Warning messages in Japanese
   - Success/error notifications in Japanese

**Expected Results**:
- ✅ All UI elements in Japanese
- ✅ No English text visible
- ✅ All functionality works identically

### Test 7: Error Handling

**Objective**: Verify robust error handling

**Test Cases**:

#### 7a. Invalid Location
1. Manually edit location field (if possible)
2. Enter non-existent path
3. Try to export

**Expected**: Error notification: "Selected export location does not exist"

#### 7b. Empty Filename (Single Export)
1. Select single dataset
2. Open export dialog
3. Clear filename field
4. Try to export

**Expected**: Warning: "Please provide a filename for the export"

#### 7c. Dataset Not Found
1. Select dataset
2. Delete from RAMAN_DATA (programmatically if testing)
3. Try to export

**Expected**: Error: "Dataset '{name}' not found"

#### 7d. Disk Full or Permission Error
1. Export to protected location or full disk
2. Verify graceful error handling

**Expected**: Error notification with exception message

### Test 8: Format Compatibility

**Objective**: Verify all export formats work

**Formats to Test**:
- CSV (Comma-separated values)
- TXT (Tab-separated values)
- ASC (ASCII format)
- Pickle (Python-specific format)

**Steps** (for each format):
1. Export dataset in format
2. Verify file created with correct extension
3. Verify file can be opened/read
4. Verify metadata JSON created (if enabled)

**Expected Results**:
- ✅ All formats export successfully
- ✅ Files have correct extensions (.csv, .txt, .asc, .pkl)
- ✅ Data integrity maintained
- ✅ Metadata JSON format consistent across all

## Test Data Requirements

### Minimum Test Data
- **Raw Dataset**: At least 1 unprocessed dataset
- **Preprocessed Dataset**: At least 1 dataset with preprocessing history
- **Multiple Datasets**: At least 3 datasets for batch testing

### Recommended Test Project Structure
```
test-project/
├── raw_data_1.csv
├── raw_data_2.csv
├── raw_data_3.csv
└── processed_data_1 (with preprocessing metadata)
```

## Validation Checklist

### Pre-Testing
- [ ] Application starts without errors
- [ ] Preprocessing page loads successfully
- [ ] Datasets visible in list
- [ ] Export button visible and enabled

### Post-Testing
- [ ] All test scenarios pass
- [ ] No errors in application logs
- [ ] Exported files readable and valid
- [ ] Metadata JSON well-formed
- [ ] UI responsive and no crashes
- [ ] Locale switching works

## Expected Metadata JSON Structure

```json
{
  "export_info": {
    "export_date": "2025-10-03T10:30:00.123456",
    "dataset_name": "processed_data",
    "data_shape": {
      "rows": 1200,
      "columns": 50
    }
  },
  "preprocessing": {
    "is_preprocessed": true,
    "processing_date": "2025-10-03T09:15:00.000000",
    "source_datasets": ["raw_data_1", "raw_data_2"],
    "pipeline": [
      {
        "category": "Range Operations",
        "method": "Cropper",
        "params": {"range": [600, 1800]},
        "enabled": true
      }
    ],
    "pipeline_summary": {
      "total_steps": 3,
      "successful_steps": 3,
      "failed_steps": 0,
      "success_rate": 100.0
    },
    "successful_steps": [...],
    "failed_steps": []
  },
  "spectral_info": {
    "num_spectra": 50,
    "spectral_axis_start": 600.0,
    "spectral_axis_end": 1800.0,
    "spectral_points": 1200
  }
}
```

## Known Limitations

1. **Location Persistence**: Only persists during application session (not saved to config file)
2. **Multiple Export Progress**: No progress bar (executes sequentially, may be slow for many datasets)
3. **Metadata for Raw Data**: Raw datasets may have minimal metadata (no preprocessing info)

## Debugging Tips

### Enable Debug Logging
```python
# Check logs/PreprocessPage.log for export-related messages
create_logs("PreprocessPage", "export_*", ...)
```

### Common Issues
- **Location not pre-filled**: Check `self._last_export_location` attribute
- **Metadata not exported**: Verify checkbox state and dataset has metadata
- **Multiple export fails**: Check individual dataset errors in logs

## Success Criteria

All tests must pass with:
- ✅ No application crashes
- ✅ No unhandled exceptions
- ✅ Correct file outputs
- ✅ Valid JSON metadata
- ✅ Proper error messages
- ✅ Both locales working

## Test Execution Log

| Test # | Description | Status | Date | Notes |
|--------|-------------|--------|------|-------|
| 1 | Single export with metadata | ⏳ Pending | - | - |
| 2 | Location validation | ⏳ Pending | - | - |
| 3 | Default location | ⏳ Pending | - | - |
| 4 | Multiple export | ⏳ Pending | - | - |
| 5 | Metadata toggle | ⏳ Pending | - | - |
| 6 | Japanese locale | ⏳ Pending | - | - |
| 7 | Error handling | ⏳ Pending | - | - |
| 8 | Format compatibility | ⏳ Pending | - | - |

---

**Test Execution Command**: `uv run main.py`  
**Log Location**: `logs/PreprocessPage.log`  
**Test Duration**: ~15-20 minutes for complete test suite
