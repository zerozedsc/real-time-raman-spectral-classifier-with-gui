# Test Plan: Dataset Selection Highlighting & Export Functionality

> **Test Date**: October 1, 2025  
> **Features**: Dataset selection visual enhancement, Export functionality  
> **Location**: `pages/preprocess_page.py`

## 🎯 Testing Objectives

1. Verify dataset selection highlighting works correctly
2. Validate export functionality for all supported formats
3. Ensure multi-language support is working
4. Test error handling and edge cases

## 📋 Test Cases

### Test Group 1: Dataset Selection Highlighting

#### TC1.1: Single Dataset Selection
- **Steps**:
  1. Launch application and open a project with multiple datasets
  2. Navigate to Preprocessing page
  3. Click on a single dataset in the input list
- **Expected**: 
  - Selected dataset shows dark blue background (#1565c0)
  - White text on selected item
  - Clear visual distinction from unselected items
- **Status**: ⏳ Pending

#### TC1.2: Multiple Dataset Selection
- **Steps**:
  1. Select first dataset
  2. Hold Ctrl and click on second dataset
  3. Verify both are highlighted
- **Expected**: 
  - Both datasets show dark blue highlighting
  - Selection persists when clicking elsewhere
- **Status**: ⏳ Pending

#### TC1.3: Hover States
- **Steps**:
  1. Hover mouse over unselected datasets
  2. Hover over selected dataset
- **Expected**:
  - Unselected: Light gray background (#f5f5f5)
  - Selected + hover: Slightly lighter blue (#1976d2)
- **Status**: ⏳ Pending

### Test Group 2: Export Functionality

#### TC2.1: Export Button Availability
- **Steps**:
  1. Check export button exists next to refresh button
  2. Verify tooltip shows correct text
- **Expected**: 
  - Button labeled "📤 Export Dataset" (EN) or "📤 データセットをエクスポート" (JA)
  - Tooltip: "Export selected dataset to file"
- **Status**: ⏳ Pending

#### TC2.2: Export Without Selection
- **Steps**:
  1. Ensure no dataset is selected
  2. Click export button
- **Expected**: 
  - Warning notification: "Please select a dataset to export"
  - No dialog opens
- **Status**: ⏳ Pending

#### TC2.3: Export Dialog - CSV Format
- **Steps**:
  1. Select a dataset
  2. Click export button
  3. Select CSV format
  4. Browse to save location
  5. Confirm export
- **Expected**: 
  - Dialog opens with format dropdown
  - CSV format creates comma-separated file
  - Success notification shows
  - File exists at specified location
- **Status**: ⏳ Pending

#### TC2.4: Export Dialog - TXT Format
- **Steps**:
  1. Select dataset
  2. Export as TXT (tab-separated)
  3. Verify file contents
- **Expected**: 
  - Tab-separated values
  - Wavenumber as first column
  - Spectra as subsequent columns
- **Status**: ⏳ Pending

#### TC2.5: Export Dialog - ASC Format
- **Steps**:
  1. Select dataset
  2. Export as ASC format
  3. Verify file contents
- **Expected**: 
  - ASCII-compatible format
  - Similar structure to TXT
- **Status**: ⏳ Pending

#### TC2.6: Export Dialog - Pickle Format
- **Steps**:
  1. Select dataset
  2. Export as Pickle (.pkl)
  3. Attempt to reload in Python
- **Expected**: 
  - Binary pickle file created
  - Can be loaded with pandas.read_pickle()
  - Preserves DataFrame structure
- **Status**: ⏳ Pending

### Test Group 3: Multi-Language Support

#### TC3.1: English Locale
- **Steps**:
  1. Set application language to English
  2. Check all export-related UI elements
- **Expected**: 
  - Export button: "Export Dataset"
  - Dialog title: "Export Dataset"
  - All labels in English
- **Status**: ⏳ Pending

#### TC3.2: Japanese Locale  
- **Steps**:
  1. Set application language to Japanese
  2. Check all export-related UI elements
- **Expected**: 
  - Export button: "データセットをエクスポート"
  - Dialog title: "データセットのエクスポート"
  - All labels in Japanese
- **Status**: ⏳ Pending

### Test Group 4: Error Handling

#### TC4.1: Invalid Export Path
- **Steps**:
  1. Try to export to invalid/non-existent path
  2. Verify error handling
- **Expected**: 
  - Error notification shows
  - User can retry with valid path
- **Status**: ⏳ Pending

#### TC4.2: File Write Permission Error
- **Steps**:
  1. Try to export to read-only location
  2. Verify error message
- **Expected**: 
  - Clear error message about permissions
  - No application crash
- **Status**: ⏳ Pending

#### TC4.3: Cancel Export Dialog
- **Steps**:
  1. Open export dialog
  2. Click Cancel button
- **Expected**: 
  - Dialog closes
  - No file is created
  - No error messages
- **Status**: ⏳ Pending

## 🔍 Manual Validation Script

### Preparation
```python
# Run this in terminal after starting the app
import time
print("=== STARTING DATASET SELECTION & EXPORT TEST ===")
print("Test start time:", time.strftime("%H:%M:%S"))
print("Waiting 45 seconds for manual testing...")
time.sleep(45)
print("Test end time:", time.strftime("%H:%M:%S"))
print("=== TEST COMPLETE - Please document results ===")
```

### During 45-Second Window
1. **0-15s**: Test dataset selection highlighting
   - Single selection
   - Multiple selection
   - Hover states
   
2. **15-30s**: Test export without selection
   - Click export with no selection
   - Verify warning message

3. **30-45s**: Test basic export
   - Select dataset
   - Open export dialog
   - Try CSV export
   - Verify success

## 📊 Results Template

```markdown
### Test Execution Results

**Date**: [Date]
**Tester**: [Name]
**Environment**: Windows/macOS/Linux

#### Dataset Selection Highlighting
- TC1.1: ✅ PASS / ❌ FAIL - [Notes]
- TC1.2: ✅ PASS / ❌ FAIL - [Notes]
- TC1.3: ✅ PASS / ❌ FAIL - [Notes]

#### Export Functionality
- TC2.1: ✅ PASS / ❌ FAIL - [Notes]
- TC2.2: ✅ PASS / ❌ FAIL - [Notes]
- TC2.3: ✅ PASS / ❌ FAIL - [Notes]
- TC2.4: ✅ PASS / ❌ FAIL - [Notes]
- TC2.5: ✅ PASS / ❌ FAIL - [Notes]
- TC2.6: ✅ PASS / ❌ FAIL - [Notes]

#### Multi-Language
- TC3.1: ✅ PASS / ❌ FAIL - [Notes]
- TC3.2: ✅ PASS / ❌ FAIL - [Notes]

#### Error Handling
- TC4.1: ✅ PASS / ❌ FAIL - [Notes]
- TC4.2: ✅ PASS / ❌ FAIL - [Notes]
- TC4.3: ✅ PASS / ❌ FAIL - [Notes]

#### Issues Found
1. [Description of issue]
2. [Description of issue]

#### Screenshots
- [Attach relevant screenshots]

#### Terminal Output
```
[Paste any relevant console output]
```
```

## 🐛 Known Issues to Watch For

1. **Selection persistence** - Does selection persist when switching between datasets?
2. **Export dialog positioning** - Does dialog appear centered on parent window?
3. **File overwrite handling** - What happens if file already exists?
4. **Large dataset export** - Performance with very large datasets?
5. **Special characters in filename** - Handling of special characters?

## 📝 Additional Observations

### Visual Quality Checklist
- [ ] Selection color contrast is sufficient
- [ ] Text remains readable when selected
- [ ] Hover states are smooth and responsive
- [ ] Export dialog is properly styled
- [ ] All buttons have appropriate icons

### Functional Quality Checklist
- [ ] All format exports create valid files
- [ ] Exported data can be re-imported
- [ ] Error messages are clear and helpful
- [ ] Dialog can be cancelled at any time
- [ ] No memory leaks during multiple exports

## 🎬 Next Steps After Testing

1. Document all test results in `RESULTS.md`
2. Create screenshots for visual verification
3. Fix any bugs found during testing
4. Update `.docs/TODOS.md` with findings
5. Update `.AGI-BANKS/RECENT_CHANGES.md`
6. Clean up any test files created

---

**Test Plan Version**: 1.0  
**Last Updated**: October 1, 2025
