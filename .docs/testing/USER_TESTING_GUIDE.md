# Quick Reference: Testing the New Features

> **For User**: Steps to validate the dataset selection highlighting and export functionality

## 🚀 Getting Started

### 1. Launch the Application
```powershell
cd "j:\Coding\研究\raman-app"
python main.py
```

### 2. Open a Project
- Click "Open Project" or open a recent project
- Ensure the project has multiple datasets loaded

### 3. Navigate to Preprocessing Page
- Click on the "前処理" (Preprocessing) tab

## ✨ New Features to Test

### Feature 1: Enhanced Dataset Selection Highlighting

**What Changed**: Selected datasets now have a much darker, more visible blue background

**How to Test**:
1. Look at the "入力データセット" (Input Datasets) section
2. Click on any dataset in the list
3. **Expected Result**: 
   - Selected item shows **dark blue background** (#1565c0)
   - Text turns **white** for better contrast
   - **Bold** font weight
   - 2px darker blue border

4. Try hovering over unselected items
5. **Expected Result**: Light gray background appears on hover

6. Try Ctrl+Click to select multiple datasets
7. **Expected Result**: All selected items show dark blue highlighting

**Screenshot Needed**: 
- Take a screenshot showing the dark blue selection

---

### Feature 2: Dataset Export Functionality

**What's New**: You can now export datasets to various file formats

**How to Test**:

#### Test 1: Export Button Visibility
1. Look in the "入力データセット" section
2. **Expected Result**: You should see TWO buttons:
   - "🔄 データセットを更新" (Refresh)
   - "📤 データセットをエクスポート" (Export)

#### Test 2: Export Without Selection
1. Click anywhere to deselect all datasets
2. Click the "📤 データセットをエクスポート" button
3. **Expected Result**: Warning notification:
   - "エクスポートするデータセットを選択してください"
   - (Please select a dataset to export)

#### Test 3: Export Dialog
1. Select ONE dataset from the list
2. Click the "📤 データセットをエクスポート" button
3. **Expected Result**: A dialog opens with:
   - Title: "データセットのエクスポート"
   - Format dropdown with 4 options:
     * CSV (カンマ区切り)
     * TXT (タブ区切り)
     * ASC (ASCII形式)
     * Pickle (バイナリ形式)
   - Location field with "参照..." (Browse) button
   - Filename field (pre-filled with dataset name)
   - "エクスポート" (Export) and "キャンセル" (Cancel) buttons

#### Test 4: CSV Export
1. In the export dialog, select "CSV" format
2. Click "参照..." to browse for save location
3. Choose a folder (e.g., Desktop)
4. Leave the default filename or change it
5. Click "エクスポート"
6. **Expected Result**:
   - Success notification appears
   - File is created in selected location
   - File can be opened in Excel/spreadsheet app

#### Test 5: Other Formats (Optional)
Repeat Test 4 with:
- TXT format (tab-separated)
- ASC format (ASCII)
- Pickle format (binary - for Python use)

#### Test 6: Cancel Button
1. Open export dialog
2. Click "キャンセル"
3. **Expected Result**: Dialog closes, no file created

**Screenshots Needed**:
- Export dialog showing all options
- Success notification
- Exported file location

---

## 🐛 What to Watch For

### Potential Issues to Report:

1. **Selection not visible enough?**
   - Is the dark blue (#1565c0) too dark or too light?
   - Is text readable on the blue background?

2. **Export button placement**
   - Is it easy to find?
   - Does it crowd the interface?

3. **Export dialog**
   - Is the layout clear?
   - Are all labels understandable in Japanese?
   - Does the browse button work properly?

4. **File creation**
   - Do exported files open correctly?
   - Is the data preserved accurately?
   - Are column headers included?

5. **Error handling**
   - If you try to save to a read-only location, is the error clear?
   - If you cancel, does it work cleanly?

---

## 📝 Feedback Template

Please provide feedback using this format:

### Dataset Selection Highlighting
**Visual Quality**: ⭐⭐⭐⭐⭐ (1-5 stars)
**Comments**: 
- Is the contrast good?
- Any suggestions for improvement?

### Export Functionality
**Ease of Use**: ⭐⭐⭐⭐⭐ (1-5 stars)
**Format Quality**: ⭐⭐⭐⭐⭐ (1-5 stars)
**Comments**:
- Was the dialog intuitive?
- Did exports work as expected?
- Any missing features?

### Overall
**Issues Found**: (None / Minor / Major)
**Would use this feature?**: Yes / No
**Additional Comments**:

---

## 📸 Screenshots to Capture

Please take screenshots of:
1. ✅ Selected dataset with dark blue highlighting
2. ✅ Export button in the UI
3. ✅ Export dialog with all options
4. ✅ Success notification after export
5. ✅ (Optional) Japanese UI showing translations

---

## 🔄 If Something Breaks

### Application Crashes
```powershell
# Check the terminal for error messages
# Note the exact steps that caused the crash
```

### Export Doesn't Work
- Check if you have write permissions to the selected folder
- Try a different folder (e.g., Desktop)
- Try a different format

### Styling Issues
- Take a screenshot
- Describe what looks wrong
- Suggest what you expected to see

---

## ✅ When Testing is Complete

1. Fill out the feedback template above
2. Gather all screenshots
3. Note any issues or suggestions
4. Share feedback with development team

---

**Thank you for testing! 🙏**

Your feedback helps improve the application for all users.
