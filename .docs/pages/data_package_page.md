# `pages/data_package_page.py` Documentation

This page is the main interface for managing all datasets within a project. It enables users to:

- Import multiple, uniquely named Raman datasets
- Assign comprehensive metadata
- Save datasets to the project before pre-processing

---

## 🆕 Recent Updates (2025)

### **Dataset Info Display Enhancement (October 21, 2025) ✨**

#### **Feature: At-a-Glance Dataset Information**
Enhanced `DatasetItemWidget` to display comprehensive dataset information directly in the list, eliminating the need to select datasets to see basic statistics.

**Information Displayed**:
- **Spectrum Count**: Number of spectra in dataset
- **Wavelength Range**: Min–Max in cm⁻¹
- **Data Points**: Number of measurement points per spectrum

**Visual Design**:
```
20211107_MM16_B                         [🗑️]
40 spectra | 379.7–3780.1 cm⁻¹ | 3000 pts
```

**Layout Structure**:
- **Line 1**: Dataset name (bold, 13px, normal color)
- **Line 2**: Dataset info (regular, 10px, gray #7f8c8d)
- **Format**: Compact single line with pipe separators (`|`)
- **Height Impact**: Minimal increase (~15px per item from ~30px → ~45px)

**Implementation Details**:
```python
class DatasetItemWidget(QWidget):
    def __init__(self, dataset_name: str, parent=None):
        # Vertical layout for name + info
        info_vbox = QVBoxLayout()
        info_vbox.setSpacing(2)
        
        # Dataset name (bold)
        name_label = QLabel(dataset_name)
        name_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        
        # Extract info from RAMAN_DATA
        df = RAMAN_DATA.get(dataset_name)
        if df is not None and not df.empty:
            num_spectra = df.shape[1]
            wavelength_min = df.index.min()
            wavelength_max = df.index.max()
            data_points = df.shape[0]
            
            info_text = f"{num_spectra} spectra | {wavelength_min:.1f}–{wavelength_max:.1f} cm⁻¹ | {data_points} pts"
            info_label = QLabel(info_text)
            info_label.setStyleSheet("font-size: 10px; color: #7f8c8d;")
```

**Data Source**:
- Info extracted live from `RAMAN_DATA` global dictionary (pandas DataFrames)
- **Spectrum count**: `df.shape[1]` (columns = individual spectra)
- **Wavelength range**: `df.index.min()` to `df.index.max()` (row indices = wavenumbers)
- **Data points**: `df.shape[0]` (rows = measurement points)

**Error Handling**:
- Gracefully handles missing data (shows name only without crash)
- Checks for `None` and empty DataFrames before extracting info

**Benefits**:
- ✅ **Instant Insights**: See dataset size without selecting items
- ✅ **Quick Identification**: Identify datasets by characteristics at a glance
- ✅ **Consistent Format**: Matches info display shown in data preview panel
- ✅ **Minimal Height Impact**: Small font keeps item height reasonable
- ✅ **No Selection Required**: Browse datasets without triggering preview reloads

**User Experience**:
- Users can quickly scan dataset list to find:
  - Datasets with specific spectrum counts (e.g., find datasets with >100 spectra)
  - Datasets covering specific wavelength ranges (e.g., find full-range datasets)
  - Datasets with matching data point counts (e.g., verify resolution consistency)
- Reduces clicks and preview loads when browsing project datasets

**Related Documentation**:
- `.AGI-BANKS/RECENT_CHANGES.md`: "October 21, 2025 - Preview Toggle & Dataset Info Enhancements"

---

## Architecture & Workflow

The page uses a **two-panel layout** for clear separation of concerns.

### Left Panel: Data Management

#### **Importer Group (`importer_group`)**

- **Dataset Name:** Auto-filled from the selected file/folder name (user-editable)
- **Data Path:** Selectable via drag-and-drop or file browser
- **Metadata Path:** Optional; file browser only allows `.json` files
- **Actions:**  
     - **Preview:** Preview the data  
     - **Add to Project:** Add dataset to the project

#### **Project Datasets Group (`loaded_group`)**

- **Component:** Scrollable `QListWidget` (`loaded_data_list`) using custom `DatasetItemWidget`
- **Features:**  
     - Handles many datasets without clutter  
     - Each item shows dataset name and a permanently visible Remove ("X") button for deletion

---

### Right Panel: Data Preview & Metadata

This panel is vertically split, with more space for data visualization.

#### **Data Preview (`preview_group`)**

- **Main Area:** Large, clear view of spectra using a `MatplotlibWidget` inside a themed frame
- **Styling:** Plot background and legend match the application's theme
- **Info Label:** Below the plot, summarizes data (spectra count, wavenumber range, etc.)

#### **Metadata Editor (`meta_editor_group`)**

- **Location:** Below the data preview
- **Component:** `QTabWidget` with tabs: Sample, Instrument, Measurement, Notes
- **Fields:** Editable when previewing new data; read-only for saved datasets
- **Save Metadata:** Button to save entered info to a new `.json` file

---

## Import & Management Workflow

1. **Input:** User selects a data source; Dataset Name auto-populates
2. **Preview:** User clicks Preview; right panel updates with plot and metadata; "Add to Project" is enabled
3. **Add to Project:** User clicks Add to Project; dataset is validated and saved as `.pkl`, project config updates, and dataset list refreshes
4. **Selection:** Clicking a dataset displays its plot and metadata
5. **Deletion:** Clicking "X" prompts for confirmation and, if confirmed, removes the dataset and its data file from the project
