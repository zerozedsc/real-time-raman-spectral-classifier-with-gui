# `pages/data_package_page.py` Documentation

This page is the main interface for managing all datasets within a project. It enables users to:

- Import multiple, uniquely named Raman datasets
- Assign comprehensive metadata
- Save datasets to the project before pre-processing

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
