# `pages/data_package_page.py` Documentation

This page is the main interface for managing all datasets within a project. It enables users to import multiple, uniquely named Raman datasets (e.g., "Tumor Spectra", "Control Group 1"), assign custom names, and save them to the project before moving to pre-processing.

---

## Architecture & Workflow

The page uses a two-panel layout:

### Left Panel: Data Management

- **Importer Group (`importer_group`):**
     - Import new datasets.
     - Fields:
          - **Dataset Name:** Auto-filled from the selected file/folder name (editable).
          - **Data Path:** Select via drag-and-drop or file browser.
          - **Metadata Path:** Optional.
     - Actions:
          - **Preview:** Load and preview the dataset.
          - **Add to Project:** Save the dataset to the project.

- **Project Datasets Group (`loaded_group`):**
     - **Component:** `QListWidget` (`loaded_data_list`)
     - Shows all dataset names currently loaded in the project (from `utils.RAMAN_DATA`).
     - Selecting a dataset updates the right panel with its details.

### Right Panel: Data Preview & Metadata

- **Components:**
     - **MatplotlibWidget (`plot_widget`):** Plots the selected or previewed dataset.
     - **QLabel (`info_label`):** Shows summary statistics (e.g., spectra count, wavenumber range).
     - **Metadata Editor (`meta_editor_group`):**
          - Editable when previewing a new dataset.
          - Read-only for datasets already in the project.
          - **Save Metadata:** Button to save metadata to a `.json` file.

---

## Import Workflow

1. **Input:** User selects a data source. Dataset Name auto-fills. Metadata file is optional.
2. **Preview:**
      - User clicks **Preview**.
      - `handle_preview_data` loads data into a temporary variable.
      - Right panel updates with plot, info, and metadata editor.
      - **Add to Project** button becomes enabled.
3. **Commit:**
      - User clicks **Add to Project**.
      - Validates unique dataset name.
      - Calls `PROJECT_MANAGER.add_dataframe_to_project()`.
      - Saves DataFrame as `.pkl` and updates project `.json` config.
      - Refreshes dataset list and clears importer fields.
