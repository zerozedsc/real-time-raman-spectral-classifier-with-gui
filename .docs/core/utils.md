# `utils.py` Documentation

## Overview

This module serves as a central hub for application-wide utilities, global instances, and helper functions. It is intended to be imported by various parts of the application to access shared resources and logic.

**Recent Updates**: Module has been cleaned and optimized - removed unused functions (`validate_raman_data_integrity`, `get_raman_data_summary`) and unused imports to maintain clean architecture.

## Key Components

### Global Instances

These objects are instantiated once and shared across the entire application to maintain a consistent state.

- **`RAMAN_DATA`** (`Dict[str, pd.DataFrame]`):  
    A global, in-memory dictionary that acts as the primary data store for the currently active project.
    - **Keys:** User-defined string names for each dataset (e.g., `"Tumor Tissue"`, `"Healthy Control"`).
    - **Values:** The corresponding `pandas.DataFrame` containing the Raman spectra.
    - This dictionary is cleared and repopulated by the `ProjectManager` whenever a new project is loaded.

- **`PROJECT_MANAGER`** (`ProjectManager` instance):  
    The single source of truth for all project-related file I/O. Handles creating, loading, saving, and managing project configuration files and their associated data.

- **`LOCALIZEMANAGER`** (`LocalizationManager` instance):  
    Manages loading and retrieving translated strings from JSON files in the `assets/locales` directory.

- **`CONFIGS`** (`dict`):  
    A dictionary loaded from `configs/app_configs.json` containing application-level configurations, such as dynamic metadata fields.

## `ProjectManager` Class

This is the most critical class in this module. It abstracts all file system interactions related to projects.

**Responsibilities:**

- **Project Creation:**  
    Creates a new `.json` project file and a corresponding subdirectory within the `projects/` folder to hold associated data.

- **Project Loading:**  
    Reads a `.json` project file, finds the paths to all associated data packages (`.pkl` files), and loads them into the global `RAMAN_DATA` dictionary.

- **Data Persistence:**  
    Provides a method (`add_dataframe_to_project`) that:
    - Saves a given `pd.DataFrame` to a `.pkl` file within the project's dedicated data folder.
    - Updates the main `.json` project file with a reference to this new `.pkl` file, linking it to a user-defined dataset name.

- **Project Saving:**  
    Persists any changes to the `current_project_data` dictionary back to its `.json` file.

- **Recent Projects:**  
    Scans the `projects/` directory to provide a list of recently modified projects for the HomePage.

## Code Quality Notes

This module follows strict code quality standards:
- **No Debug Prints**: All debug statements have been removed in favor of proper logging
- **Clean Imports**: Only necessary imports are retained
- **Optimized Functions**: Unused utility functions have been removed to maintain lean architecture
- **Global State Management**: Centralized management of application-wide instances for consistent state

## Helper Functions

- **`LOCALIZE(key, **kwargs)`**:  
    A convenient shorthand function that calls `LOCALIZEMANAGER.get()`. This makes the UI code cleaner and easier to read.