# `main.py` Documentation

## Overview

This file is the main entry point for the Raman Spectral Classifier application. Its primary responsibilities include:

- **Application Initialization:** Sets up the `QApplication` instance.
- **Global Setup:**
    - Loads all bundled fonts from the `assets/fonts` directory using `QFontDatabase`.
    - Initializes the `LocalizationManager` for multi-language support.
    - Generates and applies the main application stylesheet (QSS) based on the selected language's font configuration.
- **Main Window Creation:** Instantiates and displays the `MainWindow`.
- **Event Loop:** Starts the Qt event loop via `app.exec()`.

---

## `MainWindow` Class

The `MainWindow` class serves as the top-level container for the application's user interface.

### Key Components

- **`QStackedWidget` (`central_stack`):**  
    The core navigation mechanism, holding the two primary views:
    - **HomePage:** The initial landing screen for creating or opening projects.
    - **WorkspacePage:** The main work area containing the tabbed interface for data processing, ML, etc.
- **Toast Component:**  
    A non-blocking notification widget parented to the `MainWindow` to display application-wide alerts (e.g., "Project Loaded", "Error").

### Workflow

1. On startup, the `MainWindow` is created and immediately displays the `HomePage`.
2. The `HomePage` emits signals (`projectOpened` or `newProjectCreated`) when the user interacts with it.
3. These signals are connected to the `open_project_workspace` slot in `MainWindow`.
4. The `open_project_workspace` slot calls the `PROJECT_MANAGER` to load all project data (including pickled DataFrames) into memory.
5. If loading is successful, it calls the `load_project` method on the `workspace_page` instance to update its child widgets.
6. Finally, it switches the `central_stack`'s current widget to the `workspace_page`, transitioning the user from the home screen to the main application environment.
