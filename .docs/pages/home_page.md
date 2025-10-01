# `pages/home_page.py` Documentation

This module defines the **HomePage**, which serves as the modern, professional landing screen for the application. It provides users with clear, immediate actions to either start a new project or open an existing one, and it lists recently accessed projects for quick access.

---

## Architecture & Key Components

The HomePage has been redesigned for a clean, professional appearance that remains balanced on all screen sizes.

### Layout

- **Core Layout (`centering_layout`)**:  
    A `QHBoxLayout` uses stretchable spacers to center a single main content widget.
- **Content Container**:  
    The `content_container` widget holds all visible elements and has a `maximumWidth` set to prevent the UI from looking stretched and sparse on wide screens.
- **Content Organization**:  
    All content within the container is organized by a `QVBoxLayout`, creating a focused, single-column feel.

### Header Section

- **Hero Section**:  
    A prominent section at the top welcomes the user.
- **`QLabel` (`welcomeTitle`)**:  
    Displays the main application title, **"ラマン分光分類器"**.
- **`QLabel` (`welcomeSubtitle`)**:  
    Displays the descriptive subtitle related to your research.

### Action Cards

- **ActionCard (custom widget)**:  
    The primary user actions ("New Project" and "Open Project") are presented as large, clickable cards.
- Each ActionCard combines an icon, a title, and a descriptive text, making its purpose immediately clear.

### Recent Projects Section

- **`QGroupBox` (`recentProjectsGroup`)**:  
    A styled group box that contains the list of recent projects.
- **`QListWidget` (`recent_projects_list`)**:  
    Displays the list of projects.
- **`RecentProjectItemWidget`**:  
    Each entry in the list is a custom widget that neatly displays the project's name and its last modified date.

---

## Workflow & Signals

### Initialization

- When the application starts, the HomePage is the first screen shown.
- It immediately calls `populate_recent_projects()` to fetch and display the list of recent projects from the `ProjectManager`.

### User Actions

- **New Project**:  
    Clicking the "New Project" card triggers the `handle_new_project` method.
- **Open Project**:  
    Clicking the "Open Project" card triggers the `handle_open_project` method.
- **Open Recent**:  
    Double-clicking an item in the "Recent Projects" list triggers the `handle_recent_item_opened` method.

### Signal Emission

Upon a successful action, the HomePage emits one of two signals:

- `newProjectCreated(str)`: Emitted with the path to the newly created project file.
- `projectOpened(str)`: Emitted with the path to the selected project file.

These signals are captured by the `MainWindow` to transition from the home page to the main workspace environment.