import os
import json
import datetime
from typing import List, Dict, Any
import pandas as pd
from PySide6.QtGui import QFontDatabase, QIcon, QPixmap, QPainter 
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtCore import QSize, Qt

# This import assumes your configs.py is in a 'configs' directory
from configs.configs import *
from configs.style.stylesheets import get_main_stylesheet

# --- Global In-Memory Data Store ---
# This dictionary will hold the currently loaded DataFrames for the active project.
# The keys are the user-defined dataset names.
RAMAN_DATA: Dict[str, pd.DataFrame] = {}

class ProjectManager:
    """Handles all project-related file operations, including managing multiple datasets."""
    def __init__(self, projects_dir: str = "projects"):
        self.projects_dir = os.path.abspath(projects_dir)
        self.current_project_data: Dict[str, Any] = {}
        self._ensure_projects_dir_exists()

    def _ensure_projects_dir_exists(self):
        """Creates the base projects directory if it doesn't exist."""
        os.makedirs(self.projects_dir, exist_ok=True)

    def _get_project_data_dir(self) -> str | None:
        """Returns the path to the dedicated 'data' subdirectory for the current project."""
        if not self.current_project_data:
            create_logs("ProjectManager", "projects", "Cannot get data dir, no project loaded.", status='error')
            return None
        project_name = self.current_project_data.get("projectName", "").replace(' ', '_').lower()
        if not project_name:
            create_logs("ProjectManager", "projects", "Cannot get data dir, project has no name.", status='error')
            return None
        project_root_dir = os.path.join(self.projects_dir, project_name)
        data_dir = os.path.join(project_root_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

    def get_recent_projects(self) -> List[Dict[str, Any]]:
        """Scans the projects directory and returns a sorted list of projects by modification date."""
        projects = []
        project_folders = [d for d in os.listdir(self.projects_dir) if os.path.isdir(os.path.join(self.projects_dir, d))]

        for project_name in project_folders:
            project_path = os.path.join(self.projects_dir, project_name, f"{project_name}.json")
            if os.path.exists(project_path):
                try:
                    with open(project_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    last_modified_timestamp = os.path.getmtime(project_path)
                    projects.append({
                        "name": data.get("projectName", project_name),
                        "path": project_path,
                        "last_modified": datetime.datetime.fromtimestamp(last_modified_timestamp).strftime('%Y-%m-%d %H:%M'),
                        "timestamp": last_modified_timestamp
                    })
                except (json.JSONDecodeError, FileNotFoundError):
                    create_logs("ProjectManager", "projects", f"Could not read project file: {project_path}", status='warning')
        return sorted(projects, key=lambda p: p['timestamp'], reverse=True)

    def create_new_project(self, project_name: str) -> str | None:
        """Creates a new project JSON file and its associated data directory."""
        sanitized_name = project_name.replace(' ', '_').lower()
        project_root_dir = os.path.join(self.projects_dir, sanitized_name)
        project_path = os.path.join(project_root_dir, f"{sanitized_name}.json")

        if os.path.exists(project_path):
            return None

        project_data = {
            "projectName": project_name,
            "creationDate": datetime.datetime.now().isoformat(),
            "schemaVersion": "1.2", # Updated schema for nested metadata
            "dataPackages": {}
        }
        try:
            os.makedirs(os.path.join(project_root_dir, "data"), exist_ok=True)
            with open(project_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=4)
            return project_path
        except IOError as e:
            create_logs("ProjectManager", "projects", f"Failed to create project file: {e}", status='error')
            return None

    def add_dataframe_to_project(self, dataset_name: str, df: pd.DataFrame, metadata: dict) -> bool:
        """Saves a DataFrame as a pickle, and updates the project config."""
        data_dir = self._get_project_data_dir()
        if not data_dir:
            return False

        pickle_filename = f"{dataset_name.replace(' ', '_').lower()}.pkl"
        pickle_path = os.path.join(data_dir, pickle_filename)
        try:
            df.to_pickle(pickle_path)
        except IOError as e:
            create_logs("ProjectManager", "projects", f"Failed to save temporary data file: {e}", status='error')
            return False

        if "dataPackages" not in self.current_project_data:
            self.current_project_data["dataPackages"] = {}
        
        self.current_project_data["dataPackages"][dataset_name] = {
            "path": pickle_path,
            "metadata": metadata,
            "addedDate": datetime.datetime.now().isoformat()
        }

        self.save_current_project()
        RAMAN_DATA[dataset_name] = df
        return True

    def remove_dataframe_from_project(self, dataset_name: str) -> bool:
        """Removes a dataset from the project, including its pickle file."""
        if "dataPackages" not in self.current_project_data or dataset_name not in self.current_project_data["dataPackages"]:
            create_logs("ProjectManager", "projects", f"Dataset '{dataset_name}' not found in project config.", status='warning')
            return False

        package_info = self.current_project_data["dataPackages"].pop(dataset_name, None)
        if package_info:
            pickle_path = package_info.get("path")
            if pickle_path and os.path.exists(pickle_path):
                try:
                    os.remove(pickle_path)
                    create_logs("ProjectManager", "projects", f"Successfully deleted data file: {pickle_path}", status='info')
                except OSError as e:
                    create_logs("ProjectManager", "projects", f"Error deleting data file {pickle_path}: {e}", status='error')
                    # Don't stop, still try to save the project file
        
        # Also remove from the in-memory store
        RAMAN_DATA.pop(dataset_name, None)
        
        self.save_current_project()
        create_logs("ProjectManager", "projects", f"Successfully removed dataset '{dataset_name}' from project.", status='info')
        return True

    def load_project(self, project_path: str) -> bool:
        """Loads project JSON and all associated data packages from their pickle files."""
        global RAMAN_DATA
        try:
            with open(project_path, 'r', encoding='utf-8') as f:
                self.current_project_data = json.load(f)
            # Store the full path to the project file itself for saving later
            self.current_project_data['projectFilePath'] = project_path
            
            RAMAN_DATA.clear()
            data_packages = self.current_project_data.get("dataPackages", {})
            for name, package_info in data_packages.items():
                pickle_path = package_info.get("path")
                if pickle_path and os.path.exists(pickle_path):
                    try:
                        RAMAN_DATA[name] = pd.read_pickle(pickle_path)
                    except Exception as e:
                        create_logs("ProjectManager", "projects", f"Failed to load data package '{name}' from {pickle_path}: {e}", status='warning')
                else:
                    create_logs("ProjectManager", "projects", f"Data file for '{name}' not found at {pickle_path}", status='warning')

            create_logs("ProjectManager", "projects", f"Successfully loaded project: {self.current_project_data.get('projectName')}", status='info')
            return True
        except (FileNotFoundError, json.JSONDecodeError) as e:
            create_logs("ProjectManager", "projects", f"Failed to load project {project_path}: {e}", status='error')
            self.current_project_data = {}
            RAMAN_DATA.clear()
            return False

    def save_current_project(self):
        """Saves the current project data back to its JSON file."""
        project_path = self.current_project_data.get('projectFilePath')
        if not project_path:
            create_logs("ProjectManager", "projects", "Cannot save project: No project file path is set.", status='error')
            return

        try:
            # Create a copy to avoid saving the file path into the JSON itself
            data_to_save = self.current_project_data.copy()
            data_to_save.pop('projectFilePath', None)
            with open(project_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4)
        except IOError as e:
            create_logs("ProjectManager", "projects", f"Failed to save project {project_path}: {e}", status='error')

    def get_dataframe_metadata(self, dataset_name: str) -> Dict[str, Any] | None:
        """Retrieve metadata for a specific dataset."""
        if "dataPackages" not in self.current_project_data:
            return None
        
        package_info = self.current_project_data["dataPackages"].get(dataset_name)
        if package_info:
            return package_info.get("metadata", {})
        return None

    def update_dataframe_metadata(self, dataset_name: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for an existing dataset."""
        if "dataPackages" not in self.current_project_data:
            return False
        
        if dataset_name in self.current_project_data["dataPackages"]:
            self.current_project_data["dataPackages"][dataset_name]["metadata"] = metadata
            self.save_current_project()
            return True
        return False

    def get_all_dataframe_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all datasets in the project."""
        if "dataPackages" not in self.current_project_data:
            return {}
        
        metadata_dict = {}
        for name, package_info in self.current_project_data["dataPackages"].items():
            metadata_dict[name] = package_info.get("metadata", {})
        return metadata_dict


CONFIGS = load_config()
LOCALIZEMANAGER = LocalizationManager(default_lang="ja")
PROJECT_MANAGER = ProjectManager()

# --- Shorthand Function ---
def LOCALIZE(key, **kwargs):
    return LOCALIZEMANAGER.get(key, **kwargs)


ICON_PATHS = {
    "new_project": os.path.join(os.path.dirname(__file__), "assets/icons/new-project.svg"),
    "open_project": os.path.join(os.path.dirname(__file__), "assets/icons/load-project.svg"),
    "recent_projects": os.path.join(os.path.dirname(__file__), "assets/icons/recent-project.svg"),
}

# [100725] --- Utility Functions ---
def load_svg_icon(path: str,  color: Qt.GlobalColor = None, size: QSize = QSize(48, 48)) -> QIcon:
    """
    Loads an SVG file from disk and returns a QIcon of the given size.
    Optionally applies a color overlay to the SVG.
    """
    renderer = QSvgRenderer(path)
    pixmap = QPixmap(size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    if color is not None:
        # Apply color overlay
        mask = pixmap.createMaskFromColor(Qt.GlobalColor.transparent)
        colored_pixmap = QPixmap(size)
        colored_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(colored_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(colored_pixmap.rect(), color)
        painter.end()
        return QIcon(colored_pixmap)
    return QIcon(pixmap)

# [230725] --- RAMAN_DATA Integrity Check ---
# This function checks the integrity of the global RAMAN_DATA dictionary.
# It logs the total number of datasets and spectra, and checks for empty datasets.
def validate_raman_data_integrity():
    """Validate RAMAN_DATA integrity and log status."""
    try:
        total_datasets = len(RAMAN_DATA)
        total_spectra = sum(df.shape[1] if df.ndim > 1 else 1 for df in RAMAN_DATA.values())
        
        create_logs("RAMAN_DATA", "integrity_check", 
                   f"RAMAN_DATA contains {total_datasets} datasets with {total_spectra} total spectra", 
                   status='info')
        
        # Check for empty datasets
        empty_datasets = [name for name, df in RAMAN_DATA.items() if df.empty]
        if empty_datasets:
            create_logs("RAMAN_DATA", "empty_datasets", 
                       f"Found empty datasets: {empty_datasets}", status='warning')
        
        return True
    except Exception as e:
        create_logs("RAMAN_DATA", "integrity_error", 
                   f"Error validating RAMAN_DATA: {e}", status='error')
        return False

# [240725] --- RAMAN_DATA Summary Function ---
# This function provides a summary of the currently loaded RAMAN_DATA contents.
def get_raman_data_summary():
    """Get summary of current RAMAN_DATA contents."""
    if not RAMAN_DATA:
        return "No datasets loaded"
    
    summary = f"Loaded datasets: {len(RAMAN_DATA)}\n"
    for name, df in RAMAN_DATA.items():
        summary += f"  • {name}: {df.shape} ({df.index.min():.1f}-{df.index.max():.1f} cm⁻¹)\n"
    
    return summary