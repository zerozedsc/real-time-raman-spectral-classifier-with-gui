import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QStackedWidget, QLabel
)
from PySide6.QtCore import Qt

from components.app_tabs import AppTabBar
from pages.data_package_page import DataPackagePage
from pages.preprocess_page import PreprocessPage
from pages.home_page import HomePage
from utils import *
from configs.configs import *

class WorkspacePage(QWidget):
    """
    The main workspace container that holds the AppTabBar and the
    stacked widget for different application pages.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workspacePage")
        self._setup_ui()
        self._connect_signals()
        
        # Initially show home page with hidden tab bar
        self.show_home_page()

    def _setup_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.tab_bar = AppTabBar()
        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("workspaceStack")

        # --- Instantiate all pages ---
        self.home_page = HomePage()
        self.data_page = DataPackagePage()
        self.preprocessing_page = PreprocessPage()
        self.ml_page = QLabel(LOCALIZE("TABS.machine_learning") + " Page Content", alignment=Qt.AlignmentFlag.AlignCenter)
        self.realtime_page = QLabel(LOCALIZE("TABS.real_time") + " Page Content", alignment=Qt.AlignmentFlag.AlignCenter)
        
        # --- Add pages to the stack ---
        self.page_stack.addWidget(self.home_page)      # Index 0 - Home
        self.page_stack.addWidget(self.data_page)      # Index 1 - Data
        self.page_stack.addWidget(self.preprocessing_page)  # Index 2 - Preprocessing
        self.page_stack.addWidget(self.ml_page)        # Index 3 - ML
        self.page_stack.addWidget(self.realtime_page)  # Index 4 - Real-time

        main_layout.addWidget(self.tab_bar)
        main_layout.addWidget(self.page_stack, 1)

    def _connect_signals(self):
        """Connect all signals for navigation and project handling."""
        # Tab navigation
        self.tab_bar.tabChanged.connect(self._on_tab_changed)
        self.tab_bar.homeRequested.connect(self.show_home_page)
        
        # Home page project signals
        self.home_page.newProjectCreated.connect(self.load_project)
        self.home_page.projectOpened.connect(self.load_project)

    def _on_tab_changed(self, index):
        """Handle tab changes and trigger data loading for all pages."""
        # Map tab index to page index (tabs start from index 0 but skip home page)
        page_index = index + 1  # Offset by 1 since home is not in tabs
        
        # Switch to the selected page
        self.page_stack.setCurrentIndex(page_index)
        
        # Show tab bar when not on home page
        self.tab_bar.setVisible(True)
        
        # Automatically refresh data for the current page
        self._refresh_current_page_data()
    
    def show_home_page(self):
        """Show the home page, hide tab bar, and reset workspace state."""
        # Clear all project data and reset workspace state
        self._reset_workspace_state()
        
        # Switch to home page
        self.page_stack.setCurrentIndex(0)  # Home page is at index 0
        self.tab_bar.setVisible(False)      # Hide tab bar on home page
        self.tab_bar.clearActiveTab()       # Clear any active tab highlighting
        
        # Refresh home page data (recent projects, etc.)
        self._refresh_home_page()
        
        # Re-establish home page connections (in case they were broken)
        self._reconnect_home_signals()
        
        create_logs("WorkspacePage", "show_home", "Returned to home page and reset workspace state", status='info')
    
    def _reset_workspace_state(self):
        """Reset workspace state and clear memory when returning to home."""
        try:
            # Clear project data from all pages
            for i in range(1, self.page_stack.count()):  # Skip home page (index 0)
                widget = self.page_stack.widget(i)
                if hasattr(widget, 'clear_project_data'):
                    widget.clear_project_data()
                elif hasattr(widget, 'reset'):
                    widget.reset()
            
            # Reset global project state if available
            if hasattr(PROJECT_MANAGER, 'clear_current_project'):
                PROJECT_MANAGER.clear_current_project()
            
            create_logs("WorkspacePage", "reset_state", "Successfully reset workspace state", status='info')
            
        except Exception as e:
            create_logs("WorkspacePage", "reset_state_error", 
                       f"Error resetting workspace state: {e}", status='warning')

    def _refresh_home_page(self):
        """Refresh home page data (recent projects list)."""
        try:
            if hasattr(self.home_page, 'populate_recent_projects'):
                self.home_page.populate_recent_projects()
            create_logs("WorkspacePage", "refresh_home", "Refreshed home page data", status='info')
        except Exception as e:
            create_logs("WorkspacePage", "refresh_home_error", 
                       f"Error refreshing home page: {e}", status='warning')

    def _reconnect_home_signals(self):
        """Re-establish home page signal connections to ensure they work properly."""
        try:
            # Disconnect existing connections to avoid duplicates
            self.home_page.newProjectCreated.disconnect()
            self.home_page.projectOpened.disconnect()
        except:
            pass  # Ignore if no connections exist
        
        # Reconnect signals
        self.home_page.newProjectCreated.connect(self.load_project)
        self.home_page.projectOpened.connect(self.load_project)
        
        create_logs("WorkspacePage", "reconnect_signals", "Re-established home page signal connections", status='info')
    
    def _refresh_current_page_data(self):
        """Refresh data for the currently active page."""
        current_index = self.page_stack.currentIndex()
        current_widget = self.page_stack.widget(current_index)
        
        # Skip refreshing home page
        if current_index == 0:
            return
        
        # Call load_project_data if the page has this method
        if hasattr(current_widget, 'load_project_data'):
            try:
                current_widget.load_project_data()
                create_logs("WorkspacePage", "auto_refresh", 
                           f"Auto-refreshed data for page index {current_index}", status='info')
            except Exception as e:
                create_logs("WorkspacePage", "auto_refresh_error", 
                           f"Error auto-refreshing page {current_index}: {e}", status='error')

    def load_project(self, project_path: str):
        """Load project and refresh all pages."""
        try:
            # Update project manager state if available
            if hasattr(PROJECT_MANAGER, 'set_current_project'):
                PROJECT_MANAGER.set_current_project(project_path)
            
            # Refresh all pages that have load_project_data method (skip home page)
            for i in range(1, self.page_stack.count()):  # Start from 1 to skip home
                widget = self.page_stack.widget(i)
                if hasattr(widget, 'load_project_data'):
                    try:
                        widget.load_project_data()
                    except Exception as e:
                        create_logs("WorkspacePage", "page_refresh_error", 
                                   f"Error refreshing page {i}: {e}", status='warning')
            
            # Show the data page (index 1) and make tab bar visible when project is loaded
            self.page_stack.setCurrentIndex(1)  # Data page
            self.tab_bar.setVisible(True)
            self.tab_bar.setActiveTab(0)  # Set first tab (Data) as active
            
            create_logs("WorkspacePage", "load_project", f"Successfully loaded project: {project_path}", status='info')
            
        except Exception as e:
            create_logs("WorkspacePage", "load_project_error", 
                       f"Error loading project {project_path}: {e}", status='error')

