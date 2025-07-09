import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QStackedWidget, QLabel
)
from PySide6.QtCore import Qt

from components.app_tabs import AppTabBar
from pages.data_package_page import DataPackagePage
from utils import LOCALIZE

class WorkspacePage(QWidget):
    """
    The main workspace container that holds the AppTabBar and the
    stacked widget for different application pages.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workspacePage")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.tab_bar = AppTabBar()
        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("workspaceStack")

        self.data_page = DataPackagePage()
        self.preprocessing_page = QLabel(LOCALIZE("TABS.preprocessing") + " Page Content", alignment=Qt.AlignmentFlag.AlignCenter)
        self.ml_page = QLabel(LOCALIZE("TABS.machine_learning") + " Page Content", alignment=Qt.AlignmentFlag.AlignCenter)
        self.realtime_page = QLabel(LOCALIZE("TABS.real_time") + " Page Content", alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.page_stack.addWidget(self.data_page)
        self.page_stack.addWidget(self.preprocessing_page)
        self.page_stack.addWidget(self.ml_page)
        self.page_stack.addWidget(self.realtime_page)

        main_layout.addWidget(self.tab_bar)
        main_layout.addWidget(self.page_stack, 1)

        self.tab_bar.tabChanged.connect(self.page_stack.setCurrentIndex)

    def load_project(self, project_path: str):
        """
        Passes project data to the relevant pages after loading.
        """
        print(f"Workspace is loading project: {project_path}")
        
        # --- CRUCIAL: Trigger the data page to update its UI ---
        self.data_page.load_project_data()
        
        # Ensure the view is on the first tab when a project is loaded
        self.page_stack.setCurrentIndex(0)
        self.tab_bar.setActiveTab(0)
