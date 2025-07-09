import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QListWidget, QListWidgetItem, QFileDialog, QInputDialog, QMessageBox
)
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QIcon, QPixmap, QPainter
from PySide6.QtSvg import QSvgRenderer

# Import the global project manager instance
from utils import LOCALIZE, PROJECT_MANAGER

# --- (SVG and Icon functions remain the same) ---
SVG_NEW_FILE = """<svg ...>""" # Keep SVG data
SVG_OPEN_FOLDER = """<svg ...>""" # Keep SVG data
def create_icon_from_svg(svg_data, color):
    # ... function implementation ...
    svg_data_colored = svg_data.replace('stroke="currentColor"', f'stroke="{color}"')
    renderer = QSvgRenderer(svg_data_colored.encode('utf-8'))
    pixmap = QPixmap(renderer.defaultSize())
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


class RecentProjectItemWidget(QWidget):
    """A custom widget to display a single recent project in the list."""
    def __init__(self, project_name, last_modified, parent=None):
        super().__init__(parent)
        self.setObjectName("recentProjectItem")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.name_label = QLabel(project_name)
        self.name_label.setObjectName("projectNameLabel")
        self.path_label = QLabel(f"Last Modified: {last_modified}")
        self.path_label.setObjectName("projectPathLabel")
        layout.addWidget(self.name_label)
        layout.addWidget(self.path_label)


class HomePage(QWidget):
    """The main landing page for the application."""
    newProjectCreated = Signal(str)
    projectOpened = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("homePage")
        self._setup_ui()
        self.populate_recent_projects() # Renamed for clarity

    def _setup_ui(self):
        # --- (This method remains largely the same) ---
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self._create_sidebar(main_layout)
        self._create_main_area(main_layout)

    def _create_sidebar(self, parent_layout):
        # --- (This method remains the same) ---
        sidebar_frame = QFrame()
        sidebar_frame.setObjectName("homeSidebar")
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(20, 30, 20, 30)
        sidebar_layout.setSpacing(15)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        title_label = QLabel(LOCALIZE("MAIN_WINDOW.title"))
        title_label.setObjectName("sidebarTitle")
        subtitle_label = QLabel(LOCALIZE("HOME_PAGE.subtitle"))
        subtitle_label.setObjectName("sidebarSubtitle")
        subtitle_label.setWordWrap(True)
        new_icon = create_icon_from_svg(SVG_NEW_FILE, "#2c3e50")
        open_icon = create_icon_from_svg(SVG_OPEN_FOLDER, "#2c3e50")
        self.new_project_button = QPushButton(LOCALIZE("HOME_PAGE.new_project_button"))
        self.new_project_button.setIcon(new_icon)
        self.new_project_button.setIconSize(QSize(24, 24))
        self.new_project_button.setObjectName("sidebarButton")
        self.open_project_button = QPushButton(LOCALIZE("HOME_PAGE.open_project_button"))
        self.open_project_button.setIcon(open_icon)
        self.open_project_button.setIconSize(QSize(24, 24))
        self.open_project_button.setObjectName("sidebarButton")
        sidebar_layout.addWidget(title_label)
        sidebar_layout.addWidget(subtitle_label)
        sidebar_layout.addSpacing(35)
        sidebar_layout.addWidget(self.new_project_button)
        sidebar_layout.addWidget(self.open_project_button)
        parent_layout.addWidget(sidebar_frame)
        self.new_project_button.clicked.connect(self.handle_new_project)
        self.open_project_button.clicked.connect(self.handle_open_project)

    def _create_main_area(self, parent_layout):
        # --- (This method remains the same) ---
        main_area_widget = QWidget()
        main_area_widget.setObjectName("homeMainArea")
        main_layout = QVBoxLayout(main_area_widget)
        main_layout.setContentsMargins(50, 40, 50, 40)
        main_layout.setSpacing(20)
        header_label = QLabel(LOCALIZE("HOME_PAGE.recent_projects_title"))
        header_label.setObjectName("mainAreaTitle")
        self.recent_projects_list = QListWidget()
        self.recent_projects_list.setObjectName("recentProjectsList")
        self.recent_projects_list.itemDoubleClicked.connect(self.handle_recent_item_opened)
        main_layout.addWidget(header_label)
        main_layout.addWidget(self.recent_projects_list)
        parent_layout.addWidget(main_area_widget, 1)

    def populate_recent_projects(self):
        """Populates the list using the ProjectManager."""
        self.recent_projects_list.clear()
        
        recent_projects = PROJECT_MANAGER.get_recent_projects()
        
        if not recent_projects:
            item = QListWidgetItem(LOCALIZE("HOME_PAGE.no_recent_projects"))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.recent_projects_list.addItem(item)
        else:
            for project in recent_projects:
                list_item = QListWidgetItem(self.recent_projects_list)
                list_item.setData(Qt.ItemDataRole.UserRole, project["path"])
                
                item_widget = RecentProjectItemWidget(project["name"], project["last_modified"])
                list_item.setSizeHint(item_widget.sizeHint())
                
                self.recent_projects_list.addItem(list_item)
                self.recent_projects_list.setItemWidget(list_item, item_widget)
    
    def handle_new_project(self):
        """Handles creating a new project using the ProjectManager."""
        project_name, ok = QInputDialog.getText(self, LOCALIZE("HOME_PAGE.new_project_dialog_title"), LOCALIZE("HOME_PAGE.new_project_dialog_label"))
        
        if ok and project_name:
            project_path = PROJECT_MANAGER.create_new_project(project_name)
            
            if project_path:
                self.populate_recent_projects()
                self.newProjectCreated.emit(project_path)
            else:
                QMessageBox.warning(self, LOCALIZE("COMMON.error"), LOCALIZE("HOME_PAGE.project_exists_error", name=project_name))

    def handle_open_project(self):
        """Handles opening an existing project file."""
        projects_dir = PROJECT_MANAGER.projects_dir
        file_path, _ = QFileDialog.getOpenFileName(self, LOCALIZE("HOME_PAGE.open_project_dialog_title"), projects_dir, "Project Files (*.json)")
        if file_path:
            self.projectOpened.emit(file_path)
            
    def handle_recent_item_opened(self, item):
        """Handles opening a project from the recent list."""
        project_path = item.data(Qt.ItemDataRole.UserRole)
        if project_path:
            self.projectOpened.emit(project_path)
