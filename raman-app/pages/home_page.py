from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QListWidget, QListWidgetItem, QFileDialog, QInputDialog, QMessageBox,
    QSizePolicy, QGroupBox, QScrollArea
)
from PySide6.QtCore import Qt, QSize, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QIcon, QPixmap, QPainter, QMouseEvent, QFont
from PySide6.QtSvg import QSvgRenderer

from utils import *

class ActionCard(QWidget):
    """A modern, animated clickable card widget for primary actions."""
    clicked = Signal()

    def __init__(self, icon: QIcon, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("actionCard")
        self.setFixedSize(280, 160)
        
        self._setup_ui(icon, title, text)

    def _setup_ui(self, icon: QIcon, title: str, text: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        icon_container = QWidget()
        icon_container.setFixedHeight(48)
        icon_layout = QHBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        
        icon_label = QLabel()
        icon_label.setPixmap(icon.pixmap(QSize(32, 32)))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        icon_layout.addWidget(icon_label)
        icon_layout.addStretch()

        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        title_label.setWordWrap(True)
        
        desc_label = QLabel(text)
        desc_label.setObjectName("cardDescription")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addWidget(icon_container)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addStretch()

    def enterEvent(self, event):
        self.setProperty("hover", True)
        self.style().unpolish(self)
        self.style().polish(self)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setProperty("hover", False)
        self.style().unpolish(self)
        self.style().polish(self)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)

class RecentProjectItemWidget(QWidget):
    """Enhanced widget for displaying recent projects with better visual hierarchy."""
    def __init__(self, project_name: str, last_modified: str, parent=None):
        super().__init__(parent)
        self.setObjectName("recentProjectItem")
        self.setFixedHeight(70)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        
        icon_label = QLabel()
        project_icon = load_svg_icon(ICON_PATHS["recent_projects"], "#0078d4", QSize(24, 24))
        icon_label.setPixmap(project_icon.pixmap(QSize(24, 24)))
        icon_label.setFixedSize(24, 24)
        
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(4)
        
        name_label = QLabel(project_name)
        name_label.setObjectName("projectName")
        
        time_label = QLabel(LOCALIZE("HOME_PAGE.last_modified_label", date=last_modified))
        time_label.setObjectName("projectTime")
        
        info_layout.addWidget(name_label)
        info_layout.addWidget(time_label)
        
        layout.addWidget(icon_label)
        layout.addLayout(info_layout)
        layout.addStretch()

class HomePage(QWidget):
    """Modern, scientific-themed landing page with enhanced visual design."""
    newProjectCreated = Signal(str)
    projectOpened = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("homePage")
        self._setup_ui()
        self.populate_recent_projects()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        sidebar = self._create_sidebar()
        content_area = self._create_content_area()
        main_layout.addWidget(sidebar, 0)
        main_layout.addWidget(content_area, 1)

    def _create_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("homeSidebar")
        sidebar.setFixedWidth(320)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(32, 40, 32, 40)
        layout.setSpacing(32)
        
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)
        
        title_label = QLabel(LOCALIZE("MAIN_WINDOW.title"))
        title_label.setObjectName("homeTitle")
        
        subtitle_label = QLabel(LOCALIZE("HOME_PAGE.subtitle"))
        subtitle_label.setObjectName("homeSubtitle")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        
        actions_container = QWidget()
        actions_layout = QVBoxLayout(actions_container)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(16)
        
        new_icon = load_svg_icon(ICON_PATHS["new_project"], "#0078d4", QSize(32, 32))
        new_card = ActionCard(new_icon, LOCALIZE("HOME_PAGE.new_project_button"), LOCALIZE("HOME_PAGE.new_project_desc"))
        new_card.clicked.connect(self.handle_new_project)
        
        open_icon = load_svg_icon(ICON_PATHS["open_project"], "#0078d4", QSize(32, 32))
        open_card = ActionCard(open_icon, LOCALIZE("HOME_PAGE.open_project_button"), LOCALIZE("HOME_PAGE.open_project_desc"))
        open_card.clicked.connect(self.handle_open_project)
        
        actions_layout.addWidget(new_card)
        actions_layout.addWidget(open_card)
        
        layout.addWidget(title_container)
        layout.addWidget(actions_container)
        layout.addStretch()
        
        return sidebar

    def _create_content_area(self) -> QWidget:
        content_area = QWidget()
        content_area.setObjectName("homeContentArea")
        layout = QVBoxLayout(content_area)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(24)
        
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)
        
        recent_icon = load_svg_icon(ICON_PATHS["recent_projects"], "#0078d4", QSize(24, 24))
        icon_label = QLabel()
        icon_label.setPixmap(recent_icon.pixmap(QSize(24, 24)))
        
        header_label = QLabel(LOCALIZE("HOME_PAGE.recent_projects_title"))
        header_label.setObjectName("recentProjectsHeader")
        
        header_layout.addWidget(icon_label)
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        
        self.recent_projects_list = QListWidget()
        self.recent_projects_list.setObjectName("recentProjectsList")
        self.recent_projects_list.setFrameShape(QFrame.Shape.NoFrame)
        self.recent_projects_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.recent_projects_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.recent_projects_list.itemDoubleClicked.connect(self.handle_recent_item_opened)
        
        layout.addWidget(header_container)
        layout.addWidget(self.recent_projects_list, 1)
        
        return content_area

    def populate_recent_projects(self):
        self.recent_projects_list.clear()
        recent_projects = PROJECT_MANAGER.get_recent_projects()
        
        if not recent_projects:
            empty_item = QListWidgetItem(LOCALIZE("HOME_PAGE.no_recent_projects"))
            empty_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_item.setFlags(empty_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.recent_projects_list.addItem(empty_item)
        else:
            for project in recent_projects:
                list_item = QListWidgetItem(self.recent_projects_list)
                list_item.setData(Qt.ItemDataRole.UserRole, project["path"])
                item_widget = RecentProjectItemWidget(project["name"], project["last_modified"])
                list_item.setSizeHint(item_widget.sizeHint())
                self.recent_projects_list.addItem(list_item)
                self.recent_projects_list.setItemWidget(list_item, item_widget)

    def handle_new_project(self):
        project_name, ok = QInputDialog.getText(self, LOCALIZE("HOME_PAGE.new_project_dialog_title"), LOCALIZE("HOME_PAGE.new_project_dialog_label"))
        if ok and project_name.strip():
            project_path = PROJECT_MANAGER.create_new_project(project_name.strip())
            if project_path:
                self.populate_recent_projects()
                self.newProjectCreated.emit(project_path)
            else:
                QMessageBox.warning(self, LOCALIZE("COMMON.error"), LOCALIZE("HOME_PAGE.project_exists_error", name=project_name.strip()))

    def handle_open_project(self):
        projects_dir = PROJECT_MANAGER.projects_dir
        file_path, _ = QFileDialog.getOpenFileName(self, LOCALIZE("HOME_PAGE.open_project_dialog_title"), projects_dir, LOCALIZE("HOME_PAGE.project_file_filter"))
        if file_path:
            self.projectOpened.emit(file_path)
            
    def handle_recent_item_opened(self, item: QListWidgetItem):
        project_path = item.data(Qt.ItemDataRole.UserRole)
        if project_path:
            self.projectOpened.emit(project_path)
