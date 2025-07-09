import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QPushButton, QButtonGroup
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon

from utils import LOCALIZE

class AppTabBar(QWidget):
    """
    A custom tab bar component for main application navigation.
    Uses styled QPushButtons to act as tabs.
    """
    # Signal emitted when a tab is changed, sending the index of the tab
    tabChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("appTabBar")
        
        # --- Layout and Button Group ---
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True) # Only one button can be checked at a time

        # --- Define Tabs ---
        # The keys should match the localization files
        tabs = [
            ("data", "TABS.data"),
            ("preprocessing", "TABS.preprocessing"),
            ("machine_learning", "TABS.machine_learning"),
            ("analysis", "TABS.analysis")
        ]

        # --- Create and Add Tab Buttons ---
        for index, (object_name, loc_key) in enumerate(tabs):
            button = QPushButton(LOCALIZE(loc_key))
            button.setObjectName(object_name) # For specific styling if needed
            button.setCheckable(True)
            layout.addWidget(button)
            self.button_group.addButton(button, index)

        # Connect the button group's signal to our custom signal
        self.button_group.idClicked.connect(self.tabChanged.emit)

        # Set the first tab as active by default
        if self.button_group.button(0):
            self.button_group.button(0).setChecked(True)

    def setActiveTab(self, index: int):
        """Programmatically sets the active tab."""
        button = self.button_group.button(index)
        if button:
            button.setChecked(True)
            self.tabChanged.emit(index)


# --- Standalone Test ---
if __name__ == '__main__':
    from stylesheets import get_main_stylesheet
    MOCK_TRANSLATIONS = { "en": { "TABS": { "data": "Data", "preprocessing": "Pre-processing", "machine_learning": "ML Training", "real_time": "Real-time Prediction" }, "APP_CONFIG": {"font_family": "'Inter', 'sans-serif'"} } }
    def LOCALIZE(key):
        parts = key.split('.')
        try: return MOCK_TRANSLATIONS["en"][parts[0]][parts[1]]
        except: return key

    app = QApplication(sys.argv)
    app.setStyleSheet(get_main_stylesheet(LOCALIZE("APP_CONFIG.font_family")))
    
    window = QWidget()
    main_layout = QHBoxLayout(window)
    tab_bar = AppTabBar()
    main_layout.addWidget(tab_bar)
    
    window.setWindowTitle("AppTabBar Test")
    window.show()
    sys.exit(app.exec())
