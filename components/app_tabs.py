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
    # Signal emitted when home tab is clicked
    homeRequested = Signal()

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
        # Note: Home tab is separate and handled differently
        tabs = [
            ("data", "TABS.data"),
            ("preprocessing", "TABS.preprocessing"),
            ("machine_learning", "TABS.machine_learning"),
            ("analysis", "TABS.analysis")
        ]

        # --- Create Home Button (separate from main tabs) ---
        self.home_button = QPushButton(LOCALIZE("TABS.home"))
        self.home_button.setObjectName("home")
        self.home_button.clicked.connect(self.homeRequested.emit)
        layout.addWidget(self.home_button)
        
        # Add separator or spacing
        layout.addSpacing(10)

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

    def clearActiveTab(self):
        """Clear all active tabs (used when on home page)."""
        self.button_group.setExclusive(False)
        for button in self.button_group.buttons():
            button.setChecked(False)
        self.button_group.setExclusive(True)