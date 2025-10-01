"""
Icon management utilities for the widgets package.

This module provides centralized icon path management and loading functionality
for all widgets in the components/widgets package.
"""

import os
from typing import Optional, Union
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from utils import load_svg_icon

# Base path for all icons
ICONS_BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "icons")

# Icon file paths registry
ICON_PATHS = {
    # Widget control icons (current usage)
    "minus": "minus.svg",
    "plus": "plus.svg", 
    "trash": "trash-xmark.svg",
    "trash_bin": "trash-bin.svg",
    
    # Legacy icons (still used in some files - need to be migrated)
    "decrease_circle": "decrease-circle.svg",  # Used in preprocess_page_utils/widgets.py
    "increase_circle": "increase-circle.svg",  # Used in preprocess_page_utils/widgets.py
    
    # Navigation and view icons
    "chevron_down": "chevron-down.svg",
    "eye_open": "eye-open.svg",          # Used in preprocess_page.py
    "eye_close": "eye-close.svg",        # Used in preprocess_page.py
    "reload": "reload.svg",              # Used in preprocess_page.py, home_page.py, utils.py
    "focus_horizontal": "focus-horizontal-round.svg",  # Manual focus button
    "export": "export-button.svg",       # Export button icon
    
    # Project management icons
    "new_project": "new-project.svg",    # Used in home_page.py, utils.py
    "load_project": "load-project.svg",  # Used as "open_project" in home_page.py, utils.py
    "recent_project": "recent-project.svg", # Used as "recent_projects" in home_page.py, utils.py
    
    # Aliases for backward compatibility with utils.py ICON_PATHS
    "open_project": "load-project.svg",     # Alias for load_project
    "recent_projects": "recent-project.svg", # Alias for recent_project (plural form)
}

# Default icon sizes for different widget types
DEFAULT_SIZES = {
    "button": QSize(16, 16),
    "toolbar": QSize(24, 24),
    "large": QSize(32, 32),
}

def get_icon_path(icon_name: str) -> str:
    """
    Get the full path to an icon file.
    
    Args:
        icon_name: Name of the icon (key from ICON_PATHS)
        
    Returns:
        Full path to the icon file
        
    Raises:
        KeyError: If icon_name is not found in registry
    """
    if icon_name not in ICON_PATHS:
        raise KeyError(f"Icon '{icon_name}' not found in registry. Available icons: {list(ICON_PATHS.keys())}")
    
    return os.path.join(ICONS_BASE_PATH, ICON_PATHS[icon_name])

def load_icon(icon_name: str, size: Optional[Union[QSize, str]] = None, color: Optional[str] = None) -> QIcon:
    """
    Load an icon with optional size and color customization.
    
    Args:
        icon_name: Name of the icon (key from ICON_PATHS)
        size: Icon size - can be QSize object or string key from DEFAULT_SIZES
        color: Optional color for SVG icons (hex color or Qt color name)
        
    Returns:
        QIcon object ready for use
        
    Example:
        >>> icon = load_icon("minus", "button")
        >>> icon = load_icon("plus", QSize(20, 20), "#6c757d")
    """
    icon_path = get_icon_path(icon_name)
    
    # Handle size parameter
    if size is None:
        size = DEFAULT_SIZES["button"]
    elif isinstance(size, str):
        if size not in DEFAULT_SIZES:
            raise KeyError(f"Size '{size}' not found. Available sizes: {list(DEFAULT_SIZES.keys())}")
        size = DEFAULT_SIZES[size]
    
    # Load icon based on whether color customization is needed
    if color is not None:
        # Use the utility function for color customization
        return load_svg_icon(icon_path, color, size)
    else:
        # Use direct QIcon loading for better performance
        icon = QIcon(icon_path)
        return icon

def create_button_icon(icon_name: str, color: Optional[str] = None) -> QIcon:
    """
    Create an icon optimized for button usage.
    
    Args:
        icon_name: Name of the icon
        color: Optional color override
        
    Returns:
        QIcon sized appropriately for buttons
    """
    return load_icon(icon_name, "button", color)

def create_toolbar_icon(icon_name: str, color: Optional[str] = None) -> QIcon:
    """
    Create an icon optimized for toolbar usage.
    
    Args:
        icon_name: Name of the icon
        color: Optional color override
        
    Returns:
        QIcon sized appropriately for toolbars
    """
    return load_icon(icon_name, "toolbar", color)

def list_available_icons() -> list:
    """
    Get a list of all available icon names.
    
    Returns:
        List of icon names that can be used with load_icon()
    """
    return list(ICON_PATHS.keys())

def verify_icon_exists(icon_name: str) -> bool:
    """
    Check if an icon file actually exists on disk.
    
    Args:
        icon_name: Name of the icon to check
        
    Returns:
        True if icon file exists, False otherwise
    """
    try:
        icon_path = get_icon_path(icon_name)
        return os.path.exists(icon_path)
    except KeyError:
        return False

def get_missing_icons() -> list:
    """
    Get a list of icons that are registered but missing from disk.
    
    Returns:
        List of icon names that are registered but file doesn't exist
    """
    missing = []
    for icon_name in ICON_PATHS.keys():
        if not verify_icon_exists(icon_name):
            missing.append(icon_name)
    return missing