# Icon Management System Documentation

## Overview

The **Icon Management System** provides centralized icon loading and management for the entire Raman Spectroscopy application. Located at `components/widgets/icons.py`, this system ensures consistent icon usage, easy maintenance, and efficient loading across all application components.

## Key Features

- **Centralized Registry**: All icon paths managed in one location
- **Flexible Loading**: Support for custom sizes, colors, and optimization
- **Legacy Support**: Backward compatibility with existing icon usage patterns
- **Type Safety**: Prevents missing icon errors with registry validation
- **Performance**: Optimized loading with direct QIcon for better performance
- **Scalability**: Easy to add new icons and update existing ones

## Icon Registry

### Current Icons

| Icon Name | File | Usage | Location |
|-----------|------|-------|----------|
| `minus` | minus.svg | Widget decrease buttons | Parameter widgets |
| `plus` | plus.svg | Widget increase buttons | Parameter widgets |
| `trash` | trash-xmark.svg | Remove/delete actions | Dict parameter widgets |
| `trash_bin` | trash-bin.svg | Alternative trash icon | Various locations |
| `eye_open` | eye-open.svg | Show/visible state | Preprocess page |
| `eye_close` | eye-close.svg | Hide/invisible state | Preprocess page |
| `reload` | reload.svg | Refresh/reload actions | Home page, preprocess |
| `chevron_down` | chevron-down.svg | Dropdown indicators | Comboboxes |
| `new_project` | new-project.svg | Create new project | Home page |
| `load_project` | load-project.svg | Open existing project | Home page |
| `recent_project` | recent-project.svg | Recent projects list | Home page |

### Legacy Icons (Being Phased Out)

| Icon Name | File | Status |
|-----------|------|--------|
| `decrease_circle` | decrease-circle.svg | ⚠️ Legacy - migrate to `minus` |
| `increase_circle` | increase-circle.svg | ⚠️ Legacy - migrate to `plus` |

## Usage Examples

### Basic Icon Loading

```python
from components.widgets import load_icon, create_button_icon

# Load icon with default size (16x16 for buttons)
icon = create_button_icon("minus")

# Load icon with custom size
icon = load_icon("reload", size=QSize(24, 24))

# Load icon with color customization
icon = load_icon("eye_open", color="#2c3e50")
```

### Widget Integration

```python
from components.widgets import create_button_icon

class MyWidget(QWidget):
    def setup_ui(self):
        # Create button with icon
        self.action_btn = QPushButton()
        self.action_btn.setIcon(create_button_icon("plus"))
        self.action_btn.setFixedSize(24, 24)
```

### Size Presets

```python
from components.widgets import load_icon

# Predefined sizes
button_icon = load_icon("minus", "button")      # 16x16
toolbar_icon = load_icon("reload", "toolbar")   # 24x24  
large_icon = load_icon("new_project", "large")  # 32x32

# Custom size
custom_icon = load_icon("trash", QSize(20, 20))
```

## API Reference

### Functions

#### `load_icon(icon_name, size=None, color=None) -> QIcon`
Main icon loading function with full customization options.

**Parameters:**
- `icon_name` (str): Name of the icon from the registry
- `size` (QSize|str, optional): Icon size or size preset ("button", "toolbar", "large")
- `color` (str, optional): Color override for SVG icons (hex or color name)

**Returns:** QIcon object ready for use

#### `create_button_icon(icon_name, color=None) -> QIcon`
Convenience function for button-sized icons (16x16).

#### `create_toolbar_icon(icon_name, color=None) -> QIcon`
Convenience function for toolbar-sized icons (24x24).

#### `get_icon_path(icon_name) -> str`
Get the full filesystem path to an icon file.

#### `list_available_icons() -> list`
Get list of all registered icon names.

#### `verify_icon_exists(icon_name) -> bool`
Check if an icon file exists on disk.

#### `get_missing_icons() -> list`
Get list of registered icons that are missing files.

## Migration Guide

### From Direct Path Loading

**Before:**
```python
# Old way - hardcoded paths
icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "icons", "minus.svg")
button.setIcon(QIcon(icon_path))
```

**After:**
```python
# New way - centralized management
from components.widgets import create_button_icon
button.setIcon(create_button_icon("minus"))
```

### From load_svg_icon Utility

**Before:**
```python
# Old way - utility function
from utils import load_svg_icon
icon = load_svg_icon("assets/icons/eye-open.svg", "#2c3e50", QSize(16, 16))
```

**After:**
```python
# New way - icon management system
from components.widgets import load_icon
icon = load_icon("eye_open", size="button", color="#2c3e50")
```

### From utils.py ICON_PATHS

**Before:**
```python
# Old way - utils ICON_PATHS
from utils import ICON_PATHS, load_svg_icon
icon = load_svg_icon(ICON_PATHS["reload"], "#7f8c8d", QSize(16, 16))
```

**After:**
```python
# New way - widgets icon system  
from components.widgets import load_icon
icon = load_icon("reload", size="button", color="#7f8c8d")
```

## Adding New Icons

### 1. Add Icon File
Place the SVG file in `assets/icons/` directory.

### 2. Register Icon
Add entry to `ICON_PATHS` in `icons.py`:

```python
ICON_PATHS = {
    # ... existing icons ...
    "my_new_icon": "my-new-icon.svg",
}
```

### 3. Update Documentation
Add the new icon to this documentation and any relevant widget docs.

### 4. Test Icon
```python
from components.widgets import verify_icon_exists, create_button_icon

# Verify icon exists
assert verify_icon_exists("my_new_icon")

# Test loading
icon = create_button_icon("my_new_icon")
```

## Performance Considerations

- **Direct QIcon Loading**: For non-colored icons, uses `QIcon(path)` for better performance
- **SVG Color Customization**: Only uses `load_svg_icon` when color parameter is provided
- **Size Optimization**: Predefined size presets reduce object creation
- **Registry Validation**: Prevents runtime errors from missing icons

## Error Handling

The system provides clear error messages for common issues:

```python
# Missing icon
try:
    icon = load_icon("nonexistent_icon")
except KeyError as e:
    print(f"Icon not found: {e}")
    # Shows available icons in error message

# Invalid size preset
try:
    icon = load_icon("minus", size="invalid_size")
except KeyError as e:
    print(f"Invalid size preset: {e}")
    # Shows available size presets
```

## Future Enhancements

- **Icon Themes**: Support for light/dark theme variants
- **SVG Preprocessing**: Cached color variants for performance
- **Icon Validation**: Automated checks for missing icons during build
- **Documentation Generation**: Auto-generate icon gallery from registry
- **Plugin System**: Allow extensions to register custom icons