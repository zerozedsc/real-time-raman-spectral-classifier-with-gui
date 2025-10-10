# Widgets Component Package Documentation

## 1. Overview

The **Widgets Component Package** provides reusable, professional-grade parameter input widgets optimized for scientific applications, particularly Raman spectroscopy. This package is located at `components/widgets/` and can be used across all pages of the application for consistent UI and functionality.

---

## 2. Package Structure

```
components/
└── widgets/
    ├── __init__.py          # Package initialization and exports
    ├── parameter_widgets.py # Main widget implementations
    ├── matplotlib_widget.py # Matplotlib integration widgets
    ├── icons.py            # Centralized icon management
    └── utils.py            # Shared utilities and imports
```

### **Key Features**
- **Centralized Icon Management**: New icons.py provides unified icon loading across the application
- **Professional SVG Icons**: Uses minus.svg and plus.svg with professional gray styling
- **Real-time Parameter Validation**: Immediate feedback and constraint enforcement
- **Medical/Scientific Styling**: Professional appearance optimized for research applications
- **Extensible Parameter System**: Easy to add new parameter types and widgets
- **Matplotlib Integration**: MatplotlibWidget moved to widgets package for better organization
- **Japanese/English Localization**: Full internationalization support

---

## 3. Available Widgets

### **CustomSpinBox**
Integer input widget with professional SVG +/- buttons.

**Features:**
- SVG icons for increase/decrease buttons (minus.svg/plus.svg with professional gray styling)
- Real-time value validation and constraints
- Professional styling with hover/pressed states (#6c757d color scheme)
- Configurable range, step, and styling override protection

**Usage:**
```python
from components.widgets import CustomSpinBox

widget = CustomSpinBox()
widget.setRange(0, 100)
widget.setSingleStep(1)
widget.setValue(50)
widget.valueChanged.connect(callback)
```

### **CustomDoubleSpinBox** 
Float input widget with professional SVG +/- buttons.

**Features:**
- Same SVG icon system as CustomSpinBox
- Configurable decimal precision
- Suffix support (e.g., " cm⁻¹")
- Scientific notation support for large ranges

**Usage:**
```python
from components.widgets import CustomDoubleSpinBox

widget = CustomDoubleSpinBox()
widget.setRange(0.0, 1000.0)
widget.setDecimals(2)
widget.setSuffix(" cm⁻¹")
widget.setValue(520.5)
```

### **RangeParameterWidget**
Dual-range input with synchronized spinboxes and sliders.

**Features:**
- Professional dual CustomDoubleSpinBox inputs
- Synchronized slider controls
- Real-time range validation (min < max)
- Automatic data range clamping
- Professional styling with medical context

**Usage:**
```python
from components.widgets import RangeParameterWidget

widget = RangeParameterWidget(
    param_name="region",
    info={"type": "tuple", "range": [400, 4000]},
    default_value=(800, 1800),
    data_range=(378.5, 3517.8)  # Actual data bounds
)
```

### **DictParameterWidget**
Dynamic dictionary parameter input with add/remove functionality.

**Features:**
- Add/remove entries dynamically
- CustomDoubleSpinBox for values
- Professional trash icon for removal
- Scrollable container for multiple entries
- Real-time parameter change notifications

### **DynamicParameterWidget**
Comprehensive parameter widget factory that creates appropriate controls based on parameter metadata.

**Supported Parameter Types:**
- `int`: CustomSpinBox
- `float`: CustomDoubleSpinBox  
- `scientific`: CustomDoubleSpinBox with scientific range
- `choice`: Styled QComboBox
- `tuple`: RangeParameterWidget
- `dict`: DictParameterWidget
- `bool`: Styled QCheckBox
- `list_int`, `list_float`: Formatted QLineEdit
- `optional`: Optional QLineEdit

---

## 4. SVG Icon Integration

### **Icon Assets**
- **decrease-circle.svg**: Professional circular minus icon
- **increase-circle.svg**: Professional circular plus icon
- **Location**: `assets/icons/`

### **Implementation**
```python
# Proper SVG icon loading
decrease_icon_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "icons", "decrease-circle.svg")
self.minus_btn.setIcon(load_svg_icon(decrease_icon_path, None, QSize(16, 16)))
```

### **Styling Integration**
- Icons sized at 16x16px within 24x24px buttons
- Circular button styling with border colors matching icon semantics
- Hover/pressed states with color transitions
- Professional color scheme (red for decrease, green for increase)

---

## 5. Usage in Pages

### **Import Structure**
```python
# In any page file
import sys
import os
# Add components to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from components.widgets import (
    CustomSpinBox, 
    CustomDoubleSpinBox, 
    RangeParameterWidget,
    DynamicParameterWidget
)
```

### **Integration Example**
```python
# Create parameter widget for Raman spectroscopy
method_info = {
    "param_info": {
        "region": {"type": "tuple", "range": [400, 4000], "description": "Wavenumber range"},
        "window_length": {"type": "int", "range": [3, 21], "step": 2}
    },
    "default_params": {"region": (800, 1800), "window_length": 7}
}

widget = DynamicParameterWidget(method_info, data_range=(378.5, 3517.8))
parameters = widget.get_parameters()  # Extract all parameter values
```

---

## 6. Medical/Scientific Features

### **Contextual Design**
- **Wavenumber Units**: Automatic cm⁻¹ suffix for spectroscopic parameters
- **Range Validation**: Medical-appropriate parameter bounds enforcement
- **Professional Appearance**: Clean, medical-grade interface design
- **Accessibility**: Keyboard navigation and screen reader support

### **Real-time Validation**
- **Parameter Constraints**: Live enforcement of min/max bounds
- **Cross-parameter Validation**: Ensures min < max for ranges
- **Visual Feedback**: Immediate indication of valid/invalid states
- **Error Prevention**: Proactive validation prevents invalid parameter combinations

---

## 7. Technical Architecture

### **Component Isolation**
- **Self-contained Package**: No dependencies on page-specific code
- **Reusable Across Pages**: Can be used in any part of the application
- **Consistent Styling**: Professional medical theme throughout
- **Performance Optimized**: Efficient rendering and event handling

### **Backward Compatibility**
```python
# Aliases maintained for existing code
ParameterWidget = DynamicParameterWidget
ParameterGroupWidget = DynamicParameterWidget
```

### **Error Handling**
- **Graceful Degradation**: Falls back to emoji icons if SVG unavailable
- **Logging Integration**: Comprehensive error logging via create_logs
- **Parameter Validation**: Robust handling of invalid parameter types/values

---

## 8. Development Guidelines

### **Extending Widgets**
1. Add new widget class to `parameter_widgets.py`
2. Follow naming convention: `Custom[Type]Widget`
3. Implement consistent signal patterns (`valueChanged`, `parametersChanged`)
4. Use professional styling with medical color scheme
5. Export in `__init__.py`

### **Testing Widgets**
```python
# Test import structure
from components.widgets import CustomSpinBox
widget = CustomSpinBox()
assert widget.value() == 0

# Test SVG icons
assert widget.minus_btn.icon() is not None
assert widget.plus_btn.icon() is not None
```

### **Performance Considerations**
- **Icon Caching**: SVG icons loaded once per widget instance
- **Event Debouncing**: Parameter change events debounced for performance
- **Memory Management**: Proper widget cleanup and signal disconnection

---

## 9. Migration Guide

### **From Old Structure**
```python
# OLD (deprecated)
from .preprocess_page_utils.widgets import CustomSpinBox

# NEW (current)
from components.widgets import CustomSpinBox
```

---

## 7. Icon Management System

### **Centralized Icon Loading**
The `icons.py` module provides centralized icon management for the entire application.

**Available Functions:**
```python
from components.widgets import load_icon, create_button_icon, create_toolbar_icon

# Load any icon with custom size and color
icon = load_icon("minus", size="button", color="#6c757d")

# Quick button icon creation
button_icon = create_button_icon("plus")

# Toolbar-sized icons
toolbar_icon = create_toolbar_icon("reload")
```

**Available Icons:**
- `minus`, `plus` - Widget control icons
- `trash`, `trash_bin` - Delete/remove actions
- `eye_open`, `eye_close` - Visibility toggles
- `reload` - Refresh/reload actions
- `new_project`, `load_project`, `recent_project` - Project management
- `chevron_down` - Dropdown indicators
- `decrease_circle`, `increase_circle` - Legacy icons (being phased out)

**Icon Registration:**
```python
# Get list of all available icons
available_icons = list_available_icons()

# Get path to specific icon file
icon_path = get_icon_path("minus")
```

### **Benefits of New Structure**
- **Reusability**: Use widgets in any page (home, preprocess, analysis, etc.)
- **Maintainability**: Single source of truth for widget implementations
- **Icon Consistency**: Centralized icon management ensures consistent styling
- **Easy Migration**: Gradual migration from hardcoded paths to centralized system
- **Consistency**: Uniform styling and behavior across entire application
- **Scalability**: Easy to add new pages that use the same professional widgets

This widgets package provides the foundation for professional, medical-grade parameter input across the entire Raman spectroscopy application.