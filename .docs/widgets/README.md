# Widgets Component Documentation

This folder contains comprehensive documentation for the widgets component package of the Raman Spectroscopy application.

## Documentation Files

### Core Components
- **[widgets-component-package.md](widgets-component-package.md)** - Main package overview and structure
- **[icon-management-system.md](icon-management-system.md)** - Centralized icon loading and management
- **[parameter-constraint-system.md](parameter-constraint-system.md)** - Intelligent parameter validation and hints

### Implementation Details
- **[parameter-widgets-fixes.md](parameter-widgets-fixes.md)** - Bug fixes and improvements
- **[enhanced-parameter-widgets.md](enhanced-parameter-widgets.md)** - Advanced widget features

## Widget Components Overview

### Parameter Widgets (`parameter_widgets.py`)
Reusable parameter input widgets for scientific applications:
- `CustomSpinBox` - Integer parameter input with +/- buttons
- `CustomDoubleSpinBox` - Float parameter input with +/- buttons
- Clean UI design optimized for scientific workflows

### Enhanced Parameter Widgets (`enhanced_parameter_widgets.py`)
Advanced widgets with constraint validation:
- `ConstrainedSpinBox` - Integer input with real-time validation
- `ParameterHintWidget` - Standalone constraint display
- Integration with parameter constraint database

### Icon Management (`icons.py`)
Centralized icon system:
- Registry of all application icons
- Flexible loading with custom sizing
- Performance optimized with direct QIcon loading
- Legacy compatibility support

### Matplotlib Integration (`matplotlib_widget.py`)
Scientific plotting widgets:
- Interactive matplotlib charts
- Optimized for Raman spectroscopy data
- Clean integration with Qt interface

## Usage Guidelines

### Import Structure
```python
# Basic widgets
from components.widgets.parameter_widgets import CustomSpinBox, CustomDoubleSpinBox

# Enhanced widgets with constraints
from components.widgets.enhanced_parameter_widgets import ConstrainedSpinBox

# Icon management
from components.widgets.icons import create_button_icon, load_icon

# Matplotlib integration
from components.widgets.matplotlib_widget import MatplotlibWidget
```

### Best Practices
1. **Use Enhanced Widgets** for preprocessing parameters to get automatic validation
2. **Leverage Icon System** for consistent UI across the application
3. **Follow Parameter Naming** conventions for automatic constraint loading
4. **Check Documentation** for specific widget configuration options

## Related Systems

This widgets package integrates with:
- **Preprocessing Functions** (`functions/preprocess/`) - Parameter constraint database
- **Application Pages** (`pages/`) - UI composition and layout
- **Configuration System** (`configs/`) - Style and behavior settings

For application-wide documentation, see the main `docs/` folder at the project root.