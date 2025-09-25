# Implementation Patterns and Best Practices

## Code Architecture Patterns

### 1. Icon Management Pattern (Updated September 2025)
Centralized icon management with comprehensive path registry and utility functions.

```python
# components/widgets/icons.py
ICON_PATHS = {
    "eye_open": "eye-open.svg",
    "eye_close": "eye-close.svg",
    "minus": "minus.svg",
    "plus": "plus.svg",
    # ... more icons
}

def load_icon(icon_name: str, size: Optional[Union[QSize, str]] = None, color: Optional[str] = None) -> QIcon:
    """Load an icon with optional size and color customization."""
    icon_path = get_icon_path(icon_name)
    
    if size is None:
        size = DEFAULT_SIZES["button"]
    elif isinstance(size, str):
        size = DEFAULT_SIZES[size]
    
    if color is not None:
        return load_svg_icon(icon_path, color, size)
    else:
        return QIcon(icon_path)
```

**Best Practices:**
- Use centralized icon registry to avoid hardcoded paths
- Provide size presets ("button", "toolbar", "large") for consistency
- Support color customization for SVG icons
- Include backward compatibility aliases for existing usage

### 2. State-Aware Toggle Buttons Pattern
Interactive buttons that change appearance and behavior based on internal state.

```python
def _update_enable_button(self):
    """Update button icon and tooltip based on current state."""
    if self.step.enabled:
        # State is enabled, show action to disable
        icon = load_icon("eye_close", "button")
        tooltip = LOCALIZE("PREPROCESS.disable_step_tooltip")
    else:
        # State is disabled, show action to enable
        icon = load_icon("eye_open", "button")
        tooltip = LOCALIZE("PREPROCESS.enable_step_tooltip")
    
    self.enable_toggle_btn.setIcon(icon)
    self.enable_toggle_btn.setToolTip(tooltip)

def _toggle_enabled(self):
    """Toggle state and update UI accordingly."""
    self.step.enabled = not self.step.enabled
    self._update_enable_button()
    self._update_appearance()
    # Notify parent of state change
    self.toggled.emit(self.step_index, self.step.enabled)
```

**Key Principles:**
- Icon represents the action, not the current state
- Tooltip describes what will happen when clicked
- Emit signals to notify parent components of state changes
- Update multiple UI elements consistently when state changes

### 3. Dataset Type Tracking Pattern
Track user navigation between different data types to provide contextual behavior.

```python
def __init__(self):
    # Track dataset selection for pipeline transfer logic
    self._last_selected_was_preprocessed = False

def handle_dataset_selection(self, dataset_name: str):
    """Handle dataset selection with context-aware logic."""
    metadata = PROJECT_MANAGER.get_dataframe_metadata(dataset_name)
    is_preprocessed = metadata and metadata.get('is_preprocessed', False)
    
    if is_preprocessed:
        # Check if moving between preprocessed datasets
        if self._last_selected_was_preprocessed and len(self.pipeline_steps) > 0:
            self._show_pipeline_transfer_dialog(dataset_name)
        else:
            self._load_preprocessing_pipeline(metadata.get('preprocessing_pipeline', []))
        self._last_selected_was_preprocessed = True
    else:
        # Raw dataset selected - always clear
        self._clear_preprocessing_history()
        self._last_selected_was_preprocessed = False
```

**Benefits:**
- Provides contextual user interactions
- Prevents unwanted dialogs in wrong scenarios
- Maintains state across user navigation
- Enables intelligent pipeline management

### 4. Centralized Dataset Name Cleaning Pattern
Consistent emoji and prefix removal across the application.

```python
def _clean_dataset_name(self, item_text: str) -> str:
    """Clean dataset name by removing UI prefixes like emojis."""
    return item_text.replace("📊 ", "").replace("🔬 ", "").strip()

# Usage throughout the application
dataset_name = self._clean_dataset_name(first_item.text())  # Instead of hardcoded replace()
```

**Advantages:**
- Single source of truth for dataset name cleaning
- Easy to add new emoji types or prefixes
- Prevents bugs from missed emoji types
- Consistent behavior across all components

### 5. Widget Component Pattern
Used throughout the application for parameter input widgets and UI components.

```python
class BaseParameterWidget(QWidget):
    """Base class for all parameter input widgets."""
    
    value_changed = Signal()  # Real-time value change notification
    
    def __init__(self, param_name: str, config: dict, parent=None):
        super().__init__(parent)
        self.param_name = param_name
        self.config = config
        self._setup_ui()
        self._connect_signals()
    
    def get_value(self):
        """Get current widget value - must be implemented by subclasses."""
        raise NotImplementedError
    
    def set_value(self, value):
        """Set widget value - must be implemented by subclasses."""
        raise NotImplementedError
    
    def validate(self) -> bool:
        """Validate current value - returns True if valid."""
        return True
    
    def _setup_ui(self):
        """Setup widget UI - implemented by subclasses."""
        pass
    
    def _connect_signals(self):
        """Connect internal signals - implemented by subclasses."""
        pass
```

**Usage Pattern:**
- Inherit from base class for consistency
- Implement required methods (get_value, set_value)
- Emit value_changed signal for real-time updates
- Use validate() method for input validation
- Apply consistent styling using configuration

### 2. Method Registry Pattern
Used for dynamic method discovery and instantiation in preprocessing pipeline.

```python
class MethodRegistry:
    """Registry for preprocessing methods with automatic discovery."""
    
    def __init__(self):
        self._methods = {}  # category -> {method_name: method_info}
    
    def register_method(self, category: str, name: str, method_class, **kwargs):
        """Register a preprocessing method."""
        if category not in self._methods:
            self._methods[category] = {}
        
        self._methods[category][name] = {
            'class': method_class,
            'name': name,
            'category': category,
            **kwargs
        }
    
    def get_method_info(self, category: str, method: str) -> dict:
        """Get method information."""
        return self._methods.get(category, {}).get(method)
    
    def create_method_instance(self, category: str, method: str, params: dict):
        """Create method instance with parameters."""
        method_info = self.get_method_info(category, method)
        if not method_info:
            raise ValueError(f"Method {category}.{method} not found")
        
        method_class = method_info['class']
        return method_class(**params)
```

**Usage Benefits:**
- Dynamic method discovery
- Consistent method interfaces
- Easy addition of new methods
- Automatic parameter handling

### 3. Pipeline Step Pattern
Used for preprocessing pipeline management with enable/disable functionality.

```python
@dataclass
class PipelineStep:
    """Represents a step in the preprocessing pipeline."""
    
    category: str
    method: str
    params: dict
    enabled: bool = True
    order: int = 0
    
    def execute(self, data):
        """Execute this step on the provided data."""
        if not self.enabled:
            return data
        
        method_instance = REGISTRY.create_method_instance(
            self.category, self.method, self.params
        )
        return method_instance.process(data)
    
    def validate_params(self) -> bool:
        """Validate step parameters."""
        method_info = REGISTRY.get_method_info(self.category, self.method)
        if not method_info:
            return False
        
        # Validate parameters against method requirements
        param_info = method_info.get('param_info', {})
        for param_name, requirements in param_info.items():
            if param_name not in self.params:
                if requirements.get('required', False):
                    return False
        return True
```

### 4. Observer Pattern for Real-time Updates
Used for parameter widgets and preview generation.

```python
class PreviewManager:
    """Manages real-time preview updates."""
    
    def __init__(self):
        self._observers = []
        self._data = None
        self._pipeline = []
    
    def add_observer(self, callback):
        """Add observer for preview updates."""
        self._observers.append(callback)
    
    def remove_observer(self, callback):
        """Remove observer."""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def notify_observers(self, event_type: str, data=None):
        """Notify all observers of changes."""
        for callback in self._observers:
            try:
                callback(event_type, data)
            except Exception as e:
                print(f"Observer error: {e}")
    
    def update_pipeline(self, pipeline_steps):
        """Update pipeline and trigger preview update."""
        self._pipeline = pipeline_steps
        self._generate_preview()
    
    def _generate_preview(self):
        """Generate preview data and notify observers."""
        if self._data is None:
            return
        
        preview_data = self._apply_pipeline(self._data, self._pipeline)
        self.notify_observers('preview_updated', preview_data)
```

## UI Development Patterns

### 1. Dynamic Widget Generation
Pattern for automatically generating parameter widgets based on method signatures.

```python
def create_parameter_widget(param_name: str, param_info: dict, parent=None):
    """Factory function for creating parameter widgets."""
    
    param_type = param_info.get('type', 'float')
    widget_config = {
        'param_name': param_name,
        'label': param_info.get('label', param_name.title()),
        'tooltip': param_info.get('description', ''),
        'units': param_info.get('units', ''),
    }
    
    if param_type == 'int':
        widget = IntParameterWidget(**widget_config, parent=parent)
        if 'min' in param_info:
            widget.set_minimum(param_info['min'])
        if 'max' in param_info:
            widget.set_maximum(param_info['max'])
    
    elif param_type == 'float':
        widget = FloatParameterWidget(**widget_config, parent=parent)
        widget.set_precision(param_info.get('precision', 2))
        if 'min' in param_info:
            widget.set_minimum(param_info['min'])
        if 'max' in param_info:
            widget.set_maximum(param_info['max'])
    
    elif param_type == 'range':
        widget = RangeParameterWidget(**widget_config, parent=parent)
        widget.set_range(param_info.get('min', 0), param_info.get('max', 100))
    
    elif param_type == 'choice':
        widget = ChoiceParameterWidget(**widget_config, parent=parent)
        widget.set_choices(param_info.get('choices', []))
    
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")
    
    # Set default value if provided
    if 'default' in param_info:
        widget.set_value(param_info['default'])
    
    return widget
```

### 2. Color-Coded Status Indicators
Pattern for providing visual feedback on widget states.

```python
class StatusMixin:
    """Mixin for adding status indication to widgets."""
    
    STATUS_COLORS = {
        'valid': '#2ecc71',    # Green
        'invalid': '#e74c3c',  # Red
        'warning': '#f39c12',  # Orange
        'neutral': '#95a5a6'   # Gray
    }
    
    def set_status(self, status: str, message: str = ''):
        """Set widget status with visual feedback."""
        color = self.STATUS_COLORS.get(status, self.STATUS_COLORS['neutral'])
        
        # Update border color
        self.setStyleSheet(f"""
            QWidget {{
                border: 2px solid {color};
                border-radius: 4px;
            }}
        """)
        
        # Update tooltip with status message
        if message:
            self.setToolTip(f"Status: {status.title()}\n{message}")
        
        # Emit status change signal
        if hasattr(self, 'status_changed'):
            self.status_changed.emit(status, message)
```

### 3. Matplotlib Integration Pattern
Pattern for integrating matplotlib with PySide6 for scientific plotting.

```python
class MatplotlibWidget(QWidget):
    """Widget for embedding matplotlib plots in PySide6."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Configure matplotlib for Qt
        self._configure_matplotlib()
    
    def _configure_matplotlib(self):
        """Configure matplotlib for optimal Qt integration."""
        # Use Qt-specific backend
        matplotlib.use('Qt5Agg')
        
        # Configure for scientific plotting
        plt.style.use('default')
        self.figure.patch.set_facecolor('white')
        
        # Configure for high DPI displays
        self.canvas.setStyleSheet("background-color: white;")
    
    def plot_spectra(self, wavenumbers, intensities, auto_focus=False, **kwargs):
        """Plot spectral data with optional auto-focus."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plot data
        ax.plot(wavenumbers, intensities, **kwargs)
        
        # Apply auto-focus if requested
        if auto_focus:
            focus_range = self.detect_signal_range(wavenumbers, intensities)
            if focus_range:
                ax.set_xlim(focus_range)
        
        # Configure axes
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Intensity')
        ax.grid(True, alpha=0.3)
        
        # Update display
        self.figure.tight_layout()
        self.canvas.draw()
```

### 4. Dynamic UI Sizing Pattern
Pattern for creating UI elements that adapt to content length, especially for internationalization.

```python
def adjust_button_width_to_text(button: QPushButton, min_width: int = 80):
    """Adjust button width dynamically based on text content."""
    from PySide6.QtGui import QFontMetrics
    
    # Get current text and font
    text = button.text()
    font = button.font()
    
    # Calculate text width using font metrics
    font_metrics = QFontMetrics(font)
    text_width = font_metrics.horizontalAdvance(text)
    
    # Account for additional UI elements
    icon_width = 16 if button.icon() else 0
    spacing = 8 if text.strip() and icon_width > 0 else 0
    padding = 16  # CSS padding (8px left + 8px right)
    border = 4    # CSS border (2px left + 2px right)
    
    # Calculate total width
    total_width = text_width + icon_width + spacing + padding + border
    dynamic_width = max(min_width, total_width)
    
    button.setFixedWidth(dynamic_width)

# Integration example
class LocalizedToggleButton(QPushButton):
    """Button that auto-adjusts width for localized text."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedHeight(32)  # Fixed height, dynamic width
        
    def setText(self, text: str):
        """Override setText to trigger width adjustment."""
        super().setText(text)
        self.adjust_width_to_text()
        
    def adjust_width_to_text(self):
        """Adjust width based on current text content."""
        adjust_button_width_to_text(self, min_width=80)
```

**Benefits:**
- Responsive UI that adapts to different languages
- Prevents text truncation in longer translations
- Maintains visual consistency across locales
- Automatic adjustment without manual sizing

**Use Cases:**
- Toggle buttons with ON/OFF states
- Buttons with localized text
- Dynamic labels that change content
- UI elements with varying text lengths

## Error Handling Patterns

### 1. Graceful Degradation Pattern
Used throughout the application to handle errors without crashing.

```python
def safe_execute(func, fallback_value=None, log_errors=True):
    """Execute function with graceful error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                create_logs("safe_execute", func.__name__, 
                           f"Error in {func.__name__}: {e}", status='error')
            return fallback_value
    return wrapper

# Usage example
@safe_execute
def process_spectral_data(data, method_params):
    """Process spectral data with error handling."""
    # Processing logic here
    return processed_data
```

### 2. Validation Pattern
Used for parameter validation with user feedback.

```python
class ValidationResult:
    """Result of validation operation."""
    
    def __init__(self, is_valid: bool, message: str = '', value=None):
        self.is_valid = is_valid
        self.message = message
        self.value = value

def validate_parameter(value, param_info: dict) -> ValidationResult:
    """Validate parameter against requirements."""
    
    # Type validation
    expected_type = param_info.get('type', 'float')
    if expected_type == 'int':
        try:
            value = int(value)
        except (ValueError, TypeError):
            return ValidationResult(False, "Must be an integer")
    
    elif expected_type == 'float':
        try:
            value = float(value)
        except (ValueError, TypeError):
            return ValidationResult(False, "Must be a number")
    
    # Range validation
    if 'min' in param_info and value < param_info['min']:
        return ValidationResult(False, f"Must be >= {param_info['min']}")
    
    if 'max' in param_info and value > param_info['max']:
        return ValidationResult(False, f"Must be <= {param_info['max']}")
    
    return ValidationResult(True, "Valid", value)
```

## Performance Optimization Patterns

### 1. Caching Pattern
Used for expensive operations like data processing.

```python
from functools import lru_cache
import hashlib

class DataCache:
    """Cache for processed data with automatic invalidation."""
    
    def __init__(self, max_size=100):
        self._cache = {}
        self._max_size = max_size
        self._access_order = []
    
    def get_cache_key(self, data, pipeline_steps):
        """Generate cache key from data and pipeline."""
        # Create hash from data shape and pipeline configuration
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()
        pipeline_hash = hashlib.md5(str(pipeline_steps).encode()).hexdigest()
        return f"{data_hash}_{pipeline_hash}"
    
    def get(self, key):
        """Get cached result."""
        if key in self._cache:
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def set(self, key, value):
        """Store result in cache."""
        # Remove oldest if cache is full
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = value
        if key not in self._access_order:
            self._access_order.append(key)
```

### 2. Lazy Loading Pattern
Used for components that are expensive to initialize.

```python
class LazyLoader:
    """Lazy loader for expensive components."""
    
    def __init__(self, loader_func, *args, **kwargs):
        self._loader_func = loader_func
        self._args = args
        self._kwargs = kwargs
        self._instance = None
        self._loaded = False
    
    def __call__(self):
        """Get or create instance."""
        if not self._loaded:
            self._instance = self._loader_func(*self._args, **self._kwargs)
            self._loaded = True
        return self._instance
    
    def is_loaded(self):
        """Check if instance has been loaded."""
        return self._loaded

# Usage example
def create_expensive_component():
    # Expensive initialization here
    return ExpensiveComponent()

lazy_component = LazyLoader(create_expensive_component)

# Component is only created when first accessed
component = lazy_component()
```

## Configuration Management Patterns

### 1. Configuration Validation Pattern
Used for ensuring configuration integrity.

```python
import json
from pathlib import Path

class ConfigurationManager:
    """Manages application configuration with validation."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._config = {}
        self._defaults = self._get_defaults()
        self.load_config()
    
    def _get_defaults(self):
        """Define default configuration values."""
        return {
            'ui': {
                'theme': 'light',
                'auto_save': True,
                'preview_update_delay': 500  # ms
            },
            'processing': {
                'cache_size': 100,
                'parallel_processing': True,
                'auto_focus': {
                    'enabled': True,
                    'threshold': 0.1
                }
            },
            'data': {
                'default_format': 'csv',
                'backup_enabled': True
            }
        }
    
    def load_config(self):
        """Load configuration from file with fallback to defaults."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                self._config = self._merge_configs(self._defaults, file_config)
            else:
                self._config = self._defaults.copy()
                self.save_config()  # Create default config file
        except Exception as e:
            print(f"Error loading config: {e}")
            self._config = self._defaults.copy()
    
    def _merge_configs(self, defaults, user_config):
        """Recursively merge user config with defaults."""
        result = defaults.copy()
        for key, value in user_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set target value
        config[keys[-1]] = value
        self.save_config()
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")
```

These patterns provide a solid foundation for maintaining code quality, consistency, and extensibility throughout the Raman spectroscopy application. Each pattern addresses specific architectural needs while maintaining the overall design philosophy of the project.