# Parameter Constraint System Documentation

## Overview

The **Parameter Constraint System** provides intelligent parameter validation, user hints, and constraint enforcement for preprocessing function parameters in the Raman Spectroscopy application. This system guides users to input valid parameter values and understand the implications of their choices.

## Components

### 1. Parameter Constraints Database (`functions/preprocess/parameter_constraints.py`)

A comprehensive database of parameter constraints including:
- **Value ranges** (min/max limits)
- **Data types** (int, float, string)
- **Special constraints** (odd numbers only, log scale, etc.)
- **Typical usage ranges** for different applications
- **User-friendly hints** and explanations

### 2. Enhanced Parameter Widgets (`components/widgets/enhanced_parameter_widgets.py`)

Enhanced versions of parameter widgets that:
- **Validate input** against constraint database
- **Display real-time hints** and guidance
- **Show constraint violations** with visual feedback
- **Auto-configure** based on parameter metadata
- **Provide smart defaults** for different use cases

## Constraint Database Structure

### Parameter Categories

1. **Baseline Correction Parameters**
   - `baseline_asls_lam`: Smoothness parameter (1e3 - 1e9)
   - `baseline_asls_p`: Asymmetry parameter (0.001 - 0.1)
   - `baseline_poly_order`: Polynomial order (1 - 10)

2. **Spike Removal Parameters**
   - `spike_gaussian_kernel`: Kernel size (odd numbers only, 1-51)
   - `spike_gaussian_threshold`: Detection threshold (1.0 - 10.0)
   - `spike_median_kernel_size`: Median filter size (odd numbers only)

3. **Derivative Parameters**
   - `derivative_order`: Order (1 or 2 only)
   - `derivative_window_length`: Window size (odd numbers, > polyorder)
   - `derivative_polyorder`: Polynomial order (< window length)

4. **Normalization Parameters**
   - `normalization_vector_norm`: Type selection (l1, l2, max)
   - `normalization_minmax_range_min/max`: Range limits

5. **Calibration Parameters**
   - `calibration_shift`: Wavenumber shift (-500 to 500 cm⁻¹)
   - `calibration_stretch`: Stretch factor (0.8 - 1.2)

### Constraint Types

#### Value Constraints
```python
"min": 1e3,              # Minimum allowed value
"max": 1e9,              # Maximum allowed value
"typical_range": (1e4, 1e8),  # Recommended range
"default": 1e6           # Default value
```

#### Type Constraints
```python
"type": "int",           # Integer only
"type": "float",         # Floating point
"type": "string",        # String with options
"options": ["l1", "l2", "max"]  # Valid string options
```

#### Special Constraints
```python
"constraints": [
    "must_be_positive",      # > 0
    "odd_numbers_only",      # Odd integers only
    "between_0_and_1",       # In range [0, 1]
    "log_scale_preferred",   # Use logarithmic steps
    "less_than_window_length" # Relational constraint
]
```

## Widget Usage

### ConstrainedSpinBox

Enhanced integer parameter widget with constraint validation:

```python
from components.widgets.enhanced_parameter_widgets import ConstrainedSpinBox

# Create widget with parameter name for automatic constraint loading
kernel_spinbox = ConstrainedSpinBox(parameter_name="spike_gaussian_kernel")

# Widget automatically:
# - Sets min=1, max=51 from database
# - Enforces odd numbers only
# - Shows hint: "Must be odd number. Larger values = more smoothing. Start with 5."
# - Validates input and shows visual feedback
```

### ParameterHintWidget

Standalone hint display widget:

```python
from components.widgets.enhanced_parameter_widgets import ParameterHintWidget

# Create hint widget for any parameter
hint_widget = ParameterHintWidget(parameter_name="baseline_asls_lam")

# Displays:
# - Parameter description
# - Valid range information
# - Constraints (e.g., "Must Be Positive, Log Scale Preferred")
# - User-friendly hints
```

## Constraint Validation

### Real-time Validation

Widgets automatically validate parameter values and provide feedback:

- ✅ **Valid values**: Green background, checkmark in hint
- ❌ **Invalid values**: Red background, error message in hint
- ⚠️ **Suboptimal values**: Warning styling with guidance

### Visual Feedback

```python
# Valid parameter
hint_label.setStyleSheet("""
    QLabel {
        color: #155724;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
""")

# Invalid parameter
hint_label.setStyleSheet("""
    QLabel {
        color: #dc3545;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
""")
```

## Parameter Hints System

### Context-Aware Hints

Hints adapt based on current value and use case:

```python
# For baseline_asls_p = 0.001
"Lower values (0.001-0.01) for strong fluorescence. ✅ Current value valid (Typical range: 0.001 - 0.05)"

# For spike_gaussian_kernel = 6
"Must be odd number. Larger values = more smoothing. ❌ Current value invalid: Must be an odd number"
```

### Use-Case Specific Suggestions

```python
constraints = ParameterConstraints()

# Get suggestion for biological samples
bio_lam = constraints.suggest_parameter_value("baseline_asls_lam", "biological")
# Returns: 1e8 (higher smoothness for fluorescence)

# Get suggestion for sensitive spike detection
sensitive_threshold = constraints.suggest_parameter_value("spike_gaussian_threshold", "sensitive_spike_detection")
# Returns: 2.0 (lower threshold for more sensitivity)
```

## Implementation Details

### Constraint Database Initialization

```python
class ParameterConstraints:
    def __init__(self):
        self.constraints = self._build_constraint_database()
    
    def _build_constraint_database(self) -> Dict[str, Dict[str, Any]]:
        return {
            "parameter_name": {
                "type": "float",
                "min": 1.0,
                "max": 10.0,
                "default": 3.0,
                "description": "Parameter description",
                "constraints": ["must_be_positive"],
                "hint": "User-friendly guidance"
            }
        }
```

### Widget Constraint Application

```python
def _apply_constraints(self):
    if not self.constraints or not self.parameter_name:
        return
    
    constraint_info = self.constraints.get_constraint_info(self.parameter_name)
    
    # Set ranges from database
    if "min" in constraint_info:
        self._minimum = constraint_info["min"]
    if "max" in constraint_info:
        self._maximum = constraint_info["max"]
    
    # Apply default value
    if "default" in constraint_info:
        self.set_value(constraint_info["default"])
```

## Benefits

### For Users
- **Guided Parameter Selection**: Clear hints prevent invalid configurations
- **Educational**: Learn parameter effects and constraints
- **Error Prevention**: Catch invalid values before processing
- **Confidence**: Know parameters are in valid ranges

### For Developers
- **Centralized Constraints**: All parameter rules in one place
- **Consistent Validation**: Same rules across all widgets
- **Easy Maintenance**: Update constraints without changing UI code
- **Extensible**: Easy to add new parameters and constraints

## Future Enhancements

### Planned Features
1. **Dynamic Constraints**: Parameters that depend on other parameters
2. **Advanced Validation**: Cross-parameter validation rules
3. **Context Sensitivity**: Different constraints for different data types
4. **User Preferences**: Remember user's preferred parameter sets
5. **Expert/Beginner Modes**: Different constraint levels for different users

### Integration Opportunities
1. **Preprocessing Pipeline**: Auto-validation before processing
2. **Configuration Saving**: Store validated parameter sets
3. **Template System**: Pre-configured parameter sets for common tasks
4. **Help System**: Context-sensitive help based on current parameters

This constraint system significantly improves the user experience by providing intelligent guidance and preventing common parameter configuration errors in Raman spectroscopy preprocessing workflows.