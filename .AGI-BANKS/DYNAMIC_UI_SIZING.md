# Dynamic UI Sizing Implementation

## Overview
This document describes the implementation of dynamic UI sizing to support internationalization and responsive design in the Raman spectroscopy application.

## Problem Statement
The application's preview toggle button had a fixed width (120px) that couldn't accommodate longer localized text, particularly in Japanese where " プレビュー ON" is longer than the English " Preview ON".

## Solution Architecture

### Core Implementation
Added a new method `_adjust_button_width_to_text()` that dynamically calculates button width based on actual text content:

```python
def _adjust_button_width_to_text(self):
    """Adjust button width dynamically based on text content."""
    from PySide6.QtGui import QFontMetrics
    
    # Get current text and font
    text = self.preview_toggle_btn.text()
    font = self.preview_toggle_btn.font()
    
    # Calculate text width using font metrics
    font_metrics = QFontMetrics(font)
    text_width = font_metrics.horizontalAdvance(text)
    
    # Account for UI elements
    icon_width = 16  # Icon size
    spacing = 8 if text.strip() else 0  # Spacing between icon and text
    padding = 16  # CSS padding (8px left + 8px right)
    border = 4   # CSS border (2px left + 2px right)
    
    total_width = text_width + icon_width + spacing + padding + border
    dynamic_width = max(80, total_width)  # Minimum 80px
    
    self.preview_toggle_btn.setFixedWidth(dynamic_width)
```

### Integration Points

#### Button Creation
Changed from fixed size to fixed height only:
```python
# Before: Fixed width and height
self.preview_toggle_btn.setFixedSize(120, 32)

# After: Fixed height, dynamic width
self.preview_toggle_btn.setFixedHeight(32)
```

#### Text Updates
Modified `_update_preview_button_state()` to trigger width adjustment:
```python
def _update_preview_button_state(self, enabled):
    if enabled:
        text = LOCALIZE("PREPROCESS.UI.preview_on")
        self.preview_toggle_btn.setText(text)
    else:
        text = LOCALIZE("PREPROCESS.UI.preview_off") 
        self.preview_toggle_btn.setText(text)
    
    # Trigger dynamic width calculation
    self._adjust_button_width_to_text()
```

## Localization Support

### English Locale (en.json)
```json
{
    "PREPROCESS": {
        "UI": {
            "preview_on": " Preview ON",
            "preview_off": " Preview OFF"
        }
    }
}
```

### Japanese Locale (ja.json)
```json
{
    "PREPROCESS": {
        "UI": {
            "preview_on": " プレビュー ON",
            "preview_off": " プレビュー OFF"
        }
    }
}
```

## Technical Benefits

1. **Language Agnostic**: Automatically adapts to any text length
2. **Font Aware**: Uses actual font metrics for accurate calculations
3. **Consistent Layout**: Maintains visual harmony with fixed height
4. **Minimum Width**: Prevents buttons from becoming too narrow
5. **Performance Optimized**: Only recalculates when text changes

## Implementation Pattern

This solution establishes a reusable pattern for dynamic UI sizing:

1. **Fixed Height, Dynamic Width**: Maintains consistent vertical alignment
2. **Font Metrics Calculation**: Accurate text measurement
3. **Element Accounting**: Includes icons, padding, borders in calculation
4. **Minimum Constraints**: Ensures usable minimum sizes
5. **Localization Integration**: Works seamlessly with translation system

## Usage Guidelines

For future UI elements requiring dynamic sizing:

1. Use `setFixedHeight()` instead of `setFixedSize()` 
2. Implement text width calculation using `QFontMetrics`
3. Account for all visual elements (icons, padding, borders)
4. Set appropriate minimum width constraints
5. Trigger recalculation when text content changes

## Files Modified

- `pages/preprocess_page.py`: Core implementation
- `assets/locales/en.json`: English UI labels  
- `assets/locales/ja.json`: Japanese UI labels
- `.AI-AGENT/RECENT_CHANGES.md`: Documentation updates
- `.AI-AGENT/IMPLEMENTATION_PATTERNS.md`: Pattern documentation
- `.AI-AGENT/PROJECT_OVERVIEW.md`: Architecture updates

## Testing Results

- ✅ English UI displays correctly with appropriate button width
- ✅ Japanese UI displays correctly with wider button for longer text  
- ✅ Text never truncated in either language
- ✅ Visual consistency maintained across language switches
- ✅ Minimum width prevents overly narrow buttons
- ✅ Application starts without errors in both locales

This implementation serves as a foundation for responsive, internationalization-friendly UI components throughout the application.