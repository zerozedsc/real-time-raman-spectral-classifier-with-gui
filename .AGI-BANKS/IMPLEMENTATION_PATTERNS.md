# Implementation Patterns and Best Practices

## Code Architecture Patterns

### 0. Dynamic Section Title Pattern (October 10, 2025) 🆕
Dynamic updating of section titles to reflect current context/state.

```python
# Store reference to title label during UI setup
self.params_title_label = QLabel(LOCALIZE("PREPROCESS.parameters_title"))
self.params_title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
params_title_layout.addWidget(self.params_title_label)

# Update title dynamically when state changes
def _show_parameter_widget(self, step: PipelineStep):
    # ... create parameter widget ...
    
    # Update title with context
    category_display = step.category.replace('_', ' ').title()
    self.params_title_label.setText(
        f"{LOCALIZE('PREPROCESS.parameters_title')} - {category_display}: {step.method}"
    )

# Reset title when clearing
def _clear_parameter_widget(self):
    # ... clear widget ...
    self.params_title_label.setText(LOCALIZE("PREPROCESS.parameters_title"))
```

### 1. Pipeline Import/Export Pattern (October 10, 2025) 🆕
Complete workflow for saving and loading complex configuration data.

```python
# Export Pattern
def export_pipeline(self):
    # 1. Validate data exists
    if not self.pipeline_steps:
        self.showNotification.emit("No pipeline to export", "warning")
        return
    
    # 2. Show user-friendly dialog
    dialog = QDialog(self)
    name_edit = QLineEdit()
    desc_edit = QTextEdit()
    # ... build dialog UI ...
    
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return
    
    # 3. Get project directory
    project_name = PROJECT_MANAGER.current_project_data.get("projectName", "").replace(' ', '_').lower()
    project_root = os.path.join(PROJECT_MANAGER.projects_dir, project_name)
    pipelines_dir = os.path.join(project_root, "pipelines")
    os.makedirs(pipelines_dir, exist_ok=True)
    
    # 4. Build data structure
    pipeline_data = {
        "name": pipeline_name,
        "description": pipeline_description,
        "created_date": datetime.datetime.now().isoformat(),
        "step_count": len(self.pipeline_steps),
        "steps": []
    }
    
    for step in self.pipeline_steps:
        pipeline_data["steps"].append({
            "category": step.category,
            "method": step.method,
            "params": step.params,
            "enabled": step.enabled
        })
    
    # 5. Save to file
    filename = f"{pipeline_name.replace(' ', '_').lower()}.json"
    filepath = os.path.join(pipelines_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(pipeline_data, f, indent=2, ensure_ascii=False)
    
    # 6. Provide feedback
    self.showNotification.emit(f"Pipeline '{pipeline_name}' exported successfully", "success")

# Import Pattern
def import_pipeline(self):
    # 1. Load saved pipelines
    pipelines_dir = os.path.join(project_root, "pipelines")
    saved_pipelines = []
    
    for filename in os.listdir(pipelines_dir):
        if filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                pipeline_data = json.load(f)
                saved_pipelines.append({
                    'name': pipeline_data.get('name'),
                    'filepath': filepath,
                    'data': pipeline_data
                })
    
    # 2. Show selection dialog with rich preview
    dialog = QDialog(self)
    pipeline_list = QListWidget()
    
    for pipeline in saved_pipelines:
        item = QListWidgetItem()
        # Create custom widget with name, step count, description
        widget = create_pipeline_preview_widget(pipeline)
        item.setData(Qt.ItemDataRole.UserRole, pipeline)
        pipeline_list.addItem(item)
        pipeline_list.setItemWidget(item, widget)
    
    # 3. Confirm replacement if needed
    if self.pipeline_steps:
        confirm = QMessageBox.question(self, "Replace Pipeline?", 
                                      f"Replace {len(self.pipeline_steps)} steps?")
        if confirm != QMessageBox.StandardButton.Yes:
            return
    
    # 4. Load selected pipeline
    selected_pipeline = pipeline_list.currentItem().data(Qt.ItemDataRole.UserRole)
    self._load_pipeline_from_data(selected_pipeline['data'])
    
    self.showNotification.emit(f"Pipeline imported successfully", "success")

# External import support
def _import_external_pipeline(self, parent_dialog):
    from PySide6.QtWidgets import QFileDialog
    
    filepath, _ = QFileDialog.getOpenFileName(
        parent_dialog, "Select Pipeline File", "", "JSON Files (*.json)"
    )
    
    if filepath:
        with open(filepath, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)
        self._load_pipeline_from_data(pipeline_data)
        parent_dialog.accept()
```

### 2. Gray Border Selection Pattern (October 10, 2025) 🆕
Subtle selection indicator that preserves item's original appearance.

```python
def _update_appearance(self):
    """Update visual appearance with selection support."""
    # Determine background based on step state
    if not self.step.enabled:
        bg_color = "#f8f9fa"  # Disabled
    elif self.step.is_existing:
        bg_color = "#e3f2fd"  # Existing
    else:
        bg_color = "white"  # New
    
    # Apply normal styling
    self.setStyleSheet(f"background-color: {bg_color}; border: 1px solid #dee2e6;")
    
    # Override with selection if selected (gray border, keep background)
    if self.is_selected:
        selected_style = f"""
            QWidget {{
                background-color: {bg_color};
                border: 2px solid #6c757d;  /* Gray selection border */
                border-radius: 6px;
            }}
        """
        self.setStyleSheet(selected_style)
```

### 3. Title Bar Action Buttons Pattern (October 7, 2025)
Compact action buttons integrated into section title bars for space-efficient UI design.

```python
# Create title widget with action buttons in title bar
title_widget = QWidget()
title_layout = QHBoxLayout(title_widget)
title_layout.setContentsMargins(0, 0, 0, 0)
title_layout.setSpacing(8)

title_label = QLabel(LOCALIZE("SECTION.title"))
title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
title_layout.addWidget(title_label)

# Add hint button if needed
hint_btn = QPushButton("?")
hint_btn.setFixedSize(20, 20)
# ... hint button styling ...
title_layout.addWidget(hint_btn)

title_layout.addStretch()  # Push action buttons to the right

# Compact action buttons (24px)
refresh_btn = QPushButton()
refresh_btn.setObjectName("titleBarButton")
refresh_btn.setIcon(load_svg_icon(get_icon_path("reload"), "#0078d4", QSize(14, 14)))
refresh_btn.setFixedSize(24, 24)
refresh_btn.setStyleSheet("""
    QPushButton#titleBarButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 3px;
        padding: 2px;
    }
    QPushButton#titleBarButton:hover {
        background-color: #e7f3ff;
        border-color: #0078d4;
    }
""")
title_layout.addWidget(refresh_btn)
```

### 1. Pipeline Step Selection Visual Feedback Pattern (October 7, 2025) 🆕
Implementing visual selection state for custom widgets in list views.

```python
class PipelineStepWidget(QWidget):
    def __init__(self, step, step_index, parent=None):
        super().__init__(parent)
        self.is_selected = False  # Track selection state
        self._setup_ui()
        
    def set_selected(self, selected: bool):
        """Set selection state and update appearance."""
        self.is_selected = selected
        self._update_appearance()
        
    def _update_appearance(self):
        """Update visual appearance with selection override."""
        # ... normal state styling ...
        
        # Override with selection styling if selected
        if self.is_selected:
            selected_style = """
                QWidget {
                    background-color: #d4e6f7;
                    border: 2px solid #0078d4;
                    border-radius: 6px;
                }
            """
            self.setStyleSheet(selected_style)

# In parent widget's selection handler
def on_item_selected(self, current, previous):
    # Update visual selection state for all widgets
    for i in range(self.list_widget.count()):
        item = self.list_widget.item(i)
        widget = self.list_widget.itemWidget(item)
        if widget and hasattr(widget, 'set_selected'):
            widget.set_selected(item == current)
```

### 2. Choice Parameter Type Conversion Pattern (October 7, 2025) 🆕
Proper handling of choice parameters with type preservation in dynamic UI generation.

```python
class DynamicParameterWidget(QWidget):
    def _create_parameter_widget(self, param_name: str, info: Dict, default_value: Any):
        if param_type == "choice":
            widget = QComboBox()
            choices = info.get("choices", [])
            
            # Create mapping to preserve original types
            choice_mapping = {}
            for choice in choices:
                str_choice = str(choice)
                widget.addItem(str_choice)
                choice_mapping[str_choice] = choice
            
            widget.choice_mapping = choice_mapping  # Store for extraction
            
    def get_parameters(self) -> Dict[str, Any]:
        # Extract with proper type conversion
        if param_type == "choice":
            current_text = widget.currentText()
            if hasattr(widget, 'choice_mapping'):
                params[param_name] = widget.choice_mapping[current_text]
            else:
                # Fallback type conversion
                try:
                    params[param_name] = int(current_text)
                except ValueError:
                    params[param_name] = current_text
```

### 3. Robust Signal Handling with Dynamic Index Resolution (October 7, 2025) 🆕
Error-safe signal handling for widgets that may have stale indices.

```python
def on_step_toggled(self, step_index: int, enabled: bool):
    """Handle step toggle with robust index validation."""
    # Find actual index by searching for sender widget
    actual_step_index = None
    sender_widget = self.sender()
    
    for i in range(self.list_widget.count()):
        widget = self.list_widget.itemWidget(self.list_widget.item(i))
        if widget == sender_widget:
            actual_step_index = i
            break
    
    # Use provided index as fallback
    if actual_step_index is None:
        actual_step_index = step_index
    
    # Validate bounds
    if not (0 <= actual_step_index < len(self.data_list)):
        create_logs("error", "Invalid index", f"Index {actual_step_index} out of bounds")
        return
    
    # Proceed with safe index
    self.data_list[actual_step_index].enabled = enabled
```

### 4. Hint Button in Title Pattern (October 6 Evening, 2025) 🆕
Space-efficient hint/info button integrated into section titles, combining multiple hints in one compact button.

```python
# Create custom title widget with hint button
title_widget = QWidget()
title_layout = QHBoxLayout(title_widget)
title_layout.setContentsMargins(0, 0, 0, 0)
title_layout.setSpacing(8)

# Title label
title_label = QLabel(LOCALIZE("SECTION.title"))
title_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #2c3e50;")
title_layout.addWidget(title_label)

# Hint button with ? icon
hint_btn = QPushButton("?")
hint_btn.setObjectName("hintButton")
hint_btn.setFixedSize(20, 20)
hint_btn.setToolTip(
    LOCALIZE("SECTION.hint_1") + "\\n\\n" +
    LOCALIZE("SECTION.hint_2")  # Combine multiple hints
)
hint_btn.setCursor(Qt.PointingHandCursor)
hint_btn.setStyleSheet("""
    QPushButton#hintButton {
        background-color: #e7f3ff;
        color: #0078d4;
        border: 1px solid #90caf9;
        border-radius: 10px;
        font-weight: bold;
        font-size: 11px;
        padding: 0px;
    }
    QPushButton#hintButton:hover {
        background-color: #0078d4;
        color: white;
        border-color: #0078d4;
    }
""")
title_layout.addWidget(hint_btn)
title_layout.addStretch()

# Use custom title in QGroupBox
group_box = QGroupBox()
group_box.setTitle("")  # Empty title since using custom widget
layout = QVBoxLayout(group_box)
layout.addWidget(title_widget)
```

**Benefits**:
- Saves vertical space (no separate info row needed)
- Consolidates multiple hints in one button
- Professional appearance (? icon vs emoji)
- Clear visual hierarchy in title bar

---

### 1. Icon-Only Button Pattern (October 6, 2025) 🆕
Space-efficient button design using SVG icons with hover tooltips, eliminating text labels for compact UI.

```python
# Import required functions
from utils import load_svg_icon
from components.widgets.icons import get_icon_path

# Create icon-only button with hover tooltip
button = QPushButton()
button.setObjectName("iconOnlyButton")

# Load and set icon (with color customization)
icon = load_svg_icon(get_icon_path("icon_name"), "#0078d4", QSize(18, 18))
button.setIcon(icon)
button.setIconSize(QSize(18, 18))
button.setFixedSize(36, 36)

# Add tooltip for text (shows on hover)
button.setToolTip(LOCALIZE("PREPROCESS.button_tooltip"))
button.setCursor(Qt.PointingHandCursor)

# Apply inline or stylesheet-based styling
button.setStyleSheet("""
    QPushButton#iconOnlyButton {
        background-color: #ffffff;
        border: 2px solid #dee2e6;
        border-radius: 6px;
        padding: 6px;
    }
    QPushButton#iconOnlyButton:hover {
        background-color: #e7f3ff;
        border-color: #0078d4;
    }
    QPushButton#iconOnlyButton:pressed {
        background-color: #d0e7ff;
        border-color: #005a9e;
    }
""")

# Connect to action
button.clicked.connect(self.action_method)
```

**Variant: Professional SVG Icon Buttons (Replaces Emoji)**
```python
# Plus button with SVG icon (instead of ➕ emoji)
add_btn = QPushButton()
add_btn.setObjectName("addStepButton")
plus_icon = load_svg_icon(get_icon_path("plus"), "white", QSize(24, 24))
add_btn.setIcon(plus_icon)
add_btn.setIconSize(QSize(24, 24))
add_btn.setFixedSize(60, 50)

# Trash button with SVG icon (instead of 🗑️ emoji)
remove_btn = QPushButton()
remove_btn.setObjectName("compactButton")
trash_icon = load_svg_icon(get_icon_path("trash_bin"), "#dc3545", QSize(14, 14))
remove_btn.setIcon(trash_icon)
remove_btn.setIconSize(QSize(14, 14))
remove_btn.setFixedHeight(28)
```

**Available SVG Icons** (from `components/widgets/icons.py`):
- `plus` → plus.svg (add/create actions)
- `minus` → minus.svg (decrease/remove actions)
- `trash_bin` → trash-bin.svg (delete actions)
- `trash` → trash-xmark.svg (clear/reset actions)
- `reload` → reload.svg (refresh actions)
- `export` → export-button.svg (export actions)
- `eye_open` → eye-open.svg (show/visibility)
- `eye_close` → eye-close.svg (hide actions)
- `chevron_down` → chevron-down.svg (dropdown indicators)

**Variant: Colored Icon Button (e.g., Green for Export)**
```python
export_btn = QPushButton()
export_btn.setObjectName("iconOnlyButtonGreen")
export_icon = load_svg_icon(get_icon_path("export"), "#2e7d32", QSize(18, 18))
export_btn.setIcon(export_icon)
export_btn.setIconSize(QSize(18, 18))
export_btn.setFixedSize(36, 36)
export_btn.setToolTip(LOCALIZE("PREPROCESS.export_button_tooltip"))
export_btn.setCursor(Qt.PointingHandCursor)
export_btn.setStyleSheet("""
    QPushButton#iconOnlyButtonGreen {
        background-color: #4caf50;
        border: 2px solid #4caf50;
        border-radius: 6px;
        padding: 6px;
    }
    QPushButton#iconOnlyButtonGreen:hover {
        background-color: #45a049;
        border-color: #45a049;
    }
    QPushButton#iconOnlyButtonGreen:pressed {
        background-color: #3d8b40;
        border-color: #3d8b40;
    }
    QPushButton#iconOnlyButtonGreen:disabled {
        background-color: #a5d6a7;
        border-color: #a5d6a7;
    }
""")
export_btn.clicked.connect(self.export_dataset)
```

**Variant: Compact Control Buttons (Emoji Icons)**
```python
# For pipeline controls (Remove, Clear, Toggle)
remove_btn = QPushButton("🗑️")
remove_btn.setObjectName("compactButton")
remove_btn.setFixedHeight(28)
remove_btn.setStyleSheet("""
    QPushButton#compactButton {
        background-color: #f8f9fa;
        color: #495057;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 4px 10px;
        font-weight: 500;
        font-size: 14px;
    }
    QPushButton#compactButton:hover {
        background-color: #e9ecef;
        border-color: #adb5bd;
    }
    QPushButton#compactButton:pressed {
        background-color: #dee2e6;
    }
""")
remove_btn.setToolTip(LOCALIZE("PREPROCESS.remove_step"))
remove_btn.setCursor(Qt.PointingHandCursor)
remove_btn.clicked.connect(self.remove_pipeline_step)
```

**Key Features:**
- Saves 60-80% horizontal space compared to text buttons
- Text visible only on hover via tooltip
- Consistent size (36x36px for icon buttons, 28px height for compact)
- Clear visual feedback (hover and pressed states)
- Accessible through tooltips
- Pointer cursor indicates interactivity

**Icon Loading Best Practices:**
- Use centralized `get_icon_path()` for path management
- Use `load_svg_icon()` for color customization
- Common icon sizes: 16x16, 18x18, 20x20 pixels
- Button sizes: 32x32, 36x36, 40x40 pixels
- Match icon color to button theme

**Color Schemes:**
- **Primary (Blue)**: #0078d4 icon, white/light blue background
- **Success (Green)**: #2e7d32 icon, green background (#4caf50)
- **Neutral (Gray)**: #495057 icon/text, light gray background (#f8f9fa)
- **Hover states**: Slightly darker/lighter shade of base color

**Use Cases:**
- Toolbar buttons with limited space
- Action buttons in compact layouts
- Control buttons (add, remove, clear, toggle)
- Export, import, reload, refresh actions
- Navigation buttons

**Best Practices:**
- Always provide tooltip for accessibility
- Use recognizable icons (standard icons or emoji)
- Set cursor to pointer for click indication
- Provide clear hover/pressed states
- Group related icon buttons together
- Maintain consistent sizing within a group
- Test icon visibility on different backgrounds

**Available Icons** (from components/widgets/icons.py):
- `reload`: Refresh/reload icon
- `export`: Export/download icon
- `trash_bin`: Delete/remove icon
- `plus`: Add/create icon
- `minus`: Remove/subtract icon
- `eye_open`: Show/visible icon
- `eye_close`: Hide/invisible icon
- `focus_horizontal`: Focus/zoom icon
- `new_project`: New file/project icon
- `load_project`: Open file/project icon

---

### 1. Hover Tooltip System Pattern (October 2025)
Space-efficient information display using interactive icons with rich HTML tooltips.

```python
# Create info icons row with hover tooltips
info_row = QHBoxLayout()
info_row.setSpacing(12)
info_row.setContentsMargins(4, 4, 4, 4)

# Multi-selection hint icon
multi_select_icon = QLabel("ℹ️")
multi_select_icon.setStyleSheet("""
    QLabel {
        color: #0078d4;
        font-size: 14px;
        padding: 2px;
        border-radius: 3px;
    }
    QLabel:hover {
        background-color: #e7f3ff;
    }
""")
multi_select_icon.setToolTip(LOCALIZE("PREPROCESS.multi_select_hint"))
multi_select_icon.setCursor(Qt.PointingHandCursor)
info_row.addWidget(multi_select_icon)

# Multi-dataset processing hint icon
multi_dataset_icon = QLabel("💡")
multi_dataset_icon.setStyleSheet("""
    QLabel {
        color: #0078d4;
        font-size: 14px;
        padding: 2px;
        border-radius: 3px;
    }
    QLabel:hover {
        background-color: #e7f3ff;
    }
""")
multi_dataset_icon.setToolTip(LOCALIZE("PREPROCESS.multi_dataset_hint"))
multi_dataset_icon.setCursor(Qt.PointingHandCursor)
info_row.addWidget(multi_dataset_icon)

info_row.addStretch()
layout.addLayout(info_row)
```

**Locale String Format (HTML-formatted):**
```json
{
  "multi_select_hint": "<b>Multi-Selection:</b><br>• Hold <b>Ctrl</b> (or <b>Cmd</b>) + Click to select multiple datasets<br>• Hold <b>Shift</b> + Click to select a range",
  "multi_dataset_hint": "<b>Multi-Dataset Processing:</b><br>When multiple datasets are selected, they will be combined and processed together as one output dataset. The preprocessing pipeline applies to all selected data simultaneously."
}
```

**Key Features:**
- Saves vertical space compared to always-visible labels
- Rich HTML formatting in tooltips (bold text, bullet points, line breaks)
- Interactive hover states with visual feedback
- Medical theme colors (#0078d4 blue, #e7f3ff light blue hover)
- Pointer cursor indicates interactivity
- Icon-based design for quick recognition (ℹ️ = info, 💡 = insight)

**Best Practices:**
- Use HTML formatting for complex tooltip content
- Provide hover state styling for visual feedback
- Use emoji icons for better visual communication
- Keep tooltip text concise but informative
- Group related info icons together in horizontal layout
- Maintain consistent styling across all tooltip icons

---

### 2. Unified Selection Handler Pattern (October 2025)
Cross-tab selection synchronization with single handler for multiple QListWidget instances.

```python
# In __init__ or creation method:
# Create three separate list widgets for different tabs
self.dataset_list_all = QListWidget()
self.dataset_list_raw = QListWidget()
self.dataset_list_preprocessed = QListWidget()

# Configure and connect all list widgets to same handler
for list_widget in [self.dataset_list_all, self.dataset_list_raw, 
                    self.dataset_list_preprocessed]:
    list_widget.setObjectName("datasetList")
    list_widget.setSelectionMode(QListWidget.ExtendedSelection)
    # Connect to unified handler
    list_widget.itemSelectionChanged.connect(self._on_dataset_selection_changed)

# Keep reference to active list for backward compatibility
self.dataset_list = self.dataset_list_all

# Connect tab change signal
self.dataset_tabs.currentChanged.connect(self._on_dataset_tab_changed)

def _on_dataset_tab_changed(self, index: int):
    """Update active dataset list reference when tab changes."""
    if index == 0:
        self.dataset_list = self.dataset_list_all
    elif index == 1:
        self.dataset_list = self.dataset_list_raw
    elif index == 2:
        self.dataset_list = self.dataset_list_preprocessed
    
    # Trigger selection event for newly active tab
    self._on_dataset_selection_changed()

def _on_dataset_selection_changed(self):
    """Handle dataset selection changes across all tabs."""
    selected_items = self.dataset_list.selectedItems()
    
    if not selected_items:
        # Clear visualization and state
        self.plot_widget.clear_plot()
        return
    
    # Load and display selected data
    all_dfs = []
    for item in selected_items:
        dataset_name = self._clean_dataset_name(item.text())
        if dataset_name in RAMAN_DATA:
            all_dfs.append(RAMAN_DATA[dataset_name])
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=1)
        self.original_data = combined_df
        # Update visualization
        self._schedule_preview_update()
```

**Key Features:**
- Single handler for all tab list widgets (DRY principle)
- Tab switching automatically triggers visualization update
- Backward compatible with existing `self.dataset_list` reference
- Consistent behavior across All/Raw/Preprocessed tabs
- Signal-based architecture for loose coupling

**Best Practices:**
- Connect all related widgets to same handler for consistency
- Update active reference on tab change for backward compatibility
- Trigger selection event when switching tabs
- Use `itemSelectionChanged` signal for immediate response
- Handle empty selection gracefully



def _on_dataset_tab_changed(self, index: int):
    """Update active dataset list reference when tab changes."""
    if index == 0:
        self.dataset_list = self.dataset_list_all
    elif index == 1:
        self.dataset_list = self.dataset_list_raw
    elif index == 2:
        self.dataset_list = self.dataset_list_preprocessed
    
    # Trigger selection event for newly active tab
    self._on_dataset_selection_changed()

def _on_dataset_selection_changed(self):
    """Handle dataset selection changes across all tabs."""
    selected_items = self.dataset_list.selectedItems()
    
    if not selected_items:
        # Clear visualization and state
        self.plot_widget.clear_plot()
        return
    
    # Load and display selected data
    all_dfs = []
    for item in selected_items:
        dataset_name = self._clean_dataset_name(item.text())
        if dataset_name in RAMAN_DATA:
            all_dfs.append(RAMAN_DATA[dataset_name])
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=1)
        self.original_data = combined_df
        # Update visualization
        self._schedule_preview_update()
```

**Key Features:**
- Single handler for all tab list widgets (DRY principle)
- Tab switching automatically triggers visualization update
- Backward compatible with existing `self.dataset_list` reference
- Consistent behavior across All/Raw/Preprocessed tabs
- Signal-based architecture for loose coupling

**Best Practices:**
- Connect all related widgets to same handler for consistency
- Update active reference on tab change for backward compatibility
- Trigger selection event when switching tabs
- Use `itemSelectionChanged` signal for immediate response
- Handle empty selection gracefully

---

### 3. Modern Card-Based UI Pattern (October 2025)
Clean, professional card-based design for metrics and controls.

```python
def _create_metric_item(self, icon: str, value: str, label: str) -> QFrame:
    """Create a modern metric display item with icon, value, and label."""
    metric = QFrame()
    metric.setObjectName("metricItem")
    metric_layout = QVBoxLayout(metric)
    metric_layout.setContentsMargins(16, 12, 16, 12)
    metric_layout.setSpacing(6)
    
    # Icon row (centered)
    icon_label = QLabel(icon)
    icon_label.setStyleSheet("font-size: 20px;")
    icon_label.setAlignment(Qt.AlignCenter)
    metric_layout.addWidget(icon_label)
    
    # Value (large and prominent, centered)
    value_label = QLabel(value)
    value_label.setObjectName("metricValue")
    value_label.setAlignment(Qt.AlignCenter)
    metric_layout.addWidget(value_label)
    
    # Label (small description, centered)
    label_widget = QLabel(label)
    label_widget.setObjectName("metricLabel")
    label_widget.setAlignment(Qt.AlignCenter)
    label_widget.setWordWrap(True)
    metric_layout.addWidget(label_widget)
    
    return metric

# In confirmation dialog header:
metrics_grid = QGridLayout()
metrics_grid.setSpacing(16)

input_metric = self._create_metric_item("📊", str(len(datasets)), 
                                        LOCALIZE("datasets_label"))
metrics_grid.addWidget(input_metric, 0, 0)

steps_metric = self._create_metric_item("⚙️", str(len(steps)), 
                                        LOCALIZE("steps_label"))
metrics_grid.addWidget(steps_metric, 0, 1)

output_metric = self._create_metric_item("💾", output_name, 
                                         LOCALIZE("output_label"))
metrics_grid.addWidget(output_metric, 0, 2)

# Stylesheet for metrics:
"""
#metricItem {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ffffff, stop:1 #fafcfe);
    border: 1px solid #e1e8ed;
    border-radius: 8px;
    min-width: 140px;
    max-width: 200px;
}

#metricItem:hover {
    border-color: #90caf9;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f8fbff, stop:1 #e3f2fd);
}

#metricValue {
    font-size: 24px;
    font-weight: 700;
    color: #0078d4;
    letter-spacing: -0.5px;
}

#metricLabel {
    font-size: 11px;
    font-weight: 500;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
"""
```

**Key Features:**
- Vertical layout: Icon → Value → Label
- Centered alignment for clean appearance
- Hover states with border and gradient changes
- Grid layout for responsive metric display
- Professional typography with size/weight hierarchy
- Medical theme colors and subtle gradients

**Best Practices:**
- Use QGridLayout for responsive metric arrangement
- Implement hover states for interactivity
- Maintain consistent spacing (padding 16/12, spacing 6)
- Use uppercase labels with letter spacing for professional look
- Limit card width for better readability (140-200px)
- Center-align all content within metric cards

---

### 4. Export with Metadata Pattern (October 2025)
Comprehensive data export with automatic metadata serialization and batch processing support.

```python
def export_dataset(self):
    """Export selected dataset(s) with optional metadata."""
    selected_items = self.dataset_list.selectedItems()
    dataset_names = [self._clean_dataset_name(item.text()) for item in selected_items]
    
    # Create export dialog with dynamic UI
    if len(dataset_names) > 1:
        # Multiple export mode: show count, hide filename field
        info_label = QLabel(LOCALIZE("PREPROCESS.export_multiple_info", count=len(dataset_names)))
        layout.addWidget(info_label)
    else:
        # Single export mode: show filename field
        filename_edit = QLineEdit()
        filename_edit.setText(dataset_names[0])
    
    # Load and pre-fill last location (session persistence)
    last_export_path = getattr(self, '_last_export_location', None)
    if last_export_path and os.path.exists(last_export_path):
        location_edit.setText(last_export_path)
    
    # Validation before export
    if not export_path:
        QMessageBox.warning(self, 
                          LOCALIZE("PREPROCESS.export_warning_title"),
                          LOCALIZE("PREPROCESS.export_no_location_warning"))
        return
    
    # Store location for next time
    self._last_export_location = export_path
    
    # Export with metadata
    for dataset_name in dataset_names:
        self._export_single_dataset(dataset_name, export_path, 
                                    dataset_name, export_format, 
                                    export_metadata=True)

def _export_metadata_json(self, metadata: Dict, export_path: str, 
                          filename: str, data_shape: tuple):
    """Export preprocessing metadata to JSON."""
    export_meta = {
        "export_info": {
            "export_date": datetime.now().isoformat(),
            "dataset_name": filename,
            "data_shape": {"rows": data_shape[0], "columns": data_shape[1]}
        },
        "preprocessing": {
            "is_preprocessed": metadata.get("is_preprocessed", False),
            "pipeline": metadata.get("preprocessing_pipeline", []),
            "source_datasets": metadata.get("source_datasets", [])
        },
        "spectral_info": {
            "num_spectra": metadata.get("num_spectra", data_shape[1]),
            "spectral_axis_start": metadata.get("spectral_axis", [None])[0],
            "spectral_axis_end": metadata.get("spectral_axis", [None])[-1]
        }
    }
    
    metadata_path = os.path.join(export_path, f"{filename}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(export_meta, f, indent=2, ensure_ascii=False)
```

**Key Features:**
- Dynamic UI based on single vs. multiple dataset selection
- Location persistence across export operations (session-level)
- Explicit validation with user-friendly warning dialogs
- Structured metadata JSON with export info, preprocessing, and spectral data
- Batch processing with individual error handling
- Comprehensive localization support

**Best Practices:**
- Separate helper methods for single dataset export and metadata export
- Validate all inputs before file operations
- Store last location using `getattr()` with default to handle first use
- Use ISO format for timestamps in metadata
- Include both human-readable and machine-parseable data in JSON
- Emit appropriate notifications: success, warning (partial), error

**Benefits:**
- Improved data traceability with metadata export
- Better UX with location persistence and validation
- Efficient batch processing for multiple datasets
- Robust error handling with graceful degradation
- Full internationalization support

---

### 2. Icon Management Pattern (Updated September 2025)
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