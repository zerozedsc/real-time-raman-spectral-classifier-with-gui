# Preprocess Page Refactoring Plan

> **Comprehensive code reorganization strategy for preprocess_page.py**  
> Created: October 6, 2025  
> Status: Planning Phase

## 📊 Current State Analysis

### File Statistics
- **Total Lines**: ~3060 lines
- **Total Methods**: 75 methods
- **Inline Styles**: 40+ inline `setStyleSheet()` calls
- **Class**: Single `PreprocessPage` class handling all functionality

### Current Structure Issues

1. **Monolithic Design**
   - Single 3060-line file handling all preprocessing page logic
   - Too many responsibilities in one class
   - Difficult to navigate and maintain

2. **Inline Styling**
   - 40+ inline style definitions scattered throughout the code
   - Style duplication and inconsistency
   - Hard to maintain consistent theme

3. **Mixed Concerns**
   - UI creation, business logic, data handling all in one file
   - Parameter widget management mixed with pipeline logic
   - Export functionality embedded in page class

4. **Limited Reusability**
   - UI components tightly coupled to page logic
   - Difficult to test individual components
   - Hard to reuse patterns across pages

## 🎯 Refactoring Goals

### Primary Objectives
1. **Reduce File Size**: Target 500-800 lines for main page file
2. **Extract Styles**: Move all inline styles to `stylesheets.py`
3. **Modularize Components**: Extract reusable UI components
4. **Separate Concerns**: Separate UI, logic, and data layers
5. **Improve Testability**: Make components independently testable

### Code Quality Targets
- Single Responsibility Principle for each module
- Maximum 50 lines per method
- Maximum 300 lines per file (except main page coordinator)
- Clear separation of UI and business logic
- Reusable component patterns

## 📁 Proposed File Structure

### New Structure
```
pages/
├── preprocess_page.py                # Main coordinator (500-800 lines)
│   └── PreprocessPage class          # Orchestrates all components
│
└── preprocess_page_utils/
    ├── __init__.py                   # Package exports
    ├── __utils__.py                  # Existing utilities
    ├── thread.py                     # Existing processing thread
    ├── pipeline.py                   # Existing pipeline logic
    ├── widgets.py                    # Existing dynamic widgets
    │
    ├── ui_components.py              # NEW: UI creation methods
    │   ├── create_input_datasets_section()
    │   ├── create_pipeline_building_section()
    │   ├── create_output_configuration_section()
    │   ├── create_parameters_section()
    │   └── create_visualization_section()
    │
    ├── dataset_manager.py            # NEW: Dataset operations
    │   ├── load_project_data()
    │   ├── export_dataset()
    │   ├── _export_single_dataset()
    │   ├── _export_metadata_json()
    │   └── _clean_dataset_name()
    │
    ├── pipeline_manager.py           # NEW: Pipeline operations
    │   ├── add_pipeline_step()
    │   ├── remove_pipeline_step()
    │   ├── clear_pipeline()
    │   ├── toggle_all_existing_steps()
    │   ├── _save_to_global_memory()
    │   ├── _restore_global_pipeline_memory()
    │   └── _rebuild_pipeline_ui()
    │
    ├── preview_manager.py            # NEW: Preview operations
    │   ├── _toggle_preview_mode()
    │   ├── _manual_refresh_preview()
    │   ├── _schedule_preview_update()
    │   ├── _update_preview()
    │   ├── _show_original_data()
    │   ├── _show_preview_data()
    │   └── _should_auto_focus()
    │
    ├── parameter_manager.py          # NEW: Parameter operations
    │   ├── on_pipeline_step_selected()
    │   ├── _show_parameter_widget()
    │   ├── _clear_parameter_widget()
    │   ├── _update_step_parameters()
    │   ├── _connect_parameter_signals()
    │   └── _update_current_step_parameters()
    │
    ├── history_manager.py            # NEW: History operations
    │   ├── _show_preprocessing_history()
    │   ├── _clear_preprocessing_history()
    │   ├── _clear_preprocessing_history_display_only()
    │   ├── _load_preprocessing_pipeline()
    │   ├── _pipelines_differ()
    │   └── _show_pipeline_transfer_dialog()
    │
    └── styles.py                     # NEW: Style definitions
        ├── ICON_BUTTON_STYLES
        ├── COMPACT_BUTTON_STYLES
        ├── COMBOBOX_STYLES
        ├── PIPELINE_LIST_STYLES
        ├── DATASET_TABS_STYLES
        ├── TOOLTIP_ICON_STYLES
        ├── PREVIEW_BUTTON_STYLES
        └── DIALOG_STYLES

configs/style/
└── stylesheets.py                    # Move all inline styles here
    └── PREPROCESS_PAGE_STYLES = {
            'icon_only_button': '...',
            'icon_only_button_green': '...',
            'compact_button': '...',
            'add_step_button': '...',
            'category_combo': '...',
            'method_combo': '...',
            'pipeline_list': '...',
            'dataset_tabs': '...',
            'tooltip_icon': '...',
            'preview_button_enabled': '...',
            'preview_button_disabled': '...',
            'export_dialog': '...',
            ...
        }
```

## 🔄 Method Distribution

### PreprocessPage (Main Coordinator) - ~600 lines
```python
class PreprocessPage(QWidget):
    # Core setup (100 lines)
    - __init__()
    - _setup_ui()
    - _connect_signals()
    - _create_left_panel()
    - _create_right_panel()
    
    # Delegation methods (100 lines)
    - load_project_data() → dataset_manager
    - export_dataset() → dataset_manager
    - add_pipeline_step() → pipeline_manager
    - remove_pipeline_step() → pipeline_manager
    - clear_pipeline() → pipeline_manager
    - toggle_all_existing_steps() → pipeline_manager
    - on_pipeline_step_selected() → parameter_manager
    - run_preprocessing() → Keep in main (threading coordination)
    - cancel_preprocessing() → Keep in main (threading coordination)
    
    # Event handlers (100 lines)
    - _on_dataset_tab_changed()
    - _on_dataset_selection_changed()
    - _on_pipeline_reordered()
    - on_step_toggled()
    - on_progress_updated()
    - on_status_updated()
    - on_processing_completed()
    - on_processing_error()
    - on_step_completed()
    - on_step_failed()
    
    # UI state management (100 lines)
    - _start_processing_ui()
    - _reset_ui_state()
    - _update_preview_button_state()
    - update_method_combo()
    - _set_default_output_name()
    - _clear_default_output_name()
    
    # Threading callbacks (100 lines)
    - _on_thread_finished()
    - _apply_full_pipeline()
    - _apply_preview_pipeline()
```

### ui_components.py - ~400 lines
```python
def create_input_datasets_section(parent) -> QGroupBox:
    """Create input datasets selection group with tabs."""
    # Icon-only buttons, tabs, tooltips
    # ~80 lines

def create_pipeline_building_section(parent) -> QGroupBox:
    """Create pipeline construction section."""
    # Compact layout, category/method selection, list view
    # ~120 lines

def create_output_configuration_section(parent) -> QGroupBox:
    """Create output configuration section."""
    # Output name, progress bar, run/cancel buttons
    # ~60 lines

def create_parameters_section(parent) -> QGroupBox:
    """Create parameters section."""
    # Scrollable parameter area
    # ~60 lines

def create_visualization_section(parent) -> QGroupBox:
    """Create visualization section with preview controls."""
    # Preview toggle, manual refresh, focus, plot widget
    # ~80 lines
```

### dataset_manager.py - ~300 lines
```python
class DatasetManager:
    """Manages dataset loading, selection, and export operations."""
    
    def __init__(self, parent_page):
        self.parent = parent_page
        self._last_export_location = None
    
    def load_project_data(self):
        """Load and populate dataset lists."""
        # ~100 lines
    
    def export_dataset(self):
        """Export dialog and orchestration."""
        # ~100 lines
    
    def _export_single_dataset(self, ...):
        """Export single dataset to file."""
        # ~50 lines
    
    def _export_metadata_json(self, ...):
        """Export metadata JSON."""
        # ~30 lines
    
    def _clean_dataset_name(self, text: str) -> str:
        """Clean dataset name from list item."""
        # ~10 lines
```

### pipeline_manager.py - ~250 lines
```python
class PipelineManager:
    """Manages pipeline construction and state."""
    
    def __init__(self, parent_page):
        self.parent = parent_page
        self.steps = []
        self.global_memory = []
    
    def add_pipeline_step(self):
        """Add step to pipeline."""
        # ~40 lines
    
    def remove_pipeline_step(self):
        """Remove selected step."""
        # ~20 lines
    
    def clear_pipeline(self):
        """Clear all steps."""
        # ~20 lines
    
    def toggle_all_existing_steps(self):
        """Toggle enable/disable all steps."""
        # ~30 lines
    
    def _save_to_global_memory(self):
        """Save current pipeline to memory."""
        # ~20 lines
    
    def _restore_global_pipeline_memory(self):
        """Restore pipeline from memory."""
        # ~30 lines
    
    def _rebuild_pipeline_ui(self):
        """Rebuild pipeline list UI."""
        # ~40 lines
```

### preview_manager.py - ~300 lines
```python
class PreviewManager:
    """Manages real-time preview functionality."""
    
    def __init__(self, parent_page):
        self.parent = parent_page
        self.preview_enabled = True
        self.preview_timer = QTimer()
        self.preview_cache = {}
    
    def toggle_preview_mode(self, enabled: bool):
        """Toggle preview on/off."""
        # ~30 lines
    
    def manual_refresh_preview(self):
        """Manually refresh preview."""
        # ~20 lines
    
    def schedule_preview_update(self, delay_ms: int = 300):
        """Schedule debounced preview update."""
        # ~15 lines
    
    def update_preview(self):
        """Update preview with current pipeline."""
        # ~80 lines
    
    def show_original_data(self):
        """Show original unprocessed data."""
        # ~30 lines
    
    def show_preview_data(self, processed_data):
        """Show processed preview data."""
        # ~40 lines
    
    def should_auto_focus(self) -> bool:
        """Determine if auto-focus should be applied."""
        # ~40 lines
    
    def extract_crop_bounds(self):
        """Extract crop boundaries from pipeline."""
        # ~30 lines
```

### parameter_manager.py - ~200 lines
```python
class ParameterManager:
    """Manages parameter widgets and updates."""
    
    def __init__(self, parent_page):
        self.parent = parent_page
        self.current_step_widget = None
    
    def on_pipeline_step_selected(self, current, previous):
        """Handle pipeline step selection."""
        # ~30 lines
    
    def show_parameter_widget(self, step):
        """Display parameter widget for step."""
        # ~60 lines
    
    def clear_parameter_widget(self):
        """Clear parameter area."""
        # ~20 lines
    
    def update_step_parameters(self):
        """Update step with current parameters."""
        # ~30 lines
    
    def connect_parameter_signals(self, widget):
        """Connect parameter change signals."""
        # ~30 lines
    
    def update_current_step_parameters(self):
        """Update current step parameters."""
        # ~20 lines
```

### history_manager.py - ~250 lines
```python
class HistoryManager:
    """Manages preprocessing history display and loading."""
    
    def __init__(self, parent_page):
        self.parent = parent_page
    
    def show_preprocessing_history(self, metadata: Dict):
        """Display preprocessing history."""
        # ~80 lines
    
    def clear_preprocessing_history(self):
        """Clear history display and reset pipeline."""
        # ~20 lines
    
    def clear_preprocessing_history_display_only(self):
        """Clear only the visual display."""
        # ~20 lines
    
    def load_preprocessing_pipeline(self, pipeline_data: List[Dict], ...):
        """Load pipeline from history data."""
        # ~70 lines
    
    def pipelines_differ(self, current_steps, target_pipeline) -> bool:
        """Check if pipelines differ."""
        # ~30 lines
    
    def show_pipeline_transfer_dialog(self, dataset_name: str, ...):
        """Show dialog for pipeline transfer."""
        # ~40 lines
```

### styles.py - ~800 lines
```python
"""Stylesheet definitions for preprocess page components."""

# Icon button styles
ICON_BUTTON_STYLES = {
    'icon_only_button': """...""",  # ~30 lines
    'icon_only_button_green': """...""",  # ~30 lines
}

# Compact button styles
COMPACT_BUTTON_STYLES = {
    'compact_button': """...""",  # ~30 lines
    'add_step_button': """...""",  # ~30 lines
}

# ComboBox styles
COMBOBOX_STYLES = {
    'category_combo': """...""",  # ~30 lines
    'method_combo': """...""",  # ~30 lines
}

# List and tab styles
LIST_TAB_STYLES = {
    'pipeline_list': """...""",  # ~40 lines
    'dataset_tabs': """...""",  # ~50 lines
}

# Tooltip and hint styles
TOOLTIP_STYLES = {
    'tooltip_icon': """...""",  # ~20 lines
    'multi_select_hint': """...""",  # ~20 lines
    'multi_dataset_hint': """...""",  # ~20 lines
}

# Preview control styles
PREVIEW_STYLES = {
    'preview_button_enabled': """...""",  # ~40 lines
    'preview_button_disabled': """...""",  # ~40 lines
    'manual_refresh_button': """...""",  # ~20 lines
    'manual_focus_button': """...""",  # ~20 lines
}

# Dialog styles
DIALOG_STYLES = {
    'export_dialog': """...""",  # ~100 lines
    'confirmation_dialog': """...""",  # ~80 lines
    'history_dialog': """...""",  # ~60 lines
}

# Collect all styles for export
ALL_PREPROCESS_STYLES = {
    **ICON_BUTTON_STYLES,
    **COMPACT_BUTTON_STYLES,
    **COMBOBOX_STYLES,
    **LIST_TAB_STYLES,
    **TOOLTIP_STYLES,
    **PREVIEW_STYLES,
    **DIALOG_STYLES,
}
```

## 🔧 Migration Strategy

### Phase 1: Style Extraction (2-3 hours)
**Goal**: Move all inline styles to centralized location

1. **Create styles.py in preprocess_page_utils/**
   - Extract all inline `setStyleSheet()` calls
   - Group by component type
   - Create named style constants

2. **Update stylesheets.py**
   - Add PREPROCESS_PAGE_STYLES dictionary
   - Import from preprocess_page_utils/styles.py
   - Maintain backward compatibility

3. **Refactor preprocess_page.py**
   - Replace inline styles with references
   - Pattern: `widget.setStyleSheet(PREPROCESS_PAGE_STYLES['style_name'])`
   - Test all UI rendering

**Deliverables**:
- `preprocess_page_utils/styles.py` (800 lines)
- Updated `configs/style/stylesheets.py` (add imports)
- Reduced `preprocess_page.py` (remove ~400 lines of inline styles)

**Testing**:
- Visual regression testing
- Verify all components render correctly
- Check hover states and interactions

### Phase 2: UI Component Extraction (3-4 hours)
**Goal**: Extract UI creation methods to separate module

1. **Create ui_components.py**
   - Move `_create_input_datasets_group()`
   - Move `_create_pipeline_building_group()`
   - Move `_create_output_configuration_group()`
   - Move `_create_right_panel()` sections
   - Convert to standalone functions

2. **Update PreprocessPage**
   - Import UI creation functions
   - Replace method calls with function calls
   - Pass parent reference for signal connections

3. **Test UI Construction**
   - Verify all UI elements created correctly
   - Check signal connections work
   - Validate layout and styling

**Deliverables**:
- `preprocess_page_utils/ui_components.py` (400 lines)
- Updated `preprocess_page.py` (remove ~500 lines of UI creation)

**Testing**:
- Full UI smoke test
- Signal/slot connection verification
- Layout responsiveness check

### Phase 3: Manager Classes (4-5 hours)
**Goal**: Extract business logic to specialized manager classes

1. **Create dataset_manager.py**
   - Extract dataset loading logic
   - Extract export functionality
   - Create DatasetManager class

2. **Create pipeline_manager.py**
   - Extract pipeline operations
   - Create PipelineManager class
   - Move global memory logic

3. **Create preview_manager.py**
   - Extract preview functionality
   - Create PreviewManager class
   - Move preview timer and cache

4. **Create parameter_manager.py**
   - Extract parameter widget logic
   - Create ParameterManager class

5. **Create history_manager.py**
   - Extract history display logic
   - Create HistoryManager class

6. **Update PreprocessPage**
   - Initialize manager instances
   - Delegate method calls to managers
   - Maintain signal/slot connections

**Deliverables**:
- 5 new manager classes (1300 lines total)
- Significantly reduced `preprocess_page.py` (target 600-800 lines)

**Testing**:
- Unit tests for each manager class
- Integration testing with PreprocessPage
- End-to-end workflow testing

### Phase 4: Integration & Testing (2-3 hours)
**Goal**: Ensure all refactored code works correctly

1. **Integration Testing**
   - Full preprocessing workflow
   - Export functionality
   - Pipeline building and editing
   - Preview mode toggling
   - History loading

2. **Performance Validation**
   - No regression in UI responsiveness
   - Preview update performance
   - Large dataset handling

3. **Code Quality**
   - Run linters (ruff, black)
   - Type hint validation
   - Docstring coverage

4. **Documentation Updates**
   - Update preprocess_page.md
   - Add architecture diagrams
   - Document new module structure

**Deliverables**:
- Fully functional refactored codebase
- Updated documentation
- Test results and validation reports

### Phase 5: Optimization & Polish (1-2 hours)
**Goal**: Final optimizations and cleanup

1. **Code Review**
   - Review all new modules
   - Check for code duplication
   - Optimize imports

2. **Performance Tuning**
   - Profile critical paths
   - Optimize preview updates
   - Cache improvements

3. **Documentation Polish**
   - Final documentation review
   - Add code examples
   - Update .AGI-BANKS files

## 📊 Success Metrics

### Quantitative Metrics
- ✅ Main file reduced from 3060 to 600-800 lines (70-75% reduction)
- ✅ Zero inline styles in preprocess_page.py
- ✅ 7 new focused modules (each <400 lines)
- ✅ All tests passing
- ✅ No performance regression

### Qualitative Metrics
- ✅ Clear separation of concerns
- ✅ Improved code readability
- ✅ Easier to maintain and extend
- ✅ Better testability
- ✅ Consistent styling

## ⚠️ Risks & Mitigation

### Risk 1: Signal/Slot Connection Breaks
**Mitigation**: 
- Comprehensive testing after each phase
- Keep connection patterns consistent
- Document all signal/slot relationships

### Risk 2: Style Regression
**Mitigation**:
- Visual regression testing
- Screenshot comparisons
- Thorough UI review

### Risk 3: Performance Impact
**Mitigation**:
- Profile before and after refactoring
- Monitor memory usage
- Optimize critical paths

### Risk 4: Breaking Changes
**Mitigation**:
- Maintain backward compatibility where possible
- Test on actual project data
- Have rollback plan (git branch)

## 🔄 Rollback Plan

If critical issues arise:
1. Keep original code in git branch
2. Document all breaking changes
3. Quick rollback procedure:
   ```bash
   git checkout main
   git cherry-pick <safe-commits>
   ```

## 📅 Timeline

**Total Estimated Time**: 12-17 hours

- Phase 1 (Style Extraction): 2-3 hours
- Phase 2 (UI Components): 3-4 hours
- Phase 3 (Manager Classes): 4-5 hours
- Phase 4 (Integration): 2-3 hours
- Phase 5 (Optimization): 1-2 hours

**Recommended Approach**: Complete one phase at a time, test thoroughly before proceeding.

## ✅ Checklist

### Before Starting
- [ ] Commit all current changes
- [ ] Create refactoring branch
- [ ] Backup current codebase
- [ ] Review all existing functionality
- [ ] Set up testing environment

### Phase 1 Checklist
- [ ] Create styles.py module
- [ ] Extract all inline styles
- [ ] Update imports in preprocess_page.py
- [ ] Test UI rendering
- [ ] Commit Phase 1 changes

### Phase 2 Checklist
- [ ] Create ui_components.py module
- [ ] Extract UI creation methods
- [ ] Update PreprocessPage class
- [ ] Test UI construction
- [ ] Commit Phase 2 changes

### Phase 3 Checklist
- [ ] Create all manager classes
- [ ] Extract business logic
- [ ] Update PreprocessPage delegation
- [ ] Test all workflows
- [ ] Commit Phase 3 changes

### Phase 4 Checklist
- [ ] Full integration testing
- [ ] Performance validation
- [ ] Code quality checks
- [ ] Documentation updates
- [ ] Commit Phase 4 changes

### Phase 5 Checklist
- [ ] Code review
- [ ] Final optimizations
- [ ] Documentation polish
- [ ] Final testing
- [ ] Merge to main branch

## 📝 Notes

- This refactoring maintains all existing functionality
- No user-facing changes (except improved performance)
- All signals and slots preserved
- Backward compatible with existing code
- Can be done incrementally with testing between phases
