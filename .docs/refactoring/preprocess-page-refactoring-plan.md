# Preprocess Page Refactoring Plan
**Version**: 1.0.0  
**Date**: October 6, 2025  
**Status**: 📋 Planning Phase

## 🎯 Refactoring Objectives

### Primary Goals
1. **Reduce file size** from 3,068 lines to manageable modules (<500 lines each)
2. **Improve maintainability** with clear separation of concerns
3. **Add comprehensive documentation** with structured comments
4. **Preserve all functionality** through rigorous phase-by-phase validation
5. **Enable easier debugging** with isolated, testable components

### Success Criteria
- ✅ Each module has single responsibility
- ✅ All methods have structured docstrings
- ✅ No functionality broken or lost
- ✅ Test coverage for critical paths
- ✅ Clear import structure and dependencies
- ✅ Reduced cognitive load per file

---

## 📊 Current Architecture Analysis

### File Structure Overview
```
pages/
├── preprocess_page.py                      (3,068 lines) ⚠️ TOO LARGE
│   └── PreprocessPage (QWidget)
│       ├── 74 methods
│       ├── 12 UI component creators
│       ├── 18 data management methods
│       ├── 15 preprocessing execution methods
│       ├── 12 preview/visualization methods
│       ├── 8 memory/state management methods
│       └── 9 utility/helper methods
└── preprocess_page_utils/
    ├── __utils__.py                        (30 lines)
    ├── pipeline.py                         (898 lines)
    ├── widgets.py                          (?)
    └── ... (other utilities)
```

### Method Categories Analysis

#### 🎨 **UI Component Creators** (12 methods, ~950 lines)
Methods that build UI sections:

| Method | Lines | Purpose | Dependencies | Refactor Priority |
|--------|-------|---------|--------------|-------------------|
| `_setup_ui()` | ~25 | Main UI initialization | All creators | 🔴 High |
| `_create_left_panel()` | ~25 | Left panel container | Pipeline, Input, Output groups | 🟡 Medium |
| `_create_pipeline_building_group()` | ~250 | Pipeline builder UI | Category/Method combos, Pipeline list | 🔴 High |
| `_create_input_datasets_group()` | ~200 | Dataset selection UI | Tab widgets, List widgets | 🔴 High |
| `_create_output_configuration_group()` | ~40 | Output config UI | Text inputs | 🟢 Low |
| `_create_right_panel()` | ~130 | Right panel with viz | Plot widget, Preview controls | 🟡 Medium |

**Algorithms & Flow**:
- **Pattern**: Builder pattern - each method creates QGroupBox with layouts
- **Data Flow**: Parent → Child widget construction
- **API Usage**: PySide6.QtWidgets (QGroupBox, QVBoxLayout, QHBoxLayout)
- **Critical Info**:
  - Height constraints for non-maximized windows (140-165px datasets, 180-215px pipeline)
  - Icon-only buttons (28x28px) with tooltips
  - Compact spacing (8px) and margins (12px)
  - Signal connections for user interactions

**Refactoring Strategy**: Extract to `ui_builders.py` with one function per section

---

#### 📦 **Data Management Methods** (18 methods, ~700 lines)
Methods that handle project data, datasets, exports:

| Method | Lines | Purpose | Data Flow | Dependencies |
|--------|-------|---------|-----------|--------------|
| `load_project_data()` | ~105 | Load datasets from project | Project → Datasets → UI Lists | Project state, Dataset lists |
| `export_dataset()` | ~315 | Export dataset with metadata | Selected datasets → Files | Export dialog, File I/O |
| `_export_single_dataset()` | ~55 | Single dataset export | Dataset object → CSV/JSON | File I/O, Metadata |
| `_export_metadata_json()` | ~60 | Export metadata to JSON | Metadata dict → JSON file | JSON serialization |
| `_on_dataset_selection_changed()` | ~75 | Handle dataset selection | UI selection → Pipeline/Preview update | Pipeline loader, Preview |
| `_on_dataset_tab_changed()` | ~10 | Handle tab switching | Tab index → List widget | Tab widget state |
| `_clean_dataset_name()` | ~4 | Remove preprocessing suffix | String → Clean string | String manipulation |
| `_set_default_output_name()` | ~10 | Set default output name | Dataset name → Output field | Output text widget |
| `_clear_default_output_name()` | ~8 | Clear output name field | - → Empty field | Output text widget |

**Algorithms & Flow**:
```
load_project_data():
  1. Get current project path
  2. Read datasets from disk (raw/, preprocessed/)
  3. Parse dataset metadata (JSON)
  4. Populate UI lists (All/Raw/Preprocessed tabs)
  5. Restore selection if exists
  6. Load preprocessing history
  
export_dataset():
  1. Get selected datasets
  2. Show export dialog (location selection)
  3. For each dataset:
     a. Write data to CSV
     b. Write metadata to JSON
     c. Include preprocessing pipeline
  4. Show success notification
```

**API Usage**:
- **File I/O**: `os.path`, `os.listdir`, `open()`
- **Data**: `pandas.DataFrame.to_csv()`, `json.dump()`
- **UI**: `QListWidget.currentItem()`, `QFileDialog`

**Critical Info**:
- Dataset structure: `{name, data, wavenumber, metadata, preprocessing_history}`
- Metadata format: JSON with pipeline history, timestamps
- Export preserves full preprocessing chain
- Tab filtering logic (All/Raw/Preprocessed)

**Refactoring Strategy**: Extract to `data_manager.py` with DataManager class

---

#### ⚙️ **Pipeline Management Methods** (15 methods, ~650 lines)
Methods that handle preprocessing pipeline operations:

| Method | Lines | Purpose | Algorithm | State Changes |
|--------|-------|---------|-----------|---------------|
| `add_pipeline_step()` | ~50 | Add step to pipeline | Create step → Add to list → Update UI | `self.pipeline_steps` append |
| `remove_pipeline_step()` | ~35 | Remove selected step | Get index → Remove from list → Update indices | `self.pipeline_steps` delete |
| `clear_pipeline()` | ~15 | Clear all steps | Loop → Remove all → Clear UI | `self.pipeline_steps` clear |
| `on_step_toggled()` | ~15 | Toggle step enabled | Get step → Toggle flag → Update UI | `step.enabled` toggle |
| `toggle_all_existing_steps()` | ~40 | Toggle all existing | Loop steps → Toggle all → Update widgets | Multiple `step.enabled` |
| `on_pipeline_step_selected()` | ~20 | Handle step selection | Save prev params → Load new params | Current step state |
| `_show_parameter_widget()` | ~25 | Show params for step | Get widget factory → Create widget → Display | Parameter widget visible |
| `_clear_parameter_widget()` | ~15 | Clear parameter area | Remove widget → Hide area | Parameter widget hidden |
| `_update_step_parameters()` | ~10 | Save current params | Get widget values → Store in step | `step.params` update |
| `_on_pipeline_reordered()` | ~20 | Handle drag-drop reorder | Update indices → Reorder list → Update UI | `self.pipeline_steps` order |
| `_load_preprocessing_pipeline()` | ~45 | Load pipeline from data | Parse JSON → Create steps → Add to UI | Full pipeline restore |
| `_save_to_global_memory()` | ~15 | Save pipeline state | Serialize steps → Store in memory | `self._global_pipeline_memory` |
| `_restore_global_pipeline_memory()` | ~20 | Restore saved pipeline | Load from memory → Rebuild UI | Full pipeline restore |
| `_clear_global_memory()` | ~4 | Clear memory | Clear list | `self._global_pipeline_memory` clear |
| `_rebuild_pipeline_ui()` | ~15 | Rebuild UI from steps | Loop steps → Create widgets → Display | Full UI rebuild |

**Algorithms & Flow**:
```
add_pipeline_step():
  1. Validate category/method selection
  2. Save current step parameters (if any)
  3. Create PipelineStep object
  4. Append to self.pipeline_steps
  5. Create QListWidgetItem
  6. Create PipelineStepWidget
  7. Set item widget and size hint
  8. Select new item
  9. Show parameter widget
  10. Save to global memory
  11. Schedule preview update

Pipeline State Machine:
  - State: List of PipelineStep objects
  - UI: QListWidget with custom widgets
  - Memory: Global backup for cross-dataset retention
  - Persistence: JSON in dataset metadata
```

**API Usage**:
- **Data Structures**: `List[PipelineStep]`, `Dict[str, Any]`
- **UI**: `QListWidget`, `QListWidgetItem`, custom `PipelineStepWidget`
- **Signals**: `currentItemChanged`, `rowsMoved`

**Critical Info**:
- **Pipeline Steps**: Ordered list, can be reordered via drag-drop
- **Step States**: `enabled` flag, `is_existing` flag for imported steps
- **Parameter Storage**: Each step stores `params` dict
- **Global Memory**: Preserves pipeline across dataset switches
- **Source Tracking**: Steps track `source_dataset` for imports

**Refactoring Strategy**: Extract to `pipeline_manager.py` with PipelineManager class

---

#### 🔄 **Preprocessing Execution Methods** (10 methods, ~580 lines)
Methods that run preprocessing pipeline:

| Method | Lines | Purpose | Processing Type | Thread Safety |
|--------|-------|---------|----------------|---------------|
| `run_preprocessing()` | ~115 | Main preprocessing executor | Full pipeline | Uses QThread |
| `_on_thread_finished()` | ~40 | Thread completion handler | - | Thread-safe |
| `on_step_completed()` | ~8 | Single step completion | - | Thread-safe |
| `on_step_failed()` | ~5 | Single step error | - | Thread-safe |
| `cancel_preprocessing()` | ~8 | Cancel running process | - | Thread-safe |
| `_start_processing_ui()` | ~9 | Update UI for processing | - | Main thread |
| `_reset_ui_state()` | ~9 | Reset UI after processing | - | Main thread |
| `on_progress_updated()` | ~4 | Update progress bar | - | Thread-safe signal |
| `on_status_updated()` | ~4 | Update status text | - | Thread-safe signal |
| `on_processing_completed()` | ~95 | Handle completion | Save results | Thread-safe |
| `on_processing_error()` | ~5 | Handle errors | - | Thread-safe |

**Algorithms & Flow**:
```
run_preprocessing():
  1. Validate inputs (datasets, pipeline, output name)
  2. Get selected datasets
  3. Prepare pipeline steps (enabled only)
  4. Create preprocessing thread (QThread)
  5. Connect signals (progress, status, completion, error)
  6. Start thread execution
  7. Update UI (disable buttons, show progress)
  
Thread Execution (in QThread):
  1. For each dataset:
     a. Load data from project
     b. For each pipeline step:
        - Apply preprocessing method
        - Update progress (step N of M)
        - Handle errors
     c. Collect metadata (steps, params, timestamp)
  2. Emit completion signal with results
  
on_processing_completed():
  1. Save preprocessed data to project
  2. Save metadata (pipeline history)
  3. Reload project data (refresh lists)
  4. Show preprocessing history dialog
  5. Reset UI state
```

**API Usage**:
- **Threading**: `QThread`, `Signal`, `pyqtSlot`
- **Data Processing**: `EnhancedRamanPipeline` from `functions.preprocess`
- **File I/O**: Save to `preprocessed/` directory
- **Signals**: `progressUpdated`, `statusUpdated`, `completionSignal`, `errorSignal`

**Critical Info**:
- **Thread Safety**: All UI updates via signals
- **Error Handling**: Try-except in thread, emit error signal
- **Cancellation**: Thread checks cancellation flag
- **Metadata**: Full pipeline history saved with results
- **Multi-dataset**: Can process multiple datasets in batch

**Refactoring Strategy**: Extract to `preprocessing_executor.py` with PreprocessingExecutor class

---

#### 📊 **Preview/Visualization Methods** (12 methods, ~550 lines)
Methods that handle live preview and plotting:

| Method | Lines | Purpose | Update Trigger | Performance |
|--------|-------|---------|----------------|-------------|
| `_toggle_preview_mode()` | ~65 | Enable/disable preview | Toggle button | Immediate |
| `_update_preview()` | ~108 | Main preview update | Timer (300ms debounce) | Throttled |
| `_schedule_preview_update()` | ~15 | Schedule update | Any pipeline change | Debounced |
| `_manual_refresh_preview()` | ~9 | Force refresh | Manual button | Immediate |
| `_update_preview_button_state()` | ~40 | Update toggle button | Preview state change | Immediate |
| `_update_preview_status()` | ~33 | Update status indicator | Preview computation | Immediate |
| `_apply_full_pipeline()` | ~65 | Apply all pipeline steps | Full processing | Synchronous |
| `_apply_preview_pipeline()` | ~98 | Apply pipeline preview | Preview update | Synchronous |
| `_show_original_data()` | ~10 | Plot original data | Dataset selection | Immediate |
| `_show_preview_data()` | ~58 | Plot preview data | Preview computation | Immediate |
| `_should_auto_focus()` | ~6 | Check auto-focus flag | Cropper params | Immediate |
| `_extract_crop_bounds()` | ~12 | Get crop boundaries | Cropper step | Immediate |
| `_manual_focus()` | ~88 | Manual focus trigger | Focus button | Immediate |

**Algorithms & Flow**:
```
Preview Update Flow:
  1. User changes pipeline (add/remove/reorder/params)
  2. _schedule_preview_update() called
  3. Timer started (300ms debounce)
  4. Timer expires → _update_preview()
  5. Get current dataset
  6. Apply pipeline to data
  7. Plot results (original vs preview)
  8. Update status indicator
  
_update_preview():
  1. Check if preview enabled
  2. Check if dataset selected
  3. Check if pipeline exists
  4. Get data and wavenumber
  5. Create pipeline from steps
  6. Apply pipeline (with error handling)
  7. Plot original + preview side-by-side
  8. Auto-focus if Cropper detected
  9. Update status (Ready/Error/Processing)

Auto-Focus Logic:
  1. Check if Cropper in pipeline
  2. Get crop bounds from parameters
  3. Apply to plot x-axis limits
  4. Highlight cropped region
```

**API Usage**:
- **Plotting**: `MatplotlibWidget`, `plot_spectra()`
- **Timer**: `QTimer.singleShot()` for debouncing
- **Data**: NumPy arrays, Pandas DataFrames
- **Pipeline**: `EnhancedRamanPipeline.preview_pipeline()`

**Critical Info**:
- **Debouncing**: 300ms delay prevents excessive computation
- **Performance**: Preview uses limited data (first N spectra)
- **Auto-Focus**: Automatically zooms to crop region
- **Status States**: "Ready", "Processing", "Error", "Preview Disabled"
- **Side-by-Side**: Original (left) vs Preview (right) plots

**Refactoring Strategy**: Extract to `preview_manager.py` with PreviewManager class

---

#### 🧠 **Memory/State Management Methods** (8 methods, ~80 lines)
Methods that manage application state:

| Method | Lines | Purpose | Persistence | Scope |
|--------|-------|---------|-------------|-------|
| `_save_to_global_memory()` | ~15 | Save pipeline to memory | In-memory | Cross-dataset |
| `_restore_global_pipeline_memory()` | ~20 | Restore from memory | In-memory | Cross-dataset |
| `_clear_global_memory()` | ~4 | Clear memory | In-memory | Cross-dataset |
| `_clear_preprocessing_history()` | ~9 | Clear history data | Persistent | Dataset-specific |
| `_clear_preprocessing_history_display_only()` | ~7 | Clear UI only | UI only | Display |
| `_show_preprocessing_history()` | ~38 | Show history dialog | Display | Read-only |
| `_pipelines_differ()` | ~32 | Compare pipelines | Comparison | Read-only |
| `_show_pipeline_transfer_dialog()` | ~48 | Import pipeline dialog | Display | User choice |

**Algorithms & Flow**:
```
Global Pipeline Memory:
  - Purpose: Retain pipeline when switching datasets
  - Storage: self._global_pipeline_memory (List[PipelineStep])
  - Lifecycle:
    1. Save on any pipeline change
    2. Restore when switching datasets
    3. Clear on explicit user action
    
Pipeline Comparison:
  1. Get current pipeline steps
  2. Get target pipeline (from dataset)
  3. Compare count
  4. Compare each step (category, method, params)
  5. Return True if different
  
Pipeline Transfer:
  1. Detect existing pipeline in dataset
  2. Compare with current pipeline
  3. If different, show dialog:
     - Load from dataset
     - Keep current
     - Merge (not implemented)
  4. Apply user choice
```

**API Usage**:
- **Data Structures**: `List[PipelineStep]`, deep copy
- **UI**: `QDialog`, `QMessageBox`
- **Comparison**: Dict equality checks

**Critical Info**:
- **Global Memory**: Not persistent across app restarts
- **Dataset History**: Persistent in metadata JSON
- **Conflict Resolution**: User chooses on pipeline mismatch
- **Deep Copy**: Avoids reference issues

**Refactoring Strategy**: Extract to `state_manager.py` with StateManager class

---

#### 🔧 **Utility/Helper Methods** (9 methods, ~120 lines)
Methods that provide utility functions:

| Method | Lines | Purpose | Used By | Type |
|--------|-------|---------|---------|------|
| `update_method_combo()` | ~10 | Update method dropdown | Category selection | UI sync |
| `_connect_signals()` | ~17 | Connect all signals | Initialization | Setup |
| `_connect_parameter_signals()` | ~25 | Connect param signals | Parameter widget | Dynamic |
| `_update_current_step_parameters()` | ~10 | Save current params | Pipeline changes | State |
| `_get_data_wavenumber_range()` | ~11 | Get data range | Preview, Focus | Data |
| `_adjust_button_width_to_text()` | ~25 | Auto-size button | Preview toggle | UI |
| `_clean_dataset_name()` | ~4 | Remove suffix | Dataset display | String |

**Refactoring Strategy**: Keep in main class or extract to `utils.py`

---

## 📁 Proposed File Structure

### New Module Organization
```
pages/
├── preprocess_page.py                      (200-300 lines) ✅ MAIN COORDINATOR
│   └── PreprocessPage (QWidget)
│       ├── __init__()
│       ├── _setup_ui() → delegates to UIBuilder
│       ├── _connect_signals()
│       ├── Public API methods (10-15 methods)
│       └── Event handlers (10-15 methods)
│
└── preprocess_page_utils/
    ├── __init__.py                         (Import all)
    ├── __utils__.py                        (Shared imports)
    │
    ├── ui_builders.py                      (~400 lines) ✅ UI CONSTRUCTION
    │   └── UIBuilder class
    │       ├── build_main_layout()
    │       ├── build_left_panel()
    │       ├── build_right_panel()
    │       ├── build_pipeline_group()
    │       ├── build_input_datasets_group()
    │       ├── build_output_group()
    │       └── build_visualization_group()
    │
    ├── data_manager.py                     (~400 lines) ✅ DATA OPERATIONS
    │   └── DataManager class
    │       ├── load_project_data()
    │       ├── export_dataset()
    │       ├── export_single_dataset()
    │       ├── export_metadata()
    │       ├── get_selected_datasets()
    │       ├── parse_dataset_metadata()
    │       └── validate_dataset()
    │
    ├── pipeline_manager.py                 (~450 lines) ✅ PIPELINE LOGIC
    │   └── PipelineManager class
    │       ├── add_step()
    │       ├── remove_step()
    │       ├── clear_pipeline()
    │       ├── toggle_step()
    │       ├── reorder_steps()
    │       ├── load_from_json()
    │       ├── save_to_json()
    │       ├── get_enabled_steps()
    │       └── validate_pipeline()
    │
    ├── preprocessing_executor.py           (~400 lines) ✅ EXECUTION ENGINE
    │   ├── PreprocessingThread (QThread)
    │   │   ├── run()
    │   │   ├── apply_pipeline()
    │   │   └── handle_errors()
    │   └── PreprocessingExecutor class
    │       ├── start_preprocessing()
    │       ├── cancel_preprocessing()
    │       ├── handle_completion()
    │       └── handle_error()
    │
    ├── preview_manager.py                  (~450 lines) ✅ LIVE PREVIEW
    │   └── PreviewManager class
    │       ├── toggle_preview()
    │       ├── update_preview()
    │       ├── schedule_update()
    │       ├── apply_pipeline_preview()
    │       ├── show_original()
    │       ├── show_preview()
    │       ├── update_status()
    │       ├── auto_focus()
    │       └── manual_focus()
    │
    ├── state_manager.py                    (~250 lines) ✅ STATE PERSISTENCE
    │   └── StateManager class
    │       ├── save_to_memory()
    │       ├── restore_from_memory()
    │       ├── clear_memory()
    │       ├── compare_pipelines()
    │       ├── show_history()
    │       └── handle_conflicts()
    │
    ├── parameter_widgets.py                (Existing, optimize)
    ├── pipeline.py                         (Existing, optimize)
    └── widgets.py                          (Existing, optimize)
```

### Import Dependencies Graph
```
preprocess_page.py
├── ui_builders.py
│   ├── __utils__.py
│   ├── widgets.py
│   └── parameter_widgets.py
├── data_manager.py
│   ├── __utils__.py
│   └── state_manager.py
├── pipeline_manager.py
│   ├── __utils__.py
│   ├── pipeline.py
│   └── state_manager.py
├── preprocessing_executor.py
│   ├── __utils__.py
│   ├── pipeline_manager.py
│   └── functions.preprocess
├── preview_manager.py
│   ├── __utils__.py
│   ├── pipeline_manager.py
│   └── components.widgets.matplotlib_widget
└── state_manager.py
    ├── __utils__.py
    └── pipeline.py
```

---

## 📝 Documentation Standard

### Structured Comment Format
```python
"""
{One-line description of what this function/class does}

Detailed description if needed (algorithm, flow, design pattern).

Args:
    param_name (type): Description of parameter
    another_param (type, optional): Description with default. Defaults to None.

Returns:
    return_type: Description of return value
    
Raises:
    ExceptionType: When this exception is raised
    
Use in:
    - pages/preprocess_page.py: PreprocessPage.__init__()
    - pages/preprocess_page.py: PreprocessPage._setup_ui()
    
Example:
    >>> manager = PipelineManager()
    >>> manager.add_step("Noise Removal", "SavGol")
    
Note:
    Important information about side effects, performance, or constraints.
"""
```

### Example: Function Documentation
```python
def add_pipeline_step(self, category: str, method: str, params: Dict[str, Any] = None) -> bool:
    """
    Add a new preprocessing step to the pipeline.
    
    Creates a PipelineStep object, adds it to the internal pipeline list,
    creates a UI widget, and updates the global pipeline memory. Automatically
    schedules a preview update if preview mode is enabled.
    
    Args:
        category (str): Preprocessing category (e.g., "Noise Removal")
        method (str): Method name (e.g., "SavGol")
        params (Dict[str, Any], optional): Method parameters. Defaults to {}.
    
    Returns:
        bool: True if step added successfully, False otherwise
        
    Raises:
        ValueError: If category or method is invalid
        
    Use in:
        - pages/preprocess_page.py: PreprocessPage.on_add_button_clicked()
        - preprocess_page_utils/pipeline_manager.py: PipelineManager.import_pipeline()
        
    Example:
        >>> pipeline_mgr.add_pipeline_step("Noise Removal", "SavGol", {"window_length": 11})
        True
        
    Note:
        This method has side effects:
        - Updates self.pipeline_steps list
        - Creates QListWidgetItem and PipelineStepWidget
        - Saves to global memory
        - Schedules preview update (300ms debounce)
    """
    # Implementation...
```

### Example: Class Documentation
```python
class PipelineManager:
    """
    Manages preprocessing pipeline operations and state.
    
    Handles adding, removing, reordering, and persisting pipeline steps.
    Maintains pipeline state consistency between UI, memory, and storage.
    
    Attributes:
        pipeline_steps (List[PipelineStep]): Ordered list of pipeline steps
        _memory_backup (List[PipelineStep]): Global backup for cross-dataset retention
        
    Use in:
        - pages/preprocess_page.py: PreprocessPage (composition)
        - preprocess_page_utils/preprocessing_executor.py: PreprocessingExecutor
        - preprocess_page_utils/preview_manager.py: PreviewManager
        
    Example:
        >>> manager = PipelineManager()
        >>> manager.add_step("Noise Removal", "SavGol")
        >>> manager.add_step("Baseline Correction", "AsLS")
        >>> steps = manager.get_enabled_steps()
        >>> manager.save_to_json("pipeline.json")
        
    Note:
        Thread-safe for read operations. Write operations should be called
        from the main Qt thread only.
    """
```

---

## 🔄 Phased Refactoring Plan

### Phase 1: Setup & Preparation ✅ (1-2 hours)
**Goal**: Prepare infrastructure without breaking existing code

#### Tasks
- [ ] **1.1 Create New Module Files**
  - Create empty module files with docstrings
  - Set up `__init__.py` with imports
  - Add copyright headers and version info
  
- [ ] **1.2 Set Up Testing Framework**
  - Create `tests/pages/test_preprocess_page.py`
  - Add smoke tests for current functionality
  - Document test data requirements
  
- [ ] **1.3 Create Baseline Snapshot**
  - Run full application test
  - Document current behavior (screenshots)
  - Export sample project for regression testing
  
- [ ] **1.4 Set Up Git Branch**
  - Create feature branch: `refactor/preprocess-page`
  - Commit baseline code
  - Tag as `refactor-start`

#### Validation Checkpoints
✅ All module files created  
✅ Tests run and pass (baseline)  
✅ Git branch created  
✅ Documentation in place  

---

### Phase 2: Extract UI Builders 🎨 (3-4 hours)
**Goal**: Move UI creation to separate module

#### Tasks
- [ ] **2.1 Create UIBuilder Class**
  - Define class structure and interface
  - Add parent widget parameter
  - Set up method signatures
  
- [ ] **2.2 Extract Pipeline Building Group**
  - Copy `_create_pipeline_building_group()` → `ui_builders.py`
  - Rename to `build_pipeline_group()`
  - Add structured docstring
  - Pass necessary state (combos, lists) to parent
  - Test in isolation
  
- [ ] **2.3 Extract Input Datasets Group**
  - Copy `_create_input_datasets_group()` → `ui_builders.py`
  - Rename to `build_input_datasets_group()`
  - Add structured docstring
  - Pass tab widget and lists to parent
  - Test in isolation
  
- [ ] **2.4 Extract Output Configuration Group**
  - Copy `_create_output_configuration_group()` → `ui_builders.py`
  - Rename to `build_output_group()`
  - Add structured docstring
  - Test in isolation
  
- [ ] **2.5 Extract Visualization Group**
  - Extract from `_create_right_panel()`
  - Create `build_visualization_group()`
  - Add structured docstring
  - Test in isolation
  
- [ ] **2.6 Extract Panel Builders**
  - Copy `_create_left_panel()` → `build_left_panel()`
  - Copy `_create_right_panel()` → `build_right_panel()`
  - Use extracted group builders
  - Test in isolation
  
- [ ] **2.7 Update PreprocessPage**
  - Import UIBuilder
  - Replace `_create_*` calls with `UIBuilder` calls
  - Store returned widgets as instance variables
  - Test full integration

#### Validation Checkpoints
✅ UIBuilder class created  
✅ All UI methods extracted  
✅ Structured docstrings added  
✅ Tests pass (UI renders correctly)  
✅ No visual differences from baseline  
✅ Code compiles without errors  

---

### Phase 3: Extract Data Manager 📦 (3-4 hours)
**Goal**: Separate data operations from UI

#### Tasks
- [ ] **3.1 Create DataManager Class**
  - Define class with project_path parameter
  - Add dataset loading methods signature
  - Set up error handling pattern
  
- [ ] **3.2 Extract Load Project Data**
  - Copy `load_project_data()` → `DataManager.load_project_data()`
  - Add structured docstring
  - Return datasets dict instead of updating UI
  - Test with sample project
  
- [ ] **3.3 Extract Export Methods**
  - Copy `export_dataset()` → `DataManager.export_dataset()`
  - Copy `_export_single_dataset()` → `DataManager.export_single()`
  - Copy `_export_metadata_json()` → `DataManager.export_metadata()`
  - Add structured docstrings
  - Test export functionality
  
- [ ] **3.4 Add Dataset Utilities**
  - Move `_clean_dataset_name()`
  - Add `get_selected_datasets()`
  - Add `parse_metadata()`
  - Add `validate_dataset()`
  - Add structured docstrings
  
- [ ] **3.5 Update PreprocessPage**
  - Create DataManager instance
  - Replace direct calls with DataManager calls
  - Update UI from returned data
  - Test full integration

#### Validation Checkpoints
✅ DataManager class created  
✅ All data methods extracted  
✅ Tests pass (load/export work)  
✅ Data integrity verified  
✅ No data loss or corruption  

---

### Phase 4: Extract Pipeline Manager ⚙️ (4-5 hours)
**Goal**: Isolate pipeline logic

#### Tasks
- [ ] **4.1 Create PipelineManager Class**
  - Define class with pipeline_steps list
  - Add signal definitions (if needed)
  - Set up memory backup mechanism
  
- [ ] **4.2 Extract Add/Remove Methods**
  - Copy `add_pipeline_step()` → `PipelineManager.add_step()`
  - Copy `remove_pipeline_step()` → `PipelineManager.remove_step()`
  - Copy `clear_pipeline()` → `PipelineManager.clear()`
  - Add structured docstrings
  - Return UI update instructions instead of updating UI
  - Test pipeline operations
  
- [ ] **4.3 Extract Toggle Methods**
  - Copy `on_step_toggled()` → `PipelineManager.toggle_step()`
  - Copy `toggle_all_existing_steps()` → `PipelineManager.toggle_all()`
  - Add structured docstrings
  - Test toggle functionality
  
- [ ] **4.4 Extract Pipeline I/O**
  - Copy `_load_preprocessing_pipeline()` → `PipelineManager.load_from_json()`
  - Add `PipelineManager.save_to_json()`
  - Add structured docstrings
  - Test load/save
  
- [ ] **4.5 Extract Memory Management**
  - Copy `_save_to_global_memory()` → `PipelineManager.save_to_memory()`
  - Copy `_restore_global_pipeline_memory()` → `PipelineManager.restore_from_memory()`
  - Copy `_clear_global_memory()` → `PipelineManager.clear_memory()`
  - Add structured docstrings
  - Test memory operations
  
- [ ] **4.6 Extract Reorder Logic**
  - Copy `_on_pipeline_reordered()` → `PipelineManager.reorder()`
  - Add structured docstring
  - Test drag-drop reordering
  
- [ ] **4.7 Update PreprocessPage**
  - Create PipelineManager instance
  - Replace direct calls with PipelineManager calls
  - Update UI based on manager state
  - Connect UI signals to manager methods
  - Test full integration

#### Validation Checkpoints
✅ PipelineManager class created  
✅ All pipeline methods extracted  
✅ Tests pass (add/remove/reorder)  
✅ Memory operations work  
✅ Pipeline persistence works  

---

### Phase 5: Extract Preprocessing Executor 🔄 (4-5 hours)
**Goal**: Separate execution engine from UI

#### Tasks
- [ ] **5.1 Create PreprocessingThread Class**
  - Define QThread subclass
  - Add signal definitions (progress, status, completion, error)
  - Set up cancellation mechanism
  
- [ ] **5.2 Extract Thread Run Method**
  - Copy processing logic from `run_preprocessing()`
  - Implement `run()` method
  - Add structured docstring
  - Test thread execution
  
- [ ] **5.3 Create PreprocessingExecutor Class**
  - Define class with thread management
  - Add `start_preprocessing()` method
  - Add `cancel_preprocessing()` method
  - Add signal handlers
  
- [ ] **5.4 Extract Execution Methods**
  - Copy `run_preprocessing()` → `PreprocessingExecutor.start()`
  - Copy `_on_thread_finished()` → `PreprocessingExecutor.on_finished()`
  - Copy `on_processing_completed()` → `PreprocessingExecutor.on_completed()`
  - Copy `on_processing_error()` → `PreprocessingExecutor.on_error()`
  - Add structured docstrings
  
- [ ] **5.5 Extract UI State Methods**
  - Copy `_start_processing_ui()` → `PreprocessingExecutor.start_ui()`
  - Copy `_reset_ui_state()` → `PreprocessingExecutor.reset_ui()`
  - Add structured docstrings
  
- [ ] **5.6 Update PreprocessPage**
  - Create PreprocessingExecutor instance
  - Replace direct calls with executor calls
  - Connect executor signals to UI updates
  - Test full integration

#### Validation Checkpoints
✅ PreprocessingExecutor created  
✅ Thread execution works  
✅ Tests pass (preprocessing runs)  
✅ Progress updates work  
✅ Error handling works  
✅ Cancellation works  

---

### Phase 6: Extract Preview Manager 📊 (4-5 hours)
**Goal**: Isolate preview and visualization logic

#### Tasks
- [ ] **6.1 Create PreviewManager Class**
  - Define class with plot widget
  - Add preview state tracking
  - Set up debounce timer
  
- [ ] **6.2 Extract Preview Toggle**
  - Copy `_toggle_preview_mode()` → `PreviewManager.toggle_preview()`
  - Copy `_update_preview_button_state()` → `PreviewManager.update_button_state()`
  - Add structured docstrings
  - Test toggle functionality
  
- [ ] **6.3 Extract Preview Update**
  - Copy `_update_preview()` → `PreviewManager.update_preview()`
  - Copy `_schedule_preview_update()` → `PreviewManager.schedule_update()`
  - Copy `_manual_refresh_preview()` → `PreviewManager.manual_refresh()`
  - Add structured docstrings
  - Test preview updates
  
- [ ] **6.4 Extract Pipeline Application**
  - Copy `_apply_full_pipeline()` → `PreviewManager.apply_full_pipeline()`
  - Copy `_apply_preview_pipeline()` → `PreviewManager.apply_preview_pipeline()`
  - Add structured docstrings
  - Test pipeline application
  
- [ ] **6.5 Extract Plotting Methods**
  - Copy `_show_original_data()` → `PreviewManager.show_original()`
  - Copy `_show_preview_data()` → `PreviewManager.show_preview()`
  - Add structured docstrings
  - Test plotting
  
- [ ] **6.6 Extract Focus Methods**
  - Copy `_should_auto_focus()` → `PreviewManager.should_auto_focus()`
  - Copy `_extract_crop_bounds()` → `PreviewManager.extract_crop_bounds()`
  - Copy `_manual_focus()` → `PreviewManager.manual_focus()`
  - Add structured docstrings
  - Test focus functionality
  
- [ ] **6.7 Extract Status Updates**
  - Copy `_update_preview_status()` → `PreviewManager.update_status()`
  - Add structured docstring
  - Test status updates
  
- [ ] **6.8 Update PreprocessPage**
  - Create PreviewManager instance
  - Replace direct calls with manager calls
  - Connect manager signals to UI
  - Test full integration

#### Validation Checkpoints
✅ PreviewManager class created  
✅ All preview methods extracted  
✅ Tests pass (preview works)  
✅ Debouncing works  
✅ Auto-focus works  
✅ Performance acceptable  

---

### Phase 7: Extract State Manager 🧠 (2-3 hours)
**Goal**: Centralize state management

#### Tasks
- [ ] **7.1 Create StateManager Class**
  - Define class with memory storage
  - Add comparison methods
  - Set up conflict resolution
  
- [ ] **7.2 Extract Memory Methods**
  - Copy memory methods from PipelineManager
  - Add structured docstrings
  - Test memory operations
  
- [ ] **7.3 Extract History Methods**
  - Copy `_show_preprocessing_history()` → `StateManager.show_history()`
  - Copy `_clear_preprocessing_history()` → `StateManager.clear_history()`
  - Add structured docstrings
  - Test history operations
  
- [ ] **7.4 Extract Comparison Methods**
  - Copy `_pipelines_differ()` → `StateManager.pipelines_differ()`
  - Copy `_show_pipeline_transfer_dialog()` → `StateManager.show_transfer_dialog()`
  - Add structured docstrings
  - Test comparison and conflict resolution
  
- [ ] **7.5 Update All Managers**
  - Inject StateManager into other managers
  - Replace direct memory access with StateManager calls
  - Test full integration

#### Validation Checkpoints
✅ StateManager class created  
✅ All state methods extracted  
✅ Tests pass (state persistence)  
✅ Conflict resolution works  

---

### Phase 8: Optimize Existing Utilities 🔧 (2-3 hours)
**Goal**: Clean up and document existing utils

#### Tasks
- [ ] **8.1 Review pipeline.py**
  - Add structured docstrings to all classes/methods
  - Optimize `PipelineStepWidget` performance
  - Add "Use in" documentation
  
- [ ] **8.2 Review parameter_widgets.py**
  - Add structured docstrings to all widgets
  - Remove unused code
  - Add "Use in" documentation
  
- [ ] **8.3 Review widgets.py**
  - Add structured docstrings
  - Consolidate duplicate code
  - Add "Use in" documentation
  
- [ ] **8.4 Optimize __utils__.py**
  - Remove unused imports
  - Add import documentation
  - Organize imports by category

#### Validation Checkpoints
✅ All utilities documented  
✅ No unused code  
✅ Import structure clean  

---

### Phase 9: Update Main PreprocessPage 🎯 (3-4 hours)
**Goal**: Simplify main class to coordinator role

#### Tasks
- [ ] **9.1 Remove Extracted Methods**
  - Delete all methods moved to managers
  - Keep only coordinator methods
  - Add delegation methods where needed
  
- [ ] **9.2 Update __init__**
  - Create all manager instances
  - Pass dependencies between managers
  - Set up manager interconnections
  
- [ ] **9.3 Update _setup_ui**
  - Use UIBuilder for all UI creation
  - Store widget references
  - Connect signals to manager methods
  
- [ ] **9.4 Update _connect_signals**
  - Connect UI signals to manager methods
  - Connect manager signals to UI updates
  - Document signal flow
  
- [ ] **9.5 Add Public API Methods**
  - Define clean public interface
  - Add delegation to managers
  - Add structured docstrings
  
- [ ] **9.6 Add Documentation**
  - Add class docstring with architecture overview
  - Document manager responsibilities
  - Add usage examples

#### Validation Checkpoints
✅ Main class < 300 lines  
✅ Clear separation of concerns  
✅ Well-documented public API  
✅ Tests pass  

---

### Phase 10: Integration Testing & Documentation 🧪 (4-6 hours)
**Goal**: Verify everything works and document

#### Tasks
- [ ] **10.1 Run Full Integration Tests**
  - Test all features end-to-end
  - Verify no regressions
  - Test edge cases
  - Test error handling
  
- [ ] **10.2 Performance Testing**
  - Measure preview update latency
  - Measure preprocessing time
  - Compare with baseline
  - Optimize if needed
  
- [ ] **10.3 Update Documentation**
  - Update `.docs/pages/preprocess_page.md`
  - Add architecture diagram
  - Document manager responsibilities
  - Add troubleshooting guide
  
- [ ] **10.4 Create Migration Guide**
  - Document breaking changes (if any)
  - Add upgrade instructions
  - Document new extension points
  
- [ ] **10.5 Code Review**
  - Review all new modules
  - Check docstring completeness
  - Verify import structure
  - Check for code smells
  
- [ ] **10.6 Final Cleanup**
  - Remove debug code
  - Remove commented code
  - Format all files
  - Run linter

#### Validation Checkpoints
✅ All tests pass  
✅ No performance regressions  
✅ Documentation complete  
✅ Code review passed  
✅ Ready for merge  

---

## 🎯 Success Metrics

### Quantitative Metrics
- [ ] **File Size**: Main file reduced from 3,068 → <300 lines (-90%)
- [ ] **Module Size**: All modules <500 lines each
- [ ] **Test Coverage**: >80% code coverage
- [ ] **Cyclomatic Complexity**: <10 per method
- [ ] **Documentation**: 100% methods documented

### Qualitative Metrics
- [ ] **Readability**: Each module has single, clear purpose
- [ ] **Maintainability**: New features can be added without touching multiple files
- [ ] **Testability**: Each module can be tested in isolation
- [ ] **Debuggability**: Stack traces point to specific, small modules

---

## 📋 Risk Mitigation

### Identified Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking existing functionality | High | Medium | Phase-by-phase testing, baseline snapshots |
| Performance degradation | Medium | Low | Benchmark before/after, optimize if needed |
| Merge conflicts | Medium | Medium | Work in feature branch, frequent rebases |
| Incomplete refactor | High | Low | Phased approach, validate each phase |
| Lost functionality | High | Low | Comprehensive tests, manual testing |

### Rollback Plan
- Each phase commits separately
- Can rollback to any phase
- Git tags mark stable points
- Baseline snapshot available

---

## 📚 Additional Notes

### Design Principles
1. **Single Responsibility**: Each module does one thing well
2. **Dependency Injection**: Managers receive dependencies via constructor
3. **Interface Segregation**: Small, focused interfaces
4. **Documentation First**: Write docstrings before implementation
5. **Test-Driven**: Write tests before refactoring

### Future Enhancements
- Add plugin system for custom preprocessing methods
- Add undo/redo for pipeline operations
- Add pipeline templates
- Add batch processing UI
- Add advanced visualization options

---

**Version History**:
- v1.0.0 (2025-10-06): Initial comprehensive refactoring plan
