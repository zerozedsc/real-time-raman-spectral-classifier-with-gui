# Base Memory - AI Agent Knowledge Base

> **Core knowledge and reference system for AI-assisted development**  
> **Last Updated**: October 14, 2025 - Data Package Page Major Redesign & Batch Import

## 🎯 Purpose

This document serves as the foundational knowledge base for AI agents working on the Raman Spectroscopy Analysis Application. It provides quick access to essential information and references to detailed documentation.

## ⚠️ CRITICAL: Non-Maximized Window Design Constraint

**Updated**: October 6, 2025 (Evening #2)

### Design Principle
The application **MUST** work well in non-maximized window mode (e.g., 800x600 resolution). All UI sections must be optimized for smaller heights.

### Height Management Guidelines
1. **List Widgets**:
   - Calculate height based on actual item size measurement
   - **Dataset lists**: Show max **4 items** before scroll (**120px** height)
   - **Pipeline lists**: Show max **5 steps** before scroll (215px height)
   - **Item height**: Dataset items ~28px, Pipeline items ~40px

2. **Compact Controls**:
   - Button sizes: **28x28px** for compact headers (not 32x32px)
   - Icon sizes: **14x14px** in compact buttons (not 16x16px)
   - Spacing: **8px** between controls in compact layouts
   - Font sizes: Reduce by 1-2px in compact areas

3. **Section Headers**:
   - Explicit margins: **12px** all sides
   - Minimal spacing: **8px** between elements
   - No extra label containers (combine text + controls directly)

4. **Main Window**:
   - Minimum height: **600px** (for non-maximized mode)
   - Minimum width: **1000px**
   - Default size: 1440x900

5. **Applied in**:
   - ✅ Input Dataset Section: 100-120px (shows 4 items)
   - ✅ Pipeline Construction: 180-215px (shows 5 steps)
   - ✅ Visualization Header: Compact 28px buttons, 8px spacing
   - ✅ Visualization Plot: 300px minimum (reduced from 400px)
   - ✅ Main Window: 600px minimum height
   - ✅ Data Package Page: 12px/16px margins (Oct 14, 2025)

### Example Implementation
```python
# Dataset list - shows 4 items (actual item height ~28px)
list_widget.setMinimumHeight(100)
list_widget.setMaximumHeight(120)

# Pipeline list - shows 5 steps (actual item height ~40px)
self.pipeline_list.setMinimumHeight(180)
self.pipeline_list.setMaximumHeight(215)

# Visualization plot - compact
self.plot_widget.setMinimumHeight(300)  # Reduced from 400

# Main window constraints
self.setMinimumHeight(600)
self.setMinimumWidth(1000)

# Compact buttons
button.setFixedSize(28, 28)
icon = load_svg_icon(path, color, QSize(14, 14))
```

## 📋 Current Development Focus

### Active Tasks (See `.docs/TODOS.md` for details)
1. ✅ **Data Package Page Major Redesign** - COMPLETE (Oct 14, 2025)
   - Modern UI matching preprocessing page design
   - Multiple folder batch import (180x faster for 118 datasets)
   - Automatic metadata loading from JSON files
   - Real-time auto-preview with toggle control
   - Dataset selector for multiple dataset preview
   - Full localization (English + Japanese)
   - 456 lines of new code, production-ready
2. ✅ **Preprocessing Page Enhancements** - COMPLETE (Oct 10, 2025)
   - Dynamic parameter titles show category + step name
   - Gray border selection feedback for pipeline steps
   - Hint buttons added to all major sections
   - Complete pipeline import/export functionality
   - Clean code, no debug statements, production-ready
3. ✅ **Visualization Package Refactoring** - COMPLETE (Oct 1, 2025)
   - Extracted FigureManager to separate file (387 lines)
   - Reduced core.py to 4,405 lines (from 4,812)
   - Full backward compatibility maintained
   - Application tested and working
4. 📋 **RamanVisualizer Modularization** - PLANNED (See `.docs/functions/RAMAN_VISUALIZER_REFACTORING_PLAN.md`)
   - Phase 1-3 extraction (13-18 hours estimated)
   - Deferred to future sprint
5. ✅ **UI Improvements** - COMPLETE
   - Dataset list enhancement (4-6 items visible)
   - Export button styling (green, SVG icon)
   - Preview button width fix
6. ✅ **Export Feature Enhancements** - COMPLETE (Oct 3, 2025)
   - Metadata JSON export alongside datasets
   - Location validation with warning dialog
   - Default location persistence across exports
   - Multiple dataset batch export support

### Recent Completions (October 2025)
- ✅ **Data Package Page Major Redesign (Oct 14, 2025)**:
  - Batch import: 180x faster (30 min → 10 sec for 118 datasets)
  - Auto-metadata loading from JSON files
  - Real-time auto-preview with eye icon toggle
  - Dataset selector dropdown for multiple previews
  - Modern UI with hint buttons matching preprocessing page
  - 10 new localization keys (English + Japanese)
- ✅ **Preprocessing Page Enhancements (Oct 10, 2025)**:
  - Dynamic parameter section titles
  - Visual selection feedback with gray borders
  - Hint buttons for all major sections
  - Complete pipeline import/export system
  - Saved pipelines in projects/{project_name}/pipelines/
  - External pipeline import support
  - Rich pipeline preview in import dialog
- ✅ **Export Functionality (Oct 3, 2025)**:
  - Automatic metadata export in JSON format
  - Smart location validation and warnings
  - Last-used location persistence
  - Multi-dataset batch export capability
  - Comprehensive localization (EN/JA)
- ✅ Visualization package refactoring (visualization.py → visualization/)
- ✅ Removed original visualization.py file
- ✅ Comprehensive testing and validation
- ✅ Documentation reorganization (.docs/ structure)
- ✅ UI improvements (dataset list, export button, preview button)
- ✅ Fixed xlim padding (±50 wavenumber units)
- ✅ Enhanced parameter persistence
- ✅ Removed debug logging the Raman Spectroscopy Analysis Application. It provides quick access to essential information and references to detailed documentation.

## 📚 Documentation Structure

### Primary Documentation Hub: `.docs/`
All detailed documentation is centralized in the `.docs/` folder:
- **Task Management**: `.docs/TODOS.md` - Start here for current tasks
- **Architecture**: `.docs/main.md` - Application structure
- **Pages**: `.docs/pages/` - Page-specific documentation
- **Widgets**: `.docs/widgets/` - Widget system details
- **Functions**: `.docs/functions/` - Function library docs
- **Testing**: `.docs/testing/` - Test documentation

### AI Agent Knowledge Base: `.AGI-BANKS/`
High-level context and patterns (this folder):
- `BASE_MEMORY.md` - This file (quick reference)
- `PROJECT_OVERVIEW.md` - Architecture and design patterns
- `FILE_STRUCTURE.md` - Codebase organization
- `IMPLEMENTATION_PATTERNS.md` - Common coding patterns
- `RECENT_CHANGES.md` - Latest updates and fixes
- `DEVELOPMENT_GUIDELINES.md` - Coding standards
- `HISTORY_PROMPT.md` - Completed work archive

## 🚀 Quick Start

### For New Tasks
```
1. Check .docs/TODOS.md for current task details
2. Review PROJECT_OVERVIEW.md for architecture context
3. Check IMPLEMENTATION_PATTERNS.md for coding patterns
4. Review relevant .docs/ files for implementation details
5. Implement changes following DEVELOPMENT_GUIDELINES.md
6. Update both .docs/ and .AGI-BANKS as needed
```

### For Bug Fixes
```
1. Check RECENT_CHANGES.md for recent modifications
2. Review relevant .docs/pages/ or .docs/widgets/ files
3. Check FILE_STRUCTURE.md for file locations
4. Implement fix following patterns
5. Update TODOS.md and RECENT_CHANGES.md
```

## 🏗️ Project Architecture

### Technology Stack
- **Frontend**: PySide6 (Qt6) - Modern cross-platform GUI
- **Visualization**: matplotlib with Qt backend
- **Data Processing**: pandas, numpy, scipy
- **Configuration**: JSON-based with live reloading
- **Internationalization**: Multi-language support (EN/JA)

### Key Directories
```
raman-app/
├── main.py                    # Application entry point
├── pages/                     # Application pages (UI views)
│   ├── preprocess_page.py     # Main preprocessing interface
│   ├── data_package_page.py   # Data import/management
│   └── home_page.py           # Project management
├── components/                # Reusable UI components
│   ├── app_tabs.py           # Tab navigation
│   ├── toast.py              # Notifications
│   └── widgets/              # Custom widget library
├── functions/                 # Core processing functions
│   ├── data_loader.py        # File loading/parsing
│   ├── visualization/        # 📦 Visualization package (Oct 2025 refactor)
│   │   ├── __init__.py       # Package exports
│   │   ├── core.py           # RamanVisualizer class
│   │   └── figure_manager.py # Figure management
│   ├── preprocess/           # Preprocessing algorithms
│   └── ML.py                 # Machine learning
├── configs/                   # Configuration management
├── assets/                    # Static resources
│   ├── icons/                # SVG icons
│   ├── locales/              # EN/JA translations
│   └── fonts/                # Typography
├── .docs/                     # 📚 Detailed documentation hub
└── .AGI-BANKS/               # 🤖 AI agent knowledge base
```

## 🎨 UI/UX Patterns

### Preprocessing Page Structure
- **Left Panel**: Dataset selection, pipeline building, output config
- **Right Panel**: Parameter controls, visualization
- **Key Features**: Global pipeline memory, parameter persistence, real-time preview

### Color Scheme
- **Primary**: Blue (`#1976d2`) - Active/enabled states
- **Secondary**: Light blue (`#64b5f6`) - Disabled states  
- **Success**: Green (`#2e7d32`)
- **Warning**: Orange (minimal use)
- **Error**: Red (`#dc3545`)
- **Neutral**: Grays for backgrounds and borders

### Widget Patterns
- Enhanced parameter widgets with validation
- Real-time value updates
- Visual feedback for errors
- Tooltip-based help system

## 🔧 Common Patterns

### Data Flow
```
1. Data Loading (data_loader.py)
   ↓
2. RAMAN_DATA global store (utils.py)
   ↓
3. Page UI (e.g., preprocess_page.py)
   ↓
4. Processing Pipeline (functions/preprocess/)
   ↓
5. Visualization (matplotlib_widget.py)
```

### Pipeline System
```
1. User selects category & method
2. PipelineStep created with parameters
3. Steps stored in pipeline_steps list
4. Global memory preserves across dataset switches
5. Real-time preview updates on changes
```

### File Operations
```python
# Import Pattern (data_loader.py)
- Support: CSV, TXT, ASC, Pickle
- Directory or single file
- Wavenumber as index, spectra as columns

# Export Pattern (to be implemented)
- Similar formats as import
- Selected dataset required
- Preserve metadata where possible
```

## � Critical Development Context

### GUI Application Architecture
**Important**: This is a GUI application built with PySide6. Output and debugging require:
1. **Log Files**: Check `logs/` folder for runtime information
   - `PreprocessPage.log` - Preprocessing operations
   - `data_loading.log` - Data import/export
   - `RamanPipeline.log` - Pipeline execution
   - `config.log` - Configuration changes
2. **Terminal Output**: Run `uv run main.py` to see console output
3. **No Direct Print**: GUI apps don't show print() in typical execution

### Environment Management
**Always check current environment before operations:**
- Project uses **uv** package manager (pyproject.toml)
- Commands: `uv run python script.py` or `uv run main.py`
- Virtual environment managed by uv automatically
- Dependencies: See `pyproject.toml` and `uv.lock`

### Documentation Standards (Required)
**All functions, classes, and features MUST include docstrings in this format:**

```python
def function_name(param1, param2):
    """
    Brief description of what the function does.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
    
    Returns:
        type: Description of return value
    
    Raises:
        ExceptionType: When this exception is raised
    """
```

**For classes:**
```python
class ClassName:
    """
    Brief description of the class purpose.
    
    Attributes:
        attr1 (type): Description of attr1
        attr2 (type): Description of attr2
    """
```

Refer to existing code for examples of proper documentation style.

## �📋 Current Development Focus

### Active Tasks (See `.docs/TODOS.md` for details)
1. **Dataset Selection Highlighting** - Improve visual feedback (show 4-6 items with scroll)
2. **Export Button Enhancement** - Use SVG icon, green color, simplified text
3. **Preview Button Sizing** - Dynamic width based on text content
4. **Visualization.py Refactoring** - Convert to package folder for better organization
5. **Testing & Debugging** - Deep analysis and problem identification

### Recent Completions
- Fixed xlim padding (±50 wavenumber units)
- Enhanced parameter persistence
- Removed debug logging
- Organized documentation structure

## 🌐 Internationalization

### Locale System
- **Files**: `assets/locales/en.json`, `assets/locales/ja.json`
- **Usage**: `LOCALIZE("KEY.SUBKEY", param=value)`
- **Pattern**: Hierarchical keys (e.g., `PREPROCESS.input_datasets_title`)

### Adding Localized Strings
```json
// en.json
{
  "PREPROCESS": {
    "export_button": "Export Dataset",
    "export_formats": "Select Format"
  }
}

// ja.json  
{
  "PREPROCESS": {
    "export_button": "データセットをエクスポート",
    "export_formats": "形式を選択"
  }
}
```

## 🧪 Testing Protocol

### Validation Process
1. Create test documentation in `.docs/testing/feature-name/`
2. Implement feature with debug logging
3. Run terminal validation (45-second observation periods)
4. Document results (screenshots, terminal output)
5. Clean up test artifacts
6. Update `.docs/TODOS.md` and `.AGI-BANKS/RECENT_CHANGES.md`

### Test Documentation Structure
```
.docs/testing/
└── feature-name/
    ├── TEST_PLAN.md          # What to test
    ├── RESULTS.md            # Test outcomes
    ├── terminal_output.txt   # Console logs
    └── screenshots/          # Visual evidence
```

## 🔗 Key References

### Must-Read Documentation
- `.docs/TODOS.md` - Current tasks and priorities
- `.docs/pages/preprocess_page.md` - Main UI documentation
- `PROJECT_OVERVIEW.md` - Architecture overview
- `IMPLEMENTATION_PATTERNS.md` - Coding patterns

### External Resources
- PySide6 Documentation: https://doc.qt.io/qtforpython-6/
- matplotlib Qt Backend: https://matplotlib.org/stable/users/explain/backends.html
- pandas API: https://pandas.pydata.org/docs/

## 📝 Documentation Updates

### When to Update
- **Immediately**: Critical bugs, breaking changes
- **Per Feature**: New capabilities, UI changes
- **Per Task**: Task completion, progress updates
- **Weekly**: Review and consolidate changes

### What to Update
1. `.docs/TODOS.md` - Task progress
2. Relevant `.docs/` component files - Implementation details
3. `RECENT_CHANGES.md` - Summary of changes
4. This file (BASE_MEMORY.md) - If core patterns change

## 🎓 Development Guidelines

### Code Quality
- Follow PEP 8 style guide
- Use type hints where beneficial
- Document complex logic
- Keep functions focused and modular

### UI Development
- Maintain consistent styling
- Support both languages (EN/JA)
- Provide clear visual feedback
- Ensure accessibility

### Git Workflow
- Descriptive commit messages
- Reference issues/tasks in commits
- Keep commits focused
- Regular pushes to backup work

## ⚠️ CRITICAL: Project Loading & Memory Management (Oct 8, 2025)

### Project Loading Flow
**ALWAYS follow this exact sequence:**
```python
# In workspace_page.py load_project():
1. Clear all pages → clear_project_data() on each page
2. Load project → PROJECT_MANAGER.load_project(project_path)  # ← CRITICAL
3. Refresh pages → load_project_data() on each page
```

### Common Mistakes to Avoid
❌ **WRONG**: Calling non-existent `set_current_project()`
❌ **WRONG**: Not calling `PROJECT_MANAGER.load_project()` before `load_project_data()`
❌ **WRONG**: Forgetting to clear `RAMAN_DATA` in `clear_project_data()`

✅ **CORRECT**: `PROJECT_MANAGER.load_project(project_path)` populates `RAMAN_DATA`
✅ **CORRECT**: Clear pages → Load project → Refresh pages
✅ **CORRECT**: `RAMAN_DATA.clear()` in every `clear_project_data()` method

### Global State Management
- **RAMAN_DATA**: Dict[str, pd.DataFrame] in `utils.py` (line 16)
- **PROJECT_MANAGER**: Singleton instance in `utils.py` (line 219)
- **load_project()**: Reads pickle files, populates RAMAN_DATA (utils.py line 156)
- **clear_project_data()**: Must clear RAMAN_DATA explicitly

### Pipeline Index Safety
**ALWAYS access full pipeline list, then check if in filtered list:**
```python
# ❌ WRONG (causes index out of range):
current_step = steps[current_row]  # steps = enabled only

# ✅ CORRECT:
if current_row < len(self.pipeline_steps):
    current_step = self.pipeline_steps[current_row]  # Full list
    if current_step in steps:  # Check if enabled
        # Update parameters...
```

### Parameter Type Conversion
**Registry handles these types automatically:**
- `int` → int(value)
- `float` → float(value)
- `scientific` → float(value)  # 1e6 → 1000000.0
- `list` → ast.literal_eval(value)  # "[5,11,21]" → [5,11,21]
- `choice` → type detection from choices[0]

### Ramanspy Library Wrappers
**Created wrappers for buggy ramanspy methods:**
- `functions/preprocess/kernel_denoise.py` - Fixes numpy.uniform → numpy.random.uniform
- `functions/preprocess/background_subtraction.py` - Fixes array comparison issues

## 🆘 Troubleshooting

### Common Issues
1. **Import Errors**: Check sys.path manipulation in files
2. **Locale Missing**: Add keys to both en.json and ja.json
3. **UI Not Updating**: Check signal/slot connections
4. **Preview Issues**: Verify global memory persistence
5. **Project Won't Load**: Check PROJECT_MANAGER.load_project() is called
6. **Memory Persists**: Ensure RAMAN_DATA.clear() in clear_project_data()
7. **Pipeline Errors**: Validate index access patterns (see above)
8. **Parameter Errors**: Check param_info type definitions in registry

### Where to Look
- **UI Issues**: `.docs/pages/` documentation
- **Data Issues**: `.docs/functions/` and `data_loader.py`
- **Widget Issues**: `.docs/widgets/` documentation
- **Style Issues**: `configs/style/stylesheets.py`
- **Project Loading**: `utils.py` (ProjectManager class)
- **Pipeline Issues**: `pages/preprocess_page.py`, `pages/preprocess_page_utils/pipeline.py`
- **Type Conversion**: `functions/preprocess/registry.py`

---

**Version**: 1.1  
**Last Updated**: October 8, 2025  
**Next Review**: After full system testing

**Quick Links**:
- [TODOS](./../.docs/TODOS.md)
- [Project Overview](./PROJECT_OVERVIEW.md)
- [Recent Changes](./RECENT_CHANGES.md)
- [File Structure](./FILE_STRUCTURE.md)
- [Bug Fixes Report](../FINAL_BUG_FIX_REPORT.md)
