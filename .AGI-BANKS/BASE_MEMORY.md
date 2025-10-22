# Base Memory - AI Agent Knowledge Base

> **Core knowledge and reference system for AI-assisted development**  
> **Last Updated**: October 21, 2025 - Build Troubleshooting Guide (Part 7)

## 🔧 BUILD SYSTEM (October 21, 2025) ⭐ UPDATED

### Critical Build Workflow (Part 7) ⚠️ IMPORTANT

**MUST REBUILD After Spec Changes**:
```powershell
# 1. Edit spec file (add hidden imports, change config)
# 2. REBUILD (don't skip this!)
.\build_portable.ps1 -Clean
# 3. Test NEW exe (not old one!)
.\dist\raman_app\raman_app.exe
```

**Common Mistake**:
- Update spec file ✅
- Forget to rebuild ❌
- Test old exe ❌
- Report "fix doesn't work" ❌

**Verification**:
```powershell
# Check if rebuild needed:
(Get-Item build_scripts\raman_app.spec).LastWriteTime  # Spec modified
(Get-Item dist\raman_app).LastWriteTime  # Dist created
# If spec is newer → MUST REBUILD
```

**PowerShell Syntax**:
```powershell
# ✅ Correct:
.\build_portable.ps1 -Clean
.\build_portable.ps1 -Debug

# ❌ Wrong (Linux style):
.\build_portable.ps1 --clean
.\build_portable.ps1 --debug
```

**Path Handling**:
```powershell
# ✅ Use -LiteralPath for Unicode/special chars:
Get-Item -LiteralPath $path
Get-ChildItem -LiteralPath $path -Recurse

# ❌ Fails with Chinese/Unicode:
Get-Item $path
Get-ChildItem -Path $path
```

**Troubleshooting Reference**: `.docs/building/TROUBLESHOOTING.md`

### Build System Enhancements (Part 6)

**Critical Fixes**:
1. **PowerShell Path Handling**: Use `-LiteralPath` for paths with non-ASCII characters
2. **Scipy Hidden Imports**: Added scipy.stats submodules to prevent runtime errors
3. **Test Validation**: Check both direct and `_internal/` paths for assets
4. **Backup System**: Automatic timestamped backups before cleaning builds

**Backup System**:
```powershell
# Automatically creates backup_YYYYMMDD_HHmmss folder
# Moves (not deletes) existing build/dist before rebuild
# Example: build_backups/backup_20251021_150732/
```

**Scipy Runtime Fix**:
```python
# Required hidden imports for scipy.stats
hiddenimports += [
    'scipy.stats',
    'scipy.stats._stats_py',
    'scipy.stats.distributions',
    'scipy.stats._distn_infrastructure',
]
```

### PyInstaller Configuration
**Windows Builds**: Portable executable and NSIS installer

**Spec Files**:
- `raman_app.spec` (Portable) - 100 lines
- `raman_app_installer.spec` (Installer) - 100 lines

**Key Configuration**:
```python
# Hidden imports for complex packages
hiddenimports = [
    'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',  # Qt6
    'numpy', 'pandas', 'scipy',  # Data processing
    'matplotlib.backends.backend_qt5agg',  # Visualization
    'ramanspy', 'pybaselines',  # Spectroscopy
    'torch', 'sklearn',  # ML (optional)
]

# Data files collection
datas = [
    ('assets/icons', 'assets/icons'),
    ('assets/fonts', 'assets/fonts'),
    ('assets/locales', 'assets/locales'),
]

# Binary files
binaries = [
    ('drivers/atmcd32d.dll', 'drivers'),  # Andor SDK
    ('drivers/atmcd64d.dll', 'drivers'),
]
```

**Build Scripts** (PowerShell):
- `build_portable.ps1` - Automated portable build (190 lines)
- `build_installer.ps1` - Installer staging build (180 lines)

**Test Suite**:
- `test_build_executable.py` - Comprehensive validation (500+ lines)

**NSIS Template**:
- `raman_app_installer.nsi` - Windows installer (100 lines)

**Build Workflow**:
```powershell
# 1. Build portable
.\build_portable.ps1

# 2. Test
python test_build_executable.py

# 3. Build installer (optional)
.\build_installer.ps1

# Output:
# - dist/raman_app/ (50-80 MB)
# - dist_installer/raman_app_installer_staging/ (same size)
# - raman_app_installer.exe (30-50 MB, if NSIS available)
```

**Documentation**:
- `.docs/building/PYINSTALLER_GUIDE.md` (600+ lines)

**Typical Output Sizes**:
- Portable .exe: 50-80 MB
- Total distribution: 60-100 MB
- Installer .exe: 30-50 MB (compressed)

**Testing Validation**:
- ✓ Executable structure
- ✓ Required directories (assets, PySide6, _internal)
- ✓ Asset files (icons, fonts, locales, data)
- ✓ Binary/DLL files
- ✓ Executable launch
- ✓ Performance baseline

### Build Script Fixes (October 21, 2025 - Part 3) ✅

**PowerShell Script Corrections**:
1. **build_portable.ps1**
   - Fixed try-catch block structure
   - Removed problematic emoji characters from code
   - Improved error handling
   - Status: ✅ Syntax correct, runs without errors

2. **build_installer.ps1**
   - Fixed nested block structure
   - Corrected variable interpolation
   - Improved parenthesis matching
   - Status: ✅ All parsing errors resolved

3. **test_build_executable.py**
   - Enhanced error messages with build command hints
   - Better user guidance
   - Status: ✅ Improved user experience

**Verification** ✅:
- [x] PowerShell scripts run without syntax errors
- [x] Error messages provide clear next steps
- [x] All blocks properly closed and structured

### Path Resolution Fix (October 21, 2025 - Part 4) ✅

**Problem**: Build scripts in `build_scripts/` subfolder couldn't locate parent directory files

**Solution Implemented**:

1. **Spec Files** (`raman_app.spec`, `raman_app_installer.spec`)
   ```python
   # Get project root (parent of build_scripts/)
   spec_dir = os.path.dirname(os.path.abspath(__file__))
   project_root = os.path.dirname(spec_dir)
   sys.path.insert(0, project_root)
   ```

2. **PowerShell Scripts** (`build_portable.ps1`, `build_installer.ps1`)
   ```powershell
   # Detect script location and project root
   $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
   $ProjectRoot = Split-Path -Parent $ScriptDir
   
   # Change to project root
   Push-Location $ProjectRoot
   # ... build ...
   Pop-Location  # Restore directory
   ```

3. **Python Test** (`test_build_executable.py`)
   ```python
   script_dir = os.path.dirname(os.path.abspath(__file__))
   project_root = os.path.dirname(script_dir)
   os.chdir(project_root)
   ```

**Benefits**:
- ✅ Scripts work from any directory
- ✅ Relative paths resolve correctly
- ✅ No hardcoded paths needed
- ✅ Proper error handling
- ✅ Directory restored after build

**Status** ✅:
- [x] Spec files detect project root correctly
- [x] Build scripts use Push/Pop-Location
- [x] Test script changes to project root
- [x] All relative paths resolve properly
- [x] Error handling with directory restoration

### PyInstaller CLI Argument Fix (October 21, 2025 - Part 5) ✅

**Problem**: PyInstaller 6.16.0 rejected the unsupported `--buildpath` option used by both build scripts, preventing builds from running.

**Solution**:
```powershell
$BuildArgs = @(
    '--distpath', $OutputDir,
    '--workpath', 'build'
)

$BuildArgs += 'raman_app.spec'
```
- Removed `--buildpath` and ensured all options precede the spec file
- Mirrored the fix in `build_installer.ps1` and appended `raman_app_installer.spec` last
- Cleanup lists now target only directories that PyInstaller actually generates (`build`, `dist`, `build_installer`, `dist_installer`)
- Spec files now compute their location via `Path(sys.argv[0])` fallback when `__file__` is missing

**Benefits**:
- ✅ PyInstaller CLI invocation succeeds without errors
- ✅ Works reliably with PyInstaller 6.16.0 behaviour
- ✅ Consistent command structure between portable and installer builds
- ✅ Clearer BuildArgs assembly with inline comment guidance
- ⚠️ Build currently fails later during module data collection (follow-up fix required for `collect_data_files` usage)

---

## 🎯 Purpose

This document serves as the foundational knowledge base for AI agents working on the Raman Spectroscopy Analysis Application. It provides quick access to essential information and references to detailed documentation.

## 🧪 Testing Standards (Updated October 15, 2025)

### Critical Testing Principles

#### 0. ALWAYS Implement Robust Parameter Type Validation (NEW - October 15, 2025) 🔒
**MANDATORY**: ALL preprocessing methods MUST handle parameter type conversions robustly.

**Why This Matters**:
- UI sliders send FLOATS (1.0, 1.2) but libraries may expect INTEGERS
- User input may come as STRINGS from text fields
- Optional parameters may be None but need proper handling
- Type mismatches cause RuntimeWarnings and failures

**Critical Pattern - Two-Layer Type Validation**:

```python
# Layer 1: Registry Level (functions/preprocess/registry.py)
def create_method_instance(self, category: str, method: str, params: Dict[str, Any] = None):
    """Create instance with robust type conversion."""
    if param_type == "int":
        # CRITICAL: Two-stage conversion handles all cases
        if value is None:
            converted_params[actual_key] = None
        else:
            converted_params[actual_key] = int(float(value))  # float() handles strings, int() converts
    
    elif param_type in ("float", "scientific"):
        if value is None:
            converted_params[actual_key] = None
        else:
            converted_params[actual_key] = float(value)
    
    elif param_type == "choice":
        choices = param_info[actual_key].get("choices", [])
        if choices and isinstance(choices[0], int):
            converted_params[actual_key] = int(float(value))  # Type-aware conversion
        elif choices and isinstance(choices[0], float):
            converted_params[actual_key] = float(value)
        else:
            converted_params[actual_key] = value
    
    elif param_type == "bool":
        if isinstance(value, bool):
            converted_params[actual_key] = value
        elif isinstance(value, str):
            converted_params[actual_key] = value.lower() in ('true', '1', 'yes')
        else:
            converted_params[actual_key] = bool(value)

# Layer 2: Class Level (defensive programming)
class FABCFixed:
    def __init__(self, lam=1e6, scale=None, num_std=3.0, diff_order=2, min_length=2, ...):
        # CRITICAL: Explicit type conversion at class level
        self.lam = float(lam)  # Ensure float
        self.scale = None if scale is None else float(scale)  # None-safe
        self.num_std = float(num_std)
        self.diff_order = int(diff_order)  # MUST be int!
        self.min_length = int(min_length)  # MUST be int!
        self.weights_as_mask = bool(weights_as_mask)
```

**Testing Requirements**:
1. Test with default parameters
2. Test with float parameters (UI slider case): 1.0, 2.0
3. Test with decimal floats (worst case): 1.2, 2.7
4. Test with string parameters: "1", "2.0"
5. Test with None for optional parameters
6. Test actual execution with synthetic data

**Common Failure Patterns**:
- `expected a sequence of integers or a single integer, got '1.0'` → Need int() conversion
- `extrapolate_window must be greater than 0` → Parameter passed as wrong type
- Type conversion only at one layer → Need defensive programming at both layers

#### 1. ALWAYS Verify Library Signatures Before Registry Updates
**MANDATORY PROCESS**: When adding/updating preprocessing methods in registry, MUST verify actual library signatures:

```python
# Example: Verify ASPLS parameters
import inspect
from ramanspy.preprocessing.baseline import ASPLS
sig = inspect.signature(ASPLS.__init__)
print('ASPLS parameters:', list(sig.parameters.keys()))
```

**Why This Matters**:
- ramanspy wrappers may NOT expose all pybaselines parameters
- Documentation can be misleading or outdated
- ASPLS issue: registry had `p_initial` and `asymmetric_coef`, but ramanspy only accepts `lam, diff_order, max_iter, tol, weights, alpha`
- Result: 50% of methods were broken due to incorrect parameter definitions

**Process**:
1. Import the actual class from ramanspy/pybaselines
2. Use `inspect.signature()` to get real parameters
3. Update registry to match EXACT library signature
4. Add parameter aliases ONLY for backward compatibility
5. Test with functional data, not just instantiation

#### 1.1 Bypassing Broken ramanspy Wrappers (FABC Pattern)
**WHEN TO USE**: If ramanspy wrapper has bugs that cannot be fixed by parameter adjustments

**FABC Case Study**: ramanspy.FABC has upstream bug (line 33 passes x_data incorrectly)
- **Problem**: `np.apply_along_axis(self.method, axis, data.spectral_data, x_data)` 
  - Passes x_data to function call, but pybaselines.fabc doesn't accept it in call signature
  - x_data should be in Baseline initialization, not method call
- **Solution**: Create custom wrapper calling pybaselines.api directly

**Pattern**:
```python
# functions/preprocess/fabc_fixed.py
from pybaselines import api
import numpy as np

class FABCFixed:
    """Custom FABC bypassing ramanspy wrapper bug."""
    
    def __init__(self, lam=1e5, scale=0.5, num_std=3, diff_order=2, min_length=2):
        self.lam = lam
        self.scale = scale
        self.num_std = num_std
        self.diff_order = diff_order
        self.min_length = min_length
    
    def _get_baseline_fitter(self, x_data: np.ndarray):
        # CORRECT: x_data in initialization
        return api.Baseline(x_data=x_data)
    
    def _process_spectrum(self, spectrum, x_data):
        fitter = self._get_baseline_fitter(x_data)
        # CORRECT: No x_data in method call
        baseline, params = fitter.fabc(
            data=spectrum,
            lam=self.lam,
            scale=self.scale,
            num_std=self.num_std,
            diff_order=self.diff_order,
            min_length=self.min_length
        )
        return spectrum - baseline
    
    def __call__(self, data, spectral_axis=None):
        # Container-aware wrapper
        # Handles both SpectralContainer and numpy arrays
        # Returns same type as input
```

**Registry Integration**:
```python
# functions/preprocess/registry.py
from .fabc_fixed import FABCFixed

"FABC": {
    "class": FABCFixed,  # Use custom wrapper, not ramanspy.FABC
    "description": "Fixed FABC implementation (bypassing ramanspy bug)",
    # ... parameters
}
```

**Testing Requirements**:
- Test baseline reduction (should be >95% for fluorescence)
- Test with SpectralContainer and numpy arrays
- Verify container type preserved
- Compare with pybaselines direct call (should match exactly)

#### 2. Testing Documentation Location
**CRITICAL**: Documentation location rules:
- **Test summaries/reports (.md)**: Save in `.docs/testing/` folder
- **Test scripts (.py)**: Save in `test_script/` folder
- **Test results (.txt, .json)**: Save in `test_script/results/` folder

**Directory Structure**:
```
.docs/
└── testing/
    ├── session2_comprehensive_testing.md
    ├── session3_functional_testing.md
    └── priority_fixes_progress.md

test_script/
├── test_preprocessing_comprehensive.py
├── test_preprocessing_functional.py
├── results/
│   ├── comprehensive_test_results_20251014_220536.txt
│   └── functional_test_results_20251014_224048.json
└── README.md
```

### Test Script Location
**CRITICAL**: ALL test scripts MUST be saved in the `test_script/` folder with their output results organized in subdirectories.

### Environment Requirements
1. **Always check the Python environment**: This project uses **UV** (not pip, conda, or poetry)
2. **Run scripts with UV**: Use `uv run python <script_name>` to ensure correct environment
3. **Example**:
   ```bash
   cd test_script
   uv run python test_preprocessing_comprehensive.py
   ```

## Testing Standards

### Test Script Organization
- **Scripts Location**: All test scripts MUST be in `test_script/` folder
- **Results Location**: Timestamped outputs in `test_script/results/` subfolder
- **Documentation Location**: Test summaries/reports in `.docs/testing/` folder
- **Environment**: ALWAYS use UV package manager (`uv run python test_script.py`)
- **Outputs**: Save timestamped results with format `test_results_YYYYMMDD_HHMMSS.txt`

### Test Coverage Requirements

#### 1. Structural Testing (Necessary but NOT Sufficient)
- ✓ Method exists in registry
- ✓ Parameters defined correctly
- ✓ Can instantiate class
- ✗ Does NOT guarantee functionality

#### 2. Functional Testing (REQUIRED for Production)
- ✓ Apply methods to realistic synthetic data
- ✓ Validate expected transformations
- ✓ Check output quality (no NaN/Inf)
- ✓ Test complete workflows (pipelines)
- ✓ Measure domain-appropriate metrics

**Key Principle**: "If it instantiates but doesn't work on real data, it's still broken!"

#### 3. Library Signature Verification (MANDATORY for Registry Updates)
- ✓ Use `inspect.signature()` to verify actual parameters
- ✓ Compare with ramanspy wrapper (may differ from pybaselines)
- ✓ Test with actual data after registry updates
- ✓ Document any wrapper limitations

### Current Test Scripts

1. **test_preprocessing_comprehensive.py** (Session 2)
   - Structural validation
   - 40/40 methods pass instantiation
   - Does NOT test functionality

2. **test_preprocessing_functional.py** (Session 3) ⭐ CRITICAL
   - Functional validation with synthetic Raman spectra
   - Tests tissue-realistic data
   - **FOUND**: 50% of methods have functional issues
   - **CRITICAL**: ASPLS parameter bug blocks ALL medical pipelines

### Test Data Generation Standards

For Raman preprocessing tests:
```python
# REQUIRED: Generate realistic synthetic spectra
- Fluorescence baseline (tissue autofluorescence)
- Gaussian peaks at biomolecular positions
- Realistic noise levels
- Occasional cosmic ray spikes
- Multiple tissue types (normal, cancer, inflammation)
```

### Validation Metrics (Category-Specific)

**DO NOT use universal metrics for all methods!**

```python
# Baseline Correction:
✓ Check residual variance reduction
✓ Validate baseline removed
✗ Don't expect SNR increase (removes mean!)

# Normalization:
✓ Check appropriate scaling (SNV: mean=0, std=1)
✓ Validate data range
✗ Don't use vector norm for all types!

# Denoising:
✓ Check high-frequency noise reduction
✓ Measure signal smoothing
✓ SNR improvement valid here

# Cosmic Ray Removal:
✓ Check spike elimination
✓ Validate outlier reduction
✓ SNR improvement expected

# Derivatives:
✓ Check zero-crossings
✓ Validate peak enhancement
✗ Don't expect positive SNR (emphasizes differences!)
```

### Medical Pipeline Testing

**REQUIRED for all medical diagnostic applications**:

1. Test complete workflows end-to-end
2. Use multiple tissue types
3. Validate tissue separability after preprocessing
4. Check each step for cascading failures
5. Document expected outcomes

**Critical Lesson**: One broken step blocks entire medical workflow!

### Known Critical Issues (October 2025)

**BLOCKER BUGS**:
1. **ASPLS Parameter Naming** 🚨
   - Registry uses 'p_initial', users expect 'p'
   - Blocks 3/6 medical pipelines
   - FIX: Accept both parameter names

2. **Method Name Inconsistencies**:
   - IAsLS vs IASLS
   - AirPLS vs AIRPLS
   - ArPLS vs ARPLS
   - FIX: Add aliases

3. **Calibration Methods**:
   - Need optional runtime inputs for testing
   - Cannot use in automated pipelines currently

### Test Execution Standards
1. **Comprehensive Coverage**: Test ALL implemented methods/features, not just a subset
2. **Deep Analysis**: Perform thorough validation including:
   - Parameter definitions and consistency
   - Method instantiation
   - Range validation
   - Error handling
3. **Output Reports**: Generate both text and JSON reports with timestamps
4. **Results Tracking**: Save results to `test_script/test_results_TIMESTAMP.txt`

### Current Test Scripts
- `test_preprocessing_comprehensive.py`: Tests all 40 preprocessing methods
  - Last run: 2025-10-14, Result: 40/40 passed (100%)
  - Output: `test_results_20251014_220536.txt`

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
