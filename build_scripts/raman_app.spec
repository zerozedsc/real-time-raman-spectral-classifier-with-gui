# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Raman Spectroscopy Analysis Application - Portable Executable
Generated: October 21, 2025

This spec creates a standalone portable executable that includes all dependencies.
No installation required - just run the .exe file.

Build command:
    pyinstaller raman_app.spec

Output location:
    dist/raman_app/ (directory with raman_app.exe and dependencies)
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from pathlib import Path
import os
import sys

# Determine spec location even when __file__ is undefined
if '__file__' in globals():
    spec_path = Path(__file__).resolve()
else:
    # When PyInstaller executes the spec, __file__ may be missing.
    candidate = Path(sys.argv[0]).resolve()
    if candidate.name != 'raman_app.spec':
        candidate = Path(os.getcwd()) / 'build_scripts' / 'raman_app.spec'
    spec_path = candidate

spec_dir = spec_path.parent
project_root = spec_dir.parent if spec_dir.name == 'build_scripts' else spec_dir

# Add project root to sys.path so modules can be found (convert to string!)
sys.path.insert(0, str(project_root))

# Collect all necessary data files and modules
datas = []
binaries = []
hiddenimports = []

# ============== DATA FILES ==============
# Assets (icons, fonts, locales) - manually collect since not a package
assets_dir = os.path.join(str(project_root), 'assets')
if os.path.exists(assets_dir):
    datas += [(assets_dir, 'assets')]

# Functions module - manually collect Python files
functions_dir = os.path.join(str(project_root), 'functions')
if os.path.exists(functions_dir):
    datas += [(functions_dir, 'functions')]

# PySide6 plugins and data
datas += collect_data_files('PySide6')

# matplotlib mpl-data
datas += collect_data_files('matplotlib')

# ramanspy data files
try:
    datas += collect_data_files('ramanspy')
except Exception:
    pass

# ============== HIDDEN IMPORTS ==============
# PySide6 components
hiddenimports += [
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'PySide6.QtOpenGL',
    'PySide6.QtPrintSupport',
    'shiboken6',
]

# Data science libraries
hiddenimports += [
    'numpy',
    'pandas',
    'scipy',
    'scipy.integrate',
    'scipy.signal',
    'scipy.interpolate',
    'scipy.optimize',
    'scipy.special',
    'scipy.stats',
    'scipy.stats._stats_py',
    'scipy.stats.distributions',
    'scipy.stats._distn_infrastructure',
    'sklearn',
    'sklearn.preprocessing',
    'sklearn.decomposition',
    'sklearn.linear_model',
    'sklearn.ensemble',
]

# Visualization and processing
hiddenimports += [
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.figure',
    'matplotlib.widgets',
    'seaborn',
    'PIL',
    'imageio',
]

# Spectroscopy and analysis
hiddenimports += [
    'ramanspy',
    'ramanspy.preprocessing',
    'ramanspy.preprocessing.normalise',
    'ramanspy.preprocessing.noise',
    'ramanspy.preprocessing.baseline',
    'ramanspy.preprocessing.spectral_range',
    'ramanspy.preprocessing.smoothing',
    'renishawwire',
    'pybaselines',
    'pybaselines.api',
    'pybaselines.optimizers',
]

# ML and deep learning
hiddenimports += [
    'torch',
    'torch.nn',
    'torch.optim',
    'skl2onnx',
    'onnx',
    'sklearn.preprocessing',
]

# Additional utilities
hiddenimports += [
    'requests',
    'tqdm',
    'cloudpickle',
    'joblib',
    'pydantic',
    'pydantic_core',
    'cryptography',
]

# Collect all submodules from key packages (wrapped in try-except for robustness)
try:
    hiddenimports += collect_submodules('ramanspy')
except Exception as e:
    print(f"Warning: Could not collect ramanspy submodules: {e}")

try:
    hiddenimports += collect_submodules('pybaselines')
except Exception as e:
    print(f"Warning: Could not collect pybaselines submodules: {e}")

# ============== BINARIES ==============
# Include DLL files for Andor SDK
dll_path = os.path.join(str(project_root), 'drivers')
if os.path.exists(dll_path):
    binaries += [
        (os.path.join(dll_path, 'atmcd32d.dll'), 'drivers'),
        (os.path.join(dll_path, 'atmcd64d.dll'), 'drivers'),
    ]

# ============== ANALYSIS ==============
a = Analysis(
    [os.path.join(str(project_root), 'main.py')],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludedimports=['tkinter', 'matplotlib.backends.backend_tk'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# ============== PYZ ==============
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ============== EXE ==============
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='raman_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window (GUI only)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if needed: 'assets/icon.ico'
)

# ============== BUNDLE ==============
# Creates dist/raman_app/ directory with raman_app.exe and all dependencies
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='raman_app',
)
