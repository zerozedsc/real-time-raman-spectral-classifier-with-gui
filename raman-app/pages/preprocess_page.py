import sys
import pandas as pd
import ramanspy as rp
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QGroupBox, QListWidget, QListWidgetItem,
    QStackedWidget, QComboBox, QDoubleSpinBox, QSpinBox, QMessageBox,
    QCheckBox, QSlider, QTextEdit, QScrollArea, QFrame, QSplitter
)
from PySide6.QtCore import Qt, Signal

from utils import *
from components.matplotlib_widget import MatplotlibWidget, plot_spectra
from functions.preprocess import *
from functions.data_loader import plot_spectra

class ParameterWidget(QWidget):
    """Base class for parameter editing widgets with enhanced styling."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.layout.setSpacing(12)
        self.layout.setAlignment(Qt.AlignTop)

    def get_params(self) -> dict:
        raise NotImplementedError
    
    def add_parameter_row(self, row: int, label_text: str, widget: QWidget, tooltip: str = None):
        """Helper method to add a parameter row with consistent styling."""
        label = QLabel(label_text)
        label.setStyleSheet("QLabel { font-weight: 500; color: #2c3e50; }")
        if tooltip:
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
        
        self.layout.addWidget(label, row, 0, Qt.AlignLeft)
        self.layout.addWidget(widget, row, 1)

class CropperParamsWidget(ParameterWidget):
    """Enhanced parameter widget for spectral cropping."""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Region start
        self.start_spinbox = QDoubleSpinBox()
        self.start_spinbox.setRange(0, 10000)
        self.start_spinbox.setValue(1050)
        self.start_spinbox.setSuffix(" cm⁻¹")
        self.start_spinbox.setStyleSheet("QDoubleSpinBox { padding: 6px; }")
        self.add_parameter_row(0, LOCALIZE("PREPROCESS.PARAMS.region_start"), self.start_spinbox,
                              "分析に使用する波数範囲の開始点を設定します")

        # Region end
        self.end_spinbox = QDoubleSpinBox()
        self.end_spinbox.setRange(0, 10000)
        self.end_spinbox.setValue(1700)
        self.end_spinbox.setSuffix(" cm⁻¹")
        self.end_spinbox.setStyleSheet("QDoubleSpinBox { padding: 6px; }")
        self.add_parameter_row(1, LOCALIZE("PREPROCESS.PARAMS.region_end"), self.end_spinbox,
                              "分析に使用する波数範囲の終了点を設定します")

        # Info label
        info_label = QLabel("スペクトルを指定した波数範囲にクロップします。\n一般的にはフィンガープリント領域（1000-1700 cm⁻¹）が使用されます。")
        info_label.setStyleSheet("QLabel { color: #666; font-size: 12px; padding: 8px; background: #f8f9fa; border-radius: 4px; }")
        info_label.setWordWrap(True)
        self.layout.addWidget(info_label, 2, 0, 1, 2)

    def get_params(self) -> dict:
        return {'region': (self.start_spinbox.value(), self.end_spinbox.value())}

class SavGolParamsWidget(ParameterWidget):
    """Enhanced parameter widget for Savitzky-Golay filtering."""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Window length
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setRange(3, 99)
        self.window_spinbox.setSingleStep(2)
        self.window_spinbox.setValue(7)
        self.window_spinbox.setStyleSheet("QSpinBox { padding: 6px; }")
        self.add_parameter_row(0, LOCALIZE("PREPROCESS.PARAMS.window_length"), self.window_spinbox,
                              "フィルタリングウィンドウのサイズ（奇数のみ）。大きな値ほど強い平滑化")

        # Polynomial order
        self.poly_spinbox = QSpinBox()
        self.poly_spinbox.setRange(1, 10)
        self.poly_spinbox.setValue(3)
        self.poly_spinbox.setStyleSheet("QSpinBox { padding: 6px; }")
        self.add_parameter_row(1, LOCALIZE("PREPROCESS.PARAMS.polyorder"), self.poly_spinbox,
                              "フィッティングに使用する多項式の次数。通常は2-4が適切")

        # Derivative order
        self.deriv_spinbox = QSpinBox()
        self.deriv_spinbox.setRange(0, 3)
        self.deriv_spinbox.setValue(0)
        self.deriv_spinbox.setStyleSheet("QSpinBox { padding: 6px; }")
        self.add_parameter_row(2, "微分次数", self.deriv_spinbox,
                              "0: 平滑化のみ, 1: 1次微分, 2: 2次微分")

        # Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['interp', 'mirror', 'constant', 'wrap'])
        self.mode_combo.setCurrentText('interp')
        self.mode_combo.setStyleSheet("QComboBox { padding: 6px; }")
        self.add_parameter_row(3, LOCALIZE("PREPROCESS.PARAMS.mode"), self.mode_combo,
                              "境界処理モード")

        # Info label
        info_label = QLabel("Savitzky-Golayフィルタは、スペクトル形状を保持しながらノイズを除去します。\n医療診断では、微細な特徴を保持するため低次多項式が推奨されます。")
        info_label.setStyleSheet("QLabel { color: #666; font-size: 12px; padding: 8px; background: #f8f9fa; border-radius: 4px; }")
        info_label.setWordWrap(True)
        self.layout.addWidget(info_label, 4, 0, 1, 2)

    def get_params(self) -> dict:
        return {
            'window_length': self.window_spinbox.value(), 
            'polyorder': self.poly_spinbox.value(),
            'deriv': self.deriv_spinbox.value(),
            'mode': self.mode_combo.currentText()
        }

class ASPLSParamsWidget(ParameterWidget):
    """Enhanced parameter widget for ASPLS baseline correction."""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Lambda (smoothness)
        self.lam_spinbox = QDoubleSpinBox()
        self.lam_spinbox.setDecimals(0)
        self.lam_spinbox.setRange(1, 1e12)
        self.lam_spinbox.setValue(1e5)
        self.lam_spinbox.setStyleSheet("QDoubleSpinBox { padding: 6px; }")
        self.add_parameter_row(0, LOCALIZE("PREPROCESS.PARAMS.lam"), self.lam_spinbox,
                              "平滑化パラメータ。大きな値ほど滑らかなベースライン")

        # Asymmetry parameter
        self.p_spinbox = QDoubleSpinBox()
        self.p_spinbox.setRange(0.001, 0.999)
        self.p_spinbox.setSingleStep(0.01)
        self.p_spinbox.setValue(0.01)
        self.p_spinbox.setDecimals(3)
        self.p_spinbox.setStyleSheet("QDoubleSpinBox { padding: 6px; }")
        self.add_parameter_row(1, LOCALIZE("PREPROCESS.PARAMS.p"), self.p_spinbox,
                              "非対称パラメータ。小さな値ほど、ピークよりもベースラインを重視")

        # Difference order
        self.diff_order_spinbox = QSpinBox()
        self.diff_order_spinbox.setRange(1, 3)
        self.diff_order_spinbox.setValue(2)
        self.diff_order_spinbox.setStyleSheet("QSpinBox { padding: 6px; }")
        self.add_parameter_row(2, LOCALIZE("PREPROCESS.PARAMS.diff_order"), self.diff_order_spinbox,
                              "差分次数。通常は2が適切")

        # Maximum iterations
        self.max_iter_spinbox = QSpinBox()
        self.max_iter_spinbox.setRange(1, 1000)
        self.max_iter_spinbox.setValue(10)
        self.max_iter_spinbox.setStyleSheet("QSpinBox { padding: 6px; }")
        self.add_parameter_row(3, LOCALIZE("PREPROCESS.PARAMS.max_iter"), self.max_iter_spinbox,
                              "最大反復回数")

        # Tolerance
        self.tol_spinbox = QDoubleSpinBox()
        self.tol_spinbox.setRange(1e-9, 1e-3)
        self.tol_spinbox.setValue(1e-6)
        self.tol_spinbox.setDecimals(9)
        self.tol_spinbox.setStyleSheet("QDoubleSpinBox { padding: 6px; }")
        self.add_parameter_row(4, LOCALIZE("PREPROCESS.PARAMS.tol"), self.tol_spinbox,
                              "収束判定の許容誤差")

        # Info label
        info_label = QLabel("ASPLS（Asymmetric Least Squares Penalized Smoothing）は、\n自動的にベースラインを推定・除去する手法です。\n医療応用では、組織の蛍光背景除去に効果的です。")
        info_label.setStyleSheet("QLabel { color: #666; font-size: 12px; padding: 8px; background: #f8f9fa; border-radius: 4px; }")
        info_label.setWordWrap(True)
        self.layout.addWidget(info_label, 5, 0, 1, 2)

    def get_params(self) -> dict:
        return {
            'lam': self.lam_spinbox.value(), 
            'p': self.p_spinbox.value(),
            'diff_order': self.diff_order_spinbox.value(),
            'max_iter': self.max_iter_spinbox.value(),
            'tol': self.tol_spinbox.value()
        }

class MultiScaleConv1DParamsWidget(ParameterWidget):
    """Enhanced parameter widget for Multi-scale Convolution baseline correction."""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Kernel sizes
        self.kernel_sizes_edit = QLineEdit()
        self.kernel_sizes_edit.setText("5,11,21,41")
        self.kernel_sizes_edit.setStyleSheet("QLineEdit { padding: 6px; }")
        self.add_parameter_row(0, LOCALIZE("PREPROCESS.PARAMS.kernel_sizes"), self.kernel_sizes_edit,
                              "カーネルサイズをカンマ区切りで指定（例: 5,11,21,41）")

        # Weights
        self.weights_edit = QLineEdit()
        self.weights_edit.setText("0.25,0.25,0.25,0.25")
        self.weights_edit.setStyleSheet("QLineEdit { padding: 6px; }")
        self.add_parameter_row(1, LOCALIZE("PREPROCESS.PARAMS.weights"), self.weights_edit,
                              "各スケールの重みをカンマ区切りで指定")

        # Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['reflect', 'constant', 'nearest', 'wrap'])
        self.mode_combo.setCurrentText('reflect')
        self.mode_combo.setStyleSheet("QComboBox { padding: 6px; }")
        self.add_parameter_row(2, LOCALIZE("PREPROCESS.PARAMS.mode"), self.mode_combo,
                              "畳み込み境界処理モード")

        # Iterations
        self.iter_spinbox = QSpinBox()
        self.iter_spinbox.setRange(1, 10)
        self.iter_spinbox.setValue(1)
        self.iter_spinbox.setStyleSheet("QSpinBox { padding: 6px; }")
        self.add_parameter_row(3, LOCALIZE("PREPROCESS.PARAMS.iterations"), self.iter_spinbox,
                              "反復回数")

        # Info label
        info_label = QLabel("マルチスケール畳み込みは、異なるスケールの特徴を同時に捉えて\nベースラインを推定します。複雑な背景パターンに効果的です。")
        info_label.setStyleSheet("QLabel { color: #666; font-size: 12px; padding: 8px; background: #f8f9fa; border-radius: 4px; }")
        info_label.setWordWrap(True)
        self.layout.addWidget(info_label, 4, 0, 1, 2)

    def get_params(self) -> dict:
        kernel_sizes = [int(x.strip()) for x in self.kernel_sizes_edit.text().split(',')]
        weights_text = self.weights_edit.text().strip()
        weights = [float(x.strip()) for x in weights_text.split(',')] if weights_text else None
        
        return {
            'kernel_sizes': kernel_sizes,
            'weights': weights,
            'mode': self.mode_combo.currentText(),
            'iterations': self.iter_spinbox.value()
        }

class SNVParamsWidget(ParameterWidget):
    """Enhanced parameter widget for SNV normalization."""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Info label
        info_label = QLabel("SNV（Standard Normal Variate）正規化は、各スペクトルを\n平均0、標準偏差1に正規化します。\n\n散乱効果やサンプル間の濃度差を補正し、\n医療診断の精度向上に重要な前処理です。")
        info_label.setStyleSheet("QLabel { color: #666; font-size: 12px; padding: 8px; background: #f8f9fa; border-radius: 4px; }")
        info_label.setWordWrap(True)
        self.layout.addWidget(info_label, 0, 0, 1, 2)

    def get_params(self) -> dict:
        return {}

class VectorNormalizeParamsWidget(ParameterWidget):
    """Enhanced parameter widget for Vector normalization."""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Norm type
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(['l1', 'l2', 'max'])
        self.norm_combo.setCurrentText('l2')
        self.norm_combo.setStyleSheet("QComboBox { padding: 6px; }")
        self.add_parameter_row(0, "正規化タイプ", self.norm_combo,
                              "l1: Manhattan距離, l2: Euclidean距離, max: 最大値正規化")

        # Info label
        info_label = QLabel("ベクトル正規化は、スペクトルの強度を統一し、\n相対的な形状特徴に焦点を当てます。\n\n医療診断では、組織厚さの違いを補正する際に有効です。")
        info_label.setStyleSheet("QLabel { color: #666; font-size: 12px; padding: 8px; background: #f8f9fa; border-radius: 4px; }")
        info_label.setWordWrap(True)
        self.layout.addWidget(info_label, 1, 0, 1, 2)

    def get_params(self) -> dict:
        return {'norm': self.norm_combo.currentText()}

class PreprocessPage(QWidget):
    """Enhanced preprocessing page with modern layout and detailed parameter controls."""
    showNotification = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("preprocessPage")
        self.raman_pipeline = RamanPipeline()
        self.available_steps = {
            "スペクトル切り出し": (rp.preprocessing.misc.Cropper, CropperParamsWidget),
            "Savitzky-Golay フィルタ": (rp.preprocessing.denoise.SavGol, SavGolParamsWidget),
            "ASPLS ベースライン補正": (rp.preprocessing.baseline.ASPLS, ASPLSParamsWidget),
            "マルチスケール畳み込み": (MultiScaleConv1D, MultiScaleConv1DParamsWidget),
            "ベクトル正規化": (rp.preprocessing.normalise.Vector, VectorNormalizeParamsWidget),
            "SNV 正規化": (SNV, SNVParamsWidget)
        }
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the main UI layout matching the diagram structure."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Create main splitter for left and right panels
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(2)
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #ddd;
                border: 1px solid #bbb;
            }
            QSplitter::handle:hover {
                background-color: #0078d4;
            }
        """)

        # Left panel
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)

        # Right panel
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([400, 600])  # Left: 400px, Right: 600px
        main_splitter.setCollapsible(0, False)
        main_splitter.setCollapsible(1, False)

        main_layout.addWidget(main_splitter)

    def _create_left_panel(self) -> QWidget:
        """Create the left control panel with enhanced styling."""
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setMaximumWidth(420)
        left_panel.setStyleSheet("""
            #leftPanel {
                background-color: #f8f9fa;
                border-right: 1px solid #dee2e6;
            }
        """)
        
        layout = QVBoxLayout(left_panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Input Datasets Section
        input_group = self._create_input_datasets_group()
        layout.addWidget(input_group)

        # Pipeline Configuration Section
        pipeline_group = self._create_pipeline_configuration_group()
        layout.addWidget(pipeline_group)

        # Output Configuration Section
        output_group = self._create_output_configuration_group()
        layout.addWidget(output_group)

        layout.addStretch()
        return left_panel

    def _create_input_datasets_group(self) -> QGroupBox:
        """Create input datasets selection group."""
        input_group = QGroupBox(LOCALIZE("PREPROCESS.input_datasets_title"))
        input_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: 600;
                color: #2c3e50;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                background-color: #f8f9fa;
            }
        """)
        
        layout = QVBoxLayout(input_group)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)

        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.dataset_list.setMaximumHeight(120)
        self.dataset_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                padding: 4px;
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #f1f3f4;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
        """)
        layout.addWidget(self.dataset_list)
        
        return input_group

    def _create_pipeline_configuration_group(self) -> QGroupBox:
        """Create pipeline configuration group."""
        pipeline_group = QGroupBox(LOCALIZE("PREPROCESS.pipeline_config_title"))
        pipeline_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: 600;
                color: #2c3e50;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                background-color: #f8f9fa;
            }
        """)
        
        layout = QVBoxLayout(pipeline_group)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(12)

        # Step selector and add button
        selector_layout = QHBoxLayout()
        selector_layout.setSpacing(8)
        
        self.step_selector = QComboBox()
        self.step_selector.addItems(self.available_steps.keys())
        self.step_selector.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                font-size: 13px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(assets/icons/chevron-down.svg);
                width: 12px;
                height: 12px;
            }
        """)
        
        add_step_btn = QPushButton(LOCALIZE("PREPROCESS.add_step_button"))
        add_step_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: 500;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        add_step_btn.clicked.connect(self.add_pipeline_step)
        
        selector_layout.addWidget(self.step_selector, 1)
        selector_layout.addWidget(add_step_btn)
        layout.addLayout(selector_layout)

        # Pipeline list
        self.pipeline_list = QListWidget()
        self.pipeline_list.setDragDropMode(QListWidget.InternalMove)
        self.pipeline_list.setMaximumHeight(150)
        self.pipeline_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                padding: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f1f3f4;
                background-color: #f8f9fa;
                margin-bottom: 2px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
                border: 1px solid #1976d2;
            }
        """)
        layout.addWidget(self.pipeline_list)

        # Pipeline control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        remove_step_btn = QPushButton(LOCALIZE("PREPROCESS.remove_step_button"))
        clear_pipeline_btn = QPushButton(LOCALIZE("PREPROCESS.clear_pipeline_button"))
        
        for btn in [remove_step_btn, clear_pipeline_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #ffffff;
                    color: #6c757d;
                    border: 1px solid #ced4da;
                    padding: 6px 12px;
                    border-radius: 4px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #f8f9fa;
                    border-color: #adb5bd;
                }
            """)
        
        remove_step_btn.clicked.connect(self.remove_pipeline_step)
        clear_pipeline_btn.clicked.connect(self.pipeline_list.clear)
        
        button_layout.addWidget(remove_step_btn)
        button_layout.addWidget(clear_pipeline_btn)
        layout.addLayout(button_layout)

        return pipeline_group

    def _create_output_configuration_group(self) -> QGroupBox:
        """Create output configuration group."""
        output_group = QGroupBox(LOCALIZE("PREPROCESS.output_config_title"))
        output_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: 600;
                color: #2c3e50;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                background-color: #f8f9fa;
            }
        """)
        
        layout = QVBoxLayout(output_group)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(12)

        # Output name input
        name_layout = QVBoxLayout()
        name_layout.setSpacing(4)
        
        name_label = QLabel(LOCALIZE("PREPROCESS.output_name_label"))
        name_label.setStyleSheet("QLabel { font-size: 13px; color: #495057; }")
        
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText(LOCALIZE("PREPROCESS.output_name_placeholder"))
        self.output_name_edit.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 13px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #0078d4;
                outline: none;
            }
        """)
        
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.output_name_edit)
        layout.addLayout(name_layout)

        # Run button
        self.run_button = QPushButton(LOCALIZE("PREPROCESS.run_button"))
        self.run_button.setObjectName("ctaButton")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        layout.addWidget(self.run_button)

        return output_group

    def _create_right_panel(self) -> QWidget:
        """Create the right display panel."""
        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        
        layout = QVBoxLayout(right_panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Parameters section
        self.params_group = QGroupBox(LOCALIZE("PREPROCESS.parameters_title"))
        self.params_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: 600;
                color: #2c3e50;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                background-color: white;
            }
        """)
        
        params_layout = QVBoxLayout(self.params_group)
        params_layout.setContentsMargins(12, 16, 12, 12)
        
        # Create scrollable area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
            }
            QScrollBar:vertical {
                background-color: #f1f1f1;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #c1c1c1;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #a8a8a8;
            }
        """)
        
        self.params_stack = QStackedWidget()
        scroll_area.setWidget(self.params_stack)
        params_layout.addWidget(scroll_area)
        
        # Create parameter widgets for all available steps
        self.param_widgets = {}
        for name, (step_class, widget_class) in self.available_steps.items():
            if widget_class:
                widget = widget_class()
                self.params_stack.addWidget(widget)
                self.param_widgets[name] = widget
            else:
                widget = QWidget()
                widget_layout = QVBoxLayout(widget)
                label = QLabel(LOCALIZE("PREPROCESS.no_params_label"))
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
                widget_layout.addWidget(label)
                self.params_stack.addWidget(widget)
                self.param_widgets[name] = widget

        # Visualization section
        plot_group = QGroupBox(LOCALIZE("PREPROCESS.visualization_title"))
        plot_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: 600;
                color: #2c3e50;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                background-color: white;
            }
        """)
        
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(12, 16, 12, 12)
        
        self.plot_widget = MatplotlibWidget()
        self.plot_widget.setMinimumHeight(400)
        self.plot_widget.setStyleSheet("""
            MatplotlibWidget {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: white;
            }
        """)
        plot_layout.addWidget(self.plot_widget)

        # Add sections to right panel
        layout.addWidget(self.params_group, 1)
        layout.addWidget(plot_group, 2)

        return right_panel

    def _connect_signals(self):
        """Connect UI signals to their handlers."""
        self.dataset_list.itemSelectionChanged.connect(self.preview_raw_data)
        self.pipeline_list.currentItemChanged.connect(self.on_pipeline_step_selected)
        self.run_button.clicked.connect(self.run_preprocessing)

    def load_project_data(self):
        """Load project data and populate the dataset list."""
        self.dataset_list.clear()
        self.plot_widget.clear_plot()
        
        if RAMAN_DATA:
            self.dataset_list.addItems(sorted(RAMAN_DATA.keys()))
            if self.dataset_list.count() > 0:
                self.dataset_list.setCurrentRow(0)
        else:
            self.dataset_list.addItem(LOCALIZE("DATA_PACKAGE_PAGE.no_datasets_loaded"))
            self.dataset_list.setEnabled(False)

    def preview_raw_data(self):
        """Preview the selected raw data in the visualization panel."""
        selected_items = self.dataset_list.selectedItems()
        if not selected_items:
            self.plot_widget.clear_plot()
            return
        
        all_dfs = []
        for item in selected_items:
            dataset_name = item.text()
            if dataset_name in RAMAN_DATA:
                df = RAMAN_DATA[dataset_name]
                all_dfs.append(df)
        
        if all_dfs:
            try:
                combined_df = pd.concat(all_dfs, axis=1)
                fig = plot_spectra(combined_df, title=LOCALIZE("PREPROCESS.raw_spectra_title"))
                self.plot_widget.update_plot(fig)
            except Exception as e:
                print(f"Error previewing data: {e}")
                self.plot_widget.clear_plot()

    def add_pipeline_step(self):
        """Add a new step to the preprocessing pipeline."""
        step_name = self.step_selector.currentText()
        item = QListWidgetItem(step_name)
        self.pipeline_list.addItem(item)
        self.pipeline_list.setCurrentItem(item)

    def remove_pipeline_step(self):
        """Remove the selected step from the preprocessing pipeline."""
        current_row = self.pipeline_list.currentRow()
        if current_row >= 0:
            self.pipeline_list.takeItem(current_row)

    def on_pipeline_step_selected(self, current, previous):
        """Handle pipeline step selection to show appropriate parameters."""
        if not current:
            self.params_group.setTitle(LOCALIZE("PREPROCESS.parameters_title"))
            return

        step_name = current.text()
        self.params_group.setTitle(LOCALIZE("PREPROCESS.parameters_for_step", step=step_name))
        
        if step_name in self.param_widgets:
            widget = self.param_widgets[step_name]
            self.params_stack.setCurrentWidget(widget)

    def run_preprocessing(self):
        """Execute the preprocessing pipeline on selected data."""
        # 1. Validate inputs
        selected_items = self.dataset_list.selectedItems()
        if not selected_items:
            self.showNotification.emit(LOCALIZE("NOTIFICATIONS.no_input_data_selected"), "error")
            return
        
        output_name = self.output_name_edit.text().strip()
        if not output_name:
            self.showNotification.emit(LOCALIZE("NOTIFICATIONS.output_name_missing"), "error")
            return
        
        if output_name in RAMAN_DATA:
            reply = QMessageBox.question(
                self, 
                LOCALIZE("PREPROCESS.overwrite_confirm_title"),
                LOCALIZE("PREPROCESS.overwrite_confirm_text", name=output_name),
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # 2. Build pipeline
        pipeline_steps = []
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            step_name = item.text()
            
            if step_name not in self.available_steps:
                continue
                
            step_class, widget_class = self.available_steps[step_name]
            
            params = {}
            if widget_class and step_name in self.param_widgets:
                widget = self.param_widgets[step_name]
                try:
                    params = widget.get_params()
                except Exception as e:
                    self.showNotification.emit(f"パラメータエラー ({step_name}): {str(e)}", "error")
                    return
            
            try:
                pipeline_steps.append(step_class(**params))
            except Exception as e:
                self.showNotification.emit(f"ステップ作成エラー ({step_name}): {str(e)}", "error")
                return

        if not pipeline_steps:
            self.showNotification.emit(LOCALIZE("NOTIFICATIONS.pipeline_is_empty"), "error")
            return

        # 3. Prepare data
        try:
            input_dfs = []
            for item in selected_items:
                dataset_name = item.text()
                if dataset_name in RAMAN_DATA:
                    input_dfs.append(RAMAN_DATA[dataset_name])
            
            if not input_dfs:
                self.showNotification.emit("有効なデータが選択されていません", "error")
                return
            
            # Combine all selected datasets
            combined_df = pd.concat(input_dfs, axis=1)
            
            # 4. Execute pipeline
            result = self.raman_pipeline.preprocess(
                dfs=[combined_df],
                label=output_name,
                preprocessing_steps=pipeline_steps,
                visualize_steps=False
            )
            
            processed_spectra = result['processed']
            
            # 5. Create output DataFrame
            processed_df = pd.DataFrame(
                processed_spectra.spectral_data.T,
                index=processed_spectra.spectral_axis,
                columns=[f"{output_name}_{i}" for i in range(processed_spectra.spectral_data.shape[0])]
            )
            processed_df.index.name = 'wavenumber'

            # 6. Save to project
            metadata = {
                "source_datasets": [item.text() for item in selected_items],
                "pipeline": [step.__class__.__name__ for step in pipeline_steps],
                "parameters": result['preprocessing_info']['parameters_used'] if 'preprocessing_info' in result else {},
                "processing_date": pd.Timestamp.now().isoformat()
            }
            
            success = PROJECT_MANAGER.add_dataframe_to_project(output_name, processed_df, metadata)

            if success:
                self.showNotification.emit(LOCALIZE("NOTIFICATIONS.preprocess_success", name=output_name), "success")
                self.load_project_data()
                # Select the new dataset
                for i in range(self.dataset_list.count()):
                    if self.dataset_list.item(i).text() == output_name:
                        self.dataset_list.setCurrentRow(i)
                        break
            else:
                self.showNotification.emit(LOCALIZE("NOTIFICATIONS.dataset_add_error"), "error")

            # 7. Visualize results
            fig = plot_spectra(processed_df, title=LOCALIZE("PREPROCESS.processed_spectra_title", name=output_name))
            self.plot_widget.update_plot(fig)

        except Exception as e:
            self.showNotification.emit(LOCALIZE("NOTIFICATIONS.preprocess_error", error=str(e)), "error")
            print(f"Preprocessing Error: {e}")
            import traceback
            traceback.print_exc()

