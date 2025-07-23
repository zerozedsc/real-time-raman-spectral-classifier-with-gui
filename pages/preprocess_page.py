import sys
import pandas as pd
import ramanspy as rp
import traceback
from typing import Dict, List, Any, Optional, Tuple
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QGroupBox, QListWidget, QListWidgetItem,
    QStackedWidget, QComboBox, QDoubleSpinBox, QSpinBox, QMessageBox,
    QCheckBox, QSlider, QTextEdit, QScrollArea, QFrame, QSplitter,
    QProgressBar, QApplication, QDialog, QDialogButtonBox, QTreeWidget,
    QTreeWidgetItem, QFormLayout, QTabWidget
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QFont, QIcon

from utils import *
from components.matplotlib_widget import MatplotlibWidget, plot_spectra
from functions.preprocess import PREPROCESSING_REGISTRY, EnhancedRamanPipeline
from functions.data_loader import plot_spectra

class DynamicParameterWidget(QWidget):
    """Dynamic parameter widget that creates UI controls based on parameter info."""
    
    def __init__(self, method_info: Dict[str, Any], saved_params: Dict[str, Any] = None, parent=None):
        super().__init__(parent)
        self.method_info = method_info
        self.saved_params = saved_params or {}
        self.param_widgets = {}
        self.setObjectName("dynamicParameterWidget")
        self._setup_ui()
    
    def _setup_ui(self):
        # Clear any existing layout
        if self.layout():
            while self.layout().count():
                child = self.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            self.layout().deleteLater()
        
        layout = QFormLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        param_info = self.method_info.get("param_info", {})
        default_params = self.method_info.get("default_params", {})
        
        if not param_info:
            # No parameters
            label = QLabel(LOCALIZE("PREPROCESS.no_parameters"))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #666; font-style: italic;")
            layout.addRow(label)
            return
        
        # Add parameters with saved values taking precedence
        for param_name, info in param_info.items():
            # Use saved parameter value if available, otherwise use default
            param_value = self.saved_params.get(param_name, default_params.get(param_name))
            widget = self._create_parameter_widget(param_name, info, param_value)
            if widget:
                label = QLabel(f"{param_name}:")
                label.setToolTip(info.get("description", ""))
                layout.addRow(label, widget)
                self.param_widgets[param_name] = widget
        
        # Force layout update
        self.updateGeometry()
        self.update()
    
    def _create_parameter_widget(self, param_name: str, info: Dict[str, Any], default_value: Any) -> QWidget:
        """Create appropriate widget based on parameter type."""
        param_type = info.get("type", "float")
        
        if param_type == "int":
            widget = QSpinBox()
            range_info = info.get("range", [0, 100])
            widget.setRange(range_info[0], range_info[1])
            if "step" in info:
                widget.setSingleStep(info["step"])
            if default_value is not None:
                widget.setValue(int(default_value))
            return widget
            
        elif param_type == "float":
            widget = QDoubleSpinBox()
            range_info = info.get("range", [0.0, 1.0])
            widget.setRange(range_info[0], range_info[1])
            widget.setDecimals(3)
            if "step" in info:
                widget.setSingleStep(info["step"])
            if default_value is not None:
                widget.setValue(float(default_value))
            return widget
            
        elif param_type == "scientific":
            widget = QDoubleSpinBox()
            range_info = info.get("range", [1e-9, 1e12])
            widget.setRange(range_info[0], range_info[1])
            widget.setDecimals(0)
            widget.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
            if default_value is not None:
                widget.setValue(float(default_value))
            return widget
            
        elif param_type == "choice":
            widget = QComboBox()
            choices = info.get("choices", [])
            widget.addItems(choices)
            if default_value is not None and default_value in choices:
                widget.setCurrentText(str(default_value))
            return widget
            
        elif param_type == "tuple":
            # For tuples like regions
            widget = QLineEdit()
            if default_value is not None:
                if isinstance(default_value, (tuple, list)) and len(default_value) == 2:
                    widget.setText(f"{default_value[0]}, {default_value[1]}")
                else:
                    widget.setText(str(default_value))
            widget.setPlaceholderText(LOCALIZE("PREPROCESS.tuple_format_hint"))
            return widget
            
        elif param_type == "list_int":
            widget = QLineEdit()
            if default_value is not None:
                if isinstance(default_value, list):
                    widget.setText(", ".join(map(str, default_value)))
                else:
                    widget.setText(str(default_value))
            widget.setPlaceholderText(LOCALIZE("PREPROCESS.list_int_format_hint"))
            return widget
            
        elif param_type == "list_float":
            widget = QLineEdit()
            if default_value is not None:
                if isinstance(default_value, list):
                    widget.setText(", ".join(map(str, default_value)))
                else:
                    widget.setText(str(default_value))
            widget.setPlaceholderText(LOCALIZE("PREPROCESS.list_float_format_hint"))
            return widget
            
        else:
            # Default to text input
            widget = QLineEdit()
            if default_value is not None:
                widget.setText(str(default_value))
            return widget
    
    def get_parameters(self) -> Dict[str, Any]:
        """Extract parameters from widgets."""
        params = {}
        param_info = self.method_info.get("param_info", {})
        
        for param_name, widget in self.param_widgets.items():
            info = param_info.get(param_name, {})
            param_type = info.get("type", "float")
            
            try:
                if param_type == "int":
                    params[param_name] = widget.value()
                elif param_type in ["float", "scientific"]:
                    params[param_name] = widget.value()
                elif param_type == "choice":
                    params[param_name] = widget.currentText()
                elif param_type == "tuple":
                    text = widget.text().strip()
                    if text:
                        values = [float(x.strip()) for x in text.split(",")]
                        if len(values) == 2:
                            params[param_name] = tuple(values)
                elif param_type == "list_int":
                    text = widget.text().strip()
                    if text:
                        params[param_name] = [int(x.strip()) for x in text.split(",")]
                elif param_type == "list_float":
                    text = widget.text().strip()
                    if text:
                        values = [float(x.strip()) for x in text.split(",")]
                        params[param_name] = values if values else None
                else:
                    text = widget.text().strip()
                    if text:
                        params[param_name] = text
            except Exception as e:
                create_logs("DynamicParameterWidget", "parameter_extraction",
                           f"Error extracting parameter {param_name}: {e}", status='warning')
                
        return params


class PipelineStep:
    """Represents a step in the preprocessing pipeline."""
    
    def __init__(self, category: str, method: str, params: Dict[str, Any] = None):
        self.category = category
        self.method = method
        self.params = params or {}
        self.enabled = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "method": self.method,
            "params": self.params,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineStep':
        step = cls(data["category"], data["method"], data.get("params", {}))
        step.enabled = data.get("enabled", True)
        return step
    
    def create_instance(self) -> Any:
        """Create the actual preprocessing instance."""
        return PREPROCESSING_REGISTRY.create_method_instance(self.category, self.method, self.params)
    
    def get_display_name(self) -> str:
        """Get display name for the step."""
        category_name = LOCALIZE(f"PREPROCESS.CATEGORY.{self.category.upper()}")
        return f"{category_name} - {self.method}"


class PipelineConfirmationDialog(QDialog):
    """Enhanced dialog to confirm pipeline with detailed information."""
    
    def __init__(self, pipeline_steps: List[PipelineStep], output_name: str, selected_datasets: List[str], parent=None):
        super().__init__(parent)
        self.pipeline_steps = pipeline_steps
        self.output_name = output_name
        self.selected_datasets = selected_datasets
        self.setModal(True)
        self._setup_ui()
        self._apply_styles()
    
    def _setup_ui(self):
        self.setWindowTitle(LOCALIZE("PREPROCESS.CONFIRM_DIALOG.title"))
        self.setMinimumSize(900, 750)
        self.resize(900, 750)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header with summary
        header_frame = QFrame()
        header_frame.setObjectName("confirmDialogHeader")
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 20, 20, 20)
        header_layout.setSpacing(12)
        
        title_label = QLabel(LOCALIZE("PREPROCESS.CONFIRM_DIALOG.header"))
        title_label.setObjectName("confirmDialogTitle")
        header_layout.addWidget(title_label)
        
        # Summary statistics
        summary_text = self._create_summary_text()
        summary_label = QLabel(summary_text)
        summary_label.setObjectName("confirmDialogSummary")
        summary_label.setWordWrap(True)
        header_layout.addWidget(summary_label)
        
        layout.addWidget(header_frame)
        
        # Tabs for different sections
        tabs = QTabWidget()
        tabs.setObjectName("confirmDialogTabs")
        
        # Input datasets tab
        input_tab = self._create_input_tab()
        tabs.addTab(input_tab, LOCALIZE("PREPROCESS.CONFIRM_DIALOG.input_tab"))
        
        # Pipeline tab
        pipeline_tab = self._create_pipeline_tab()
        tabs.addTab(pipeline_tab, LOCALIZE("PREPROCESS.CONFIRM_DIALOG.pipeline_tab"))
        
        layout.addWidget(tabs)
        
        # Buttons
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton(LOCALIZE("COMMON.cancel"))
        cancel_btn.setObjectName("cancelButton")
        cancel_btn.clicked.connect(self.reject)
        
        start_btn = QPushButton(LOCALIZE("PREPROCESS.CONFIRM_DIALOG.start_processing"))
        start_btn.setObjectName("startButton")
        start_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(start_btn)
        
        layout.addWidget(button_frame)
    
    def _apply_styles(self):
        """Apply comprehensive styling to the dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
            }
            
            #confirmDialogHeader {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e3f2fd, stop:1 #bbdefb);
                border: 1px solid #90caf9;
                border-radius: 12px;
            }
            
            #confirmDialogTitle {
                font-size: 18px;
                font-weight: 700;
                color: #1565c0;
                margin: 0px;
            }
            
            #confirmDialogSummary {
                font-size: 14px;
                color: #424242;
                line-height: 1.5;
            }
            
            #confirmDialogTabs {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            
            #confirmDialogTabs::pane {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: white;
                margin-top: -1px;
            }
            
            #confirmDialogTabs::tab-bar {
                alignment: left;
            }
            
            #confirmDialogTabs QTabBar::tab {
                background: #f5f5f5;
                border: 1px solid #e0e0e0;
                padding: 12px 20px;
                margin-right: 2px;
                font-weight: 500;
                min-width: 120px;
            }
            
            #confirmDialogTabs QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
                color: #1976d2;
                font-weight: 600;
            }
            
            #confirmDialogTabs QTabBar::tab:hover:!selected {
                background: #eeeeee;
            }
            
            QListWidget {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background-color: #fafafa;
                padding: 8px;
            }
            
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #f0f0f0;
                background-color: white;
                margin-bottom: 4px;
                border-radius: 4px;
            }
            
            QTreeWidget {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background-color: white;
                alternate-background-color: #f9f9f9;
            }
            
            QTreeWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
            
            QTreeWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
            
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background-color: white;
                padding: 12px;
            }
            
            #startButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4caf50, stop:1 #388e3c);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 14px;
                min-width: 120px;
            }
            
            #startButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #66bb6a, stop:1 #4caf50);
            }
            
            #startButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #388e3c, stop:1 #2e7d32);
            }
            
            #cancelButton {
                background-color: #f5f5f5;
                color: #424242;
                border: 1px solid #e0e0e0;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: 500;
                font-size: 14px;
                min-width: 120px;
                margin-right: 12px;
            }
            
            #cancelButton:hover {
                background-color: #eeeeee;
                border-color: #bdbdbd;
            }
            
            QFrame {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 16px;
            }
        """)
    
    def _create_summary_text(self) -> str:
        """Create localized summary text for the header."""
        total_datasets = len(self.selected_datasets)
        total_steps = len(self.pipeline_steps)
        
        # Estimate processing time (rough calculation)
        estimated_time = total_steps * 2  # 2 seconds per step average
        
        return LOCALIZE("PREPROCESS.CONFIRM_DIALOG.summary_text", 
                       datasets=total_datasets,
                       steps=total_steps, 
                       time=estimated_time,
                       output_name=self.output_name)
    
    def _create_input_tab(self) -> QWidget:
        """Create input datasets tab."""
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        input_layout.setContentsMargins(20, 20, 20, 20)
        input_layout.setSpacing(16)
        
        # Header
        header_label = QLabel(LOCALIZE("PREPROCESS.CONFIRM_DIALOG.selected_datasets_header"))
        header_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #424242; margin-bottom: 8px;")
        input_layout.addWidget(header_label)
        
        # Dataset list
        datasets_list = QListWidget()
        datasets_list.setMaximumHeight(200)
        for dataset in self.selected_datasets:
            item = QListWidgetItem(f"📊 {dataset}")
            datasets_list.addItem(item)
        input_layout.addWidget(datasets_list)
        
        # Info
        info_label = QLabel(LOCALIZE("PREPROCESS.CONFIRM_DIALOG.datasets_info", count=len(self.selected_datasets)))
        info_label.setStyleSheet("font-style: italic; color: #666; font-size: 13px;")
        input_layout.addWidget(info_label)
        
        input_layout.addStretch()
        return input_tab
    
    def _create_pipeline_tab(self) -> QWidget:
        """Create pipeline tab with detailed step information."""
        pipeline_tab = QWidget()
        pipeline_layout = QVBoxLayout(pipeline_tab)
        pipeline_layout.setContentsMargins(20, 20, 20, 20)
        pipeline_layout.setSpacing(16)
        
        # Header
        header_label = QLabel(LOCALIZE("PREPROCESS.CONFIRM_DIALOG.pipeline_steps_header"))
        header_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #424242; margin-bottom: 8px;")
        pipeline_layout.addWidget(header_label)
        
        # Pipeline tree
        pipeline_tree = QTreeWidget()
        pipeline_tree.setHeaderLabels([
            LOCALIZE("PREPROCESS.CONFIRM_DIALOG.step"), 
            LOCALIZE("PREPROCESS.CONFIRM_DIALOG.method"),
            LOCALIZE("PREPROCESS.CONFIRM_DIALOG.purpose"),
            LOCALIZE("PREPROCESS.CONFIRM_DIALOG.parameters")
        ])
        
        for i, step in enumerate(self.pipeline_steps):
            category_display = LOCALIZE(f"PREPROCESS.CATEGORY.{step.category.upper()}")
            purpose = self._get_step_purpose(step.category, step.method)
            params_text = self._format_parameters(step.params) if step.params else LOCALIZE("PREPROCESS.CONFIRM_DIALOG.no_params")
            
            item = QTreeWidgetItem([
                f"{i + 1}. {category_display}",
                step.method,
                purpose,
                params_text
            ])
            pipeline_tree.addTopLevelItem(item)
        
        pipeline_tree.expandAll()
        for i in range(4):
            pipeline_tree.resizeColumnToContents(i)
        pipeline_layout.addWidget(pipeline_tree)
        
        return pipeline_tab
    
    def _get_step_purpose(self, category: str, method: str) -> str:
        """Get localized purpose/benefit of each preprocessing step."""
        return LOCALIZE(f"PREPROCESS.PURPOSE.{category.upper()}")
    
    def _format_parameters(self, params: Dict[str, Any]) -> str:
        """Format parameters for display."""
        if not params:
            return LOCALIZE("PREPROCESS.CONFIRM_DIALOG.default_params")
        
        formatted = []
        for key, value in params.items():
            if isinstance(value, (int, float)):
                formatted.append(f"{key}={value}")
            elif isinstance(value, (tuple, list)):
                formatted.append(f"{key}={value}")
            else:
                formatted.append(f"{key}={str(value)[:20]}")
        
        return ", ".join(formatted[:3])  # Limit to 3 parameters for readability
    
    def _generate_benefits_content(self) -> str:
        """Generate HTML content describing expected benefits."""
        categories_used = set(step.category for step in self.pipeline_steps)
        
        html_content = "<ul style='margin-left: 20px;'>"
        for category in categories_used:
            title = LOCALIZE(f"PREPROCESS.BENEFITS.{category.upper()}.title")
            benefits = LOCALIZE(f"PREPROCESS.BENEFITS.{category.upper()}.benefits")
            
            html_content += f"<li><b>{title}</b><ul>"
            for benefit in benefits:
                html_content += f"<li>{benefit}</li>"
            html_content += "</ul></li>"
        
        html_content += "</ul>"
        
        # Add overall benefits
        overall_benefits = LOCALIZE("PREPROCESS.BENEFITS.overall")
        html_content += f"<br><b>{LOCALIZE('PREPROCESS.BENEFITS.overall_title')}:</b><ul style='margin-left: 20px;'>"
        for benefit in overall_benefits:
            html_content += f"<li>{benefit}</li>"
        html_content += "</ul>"
        
        return html_content
    
    def _calculate_quality_metrics(self) -> List[str]:
        """Calculate expected quality improvements."""
        metrics = []
        categories_used = set(step.category for step in self.pipeline_steps)
        
        for category in categories_used:
            metric = LOCALIZE(f"PREPROCESS.METRICS.{category.upper()}")
            if metric:
                metrics.append(metric)
        
        # Overall metrics
        metrics.extend([
            LOCALIZE("PREPROCESS.METRICS.total_steps", steps=len(self.pipeline_steps)),
            LOCALIZE("PREPROCESS.METRICS.data_integrity"),
            LOCALIZE("PREPROCESS.METRICS.reproducibility")
        ])
        
        return metrics


class PreprocessingThread(QThread):
    """Background thread for preprocessing operations with detailed progress updates."""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    step_completed = Signal(str, int)  # step_name, step_number
    processing_completed = Signal(dict)
    processing_error = Signal(str)
    step_failed = Signal(str, str)  # step_name, error_message
    
    def __init__(self, pipeline_steps: List[PipelineStep], input_dfs: List[pd.DataFrame], 
                 output_name: str, parent=None):
        super().__init__(parent)
        self.pipeline_steps = pipeline_steps
        self.input_dfs = input_dfs
        self.output_name = output_name
        self.is_cancelled = False
        self.failed_steps = []
        
    def run(self):
        """Execute the preprocessing pipeline sequentially with error handling."""
        try:
            create_logs("PreprocessingThread", "start", 
                       f"Starting preprocessing with {len(self.input_dfs)} datasets", status='info')
            
            self.status_updated.emit(LOCALIZE("PREPROCESS.STATUS.preparing_data"))
            self.progress_updated.emit(5)
            
            if self.is_cancelled:
                return
            
            # Merge input data first
            self.status_updated.emit(LOCALIZE("PREPROCESS.STATUS.merging_data"))
            self.progress_updated.emit(10)
            
            # Combine all DataFrames
            try:
                merged_df = pd.concat(self.input_dfs, axis=1)
                create_logs("PreprocessingThread", "data_merge", 
                           f"Merged data shape: {merged_df.shape}", status='info')
            except Exception as e:
                error_msg = f"Error merging input data: {str(e)}"
                create_logs("PreprocessingThread", "merge_error", error_msg, status='error')
                self.processing_error.emit(error_msg)
                return
            
            # Initialize SpectralContainer with better error handling
            try:
                if merged_df.index.name == 'wavenumber':
                    wavenumbers = merged_df.index.values
                    intensities = merged_df.values.T
                else:
                    if 'wavenumber' in merged_df.columns:
                        wavenumbers = merged_df['wavenumber'].values
                        intensities = merged_df.drop('wavenumber', axis=1).values.T
                    else:
                        wavenumbers = merged_df.iloc[:, 0].values
                        intensities = merged_df.iloc[:, 1:].values.T
                
                create_logs("PreprocessingThread", "data_structure", 
                           f"Wavenumbers shape: {wavenumbers.shape}, Intensities shape: {intensities.shape}", 
                           status='info')
                
                spectra = rp.SpectralContainer(intensities, wavenumbers)
                
            except Exception as e:
                error_msg = f"Error creating SpectralContainer: {str(e)}"
                create_logs("PreprocessingThread", "container_error", error_msg, status='error')
                self.processing_error.emit(error_msg)
                return
            
            self.progress_updated.emit(20)
            
            # Process each step sequentially
            total_steps = len(self.pipeline_steps)
            successful_steps = []
            skipped_steps = []
            
            create_logs("PreprocessingThread", "pipeline_start", 
                       f"Processing {total_steps} pipeline steps", status='info')
            
            for i, step in enumerate(self.pipeline_steps):
                if self.is_cancelled:
                    create_logs("PreprocessingThread", "cancelled", "Processing cancelled by user", status='info')
                    return
                
                step_name = step.method
                step_progress_start = 20 + int((i / total_steps) * 60)
                step_progress_end = 20 + int(((i + 1) / total_steps) * 60)
                
                # Check if this step should be skipped (only skip if it's existing AND not enabled)
                if hasattr(step, 'is_existing') and step.is_existing and not step.enabled:
                    self.status_updated.emit(LOCALIZE("PREPROCESS.STATUS.skipping_existing_step", 
                                                    step=step_name, 
                                                    number=i+1, 
                                                    total=total_steps))
                    
                    skipped_steps.append({
                        'step_name': step_name,
                        'step_index': i + 1,
                        'category': step.category,
                        'reason': 'existing_step_disabled'
                    })
                    
                    self.progress_updated.emit(step_progress_end)
                    continue
                
                # Process the step (both new steps and enabled existing steps)
                self.status_updated.emit(LOCALIZE("PREPROCESS.STATUS.processing_step", 
                                                step=step_name, 
                                                number=i+1, 
                                                total=total_steps))
                self.progress_updated.emit(step_progress_start)
                
                try:
                    create_logs("PreprocessingThread", "step_start", 
                               f"Starting step {i+1}/{total_steps}: {step_name} with params: {step.params}", 
                               status='info')
                    
                    # Create preprocessing instance
                    instance = step.create_instance()
                    
                    # Apply preprocessing step
                    pre_shape = spectra.spectral_data.shape
                    pre_axis_shape = spectra.spectral_axis.shape
                    
                    spectra = instance.apply(spectra)
                    
                    post_shape = spectra.spectral_data.shape
                    post_axis_shape = spectra.spectral_axis.shape
                    
                    # Log successful step
                    successful_steps.append({
                        'step_name': step_name,
                        'step_index': i + 1,
                        'category': step.category,
                        'parameters': step.params,
                        'data_change': {
                            'input_shape': pre_shape,
                            'output_shape': post_shape,
                            'axis_input_shape': pre_axis_shape,
                            'axis_output_shape': post_axis_shape
                        }
                    })
                    
                    self.step_completed.emit(step_name, i + 1)
                    self.progress_updated.emit(step_progress_end)
                    
                    create_logs("PreprocessingThread", "step_success",
                            f"Step {i+1}/{total_steps} ({step_name}) completed successfully. "
                            f"Data shape: {pre_shape} -> {post_shape}", 
                            status='info')
                    
                except Exception as e:
                    # Log failed step but continue processing
                    error_msg = f"Step {i+1} ({step_name}) failed: {str(e)}"
                    self.failed_steps.append({
                        'step_name': step_name,
                        'step_index': i + 1,
                        'category': step.category,
                        'error': str(e),
                        'error_type': type(e).__name__
                    })
                    
                    self.step_failed.emit(step_name, str(e))
                    
                    create_logs("PreprocessingThread", "step_error",
                            f"Step {i+1}/{total_steps} ({step_name}) failed: {e}. Continuing with remaining steps.", 
                            status='error')
                    
                    # Continue to next step
                    continue
            
            if self.is_cancelled:
                return
            
            # Finalize results
            self.progress_updated.emit(85)
            self.status_updated.emit(LOCALIZE("PREPROCESS.STATUS.finalizing_results"))
            
            # Create output DataFrame
            try:
                processed_df = pd.DataFrame(
                    spectra.spectral_data.T,
                    index=spectra.spectral_axis,
                    columns=[f"{self.output_name}_{i}" for i in range(spectra.spectral_data.shape[0])]
                )
                processed_df.index.name = 'wavenumber'
                
                create_logs("PreprocessingThread", "output_created", 
                           f"Output DataFrame created with shape: {processed_df.shape}", status='info')
                
            except Exception as e:
                error_msg = f"Error creating output DataFrame: {str(e)}"
                create_logs("PreprocessingThread", "output_error", error_msg, status='error')
                self.processing_error.emit(error_msg)
                return
            
            self.progress_updated.emit(100)
            self.status_updated.emit(LOCALIZE("PREPROCESS.STATUS.completed"))
            
            # Return comprehensive results
            result_data = {
                'processed_df': processed_df,
                'successful_steps': successful_steps,
                'failed_steps': self.failed_steps,
                'skipped_steps': skipped_steps,
                'total_steps': total_steps,
                'success_rate': len(successful_steps) / total_steps if total_steps > 0 else 0,
                'spectra': spectra,
                'original_data': merged_df
            }
            
            create_logs("PreprocessingThread", "completed", 
                       f"Processing completed. Success rate: {result_data['success_rate']:.1%}", 
                       status='info')
            
            self.processing_completed.emit(result_data)
            
        except Exception as e:
            error_msg = f"{LOCALIZE('PREPROCESS.STATUS.error')}: {str(e)}"
            create_logs("PreprocessingThread", "critical_error",
                    f"Critical preprocessing error: {e}\n{traceback.format_exc()}", 
                    status='error')
            self.processing_error.emit(error_msg)
    
    def cancel(self):
        """Cancel the preprocessing operation."""
        self.is_cancelled = True
        create_logs("PreprocessingThread", "cancel_requested", "Cancellation requested", status='info')
        self.quit()
        self.wait()


class FailedStepsDialog(QDialog):
    """Dialog to show detailed information about failed preprocessing steps."""
    
    def __init__(self, failed_steps: List[Dict], successful_steps: List[Dict], parent=None):
        super().__init__(parent)
        self.failed_steps = failed_steps
        self.successful_steps = successful_steps
        self.setModal(True)
        self._setup_ui()
        self._apply_styles()
    
    def _setup_ui(self):
        self.setWindowTitle(LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.title"))
        self.setMinimumSize(700, 500)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Header
        header_label = QLabel(LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.header"))
        header_label.setObjectName("failedDialogHeader")
        layout.addWidget(header_label)
        
        # Summary
        summary_text = LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.summary",
                               failed=len(self.failed_steps),
                               successful=len(self.successful_steps),
                               total=len(self.failed_steps) + len(self.successful_steps))
        summary_label = QLabel(summary_text)
        summary_label.setObjectName("failedDialogSummary")
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)
        
        # Failed steps details
        if self.failed_steps:
            failed_group = QGroupBox(LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.failed_steps_title"))
            failed_layout = QVBoxLayout(failed_group)
            
            failed_tree = QTreeWidget()
            failed_tree.setHeaderLabels([
                LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.step"),
                LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.category"),
                LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.error_type"),
                LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.error_message")
            ])
            
            for step in self.failed_steps:
                item = QTreeWidgetItem([
                    f"{step['step_index']}. {step['step_name']}",
                    LOCALIZE(f"PREPROCESS.CATEGORY.{step['category'].upper()}"),
                    step['error_type'],
                    step['error'][:100] + "..." if len(step['error']) > 100 else step['error']
                ])
                item.setToolTip(3, step['error'])  # Full error message in tooltip
                failed_tree.addTopLevelItem(item)
            
            failed_tree.expandAll()
            for i in range(4):
                failed_tree.resizeColumnToContents(i)
            
            failed_layout.addWidget(failed_tree)
            layout.addWidget(failed_group)
        
        # Successful steps summary
        if self.successful_steps:
            success_group = QGroupBox(LOCALIZE("PREPROCESS.FAILED_STEPS_DIALOG.successful_steps_title"))
            success_layout = QVBoxLayout(success_group)
            
            success_list = QListWidget()
            success_list.setMaximumHeight(150)
            
            for step in self.successful_steps:
                category_key = f"PREPROCESS.CATEGORY.{step['category'].upper()}"
                category_display = LOCALIZE(category_key)
                item_text = f"✓ {step['step_index']}. {step['step_name']} ({category_display})"
                item = QListWidgetItem(item_text)
                success_list.addItem(item)
            
            success_layout.addWidget(success_list)
            layout.addWidget(success_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_btn = QPushButton(LOCALIZE("COMMON.close"))
        close_btn.setObjectName("closeButton")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addWidget(QWidget())  # Spacer
        layout.addLayout(button_layout)
    
    def _apply_styles(self):
        """Apply styling to the failed steps dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
            }
            
            #failedDialogHeader {
                font-size: 18px;
                font-weight: 700;
                color: #d32f2f;
                margin-bottom: 8px;
            }
            
            #failedDialogSummary {
                font-size: 14px;
                color: #424242;
                background-color: #fff3e0;
                border: 1px solid #ffcc02;
                border-radius: 6px;
                padding: 12px;
                margin-bottom: 16px;
            }
            
            QGroupBox {
                font-size: 14px;
                font-weight: 600;
                color: #424242;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 12px;
                background-color: white;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                background-color: white;
            }
            
            QTreeWidget, QListWidget {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background-color: white;
            }
            
            QTreeWidget::item, QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
            
            #closeButton {
                background-color: #1976d2;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: 600;
                min-width: 100px;
            }
            
            #closeButton:hover {
                background-color: #1565c0;
            }
        """)


class PipelineStepWidget(QWidget):
    """Custom widget for displaying pipeline steps with toggle functionality."""
    
    toggled = Signal(int, bool)  # step_index, enabled
    
    def __init__(self, step: 'PipelineStep', step_index: int, parent=None):
        super().__init__(parent)
        self.step = step
        self.step_index = step_index
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # Toggle checkbox for existing steps
        if hasattr(self.step, 'is_existing') and self.step.is_existing:
            self.toggle_checkbox = QCheckBox()
            self.toggle_checkbox.setChecked(False)  # Default: don't reuse existing steps
            self.toggle_checkbox.setToolTip(LOCALIZE("PREPROCESS.toggle_existing_step_tooltip"))
            self.toggle_checkbox.toggled.connect(self._on_toggled)
            layout.addWidget(self.toggle_checkbox)
            
            # Visual indicator for existing steps
            status_label = QLabel("⚙️")
            status_label.setToolTip(LOCALIZE("PREPROCESS.existing_step_indicator"))
            layout.addWidget(status_label)
        else:
            # Visual indicator for new steps
            status_label = QLabel("➕")
            status_label.setToolTip(LOCALIZE("PREPROCESS.new_step_indicator"))
            layout.addWidget(status_label)
        
        # Step name label
        display_name = self.step.get_display_name()
        self.name_label = QLabel(display_name)
        self.name_label.setWordWrap(True)
        layout.addWidget(self.name_label, 1)
        
        # Update appearance based on step status
        self._update_appearance()
    
    def _on_toggled(self, checked: bool):
        """Handle toggle state change."""
        self.toggled.emit(self.step_index, checked)
        self._update_appearance()
    
    def _update_appearance(self):
        """Update visual appearance based on step state."""
        if hasattr(self.step, 'is_existing') and self.step.is_existing:
            if hasattr(self, 'toggle_checkbox') and self.toggle_checkbox.isChecked():
                # Existing step enabled for reuse
                self.name_label.setStyleSheet("""
                    QLabel {
                        color: #2e7d32;
                        font-weight: 500;
                    }
                """)
                self.setToolTip(LOCALIZE("PREPROCESS.existing_step_enabled_tooltip"))
            else:
                # Existing step disabled (default)
                self.name_label.setStyleSheet("""
                    QLabel {
                        color: #757575;
                        font-style: italic;
                    }
                """)
                self.setToolTip(LOCALIZE("PREPROCESS.existing_step_disabled_tooltip"))
        else:
            # New step
            self.name_label.setStyleSheet("""
                QLabel {
                    color: #1976d2;
                    font-weight: 500;
                }
            """)
            self.setToolTip(LOCALIZE("PREPROCESS.new_step_tooltip"))
    
    def is_enabled(self) -> bool:
        """Check if step is enabled for processing."""
        if hasattr(self.step, 'is_existing') and self.step.is_existing:
            return hasattr(self, 'toggle_checkbox') and self.toggle_checkbox.isChecked()
        return True  # New steps are always enabled
    
    def set_enabled(self, enabled: bool):
        """Set the enabled state of the step."""
        if hasattr(self, 'toggle_checkbox'):
            self.toggle_checkbox.setChecked(enabled)


class PreprocessPage(QWidget):
    """Enhanced preprocessing page with dynamic pipeline building and comprehensive parameter controls."""
    showNotification = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("preprocessPage")
        self.processing_thread = None
        self.pipeline_steps: List[PipelineStep] = []
        self.current_step_widget = None
        self._setup_ui()
        self._connect_signals()
        
        # Auto-refresh data when page is shown
        QTimer.singleShot(100, self.load_project_data)

    def _setup_ui(self):
        """Setup the main UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(2)

        # Left panel - Controls
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)

        # Right panel - Parameters and Visualization
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([400, 600])
        main_splitter.setCollapsible(0, False)
        main_splitter.setCollapsible(1, False)

        main_layout.addWidget(main_splitter)

    def _create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setMaximumWidth(450)
        
        layout = QVBoxLayout(left_panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Input datasets section
        input_group = self._create_input_datasets_group()
        layout.addWidget(input_group)

        # Pipeline building section
        pipeline_group = self._create_pipeline_building_group()
        layout.addWidget(pipeline_group)

        # Output configuration section
        output_group = self._create_output_configuration_group()
        layout.addWidget(output_group)

        layout.addStretch()
        return left_panel

    def _create_pipeline_building_group(self) -> QGroupBox:
        """Create pipeline building group with categories."""
        pipeline_group = QGroupBox(LOCALIZE("PREPROCESS.pipeline_building_title"))
        
        layout = QVBoxLayout(pipeline_group)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(12)

        # Category and method selection
        selection_layout = QVBoxLayout()
        
        # Category selection
        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel(LOCALIZE("PREPROCESS.category")))
        self.category_combo = QComboBox()
        categories = PREPROCESSING_REGISTRY.get_categories()
        for category in categories:
            display_name = LOCALIZE(f"PREPROCESS.CATEGORY.{category.upper()}")
            self.category_combo.addItem(display_name, category)
        cat_layout.addWidget(self.category_combo)
        selection_layout.addLayout(cat_layout)
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel(LOCALIZE("PREPROCESS.method")))
        self.method_combo = QComboBox()
        method_layout.addWidget(self.method_combo)
        selection_layout.addLayout(method_layout)
        
        layout.addLayout(selection_layout)

        # Add step button with better icon
        add_step_btn = QPushButton("+")
        add_step_btn.setObjectName("iconButton")
        add_step_btn.setToolTip(LOCALIZE("PREPROCESS.add_step_button"))
        add_step_btn.clicked.connect(self.add_pipeline_step)
        
        # Center the add button
        add_button_layout = QHBoxLayout()
        add_button_layout.addStretch()
        add_button_layout.addWidget(add_step_btn)
        add_button_layout.addStretch()
        layout.addLayout(add_button_layout)

        # Pipeline steps list with custom widgets
        self.pipeline_list = QListWidget()
        self.pipeline_list.setMaximumHeight(250)
        self.pipeline_list.setDragDropMode(QListWidget.InternalMove)
        layout.addWidget(self.pipeline_list)

        # Pipeline control buttons with improved icons
        button_layout = QHBoxLayout()
        
        # Remove button
        remove_btn = QPushButton("×")
        remove_btn.setObjectName("iconButton")
        remove_btn.setToolTip(LOCALIZE("PREPROCESS.remove_step"))
        remove_btn.clicked.connect(self.remove_pipeline_step)
        
        # Clear button
        clear_btn = QPushButton("⌫")
        clear_btn.setObjectName("iconButton")
        clear_btn.setToolTip(LOCALIZE("PREPROCESS.clear_pipeline"))
        clear_btn.clicked.connect(self.clear_pipeline)
        
        # Toggle all existing steps button
        self.toggle_all_btn = QPushButton("↻")
        self.toggle_all_btn.setObjectName("iconButton")
        self.toggle_all_btn.setToolTip(LOCALIZE("PREPROCESS.toggle_all_existing"))
        self.toggle_all_btn.setVisible(False)  # Initially hidden
        self.toggle_all_btn.clicked.connect(self.toggle_all_existing_steps)
        
        # Add buttons with spacing
        button_layout.addWidget(remove_btn)
        button_layout.addSpacing(12)
        button_layout.addWidget(clear_btn)
        button_layout.addSpacing(12)
        button_layout.addWidget(self.toggle_all_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)

        return pipeline_group

    def _create_input_datasets_group(self) -> QGroupBox:
        """Create input datasets selection group."""
        input_group = QGroupBox(LOCALIZE("PREPROCESS.input_datasets_title"))
        
        layout = QVBoxLayout(input_group)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)

        # Refresh button with better styling
        refresh_btn = QPushButton("🔄 " + LOCALIZE("PREPROCESS.refresh_datasets"))
        refresh_btn.setObjectName("refreshButton")
        refresh_btn.setToolTip(LOCALIZE("PREPROCESS.refresh_datasets_tooltip"))
        refresh_btn.clicked.connect(self.load_project_data)
        layout.addWidget(refresh_btn)

        # Dataset list
        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.dataset_list.setMaximumHeight(120)
        layout.addWidget(self.dataset_list)
        
        return input_group

    def _create_output_configuration_group(self) -> QGroupBox:
        """Create output configuration group."""
        output_group = QGroupBox(LOCALIZE("PREPROCESS.output_config_title"))
        
        layout = QVBoxLayout(output_group)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(12)

        # Output name input
        name_layout = QVBoxLayout()
        name_layout.addWidget(QLabel(LOCALIZE("PREPROCESS.output_name_label")))
        
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText(LOCALIZE("PREPROCESS.output_name_placeholder"))
        name_layout.addWidget(self.output_name_edit)
        layout.addLayout(name_layout)

        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label (initially hidden)
        self.status_label = QLabel()
        self.status_label.setVisible(False)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Run button with better styling
        self.run_button = QPushButton("▶ " + LOCALIZE("PREPROCESS.run_button"))
        self.run_button.setObjectName("ctaButton")
        self.run_button.setToolTip(LOCALIZE("PREPROCESS.run_button_tooltip"))
        layout.addWidget(self.run_button)
        
        # Cancel button with better styling (initially hidden)
        self.cancel_button = QPushButton("⏹ " + LOCALIZE("PREPROCESS.cancel_button"))
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setToolTip(LOCALIZE("PREPROCESS.cancel_button_tooltip"))
        self.cancel_button.setVisible(False)
        layout.addWidget(self.cancel_button)

        return output_group

    def _create_right_panel(self) -> QWidget:
        """Create the right panel for parameters and visualization."""
        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        
        layout = QVBoxLayout(right_panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Parameters section
        self.params_group = QGroupBox(LOCALIZE("PREPROCESS.parameters_title"))
        params_layout = QVBoxLayout(self.params_group)
        
        # Scrollable area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)
        
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        
        # Default message
        default_label = QLabel(LOCALIZE("PREPROCESS.set_step_params"))
        default_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        default_label.setFixedHeight(30)
        self.params_layout.addWidget(default_label)
        
        scroll_area.setWidget(self.params_container)
        params_layout.addWidget(scroll_area)
        
        layout.addWidget(self.params_group, 1)

        # Visualization section
        plot_group = QGroupBox(LOCALIZE("PREPROCESS.visualization_title"))
        plot_layout = QVBoxLayout(plot_group)
        
        self.plot_widget = MatplotlibWidget()
        self.plot_widget.setMinimumHeight(400)
        plot_layout.addWidget(self.plot_widget)

        layout.addWidget(plot_group, 2)

        return right_panel

    def _connect_signals(self):
        """Connect UI signals to their handlers."""
        self.category_combo.currentTextChanged.connect(self.update_method_combo)
        
        # Use itemSelectionChanged instead of itemClicked for better responsiveness
        self.dataset_list.itemSelectionChanged.connect(self.preview_raw_data)
        
        self.pipeline_list.currentItemChanged.connect(self.on_pipeline_step_selected)
        self.run_button.clicked.connect(self.run_preprocessing)
        self.cancel_button.clicked.connect(self.cancel_preprocessing)
        
        # Initialize method combo
        self.update_method_combo()

    def update_method_combo(self):
        """Update method combo based on selected category."""
        self.method_combo.clear()
        
        current_category = self.category_combo.currentData()
        if current_category:
            methods = PREPROCESSING_REGISTRY.get_methods_by_category(current_category)
            for method_name, method_info in methods.items():
                self.method_combo.addItem(method_name)

    def load_project_data(self):
        """Load project data and populate the dataset list with preprocessing info."""
        try:
            # Clear existing UI
            self.dataset_list.clear()
            self.plot_widget.clear_plot()
            
            create_logs("PreprocessPage", "data_loading", 
                       f"Loading {len(RAMAN_DATA)} datasets from RAMAN_DATA", status='info')
            
            if RAMAN_DATA:
                # Sort datasets: raw data first, then preprocessed
                dataset_items = []
                
                for dataset_name in sorted(RAMAN_DATA.keys()):
                    item = QListWidgetItem()
                    
                    # Check if this is preprocessed data
                    try:
                        metadata = PROJECT_MANAGER.get_dataframe_metadata(dataset_name)
                        is_preprocessed = metadata and metadata.get('is_preprocessed', False)
                        
                        if is_preprocessed:
                            item.setText(f"🔬 {dataset_name}")
                            item.setToolTip(LOCALIZE("PREPROCESS.preprocessed_data_tooltip", 
                                                   steps=len(metadata.get('preprocessing_pipeline', []))))
                            dataset_items.append((1, item))  # Preprocessed data (sort order 1)
                        else:
                            item.setText(f"📊 {dataset_name}")
                            item.setToolTip(LOCALIZE("PREPROCESS.raw_data_tooltip"))
                            dataset_items.append((0, item))  # Raw data (sort order 0)
                            
                    except Exception as e:
                        # Fallback for metadata access errors
                        item.setText(f"📊 {dataset_name}")
                        item.setToolTip(LOCALIZE("PREPROCESS.raw_data_tooltip"))
                        dataset_items.append((0, item))
                        create_logs("PreprocessPage", "metadata_error", 
                                   f"Error accessing metadata for {dataset_name}: {e}", status='warning')
                    
                    # Ensure the item is selectable
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                
                # Sort and add items (raw data first, then preprocessed)
                dataset_items.sort(key=lambda x: x[0])
                for _, item in dataset_items:
                    self.dataset_list.addItem(item)
                
                # Enable the dataset list
                self.dataset_list.setEnabled(True)
                
                # Auto-select first item if available
                if self.dataset_list.count() > 0:
                    self.dataset_list.setCurrentRow(0)
                    self.showNotification.emit(
                        LOCALIZE("PREPROCESS.datasets_loaded", count=len(RAMAN_DATA)), 
                        "info"
                    )
            else:
                # No data available
                item = QListWidgetItem(LOCALIZE("PREPROCESS.no_datasets_available"))
                item.setFlags(Qt.ItemFlag.NoItemFlags)  # Make it non-selectable
                self.dataset_list.addItem(item)
                self.dataset_list.setEnabled(False)
                self.showNotification.emit(LOCALIZE("PREPROCESS.no_data_warning"), "warning")
                
        except Exception as e:
            create_logs("PreprocessPage", "load_data_error", 
                       f"Critical error loading project data: {e}", status='error')
            self.showNotification.emit(f"Error loading data: {str(e)}", "error")   
    
    def preview_raw_data(self):
        """Preview the selected data and show preprocessing history if applicable."""
        selected_items = self.dataset_list.selectedItems()
        if not selected_items:
            self.plot_widget.clear_plot()
            self._clear_preprocessing_history()
            return
        
        # Handle single selection for preprocessing history
        if len(selected_items) == 1:
            dataset_name = selected_items[0].text().replace('🔬 ', '').replace('📊 ', '')
            try:
                metadata = PROJECT_MANAGER.get_dataframe_metadata(dataset_name)
                
                if metadata and metadata.get('is_preprocessed', False):
                    self._show_preprocessing_history(metadata)
                    # Load existing pipeline for editing
                    self._load_preprocessing_pipeline(metadata.get('preprocessing_pipeline', []))
                else:
                    self._clear_preprocessing_history()
            except Exception as e:
                create_logs("PreprocessPage", "history_error", 
                        f"Error loading preprocessing history for {dataset_name}: {e}", status='warning')
                self._clear_preprocessing_history()
        else:
            self._clear_preprocessing_history()
    
        # Show spectral data
        all_dfs = []
        for item in selected_items:
            dataset_name = item.text().replace('🔬 ', '').replace('📊 ', '')
            if dataset_name in RAMAN_DATA:
                df = RAMAN_DATA[dataset_name]
                all_dfs.append(df)
        
        if all_dfs:
            try:
                combined_df = pd.concat(all_dfs, axis=1)
                fig = plot_spectra(combined_df, title=LOCALIZE("PREPROCESS.spectra_preview_title"))
                self.plot_widget.update_plot(fig)
            except Exception as e:
                create_logs("PreprocessPage", "preview_error", 
                        f"Error previewing data: {e}", status='error')
                self.plot_widget.clear_plot()

    def add_pipeline_step(self):
        """Add a new step to the preprocessing pipeline."""
        current_category = self.category_combo.currentData()
        current_method = self.method_combo.currentText()
        
        if not current_category or not current_method:
            self.showNotification.emit(LOCALIZE("PREPROCESS.select_category_method"), "warning")
            return
        
        # Save current step parameters before adding new step
        if self.current_step_widget and self.pipeline_list.currentItem():
            current_step_index = self.pipeline_list.currentItem().data(Qt.ItemDataRole.UserRole)
            if current_step_index is not None and 0 <= current_step_index < len(self.pipeline_steps):
                current_step = self.pipeline_steps[current_step_index]
                current_step.params = self.current_step_widget.get_parameters()
        
        # Create pipeline step (new steps are not marked as existing)
        step = PipelineStep(current_category, current_method)
        step.is_existing = False  # New step
        self.pipeline_steps.append(step)
        
        # Add to UI list with custom widget
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, len(self.pipeline_steps) - 1)
        self.pipeline_list.addItem(item)
        
        # Create and set custom widget
        step_widget = PipelineStepWidget(step, len(self.pipeline_steps) - 1)
        step_widget.toggled.connect(self.on_step_toggled)
        item.setSizeHint(step_widget.sizeHint())
        self.pipeline_list.setItemWidget(item, step_widget)
        
        # Clear parameter widget before selecting new item
        self._clear_parameter_widget()
        
        # Set current item and force parameter widget update
        self.pipeline_list.setCurrentItem(item)
        QTimer.singleShot(100, lambda: self._show_parameter_widget(step))
        
        self.showNotification.emit(
            LOCALIZE("PREPROCESS.step_added", step=current_method), 
            "info"
        )

    def remove_pipeline_step(self):
        """Remove the selected step from the preprocessing pipeline."""
        current_row = self.pipeline_list.currentRow()
        if current_row >= 0:
            # Remove from pipeline steps
            del self.pipeline_steps[current_row]
            
            # Remove from UI
            self.pipeline_list.takeItem(current_row)
            
            # Update indices in remaining items and their widgets
            for i in range(self.pipeline_list.count()):
                item = self.pipeline_list.item(i)
                item.setData(Qt.ItemDataRole.UserRole, i)
                
                # Update step widget index
                step_widget = self.pipeline_list.itemWidget(item)
                if isinstance(step_widget, PipelineStepWidget):
                    step_widget.step_index = i
            
            # Clear parameter widget if no items left
            if self.pipeline_list.count() == 0:
                self._clear_parameter_widget()
                self.toggle_all_btn.setVisible(False)
            else:
                # Select the next item or previous if at end
                new_row = min(current_row, self.pipeline_list.count() - 1)
                self.pipeline_list.setCurrentRow(new_row)
            
            self.showNotification.emit(LOCALIZE("PREPROCESS.step_removed"), "info")

    def clear_pipeline(self):
        """Clear all pipeline steps."""
        self.pipeline_steps.clear()
        self.pipeline_list.clear()
        self._clear_parameter_widget()
        self.toggle_all_btn.setVisible(False)
        self.showNotification.emit(LOCALIZE("PREPROCESS.pipeline_cleared"), "info")

    def on_step_toggled(self, step_index: int, enabled: bool):
        """Handle step toggle state change."""
        if 0 <= step_index < len(self.pipeline_steps):
            step = self.pipeline_steps[step_index]
            step.enabled = enabled
            
            create_logs("PreprocessPage", "step_toggled", 
                       f"Step {step_index} ({step.method}) {'enabled' if enabled else 'disabled'}", 
                       status='info')

    def toggle_all_existing_steps(self):
        """Toggle all existing steps on/off."""
        # Check current state of existing steps
        existing_enabled_count = 0
        existing_total_count = 0
        
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            step_widget = self.pipeline_list.itemWidget(item)
            
            if isinstance(step_widget, PipelineStepWidget):
                step = step_widget.step
                if hasattr(step, 'is_existing') and step.is_existing:
                    existing_total_count += 1
                    if step_widget.is_enabled():
                        existing_enabled_count += 1
        
        # If more than half are enabled, disable all; otherwise enable all
        new_state = existing_enabled_count < (existing_total_count / 2)
        
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            step_widget = self.pipeline_list.itemWidget(item)
            
            if isinstance(step_widget, PipelineStepWidget):
                step = step_widget.step
                if hasattr(step, 'is_existing') and step.is_existing:
                    step_widget.set_enabled(new_state)
        
        # Update button text
        if new_state:
            self.toggle_all_btn.setText(LOCALIZE("PREPROCESS.disable_all_existing"))
        else:
            self.toggle_all_btn.setText(LOCALIZE("PREPROCESS.enable_all_existing"))
    
     
    
    def on_pipeline_step_selected(self, current, previous):
        """Handle pipeline step selection to show appropriate parameters."""
        # Save parameters from the previously selected step
        if previous and self.current_step_widget:
            prev_step_index = previous.data(Qt.ItemDataRole.UserRole)
            if prev_step_index is not None and 0 <= prev_step_index < len(self.pipeline_steps):
                prev_step = self.pipeline_steps[prev_step_index]
                # Save current parameters to the step
                prev_step.params = self.current_step_widget.get_parameters()
        
        if not current:
            self._clear_parameter_widget()
            return

        step_index = current.data(Qt.ItemDataRole.UserRole)
        if step_index is not None and 0 <= step_index < len(self.pipeline_steps):
            step = self.pipeline_steps[step_index]
            # Show parameter widget with saved parameters
            QTimer.singleShot(50, lambda: self._show_parameter_widget(step))

    def _show_parameter_widget(self, step: PipelineStep):
        """Show parameter widget for the selected step."""
        # Clear existing widget completely
        self._clear_parameter_widget()
        
        # Get method info
        method_info = PREPROCESSING_REGISTRY.get_method_info(step.category, step.method)
        if not method_info:
            return
        
        # Create parameter widget with saved parameters
        self.current_step_widget = DynamicParameterWidget(method_info, step.params)
        self.params_layout.addWidget(self.current_step_widget)
        
        # Update group title
        self.params_group.setTitle(
            LOCALIZE("PREPROCESS.parameters_for_step", step=step.method)
        )

    def _clear_parameter_widget(self):
        """Clear the parameter widget completely."""
        # Remove all widgets from the layout
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Clean up current step widget reference
        if self.current_step_widget:
            self.current_step_widget.deleteLater()
            self.current_step_widget = None
        
        # Reset to default message
        default_label = QLabel(LOCALIZE("PREPROCESS.set_step_params"))
        default_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.params_layout.addWidget(default_label)
        
        self.params_group.setTitle(LOCALIZE("PREPROCESS.parameters_title"))

    def _update_step_parameters(self):
        """Update parameters for the currently selected step."""
        current_item = self.pipeline_list.currentItem()
        if not current_item or not self.current_step_widget:
            return
        
        step_index = current_item.data(Qt.ItemDataRole.UserRole)
        if step_index is not None and 0 <= step_index < len(self.pipeline_steps):
            step = self.pipeline_steps[step_index]
            step.params = self.current_step_widget.get_parameters()

    def run_preprocessing(self):
        """Execute the preprocessing pipeline."""
        # Validate inputs
        selected_items = self.dataset_list.selectedItems()
        if not selected_items:
            self.showNotification.emit(LOCALIZE("PREPROCESS.no_datasets_selected"), "error")
            return
        
        if not self.pipeline_steps:
            self.showNotification.emit(LOCALIZE("PREPROCESS.no_pipeline_steps"), "error")
            return
        
        # Check if any steps are enabled
        enabled_steps = [step for step in self.pipeline_steps if step.enabled]
        if not enabled_steps:
            self.showNotification.emit(LOCALIZE("PREPROCESS.no_enabled_steps"), "error")
            return
        
        output_name = self.output_name_edit.text().strip()
        if not output_name:
            self.showNotification.emit(LOCALIZE("PREPROCESS.no_output_name"), "error")
            return
        
        # Update step enabled states from UI
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            step_widget = self.pipeline_list.itemWidget(item)
            
            if isinstance(step_widget, PipelineStepWidget) and i < len(self.pipeline_steps):
                self.pipeline_steps[i].enabled = step_widget.is_enabled()
        
        # Save current step parameters before processing
        if self.current_step_widget and self.pipeline_list.currentItem():
            current_step_index = self.pipeline_list.currentItem().data(Qt.ItemDataRole.UserRole)
            if current_step_index is not None and 0 <= current_step_index < len(self.pipeline_steps):
                current_step = self.pipeline_steps[current_step_index]
                current_step.params = self.current_step_widget.get_parameters()
        
        # Check if output name already exists
        if output_name in RAMAN_DATA:
            reply = QMessageBox.question(
                self, 
                LOCALIZE("PREPROCESS.overwrite_title"),
                LOCALIZE("PREPROCESS.overwrite_message", name=output_name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Collect input DataFrames - FIX THE DATASET NAME EXTRACTION
        input_dfs = []
        selected_datasets = []
        for item in selected_items:
            # Remove the emoji prefixes to get the actual dataset name
            dataset_name = item.text().replace('🔬 ', '').replace('📊 ', '')
            if dataset_name in RAMAN_DATA:
                input_dfs.append(RAMAN_DATA[dataset_name])
                selected_datasets.append(dataset_name)
        
        if not input_dfs:
            self.showNotification.emit(LOCALIZE("PREPROCESS.no_valid_datasets"), "error")
            create_logs("PreprocessPage", "validation_error", 
                       f"No valid datasets found. Selected items: {[item.text() for item in selected_items]}", 
                       status='error')
            return
        
        # Show confirmation dialog with only enabled steps
        try:
            enabled_pipeline_steps = [step for step in self.pipeline_steps if step.enabled]
            dialog = PipelineConfirmationDialog(enabled_pipeline_steps, output_name, selected_datasets, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
        except Exception as e:
            create_logs("PreprocessPage", "dialog_error", 
                       f"Error showing confirmation dialog: {e}", status='error')
            self.showNotification.emit(f"Dialog error: {str(e)}", "error")
            return
        
        # Start processing with only enabled steps
        self._start_processing_ui()
        
        try:
            # Filter to only enabled steps for processing
            enabled_steps = [step for step in self.pipeline_steps if step.enabled]
            
            # Create and start processing thread
            self.processing_thread = PreprocessingThread(
                enabled_steps,  # Pass only enabled steps
                input_dfs, 
                output_name, 
                self
            )
            
            # Connect signals with enhanced error handling
            self.processing_thread.progress_updated.connect(self.on_progress_updated)
            self.processing_thread.status_updated.connect(self.on_status_updated)
            self.processing_thread.step_completed.connect(self.on_step_completed)
            self.processing_thread.step_failed.connect(self.on_step_failed)
            self.processing_thread.processing_completed.connect(self.on_processing_completed)
            self.processing_thread.processing_error.connect(self.on_processing_error)
            self.processing_thread.finished.connect(self._on_thread_finished)
            
            # Start the thread
            self.processing_thread.start()
            
            create_logs("PreprocessPage", "processing_started", 
                       f"Started preprocessing with {len(input_dfs)} datasets and {len(enabled_steps)} enabled steps", 
                       status='info')
            
            self.showNotification.emit(LOCALIZE("PREPROCESS.processing_started"), "info")
            
        except Exception as e:
            create_logs("PreprocessPage", "thread_creation_error", 
                       f"Error creating processing thread: {e}", status='error')
            self.showNotification.emit(f"Processing error: {str(e)}", "error")
            self._reset_ui_state()

    def _on_thread_finished(self):
        """Handle thread completion (success or failure) with proper cleanup."""
        try:
            create_logs("PreprocessPage", "thread_finished", "Processing thread finished", status='info')
            
            # Ensure thread is properly cleaned up
            if self.processing_thread:
                # Disconnect all signals to prevent memory leaks
                try:
                    self.processing_thread.progress_updated.disconnect()
                    self.processing_thread.status_updated.disconnect()
                    self.processing_thread.step_completed.disconnect()
                    self.processing_thread.step_failed.disconnect()
                    self.processing_thread.processing_completed.disconnect()
                    self.processing_thread.processing_error.disconnect()
                    self.processing_thread.finished.disconnect()
                except:
                    pass  # Ignore if signals are already disconnected
                
                # Wait for thread to finish if still running
                if self.processing_thread.isRunning():
                    self.processing_thread.wait(3000)  # Wait up to 3 seconds
                
                # Clean up thread reference
                self.processing_thread.deleteLater()
                self.processing_thread = None
                
                create_logs("PreprocessPage", "thread_cleanup", "Processing thread cleaned up successfully", status='info')
            
            # Only reset UI if it wasn't already reset by completion handlers
            # This handles cases where the thread finished due to cancellation or error
            if self.progress_bar.isVisible():
                self._reset_ui_state()
                create_logs("PreprocessPage", "ui_reset", "UI state reset after thread completion", status='info')
                
        except Exception as e:
            create_logs("PreprocessPage", "thread_cleanup_error", 
                        f"Error during thread cleanup: {e}", status='error')
            # Force UI reset even if cleanup failed
            self._reset_ui_state()

    def on_step_completed(self, step_name: str, step_number: int):
        """Handle step completion updates."""
        total_steps = len(self.pipeline_steps)
        self.status_label.setText(LOCALIZE("PREPROCESS.STATUS.step_completed", 
                                        step=step_name, 
                                        number=step_number,
                                        total=total_steps))
    
    def on_step_failed(self, step_name: str, error_message: str):
        """Handle individual step failures with logging only."""
        create_logs("PreprocessPage", "step_failure",
                f"Step '{step_name}' failed: {error_message}", status='warning')

    def cancel_preprocessing(self):
        """Cancel the current preprocessing operation."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.cancel()
            self.processing_thread.wait(2000)
            self._reset_ui_state()
            self.showNotification.emit(LOCALIZE("PREPROCESS.processing_cancelled"), "warning")

    def _start_processing_ui(self):
        """Set UI to processing state."""
        self.run_button.setEnabled(False)
        self.run_button.setText("⏳ " + LOCALIZE("PREPROCESS.processing"))
        self.cancel_button.setVisible(True)
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.progress_bar.setValue(0)

    def _reset_ui_state(self):
        """Reset UI to normal state after processing."""
        self.run_button.setEnabled(True)
        self.run_button.setText("▶ " + LOCALIZE("PREPROCESS.run_button"))
        self.cancel_button.setVisible(False)
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.progress_bar.setValue(0)
    
    def on_progress_updated(self, value):
        """Handle progress updates from processing thread."""
        self.progress_bar.setValue(value)

    def on_status_updated(self, status):
        """Handle status updates from processing thread."""
        self.status_label.setText(status)

    def on_processing_completed(self, result_data):
        """Handle completion of preprocessing with detailed feedback."""
        try:
            processed_df = result_data['processed_df']
            successful_steps = result_data['successful_steps']
            failed_steps = result_data['failed_steps']
            total_steps = result_data['total_steps']
            success_rate = result_data['success_rate']
            
            output_name = self.output_name_edit.text().strip()
            
            # Show failed steps dialog if there are failures
            if failed_steps:
                failed_dialog = FailedStepsDialog(failed_steps, successful_steps, self)
                failed_dialog.exec()
            
            # Create comprehensive metadata including preprocessing history
            selected_items = self.dataset_list.selectedItems()
            
            # Save pipeline steps data for future reference
            pipeline_data = []
            for step in self.pipeline_steps:
                pipeline_data.append({
                    "category": step.category,
                    "method": step.method,
                    "params": step.params,
                    "enabled": step.enabled
                })
            
            metadata = {
                "source_datasets": [item.text() for item in selected_items],
                "preprocessing_pipeline": pipeline_data,  # Save the complete pipeline
                "pipeline_summary": {
                    "total_steps": total_steps,
                    "successful_steps": len(successful_steps),
                    "failed_steps": len(failed_steps),
                    "success_rate": success_rate
                },
                "successful_pipeline": successful_steps,
                "failed_pipeline": failed_steps,
                "processing_date": pd.Timestamp.now().isoformat(),
                "processing_type": "spectral_preprocessing",
                "spectral_axis": processed_df.index.tolist(),
                "num_spectra": processed_df.shape[1],
                "data_shape": processed_df.shape,
                "is_preprocessed": True  # Flag to identify preprocessed data
            }
            
            # Save to project
            success = PROJECT_MANAGER.add_dataframe_to_project(output_name, processed_df, metadata)
            
            if success:
                # Show success message with summary
                if failed_steps:
                    message = LOCALIZE("PREPROCESS.processing_success_with_failures", 
                                    name=output_name, 
                                    successful=len(successful_steps),
                                    failed=len(failed_steps),
                                    total=total_steps)
                    notification_type = "warning"
                else:
                    message = LOCALIZE("PREPROCESS.processing_success_complete", 
                                    name=output_name, 
                                    steps=len(successful_steps))
                    notification_type = "success"
                
                self.showNotification.emit(message, notification_type)
                self.load_project_data()
                
                # Select the new dataset
                for i in range(self.dataset_list.count()):
                    if self.dataset_list.item(i).text() == output_name:
                        self.dataset_list.setCurrentRow(i)
                        break
                        
                # Visualize results
                fig = plot_spectra(
                    processed_df, 
                    title=LOCALIZE("PREPROCESS.processed_spectra_title", name=output_name)
                )
                self.plot_widget.update_plot(fig)
                
                # Clear form
                self.clear_pipeline()
                self.output_name_edit.clear()
                
            else:
                self.showNotification.emit(LOCALIZE("PREPROCESS.save_error"), "error")
                
        except Exception as e:
            error_msg = LOCALIZE("PREPROCESS.result_error", error=str(e))
            self.showNotification.emit(error_msg, "error")
            create_logs("PreprocessPage", "result_error", 
                    f"Error processing results: {e}\n{traceback.format_exc()}", 
                    status='error')
        finally:
            self._reset_ui_state()

    def on_processing_error(self, error_msg):
        """Handle processing errors."""
        self.showNotification.emit(error_msg, "error")
        self._reset_ui_state()

    def _show_preprocessing_history(self, metadata: Dict):
        """Show preprocessing history information."""
        pipeline_summary = metadata.get('pipeline_summary', {})
        processing_date = metadata.get('processing_date', '')
        
        if processing_date:
            try:
                date_obj = pd.to_datetime(processing_date)
                formatted_date = date_obj.strftime('%Y-%m-%d %H:%M')
            except:
                formatted_date = processing_date
        else:
            formatted_date = LOCALIZE("COMMON.unknown")
        
        history_text = LOCALIZE("PREPROCESS.history_info",
                            date=formatted_date,
                            total_steps=pipeline_summary.get('total_steps', 0),
                            successful_steps=pipeline_summary.get('successful_steps', 0),
                            failed_steps=pipeline_summary.get('failed_steps', 0))
        
        # Show history in parameters area temporarily
        history_label = QLabel(history_text)
        history_label.setObjectName("preprocessingHistory")
        history_label.setWordWrap(True)
        history_label.setStyleSheet("""
            QLabel#preprocessingHistory {
                background-color: #e8f5e8;
                border: 1px solid #4caf50;
                border-radius: 6px;
                padding: 12px;
                color: #2e7d32;
            }
        """)
        
        # Clear existing parameter widget and show history
        self._clear_parameter_widget()
        self.params_layout.addWidget(history_label)
        self.params_group.setTitle(LOCALIZE("PREPROCESS.preprocessing_history_title"))

    def _clear_preprocessing_history(self):
        """Clear preprocessing history display."""
        self._clear_parameter_widget()

    def _load_preprocessing_pipeline(self, pipeline_data: List[Dict]):
        """Load existing preprocessing pipeline for editing/extension."""
        self.pipeline_steps.clear()
        self.pipeline_list.clear()
        
        has_existing_steps = False
        
        for i, step_data in enumerate(pipeline_data):
            step = PipelineStep(
                step_data['category'],
                step_data['method'], 
                step_data.get('params', {})
            )
            step.enabled = step_data.get('enabled', True)
            
            # Mark as existing step (will be skipped by default unless toggled)
            step.is_existing = True
            has_existing_steps = True
            
            self.pipeline_steps.append(step)
            
            # Add to UI list with custom widget
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.pipeline_list.addItem(item)
            
            # Create and set custom widget
            step_widget = PipelineStepWidget(step, i)
            step_widget.toggled.connect(self.on_step_toggled)
            item.setSizeHint(step_widget.sizeHint())
            self.pipeline_list.setItemWidget(item, step_widget)
        
        # Show toggle all button if there are existing steps
        if has_existing_steps:
            self.toggle_all_btn.setVisible(True)
            self.toggle_all_btn.setText(LOCALIZE("PREPROCESS.enable_all_existing"))

    

