from .__utils__ import *

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

