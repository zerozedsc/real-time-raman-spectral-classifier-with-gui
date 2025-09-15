import sys
import os
# Add components to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from components.widgets import *
from .preprocess_page_utils import *


class PreprocessPage(QWidget):
    """Enhanced preprocessing page with dynamic pipeline building and comprehensive parameter controls."""
    showNotification = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("preprocessPage")
        self.processing_thread = None
        self.pipeline_steps: List[PipelineStep] = []
        self.current_step_widget = None
        
        # Real-time preview attributes
        self.preview_data = None  # Current dataset for preview
        self.original_data = None  # Original unprocessed data
        self.processed_data = None  # Current processed data
        self.preview_cache = {}  # Cache for processed steps
        self.preview_timer = QTimer()  # Debounce timer for parameter changes
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._update_preview)
        self.preview_enabled = True  # Toggle for preview functionality
        
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
        
        # Preview controls - Enhanced UI
        preview_controls = QHBoxLayout()
        preview_controls.setSpacing(15)
        
        # Preview mode toggle with enhanced styling
        preview_toggle_container = QHBoxLayout()
        preview_toggle_container.setSpacing(8)
        
        # Create toggle button with eye icons
        self.preview_toggle_btn = QPushButton()
        self.preview_toggle_btn.setCheckable(True)
        self.preview_toggle_btn.setChecked(True)
        self.preview_toggle_btn.setFixedSize(120, 32)
        self.preview_toggle_btn.setToolTip(LOCALIZE("PREPROCESS.real_time_preview_tooltip"))
        
        # Load eye icons
        self.eye_open_icon = load_svg_icon("assets/icons/eye-open.svg", "#2c3e50", QSize(16, 16))
        self.eye_close_icon = load_svg_icon("assets/icons/eye-close.svg", "#7f8c8d", QSize(16, 16))
        
        # Set initial state
        self._update_preview_button_state(True)
        
        self.preview_toggle_btn.toggled.connect(self._toggle_preview_mode)
        preview_toggle_container.addWidget(self.preview_toggle_btn)
        
        preview_controls.addLayout(preview_toggle_container)
        
        # Manual refresh button with SVG icon
        self.manual_refresh_btn = QPushButton()
        reload_icon = load_svg_icon(ICON_PATHS["reload"], "#7f8c8d", QSize(16, 16))
        self.manual_refresh_btn.setIcon(reload_icon)
        self.manual_refresh_btn.setIconSize(QSize(16, 16))
        self.manual_refresh_btn.setFixedSize(32, 32)
        self.manual_refresh_btn.setToolTip(LOCALIZE("PREPROCESS.manual_refresh_tooltip"))
        self.manual_refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #ecf0f1;
                border: 2px solid #bdc3c7;
                border-radius: 16px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #d5dbdb;
                border-color: #85929e;
            }
            QPushButton:pressed {
                background-color: #aeb6bf;
            }
        """)
        self.manual_refresh_btn.clicked.connect(self._manual_refresh_preview)
        preview_controls.addWidget(self.manual_refresh_btn)
        
        # Manual focus button with SVG icon
        self.manual_focus_btn = QPushButton()
        focus_icon = load_svg_icon("assets/icons/focus-horizontal-round.svg", "#7f8c8d", QSize(16, 16))
        self.manual_focus_btn.setIcon(focus_icon)
        self.manual_focus_btn.setIconSize(QSize(16, 16))
        self.manual_focus_btn.setFixedSize(32, 32)
        self.manual_focus_btn.setToolTip("Manual Focus - Focus on signal regions")
        self.manual_focus_btn.setStyleSheet("""
            QPushButton {
                background-color: #ecf0f1;
                border: 2px solid #bdc3c7;
                border-radius: 16px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #d5dbdb;
                border-color: #85929e;
            }
            QPushButton:pressed {
                background-color: #aeb6bf;
            }
        """)
        self.manual_focus_btn.clicked.connect(self._manual_focus)
        preview_controls.addWidget(self.manual_focus_btn)
        
        preview_controls.addStretch()
        
        # Enhanced status indicator with label
        status_container = QHBoxLayout()
        status_container.setSpacing(5)
        
        status_label = QLabel(LOCALIZE("PREPROCESS.preview_status"))
        status_label.setStyleSheet("font-size: 11px; color: #7f8c8d; font-weight: bold;")
        status_container.addWidget(status_label)
        
        self.preview_status = QLabel("●")
        self.preview_status.setStyleSheet("color: #27ae60; font-size: 14px;")
        self.preview_status.setToolTip(LOCALIZE("PREPROCESS.preview_status_ready"))
        status_container.addWidget(self.preview_status)
        
        self.preview_status_text = QLabel("Ready")
        self.preview_status_text.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
        status_container.addWidget(self.preview_status_text)
        
        preview_controls.addLayout(status_container)
        
        plot_layout.addLayout(preview_controls)
        
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
        
        # Connect drag and drop reordering to automatic preview update
        self.pipeline_list.model().rowsMoved.connect(self._on_pipeline_reordered)
        
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
            self._clear_default_output_name()
            # Clear preview data
            self.original_data = None
            self.processed_data = None
            return
        
        # Set default output name based on first selected dataset
        first_dataset_name = selected_items[0].text().replace('🔬 ', '').replace('📊 ', '')
        self._set_default_output_name(first_dataset_name)
        
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
    
        # Show spectral data and store for preview
        all_dfs = []
        for item in selected_items:
            dataset_name = item.text().replace('🔬 ', '').replace('📊 ', '')
            if dataset_name in RAMAN_DATA:
                df = RAMAN_DATA[dataset_name]
                all_dfs.append(df)
        
        if all_dfs:
            try:
                combined_df = pd.concat(all_dfs, axis=1)
                
                # Store original data for preview system as DataFrame to preserve index/wavenumbers
                self.original_data = combined_df
                
                # Show data with current preview mode
                if self.preview_enabled and self.pipeline_steps:
                    # Trigger preview update
                    self._schedule_preview_update()
                else:
                    # Show original data without auto-focus (no pipeline steps yet)
                    fig = plot_spectra(combined_df, title=LOCALIZE("PREPROCESS.spectra_preview_title"), auto_focus=False)
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
        
        # Trigger automatic preview update
        self._schedule_preview_update()
        
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
            
            # Trigger automatic preview update
            self._schedule_preview_update()
            
            self.showNotification.emit(LOCALIZE("PREPROCESS.step_removed"), "info")

    def clear_pipeline(self):
        """Clear all pipeline steps."""
        self.pipeline_steps.clear()
        self.pipeline_list.clear()
        self._clear_parameter_widget()
        self.toggle_all_btn.setVisible(False)
        
        # Trigger automatic preview update to show original data
        if self.preview_enabled:
            self._schedule_preview_update()
        
        self.showNotification.emit(LOCALIZE("PREPROCESS.pipeline_cleared"), "info")

    def on_step_toggled(self, step_index: int, enabled: bool):
        """Handle step toggle state change."""
        if 0 <= step_index < len(self.pipeline_steps):
            step = self.pipeline_steps[step_index]
            step.enabled = enabled
            
            # Trigger automatic preview update
            self._schedule_preview_update()
            
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

    def _get_data_wavenumber_range(self) -> tuple:
        """Get the wavenumber range from the currently loaded data."""
        if self.original_data is not None and not self.original_data.empty:
            wavenumbers = self.original_data.index.values
            min_wn = float(wavenumbers.min())
            max_wn = float(wavenumbers.max())
            return (min_wn, max_wn)
        else:
            # Fallback to default range if no data is loaded
            return (400.0, 4000.0)
    
    def _show_parameter_widget(self, step: PipelineStep):
        """Show parameter widget for the selected step."""
        # Clear existing widget completely
        self._clear_parameter_widget()
        
        # Get method info
        method_info = PREPROCESSING_REGISTRY.get_method_info(step.category, step.method)
        if not method_info:
            return
        
        # Get actual data range for tuple parameters
        data_range = self._get_data_wavenumber_range()
        
        # Create parameter widget with saved parameters and data range
        self.current_step_widget = DynamicParameterWidget(method_info, step.params, data_range)
        self.params_layout.addWidget(self.current_step_widget)
        
        # Connect parameter signals for automatic preview updates
        self._connect_parameter_signals(self.current_step_widget)
        
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

    def _set_default_output_name(self, dataset_name: str):
        """Set default output name based on selected dataset name with '-pp' suffix."""
        # Only set default if the output name field is empty or contains a previous default
        current_text = self.output_name_edit.text().strip()
        
        # Check if current text is empty or appears to be a previous default (ends with -pp)
        if not current_text or current_text.endswith('-pp'):
            default_name = f"{dataset_name}-pp"
            self.output_name_edit.setText(default_name)
            # Set placeholder to show the default behavior
            self.output_name_edit.setPlaceholderText(LOCALIZE("PREPROCESS.output_name_placeholder"))
    
    def _clear_default_output_name(self):
        """Clear the default output name when no datasets are selected."""
        current_text = self.output_name_edit.text().strip()
        
        # Only clear if it appears to be a default name (ends with -pp)
        if current_text.endswith('-pp'):
            self.output_name_edit.clear()

    # ============== AUTOMATIC PREVIEW SYSTEM ==============
    
    def _update_preview_button_state(self, enabled: bool):
        """Update preview button appearance based on state."""
        if enabled:
            self.preview_toggle_btn.setIcon(self.eye_open_icon)
            self.preview_toggle_btn.setText("プレビュー")
            self.preview_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 6px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #21618c;
                }
            """)
        else:
            self.preview_toggle_btn.setIcon(self.eye_close_icon)
            self.preview_toggle_btn.setText("オフ")
            self.preview_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #95a5a6;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 6px;
                }
                QPushButton:hover {
                    background-color: #7f8c8d;
                }
                QPushButton:pressed {
                    background-color: #6c7b7d;
                }
            """)

    def _toggle_preview_mode(self, enabled: bool):
        """Toggle real-time preview mode."""
        self.preview_enabled = enabled
        self._update_preview_button_state(enabled)
        
        if enabled:
            self._update_preview_status("ready", "green")
            # Trigger preview update if data is available
            if self.original_data is not None:
                self._schedule_preview_update()
        else:
            self._update_preview_status("disabled", "gray")
            # Show original data only
            self._show_original_data()
    
    def _update_preview_status(self, status: str, color: str):
        """Update the preview status indicator."""
        color_map = {
            "green": "#27ae60",
            "orange": "#f39c12", 
            "red": "#e74c3c",
            "gray": "#95a5a6"
        }
        
        status_text_map = {
            "ready": "Ready",
            "processing": "Processing...", 
            "error": "Error",
            "disabled": "Disabled"
        }
        
        mapped_color = color_map.get(color, color)
        status_text = status_text_map.get(status, status.title())
        
        self.preview_status.setStyleSheet(f"color: {mapped_color}; font-size: 14px;")
        self.preview_status_text.setStyleSheet(f"font-size: 11px; color: {mapped_color}; font-weight: bold;")
        self.preview_status_text.setText(status_text)
        
        status_messages = {
            "ready": LOCALIZE("PREPROCESS.preview_status_ready"),
            "processing": LOCALIZE("PREPROCESS.preview_status_processing"), 
            "error": LOCALIZE("PREPROCESS.preview_status_error"),
            "disabled": LOCALIZE("PREPROCESS.preview_status_disabled")
        }
        tooltip = status_messages.get(status, status)
        self.preview_status.setToolTip(tooltip)
        self.preview_status_text.setToolTip(tooltip)
    
    def _manual_refresh_preview(self):
        """Manually trigger a preview refresh."""
        if self.original_data is not None:
            self._update_preview_status("processing", "orange")
            # Use immediate update instead of scheduled
            QTimer.singleShot(100, self._update_preview)  # Small delay for UI responsiveness
        else:
            self.showNotification.emit(LOCALIZE("PREPROCESS.no_data_for_preview"), "warning")
    
    def _schedule_preview_update(self, delay_ms: int = 300):
        """Schedule a preview update with debouncing."""
        if not self.preview_enabled:
            return
            
        # Stop any existing timer
        self.preview_timer.stop()
        self._update_preview_status("processing", "orange")
        
        # Start new timer with delay for debouncing
        self.preview_timer.start(delay_ms)
    
    def _apply_preview_pipeline(self, data, steps):
        """Apply preprocessing steps to data for preview - DataFrame-first approach."""
        try:
            create_logs("preview_pipeline_debug", "PreprocessPage", 
                       f"_apply_preview_pipeline called with {len(steps)} steps", status='debug')
            
            # Return original data if no steps
            if not steps:
                create_logs("preview_pipeline_debug", "PreprocessPage", "No steps to apply, returning original data", status='debug')
                return data
            
            # Work directly with DataFrame (original approach)
            processed_data = data.copy()
            
            create_logs("preview_pipeline_debug", "PreprocessPage", 
                       f"Starting with DataFrame shape: {processed_data.shape}", status='debug')
            
            # Update current step parameters from widget before applying pipeline
            current_row = self.pipeline_list.currentRow()
            if current_row >= 0 and self.current_step_widget:
                current_step = steps[current_row]
                current_params = self.current_step_widget.get_parameters()
                current_step.params = current_params
            
            # Apply each enabled step
            for step in steps:
                # Skip disabled steps
                if not step.enabled:
                    create_logs("preview_step_debug", "PreprocessPage", 
                               f"Skipping disabled step: {step.method}", status='debug')
                    continue
                    
                try:
                    create_logs("preview_step_debug", "PreprocessPage", 
                               f"Applying step: {step.method} with params: {step.params}", status='debug')
                    
                    # Get preprocessing method instance
                    method_info = PREPROCESSING_REGISTRY.get_method_info(step.category, step.method)
                    
                    if not method_info:
                        create_logs("preview_method_error", "PreprocessPage", 
                                   f"Method {step.category}.{step.method} not found in registry", status='warning')
                        continue
                    
                    method_instance = PREPROCESSING_REGISTRY.create_method_instance(
                        step.category, step.method, step.params
                    )
                    
                    create_logs("preview_step_debug", "PreprocessPage", 
                               f"Created method instance: {type(method_instance)}", status='debug')
                    
                    # Handle different method types
                    if step.method in ['WavenumberCalibration', 'IntensityCalibration']:
                        # Skip calibration methods in preview
                        create_logs("preview_skip", "PreprocessPage", 
                                   f"Skipping {step.method} in preview (requires special parameters)", status='info')
                        continue
                    
                    elif hasattr(method_instance, 'apply'):
                        # Convert DataFrame to SpectralContainer for this step only
                        wavenumbers = processed_data.index.values
                        intensities = processed_data.values.T  # Shape: (n_spectra, n_wavenumbers)
                        
                        create_logs("preview_conversion_debug", "PreprocessPage", 
                                   f"Converting to SpectralContainer: intensities {intensities.shape}, wavenumbers {wavenumbers.shape}", 
                                   status='debug')
                        
                        # Use sample for faster preview
                        if intensities.shape[0] > 10:
                            sample_indices = list(range(0, intensities.shape[0], max(1, intensities.shape[0] // 5)))
                            intensities = intensities[sample_indices]
                            create_logs("preview_sampling_debug", "PreprocessPage", 
                                       f"Sampled to {intensities.shape[0]} spectra", status='debug')
                        
                        # Create SpectralContainer for this step only
                        import ramanspy as rp
                        temp_spectral = rp.SpectralContainer(intensities, wavenumbers)
                        
                        # Apply processing
                        result = method_instance.apply(temp_spectral)
                        
                        create_logs("preview_processing_debug", "PreprocessPage", 
                                   f"Applied {step.method}, result type: {type(result)}", status='debug')
                        
                        # Convert result back to DataFrame immediately
                        if hasattr(result, 'spectral_data') and hasattr(result, 'spectral_axis'):
                            import pandas as pd
                            processed_data = pd.DataFrame(
                                result.spectral_data.T,  # Transpose back to (wavenumbers, spectra)
                                index=result.spectral_axis,
                                columns=[f"preview_{i}" for i in range(result.spectral_data.shape[0])]
                            )
                            processed_data.index.name = 'wavenumber'
                            
                            create_logs("preview_conversion_debug", "PreprocessPage", 
                                       f"Converted back to DataFrame: {processed_data.shape}", status='debug')
                        
                    else:
                        create_logs("preview_skip", "PreprocessPage", 
                                   f"Method {step.method} does not have apply method", status='warning')
                        
                except Exception as e:
                    # Log the error but continue with next step
                    create_logs("preview_method_error", "PreprocessPage", 
                               f"Error applying {step.method}: {str(e)}", status='error')
                    continue
            
            create_logs("preview_final_debug", "PreprocessPage", 
                       f"Final processed data shape: {processed_data.shape}", status='debug')
            
            return processed_data
            
        except Exception as e:
            # Log the error and return original data if pipeline fails
            create_logs("preview_pipeline_error", "PreprocessPage", 
                       f"Pipeline failed: {str(e)}", status='error')
            return data

    def _should_auto_focus(self) -> bool:
        """
        Check if auto-focus should be enabled based on pipeline contents.
        Auto-focus is only enabled when there are range-limiting preprocessing steps.
        """
        if not self.pipeline_steps:
            return False
        
        # Check for range-limiting steps that benefit from auto-focus
        range_limiting_steps = {
            'Cropper',  # Primary range-limiting step
            'Range Selector', 
            'Spectral Cropper',
            'Wavenumber Filter',
        }
        
        enabled_steps = [step for step in self.pipeline_steps if step.enabled]
        
        for step in enabled_steps:
            step_name = getattr(step, 'method_name', '')
            if any(range_step.lower() in step_name.lower() for range_step in range_limiting_steps):
                return True
        
        return False

    def _show_original_data(self):
        """Show original unprocessed data."""
        if self.original_data is not None:
            # For original data, only auto-focus if there are range-limiting steps
            auto_focus = self._should_auto_focus()
            self.plot_widget.plot_spectra(self.original_data, title="Original Data", auto_focus=auto_focus)
    
    def _show_preview_data(self, processed_data):
        """Show processed data with original data overlay."""
        if self.original_data is not None and processed_data is not None:
            try:
                # Convert processed_data from SpectralContainer if needed
                if hasattr(processed_data, 'spectral_data'):
                    # SpectralContainer format - extract data
                    processed_array = processed_data.spectral_data
                    wavenumbers = processed_data.spectral_axis
                else:
                    # Already numpy array
                    processed_array = processed_data
                    wavenumbers = None
                
                # Get original data for comparison
                if hasattr(self.original_data, 'values'):
                    # DataFrame format
                    original_array = self.original_data.values.T  # Transpose to match format
                    original_wavenumbers = self.original_data.index.values
                else:
                    original_array = self.original_data
                    original_wavenumbers = None
                
                # Sample data for visualization if too large
                if original_array.shape[0] > 10:
                    sample_indices = np.arange(0, original_array.shape[0], max(1, original_array.shape[0] // 10))
                    sample_original = original_array[sample_indices]
                else:
                    sample_original = original_array
                
                # Plot both original and processed data with conditional auto-focus
                auto_focus = self._should_auto_focus()
                
                if wavenumbers is not None and original_wavenumbers is not None:
                    # Both have wavenumber info
                    self.plot_widget.plot_comparison_spectra_with_wavenumbers(
                        sample_original, processed_array,
                        original_wavenumbers, wavenumbers,
                        titles=["Original", "Processed"],
                        colors=["lightblue", "darkblue"],
                        auto_focus=auto_focus
                    )
                else:
                    # Fallback to original method
                    self.plot_widget.plot_comparison_spectra(
                        sample_original, processed_array,
                        titles=["Original", "Processed"],
                        colors=["lightblue", "darkblue"]
                    )
                
            except Exception as e:
                create_logs("preview_display_error", "PreprocessPage", 
                           f"Error displaying preview data: {str(e)}", status='error')
                # Fallback to showing original data
                self._show_original_data()
    
    def _on_pipeline_reordered(self):
        """Handle pipeline reordering via drag and drop."""
        # Update pipeline_steps order to match UI order
        new_order = []
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            old_index = item.data(Qt.ItemDataRole.UserRole)
            if old_index is not None and old_index < len(self.pipeline_steps):
                new_order.append(self.pipeline_steps[old_index])
        
        self.pipeline_steps = new_order
        
        # Update item data to reflect new order
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            item.setData(Qt.ItemDataRole.UserRole, i)
        
        # Trigger preview update
        self._schedule_preview_update()
    
    def _get_data_wavenumber_range(self):
        """Get the wavenumber range from the currently loaded data."""
        if self.original_data is not None and not self.original_data.empty:
            wavenumbers = self.original_data.index.values
            min_wn = float(wavenumbers.min())
            max_wn = float(wavenumbers.max())
            return (min_wn, max_wn)
        else:
            # Fallback to default range if no data is loaded
            return (400.0, 4000.0)
    
    def _should_auto_focus(self) -> bool:
        """Check if auto-focus should be enabled based on pipeline contents."""
        range_limiting_steps = ['Cropper', 'Range Selector', 'Spectral Window', 'Baseline Range']
        enabled_steps = [step for step in self.pipeline_steps if step.enabled]
        return any(step.method in range_limiting_steps for step in enabled_steps)
    
    def _schedule_preview_update(self, delay_ms: int = 500):
        """Schedule a preview update with debouncing."""
        if not self.preview_enabled:
            return
        
        # Stop any pending timer
        self.preview_timer.stop()
        
        # Start new timer with specified delay
        self.preview_timer.setInterval(delay_ms)
        self.preview_timer.start()
    
    def _update_preview(self):
        """Update the preview plot with current pipeline."""
        try:
            if not self.preview_enabled:
                return
            
            # Get current selected datasets
            selected_items = self.dataset_list.selectedItems()
            if not selected_items:
                self.plot_widget.clear_plot()
                return
            
            # Use first selected dataset for preview
            first_item = selected_items[0]
            dataset_name = first_item.text().replace("📊 ", "").replace("🔄 ", "")
            
            if dataset_name not in RAMAN_DATA:
                self.plot_widget.clear_plot()
                return
            
            original_data = RAMAN_DATA[dataset_name]
            self.original_data = original_data  # Ensure original_data is set
            
            # Apply current pipeline to get processed data
            enabled_steps = [step for step in self.pipeline_steps if step.enabled]
            processed_data = self._apply_preview_pipeline(original_data, enabled_steps)
            
            # Determine if auto-focus should be used
            auto_focus = self._should_auto_focus()
            
            # Plot comparison with conditional auto-focus
            if processed_data is not None:
                # Check if data has been modified by comparing shapes and values
                data_modified = False
                try:
                    create_logs("preview_comparison_debug", "PreprocessPage", 
                               f"Comparing data - original: {original_data.shape}, processed: {processed_data.shape}", 
                               status='debug')
                    
                    # Compare processed DataFrame with original DataFrame
                    if hasattr(processed_data, 'shape') and hasattr(original_data, 'shape'):
                        # Both are DataFrames - do shape and content comparison
                        if processed_data.shape != original_data.shape:
                            data_modified = True
                            create_logs("preview_comparison_debug", "PreprocessPage", 
                                       "Shape comparison: data_modified = True", status='debug')
                        elif hasattr(processed_data, 'equals') and hasattr(original_data, 'equals'):
                            # DataFrame comparison
                            if not processed_data.equals(original_data):
                                data_modified = True
                                create_logs("preview_comparison_debug", "PreprocessPage", 
                                           "Content comparison: data_modified = True", status='debug')
                            else:
                                create_logs("preview_comparison_debug", "PreprocessPage", 
                                           "Content comparison: data unchanged", status='debug')
                        else:
                            # Fallback to numpy array comparison
                            import numpy as np
                            if not np.array_equal(processed_data.values, original_data.values):
                                data_modified = True
                                create_logs("preview_comparison_debug", "PreprocessPage", 
                                           "Array comparison: data_modified = True", status='debug')
                    else:
                        # Fallback - assume modified if we can't compare properly
                        data_modified = True
                        create_logs("preview_comparison_debug", "PreprocessPage", 
                                   "Fallback: data_modified = True", status='debug')
                except Exception as e:
                    # If comparison fails, assume data is modified
                    data_modified = True
                    create_logs("preview_comparison_debug", "PreprocessPage", 
                               f"Exception in comparison: {e}, data_modified = True", status='debug')
                
                create_logs("preview_display_debug", "PreprocessPage", 
                           f"Final data_modified: {data_modified}", status='debug')
                
                if data_modified:
                    # Both original_data and processed_data are now DataFrames
                    # Use plot_spectra for both as it handles DataFrames correctly
                    # instead of trying to extract arrays manually
                    
                    # Create a simple comparison by showing both datasets  
                    self.plot_widget.plot_spectra(
                        processed_data,
                        title="Preprocessed Data (Preview)",
                        auto_focus=auto_focus
                    )
                else:
                    # Show original data only
                    self.plot_widget.plot_spectra(
                        original_data,
                        title="Original Data",
                        auto_focus=False  # No auto-focus for original data view
                    )
            else:
                # Show original data only
                self.plot_widget.plot_spectra(
                    original_data,
                    title="Original Data",
                    auto_focus=False  # No auto-focus for original data view
                )
            
            # Update status
            self.preview_status.setStyleSheet("color: #27ae60; font-size: 14px;")
            self.preview_status_text.setText("Ready")
            self.preview_status_text.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            
        except Exception as e:
            create_logs("PreprocessPage", "_update_preview", f"Error updating preview: {e}", status='error')
            self.preview_status.setStyleSheet("color: #e74c3c; font-size: 14px;")
            self.preview_status_text.setText("Error")
            self.preview_status_text.setStyleSheet("font-size: 11px; color: #e74c3c; font-weight: bold;")
    
    def _manual_refresh_preview(self):
        """Manually refresh the preview."""
        self._update_preview()
    
    def _manual_focus(self):
        """Manually apply focus to the current plot."""
        try:
            import numpy as np
            # Get current selected datasets
            selected_items = self.dataset_list.selectedItems()
            if not selected_items:
                return
            
            # Use first selected dataset
            first_item = selected_items[0]
            dataset_name = first_item.text().replace("📊 ", "").replace("🔄 ", "")
            
            if dataset_name not in RAMAN_DATA:
                return
            
            original_data = RAMAN_DATA[dataset_name]
            enabled_steps = [step for step in self.pipeline_steps if step.enabled]
            processed_data = self._apply_preview_pipeline(original_data, enabled_steps)
            
            # Force auto-focus on
            if processed_data is not None:
                # Check if data has been modified
                data_modified = False
                try:
                    # Compare processed DataFrame with original DataFrame
                    if hasattr(processed_data, 'shape') and hasattr(original_data, 'shape'):
                        # Both are DataFrames - do shape and content comparison
                        if processed_data.shape != original_data.shape:
                            data_modified = True
                        elif hasattr(processed_data, 'equals') and hasattr(original_data, 'equals'):
                            # DataFrame comparison
                            if not processed_data.equals(original_data):
                                data_modified = True
                        else:
                            # Fallback to numpy array comparison
                            if not np.array_equal(processed_data.values, original_data.values):
                                data_modified = True
                    else:
                        # Fallback - assume modified if we can't compare properly
                        data_modified = True
                except Exception:
                    # If comparison fails, assume data is modified
                    data_modified = True
                
                if data_modified:
                    # Both original_data and processed_data are now DataFrames
                    # Extract data for comparison plot
                    original_array = original_data.values
                    original_wavenumbers = original_data.index.values
                    processed_array = processed_data.values
                    processed_wavenumbers = processed_data.index.values
                    
                    # Show comparison plot with forced auto-focus
                    self.plot_widget.plot_comparison_spectra_with_wavenumbers(
                        original_array,
                        processed_array,
                        original_wavenumbers,
                        processed_wavenumbers,
                        titles=["Original", "Processed"],
                        auto_focus=True  # Force auto-focus
                    )
                else:
                    # Show original data with forced auto-focus
                    self.plot_widget.plot_spectra(
                        original_data,
                        title="Original Data (Focused)",
                        auto_focus=True  # Force auto-focus
                    )
            else:
                # Show original data with forced auto-focus
                self.plot_widget.plot_spectra(
                    original_data,
                    title="Original Data (Focused)",
                    auto_focus=True  # Force auto-focus
                )
            
        except Exception as e:
            create_logs("PreprocessPage", "_manual_focus", f"Error in manual focus: {e}", status='error')
    
    def _toggle_preview_mode(self, enabled):
        """Toggle preview mode on/off."""
        self.preview_enabled = enabled
        self._update_preview_button_state(enabled)
        
        if enabled:
            # Ensure we have data loaded when turning preview back on
            if self.original_data is None or self.original_data.empty:
                self.preview_raw_data()  # This will load data and trigger preview
            else:
                self._update_preview()
        else:
            self.plot_widget.clear_plot()
    
    def _update_preview_button_state(self, enabled):
        """Update preview button appearance based on state."""
        if enabled:
            self.preview_toggle_btn.setIcon(self.eye_open_icon)
            self.preview_toggle_btn.setText(" Preview ON")
            self.preview_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e8f5e8;
                    border: 2px solid #27ae60;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                    color: #27ae60;
                }
                QPushButton:hover {
                    background-color: #d5e8d4;
                }
            """)
        else:
            self.preview_toggle_btn.setIcon(self.eye_close_icon)
            self.preview_toggle_btn.setText(" Preview OFF")
            self.preview_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f8f8f8;
                    border: 2px solid #7f8c8d;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                    color: #7f8c8d;
                }
                QPushButton:hover {
                    background-color: #ecf0f1;
                }
            """)
    
    def preview_raw_data(self):
        """Preview raw data when dataset selection changes."""
        selected_items = self.dataset_list.selectedItems()
        if not selected_items:
            self.plot_widget.clear_plot()
            return
        
        first_item = selected_items[0]
        dataset_name = first_item.text().replace("📊 ", "").replace("🔄 ", "")
        
        if dataset_name in RAMAN_DATA:
            self.original_data = RAMAN_DATA[dataset_name]
            
            # Show original data without auto-focus initially
            self.plot_widget.plot_spectra(
                self.original_data,
                title=f"Dataset: {dataset_name}",
                auto_focus=False
            )
            
            # Trigger preview update if preview is enabled
            if self.preview_enabled:
                self._schedule_preview_update()
    
    def on_pipeline_step_selected(self, current, previous):
        """Handle pipeline step selection change."""
        if current is None:
            self._clear_parameter_widget()
            return
        
        step_index = current.data(Qt.ItemDataRole.UserRole)
        if step_index is not None and 0 <= step_index < len(self.pipeline_steps):
            step = self.pipeline_steps[step_index]
            self._show_parameter_widget(step)
            
            # Trigger preview update when step is selected
            self._schedule_preview_update()
    
    def _clear_parameter_widget(self):
        """Clear the parameter widget area."""
        for i in reversed(range(self.params_layout.count())):
            child = self.params_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Show default message
        default_label = QLabel(LOCALIZE("PREPROCESS.set_step_params"))
        default_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        default_label.setFixedHeight(30)
        self.params_layout.addWidget(default_label)
        
        self.current_step_widget = None
    
    def _show_parameter_widget(self, step: PipelineStep):
        """Show parameter widget for the given step."""
        # Clear existing widget
        self._clear_parameter_widget()
        
        # Get method info
        method_info = PREPROCESSING_REGISTRY.get_method_info(step.category, step.method)
        if not method_info:
            return
        
        # Create parameter widget with proper data range
        data_range = self._get_data_wavenumber_range()
        param_widget = DynamicParameterWidget(method_info, step.params, data_range, self)
        
        # Remove default message and add parameter widget
        for i in reversed(range(self.params_layout.count())):
            child = self.params_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        self.params_layout.addWidget(param_widget)
        self.current_step_widget = param_widget
        
        # Connect parameter changes to preview updates
        self._connect_parameter_signals(param_widget)
    
    def _on_pipeline_reordered(self, parent, start, end, destination, row):
        """Handle pipeline reordering via drag and drop."""
        # Reconstruct pipeline_steps list in new order
        new_order = []
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            old_index = item.data(Qt.ItemDataRole.UserRole)
            if old_index is not None and old_index < len(self.pipeline_steps):
                new_order.append(self.pipeline_steps[old_index])
        
        self.pipeline_steps = new_order
        
        # Update item data to reflect new order
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            item.setData(Qt.ItemDataRole.UserRole, i)
        
        # Trigger preview update
        self._schedule_preview_update()
    
    def _connect_parameter_signals(self, param_widget):
        """Connect parameter widget signals for automatic preview updates."""
        if not param_widget:
            return
        
        # Connect all parameter widget signals to trigger preview updates
        for widget in param_widget.param_widgets.values():
            # Connect custom parameter widgets with parametersChanged signal
            if hasattr(widget, 'parametersChanged'):
                widget.parametersChanged.connect(lambda: self._schedule_preview_update())
                # Connect real-time updates for sliders with immediate response
                if hasattr(widget, 'realTimeUpdate'):
                    widget.realTimeUpdate.connect(lambda: self._schedule_preview_update(delay_ms=50))
            # Connect standard Qt widgets
            elif hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(lambda: self._schedule_preview_update())
            elif hasattr(widget, 'textChanged'):
                widget.textChanged.connect(lambda: self._schedule_preview_update())
            elif hasattr(widget, 'currentTextChanged'):
                widget.currentTextChanged.connect(lambda: self._schedule_preview_update())
            elif hasattr(widget, 'toggled'):
                widget.toggled.connect(lambda: self._schedule_preview_update())

    

