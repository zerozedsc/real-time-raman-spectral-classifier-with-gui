import sys
import os
import json
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QGroupBox, QScrollArea, QFrame, QFileDialog,
    QListWidget, QListWidgetItem, QMessageBox, QTabWidget, QTextEdit, QSizePolicy, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, QUrl, QSize
from PySide6.QtGui import QIcon

from functions.data_loader import plot_spectra, load_data_from_path, load_metadata_from_json
from utils import LOCALIZE, PROJECT_MANAGER, CONFIGS, RAMAN_DATA
from components.matplotlib_widget import MatplotlibWidget, plot_spectra

class DragDropLabel(QLabel):
    pathDropped = Signal(str)
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True); self.setAlignment(Qt.AlignCenter); self.setWordWrap(True)
        self.setObjectName("dragDropLabel"); self.setMinimumHeight(80)
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction(); self.setStyleSheet("border: 2px dashed #0078d4; background-color: #eaf2f8;")
        else: event.ignore()
    def dragLeaveEvent(self, event): self.setStyleSheet("")
    def dropEvent(self, event):
        self.setStyleSheet("");
        if urls := event.mimeData().urls(): self.pathDropped.emit(urls[0].toLocalFile())

class DatasetItemWidget(QWidget):
    removeRequested = Signal(str)
    
    def __init__(self, dataset_name: str, parent=None):
        super().__init__(parent)
        self.dataset_name = dataset_name
        self.setObjectName("datasetItemWidget")
        layout = QHBoxLayout(self); layout.setContentsMargins(0, 5, 5, 5); layout.setSpacing(10)
        name_label = QLabel(dataset_name); name_label.setObjectName("datasetItemLabel")
        remove_button = QPushButton("V"); remove_button.setObjectName("removeListItemButton"); remove_button.setFixedSize(26, 26); remove_button.setToolTip(f"Remove '{dataset_name}'"); remove_button.clicked.connect(lambda: self.removeRequested.emit(self.dataset_name))
        layout.addWidget(name_label); layout.addStretch(); layout.addWidget(remove_button)

class DataPackagePage(QWidget):
    showNotification = Signal(str, str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.preview_dataframe = None; self.preview_metadata = {}
        self._setup_ui()
        self._connect_signals()
        self.load_project_data()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self); main_layout.setContentsMargins(20, 20, 20, 20); main_layout.setSpacing(20)
        self._create_left_panel(main_layout)
        self._create_right_panel(main_layout)
        main_layout.setStretch(0, 1); main_layout.setStretch(1, 2)

    def _create_left_panel(self, parent_layout):
        left_vbox = QVBoxLayout(); left_vbox.setSpacing(20)
        importer_group = QGroupBox(LOCALIZE("DATA_PACKAGE_PAGE.importer_title")); importer_layout = QGridLayout(importer_group); importer_layout.setSpacing(10)
        importer_layout.addWidget(QLabel(f"<b>{LOCALIZE('DATA_PACKAGE_PAGE.dataset_name_label')}:</b>"), 0, 0, 1, 2)
        self.dataset_name_edit = QLineEdit(); self.dataset_name_edit.setPlaceholderText(LOCALIZE("DATA_PACKAGE_PAGE.dataset_name_placeholder")); importer_layout.addWidget(self.dataset_name_edit, 1, 0, 1, 2)
        self.data_drop_label = DragDropLabel(LOCALIZE("DATA_PACKAGE_PAGE.drag_drop_data_prompt")); importer_layout.addWidget(self.data_drop_label, 2, 0, 1, 2)
        self.data_path_edit = QLineEdit(); self.data_path_edit.setReadOnly(True); importer_layout.addWidget(self.data_path_edit, 3, 0)
        browse_data_btn = QPushButton(LOCALIZE("DATA_PACKAGE_PAGE.browse_data_button")); importer_layout.addWidget(browse_data_btn, 3, 1)
        self.meta_drop_label = DragDropLabel(LOCALIZE("DATA_PACKAGE_PAGE.drag_drop_meta_prompt")); importer_layout.addWidget(self.meta_drop_label, 4, 0, 1, 2)
        self.meta_path_edit = QLineEdit(); self.meta_path_edit.setReadOnly(True); importer_layout.addWidget(self.meta_path_edit, 5, 0)
        browse_meta_btn = QPushButton(LOCALIZE("DATA_PACKAGE_PAGE.browse_meta_button")); importer_layout.addWidget(browse_meta_btn, 5, 1)
        self.preview_button = QPushButton(LOCALIZE("DATA_PACKAGE_PAGE.preview_button")); importer_layout.addWidget(self.preview_button, 6, 0)
        self.add_to_project_button = QPushButton(LOCALIZE("DATA_PACKAGE_PAGE.add_to_project_button")); self.add_to_project_button.setObjectName("ctaButton"); self.add_to_project_button.setEnabled(False); importer_layout.addWidget(self.add_to_project_button, 6, 1)
        
        loaded_group = QGroupBox(LOCALIZE("DATA_PACKAGE_PAGE.loaded_datasets_title"))
        loaded_layout = QVBoxLayout(loaded_group)
        self.loaded_data_list = QListWidget(); self.loaded_data_list.setObjectName("loadedDataListWidget"); self.loaded_data_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        loaded_layout.addWidget(self.loaded_data_list)
        
        left_vbox.addWidget(importer_group); left_vbox.addWidget(loaded_group, 1)
        parent_layout.addLayout(left_vbox)

        self.data_drop_label.pathDropped.connect(self._set_data_path); self.meta_drop_label.pathDropped.connect(self.meta_path_edit.setText)
        browse_data_btn.clicked.connect(self.browse_for_data); browse_meta_btn.clicked.connect(self.browse_for_metadata)
        self.loaded_data_list.currentItemChanged.connect(self.display_selected_dataset)

    def _create_right_panel(self, parent_layout):
        right_vbox = QVBoxLayout()
        preview_group = QGroupBox(LOCALIZE("DATA_PACKAGE_PAGE.preview_title"))
        preview_layout = QVBoxLayout(preview_group)
        plot_frame = QFrame(); plot_frame.setObjectName("plotFrame"); plot_layout = QVBoxLayout(plot_frame); self.plot_widget = MatplotlibWidget(); plot_layout.addWidget(self.plot_widget)
        self.info_label = QLabel(LOCALIZE("DATA_PACKAGE_PAGE.no_data_preview")); self.info_label.setAlignment(Qt.AlignCenter); self.info_label.setWordWrap(True)
        preview_layout.addWidget(plot_frame); preview_layout.addWidget(self.info_label)

        self.meta_editor_group = QGroupBox(LOCALIZE("DATA_PACKAGE_PAGE.metadata_editor_title")); self.meta_editor_group.setCheckable(True); self.meta_editor_group.setChecked(False)
        editor_vbox = QVBoxLayout(self.meta_editor_group)
        self.meta_tabs = QTabWidget()
        self._populate_metadata_fields()
        self.save_meta_button = QPushButton(LOCALIZE("DATA_PACKAGE_PAGE.save_meta_button"))
        editor_vbox.addWidget(self.meta_tabs); editor_vbox.addWidget(self.save_meta_button, 0, Qt.AlignRight)

        right_vbox.addWidget(preview_group); right_vbox.addWidget(self.meta_editor_group)
        right_vbox.setStretch(0, 3); right_vbox.setStretch(1, 2)
        parent_layout.addLayout(right_vbox)

    def _populate_metadata_fields(self):
        self.metadata_widgets = {}
        meta_structure = CONFIGS.get("metadata_structure", {})
        for tab_key, fields in meta_structure.items():
            tab_widget = QWidget(); tab_layout = QGridLayout(tab_widget); tab_name = LOCALIZE(f"DATA_PACKAGE_PAGE.metadata_tab_{tab_key}")
            self.meta_tabs.addTab(tab_widget, tab_name); self.metadata_widgets[tab_key] = {}
            row = 0
            for field_key, field_info in fields.items():
                field_name = LOCALIZE(f"DATA_PACKAGE_PAGE.metadata_field_{field_key}"); label = QLabel(f"{field_name}:")
                field_type = field_info.get("type", "LineEdit")
                if field_type == "TextEdit": widget = QTextEdit(); widget.setMinimumHeight(80)
                else: widget = QLineEdit()
                widget.setPlaceholderText(field_info.get("placeholder", "")); tab_layout.addWidget(label, row, 0); tab_layout.addWidget(widget, row, 1)
                self.metadata_widgets[tab_key][field_key] = widget; row += 1
            tab_layout.setRowStretch(row, 1)

    def _connect_signals(self):
        self.preview_button.clicked.connect(self.handle_preview_data)
        self.add_to_project_button.clicked.connect(self.handle_add_to_project)
        self.save_meta_button.clicked.connect(self.save_metadata_as_json)
        self.meta_editor_group.toggled.connect(self.toggle_metadata_editing)

    def _set_data_path(self, path: str):
        self.data_path_edit.setText(path)
        if path:
            base_name, _ = os.path.splitext(os.path.basename(path))
            if not self.dataset_name_edit.text().strip(): self.dataset_name_edit.setText(base_name.replace('_', ' ').replace('-', ' ').title())

    def load_project_data(self):
        current_selection_name = self.loaded_data_list.currentItem().data(Qt.UserRole) if self.loaded_data_list.currentItem() else None
        self.loaded_data_list.clear()
        if not RAMAN_DATA:
            placeholder_item = QListWidgetItem(LOCALIZE("DATA_PACKAGE_PAGE.no_datasets_loaded")); placeholder_item.setFlags(placeholder_item.flags() & ~Qt.ItemIsSelectable)
            self.loaded_data_list.addItem(placeholder_item); self.loaded_data_list.setEnabled(False); self.update_preview_display(None, {})
        else:
            self.loaded_data_list.setEnabled(True)
            for name in sorted(RAMAN_DATA.keys()):
                item_widget = DatasetItemWidget(name); item_widget.removeRequested.connect(self.handle_remove_dataset)
                list_item = QListWidgetItem(self.loaded_data_list); list_item.setSizeHint(item_widget.sizeHint()); list_item.setData(Qt.UserRole, name)
                self.loaded_data_list.addItem(list_item); self.loaded_data_list.setItemWidget(list_item, item_widget)
            if current_selection_name:
                for i in range(self.loaded_data_list.count()):
                    if self.loaded_data_list.item(i).data(Qt.UserRole) == current_selection_name: self.loaded_data_list.setCurrentRow(i); break
            elif self.loaded_data_list.count() > 0: self.loaded_data_list.setCurrentRow(0)

    def display_selected_dataset(self, current_item: QListWidgetItem, previous_item: QListWidgetItem):
        if not current_item or not self.loaded_data_list.isEnabled(): self.update_preview_display(None, {}); return
        dataset_name = current_item.data(Qt.UserRole)
        if not dataset_name: return
        df = RAMAN_DATA.get(dataset_name)
        metadata = PROJECT_MANAGER.current_project_data.get("dataPackages", {}).get(dataset_name, {}).get("metadata", {})
        self.update_preview_display(df, metadata, is_preview=False)

    def browse_for_data(self):
        dialog = QFileDialog(self, LOCALIZE("DATA_PACKAGE_PAGE.browse_data_dialog_title")); dialog.setFileMode(QFileDialog.FileMode.ExistingFiles); dialog.setOption(QFileDialog.Option.ShowDirsOnly, False)
        if dialog.exec(): self._set_data_path(dialog.selectedFiles()[0])

    def browse_for_metadata(self):
        path, _ = QFileDialog.getOpenFileName(self, LOCALIZE("DATA_PACKAGE_PAGE.browse_meta_dialog_title"), "", f"JSON Files (*.json)")
        if path: self.meta_path_edit.setText(path)

    def handle_preview_data(self):
        data_path = self.data_path_edit.text()
        if not data_path: self.showNotification.emit(LOCALIZE("NOTIFICATIONS.data_source_missing"), "error"); return
        df = load_data_from_path(data_path)
        if isinstance(df, str): self.showNotification.emit(LOCALIZE("NOTIFICATIONS.data_load_error", error=df), "error"); self.preview_dataframe = None; self.add_to_project_button.setEnabled(False)
        else: self.preview_dataframe = df; self.showNotification.emit(LOCALIZE("NOTIFICATIONS.data_load_success"), "success"); self.add_to_project_button.setEnabled(True)
        meta_path = self.meta_path_edit.text()
        if meta_path:
            meta = load_metadata_from_json(meta_path)
            if isinstance(meta, str): self.showNotification.emit(LOCALIZE("NOTIFICATIONS.meta_load_error", error=meta), "error"); self.preview_metadata = {}
            else: self.preview_metadata = meta
        else: self.preview_metadata = {}
        self.update_preview_display(self.preview_dataframe, self.preview_metadata, is_preview=True)

    def handle_add_to_project(self):
        dataset_name = self.dataset_name_edit.text().strip()
        if not dataset_name: self.showNotification.emit(LOCALIZE("NOTIFICATIONS.dataset_name_missing"), "error"); return
        if dataset_name in RAMAN_DATA: self.showNotification.emit(LOCALIZE("NOTIFICATIONS.dataset_name_exists", name=dataset_name), "error"); return
        if self.preview_dataframe is None: self.showNotification.emit(LOCALIZE("NOTIFICATIONS.no_data_to_add"), "error"); return
        if self.meta_editor_group.isChecked(): self.preview_metadata = self._get_metadata_from_editor()
        success = PROJECT_MANAGER.add_dataframe_to_project(dataset_name, self.preview_dataframe, self.preview_metadata)
        if success: self.showNotification.emit(LOCALIZE("NOTIFICATIONS.dataset_add_success", name=dataset_name), "success"); self.load_project_data(); self.clear_importer_fields()
        else: self.showNotification.emit(LOCALIZE("NOTIFICATIONS.dataset_add_error"), "error")

    def handle_remove_dataset(self, name: str):
        reply = QMessageBox.question(self, LOCALIZE("DATA_PACKAGE_PAGE.remove_dataset_confirm_title"), LOCALIZE("DATA_PACKAGE_PAGE.remove_dataset_confirm_text", name=name), QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if PROJECT_MANAGER.remove_dataframe_from_project(name): self.showNotification.emit(LOCALIZE("NOTIFICATIONS.dataset_remove_success", name=name), "success"); self.load_project_data()
            else: self.showNotification.emit(LOCALIZE("NOTIFICATIONS.dataset_remove_error", name=name), "error")

    def update_preview_display(self, df: pd.DataFrame, metadata: dict, is_preview: bool = True):
        self.plot_widget.update_plot(plot_spectra(df))
        if df is not None and not df.empty:
            info_text = (f"<b>{LOCALIZE('DATA_PACKAGE_PAGE.info_num_spectra')}:</b> {df.shape[1]} | " f"<b>{LOCALIZE('DATA_PACKAGE_PAGE.info_wavenumber_range')}:</b> {df.index.min():.2f} - {df.index.max():.2f} cm⁻¹ | " f"<b>{LOCALIZE('DATA_PACKAGE_PAGE.info_data_points')}:</b> {df.shape[0]}")
            self.info_label.setText(info_text)
        else: self.info_label.setText(LOCALIZE("DATA_PACKAGE_PAGE.no_data_preview"))
        self._set_metadata_in_editor(metadata); self.meta_editor_group.setChecked(bool(metadata) or is_preview); self.toggle_metadata_editing(self.meta_editor_group.isChecked(), read_only=not is_preview)

    def _get_metadata_from_editor(self) -> dict:
        metadata = {};
        for tab_key, fields in self.metadata_widgets.items():
            metadata[tab_key] = {}
            for field_key, widget in fields.items(): metadata[tab_key][field_key] = widget.toPlainText() if isinstance(widget, QTextEdit) else widget.text()
        return metadata

    def _set_metadata_in_editor(self, metadata: dict):
        for tab_key, fields in self.metadata_widgets.items():
            for field_key, widget in fields.items():
                value = metadata.get(tab_key, {}).get(field_key, "")
                if isinstance(widget, QTextEdit): widget.setPlainText(str(value))
                else: widget.setText(str(value))

    def toggle_metadata_editing(self, checked, read_only=False):
        self.meta_editor_group.setEnabled(checked)
        for tab in self.metadata_widgets.values():
            for widget in tab.values(): widget.setReadOnly(read_only)
        self.save_meta_button.setVisible(checked and not read_only)

    def save_metadata_as_json(self):
        data_dir = PROJECT_MANAGER._get_project_data_dir()
        if not data_dir: self.showNotification.emit(LOCALIZE("NOTIFICATIONS.no_project_loaded_for_save"), "error"); return
        dataset_name = self.dataset_name_edit.text().strip()
        default_filename = f"{dataset_name}_metadata.json" if dataset_name else "metadata.json"
        path, _ = QFileDialog.getSaveFileName(self, LOCALIZE("DATA_PACKAGE_PAGE.save_meta_dialog_title"), os.path.join(data_dir, default_filename), f"JSON Files (*.json)")
        if path:
            manual_meta = self._get_metadata_from_editor()
            try:
                with open(path, 'w', encoding='utf-8') as f: json.dump(manual_meta, f, indent=4)
                self.showNotification.emit(LOCALIZE("NOTIFICATIONS.meta_save_success"), "success"); self.meta_path_edit.setText(path); self.preview_metadata = manual_meta
            except Exception as e: self.showNotification.emit(LOCALIZE("NOTIFICATIONS.meta_save_error", error=e), "error")

    def clear_importer_fields(self):
        self.dataset_name_edit.clear(); self.data_path_edit.clear(); self.meta_path_edit.clear()
        self.preview_dataframe = None; self.preview_metadata = {}; self.add_to_project_button.setEnabled(False)
        self._set_metadata_in_editor({}); self.meta_editor_group.setChecked(False)
