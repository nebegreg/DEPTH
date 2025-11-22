#!/usr/bin/env python3
"""
Depth Anything v3 Professional GUI Application
===============================================

A comprehensive PyQt6 application for monocular/multi-view depth estimation,
3D reconstruction, pose estimation, and real-time tracking using Depth Anything v3.

Features:
- Monocular depth estimation
- Multi-view depth estimation
- Camera pose estimation
- 3D Gaussian reconstruction
- Real-time video/webcam processing
- Batch processing
- Multiple export formats (GLB, PLY, NPZ, etc.)
- Interactive 3D visualization
- GPU acceleration with performance optimization

Author: Claude
License: MIT
"""

import sys
import os
import glob
import time
from pathlib import Path
from typing import Optional, List
import numpy as np
import cv2
import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QTextEdit, QFileDialog,
    QTabWidget, QGroupBox, QGridLayout, QProgressBar, QSpinBox,
    QCheckBox, QSplitter, QScrollArea, QToolBar, QStatusBar,
    QMessageBox, QLineEdit, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QPixmap, QImage, QAction, QIcon, QPalette, QColor

# Import Depth Anything v3
sys.path.insert(0, str(Path(__file__).parent / 'Depth-Anything-3-main' / 'src'))
from depth_anything_3.api import DepthAnything3


class DepthWorker(QThread):
    """Worker thread for depth estimation to keep UI responsive"""

    finished = pyqtSignal(object)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, model, images, mode='monocular', export_dir=None, export_format=None):
        super().__init__()
        self.model = model
        self.images = images
        self.mode = mode
        self.export_dir = export_dir
        self.export_format = export_format
        self._is_running = True

    def run(self):
        """Execute depth estimation"""
        try:
            self.progress.emit(10, "Preprocessing images...")

            # Run inference based on mode
            if self.mode == 'monocular':
                prediction = self.model.inference(
                    self.images,
                    export_dir=self.export_dir,
                    export_format=self.export_format
                )
            elif self.mode == 'multiview':
                self.progress.emit(30, "Processing multi-view depth...")
                prediction = self.model.inference(
                    self.images,
                    export_dir=self.export_dir,
                    export_format=self.export_format
                )
            elif self.mode == 'pose_estimation':
                self.progress.emit(30, "Estimating camera poses...")
                prediction = self.model.inference(
                    self.images,
                    export_dir=self.export_dir,
                    export_format=self.export_format
                )
            elif self.mode == 'gaussian':
                self.progress.emit(30, "Generating 3D Gaussians...")
                prediction = self.model.inference(
                    self.images,
                    export_dir=self.export_dir,
                    export_format='glb'
                )
            else:
                prediction = self.model.inference(self.images)

            self.progress.emit(90, "Finalizing results...")

            if self._is_running:
                self.finished.emit(prediction)
                self.progress.emit(100, "Complete!")

        except Exception as e:
            self.error.emit(f"Error during processing: {str(e)}")

    def stop(self):
        """Stop the worker"""
        self._is_running = False


class VideoWorker(QThread):
    """Worker thread for video/webcam processing"""

    frame_ready = pyqtSignal(object, object)  # frame, depth_map
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, source, fps=30):
        super().__init__()
        self.model = model
        self.source = source  # file path or camera index
        self.fps = fps
        self._is_running = True

    def run(self):
        """Process video stream"""
        try:
            # Open video source
            if isinstance(self.source, int):
                cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(str(self.source))

            if not cap.isOpened():
                self.error.emit("Cannot open video source")
                return

            frame_delay = int(1000 / self.fps)

            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run inference
                prediction = self.model.inference([frame_rgb])
                depth_map = prediction.depth[0]

                self.frame_ready.emit(frame_rgb, depth_map)
                self.msleep(frame_delay)

            cap.release()
            self.finished.emit()

        except Exception as e:
            self.error.emit(f"Video processing error: {str(e)}")

    def stop(self):
        """Stop video processing"""
        self._is_running = False


class DepthAnythingGUI(QMainWindow):
    """Main GUI application for Depth Anything v3"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth Anything v3 - Professional Edition")
        self.setGeometry(100, 100, 1600, 900)

        # Model and state
        self.model = None
        self.current_prediction = None
        self.current_images = []
        self.video_worker = None
        self.depth_worker = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Setup UI
        self.init_ui()
        self.setup_toolbar()
        self.setup_statusbar()
        self.apply_dark_theme()

        # Auto-load model
        QTimer.singleShot(100, self.auto_load_model)

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel - Controls
        left_panel = self.create_control_panel()

        # Right panel - Visualization
        right_panel = self.create_visualization_panel()

        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)

    def create_control_panel(self) -> QWidget:
        """Create the left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Model selection
        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout()

        model_layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "DA3NESTED-GIANT-LARGE (1.4B params)",
            "DA3-GIANT (1.15B params)",
            "DA3-LARGE (0.35B params)",
            "DA3-BASE (0.12B params)",
            "DA3-SMALL (0.08B params)",
            "DA3METRIC-LARGE (Metric)",
            "DA3MONO-LARGE (Mono)"
        ])
        self.model_combo.setCurrentIndex(2)  # Default to LARGE
        model_layout.addWidget(self.model_combo, 0, 1)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn, 1, 0, 1, 2)

        self.model_status = QLabel("Model: Not loaded")
        self.model_status.setStyleSheet("color: orange;")
        model_layout.addWidget(self.model_status, 2, 0, 1, 2)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Mode selection
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QVBoxLayout()

        self.mode_buttons = QButtonGroup()
        modes = [
            ("Monocular Depth", "monocular"),
            ("Multi-View Depth", "multiview"),
            ("Pose Estimation", "pose_estimation"),
            ("3D Gaussians", "gaussian"),
            ("Real-time Video", "video"),
            ("Webcam", "webcam")
        ]

        for idx, (label, value) in enumerate(modes):
            radio = QRadioButton(label)
            radio.setProperty("mode", value)
            self.mode_buttons.addButton(radio, idx)
            mode_layout.addWidget(radio)
            if idx == 0:
                radio.setChecked(True)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Input selection
        input_group = QGroupBox("Input Source")
        input_layout = QVBoxLayout()

        self.load_images_btn = QPushButton("Load Images")
        self.load_images_btn.clicked.connect(self.load_images)
        input_layout.addWidget(self.load_images_btn)

        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self.load_video)
        input_layout.addWidget(self.load_video_btn)

        self.load_folder_btn = QPushButton("Load Folder (Batch)")
        self.load_folder_btn.clicked.connect(self.load_folder)
        input_layout.addWidget(self.load_folder_btn)

        self.images_label = QLabel("No images loaded")
        input_layout.addWidget(self.images_label)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QGridLayout()

        options_layout.addWidget(QLabel("Export Format:"), 0, 0)
        self.export_format = QComboBox()
        self.export_format.addItems(["None", "GLB", "PLY", "NPZ", "Depth Image", "All"])
        options_layout.addWidget(self.export_format, 0, 1)

        options_layout.addWidget(QLabel("FPS (video):"), 1, 0)
        self.fps_spinner = QSpinBox()
        self.fps_spinner.setRange(1, 60)
        self.fps_spinner.setValue(15)
        options_layout.addWidget(self.fps_spinner, 1, 1)

        self.use_metric_depth = QCheckBox("Use Metric Depth")
        options_layout.addWidget(self.use_metric_depth, 2, 0, 1, 2)

        self.show_confidence = QCheckBox("Show Confidence Map")
        options_layout.addWidget(self.show_confidence, 3, 0, 1, 2)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Process button
        self.process_btn = QPushButton("Process")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        layout.addWidget(self.process_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        return panel

    def create_visualization_panel(self) -> QWidget:
        """Create the right visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tabs for different visualizations
        self.viz_tabs = QTabWidget()

        # Original image tab
        self.original_scroll = QScrollArea()
        self.original_label = QLabel("Load an image to begin")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(640, 480)
        self.original_scroll.setWidget(self.original_label)
        self.original_scroll.setWidgetResizable(True)
        self.viz_tabs.addTab(self.original_scroll, "Original")

        # Depth map tab
        self.depth_scroll = QScrollArea()
        self.depth_label = QLabel("Process to see depth map")
        self.depth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_label.setMinimumSize(640, 480)
        self.depth_scroll.setWidget(self.depth_label)
        self.depth_scroll.setWidgetResizable(True)
        self.viz_tabs.addTab(self.depth_scroll, "Depth Map")

        # Confidence map tab
        self.conf_scroll = QScrollArea()
        self.conf_label = QLabel("Confidence map (if available)")
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.conf_label.setMinimumSize(640, 480)
        self.conf_scroll.setWidget(self.conf_label)
        self.conf_scroll.setWidgetResizable(True)
        self.viz_tabs.addTab(self.conf_scroll, "Confidence")

        # 3D visualization tab
        self.viz_3d_label = QLabel("3D visualization (requires Open3D viewer)")
        self.viz_3d_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viz_3d_btn = QPushButton("Open 3D Viewer")
        self.viz_3d_btn.clicked.connect(self.open_3d_viewer)
        viz_3d_widget = QWidget()
        viz_3d_layout = QVBoxLayout(viz_3d_widget)
        viz_3d_layout.addWidget(self.viz_3d_label)
        viz_3d_layout.addWidget(self.viz_3d_btn)
        viz_3d_layout.addStretch()
        self.viz_tabs.addTab(viz_3d_widget, "3D View")

        # Stats tab
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.viz_tabs.addTab(self.stats_text, "Statistics")

        layout.addWidget(self.viz_tabs)

        return panel

    def setup_toolbar(self):
        """Setup toolbar with common actions"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(32, 32))
        self.addToolBar(toolbar)

        # File actions
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.load_images)
        toolbar.addAction(open_action)

        save_action = QAction("Export", self)
        save_action.triggered.connect(self.export_results)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        # View actions
        clear_action = QAction("Clear", self)
        clear_action.triggered.connect(self.clear_all)
        toolbar.addAction(clear_action)

        toolbar.addSeparator()

        # Help
        help_action = QAction("Help", self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)

    def setup_statusbar(self):
        """Setup status bar"""
        self.statusBar().showMessage("Ready")

    def apply_dark_theme(self):
        """Apply modern dark theme"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

        self.setPalette(dark_palette)

        # Additional stylesheet
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                background-color: #555;
            }
            QPushButton:hover {
                background-color: #666;
            }
            QPushButton:pressed {
                background-color: #444;
            }
        """)

    def log(self, message: str, level: str = "INFO"):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        color = {
            "INFO": "white",
            "SUCCESS": "lightgreen",
            "WARNING": "yellow",
            "ERROR": "red"
        }.get(level, "white")

        self.log_output.append(
            f'<span style="color: gray;">[{timestamp}]</span> '
            f'<span style="color: {color};">[{level}]</span> {message}'
        )
        self.statusBar().showMessage(message)

    def auto_load_model(self):
        """Auto-load default model on startup"""
        self.log("Auto-loading default model...", "INFO")
        self.load_model()

    def load_model(self):
        """Load the selected Depth Anything model"""
        try:
            self.log("Loading model...", "INFO")
            self.load_model_btn.setEnabled(False)
            QApplication.processEvents()

            # Get model name
            model_text = self.model_combo.currentText()
            model_map = {
                "DA3NESTED-GIANT-LARGE (1.4B params)": "depth-anything/DA3NESTED-GIANT-LARGE",
                "DA3-GIANT (1.15B params)": "depth-anything/DA3-GIANT",
                "DA3-LARGE (0.35B params)": "depth-anything/DA3-LARGE",
                "DA3-BASE (0.12B params)": "depth-anything/DA3-BASE",
                "DA3-SMALL (0.08B params)": "depth-anything/DA3-SMALL",
                "DA3METRIC-LARGE (Metric)": "depth-anything/DA3METRIC-LARGE",
                "DA3MONO-LARGE (Mono)": "depth-anything/DA3MONO-LARGE"
            }

            model_name = model_map.get(model_text, "depth-anything/DA3-LARGE")

            # Load model
            self.model = DepthAnything3.from_pretrained(model_name)
            self.model = self.model.to(device=self.device)
            self.model.eval()

            self.log(f"Model loaded successfully on {self.device}", "SUCCESS")
            self.model_status.setText(f"Model: {model_text} ({self.device.upper()})")
            self.model_status.setStyleSheet("color: lightgreen;")
            self.process_btn.setEnabled(len(self.current_images) > 0)

        except Exception as e:
            self.log(f"Failed to load model: {str(e)}", "ERROR")
            self.model_status.setText("Model: Load failed")
            self.model_status.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
        finally:
            self.load_model_btn.setEnabled(True)

    def load_images(self):
        """Load images for processing"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )

        if files:
            self.current_images = files
            self.images_label.setText(f"{len(files)} image(s) loaded")
            self.log(f"Loaded {len(files)} images", "SUCCESS")

            # Display first image
            if files:
                pixmap = QPixmap(files[0])
                self.original_label.setPixmap(
                    pixmap.scaled(
                        self.original_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                )

            if self.model is not None:
                self.process_btn.setEnabled(True)

    def load_video(self):
        """Load video file"""
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.flv)"
        )

        if file:
            self.current_images = [file]
            self.images_label.setText(f"Video: {Path(file).name}")
            self.log(f"Loaded video: {Path(file).name}", "SUCCESS")

            # Auto-select video mode
            for i, button in enumerate(self.mode_buttons.buttons()):
                if button.property("mode") == "video":
                    button.setChecked(True)
                    break

            if self.model is not None:
                self.process_btn.setEnabled(True)

    def load_folder(self):
        """Load folder for batch processing"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")

        if folder:
            # Find all images in folder
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp']
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(folder, ext)))
                files.extend(glob.glob(os.path.join(folder, ext.upper())))

            files = sorted(files)

            if files:
                self.current_images = files
                self.images_label.setText(f"{len(files)} images from folder")
                self.log(f"Loaded {len(files)} images from folder", "SUCCESS")

                if self.model is not None:
                    self.process_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Warning", "No images found in folder")

    def process(self):
        """Process images based on selected mode"""
        if not self.model:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return

        if not self.current_images:
            QMessageBox.warning(self, "Warning", "Please load images first")
            return

        # Get selected mode
        selected_mode = None
        for button in self.mode_buttons.buttons():
            if button.isChecked():
                selected_mode = button.property("mode")
                break

        if selected_mode in ['video', 'webcam']:
            self.process_video(selected_mode)
        else:
            self.process_images(selected_mode)

    def process_images(self, mode: str):
        """Process static images"""
        try:
            self.log(f"Starting {mode} processing...", "INFO")
            self.process_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Prepare export
            export_format_map = {
                "None": None,
                "GLB": "glb",
                "PLY": "ply",
                "NPZ": "npz",
                "Depth Image": "depth_image",
                "All": "all"
            }
            export_format = export_format_map[self.export_format.currentText()]
            export_dir = None

            if export_format:
                export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
                if not export_dir:
                    export_format = None

            # Create worker
            self.depth_worker = DepthWorker(
                self.model,
                self.current_images,
                mode=mode,
                export_dir=export_dir,
                export_format=export_format
            )

            self.depth_worker.progress.connect(self.on_progress)
            self.depth_worker.finished.connect(self.on_processing_finished)
            self.depth_worker.error.connect(self.on_error)

            self.depth_worker.start()

        except Exception as e:
            self.log(f"Processing error: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", f"Processing failed:\n{str(e)}")
            self.process_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def process_video(self, mode: str):
        """Process video or webcam stream"""
        try:
            self.log(f"Starting {mode} mode...", "INFO")

            if mode == 'webcam':
                source = 0  # Default camera
            else:
                source = self.current_images[0]

            fps = self.fps_spinner.value()

            # Create video worker
            self.video_worker = VideoWorker(self.model, source, fps)
            self.video_worker.frame_ready.connect(self.on_frame_ready)
            self.video_worker.finished.connect(self.on_video_finished)
            self.video_worker.error.connect(self.on_error)

            # Change button to stop
            self.process_btn.setText("Stop")
            self.process_btn.clicked.disconnect()
            self.process_btn.clicked.connect(self.stop_video)

            self.video_worker.start()

        except Exception as e:
            self.log(f"Video processing error: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", f"Video processing failed:\n{str(e)}")

    def stop_video(self):
        """Stop video processing"""
        if self.video_worker:
            self.video_worker.stop()
            self.video_worker.wait()

        self.process_btn.setText("Process")
        self.process_btn.clicked.disconnect()
        self.process_btn.clicked.connect(self.process)

        self.log("Video processing stopped", "INFO")

    def on_progress(self, value: int, message: str):
        """Handle progress updates"""
        self.progress_bar.setValue(value)
        self.log(message, "INFO")

    def on_processing_finished(self, prediction):
        """Handle processing completion"""
        try:
            self.current_prediction = prediction
            self.log("Processing complete!", "SUCCESS")

            # Display results
            self.display_results(prediction)

            # Show statistics
            self.show_statistics(prediction)

        except Exception as e:
            self.log(f"Error displaying results: {str(e)}", "ERROR")
        finally:
            self.process_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def on_frame_ready(self, frame, depth_map):
        """Handle real-time video frame"""
        try:
            # Display original frame
            h, w = frame.shape[:2]
            bytes_per_line = 3 * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.original_label.setPixmap(
                pixmap.scaled(
                    self.original_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )
            )

            # Display depth map
            depth_normalized = ((depth_map - depth_map.min()) /
                              (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
            depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

            h, w = depth_rgb.shape[:2]
            bytes_per_line = 3 * w
            q_img_depth = QImage(depth_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap_depth = QPixmap.fromImage(q_img_depth)
            self.depth_label.setPixmap(
                pixmap_depth.scaled(
                    self.depth_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )
            )

        except Exception as e:
            self.log(f"Frame display error: {str(e)}", "ERROR")

    def on_video_finished(self):
        """Handle video processing completion"""
        self.log("Video processing finished", "SUCCESS")
        self.process_btn.setText("Process")
        self.process_btn.clicked.disconnect()
        self.process_btn.clicked.connect(self.process)

    def on_error(self, error_msg: str):
        """Handle errors"""
        self.log(error_msg, "ERROR")
        QMessageBox.critical(self, "Error", error_msg)
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def display_results(self, prediction):
        """Display prediction results"""
        try:
            # Display first depth map
            if prediction.depth is not None and len(prediction.depth) > 0:
                depth_map = prediction.depth[0]

                # Normalize and colorize
                depth_normalized = ((depth_map - depth_map.min()) /
                                  (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
                depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

                # Convert to QPixmap
                h, w = depth_rgb.shape[:2]
                bytes_per_line = 3 * w
                q_img = QImage(depth_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)

                self.depth_label.setPixmap(
                    pixmap.scaled(
                        self.depth_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                )

            # Display confidence map if available
            if prediction.conf is not None and len(prediction.conf) > 0:
                conf_map = prediction.conf[0]

                conf_normalized = ((conf_map - conf_map.min()) /
                                 (conf_map.max() - conf_map.min()) * 255).astype(np.uint8)
                conf_colored = cv2.applyColorMap(conf_normalized, cv2.COLORMAP_JET)
                conf_rgb = cv2.cvtColor(conf_colored, cv2.COLOR_BGR2RGB)

                h, w = conf_rgb.shape[:2]
                bytes_per_line = 3 * w
                q_img = QImage(conf_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)

                self.conf_label.setPixmap(
                    pixmap.scaled(
                        self.conf_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                )

        except Exception as e:
            self.log(f"Display error: {str(e)}", "ERROR")

    def show_statistics(self, prediction):
        """Show prediction statistics"""
        try:
            stats = []
            stats.append("=== Prediction Statistics ===\n")

            if prediction.depth is not None:
                stats.append(f"Number of images: {len(prediction.depth)}")
                stats.append(f"Depth map shape: {prediction.depth[0].shape}")
                stats.append(f"Depth range: [{prediction.depth[0].min():.3f}, {prediction.depth[0].max():.3f}]")
                stats.append(f"Mean depth: {prediction.depth[0].mean():.3f}")
                stats.append(f"Std depth: {prediction.depth[0].std():.3f}\n")

            if prediction.conf is not None:
                stats.append(f"Confidence available: Yes")
                stats.append(f"Mean confidence: {prediction.conf[0].mean():.3f}\n")

            if prediction.extrinsics is not None:
                stats.append(f"Camera extrinsics shape: {prediction.extrinsics.shape}")
                stats.append("First camera extrinsics:")
                stats.append(str(prediction.extrinsics[0]) + "\n")

            if prediction.intrinsics is not None:
                stats.append(f"Camera intrinsics shape: {prediction.intrinsics.shape}")
                stats.append("First camera intrinsics:")
                stats.append(str(prediction.intrinsics[0]))

            self.stats_text.setText("\n".join(stats))

        except Exception as e:
            self.log(f"Statistics error: {str(e)}", "ERROR")

    def open_3d_viewer(self):
        """Open 3D point cloud viewer"""
        if self.current_prediction is None:
            QMessageBox.warning(self, "Warning", "No processed data available")
            return

        try:
            import open3d as o3d

            # Create point cloud from depth
            depth = self.current_prediction.depth[0]
            intrinsics = self.current_prediction.intrinsics[0] if self.current_prediction.intrinsics is not None else None

            if intrinsics is None:
                # Use default intrinsics
                h, w = depth.shape
                fx = fy = w
                cx, cy = w / 2, h / 2
            else:
                fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            # Create point cloud
            h, w = depth.shape
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

            z = depth
            x = (x - cx) * z / fx
            y = (y - cy) * z / fy

            points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

            # Filter invalid points
            valid = ~np.isnan(points).any(axis=1)
            points = points[valid]

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Add color if available
            if self.current_prediction.processed_images is not None:
                colors = self.current_prediction.processed_images[0].reshape(-1, 3)[valid] / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)

            # Visualize
            o3d.visualization.draw_geometries([pcd])

        except ImportError:
            QMessageBox.warning(
                self,
                "Warning",
                "Open3D not installed. Install with: pip install open3d"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"3D visualization error:\n{str(e)}")

    def export_results(self):
        """Export current results"""
        if self.current_prediction is None:
            QMessageBox.warning(self, "Warning", "No results to export")
            return

        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return

        try:
            self.log("Exporting results...", "INFO")

            # Save depth maps as images
            for i, depth in enumerate(self.current_prediction.depth):
                depth_normalized = ((depth - depth.min()) /
                                  (depth.max() - depth.min()) * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

                output_path = os.path.join(export_dir, f"depth_{i:04d}.png")
                cv2.imwrite(output_path, depth_colored)

            # Save as NPZ
            npz_path = os.path.join(export_dir, "prediction.npz")
            np.savez(
                npz_path,
                depth=self.current_prediction.depth,
                conf=self.current_prediction.conf if self.current_prediction.conf is not None else np.array([]),
                extrinsics=self.current_prediction.extrinsics if self.current_prediction.extrinsics is not None else np.array([]),
                intrinsics=self.current_prediction.intrinsics if self.current_prediction.intrinsics is not None else np.array([])
            )

            self.log(f"Results exported to {export_dir}", "SUCCESS")
            QMessageBox.information(self, "Success", f"Results exported to:\n{export_dir}")

        except Exception as e:
            self.log(f"Export error: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def clear_all(self):
        """Clear all data and reset UI"""
        self.current_images = []
        self.current_prediction = None

        self.original_label.setText("Load an image to begin")
        self.original_label.setPixmap(QPixmap())
        self.depth_label.setText("Process to see depth map")
        self.depth_label.setPixmap(QPixmap())
        self.conf_label.setText("Confidence map (if available)")
        self.conf_label.setPixmap(QPixmap())
        self.stats_text.clear()

        self.images_label.setText("No images loaded")
        self.process_btn.setEnabled(False)

        self.log("All data cleared", "INFO")

    def show_help(self):
        """Show help dialog"""
        help_text = """
        <h2>Depth Anything v3 - Professional Edition</h2>

        <h3>Quick Start:</h3>
        <ol>
            <li>Select a model from the dropdown (DA3-LARGE recommended for most uses)</li>
            <li>Click "Load Model" to initialize the model</li>
            <li>Choose a processing mode (Monocular, Multi-View, etc.)</li>
            <li>Load images, video, or select webcam input</li>
            <li>Click "Process" to run depth estimation</li>
        </ol>

        <h3>Processing Modes:</h3>
        <ul>
            <li><b>Monocular Depth:</b> Single image depth estimation</li>
            <li><b>Multi-View Depth:</b> Consistent depth from multiple views</li>
            <li><b>Pose Estimation:</b> Estimate camera poses and intrinsics</li>
            <li><b>3D Gaussians:</b> Generate 3D Gaussian reconstruction</li>
            <li><b>Real-time Video:</b> Process video files frame-by-frame</li>
            <li><b>Webcam:</b> Real-time depth from webcam feed</li>
        </ul>

        <h3>Features:</h3>
        <ul>
            <li>GPU acceleration (CUDA if available)</li>
            <li>Multiple export formats (GLB, PLY, NPZ, depth images)</li>
            <li>Interactive 3D visualization</li>
            <li>Batch processing for folders</li>
            <li>Real-time video/webcam support</li>
            <li>Confidence map visualization</li>
        </ul>

        <h3>System Requirements:</h3>
        <ul>
            <li>Python 3.8+</li>
            <li>PyQt6</li>
            <li>PyTorch 2.0+</li>
            <li>CUDA-capable GPU (recommended)</li>
            <li>8GB+ RAM (16GB+ for large models)</li>
        </ul>

        <p>For more information, visit:
        <a href="https://depth-anything-3.github.io/">Depth Anything v3 Project Page</a></p>
        """

        QMessageBox.about(self, "Help", help_text)

    def closeEvent(self, event):
        """Handle application close"""
        # Stop any running workers
        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.wait()

        if self.depth_worker and self.depth_worker.isRunning():
            self.depth_worker.stop()
            self.depth_worker.wait()

        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Depth Anything v3 Professional")
    app.setOrganizationName("Depth Anything")

    window = DepthAnythingGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
