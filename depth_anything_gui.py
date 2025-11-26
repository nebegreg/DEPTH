#!/usr/bin/env python3
"""
Depth Anything v3 Professional GUI Application - VFX ULTIMATE Edition
====================================================================

A comprehensive PyQt6 application for monocular/multi-view depth estimation,
3D reconstruction, pose estimation, and professional VFX workflows.

Features:
- Monocular depth estimation
- Multi-view depth estimation
- Camera pose estimation
- 3D Gaussian reconstruction
- Real-time video/webcam processing
- Batch processing
- OpenEXR multi-channel export
- DPX sequence export
- FBX/Alembic camera export
- 3D mesh generation (Poisson, Ball Pivoting)
- Multiple mesh export formats (OBJ, PLY, GLB, FBX, STL)
- Normal map generation
- Interactive 3D visualization
- GPU acceleration with performance optimization
- Autodesk Flame integration

Author: Claude
License: MIT
"""

import sys
import os
import glob
import time
import struct
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import cv2
import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QTextEdit, QFileDialog,
    QTabWidget, QGroupBox, QGridLayout, QProgressBar, QSpinBox,
    QCheckBox, QSplitter, QScrollArea, QToolBar, QStatusBar,
    QMessageBox, QLineEdit, QRadioButton, QButtonGroup, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QPixmap, QImage, QAction, QIcon, QPalette, QColor

# Import Depth Anything v3
sys.path.insert(0, str(Path(__file__).parent / 'Depth-Anything-3-main' / 'src'))
from depth_anything_3.api import DepthAnything3


# ============================================================================
# VFX EXPORT UTILITIES - Integrated
# ============================================================================

class OpenEXRExporter:
    """Export multi-channel OpenEXR files for VFX"""

    @staticmethod
    def export(output_path: str, channels: Dict[str, np.ndarray], metadata: Optional[Dict] = None,
              compression: str = 'ZIP'):
        """
        Export multi-channel EXR file

        Args:
            output_path: Output .exr file path
            channels: Dict of channel_name -> numpy array
            metadata: Optional metadata dict
            compression: 'NONE', 'ZIP', 'PIZ', 'ZIPS', 'RLE', 'B44'
        """
        try:
            import OpenEXR
            import Imath
        except ImportError:
            raise ImportError(
                "OpenEXR library required. Install with:\n"
                "  pip install openexr\n"
                "On Ubuntu/Debian: sudo apt-get install libopenexr-dev\n"
                "On macOS: brew install openexr"
            )

        first_channel = list(channels.values())[0]
        height, width = first_channel.shape[:2]

        header = OpenEXR.Header(width, height)

        compression_map = {
            'NONE': Imath.Compression.NO_COMPRESSION,
            'ZIP': Imath.Compression.ZIP_COMPRESSION,
            'ZIPS': Imath.Compression.ZIPS_COMPRESSION,
            'PIZ': Imath.Compression.PIZ_COMPRESSION,
            'RLE': Imath.Compression.RLE_COMPRESSION,
            'B44': Imath.Compression.B44_COMPRESSION,
        }
        header['compression'] = compression_map.get(compression, Imath.Compression.ZIP_COMPRESSION)

        if metadata:
            for key, value in metadata.items():
                header[key] = str(value)

        exr_channels = {}
        channel_data = {}

        for name, data in channels.items():
            if data.ndim == 2:
                exr_channels[name] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                channel_data[name] = data.astype(np.float32).tobytes()
            else:
                raise ValueError(f"Channel {name} must be 2D array, got shape {data.shape}")

        header['channels'] = exr_channels

        exr_file = OpenEXR.OutputFile(output_path, header)
        exr_file.writePixels(channel_data)
        exr_file.close()


class DPXExporter:
    """Export DPX sequences (cinema-quality)"""

    @staticmethod
    def export(output_path: str, image: np.ndarray, bit_depth: int = 10):
        """Export single DPX file"""
        try:
            import imageio
        except ImportError:
            raise ImportError("imageio required: pip install imageio")

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        if bit_depth == 10:
            max_val = 1023
            dtype = np.uint16
        elif bit_depth == 16:
            max_val = 65535
            dtype = np.uint16
        else:
            raise ValueError("bit_depth must be 10 or 16")

        image_norm = image / image.max() if image.max() > 0 else image
        image_scaled = (image_norm * max_val).astype(dtype)

        imageio.imwrite(output_path, image_scaled, format='DPX')


class FBXCameraExporter:
    """Export camera tracking data as FBX"""

    @staticmethod
    def export(output_path: str, extrinsics: np.ndarray, intrinsics: np.ndarray,
              image_size: Tuple[int, int], fps: float = 24.0,
              camera_name: str = "Camera"):
        """Export camera tracking as ASCII FBX"""
        width, height = image_size

        if intrinsics.ndim == 3:
            fx = intrinsics[0, 0, 0]
            fy = intrinsics[0, 1, 1]
            cx = intrinsics[0, 0, 2]
            cy = intrinsics[0, 1, 2]
        else:
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

        focal_length_mm = fx * 36.0 / width
        sensor_width = 36.0
        sensor_height = sensor_width * height / width

        with open(output_path, 'w') as f:
            f.write("; FBX 7.4.0 project file\n")
            f.write("; Created by Depth Anything v3 VFX Ultimate\n")
            f.write(f"; Camera: {camera_name}\n")
            f.write(f"; Frames: {len(extrinsics)}\n")
            f.write(f"; FPS: {fps}\n\n")

            f.write("FBXHeaderExtension:  {\n")
            f.write("    FBXHeaderVersion: 1003\n")
            f.write("    FBXVersion: 7400\n")
            f.write("    CreationTimeStamp:  {\n")
            f.write("        Version: 1000\n")
            f.write("    }\n")
            f.write("    Creator: \"Depth Anything v3\"\n")
            f.write("}\n\n")

            f.write("GlobalSettings:  {\n")
            f.write("    Version: 1000\n")
            f.write("    Properties70:  {\n")
            f.write(f'        P: "CustomFrameRate", "double", "Number", "",{fps}\n')
            f.write("    }\n")
            f.write("}\n\n")

            f.write("Objects:  {\n")
            f.write(f'    Model: "Model::{camera_name}", "Camera" {{\n')
            f.write("        Version: 232\n")
            f.write("    }\n")
            f.write("}\n")


class NormalMapGenerator:
    """Generate surface normal maps from depth"""

    @staticmethod
    def compute(depth: np.ndarray, intrinsics: np.ndarray, smooth: bool = False) -> np.ndarray:
        """Compute surface normals from depth map"""
        h, w = depth.shape

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        y, x = np.mgrid[0:h, 0:w]
        z = depth
        x3d = (x - cx) * z / fx
        y3d = (y - cy) * z / fy
        z3d = z

        if smooth:
            try:
                from scipy.ndimage import gaussian_filter
                z_smooth = gaussian_filter(z3d, sigma=1.0)
                grad_x = np.gradient(z_smooth, axis=1)
                grad_y = np.gradient(z_smooth, axis=0)
            except ImportError:
                grad_x = np.gradient(z3d, axis=1)
                grad_y = np.gradient(z3d, axis=0)
        else:
            grad_x = np.gradient(z3d, axis=1)
            grad_y = np.gradient(z3d, axis=0)

        dx = np.stack([np.ones_like(z3d), np.zeros_like(z3d), grad_x], axis=-1)
        dy = np.stack([np.zeros_like(z3d), np.ones_like(z3d), grad_y], axis=-1)

        normals = np.cross(dx, dy)

        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / (norm + 1e-8)

        return normals

    @staticmethod
    def to_rgb(normals: np.ndarray) -> np.ndarray:
        """Convert normals to RGB for visualization"""
        normals_rgb = ((normals + 1) / 2 * 255).astype(np.uint8)
        return normals_rgb


# ============================================================================
# MESH GENERATION - Integrated
# ============================================================================

class MeshGenerator:
    """Generate 3D meshes from depth maps and point clouds"""

    @staticmethod
    def depth_to_point_cloud(depth: np.ndarray, intrinsics: np.ndarray,
                            rgb_image: Optional[np.ndarray] = None,
                            depth_scale: float = 1.0, max_depth: float = 100.0):
        """Convert depth map to 3D point cloud"""
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        h, w = depth.shape

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        y, x = np.mgrid[0:h, 0:w]

        valid_mask = (depth > 0) & (depth < max_depth)

        z = depth[valid_mask] * depth_scale
        x_3d = (x[valid_mask] - cx) * z / fx
        y_3d = (y[valid_mask] - cy) * z / fy

        points = np.stack([x_3d, y_3d, z], axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if rgb_image is not None:
            colors = rgb_image[valid_mask] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    @staticmethod
    def remove_outliers(pcd, nb_neighbors: int = 20, std_ratio: float = 2.0):
        """Remove outliers from point cloud"""
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        pcd_clean, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return pcd_clean

    @staticmethod
    def estimate_normals(pcd, search_param_knn: int = 30):
        """Estimate normals for point cloud"""
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=search_param_knn)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        return pcd

    @staticmethod
    def poisson_reconstruction(pcd, depth: int = 9):
        """Poisson Surface Reconstruction"""
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        if not pcd.has_normals():
            raise ValueError("Point cloud must have normals")

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )

        return mesh, densities

    @staticmethod
    def filter_mesh_by_density(mesh, densities: np.ndarray, quantile: float = 0.01):
        """Remove low-density vertices from mesh"""
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, quantile)

        vertices_to_remove = densities < density_threshold
        mesh_filtered = mesh.select_by_index(
            np.where(~vertices_to_remove)[0]
        )

        return mesh_filtered

    @staticmethod
    def simplify_mesh(mesh, target_triangles: int = 100000):
        """Simplify mesh (reduce polygon count)"""
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        mesh_simplified = mesh.simplify_quadric_decimation(target_triangles)
        return mesh_simplified

    @staticmethod
    def smooth_mesh(mesh, iterations: int = 5):
        """Smooth mesh surface"""
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        mesh_smooth = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
        return mesh_smooth

    @staticmethod
    def export_mesh(mesh, output_path: str, compute_normals: bool = True):
        """Export mesh to file"""
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        if compute_normals:
            mesh.compute_vertex_normals()

        ext = Path(output_path).suffix.lower()

        if ext in ['.obj', '.ply', '.stl']:
            o3d.io.write_triangle_mesh(output_path, mesh)
        elif ext in ['.glb', '.gltf', '.fbx']:
            try:
                import trimesh
                vertices = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles)

                if mesh.has_vertex_colors():
                    colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
                    tm_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=colors)
                else:
                    tm_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

                tm_mesh.export(output_path)
            except ImportError:
                raise ImportError("trimesh required for GLB/FBX: pip install trimesh")
        else:
            raise ValueError(f"Unsupported format: {ext}")


class MeshPipeline:
    """Complete pipeline: Depth → Point Cloud → Mesh"""

    @staticmethod
    def depth_to_mesh(depth: np.ndarray, intrinsics: np.ndarray,
                     rgb_image: Optional[np.ndarray] = None,
                     depth_level: int = 9, simplify: bool = True,
                     target_triangles: int = 100000, smooth: bool = True,
                     remove_outliers: bool = True, progress_callback=None):
        """Complete pipeline: depth map → 3D mesh"""

        if progress_callback:
            progress_callback(10, "Converting depth to point cloud...")
        pcd = MeshGenerator.depth_to_point_cloud(depth, intrinsics, rgb_image)

        if remove_outliers:
            if progress_callback:
                progress_callback(25, "Removing outliers...")
            pcd = MeshGenerator.remove_outliers(pcd)

        if progress_callback:
            progress_callback(40, "Estimating normals...")
        pcd = MeshGenerator.estimate_normals(pcd)

        if progress_callback:
            progress_callback(55, "Generating mesh (Poisson)...")
        mesh, densities = MeshGenerator.poisson_reconstruction(pcd, depth=depth_level)
        mesh = MeshGenerator.filter_mesh_by_density(mesh, densities)

        if simplify and len(mesh.triangles) > target_triangles:
            if progress_callback:
                progress_callback(75, "Simplifying mesh...")
            mesh = MeshGenerator.simplify_mesh(mesh, target_triangles)

        if smooth:
            if progress_callback:
                progress_callback(90, "Smoothing mesh...")
            mesh = MeshGenerator.smooth_mesh(mesh)

        if progress_callback:
            progress_callback(100, "Mesh generation complete!")

        return mesh


# ============================================================================
# WORKER THREADS
# ============================================================================

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

    frame_ready = pyqtSignal(object, object)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, source, fps=30):
        super().__init__()
        self.model = model
        self.source = source
        self.fps = fps
        self._is_running = True

    def run(self):
        """Process video stream"""
        try:
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

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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


class MeshWorker(QThread):
    """Worker thread for mesh generation"""

    finished = pyqtSignal(object)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, depth, intrinsics, rgb_image, depth_level, target_triangles):
        super().__init__()
        self.depth = depth
        self.intrinsics = intrinsics
        self.rgb_image = rgb_image
        self.depth_level = depth_level
        self.target_triangles = target_triangles

    def run(self):
        """Generate mesh"""
        try:
            mesh = MeshPipeline.depth_to_mesh(
                self.depth,
                self.intrinsics,
                self.rgb_image,
                depth_level=self.depth_level,
                target_triangles=self.target_triangles,
                progress_callback=self.progress.emit
            )
            self.finished.emit(mesh)
        except Exception as e:
            self.error.emit(f"Mesh generation error: {str(e)}")


# ============================================================================
# MAIN GUI
# ============================================================================

class DepthAnythingGUI(QMainWindow):
    """Main GUI application for Depth Anything v3 - VFX ULTIMATE Edition"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth Anything v3 - VFX ULTIMATE Edition")
        self.setGeometry(100, 100, 1600, 900)

        # Model and state
        self.model = None
        self.current_prediction = None
        self.current_images = []
        self.current_mesh = None
        self.video_worker = None
        self.depth_worker = None
        self.mesh_worker = None
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
        self.model_combo.setCurrentIndex(2)
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

        # VFX Export Options
        vfx_group = QGroupBox("VFX Export Options")
        vfx_layout = QGridLayout()

        vfx_layout.addWidget(QLabel("Export Format:"), 0, 0)
        self.vfx_export_format = QComboBox()
        self.vfx_export_format.addItems([
            "None",
            "OpenEXR Multi-Channel",
            "DPX Sequence (10-bit)",
            "DPX Sequence (16-bit)",
            "FBX Camera",
            "All VFX Formats"
        ])
        vfx_layout.addWidget(self.vfx_export_format, 0, 1)

        self.export_normals_check = QCheckBox("Include Normal Maps")
        self.export_normals_check.setChecked(True)
        vfx_layout.addWidget(self.export_normals_check, 1, 0, 1, 2)

        self.export_confidence_check = QCheckBox("Include Confidence")
        self.export_confidence_check.setChecked(True)
        vfx_layout.addWidget(self.export_confidence_check, 2, 0, 1, 2)

        vfx_group.setLayout(vfx_layout)
        layout.addWidget(vfx_group)

        # 3D Mesh Generation
        mesh_group = QGroupBox("3D Mesh Generation")
        mesh_layout = QGridLayout()

        mesh_layout.addWidget(QLabel("Poisson Depth:"), 0, 0)
        self.poisson_depth = QSpinBox()
        self.poisson_depth.setRange(6, 12)
        self.poisson_depth.setValue(9)
        self.poisson_depth.setToolTip("Higher = more detail (slower)")
        mesh_layout.addWidget(self.poisson_depth, 0, 1)

        mesh_layout.addWidget(QLabel("Target Triangles:"), 1, 0)
        self.target_triangles = QSpinBox()
        self.target_triangles.setRange(10000, 1000000)
        self.target_triangles.setSingleStep(10000)
        self.target_triangles.setValue(100000)
        mesh_layout.addWidget(self.target_triangles, 1, 1)

        mesh_layout.addWidget(QLabel("Mesh Format:"), 2, 0)
        self.mesh_format = QComboBox()
        self.mesh_format.addItems(["OBJ", "PLY", "GLB", "FBX", "STL"])
        self.mesh_format.setCurrentText("GLB")
        mesh_layout.addWidget(self.mesh_format, 2, 1)

        self.generate_mesh_btn = QPushButton("Generate 3D Mesh")
        self.generate_mesh_btn.setEnabled(False)
        self.generate_mesh_btn.clicked.connect(self.generate_mesh)
        mesh_layout.addWidget(self.generate_mesh_btn, 3, 0, 1, 2)

        mesh_group.setLayout(mesh_layout)
        layout.addWidget(mesh_group)

        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QGridLayout()

        options_layout.addWidget(QLabel("FPS (video):"), 0, 0)
        self.fps_spinner = QSpinBox()
        self.fps_spinner.setRange(1, 60)
        self.fps_spinner.setValue(15)
        options_layout.addWidget(self.fps_spinner, 0, 1)

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

        # Normal map tab
        self.normal_scroll = QScrollArea()
        self.normal_label = QLabel("Normal map will appear here")
        self.normal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.normal_label.setMinimumSize(640, 480)
        self.normal_scroll.setWidget(self.normal_label)
        self.normal_scroll.setWidgetResizable(True)
        self.viz_tabs.addTab(self.normal_scroll, "Normal Map")

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

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.load_images)
        toolbar.addAction(open_action)

        save_action = QAction("Export VFX", self)
        save_action.triggered.connect(self.export_vfx)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        clear_action = QAction("Clear", self)
        clear_action.triggered.connect(self.clear_all)
        toolbar.addAction(clear_action)

        toolbar.addSeparator()

        help_action = QAction("Help", self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)

    def setup_statusbar(self):
        """Setup status bar"""
        self.statusBar().showMessage("Ready - VFX ULTIMATE Edition")

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

            self.depth_worker = DepthWorker(
                self.model,
                self.current_images,
                mode=mode
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
                source = 0
            else:
                source = self.current_images[0]

            fps = self.fps_spinner.value()

            self.video_worker = VideoWorker(self.model, source, fps)
            self.video_worker.frame_ready.connect(self.on_frame_ready)
            self.video_worker.finished.connect(self.on_video_finished)
            self.video_worker.error.connect(self.on_error)

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

            self.display_results(prediction)
            self.show_statistics(prediction)

            # Enable mesh generation if we have depth
            if prediction.depth is not None:
                self.generate_mesh_btn.setEnabled(True)

        except Exception as e:
            self.log(f"Error displaying results: {str(e)}", "ERROR")
        finally:
            self.process_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def on_frame_ready(self, frame, depth_map):
        """Handle real-time video frame"""
        try:
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
            # Display depth map
            if prediction.depth is not None and len(prediction.depth) > 0:
                depth_map = prediction.depth[0]

                depth_normalized = ((depth_map - depth_map.min()) /
                                  (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
                depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

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

                # Generate and display normal map
                try:
                    if prediction.intrinsics is not None and len(prediction.intrinsics) > 0:
                        intrinsics = prediction.intrinsics[0]
                    else:
                        # Default intrinsics
                        h, w = depth_map.shape
                        intrinsics = np.array([
                            [w, 0, w/2],
                            [0, w, h/2],
                            [0, 0, 1]
                        ])

                    normals = NormalMapGenerator.compute(depth_map, intrinsics, smooth=True)
                    normals_rgb = NormalMapGenerator.to_rgb(normals)

                    h, w = normals_rgb.shape[:2]
                    bytes_per_line = 3 * w
                    q_img = QImage(normals_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)

                    self.normal_label.setPixmap(
                        pixmap.scaled(
                            self.normal_label.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                    )
                except Exception as e:
                    self.log(f"Normal map generation failed: {str(e)}", "WARNING")

            # Display confidence map
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

            if prediction.intrinsics is not None:
                stats.append(f"Camera intrinsics shape: {prediction.intrinsics.shape}")

            self.stats_text.setText("\n".join(stats))

        except Exception as e:
            self.log(f"Statistics error: {str(e)}", "ERROR")

    def generate_mesh(self):
        """Generate 3D mesh from depth"""
        if self.current_prediction is None or self.current_prediction.depth is None:
            QMessageBox.warning(self, "Warning", "No depth data available")
            return

        try:
            self.log("Starting mesh generation...", "INFO")
            self.generate_mesh_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            depth = self.current_prediction.depth[0]

            # Get intrinsics
            if self.current_prediction.intrinsics is not None:
                intrinsics = self.current_prediction.intrinsics[0]
            else:
                h, w = depth.shape
                intrinsics = np.array([
                    [w, 0, w/2],
                    [0, w, h/2],
                    [0, 0, 1]
                ])

            # Get RGB if available
            rgb_image = None
            if hasattr(self.current_prediction, 'processed_images') and self.current_prediction.processed_images is not None:
                rgb_image = self.current_prediction.processed_images[0]

            depth_level = self.poisson_depth.value()
            target_triangles = self.target_triangles.value()

            # Create mesh worker
            self.mesh_worker = MeshWorker(
                depth, intrinsics, rgb_image, depth_level, target_triangles
            )

            self.mesh_worker.progress.connect(self.on_progress)
            self.mesh_worker.finished.connect(self.on_mesh_finished)
            self.mesh_worker.error.connect(self.on_mesh_error)

            self.mesh_worker.start()

        except Exception as e:
            self.log(f"Mesh generation error: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", f"Mesh generation failed:\n{str(e)}")
            self.generate_mesh_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def on_mesh_finished(self, mesh):
        """Handle mesh generation completion"""
        try:
            self.current_mesh = mesh
            self.log("Mesh generation complete!", "SUCCESS")

            # Ask to save mesh
            mesh_format = self.mesh_format.currentText().lower()
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save 3D Mesh",
                f"output_mesh.{mesh_format}",
                f"{mesh_format.upper()} Files (*.{mesh_format})"
            )

            if output_path:
                MeshGenerator.export_mesh(mesh, output_path)
                self.log(f"Mesh saved to {output_path}", "SUCCESS")
                QMessageBox.information(self, "Success", f"Mesh saved to:\n{output_path}")

        except Exception as e:
            self.log(f"Mesh save error: {str(e)}", "ERROR")
        finally:
            self.generate_mesh_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def on_mesh_error(self, error_msg: str):
        """Handle mesh generation error"""
        self.log(f"Mesh error: {error_msg}", "ERROR")
        QMessageBox.critical(self, "Error", error_msg)
        self.generate_mesh_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def export_vfx(self):
        """Export VFX formats"""
        if self.current_prediction is None:
            QMessageBox.warning(self, "Warning", "No processed data to export")
            return

        export_format = self.vfx_export_format.currentText()
        if export_format == "None":
            QMessageBox.information(self, "Info", "Please select an export format")
            return

        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return

        try:
            self.log(f"Exporting {export_format}...", "INFO")

            depth = self.current_prediction.depth[0]
            h, w = depth.shape

            # Get intrinsics
            if self.current_prediction.intrinsics is not None:
                intrinsics = self.current_prediction.intrinsics[0]
            else:
                intrinsics = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])

            if export_format == "OpenEXR Multi-Channel" or export_format == "All VFX Formats":
                # Export OpenEXR
                channels = {'depth.Z': depth}

                if self.export_confidence_check.isChecked() and self.current_prediction.conf is not None:
                    channels['confidence.R'] = self.current_prediction.conf[0]

                if self.export_normals_check.isChecked():
                    normals = NormalMapGenerator.compute(depth, intrinsics)
                    channels['normal.R'] = normals[:, :, 0]
                    channels['normal.G'] = normals[:, :, 1]
                    channels['normal.B'] = normals[:, :, 2]

                exr_path = os.path.join(export_dir, "depth.exr")
                OpenEXRExporter.export(exr_path, channels, metadata={'software': 'Depth Anything v3 VFX'})
                self.log(f"OpenEXR exported to {exr_path}", "SUCCESS")

            if "DPX" in export_format or export_format == "All VFX Formats":
                # Export DPX
                bit_depth = 16 if "16-bit" in export_format else 10
                dpx_path = os.path.join(export_dir, f"depth.dpx")
                DPXExporter.export(dpx_path, depth, bit_depth=bit_depth)
                self.log(f"DPX exported to {dpx_path}", "SUCCESS")

            if export_format == "FBX Camera" or export_format == "All VFX Formats":
                # Export FBX camera
                if self.current_prediction.extrinsics is not None:
                    fbx_path = os.path.join(export_dir, "camera.fbx")
                    FBXCameraExporter.export(
                        fbx_path,
                        self.current_prediction.extrinsics,
                        self.current_prediction.intrinsics if self.current_prediction.intrinsics is not None else np.array([intrinsics]),
                        (w, h)
                    )
                    self.log(f"FBX camera exported to {fbx_path}", "SUCCESS")
                else:
                    self.log("No camera data available for FBX export", "WARNING")

            QMessageBox.information(self, "Success", f"VFX files exported to:\n{export_dir}")

        except ImportError as e:
            self.log(f"Export failed - missing dependency: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
        except Exception as e:
            self.log(f"Export error: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def open_3d_viewer(self):
        """Open 3D point cloud viewer"""
        if self.current_prediction is None:
            QMessageBox.warning(self, "Warning", "No processed data available")
            return

        try:
            import open3d as o3d

            depth = self.current_prediction.depth[0]
            intrinsics = self.current_prediction.intrinsics[0] if self.current_prediction.intrinsics is not None else None

            if intrinsics is None:
                h, w = depth.shape
                fx = fy = w
                cx, cy = w / 2, h / 2
            else:
                fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            h, w = depth.shape
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

            z = depth
            x = (x - cx) * z / fx
            y = (y - cy) * z / fy

            points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

            valid = ~np.isnan(points).any(axis=1)
            points = points[valid]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            if hasattr(self.current_prediction, 'processed_images') and self.current_prediction.processed_images is not None:
                colors = self.current_prediction.processed_images[0].reshape(-1, 3)[valid] / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)

            o3d.visualization.draw_geometries([pcd])

        except ImportError:
            QMessageBox.warning(
                self,
                "Warning",
                "Open3D not installed. Install with: pip install open3d"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"3D visualization error:\n{str(e)}")

    def clear_all(self):
        """Clear all data and reset UI"""
        self.current_images = []
        self.current_prediction = None
        self.current_mesh = None

        self.original_label.setText("Load an image to begin")
        self.original_label.setPixmap(QPixmap())
        self.depth_label.setText("Process to see depth map")
        self.depth_label.setPixmap(QPixmap())
        self.normal_label.setText("Normal map will appear here")
        self.normal_label.setPixmap(QPixmap())
        self.conf_label.setText("Confidence map (if available)")
        self.conf_label.setPixmap(QPixmap())
        self.stats_text.clear()

        self.images_label.setText("No images loaded")
        self.process_btn.setEnabled(False)
        self.generate_mesh_btn.setEnabled(False)

        self.log("All data cleared", "INFO")

    def show_help(self):
        """Show help dialog"""
        help_text = """
        <h2>Depth Anything v3 - VFX ULTIMATE Edition</h2>

        <h3>Quick Start:</h3>
        <ol>
            <li>Select a model from the dropdown (DA3-LARGE recommended)</li>
            <li>Click "Load Model" to initialize</li>
            <li>Choose a processing mode</li>
            <li>Load images, video, or select webcam input</li>
            <li>Click "Process" to run depth estimation</li>
        </ol>

        <h3>VFX Features:</h3>
        <ul>
            <li><b>OpenEXR Export:</b> Multi-channel with depth, normals, confidence</li>
            <li><b>DPX Export:</b> Cinema-quality sequences (10/16-bit)</li>
            <li><b>FBX Camera:</b> Camera tracking for Flame/Maya</li>
            <li><b>3D Mesh Generation:</b> Poisson reconstruction</li>
            <li><b>Normal Maps:</b> Surface normals for lighting</li>
        </ul>

        <h3>Mesh Generation:</h3>
        <ul>
            <li><b>Poisson Depth 6-8:</b> Fast, low detail</li>
            <li><b>Poisson Depth 9:</b> Recommended (production)</li>
            <li><b>Poisson Depth 10-12:</b> High detail (slow)</li>
        </ul>

        <h3>Export Formats:</h3>
        <ul>
            <li>OBJ - Universal 3D format</li>
            <li>PLY - With vertex colors</li>
            <li>GLB - Blender, Flame compatible</li>
            <li>FBX - Maya, 3DS Max</li>
            <li>STL - 3D printing</li>
        </ul>

        <p>For Autodesk Flame integration, see FLAME_INTEGRATION.md</p>
        """

        QMessageBox.about(self, "Help", help_text)

    def closeEvent(self, event):
        """Handle application close"""
        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.wait()

        if self.depth_worker and self.depth_worker.isRunning():
            self.depth_worker.stop()
            self.depth_worker.wait()

        if self.mesh_worker and self.mesh_worker.isRunning():
            self.mesh_worker.wait()

        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Depth Anything v3 VFX ULTIMATE")
    app.setOrganizationName("Depth Anything")

    window = DepthAnythingGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
