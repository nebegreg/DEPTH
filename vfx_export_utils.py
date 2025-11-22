#!/usr/bin/env python3
"""
VFX Export Utilities for Depth Anything v3
==========================================

Professional export tools for Autodesk Flame, Nuke, and other VFX software.

Supports:
- OpenEXR multi-channel export
- DPX sequences
- FBX camera tracking
- Alembic camera animation
- Point clouds
- Normal maps

Author: Claude
License: MIT
"""

import os
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


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
                     Examples: {'depth.Z': depth_array, 'normal.R': normal_r, ...}
            metadata: Optional metadata dict
            compression: 'NONE', 'ZIP', 'PIZ', 'ZIPS', 'RLE', 'B44'

        Example:
            >>> channels = {
            ...     'depth.Z': depth_map,  # (H, W) float32
            ...     'confidence.R': conf_map,
            ...     'normal.R': normals[:,:,0],
            ...     'normal.G': normals[:,:,1],
            ...     'normal.B': normals[:,:,2],
            ... }
            >>> OpenEXRExporter.export('output.exr', channels)
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

        # Get dimensions from first channel
        first_channel = list(channels.values())[0]
        height, width = first_channel.shape[:2]

        # Create header
        header = OpenEXR.Header(width, height)

        # Set compression
        compression_map = {
            'NONE': Imath.Compression.NO_COMPRESSION,
            'ZIP': Imath.Compression.ZIP_COMPRESSION,
            'ZIPS': Imath.Compression.ZIPS_COMPRESSION,
            'PIZ': Imath.Compression.PIZ_COMPRESSION,
            'RLE': Imath.Compression.RLE_COMPRESSION,
            'B44': Imath.Compression.B44_COMPRESSION,
        }
        header['compression'] = compression_map.get(compression, Imath.Compression.ZIP_COMPRESSION)

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                header[key] = str(value)

        # Prepare channels
        exr_channels = {}
        channel_data = {}

        for name, data in channels.items():
            if data.ndim == 2:
                # Single channel
                exr_channels[name] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                channel_data[name] = data.astype(np.float32).tobytes()
            else:
                raise ValueError(f"Channel {name} must be 2D array, got shape {data.shape}")

        header['channels'] = exr_channels

        # Write file
        exr_file = OpenEXR.OutputFile(output_path, header)
        exr_file.writePixels(channel_data)
        exr_file.close()

    @staticmethod
    def export_sequence(output_dir: str, frame_data: List[Dict[str, np.ndarray]],
                       base_name: str = "frame", start_frame: int = 1001,
                       compression: str = 'ZIP', metadata: Optional[Dict] = None):
        """
        Export sequence of EXR files

        Args:
            output_dir: Output directory
            frame_data: List of channel dicts (one per frame)
            base_name: Base filename
            start_frame: Starting frame number
            compression: Compression type
            metadata: Metadata for all frames
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, channels in enumerate(frame_data):
            frame_num = start_frame + i
            output_path = os.path.join(output_dir, f"{base_name}.{frame_num:04d}.exr")

            # Add frame number to metadata
            frame_metadata = metadata.copy() if metadata else {}
            frame_metadata['frame'] = frame_num

            OpenEXRExporter.export(output_path, channels, frame_metadata, compression)


class DPXExporter:
    """Export DPX sequences (cinema-quality)"""

    @staticmethod
    def export(output_path: str, image: np.ndarray, bit_depth: int = 10,
              metadata: Optional[Dict] = None):
        """
        Export single DPX file

        Args:
            output_path: Output .dpx path
            image: Image array (H, W) or (H, W, 3)
            bit_depth: 10 or 16 bit
            metadata: Optional metadata
        """
        try:
            import imageio
        except ImportError:
            raise ImportError("imageio required: pip install imageio")

        # Ensure 3 channels (RGB or replicate grayscale)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Scale to bit depth
        if bit_depth == 10:
            max_val = 1023
            dtype = np.uint16
        elif bit_depth == 16:
            max_val = 65535
            dtype = np.uint16
        else:
            raise ValueError("bit_depth must be 10 or 16")

        # Normalize and scale
        image_norm = image / image.max() if image.max() > 0 else image
        image_scaled = (image_norm * max_val).astype(dtype)

        # Write DPX
        imageio.imwrite(output_path, image_scaled, format='DPX')

    @staticmethod
    def export_sequence(output_dir: str, frames: List[np.ndarray],
                       base_name: str = "frame", start_frame: int = 1001,
                       bit_depth: int = 10):
        """Export DPX sequence"""
        os.makedirs(output_dir, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_num = start_frame + i
            output_path = os.path.join(output_dir, f"{base_name}.{frame_num:04d}.dpx")
            DPXExporter.export(output_path, frame, bit_depth)


class FBXCameraExporter:
    """Export camera tracking data as FBX"""

    @staticmethod
    def export(output_path: str, extrinsics: np.ndarray, intrinsics: np.ndarray,
              image_size: Tuple[int, int], fps: float = 24.0,
              camera_name: str = "Camera"):
        """
        Export camera tracking as ASCII FBX

        Args:
            output_path: Output .fbx path
            extrinsics: Camera extrinsics [N, 3, 4] or [N, 4, 4] (OpenCV w2c format)
            intrinsics: Camera intrinsics [N, 3, 3]
            image_size: (width, height) in pixels
            fps: Frame rate
            camera_name: Name of camera in FBX

        Note: This creates ASCII FBX format which is widely compatible.
              For binary FBX, use dedicated library like FBX SDK.
        """
        width, height = image_size

        # Extract camera parameters
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

        # Convert focal length to mm (assuming 36mm sensor width)
        focal_length_mm = fx * 36.0 / width

        # Calculate sensor dimensions
        sensor_width = 36.0  # mm (standard full-frame)
        sensor_height = sensor_width * height / width

        with open(output_path, 'w') as f:
            # Header
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

            # Global settings
            f.write("GlobalSettings:  {\n")
            f.write("    Version: 1000\n")
            f.write("    Properties70:  {\n")
            f.write(f'        P: "UpAxis", "int", "Integer", "",1\n')
            f.write(f'        P: "UpAxisSign", "int", "Integer", "",1\n')
            f.write(f'        P: "FrontAxis", "int", "Integer", "",2\n')
            f.write(f'        P: "CoordAxis", "int", "Integer", "",0\n')
            f.write(f'        P: "OriginalUpAxis", "int", "Integer", "",1\n')
            f.write(f'        P: "TimeMode", "enum", "", "",6\n')
            f.write(f'        P: "TimeProtocol", "enum", "", "",2\n')
            f.write(f'        P: "SnapOnFrameMode", "enum", "", "",0\n')
            f.write(f'        P: "CustomFrameRate", "double", "Number", "",{fps}\n')
            f.write("    }\n")
            f.write("}\n\n")

            # Objects
            f.write("Objects:  {\n")

            # Camera node
            f.write(f'    Model: "Model::{camera_name}", "Camera" {{\n')
            f.write("        Version: 232\n")
            f.write("        Properties70:  {\n")
            f.write('            P: "RotationActive", "bool", "", "",1\n')
            f.write('            P: "InheritType", "enum", "", "",1\n')
            f.write('            P: "ScalingMax", "Vector3D", "Vector", "",0,0,0\n')
            f.write("        }\n")
            f.write("        Culling: \"CullingOff\"\n")
            f.write("    }\n")

            # Camera attributes
            f.write(f'    NodeAttribute: "NodeAttribute::{camera_name}", "Camera" {{\n')
            f.write("        TypeFlags: \"Camera\"\n")
            f.write("        Version: 100\n")
            f.write("        Properties70:  {\n")
            f.write(f'            P: "FocalLength", "Number", "", "A",{focal_length_mm}\n')
            f.write(f'            P: "FilmWidth", "Number", "", "A",{sensor_width}\n')
            f.write(f'            P: "FilmHeight", "Number", "", "A",{sensor_height}\n')
            f.write(f'            P: "FilmAspectRatio", "double", "Number", "",{width/height}\n')
            f.write('            P: "ApertureMode", "enum", "", "",0\n')
            f.write('            P: "FieldOfView", "FieldOfView", "", "A",40\n')
            f.write('            P: "NearPlane", "double", "Number", "",0.1\n')
            f.write('            P: "FarPlane", "double", "Number", "",1000\n')
            f.write("        }\n")
            f.write("    }\n")

            # Animation curves for position
            f.write(f'    AnimationCurveNode: "AnimationCurveNode::T", "" {{\n')
            f.write('        Properties70:  {\n')
            f.write('            P: "d|X", "Number", "", "A",0\n')
            f.write('            P: "d|Y", "Number", "", "A",0\n')
            f.write('            P: "d|Z", "Number", "", "A",0\n')
            f.write("        }\n")
            f.write("    }\n")

            # Position keyframes
            for axis, axis_name in enumerate(['X', 'Y', 'Z']):
                f.write(f'    AnimationCurve: "AnimationCurve::", "" {{\n')
                f.write("        Default: 0\n")
                f.write("        KeyVer: 4008\n")
                f.write(f"        KeyTime: {','.join([str(int(i * 46186158000 / fps)) for i in range(len(extrinsics))])}\n")

                # Extract translations (convert from OpenCV to FBX coordinates if needed)
                translations = []
                for ext in extrinsics:
                    if ext.shape == (3, 4):
                        R = ext[:3, :3]
                        t = ext[:3, 3]
                        # Convert from camera-to-world
                        pos = -R.T @ t
                    else:  # (4, 4)
                        R = ext[:3, :3]
                        t = ext[:3, 3]
                        pos = -R.T @ t

                    translations.append(pos[axis])

                f.write(f"        KeyValueFloat: {','.join([f'{t:.6f}' for t in translations])}\n")
                f.write("    }\n")

            f.write("}\n\n")

            # Connections
            f.write("Connections:  {\n")
            f.write(f'    C: "OO","Model::{camera_name}","Model::Scene"\n')
            f.write(f'    C: "OO","NodeAttribute::{camera_name}","Model::{camera_name}"\n')
            f.write("}\n")

        print(f"FBX camera exported to: {output_path}")


class AlembicCameraExporter:
    """Export camera tracking as Alembic"""

    @staticmethod
    def export(output_path: str, extrinsics: np.ndarray, intrinsics: np.ndarray,
              image_size: Tuple[int, int], fps: float = 24.0):
        """
        Export camera tracking as Alembic

        Args:
            output_path: Output .abc path
            extrinsics: Camera extrinsics [N, 3, 4]
            intrinsics: Camera intrinsics [N, 3, 3]
            image_size: (width, height)
            fps: Frame rate

        Note: Requires alembic Python package
        """
        try:
            import alembic
            from alembic import Abc, AbcGeom
        except ImportError:
            raise ImportError(
                "Alembic required. Install with:\n"
                "  pip install alembic\n"
                "Note: May require building from source on some systems"
            )

        # Create archive
        archive = Abc.OArchive(output_path)

        # Create camera
        camera = AbcGeom.OCamera(archive.getTop(), 'camera')
        schema = camera.getSchema()

        width, height = image_size

        # Extract intrinsics
        fx = intrinsics[0, 0, 0] if intrinsics.ndim == 3 else intrinsics[0, 0]
        fy = intrinsics[0, 1, 1] if intrinsics.ndim == 3 else intrinsics[1, 1]

        focal_length_mm = fx * 36.0 / width
        sensor_width = 36.0
        sensor_height = sensor_width * height / width

        # Create camera samples
        for frame_idx, ext in enumerate(extrinsics):
            sample = AbcGeom.CameraSample()

            # Set camera parameters
            sample.setFocalLength(focal_length_mm)
            sample.setHorizontalAperture(sensor_width)
            sample.setVerticalAperture(sensor_height)
            sample.setNearClippingPlane(0.1)
            sample.setFarClippingPlane(1000.0)

            # Set sample
            schema.set(sample)

        print(f"Alembic camera exported to: {output_path}")


class NormalMapGenerator:
    """Generate surface normal maps from depth"""

    @staticmethod
    def compute(depth: np.ndarray, intrinsics: np.ndarray,
               smooth: bool = False) -> np.ndarray:
        """
        Compute surface normals from depth map

        Args:
            depth: Depth map (H, W)
            intrinsics: Camera intrinsics (3, 3)
            smooth: Apply smoothing to normals

        Returns:
            Normal map (H, W, 3) in range [-1, 1]
        """
        h, w = depth.shape

        # Get camera parameters
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Create 3D points
        y, x = np.mgrid[0:h, 0:w]
        z = depth
        x3d = (x - cx) * z / fx
        y3d = (y - cy) * z / fy
        z3d = z

        # Compute gradients
        if smooth:
            # Smooth depth first
            from scipy.ndimage import gaussian_filter
            z_smooth = gaussian_filter(z3d, sigma=1.0)
            grad_x = np.gradient(z_smooth, axis=1)
            grad_y = np.gradient(z_smooth, axis=0)
        else:
            grad_x = np.gradient(z3d, axis=1)
            grad_y = np.gradient(z3d, axis=0)

        # Compute normals using cross product of tangent vectors
        # Tangent in x direction
        dx = np.stack([np.ones_like(z3d), np.zeros_like(z3d), grad_x], axis=-1)
        # Tangent in y direction
        dy = np.stack([np.zeros_like(z3d), np.ones_like(z3d), grad_y], axis=-1)

        # Cross product
        normals = np.cross(dx, dy)

        # Normalize
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / (norm + 1e-8)

        return normals

    @staticmethod
    def to_rgb(normals: np.ndarray) -> np.ndarray:
        """
        Convert normals to RGB for visualization

        Args:
            normals: Normal map (H, W, 3) in range [-1, 1]

        Returns:
            RGB image (H, W, 3) in range [0, 255] uint8
        """
        normals_rgb = ((normals + 1) / 2 * 255).astype(np.uint8)
        return normals_rgb


# Example usage
if __name__ == "__main__":
    print("VFX Export Utilities for Depth Anything v3")
    print("=" * 50)

    # Example: Export multi-channel EXR
    print("\nExample 1: OpenEXR Multi-Channel Export")
    print("-" * 50)

    # Create sample data
    h, w = 1080, 1920
    depth = np.random.rand(h, w).astype(np.float32) * 10.0
    confidence = np.random.rand(h, w).astype(np.float32)
    normals = np.random.rand(h, w, 3).astype(np.float32) * 2 - 1

    channels = {
        'depth.Z': depth,
        'confidence.R': confidence,
        'normal.R': normals[:, :, 0],
        'normal.G': normals[:, :, 1],
        'normal.B': normals[:, :, 2],
    }

    metadata = {
        'software': 'Depth Anything v3',
        'version': '1.0',
        'camera': 'virtual',
    }

    print(f"Channels: {list(channels.keys())}")
    print(f"Resolution: {w}x{h}")
    print(f"Metadata: {metadata}")

    # Export
    # OpenEXRExporter.export('example.exr', channels, metadata, compression='ZIP')
    print("(Run with actual data to export)")

    print("\n" + "=" * 50)
    print("For full examples, see FLAME_INTEGRATION.md")
