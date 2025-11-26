#!/usr/bin/env python3
"""
Installation and Functionality Test Script
==========================================

This script tests all components of the Depth Anything v3 VFX Ultimate Edition
to ensure everything is properly installed and working.

Run this BEFORE using the applications to verify your setup.

Author: Claude
"""

import sys
import os
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def test_header(name):
    """Print test section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{name}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")

def test_ok(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.RESET} {message}")

def test_fail(message):
    """Print failure message"""
    print(f"{Colors.RED}✗{Colors.RESET} {message}")

def test_warn(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {message}")

def test_info(message):
    """Print info message"""
    print(f"  {message}")


def test_python_version():
    """Test Python version"""
    test_header("Python Version")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 8:
        test_ok(f"Python {version_str} (compatible)")
    else:
        test_fail(f"Python {version_str} (requires 3.8+)")
        return False

    return True


def test_core_dependencies():
    """Test core Python dependencies"""
    test_header("Core Dependencies")

    all_ok = True

    # NumPy
    try:
        import numpy as np
        test_ok(f"NumPy {np.__version__}")
    except ImportError:
        test_fail("NumPy not installed: pip install numpy")
        all_ok = False

    # PyTorch
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            test_ok(f"PyTorch {torch.__version__} (CUDA: {torch.version.cuda})")
            test_info(f"GPU: {torch.cuda.get_device_name(0)}")
            test_info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            test_warn(f"PyTorch {torch.__version__} (CPU only - slower performance)")
    except ImportError:
        test_fail("PyTorch not installed: pip install torch torchvision")
        all_ok = False

    # OpenCV
    try:
        import cv2
        test_ok(f"OpenCV {cv2.__version__}")
    except ImportError:
        test_fail("OpenCV not installed: pip install opencv-python")
        all_ok = False

    # PIL
    try:
        from PIL import Image
        test_ok(f"Pillow (PIL) installed")
    except ImportError:
        test_fail("Pillow not installed: pip install Pillow")
        all_ok = False

    return all_ok


def test_gui_dependencies():
    """Test GUI dependencies"""
    test_header("GUI Dependencies")

    all_ok = True

    # PyQt6
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QPixmap
        test_ok("PyQt6 installed")
    except ImportError:
        test_fail("PyQt6 not installed: pip install PyQt6")
        test_info("GUI applications will not work without PyQt6")
        all_ok = False

    return all_ok


def test_depth_anything():
    """Test Depth Anything v3 installation"""
    test_header("Depth Anything v3")

    # Check if source exists
    da3_path = Path(__file__).parent / 'Depth-Anything-3-main' / 'src'
    if not da3_path.exists():
        test_fail("Depth-Anything-3-main/src not found")
        test_info("Extract Depth-Anything-3-main.zip first")
        return False

    test_ok(f"Source directory found: {da3_path}")

    # Try to import
    sys.path.insert(0, str(da3_path))

    try:
        from depth_anything_3.api import DepthAnything3
        test_ok("Depth Anything v3 API imported successfully")

        # Check requirements
        required_modules = ['einops', 'omegaconf', 'safetensors', 'huggingface_hub']
        for module in required_modules:
            try:
                __import__(module)
                test_ok(f"  {module} installed")
            except ImportError:
                test_fail(f"  {module} not installed")
                test_info(f"  Install with: pip install {module}")
                return False

        return True

    except Exception as e:
        test_fail(f"Cannot import Depth Anything v3: {e}")
        test_info("cd Depth-Anything-3-main && pip install -e .")
        return False


def test_vfx_dependencies():
    """Test VFX-specific dependencies"""
    test_header("VFX Dependencies (Optional)")

    # Open3D
    try:
        import open3d as o3d
        test_ok(f"Open3D {o3d.__version__} (for 3D meshes)")
    except ImportError:
        test_warn("Open3D not installed (optional): pip install open3d")
        test_info("3D mesh generation will not work without Open3D")

    # OpenEXR
    try:
        import OpenEXR
        test_ok("OpenEXR installed (for multi-channel export)")
    except ImportError:
        test_warn("OpenEXR not installed (optional)")
        test_info("Multi-channel EXR export will not work")
        test_info("Install: pip install openexr (may need system libs)")

    # Trimesh
    try:
        import trimesh
        test_ok("Trimesh installed (for GLB/FBX export)")
    except ImportError:
        test_warn("Trimesh not installed (optional): pip install trimesh")
        test_info("GLB/FBX export will not work without trimesh")

    # imageio
    try:
        import imageio
        test_ok("imageio installed (for video/DPX)")
    except ImportError:
        test_warn("imageio not installed: pip install imageio imageio-ffmpeg")


def test_application_files():
    """Test that application files exist"""
    test_header("Application Files")

    files = {
        'depth_anything_gui.py': 'Standard GUI Application',
        'depth_anything_vfx_ultimate.py': 'VFX Ultimate Edition',
        'vfx_export_utils.py': 'VFX Export Utilities',
        'mesh_generator.py': 'Mesh Generation Module',
        'example_vfx_export.py': 'VFX Export Examples',
        'example_mesh_generation.py': 'Mesh Generation Examples',
    }

    all_ok = True
    for file, description in files.items():
        path = Path(__file__).parent / file
        if path.exists():
            test_ok(f"{file}: {description}")
        else:
            test_fail(f"{file} not found")
            all_ok = False

    return all_ok


def test_documentation():
    """Test that documentation exists"""
    test_header("Documentation")

    docs = {
        'START_HERE.md': 'Navigation Guide',
        'README_GUI.md': 'Standard GUI Documentation',
        'README_VFX_ULTIMATE.md': 'VFX Features Overview',
        'FLAME_INTEGRATION.md': 'Autodesk Flame Integration',
        'MESH_GENERATION.md': '3D Mesh Generation Guide',
        'QUICKSTART.md': 'Quick Start Guide',
    }

    all_ok = True
    for file, description in docs.items():
        path = Path(__file__).parent / file
        if path.exists():
            test_ok(f"{file}: {description}")
        else:
            test_warn(f"{file} not found")

    return all_ok


def test_import_applications():
    """Test importing application modules"""
    test_header("Application Import Test")

    all_ok = True

    # Test VFX utilities
    try:
        from vfx_export_utils import OpenEXRExporter, DPXExporter
        test_ok("vfx_export_utils imports successfully")
    except Exception as e:
        test_fail(f"vfx_export_utils import failed: {e}")
        all_ok = False

    # Test mesh generator
    try:
        from mesh_generator import MeshGenerator, MeshPipeline
        test_ok("mesh_generator imports successfully")
    except Exception as e:
        test_fail(f"mesh_generator import failed: {e}")
        all_ok = False

    # Test VFX ultimate wrapper
    try:
        from depth_anything_vfx_ultimate import vfx_utils
        test_ok("depth_anything_vfx_ultimate imports successfully")
    except Exception as e:
        test_fail(f"depth_anything_vfx_ultimate import failed: {e}")
        all_ok = False

    return all_ok


def print_summary(results):
    """Print test summary"""
    test_header("Test Summary")

    passed = sum(results.values())
    total = len(results)

    print()
    for test_name, result in results.items():
        if result:
            test_ok(test_name)
        else:
            test_fail(test_name)

    print()
    print(f"{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.RESET}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All tests passed! System is ready to use.{Colors.RESET}")
        print()
        print("Next steps:")
        print("  1. Read START_HERE.md for navigation")
        print("  2. Run: python depth_anything_gui.py")
        print("  3. Or run: python depth_anything_vfx_ultimate.py")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Some tests failed. Please fix issues above.{Colors.RESET}")
        print()
        print("Common fixes:")
        print("  1. Install missing dependencies: pip install -r requirements_gui.txt")
        print("  2. Install Depth Anything v3: cd Depth-Anything-3-main && pip install -e .")
        print("  3. For VFX features: pip install -r requirements_vfx_ultimate.txt")


def main():
    """Main test function"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*60)
    print("Depth Anything v3 - Installation Test")
    print("="*60)
    print(f"{Colors.RESET}")

    results = {}

    # Run tests
    results["Python Version"] = test_python_version()
    results["Core Dependencies"] = test_core_dependencies()
    results["GUI Dependencies"] = test_gui_dependencies()
    results["Depth Anything v3"] = test_depth_anything()
    results["VFX Dependencies"] = True  # Optional, always pass
    test_vfx_dependencies()
    results["Application Files"] = test_application_files()
    results["Documentation"] = True  # Optional, always pass
    test_documentation()
    results["Application Imports"] = test_import_applications()

    # Print summary
    print_summary(results)

    print()


if __name__ == "__main__":
    main()
