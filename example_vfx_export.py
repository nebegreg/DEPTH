#!/usr/bin/env python3
"""
Example: VFX Professional Export Workflow
==========================================

This script demonstrates how to use Depth Anything v3 with professional
VFX export formats for Autodesk Flame, Nuke, and other software.

Author: Claude
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add Depth Anything v3 to path
sys.path.insert(0, str(Path(__file__).parent / 'Depth-Anything-3-main' / 'src'))
from depth_anything_3.api import DepthAnything3

# Import VFX export utilities
from vfx_export_utils import (
    OpenEXRExporter,
    DPXExporter,
    FBXCameraExporter,
    NormalMapGenerator
)


def example_1_openexr_export():
    """Example 1: Export multi-channel OpenEXR for Flame"""
    print("\n" + "="*60)
    print("Example 1: OpenEXR Multi-Channel Export for Autodesk Flame")
    print("="*60)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading DA3-LARGE model...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
    model = model.to(device=device)
    model.eval()

    # Example image path (replace with your image)
    image_path = "Depth-Anything-3-main/assets/examples/SOH/000.png"

    if not os.path.exists(image_path):
        print(f"Warning: Example image not found at {image_path}")
        print("Please provide your own image path")
        return

    # Run inference
    print("Running inference...")
    prediction = model.inference([image_path])

    # Extract data
    depth = prediction.depth[0]  # (H, W)
    confidence = prediction.conf[0] if prediction.conf is not None else np.ones_like(depth)
    intrinsics = prediction.intrinsics[0] if prediction.intrinsics is not None else np.eye(3)

    # Compute normal maps
    print("Computing normal maps...")
    normals = NormalMapGenerator.compute(depth, intrinsics)

    # Prepare channels for OpenEXR
    channels = {
        'depth.Z': depth.astype(np.float32),
        'confidence.R': confidence.astype(np.float32),
        'normal.R': normals[:, :, 0].astype(np.float32),
        'normal.G': normals[:, :, 1].astype(np.float32),
        'normal.B': normals[:, :, 2].astype(np.float32),
    }

    # Metadata
    metadata = {
        'software': 'Depth Anything v3 VFX Ultimate',
        'model': 'DA3-LARGE',
        'device': device,
        'image': os.path.basename(image_path),
    }

    # Export
    output_path = "output_vfx/example_multichannel.exr"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting to {output_path}...")
    try:
        OpenEXRExporter.export(
            output_path,
            channels,
            metadata=metadata,
            compression='ZIP'
        )
        print(f"✓ Success! Exported to {output_path}")
        print(f"  Channels: {list(channels.keys())}")
        print(f"  Resolution: {depth.shape[1]}x{depth.shape[0]}")

        # How to import in Flame
        print("\nTo import in Autodesk Flame:")
        print("1. Media Panel → Import → Image")
        print(f"2. Select: {output_path}")
        print("3. Format Settings:")
        print("   - File Type: OpenEXR")
        print("   - Channels: Multi-channel (select All)")
        print("   - Color Space: Linear")
        print("4. Use depth.Z channel in Action for DOF, fog, etc.")

    except ImportError as e:
        print(f"✗ OpenEXR not available: {e}")
        print("Install with: pip install openexr")
        print("Or see requirements_vfx_ultimate.txt for instructions")


def example_2_dpx_sequence():
    """Example 2: Export DPX sequence (cinema-quality)"""
    print("\n" + "="*60)
    print("Example 2: DPX Sequence Export (Cinema Quality)")
    print("="*60)

    # For this example, we'll create sample data
    # In production, you'd process actual image sequence

    print("Creating sample depth maps...")
    frames = []
    for i in range(10):  # 10 frames example
        # Simulated depth map
        depth = np.random.rand(1080, 1920).astype(np.float32) * 100
        frames.append(depth)

    # Export DPX sequence
    output_dir = "output_vfx/dpx_sequence"
    print(f"Exporting to {output_dir}...")

    DPXExporter.export_sequence(
        output_dir=output_dir,
        frames=frames,
        base_name="depth",
        start_frame=1001,  # VFX standard starting frame
        bit_depth=10  # 10-bit DPX (cinema standard)
    )

    print(f"✓ Success! Exported {len(frames)} frames")
    print(f"  Format: DPX 10-bit")
    print(f"  Range: 1001-{1001+len(frames)-1}")
    print(f"  Location: {output_dir}/")

    print("\nTo import in Autodesk Flame:")
    print("1. Media Panel → Import → DPX Sequence")
    print(f"2. Select first frame: depth.1001.dpx")
    print("3. Auto-detects sequence range")
    print("4. Use for high-end color grading workflow")


def example_3_fbx_camera_tracking():
    """Example 3: Export FBX camera tracking"""
    print("\n" + "="*60)
    print("Example 3: FBX Camera Tracking Export")
    print("="*60)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading DA3-LARGE model...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
    model = model.to(device=device)
    model.eval()

    # Example: Process multiple images for camera tracking
    # In production, you'd use actual image sequence
    image_dir = "Depth-Anything-3-main/assets/examples/SOH"

    if not os.path.exists(image_dir):
        print(f"Warning: Example directory not found at {image_dir}")
        print("Skipping camera tracking example")
        return

    # Get images
    import glob
    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))[:5]  # First 5 images

    if not images:
        print("No images found in example directory")
        return

    print(f"Processing {len(images)} images for camera tracking...")

    # Run inference with pose estimation
    prediction = model.inference(images)

    # Extract camera data
    if prediction.extrinsics is None or prediction.intrinsics is None:
        print("Warning: No camera tracking data available")
        print("Make sure to use a model that supports pose estimation (DA3-GIANT/LARGE)")
        return

    extrinsics = prediction.extrinsics  # [N, 3, 4]
    intrinsics = prediction.intrinsics  # [N, 3, 3]

    # Image size
    import cv2
    img = cv2.imread(images[0])
    height, width = img.shape[:2]

    # Export FBX camera
    output_path = "output_vfx/camera_tracking.fbx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting camera to {output_path}...")

    FBXCameraExporter.export(
        output_path=output_path,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_size=(width, height),
        fps=24.0,
        camera_name="TrackedCamera"
    )

    print(f"✓ Success! Exported camera tracking")
    print(f"  Frames: {len(extrinsics)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: 24.0")

    print("\nTo import in Autodesk Flame:")
    print("1. Action → Scene → Import → FBX Scene")
    print(f"2. Select: {output_path}")
    print("3. Options:")
    print("   - Import Cameras: ✓")
    print("   - Import Animation: ✓")
    print("4. Camera now matches original footage movement")
    print("5. Use for CG integration, match-move, etc.")


def example_4_complete_workflow():
    """Example 4: Complete VFX workflow (all formats)"""
    print("\n" + "="*60)
    print("Example 4: Complete VFX Workflow - All Export Formats")
    print("="*60)

    print("""
This example shows a complete professional VFX workflow:

1. Import image sequence (e.g., DPX plates from camera)
2. Process with Depth Anything v3
3. Export ALL formats for pipeline:
   - OpenEXR multi-channel (for compositing)
   - DPX sequence (for color grading)
   - FBX camera (for 3D integration)
   - Normal maps (for lighting)
   - Point clouds (for 3D reconstruction)

Production Workflow:
====================

# Step 1: Organize your footage
project/
├── plates/
│   └── shot_010/
│       ├── plate.1001.dpx
│       ├── plate.1002.dpx
│       └── ...

# Step 2: Process with Depth Anything v3
python process_batch.py \\
    --input project/plates/shot_010/plate.%04d.dpx \\
    --output project/depth/shot_010/ \\
    --model DA3-LARGE \\
    --export exr,dpx,fbx,ply

# Step 3: Import in Flame
- Depth maps → Compositing (DOF, fog, masking)
- Camera FBX → 3D Scene (CG integration)
- DPX sequence → Color grading

# Step 4: Deliver
- Final composite → DPX/ProRes for DI
- VFX metadata → Production tracking
""")

    print("\nFor real production use, see:")
    print("- README_VFX_ULTIMATE.md : Complete workflows")
    print("- FLAME_INTEGRATION.md : Flame-specific guide")
    print("- vfx_export_utils.py : Export utility functions")


def main():
    """Main function - run all examples"""
    print("="*60)
    print("Depth Anything v3 - VFX Professional Export Examples")
    print("="*60)

    print("\nThese examples demonstrate professional VFX export workflows")
    print("for Autodesk Flame, Nuke, and other post-production software.")

    # Check dependencies
    print("\nChecking dependencies...")
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError:
        print("✗ PyTorch required: pip install torch")
        return

    try:
        import OpenEXR
        print("✓ OpenEXR available")
        has_exr = True
    except ImportError:
        print("⚠ OpenEXR not available (optional)")
        print("  Install with: pip install openexr")
        print("  See requirements_vfx_ultimate.txt for instructions")
        has_exr = False

    # Run examples
    try:
        if has_exr:
            example_1_openexr_export()

        example_2_dpx_sequence()
        example_3_fbx_camera_tracking()
        example_4_complete_workflow()

        print("\n" + "="*60)
        print("Examples complete!")
        print("="*60)

        print("\nNext steps:")
        print("1. Check output_vfx/ directory for exported files")
        print("2. Import in Autodesk Flame following instructions above")
        print("3. Explore README_VFX_ULTIMATE.md for production workflows")
        print("4. See FLAME_INTEGRATION.md for detailed Flame integration")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
