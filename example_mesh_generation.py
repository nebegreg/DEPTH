#!/usr/bin/env python3
"""
Example: 3D Mesh Generation from Depth Anything v3
===================================================

This example demonstrates how to generate high-quality 3D meshes from
Depth Anything v3 depth maps using Poisson surface reconstruction.

Workflow:
1. Load image
2. Estimate depth with Depth Anything v3
3. Convert depth to point cloud
4. Remove outliers
5. Estimate normals
6. Poisson reconstruction
7. Post-processing (simplify, smooth)
8. Export mesh

Output formats:
- OBJ (universal)
- PLY (with colors)
- GLB (Blender, Flame, Unity)
- STL (3D printing)

Author: Claude
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add Depth Anything v3 to path
sys.path.insert(0, str(Path(__file__).parent / 'Depth-Anything-3-main' / 'src'))

# Import Depth Anything v3
from depth_anything_3.api import DepthAnything3

# Import mesh generator
from mesh_generator import MeshGenerator, MeshPipeline


def example_1_basic_mesh():
    """Example 1: Basic mesh generation from single image"""
    print("\n" + "="*60)
    print("Example 1: Basic 3D Mesh from Single Image")
    print("="*60)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("\nLoading DA3-LARGE model...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
    model = model.to(device=device)
    model.eval()

    # Example image
    image_path = "Depth-Anything-3-main/assets/examples/SOH/000.png"

    if not os.path.exists(image_path):
        print(f"Warning: Example image not found at {image_path}")
        print("Please provide your own image")
        return

    # Run depth estimation
    print(f"\nProcessing image: {image_path}")
    prediction = model.inference([image_path])

    # Extract data
    depth = prediction.depth[0]  # (H, W)
    intrinsics = prediction.intrinsics[0] if prediction.intrinsics is not None else None

    # Load RGB image for colors
    import cv2
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Create intrinsics if not available
    if intrinsics is None:
        h, w = depth.shape
        intrinsics = np.array([
            [w, 0, w/2],
            [0, w, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        print("Using default intrinsics")

    # Generate mesh using pipeline
    print("\n" + "="*60)
    print("Generating 3D Mesh...")
    print("="*60)

    mesh = MeshPipeline.depth_to_mesh(
        depth=depth,
        intrinsics=intrinsics,
        rgb_image=rgb_image,
        method='poisson',
        depth_level=9,  # 9-10 recommended for good detail
        simplify=True,
        target_triangles=100000,
        smooth=True,
        remove_outliers=True
    )

    # Export mesh
    output_dir = "output_meshes"
    os.makedirs(output_dir, exist_ok=True)

    # Export to multiple formats
    formats = {
        'obj': 'Universal format (no colors)',
        'ply': 'With vertex colors',
        'glb': 'For Blender/Flame/Unity',
    }

    for ext, description in formats.items():
        output_path = os.path.join(output_dir, f"mesh_example1.{ext}")
        try:
            MeshGenerator.export_mesh(mesh, output_path)
            print(f"✓ Exported {ext.upper()}: {output_path} ({description})")
        except Exception as e:
            print(f"✗ Failed to export {ext.upper()}: {e}")

    print("\n" + "="*60)
    print("Mesh generation complete!")
    print("="*60)

    # Visualize
    print("\nTo visualize the mesh:")
    print("  1. Install Blender (free)")
    print("  2. File → Import → glTF 2.0 (.glb)")
    print(f"  3. Select: {output_dir}/mesh_example1.glb")
    print("\nOr use Open3D:")
    print("  python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_triangle_mesh('output_meshes/mesh_example1.ply')])\"")


def example_2_high_quality_mesh():
    """Example 2: High-quality mesh with fine-tuning"""
    print("\n" + "="*60)
    print("Example 2: High-Quality Mesh with Fine-Tuning")
    print("="*60)

    # Setup (same as example 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
    model = model.to(device=device)
    model.eval()

    image_path = "Depth-Anything-3-main/assets/examples/SOH/000.png"
    if not os.path.exists(image_path):
        print("Example image not found")
        return

    # Process
    prediction = model.inference([image_path])
    depth = prediction.depth[0]
    intrinsics = prediction.intrinsics[0]

    import cv2
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    if intrinsics is None:
        h, w = depth.shape
        intrinsics = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)

    print("\nStep-by-step mesh generation with fine control:")

    # Step 1: Point cloud
    print("\n[1/7] Depth → Point cloud")
    pcd = MeshGenerator.depth_to_point_cloud(
        depth=depth,
        intrinsics=intrinsics,
        rgb_image=rgb_image,
        depth_scale=1.0,
        max_depth=50.0  # Filter far points
    )

    # Step 2: Outlier removal (aggressive)
    print("\n[2/7] Remove outliers (aggressive)")
    pcd = MeshGenerator.remove_outliers(
        pcd,
        nb_neighbors=30,  # More neighbors = stricter
        std_ratio=1.5,    # Lower = remove more
        method='statistical'
    )

    # Step 3: Normal estimation (high quality)
    print("\n[3/7] Estimate normals (high quality)")
    pcd = MeshGenerator.estimate_normals(
        pcd,
        search_param_knn=50,  # More neighbors = smoother normals
        orient_normals=True
    )

    # Step 4: Poisson (high detail)
    print("\n[4/7] Poisson reconstruction (high detail)")
    mesh, densities = MeshGenerator.poisson_reconstruction(
        pcd,
        depth=10,  # Higher = more detail (but slower & larger file)
        linear_fit=False
    )

    # Step 5: Filter by density
    print("\n[5/7] Filter low-density areas")
    mesh = MeshGenerator.filter_mesh_by_density(
        mesh,
        densities,
        quantile=0.02  # Remove bottom 2%
    )

    # Step 6: Simplify
    print("\n[6/7] Simplify mesh")
    mesh = MeshGenerator.simplify_mesh(
        mesh,
        target_triangles=200000,  # High polygon count
        method='quadric'  # Better quality
    )

    # Step 7: Smooth
    print("\n[7/7] Smooth surface")
    mesh = MeshGenerator.smooth_mesh(
        mesh,
        iterations=10,  # More iterations = smoother
        method='taubin'  # Taubin preserves volume better than Laplacian
    )

    # Export
    output_path = "output_meshes/mesh_high_quality.glb"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    MeshGenerator.export_mesh(mesh, output_path)

    print(f"\n✓ High-quality mesh exported: {output_path}")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Triangles: {len(mesh.triangles)}")


def example_3_multiview_mesh():
    """Example 3: Mesh from multiple views (better quality)"""
    print("\n" + "="*60)
    print("Example 3: Multi-View Mesh Fusion")
    print("="*60)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
    model = model.to(device=device)
    model.eval()

    # Multiple images
    import glob
    image_dir = "Depth-Anything-3-main/assets/examples/SOH"
    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))[:3]  # First 3

    if len(images) < 2:
        print("Need at least 2 images for multi-view")
        return

    print(f"\nProcessing {len(images)} images for multi-view reconstruction...")

    # Process all images
    prediction = model.inference(images)

    # Merge point clouds from all views
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D required: pip install open3d")
        return

    print("\nMerging point clouds from all views...")
    merged_pcd = o3d.geometry.PointCloud()

    for i in range(len(images)):
        depth = prediction.depth[i]
        intrinsics = prediction.intrinsics[i] if prediction.intrinsics is not None else None

        # Load RGB
        import cv2
        rgb = cv2.imread(images[i])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Create intrinsics if needed
        if intrinsics is None:
            h, w = depth.shape
            intrinsics = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)

        # Create point cloud for this view
        pcd = MeshGenerator.depth_to_point_cloud(depth, intrinsics, rgb)

        # Transform based on camera pose (if available)
        if prediction.extrinsics is not None:
            # Apply camera extrinsics
            ext = prediction.extrinsics[i]
            R = ext[:3, :3]
            t = ext[:3, 3]
            transformation = np.eye(4)
            transformation[:3, :3] = R
            transformation[:3, 3] = t
            pcd.transform(transformation)

        # Merge
        merged_pcd += pcd

    print(f"Merged point cloud: {len(merged_pcd.points)} points")

    # Remove overlapping points
    print("Removing duplicate points...")
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.02)
    print(f"After downsampling: {len(merged_pcd.points)} points")

    # Remove outliers
    merged_pcd = MeshGenerator.remove_outliers(merged_pcd)

    # Estimate normals
    merged_pcd = MeshGenerator.estimate_normals(merged_pcd)

    # Generate mesh
    mesh, densities = MeshGenerator.poisson_reconstruction(merged_pcd, depth=9)
    mesh = MeshGenerator.filter_mesh_by_density(mesh, densities, quantile=0.01)
    mesh = MeshGenerator.simplify_mesh(mesh, target_triangles=150000)
    mesh = MeshGenerator.smooth_mesh(mesh, iterations=5)

    # Export
    output_path = "output_meshes/mesh_multiview.glb"
    MeshGenerator.export_mesh(mesh, output_path)

    print(f"\n✓ Multi-view mesh exported: {output_path}")
    print("  Better quality than single-view!")


def example_4_mesh_for_flame():
    """Example 4: Mesh optimized for Autodesk Flame"""
    print("\n" + "="*60)
    print("Example 4: Mesh for Autodesk Flame")
    print("="*60)

    print("""
    For Autodesk Flame integration:

    1. Generate mesh (any example above)

    2. Export as GLB:
       - Best format for Flame
       - Includes colors and materials
       - Compact file size

    3. Import in Flame:
       - Action → Scene → Import → glTF 2.0
       - Select mesh.glb file
       - Geometry appears in 3D scene

    4. Use cases:
       - Reference geometry for CG placement
       - Collision detection
       - Light/shadow interaction
       - Camera projection mapping

    5. Combine with camera tracking:
       - Import mesh.glb (geometry)
       - Import camera.fbx (tracking)
       - Perfect match-move alignment

    Recommended settings for Flame:
    - Depth level: 9 (good balance)
    - Simplify: Yes (100k-200k triangles)
    - Smooth: Yes (cleaner surface)
    - Format: GLB (best compatibility)
    """)

    # Example workflow
    print("\nExample export for Flame:")
    print("  MeshPipeline.depth_to_mesh(..., depth_level=9, target_triangles=150000)")
    print("  MeshGenerator.export_mesh(mesh, 'scene_geometry.glb')")


def example_5_comparison_methods():
    """Example 5: Compare Poisson vs Ball Pivoting"""
    print("\n" + "="*60)
    print("Example 5: Comparison of Reconstruction Methods")
    print("="*60)

    print("""
    Two main reconstruction algorithms available:

    1. POISSON SURFACE RECONSTRUCTION
       Pros:
       - Smooth, watertight meshes
       - Handles noise well
       - Good for organic shapes
       - Industry standard

       Cons:
       - Can over-smooth details
       - Creates closed surfaces (may fill holes)
       - Slower

       Use for:
       - Characters, organic objects
       - Smooth surfaces
       - VFX assets
       - Autodesk Flame geometry

    2. BALL PIVOTING ALGORITHM (BPA)
       Pros:
       - Preserves sharp features
       - Non-watertight (good for scans)
       - Faster
       - Better for noisy data

       Cons:
       - May have holes
       - More sensitive to parameters
       - Less smooth

       Use for:
       - Scanned data
       - Sharp edges (buildings)
       - When holes are OK

    Recommendation for Depth Anything v3:
    → Use POISSON (default)
    → Depth level 9-10 for most cases
    → Use BPA only if Poisson over-smooths
    """)


def main():
    """Run all examples"""
    print("="*60)
    print("3D Mesh Generation Examples - Depth Anything v3")
    print("="*60)

    print("\nAvailable examples:")
    print("1. Basic mesh generation")
    print("2. High-quality mesh with fine-tuning")
    print("3. Multi-view mesh fusion")
    print("4. Mesh for Autodesk Flame")
    print("5. Comparison of methods")

    print("\n" + "="*60)

    # Check dependencies
    try:
        import open3d
        print("✓ Open3D available")
    except ImportError:
        print("✗ Open3D required: pip install open3d")
        return

    try:
        import trimesh
        print("✓ Trimesh available (for GLB export)")
    except ImportError:
        print("⚠ Trimesh recommended for GLB export: pip install trimesh")

    # Run examples
    try:
        example_1_basic_mesh()
        # example_2_high_quality_mesh()
        # example_3_multiview_mesh()
        example_4_mesh_for_flame()
        example_5_comparison_methods()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check output_meshes/ directory")
    print("2. Open meshes in Blender or other 3D software")
    print("3. Import in Autodesk Flame (GLB format)")
    print("4. Combine with camera tracking for perfect integration")


if __name__ == "__main__":
    main()
