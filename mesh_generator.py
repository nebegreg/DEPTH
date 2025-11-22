#!/usr/bin/env python3
"""
3D Mesh Generation from Depth Maps
===================================

Generate high-quality 3D meshes from depth maps using:
- Poisson Surface Reconstruction
- Ball Pivoting Algorithm
- Alpha Shapes
- Outlier removal
- Normal estimation

Export formats:
- OBJ (universal)
- PLY (with colors)
- GLB (Blender, Flame)
- FBX (Maya, 3DS Max)
- STL (3D printing)

Author: Claude
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path


class MeshGenerator:
    """Generate 3D meshes from depth maps and point clouds"""

    @staticmethod
    def depth_to_point_cloud(
        depth: np.ndarray,
        intrinsics: np.ndarray,
        rgb_image: Optional[np.ndarray] = None,
        depth_scale: float = 1.0,
        max_depth: float = 100.0
    ) -> 'o3d.geometry.PointCloud':
        """
        Convert depth map to 3D point cloud

        Args:
            depth: Depth map (H, W) in meters
            intrinsics: Camera intrinsics (3, 3)
            rgb_image: Optional RGB image (H, W, 3) for colors
            depth_scale: Scale factor for depth values
            max_depth: Maximum depth threshold (filter far points)

        Returns:
            Open3D PointCloud object
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        h, w = depth.shape

        # Get camera parameters
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Create pixel coordinates
        y, x = np.mgrid[0:h, 0:w]

        # Filter by max depth
        valid_mask = (depth > 0) & (depth < max_depth)

        # Convert to 3D points
        z = depth[valid_mask] * depth_scale
        x_3d = (x[valid_mask] - cx) * z / fx
        y_3d = (y[valid_mask] - cy) * z / fy

        # Stack coordinates
        points = np.stack([x_3d, y_3d, z], axis=-1)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Add colors if RGB image provided
        if rgb_image is not None:
            colors = rgb_image[valid_mask] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    @staticmethod
    def remove_outliers(
        pcd: 'o3d.geometry.PointCloud',
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
        method: str = 'statistical'
    ) -> 'o3d.geometry.PointCloud':
        """
        Remove outliers from point cloud

        Args:
            pcd: Input point cloud
            nb_neighbors: Number of neighbors to analyze for statistical outlier removal
            std_ratio: Standard deviation ratio threshold
            method: 'statistical' or 'radius'

        Returns:
            Cleaned point cloud
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        print(f"Point cloud before outlier removal: {len(pcd.points)} points")

        if method == 'statistical':
            # Statistical outlier removal
            # Remove points that are further away from their neighbors in average
            pcd_clean, ind = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
        elif method == 'radius':
            # Radius outlier removal
            # Remove points that have few neighbors in a given sphere
            pcd_clean, ind = pcd.remove_radius_outlier(
                nb_points=nb_neighbors,
                radius=0.05
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"Point cloud after outlier removal: {len(pcd_clean.points)} points")
        print(f"Removed {len(pcd.points) - len(pcd_clean.points)} outliers")

        return pcd_clean

    @staticmethod
    def estimate_normals(
        pcd: 'o3d.geometry.PointCloud',
        search_param_knn: int = 30,
        search_param_radius: Optional[float] = None,
        orient_normals: bool = True
    ) -> 'o3d.geometry.PointCloud':
        """
        Estimate normals for point cloud

        Normals are vectors perpendicular to the surface, required for
        Poisson reconstruction.

        Args:
            pcd: Input point cloud
            search_param_knn: Number of neighbors for normal estimation (KNN)
            search_param_radius: Radius for neighbor search (alternative to KNN)
            orient_normals: Orient normals consistently

        Returns:
            Point cloud with estimated normals
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        print("Estimating normals...")

        # Estimate normals
        if search_param_radius is not None:
            # Hybrid search (radius + KNN)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=search_param_radius,
                    max_nn=search_param_knn
                )
            )
        else:
            # KNN search only
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=search_param_knn)
            )

        # Orient normals consistently (camera viewpoint)
        if orient_normals:
            # Orient towards camera (assuming camera at origin looking down +Z)
            pcd.orient_normals_consistent_tangent_plane(k=15)

        print(f"Normals estimated for {len(pcd.normals)} points")

        return pcd

    @staticmethod
    def poisson_reconstruction(
        pcd: 'o3d.geometry.PointCloud',
        depth: int = 9,
        width: int = 0,
        scale: float = 1.1,
        linear_fit: bool = False
    ) -> Tuple['o3d.geometry.TriangleMesh', np.ndarray]:
        """
        Poisson Surface Reconstruction

        Creates a smooth, watertight mesh from oriented point cloud.

        Args:
            pcd: Point cloud WITH normals
            depth: Octree depth (controls mesh detail)
                   - 6-8: Low detail, fast
                   - 9-10: Medium detail (recommended)
                   - 11-12: High detail, slow, large files
            width: Octree width (0 = auto)
            scale: Scale factor for bounding box
            linear_fit: Use linear interpolation

        Returns:
            Tuple of (mesh, densities)
            - mesh: Triangle mesh
            - densities: Vertex density values (for filtering)
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        print(f"Running Poisson reconstruction (depth={depth})...")

        # Check normals exist
        if not pcd.has_normals():
            raise ValueError("Point cloud must have normals. Run estimate_normals() first.")

        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=width,
            scale=scale,
            linear_fit=linear_fit
        )

        print(f"Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        return mesh, densities

    @staticmethod
    def filter_mesh_by_density(
        mesh: 'o3d.geometry.TriangleMesh',
        densities: np.ndarray,
        quantile: float = 0.01
    ) -> 'o3d.geometry.TriangleMesh':
        """
        Remove low-density vertices (outliers) from mesh

        Poisson reconstruction can create spurious geometry in low-density areas.
        This removes vertices below a density threshold.

        Args:
            mesh: Input mesh
            densities: Vertex densities from Poisson reconstruction
            quantile: Remove vertices below this density quantile (0.01 = bottom 1%)

        Returns:
            Filtered mesh
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        print(f"Filtering mesh by density (quantile={quantile})...")

        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, quantile)

        # Remove low-density vertices
        vertices_to_remove = densities < density_threshold
        mesh_filtered = mesh.select_by_index(
            np.where(~vertices_to_remove)[0]
        )

        print(f"Removed {vertices_to_remove.sum()} low-density vertices")
        print(f"Filtered mesh: {len(mesh_filtered.vertices)} vertices, {len(mesh_filtered.triangles)} triangles")

        return mesh_filtered

    @staticmethod
    def ball_pivoting_reconstruction(
        pcd: 'o3d.geometry.PointCloud',
        radii: Optional[List[float]] = None
    ) -> 'o3d.geometry.TriangleMesh':
        """
        Ball Pivoting Algorithm (BPA) reconstruction

        Alternative to Poisson. Better for non-watertight surfaces.

        Args:
            pcd: Point cloud WITH normals
            radii: List of ball radii for multi-scale reconstruction
                   If None, auto-computed from point cloud

        Returns:
            Triangle mesh
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        print("Running Ball Pivoting Algorithm...")

        # Check normals
        if not pcd.has_normals():
            raise ValueError("Point cloud must have normals. Run estimate_normals() first.")

        # Auto-compute radii if not provided
        if radii is None:
            # Estimate average distance between points
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist, avg_dist * 2, avg_dist * 4]
            print(f"Auto-computed radii: {radii}")

        # BPA reconstruction
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii)
        )

        print(f"BPA mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        return mesh

    @staticmethod
    def simplify_mesh(
        mesh: 'o3d.geometry.TriangleMesh',
        target_triangles: int = 100000,
        method: str = 'quadric'
    ) -> 'o3d.geometry.TriangleMesh':
        """
        Simplify mesh (reduce polygon count)

        Args:
            mesh: Input mesh
            target_triangles: Target number of triangles
            method: 'quadric' (better quality) or 'average' (faster)

        Returns:
            Simplified mesh
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        original_triangles = len(mesh.triangles)
        print(f"Simplifying mesh from {original_triangles} to ~{target_triangles} triangles...")

        if method == 'quadric':
            # Quadric decimation (better quality)
            mesh_simplified = mesh.simplify_quadric_decimation(target_triangles)
        elif method == 'average':
            # Vertex clustering (faster)
            voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 100
            mesh_simplified = mesh.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Average
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"Simplified mesh: {len(mesh_simplified.triangles)} triangles ({len(mesh_simplified.triangles)/original_triangles*100:.1f}% of original)")

        return mesh_simplified

    @staticmethod
    def smooth_mesh(
        mesh: 'o3d.geometry.TriangleMesh',
        iterations: int = 1,
        method: str = 'laplacian'
    ) -> 'o3d.geometry.TriangleMesh':
        """
        Smooth mesh surface

        Args:
            mesh: Input mesh
            iterations: Number of smoothing iterations
            method: 'laplacian' or 'taubin'

        Returns:
            Smoothed mesh
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        print(f"Smoothing mesh ({method}, {iterations} iterations)...")

        if method == 'laplacian':
            mesh_smooth = mesh.filter_smooth_laplacian(
                number_of_iterations=iterations
            )
        elif method == 'taubin':
            mesh_smooth = mesh.filter_smooth_taubin(
                number_of_iterations=iterations
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return mesh_smooth

    @staticmethod
    def export_mesh(
        mesh: 'o3d.geometry.TriangleMesh',
        output_path: str,
        compute_normals: bool = True
    ):
        """
        Export mesh to file

        Supported formats:
        - .obj (universal, no colors)
        - .ply (with colors)
        - .stl (3D printing)
        - .glb (Blender, Flame - requires trimesh)
        - .fbx (Maya, 3DS Max - requires trimesh)

        Args:
            mesh: Mesh to export
            output_path: Output file path
            compute_normals: Compute vertex normals before export
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")

        if compute_normals:
            mesh.compute_vertex_normals()

        ext = Path(output_path).suffix.lower()

        if ext in ['.obj', '.ply', '.stl']:
            # Open3D native export
            o3d.io.write_triangle_mesh(output_path, mesh)
            print(f"Mesh exported to {output_path}")

        elif ext in ['.glb', '.gltf', '.fbx']:
            # Use trimesh for GLB/FBX export
            try:
                import trimesh
            except ImportError:
                raise ImportError("trimesh required for GLB/FBX export: pip install trimesh")

            # Convert Open3D mesh to trimesh
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            if mesh.has_vertex_colors():
                colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
                tm_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=colors)
            else:
                tm_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

            # Export
            tm_mesh.export(output_path)
            print(f"Mesh exported to {output_path} (via trimesh)")

        else:
            raise ValueError(f"Unsupported format: {ext}")


class MeshPipeline:
    """Complete pipeline: Depth → Point Cloud → Mesh"""

    @staticmethod
    def depth_to_mesh(
        depth: np.ndarray,
        intrinsics: np.ndarray,
        rgb_image: Optional[np.ndarray] = None,
        method: str = 'poisson',
        depth_level: int = 9,
        simplify: bool = True,
        target_triangles: int = 100000,
        smooth: bool = True,
        remove_outliers: bool = True
    ) -> 'o3d.geometry.TriangleMesh':
        """
        Complete pipeline: depth map → 3D mesh

        Args:
            depth: Depth map (H, W)
            intrinsics: Camera intrinsics (3, 3)
            rgb_image: Optional RGB image for colors
            method: 'poisson' or 'ball_pivoting'
            depth_level: Poisson depth (9-10 recommended)
            simplify: Simplify mesh
            target_triangles: Target polygon count
            smooth: Smooth mesh surface
            remove_outliers: Remove outlier points

        Returns:
            3D mesh
        """
        print("="*60)
        print("3D Mesh Generation Pipeline")
        print("="*60)

        # Step 1: Depth to point cloud
        print("\n[1/6] Converting depth to point cloud...")
        pcd = MeshGenerator.depth_to_point_cloud(depth, intrinsics, rgb_image)

        # Step 2: Remove outliers
        if remove_outliers:
            print("\n[2/6] Removing outliers...")
            pcd = MeshGenerator.remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0)
        else:
            print("\n[2/6] Skipping outlier removal")

        # Step 3: Estimate normals
        print("\n[3/6] Estimating normals...")
        pcd = MeshGenerator.estimate_normals(pcd, search_param_knn=30)

        # Step 4: Mesh reconstruction
        print(f"\n[4/6] Mesh reconstruction ({method})...")
        if method == 'poisson':
            mesh, densities = MeshGenerator.poisson_reconstruction(pcd, depth=depth_level)
            # Filter low-density areas
            mesh = MeshGenerator.filter_mesh_by_density(mesh, densities, quantile=0.01)
        elif method == 'ball_pivoting':
            mesh = MeshGenerator.ball_pivoting_reconstruction(pcd)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Step 5: Simplify
        if simplify and len(mesh.triangles) > target_triangles:
            print(f"\n[5/6] Simplifying mesh to ~{target_triangles} triangles...")
            mesh = MeshGenerator.simplify_mesh(mesh, target_triangles)
        else:
            print("\n[5/6] Skipping simplification")

        # Step 6: Smooth
        if smooth:
            print("\n[6/6] Smoothing mesh...")
            mesh = MeshGenerator.smooth_mesh(mesh, iterations=5, method='laplacian')
        else:
            print("\n[6/6] Skipping smoothing")

        print("\n" + "="*60)
        print("Mesh generation complete!")
        print(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        print("="*60)

        return mesh


# Example usage
if __name__ == "__main__":
    print("3D Mesh Generation from Depth Maps")
    print("="*60)

    # Example: Create mesh from synthetic depth
    print("\nExample: Generating mesh from synthetic depth data")

    # Create sample depth map (sphere)
    h, w = 480, 640
    y, x = np.ogrid[-h/2:h/2, -w/2:w/2]
    radius = 200
    depth = np.sqrt(x**2 + y**2)
    depth = np.clip(radius - depth, 0, radius) / 50.0  # Normalize

    # Create sample intrinsics
    intrinsics = np.array([
        [500, 0, w/2],
        [0, 500, h/2],
        [0, 0, 1]
    ])

    # Optional: Create sample RGB image
    rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    # Generate mesh
    try:
        mesh = MeshPipeline.depth_to_mesh(
            depth=depth,
            intrinsics=intrinsics,
            rgb_image=rgb,
            method='poisson',
            depth_level=9,
            simplify=True,
            target_triangles=50000,
            smooth=True
        )

        # Export
        output_path = "example_mesh.ply"
        MeshGenerator.export_mesh(mesh, output_path)

        print(f"\n✓ Mesh exported to: {output_path}")
        print("\nVisualize with:")
        print("  python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_triangle_mesh('example_mesh.ply')])\"")

    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        print("Install with: pip install open3d")

    print("\n" + "="*60)
    print("For production use, see:")
    print("- Integration in depth_anything_vfx_ultimate.py")
    print("- Documentation in README_VFX_ULTIMATE.md")
    print("="*60)
