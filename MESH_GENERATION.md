# 3D Mesh Generation from Depth Maps
## Complete Guide - Depth Anything v3

<div align="center">

![Mesh Generation](https://img.shields.io/badge/Mesh-Poisson%20%2B%20BPA-blue)
![Open3D](https://img.shields.io/badge/Open3D-Required-green)
![Formats](https://img.shields.io/badge/Export-OBJ%20%7C%20PLY%20%7C%20GLB-orange)

**Generate production-quality 3D meshes from depth maps using Poisson Surface Reconstruction**

</div>

---

## 🎯 Overview

Ce guide explique comment générer des meshes 3D de haute qualité à partir des depth maps de Depth Anything v3, en utilisant l'algorithme de **Poisson Surface Reconstruction**.

### Pourquoi générer des meshes ?

**Point clouds** (nuages de points) :
- ✅ Rapide à générer
- ✅ Facile à manipuler
- ✗ Pas de surface continue
- ✗ Difficile à texturer
- ✗ Pas de rendu photo-réaliste

**Meshes** (maillages 3D) :
- ✅ Surface continue et fermée
- ✅ Rendu photo-réaliste
- ✅ Texturable
- ✅ Compatible tous logiciels 3D
- ✅ Utilisable pour CG, jeux, VFX

---

## 🚀 Quick Start

### Installation

```bash
# Open3D (ESSENTIEL)
pip install open3d

# Trimesh (pour export GLB/FBX)
pip install trimesh

# Vérification
python -c "import open3d as o3d; print('Open3D OK ✓')"
```

### Exemple Ultra-Rapide

```python
from depth_anything_3.api import DepthAnything3
from mesh_generator import MeshPipeline
import torch

# 1. Load model
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
model = model.to(device="cuda")

# 2. Get depth
prediction = model.inference(["image.jpg"])
depth = prediction.depth[0]
intrinsics = prediction.intrinsics[0]

# 3. Generate mesh (ONE LINE!)
mesh = MeshPipeline.depth_to_mesh(
    depth=depth,
    intrinsics=intrinsics,
    method='poisson',
    depth_level=9
)

# 4. Export
from mesh_generator import MeshGenerator
MeshGenerator.export_mesh(mesh, "output.glb")
```

**C'est tout ! Mesh 3D prêt en 4 lignes de code** ✓

---

## 📐 Algorithm: Poisson Surface Reconstruction

### Principe

L'algorithme de Poisson résout une équation de Poisson 3D pour créer une surface lisse qui passe par les points du nuage.

**Pipeline complet** :

```
Depth Map
    ↓
[1] Point Cloud Generation
    ↓ (X, Y, Z coordinates)
[2] Outlier Removal
    ↓ (remove noise)
[3] Normal Estimation
    ↓ (vectors perpendicular to surface)
[4] Poisson Reconstruction
    ↓ (solve 3D Poisson equation)
[5] Density Filtering
    ↓ (remove low-confidence areas)
[6] Post-Processing
    ↓ (simplify, smooth)
Final Mesh
```

### Pourquoi Poisson ?

**Avantages** :
- ✅ Surfaces lisses et continues
- ✅ Watertight (fermé, étanche)
- ✅ Résistant au bruit
- ✅ Standard industrie (Pixar, ILM, etc.)
- ✅ Contrôle fin du niveau de détail

**Alternatives** :
- **Ball Pivoting Algorithm (BPA)** : Préserve détails, non-watertight
- **Alpha Shapes** : Rapide, bonne pour scans
- **Marching Cubes** : Pour volumes

**Recommandation** : Poisson pour 95% des cas

---

## 🛠️ Step-by-Step Guide

### Step 1: Depth to Point Cloud

**Objectif** : Convertir depth map 2D en points 3D

```python
from mesh_generator import MeshGenerator
import numpy as np

# Depth map (H, W) - distance en mètres
depth = prediction.depth[0]

# Camera intrinsics (3, 3)
intrinsics = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
])

# Optional: RGB image for colors
import cv2
rgb_image = cv2.imread("image.jpg")
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

# Convert
pcd = MeshGenerator.depth_to_point_cloud(
    depth=depth,
    intrinsics=intrinsics,
    rgb_image=rgb_image,
    depth_scale=1.0,     # Scale depth values
    max_depth=100.0      # Filter far points
)

print(f"Point cloud: {len(pcd.points)} points")
```

**Paramètres** :
- `depth_scale` : Multiplier pour ajuster l'échelle (1.0 = mètres)
- `max_depth` : Ignorer points au-delà de cette distance (élimine background)

---

### Step 2: Outlier Removal

**Objectif** : Éliminer les points aberrants (bruit)

Les depth maps peuvent contenir du bruit ou des artefacts. Ces outliers créent des spikes ou des trous dans le mesh final.

```python
pcd_clean = MeshGenerator.remove_outliers(
    pcd,
    nb_neighbors=20,    # Number of neighbors to analyze
    std_ratio=2.0,      # Standard deviation threshold
    method='statistical'  # or 'radius'
)

print(f"Removed {len(pcd.points) - len(pcd_clean.points)} outliers")
```

**Méthodes** :

**Statistical Outlier Removal** (recommandé) :
- Calcule distance moyenne aux voisins pour chaque point
- Supprime points trop éloignés (> std_ratio × std_dev)
- `nb_neighbors` : plus élevé = plus strict (20-30 bon)
- `std_ratio` : plus bas = plus agressif (2.0 bon départ)

**Radius Outlier Removal** :
- Supprime points avec < nb_neighbors dans radius donné
- Bon pour scans uniformes

**Recommandation** : Statistical avec nb_neighbors=20, std_ratio=2.0

---

### Step 3: Normal Estimation

**Objectif** : Calculer vecteurs perpendiculaires à la surface

Les **normals** sont des vecteurs qui pointent perpendiculairement à la surface. Essentiels pour Poisson reconstruction.

```python
pcd_with_normals = MeshGenerator.estimate_normals(
    pcd_clean,
    search_param_knn=30,          # Number of neighbors
    search_param_radius=None,     # Or use radius search
    orient_normals=True           # Orient consistently
)

print(f"Normals estimated for {len(pcd_with_normals.normals)} points")
```

**Paramètres** :

- `search_param_knn` :
  - Nombre de voisins pour estimation
  - Plus élevé = normals plus lisses
  - Recommandé : 30-50

- `search_param_radius` :
  - Alternative : rayon de recherche
  - Utile si densité variable
  - None = utilise KNN

- `orient_normals` :
  - Oriente normals de manière cohérente
  - **IMPORTANT** : active toujours pour Poisson
  - Assume caméra à l'origine

**Visualisation des normals** :
```python
import open3d as o3d
o3d.visualization.draw_geometries(
    [pcd_with_normals],
    point_show_normal=True  # Show normal vectors
)
```

---

### Step 4: Poisson Reconstruction

**Objectif** : Créer surface mesh à partir du point cloud

C'est l'étape magique ! L'algorithme résout l'équation de Poisson pour créer une surface lisse.

```python
mesh, densities = MeshGenerator.poisson_reconstruction(
    pcd_with_normals,
    depth=9,           # Octree depth (detail level)
    width=0,           # 0 = auto
    scale=1.1,         # Bounding box scale
    linear_fit=False   # Use quadratic fit
)

print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
```

**Paramètre CLÉ : `depth`** (niveau de détail)

| Depth | Vertices | Triangles | Quality | Speed | File Size | Use Case |
|-------|----------|-----------|---------|-------|-----------|----------|
| 6 | ~4K | ~8K | Low | Fast | Small | Preview |
| 7 | ~16K | ~32K | Medium-Low | Fast | Small | Draft |
| 8 | ~65K | ~130K | Medium | Medium | Medium | Standard |
| **9** | **~260K** | **~520K** | **Good** | **Medium** | **Medium** | **Recommended** |
| **10** | **~1M** | **~2M** | **High** | **Slow** | **Large** | **High-quality** |
| 11 | ~4M | ~8M | Very High | Very Slow | Very Large | Extreme detail |
| 12 | ~16M+ | ~32M+ | Maximum | Extremely Slow | Huge | Overkill |

**Recommandations** :
- **Preview/Draft** : depth=7-8
- **Production standard** : depth=9 ✓
- **High-quality VFX** : depth=10
- **Extreme detail** : depth=11 (rarement nécessaire)

**Autres paramètres** :

- `width` :
  - Largeur octree (0 = auto)
  - Laisser à 0 sauf cas spéciaux

- `scale` :
  - Facteur d'échelle bounding box
  - 1.1 = 10% marge (bon par défaut)
  - Augmenter si mesh coupé

- `linear_fit` :
  - False = quadratic (meilleure qualité)
  - True = linear (plus rapide)

**Output** :
- `mesh` : Triangle mesh final
- `densities` : Densité par vertex (pour filtrage)

---

### Step 5: Density Filtering

**Objectif** : Éliminer géométrie aberrante dans zones low-density

Poisson peut créer de la géométrie parasite dans les zones à faible densité de points.

```python
mesh_filtered = MeshGenerator.filter_mesh_by_density(
    mesh,
    densities,
    quantile=0.01  # Remove bottom 1%
)

print(f"Removed {len(mesh.vertices) - len(mesh_filtered.vertices)} low-density vertices")
```

**Paramètre : `quantile`**
- 0.01 = supprime bottom 1% (recommandé)
- 0.02 = bottom 2% (plus agressif)
- 0.05 = bottom 5% (très agressif)

**Avant/Après** :
```
Before: Spurious geometry, floating triangles
After: Clean surface, no artifacts
```

---

### Step 6: Post-Processing

#### A. Simplification (reduce polygon count)

**Objectif** : Réduire nombre de polygones sans perte visible de qualité

```python
mesh_simplified = MeshGenerator.simplify_mesh(
    mesh_filtered,
    target_triangles=100000,  # Target count
    method='quadric'          # or 'average'
)

reduction = (1 - len(mesh_simplified.triangles) / len(mesh_filtered.triangles)) * 100
print(f"Reduced by {reduction:.1f}%")
```

**Méthodes** :

**Quadric Decimation** (recommandé) :
- Meilleure qualité
- Préserve features importants
- Plus lent
- **Use for** : Final assets, VFX

**Vertex Clustering** :
- Plus rapide
- Qualité moyenne
- **Use for** : Preview, draft

**Target triangle counts** :

| Use Case | Triangles | Quality | Performance |
|----------|-----------|---------|-------------|
| Real-time preview | 10K-50K | Low | Excellent |
| Game asset (LOD 0) | 50K-100K | Medium | Good |
| VFX background | 100K-200K | Good | Medium |
| **VFX hero asset** | **200K-500K** | **High** | **Medium** |
| Film close-up | 500K-1M | Very High | Slow |
| Extreme detail | 1M+ | Maximum | Very Slow |

**Recommandation** : 100K-200K pour la plupart des cas VFX

#### B. Smoothing

**Objectif** : Lisser la surface pour éliminer micro-détails

```python
mesh_smooth = MeshGenerator.smooth_mesh(
    mesh_simplified,
    iterations=5,      # Number of smoothing passes
    method='laplacian'  # or 'taubin'
)
```

**Méthodes** :

**Laplacian Smoothing** :
- Simple et efficace
- Peut réduire légèrement le volume
- Iterations : 1-10

**Taubin Smoothing** (recommandé) :
- Préserve mieux le volume
- Deux passes : inflate + deflate
- Meilleure qualité
- Iterations : 5-20

**Recommandation** : Taubin avec 5-10 iterations

---

## 🎨 Export Formats

### Supported Formats

```python
from mesh_generator import MeshGenerator

# OBJ - Universal
MeshGenerator.export_mesh(mesh, "output.obj")
# Pros: Universal, simple
# Cons: No colors in standard OBJ

# PLY - With colors
MeshGenerator.export_mesh(mesh, "output.ply")
# Pros: Vertex colors, normals, compact
# Cons: Less software support than OBJ

# GLB - Blender, Flame, Unity, Unreal
MeshGenerator.export_mesh(mesh, "output.glb")
# Pros: Colors, materials, animations, compact
# Cons: Requires trimesh library

# STL - 3D Printing
MeshGenerator.export_mesh(mesh, "output.stl")
# Pros: 3D printer compatible
# Cons: No colors

# FBX - Maya, 3DS Max
MeshGenerator.export_mesh(mesh, "output.fbx")
# Pros: Industry standard
# Cons: Requires trimesh, complex format
```

### Format Recommendations

**For Autodesk Flame** :
- ✅ **GLB** (primary)
- ✅ OBJ (fallback)

**For Blender** :
- ✅ GLB
- ✅ PLY (with colors)

**For Maya/3DS Max** :
- ✅ FBX
- ✅ OBJ

**For Nuke** :
- ✅ OBJ
- ✅ FBX

**For 3D Printing** :
- ✅ STL

---

## 🔥 Complete Examples

### Example 1: Standard Workflow

```python
from depth_anything_3.api import DepthAnything3
from mesh_generator import MeshPipeline
import torch
import cv2

# Load model
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
model = model.to(device="cuda")

# Get depth
prediction = model.inference(["image.jpg"])
depth = prediction.depth[0]
intrinsics = prediction.intrinsics[0]

# Load RGB for colors
rgb = cv2.imread("image.jpg")
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

# ONE-LINE MESH GENERATION
mesh = MeshPipeline.depth_to_mesh(
    depth=depth,
    intrinsics=intrinsics,
    rgb_image=rgb,
    method='poisson',
    depth_level=9,
    simplify=True,
    target_triangles=150000,
    smooth=True
)

# Export
from mesh_generator import MeshGenerator
MeshGenerator.export_mesh(mesh, "final_mesh.glb")
```

### Example 2: High-Quality (Manual Control)

```python
from mesh_generator import MeshGenerator

# Step-by-step with full control

# 1. Point cloud
pcd = MeshGenerator.depth_to_point_cloud(
    depth, intrinsics, rgb,
    max_depth=50.0  # Filter background
)

# 2. Remove outliers (aggressive)
pcd = MeshGenerator.remove_outliers(
    pcd,
    nb_neighbors=30,
    std_ratio=1.5  # More strict
)

# 3. Estimate normals (high quality)
pcd = MeshGenerator.estimate_normals(
    pcd,
    search_param_knn=50  # More neighbors
)

# 4. Poisson (high detail)
mesh, densities = MeshGenerator.poisson_reconstruction(
    pcd,
    depth=10  # High detail
)

# 5. Filter
mesh = MeshGenerator.filter_mesh_by_density(mesh, densities, quantile=0.02)

# 6. Simplify
mesh = MeshGenerator.simplify_mesh(mesh, target_triangles=200000, method='quadric')

# 7. Smooth
mesh = MeshGenerator.smooth_mesh(mesh, iterations=10, method='taubin')

# Export
MeshGenerator.export_mesh(mesh, "high_quality.glb")
```

### Example 3: Multi-View Fusion

```python
import open3d as o3d

# Process multiple images
images = ["view1.jpg", "view2.jpg", "view3.jpg"]
prediction = model.inference(images)

# Merge point clouds
merged_pcd = o3d.geometry.PointCloud()

for i in range(len(images)):
    depth = prediction.depth[i]
    intrinsics = prediction.intrinsics[i]
    rgb = cv2.imread(images[i])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    pcd = MeshGenerator.depth_to_point_cloud(depth, intrinsics, rgb)

    # Apply camera transform
    if prediction.extrinsics is not None:
        ext = prediction.extrinsics[i]
        # Transform to world coordinates
        # ...

    merged_pcd += pcd

# Downsample to remove duplicates
merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.02)

# Continue with normal workflow
merged_pcd = MeshGenerator.remove_outliers(merged_pcd)
merged_pcd = MeshGenerator.estimate_normals(merged_pcd)
mesh, densities = MeshGenerator.poisson_reconstruction(merged_pcd, depth=9)
# ...
```

---

## 🎬 Autodesk Flame Integration

### Workflow: Depth Map → 3D Mesh → Flame

**Step 1: Generate Mesh**
```bash
python example_mesh_generation.py
# Outputs: scene_geometry.glb
```

**Step 2: Import in Flame**
```
Action → Scene → Import → glTF 2.0 (.glb)
Select: scene_geometry.glb
```

**Step 3: Use in Scene**
- Geometry appears in 3D viewport
- Can be used for:
  - Reference geometry
  - CG element placement
  - Collision detection
  - Shadow casting
  - Camera projection mapping

**Step 4: Combine with Camera Tracking**
```
1. Import mesh.glb (geometry)
2. Import camera.fbx (tracking from Depth Anything)
3. Perfect alignment!
4. Place CG elements on real geometry
```

### Use Cases in Flame

**1. CG Integration**
```
Real plate + Tracked camera + Mesh geometry
→ Place CG spaceship on correct surface
→ Perfect occlusion
→ Realistic shadows
```

**2. Depth-Based Grading**
```
Mesh provides accurate 3D reference
→ Color grade by distance zones
→ More accurate than depth maps
```

**3. Match-Move Reference**
```
Mesh as visual guide for manual adjustments
→ See 3D structure clearly
→ Verify camera solve accuracy
```

**4. Lighting Reference**
```
Use mesh for:
→ Virtual light placement
→ Shadow direction
→ Reflection mapping
```

---

## 🎯 Best Practices

### 1. Parameter Tuning

**Start with defaults** :
```python
mesh = MeshPipeline.depth_to_mesh(
    depth, intrinsics, rgb,
    method='poisson',
    depth_level=9,        # Good balance
    simplify=True,
    target_triangles=150000,
    smooth=True
)
```

**Iterate** :
- Too noisy? → Increase outlier removal (std_ratio=1.5)
- Not enough detail? → Increase depth (10)
- Too many polygons? → Decrease target_triangles
- Surface too rough? → More smoothing iterations

### 2. Quality vs Performance

| Priority | Depth | Triangles | Outlier | Normal KNN | Smooth |
|----------|-------|-----------|---------|------------|--------|
| **Speed** | 7-8 | 50K | Light | 20 | 1-3 |
| **Balanced** | 9 | 100-150K | Medium | 30 | 5 |
| **Quality** | 10 | 200-300K | Aggressive | 50 | 10 |

### 3. Memory Management

**Large meshes** (depth=11+) peuvent utiliser beaucoup de RAM :
- **8GB RAM** : depth ≤ 9
- **16GB RAM** : depth ≤ 10
- **32GB+ RAM** : depth 11-12 OK

**Optimizations** :
```python
# Process in smaller chunks
# Or simplify immediately after reconstruction
mesh, densities = MeshGenerator.poisson_reconstruction(pcd, depth=10)
mesh = MeshGenerator.filter_mesh_by_density(mesh, densities)
mesh = MeshGenerator.simplify_mesh(mesh, 100000)  # Reduce immediately
```

### 4. Color Preservation

Pour préserver les couleurs :
```python
# Always provide RGB image
rgb = cv2.imread("image.jpg")
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

pcd = MeshGenerator.depth_to_point_cloud(depth, intrinsics, rgb)

# Colors transfer through pipeline:
# Point cloud → Mesh → Export
```

### 5. Coordinate Systems

**Depth Anything v3** uses OpenCV coordinates :
- X: Right
- Y: Down
- Z: Forward (depth)

**Flame/Blender** may use different systems :
- May need to transform mesh after import
- Usually handled automatically by import

---

## 🐛 Troubleshooting

### Problème : Mesh has holes

**Causes** :
- Outliers not removed
- Poisson depth too low
- Point cloud too sparse

**Solutions** :
```python
# 1. More aggressive outlier removal
pcd = MeshGenerator.remove_outliers(pcd, nb_neighbors=30, std_ratio=1.5)

# 2. Increase Poisson depth
mesh, densities = MeshGenerator.poisson_reconstruction(pcd, depth=10)

# 3. Less aggressive density filtering
mesh = MeshGenerator.filter_mesh_by_density(mesh, densities, quantile=0.005)
```

### Problème : Mesh too smooth (lost details)

**Solutions** :
```python
# 1. Increase Poisson depth
depth=10  # or 11

# 2. Less smoothing
iterations=1  # or skip smoothing

# 3. Try Ball Pivoting instead
mesh = MeshGenerator.ball_pivoting_reconstruction(pcd)
```

### Problème : Spurious geometry / floating triangles

**Solutions** :
```python
# More aggressive density filtering
mesh = MeshGenerator.filter_mesh_by_density(mesh, densities, quantile=0.02)

# Or manual cleanup in Blender
```

### Problème : Out of memory

**Solutions** :
```python
# 1. Lower Poisson depth
depth=8  # instead of 10

# 2. Downsample point cloud first
pcd = pcd.voxel_down_sample(voxel_size=0.01)

# 3. Simplify immediately
mesh = MeshGenerator.simplify_mesh(mesh, 50000)
```

### Problème : Mesh import fails in Flame

**Solutions** :
```
# 1. Use GLB format (best)
MeshGenerator.export_mesh(mesh, "output.glb")

# 2. Try OBJ as fallback
MeshGenerator.export_mesh(mesh, "output.obj")

# 3. Check file size (< 500MB recommended)
# Simplify if too large
```

---

## 📚 Additional Resources

### Documentation
- [Open3D Docs](http://www.open3d.org/docs/)
- [Poisson Reconstruction Paper](http://hhoppe.com/poissonrecon.pdf)
- [Trimesh Docs](https://trimsh.org/)

### Tutorials
- [Open3D Tutorial - Surface Reconstruction](http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html)
- [Point Cloud to Mesh Guide](http://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html)

### Tools
- **Blender** : Free, visualize and edit meshes
- **MeshLab** : Free, mesh processing
- **CloudCompare** : Free, point cloud viewer

---

## 🎓 Summary

### Quick Reference

**ONE-LINE mesh generation** :
```python
mesh = MeshPipeline.depth_to_mesh(depth, intrinsics, rgb, method='poisson', depth_level=9)
```

**Recommended parameters** :
- Poisson depth: **9** (standard) or **10** (high quality)
- Target triangles: **100K-200K**
- Outlier removal: nb_neighbors=**20**, std_ratio=**2.0**
- Normal estimation: search_param_knn=**30**
- Smoothing: Taubin, **5-10** iterations

**Best formats** :
- Flame: **GLB**
- Blender: **GLB** or **PLY**
- Maya: **FBX** or **OBJ**

**Production workflow** :
```
Depth Anything v3 → Depth Map → Point Cloud →
→ Outlier Removal → Normal Estimation →
→ Poisson Reconstruction → Filtering →
→ Simplification → Smoothing →
→ Export GLB → Import in Flame
```

---

<div align="center">

**🎬 Ready to create amazing 3D meshes! 🎬**

[⬆ Back to Top](#3d-mesh-generation-from-depth-maps)

</div>
