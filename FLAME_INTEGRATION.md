# Autodesk Flame Integration Guide
# Depth Anything v3 → Flame VFX Pipeline

## 🎬 Vue d'ensemble

Ce guide explique comment intégrer les depth maps et données 3D de Depth Anything v3 dans Autodesk Flame pour des workflows VFX professionnels.

**Sources consultées** :
- [Autodesk Flame 2025 - Camera Tracking](https://help.autodesk.com/view/FLAME/2025/ENU/?guid=GUID-70B64EE8-0402-4842-ACF6-10D8492CCFC4)
- [Flame - Importing 3D Models](https://help.autodesk.com/view/FLAME/2025/ENU/?guid=GUID-CA0A558A-C81B-4F08-90A9-559CDE389C00)
- [Working with FBX and Alembic Scene Nodes](https://help.autodesk.com/view/FLAME/2025/ENU/?guid=GUID-49474588-6833-4870-9B1A-B9989D4C446B)
- [OpenEXR Format Settings](https://help.autodesk.com/view/FLAME/2023/ENU/?guid=GUID-C1DD8D7D-4F2D-4399-A216-3FB972710424)

---

## 📋 Formats Supportés par Flame

### 1. **Caméra Tracking**
- **FBX** : Format standard pour caméras 3D
- **Alembic (.abc)** : Format d'échange 3D (préféré pour animation)

### 2. **Séquences d'Images**
- **OpenEXR (.exr)** : Standard industrie, multi-channel, 16/32-bit float
- **DPX (.dpx)** : Cinema-quality, 10/16-bit
- **TIFF (.tif)** : HDR jusqu'à 32 bpc
- **PNG (.png)** : Acceptable mais moins professionnel

### 3. **Données 3D**
- **Point Clouds** : PLY, OBJ
- **Z-Depth Maps** : EXR, TIFF 32-bit

---

## 🔄 Workflows Depth Anything v3 → Flame

### Workflow 1 : Depth Maps pour Compositing

#### Étape 1 : Export depuis Depth Anything v3

```python
# Dans l'application GUI
1. Load Images ou Import Sequence
2. Mode → Monocular Depth
3. Export Format → OpenEXR Multi-Channel
4. Options:
   ✓ Include Normal Maps
   ✓ Include Confidence Maps
5. Process
```

**Output** :
- `frame_####.exr` avec channels :
  - `depth.Z` : Profondeur (float 32-bit)
  - `confidence.R` : Confiance (float 32-bit)
  - `normal.R/G/B` : Normal maps (float 32-bit × 3)
  - `rgba.R/G/B/A` : Image originale (optionnel)

#### Étape 2 : Import dans Flame

```
1. Media Panel → Import → Image Sequence
2. Sélectionner premier frame (frame_1001.exr)
3. Format Settings:
   - File Type: OpenEXR
   - Channels: Multi-channel
   - Color Space: Linear
   - Bit Depth: 32-bit float
4. Frame Range: 1001-1100 (auto-détecté)
5. Import
```

#### Étape 3 : Utilisation dans Action

```
1. Créer Action node
2. Media → Importer séquence depth
3. Connecter depth channel au Z-Depth input
4. Utiliser pour:
   - Depth of Field
   - Fog/Atmosphere
   - Color Correction par distance
   - Masking 3D
   - Depth-based keying
```

**Exemple - Depth of Field** :
```
Action → Lens → Depth of Field
- Z-Depth Source: Imported EXR (depth channel)
- Focus Distance: Ajuster selon depth values
- Falloff: Contrôle progressif
```

---

### Workflow 2 : Camera Tracking pour Match-Move

#### Étape 1 : Export Camera Tracking

```python
# Dans l'application
1. Load Video/Sequence
2. Mode → Pose Estimation (ou Multi-View)
3. Export Format → FBX Camera Tracking ou Alembic Camera
4. Process
```

**Output** :
- `camera_tracking.fbx` : Caméra avec animation
- `camera_tracking.abc` : Alternative Alembic
- Includes:
  - Camera position (translation)
  - Camera rotation
  - Focal length
  - Sensor size
  - Lens distortion (si disponible)

#### Étape 2 : Import Camera dans Flame

**Méthode FBX** :
```
1. Action → Scene → Import → FBX Scene
2. Sélectionner camera_tracking.fbx
3. Options:
   - Import Cameras: ✓
   - Import Animation: ✓
   - Frame Rate: Match source (24/25/30 fps)
   - Start Frame: 1001 (ou votre timecode)
4. Load
5. Camera apparaît dans Scene hierarchy
```

**Méthode Alembic** :
```
1. Action → Scene → Import → Alembic Scene
2. Sélectionner camera_tracking.abc
3. Scene Settings:
   - Alembic Time: Frame
   - FPS: Match source
4. Camera est maintenant animée dans la timeline
```

#### Étape 3 : Utilisation Match-Move

```
1. Caméra importée devient Current Camera
2. Ajouter éléments 3D :
   - 3D Text
   - Imported Models
   - Particle Systems
   - Lights
3. Éléments 3D suivent automatiquement la caméra
4. Render avec depth integration
```

**Tips** :
- Vérifier l'échelle : Flame peut avoir scale différent
- Ajuster focal length si nécessaire
- Utiliser Camera Tracker panel pour refinement

---

### Workflow 3 : Point Cloud pour 3D Reconstruction

#### Étape 1 : Export Point Cloud

```python
# Dans l'application
1. Load Images (multi-view pour meilleure qualité)
2. Mode → Multi-View Depth
3. Export Format → Point Cloud (PLY)
4. Process
```

**Output** :
- `pointcloud.ply` : Point cloud avec couleurs
- `pointcloud_dense.ply` : Version dense (optionnel)

#### Étape 2 : Import dans Flame

```
1. Action → Geometry → Import
2. Formats supportés:
   - FBX (avec geometry)
   - OBJ (convertir PLY si nécessaire)
   - Alembic (pour animated geometry)

3. Pour PLY:
   Option A - Direct (si supporté):
     Import → PLY file

   Option B - Conversion:
     a. Ouvrir dans MeshLab/Blender
     b. Export as OBJ ou FBX
     c. Import OBJ/FBX dans Flame
```

#### Étape 3 : Utilisation 3D Scene

```
1. Point cloud visible dans 3D viewport
2. Utiliser pour:
   - Reference geometry
   - Collision detection
   - Lighting reference
   - Camera projection mapping
3. Combiner avec tracked camera pour match-move parfait
```

---

### Workflow 4 : Séquences DPX (Cinema Quality)

#### Étape 1 : Export DPX

```python
# Dans l'application
1. Load Sequence
2. Export Format → DPX Sequence
3. Options:
   - Bit Depth: 10-bit ou 16-bit
   - Color Space: Linear ou Log
   - Start Frame: 1001
4. Process
```

**Output** :
- `depth_1001.dpx` à `depth_1100.dpx`
- Naming convention: `basename.####.dpx`
- 10-bit: ~36MB par frame (4K)
- 16-bit: ~72MB par frame (4K)

#### Étape 2 : Import DPX dans Flame

```
1. Media Panel → Import → DPX Sequence
2. Auto-détection de la séquence
3. Settings:
   - Bit Depth: 10 ou 16
   - Color Space: Linear (pour depth)
   - Scan Format: 4K/2K/HD
4. Import comme clip
```

#### Étape 3 : Color Management

```
Flame gère automatiquement DPX color space:
- Input: Linear
- Working: ACES ou Linear
- Output: Selon deliverable

Pour depth maps:
- Garder en Linear
- Pas de color correction
- Utiliser comme data pass
```

---

## 🛠️ Best Practices

### 1. **Naming Conventions**

**Séquences d'images** :
```
✓ GOOD:
  shot010_depth_v001.1001.exr
  shot010_depth_v001.1002.exr
  ...
  shot010_depth_v001.1100.exr

✗ AVOID:
  depth1.exr (pas de padding)
  depth_01.jpg (format non-professionnel)
  myimage.png (pas de frame number)
```

**Caméras** :
```
✓ GOOD:
  shot010_camera_v001.fbx
  shot010_cam_tracked.abc

✗ AVOID:
  camera.fbx (pas de version)
  cam.abc (trop générique)
```

### 2. **Frame Ranges**

**Convention industrie** :
- **1001** : Start frame standard (évite frame 0 issues)
- **Handle frames** : +10 frames avant/après pour transitions
- **Timecode** : Sync avec editorial

**Example** :
```
Shot duration: 100 frames
Editorial cut: 1001-1100
VFX delivery: 991-1110 (handles +10)
```

### 3. **File Organization**

**Structure recommandée** :
```
project/
├── shots/
│   └── shot010/
│       ├── plates/          # Original footage
│       ├── depth/           # Depth maps
│       │   ├── exr/         # OpenEXR multi-channel
│       │   ├── dpx/         # DPX sequences
│       │   └── preview/     # PNG previews
│       ├── tracking/        # Camera data
│       │   ├── fbx/
│       │   └── abc/
│       └── pointclouds/     # 3D data
└── flame_project/
    └── import/              # Symlinks or copies
```

### 4. **OpenEXR Multi-Channel**

**Channels recommandés** :
```python
channels = {
    'depth.Z': depth_map,           # Main depth (32-bit float)
    'confidence.R': confidence,      # Confidence map
    'normal.R': normals[:,:,0],     # Normal X
    'normal.G': normals[:,:,1],     # Normal Y
    'normal.B': normals[:,:,2],     # Normal Z
    'rgba.R': image[:,:,0],         # Original image (optional)
    'rgba.G': image[:,:,1],
    'rgba.B': image[:,:,2],
    'rgba.A': alpha,
}
```

**Import dans Flame** :
```
- Flame lit automatiquement tous les channels
- Access via Channel menu dans Action
- Combiner plusieurs passes dans un seul EXR
```

### 5. **Depth Range Normalization**

**Pour Flame** :
```python
# Normaliser depth pour Z-Depth usage
depth_normalized = (depth - depth_min) / (depth_max - depth_min)

# Ou garder metric depth si modèle support
depth_metric = depth  # en mètres

# Flame préfère:
- Near plane: 0.1 à 1.0
- Far plane: 100 à 1000
```

### 6. **Color Space**

**Depth maps** :
- Toujours **Linear** (pas de gamma correction)
- 32-bit float pour précision maximale
- Pas de color management sur depth data

**Images RGB** :
- Match color space du projet Flame
- ACES si pipeline ACES
- Rec709 pour broadcast
- Log pour capture camera (RED, ARRI)

---

## 🎯 Cas d'Usage Spécifiques

### A. Depth of Field Réaliste

**Workflow** :
```
1. Export depth maps (EXR 32-bit)
2. Import dans Flame Action
3. Action → Lens → Depth of Field
4. Z-Depth source: depth.Z channel
5. Ajuster:
   - Focus point (basé sur depth values)
   - F-stop (intensité blur)
   - Bokeh shape
6. Real-time preview
```

**Avantages** :
- DoF physiquement correct
- Pas d'artifacts autour des edges
- Ajustable en post sans re-render

### B. Atmospheric Fog

**Workflow** :
```
1. Depth maps EXR importés
2. Action → Lens → Fog
3. Depth-based falloff:
   - Near fog: 0
   - Far fog: 100% (basé sur depth)
4. Color et density ajustables
5. Combine avec color correction
```

### C. Selective Color Grading

**Workflow** :
```
1. Import depth + original image
2. Action → Color → Selective
3. Masking basé sur depth:
   - Foreground: depth < 10m
   - Midground: 10m < depth < 50m
   - Background: depth > 50m
4. Grade chaque zone séparément
5. Feather automatique via depth gradient
```

### D. 3D Object Integration

**Workflow** :
```
1. Import camera tracking (FBX)
2. Import depth maps (EXR)
3. Import 3D models
4. Action Scene:
   - Position models dans 3D space
   - Camera tracking appliqué automatiquement
   - Depth maps pour occlusion
5. Render avec:
   - Shadows
   - Reflections
   - Depth integration pour compositing réaliste
```

---

## ⚠️ Troubleshooting

### Problème 1 : FBX Camera import décalé d'une frame

**Symptôme** : Animation camera décalée (bug Flame connu)

**Solution** :
```
1. Après import FBX
2. Action → Camera → Edit Animation
3. Shift all keyframes by -1 frame
4. Ou exporter avec offset +1 depuis Depth Anything
```

### Problème 2 : Depth maps trop sombres/claires

**Symptôme** : Depth visualization incorrecte

**Solution** :
```
1. Vérifier que depth est en Linear color space
2. Flame → Color → Color Management: OFF pour depth
3. Utiliser depth.Z channel directement (pas RGB)
4. Ajuster range dans Depth of Field settings
```

### Problème 3 : EXR multi-channel non reconnu

**Symptôme** : Channels pas visibles

**Solution** :
```
1. Import settings → Format: OpenEXR
2. Channels: All (not RGB only)
3. Vérifier EXR file:
   > exrheader file.exr
   Channels should list: depth.Z, normal.R, etc.
4. Re-export depuis Depth Anything si nécessaire
```

### Problème 4 : Scale/Units mismatch

**Symptôme** : Caméra ou models trop grands/petits

**Solution** :
```
1. Flame utilise centimètres par défaut
2. Depth Anything export en mètres
3. Action → Scene → Settings:
   - Units: Meters
   - Scale: 1.0
4. Ou multiplier depth par 100 lors export
```

### Problème 5 : Performance lente avec EXR 4K

**Symptôme** : Playback lent, cache plein

**Solution** :
```
1. Utiliser OpenEXR compression: ZIP ou PIZ
2. Flame → Proxy mode: Half res pour playback
3. Cache settings: Augmenter RAM allocation
4. Render proxies pour preview
5. Full res seulement pour final render
```

---

## 📊 Performance & Optimization

### Tailles de Fichiers

**OpenEXR (4K, 32-bit float)** :
- No compression: ~130 MB/frame
- ZIP compression: ~40 MB/frame
- PIZ compression: ~35 MB/frame

**DPX (4K)** :
- 10-bit: ~36 MB/frame
- 16-bit: ~72 MB/frame

**Recommandation** :
- **Working**: EXR avec ZIP (bon compromis qualité/taille)
- **Archive**: EXR non-compressé ou DPX 16-bit
- **Proxy**: PNG 1080p pour previews

### Temps de Processing

**Depth Anything v3 (GPU RTX 3090)** :
- 1080p: ~0.15s/frame (DA3-LARGE)
- 4K: ~0.6s/frame
- Batch 100 frames: ~1 minute (1080p)

**Flame Import** :
- EXR sequence (100 frames): ~10 secondes
- FBX camera: instantané
- Point cloud (1M points): ~5 secondes

### Workflow Optimization

**Pipeline efficace** :
```
1. Depth Anything processing en batch overnight
2. Export tous formats en parallèle
3. Structure de dossiers organisée
4. Import dans Flame le matin
5. Iterative grading/compositing dans la journée
```

---

## 🎓 Exemples Pratiques

### Exemple 1 : Shot Commercial - Depth of Field

**Setup** :
- Footage: 4K, 24fps, 100 frames
- Besoin: DoF réglable en post

**Steps** :
```
1. Depth Anything:
   - Import video commercial.mp4
   - Mode: Monocular Depth
   - Export: OpenEXR (depth only)
   - Output: shot_depth.1001-1100.exr

2. Flame:
   - Import EXR sequence
   - Action → Lens → DoF
   - Depth source: depth.Z
   - Focus: F11 (ajustable)
   - Bokeh: Circular

3. Résultat:
   - DoF cinématique
   - Focus adjustable sans re-render
   - Livraison client rapide
```

### Exemple 2 : Film VFX - CG Integration

**Setup** :
- Footage: 2K anamorphic, 24fps, 240 frames
- Besoin: Ajouter spaceship CG

**Steps** :
```
1. Depth Anything:
   - Import DPX sequence
   - Mode: Multi-View Depth + Pose Estimation
   - Export:
     * EXR depth maps
     * FBX camera tracking
     * Point cloud PLY

2. Flame:
   - Import camera FBX
   - Import depth EXR
   - Import spaceship model (FBX)
   - Action 3D Scene:
     * Position spaceship
     * Camera auto-matched
     * Depth for occlusion
   - Lighting to match plate
   - Render composite

3. Résultat:
   - Match-move parfait
   - Occlusion réaliste
   - Integration seamless
```

### Exemple 3 : Music Video - Stylized Grading

**Setup** :
- Footage: 1080p, 30fps, 200 frames
- Besoin: Color grade par profondeur

**Steps** :
```
1. Depth Anything:
   - Import video
   - Mode: Monocular Depth
   - Export: EXR multi-channel (depth + original)

2. Flame:
   - Import EXR
   - Action → Color → Selective
   - Masks basées sur depth:
     * FG (0-5m): Warm, saturated
     * MG (5-20m): Normal
     * BG (20m+): Cool, desaturated
   - Animated grade sur timeline

3. Résultat:
   - Look unique et stylisé
   - Depth-based artistically
   - Client impressionné
```

---

## 📚 Ressources

### Documentation Flame

- [Autodesk Flame 2025 Help](https://help.autodesk.com/view/FLAME/2025/ENU/)
- [Camera Tracking](https://help.autodesk.com/view/FLAME/2025/ENU/?guid=GUID-70B64EE8-0402-4842-ACF6-10D8492CCFC4)
- [FBX/Alembic Import](https://help.autodesk.com/view/FLAME/2025/ENU/?guid=GUID-49474588-6833-4870-9B1A-B9989D4C446B)
- [OpenEXR Settings](https://help.autodesk.com/view/FLAME/2023/ENU/?guid=GUID-C1DD8D7D-4F2D-4399-A216-3FB972710424)

### Depth Anything v3

- [Project Page](https://depth-anything-3.github.io/)
- [GitHub Repository](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [Paper (arXiv)](https://arxiv.org/abs/2511.10647)

### VFX Standards

- [VFX Reference Platform](https://vfxplatform.com/)
- [OpenEXR Specification](https://www.openexr.com/)
- [DPX Format Guide](https://www.smpte.org/)
- [FBX SDK Documentation](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-0)

### Forums & Community

- [Logik Forums - Flame](https://forum.logik.tv/)
- [Creative COW - Flame](https://creativecow.net/forums/forum/autodesk-flame/)
- [Depth Anything Discussions](https://github.com/ByteDance-Seed/Depth-Anything-3/discussions)

---

## ✅ Checklist de Production

**Avant Processing** :
- [ ] Footage en résolution finale
- [ ] Frame range confirmé avec editorial
- [ ] Color space documenté
- [ ] Handles frames inclus (+10)

**Processing Depth Anything** :
- [ ] Modèle approprié sélectionné
- [ ] Export format = OpenEXR multi-channel
- [ ] Naming convention respectée
- [ ] Métadonnées inclusesfooter: "footer"
- [ ] QC sur sample frames

**Import Flame** :
- [ ] Séquences importées correctement
- [ ] Frame range vérifié
- [ ] Channels accessibles
- [ ] Color space correct

**Compositing/Grading** :
- [ ] Depth maps fonctionnels
- [ ] Camera tracking aligné
- [ ] Occlusion correcte
- [ ] Client approval sur preview

**Delivery** :
- [ ] Format final selon specs
- [ ] Timecode correct
- [ ] Métadonnées complètes
- [ ] Archivage organisé

---

<div align="center">

**🎬 Depth Anything v3 × Autodesk Flame = VFX Excellence 🎬**

[⬆ Retour en haut](#autodesk-flame-integration-guide)

</div>
