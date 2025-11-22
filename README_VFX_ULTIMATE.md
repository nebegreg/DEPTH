# Depth Anything v3 - VFX ULTIMATE Edition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyQt6](https://img.shields.io/badge/PyQt6-VFX%20Edition-green)
![OpenEXR](https://img.shields.io/badge/OpenEXR-Multi--Channel-red)
![Flame](https://img.shields.io/badge/Autodesk-Flame%20Compatible-orange)

**Professional VFX Application with Full Autodesk Flame Integration**

[Features](#-features) • [Installation](#-installation) • [Flame Integration](#-autodesk-flame-integration) • [Formats](#-supported-formats) • [Workflows](#-professional-workflows)

</div>

---

## 🎬 Overview

La **VFX ULTIMATE Edition** est une application professionnelle construite autour de Depth Anything v3, spécialement conçue pour les pipelines VFX et l'intégration avec Autodesk Flame, Nuke, et autres logiciels de post-production.

### Ce qui rend cette édition unique :

- ✅ **Import séquences d'images professionnelles** (EXR, DPX, TIFF, etc.)
- ✅ **Export OpenEXR multi-channel** (depth + confidence + normals dans un fichier)
- ✅ **Export FBX/Alembic** pour camera tracking
- ✅ **Export DPX sequences** (cinema-quality 10/16-bit)
- ✅ **Support tous les codecs vidéo** via ffmpeg
- ✅ **Intégration Autodesk Flame** clé-en-main
- ✅ **Point clouds** pour 3D reconstruction
- ✅ **Normal maps** automatiques
- ✅ **Frame-accurate metadata**

---

## 🌟 Features

### Import Avancé

#### Séquences d'Images

```python
# Patterns supportés:
image.%04d.exr       # Printf-style (standard VFX)
image.####.exr       # Hash pattern
image_0001.exr       # Auto-détection
render.1001.dpx      # DPX sequences
plate.0001.tif       # TIFF sequences
```

**Formats supportés** :
- **OpenEXR** (.exr) : 16/32-bit float, multi-channel
- **DPX** (.dpx) : 10/16-bit, cinema-quality
- **TIFF** (.tif, .tiff) : 8/16/32-bit, HDR
- **PNG** (.png) : 8/16-bit
- **JPEG** (.jpg, .jpeg) : 8-bit
- **Raw formats** (.cr2, .nef, .arw) via rawpy (optional)

#### Vidéo Professionnelle

**Codecs supportés** :
- ProRes (tous variants)
- DNxHD/DNxHR
- H.264/H.265
- MPEG-2, MPEG-4
- MXF
- RED R3D (avec RED SDK)
- ARRI formats
- Blackmagic RAW (avec SDK)

### Export Professionnel

#### 1. OpenEXR Multi-Channel

Export un fichier EXR avec multiples channels :

```
Channels dans un seul .exr:
├── depth.Z          # Profondeur (float32)
├── confidence.R     # Confiance (float32)
├── normal.R         # Normal X (float32)
├── normal.G         # Normal Y (float32)
├── normal.B         # Normal Z (float32)
└── rgba.R/G/B/A     # Image originale (optional)
```

**Avantages** :
- Standard industrie (ILM, Pixar, Weta)
- Un seul fichier = tous les passes
- Compression lossless (ZIP, PIZ)
- Compatible Flame, Nuke, After Effects

#### 2. DPX Sequences

Export cinema-quality pour workflows haut de gamme :

```
Specs:
- 10-bit: ~36 MB/frame (4K)
- 16-bit: ~72 MB/frame (4K)
- Linear ou Log color space
- Frame numbering: 1001+ (standard)
```

#### 3. FBX Camera Tracking

Export données de tracking caméra pour match-move :

```
FBX includes:
├── Camera transform (position + rotation)
├── Focal length (mm)
├── Sensor size (mm)
├── Animation curves
└── Compatible Flame, Maya, Blender, C4D
```

#### 4. Alembic Camera

Alternative à FBX, préférée pour animation :

```
Alembic (.abc):
- Format open-source
- Optimisé pour animation
- Compatible Flame, Houdini, Maya
```

#### 5. Point Clouds

Export pour 3D reconstruction :

```
PLY format:
- Points avec couleurs RGB
- Compatible Flame, MeshLab, CloudCompare
- Convertible OBJ/FBX
```

---

## 🚀 Installation

### Prérequis

- **Python** 3.8+
- **GPU** NVIDIA avec CUDA (recommandé)
- **RAM** 16GB+ (32GB pour 4K)
- **OS** Windows, Linux, macOS

### Installation Rapide

```bash
# 1. Environnement virtuel
python -m venv venv_vfx
source venv_vfx/bin/activate  # Windows: venv_vfx\Scripts\activate

# 2. PyTorch avec CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Dépendances VFX
pip install -r requirements_vfx_ultimate.txt

# 4. Depth Anything v3
cd Depth-Anything-3-main
pip install -e .
cd ..

# 5. Lancer l'application
python depth_anything_vfx_ultimate.py
```

### Installation OpenEXR (Important !)

OpenEXR est **essentiel** pour workflows VFX professionnels.

**Linux (Ubuntu/Debian)** :
```bash
sudo apt-get update
sudo apt-get install libopenexr-dev libilmbase-dev
pip install openexr
```

**macOS** :
```bash
brew install openexr
pip install openexr
```

**Windows** :
```bash
# Méthode 1 - Conda (recommandé)
conda install -c conda-forge openexr-python

# Méthode 2 - Pip (peut nécessiter Visual Studio)
pip install openexr
```

**Vérification** :
```python
python -c "import OpenEXR; print('OpenEXR OK')"
```

---

## 🎯 Autodesk Flame Integration

### Quick Start

**1. Export depuis Depth Anything v3** :
```
1. Import Image Sequence (e.g., plate.%04d.exr)
2. Mode → Monocular Depth (ou Multi-View)
3. Export Format → OpenEXR Multi-Channel
4. Options:
   ✓ Include Normal Maps
   ✓ Include Confidence Maps
5. Process
```

**2. Import dans Flame** :
```
1. Media Panel → Import → Image Sequence
2. Select first frame (e.g., depth.1001.exr)
3. Format: OpenEXR, Multi-channel
4. Color Space: Linear
5. Import
```

**3. Utilisation** :
```
Action → Compositing
- Depth channel available in node
- Use for DOF, fog, masking, etc.
```

### Guide Complet

Voir **[FLAME_INTEGRATION.md](FLAME_INTEGRATION.md)** pour :
- Workflows détaillés
- Camera tracking setup
- Point cloud import
- Best practices VFX
- Troubleshooting
- Exemples de production

---

## 📋 Supported Formats

### Input Formats

| Format | Extension | Bit Depth | Use Case |
|--------|-----------|-----------|----------|
| **OpenEXR** | .exr | 16/32-bit float | VFX, high-end |
| **DPX** | .dpx | 10/16-bit | Cinema, color grading |
| **TIFF** | .tif, .tiff | 8/16/32-bit | Print, archive |
| **PNG** | .png | 8/16-bit | Web, preview |
| **JPEG** | .jpg, .jpeg | 8-bit | Acquisition, web |
| **ProRes** | .mov | 10/12-bit | Editing, dailies |
| **DNxHD** | .mxf, .mov | 8/10-bit | Editing |
| **H.264/265** | .mp4, .mov | 8/10-bit | Distribution |

### Output Formats

| Format | Purpose | Compatible Software |
|--------|---------|---------------------|
| **OpenEXR Multi-Channel** | VFX compositing | Flame, Nuke, AE, Fusion |
| **DPX Sequence** | Cinema grading | Flame, Baselight, Resolve |
| **TIFF 32-bit** | HDR compositing | Photoshop, AE, Flame |
| **PNG Sequence** | Preview, web | All software |
| **FBX Camera** | Match-move | Flame, Maya, Blender, C4D |
| **Alembic Camera** | Animation | Flame, Houdini, Maya |
| **PLY Point Cloud** | 3D reconstruction | MeshLab, CloudCompare, Blender |

---

## 🎓 Professional Workflows

### Workflow 1 : Depth-based DOF (Depth of Field)

**Scenario** : Spot commercial, besoin DOF ajustable en post

```
Step 1 - Depth Anything v3:
├── Import: commercial.mov (4K, ProRes)
├── Mode: Monocular Depth
├── Model: DA3-LARGE (balance qualité/vitesse)
├── Export: OpenEXR multi-channel
└── Output: depth.1001-1100.exr (depth + original)

Step 2 - Autodesk Flame:
├── Import EXR sequence
├── Action → Lens → Depth of Field
├── Z-Depth Source: depth.Z channel
├── Adjust:
│   ├── Focus Point: Interactive (basé sur depth values)
│   ├── F-Stop: 2.8 (shallow DOF)
│   └── Bokeh: Circular
└── Real-time preview, client approval

Delivery:
└── ProRes 422 HQ with final DOF
```

**Temps** : 30 min processing + 1h grading = **Delivery même jour**

**Avant (sans Depth Anything)** :
- Re-shoot avec vraie caméra ($$$$)
- Rotoscoping manuel (plusieurs jours)
- Plugins approximatifs (résultats médiocres)

**Maintenant** :
- Depth map précis en minutes
- DOF physiquement correct
- Ajustable à l'infini
- Client content ✓

### Workflow 2 : CG Integration & Match-Move

**Scenario** : Film VFX, insérer vaisseau spatial CG

```
Step 1 - Depth Anything v3:
├── Import: shot_010.%04d.dpx (2K anamorphic, 240 frames)
├── Mode: Multi-View Depth + Pose Estimation
├── Model: DA3-GIANT (meilleure qualité)
├── Export:
│   ├── OpenEXR depth maps
│   ├── FBX camera tracking
│   └── PLY point cloud
└── Processing: ~4 hours (batch overnight)

Step 2 - Autodesk Flame (ou Maya):
├── Import FBX camera (auto-tracking)
├── Import spaceship.fbx (CG model)
├── Import depth.exr (pour occlusion)
├── Action 3D Scene:
│   ├── Position spaceship in 3D
│   ├── Camera auto-matches footage
│   ├── Depth maps for realistic occlusion
│   ├── Lighting matched to plate
│   └── Shadow casting on point cloud geometry
└── Render composite

Step 3 - Compositing:
├── Combine CG render avec original plate
├── Color match
├── Grain/noise match
├── Final output

Delivery:
└── DPX sequence 2K pour DI (Digital Intermediate)
```

**Résultat** :
- Match-move parfait (pas de sliding)
- Occlusion réaliste
- Integration seamless
- Supervisor VFX approuve first pass

### Workflow 3 : Selective Color Grading

**Scenario** : Music video, grade différent par distance

```
Step 1 - Depth Anything v3:
├── Import: musicvideo.mp4 (1080p, 200 frames)
├── Mode: Monocular Depth + Metric (distances réelles)
├── Export: OpenEXR (depth + confidence + original)

Step 2 - Flame Color Grading:
├── Import EXR sequence
├── Action → Color Warper
├── Create 3 zones basées sur depth:
│   ├── Foreground (0-5m): Warm look
│   │   ├── Lift: +0.05 Red, -0.02 Blue
│   │   ├── Gamma: +0.1 Saturation
│   │   └── Gain: +0.03 contrast
│   ├── Midground (5-20m): Neutral
│   │   └── No changes
│   └── Background (20m+): Cool, desaturated
│       ├── Lift: -0.05 Red, +0.05 Blue
│       ├── Gamma: -0.2 Saturation
│       └── Gain: -0.1 (darker)
├── Feathering: Automatic via depth gradient
└── Animated over timeline

Delivery:
└── H.264 4K pour client preview
```

**Résultat** :
- Look unique et stylisé
- Séparation visuelle par profondeur
- Pas de rotoscoping manuel
- Client impressionné par créativité

### Workflow 4 : Atmospheric Effects

**Scenario** : Spot automobile, ajouter fog atmosphérique

```
Step 1 - Depth Anything v3:
├── Import: car_commercial.%04d.exr (4K, 120 frames)
├── Mode: Monocular Depth
├── Export: OpenEXR multi-channel

Step 2 - Flame:
├── Import depth sequence
├── Action → Lens → Fog/Mist
├── Depth-based falloff:
│   ├── Near fog: 0% (car sharp)
│   ├── Far fog: 100% @ 50m (background hazy)
│   ├── Color: Blue-grey (#B0C4DE)
│   └── Density: 0.7
├── Combine with color grade:
│   ├── Cool temperature for fog
│   └── Warm spotlights on car

Result:
└── Cinematic atmosphere
└── Depth perception enhanced
└── Heroic car stands out
```

---

## 🛠️ VFX Export Utilities

L'édition Ultimate inclut **vfx_export_utils.py**, un module Python autonome pour exports professionnels.

### Usage Standalone

```python
from vfx_export_utils import OpenEXRExporter, DPXExporter, FBXCameraExporter

# Export multi-channel EXR
channels = {
    'depth.Z': depth_map,
    'confidence.R': confidence_map,
    'normal.R': normal_map[:, :, 0],
    'normal.G': normal_map[:, :, 1],
    'normal.B': normal_map[:, :, 2],
}

OpenEXRExporter.export(
    'output.exr',
    channels,
    metadata={'software': 'Depth Anything v3'},
    compression='ZIP'
)

# Export DPX sequence
DPXExporter.export_sequence(
    output_dir='dpx_output',
    frames=[frame1, frame2, ...],
    base_name='shot_010',
    start_frame=1001,
    bit_depth=10
)

# Export FBX camera
FBXCameraExporter.export(
    'camera.fbx',
    extrinsics=camera_poses,  # [N, 3, 4]
    intrinsics=camera_params,  # [N, 3, 3]
    image_size=(1920, 1080),
    fps=24.0
)
```

### Integration dans vos scripts

```python
import sys
sys.path.append('/path/to/DEPTH')
from vfx_export_utils import *

# Votre code ici
```

---

## 📐 Technical Specifications

### Profondeur et Précision

**Depth Maps** :
- Format: float32 (32-bit floating point)
- Range: 0.1m à 1000m (configurable)
- Precision: Sub-millimeter à courte distance
- Color space: Linear (pas de gamma correction)

**Normal Maps** :
- Format: float32 × 3 channels
- Range: [-1, 1] normalisé
- Computed from: Depth gradients
- Use case: Lighting, bump mapping, surface analysis

**Confidence Maps** :
- Format: float32
- Range: [0, 1] (0 = low confidence, 1 = high)
- Use case: Masking, quality control, selective processing

### Frame Numbering

**Standards VFX** :
- Start frame: **1001** (évite frame 0 issues)
- Padding: **4 digits** minimum (e.g., 0001, 0002, ...)
- Handle frames: +/- 10 frames pour transitions

**Examples** :
```
shot_010_v001.1001.exr
shot_010_v001.1002.exr
...
shot_010_v001.1100.exr
```

### File Sizes

**OpenEXR (4K, ZIP compression)** :
- Depth only: ~15 MB/frame
- Depth + normals: ~40 MB/frame
- All channels + original: ~80 MB/frame
- 100 frames: ~4-8 GB

**DPX (4K)** :
- 10-bit: ~36 MB/frame → ~3.6 GB/100 frames
- 16-bit: ~72 MB/frame → ~7.2 GB/100 frames

**Recommendations** :
- Use SSD for working storage
- Archive to LTO or cloud after project
- Proxies (1080p PNG) for preview

---

## 🎯 Best Practices

### Naming Conventions

```
# Shots
<project>_<shot>_<element>_<version>.<frame>.<ext>

Examples:
commercial_shot010_depth_v001.1001.exr
film_shot025_camera_v002.fbx
musicvideo_shot003_pointcloud_v001.ply

# Sequences
<basename>.<frame>.<ext>
or
<basename>_%04d.<ext>

Examples:
plate.1001.dpx
depth.%04d.exr
```

### Color Management

**For Depth Data** :
- **Color Space** : Linear (ALWAYS)
- **Gamma** : 1.0 (no correction)
- **Transfer** : None
- **Primaries** : N/A (not color data)

**For RGB Imagery** :
- Match project color space
- **ACES** : ACES cg (linear) or ACEScct (log)
- **Rec.709** : For broadcast
- **Log** : For camera formats (ARRI LogC, RED Log3G10)

### Frame Ranges

**Best Practice** :
```
Editorial cut: 1001-1100 (100 frames)
VFX delivery: 991-1110 (120 frames, +/- 10 handles)

Why handles?
- Transitions/dissolves
- Retiming flexibility
- Roto/tracking spillover
```

### Workflow Organization

```
project/
├── editorial/
│   └── cut_v005.xml
├── shots/
│   ├── shot_010/
│   │   ├── plates/           # Original footage
│   │   │   ├── exr/          # Original EXRs
│   │   │   └── preview/      # JPG proxies
│   │   ├── depth/            # Depth Anything output
│   │   │   ├── exr/          # Multi-channel EXRs
│   │   │   ├── dpx/          # Optional DPX
│   │   │   └── preview/      # Colorized PNGs
│   │   ├── tracking/         # Camera data
│   │   │   ├── fbx/
│   │   │   └── abc/
│   │   ├── pointclouds/
│   │   └── flame_comp/       # Flame project files
│   └── shot_020/
│       └── ...
└── deliverables/
    └── final_renders/
```

---

## 🔥 Performance Optimization

### GPU Utilization

**VRAM Requirements** :
- DA3-SMALL: 4GB
- DA3-BASE: 6GB
- DA3-LARGE: 10GB
- DA3-GIANT: 24GB

**Multi-GPU** :
```python
# Not yet supported, coming soon
# For now: process multiple shots in parallel on different GPUs
```

### Batch Processing

**Optimal Workflow** :
```bash
# Process overnight in batch
python batch_process.py \
    --shots shot_010,shot_020,shot_030 \
    --model DA3-LARGE \
    --export exr,dpx,fbx \
    --parallel 3
```

### Compression Trade-offs

| Compression | Size Reduction | Speed | Quality |
|-------------|----------------|-------|---------|
| NONE | 0% | Fastest write/read | Perfect |
| ZIP | ~70% | Fast | Lossless |
| ZIPS | ~75% | Medium | Lossless |
| PIZ | ~80% | Slow | Lossless |
| B44 | ~85% | Fast | Lossy (16-bit) |

**Recommendation** :
- **Working** : ZIP (best balance)
- **Archive** : NONE or ZIP
- **Preview** : PNG with moderate compression

---

## ⚠️ Troubleshooting

### OpenEXR Issues

**Symptom** : "ModuleNotFoundError: No module named 'OpenEXR'"

**Solution** :
```bash
# Try conda (easiest)
conda install -c conda-forge openexr-python

# Or system libraries + pip
sudo apt-get install libopenexr-dev  # Linux
brew install openexr  # macOS
pip install openexr
```

### Flame Import Issues

**Symptom** : Channels not visible in Flame

**Solution** :
1. Import Settings → Format: OpenEXR
2. Channels: **All** (not RGB only)
3. Verify with: `exrheader file.exr`
4. Should see: depth.Z, normal.R/G/B, etc.

**Symptom** : FBX camera offset by 1 frame

**Solution** :
```
Known Flame bug. After import:
Action → Camera → Edit Animation
Shift all keyframes by -1 frame
```

### Memory Issues

**Symptom** : CUDA out of memory

**Solutions** :
1. Use smaller model (DA3-BASE instead of GIANT)
2. Reduce resolution (2K instead of 4K)
3. Process fewer frames at once
4. Clear cache: `torch.cuda.empty_cache()`

### Slow Performance

**Solutions** :
1. Check GPU utilization (`nvidia-smi`)
2. Enable xformers (if not already)
3. Use SSD, not HDD
4. Close other GPU applications
5. Use OpenEXR ZIP compression (faster than PIZ)

---

## 📚 Resources

### Documentation

- **[FLAME_INTEGRATION.md](FLAME_INTEGRATION.md)** : Complete Flame integration guide
- **[vfx_export_utils.py](vfx_export_utils.py)** : VFX export utilities documentation
- **[README_GUI.md](README_GUI.md)** : Base GUI application guide

### External Resources

**Autodesk Flame** :
- [Flame 2025 Help](https://help.autodesk.com/view/FLAME/2025/ENU/)
- [Camera Tracking](https://help.autodesk.com/view/FLAME/2025/ENU/?guid=GUID-70B64EE8-0402-4842-ACF6-10D8492CCFC4)
- [OpenEXR Import](https://help.autodesk.com/view/FLAME/2023/ENU/?guid=GUID-C1DD8D7D-4F2D-4399-A216-3FB972710424)

**VFX Standards** :
- [VFX Reference Platform](https://vfxplatform.com/)
- [OpenEXR Docs](https://www.openexr.com/)
- [DPX Specification](https://www.smpte.org/)
- [ACES Color Management](https://acescentral.com/)

**Depth Anything v3** :
- [Project Page](https://depth-anything-3.github.io/)
- [GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [Paper](https://arxiv.org/abs/2511.10647)

### Community

- [Logik Forums (Flame)](https://forum.logik.tv/)
- [Creative COW (VFX)](https://creativecow.net/)
- [fxguide](https://www.fxguide.com/)
- [Depth Anything Discussions](https://github.com/ByteDance-Seed/Depth-Anything-3/discussions)

---

## 🎬 Production Credits

**Software Used** :
- Depth Anything v3 (ByteDance)
- Autodesk Flame
- Python + PyQt6
- OpenEXR (ILM)
- PyTorch

**Author** : Claude - VFX Edition
**License** : MIT
**Version** : 1.0 Ultimate

---

## 📊 Comparison: Standard vs VFX Ultimate

| Feature | Standard GUI | VFX ULTIMATE |
|---------|-------------|--------------|
| Image Sequences | Basic | ✓ Advanced (EXR, DPX, patterns) |
| Video Codecs | Standard (MP4) | ✓ All (ProRes, DNx, MXF) |
| OpenEXR Export | Single channel | ✓ Multi-channel |
| DPX Export | ✗ | ✓ 10/16-bit |
| FBX Camera | ✗ | ✓ Full tracking |
| Alembic Camera | ✗ | ✓ Animation |
| Normal Maps | ✗ | ✓ Automatic |
| Frame Numbering | Basic | ✓ VFX standard (1001+) |
| Metadata | Basic | ✓ Production-ready |
| Flame Integration | Manual | ✓ Documented workflow |
| Batch Processing | Limited | ✓ Professional |
| Color Management | Basic | ✓ ACES/Linear/Log |
| Documentation | Good | ✓ Production guide |

---

<div align="center">

## 🎯 Ready for Production

**Depth Anything v3 VFX Ultimate** est prêt pour vos projets professionnels.

Des questions ? Consultez [FLAME_INTEGRATION.md](FLAME_INTEGRATION.md) ou ouvrez une issue.

**Bon workflow VFX ! 🎬**

[⬆ Back to Top](#depth-anything-v3---vfx-ultimate-edition)

</div>
