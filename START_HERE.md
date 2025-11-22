# 🚀 Depth Anything v3 - VFX ULTIMATE Edition
## START HERE - Guide de Démarrage

Bienvenue dans la **VFX ULTIMATE Edition** de Depth Anything v3 !

Cette application transforme Depth Anything v3 en un outil VFX professionnel avec intégration complète pour **Autodesk Flame**, Nuke, et autres logiciels de post-production.

---

## 📚 Quelle Documentation Consulter ?

Vous avez maintenant **DEUX applications** disponibles :

### 1️⃣ **Application Standard** (Recommandée pour débuter)

**Fichiers** :
- `depth_anything_gui.py` - Application PyQt6 standard
- `README_GUI.md` - Documentation complète
- `QUICKSTART.md` - Démarrage rapide
- `requirements_gui.txt` - Dépendances

**Pour qui** : Utilisateurs généraux, tests, démos, projets personnels

**Lancer** :
```bash
# Linux/macOS
./launch_gui.sh

# Windows
launch_gui.bat

# Ou directement
python depth_anything_gui.py
```

**Features** :
- Interface moderne et intuitive
- 6 modes de traitement
- Export GLB, PLY, NPZ, images
- Vidéo et webcam temps réel
- 3D visualization
- GPU/CPU support

---

### 2️⃣ **VFX ULTIMATE Edition** (Pour professionnels VFX)

**Fichiers** :
- `depth_anything_vfx_ultimate.py` - Application VFX avancée
- `README_VFX_ULTIMATE.md` - Guide VFX complet ⭐ **COMMENCER ICI pour VFX**
- `FLAME_INTEGRATION.md` - Guide Autodesk Flame ⭐ **Intégration Flame**
- `vfx_export_utils.py` - Utilitaires export professionnel
- `example_vfx_export.py` - Exemples de code
- `requirements_vfx_ultimate.txt` - Dépendances VFX

**Pour qui** : Professionnels VFX, studios de post-production, intégration Flame/Nuke

**Lancer** :
```bash
python depth_anything_vfx_ultimate.py
```

**Features EXCLUSIVES** :
- ✅ Import séquences d'images (EXR, DPX, TIFF)
- ✅ Export OpenEXR multi-channel
- ✅ Export DPX sequences (10/16-bit)
- ✅ Export FBX/Alembic camera tracking
- ✅ Normal maps automatiques
- ✅ Support ProRes, DNxHD, MXF
- ✅ Frame numbering VFX standard (1001+)
- ✅ **Intégration Autodesk Flame clé-en-main**

---

## 🎯 Guides Recommandés par Cas d'Usage

### Je veux juste essayer Depth Anything v3
→ **Lire** : `QUICKSTART.md`
→ **Lancer** : `./launch_gui.sh` ou `launch_gui.bat`
→ **Application** : Standard GUI

### Je fais de la VFX professionnelle
→ **Lire** : `README_VFX_ULTIMATE.md` ⭐
→ **Installer** : `pip install -r requirements_vfx_ultimate.txt`
→ **Application** : VFX Ultimate

### J'utilise Autodesk Flame
→ **Lire** : `FLAME_INTEGRATION.md` ⭐⭐⭐
→ **Workflows** : DOF, camera tracking, color grading
→ **Formats** : OpenEXR multi-channel, FBX camera

### Je veux intégrer dans mon pipeline Python
→ **Lire** : `vfx_export_utils.py` (documentation inline)
→ **Examples** : `example_vfx_export.py`
→ **Import** : `from vfx_export_utils import OpenEXRExporter, ...`

### Je travaille sur Nuke
→ **Lire** : `README_VFX_ULTIMATE.md` section "Nuke Integration"
→ **Format** : OpenEXR multi-channel (depth.Z, normal.R/G/B)
→ **Workflow** : Import EXR → Shuffle channels → Use in comp

### Je veux exporter pour After Effects
→ **Application** : Standard GUI suffit
→ **Format** : TIFF 32-bit ou PNG sequences
→ **Lire** : `README_GUI.md`

---

## 📦 Installation

### Installation Rapide - Standard Edition

```bash
# 1. Script automatique
./launch_gui.sh  # Linux/macOS
launch_gui.bat   # Windows

# Ou manuel:
python -m venv venv
source venv/bin/activate
pip install -r requirements_gui.txt
cd Depth-Anything-3-main && pip install -e . && cd ..
python depth_anything_gui.py
```

### Installation Complète - VFX ULTIMATE

```bash
# 1. Environnement
python -m venv venv_vfx
source venv_vfx/bin/activate  # Windows: venv_vfx\Scripts\activate

# 2. PyTorch avec CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. OpenEXR (IMPORTANT pour VFX)
# Linux:
sudo apt-get install libopenexr-dev libilmbase-dev
pip install openexr

# macOS:
brew install openexr
pip install openexr

# Windows:
conda install -c conda-forge openexr-python

# 4. Dépendances VFX
pip install -r requirements_vfx_ultimate.txt

# 5. Depth Anything v3
cd Depth-Anything-3-main && pip install -e . && cd ..

# 6. Lancer
python depth_anything_vfx_ultimate.py
```

**Vérification OpenEXR** :
```bash
python -c "import OpenEXR; print('OpenEXR OK ✓')"
```

---

## 🎬 Quick Start - Autodesk Flame

**Workflow le plus simple** : Depth of Field

```
1. Depth Anything v3:
   - Lancer: python depth_anything_vfx_ultimate.py
   - Import video: commercial.mp4
   - Mode: Monocular Depth
   - Export: OpenEXR Multi-Channel
   - Process → Choisir output folder

2. Autodesk Flame:
   - Media Panel → Import → Image Sequence
   - Sélectionner: depth.1001.exr
   - Format: OpenEXR, Multi-channel, Linear
   - Import

3. Utilisation:
   - Action → Lens → Depth of Field
   - Z-Depth Source: depth.Z channel
   - Ajuster Focus Point, F-Stop
   - Real-time preview!

✓ DOF cinématique en 5 minutes
```

**Pour workflows avancés** : Voir `FLAME_INTEGRATION.md`

---

## 📁 Structure du Projet

```
DEPTH/
├── 📄 START_HERE.md                    ← VOUS ÊTES ICI
│
├── ⭐ ÉDITION STANDARD
│   ├── depth_anything_gui.py           - Application principale
│   ├── README_GUI.md                   - Doc complète (500 lignes)
│   ├── QUICKSTART.md                   - Démarrage rapide
│   ├── requirements_gui.txt            - Dépendances
│   ├── launch_gui.sh                   - Launcher Linux/macOS
│   └── launch_gui.bat                  - Launcher Windows
│
├── ⭐⭐⭐ ÉDITION VFX ULTIMATE
│   ├── depth_anything_vfx_ultimate.py  - Application VFX
│   ├── README_VFX_ULTIMATE.md          - Guide VFX complet (600 lignes)
│   ├── FLAME_INTEGRATION.md            - Guide Flame (700 lignes)
│   ├── vfx_export_utils.py             - Utilitaires export
│   ├── example_vfx_export.py           - Exemples code
│   └── requirements_vfx_ultimate.txt   - Dépendances VFX
│
└── 📦 DEPTH ANYTHING V3 SOURCE
    └── Depth-Anything-3-main/          - Code source DA3
```

---

## 🎯 Workflow Recommandé

### Pour Utilisateurs Flame

**1. Lire d'abord** :
- `FLAME_INTEGRATION.md` (OBLIGATOIRE - tout y est !)

**2. Installer** :
- OpenEXR (essentiel)
- `requirements_vfx_ultimate.txt`

**3. Premier test** :
- Workflow DOF (section "Quick Start")
- 5 minutes pour voir les résultats

**4. Production** :
- Suivre workflows dans FLAME_INTEGRATION.md
- DOF, camera tracking, color grading, etc.

### Pour Autres VFX Software

**Nuke** :
- Même workflow que Flame
- OpenEXR multi-channel standard
- Voir README_VFX_ULTIMATE.md

**After Effects** :
- Application standard suffit
- Export TIFF 32-bit ou PNG
- Voir README_GUI.md

**Blender** :
- FBX pour camera tracking
- PLY pour point clouds
- Voir README_VFX_ULTIMATE.md

---

## 🆘 Aide & Support

### Problème OpenEXR ne s'installe pas
→ Voir `requirements_vfx_ultimate.txt` section "Troubleshooting"
→ Alternative conda : `conda install -c conda-forge openexr-python`

### Application ne démarre pas
→ Voir `QUICKSTART.md` section "Troubleshooting"
→ Vérifier Python 3.8+, PyTorch installé

### Flame n'importe pas mes fichiers
→ Voir `FLAME_INTEGRATION.md` section "Troubleshooting"
→ Vérifier format EXR multi-channel, linear color space

### Camera tracking décalé d'1 frame
→ Bug connu Flame
→ Solution dans `FLAME_INTEGRATION.md`

### Performances lentes
→ Voir `README_VFX_ULTIMATE.md` section "Performance"
→ Utiliser modèle plus petit, réduire résolution

---

## 📊 Comparaison Éditions

| Feature | Standard | VFX ULTIMATE |
|---------|----------|--------------|
| Interface PyQt6 | ✓ | ✓ |
| Depth estimation | ✓ | ✓ |
| GPU/CPU support | ✓ | ✓ |
| Export GLB/PLY/NPZ | ✓ | ✓ |
| **Import séquences EXR/DPX** | ✗ | ✓ |
| **OpenEXR multi-channel** | ✗ | ✓ |
| **DPX sequences export** | ✗ | ✓ |
| **FBX/Alembic camera** | ✗ | ✓ |
| **Normal maps** | ✗ | ✓ |
| **Flame integration** | ✗ | ✓ |
| **Frame numbering 1001+** | ✗ | ✓ |
| **Production workflows** | ✗ | ✓ |

**Recommandation** :
- **Standard** : Parfait pour 90% des utilisateurs
- **VFX ULTIMATE** : Indispensable si vous utilisez Flame/Nuke professionnellement

---

## 🔗 Liens Utiles

### Documentation Projet
- [Depth Anything v3 Project Page](https://depth-anything-3.github.io/)
- [GitHub Repository](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [Paper (arXiv)](https://arxiv.org/abs/2511.10647)

### Autodesk Flame
- [Flame 2025 Help](https://help.autodesk.com/view/FLAME/2025/ENU/)
- [Camera Tracking](https://help.autodesk.com/view/FLAME/2025/ENU/?guid=GUID-70B64EE8-0402-4842-ACF6-10D8492CCFC4)
- [OpenEXR Import](https://help.autodesk.com/view/FLAME/2023/ENU/?guid=GUID-C1DD8D7D-4F2D-4399-A216-3FB972710424)

### VFX Resources
- [OpenEXR.com](https://www.openexr.com/)
- [VFX Reference Platform](https://vfxplatform.com/)
- [fxguide](https://www.fxguide.com/)
- [Logik Forums](https://forum.logik.tv/)

---

## ✅ Checklist Premier Lancement

### Édition Standard
- [ ] Python 3.8+ installé
- [ ] PyTorch installé (CUDA si GPU)
- [ ] `pip install -r requirements_gui.txt`
- [ ] Depth Anything v3 installé (`cd Depth-Anything-3-main && pip install -e .`)
- [ ] Lancer : `python depth_anything_gui.py`
- [ ] Charger une image test
- [ ] Process → Voir depth map ✓

### Édition VFX ULTIMATE
- [ ] Tout ci-dessus +
- [ ] **OpenEXR installé** (vérifier avec `python -c "import OpenEXR"`)
- [ ] `pip install -r requirements_vfx_ultimate.txt`
- [ ] Lire `FLAME_INTEGRATION.md` (si utilise Flame)
- [ ] Lire `README_VFX_ULTIMATE.md`
- [ ] Tester `python example_vfx_export.py`
- [ ] Premier export OpenEXR multi-channel ✓

---

## 🎓 Prochaines Étapes

### Nouveau à Depth Anything
1. Lire `QUICKSTART.md`
2. Lancer application standard
3. Tester avec vos images
4. Explorer les modes (monocular, multiview, etc.)

### Professionnel VFX / Flame User
1. **Lire `FLAME_INTEGRATION.md`** (ESSENTIEL)
2. Installer OpenEXR
3. Tester workflow DOF (5 min)
4. Explorer autres workflows (camera tracking, etc.)
5. Intégrer dans votre pipeline

### Développeur / Pipeline TD
1. Lire `vfx_export_utils.py`
2. Étudier `example_vfx_export.py`
3. Intégrer dans vos scripts Python
4. Automatiser avec batch processing

---

## 💬 Questions Fréquentes

**Q: Quelle édition choisir ?**
A: Standard pour usage général, VFX ULTIMATE si vous utilisez Flame/Nuke professionnellement.

**Q: OpenEXR est obligatoire ?**
A: Non pour Standard, OUI pour VFX ULTIMATE (c'est le standard industrie).

**Q: Ça marche sur CPU ?**
A: Oui, mais beaucoup plus lent. GPU NVIDIA avec CUDA fortement recommandé.

**Q: Compatible avec Nuke ?**
A: Oui ! Même workflow que Flame, voir README_VFX_ULTIMATE.md

**Q: Je peux l'utiliser commercialement ?**
A: Dépend du modèle :
- DA3-GIANT, DA3-LARGE, DA3NESTED : Non-commercial (CC BY-NC)
- DA3-BASE, DA3-SMALL : Oui (Apache 2.0)

**Q: Besoin d'internet ?**
A: Seulement au premier lancement pour télécharger le modèle depuis Hugging Face.

---

## 🎬 Bon Workflow !

Vous êtes maintenant prêt à exploiter toute la puissance de Depth Anything v3 !

**Pour commencer MAINTENANT** :
- Utilisateur général → `./launch_gui.sh`
- Professionnel Flame → Lire `FLAME_INTEGRATION.md`

**Questions ?** Consultez les guides ou ouvrez une issue sur GitHub.

---

<div align="center">

**Made with ❤️ for the VFX Community**

Depth Anything v3 × PyQt6 × Autodesk Flame = 🔥

</div>
