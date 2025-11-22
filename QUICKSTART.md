# 🚀 Guide de Démarrage Rapide - Depth Anything v3 GUI

## Installation Express (5 minutes)

### Linux / macOS

```bash
# 1. Extraire l'archive Depth Anything v3 (déjà fait)
unzip Depth-Anything-3-main.zip

# 2. Lancer le script d'installation automatique
./launch_gui.sh
```

C'est tout ! Le script va :
- ✅ Créer l'environnement virtuel
- ✅ Installer toutes les dépendances
- ✅ Lancer l'application

### Windows

```cmd
REM 1. Extraire l'archive Depth Anything v3 (déjà fait si vous lisez ceci)

REM 2. Double-cliquer sur launch_gui.bat
REM Ou depuis le terminal :
launch_gui.bat
```

Le script fait tout automatiquement !

---

## Installation Manuelle (si scripts ne fonctionnent pas)

### 1. Environnement Virtuel

```bash
# Créer
python -m venv venv

# Activer
source venv/bin/activate  # Linux/macOS
# OU
venv\Scripts\activate  # Windows
```

### 2. PyTorch avec CUDA (si GPU disponible)

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OU CPU seulement (plus lent)
pip install torch torchvision
```

### 3. Dépendances GUI

```bash
pip install -r requirements_gui.txt
```

### 4. Depth Anything v3

```bash
cd Depth-Anything-3-main
pip install -e .
cd ..
```

### 5. Lancer

```bash
python depth_anything_gui.py
```

---

## Premier Test (2 minutes)

### Test 1 : Image Simple

1. **Lancer l'app** → Le modèle DA3-LARGE se charge automatiquement
2. **Télécharger une image test** :
   ```bash
   # Exemple d'image
   wget https://images.unsplash.com/photo-1506905925346-21bda4d32df4 -O test.jpg
   ```
3. **Dans l'app** :
   - Cliquer "Load Images" → Sélectionner `test.jpg`
   - Mode → "Monocular Depth" (déjà sélectionné)
   - Cliquer "Process"
4. **Voir le résultat** dans l'onglet "Depth Map"

✅ Si vous voyez une carte de profondeur colorée → **Succès !**

### Test 2 : Webcam Temps Réel

1. **Mode** → Cocher "Webcam"
2. **FPS** → 15
3. **Process** → Voir votre profondeur en direct !
4. **Stop** pour arrêter

### Test 3 : Export 3D

1. **Charger une image**
2. **Mode** → "Monocular Depth"
3. **Export Format** → "GLB"
4. **Process** → Sélectionner dossier de sortie
5. **Ouvrir le .glb** dans Blender, Windows 3D Viewer, etc.

---

## Configuration Requise

### Minimum
- **CPU** : Dual-core 2.0+ GHz
- **RAM** : 8 GB
- **GPU** : Optionnel (CPU fonctionne)
- **Espace** : 5 GB

### Recommandé
- **CPU** : Quad-core 3.0+ GHz
- **RAM** : 16 GB
- **GPU** : NVIDIA GTX 1060+ (6GB VRAM) avec CUDA
- **Espace** : 10 GB

### Optimal
- **CPU** : 8+ cores
- **RAM** : 32 GB+
- **GPU** : NVIDIA RTX 3090+ (24GB VRAM)
- **Espace** : 20 GB

---

## Choix du Modèle

| Situation | Modèle Recommandé | Pourquoi |
|-----------|------------------|----------|
| **Premier essai** | DA3-LARGE | Bon équilibre qualité/vitesse |
| **Production, meilleure qualité** | DA3NESTED-GIANT-LARGE | Meilleure précision + métrique |
| **Temps réel (webcam)** | DA3-BASE | Rapide, qualité correcte |
| **GPU faible (4GB VRAM)** | DA3-SMALL | Léger, tient en mémoire |
| **CPU uniquement** | DA3-BASE | Acceptable sur CPU moderne |
| **Mesures précises** | DA3METRIC-LARGE | Profondeur métrique réelle |

---

## Problèmes Courants

### ❌ "CUDA out of memory"

**Solution** :
```python
# Dans l'app, changer le modèle pour un plus petit
DA3-LARGE → DA3-BASE → DA3-SMALL
```

### ❌ "ModuleNotFoundError: No module named 'PyQt6'"

**Solution** :
```bash
# Vérifier que venv est activé
source venv/bin/activate  # ou venv\Scripts\activate

# Réinstaller
pip install -r requirements_gui.txt
```

### ❌ "Cannot download model from HuggingFace"

**Solution 1 : Miroir**
```bash
export HF_ENDPOINT=https://hf-mirror.com
python depth_anything_gui.py
```

**Solution 2 : Téléchargement manuel**
1. Aller sur https://huggingface.co/depth-anything/DA3-LARGE
2. Télécharger les fichiers
3. Placer dans `~/.cache/huggingface/hub/`

### ❌ "Application crashes on start"

**Vérifications** :
```bash
# 1. Python version (doit être 3.8+)
python --version

# 2. PyQt6 installé
python -c "import PyQt6; print('OK')"

# 3. PyTorch installé
python -c "import torch; print('OK')"

# 4. Logs détaillés
python depth_anything_gui.py 2>&1 | tee debug.log
```

---

## Exemples d'Images

### Télécharger des images de test

```bash
# Paysage
wget https://images.unsplash.com/photo-1506905925346-21bda4d32df4 -O landscape.jpg

# Intérieur
wget https://images.unsplash.com/photo-1616486338812-3dadae4b4ace -O interior.jpg

# Portrait
wget https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d -O portrait.jpg

# Urbain
wget https://images.unsplash.com/photo-1477959858617-67f85cf4f1df -O city.jpg
```

### Tester multi-vues

```bash
# Créer un dossier
mkdir test_multiview

# Télécharger une séquence (exemple : photos d'un objet sous différents angles)
# Placer 5-10 images dans test_multiview/

# Dans l'app :
# Load Folder → test_multiview
# Mode → Multi-View Depth
# Export Format → GLB
# Process
```

---

## Raccourcis Clavier (dans l'app)

- `Ctrl+O` : Open (Load Images)
- `Ctrl+S` : Save (Export)
- `Ctrl+Q` : Quit
- `F1` : Help

---

## Workflow Recommandé

### Cas d'usage : Reconstruction 3D d'un objet

1. **Capturer** : Prendre 10-20 photos de l'objet sous différents angles
2. **Importer** : Load Folder → Sélectionner dossier
3. **Traiter** :
   - Model : DA3-LARGE ou GIANT
   - Mode : Multi-View Depth
   - Export : GLB
   - Process
4. **Visualiser** : Onglet 3D View → Open 3D Viewer
5. **Exporter** : Ouvrir .glb dans Blender pour nettoyage/texturing

### Cas d'usage : Profondeur pour vidéo

1. **Importer** : Load Video → video.mp4
2. **Configurer** :
   - Model : DA3-LARGE
   - Mode : Real-time Video
   - FPS : 15-30
   - Export : NPZ (pour post-processing) + Depth Images
3. **Traiter** : Process
4. **Post-prod** : Utiliser .npz dans Python ou images dans After Effects

### Cas d'usage : Application AR (tracking temps réel)

1. **Setup** :
   - Model : DA3-BASE (vitesse)
   - Mode : Webcam
   - FPS : 30
   - Show Confidence : ON
2. **Lancer** : Process
3. **Intégrer** : Utiliser depth stream pour occlusion/placement objets 3D

---

## Performance Attendue

### Temps de Traitement (image 1080p)

| GPU | DA3-SMALL | DA3-BASE | DA3-LARGE | DA3-GIANT |
|-----|-----------|----------|-----------|-----------|
| **RTX 4090** | 0.05s | 0.08s | 0.15s | 0.4s |
| **RTX 3090** | 0.08s | 0.12s | 0.25s | 0.7s |
| **RTX 2080** | 0.15s | 0.25s | 0.5s | 1.5s |
| **GTX 1080** | 0.3s | 0.5s | 1.0s | 3.0s |
| **CPU (i7)** | 3s | 5s | 10s | 30s+ |

### FPS Webcam (640x480)

| GPU | DA3-SMALL | DA3-BASE | DA3-LARGE |
|-----|-----------|----------|-----------|
| **RTX 4090** | 60+ FPS | 45 FPS | 25 FPS |
| **RTX 3090** | 45 FPS | 30 FPS | 15 FPS |
| **RTX 2080** | 30 FPS | 20 FPS | 10 FPS |
| **GTX 1080** | 20 FPS | 12 FPS | 6 FPS |

---

## Prochaines Étapes

Maintenant que l'app fonctionne, consultez :

- 📖 **README_GUI.md** : Documentation complète
- 🔗 **https://depth-anything-3.github.io/** : Project page officiel
- 📄 **https://arxiv.org/abs/2511.10647** : Paper technique
- 🤗 **https://huggingface.co/depth-anything** : Tous les modèles

---

## Support

**Problème non résolu ?**

1. Vérifier README_GUI.md section Troubleshooting
2. Chercher dans GitHub Issues : https://github.com/ByteDance-Seed/Depth-Anything-3/issues
3. Ouvrir une nouvelle issue avec :
   - OS et version
   - Python version
   - GPU/CPU
   - Logs d'erreur complets

---

<div align="center">

**✨ Profitez de Depth Anything v3 ! ✨**

[⬆ Retour en haut](#-guide-de-démarrage-rapide---depth-anything-v3-gui)

</div>
