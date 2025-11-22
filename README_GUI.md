# Depth Anything v3 - Application GUI Professionnelle

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyQt6](https://img.shields.io/badge/PyQt6-6.4%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Application GUI complète pour l'estimation de profondeur, la reconstruction 3D et le tracking avec Depth Anything v3**

[Fonctionnalités](#fonctionnalités) • [Installation](#installation) • [Utilisation](#utilisation) • [Modes](#modes) • [Documentation](#documentation)

</div>

---

## 🌟 Vue d'ensemble

Cette application PyQt6 professionnelle exploite toutes les capacités de **Depth Anything v3**, le modèle state-of-the-art pour l'estimation de profondeur et la reconstruction 3D. Elle offre une interface moderne et intuitive pour :

- ✅ **Estimation de profondeur monoculaire** : Carte de profondeur à partir d'une seule image
- ✅ **Estimation multi-vues** : Profondeur cohérente à partir de plusieurs images
- ✅ **Estimation de pose caméra** : Extrinsics et intrinsics automatiques
- ✅ **Reconstruction 3D Gaussians** : Génération de scènes 3D photo-réalistes
- ✅ **Traitement vidéo temps réel** : Profondeur en direct sur vidéos et webcam
- ✅ **Export multi-formats** : GLB, PLY, NPZ, images de profondeur
- ✅ **Visualisation 3D interactive** : Point clouds avec Open3D
- ✅ **Traitement batch** : Processus automatisé pour dossiers entiers

## 🚀 Fonctionnalités

### Interface Utilisateur

- **Design moderne** : Thème sombre professionnel avec interface responsive
- **Multi-onglets** : Visualisation séparée pour image originale, depth map, confidence, 3D, statistiques
- **Contrôles intuitifs** : Sélection facile des modes, paramètres et options d'export
- **Logs en temps réel** : Suivi détaillé des opérations avec codes couleur
- **Barre de progression** : Feedback visuel pendant le traitement

### Modes de Traitement

#### 1. **Monocular Depth** 🖼️
Estimation de profondeur à partir d'une seule image RGB.

**Cas d'usage** :
- Photos standards
- Images aériennes
- Scènes d'intérieur/extérieur

**Output** :
- Carte de profondeur normalisée
- Carte de confiance
- Export en multiples formats

#### 2. **Multi-View Depth** 📸📸
Estimation cohérente de profondeur à partir de plusieurs vues.

**Cas d'usage** :
- Séquences d'images
- Captures multi-angles
- Reconstruction de scènes

**Output** :
- Cartes de profondeur cohérentes
- Fusion haute qualité
- Modèles 3D exportables

#### 3. **Pose Estimation** 📷
Estimation des poses caméra et paramètres intrinsèques.

**Cas d'usage** :
- Calibration caméra automatique
- SLAM visuel
- Localisation 3D

**Output** :
- Extrinsics (rotation + translation)
- Intrinsics (matrice K)
- Trajectoire caméra

#### 4. **3D Gaussians** 🎨
Génération de Gaussians 3D pour novel view synthesis.

**Cas d'usage** :
- Synthèse de nouvelles vues
- Rendu photo-réaliste
- Réalité virtuelle

**Output** :
- Fichiers GLB/PLY
- Gaussians 3D
- Rendus haute fidélité

#### 5. **Real-time Video** 🎥
Traitement de fichiers vidéo frame par frame.

**Cas d'usage** :
- Analyse vidéo
- Séquences temporelles
- Effets visuels

**Features** :
- Contrôle FPS
- Prévisualisation temps réel
- Export séquences

#### 6. **Webcam** 📹
Flux en direct depuis webcam avec profondeur temps réel.

**Cas d'usage** :
- Démos interactives
- Applications AR
- Télémétrie en direct

**Features** :
- Latence minimale
- Affichage synchronisé
- Contrôle résolution

### Modèles Disponibles

| Modèle | Paramètres | Capacités | Recommandé pour |
|--------|-----------|-----------|-----------------|
| **DA3NESTED-GIANT-LARGE** | 1.4B | Toutes + Métrique | Production, meilleure qualité |
| **DA3-GIANT** | 1.15B | Toutes + Gaussians | Haute performance |
| **DA3-LARGE** | 0.35B | Complet | Usage général (recommandé) |
| **DA3-BASE** | 0.12B | Standard | Ressources limitées |
| **DA3-SMALL** | 0.08B | Léger | Mobile, edge devices |
| **DA3METRIC-LARGE** | 0.35B | Métrique mono | Mesures précises |
| **DA3MONO-LARGE** | 0.35B | Mono haute qualité | Profondeur relative |

### Formats d'Export

- **GLB** : Format 3D standard (compatible Blender, Unity, etc.)
- **PLY** : Point cloud format (MeshLab, CloudCompare)
- **NPZ** : Données NumPy (post-processing Python)
- **Depth Images** : PNG/JPG colorisés (visualisation)
- **All** : Export tous les formats simultanément

## 📦 Installation

### Prérequis

- **Python** : 3.8 ou supérieur
- **GPU** : CUDA-capable recommandé (8GB+ VRAM pour large models)
- **RAM** : 8GB minimum, 16GB+ recommandé
- **OS** : Windows, Linux, macOS

### Installation Rapide

```bash
# 1. Cloner ou télécharger ce repository
cd DEPTH

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Installer PyTorch avec support CUDA (si GPU disponible)
# Visitez https://pytorch.org/get-started/locally/ pour votre configuration
# Exemple pour CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Installer les dépendances GUI
pip install -r requirements_gui.txt

# 5. Installer Depth Anything v3
cd Depth-Anything-3-main
pip install -e .
cd ..

# 6. Lancer l'application
python depth_anything_gui.py
```

### Installation CPU uniquement

Si vous n'avez pas de GPU CUDA :

```bash
pip install torch torchvision  # Version CPU
pip install -r requirements_gui.txt
cd Depth-Anything-3-main && pip install -e . && cd ..
python depth_anything_gui.py
```

**Note** : Les performances seront significativement plus lentes sur CPU.

### Dépendances Optionnelles

```bash
# Pour xformers (accélération attention, recommandé)
pip install xformers

# Pour 3D Gaussians (fonctionnalités avancées)
pip install diff-gaussian-rasterization simple-knn

# Pour développement
pip install pre-commit black flake8
```

## 🎯 Utilisation

### Démarrage Rapide

1. **Lancer l'application**
   ```bash
   python depth_anything_gui.py
   ```

2. **Chargement du modèle**
   - Le modèle DA3-LARGE se charge automatiquement au démarrage
   - Ou sélectionnez un autre modèle et cliquez "Load Model"

3. **Charger des images**
   - Cliquez "Load Images" pour une ou plusieurs images
   - Ou "Load Video" pour un fichier vidéo
   - Ou "Load Folder" pour traitement batch

4. **Sélectionner le mode**
   - Cochez le mode souhaité (Monocular, Multi-View, etc.)

5. **Configurer l'export** (optionnel)
   - Choisissez le format d'export
   - Sélectionnez les options

6. **Traiter**
   - Cliquez "Process"
   - Visualisez les résultats dans les onglets

### Exemples d'Utilisation

#### Exemple 1 : Depth map simple

```
1. Load Model (DA3-LARGE)
2. Load Images → Sélectionner photo.jpg
3. Mode → Monocular Depth
4. Process
5. Voir résultat dans onglet "Depth Map"
```

#### Exemple 2 : Reconstruction 3D multi-vues

```
1. Load Model (DA3-LARGE ou GIANT)
2. Load Folder → Sélectionner dossier avec séquence d'images
3. Mode → Multi-View Depth
4. Export Format → GLB
5. Process
6. Onglet "3D View" → Open 3D Viewer
```

#### Exemple 3 : Webcam temps réel

```
1. Load Model (DA3-BASE pour rapidité)
2. Mode → Webcam
3. FPS → 15-30
4. Process
5. Arrêter avec bouton "Stop"
```

#### Exemple 4 : Traitement vidéo avec export

```
1. Load Model (DA3-LARGE)
2. Load Video → video.mp4
3. Mode → Real-time Video
4. Export Format → NPZ
5. FPS → 15
6. Process
```

## 🖥️ Interface Détaillée

### Panneau de Contrôle (Gauche)

#### Model Configuration
- **Dropdown** : Sélection du modèle
- **Load Model** : Charge le modèle en mémoire GPU/CPU
- **Status** : Indique l'état du modèle (couleur : orange=non chargé, vert=prêt, rouge=erreur)

#### Processing Mode
- **Radio buttons** : Sélection exclusive du mode de traitement
- Modes disponibles selon le modèle chargé

#### Input Source
- **Load Images** : Sélecteur multi-fichiers (PNG, JPG, JPEG, BMP, TIFF, WEBP)
- **Load Video** : Sélecteur fichier vidéo (MP4, AVI, MOV, MKV, FLV)
- **Load Folder** : Sélecteur dossier pour batch processing
- **Label** : Affiche nombre de fichiers chargés

#### Processing Options
- **Export Format** : Format de sortie souhaité
- **FPS** : Taux de rafraîchissement pour vidéo/webcam (1-60)
- **Use Metric Depth** : Profondeur métrique (si modèle support)
- **Show Confidence Map** : Afficher carte de confiance

#### Actions
- **Process** : Lance le traitement (devient "Stop" en mode vidéo)
- **Progress Bar** : Progression du traitement

#### Log
- Historique horodaté des opérations
- Codes couleur : Blanc=INFO, Vert=SUCCESS, Jaune=WARNING, Rouge=ERROR

### Panneau de Visualisation (Droite)

#### Onglets

1. **Original** : Image source (scroll si grande)
2. **Depth Map** : Carte de profondeur colorisée (colormap Inferno)
3. **Confidence** : Carte de confiance (colormap Jet)
4. **3D View** : Bouton pour ouvrir visualiseur 3D Open3D
5. **Statistics** : Statistiques détaillées sur la prédiction

### Barre d'Outils (Haut)

- **Open** : Raccourci pour charger images
- **Export** : Exporter résultats actuels
- **Clear** : Réinitialiser l'application
- **Help** : Aide et informations

### Barre de Statut (Bas)

- Affiche le dernier message de log

## 🔧 Configuration Avancée

### Optimisation Performances

#### GPU
```python
# L'application utilise automatiquement CUDA si disponible
# Pour forcer CPU :
# Modifier ligne dans depth_anything_gui.py:
self.device = "cpu"  # au lieu de auto-detection
```

#### Mémoire
Pour les GPUs avec mémoire limitée :
- Utiliser DA3-BASE ou DA3-SMALL
- Réduire la résolution des images
- Traiter en batch plus petits

#### Vitesse
Pour traitement temps réel optimal :
- DA3-SMALL ou DA3-BASE
- Réduire FPS (15 au lieu de 30)
- Réduire résolution entrée

### Personnalisation

#### Changer le thème
Modifier la fonction `apply_dark_theme()` dans `depth_anything_gui.py`

#### Ajouter des colormaps
Modifier la ligne `cv2.COLORMAP_INFERNO` vers :
- `cv2.COLORMAP_JET`
- `cv2.COLORMAP_TURBO`
- `cv2.COLORMAP_VIRIDIS`
- etc.

#### Export personnalisé
Ajouter des formats dans la fonction `export_results()`

## 📊 Statistiques et Métriques

L'onglet "Statistics" affiche :

- **Nombre d'images** : Nombre de frames traités
- **Shape** : Dimensions des depth maps (H×W)
- **Depth range** : [min, max] de profondeur
- **Mean/Std** : Statistiques de distribution
- **Confidence** : Moyenne de confiance (si disponible)
- **Extrinsics** : Matrices de pose caméra (si estimées)
- **Intrinsics** : Matrices de paramètres caméra (si estimées)

## 🐛 Troubleshooting

### Problème : "Failed to load model"

**Solutions** :
- Vérifier connexion internet (téléchargement depuis Hugging Face)
- Utiliser un miroir HuggingFace : `export HF_ENDPOINT=https://hf-mirror.com`
- Télécharger manuellement le modèle depuis HuggingFace

### Problème : "CUDA out of memory"

**Solutions** :
- Utiliser un modèle plus petit (DA3-BASE, DA3-SMALL)
- Réduire résolution des images
- Fermer autres applications GPU
- Utiliser CPU (plus lent)

### Problème : "xformers not available"

**Solutions** :
- Sur GPU anciens, xformers peut ne pas être supporté
- Voir https://github.com/ByteDance-Seed/Depth-Anything-3/issues/11
- L'application fonctionne sans (légèrement plus lent)

### Problème : "Video not opening"

**Solutions** :
- Vérifier codec vidéo (MP4/H264 recommandé)
- Installer codecs supplémentaires : `pip install opencv-python-headless`
- Convertir vidéo avec ffmpeg

### Problème : "Webcam not detected"

**Solutions** :
- Vérifier permissions caméra
- Essayer index différent (0, 1, 2...)
- Fermer autres applications utilisant la webcam

## 📚 Documentation Technique

### Architecture

```
depth_anything_gui.py
├── DepthAnythingGUI (QMainWindow)
│   ├── Model Management
│   ├── UI Components
│   └── Event Handlers
├── DepthWorker (QThread)
│   └── Async processing for images
└── VideoWorker (QThread)
    └── Async processing for video streams
```

### Workflow

```
User Input → Load Model → Load Data → Select Mode → Process
                                                        ↓
    ← Display Results ← Post-process ← Inference ← Preprocess
```

### API Depth Anything v3

L'application utilise l'API officielle :

```python
from depth_anything_3.api import DepthAnything3

# Charger modèle
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
model = model.to(device="cuda")

# Inférence
prediction = model.inference(images)

# Accès résultats
prediction.depth         # [N, H, W] float32 - Cartes de profondeur
prediction.conf          # [N, H, W] float32 - Cartes de confiance
prediction.extrinsics    # [N, 3, 4] float32 - Poses caméra
prediction.intrinsics    # [N, 3, 3] float32 - Paramètres caméra
prediction.processed_images  # [N, H, W, 3] uint8 - Images traitées
```

## 🎓 Cas d'Usage Avancés

### 1. Robotique - Navigation et Mapping

```python
# Utiliser mode Webcam avec DA3-BASE pour temps réel
# Export NPZ pour post-processing
# Intégrer avec ROS/ROS2 pour navigation
```

### 2. Réalité Augmentée - Occlusion

```python
# Mode Monocular pour profondeur instantanée
# Utiliser confidence map pour masking
# Intégration avec ARKit/ARCore
```

### 3. Cinéma - Effects VFX

```python
# Mode Video sur séquences haute résolution
# Export multi-formats pour pipeline 3D
# Utilisation dans Nuke, After Effects, Blender
```

### 4. Architecture - Reconstruction 3D

```python
# Mode Multi-View avec DA3-GIANT
# Export GLB/PLY pour modélisation
# Import dans Blender, SketchUp, Revit
```

### 5. Recherche - Dataset Augmentation

```python
# Batch processing sur folders
# Export NPZ pour réutilisation
# Génération synthetic depth labels
```

## 🌐 Ressources

### Liens Utiles

- **Depth Anything v3 Project** : https://depth-anything-3.github.io/
- **GitHub Repository** : https://github.com/ByteDance-Seed/Depth-Anything-3
- **Paper (arXiv)** : https://arxiv.org/abs/2511.10647
- **Hugging Face Models** : https://huggingface.co/depth-anything
- **PyQt6 Documentation** : https://doc.qt.io/qtforpython-6/

### Citations

Si vous utilisez cette application dans vos travaux, veuillez citer :

```bibtex
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```

## 📝 Changelog

### Version 1.0.0 (Initial Release)

- ✅ Interface PyQt6 complète
- ✅ Support 7 modèles DA3
- ✅ 6 modes de traitement
- ✅ Visualisation temps réel
- ✅ Export multi-formats
- ✅ Visualisation 3D
- ✅ Traitement batch
- ✅ Support GPU/CPU
- ✅ Thème dark moderne
- ✅ Documentation complète

## 📄 License

MIT License - Libre d'utilisation pour projets personnels et commerciaux.

Note : Les modèles Depth Anything v3 ont leurs propres licences :
- DA3-GIANT, DA3-LARGE, DA3NESTED : CC BY-NC 4.0 (Non-commercial)
- DA3-BASE, DA3-SMALL, DA3METRIC, DA3MONO : Apache 2.0 (Commercial OK)

## 🤝 Contribution

Contributions bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 💬 Support

Pour questions et support :

- **Issues** : Ouvrir une issue sur GitHub
- **Email** : [votre email]
- **Discussions** : GitHub Discussions

---

<div align="center">

**Développé avec ❤️ pour la communauté Computer Vision**

[⬆ Retour en haut](#depth-anything-v3---application-gui-professionnelle)

</div>
