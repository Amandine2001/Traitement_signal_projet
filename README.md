# Traitement signal projet (Maxime, Amandine, Johann)

Projet en traitement numérique du signal avec un dataset `MIMII` pour l’investigation et l’inspection des machines industrielles défectueuses, issu du site https://zenodo.org/records/3384388. 

**Objectif**: détecter l'état normal/anormal d'un équipement mécanique à partir de fichiers `.WAV`, en extrayant des descripteurs audios puis en entrainant deux modèles de classification binaire (SVM et Random Forest) puis faire une classification multiclasse (normal, défaut roulement, défaut moteur, défaut ventilation).

## Vue d'ensemble du pipeline

Le flux principal est dans `main.py`:

1. Charger les fichiers audio depuis le dossier `audio` (si `features.csv` n'existe pas).
2. Extraire les features acoustiques pour chaque fichier.
3. Sauvegarder ces features dans `features.csv`.
4. Rééquilibrer les classes à l'aide d'un sous-échantillonnage de la classe `normal`.
5. Encoder les labels en valeurs numeriques.
6. Entrainer et evaluer:
	 - un SVM lineaire
	 - une Random Forest
7. Afficher les métriques (classification report, matrice de confusion) et des heatmaps.

## Structure du projet

- `main.py`: orchestration complete du workflow.
- `chargement_audio.py`: parcours recursif des fichiers `.wav` et creation des exemples labels.
- `bandeSonore.py`: classe d'analyse audio (chargement, MFCC, chroma, tempo, centroid, ZCR, etc.).
- `extract.py`: transformation de la liste d'exemples en DataFrame de features.
- `setData.py`: reequilibrage des classes.
- `model.py`: encodage des labels + entrainement/evaluation des modeles ML.
- `features.csv`: jeu de donnees tabulaire des features deja extraites.

## Format de donnees attendu

La fonction `load_audio_files` suppose une arborescence sous `audio` contenant les labels dans le chemin:

`.../<mecanisme>/<modele>/<normalite>/fichier.wav`

Exemple de labels utilises dans le code:

- `mecanisme`: `fan`, `pump`, `slider`, `valve`
- `normalite`: `normal`, `abnormal`
- `modele`: converti ensuite en codes categoriels

## Features extraites

Les colonnes generees dans `features.csv` sont:

- `name`: nom du fichier audio
- `mecanisme`: type de mecanisme
- `modele`: identifiant du modele machine
- `normalite`: label cible (normal/abnormal)
- `centroid`: centroide spectral moyen
- `mfcc_mean`: moyenne globale des MFCC
- `chroma_mean`: moyenne globale de la chroma
- `tempo`: tempo estime (BPM)
- `zcr`: zero crossing rate moyen

## Detail des modules

### 1) `bandeSonore.py`

La classe `bandeSonore` charge un fichier et calcule automatiquement:

- signal brut `y` et frequence d'echantillonnage `sr`
- centroide spectral
- STFT
- MFCC
- chroma
- note dominante
- tempo
- ZCR
- vecteur de features concatene

Le module contient aussi des fonctions utilitaires pour:

- affichage (forme d'onde, spectrogramme, MFCC, chroma)
- separation harmonique/percussive (HPSS)
- filtrage passe-bas/passe-haut

### 2) `chargement_audio.py`

- Parcourt recursivement `audio`.
- Ne garde que les `.wav`.
- Cree pour chaque fichier une entree de la forme:
	- `audio`: instance de `bandeSonore`
	- `mecanisme`, `modele`, `normalite`: labels extraits du chemin.

### 3) `extract.py`

- Convertit la liste d'exemples en `pandas.DataFrame`.
- Utilise certaines statistiques resumes des objets audio:
	- `mfcc.mean()`, `chroma.mean()`, `tempo[0]`, `zcr.mean()`.
- Ecrit ensuite les donnees dans `features.csv`.

### 4) `setData.py`

- Lit `features.csv`.
- Separer `normal` vs `anormal`.
- Sous-echantillonne la classe `normal` pour avoir autant d'exemples que la classe anormale.
- Retourne un DataFrame reequilibre.

### 5) `model.py`

- `remap_in_numeric_labels(df)`:
	- `normalite`: `normal -> 0`, `abnormal -> 1`
	- `mecanisme`: mapping manuel (`fan`, `pump`, `slider`, `valve`)
	- `modele`: conversion en codes categoriels

- `model_svm(df)`:
	- split train/test (80/20)
	- entrainement SVM lineaire
	- evaluation + matrice de confusion + heatmap

- `model_randomforest(df)`:
	- split train/test (80/20)
	- entrainement RandomForest
	- evaluation + matrice de confusion + heatmap

## Installation

### Option 1: pip + requirements

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option 2: pyproject.toml (a completer)

Le fichier `pyproject.toml` existe mais ne reference pas encore les dependances du projet. Pour un usage immediat, preferer `requirements.txt`.

## Execution

```bash
python main.py
```

Comportement:

- Si `features.csv` est absent: extraction audio complete puis creation du CSV.
- Si `features.csv` existe deja: reutilisation directe des features tabulaires.

## Resultats affiches

Pendant l'execution, le script affiche:

- le nombre de fichiers charges
- un rapport de classification pour chaque modele
- la matrice de confusion brute
- une heatmap de la matrice de confusion

## Ameliorations possibles

- Ajouter une normalisation des features (StandardScaler) avant SVM.
- Faire une validation croisee (k-fold) au lieu d'un seul split train/test.
- Sauvegarder les modeles entraines (`joblib`).
- Ajouter des tests unitaires pour les modules d'extraction et de preprocessing.
- Completer `pyproject.toml` avec les dependances reelles.
