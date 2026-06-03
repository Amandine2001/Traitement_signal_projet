# Traitement de signal - Projet de classification audio

Projet de traitement du signal pour la détection d’anomalies sur machines industrielles avec le dataset `MIMII` (source : https://zenodo.org/records/3384388).

## Objectif

Détecter l’état d’une machine (`normal` vs `abnormal`) à partir de fichiers audio `.wav`, puis analyser les anomalies pour proposer une classification multi-classe.

## Pipeline principal

Le point d’entrée est `main.py`.

1. Si `features.csv` est absent, charger les fichiers audio depuis le dossier `audio`.
2. Extraire les descripteurs audio pour chaque fichier.
3. Enregistrer les features dans `features.csv`.
4. Rééquilibrer le dataset en sous-échantillonnant la classe `normal`.
5. Encoder les labels en valeurs numériques.
6. Entraîner et évaluer :
   - un SVM linéaire
   - une Random Forest
7. Visualiser les matrices de confusion et analyser les clusters d’anomalies.

## Arborescence et modules

- `main.py` : orchestration du workflow complet.
- `chargement_audio.py` : exploration récursive du dossier `audio`, extraction des chemins et des labels.
- `bandeSonore.py` : classe de traitement audio et extraction des descripteurs.
- `extract.py` : construction du `DataFrame` des features et export CSV.
- `setData.py` : équilibrage par sous-échantillonnage de la classe `normal`.
- `model.py` : encodage des labels et entraînement des modèles de classification.
- `classification_anomalie.py` : analyse non supervisée des données anormales et visualisation des clusters.
- `features.csv` : fichier de features tabulaire généré par le pipeline.

## Format des données audio attendues

La structure attendue est :

`audio/<mecanisme>/<modele>/<normalite>/fichier.wav`

Exemples de labels :

- `mecanisme` : `fan`, `pump`, `slider`, `valve`
- `normalite` : `normal`, `abnormal`
- `modele` : identifiant machine converti ensuite en codes numériques

## Features extraites

Le fichier `features.csv` contient au moins les colonnes suivantes :

- `name`
- `mecanisme`
- `modele`
- `normalite`
- `centroid`
- `mfcc_mean`
- `chroma_mean`
- `tempo`
- `zcr`

## Description des modules

### `bandeSonore.py`

Cette classe charge un fichier audio et calcule :

- signal brut `y` et fréquence d’échantillonnage `sr`
- centroïde spectral
- MFCC
- chroma
- tempo
- ZCR

### `chargement_audio.py`

- Parcourt récursivement le dossier `audio`.
- Garde uniquement les fichiers `.wav`.
- Extrait `mecanisme`, `modele` et `normalite` depuis le chemin.

### `extract.py`

- Convertit chaque audio en dictionnaire de features.
- Construit un `pandas.DataFrame`.
- Exporte les données vers `features.csv`.

### `setData.py`

- Charge `features.csv`.
- Sépare les exemples `normal` et `abnormal`.
- Sous-échantillonne la classe `normal` pour équilibrer le dataset.

### `model.py`

- `remap_in_numeric_labels(df)`
  - `normalite` : `normal -> 0`, `abnormal -> 1`
  - `mecanisme` : mapping manuel (`fan`, `pump`, `slider`, `valve`)
  - `modele` : conversion en codes catégoriels
- `model_svm(df)` : entraînement d’un SVM linéaire et génération de matrice de confusion.
- `model_randomforest(df)` : entraînement d’une forêt aléatoire et génération de matrice de confusion.

### `classification_anomalie.py`

- filtre les exemples `abnormal`
- normalise les features et effectue un clustering KMeans
- visualise les clusters avec PCA, heatmaps et boxplots
- interprète chaque cluster en type de défaut (`défaut_roulement`, `défaut_moteur`, `défaut_ventilation`)
- génère des graphiques d’analyse des défauts

## Installation

Préconisé : utiliser `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `pyproject.toml` existe, mais les dépendances ne sont pas renseignées.

## Exécution

```bash
python main.py
```

Comportement :

- si `features.csv` est absent, le pipeline extrait d’abord les features audio puis crée le fichier CSV.
- si `features.csv` existe, le pipeline charge directement les données tabulaires.

## Résultats attendus

Pendant l’exécution, le projet affiche :

- le nombre de fichiers audio chargés
- un extrait du DataFrame rééquilibré
- un rapport de classification pour chaque modèle
- une matrice de confusion

En plus, plusieurs fichiers image sont sauvegardés :

- `matrice_confusion_svm.png`
- `matrice_confusion_randomforest.png`
- `clusters_pca.png`, `clusters_heatmap.png`, `clusters_<feature>.png`
- `defects_countplot.png`, `defects_box_<feature>.png`, `defects_mean_features.png`

## Améliorations possibles

- normaliser systématiquement les features avant entraînement
- ajouter une validation croisée (k-fold)
- sauvegarder les modèles entraînés avec `joblib`
- documenter `pyproject.toml` et les dépendances
- ajouter des tests unitaires pour l’extraction et le prétraitement
- pouvoir faire les spectrogramme d'un fichier normal et un fichier anormal pour chaque mécanisme et les comparer entre eux (malheureusement cela est complexe à réaliser s'il y a une impossibilité de téléchargement des fichiers `.WAV`).

## Analyse du signal et ses caractéristiques

Grâce à la classe `bandeSonore` et le module `chargement_audio`, il a été possible de convertir l'ensemble de ces fichiers (prenant environ 40Go) en un fichier `.CSV` exploitable par la suite en dataframe. En effet, cette classe permet de récupérer l'ensemble des caractéristiques demandées à l'exception des stft permettant de tracer les spectogrammes de chaque graphe (à savoir : fft, zcr, centroid, mfcc, chroma, tempo et zcr).

## Classification

Nous avons fait trois modèles de classification : 
- SVM et RandomForestClassifier pour savoir si le modèle repère lorsque le sons est normal ou non ;
- Multiple pour savoir de quelle anomalie il s'agit.

### SVM 

Le modèle SVM obtient une précision globale de 60 %, ce qui indique une performance modérée, légèrement supérieure à un classement aléatoire. Les deux classes sont relativement équilibrées, mais le modèle présente des difficultés à bien les séparer.

La classe 0 (normale) est légèrement mieux prédite en termes de précision, mais souffre d’un recall plus faible, ce qui signifie que plusieurs échantillons de cette classe sont mal détectés. À l’inverse, la classe 1 (anormale) est mieux détectée (meilleur recall), mais avec davantage de fausses prédictions.

La matrice de confusion confirme cette confusion entre les classes, avec un nombre important d’erreurs croisées. Cela suggère que les caractéristiques utilisées ne permettent pas une séparation claire entre les deux classes ou que le modèle SVM n’est pas suffisamment adapté sans optimisation supplémentaire.

En conclusion, le modèle montre une capacité de discrimination limitée et nécessiterait soit une amélioration des features, soit un ajustement des hyperparamètres pour obtenir de meilleures performances.

### RandomForestClassifier

Le modèle Random Forest obtient une accuracy de 81 %, ce qui indique une performance globalement bonne et nettement supérieure au hasard. Les scores de précision, recall et F1-score sont également équilibrés pour les deux classes, ce qui montre que le modèle généralise correctement sans privilégier fortement une classe par rapport à l’autre.

La classe 0 (normale) est bien détectée avec une bonne précision (0.80) et un bon recall (0.84), ce qui signifie que la majorité des échantillons de cette classe sont correctement identifiés. La classe 1 (anormale) présente des performances légèrement similaires, avec une précision de 0.82 et un recall de 0.79, indiquant une légère difficulté à capturer tous les cas de cette classe, mais des prédictions globalement fiables.

La matrice de confusion confirme ces résultats, avec une majorité de bonnes prédictions (561 pour la classe 0 et 511 pour la classe 1) et un nombre limité d’erreurs de classification.

En conclusion, le modèle Random Forest montre de bonnes performances globales, avec une capacité de classification stable et équilibrée entre les deux classes, ce qui en fait un modèle nettement plus performant et robuste que le SVM sur ce jeu de données.

### Multiple

Pour la gestion des defauts, nous n'avions pas suffisamment d'information dans le dataset. Nous avons donc choisi de nous baser sur les caractéristiques du signal (zcr, centroide, etc) pour déterminer le type d'anomalie associée. 

Ainsi, nous avons pu générer dans un premier temps plusieurs boxplots montrant la répartition des clusters en fonction de chaque caractéristiques et de voir les différents outliers possibles. L'ensemble de ces graphes sont d'abord générer avec un cluster portant un numéro. 

Ensuite, la visualisation `clusters_PCA.png` permet de bien voir les trois groupes d'anomalie (moteur, ventilation et rouloement). 

D'autres graphes sont générer suite à l'application d'une fonction permettant de baser sur les caractéristiques du signal, comme évoquer précédemment.

Enfin, un compte des type de défaut est effectué et peut être visualisé sous forme de barplot (`defects_countplot.png`) ainsi que la moyenne de chaque caractéristique pour chaque défaut relevé (`defects_mean_features.png`).

Ainsi, il y a seulement 4 machines ayant un defaut moteur, environ 200 pour un problème de ventilation et les 3000 restantes un problème de roulement.