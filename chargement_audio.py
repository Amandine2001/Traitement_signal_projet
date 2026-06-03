"""Chargement des fichiers audio du jeu de données d'entraînement.

Ce module parcourt une arborescence de dossiers, repère les fichiers WAV
et construit une liste d'objets `bandeSonore` enrichis avec les labels
extraits du chemin.
"""

import os
from bandeSonore import bandeSonore

def load_audio_files(datapath):
    """Charge tous les fichiers WAV trouvés sous `datapath`.

    La fonction suppose une structure de dossiers où le chemin relatif
    contient les informations de classement sous la forme:
    `.../<mecanisme>/<modele>/<normalite>/fichier.wav`.
    """
    training_data = []
    count = 0
    print(f"Loading audio files from: {datapath}")

    # Parcours récursif de tout le répertoire de données.
    for root, dirs, files in os.walk(datapath):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)

                # Extraction du chemin relatif pour récupérer les labels.
                relative_path = os.path.relpath(root, datapath)

                # Le dossier doit fournir les classes attendues dans cet ordre.
                _, mecanisme, modele, normalite = relative_path.split(os.sep)

                # Chaque entrée contient l'objet audio et ses étiquettes.
                training_data.append({
                    "audio": bandeSonore(file, file_path),
                    "mecanisme": mecanisme,
                    "modele": modele,
                    "normalite": normalite
                })

                # Journalisation simple pour suivre l'avancement du chargement.
                print(
                    f"count: {count} | file: {file} | mecanisme: {mecanisme} | modele: {modele} | normalite: {normalite}"
                )

                count += 1

    return training_data
