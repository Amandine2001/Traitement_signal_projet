import os
import shutil


def save_png(original_folder, output_folder):
    # Crée le dossier output s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # Parcourt les fichiers du dossier source
    for fichier in os.listdir(original_folder):
        if fichier.lower().endswith(".png"):
            original_path = os.path.join(original_folder, fichier)
            destination_path = os.path.join(output_folder, fichier)

            shutil.move(original_path, destination_path)

    return print("Every files PNG are moved")
