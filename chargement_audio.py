import os
from bandeSonore import bandeSonore


def load_audio_files(datapath):
    training_data: list[dict[str, bandeSonore | str]] = []
    count = 0
    print(f"Loading audio files from: {datapath}")
    for root, dirs, files in os.walk(datapath):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)

                # Chemin relatif à DATA_PATH
                relative_path = os.path.relpath(root, datapath)

                # mecanisme/modele/normalite
                _, mecanisme, modele, normalite = relative_path.split(os.sep)

                training_data.append(
                    {
                        "audio": bandeSonore(file, file_path),
                        "mecanisme": mecanisme,
                        "modele": modele,
                        "normalite": normalite,
                    }
                )

                print(
                    f"count: {count} | file: {file} | mecanisme: {mecanisme} | modele: {modele} | normalite: {normalite}"
                )

                count += 1

    return training_data
