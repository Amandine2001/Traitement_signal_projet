import bandeSonore
import pandas as pd


def create_dataframe(
    features_list: list[dict[str, bandeSonore.bandeSonore | str]],
):

    data = list()

    for item in features_list:
        audio: bandeSonore.bandeSonore = item["audio"]  # ty:ignore[invalid-assignment]
        mecanisme: str = item["mecanisme"]  # ty:ignore[invalid-assignment]
        modele: str = item["modele"]  # ty:ignore[invalid-assignment]
        normalite: str = item["normalite"]  # ty:ignore[invalid-assignment]

        data.append(
            {
                "name": audio.name,
                "mecanisme": mecanisme,
                "modele": modele,
                "normalite": normalite,
                "centroid": audio.centroid,
                "mfcc_mean": audio.mfcc.mean(),
                "chroma_mean": audio.chroma.mean(),
                "tempo": audio.tempo[0] if audio.tempo else None,
                "zcr": audio.zcr.mean() if audio.zcr is not None else None,  # ty:ignore[unresolved-attribute]
            }
        )

    df = pd.DataFrame(data)

    return df


def save_dataframe(df: pd.DataFrame, filename: str = "features.csv"):
    df.to_csv(filename, index=False)
