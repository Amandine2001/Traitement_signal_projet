import bandeSonore
import pandas as pd


def create_features_dataframe(
    features_list: list[dict[str, bandeSonore.bandeSonore | str]],
):

    df = pd.DataFrame(
        {
            "mecanisme": item["mecanisme"] if "mecanisme" in item else None,
            "modele": item["modele"] if "modele" in item else None,
            "normalite": item["normalite"] if "normalite" in item else None,
            "centroid": item["audio"].centroid
            if isinstance(item["audio"], bandeSonore.bandeSonore)
            else None,
            "mfcc_mean": item["audio"].mfcc.mean()
            if isinstance(item["audio"], bandeSonore.bandeSonore)
            else None,
            "chroma_mean": item["audio"].chroma.mean()
            if isinstance(item["audio"], bandeSonore.bandeSonore)
            else None,
            "tempo": item["audio"].tempo
            if isinstance(item["audio"], bandeSonore.bandeSonore)
            else None,
            "zcr": item["audio"].zcr
            if isinstance(item["audio"], bandeSonore.bandeSonore)
            else None,
        }
        for item in features_list
    )

    return df
